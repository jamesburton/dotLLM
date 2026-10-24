using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side KV cache. Per-layer device-local buffers of shape
/// <c>[maxSeqLen, numKvHeads * headDim]</c>, FP32 or (issue #1171) FP16 (<see cref="Storage"/>). The host never touches
/// cached K/V — updates are recorded as <c>vkCmdCopyBuffer</c> from the
/// host-visible activation buffers to the device-local cache, either
/// synchronously (legacy <see cref="UpdateDevice"/>) or appended to a caller-
/// supplied command buffer (<see cref="RecordUpdate"/>, used by the
/// fence-pipelined forward pass).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <c>DotLLM.Engine.KvCache.SimpleKvCache</c> semantics: <c>Update</c>
/// appends new K/V rows at the supplied position indices. The attention kernel
/// reads straight from the device buffers via <see cref="GetKeysBuffer"/> /
/// <see cref="GetValuesBuffer"/>; no staging copies are required.
/// </para>
/// <para>
/// Implements <see cref="IKvCache"/> so code that already knows about the CPU
/// cache semantics (text-generation loop, tests) can swap the backing store
/// transparently. The <see cref="IKvCache.Update(ITensor, ITensor, ReadOnlySpan{int}, int)"/>
/// and <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/>
/// overloads expect CPU-resident tensor pointers (the caller is responsible
/// for uploading); we only use the device-side path from
/// <see cref="VulkanTransformerModel"/>, but the IKvCache contract lets this
/// object satisfy the same API.
/// </para>
/// </remarks>
public sealed class VulkanKvCache : IKvCache, IPerLayerKvCache, IHostStagedKvCache, IWindowedKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    // Per-layer row stride (numKvHeads × headDim, FP32 elements). Uniform across
    // layers for every dense/GQA/MoE model; PER-LAYER for Gemma-4, whose sliding
    // and global layers carry different KV-head counts AND head dims (e.g. sliding
    // 8×256 vs global 2×512) — so each layer's cached K/V row is a different width.
    private readonly int[] _kvStride;
    private readonly VulkanKvStorage _storage;
    private readonly int _elemBytes;

    // Compact sliding-window rows (#1176): null unless the geometry marks windowed slots. Grow-only, so a resize (which swaps
    // buffer handles and invalidates descriptor sets) happens only in EnsureWindowCapacity, before recording.
    private readonly SlidingWindowRows? _rows;

    // TILE_KV of the reference sink kernel (attention_sinks_f32.glsl), which tiles the KV range from row 0. Ring bases stay multiples of it,
    // so the kernel sees the same tiles as on a full-length cache and the ring is bit-identical to it (the decode / small-row kernels
    // split relative to the window start and do not care). The GQA flash sink twins (#1200) tile in 16 / 32 rows, which divide it.
    private const int KvTileRows = 256;
    private readonly VulkanDevice.Buffer? _moveScratch; // overlapping compaction moves stage through here (K then V half)
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>
    /// Element format of the cached rows. <see cref="VulkanKvStorage.F16"/> rows are written by the F16 convert kernel and read in place by the
    /// <c>_kvf16</c> attention twins; only models that create those twins (Flint, issue #1171) accept such a cache.
    /// </summary>
    public VulkanKvStorage Storage => _storage;

    /// <summary>Device bytes held by the K and V buffers of every layer.</summary>
    public long AllocatedBytes
    {
        get
        {
            if (_rows is not null) return _rows.AllocatedBytes(i => 2L * _kvStride[i] * _elemBytes);
            long total = 0;
            for (int i = 0; i < _numLayers; i++) total += 2L * _maxSeqLen * _kvStride[i] * _elemBytes;
            return total;
        }
    }

    /// <summary>
    /// Creates the per-layer K/V buffers with a UNIFORM <c>numKvHeads × headDim</c>
    /// row stride (every layer the same width — the dense / GQA / MoE case).
    /// Memory is not zeroed — the forward pass only reads positions it has written.
    /// </summary>
    public VulkanKvCache(VulkanDevice device, int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        : this(device, BuildUniformStrides(numLayers, numKvHeads, headDim), maxSeqLen)
    {
    }

    /// <summary>
    /// Creates the per-layer K/V buffers from a Core <see cref="KvGeometry"/> descriptor.
    /// Byte-identical to the uniform constructor for every dense/GQA/MoE model and supplies
    /// distinct per-layer strides for Gemma-4. This is the constructor
    /// <see cref="VulkanTransformerModel.CreateKvCache(int)"/> uses (via
    /// <see cref="KvGeometry.FromConfig(DotLLM.Core.Models.ModelConfig)"/>), so the per-layer-stride derivation lives in one
    /// place (Core) rather than being rebuilt inline.
    /// </summary>
    public VulkanKvCache(VulkanDevice device, KvGeometry geometry, int maxSeqLen)
        : this(device, geometry, maxSeqLen, VulkanKvStorage.F32)
    {
    }

    /// <summary>
    /// As <see cref="VulkanKvCache(VulkanDevice, KvGeometry, int)"/> with an explicit element format (issue #1171). A geometry with
    /// sliding windows (<see cref="KvGeometry.HasSlidingWindows"/>) keeps only the recent rows of those slots (issue #1176): see
    /// <see cref="IWindowedKvCache"/> and <see cref="EnsureWindowCapacity"/>.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="geometry">Per-layer KV geometry.</param>
    /// <param name="maxSeqLen">Positions per layer.</param>
    /// <param name="storage">Element format of the cached rows.</param>
    public VulkanKvCache(VulkanDevice device, KvGeometry geometry, int maxSeqLen, VulkanKvStorage storage)
        : this(device, ExtractStrides(geometry), maxSeqLen, storage,
            geometry.HasSlidingWindows ? new SlidingWindowRows(geometry, maxSeqLen, growOnly: true, baseAlignment: KvTileRows) : null)
    {
    }

    private static int[] ExtractStrides(KvGeometry geometry)
    {
        var strides = new int[geometry.LayerCount];
        for (int l = 0; l < strides.Length; l++)
            strides[l] = geometry.KvStrideOf(l);
        return strides;
    }

    /// <summary>
    /// Creates the per-layer K/V buffers from an explicit PER-LAYER row stride
    /// (FP32 elements per token position, i.e. <c>numKvHeads × headDim</c> for that
    /// layer). Used by Gemma-4, whose sliding and global layers have different
    /// KV-head counts and head dims, so each layer's cached row is a different width.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="kvStridePerLayer">Per-layer row width in elements.</param>
    /// <param name="maxSeqLen">Positions per layer.</param>
    /// <param name="storage">Element format of the cached rows (F16 needs an even row width: two halves per word).</param>
    public VulkanKvCache(VulkanDevice device, int[] kvStridePerLayer, int maxSeqLen, VulkanKvStorage storage = VulkanKvStorage.F32)
        : this(device, kvStridePerLayer, maxSeqLen, storage, rows: null)
    {
    }

    private VulkanKvCache(VulkanDevice device, int[] kvStridePerLayer, int maxSeqLen, VulkanKvStorage storage, SlidingWindowRows? rows)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(kvStridePerLayer);
        if (kvStridePerLayer.Length == 0) throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _device = device;
        _numLayers = kvStridePerLayer.Length;
        _maxSeqLen = maxSeqLen;
        _kvStride = (int[])kvStridePerLayer.Clone();
        _storage = storage;
        _elemBytes = storage.ElementBytes();
        _rows = rows;

        _keys = new VulkanDevice.Buffer[_numLayers];
        _values = new VulkanDevice.Buffer[_numLayers];

        for (int i = 0; i < _numLayers; i++)
        {
            if (_kvStride[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer),
                    $"Per-layer KV stride must be positive; layer {i} = {_kvStride[i]}.");
            if (storage == VulkanKvStorage.F16 && (_kvStride[i] & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer), $"An F16 KV row must hold an even element count; layer {i} = {_kvStride[i]}.");
            long bytesPerLayer = (long)(rows?.CapacityOf(i) ?? maxSeqLen) * _kvStride[i] * _elemBytes;
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
        }

        if (rows is not null)
        {
            // K and V of the largest possible compaction stage side by side.
            long scratch = 0;
            for (int i = 0; i < _numLayers; i++)
                scratch = Math.Max(scratch, 2L * rows.MaxKeptRows(i) * _kvStride[i] * _elemBytes);
            _moveScratch = device.AllocateDeviceLocal(scratch);
        }
    }

    private static int[] BuildUniformStrides(int numLayers, int numKvHeads, int headDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        var strides = new int[numLayers];
        Array.Fill(strides, numKvHeads * headDim);
        return strides;
    }

    /// <summary>Returns the device buffer holding cached keys for the given layer.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns the device buffer holding cached values for the given layer.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int layerIndex) => _values[layerIndex];

    /// <summary>
    /// Copies new <paramref name="kDev"/> / <paramref name="vDev"/> rows into
    /// the device-local cached K/V buffers at the given positions. Source
    /// buffers are the current forward pass's host-visible K/V activations;
    /// the cache destination is device-local (VRAM on a dGPU, driver-tiled on
    /// UMA). Issues a synchronous <c>vkCmdCopyBuffer</c> + fence wait.
    /// Prefer <see cref="RecordUpdate"/> from the fence-pipelined forward pass.
    /// </summary>
    internal void UpdateDevice(
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        RequireF32("UpdateDevice");
        ThrowIfWindowed(layerIndex, "UpdateDevice");

        int rowBytes = _kvStride[layerIndex] * sizeof(float);

        // Single contiguous range if positions are consecutive — one copy call
        // covers the whole seqLen. Otherwise fall back to per-row copies.
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        if (contiguous)
        {
            int startPos = positions[0];
            ulong totalBytes = (ulong)rowBytes * (ulong)seqLen;
            _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
            _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
                _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>.
    /// The caller is responsible for the <c>TRANSFER → COMPUTE_SHADER</c>
    /// barrier that follows (so the attention kernel reads the freshly
    /// written cache rows), and for advancing <see cref="CurrentLength"/>
    /// after the batch commits.
    /// </summary>
    /// <remarks>An F16 cache is written by <paramref name="f16Writer"/> (a compute dispatch: the caller's following barrier is compute to compute).</remarks>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex,
        Kernels.IVulkanKvF16Writer? f16Writer = null)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int rowBytes = _kvStride[layerIndex] * sizeof(float);
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        // Compact sliding-window slot (#1176): compact on the device first, then write at row = position - base.
        int rowBase = 0;
        if (_rows is not null && _rows.IsWindowed(layerIndex))
        {
            if (!contiguous)
                throw new InvalidOperationException("A compact sliding-window KV slot is written from contiguous positions only.");
            RecordWindowMove(cmdBuf, layerIndex, positions[0], maxPos);
            rowBase = _rows.BaseOf(layerIndex);
        }

        if (_storage == VulkanKvStorage.F16)
        {
            // The F32 rows the projections produced are converted on the device; there is no copy-engine path for F16.
            if (!contiguous || f16Writer is null)
                throw new InvalidOperationException("An F16 KV cache is written by the convert kernel from contiguous positions only.");
            long elemOffset = (long)(positions[0] - rowBase) * _kvStride[layerIndex], count = (long)seqLen * _kvStride[layerIndex];
            f16Writer.RecordF32ToF16(cmdBuf, kDev, _keys[layerIndex], elemOffset, count);
            f16Writer.RecordF32ToF16(cmdBuf, vDev, _values[layerIndex], elemOffset, count);
            if (maxPos + 1 > _currentLength) _currentLength = maxPos + 1;
            return;
        }

        // Hazard-scoped barriers (issue #144): declare the append's access set
        // (reads the fresh K/V activations, writes the per-layer cache rows)
        // so the tracker emits the RoPE→copy barrier and the later
        // copy→attention barrier only when actually pending. One declaration
        // per buffer pair covers both the contiguous and per-row forms.
        _device.ActiveHazards?.OnTransfer(kDev.Handle, _keys[layerIndex].Handle);
        _device.ActiveHazards?.OnTransfer(vDev.Handle, _values[layerIndex].Handle);

        if (contiguous)
        {
            int startPos = positions[0] - rowBase;
            var region = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = (ulong)((long)startPos * rowBytes),
                size = (ulong)rowBytes * (ulong)seqLen,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
            VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                var region = new VkBufferCopy
                {
                    srcOffset = (ulong)((long)i * rowBytes),
                    dstOffset = (ulong)((long)pos * rowBytes),
                    size = (ulong)rowBytes,
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Validates <paramref name="positions"/> and advances <see cref="CurrentLength"/> as if
    /// <see cref="RecordUpdate"/> had run, WITHOUT recording any copy commands. Used by the
    /// fused RoPE+KV-write path (<c>RopeKvWriteF32Kernel</c>, issue #380), which writes cache
    /// rows directly via its own compute dispatch instead of going through <see cref="RecordUpdate"/>'s
    /// <c>vkCmdCopyBuffer</c> calls, but still needs the same length-tracking side effect.
    /// </summary>
    internal void AdvanceCurrentLength(ReadOnlySpan<int> positions, int seqLen)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        if (_rows is not null)
            throw new InvalidOperationException("The fused RoPE+KV write addresses rows by absolute position; a compact sliding-window cache cannot take it.");

        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    private int ValidateAndFindMaxPos(ReadOnlySpan<int> positions, int seqLen)
    {
        int maxPos = -1;
        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        return maxPos;
    }

    private static bool IsContiguousAscending(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != positions[i - 1] + 1)
                return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetKeysBuffer, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetValuesBuffer, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    /// <remarks>A compact sliding-window cache refuses a cut past its 64-position margin (<see cref="CanTruncateTo"/>).</remarks>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => Truncate(0);

    /// <inheritdoc/>
    public int RowBaseOf(int layerIndex) => _rows?.BaseOf(layerIndex) ?? 0;

    /// <summary>True when some layers keep only their sliding-window rows (issue #1176).</summary>
    public bool HasCompactWindows => _rows is not null;

    /// <summary>
    /// True when the cache can be cut back to <paramref name="length"/>: always for a full cache; for compact sliding-window
    /// layers only while they still hold the rows a query at that position needs.
    /// </summary>
    /// <param name="length">The length to cut back to.</param>
    /// <returns>Whether <see cref="Rollback"/> / <see cref="SetCurrentLength"/> to that length succeeds.</returns>
    public bool CanTruncateTo(int length) => _rows is null || _rows.CanTruncateTo(length);

    /// <summary>
    /// Grows any compact sliding-window slot too small for an append of <paramref name="positions"/> (a prefill chunk longer
    /// than the ring's free half). Must run before the forward starts recording: a grown slot gets new buffer handles, so the
    /// caller drops its cached descriptor sets when this returns true. The kept rows move with a device-side copy.
    /// </summary>
    /// <param name="positions">Positions the next forward appends.</param>
    /// <returns>True when any buffer was replaced.</returns>
    public bool EnsureWindowCapacity(ReadOnlySpan<int> positions)
    {
        if (_rows is null || positions.Length == 0) return false;
        int maxPos = ValidateAndFindMaxPos(positions, positions.Length);
        int minPos = int.MaxValue;
        foreach (int p in positions) minPos = Math.Min(minPos, p);
        bool resized = false;
        for (int l = 0; l < _numLayers; l++)
        {
            if (!_rows.TryPlan(l, minPos, maxPos, _currentLength, out var m) || !m.Resize) continue;
            long rowBytes = (long)_kvStride[l] * _elemBytes;
            _keys[l] = Regrow(_keys[l], m, rowBytes);
            _values[l] = Regrow(_values[l], m, rowBytes);
            _rows.Commit(l, m);
            resized = true;
        }
        return resized;

        VulkanDevice.Buffer Regrow(VulkanDevice.Buffer old, WindowMove m, long rowBytes)
        {
            var grown = _device.AllocateDeviceLocal(m.Capacity * rowBytes);
            if (m.Keep > 0)
                _device.CopyBufferRangeSynchronous(old, grown, (ulong)(m.Shift * rowBytes), 0, (ulong)(m.Keep * rowBytes));
            old.Dispose();
            return grown;
        }
    }

    /// <summary>
    /// Records the compaction of a windowed slot when the append <c>[minPos, maxPos]</c> would overflow its ring: the kept rows
    /// move to row 0 on the device, directly when source and destination do not overlap, else staged through the move scratch
    /// (<c>vkCmdCopyBuffer</c> forbids overlapping regions).
    /// </summary>
    private void RecordWindowMove(nint cmdBuf, int layer, int minPos, int maxPos)
    {
        if (!_rows!.TryPlan(layer, minPos, maxPos, _currentLength, out var m)) return;
        if (m.Resize)
            throw new InvalidOperationException(
                $"Layer {layer}'s sliding-window ring is too small for positions {minPos}..{maxPos}; call {nameof(EnsureWindowCapacity)} before recording.");
        if (m.Shift > 0 && m.Keep > 0)
        {
            ulong rowBytes = (ulong)_kvStride[layer] * (ulong)_elemBytes;
            ulong src = (ulong)m.Shift * rowBytes, bytes = (ulong)m.Keep * rowBytes;
            nint k = _keys[layer].Handle, v = _values[layer].Handle;
            if (m.Shift >= m.Keep)
            {
                CopyRange(cmdBuf, k, k, src, 0, bytes, first: true);
                CopyRange(cmdBuf, v, v, src, 0, bytes, first: false);
            }
            else
            {
                nint s = _moveScratch!.Handle;
                CopyRange(cmdBuf, k, s, src, 0, bytes, first: true);
                CopyRange(cmdBuf, v, s, src, bytes, bytes, first: false);
                LegacyBarrier(cmdBuf);
                CopyRange(cmdBuf, s, k, 0, 0, bytes, first: false);
                CopyRange(cmdBuf, s, v, bytes, 0, bytes, first: false);
            }

            LegacyBarrier(cmdBuf);
        }

        _rows.Commit(layer, m);
    }

    /// <summary>One <c>vkCmdCopyBuffer</c> region, declared to the hazard tracker (or preceded by a blanket barrier when untracked and <paramref name="first"/>).</summary>
    private void CopyRange(nint cmdBuf, nint src, nint dst, ulong srcOffset, ulong dstOffset, ulong size, bool first)
    {
        if (_device.ActiveHazards is { } hz) hz.OnTransfer(src, dst);
        else if (first) Kernels.KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src, dst, 1, region);
    }

    /// <summary>Blanket compute/transfer barrier on the untracked path; the hazard tracker inserts its own.</summary>
    private void LegacyBarrier(nint cmdBuf)
    {
        if (_device.ActiveHazards is null) Kernels.KernelSupport.ComputeTransferFullBarrier(cmdBuf);
    }

    private void Truncate(int length)
    {
        if (_rows is not null)
        {
            if (!_rows.CanTruncateTo(length))
                throw new InvalidOperationException(
                    $"Cannot cut the KV-cache back to {length}: a sliding-window layer already dropped rows that length needs "
                    + $"(margin {SlidingWindowRows.Margin} positions).");
            if (length == 0) _rows.Reset();
        }

        _currentLength = length;
    }

    private void ThrowIfWindowed(int layerIndex, string operation)
    {
        if (_rows is not null && _rows.IsWindowed(layerIndex))
            throw new NotSupportedException(
                $"VulkanKvCache.{operation}: layer {layerIndex} keeps only its sliding-window rows; build the cache with "
                + "KvGeometry.FromConfig(config) for full-history consumers.");
    }

    /// <summary>
    /// Number of layers in this cache.
    /// </summary>
    public int LayerCount => _numLayers;

    /// <summary>
    /// Per-row stride (<c>numKvHeads * headDim</c>, FP32 elements) of layer 0.
    /// For uniform caches this is the model-wide stride; for Gemma-4 (per-layer
    /// strides) use <see cref="KvStrideOf"/> to get a specific layer's width.
    /// </summary>
    public int KvStride => _kvStride[0];

    /// <summary>Per-row stride (FP32 elements) for the given layer.</summary>
    public int KvStrideOf(int layerIndex) => _kvStride[layerIndex];

    /// <summary>
    /// Ingests host-resident K/V rows (FP32, layout <c>[length, kvStride]</c>)
    /// for the given <paramref name="layerIndex"/> at positions <c>[0, length)</c>.
    /// Used by the hybrid CPU-prefill / iGPU-decode handoff: after CPU prefill
    /// has populated a <c>SimpleKvCache</c>, each layer's contiguous host buffer
    /// is uploaded into the device-local Vulkan KV cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Performs one host→staging map+copy plus one
    /// <c>vkCmdCopyBuffer</c> (staging→device) per call. The staging buffer is
    /// allocated and freed inside this method — handoff happens once per
    /// generation, never on the decode hot path. On a UMA APU (Strix Halo)
    /// the physical bytes never leave system DRAM; the driver does swizzle the
    /// layout in the device-local heap.
    /// </para>
    /// <para>
    /// Advances <see cref="CurrentLength"/> to <c>max(CurrentLength, length)</c>
    /// so the subsequent device decode sees positions <c>[0, length)</c> as
    /// already-cached. Both <c>keys</c> and <c>values</c> must cover exactly
    /// <c>length × KvStride</c> FP32 elements.
    /// </para>
    /// </remarks>
    public unsafe void IngestFromHost(int layerIndex, int length,
        ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        RequireF32("IngestFromHost");
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        ThrowIfWindowed(layerIndex, "IngestFromHost");
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "length must be positive.");
        if (length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"length {length} exceeds cache MaxLength {_maxSeqLen}.");
        long expectedFloats = (long)length * _kvStride[layerIndex];
        if (keys.Length != expectedFloats || values.Length != expectedFloats)
            throw new ArgumentException(
                $"keys/values must contain exactly length × kvStride = {expectedFloats} floats; "
                + $"got keys={keys.Length}, values={values.Length}.");

        long bytes = expectedFloats * sizeof(float);

        // Upload keys then values. Use the device's host-visible Allocate as a
        // one-shot staging buffer per call, then synchronous copy → device-local
        // KV destination. The synchronous variant is fine here: handoff is
        // off the per-token hot path.
        using (var stagingK = _device.Allocate(bytes))
        {
            MapAndCopy(stagingK, keys);
            _device.CopyBufferRangeSynchronous(stagingK, _keys[layerIndex],
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        }

        using (var stagingV = _device.Allocate(bytes))
        {
            MapAndCopy(stagingV, values);
            _device.CopyBufferRangeSynchronous(stagingV, _values[layerIndex],
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        }

        if (length > _currentLength)
            _currentLength = length;
    }

    /// <summary>
    /// Sets the visible length without changing buffer contents. Used after
    /// <see cref="IngestFromHost"/> calls for every layer to advance the
    /// observed length atomically across layers (the per-layer call already
    /// advances individually; this is a no-op for single-layer ingest but
    /// makes the multi-layer code path explicit at the call site).
    /// </summary>
    public void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    // ── IHostStagedKvCache: device↔host staging for cross-device KV handoff ──
    // The two halves of a device→host→device transfer: DownloadLayer reads this (prefill-device) cache's
    // K/V to host; the decode-device cache's UploadLayer (== IngestFromHost) writes it back to its device.
    // This is the production transport for a two-GPU prefill→decode handoff and is exercised by the
    // DisaggregatedScheduler's StagedKvHandoffTransfer.

    /// <inheritdoc/>
    public int StagedLayerElementCount(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _currentLength * _kvStride[layerIndex];
    }

    /// <inheritdoc/>
    public void DownloadLayer(int layerIndex, Span<float> keys, Span<float> values)
    {
        RequireF32("DownloadLayer");
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        ThrowIfWindowed(layerIndex, "DownloadLayer");
        int count = _currentLength * _kvStride[layerIndex];
        if (count == 0) return;
        if (keys.Length < count || values.Length < count)
            throw new ArgumentException($"keys/values must hold at least currentLength × kvStride = {count} floats.");
        // Device-local K/V → host. On a dGPU this is a real VRAM read-back; on UMA the bytes never leave DRAM.
        _device.Download(_keys[layerIndex], keys[..count]);
        _device.Download(_values[layerIndex], values[..count]);
    }

    /// <inheritdoc/>
    public void UploadLayer(int layerIndex, int length, ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
        => IngestFromHost(layerIndex, length, keys, values);

    // shortcut: the host-staged F32 transport (handoff, ingest) has no F16 arm yet; add one when an F16 model needs cross-device handoff.
    private void RequireF32(string operation)
    {
        if (_storage != VulkanKvStorage.F32)
            throw new NotSupportedException($"VulkanKvCache.{operation} moves F32 rows; this cache stores {_storage}.");
    }

    private unsafe void MapAndCopy(VulkanDevice.Buffer staging, ReadOnlySpan<float> source)
    {
        // 64-bit: source.Length is an element count, so `length * 4` wraps int at
        // 512 M floats (2 GiB) — the same element-count × element-size overflow
        // fixed on the CPU side in #429. A per-layer KV slab of that size is only
        // out of reach on drivers that cap maxMemoryAllocationSize at 2 GiB.
        long byteLen = (long)source.Length * sizeof(float);
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)byteLen, "vkMapMemory IngestFromHost staging");
        try
        {
            fixed (float* src = source)
            {
                System.Buffer.MemoryCopy(src, (void*)mapped, byteLen, byteLen);
            }
        }
        finally
        {
            Interop.VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
        }

        _moveScratch?.Dispose();
    }
}
