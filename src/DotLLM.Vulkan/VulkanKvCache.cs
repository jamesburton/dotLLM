using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side KV cache. Per-layer device-local buffers of shape
/// <c>[maxSeqLen, numKvHeads * headDim]</c> FP32. The host never touches
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
public sealed class VulkanKvCache : IKvCache, IPerLayerKvCache, IHostStagedKvCache
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
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

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
    /// <see cref="VulkanTransformerModel.CreateKvCache"/> uses (via
    /// <see cref="KvGeometry.FromConfig"/>), so the per-layer-stride derivation lives in one
    /// place (Core) rather than being rebuilt inline.
    /// </summary>
    public VulkanKvCache(VulkanDevice device, KvGeometry geometry, int maxSeqLen)
        : this(device, ExtractStrides(geometry), maxSeqLen)
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
    public VulkanKvCache(VulkanDevice device, int[] kvStridePerLayer, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(kvStridePerLayer);
        if (kvStridePerLayer.Length == 0) throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _device = device;
        _numLayers = kvStridePerLayer.Length;
        _maxSeqLen = maxSeqLen;
        _kvStride = (int[])kvStridePerLayer.Clone();

        _keys = new VulkanDevice.Buffer[_numLayers];
        _values = new VulkanDevice.Buffer[_numLayers];

        for (int i = 0; i < _numLayers; i++)
        {
            if (_kvStride[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer),
                    $"Per-layer KV stride must be positive; layer {i} = {_kvStride[i]}.");
            long bytesPerLayer = (long)maxSeqLen * _kvStride[i] * sizeof(float);
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
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
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int rowBytes = _kvStride[layerIndex] * sizeof(float);
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        // Hazard-scoped barriers (issue #144): declare the append's access set
        // (reads the fresh K/V activations, writes the per-layer cache rows)
        // so the tracker emits the RoPE→copy barrier and the later
        // copy→attention barrier only when actually pending. One declaration
        // per buffer pair covers both the contiguous and per-row forms.
        _device.ActiveHazards?.OnTransfer(kDev.Handle, _keys[layerIndex].Handle);
        _device.ActiveHazards?.OnTransfer(vDev.Handle, _values[layerIndex].Handle);

        if (contiguous)
        {
            int startPos = positions[0];
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
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

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
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
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
        _currentLength = length;
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
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
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
    }
}
