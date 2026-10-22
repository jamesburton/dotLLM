using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>Element format of a <see cref="VulkanNemotronHKvCache"/>'s K/V rows.</summary>
public enum VulkanKvStorage
{
    /// <summary>IEEE binary32 (the default; every attention kernel reads it).</summary>
    F32 = 0,

    /// <summary>
    /// IEEE binary16 (two per 32-bit word), written by a convert kernel. Read in place by the Qwen4-Exp QSA gather and, since issue
    /// #1102, by the <c>_kvf16</c> twins of the split-KV, flash (F32 and coopmat) and naive attention kernels, which the dense hybrid
    /// (Bonsai 2) host selects. Any other kernel would misread the bytes: a model must route every layer to one of those.
    /// </summary>
    F16 = 1,
}

/// <summary>Per-format facts the attention kernels and the KV cache share (issue #1102).</summary>
public static class VulkanKvStorageExtensions
{
    /// <summary>Bytes per cached element: 4 for <see cref="VulkanKvStorage.F32"/>, 2 for <see cref="VulkanKvStorage.F16"/>.</summary>
    /// <param name="storage">The storage format.</param>
    /// <returns>Bytes per element.</returns>
    public static int ElementBytes(this VulkanKvStorage storage) => storage switch
    {
        VulkanKvStorage.F32 => 4,
        VulkanKvStorage.F16 => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(storage)),
    };

    /// <summary>
    /// The SPIR-V name of the attention shader <paramref name="f32ShaderName"/> (no extension) compiled for this KV format: the name
    /// itself for F32, <c>{name}_kvf16</c> for F16 (the wrapper that includes the same body with <c>KV_F16</c> defined).
    /// </summary>
    /// <param name="storage">The storage format.</param>
    /// <param name="f32ShaderName">The F32 shader's base name, e.g. <c>attention_f32_splitkv</c>.</param>
    /// <returns>The base name of the matching shader.</returns>
    public static string ShaderName(this VulkanKvStorage storage, string f32ShaderName) => storage switch
    {
        VulkanKvStorage.F32 => f32ShaderName,
        VulkanKvStorage.F16 => f32ShaderName + "_kvf16",
        _ => throw new ArgumentOutOfRangeException(nameof(storage)),
    };

    /// <summary>
    /// Resolves the <c>--cache-type-k/-v</c> (<c>--kv-quant-k/-v</c>) values for a Vulkan host whose cache has one format for K and V
    /// (issue #1102). A null or empty side is unspecified and follows the other side; both unspecified is F32 (the default).
    /// </summary>
    /// <param name="key">The key-side value as given, or null.</param>
    /// <param name="value">The value-side value as given, or null.</param>
    /// <returns>The storage both sides resolve to.</returns>
    /// <exception cref="NotSupportedException">A value is not f32 / f16, or the two sides disagree.</exception>
    public static VulkanKvStorage ParseOptions(string? key, string? value)
    {
        VulkanKvStorage? k = ParseOne(key), v = ParseOne(value);
        if (k is not null && v is not null && k != v)
            throw new NotSupportedException(
                $"The Vulkan KV cache stores K and V in one format; got K {key} and V {value}. Pass the same value for both.");
        return k ?? v ?? VulkanKvStorage.F32;
    }

    private static VulkanKvStorage? ParseOne(string? raw)
    {
        string? s = raw?.Trim().ToLowerInvariant();
        return s switch
        {
            null or "" => null,
            "f32" or "fp32" => VulkanKvStorage.F32,
            "f16" or "fp16" => VulkanKvStorage.F16,
            _ => throw new NotSupportedException(
                $"KV storage '{raw}' is not available on Vulkan for this model; use f16 (half the KV of the f32 default)."),
        };
    }
}

/// <summary>
/// Sparse KV cache for the Vulkan NemotronH path. Mirrors
/// <see cref="VulkanKvCache"/> but only allocates per-layer K/V buffers for layers whose
/// <c>HybridLayerKind</c> is <c>Attention</c> — SSM and FFN layers don't need a KV cache.
/// </summary>
/// <remarks>
/// The internal physical layout is a dense array indexed by KV-slot in
/// <c>[0, attentionLayerCount)</c>. <c>kvSlotForLayer[absoluteLayerIndex]</c> maps
/// model-level layer indices into slot indices; a value of <c>-1</c> indicates a
/// non-attention layer and the cache rejects requests against that index.
/// </remarks>
public sealed class VulkanNemotronHKvCache : IKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _growthLimit;
    private int _maxSeqLen;
    private readonly int _kvStride;
    private readonly VulkanKvStorage _storage;
    private readonly int _elemBytes;
    private readonly bool _lazy;
    private int _allocLength;
    private int _currentLength;
    private bool _disposed;

    /// <summary>Smallest step a lazy cache grows by (positions); growth is otherwise 25% of what is held, rounded up to this.</summary>
    internal static int LazyGrowthQuantum { get; set; } = 1024;

    private static long s_bufferEpoch;

    /// <summary>
    /// Bumped whenever any cache of this type frees device buffers (a growth re-creating them, or a dispose). Vulkan recycles a
    /// freed buffer handle into later allocations, so a kernel's handle-keyed descriptor cache can then bind a set that names the
    /// freed buffer and silently read zeros. A model that records against these caches drops its descriptor caches when the epoch
    /// it last saw has moved (issue #1100).
    /// </summary>
    internal static long BufferEpoch => Interlocked.Read(ref s_bufferEpoch);

    /// <summary>Element format of the K/V rows.</summary>
    public VulkanKvStorage Storage => _storage;

    /// <summary>Bytes per cached element (4 for F32, 2 for F16).</summary>
    public int ElementBytes => _elemBytes;

    /// <summary>Positions the device buffers can hold right now (equals <see cref="MaxLength"/> unless the cache is lazy).</summary>
    public int AllocatedLength => _allocLength;

    /// <summary>Device bytes currently held by the K/V buffers.</summary>
    public long AllocatedBytes => 2L * _attentionLayerCount * _allocLength * _kvStride * _elemBytes;

    /// <summary>Device bytes per cached position across all layers (K and V).</summary>
    public long BytesPerPosition => 2L * _attentionLayerCount * _kvStride * _elemBytes;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of attention layers (KV slots).</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>
    /// Creates the per-attention-layer K/V buffers. <paramref name="kvSlotForLayer"/> is the
    /// physical-layer-index -> slot mapping (<c>-1</c> for non-attention layers); the
    /// <c>attentionLayerCount</c> is derived as the maximum slot + 1.
    /// </summary>
    /// <remarks>
    /// <c>storage</c>: element format; <see cref="VulkanKvStorage.F16"/> halves the footprint but needs the QSA hook (see the enum).
    /// <c>lazy</c>: allocate nothing up front and grow with <see cref="EnsureCapacity"/> (the caller must call it, before recording, for every
    /// forward and then drop every descriptor cache that may name the old buffers); <see cref="MaxLength"/> stays the logical limit.
    /// <c>growthLimit</c>: the largest <see cref="MaxLength"/> <see cref="TryGrowTo"/> may raise the cache to (the model's context window);
    /// at most <c>maxSeqLen</c>, the default, means the cache never grows.
    /// </remarks>
    public VulkanNemotronHKvCache(VulkanDevice device, int[] kvSlotForLayer, int attentionLayerCount,
        int numKvHeads, int headDim, int maxSeqLen, VulkanKvStorage storage = VulkanKvStorage.F32, bool lazy = false,
        int growthLimit = 0)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(kvSlotForLayer);
        if (attentionLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(attentionLayerCount));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _device = device;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _growthLimit = Math.Max(maxSeqLen, growthLimit);
        _kvStride = numKvHeads * headDim;
        _storage = storage;
        _elemBytes = storage.ElementBytes();
        _lazy = lazy;
        if (storage == VulkanKvStorage.F16 && (_kvStride & 1) != 0)
            throw new ArgumentException("An F16 KV cache needs an even row width.", nameof(numKvHeads));

        _keys = new VulkanDevice.Buffer[attentionLayerCount];
        _values = new VulkanDevice.Buffer[attentionLayerCount];

        if (attentionLayerCount == 0) { _allocLength = maxSeqLen; return; }
        if (lazy) return;

        long bytesPerLayer = (long)maxSeqLen * _kvStride * _elemBytes;
        for (int i = 0; i < attentionLayerCount; i++)
        {
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
        }
        _allocLength = maxSeqLen;
    }

    /// <summary>
    /// Lazy caches only: makes the device buffers hold at least <paramref name="positions"/> positions, preserving the first
    /// <see cref="CurrentLength"/> rows. Returns true when buffers were re-created - the caller MUST then drop every descriptor cache that may
    /// name the old handles (freed handles are recycled into the new buffers; a stale cached set silently reads zeros). Call BEFORE recording.
    /// Growth is 25% (at least <see cref="LazyGrowthQuantum"/>) so the transient cost is one layer's K and V, copied layer by layer.
    /// </summary>
    internal bool EnsureCapacity(int positions)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (positions > _maxSeqLen) throw new ArgumentOutOfRangeException(nameof(positions), $"{positions} exceeds the cache limit {_maxSeqLen}.");
        if (positions <= _allocLength) return false;
        int target = Math.Max(positions, _allocLength + Math.Max(LazyGrowthQuantum, _allocLength / 4));
        target = (int)Math.Min(_maxSeqLen, ((long)target + LazyGrowthQuantum - 1) / LazyGrowthQuantum * LazyGrowthQuantum);
        GrowTo(target);
        return true;
    }

    /// <summary>
    /// Raises <see cref="MaxLength"/> to <paramref name="maxLength"/>, up to the growth limit the cache was created with (issue #1100), so a
    /// recurrent-prefix hit keeps its rows instead of re-prefilling. A lazy cache only raises the limit: the forward grows its buffers before
    /// recording. An eager cache re-creates its buffers now, copying the live rows layer by layer; the caller's descriptor caches then go stale
    /// (see <see cref="BufferEpoch"/>). Returns <see langword="false"/> above the limit or when the device cannot hold the larger buffers,
    /// leaving the cache as it was.
    /// </summary>
    /// <param name="maxLength">Positions the cache must hold.</param>
    /// <returns><see langword="true"/> when <see cref="MaxLength"/> is now at least <paramref name="maxLength"/>.</returns>
    public bool TryGrowTo(int maxLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxLength <= _maxSeqLen) return true;
        if (maxLength > _growthLimit) return false;
        if (!_lazy && _attentionLayerCount > 0)
        {
            try { GrowTo(maxLength); }
            catch (VulkanException e) when (e.ErrorCode is VulkanOutOfDeviceMemory or VulkanOutOfHostMemory)
            {
                return false;
            }
        }
        else if (_attentionLayerCount == 0)
        {
            _allocLength = maxLength;
        }

        _maxSeqLen = maxLength;
        return true;
    }

    /// <summary>
    /// Call before recording a forward (issue #1100): grows a lazy <paramref name="kvCache"/> to hold <paramref name="positionsEnd"/>
    /// positions, mapping a device allocation failure to <see cref="Core.Configuration.DeviceMemoryAdmissionException"/>. Returns
    /// <see langword="true"/> when any cache of this type freed device buffers since <paramref name="seenEpoch"/> (this growth, another
    /// cache's, a restore or a dispose); the caller must then drop its kernels' descriptor caches, because a recycled handle would bind a
    /// set naming the freed buffer.
    /// </summary>
    /// <param name="kvCache">The forward's cache; anything but a <see cref="VulkanNemotronHKvCache"/> is only checked for the epoch.</param>
    /// <param name="positionsEnd">One past the largest position the forward writes.</param>
    /// <param name="seenEpoch">The caller's last seen <see cref="BufferEpoch"/>; updated.</param>
    /// <returns><see langword="true"/> when descriptor caches must be dropped.</returns>
    internal static bool PrepareForForward(IKvCache? kvCache, int positionsEnd, ref long seenEpoch)
    {
        if (kvCache is VulkanNemotronHKvCache vk)
        {
            try { vk.EnsureCapacity(positionsEnd); }
            catch (VulkanException e) when (e.ErrorCode is VulkanOutOfDeviceMemory or VulkanOutOfHostMemory)
            {
                throw new Core.Configuration.DeviceMemoryAdmissionException(
                    $"Out of device memory growing the KV cache to {positionsEnd} positions " +
                    $"({vk.BytesPerPosition * (long)positionsEnd / (1024 * 1024)} MiB). Shorten the conversation or lower --ctx-size. " +
                    $"({e.Message})");
            }
        }

        long epoch = BufferEpoch;
        if (epoch == seenEpoch) return false;
        seenEpoch = epoch;
        return true;
    }

    /// <summary>Largest value in <paramref name="positions"/> plus one: the KV length a forward over them needs.</summary>
    /// <param name="positions">The forward's positions (non-empty; need not be ascending).</param>
    /// <returns>One past the largest position.</returns>
    internal static int PositionsEnd(ReadOnlySpan<int> positions)
    {
        int max = positions[0];
        for (int i = 1; i < positions.Length; i++) if (positions[i] > max) max = positions[i];
        return max + 1;
    }

    /// <summary><c>VK_ERROR_OUT_OF_DEVICE_MEMORY</c>.</summary>
    internal const int VulkanOutOfDeviceMemory = -2;

    /// <summary><c>VK_ERROR_OUT_OF_HOST_MEMORY</c>.</summary>
    internal const int VulkanOutOfHostMemory = -1;

    private void GrowTo(int newLength)
    {
        long newBytes = (long)newLength * _kvStride * _elemBytes;
        ulong keep = (ulong)_currentLength * (ulong)_kvStride * (ulong)_elemBytes;
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            var nk = _device.AllocateDeviceLocal(newBytes);
            VulkanDevice.Buffer nv;
            try { nv = _device.AllocateDeviceLocal(newBytes); }
            catch { nk.Dispose(); throw; }
            if (keep > 0 && _keys[i] is not null)
            {
                using var ctx = _device.CreateSubmitContext();
                ctx.Begin();
                nint cmdBuf = ctx.CommandBuffer;
                Kernels.KernelSupport.ComputeToTransferBarrier(cmdBuf);
                var region = new VkBufferCopy { srcOffset = 0, dstOffset = 0, size = keep };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, _keys[i].Handle, nk.Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, _values[i].Handle, nv.Handle, 1, region);
                Kernels.KernelSupport.TransferToComputeBarrier(cmdBuf);
                ctx.SubmitAndWait();
            }
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
            _keys[i] = nk;
            _values[i] = nv;

            // Per layer, so a later layer's failed allocation still leaves the epoch past the handles freed so far.
            Interlocked.Increment(ref s_bufferEpoch);
        }
        _allocLength = newLength;
    }

    private int RequireSlot(int absoluteLayerIndex)
    {
        if ((uint)absoluteLayerIndex >= (uint)_kvSlotForLayer.Length)
            throw new ArgumentOutOfRangeException(nameof(absoluteLayerIndex));
        int slot = _kvSlotForLayer[absoluteLayerIndex];
        if (slot < 0)
            throw new InvalidOperationException(
                $"Layer {absoluteLayerIndex} is not an attention layer; no KV-cache slot exists.");
        return slot;
    }

    /// <summary>Returns the device buffer holding cached keys for the given (absolute) layer index.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int absoluteLayerIndex) => _keys[RequireSlot(absoluteLayerIndex)];

    /// <summary>Returns the device buffer holding cached values for the given (absolute) layer index.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int absoluteLayerIndex) => _values[RequireSlot(absoluteLayerIndex)];

    internal void ValidateSnapshotLayout(VulkanDevice device, ReadOnlySpan<HybridLayerKind> layerKinds, int numKvHeads, int headDim)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // Any storage: a snapshot copies whole rows byte for byte, so F16 rows (issue #1102) round-trip exactly.
        if (!ReferenceEquals(_device, device) ||
            _numKvHeads != numKvHeads || _headDim != headDim || _kvSlotForLayer.Length != layerKinds.Length)
            throw new ArgumentException("KV snapshot device, storage or geometry mismatch.");
        int slot = 0;
        for (int i = 0; i < layerKinds.Length; i++)
        {
            int expected = layerKinds[i] == HybridLayerKind.Attention ? slot++ : -1;
            if (_kvSlotForLayer[i] != expected)
                throw new ArgumentException("KV snapshot layer mapping mismatch.");
        }
        if (slot != _attentionLayerCount)
            throw new ArgumentException("KV snapshot attention layer count mismatch.");
    }

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>. Caller is
    /// responsible for the <c>TRANSFER → COMPUTE_SHADER</c> barrier before the next attention
    /// dispatch reads from the cache.
    /// </summary>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int absoluteLayerIndex,
        Kernels.IVulkanKvF16Writer? f16Writer = null)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int slot = RequireSlot(absoluteLayerIndex);
        int rowBytes = _kvStride * _elemBytes;
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        if (_storage == VulkanKvStorage.F16)
        {
            // The F32 rows the projections produced are converted on the device; there is no copy-engine path for F16.
            if (!contiguous || f16Writer is null)
                throw new InvalidOperationException("An F16 KV cache is written by the convert kernel from contiguous positions only.");
            long elemOffset = (long)positions[0] * _kvStride, count = (long)seqLen * _kvStride;
            f16Writer.RecordF32ToF16(cmdBuf, kDev, _keys[slot], elemOffset, count);
            f16Writer.RecordF32ToF16(cmdBuf, vDev, _values[slot], elemOffset, count);
            if (maxPos + 1 > _currentLength) _currentLength = maxPos + 1;
            return;
        }

        if (contiguous)
        {
            int startPos = positions[0];
            var region = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = (ulong)((long)startPos * rowBytes),
                size = (ulong)rowBytes * (ulong)seqLen,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[slot].Handle, 1, region);
            VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[slot].Handle, 1, region);
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
                VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[slot].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[slot].Handle, 1, region);
            }
        }

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
            if ((uint)pos >= (uint)_allocLength)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds the {(_lazy ? "allocated" : "max")} cache length {_allocLength} (limit {_maxSeqLen}).");
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
            "VulkanNemotronHKvCache is updated via RecordUpdate from the Vulkan forward pass.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanNemotronHKvCache is updated via RecordUpdate from the Vulkan forward pass.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache exposes device buffers, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache exposes device buffers, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>
    /// Device-to-device copy of the first <paramref name="length"/> positions into a new, independent cache sized exactly
    /// <paramref name="length"/> (one submission for every layer). Layout is <c>[position][kvStride]</c> in the cache's element format
    /// (F32 or F16), so each layer's prefix is one contiguous byte range and the copy keeps the format.
    /// </summary>
    internal VulkanNemotronHKvCache SnapshotPrefix(int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)length > (uint)_currentLength || length == 0) throw new ArgumentOutOfRangeException(nameof(length));
        var copy = new VulkanNemotronHKvCache(_device, _kvSlotForLayer, _attentionLayerCount, _numKvHeads, _headDim, length, _storage);
        copy.CopyPrefixFrom(this, length);
        copy._currentLength = length;
        return copy;
    }

    /// <summary>
    /// Overwrites this cache's first <paramref name="length"/> positions with <paramref name="source"/>'s and sets the visible
    /// length to <paramref name="length"/> (the restore half of <see cref="SnapshotPrefix"/>).
    /// </summary>
    internal void RestorePrefixFrom(VulkanNemotronHKvCache source, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (length > _maxSeqLen) throw new ArgumentOutOfRangeException(nameof(length));
        if (source._storage != _storage) throw new ArgumentException("KV storage mismatch.", nameof(source));
        EnsureCapacity(length);   // lazy destinations only grow here; the caller drops its descriptor caches when it restores into one
        CopyPrefixFrom(source, length);
        _currentLength = length;
    }

    private void CopyPrefixFrom(VulkanNemotronHKvCache source, int length)
    {
        if (source._attentionLayerCount != _attentionLayerCount || source._kvStride != _kvStride || source._elemBytes != _elemBytes)
            throw new ArgumentException("KV cache geometry mismatch.", nameof(source));
        if (_attentionLayerCount == 0 || length == 0) return;
        ulong bytes = (ulong)length * (ulong)_kvStride * (ulong)_elemBytes;
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        nint cmdBuf = ctx.CommandBuffer;
        Kernels.KernelSupport.ComputeToTransferBarrier(cmdBuf);
        var region = new Interop.VkBufferCopy { srcOffset = 0, dstOffset = 0, size = bytes };
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            Interop.VulkanApi.vkCmdCopyBuffer(cmdBuf, source._keys[i].Handle, _keys[i].Handle, 1, region);
            Interop.VulkanApi.vkCmdCopyBuffer(cmdBuf, source._values[i].Handle, _values[i].Handle, 1, region);
        }
        Kernels.KernelSupport.TransferToComputeBarrier(cmdBuf);
        ctx.SubmitAndWait();
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
        }
        Interlocked.Increment(ref s_bufferEpoch);
    }
}
