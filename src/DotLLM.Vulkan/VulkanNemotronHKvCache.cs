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
    /// IEEE binary16, written by a convert kernel. Only the Qwen4-Exp QSA gather kernel reads it (the other attention kernels would
    /// misread the bytes), so an F16 cache is only valid on a model whose attention hook handles every layer.
    /// </summary>
    F16 = 1,
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
    private readonly int _maxSeqLen;
    private readonly int _kvStride;
    private readonly VulkanKvStorage _storage;
    private readonly int _elemBytes;
    private readonly bool _lazy;
    private int _allocLength;
    private int _currentLength;
    private bool _disposed;

    /// <summary>Smallest step a lazy cache grows by (positions); growth is otherwise 25% of what is held, rounded up to this.</summary>
    internal static int LazyGrowthQuantum { get; set; } = 1024;

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
    /// </remarks>
    public VulkanNemotronHKvCache(VulkanDevice device, int[] kvSlotForLayer, int attentionLayerCount,
        int numKvHeads, int headDim, int maxSeqLen, VulkanKvStorage storage = VulkanKvStorage.F32, bool lazy = false)
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
        _kvStride = numKvHeads * headDim;
        _storage = storage;
        _elemBytes = storage == VulkanKvStorage.F16 ? 2 : 4;
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
        if (!ReferenceEquals(_device, device) || _storage != VulkanKvStorage.F32 ||
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
        Kernels.Qwen4ExpQsaKernels? f16Writer = null)
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
    /// <paramref name="length"/> (one submission for every layer). Layout is <c>[position][kvStride]</c> F32, so each layer's
    /// prefix is one contiguous byte range.
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
    }
}
