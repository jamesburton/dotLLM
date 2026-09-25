using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

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
    private int _currentLength;
    private bool _disposed;

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
    public VulkanNemotronHKvCache(VulkanDevice device, int[] kvSlotForLayer, int attentionLayerCount,
        int numKvHeads, int headDim, int maxSeqLen)
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

        _keys = new VulkanDevice.Buffer[attentionLayerCount];
        _values = new VulkanDevice.Buffer[attentionLayerCount];

        if (attentionLayerCount == 0) return;

        long bytesPerLayer = (long)maxSeqLen * _kvStride * sizeof(float);
        for (int i = 0; i < attentionLayerCount; i++)
        {
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
        }
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

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>. Caller is
    /// responsible for the <c>TRANSFER → COMPUTE_SHADER</c> barrier before the next attention
    /// dispatch reads from the cache.
    /// </summary>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int absoluteLayerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int slot = RequireSlot(absoluteLayerIndex);
        int rowBytes = _kvStride * sizeof(float);
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

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
