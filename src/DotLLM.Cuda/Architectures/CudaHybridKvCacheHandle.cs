using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Length-only <see cref="IKvCache"/> handle for the CUDA hybrid models
/// (<see cref="CudaQwen3MoeHybridTransformerModel"/>, <see cref="CudaQwen3HybridDenseTransformerModel"/>).
/// Those models own their K/V storage internally (a per-attention-layer F16 device cache —
/// see <c>EnsureF16KvCache</c>/<c>WriteF16KvRows</c>) and only read <see cref="MaxLength"/>
/// from the <see cref="IKvCache"/> passed into <c>Forward</c>; every storage-bearing member
/// of this interface is unreachable from that call path and throws, mirroring the Vulkan
/// hybrid path's <c>VulkanNemotronHKvCache</c>.
/// </summary>
public sealed class CudaHybridKvCacheHandle : IKvCache
{
    private int _currentLength;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength { get; }

    /// <summary>Creates a handle whose <see cref="MaxLength"/> is <paramref name="maxLength"/>.</summary>
    public CudaHybridKvCacheHandle(int maxLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        MaxLength = maxLength;
    }

    /// <summary>Advances the visible length. Called by the owning model as it writes new rows.</summary>
    public void Advance(int newLength)
    {
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException(
            "CudaHybridKvCacheHandle is length-only; storage is owned by the CUDA hybrid model.");

    /// <inheritdoc/>
    public void Dispose()
    {
        // No device storage owned by this handle — nothing to free.
    }
}
