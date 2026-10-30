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
    private readonly Func<int, bool>? _canGrowTo;
    private int _currentLength;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength { get; private set; }

    /// <summary>
    /// <see langword="false"/>: the rows live in the owning model's single-slot storage, which a forward on any other handle
    /// overwrites from position 0 (issue #1025).
    /// </summary>
    public bool HasExclusiveStorage => false;

    /// <summary>Creates a handle whose <see cref="MaxLength"/> is <paramref name="maxLength"/>.</summary>
    /// <param name="maxLength">Positions the handle covers.</param>
    /// <param name="canGrowTo">
    /// Supplied by an owning model that keeps its stored rows while its storage grows: it admits or refuses a growth to the
    /// given length (for example when the copy would not fit in free VRAM). Null, the default, means the handle never grows:
    /// a model whose growth discards rows must leave it null.
    /// </param>
    public CudaHybridKvCacheHandle(int maxLength, Func<int, bool>? canGrowTo = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        MaxLength = maxLength;
        _canGrowTo = canGrowTo;
    }

    /// <inheritdoc/>
    /// <remarks>Only bookkeeping here. The model grows its storage, keeping the rows, on the next forward that sees the larger <see cref="MaxLength"/>.</remarks>
    public bool TryGrowTo(int maxLength)
    {
        if (maxLength <= MaxLength) return true;
        if (_canGrowTo is null || !_canGrowTo(maxLength)) return false;
        MaxLength = maxLength;
        return true;
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
