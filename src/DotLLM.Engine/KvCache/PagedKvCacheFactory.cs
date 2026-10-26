using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Factory for creating <see cref="PagedKvCache"/> instances backed by a shared <see cref="KvBlockPool"/>.
/// The pool is shared across all sequences, enabling efficient memory utilization for batch serving.
/// </summary>
public sealed class PagedKvCacheFactory : IDisposable
{
    private readonly KvBlockPool _pool;
    private readonly int _numLayers;
    private readonly int _kvStride;

    /// <summary>The underlying block pool shared by all caches created from this factory.</summary>
    public KvBlockPool Pool => _pool;

    /// <summary>Total unmanaged bytes reserved by the shared block pool.</summary>
    public long AllocatedBytes => _pool.AllocatedBytes;

    /// <summary>
    /// Creates a new factory with a shared block pool.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads per layer.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="blockSize">Number of tokens per block (default: 16).</param>
    /// <param name="maxTotalTokens">Maximum total tokens across all sequences
    /// (determines pool size; default: 65536).</param>
    /// <param name="elementType">Block row type: <see cref="DType.Float16"/> when null (the default),
    /// <see cref="DType.Q8_0"/>, or <see cref="DType.Float32"/> for reference runs only.</param>
    public PagedKvCacheFactory(int numLayers, int numKvHeads, int headDim,
                                int blockSize = 16, int maxTotalTokens = 65536, DType? elementType = null)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (maxTotalTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTotalTokens));

        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;

        int totalBlocks = (maxTotalTokens + blockSize - 1) / blockSize;
        _pool = new KvBlockPool(numLayers, numKvHeads, headDim, blockSize, totalBlocks, elementType ?? DType.Float16);
    }

    /// <summary>
    /// Creates a new paged KV-cache for a single sequence.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for this cache.</param>
    /// <returns>A new <see cref="PagedKvCache"/> backed by the shared pool.</returns>
    public PagedKvCache Create(int maxSeqLen) => new(_pool, _numLayers, _kvStride, maxSeqLen);

    /// <inheritdoc/>
    public void Dispose() => _pool.Dispose();
}
