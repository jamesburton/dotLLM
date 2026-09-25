using DotLLM.Core.Attention;

namespace DotLLM.Cuda;

/// <summary>
/// Factory for creating <see cref="CudaPagedKvCache"/> instances backed by a shared
/// <see cref="CudaKvBlockPool"/> (issue #252). Mirrors
/// <see cref="DotLLM.Engine.KvCache.PagedKvCacheFactory"/> — the pool is shared across all
/// sequences on this GPU, enabling efficient VRAM utilization for batch serving.
/// </summary>
public sealed class CudaPagedKvCacheFactory : IDisposable
{
    private readonly CudaKvBlockPool _pool;
    private readonly int _numLayers;

    /// <summary>The underlying block pool shared by all caches created from this factory.</summary>
    public CudaKvBlockPool Pool => _pool;

    /// <summary>Number of transformer layers this factory's pool was sized for.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Total unmanaged device bytes reserved by the shared block pool.</summary>
    public long AllocatedBytes => _pool.AllocatedBytes;

    /// <summary>
    /// Creates a new factory with a shared GPU-resident block pool.
    /// </summary>
    /// <param name="geometry">Per-layer KV row geometry (see <see cref="KvGeometry"/>).</param>
    /// <param name="blockSize">Number of tokens per block (default: 16, matches the CPU pool).</param>
    /// <param name="maxTotalTokens">Maximum total tokens across all sequences
    /// (determines pool size; default: 65536).</param>
    /// <param name="context">Optional CUDA context for explicit multi-device placement; null uses the current context.</param>
    public CudaPagedKvCacheFactory(KvGeometry geometry, int blockSize = 16, int maxTotalTokens = 65536,
                                    CudaContext? context = null)
    {
        if (maxTotalTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTotalTokens));

        _numLayers = geometry.LayerCount;
        int totalBlocks = (maxTotalTokens + blockSize - 1) / blockSize;
        _pool = new CudaKvBlockPool(geometry, blockSize, totalBlocks, context);
    }

    /// <summary>
    /// Creates a new factory with a shared GPU-resident block pool, using a single uniform
    /// <c>numKvHeads * headDim</c> stride. Byte-identical to the <see cref="KvGeometry"/>
    /// constructor with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    public CudaPagedKvCacheFactory(int numLayers, int numKvHeads, int headDim,
                                    int blockSize = 16, int maxTotalTokens = 65536,
                                    CudaContext? context = null)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), blockSize, maxTotalTokens, context)
    {
    }

    /// <summary>
    /// Creates a new GPU-resident paged KV-cache for a single sequence.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for this cache.</param>
    /// <returns>A new <see cref="CudaPagedKvCache"/> backed by the shared pool.</returns>
    public CudaPagedKvCache Create(int maxSeqLen) => new(_pool, maxSeqLen);

    /// <inheritdoc/>
    public void Dispose() => _pool.Dispose();
}
