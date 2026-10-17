using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// Key-Value cache for autoregressive attention. Stores projected K and V tensors across decoding steps.
/// </summary>
public interface IKvCache : IDisposable
{
    /// <summary>Current number of cached positions.</summary>
    int CurrentLength { get; }

    /// <summary>Maximum number of positions this cache can hold.</summary>
    int MaxLength { get; }

    /// <summary>
    /// Appends new key and value projections at the given positions.
    /// </summary>
    /// <param name="keys">Key projections for the new tokens.</param>
    /// <param name="values">Value projections for the new tokens.</param>
    /// <param name="positions">Position indices for the new entries.</param>
    /// <param name="layerIndex">Transformer layer index.</param>
    void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex);

    /// <summary>Gets the cached key tensor for a given layer.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <returns>Key tensor covering all cached positions.</returns>
    ITensor GetKeys(int layerIndex);

    /// <summary>Gets the cached value tensor for a given layer.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <returns>Value tensor covering all cached positions.</returns>
    ITensor GetValues(int layerIndex);

    /// <summary>
    /// Zero-allocation update using <see cref="TensorRef"/>. Preferred on the inference hot path.
    /// </summary>
    /// <param name="keys">Key projections as a lightweight tensor reference.</param>
    /// <param name="values">Value projections as a lightweight tensor reference.</param>
    /// <param name="positions">Position indices for the new entries.</param>
    /// <param name="layerIndex">Transformer layer index.</param>
    void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex);

    /// <summary>Gets cached keys as a zero-allocation <see cref="TensorRef"/>.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    TensorRef GetKeysRef(int layerIndex);

    /// <summary>Gets cached values as a zero-allocation <see cref="TensorRef"/>.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    TensorRef GetValuesRef(int layerIndex);

    /// <summary>
    /// Rolls back the cache to the given length, discarding entries beyond that position.
    /// Used by speculative decoding to discard rejected draft tokens.
    /// Allocated memory is retained and overwritten on subsequent Update calls.
    /// </summary>
    /// <param name="length">The new current length (must be &lt;= <see cref="CurrentLength"/>).</param>
    void Rollback(int length);

    /// <summary>
    /// Grows <see cref="MaxLength"/> to at least <paramref name="maxLength"/> and keeps the committed rows. The engine uses it
    /// when a later, longer prompt reuses a cached prefix (issue #1025). The default only reports whether the cache is
    /// already large enough: a cache that cannot grow returns <see langword="false"/> and the caller allocates a new one.
    /// </summary>
    /// <param name="maxLength">Positions the cache must hold.</param>
    /// <returns><see langword="true"/> when <see cref="MaxLength"/> is now at least <paramref name="maxLength"/>.</returns>
    bool TryGrowTo(int maxLength) => maxLength <= MaxLength;

    /// <summary>
    /// <see langword="false"/> when this cache is a view over storage that another cache's forwards overwrite, for example
    /// a model-owned single-slot KV behind a length-only handle. Its rows are then valid only until a forward runs on a
    /// different cache, so a cached prefix held in it must be dropped at that point (issue #1025).
    /// </summary>
    bool HasExclusiveStorage => true;
}
