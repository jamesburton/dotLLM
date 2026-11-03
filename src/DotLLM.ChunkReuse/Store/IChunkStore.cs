namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Content-addressed store of encoded chunks. A damaged, foreign or mismatched entry is a miss, never a wrong hit.
/// </summary>
/// <remarks>
/// The store does not check how an entry was computed. Callers must put only K/V that match the key: for the default
/// <see cref="ChunkPositioning.Isolated"/> keys, an isolated <see cref="ChunkEncoder"/> encode at positions <c>0..n-1</c> by the
/// model the key's fingerprint names. A recompute output (K/V conditioned on preceding chunks) put under an isolated key would be
/// accepted by every later reader.
/// </remarks>
public interface IChunkStore
{
    /// <summary>Loads the entry for <paramref name="key"/>.</summary>
    /// <param name="key">Chunk key.</param>
    /// <param name="entry">The entry when found and valid.</param>
    /// <returns>Whether a valid entry exists.</returns>
    bool TryGet(ChunkKey key, out ChunkKvEntry entry);

    /// <summary>Stores <paramref name="entry"/> under <paramref name="key"/>; a valid existing entry is kept (same key, same content).</summary>
    /// <param name="key">Chunk key.</param>
    /// <param name="entry">The entry.</param>
    void Put(ChunkKey key, ChunkKvEntry entry);

    /// <summary>Whether a valid entry exists for <paramref name="key"/>.</summary>
    /// <param name="key">Chunk key.</param>
    /// <returns>Whether it exists.</returns>
    bool Contains(ChunkKey key);

    /// <summary>The keys of the stored entries.</summary>
    /// <returns>The keys.</returns>
    IEnumerable<ChunkKey> Keys();

    /// <summary>Total size of the stored entries in bytes.</summary>
    long TotalBytes { get; }

    /// <summary>Removes the entry for <paramref name="key"/>; a missing entry is a no-op.</summary>
    /// <param name="key">Chunk key.</param>
    void Evict(ChunkKey key);
}
