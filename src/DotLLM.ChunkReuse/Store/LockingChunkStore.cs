namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// An <see cref="IChunkStore"/> whose writes take the store lock (<see cref="ChunkIndex.AcquireStoreLock"/>) for the length
/// of one write: each <see cref="Put"/> and <see cref="Evict"/> is one short put batch (integration spec L3). Reads pass
/// through unlocked (the store's files are replaced by rename, so a reader never sees a partial entry). Use it for a writer
/// that encodes between puts (the CLI's offline build), so the lock is never held across an encode; a writer that must
/// group several writes with a ledger save (the server) takes the lock itself around the group and uses the inner store.
/// </summary>
/// <remarks>
/// Do not use it inside a scope that already holds the store lock in the same flow: the store lock is not re-entrant, so the
/// write would wait for itself. It is detected at once and throws <see cref="InvalidOperationException"/> (nothing is written),
/// instead of waiting out the lock timeout; use the inner store inside such a scope. A lock timeout is not swallowed by
/// <see cref="StoredContextAssembler"/> (unlike a busy entry): it propagates, so a writer stops after one timeout instead of
/// waiting it once per chunk.
/// </remarks>
/// <param name="inner">The store to write to (typically a <see cref="FileChunkStore"/> at <paramref name="storeRoot"/>).</param>
/// <param name="storeRoot">The store directory whose lock is taken.</param>
/// <param name="lockTimeout">How long each write waits for the lock.</param>
public sealed class LockingChunkStore(IChunkStore inner, string storeRoot, TimeSpan lockTimeout) : IChunkStore
{
    private readonly IChunkStore _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    private readonly string _root = string.IsNullOrEmpty(storeRoot) ? throw new ArgumentException("The store root is required.", nameof(storeRoot)) : storeRoot;
    private readonly TimeSpan _timeout = lockTimeout >= TimeSpan.Zero ? lockTimeout : throw new ArgumentOutOfRangeException(nameof(lockTimeout));

    /// <inheritdoc/>
    public bool TryGet(ChunkKey key, out ChunkKvEntry entry) => _inner.TryGet(key, out entry);

    /// <inheritdoc/>
    /// <exception cref="ChunkStoreLockTimeoutException">The store lock was not obtained within the timeout; nothing was written.</exception>
    /// <exception cref="InvalidOperationException">The calling flow already holds this store's lock.</exception>
    public void Put(ChunkKey key, ChunkKvEntry entry)
    {
        ThrowIfFlowHoldsTheLock();
        using IDisposable held = ChunkIndex.AcquireStoreLock(_root, _timeout);
        _inner.Put(key, entry);
    }

    private void ThrowIfFlowHoldsTheLock()
    {
        if (ChunkIndex.FlowHoldsStoreLock(_root))
            throw new InvalidOperationException($"The store lock of '{_root}' is already held by this flow: the store lock is not re-entrant, "
                + "so a LockingChunkStore here would wait for itself. Write through the inner store inside the lock scope.");
    }

    /// <inheritdoc/>
    public bool Contains(ChunkKey key) => _inner.Contains(key);

    /// <inheritdoc/>
    public IEnumerable<ChunkKey> Keys() => _inner.Keys();

    /// <inheritdoc/>
    public long TotalBytes => _inner.TotalBytes;

    /// <inheritdoc/>
    /// <exception cref="ChunkStoreLockTimeoutException">The store lock was not obtained within the timeout; nothing was deleted.</exception>
    /// <exception cref="InvalidOperationException">The calling flow already holds this store's lock.</exception>
    public void Evict(ChunkKey key)
    {
        ThrowIfFlowHoldsTheLock();
        using IDisposable held = ChunkIndex.AcquireStoreLock(_root, _timeout);
        _inner.Evict(key);
    }
}
