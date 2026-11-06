namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// A held store lock (<see cref="ChunkIndex.AcquireStoreLock"/>): the exclusive handle on the store root's
/// <see cref="ChunkIndex.StoreLockFileName"/>. It is also the <b>proof</b> that the lock is held: saves that must happen under
/// the store lock (<see cref="StoreRegistry.Save(StoreLockHandle)"/>, <see cref="ServerLedger.Save(StoreLockHandle)"/>) take
/// it and check it is live and for their store root (<see cref="EnsureHeldFor"/>), whatever thread or async flow they run on.
/// </summary>
/// <remarks>
/// Acquire and dispose a handle in the same execution flow, and do not start long-lived work inside its scope: the in-process
/// lock-order check (an index lock is refused while the same flow holds this store's lock) follows the flow, so a handle
/// acquired in a child flow (e.g. returned from <c>Task.Run</c> or an async helper) is not seen by it. Proof by handle is not
/// affected.
/// </remarks>
public sealed class StoreLockHandle : IDisposable
{
    private readonly FileStream _stream;
    private int _disposed;

    internal StoreLockHandle(string storeRoot, FileStream stream)
    {
        StoreRoot = storeRoot;
        _stream = stream;
    }

    /// <summary>The store root (absolute, without a trailing separator) whose lock this is.</summary>
    public string StoreRoot { get; }

    /// <summary>Whether the lock is still held (not disposed).</summary>
    public bool IsHeld => Volatile.Read(ref _disposed) == 0;

    /// <summary>
    /// Throws unless this handle is held and is the lock of <paramref name="storeRoot"/> (compared with
    /// <see cref="ChunkIndex.PathComparer"/> after full-path normalisation).
    /// </summary>
    /// <param name="storeRoot">The store root a write is about to change.</param>
    /// <exception cref="InvalidOperationException">The handle was disposed, or belongs to another store root.</exception>
    public void EnsureHeldFor(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        if (!IsHeld) throw new InvalidOperationException($"The store lock of '{StoreRoot}' was already released; take it again for this write.");
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(storeRoot));
        if (!ChunkIndex.PathComparer.Equals(root, StoreRoot))
            throw new InvalidOperationException($"The store lock held is for '{StoreRoot}', not for '{root}'.");
    }

    /// <summary>Releases the lock (idempotent; safe from any thread).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _stream.Dispose();
    }
}
