namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// The store is busy: <see cref="FileChunkStore.Put"/> could not replace an invalid file at the entry's path because another
/// handle holds it (on Windows a mapped file cannot be replaced; nothing was stored, the key stays a miss and a later Put can
/// repair it; on Linux and macOS the rename replaces a held file, so this case does not arise there), or, as <see cref="ChunkStoreLockTimeoutException"/>, a store or index lock was held by another writer for the
/// whole timeout. Either way nothing was written; retrying later is safe.
/// </summary>
/// <remarks>
/// A busy <i>entry</i> is transient and local: <see cref="StoredContextAssembler"/> serves the encoded K/V and leaves the key a
/// miss. A <see cref="ChunkStoreLockTimeoutException"/> is not tolerated that way (another writer kept the store lock past the
/// timeout): the assembler lets it propagate, and callers stop and report "the store is unavailable" (CLI <c>store_busy</c>,
/// server <c>chunk_store_unavailable</c>). Catch this base type to report either.
/// </remarks>
public class ChunkStoreBusyException : IOException
{
    /// <summary>Creates the exception for an entry that could not be replaced.</summary>
    /// <param name="path">The entry path that could not be replaced.</param>
    /// <param name="inner">The rename failure.</param>
    public ChunkStoreBusyException(string path, Exception inner)
        : base($"Chunk store entry '{path}' is invalid and held open by another reader; it was not replaced.", inner)
        => Path = path;

    /// <summary>Creates the exception with a custom message (derived types).</summary>
    /// <param name="path">The file that was busy.</param>
    /// <param name="message">The message.</param>
    /// <param name="inner">The underlying failure, or null.</param>
    protected ChunkStoreBusyException(string path, string message, Exception? inner)
        : base(message, inner)
        => Path = path;

    /// <summary>The busy file: the entry path that could not be replaced, or the lock file that could not be taken.</summary>
    public string Path { get; }
}

/// <summary>
/// A store lock (<see cref="ChunkIndex.AcquireStoreLock"/>) or an index lock (<see cref="ChunkIndex.AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>)
/// was held by another writer (another process, or another handle in this one) for the whole timeout. Nothing was written.
/// </summary>
public sealed class ChunkStoreLockTimeoutException : ChunkStoreBusyException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="lockPath">The lock file.</param>
    /// <param name="timeout">How long the caller waited.</param>
    /// <param name="inner">The last open failure (sharing violation), or null.</param>
    public ChunkStoreLockTimeoutException(string lockPath, TimeSpan timeout, Exception? inner)
        : base(lockPath, $"The chunk store lock '{lockPath}' is held by another writer (waited {timeout.TotalSeconds:0.###} s).", inner)
        => Timeout = timeout;

    /// <summary>How long the caller waited.</summary>
    public TimeSpan Timeout { get; }
}
