namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Ownership of a store root by its sole writer (integration spec L1; Phase I Task 15): an exclusively opened
/// (<see cref="FileShare.None"/>) <see cref="FileName"/> in the store root, held by the writing server for its whole lifetime
/// and released on dispose or when the process dies. A second server on the same root cannot take it and refuses to start;
/// an offline CLI writer (Task 17) refuses to write while it is held (<see cref="IsHeldByAnother"/>).
/// </summary>
/// <remarks>
/// <para>
/// This is not the store lock (<see cref="ChunkIndex.AcquireStoreLock"/>): that one is short-scoped (one put batch with its
/// ledger save, one eviction pass) and every writer takes it around each write, the owner included. The owner lock only says
/// who the writer is. The file is never deleted: an existing file is not a held lock, only an open handle is.
/// </para>
/// <para>Same platform notes as the store lock (<see cref="ChunkIndex"/> remarks): verified on Windows (NTFS).</para>
/// </remarks>
public sealed class StoreOwnerLock : IDisposable
{
    /// <summary>File name of the owner lock in the store root (not <c>index-*</c>, not <c>.ckv</c>: never read as data).</summary>
    public const string FileName = ".chunkstore.owner.lock";

    private readonly FileStream _stream;
    private int _disposed;

    private StoreOwnerLock(string storeRoot, FileStream stream)
    {
        StoreRoot = storeRoot;
        _stream = stream;
    }

    /// <summary>The store root (absolute, without a trailing separator).</summary>
    public string StoreRoot { get; }

    /// <summary>Whether the lock is still held.</summary>
    public bool IsHeld => Volatile.Read(ref _disposed) == 0;

    /// <summary>
    /// Tries once to take ownership of <paramref name="storeRoot"/> (created if missing). Never waits.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>The held lock, or null when another holder (process, or handle in this process) owns the store.</returns>
    /// <exception cref="IOException">The lock file cannot be created for another reason than a sharing violation.</exception>
    /// <exception cref="UnauthorizedAccessException">The store root is not writable.</exception>
    public static StoreOwnerLock? TryAcquire(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(storeRoot));
        Directory.CreateDirectory(root);
        string path = Path.Combine(root, FileName);
        try
        {
            return new StoreOwnerLock(root, new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None));
        }
        catch (IOException ex) when (IsSharingViolation(ex))
        {
            return null;
        }
    }

    /// <summary>
    /// Whether another holder owns <paramref name="storeRoot"/> right now (probes by opening the lock file exclusively and
    /// closing it at once). A missing store root or lock file is not owned. A probe failure other than a sharing violation is
    /// reported as owned (fail closed: a writer must not assume it is alone).
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>True when owned by another holder (or undeterminable).</returns>
    public static bool IsHeldByAnother(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        string path = Path.Combine(Path.GetFullPath(storeRoot), FileName);
        try
        {
            using var probe = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            return false;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            return false;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return true;
        }
    }

    // ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33) on Windows; EWOULDBLOCK from the flock emulation elsewhere.
    private static bool IsSharingViolation(IOException ex) =>
        ex is not (FileNotFoundException or DirectoryNotFoundException or PathTooLongException)
        && (!OperatingSystem.IsWindows() || (ex.HResult & 0xFFFF) is 32 or 33);

    /// <summary>Releases ownership (idempotent).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _stream.Dispose();
    }
}
