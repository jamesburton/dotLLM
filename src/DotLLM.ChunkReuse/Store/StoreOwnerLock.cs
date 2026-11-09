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
    /// Tries to take ownership of <paramref name="storeRoot"/> (created if missing, owner-only on POSIX). Never waits for a real
    /// owner: it retries only for a short grace (300 ms) so that a probe cannot make it fail. A writer holds the returned lock for
    /// its WHOLE write run (the server for its lifetime, the offline CLI for its build): that is what excludes the server, not a
    /// check followed by writes.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>The held lock, or null when another holder (process, or handle in this process) owns the store.</returns>
    /// <exception cref="IOException">The lock file cannot be created for another reason than a sharing violation.</exception>
    /// <exception cref="UnauthorizedAccessException">The store root is not writable.</exception>
    public static StoreOwnerLock? TryAcquire(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        string spelled = Path.TrimEndingDirectorySeparator(Path.GetFullPath(storeRoot));
        StorePaths.CreateStoreDirectory(spelled);
        string root = StorePaths.CanonicalRoot(spelled);
        string path = Path.Combine(root, FileName);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                return new StoreOwnerLock(root, StorePaths.OpenExclusive(path));
            }
            catch (IOException ex) when (IsSharingViolation(ex))
            {
                // A probe (IsHeldByAnother) holds the file for an instant; a real owner holds it for its lifetime. A short grace
                // keeps a probe from making a starting server refuse, and costs a genuine second server a fraction of a second.
                AfterBusyAttempt?.Invoke();
                if (sw.Elapsed >= BusyGrace) return null;
                Thread.Sleep(10);
            }
        }
    }

    /// <summary>How long <see cref="TryAcquire"/> keeps trying while the file is busy before it reports another owner.</summary>
    private static readonly TimeSpan BusyGrace = TimeSpan.FromMilliseconds(300);

    /// <summary>
    /// Whether another holder owns <paramref name="storeRoot"/> right now (probes by opening the lock file exclusively and
    /// closing it at once). A missing store root or lock file is not owned. A probe failure other than a sharing violation is
    /// reported as owned (fail closed: a writer must not assume it is alone). <b>Advisory only</b>: the answer can be stale the
    /// moment it is returned (check-then-act). A writer must not use it as its guard: take <see cref="TryAcquire"/> and hold it
    /// for the whole write run; use this for diagnostics (a status line, a clearer error message).
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>True when owned by another holder (or undeterminable).</returns>
    public static bool IsHeldByAnother(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        // The same real root as TryAcquire (links and junctions resolved, ".." applied after them), so a spelling through "link/.." probes the
        // store the owner holds and not a lexically collapsed neighbour.
        string path = Path.Combine(StorePaths.CanonicalRoot(storeRoot), FileName);
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

    internal static bool IsSharingViolation(IOException ex) => StorePaths.IsSharingViolation(ex);

    /// <summary>Test seam, per async flow: runs after each busy attempt of <see cref="TryAcquire"/>.</summary>
    internal static Action? AfterBusyAttempt
    {
        get => s_afterBusyAttempt.Value;
        set => s_afterBusyAttempt.Value = value;
    }

    private static readonly AsyncLocal<Action?> s_afterBusyAttempt = new();

    /// <summary>Releases ownership (idempotent).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _stream.Dispose();
    }
}
