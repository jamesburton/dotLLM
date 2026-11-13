namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// File-system conventions shared by the store's lock files and directories: private modes on POSIX, the real path of a store
/// root, and the one classifier for "another holder has the file".
/// </summary>
/// <remarks>
/// <para>
/// <b>Permissions.</b> A store holds token ids (effectively the source text), K/V rows, and ledger and index files that name
/// source paths (including per-API-key namespaces). On POSIX a directory this code creates is <c>0700</c> and a lock file it
/// creates is <c>0600</c> (the process umask can only tighten them), so another local user can neither read the store nor open a
/// lock file to keep a writer out. An <b>existing</b> directory keeps its mode: an operator who shares a store on purpose
/// (group read) chooses that. On Windows a new directory inherits the ACL of its parent: put the store under a directory that
/// only the service account (and administrators) can read, and do not grant others write access to the lock files.
/// </para>
/// </remarks>
internal static class StorePaths
{
    // Per async flow: a test records which directories were synced (a process-wide hook raced between parallel test classes).
    private static readonly AsyncLocal<Action<string>?> s_directorySyncedForTest = new();

    /// <summary>Test hook: called with the directory each time <see cref="SyncDirectory"/> runs (on every OS).</summary>
    internal static Action<string>? DirectorySyncedForTest
    {
        get => s_directorySyncedForTest.Value;
        set => s_directorySyncedForTest.Value = value;
    }

    /// <summary>
    /// Flushes the directory entry of a file just renamed into <paramref name="directory"/>, so a crash cannot roll the rename back
    /// (a stale ledger or index; the effect is only re-encoding). POSIX only: Windows has no directory fsync and NTFS journals the
    /// rename. Best effort: a failure (a file system that refuses fsync on a directory, no libc) is ignored, as the file itself
    /// was already flushed before the rename.
    /// </summary>
    public static void SyncDirectory(string directory)
    {
        DirectorySyncedForTest?.Invoke(directory);
        if (OperatingSystem.IsWindows()) return;
        try
        {
            int fd = NativeSync.Open(directory, NativeSync.ReadOnlyDirectory);
            if (fd < 0) return;
            try { _ = NativeSync.Fsync(fd); }
            finally { _ = NativeSync.Close(fd); }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException) { }
    }

    private static class NativeSync
    {
        // O_RDONLY | O_DIRECTORY: 0x10000 on Linux x86-64/arm64 (0x100000 on macOS); plain O_RDONLY (0) opens a directory on both,
        // which is all fsync needs, so use that and keep the value portable.
        public const int ReadOnlyDirectory = 0;

#pragma warning disable SYSLIB1054, CA5392 // best-effort libc calls with blittable arguments; no marshalling source generator needed
        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "open", SetLastError = true)]
        public static extern int Open([System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.LPUTF8Str)] string path, int flags);

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "fsync", SetLastError = true)]
        public static extern int Fsync(int fd);

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "close", SetLastError = true)]
        public static extern int Close(int fd);
#pragma warning restore SYSLIB1054, CA5392
    }

    /// <summary>Mode of a directory created by the store (owner only).</summary>
    public const UnixFileMode PrivateDirectoryMode = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute;

    /// <summary>Mode of a lock file created by the store (owner read/write).</summary>
    public const UnixFileMode PrivateFileMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;

    /// <summary>
    /// Creates <paramref name="path"/> owner-only on POSIX; an existing directory is left as it is. Missing PARENTS are created with the
    /// default mode: they are not the store's (a shared parent must stay traversable for the operator's other users).
    /// </summary>
    public static void CreateStoreDirectory(string path)
    {
        if (OperatingSystem.IsWindows() || Directory.Exists(path)) { Directory.CreateDirectory(path); return; }
        string? parent = Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
        if (!string.IsNullOrEmpty(parent)) Directory.CreateDirectory(parent);
        Directory.CreateDirectory(path, PrivateDirectoryMode);
    }

    /// <summary>
    /// Creates a new file (it must not exist) for exclusive writing, owner read/write only on POSIX: the temp files that become
    /// chunk, index, ledger and registry files, so a store under a group-readable root does not leak its contents through them.
    /// </summary>
    public static FileStream CreateNewFile(string path, int bufferSize = 4096)
    {
        var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None, BufferSize = bufferSize };
        if (!OperatingSystem.IsWindows()) options.UnixCreateMode = PrivateFileMode;
        return new FileStream(path, options);
    }

    /// <summary>Opens (creating owner-only on POSIX) <paramref name="path"/> exclusively: no other handle may be open on it.</summary>
    public static FileStream OpenExclusive(string path)
    {
        var options = new FileStreamOptions { Mode = FileMode.OpenOrCreate, Access = FileAccess.ReadWrite, Share = FileShare.None };
        if (!OperatingSystem.IsWindows()) options.UnixCreateMode = PrivateFileMode;
        return new FileStream(path, options);
    }

    /// <summary>
    /// Whether <paramref name="ex"/>, thrown by opening a file with <see cref="FileShare.None"/>, means another handle holds it.
    /// Windows: ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33). Linux: EAGAIN (11), what .NET reports for a refused
    /// <c>flock</c>. macOS / BSD: EWOULDBLOCK (35). Anything else (disk full, an I/O error, a path problem) is not "busy".
    /// </summary>
    public static bool IsSharingViolation(IOException ex)
    {
        if (ex is FileNotFoundException or DirectoryNotFoundException or PathTooLongException) return false;
        if (OperatingSystem.IsWindows()) return (ex.HResult & 0xFFFF) is 32 or 33;
        return ex.HResult == (OperatingSystem.IsMacOS() || OperatingSystem.IsFreeBSD() ? 35 : 11);
    }

    /// <summary>
    /// The absolute path of <paramref name="path"/> with every symbolic link and junction in it resolved (and no trailing separator),
    /// so two spellings of one directory compare equal. Components that do not exist yet are kept as written. Windows 8.3 short
    /// names are expanded by <see cref="Path.GetFullPath(string)"/>.
    /// </summary>
    public static string CanonicalRoot(string path)
    {
        // A ".." after a link means the link target's parent, so dots are applied while walking (never lexically beforehand).
        string absolute = Path.IsPathRooted(path) ? path : Path.Combine(Environment.CurrentDirectory, path);
        return Path.TrimEndingDirectorySeparator(Path.GetFullPath(Resolve(absolute, 0)));
    }

    private static string Resolve(string full, int depth)
    {
        if (depth > 32) return full; // a link cycle: stop, the caller's comparison still works on the spelling
        string root = Path.GetPathRoot(full) ?? "";
        string current = root;
        foreach (string part in full[root.Length..].Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            if (part == ".") continue;
            if (part == "..")
            {
                current = Path.GetDirectoryName(current) is { Length: > 0 } up ? up : root;
                continue;
            }
            current = Path.Combine(current, part);
            string? target;
            try { target = new DirectoryInfo(current).LinkTarget; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { target = null; }
            if (target is not null)
                current = Resolve(Path.IsPathRooted(target) ? target : Path.Combine(Path.GetDirectoryName(current) ?? root, target), depth + 1);
        }
        return current;
    }
}
