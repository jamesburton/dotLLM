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
    /// <summary>Mode of a directory created by the store (owner only).</summary>
    public const UnixFileMode PrivateDirectoryMode = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute;

    /// <summary>Mode of a lock file created by the store (owner read/write).</summary>
    public const UnixFileMode PrivateFileMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;

    /// <summary>Creates <paramref name="path"/> (and missing parents) owner-only on POSIX; an existing directory is left as it is.</summary>
    public static void CreateStoreDirectory(string path)
    {
        if (OperatingSystem.IsWindows()) Directory.CreateDirectory(path);
        else Directory.CreateDirectory(path, PrivateDirectoryMode);
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
    public static string CanonicalRoot(string path) => Resolve(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)), 0);

    private static string Resolve(string full, int depth)
    {
        if (depth > 32) return full; // a link cycle: stop, the caller's comparison still works on the spelling
        string root = Path.GetPathRoot(full) ?? "";
        string current = root;
        foreach (string part in full[root.Length..].Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            string? target;
            try { target = new DirectoryInfo(current).LinkTarget; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { target = null; }
            if (target is not null)
                current = Resolve(Path.TrimEndingDirectorySeparator(Path.GetFullPath(target, Path.GetDirectoryName(current) ?? root)), depth + 1);
        }
        return current;
    }
}
