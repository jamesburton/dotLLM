namespace DotLLM.HuggingFace;

/// <summary>
/// Locates and writes the canonical Hugging Face hub cache (#454).
/// </summary>
/// <remarks>
/// <para>
/// Downloads performed by the server land in the <b>hub cache</b> — the same
/// <c>blobs</c>/<c>snapshots</c>/<c>refs</c> layout <c>huggingface_hub</c> and <c>hf download</c>
/// use — rather than an ad-hoc directory, so a file fetched here is shared with every other tool
/// on the machine instead of duplicated. A hardlink is then created under
/// <see cref="HuggingFaceDownloader.DefaultModelsDirectory"/> so that
/// <see cref="HuggingFaceDownloader.ListLocalModels"/> and
/// <c>ServerStartup.ResolveModelPath</c> (which both walk a flat
/// <c>{owner}/{repo}/*.gguf</c> tree) still find it. One physical copy on disk, referenced from
/// both places.
/// </para>
/// <para>
/// Layout produced, rooted at <see cref="CacheRoot"/>:
/// <code>
/// models--{owner}--{repo}/
///   refs/{revision}                  -> text file containing the commit sha
///   blobs/{etag}                     -> the actual bytes ({etag}.incomplete while downloading)
///   snapshots/{commit}/{filename}    -> hardlink to the blob
/// </code>
/// </para>
/// </remarks>
public static class HubCache
{
    /// <summary>
    /// Resolves the hub cache root, honouring the standard environment variables in the same
    /// precedence order <c>huggingface_hub</c> uses:
    /// <c>HF_HUB_CACHE</c> → <c>HF_HOME</c>/hub → <c>~/.cache/huggingface/hub</c>.
    /// Never hardcodes a drive — this box, for example, relocates the cache via the env var.
    /// </summary>
    public static string CacheRoot
    {
        get
        {
            var hubCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
            if (!string.IsNullOrWhiteSpace(hubCache)) return hubCache;

            var hfHome = Environment.GetEnvironmentVariable("HF_HOME");
            if (!string.IsNullOrWhiteSpace(hfHome)) return Path.Combine(hfHome, "hub");

            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "huggingface", "hub");
        }
    }

    /// <summary>The <c>models--owner--repo</c> folder name for a repo id.</summary>
    public static string RepoFolderName(string repoId) =>
        "models--" + repoId.Trim('/').Replace("/", "--", StringComparison.Ordinal);

    /// <summary>Absolute path of the repo folder inside <paramref name="cacheRoot"/> (defaults to <see cref="CacheRoot"/>).</summary>
    public static string RepoDirectory(string repoId, string? cacheRoot = null) =>
        Path.Combine(cacheRoot ?? CacheRoot, RepoFolderName(repoId));

    /// <summary>Absolute path of a blob given its etag/hash.</summary>
    public static string BlobPath(string repoId, string etag, string? cacheRoot = null) =>
        Path.Combine(RepoDirectory(repoId, cacheRoot), "blobs", SanitizeEtag(etag));

    /// <summary>Absolute path of the in-progress (resumable) blob for an etag.</summary>
    public static string IncompleteBlobPath(string repoId, string etag, string? cacheRoot = null) =>
        BlobPath(repoId, etag, cacheRoot) + ".incomplete";

    /// <summary>Absolute path of a file inside a snapshot revision.</summary>
    public static string SnapshotFilePath(string repoId, string commit, string filename, string? cacheRoot = null) =>
        Path.Combine(RepoDirectory(repoId, cacheRoot), "snapshots", commit, filename.Replace('/', Path.DirectorySeparatorChar));

    /// <summary>
    /// Mirror path under <see cref="HuggingFaceDownloader.DefaultModelsDirectory"/>
    /// (<c>{models}/{owner}/{repo}/{filename}</c>) that the existing flat-tree model resolver
    /// understands.
    /// </summary>
    public static string MirrorPath(string repoId, string filename, string? modelsDir = null) =>
        Path.Combine(
            modelsDir ?? HuggingFaceDownloader.DefaultModelsDirectory,
            repoId.Replace('/', Path.DirectorySeparatorChar),
            Path.GetFileName(filename));

    /// <summary>
    /// Strips the quotes and weak-validator prefix HTTP <c>ETag</c> values carry, and any
    /// path separators, so the value is safe to use as a file name.
    /// </summary>
    public static string SanitizeEtag(string etag)
    {
        var s = etag.Trim();
        if (s.StartsWith("W/", StringComparison.Ordinal)) s = s[2..];
        s = s.Trim('"');
        foreach (var c in Path.GetInvalidFileNameChars())
            s = s.Replace(c, '_');
        return s.Length == 0 ? "unknown" : s;
    }

    /// <summary>
    /// Links <paramref name="target"/> to <paramref name="source"/>, preferring a hardlink (one
    /// physical copy on disk, no administrator rights needed on the same volume) and falling back
    /// to a file copy only when a hardlink is impossible (different volume, filesystem without
    /// hardlink support). Returns <see langword="true"/> when a hardlink was created.
    /// </summary>
    public static bool LinkOrCopy(string source, string target)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);

        // An identically-sized file already sitting at the target is almost certainly the link
        // we would recreate. Leaving it alone is not just an optimisation: on Windows a model
        // that is currently memory-mapped by a loaded model cannot be deleted, so re-pulling a
        // resident model would otherwise fail with a sharing violation *after* a full download.
        if (File.Exists(target) && new FileInfo(target).Length == new FileInfo(source).Length)
            return true;

        if (File.Exists(target)) File.Delete(target);

        try
        {
            if (NativeLink.TryCreateHardLink(source, target))
                return true;
        }
        catch
        {
            // fall through to the copy below
        }

        File.Copy(source, target, overwrite: true);
        return false;
    }
}

/// <summary>
/// Hardlink creation. The BCL has no cross-platform hardlink API (only
/// <see cref="File.CreateSymbolicLink(string, string)"/>, which needs Developer Mode or elevation
/// on Windows), so the two platform calls are bound directly.
/// </summary>
internal static partial class NativeLink
{
    [System.Runtime.InteropServices.LibraryImport("kernel32.dll", EntryPoint = "CreateHardLinkW",
        StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf16,
        SetLastError = true)]
    [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
    private static partial bool CreateHardLinkW(string lpFileName, string lpExistingFileName, nint lpSecurityAttributes);

    [System.Runtime.InteropServices.LibraryImport("libc", EntryPoint = "link",
        StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf8,
        SetLastError = true)]
    private static partial int link(string oldpath, string newpath);

    /// <summary>
    /// Creates a hardlink at <paramref name="target"/> pointing at <paramref name="source"/>.
    /// Returns <see langword="false"/> (rather than throwing) when the platform refuses —
    /// typically a cross-volume link, which is the expected reason to fall back to a copy.
    /// </summary>
    internal static bool TryCreateHardLink(string source, string target)
    {
        if (OperatingSystem.IsWindows())
            return CreateHardLinkW(target, source, 0);
        return link(source, target) == 0;
    }
}
