using System.Diagnostics;
using DotLLM.HuggingFace;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Shared helper for downloading GGUF models in test fixtures.
/// Uses a 10-minute HTTP timeout (default 100s is too short for multi-GB models on slow connections).
/// Downloads to <see cref="CacheDirectory"/> — the CLI's model store — and never re-downloads a
/// file that already exists in any store <see cref="TestFixtureResolver"/> knows about (#1279).
/// Shows download progress on stderr so it's visible during <c>dotnet test</c>.
/// </summary>
internal static class TestModelDownloader
{
    /// <summary>Environment variable overriding the test model store root.</summary>
    internal const string CacheDirEnvVar = "DOTLLM_TEST_CACHE_DIR";

    /// <summary>
    /// Root directory that test downloads write to: <c>$DOTLLM_TEST_CACHE_DIR</c> when set (an
    /// override, e.g. for CI), otherwise <see cref="HuggingFaceDownloader.DefaultModelsDirectory"/>
    /// (<c>~/.dotllm/models/</c>, the store <c>dotllm model pull</c> writes). Models are laid out
    /// underneath as <c>{owner}/{repo}/{filename}</c>.
    /// </summary>
    /// <remarks>
    /// Evaluated per access rather than cached in a static initializer so tests can
    /// exercise the override. Note that <c>~</c> is not expanded by .NET; relative
    /// values are resolved against the current directory via <see cref="Path.GetFullPath(string)"/>.
    /// </remarks>
    internal static string CacheDirectory => ResolveCacheDirectory(
        Environment.GetEnvironmentVariable(CacheDirEnvVar));

    /// <summary>
    /// Resolves the store root from a raw environment variable value.
    /// Exposed for testing; <paramref name="envValue"/> null/empty/whitespace selects the default.
    /// </summary>
    /// <param name="envValue">Raw value of <see cref="CacheDirEnvVar"/>.</param>
    /// <returns>Absolute store root.</returns>
    internal static string ResolveCacheDirectory(string? envValue) =>
        string.IsNullOrWhiteSpace(envValue)
            ? HuggingFaceDownloader.DefaultModelsDirectory
            : Path.GetFullPath(envValue);

    /// <summary>
    /// Returns the local model path, downloading from Hugging Face only when no store holds it.
    /// </summary>
    /// <param name="repoId">Hugging Face repo id (<c>owner/repo</c>).</param>
    /// <param name="filename">File name inside the repo.</param>
    /// <returns>Absolute path of the model file.</returns>
    public static async Task<string> EnsureModelAsync(string repoId, string filename)
    {
        // Resolve once so the probe and the download target can't disagree if the
        // environment variable changes mid-run.
        string cacheDir = CacheDirectory;
        string? existing = TryUseExisting(repoId, filename, cacheDir, extraDirectories: null);
        if (existing is not null)
            return existing;

        Console.Error.WriteLine($"[dotLLM] Downloading {filename} from {repoId}...");

        var sw = Stopwatch.StartNew();
        long lastReportedBytes = 0;
        var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(p =>
        {
            // Throttle: report every ~10 MB or when done
            if (p.bytesDownloaded - lastReportedBytes < 10 * 1024 * 1024)
                return;
            lastReportedBytes = p.bytesDownloaded;

            double downloadedMB = p.bytesDownloaded / (1024.0 * 1024.0);
            if (p.totalBytes.HasValue)
            {
                double totalMB = p.totalBytes.Value / (1024.0 * 1024.0);
                int pct = (int)(p.bytesDownloaded * 100 / p.totalBytes.Value);
                Console.Error.WriteLine($"[dotLLM]   {downloadedMB:F0} / {totalMB:F0} MB ({pct}%)");
            }
            else
            {
                Console.Error.WriteLine($"[dotLLM]   {downloadedMB:F0} MB downloaded...");
            }
        });

        using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        using var downloader = new HuggingFaceDownloader(httpClient);
        var result = await downloader.DownloadFileAsync(repoId, filename, cacheDir, progress);

        sw.Stop();
        double finalMB = new FileInfo(result).Length / (1024.0 * 1024.0);
        Console.Error.WriteLine($"[dotLLM]   Done: {finalMB:F0} MB in {sw.Elapsed.TotalSeconds:F1}s");

        return result;
    }

    /// <summary>
    /// Finds <paramref name="filename"/> in any model store without downloading. A hit outside
    /// <paramref name="cacheDir"/> is hardlinked into <c>{cacheDir}/{owner}/{repo}/</c> when both
    /// sit on one volume, and otherwise used in place — never copied.
    /// </summary>
    /// <param name="repoId">Hugging Face repo id (<c>owner/repo</c>).</param>
    /// <param name="filename">File name inside the repo.</param>
    /// <param name="cacheDir">Store root the file should live under.</param>
    /// <param name="extraDirectories">Additional directories to probe (tests use this to stage a
    /// hit without touching the real stores).</param>
    /// <returns>Path to use, or <c>null</c> when no store holds the file.</returns>
    internal static string? TryUseExisting(
        string repoId, string filename, string cacheDir, IReadOnlyList<string>? extraDirectories)
    {
        string target = Path.Combine(cacheDir, repoId.Replace('/', Path.DirectorySeparatorChar), filename);
        if (File.Exists(target))
            return target;

        int slash = repoId.IndexOf('/', StringComparison.Ordinal);
        string org = slash < 0 ? repoId : repoId[..slash];
        string repo = slash < 0 ? string.Empty : repoId[(slash + 1)..];
        FixtureLocation found = TestFixtureResolver.ResolveFile([], org, repo, [filename], extraDirectories);
        return found.Found ? AdoptExisting(found.Path, target) : null;
    }

    /// <summary>
    /// Hardlinks <paramref name="source"/> to <paramref name="target"/>, returning the target, or
    /// returns <paramref name="source"/> unchanged when a hardlink is impossible (cross-volume).
    /// </summary>
    /// <param name="source">Existing model file in some store.</param>
    /// <param name="target">Desired path under the dotLLM model store.</param>
    /// <returns>The path the caller should open.</returns>
    internal static string AdoptExisting(string source, string target)
    {
        try
        {
            // HF snapshot entries are usually symlinks to blobs/: link the blob's bytes, never the
            // reparse point, so the dotLLM path stays readable where following HF links is refused
            // (ERROR_UNTRUSTED_MOUNT_POINT on Strix).
            string bytes = new FileInfo(source).ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? source;
            if (HubCache.TryCreateHardLink(bytes, target))
            {
                Console.Error.WriteLine($"[dotLLM] Hardlinked {bytes} -> {target}");
                return target;
            }
        }
        catch (IOException)
        {
            // fall through: use the file where it is
        }
        catch (UnauthorizedAccessException)
        {
            // fall through: use the file where it is
        }

        return source;
    }
}
