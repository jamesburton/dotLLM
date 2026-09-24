namespace DotLLM.HuggingFace;

/// <summary>
/// Downloads files from HuggingFace Hub with progress reporting and resume support via HTTP Range headers.
/// </summary>
public sealed class HuggingFaceDownloader : IDisposable
{
    private const string DefaultCdnBase = "https://huggingface.co";

    /// <summary>Default local model storage directory: <c>~/.dotllm/models/</c>.</summary>
    public static string DefaultModelsDirectory =>
        Environment.GetEnvironmentVariable("DOTLLM_MODELS_DIR")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "models");

    private readonly HttpClient _httpClient;
    private readonly bool _ownsClient;
    private readonly string _cdnBase;

    /// <summary>
    /// Creates a new downloader.
    /// </summary>
    /// <param name="httpClient">Optional pre-configured <see cref="HttpClient"/>.</param>
    /// <param name="token">Optional HuggingFace token. Falls back to <c>HF_TOKEN</c> env var.</param>
    /// <param name="cdnBase">
    /// Base URL to resolve repo files against. Defaults to <c>https://huggingface.co</c>; tests
    /// point it at a local stub so no test ever reaches the real Hub.
    /// </param>
    public HuggingFaceDownloader(HttpClient? httpClient = null, string? token = null, string? cdnBase = null)
    {
        _cdnBase = (cdnBase ?? DefaultCdnBase).TrimEnd('/');
        if (httpClient is not null)
        {
            _httpClient = httpClient;
            _ownsClient = false;
        }
        else
        {
            _httpClient = new HttpClient();
            _ownsClient = true;
        }

        token ??= Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            _httpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("dotLLM/0.1");
    }

    /// <summary>
    /// Downloads a file from a HuggingFace repository to a local directory.
    /// Supports resuming interrupted downloads via HTTP Range headers.
    /// </summary>
    /// <param name="repoId">Repository ID, e.g. "TheBloke/Llama-2-7B-GGUF".</param>
    /// <param name="filename">Filename within the repo, e.g. "llama-2-7b.Q4_K_M.gguf".</param>
    /// <param name="destinationDir">Target directory. Defaults to <see cref="DefaultModelsDirectory"/>.</param>
    /// <param name="progress">Optional progress callback: (bytesDownloaded, totalBytes). totalBytes may be null if unknown.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Full path to the downloaded file.</returns>
    public async Task<string> DownloadFileAsync(
        string repoId,
        string filename,
        string? destinationDir = null,
        IProgress<(long bytesDownloaded, long? totalBytes)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        destinationDir ??= DefaultModelsDirectory;

        // Organize by repo: ~/.dotllm/models/{owner}/{repo}/
        var repoDir = Path.Combine(destinationDir, repoId.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(repoDir);

        var destPath = Path.Combine(repoDir, filename);
        var partPath = destPath + ".part";

        var url = $"{DefaultCdnBase}/{repoId}/resolve/main/{filename}";

        long existingBytes = 0;
        if (File.Exists(partPath))
            existingBytes = new FileInfo(partPath).Length;

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (existingBytes > 0)
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingBytes, null);

        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);

        // If server doesn't support range or file is complete, start from scratch
        if (existingBytes > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent)
            existingBytes = 0;

        response.EnsureSuccessStatusCode();

        long? totalBytes = response.Content.Headers.ContentLength.HasValue
            ? response.Content.Headers.ContentLength.Value + existingBytes
            : null;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var fileStream = new FileStream(
            partPath,
            existingBytes > 0 ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 81920);

        var buffer = new byte[81920];
        long totalRead = existingBytes;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken).ConfigureAwait(false);
            totalRead += bytesRead;
            progress?.Report((totalRead, totalBytes));
        }

        // Rename .part to final path once download is complete
        fileStream.Close();
        if (File.Exists(destPath))
            File.Delete(destPath);
        File.Move(partPath, destPath);

        return destPath;
    }

    /// <summary>
    /// Result of <see cref="DownloadToHubCacheAsync"/>: where the bytes physically live and the
    /// two paths that reference them.
    /// </summary>
    /// <param name="BlobPath">The single physical copy, inside the hub cache's <c>blobs</c> folder.</param>
    /// <param name="SnapshotPath">Hub-cache snapshot entry (hardlink to the blob).</param>
    /// <param name="ModelPath">
    /// Mirror path under <see cref="DefaultModelsDirectory"/> (hardlink to the blob) so the
    /// server's flat-tree model resolver finds it.
    /// </param>
    /// <param name="SizeBytes">Final file size.</param>
    /// <param name="Hardlinked">False when a hardlink was impossible and the file had to be copied.</param>
    public readonly record struct HubCacheDownloadResult(
        string BlobPath, string SnapshotPath, string ModelPath, long SizeBytes, bool Hardlinked);

    /// <summary>
    /// Downloads a repo file into the <b>Hugging Face hub cache</b> (#454) — the canonical
    /// <c>blobs</c>/<c>snapshots</c>/<c>refs</c> layout shared with <c>huggingface_hub</c> —
    /// and hardlinks it into <see cref="DefaultModelsDirectory"/> so the existing flat-tree
    /// resolver still finds it. Resumable (an interrupted download leaves
    /// <c>blobs/{etag}.incomplete</c>, which a later call continues via a Range request) and
    /// cancellable.
    /// </summary>
    /// <remarks>
    /// Deliberately a <b>new</b> method rather than a change to
    /// <see cref="DownloadFileAsync"/>: the CLI <c>pull</c> command depends on that method's
    /// existing <c>~/.dotllm/models</c> behaviour.
    /// </remarks>
    /// <param name="repoId">Repository ID, e.g. "bartowski/Qwen2.5-3B-GGUF".</param>
    /// <param name="filename">Filename within the repo.</param>
    /// <param name="revision">Git revision to resolve. Defaults to <c>main</c>.</param>
    /// <param name="cacheRoot">Hub cache root. Defaults to <see cref="HubCache.CacheRoot"/>.</param>
    /// <param name="modelsDir">Mirror root. Defaults to <see cref="DefaultModelsDirectory"/>.</param>
    /// <param name="progress">Progress callback: (bytesDownloaded, totalBytes).</param>
    /// <param name="cancellationToken">Cancels the transfer, leaving the <c>.incomplete</c> file for resume.</param>
    public async Task<HubCacheDownloadResult> DownloadToHubCacheAsync(
        string repoId,
        string filename,
        string? revision = null,
        string? cacheRoot = null,
        string? modelsDir = null,
        IProgress<(long bytesDownloaded, long? totalBytes)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        revision ??= "main";
        cacheRoot ??= HubCache.CacheRoot;

        var url = $"{_cdnBase}/{repoId}/resolve/{revision}/{filename}";

        // A HEAD gives the blob identity (etag) and the resolved commit without transferring
        // bytes, so the resume file can be named before the first byte arrives.
        //
        // It MUST NOT follow redirects. Every GGUF on the Hub is an LFS/Xet object, so
        // /resolve/ answers 302 -> a CDN host, and X-Linked-ETag (the sha256 huggingface_hub
        // names the blob by), X-Repo-Commit and X-Linked-Size ride on that 302 only — the CDN's
        // own response carries a different, unrelated ETag and no commit. Following the redirect
        // here would name the blob by the CDN's ETag (a second physical copy alongside whatever
        // `hf download` already wrote, defeating the point of using the hub cache at all) and
        // write snapshots/main/ instead of snapshots/{sha}/. Verified against the live Hub with a
        // HEAD request, not assumed.
        string etag;
        string commit;
        long? headLength;
        using (var head = new HttpRequestMessage(HttpMethod.Head, url))
        using (var headResponse = await MetadataClient.SendAsync(head, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false))
        {
            if (headResponse.StatusCode is not System.Net.HttpStatusCode.Found
                and not System.Net.HttpStatusCode.MovedPermanently
                and not System.Net.HttpStatusCode.TemporaryRedirect
                and not System.Net.HttpStatusCode.PermanentRedirect)
            {
                headResponse.EnsureSuccessStatusCode();
            }

            etag = ReadEtag(headResponse);
            commit = ReadHeader(headResponse, "X-Repo-Commit") ?? revision;
            headLength = headResponse.Content.Headers.ContentLength
                ?? (long.TryParse(ReadHeader(headResponse, "X-Linked-Size"), out var linked) ? linked : null);
        }

        var blobPath = HubCache.BlobPath(repoId, etag, cacheRoot);
        var partPath = HubCache.IncompleteBlobPath(repoId, etag, cacheRoot);
        var snapshotPath = HubCache.SnapshotFilePath(repoId, commit, filename, cacheRoot);
        var mirrorPath = HubCache.MirrorPath(repoId, filename, modelsDir);
        Directory.CreateDirectory(Path.GetDirectoryName(blobPath)!);

        if (!File.Exists(blobPath))
        {
            long existingBytes = File.Exists(partPath) ? new FileInfo(partPath).Length : 0;

            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            if (existingBytes > 0)
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingBytes, null);

            using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            if (existingBytes > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent)
                existingBytes = 0; // server ignored the Range - restart
            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength.HasValue
                ? response.Content.Headers.ContentLength.Value + existingBytes
                : headLength;

            progress?.Report((existingBytes, totalBytes));

            await using (var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            await using (var fileStream = new FileStream(partPath,
                existingBytes > 0 ? FileMode.Append : FileMode.Create,
                FileAccess.Write, FileShare.None, bufferSize: 81920))
            {
                var buffer = new byte[81920];
                long totalRead = existingBytes;
                int bytesRead;
                while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                {
                    // Deliberately NOT passing the token to WriteAsync: a cancel between the read
                    // and the write would drop bytes already consumed from the socket, so the
                    // .incomplete length would no longer match what was received and the resume
                    // Range would be wrong.
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), CancellationToken.None).ConfigureAwait(false);
                    totalRead += bytesRead;
                    progress?.Report((totalRead, totalBytes));
                }
                await fileStream.FlushAsync(CancellationToken.None).ConfigureAwait(false);
            }

            File.Move(partPath, blobPath, overwrite: true);
        }

        bool hardlinked = HubCache.LinkOrCopy(blobPath, snapshotPath);
        hardlinked &= HubCache.LinkOrCopy(blobPath, mirrorPath);

        var refsPath = Path.Combine(HubCache.RepoDirectory(repoId, cacheRoot), "refs", revision);
        Directory.CreateDirectory(Path.GetDirectoryName(refsPath)!);
        await File.WriteAllTextAsync(refsPath, commit, CancellationToken.None).ConfigureAwait(false);

        return new HubCacheDownloadResult(
            blobPath, snapshotPath, mirrorPath, new FileInfo(blobPath).Length, hardlinked);
    }

    private HttpClient? _metadataClient;

    /// <summary>
    /// Redirect-free client used only for the metadata HEAD. Separate from
    /// <see cref="_httpClient"/> so the body GET keeps its normal auto-redirect behaviour (and
    /// .NET's stripping of the Authorization header when the redirect crosses to the CDN host),
    /// while the HEAD can read the headers the Hub puts on the 302 itself. Created lazily so the
    /// ordinary <see cref="DownloadFileAsync"/> path allocates nothing extra.
    /// </summary>
    private HttpClient MetadataClient
    {
        get
        {
            if (_metadataClient is not null) return _metadataClient;

            var handler = new SocketsHttpHandler { AllowAutoRedirect = false };
            var client = new HttpClient(handler, disposeHandler: true);
            foreach (var header in _httpClient.DefaultRequestHeaders)
                client.DefaultRequestHeaders.TryAddWithoutValidation(header.Key, header.Value);
            return _metadataClient = client;
        }
    }

    private static string ReadEtag(HttpResponseMessage response)
    {
        var raw = ReadHeader(response, "X-Linked-Etag") ?? ReadHeader(response, "ETag");
        return raw is null ? "unknown" : HubCache.SanitizeEtag(raw);
    }

    private static string? ReadHeader(HttpResponseMessage response, string name)
    {
        if (response.Headers.TryGetValues(name, out var values))
        {
            var v = values.FirstOrDefault();
            if (!string.IsNullOrWhiteSpace(v)) return v;
        }
        if (response.Content.Headers.TryGetValues(name, out var cvalues))
        {
            var v = cvalues.FirstOrDefault();
            if (!string.IsNullOrWhiteSpace(v)) return v;
        }
        return null;
    }

    /// <summary>
    /// Lists locally downloaded models in the models directory.
    /// </summary>
    /// <param name="modelsDir">Models directory. Defaults to <see cref="DefaultModelsDirectory"/>.</param>
    /// <returns>List of local model entries.</returns>
    public static List<LocalModel> ListLocalModels(string? modelsDir = null)
    {
        modelsDir ??= DefaultModelsDirectory;
        var models = new List<LocalModel>();

        if (!Directory.Exists(modelsDir))
            return models;

        // Walk: {modelsDir}/{owner}/{repo}/*.gguf
        foreach (var ownerDir in Directory.GetDirectories(modelsDir))
        {
            var owner = Path.GetFileName(ownerDir);
            foreach (var repoDir in Directory.GetDirectories(ownerDir))
            {
                var repo = Path.GetFileName(repoDir);
                var repoId = $"{owner}/{repo}";
                foreach (var file in Directory.GetFiles(repoDir, "*.gguf"))
                {
                    var info = new FileInfo(file);
                    models.Add(new LocalModel(repoId, info.Name, info.FullName, info.Length, info.LastWriteTimeUtc));
                }
            }
        }

        return models;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _metadataClient?.Dispose();
        _metadataClient = null;
        if (_ownsClient)
            _httpClient.Dispose();
    }
}
