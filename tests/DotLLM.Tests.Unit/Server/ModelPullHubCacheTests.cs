using System.Net;
using System.Text;
using DotLLM.HuggingFace;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for <c>POST /v1/models/pull</c>'s download machinery (#454): the Hugging Face hub-cache
/// layout, hardlink mirroring, resume and cancellation.
/// </summary>
/// <remarks>
/// <para>
/// <b>No test here touches the real Hugging Face Hub.</b> Every case runs against a local
/// <see cref="HttpListener"/> stub on a loopback port serving a few kilobytes, with
/// <c>HubCache</c> and the models mirror both pointed at a temp directory. Shared bandwidth and
/// the multi-gigabyte size of real GGUFs make an actual download unacceptable in a unit suite.
/// </para>
/// <para>
/// These are discriminating rather than tautological: the layout assertions fail against a
/// downloader that writes into a flat <c>~/.dotllm/models</c> tree (the pre-#454 behaviour of
/// <see cref="HuggingFaceDownloader.DownloadFileAsync"/>), the resume assertion fails against a
/// downloader that ignores an existing <c>.incomplete</c> file, and the cancel assertion fails
/// against one that deletes partial state on abort.
/// </para>
/// </remarks>
public sealed class ModelPullHubCacheTests : IDisposable
{
    private const string RepoId = "acme/test-gguf";
    private const string Filename = "tiny-Q4_K_M.gguf";
    private const string Etag = "deadbeefcafe";
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-454-" + Guid.NewGuid().ToString("N"));
    private string CacheRoot => Path.Combine(_root, "hub");
    private string ModelsDir => Path.Combine(_root, "models");

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    // ───────────────────────────── path layout ────────────────────────────────

    [Fact]
    public void HubCache_RepoFolderName_MatchesHuggingFaceHubConvention()
    {
        Assert.Equal("models--acme--test-gguf", HubCache.RepoFolderName("acme/test-gguf"));
    }

    [Fact]
    public void HubCache_Root_HonoursHfHubCacheEnvVar()
    {
        var prev = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        try
        {
            Environment.SetEnvironmentVariable("HF_HUB_CACHE", @"X:\somewhere\hub");
            Assert.Equal(@"X:\somewhere\hub", HubCache.CacheRoot);
        }
        finally { Environment.SetEnvironmentVariable("HF_HUB_CACHE", prev); }
    }

    [Fact]
    public void HubCache_Root_FallsBackToHfHomeThenDefault()
    {
        var prevCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        var prevHome = Environment.GetEnvironmentVariable("HF_HOME");
        try
        {
            Environment.SetEnvironmentVariable("HF_HUB_CACHE", null);
            Environment.SetEnvironmentVariable("HF_HOME", Path.Combine("Y:", "hf"));
            Assert.Equal(Path.Combine("Y:", "hf", "hub"), HubCache.CacheRoot);

            Environment.SetEnvironmentVariable("HF_HOME", null);
            Assert.EndsWith(Path.Combine(".cache", "huggingface", "hub"), HubCache.CacheRoot);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HF_HUB_CACHE", prevCache);
            Environment.SetEnvironmentVariable("HF_HOME", prevHome);
        }
    }

    [Fact]
    public void HubCache_SanitizeEtag_StripsQuotesAndWeakPrefix()
    {
        Assert.Equal("abc123", HubCache.SanitizeEtag("\"abc123\""));
        Assert.Equal("abc123", HubCache.SanitizeEtag("W/\"abc123\""));
        Assert.DoesNotContain(Path.DirectorySeparatorChar, HubCache.SanitizeEtag("a/b"));
    }

    // ─────────────────────────── download behaviour ───────────────────────────

    [Fact]
    public async Task Download_LandsInHubCacheAndMirrors_AsOnePhysicalCopy()
    {
        var payload = MakePayload(4096);
        using var stub = new StubHub(payload);

        using var downloader = new HuggingFaceDownloader(cdnBase: stub.BaseUrl);
        var result = await downloader.DownloadToHubCacheAsync(
            RepoId, Filename, revision: "main", cacheRoot: CacheRoot, modelsDir: ModelsDir);

        // Hub-cache layout, exactly as huggingface_hub writes it.
        Assert.Equal(Path.Combine(CacheRoot, "models--acme--test-gguf", "blobs", Etag), result.BlobPath);
        Assert.Equal(
            Path.Combine(CacheRoot, "models--acme--test-gguf", "snapshots", Commit, Filename),
            result.SnapshotPath);
        Assert.True(File.Exists(result.BlobPath));
        Assert.True(File.Exists(result.SnapshotPath));

        // refs/main records the resolved commit.
        var refPath = Path.Combine(CacheRoot, "models--acme--test-gguf", "refs", "main");
        Assert.Equal(Commit, await File.ReadAllTextAsync(refPath));

        // Mirrored where the server's flat-tree resolver looks.
        Assert.Equal(Path.Combine(ModelsDir, "acme", "test-gguf", Filename), result.ModelPath);
        Assert.True(File.Exists(result.ModelPath));

        // Same bytes everywhere, and no .incomplete left behind.
        Assert.Equal(payload, await File.ReadAllBytesAsync(result.BlobPath));
        Assert.Equal(payload, await File.ReadAllBytesAsync(result.ModelPath));
        Assert.False(File.Exists(result.BlobPath + ".incomplete"));
        Assert.Equal(payload.Length, result.SizeBytes);
    }

    /// <summary>
    /// The mirror must be a hardlink, not a copy — one physical file on disk, per the repo's
    /// model-storage rule. Verified by writing through one path and reading the other.
    /// </summary>
    [Fact]
    public async Task Download_MirrorIsAHardlink_NotASecondCopy()
    {
        using var stub = new StubHub(MakePayload(1024));
        using var downloader = new HuggingFaceDownloader(cdnBase: stub.BaseUrl);
        var result = await downloader.DownloadToHubCacheAsync(
            RepoId, Filename, "main", CacheRoot, ModelsDir);

        Assert.True(result.Hardlinked, "expected a hardlink; a copy means two physical copies on disk");

        // Both names resolve to the same inode/file record: a write through one is visible
        // through the other.
        using (var fs = new FileStream(result.BlobPath, FileMode.Open, FileAccess.Write, FileShare.ReadWrite))
        {
            fs.Seek(0, SeekOrigin.Begin);
            fs.WriteByte(0x7F);
        }
        using var read = new FileStream(result.ModelPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        Assert.Equal(0x7F, read.ReadByte());
    }

    /// <summary>A partial <c>.incomplete</c> file must be continued via a Range request, not re-fetched.</summary>
    [Fact]
    public async Task Download_ResumesFromIncompleteFile()
    {
        var payload = MakePayload(8192);
        using var stub = new StubHub(payload);

        // Pre-seed the first half, exactly as an interrupted attempt would leave it.
        var partPath = HubCache.IncompleteBlobPath(RepoId, Etag, CacheRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(partPath)!);
        await File.WriteAllBytesAsync(partPath, payload[..4096]);

        using var downloader = new HuggingFaceDownloader(cdnBase: stub.BaseUrl);
        var result = await downloader.DownloadToHubCacheAsync(
            RepoId, Filename, "main", CacheRoot, ModelsDir);

        Assert.Equal(payload, await File.ReadAllBytesAsync(result.BlobPath));
        Assert.Equal(4096, stub.LastRangeFrom);       // asked only for the remainder
        Assert.Equal(4096, stub.LastBodyBytesServed); // and only the remainder was sent
    }

    /// <summary>A completed blob must short-circuit: no GET at all, just re-link.</summary>
    [Fact]
    public async Task Download_AlreadyCached_DoesNotRefetchTheBody()
    {
        var payload = MakePayload(2048);
        using var stub = new StubHub(payload);
        using var downloader = new HuggingFaceDownloader(cdnBase: stub.BaseUrl);

        await downloader.DownloadToHubCacheAsync(RepoId, Filename, "main", CacheRoot, ModelsDir);
        int getsAfterFirst = stub.GetCount;

        var again = await downloader.DownloadToHubCacheAsync(RepoId, Filename, "main", CacheRoot, ModelsDir);

        Assert.Equal(getsAfterFirst, stub.GetCount);
        Assert.True(File.Exists(again.ModelPath));
    }

    /// <summary>Cancelling must leave the partial bytes so a later pull resumes rather than restarts.</summary>
    [Fact]
    public async Task Cancel_LeavesIncompleteFileForResume()
    {
        var payload = MakePayload(1 << 20); // 1 MiB, throttled by the stub
        using var stub = new StubHub(payload, throttleChunk: 16 * 1024, throttleDelayMs: 25);
        using var cts = new CancellationTokenSource();

        using var downloader = new HuggingFaceDownloader(cdnBase: stub.BaseUrl);
        var task = downloader.DownloadToHubCacheAsync(
            RepoId, Filename, "main", CacheRoot, ModelsDir,
            progress: new Progress<(long, long?)>(p => { if (p.Item1 > 32 * 1024) cts.Cancel(); }),
            cancellationToken: cts.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await task);

        var partPath = HubCache.IncompleteBlobPath(RepoId, Etag, CacheRoot);
        Assert.True(File.Exists(partPath), "cancel must preserve the .incomplete file so a re-pull resumes");
        Assert.True(new FileInfo(partPath).Length > 0);
        Assert.False(File.Exists(HubCache.BlobPath(RepoId, Etag, CacheRoot)));
    }

    // ──────────────────────────── job manager ─────────────────────────────────

    [Fact]
    public async Task PullManager_RunsJobToCompletion_AndReportsProgress()
    {
        var payload = MakePayload(4096);
        using var stub = new StubHub(payload);
        using var manager = new ModelPullManager(
            () => new HuggingFaceDownloader(cdnBase: stub.BaseUrl), CacheRoot, ModelsDir);

        var job = manager.Start(RepoId, Filename, revision: null);
        int progressTicks = 0;
        job.Progress += _ => Interlocked.Increment(ref progressTicks);

        await job.WaitAsync();

        var dto = job.ToDto();
        Assert.Equal("completed", dto.Status);
        Assert.Equal(payload.Length, dto.BytesDownloaded);
        Assert.Equal(100d, dto.Percent);
        Assert.NotNull(dto.ModelPath);
        Assert.NotNull(dto.BlobPath);
        Assert.Null(dto.Error);
        Assert.True(progressTicks > 0);
    }

    /// <summary>Two POSTs for the same file must not race two writers against one <c>.incomplete</c>.</summary>
    [Fact]
    public void PullManager_DeduplicatesConcurrentJobsForTheSameTarget()
    {
        using var stub = new StubHub(MakePayload(1 << 20), throttleChunk: 8192, throttleDelayMs: 20);
        using var manager = new ModelPullManager(
            () => new HuggingFaceDownloader(cdnBase: stub.BaseUrl), CacheRoot, ModelsDir);

        var first = manager.Start(RepoId, Filename, null);
        var second = manager.Start(RepoId, Filename, null);

        Assert.Same(first, second);
        Assert.Single(manager.List());
        manager.Cancel(first.Id);
    }

    [Fact]
    public void PullManager_DifferentFiles_AreDistinctJobs()
    {
        using var stub = new StubHub(MakePayload(512));
        using var manager = new ModelPullManager(
            () => new HuggingFaceDownloader(cdnBase: stub.BaseUrl), CacheRoot, ModelsDir);

        var a = manager.Start(RepoId, Filename, null);
        var b = manager.Start(RepoId, "other.gguf", null);

        Assert.NotSame(a, b);
        Assert.Equal(2, manager.List().Count);
    }

    [Fact]
    public async Task PullManager_Cancel_MarksJobCancelled()
    {
        using var stub = new StubHub(MakePayload(1 << 20), throttleChunk: 8192, throttleDelayMs: 20);
        using var manager = new ModelPullManager(
            () => new HuggingFaceDownloader(cdnBase: stub.BaseUrl), CacheRoot, ModelsDir);

        var job = manager.Start(RepoId, Filename, null);
        Assert.True(manager.Cancel(job.Id));
        await job.WaitAsync();

        Assert.Equal("cancelled", job.ToDto().Status);
    }

    [Fact]
    public async Task PullManager_FailedDownload_IsReportedNotThrown()
    {
        using var stub = new StubHub(MakePayload(64), failWith: HttpStatusCode.NotFound);
        using var manager = new ModelPullManager(
            () => new HuggingFaceDownloader(cdnBase: stub.BaseUrl), CacheRoot, ModelsDir);

        var job = manager.Start(RepoId, Filename, null);
        await job.WaitAsync();

        var dto = job.ToDto();
        Assert.Equal("failed", dto.Status);
        Assert.NotNull(dto.Error);
    }

    [Fact]
    public void PullManager_Get_UnknownId_ReturnsNull()
    {
        using var manager = new ModelPullManager(() => new HuggingFaceDownloader(), CacheRoot, ModelsDir);
        Assert.Null(manager.Get("nope"));
        Assert.False(manager.Cancel("nope"));
    }

    // ────────────────────────────── stub server ───────────────────────────────

    private static byte[] MakePayload(int size)
    {
        var bytes = new byte[size];
        for (int i = 0; i < size; i++) bytes[i] = (byte)(i * 31 % 251);
        return bytes;
    }

    /// <summary>
    /// Minimal stand-in for <c>huggingface.co/{repo}/resolve/{rev}/{file}</c>: answers HEAD with
    /// the etag/commit headers the hub-cache layout is derived from, and GET with optional
    /// <c>Range</c> support. Bound to a loopback port; never reaches the network.
    /// </summary>
    private sealed class StubHub : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly byte[] _payload;
        private readonly int _throttleChunk;
        private readonly int _throttleDelayMs;
        private readonly HttpStatusCode? _failWith;
        private readonly CancellationTokenSource _cts = new();

        public string BaseUrl { get; }
        public long LastRangeFrom { get; private set; }
        public long LastBodyBytesServed { get; private set; }
        public int GetCount;

        public StubHub(byte[] payload, int throttleChunk = 0, int throttleDelayMs = 0, HttpStatusCode? failWith = null)
        {
            _payload = payload;
            _throttleChunk = throttleChunk;
            _throttleDelayMs = throttleDelayMs;
            _failWith = failWith;

            int port = FreePort();
            BaseUrl = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(BaseUrl + "/");
            _listener.Start();
            _ = Task.Run(LoopAsync);
        }

        private static int FreePort()
        {
            var l = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        private async Task LoopAsync()
        {
            while (!_cts.IsCancellationRequested)
            {
                HttpListenerContext ctx;
                try { ctx = await _listener.GetContextAsync(); }
                catch { return; }
                _ = Task.Run(() => HandleAsync(ctx));
            }
        }

        private async Task HandleAsync(HttpListenerContext ctx)
        {
            try
            {
                if (_failWith is { } fail)
                {
                    ctx.Response.StatusCode = (int)fail;
                    ctx.Response.Close();
                    return;
                }

                ctx.Response.Headers["ETag"] = $"\"{Etag}\"";
                ctx.Response.Headers["X-Linked-Etag"] = $"\"{Etag}\"";
                ctx.Response.Headers["X-Repo-Commit"] = Commit;

                if (ctx.Request.HttpMethod == "HEAD")
                {
                    ctx.Response.ContentLength64 = _payload.Length;
                    ctx.Response.Close();
                    return;
                }

                Interlocked.Increment(ref GetCount);

                long from = 0;
                var range = ctx.Request.Headers["Range"];
                if (!string.IsNullOrEmpty(range) && range.StartsWith("bytes=", StringComparison.Ordinal))
                {
                    from = long.Parse(range["bytes=".Length..].Split('-')[0]);
                    ctx.Response.StatusCode = (int)HttpStatusCode.PartialContent;
                    ctx.Response.Headers["Content-Range"] = $"bytes {from}-{_payload.Length - 1}/{_payload.Length}";
                }
                LastRangeFrom = from;

                var slice = _payload.AsMemory((int)from);
                LastBodyBytesServed = slice.Length;
                ctx.Response.ContentLength64 = slice.Length;

                if (_throttleChunk > 0)
                {
                    for (int off = 0; off < slice.Length; off += _throttleChunk)
                    {
                        int n = Math.Min(_throttleChunk, slice.Length - off);
                        await ctx.Response.OutputStream.WriteAsync(slice.Slice(off, n));
                        await ctx.Response.OutputStream.FlushAsync();
                        if (_throttleDelayMs > 0) await Task.Delay(_throttleDelayMs);
                    }
                }
                else
                {
                    await ctx.Response.OutputStream.WriteAsync(slice);
                }
                ctx.Response.Close();
            }
            catch
            {
                try { ctx.Response.Abort(); } catch { }
            }
        }

        public void Dispose()
        {
            _cts.Cancel();
            try { _listener.Stop(); } catch { }
            try { _listener.Close(); } catch { }
            _cts.Dispose();
        }
    }

    private static string ToText(byte[] bytes) => Encoding.UTF8.GetString(bytes);
}
