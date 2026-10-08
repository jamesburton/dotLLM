using System.Net;
using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Unit.HuggingFace;

/// <summary>Issue #756: the resolver treats a <c>-0000N-of-0000M.gguf</c> set as ONE model - pulled whole, listed once, deleted whole.</summary>
public sealed class SplitGgufResolverTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-756-" + Guid.NewGuid().ToString("N"));
    private string Hub => Path.Combine(_root, "hub");
    private string Mirror => Path.Combine(_root, "models");

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    private void AddHub(string repo, string file, int size, char fill)
    {
        string repoDir = HubCache.RepoDirectory(repo, Hub);
        string blob = Path.Combine(repoDir, "blobs", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, Enumerable.Repeat((byte)fill, size).ToArray());
        string snap = HubCache.SnapshotFilePath(repo, "c0ffee", file, Hub);
        Directory.CreateDirectory(Path.GetDirectoryName(snap)!);
        HubCache.LinkOrCopy(blob, snap);
    }

    [Theory]
    [InlineData("m-00001-of-00003.gguf", true, 1, 3)]
    [InlineData("sub/dir/m-00002-of-00003.gguf", true, 2, 3)]
    [InlineData("m-Q4_K_M.gguf", false, 0, 0)]
    [InlineData("m-00001-of-00001.gguf", false, 0, 0)]
    public void TryParse_RecognisesOnlyRealSets(string name, bool split, int no, int count)
    {
        Assert.Equal(split, SplitGguf.TryParse(name, out _, out int n, out int c));
        if (split) { Assert.Equal(no, n); Assert.Equal(count, c); }
    }

    [Fact]
    public void ShardNames_ListsEveryShardFromAnyMember()
    {
        Assert.Equal(["q/m-00001-of-00003.gguf", "q/m-00002-of-00003.gguf", "q/m-00003-of-00003.gguf"],
            SplitGguf.ShardNames("q/m-00001-of-00003.gguf"));
        Assert.Equal(["plain.gguf"], SplitGguf.ShardNames("plain.gguf"));
    }

    [Fact]
    public void EnumerateLocal_ListsASetOnceUnderShard1_WithTheSummedSize()
    {
        AddHub("acme/big", "Q4/m-00001-of-00003.gguf", 100, 'a');
        AddHub("acme/big", "Q4/m-00002-of-00003.gguf", 200, 'b');
        AddHub("acme/big", "Q4/m-00003-of-00003.gguf", 300, 'c');

        var all = ModelResolver.EnumerateLocal(Mirror, Hub);

        var only = Assert.Single(all);
        Assert.Equal("Q4/m-00001-of-00003.gguf", only.Filename);
        Assert.Equal(600, only.SizeBytes);
        Assert.Equal(only.FullPath, ModelResolver.ResolveLocal("acme/big:Q4", null, Mirror, Hub));
        Assert.Equal(600, ModelResolver.FileLength(only.FullPath));
    }

    [Fact]
    public void DeleteLocal_RemovesEveryShardAndBlob_AndLeavesOtherModelsAlone()
    {
        AddHub("acme/big", "m-00001-of-00002.gguf", 100, 'a');
        AddHub("acme/big", "m-00002-of-00002.gguf", 250, 'b');
        AddHub("acme/big", "keep-Q8_0.gguf", 77, 'k');
        var set = ModelResolver.EnumerateLocal(Mirror, Hub).Single(m => m.Filename.StartsWith("m-0"));

        long freed = ModelResolver.DeleteLocal(set, Mirror, Hub);

        Assert.Equal(350, freed);
        Assert.Equal(["keep-Q8_0.gguf"], ModelResolver.EnumerateLocal(Mirror, Hub).Select(m => m.Filename));
        Assert.Single(Directory.GetFiles(Path.Combine(HubCache.RepoDirectory("acme/big", Hub), "blobs")));
    }

    [Fact]
    public async Task Pull_OfShard1_FetchesAllShards_IntoTheHubCacheAsHardlinks()
    {
        byte[][] payloads = [Payload(1500, 1), Payload(2500, 2), Payload(900, 3)];
        string[] names = SplitGguf.ShardNames("m-00001-of-00003.gguf").ToArray();
        using var hub = new PerFileHub(names, payloads);
        using var downloader = new HuggingFaceDownloader(cdnBase: hub.BaseUrl);

        var reported = new List<(long, long?)>();
        var result = await downloader.DownloadModelToHubCacheAsync(
            "acme/split", names[0], "main", Hub, Mirror, new InlineProgress(reported.Add));

        Assert.Equal(Path.Combine(Mirror, "acme", "split", names[0]), result.ModelPath);
        Assert.Equal(1500 + 2500 + 900, result.SizeBytes);
        for (int i = 0; i < 3; i++)
        {
            string mirror = Path.Combine(Mirror, "acme", "split", names[i]);
            string snap = HubCache.SnapshotFilePath("acme/split", hub.Commit, names[i], Hub);
            Assert.Equal(payloads[i], await File.ReadAllBytesAsync(mirror));
            Assert.Equal(payloads[i], await File.ReadAllBytesAsync(snap));
        }
        // One physical copy per shard: exactly three blobs, no extra copies.
        Assert.Equal(3, Directory.GetFiles(Path.Combine(HubCache.RepoDirectory("acme/split", Hub), "blobs")).Length);
        // Progress is cumulative across shards and ends at the full size.
        Assert.Equal(1500 + 2500 + 900, reported[^1].Item1);
        Assert.True(reported.Select(r => r.Item1).Zip(reported.Select(r => r.Item1).Skip(1)).All(p => p.Second >= p.First));
    }

    [Fact]
    public async Task Pull_OfAnOrdinaryFile_IsExactlyTheSingleFileDownload()
    {
        byte[][] payloads = [Payload(700, 9)];
        using var hub = new PerFileHub(["plain-Q4_K_M.gguf"], payloads);
        using var downloader = new HuggingFaceDownloader(cdnBase: hub.BaseUrl);

        var r = await downloader.DownloadModelToHubCacheAsync("acme/plain", "plain-Q4_K_M.gguf", "main", Hub, Mirror);

        Assert.Equal(700, r.SizeBytes);
        Assert.Equal(1, hub.GetCount);
    }

    private static byte[] Payload(int n, int seed)
    {
        var b = new byte[n];
        new Random(seed).NextBytes(b);
        return b;
    }

    private sealed class InlineProgress(Action<(long, long?)> sink) : IProgress<(long bytesDownloaded, long? totalBytes)>
    {
        public void Report((long bytesDownloaded, long? totalBytes) value) => sink((value.bytesDownloaded, value.totalBytes));
    }

    /// <summary>Loopback hub serving a different payload and ETag per file name, with the HEAD identity headers the downloader requires.</summary>
    private sealed class PerFileHub : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Dictionary<string, byte[]> _files;
        public string BaseUrl { get; }
        public string Commit => "0123456789abcdef0123456789abcdef01234567";
        public int GetCount;

        public PerFileHub(string[] names, byte[][] payloads)
        {
            _files = names.Zip(payloads).ToDictionary(p => p.First, p => p.Second);
            var l = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            BaseUrl = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(BaseUrl + "/");
            _listener.Start();
            _ = Task.Run(async () =>
            {
                while (true)
                {
                    HttpListenerContext ctx;
                    try { ctx = await _listener.GetContextAsync(); } catch { return; }
                    _ = Task.Run(() => Handle(ctx));
                }
            });
        }

        private void Handle(HttpListenerContext ctx)
        {
            try
            {
                string path = ctx.Request.Url!.AbsolutePath;
                string file = _files.Keys.FirstOrDefault(k => path.EndsWith("/" + k, StringComparison.Ordinal)) ?? "";
                if (!_files.TryGetValue(file, out var payload)) { ctx.Response.StatusCode = 404; ctx.Response.Close(); return; }
                string etag = "etag" + Convert.ToHexString(System.Security.Cryptography.SHA1.HashData(payload))[..12].ToLowerInvariant();
                ctx.Response.Headers["ETag"] = $"\"{etag}\"";
                ctx.Response.Headers["X-Linked-Etag"] = $"\"{etag}\"";
                ctx.Response.Headers["X-Repo-Commit"] = Commit;
                ctx.Response.Headers["X-Linked-Size"] = payload.Length.ToString();
                ctx.Response.ContentLength64 = payload.Length;
                if (ctx.Request.HttpMethod != "HEAD")
                {
                    Interlocked.Increment(ref GetCount);
                    ctx.Response.OutputStream.Write(payload);
                }
                ctx.Response.Close();
            }
            catch { try { ctx.Response.Abort(); } catch { } }
        }

        public void Dispose() { try { _listener.Close(); } catch { } }
    }
}
