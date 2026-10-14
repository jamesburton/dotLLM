using System.Net;
using System.Security.Cryptography;
using System.Text;
using DotLLM.HuggingFace;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.HuggingFace;

/// <summary>Issue #718: reading an ollama store, pulling from a registry, and resolving ollama names. No test touches the real registry.</summary>
[Collection("SequentialFileIO")]
public sealed class OllamaInteropTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-718-" + Guid.NewGuid().ToString("N"));
    private string Store => Path.Combine(_root, "ollama");
    private string Models => Path.Combine(_root, "models");
    private string Profiles => Path.Combine(_root, "profiles");

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    private static byte[] FakeGguf(int size, byte fill = 7)
    {
        var b = new byte[size];
        Array.Fill(b, fill);
        "GGUF"u8.CopyTo(b);
        return b;
    }

    private static string Sha(byte[] data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();

    /// <summary>Writes a manifest + blobs the way ollama lays them out.</summary>
    private string AddOllamaModel(string ns, string name, string tag, byte[] gguf, string? system = null, string? paramsJson = null, string host = "registry.ollama.ai")
    {
        string blobs = Path.Combine(Store, "blobs");
        Directory.CreateDirectory(blobs);
        var layers = new List<string>();
        string Put(byte[] data, string mediaType)
        {
            string d = Sha(data);
            File.WriteAllBytes(Path.Combine(blobs, "sha256-" + d), data);
            layers.Add($"{{\"mediaType\":\"{mediaType}\",\"digest\":\"sha256:{d}\",\"size\":{data.Length}}}");
            return d;
        }
        Put(gguf, "application/vnd.ollama.image.model");
        if (system is not null) Put(Encoding.UTF8.GetBytes(system), "application/vnd.ollama.image.system");
        if (paramsJson is not null) Put(Encoding.UTF8.GetBytes(paramsJson), "application/vnd.ollama.image.params");
        Put(Encoding.UTF8.GetBytes("license text"), "application/vnd.ollama.image.license");
        string dir = Path.Combine(Store, "manifests", host, ns, name);
        Directory.CreateDirectory(dir);
        string manifest = Path.Combine(dir, tag);
        File.WriteAllText(manifest, "{\"schemaVersion\":2,\"layers\":[" + string.Join(',', layers) + "]}");
        return Path.Combine(blobs, "sha256-" + Sha(gguf));
    }

    [Theory]
    [InlineData("llama3.2", "llama3.2", "latest", false)]
    [InlineData("llama3.2:3b", "llama3.2", "3b", false)]
    [InlineData("Llama3.2:3B", "llama3.2", "3b", false)]
    [InlineData("ollama:llama3.2:3b", "llama3.2", "3b", true)]
    [InlineData("ollama:user/model:q4", "user/model", "q4", true)]
    [InlineData("registry.ollama.ai/library/phi4-mini:latest", "phi4-mini", "latest", true)]
    [InlineData("ollama.com/library/gemma3", "gemma3", "latest", true)]
    public void Parse_OllamaForms(string arg, string name, string tag, bool isExplicit)
    {
        var r = OllamaRef.TryParse(arg)!.Value;
        Assert.Equal(name, r.Name);
        Assert.Equal(tag, r.Tag);
        Assert.Equal(isExplicit, r.Explicit);
    }

    [Theory]
    [InlineData("owner/repo")]               // a Hugging Face reference, not an ollama one
    [InlineData("owner/repo:Q4_K_M")]
    [InlineData("model.gguf")]
    [InlineData(@"C:\models\x")]
    [InlineData("/models/x")]
    [InlineData("")]
    [InlineData("a b")]
    public void Parse_NonOllamaForms_AreRejected(string arg) => Assert.Null(OllamaRef.TryParse(arg));

    [Fact]
    public void Store_ListsModels_WithSystemAndParams_AndTreatsLibraryAsImplicit()
    {
        string blob = AddOllamaModel("library", "tiny", "latest", FakeGguf(2048), "You are tiny.", "{\"temperature\":0.3,\"top_k\":20,\"stop\":[\"<|end|>\"],\"num_ctx\":4096,\"num_predict\":64}");
        AddOllamaModel("someone", "custom", "q4", FakeGguf(1024, 9));
        AddOllamaModel("library", "remote", "x", FakeGguf(512, 5), host: "example.com");

        var all = OllamaStore.ListAll(Store);

        Assert.Equal(["example.com/remote", "someone/custom", "tiny"], all.Select(m => m.Ref.Name));
        var tiny = all.Single(m => m.Ref.Name == "tiny");
        Assert.Equal(blob, tiny.BlobPath);
        Assert.Equal(2048, tiny.SizeBytes);
        Assert.Equal("You are tiny.", tiny.System);

        var profile = OllamaStore.ToProfile(tiny, tiny.BlobPath);
        Assert.Equal(0.3f, profile.Temperature);
        Assert.Equal(20, profile.TopK);
        Assert.Equal(64, profile.MaxTokens);
        Assert.Equal(["<|end|>"], profile.Stop!);
        Assert.Equal("You are tiny.", profile.System);
        Assert.Equal(blob, profile.From);   // num_ctx is ignored, not mapped
    }

    [Fact]
    public void Store_SkipsManifestsWhoseBlobIsMissing_AndCorruptManifests()
    {
        string blob = AddOllamaModel("library", "gone", "latest", FakeGguf(100));
        File.Delete(blob);
        string dir = Path.Combine(Store, "manifests", "registry.ollama.ai", "library", "junk");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "latest"), "{ not json");
        Assert.Empty(OllamaStore.ListAll(Store));
        Assert.Null(OllamaStore.TryFind(new OllamaRef("gone", "latest", false), Store));
    }

    [Fact]
    public void Resolver_FindsOllamaModelsInPlace_ByNameAndTag_OnlyWhenAsked()
    {
        string blob = AddOllamaModel("library", "tiny", "3b", FakeGguf(2048));

        Assert.Equal(blob, ModelResolver.ResolveLocal("tiny:3b", null, Models, null, includeOllama: true, ollamaRoot: Store));
        Assert.Equal(blob, ModelResolver.ResolveLocal("ollama:tiny:3b", null, Models, null, includeOllama: true, ollamaRoot: Store));
        Assert.Null(ModelResolver.ResolveLocal("tiny:latest", null, Models, null, includeOllama: true, ollamaRoot: Store));
        Assert.Null(ModelResolver.ResolveLocal("tiny:3b", null, Models, null));   // ollama lookup is opt-in: the default resolver never reads another tool's store

        var listed = ModelResolver.EnumerateLocal(Models, Path.Combine(_root, "hub"), includeOllama: true, ollamaRoot: Store);
        var m = Assert.Single(listed);
        Assert.Equal("ollama/tiny", m.RepoId);
        Assert.Equal("tiny-3b.gguf", m.Filename);
        Assert.Equal(blob, m.FullPath);
        Assert.Empty(ModelResolver.EnumerateLocal(Models, Path.Combine(_root, "hub")));
    }

    [Fact]
    public void Resolver_AcceptsAnAbsoluteGgufPathWithoutExtension_ByMagic_ButNotOtherFiles()
    {
        string blob = AddOllamaModel("library", "tiny", "latest", FakeGguf(512));
        Assert.Equal(Path.GetFullPath(blob), ModelResolver.ResolveLocal(blob, null, Models, null));

        string notGguf = Path.Combine(_root, "notes.txt");
        File.WriteAllText(notGguf, "hello world");
        Assert.Null(ModelResolver.ResolveLocal(notGguf, null, Models, null));
    }

    [Fact]
    public void ModelIdFor_AnOllamaBlobIsKeyedByItsOllamaName_NotByTheSha()
    {
        string blob = AddOllamaModel("library", "tiny", "3b", FakeGguf(512));
        var prev = Environment.GetEnvironmentVariable("OLLAMA_MODELS");
        try
        {
            Environment.SetEnvironmentVariable("OLLAMA_MODELS", Store);
            Assert.Equal("tiny:3b", ServerStartup.ModelIdFor("tiny:3b", blob));
            Assert.Equal("tiny:3b", ServerStartup.ModelIdFor("ollama:tiny:3b", blob));
            Assert.StartsWith("sha256-", ServerStartup.ModelIdFor(blob, blob));   // addressed by path: nothing better to call it
        }
        finally { Environment.SetEnvironmentVariable("OLLAMA_MODELS", prev); }
    }

    // ───────────────────────────── registry pull (stub server) ─────────────────────────────

    private sealed class StubRegistry : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Task _loop;
        public string BaseUrl { get; }
        public Dictionary<string, byte[]> Blobs { get; } = new();
        public string ManifestJson { get; set; } = "";
        public int FailAfterBytes { get; set; } = -1;   // cut the first blob response short to simulate an interrupted download
        public List<string> RangeHeaders { get; } = [];

        public StubRegistry()
        {
            int port;
            using (var probe = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0)) { probe.Start(); port = ((IPEndPoint)probe.LocalEndpoint).Port; }
            BaseUrl = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(BaseUrl + "/");
            _listener.Start();
            _loop = Task.Run(async () =>
            {
                while (_listener.IsListening)
                {
                    HttpListenerContext c;
                    try { c = await _listener.GetContextAsync(); } catch { return; }
                    try { Handle(c); } catch { try { c.Response.Abort(); } catch { } }
                }
            });
        }

        private void Handle(HttpListenerContext c)
        {
            string path = c.Request.Url!.AbsolutePath;
            if (path.Contains("/manifests/"))
            {
                if (ManifestJson.Length == 0) { c.Response.StatusCode = 404; c.Response.Close(); return; }
                var b = Encoding.UTF8.GetBytes(ManifestJson);
                c.Response.ContentType = "application/vnd.docker.distribution.manifest.v2+json";
                c.Response.OutputStream.Write(b); c.Response.Close(); return;
            }
            string digest = path[(path.LastIndexOf('/') + 1)..];
            if (!Blobs.TryGetValue(digest, out var data)) { c.Response.StatusCode = 404; c.Response.Close(); return; }

            int start = 0;
            if (c.Request.Headers["Range"] is { } range)
            {
                RangeHeaders.Add(range);
                start = int.Parse(range.Replace("bytes=", "").TrimEnd('-'));
                c.Response.StatusCode = 206;
            }
            int count = data.Length - start;
            if (FailAfterBytes >= 0 && start == 0) { count = FailAfterBytes; FailAfterBytes = -1; c.Response.ContentLength64 = data.Length - start; c.Response.OutputStream.Write(data, start, count); c.Response.Abort(); return; }
            c.Response.ContentLength64 = count;
            c.Response.OutputStream.Write(data, start, count);
            c.Response.Close();
        }

        public void Dispose() { try { _listener.Stop(); _listener.Close(); } catch { } }
    }

    private static StubRegistry NewRegistry(byte[] gguf, string system, string paramsJson)
    {
        var r = new StubRegistry();
        string g = Sha(gguf), s = Sha(Encoding.UTF8.GetBytes(system)), p = Sha(Encoding.UTF8.GetBytes(paramsJson));
        r.Blobs["sha256:" + g] = gguf;
        r.Blobs["sha256:" + s] = Encoding.UTF8.GetBytes(system);
        r.Blobs["sha256:" + p] = Encoding.UTF8.GetBytes(paramsJson);
        r.ManifestJson = "{\"layers\":[" +
            $"{{\"mediaType\":\"application/vnd.ollama.image.model\",\"digest\":\"sha256:{g}\",\"size\":{gguf.Length}}}," +
            $"{{\"mediaType\":\"application/vnd.ollama.image.system\",\"digest\":\"sha256:{s}\",\"size\":{system.Length}}}," +
            $"{{\"mediaType\":\"application/vnd.ollama.image.params\",\"digest\":\"sha256:{p}\",\"size\":{paramsJson.Length}}}]}}";
        return r;
    }

    [Fact]
    public async Task Pull_DownloadsVerifiesAndSavesAProfileWithSystemAndParams()
    {
        var gguf = FakeGguf(3 << 20);
        using var reg = NewRegistry(gguf, "Be brief.", "{\"temperature\":0.5}");
        using var client = new OllamaRegistry(new HttpClient(), reg.BaseUrl);

        string path = await ModelResolver.PullOllamaAsync(new OllamaRef("tiny", "3b", true), client, null, default, Models, Profiles);

        Assert.Equal(OllamaRegistry.TargetPath(new OllamaRef("tiny", "3b", true), Models), path);
        Assert.Equal(gguf, await File.ReadAllBytesAsync(path));
        Assert.False(File.Exists(path + ".incomplete"));
        var profile = ModelProfileStore.TryGet("tiny:3b", Profiles)!;
        Assert.Equal(path, profile.From);
        Assert.Equal("Be brief.", profile.System);
        Assert.Equal(0.5f, profile.Temperature);
    }

    [Fact]
    public async Task Pull_ResumesAnInterruptedDownload_WithARangeRequest_AndStillVerifiesTheWholeFile()
    {
        var gguf = FakeGguf(4 << 20);
        for (int i = 4; i < gguf.Length; i++) gguf[i] = (byte)(i * 31);   // non-constant so a wrong splice changes the digest
        using var reg = NewRegistry(gguf, "s", "{}");
        reg.FailAfterBytes = 1 << 20;
        using var client = new OllamaRegistry(new HttpClient(), reg.BaseUrl);
        var r = new OllamaRef("tiny", "latest", true);

        await Assert.ThrowsAnyAsync<Exception>(() => client.PullAsync(r, null, default, Models));
        string part = OllamaRegistry.TargetPath(r, Models) + ".incomplete";
        Assert.True(File.Exists(part));
        Assert.InRange(new FileInfo(part).Length, 1, gguf.Length - 1);

        var (path, _) = await client.PullAsync(r, null, default, Models);

        Assert.Equal(gguf, await File.ReadAllBytesAsync(path));
        Assert.Single(reg.RangeHeaders);   // resumed, did not restart
        Assert.False(File.Exists(part));
    }

    [Fact]
    public async Task Pull_DiscardsAndRejectsABlobWhoseDigestDoesNotMatch()
    {
        var gguf = FakeGguf(1 << 20);
        using var reg = NewRegistry(gguf, "s", "{}");
        string digest = reg.Blobs.Keys.First(k => reg.Blobs[k].Length == gguf.Length);
        var corrupt = (byte[])gguf.Clone();
        corrupt[1000] ^= 0xFF;
        reg.Blobs[digest] = corrupt;   // the server serves bytes that do not hash to the manifest's digest
        using var client = new OllamaRegistry(new HttpClient(), reg.BaseUrl);
        var r = new OllamaRef("tiny", "latest", true);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => client.PullAsync(r, null, default, Models));

        Assert.Contains("sha256 mismatch", ex.Message);
        Assert.False(File.Exists(OllamaRegistry.TargetPath(r, Models)));
        Assert.False(File.Exists(OllamaRegistry.TargetPath(r, Models) + ".incomplete"));
    }

    [Fact]
    public async Task Pull_UnknownModel_ReportsNotFound()
    {
        using var reg = new StubRegistry();   // empty manifest -> 404
        using var client = new OllamaRegistry(new HttpClient(), reg.BaseUrl);
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => client.PullAsync(new OllamaRef("nope", "latest", false), null, default, Models));
        Assert.Contains("not found on the ollama registry", ex.Message);
    }

    [Fact]
    public async Task Pull_AlreadyPulledFile_IsNotDownloadedAgain()
    {
        var gguf = FakeGguf(1 << 20);
        using var reg = NewRegistry(gguf, "s", "{}");
        using var client = new OllamaRegistry(new HttpClient(), reg.BaseUrl);
        var r = new OllamaRef("tiny", "latest", true);
        await client.PullAsync(r, null, default, Models);
        reg.Blobs.Remove(reg.Blobs.Keys.First(k => reg.Blobs[k].Length == gguf.Length));   // the model blob is gone: a second download would 404
        var (path, _) = await client.PullAsync(r, null, default, Models);
        Assert.True(File.Exists(path));
    }
}
