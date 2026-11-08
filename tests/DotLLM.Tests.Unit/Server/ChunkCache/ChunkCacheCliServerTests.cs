using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkCache;
using DotLLM.ChunkReuse.Store;
using DotLLM.Server.ChunkCache;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The <c>chunkcache</c> CLI against the real server code (Phase I Task 17): the capabilities endpoint names the index, a chunk
/// the CLI built is a <c>hit</c> in the server's explain, and the explain verb prints the server's own plan. The CLI runs in
/// process with the default HTTP handler, so the bytes cross a real loopback socket into the real endpoint handlers.
/// Prewarm and evict are not on this server yet (Task 16 is on the research branch): those verbs are covered with a fake
/// server in <c>ChunkCacheCliTests</c>.
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkCacheCliServerTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-cli-server-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    public ChunkCacheCliServerTests() => _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private const string Source = """
        namespace Demo;

        /// <summary>Arithmetic helpers.</summary>
        public static class Calc
        {
            /// <summary>Adds two numbers.</summary>
            public static int Add(int a, int b)
            {
                return a + b;
            }

            /// <summary>Multiplies two numbers.</summary>
            public static int Mul(int a, int b)
            {
                return a * b;
            }
        }

        """;

    private string NewDir(string name)
    {
        string d = Path.Combine(_scratch, name + "-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(d);
        return d;
    }

    private string NewTree()
    {
        string tree = NewDir("tree");
        Directory.CreateDirectory(Path.Combine(tree, "src"));
        File.WriteAllText(Path.Combine(tree, "src", "Calc.cs"), Source.Replace("\r\n", "\n", StringComparison.Ordinal));
        return tree;
    }

    private static (int Code, string Out, string Err) Cli(params string[] args)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = ChunkCacheCli.Run(args, o, e, Path.GetTempPath(), httpHandler: null, getEnv: _ => null);
        return (code, o.ToString(), e.ToString());
    }

    private static ChunkCacheOptions On(string store) => ChunkCacheOptions.Create(true, null, store, null, false);

    [Fact]
    public async Task Build_IndexOnly_IsNamedByTheServersOwnModelAndProducer()
    {
        string store = NewDir("store");
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(store));
        string url = server.Http.BaseAddress!.ToString();
        JsonNode caps = JsonNode.Parse(await server.GetAsync("/v1/chunk-cache"))!;
        string id = caps["model"]!["id"]!.GetValue<string>(), producer = caps["model"]!["producer"]!.GetValue<string>();

        var r = Cli("build", "--model", _model, "--store", store, "--directory", NewTree(), "--server", url, "--index-only", "--json");
        Assert.True(r.Code == 0, r.Err);
        string expected = Path.GetFileName(ChunkIndex.DefaultPath(store, "symbol-512", id, producer, ChunkNamespace.Default));
        Assert.Equal([expected], Directory.GetFiles(store, "index-*.json").Select(Path.GetFileName));
        Assert.Equal("index_only", JsonDocument.Parse(r.Out).RootElement.GetProperty("mode").GetString());

        // A server that does not serve the convention is a clean error, not a crash (the flag-off 404).
        await using var off = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Disabled);
        var none = Cli("build", "--model", _model, "--store", NewDir("store"), "--directory", NewTree(), "--server", off.Http.BaseAddress!.ToString(), "--index-only", "--json");
        Assert.Equal(1, none.Code);
        Assert.Contains("chunk-cache", none.Err, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Select_Server_StoredComesFromTheServersExplain()
    {
        string store = NewDir("store");
        string tree = NewTree();
        // Offline (no server running yet owns the store): the CLI encodes the tree's chunks into the store.
        var built = Cli("build", "--model", _model, "--store", store, "--directory", tree, "--offline", "--json");
        Assert.True(built.Code == 0, built.Err);
        // A new file the store has not seen.
        File.WriteAllText(Path.Combine(tree, "src", "Fresh.cs"), "namespace Demo;\n\npublic static class Fresh\n{\n    public static int Zero() => 0;\n}\n");

        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(store));
        var r = Cli("select", "--model", _model, "--store", store, "--directory", tree, "--path", "src/*.cs", "--server", server.Http.BaseAddress!.ToString(), "--json");
        Assert.True(r.Code == 0, r.Err);
        JsonElement[] chunks = JsonDocument.Parse(r.Out).RootElement.GetProperty("chunks").EnumerateArray().Select(c => c.Clone()).ToArray();
        JsonElement[] calc = chunks.Where(c => c.GetProperty("path").GetString() == "src/Calc.cs").ToArray();
        JsonElement[] fresh = chunks.Where(c => c.GetProperty("path").GetString() == "src/Fresh.cs").ToArray();
        Assert.NotEmpty(calc);
        Assert.NotEmpty(fresh);
        Assert.All(calc, c => Assert.True(c.GetProperty("stored").GetBoolean(), r.Err)); // the server finds what the CLI built: hit
        Assert.All(fresh, c => Assert.False(c.GetProperty("stored").GetBoolean())); // and not what nobody built: miss
    }

    [Fact]
    public async Task Explain_Verb_PrintsTheServersPlan()
    {
        string store = NewDir("store");
        await using var server = await ChunkCacheTestServer.StartAsync(_model, On(store));
        string user = "Look at this.\n" + "public static int Add(int a, int b)\n{\n    return a + b;\n}\n" + "What does Add return?";
        string chunk = "public static int Add(int a, int b)\n{\n    return a + b;\n}\n";
        var cc = new JsonObject { ["v"] = 1, ["policy"] = "exact", ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("add", 0, user, chunk)) };
        string request = Path.Combine(NewDir("in"), "request.json");
        File.WriteAllText(request, ChunkCacheTestServer.OpenAi([("user", user)], cc));

        var r = Cli("explain", "--server", server.Http.BaseAddress!.ToString(), "--request", request, "--json");
        Assert.True(r.Code == 0, r.Err);
        JsonElement doc = JsonDocument.Parse(r.Out).RootElement;
        Assert.Equal("exact", doc.GetProperty("policy_would_apply").GetString());
        JsonElement c = Assert.Single(doc.GetProperty("chunks").EnumerateArray());
        Assert.Equal("add", c.GetProperty("id").GetString());
        Assert.Equal("miss", c.GetProperty("status").GetString()); // nothing stored yet
        Assert.True(doc.GetProperty("predicted_ms").GetProperty("full_prefill").GetDouble() > 0);
    }
}
