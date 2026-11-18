using System.Diagnostics;
using System.IO.Pipes;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkCache;
using DotLLM.ChunkReuse.Store;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase I Task 18: <c>chunkcache mcp</c>, the thin MCP server. Every test talks to the server in MCP's stdio framing (one
/// JSON-RPC message per line, as the bytes a client writes), either through a pair of OS pipes to the in-process server or
/// to a real <c>dotnet chunkcache.dll mcp</c> child process. Server calls of the tools go to a per-test fake
/// <see cref="HttpMessageHandler"/> handed to the host as an instance option (no static hook).
/// </summary>
public sealed class ChunkCacheMcpTests : IClassFixture<ChunkCacheCliTests.Fixture>
{
    private const string FakeUrl = "http://fake.test:8080";
    private readonly ChunkCacheCliTests.Fixture _fx;

    public ChunkCacheMcpTests(ChunkCacheCliTests.Fixture fx) => _fx = fx;

    // ------------------------------------------------------------------------------------------------ tools/list

    [Fact]
    public async Task ToolsList_HasSixTools_WithSchemas()
    {
        await using McpWire wire = await McpWire.StartAsync([]);
        JsonElement list = (await wire.CallAsync("tools/list", "{}")).GetProperty("result");
        JsonElement[] tools = list.GetProperty("tools").EnumerateArray().ToArray();

        Assert.Equal(
            new[] { "chunkcache_build", "chunkcache_evict", "chunkcache_explain", "chunkcache_prewarm", "chunkcache_select", "chunkcache_status" },
            tools.Select(t => t.GetProperty("name").GetString()!).Order(StringComparer.Ordinal).ToArray());
        foreach (JsonElement tool in tools)
        {
            string name = tool.GetProperty("name").GetString()!;
            Assert.False(string.IsNullOrWhiteSpace(tool.GetProperty("description").GetString()), name);
            JsonElement input = tool.GetProperty("inputSchema");
            Assert.Equal("object", input.GetProperty("type").GetString());
            Assert.Equal(JsonValueKind.False, input.GetProperty("additionalProperties").ValueKind); // unknown arguments are refused
            // The advertised outputSchema is the §7.2 schema of the same verb (the test fixture is the contract's copy) with
            // additionalProperties: false at every object that lists its properties.
            JsonElement output = tool.GetProperty("outputSchema");
            string verb = name["chunkcache_".Length..];
            JsonNode expected = JsonNode.Parse(File.ReadAllText(SchemaFixture(verb)))!;
            Assert.Equal(PropertyNames(expected), PropertyNames(JsonNode.Parse(output.GetRawText())!));
            Assert.Equal(JsonValueKind.False, output.GetProperty("additionalProperties").ValueKind);
        }
        JsonElement evict = tools.Single(t => t.GetProperty("name").GetString() == "chunkcache_evict");
        Assert.True(evict.GetProperty("inputSchema").GetProperty("properties").GetProperty("dry_run").GetProperty("default").GetBoolean());
        JsonElement build = tools.Single(t => t.GetProperty("name").GetString() == "chunkcache_build");
        Assert.False(build.GetProperty("inputSchema").GetProperty("properties").GetProperty("encode").GetProperty("default").GetBoolean());
        Assert.Equal("path", build.GetProperty("inputSchema").GetProperty("required")[0].GetString());
    }

    // ------------------------------------------------------------------------------------------------ one result = the CLI's result

    [Fact]
    public async Task EachTool_StructuredContentEqualsCliJson()
    {
        string tree = NewTree();
        string store = NewDir("store");
        string texts = Path.Combine(NewDir("in"), "texts.json");
        File.WriteAllText(texts, """[{"id": "a", "text": "class A { }"}]""");
        string request = Path.Combine(Path.GetDirectoryName(texts)!, "request.json");
        File.WriteAllText(request, """{"model": "m", "messages": [{"role": "user", "content": "hi"}]}""");
        var fake = new FakeServer(s => (s.Method, s.Path) switch
        {
            ("POST", "/v1/chunk-cache/prewarm") => (HttpStatusCode.OK, """{"requested":1,"already_stored":0,"queued":1,"queued_tokens":10,"refused":0,"refused_reasons":[]}"""),
            ("POST", "/v1/chunk-cache/evict") => (HttpStatusCode.OK, """{"evicted":2,"freed_bytes":1024,"kept_referenced":3,"dry_run":true}"""),
            ("POST", "/v1/chunk-cache/explain") => (HttpStatusCode.OK, """{"policy_would_apply":"exact","reason":null,"chunks":[{"id":"a","status":"hit","why":null}],"predicted_ms":{"reuse":1.5,"full_prefill":3.0},"warnings":[]}"""),
            _ => (HttpStatusCode.NotFound, "{}"),
        });
        await using McpWire wire = await McpWire.StartAsync(["--store", store, "--model", _fx.ModelA], fake);
        string treeJson = JsonSerializer.Serialize(tree);

        // (tool, MCP arguments, equivalent CLI command line)
        var cases = new (string Tool, string Args, string[] Cli, string[] Dropped)[]
        {
            ("build", $$"""{"path": {{treeJson}}}""",
                ["build", "--model", _fx.ModelA, "--store", NewDir("cli-store"), "--directory", tree, "--index-only", "--json"], ["mode", "store", "root"]),
            ("status", $$"""{"path": {{treeJson}}}""",
                ["status", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--json"], ["index_path", "scanned", "ledger"]),
            ("select", $$"""{"path": {{treeJson}}, "paths": ["src/Calc.cs"], "include_text": true}""",
                ["select", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--path", "src/Calc.cs", "--include-text", "--json"], ["order"]),
            ("evict", """{"server": "http://fake.test:8080", "mode": "orphans"}""",
                ["evict", "--server", FakeUrl, "--mode", "orphans", "--json"], ["mode", "server"]),
            ("prewarm", $$"""{"server": "http://fake.test:8080", "texts": [{"id": "a", "text": "class A { }"}]}""",
                ["prewarm", "--server", FakeUrl, "--texts", texts, "--json"], ["requests", "skipped_ineligible"]),
            ("explain", """{"server": "http://fake.test:8080", "request": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}}""",
                ["explain", "--server", FakeUrl, "--request", request, "--json"], ["server"]),
        };
        foreach ((string tool, string args, string[] cli, string[] dropped) in cases)
        {
            JsonElement result = await wire.CallToolAsync("chunkcache_" + tool, args);
            Assert.False(result.TryGetProperty("isError", out JsonElement e) && e.GetBoolean(), $"{tool}: {result}");

            var o = new StringWriter();
            int code = ChunkCacheCli.Run(cli, o, new StringWriter(), _fx.Scratch, fake, _ => null);
            Assert.Equal(0, code);
            JsonNode cliJson = JsonNode.Parse(o.ToString())!;
            JsonNode expected = Project(cliJson, JsonNode.Parse(File.ReadAllText(SchemaFixture(tool)))!);
            JsonNode actual = JsonNode.Parse(result.GetProperty("structuredContent").GetRawText())!;
            if (tool == "build") { expected["elapsed_ms"] = 0; actual["elapsed_ms"] = 0; } // wall clock

            Assert.Equal(expected.ToJsonString(), actual.ToJsonString());
            // The text content is the same document, serialised.
            JsonElement text = result.GetProperty("content")[0];
            Assert.Equal("text", text.GetProperty("type").GetString());
            JsonNode textJson = JsonNode.Parse(text.GetProperty("text").GetString()!)!;
            if (tool == "build") textJson["elapsed_ms"] = 0;
            Assert.Equal(actual.ToJsonString(), textJson.ToJsonString());
            // Projection onto the schema: CLI-only fields do not reach the client (control: the CLI does print them).
            foreach (string extra in dropped)
            {
                if (cliJson.AsObject().ContainsKey(extra))
                    Assert.False(actual.AsObject().ContainsKey(extra), $"{tool}: CLI-only field {extra} leaked");
            }
            Assert.True(dropped.Any(d => cliJson.AsObject().ContainsKey(d)), $"{tool}: control, the CLI prints an extra field");
        }
    }

    // ------------------------------------------------------------------------------------------------ defaults that protect

    [Fact]
    public async Task Build_DefaultIsIndexOnly()
    {
        string tree = NewTree();
        string store = NewDir("store");
        using ModelHandle model = OpenModel();
        var fake = PrewarmingServer(model);
        await using McpWire wire = await McpWire.StartAsync(["--store", store, "--model", _fx.ModelA, "--server", FakeUrl], fake);

        JsonElement index = await wire.CallToolAsync("chunkcache_build", $$"""{"path": {{JsonSerializer.Serialize(tree)}}}""");
        Assert.False(IsError(index), index.ToString());
        Assert.Equal(0, index.GetProperty("structuredContent").GetProperty("encode_queued_tokens").GetInt32());
        Assert.True(index.GetProperty("structuredContent").GetProperty("chunks").GetInt32() > 0);
        Assert.DoesNotContain(fake.Requests, r => r.Method == "POST"); // nothing was queued on the server
        Assert.Empty(new FileChunkStore(store).Keys());               // nothing was encoded here either
        Assert.NotEmpty(Directory.GetFiles(store, "index-*.json"));   // the index was written

        // Control: encode: true queues the missing chunks on the server (and only then); max_encode_tokens bounds the tokens.
        JsonElement all = await wire.CallToolAsync("chunkcache_build", $$"""{"path": {{JsonSerializer.Serialize(tree)}}, "encode": true}""");
        Assert.False(IsError(all), all.ToString());
        long queuedAll = all.GetProperty("structuredContent").GetProperty("encode_queued_tokens").GetInt64();
        Assert.True(queuedAll > 100);
        Assert.Contains(fake.Requests, r => r is { Method: "POST", Path: "/v1/chunk-cache/prewarm" });

        string store2 = NewDir("store2");
        await using McpWire wire2 = await McpWire.StartAsync(["--store", store2, "--model", _fx.ModelA, "--server", FakeUrl], PrewarmingServer(model));
        long budget = queuedAll - 1; // a strict token cap: every chunk but the last fits, so a chunk-count reading would queue them all
        JsonElement capped = await wire2.CallToolAsync("chunkcache_build", $$"""{"path": {{JsonSerializer.Serialize(tree)}}, "encode": true, "max_encode_tokens": {{budget}}}""");
        Assert.False(IsError(capped), capped.ToString());
        long queued = capped.GetProperty("structuredContent").GetProperty("encode_queued_tokens").GetInt64();
        Assert.InRange(queued, 1, budget); // a strict budget in tokens, never a chunk count
        Assert.True(queued < queuedAll);
    }

    [Fact]
    public async Task Evict_DefaultsToDryRun()
    {
        var fake = new FakeServer(s => s is { Method: "POST", Path: "/v1/chunk-cache/evict" }
            ? (HttpStatusCode.OK, """{"evicted":1,"freed_bytes":5,"kept_referenced":0}""")
            : (HttpStatusCode.NotFound, "{}"));
        await using McpWire wire = await McpWire.StartAsync(["--server", FakeUrl], fake);

        JsonElement defaulted = await wire.CallToolAsync("chunkcache_evict", "{}");
        Assert.False(IsError(defaulted), defaulted.ToString());
        Assert.True(defaulted.GetProperty("structuredContent").GetProperty("dry_run").GetBoolean());
        JsonNode sent = JsonNode.Parse(Assert.Single(fake.Requests, r => r.Method == "POST").Body)!;
        Assert.True(sent["dry_run"]!.GetValue<bool>());
        Assert.Equal("orphans", sent["mode"]!.GetValue<string>());

        // Control: an explicit dry_run: false is what turns the deletion on.
        JsonElement applied = await wire.CallToolAsync("chunkcache_evict", """{"dry_run": false}""");
        Assert.False(IsError(applied), applied.ToString());
        JsonNode sent2 = JsonNode.Parse(fake.Requests.Last(r => r.Method == "POST").Body)!;
        Assert.False(sent2["dry_run"]!.GetValue<bool>());
    }

    [Fact]
    public async Task Evict_Offline_DryRunDeletesNothing_ApplyNeedsExplicitFalse()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, ChunkCacheCli.Run(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--offline", "--chunk-tokens", "64", "--lab", "--json"],
            new StringWriter(), new StringWriter(), _fx.Scratch, null, _ => null));
        int before = new FileChunkStore(store).Keys().Count();
        Assert.True(before > 0);
        await using McpWire wire = await McpWire.StartAsync(["--store", store]);

        JsonElement dry = await wire.CallToolAsync("chunkcache_evict", "{}");
        Assert.False(IsError(dry), dry.ToString());
        Assert.True(dry.GetProperty("structuredContent").GetProperty("dry_run").GetBoolean());
        Assert.Equal(before, new FileChunkStore(store).Keys().Count());
    }

    // ------------------------------------------------------------------------------------------------ errors

    [Fact]
    public async Task UnknownArgument_RejectedBySchema()
    {
        string store = NewDir("store");
        var fake = new FakeServer(_ => (HttpStatusCode.OK, "{}"));
        await using McpWire wire = await McpWire.StartAsync(["--store", store, "--model", _fx.ModelA], fake);
        string tree = JsonSerializer.Serialize(NewTree());

        foreach ((string tool, string args, string needle) in new[]
        {
            ("chunkcache_build", $$"""{"path": {{tree}}, "frobnicate": true}""", "frobnicate"),
            ("chunkcache_evict", """{"all_for_model": true}""", "all_for_model"),
            ("chunkcache_status", """{"list_missing": "yes"}""", "list_missing"),       // wrong type
            ("chunkcache_evict", """{"mode": "everything"}""", "mode"),                  // not in the enum (and old CLI names are not tool modes)
            ("chunkcache_build", """{}""", "path"),                                      // required
            ("chunkcache_select", """{"hops": 101}""", "hops"),                          // maximum 100
            ("chunkcache_prewarm", """{"select": {"nope": 1}, "server": "http://x"}""", "nope"),
        })
        {
            JsonElement result = await wire.CallToolAsync(tool, args);
            Assert.True(IsError(result), $"{tool} {args}: {result}");
            Assert.Contains(needle, ErrorText(result), StringComparison.Ordinal);
        }
        Assert.Empty(fake.Requests);                                   // refused before anything ran
        Assert.Empty(Directory.EnumerateFileSystemEntries(store));      // and nothing was written
        Assert.Equal(6, (await wire.CallAsync("tools/list", "{}")).GetProperty("result").GetProperty("tools").GetArrayLength()); // the server still answers
    }

    [Fact]
    public async Task ToolError_IsResultWithIsError()
    {
        string store = Path.Combine(NewDir("none"), "no-such-store");
        await using McpWire wire = await McpWire.StartAsync(["--model", _fx.ModelA]);

        JsonElement response = await wire.CallAsync("tools/call", $$$"""{"name": "chunkcache_status", "arguments": {"store": {{{JsonSerializer.Serialize(store)}}}}}""");
        Assert.False(response.TryGetProperty("error", out _), "a tool failure is not a JSON-RPC error");
        JsonElement result = response.GetProperty("result");
        Assert.True(IsError(result));
        JsonElement err = JsonDocument.Parse(result.GetProperty("content")[0].GetProperty("text").GetString()!).RootElement.GetProperty("error");
        Assert.Equal("error", err.GetProperty("code").GetString());
        Assert.Contains("Store not found", err.GetProperty("message").GetString(), StringComparison.Ordinal);
        Assert.False(result.TryGetProperty("structuredContent", out JsonElement sc) && sc.ValueKind == JsonValueKind.Object);

        // A usage error of the verb (no store anywhere) is the same shape; the server lives on and answers the next call.
        JsonElement usage = await wire.CallToolAsync("chunkcache_status", "{}");
        Assert.True(IsError(usage));
        Assert.Equal("usage", JsonDocument.Parse(ErrorText(usage)).RootElement.GetProperty("error").GetProperty("code").GetString());
        // A tool that does not exist is a protocol error (unlike a failing tool).
        JsonElement unknown = await wire.CallAsync("tools/call", """{"name": "chunkcache_compare", "arguments": {}}""");
        Assert.True(unknown.TryGetProperty("error", out _));
        Assert.Equal(6, (await wire.CallAsync("tools/list", "{}")).GetProperty("result").GetProperty("tools").GetArrayLength());
    }

    [Fact]
    public async Task ApiKey_NeverOnTheWire()
    {
        const string key = "sk-test-9f8e7d6c5b4a39281716";
        var fake = new FakeServer(_ => (HttpStatusCode.InternalServerError, $$$"""{"error":{"message":"bad key {{{key}}}"}}"""));
        await using McpWire wire = await McpWire.StartAsync(["--server", FakeUrl], fake, env: n => n == "CHUNKCACHE_API_KEY" ? key : null);
        JsonElement result = await wire.CallToolAsync("chunkcache_evict", "{}");
        Assert.True(IsError(result));
        Assert.Equal(key, Assert.Single(fake.Requests).Headers["X-API-Key"]); // control: the key was used
        Assert.DoesNotContain(key, string.Join('\n', wire.AllFrames), StringComparison.Ordinal);
        Assert.DoesNotContain(key, wire.Stderr, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ the real process

    [Fact]
    public async Task RealProcess_StdioFraming_ListsAndCalls()
    {
        string dll = Path.Combine(AppContext.BaseDirectory, "chunkcache.dll");
        Assert.True(File.Exists(dll), dll);
        string store = NewDir("store");
        var psi = new ProcessStartInfo("dotnet")
        {
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            StandardOutputEncoding = new UTF8Encoding(false), StandardErrorEncoding = new UTF8Encoding(false), WorkingDirectory = _fx.Scratch,
        };
        psi.ArgumentList.Add(dll);
        psi.ArgumentList.Add("mcp");
        psi.ArgumentList.Add("--store");
        psi.ArgumentList.Add(store);
        using Process p = Process.Start(psi)!;
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        try
        {
            async Task<JsonElement> Roundtrip(string frame, int id)
            {
                await p.StandardInput.WriteAsync(frame + "\n"); // the bytes of a client, one message per line
                await p.StandardInput.FlushAsync(timeout.Token);
                while (true)
                {
                    string line = await p.StandardOutput.ReadLineAsync(timeout.Token) ?? throw new EndOfStreamException("server closed stdout: " + await p.StandardError.ReadToEndAsync(timeout.Token));
                    JsonElement msg = JsonDocument.Parse(line).RootElement.Clone(); // stdout carries only JSON-RPC frames
                    Assert.Equal("2.0", msg.GetProperty("jsonrpc").GetString());
                    if (msg.TryGetProperty("id", out JsonElement mid) && mid.GetInt32() == id) return msg;
                }
            }
            JsonElement init = await Roundtrip("""{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"wire-test","version":"1"}}}""", 1);
            Assert.Equal("chunkcache", init.GetProperty("result").GetProperty("serverInfo").GetProperty("name").GetString());
            Assert.True(init.GetProperty("result").GetProperty("capabilities").TryGetProperty("tools", out _));
            await p.StandardInput.WriteAsync("""{"jsonrpc":"2.0","method":"notifications/initialized"}""" + "\n");
            JsonElement tools = await Roundtrip("""{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}""", 2);
            Assert.Equal(6, tools.GetProperty("result").GetProperty("tools").GetArrayLength());
            // A real tool run in the child: status of an empty store is a tool error (no index), as a result, with the progress text off stdout.
            JsonElement call = await Roundtrip("""{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chunkcache_evict","arguments":{}}}""", 3);
            Assert.True(call.GetProperty("result").GetProperty("isError").GetBoolean()); // no --server and no store layout to evict from
        }
        finally
        {
            p.StandardInput.Close();
            if (!p.WaitForExit(15000)) p.Kill(entireProcessTree: true);
        }
        Assert.Equal(0, p.ExitCode); // EOF on stdin ends the server cleanly
    }

    [Fact]
    public async Task Mcp_Help_AndUsageErrors()
    {
        var o = new StringWriter();
        var e = new StringWriter();
        Assert.Equal(0, await McpCommand.RunAsync(["--help"], Stream.Null, Stream.Null, e, o, _fx.Scratch, null, _ => null, CancellationToken.None));
        Assert.Contains("chunkcache mcp", o.ToString(), StringComparison.Ordinal);
        Assert.Contains("--store", o.ToString(), StringComparison.Ordinal);
        var e2 = new StringWriter();
        Assert.Equal(2, await McpCommand.RunAsync(["--nope"], Stream.Null, Stream.Null, e2, new StringWriter(), _fx.Scratch, null, _ => null, CancellationToken.None));
        Assert.Contains("--nope", e2.ToString(), StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private static bool IsError(JsonElement result) => result.TryGetProperty("isError", out JsonElement e) && e.ValueKind == JsonValueKind.True;

    private static string ErrorText(JsonElement result) => result.GetProperty("content")[0].GetProperty("text").GetString()!;

    private static string SchemaFixture(string verb) =>
        Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1", "schemas", verb + ".json");

    private static List<string> PropertyNames(JsonNode schema)
    {
        var names = new List<string>();
        void Walk(JsonNode? n, string path)
        {
            if (n is JsonObject o)
            {
                if (o["properties"] is JsonObject props)
                    foreach ((string k, JsonNode? v) in props) { names.Add(path + "." + k); Walk(v, path + "." + k); }
                if (o["items"] is { } items) Walk(items, path + "[]");
            }
        }
        Walk(schema, "$");
        names.Sort(StringComparer.Ordinal);
        return names;
    }

    /// <summary>Test-side projection of a CLI document onto a schema: only the properties the schema lists, recursively.</summary>
    private static JsonNode Project(JsonNode value, JsonNode schema)
    {
        if (value is JsonObject obj && schema["properties"] is JsonObject props)
        {
            var result = new JsonObject();
            foreach ((string k, JsonNode? v) in obj)
                if (props[k] is { } sub && v is not null) result[k] = Project(v, sub);
            return result;
        }
        if (value is JsonArray arr && schema["items"] is { } items)
        {
            var result = new JsonArray();
            foreach (JsonNode? v in arr) result.Add(v is null ? null : Project(v, items));
            return result;
        }
        return value.DeepClone();
    }

    private ModelHandle OpenModel() => ModelHandle.Open(new CliContext(TextWriter.Null, TextWriter.Null, _fx.Scratch), _fx.ModelA, fullHash: false, threads: 1);

    private string NewDir(string name)
    {
        string d = Path.Combine(_fx.Scratch, name + "-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(d);
        return d;
    }

    private string NewTree()
    {
        string d = NewDir("tree");
        foreach (string f in Directory.EnumerateFiles(_fx.TreeTemplate, "*", SearchOption.AllDirectories))
        {
            string t = Path.Combine(d, Path.GetRelativePath(_fx.TreeTemplate, f));
            Directory.CreateDirectory(Path.GetDirectoryName(t)!);
            File.Copy(f, t);
        }
        return d;
    }

    private static string CapsJson(ModelHandle model)
    {
        JsonObject caps = JsonNode.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1", "capabilities.json")))!.AsObject();
        JsonObject m = caps["model"]!.AsObject();
        m["id"] = model.ModelId;
        m["producer"] = model.Producer;
        m["tokenizer_id"] = "tok-test";
        return caps.ToJsonString();
    }

    /// <summary>A server with the model's identity that accepts prewarm texts and answers with the tokens it was sent.</summary>
    private static FakeServer PrewarmingServer(ModelHandle model) => new(s =>
    {
        if (s is { Method: "GET", Path: "/v1/chunk-cache" }) return (HttpStatusCode.OK, CapsJson(model));
        if (s is { Method: "POST", Path: "/v1/chunk-cache/prewarm" })
        {
            JsonArray texts = JsonNode.Parse(s.Body)!["texts"]!.AsArray();
            long tokens = texts.Sum(t => DotLLM.ChunkReuse.CodeCorpus.ChunkTokenization.Encode(model.Tokenizer, t!["text"]!.GetValue<string>()).Length);
            return (HttpStatusCode.OK, $$"""{"requested":{{texts.Count}},"already_stored":0,"queued":{{texts.Count}},"queued_tokens":{{tokens}},"refused":0,"refused_reasons":[]}""");
        }
        return (HttpStatusCode.NotFound, "{}");
    });

    private sealed record Seen(string Method, string Path, Dictionary<string, string> Headers, string Body);

    private sealed class FakeServer(Func<Seen, (HttpStatusCode Status, string Body)> answer) : HttpMessageHandler
    {
        public List<Seen> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
            var headers = request.Headers.ToDictionary(h => h.Key, h => string.Join(",", h.Value), StringComparer.OrdinalIgnoreCase);
            var seen = new Seen(request.Method.Method, request.RequestUri!.AbsolutePath, headers, body);
            lock (Requests) Requests.Add(seen);
            (HttpStatusCode status, string text) = answer(seen);
            return new HttpResponseMessage(status) { Content = new StringContent(text, Encoding.UTF8, "application/json") };
        }
    }

    /// <summary>
    /// The in-process server behind two OS pipes; the test writes and reads the frames as bytes (UTF-8, one JSON-RPC message per
    /// line), exactly what a stdio client does. Everything the server wrote is kept in <see cref="AllFrames"/>.
    /// </summary>
    private sealed class McpWire : IAsyncDisposable
    {
        private readonly AnonymousPipeServerStream _toServer = new(PipeDirection.Out);
        private readonly AnonymousPipeServerStream _fromServer = new(PipeDirection.In);
        private readonly AnonymousPipeClientStream _serverIn;
        private readonly AnonymousPipeClientStream _serverOut;
        private readonly StreamReader _reader;
        private readonly CancellationTokenSource _cts = new(TimeSpan.FromMinutes(5));
        private readonly StringWriter _stderr = new();
        private Task<int> _run = Task.FromResult(-1);
        private int _id = 100;

        private McpWire()
        {
            _serverIn = new AnonymousPipeClientStream(PipeDirection.In, _toServer.ClientSafePipeHandle);
            _serverOut = new AnonymousPipeClientStream(PipeDirection.Out, _fromServer.ClientSafePipeHandle);
            _reader = new StreamReader(_fromServer, new UTF8Encoding(false));
        }

        public List<string> AllFrames { get; } = [];

        public string Stderr => _stderr.ToString();

        public static async Task<McpWire> StartAsync(string[] args, HttpMessageHandler? http = null, Func<string, string?>? env = null)
        {
            var wire = new McpWire();
            string[] full = [.. args];
            wire._run = Task.Run(() => McpCommand.RunAsync(full, wire._serverIn, wire._serverOut, wire._stderr, new StringWriter(), Path.GetTempPath(), http, env ?? (_ => null), wire._cts.Token));
            JsonElement init = await wire.CallAsync("initialize",
                """{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"wire-test","version":"1"}}""", 1);
            Assert.Equal("chunkcache", init.GetProperty("result").GetProperty("serverInfo").GetProperty("name").GetString());
            await wire.SendAsync("""{"jsonrpc":"2.0","method":"notifications/initialized"}""");
            return wire;
        }

        private async Task SendAsync(string frame)
        {
            Assert.DoesNotContain('\n', frame);
            byte[] bytes = Encoding.UTF8.GetBytes(frame + "\n");
            await _toServer.WriteAsync(bytes, _cts.Token);
            await _toServer.FlushAsync(_cts.Token);
        }

        public async Task<JsonElement> CallAsync(string method, string paramsJson, int? id = null)
        {
            int myId = id ?? Interlocked.Increment(ref _id);
            await SendAsync($$"""{"jsonrpc":"2.0","id":{{myId}},"method":"{{method}}","params":{{paramsJson}}}""");
            while (true)
            {
                string line = await _reader.ReadLineAsync(_cts.Token) ?? throw new EndOfStreamException("the server closed its output; stderr: " + Stderr);
                AllFrames.Add(line);
                JsonElement msg = JsonDocument.Parse(line).RootElement.Clone(); // every line of stdout is a JSON-RPC message
                Assert.Equal("2.0", msg.GetProperty("jsonrpc").GetString());
                if (msg.TryGetProperty("id", out JsonElement mid) && mid.ValueKind == JsonValueKind.Number && mid.GetInt32() == myId) return msg;
            }
        }

        public async Task<JsonElement> CallToolAsync(string name, string argumentsJson)
        {
            JsonElement response = await CallAsync("tools/call", $$"""{"name": "{{name}}", "arguments": {{argumentsJson}}}""");
            Assert.False(response.TryGetProperty("error", out JsonElement err), "protocol error: " + err);
            return response.GetProperty("result");
        }

        public async ValueTask DisposeAsync()
        {
            _toServer.Dispose(); // EOF on the server's stdin: it ends
            try { await _run.WaitAsync(TimeSpan.FromSeconds(20)); }
            catch (TimeoutException) { _cts.Cancel(); }
            catch (OperationCanceledException) { }
            _reader.Dispose();
            _stderr.Dispose();
            _fromServer.Dispose();
            _serverIn.Dispose();
            _serverOut.Dispose();
            _cts.Dispose();
        }
    }
}
