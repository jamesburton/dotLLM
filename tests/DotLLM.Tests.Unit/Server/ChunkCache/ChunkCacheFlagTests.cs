using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.Cli.Commands;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Spectre.Console.Cli;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>The copied wire-contract fixtures are byte-identical to the contract set (ChunkedkvCache <c>docs/contracts/chunk-cache-v1/</c>).</summary>
public sealed class ContractFixtureTests
{
    /// <summary>
    /// SHA-256 of <c>fixtures.sha256</c> itself, so a fixture edited together with its list entry still fails here.
    /// Changing it means the contract changed: re-copy the whole set from the contract folder.
    /// </summary>
    private const string ListSha256 = "b93229171a5179e488a96d2d2605b5a5cce641e24ee7816a1adf22a50b61511c";

    [Fact]
    public void FixtureList_IsTheContractList() =>
        Assert.Equal(ListSha256, Sha256Hex(ContractFixtures.Bytes("fixtures.sha256")));

    [Fact]
    public void FixtureCopy_MatchesContractSha256()
    {
        var listed = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(Path.Combine(ContractFixtures.Dir, "fixtures.sha256")).Where(l => l.Length > 0))
        {
            string[] f = line.Split("  ", 2);
            listed.Add(f[1], f[0]);
        }

        Assert.Equal(12, listed.Count);
        foreach ((string name, string hex) in listed)
            Assert.True(string.Equals(hex, Sha256Hex(ContractFixtures.Bytes(name)), StringComparison.Ordinal), $"{name}: bytes differ from the contract");

        string[] onDisk = Directory.GetFiles(ContractFixtures.Dir, "*.json").Select(Path.GetFileName).OfType<string>()
            .Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(listed.Keys.Order(StringComparer.Ordinal), onDisk);
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
}

/// <summary><c>--chunk-cache</c> and <c>--chunk-cache-max-chunks</c> on both option paths.</summary>
[Collection("MaxMessagesEnv")] // ServerOptions.Parse reads DOTLLM_MAX_MESSAGES
public sealed class ChunkCacheOptionsTests
{
    [Fact]
    public void Default_IsOff_With4096()
    {
        Assert.Equal(ChunkCacheOptions.Disabled, new ServerOptions { Model = "m.gguf" }.ChunkCache);
        Assert.False(ChunkCacheOptions.Disabled.Enabled);
        Assert.Equal(4096, ChunkCacheOptions.Disabled.MaxChunks);
        Assert.Equal(ChunkCacheOptions.Disabled, ServerOptions.Parse(["--model", "m.gguf"]).ChunkCache);
    }

    [Fact]
    public void Parse_Flags()
    {
        Assert.Equal(new ChunkCacheOptions(true, 4096), ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache"]).ChunkCache);
        Assert.Equal(new ChunkCacheOptions(false, 7), ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache-max-chunks", "7"]).ChunkCache);
        Assert.Equal(new ChunkCacheOptions(true, 1), ServerOptions.Parse(["--chunk-cache-max-chunks", "1", "--chunk-cache", "--model", "m.gguf"]).ChunkCache);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void MaxChunks_BelowOne_Rejected(int max)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ChunkCacheOptions.Create(true, max));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache-max-chunks", max.ToString(System.Globalization.CultureInfo.InvariantCulture)]));
        Assert.False(new ServeCommand.Settings { ChunkCacheMaxChunks = max }.Validate().Successful);
    }

    /// <summary>#1632: <c>--chunk-cache-max-chunks</c> has an upper bound too, so the per-request work stays bounded.</summary>
    [Fact]
    public void MaxChunks_AboveTheCap_Rejected()
    {
        int over = ChunkCacheOptions.AbsoluteMaxChunks + 1;
        Assert.Equal(ChunkCacheOptions.AbsoluteMaxChunks, ChunkCacheOptions.Create(true, ChunkCacheOptions.AbsoluteMaxChunks).MaxChunks);
        Assert.Throws<ArgumentOutOfRangeException>(() => ChunkCacheOptions.Create(true, over));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache-max-chunks", over.ToString(System.Globalization.CultureInfo.InvariantCulture)]));
        Assert.False(new ServeCommand.Settings { ChunkCacheMaxChunks = over }.Validate().Successful);
        Assert.True(new ServeCommand.Settings { ChunkCache = true, ChunkCacheMaxChunks = ChunkCacheOptions.AbsoluteMaxChunks }.Validate().Successful);
    }

    /// <summary><c>--chunk-store</c> (Task 13, read-only lookups) on both option paths; blank means none.</summary>
    [Fact]
    public void ChunkStore_ParsedOnBothPaths()
    {
        Assert.Null(ChunkCacheOptions.Disabled.StoreRoot);
        // Resolved to a full path at parse time (not lazily against a later working directory).
        Assert.Equal(new ChunkCacheOptions(true, 4096, Path.GetFullPath("s")), ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-store", "s"]).ChunkCache);
        Assert.Null(ChunkCacheOptions.Create(true, null, "  ").StoreRoot);
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-cache", "--chunk-store"]));
        // A store without the flag is a configuration error, not silently ignored.
        Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--chunk-store", "s"]));
        Assert.False(new ServeCommand.Settings { ChunkStore = "s" }.Validate().Successful);
        Assert.True(new ServeCommand.Settings { ChunkStore = "s", ChunkCache = true }.Validate().Successful);
        var names = typeof(ServeCommand.Settings).GetProperties()
            .Select(p => p.GetCustomAttribute<CommandOptionAttribute>())
            .Where(a => a is not null).SelectMany(a => a!.LongNames).ToHashSet(StringComparer.Ordinal);
        Assert.Contains("chunk-store", names);
    }

    [Fact]
    public void ServeCommand_DeclaresBothFlags_AndValidatesDefaults()
    {
        var names = typeof(ServeCommand.Settings).GetProperties()
            .Select(p => p.GetCustomAttribute<CommandOptionAttribute>())
            .Where(a => a is not null).SelectMany(a => a!.LongNames).ToHashSet(StringComparer.Ordinal);
        Assert.Contains("chunk-cache", names);
        Assert.Contains("chunk-cache-max-chunks", names);
        Assert.True(new ServeCommand.Settings().Validate().Successful);
        Assert.True(new ServeCommand.Settings { ChunkCache = true, ChunkCacheMaxChunks = 1 }.Validate().Successful);
    }
}

/// <summary>
/// Flag off: an annotated body behaves byte for byte like the same body without annotations, over real HTTP, for
/// valid, defective, strict and malformed annotations alike. OpenAI array content stays the binding 400 with the flag
/// on and off (ruling 19). Same harness as <see cref="ServerPinHttpTests"/>: a 1-token context makes every chat request
/// that reaches the handler stop at the prompt-length 400, whose body carries the prompt's token count.
/// </summary>
public sealed class ChunkCacheFlagHttpTests : IAsyncLifetime
{
    private readonly List<(WebApplication App, ServerState State, HttpClient Http)> _servers = [];

    private HttpClient Off => _servers[0].Http;
    private HttpClient OnServer => _servers[1].Http;

    public async Task InitializeAsync()
    {
        foreach (var options in new[] { ChunkCacheOptions.Disabled, new ChunkCacheOptions(true, 4096) })
        {
            var model = new ServerPinHttpTests.NonComputingModel(maxSeq: 1);
            var state = new ServerState
            {
                Options = new ServerOptions { Model = "test-model", ModelId = "test-model", ChunkCache = options },
                Model = model,
                Config = model.Config,
                Tokenizer = ServerPinTests.Tok,
                ChatTemplate = ServerPinTests.ServerTemplate(ServerPinTests.Qwen3Fixture),
                Generator = new TextGenerator(model, ServerPinTests.Tok),
                IsReady = true,
            };
            var builder = WebApplication.CreateSlimBuilder();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Services.AddDotLLM(state);
            var app = builder.Build();
            app.MapDotLLMEndpoints(serveUi: false);
            await app.StartAsync();
            _servers.Add((app, state, new HttpClient { BaseAddress = new Uri(app.Urls.First()) }));
        }
    }

    public async Task DisposeAsync()
    {
        foreach (var (app, state, http) in _servers)
        {
            http.Dispose();
            await app.DisposeAsync();
            state.Dispose();
        }
    }

    private static async Task<(HttpStatusCode Status, string Body)> PostAsync(HttpClient http, string path, string body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.Add("anthropic-version", "2023-06-01");
        using var resp = await http.SendAsync(request);
        return (resp.StatusCode, await resp.Content.ReadAsStringAsync());
    }

    private static readonly string BodyHash = ChunkAnnotationParserTests.Sha("hello world");

    /// <summary>chunk_cache values: valid, strict with a bad span, version 2, conflicting, and malformed shapes.</summary>
    private static readonly string[] Manifests =
    [
        $$$"""{"v":1,"policy":"shadow","chunks":[{"id":"a","hash":"{{{BodyHash}}}","at":{"message":0,"part":0}}]}""",
        """{"v":1,"strict":true,"chunks":[{"id":"a","hash":"sha256:00","at":{"message":0,"part":7,"start":3,"end":1}}]}""",
        """{"v":2,"strict":true}""",
        """{"chunks":"nope","strict":true}""",
        "5",
        "null",
        "[1,2]",
    ];

    [Fact]
    public async Task FlagOff_OpenAi_AnnotatedBodyIsByteIdenticalToPlain()
    {
        const string plain = """{"messages":[{"role":"user","content":"hello world"}]}""";
        var expected = await PostAsync(Off, "/v1/chat/completions", plain);
        Assert.Equal(HttpStatusCode.BadRequest, expected.Status);
        Assert.Contains("context_length_exceeded", expected.Body, StringComparison.Ordinal);

        foreach (string cc in Manifests)
        {
            string annotated = $$$"""{"messages":[{"role":"user","content":"hello world","chunk":{"v":1,"id":"x","hash":"sha256:00"}}],"chunk_cache":{{{cc}}}}""";
            Assert.Equal(expected, await PostAsync(Off, "/v1/chat/completions", annotated));
        }
    }

    [Fact]
    public async Task FlagOff_Anthropic_AnnotatedBodyIsByteIdenticalToPlain()
    {
        const string plain = """{"messages":[{"role":"user","content":[{"type":"text","text":"glue\n"},{"type":"text","text":"hello world"}]}]}""";
        var expected = await PostAsync(Off, "/v1/messages/count_tokens", plain);
        Assert.Equal(HttpStatusCode.OK, expected.Status);

        var inline = JsonNode.Parse(plain)!;
        inline["messages"]![0]!["content"]![1]!["chunk"] = new JsonObject { ["v"] = 1, ["id"] = "x", ["hash"] = BodyHash };
        Assert.Equal(expected, await PostAsync(Off, "/v1/messages/count_tokens", inline.ToJsonString()));

        foreach (string cc in Manifests)
        {
            var annotated = JsonNode.Parse(inline.ToJsonString())!; // inline + manifest: a conflict when the flag is on
            annotated["chunk_cache"] = JsonNode.Parse(cc);
            Assert.Equal(expected, await PostAsync(Off, "/v1/messages/count_tokens", annotated.ToJsonString()));
        }
    }

    /// <summary>Ruling 19: OpenAI array content is the same bare 400 (empty body) with the flag on as with it off.</summary>
    [Fact]
    public async Task OpenAi_ArrayContent_StillThe400_FlagOnAndOff()
    {
        string body = $$$"""
            {"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"hello world","chunk":{"v":1,"id":"x","hash":"{{{BodyHash}}}"}}]}],
             "chunk_cache":{"v":1}}
            """;
        foreach (var http in new[] { Off, OnServer })
        {
            var (status, text) = await PostAsync(http, "/v1/chat/completions", body);
            Assert.Equal(HttpStatusCode.BadRequest, status);
            Assert.Equal("", text);
        }
    }
}
