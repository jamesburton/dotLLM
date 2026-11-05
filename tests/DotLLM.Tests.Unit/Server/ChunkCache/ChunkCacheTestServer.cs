using System.Text;
using System.Text.Json.Nodes;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Tests.Unit.ChunkReuse;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// Runs every test that drives chunk-annotated requests through a real server one class at a time: they read the
/// process-wide <c>/metrics</c> chunk-cache counters, which a concurrent class would also move.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ChunkCacheServerCollection
{
    public const string Name = "ChunkCacheServer";
}

/// <summary>
/// A real dotLLM server over HTTP on loopback, loaded by <see cref="ServerStartup.LoadModel"/> from the synthetic
/// byte-level qwen3 GGUF (<see cref="SyntheticDenseGguf"/>) with the Qwen3 special tokens and chat template: real
/// tokenizer, template, generation, prefix cache (paged trie + scheduler, or the single-slot prefix cache), CPU only.
/// </summary>
internal sealed class ChunkCacheTestServer : IAsyncDisposable
{
    /// <summary>The special tokens the synthetic GGUF declares (Qwen3 chat framing and reasoning tags).</summary>
    internal static readonly string[] Specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"];

    private ChunkCacheTestServer(WebApplication app, ServerState state, HttpClient http)
    {
        App = app;
        State = state;
        Http = http;
    }

    public WebApplication App { get; }

    public ServerState State { get; }

    public HttpClient Http { get; }

    /// <summary>Writes the synthetic model to <paramref name="path"/> (once per path).</summary>
    public static string WriteModel(string path)
    {
        if (!File.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3", seed: 13, byteLevelVocab: true, specials: Specials,
                chatTemplate: ServerPinTests.FixtureTemplate(ServerPinTests.Qwen3Fixture)));
        }
        return path;
    }

    /// <summary>Loads <paramref name="modelPath"/> and starts a server on a free loopback port.</summary>
    /// <param name="modelPath">The synthetic GGUF.</param>
    /// <param name="chunkCache">The chunk-cache options (flag on or off, store).</param>
    /// <param name="paged">True: the default paged KV with the scheduler, which <c>ServerStartup</c> gives the cross-request
    /// prefix trie (#1596); false: the single-slot prefix cache.</param>
    /// <param name="configure">Runs on the app before the endpoints are mapped (test middleware: a remote address, the rate limiter).</param>
    /// <param name="logs">Receives the host's <c>ILogger</c> output (every level).</param>
    public static async Task<ChunkCacheTestServer> StartAsync(string modelPath, ChunkCacheOptions chunkCache, bool paged = true,
        Action<WebApplication>? configure = null, Microsoft.Extensions.Logging.ILoggerProvider? logs = null)
    {
        var options = new ServerOptions
        {
            Model = modelPath,
            ModelId = "tiny",
            Device = "cpu",
            Threads = 1,
            UsePaged = paged,
            ChunkCache = chunkCache,
            Metrics = true,
        };
        var state = ServerStartup.LoadModel(modelPath, options);
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddDotLLM(state);
        if (logs is not null)
        {
            builder.Logging.AddProvider(logs);
            builder.Logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Trace);
        }
        var app = builder.Build();
        configure?.Invoke(app);
        app.MapDotLLMEndpoints(serveUi: false);
        await app.StartAsync();
        state.StartSchedulerLoop(); // as ServerStartup does once the host runs (paged: the scheduler serves every request)
        return new ChunkCacheTestServer(app, state, new HttpClient { BaseAddress = new Uri(app.Urls.First()), Timeout = TimeSpan.FromMinutes(2) });
    }

    public async Task<(int Status, string Body)> PostAsync(string path, string body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.Add("anthropic-version", "2023-06-01");
        using var resp = await Http.SendAsync(request);
        return ((int)resp.StatusCode, await resp.Content.ReadAsStringAsync());
    }

    public async Task<string> GetAsync(string path) => await Http.GetStringAsync(path);

    /// <summary>Sends a request with extra headers (e.g. <c>X-API-Key</c>); <paramref name="body"/> null sends none.</summary>
    public async Task<(int Status, string Body)> SendAsync(HttpMethod method, string path, string? body, params (string Name, string Value)[] headers)
    {
        using var request = new HttpRequestMessage(method, path);
        if (body is not null)
            request.Content = new StringContent(body, Encoding.UTF8, "application/json");
        request.Headers.Add("anthropic-version", "2023-06-01");
        foreach (var (name, value) in headers)
            request.Headers.TryAddWithoutValidation(name, value);
        using var resp = await Http.SendAsync(request);
        return ((int)resp.StatusCode, await resp.Content.ReadAsStringAsync());
    }

    public async ValueTask DisposeAsync()
    {
        Http.Dispose();
        await App.StopAsync();
        await App.DisposeAsync();
        State.Dispose();
    }

    // ------------------------------------------------------------------------------------------------ bodies

    /// <summary>
    /// The UTF-8 byte range of <paramref name="text"/> inside <paramref name="content"/> (first occurrence) as a manifest
    /// entry with its real hash.
    /// </summary>
    public static JsonObject Entry(string id, int message, string content, string text)
    {
        int at = content.IndexOf(text, StringComparison.Ordinal);
        Assert.True(at >= 0, "span text not in content");
        int start = Encoding.UTF8.GetByteCount(content.AsSpan(0, at));
        return new JsonObject
        {
            ["id"] = id,
            ["hash"] = ChunkAnnotationParserTests.Sha(text),
            ["at"] = new JsonObject { ["message"] = message, ["part"] = 0, ["start"] = start, ["end"] = start + Encoding.UTF8.GetByteCount(text) },
        };
    }

    /// <summary>An OpenAI chat body (greedy, short) with optional <c>chunk_cache</c>.</summary>
    public static string OpenAi((string Role, string Content)[] messages, JsonObject? chunkCache, bool stream = false, int maxTokens = 6)
    {
        var body = new JsonObject
        {
            ["messages"] = new JsonArray(messages.Select(m => (JsonNode)new JsonObject { ["role"] = m.Role, ["content"] = m.Content }).ToArray()),
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0,
            ["seed"] = 1,
        };
        if (stream)
        {
            body["stream"] = true;
            body["stream_options"] = new JsonObject { ["include_usage"] = true };
        }
        if (chunkCache is not null)
            body["chunk_cache"] = chunkCache;
        return body.ToJsonString();
    }

    /// <summary>The same conversation as an Anthropic body (string content; a system message becomes <c>system</c>) with optional <c>chunk_cache</c>.</summary>
    public static string Anthropic((string Role, string Content)[] messages, JsonObject? chunkCache, bool stream = false, int maxTokens = 6)
    {
        var body = new JsonObject
        {
            ["messages"] = new JsonArray(messages.Where(m => m.Role != "system")
                .Select(m => (JsonNode)new JsonObject { ["role"] = m.Role, ["content"] = m.Content }).ToArray()),
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0,
        };
        if (messages.FirstOrDefault(m => m.Role == "system") is { Content: { } system })
            body["system"] = system;
        if (stream)
            body["stream"] = true;
        if (chunkCache is not null)
            body["chunk_cache"] = chunkCache;
        return body.ToJsonString();
    }

    // ------------------------------------------------------------------------------------------------ responses

    /// <summary>The <c>data:</c> JSON payloads of an SSE body, in order (<c>[DONE]</c> excluded).</summary>
    public static List<JsonNode> SseData(string body) =>
        body.Split('\n')
            .Where(l => l.StartsWith("data: ", StringComparison.Ordinal) && l != "data: [DONE]")
            .Select(l => JsonNode.Parse(l["data: ".Length..])!)
            .ToList();

    /// <summary>
    /// The body with every value that varies run to run blanked: ids, <c>created</c>, and every number under
    /// <c>timings</c> / <c>timings_ms</c> (keys kept). What remains must be byte-identical between runs that generate alike.
    /// </summary>
    public static string Normalize(string body)
    {
        if (body.StartsWith("data: ", StringComparison.Ordinal) || body.StartsWith("event: ", StringComparison.Ordinal))
        {
            var sb = new StringBuilder();
            foreach (string line in body.Split('\n'))
            {
                if (line.StartsWith("data: {", StringComparison.Ordinal))
                    sb.Append("data: ").Append(NormalizeJson(JsonNode.Parse(line["data: ".Length..])!)).Append('\n');
                else
                    sb.Append(line).Append('\n');
            }
            return sb.ToString();
        }
        return NormalizeJson(JsonNode.Parse(body)!);
    }

    private static string NormalizeJson(JsonNode node)
    {
        Blank(node, inTimings: false);
        return node.ToJsonString();
    }

    private static void Blank(JsonNode? node, bool inTimings)
    {
        switch (node)
        {
            case JsonObject o:
                foreach (var key in o.Select(p => p.Key).ToList())
                {
                    var v = o[key];
                    if (key is "id" && v is JsonValue)
                        o[key] = "X";
                    else if (key is "created")
                        o[key] = 0;
                    else if (inTimings && v is JsonValue)
                        o[key] = 0;
                    else
                        Blank(v, inTimings || key is "timings" or "timings_ms");
                }
                break;
            case JsonArray a:
                foreach (var item in a)
                    Blank(item, inTimings);
                break;
        }
    }
}
