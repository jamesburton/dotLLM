using System.Net;
using System.Text;
using DotLLM.HuggingFace;
using DotLLM.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1021: a bare model name shared by two repos (<c>q4_k_m</c> -> Holo4-27B and Holo4-35B) must be rejected on EVERY surface that
/// resolves a model, with a 400 that lists both candidate ids - never a silent load of the larger one, never a 500. Real HTTP against a
/// bare server (no model loaded, no GPU) with every model store pointed at a temp dir.
/// </summary>
/// <remarks>
/// The fixture files carry a GGUF magic but no valid header, so had either been chosen the response would be a GGUF parse error - not
/// the ambiguity message. Asserting the message therefore also pins that resolution happens before any load is attempted.
/// </remarks>
[Collection("SequentialFileIO")]   // the model-store env vars are process-global
public sealed class AmbiguousModelNameEndpointTests : IAsyncLifetime
{
    private const string IdA = "Hcompany/Holo4-27B-GGUF/q4_k_m.gguf";
    private const string IdB = "Hcompany/Holo4-35B-A3B-GGUF/q4_k_m.gguf";
    private static readonly string[] EnvVars = ["DOTLLM_MODELS_DIR", "DOTLLM_PROFILES_DIR", "OLLAMA_MODELS", "HF_HUB_CACHE"];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-1021-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, string?> _saved = new();
    private WebApplication? _app;
    private ServerState? _state;
    private HttpClient _http = new();

    public async Task InitializeAsync()
    {
        foreach (var v in EnvVars) _saved[v] = Environment.GetEnvironmentVariable(v);
        Environment.SetEnvironmentVariable("DOTLLM_MODELS_DIR", Path.Combine(_root, "models"));
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", Path.Combine(_root, "profiles"));
        Environment.SetEnvironmentVariable("OLLAMA_MODELS", Path.Combine(_root, "ollama"));
        Environment.SetEnvironmentVariable("HF_HUB_CACHE", Path.Combine(_root, "hub"));

        foreach (string repo in new[] { "Hcompany/Holo4-27B-GGUF", "Hcompany/Holo4-35B-A3B-GGUF" })
        {
            string file = HubCache.SnapshotFilePath(repo, "c0ffee", "q4_k_m.gguf", Path.Combine(_root, "hub"));
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            File.WriteAllBytes(file, "GGUF"u8.ToArray().Concat(new byte[96]).ToArray());
        }

        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        _state = ServerStartup.CreateBareState(new ServerOptions { Model = "none", AllowModelAdminApi = true });
        builder.Services.AddSingleton(_state);
        _app = builder.Build();
        _app.MapDotLLMEndpoints(serveUi: false);
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public async Task DisposeAsync()
    {
        _http.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        _state?.Dispose();
        foreach (var (k, v) in _saved) Environment.SetEnvironmentVariable(k, v);
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    [Theory]
    [InlineData("/v1/chat/completions", "{\"model\":\"q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/v1/chat/completions", "{\"model\":\"q4_k_m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/v1/completions", "{\"model\":\"q4_k_m\",\"prompt\":\"hi\"}")]
    [InlineData("/v1/messages", "{\"model\":\"q4_k_m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/v1/messages/count_tokens", "{\"model\":\"q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/v1/embeddings", "{\"model\":\"q4_k_m\",\"input\":\"hi\"}")]
    [InlineData("/v1/models/load", "{\"model\":\"q4_k_m\"}")]
    [InlineData("/api/chat", "{\"model\":\"q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/api/generate", "{\"model\":\"q4_k_m\",\"prompt\":\"hi\"}")]
    [InlineData("/api/show", "{\"model\":\"q4_k_m\"}")]
    public async Task AmbiguousBareName_Is400_ListingEveryCandidateId(string path, string body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.Add("anthropic-version", "2023-06-01");   // required by /v1/messages; ignored elsewhere

        var resp = await _http.SendAsync(request);
        string text = await resp.Content.ReadAsStringAsync();

        Assert.True(resp.StatusCode == HttpStatusCode.BadRequest, $"{path}: {(int)resp.StatusCode} {text}");
        Assert.Contains("ambiguous", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(IdA, text);
        Assert.Contains(IdB, text);
        Assert.Null(_state!.Model);   // nothing was loaded
        if (path.StartsWith("/v1/", StringComparison.Ordinal) && !path.StartsWith("/v1/messages", StringComparison.Ordinal))
            Assert.Contains("\"code\":\"ambiguous_model\"", text);   // its own code, not model_not_found (OpenAI envelope only)
    }

    /// <summary>
    /// Review of #1021: an id the server handed out goes back to the exact file it was minted for, not to whatever the name resolves to.
    /// "minted-1021" resolves to nothing by name, so reaching the (unparseable) fixture - a load error, not "not found" - proves the
    /// issued-id map was consulted.
    /// </summary>
    [Fact]
    public async Task IssuedId_ReactivatesFromTheExactPathItWasMintedFor()
    {
        string file = HubCache.SnapshotFilePath("Hcompany/Holo4-27B-GGUF", "c0ffee", "q4_k_m.gguf", Path.Combine(_root, "hub"));
        Assert.NotNull(await _state!.EnsureActiveAsync("minted-1021", keepAliveOverride: null, CancellationToken.None) is { } before
            && before.Contains("not found", StringComparison.OrdinalIgnoreCase) ? before : null);

        _state.RememberIssuedId("minted-1021", file);
        string? error = await _state.EnsureActiveAsync("minted-1021", keepAliveOverride: null, CancellationToken.None);

        Assert.NotNull(error);
        Assert.DoesNotContain("not found", error!, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("ambiguous", error!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task FullCandidateId_IsNotAmbiguous_ItReachesTheLoad()
    {
        // The listed id is the way out: it resolves, so the request proceeds to a load (which fails here only because the fixture is
        // not a real model) instead of being refused as ambiguous.
        var resp = await _http.PostAsync("/v1/chat/completions", new StringContent(
            $"{{\"model\":\"{IdA}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}", Encoding.UTF8, "application/json"));
        string text = await resp.Content.ReadAsStringAsync();

        Assert.DoesNotContain("ambiguous", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("not found", text, StringComparison.OrdinalIgnoreCase);
    }
}
