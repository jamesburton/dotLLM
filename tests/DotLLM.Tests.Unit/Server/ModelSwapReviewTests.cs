using System.Net.Http.Headers;
using System.Text;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Review of #1023: keep-alive during the ACTIVATION wait on every streaming surface, Anthropic-shaped in-stream errors on
/// <c>/v1/messages</c>, and post-lock re-validation for model-reading (non-generator) work. Real HTTP against a bare server; the gate is
/// held by a test "request" so activation has to wait, exactly as a request queued behind a load does.
/// </summary>
[Collection("SequentialFileIO")]   // the model-store env vars are process-global
public sealed class ModelSwapReviewTests : IAsyncLifetime
{
    private static readonly string[] EnvVars = ["DOTLLM_MODELS_DIR", "DOTLLM_PROFILES_DIR", "OLLAMA_MODELS", "HF_HUB_CACHE"];
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-1023r-" + Guid.NewGuid().ToString("N"));
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

        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        _state = ServerStartup.CreateBareState(new ServerOptions { Model = "none" });
        _state.KeepAliveInterval = TimeSpan.FromMilliseconds(50);
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

    /// <summary>
    /// Sends a streaming request while the gate is held, waits until a keep-alive comment has arrived, releases the gate (the
    /// activation then fails: the model does not exist), and returns the whole body.
    /// </summary>
    private async Task<string> StreamWhileGateHeldAsync(string path, string body)
    {
        var release = new TaskCompletionSource();
        var holding = new TaskCompletionSource();
        var holder = _state!.ExecuteAsync(async () => { holding.SetResult(); await release.Task; }, CancellationToken.None);
        await holding.Task;

        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.Add("anthropic-version", "2023-06-01");
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);   // committed by the keep-alive

        await using var stream = await response.Content.ReadAsStreamAsync();
        var text = new StringBuilder();
        var buffer = new byte[4096];
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (!text.ToString().Contains(": waiting for", StringComparison.Ordinal) && DateTime.UtcNow < deadline)
        {
            int n = await stream.ReadAsync(buffer);
            if (n == 0) break;
            text.Append(Encoding.UTF8.GetString(buffer, 0, n));
        }

        release.SetResult();
        await holder;
        int read;
        while ((read = await stream.ReadAsync(buffer)) > 0) text.Append(Encoding.UTF8.GetString(buffer, 0, read));
        return text.ToString();
    }

    [Theory]
    [InlineData("/v1/chat/completions", "{\"model\":\"nope-1023\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/v1/completions", "{\"model\":\"nope-1023\",\"stream\":true,\"prompt\":\"hi\"}")]
    public async Task OpenAiStreams_KeepAliveDuringTheActivationWait_ThenTheErrorInStream(string path, string body)
    {
        string raw = await StreamWhileGateHeldAsync(path, body);

        Assert.StartsWith(": waiting for busy: generate", raw);
        Assert.Contains("data: {\"error\":", raw.Replace("data: {\"type\":\"error\",\"error\":", "data: {\"error\":"));
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public async Task MessagesStream_KeepAliveDuringTheActivationWait_ThenAnAnthropicErrorEvent_WithoutDone()
    {
        string raw = await StreamWhileGateHeldAsync("/v1/messages",
            "{\"model\":\"nope-1023\",\"max_tokens\":8,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");

        Assert.StartsWith(": waiting for busy: generate", raw);
        Assert.Contains("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\"", raw);
        Assert.DoesNotContain("[DONE]", raw);   // OpenAI's terminator means nothing to an Anthropic SDK
    }

    private sealed class StartedResponseFeature : HttpResponseFeature
    {
        public override bool HasStarted => true;
    }

    [Fact]
    public async Task ModelUnavailable_OnAStartedMessagesStream_IsAnAnthropicErrorEvent()
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Features.Set<IHttpResponseFeature>(new StartedResponseFeature());
        ctx.Features.Set<IHttpResponseBodyFeature>(new StreamResponseBodyFeature(body));
        ctx.Request.Path = "/v1/messages";

        var app = new ApplicationBuilder(new ServiceCollection().BuildServiceProvider());
        ModelUnavailableException.UseModelUnavailableHandler(app);
        app.Run(_ => throw new ModelUnavailableException("owner/repo/x.gguf", "reload failed"));
        await app.Build()(ctx);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.StartsWith("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Model ", raw);
        Assert.Contains("owner/repo/x.gguf", raw);
        Assert.DoesNotContain("[DONE]", raw);
    }

    [Fact]
    public async Task ModelReadingWork_WhoseModelIsGone_ThrowsModelUnavailable_AndNeverRuns()
    {
        bool ran = false;
        await Assert.ThrowsAsync<ModelUnavailableException>(() =>
            _state!.ExecuteOnModelAsync("owner/repo/gone-1023.gguf", () => { ran = true; return Task.CompletedTask; }, CancellationToken.None));
        Assert.False(ran);
    }

    [Fact]
    public void DeviceLocalEviction_EvictsOnlyMatchingModels()
    {
        var residency = new ModelResidencyManager { MaxResidentModels = 5 };
        var older = new ResidentModelSnapshot { Key = "cpu-model", Options = new ServerOptions { Model = "a", ResolvedDevice = "cpu" }, LastUsedUtc = DateTimeOffset.UtcNow.AddMinutes(-5) };
        var newer = new ResidentModelSnapshot { Key = "gpu-model", Options = new ServerOptions { Model = "b", ResolvedDevice = "gpu:0" }, LastUsedUtc = DateTimeOffset.UtcNow };
        residency.Stash(older);
        residency.Stash(newer);

        var evicted = residency.EvictLeastRecentlyUsed(s => s.Options.ResolvedDevice == "gpu:0");

        Assert.Same(newer, evicted);            // the older CPU-resident model frees nothing on the GPU, so it stays
        Assert.True(residency.Contains("cpu-model"));
        Assert.Equal(0, residency.StashedBytes(s => s.Options.ResolvedDevice == "gpu:0"));
    }

    /// <summary>
    /// Review round 2 of #1023, on the endpoint path: <c>/v1/models/load</c>'s pre-free fit check must judge the placement the load will
    /// USE (request, then profile, then server default) - it used the current model's gpu_layers and ignored a profile's device, so a
    /// partial offload of a big model was refused and a profile pinned elsewhere was checked against the wrong device.
    /// </summary>
    [Theory]
    [InlineData("{\"model\":\"acme/big-GGUF/big.gguf\"}", true)]                       // full offload of 4 KiB onto a 1 KiB "GPU": refused
    [InlineData("{\"model\":\"acme/big-GGUF/big.gguf\",\"gpu_layers\":20}", false)]  // partial offload: the loader decides, not the fit check
    [InlineData("{\"model\":\"pinned-cpu-1023\"}", false)]                              // the profile pins the CPU: no GPU fit check applies
    public async Task ModelsLoad_FitCheckUsesTheLoadsOwnPlacement(string body, bool refused)
    {
        string file = DotLLM.HuggingFace.HubCache.SnapshotFilePath("acme/big-GGUF", "c0ffee", "big.gguf", Path.Combine(_root, "hub"));
        Directory.CreateDirectory(Path.GetDirectoryName(file)!);
        File.WriteAllBytes(file, "GGUF"u8.ToArray().Concat(new byte[4092]).ToArray());
        DotLLM.HuggingFace.ModelProfileStore.Save("pinned-cpu-1023", new DotLLM.HuggingFace.ModelProfile { From = "acme/big-GGUF/big.gguf", Device = "cpu" });
        _state!.Options = _state.Options with { Device = "gpu", AllowModelAdminApi = true };
        _state.CudaMemoryProbe = _ => (Free: 1, Total: 1024);

        var response = await _http.PostAsync("/v1/models/load", new StringContent(body, Encoding.UTF8, "application/json"));
        string text = await response.Content.ReadAsStringAsync();

        Assert.Equal(refused, text.Contains("cannot load there", StringComparison.Ordinal));
    }
}
