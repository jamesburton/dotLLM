using System.Net;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>Issue #720: the ollama-compatible /api surface, over real HTTP against a bare server (no model) with every model store pointed at a temp dir.</summary>
[Collection("SequentialFileIO")]   // the model-store env vars are process-global
public sealed class OllamaApiEndpointTests : IAsyncLifetime
{
    private static readonly string[] EnvVars = ["DOTLLM_MODELS_DIR", "DOTLLM_PROFILES_DIR", "OLLAMA_MODELS", "HF_HUB_CACHE"];
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-720-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, string?> _saved = new();
    private WebApplication? _app;
    private ServerState? _state;
    private HttpClient _http = new();

    private async Task StartAsync(bool admin)
    {
        foreach (var v in EnvVars) _saved[v] = Environment.GetEnvironmentVariable(v);
        Environment.SetEnvironmentVariable("DOTLLM_MODELS_DIR", Path.Combine(_root, "models"));
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", Path.Combine(_root, "profiles"));
        Environment.SetEnvironmentVariable("OLLAMA_MODELS", Path.Combine(_root, "ollama"));
        Environment.SetEnvironmentVariable("HF_HUB_CACHE", Path.Combine(_root, "hub"));

        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        _state = ServerStartup.CreateBareState(new ServerOptions { Model = "none", AllowModelAdminApi = admin });
        builder.Services.AddSingleton(_state);
        _app = builder.Build();
        _app.MapDotLLMEndpoints(serveUi: false);
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public Task InitializeAsync() => Task.CompletedTask;

    public async Task DisposeAsync()
    {
        _http.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        _state?.Dispose();
        foreach (var (k, v) in _saved) Environment.SetEnvironmentVariable(k, v);
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    private static StringContent Json(string body) => new(body, Encoding.UTF8, "application/json");

    [Fact]
    public async Task Version_ReportsTheEmulatedApiLevel()
    {
        await StartAsync(admin: false);
        using var doc = JsonDocument.Parse(await _http.GetStringAsync("/api/version"));
        Assert.Equal(OllamaApiEndpoint.EmulatedVersion, doc.RootElement.GetProperty("version").GetString());
    }

    [Fact]
    public async Task Tags_ListsProfilesAndLocalAndOllamaStoreModels_WithTheFieldsClientsRead()
    {
        await StartAsync(admin: false);
        // a hub-cache model, an ollama-store model, and a profile over the first
        string hubFile = HubCache.SnapshotFilePath("acme/tiny-GGUF", "c0ffee", "tiny-Q4_K_M.gguf", Path.Combine(_root, "hub"));
        Directory.CreateDirectory(Path.GetDirectoryName(hubFile)!);
        File.WriteAllBytes(hubFile, "GGUF"u8.ToArray().Concat(new byte[996]).ToArray());
        ModelProfileStore.Save("helper:v1", new ModelProfile { From = "acme/tiny-GGUF:Q4_K_M", System = "hi" });

        using var doc = JsonDocument.Parse(await _http.GetStringAsync("/api/tags"));
        var models = doc.RootElement.GetProperty("models").EnumerateArray().ToList();

        var helper = models.Single(m => m.GetProperty("name").GetString() == "helper:v1");
        var tiny = models.Single(m => m.GetProperty("name").GetString() == "tiny-Q4_K_M");
        foreach (var m in new[] { helper, tiny })
        {
            Assert.Equal(m.GetProperty("name").GetString(), m.GetProperty("model").GetString());
            Assert.Equal(1000, m.GetProperty("size").GetInt64());
            Assert.Equal(64, m.GetProperty("digest").GetString()!.Length);
            Assert.Equal("gguf", m.GetProperty("details").GetProperty("format").GetString());
            Assert.Equal("Q4_K_M", m.GetProperty("details").GetProperty("quantization_level").GetString());
            Assert.True(DateTimeOffset.TryParse(m.GetProperty("modified_at").GetString(), out _));
        }
    }

    [Fact]
    public async Task Show_ForAProfile_SynthesisesAModelfile()
    {
        await StartAsync(admin: false);
        string gguf = Path.Combine(_root, "base.gguf");
        Directory.CreateDirectory(_root);
        File.WriteAllBytes(gguf, "GGUF"u8.ToArray().Concat(new byte[100]).ToArray());
        ModelProfileStore.Save("terse", new ModelProfile { From = gguf, System = "Be brief.", Temperature = 0.2f, Stop = ["###"] });

        var resp = await _http.PostAsync("/api/show", Json("{\"model\":\"terse\"}"));
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        string modelfile = doc.RootElement.GetProperty("modelfile").GetString()!;
        Assert.Contains("FROM " + gguf, modelfile);
        Assert.Contains("SYSTEM \"\"\"Be brief.\"\"\"", modelfile);
        Assert.Contains("PARAMETER temperature 0.2", modelfile);
        Assert.Contains("PARAMETER stop \"###\"", modelfile);
        Assert.Equal("completion", doc.RootElement.GetProperty("capabilities")[0].GetString());

        var missing = await _http.PostAsync("/api/show", Json("{\"model\":\"nope\"}"));
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
        Assert.Contains("not found", await missing.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Ps_IsEmptyWithNoModelLoaded()
    {
        await StartAsync(admin: false);
        using var doc = JsonDocument.Parse(await _http.GetStringAsync("/api/ps"));
        Assert.Empty(doc.RootElement.GetProperty("models").EnumerateArray());
    }

    [Theory]
    [InlineData("/api/chat", "{\"model\":\"nope:1b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")]
    [InlineData("/api/generate", "{\"model\":\"nope:1b\",\"prompt\":\"hi\"}")]
    public async Task GenerativeRoutes_UnknownModel_Is404WithTheOllamaErrorShape(string path, string body)
    {
        await StartAsync(admin: false);
        var resp = await _http.PostAsync(path, Json(body));
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        Assert.Contains("not found, try pulling it first", doc.RootElement.GetProperty("error").GetString());
    }

    [Theory]
    [InlineData("/api/chat", "{\"messages\":[]}", "model is required")]
    [InlineData("/api/generate", "{}", "model is required")]
    [InlineData("/api/chat", "{ not json", "invalid JSON")]
    public async Task BadRequests_Are400(string path, string body, string message)
    {
        await StartAsync(admin: false);
        var resp = await _http.PostAsync(path, Json(body));
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains(message, await resp.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("/api/create")]
    [InlineData("/api/copy")]
    [InlineData("/api/push")]
    public async Task UnimplementedRoutes_Answer501WithAPointer(string path)
    {
        await StartAsync(admin: false);
        var resp = await _http.PostAsync(path, Json("{}"));
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
        Assert.Contains("not implemented", await resp.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task PullAndDelete_AreAdminGated()
    {
        await StartAsync(admin: false);
        Assert.Equal(HttpStatusCode.Forbidden, (await _http.PostAsync("/api/pull", Json("{\"model\":\"owner/repo\"}"))).StatusCode);
        var del = new HttpRequestMessage(HttpMethod.Delete, "/api/delete") { Content = Json("{\"model\":\"x\"}") };
        var resp = await _http.SendAsync(del);
        Assert.Equal(HttpStatusCode.Forbidden, resp.StatusCode);
        Assert.Contains("--allow-model-admin", await resp.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Delete_RemovesAProfile_NotItsBaseModel_AndReports404ForUnknownNames()
    {
        await StartAsync(admin: true);
        string gguf = Path.Combine(_root, "base.gguf");
        Directory.CreateDirectory(_root);
        File.WriteAllBytes(gguf, "GGUF"u8.ToArray());
        ModelProfileStore.Save("gone", new ModelProfile { From = gguf });

        var resp = await _http.SendAsync(new HttpRequestMessage(HttpMethod.Delete, "/api/delete") { Content = Json("{\"model\":\"gone\"}") });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Null(ModelProfileStore.TryGet("gone"));
        Assert.True(File.Exists(gguf));

        var missing = await _http.SendAsync(new HttpRequestMessage(HttpMethod.Delete, "/api/delete") { Content = Json("{\"model\":\"never-existed\"}") });
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
    }

    [Fact]
    public async Task Delete_OfAPulledOllamaModel_RemovesItsFile_UnlessAnotherProfileUsesIt()
    {
        await StartAsync(admin: true);
        string dir = Path.Combine(_root, "models", "ollama", "tiny");
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "tiny-latest.gguf");
        File.WriteAllBytes(file, "GGUF"u8.ToArray());
        ModelProfileStore.Save("tiny:latest", new ModelProfile { From = file });
        ModelProfileStore.Save("tiny-alias", new ModelProfile { From = file });

        async Task<HttpStatusCode> Delete(string name) =>
            (await _http.SendAsync(new HttpRequestMessage(HttpMethod.Delete, "/api/delete") { Content = Json("{\"model\":\"" + name + "\"}") })).StatusCode;

        Assert.Equal(HttpStatusCode.OK, await Delete("tiny:latest"));
        Assert.True(File.Exists(file));            // another profile still points at it
        Assert.Equal(HttpStatusCode.OK, await Delete("tiny-alias"));
        Assert.False(File.Exists(file));          // last reference gone: the pulled file goes too
        Assert.False(Directory.Exists(dir));
    }

    [Fact]
    public async Task Pull_RejectsANameThatIsNeitherARepoNorAnOllamaModel()
    {
        await StartAsync(admin: true);
        var resp = await _http.PostAsync("/api/pull", Json("{\"model\":\"C:\\\\not\\\\a\\\\model.gguf\"}"));
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    // ───────────────────────────── pure helpers ─────────────────────────────

    [Theory]
    [InlineData("\"5m\"", 300.0)]
    [InlineData("\"30s\"", 30.0)]
    [InlineData("\"1h\"", 3600.0)]
    [InlineData("\"500ms\"", 0.5)]
    [InlineData("\"-1\"", -1.0)]
    [InlineData("0", 0.0)]
    [InlineData("-1", -1.0)]
    [InlineData("120", 120.0)]
    [InlineData("\"0\"", 0.0)]
    public void KeepAlive_AcceptsNumbersAndDurationStrings(string json, double seconds)
    {
        using var doc = JsonDocument.Parse("{\"keep_alive\":" + json + "}");
        Assert.Equal(seconds, OllamaApiEndpoint.ParseKeepAlive(doc.RootElement)!.Value, 6);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"keep_alive\":\"soon\"}")]
    [InlineData("{\"keep_alive\":true}")]
    public void KeepAlive_AbsentOrUnparseable_IsNull(string json)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.Null(OllamaApiEndpoint.ParseKeepAlive(doc.RootElement));
    }

    [Fact]
    public void Options_MapOverTheEffectiveDefaults_AndFormatSelectsConstrainedDecoding()
    {
        var defaults = new SamplingDefaults { Temperature = 0.8f, TopP = 0.95f, MaxTokens = 256, StopSequences = ["END"] };
        using var doc = JsonDocument.Parse("""
            {"options":{"temperature":0.1,"top_k":20,"num_predict":32,"seed":7,"repeat_penalty":1.1,"stop":["###"]},"format":"json"}
            """);

        var o = OllamaApiEndpoint.BuildOptions(doc.RootElement, defaults, ThreadingConfig.Auto);

        Assert.Equal(0.1f, o.Temperature);
        Assert.Equal(0.95f, o.TopP);          // not in the request: the default stays
        Assert.Equal(20, o.TopK);
        Assert.Equal(32, o.MaxTokens);
        Assert.Equal(7, o.Seed);
        Assert.Equal(1.1f, o.RepetitionPenalty);
        Assert.Equal(["###", "END"], o.StopSequences);   // request stops first, profile stops appended
        Assert.IsType<ResponseFormat.JsonObject>(o.ResponseFormat);

        using var schema = JsonDocument.Parse("""{"format":{"type":"object","properties":{"a":{"type":"string"}}},"options":{"num_predict":-1}}""");
        var s = OllamaApiEndpoint.BuildOptions(schema.RootElement, defaults, ThreadingConfig.Auto);
        Assert.IsType<ResponseFormat.JsonSchema>(s.ResponseFormat);
        Assert.Equal(256, s.MaxTokens);   // -1 = unlimited in ollama: the server default applies
    }
}
