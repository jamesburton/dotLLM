using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using DotLLM.Cli.Commands;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.SafeTensors;
using DotLLM.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

[Collection("SequentialFileIO")]
public sealed class SafetensorsApplicationTests : IDisposable
{
    private readonly HfApplicationFixture _fixture = new();
    private const string Embedding = "model.embed_tokens.weight";
    private const string Template = "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";

    private static ServerOptions Options(string directory, string device = "cpu", int resident = 1) => new()
    {
        Model = directory, ModelId = directory, Device = device, Threads = 1, DecodeThreads = 1,
        UsePaged = false, PromptCacheEnabled = false, Warmup = WarmupOptions.Disabled,
        MaxResidentModels = resident, MtpEnabled = false,
    };

    [Theory]
    [InlineData("cpu")]
    [InlineData("auto")]
    public async Task Server_LoadsHfTokenizerTemplateAndPhysicalMetadata(string device)
    {
        string directory = WriteCheckpoint("model");
        Assert.Equal(directory, ServerStartup.ResolveModelPath(directory, null));
        using var state = ServerStartup.LoadModel(directory, Options(directory, device));
        Assert.Equal("cpu", state.Options.ResolvedDevice);
        Assert.Equal(device == "auto", state.DeviceFallbackWarning is not null);
        Assert.Null(state.CurrentGguf);
        Assert.NotNull(state.CurrentSafetensors);
        Assert.Equal("user:Hello;assistant:", state.ChatTemplate!.Apply(
            [new ChatMessage { Role = "user", Content = "Hello" }], new ChatTemplateOptions { AddGenerationPrompt = true }));
        Assert.Equal("Hello", state.Tokenizer!.Decode(state.Tokenizer.Encode("Hello")));
        var meta = Assert.Single(state.ListResidentModels()).Meta!;
        Assert.Equal(18_592, meta.NParams);
        Assert.Equal(37_376, meta.Size);
        Assert.Equal("F16 - 16.08 bpw", meta.Ftype);
        Assert.Equal(new FileInfo(Path.Combine(directory, "model.safetensors")).Length, state.EstimatedBytes);
        string first = await Generate(state);
        Assert.Equal(first, await Generate(state)); // A request must start with an independent cache.
    }

    [Fact]
    public async Task Residency_PreservesSourceOnReactivation_AndDisposesBothOnUnload()
    {
        string a = WriteCheckpoint("a"), b = WriteCheckpoint("b");
        using var state = ServerStartup.CreateBareState(Options(a, resident: 2));
        Assert.Null(await state.EnsureActiveAsync(a, null, CancellationToken.None));
        var sourceA = state.CurrentSafetensors!;
        var modelA = state.Model;
        Assert.Null(await state.EnsureActiveAsync(b, null, CancellationToken.None));
        var sourceB = state.CurrentSafetensors!;
        Assert.NotSame(sourceA, sourceB);
        Assert.Equal(1, state.Residency.StashedCount);
        Assert.Equal(18_592, state.ListResidentModels().Single(m => m.Key == a).Meta!.NParams);
        Assert.Equal(Path.Combine(a, "model.safetensors"), sourceA.GetTensorFileLocation(Embedding).Path);
        Assert.Null(await state.EnsureActiveAsync(a, null, CancellationToken.None));
        Assert.Same(sourceA, state.CurrentSafetensors);
        Assert.Same(modelA, state.Model);
        Assert.Equal(2, (await state.UnloadAsync(null, all: true, CancellationToken.None)).Count);
        Assert.Null(state.CurrentSafetensors);
        Assert.Throws<ObjectDisposedException>(() => sourceA.GetTensorFileLocation(Embedding));
        Assert.Throws<ObjectDisposedException>(() => sourceB.GetTensorFileLocation(Embedding));
        AssertCanOpenExclusively(a);
        AssertCanOpenExclusively(b);
    }

    [Fact]
    public async Task Residency_EvictionDisposesSource_AndShutdownDisposesActiveSource()
    {
        string a = WriteCheckpoint("evict-a"), b = WriteCheckpoint("evict-b");
        var state = ServerStartup.CreateBareState(Options(a));
        ISafetensorsTensorSource sourceB;
        try
        {
            Assert.Null(await state.EnsureActiveAsync(a, null, CancellationToken.None));
            var sourceA = state.CurrentSafetensors!;
            Assert.Null(await state.EnsureActiveAsync(b, null, CancellationToken.None));
            sourceB = state.CurrentSafetensors!;
            Assert.Throws<ObjectDisposedException>(() => sourceA.GetTensorFileLocation(Embedding));
            Assert.Equal(0, state.Residency.StashedCount);
        }
        finally { state.Dispose(); }
        Assert.Throws<ObjectDisposedException>(() => sourceB.GetTensorFileLocation(Embedding));
        AssertCanOpenExclusively(a);
        AssertCanOpenExclusively(b);
    }

    [Fact]
    public void MalformedDeclaredTemplate_FailsBeforeOpeningCorruptWeights()
    {
        string directory = WriteCheckpoint("bad-template");
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), "{\"chat_template\":17}");
        File.WriteAllBytes(Path.Combine(directory, "model.safetensors"), [1, 2, 3]);
        var error = Assert.Throws<InvalidDataException>(() => ServerStartup.LoadModel(directory, Options(directory)));
        Assert.Contains("chat template", error.Message);
        AssertCanOpenExclusively(directory);
    }

    [Fact]
    public void FailedModelLoad_ReleasesMappedSource()
    {
        string directory = WriteCheckpoint("bad-weights");
        new SafetensorsFixtureBuilder().AddFloat32("unrelated", [1], [1f])
            .WriteTo(Path.Combine(directory, "model.safetensors"));
        Assert.ThrowsAny<Exception>(() => ServerStartup.LoadModel(directory, Options(directory)));
        AssertCanOpenExclusively(directory);
    }

    [Fact]
    public void GgufStateConstructionFailure_ReleasesItsModelAndFileMapping()
    {
        string path = _fixture.WriteGgufForLifetimeCheck();
        // This cache flag is parsed after the model has been built, and fails before a ready state is returned.
        var error = Assert.Throws<ArgumentException>(() => ServerStartup.LoadModel(path, Options(path) with { CacheTypeK = "invalid-cache-type" }));
        Assert.Contains("invalid-cache-type", error.Message);
        using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        Assert.True(exclusive.CanWrite);
    }

    [Fact]
    public void NamedToolTemplate_IsSelectedPerRequest()
    {
        string directory = WriteCheckpoint("tool-template");
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), JsonSerializer.Serialize(new
        {
            chat_template = new Dictionary<string, string> { ["default"] = Template, ["tool_use"] = "tools:{{ tools | tojson }};" + Template },
        }));
        using var state = ServerStartup.LoadModel(directory, Options(directory));
        ChatMessage[] messages = [new() { Role = "user", Content = "Hi" }];
        Assert.Equal("user:Hi;assistant:", state.ChatTemplate!.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true }));
        string withTools = state.ChatTemplate.Apply(messages, new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = [new ToolDefinition("weather", "Weather", "{\"type\":\"object\"}")],
        });
        Assert.StartsWith("tools:", withTools);
        Assert.Contains("weather", withTools);
        Assert.EndsWith("user:Hi;assistant:", withTools);
    }

    [Fact]
    public async Task AdminHttpLoadAndUnload_TransfersAndReleasesHfSource()
    {
        string directory = WriteCheckpoint("admin");
        using var state = ServerStartup.CreateBareState(Options(directory) with { AllowModelAdminApi = true });
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Logging.ClearProviders();
        builder.Services.AddSingleton(state);
        await using var app = builder.Build();
        app.MapDotLLMEndpoints();
        await app.StartAsync();
        using var client = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
        using var loaded = await client.PostAsJsonAsync("/v1/models/load", new { model = directory, device = "cpu" });
        Assert.True(loaded.IsSuccessStatusCode, await loaded.Content.ReadAsStringAsync());
        var source = Assert.IsAssignableFrom<ISafetensorsTensorSource>(state.CurrentSafetensors);
        Assert.Equal(directory, state.LoadedModelPath);
        using var listing = JsonDocument.Parse(await client.GetStringAsync("/v1/models"));
        var item = listing.RootElement.GetProperty("data")[0];
        Assert.Equal(state.Options.ModelId, item.GetProperty("id").GetString());
        Assert.Equal(37_376, item.GetProperty("meta").GetProperty("size").GetInt64());
        using var unloaded = await client.PostAsJsonAsync("/v1/models/unload", new { all = true });
        Assert.True(unloaded.IsSuccessStatusCode, await unloaded.Content.ReadAsStringAsync());
        Assert.Null(state.CurrentSafetensors);
        Assert.Throws<ObjectDisposedException>(() => source.GetTensorFileLocation(Embedding));
        AssertCanOpenExclusively(directory);
    }

    [Theory]
    [InlineData("cuda:0")]
    [InlineData("vulkan:0")]
    public void ExplicitUnsupportedHfDevice_DoesNotOpenWeightsOrFallback(string device)
    {
        var error = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadSafetensors(
            "does-not-exist", device, null, default, _ => { }, _ => { }));
        Assert.Contains("no CPU fallback", error.Message);
        Assert.Contains(device, error.Message);
    }

    [Theory]
    [InlineData("cpu", 1)]
    [InlineData("auto", -1)]
    [InlineData("hip", 0)]
    public void UnsupportedLayerPlacement_FailsBeforeOpeningWeights(string device, int layers)
        => Assert.Throws<NotSupportedException>(() => DeviceModelLoader.LoadSafetensors(
            "does-not-exist", device, layers, default, _ => { }, _ => { }));

    [Theory]
    [InlineData("tensor-overlay")]
    [InlineData("draft")]
    [InlineData("context")]
    public void UnsupportedServerOptions_FailBeforeTokenizerOrWeights(string option)
    {
        string directory = WriteCheckpoint("unsupported-" + option);
        File.Delete(Path.Combine(directory, "tokenizer.json"));
        var options = option switch
        {
            "tensor-overlay" => Options(directory) with { TensorOverlayPath = "missing.gguf" },
            "draft" => Options(directory) with { SpeculativeModel = "missing.gguf" },
            _ => Options(directory) with { ContextSize = 16 },
        };
        Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(directory, options));
    }

    [Fact]
    public async Task CliRun_UsesTheSameHfCpuRoute()
    {
        string directory = WriteCheckpoint("cli");
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true,
            UseShellExecute = false, CreateNoWindow = true };
        foreach (string arg in new[] { typeof(RunCommand).Assembly.Location, "run", directory, "--device", "cpu", "-p", "Hello",
                     "-n", "2", "--temp", "0", "--threads", "1", "--json" }) start.ArgumentList.Add(arg);
        start.Environment["DOTLLM_NO_PROMPT"] = "1";
        using var process = Process.Start(start)!;
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(), stderr = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        try { await process.WaitForExitAsync(timeout.Token); }
        catch { process.Kill(entireProcessTree: true); throw; }
        string output = await stdout, error = await stderr;
        Assert.True(process.ExitCode == 0, output + error);
        using var result = JsonDocument.Parse(output);
        Assert.Equal("cpu", result.RootElement.GetProperty("backend").GetProperty("resolved_device").GetString());
        using var state = ServerStartup.LoadModel(directory, Options(directory));
        string rendered = state.ChatTemplate!.Apply([new ChatMessage { Role = "user", Content = "Hello" }],
            new ChatTemplateOptions { AddGenerationPrompt = true });
        Assert.Equal(await Generate(state, rendered), result.RootElement.GetProperty("text").GetString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CliServe_LoadsHfDirectory_Generates_AndShutsDown(bool sharded)
    {
        string directory = _fixture.WriteCheckpoint("cli-serve", sharded);
        string expected;
        using (var state = ServerStartup.LoadModel(directory, Options(directory)))
            expected = await Generate(state);
        using var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true,
            UseShellExecute = false, CreateNoWindow = true };
        foreach (string arg in new[] { typeof(ServeCommand).Assembly.Location, "serve", directory, "--device", "cpu",
                     "--host", "127.0.0.1", "--port", port.ToString(System.Globalization.CultureInfo.InvariantCulture),
                     "--threads", "1", "--no-warmup", "--no-ui", "--no-prompt-cache", "--no-paged", "--no-mtp", "--allow-model-admin" })
            start.ArgumentList.Add(arg);
        start.Environment["DOTLLM_NO_PROMPT"] = "1";
        using var process = Process.Start(start)!;
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(), stderr = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        using var client = new HttpClient(new HttpClientHandler { UseProxy = false })
            { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(2) };
        try
        {
            string? props = null;
            while (props is null)
            {
                Assert.False(process.HasExited, process.HasExited ? await stdout + await stderr : "");
                try { props = await client.GetStringAsync("/props", timeout.Token); }
                catch (HttpRequestException) { await Task.Delay(50, timeout.Token); }
                catch (TaskCanceledException) when (!timeout.IsCancellationRequested) { await Task.Delay(50, timeout.Token); }
            }
            using var metadata = JsonDocument.Parse(props);
            string modelId = metadata.RootElement.GetProperty("model_id").GetString()!;
            Assert.Equal(directory, metadata.RootElement.GetProperty("model_path").GetString());
            Assert.Equal("cpu", metadata.RootElement.GetProperty("resolved_device").GetString());
            using var completion = await client.PostAsJsonAsync("/v1/completions",
                new { model = modelId, prompt = "Hello", max_tokens = 2, temperature = 0 }, timeout.Token);
            Assert.True(completion.IsSuccessStatusCode, await completion.Content.ReadAsStringAsync(timeout.Token));
            using var result = JsonDocument.Parse(await completion.Content.ReadAsStringAsync(timeout.Token));
            Assert.Equal(expected, result.RootElement.GetProperty("choices")[0].GetProperty("text").GetString());
            using var shutdown = await client.PostAsJsonAsync("/v1/admin/shutdown", new { }, timeout.Token);
            Assert.Equal(System.Net.HttpStatusCode.Accepted, shutdown.StatusCode);
            await process.WaitForExitAsync(timeout.Token);
            Assert.True(process.ExitCode == 0, await stdout + await stderr);
        }
        finally
        {
            if (!process.HasExited) { process.Kill(entireProcessTree: true); await process.WaitForExitAsync(); }
        }
        foreach (string file in Directory.EnumerateFiles(directory, "*.safetensors"))
        {
            using var exclusive = new FileStream(file, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            Assert.True(exclusive.CanWrite);
        }
    }

    private static async Task<string> Generate(ServerState state, string prompt = "Hello")
    {
        var text = new System.Text.StringBuilder();
        await foreach (string part in state.Generator!.GenerateStreamingAsync(prompt, new InferenceOptions { MaxTokens = 2, Temperature = 0 }))
            text.Append(part);
        return text.ToString();
    }

    private string WriteCheckpoint(string name) => _fixture.WriteCheckpoint(name);

    private static void AssertCanOpenExclusively(string directory)
    {
        using var file = new FileStream(Path.Combine(directory, "model.safetensors"), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        Assert.True(file.CanWrite);
    }

    public void Dispose()
    {
        _fixture.Dispose();
    }
}
