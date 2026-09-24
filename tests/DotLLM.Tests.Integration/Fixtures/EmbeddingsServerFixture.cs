using DotLLM.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Boots the real dotLLM server in-process, on a loopback port, with SmolLM2-135M-Instruct Q8_0
/// loaded on the CPU backend — for the <c>POST /v1/embeddings</c> HTTP tests (issue #451).
/// </summary>
/// <remarks>
/// The whole server is built through <see cref="ServerStartup.BuildApp"/>, not a hand-rolled
/// endpoint map, so the tests exercise the same JSON context, model activation and request gate
/// that production traffic hits. When the fixture model is not on disk, <see cref="SkipReason"/>
/// is set and every test skips with the list of probed locations.
/// </remarks>
public sealed class EmbeddingsServerFixture : IAsyncLifetime
{
    private WebApplication? _app;

    /// <summary>Client bound to the running server's base address, or null when skipping.</summary>
    public HttpClient? Client { get; private set; }

    /// <summary>Non-null when the fixture model could not be resolved; tests skip with this message.</summary>
    public string? SkipReason { get; private set; }

    /// <summary>Hidden size of the loaded model (the embedding dimension).</summary>
    public int HiddenSize { get; private set; }

    /// <summary>Context length of the loaded model.</summary>
    public int MaxSequenceLength { get; private set; }

    /// <summary>The server's model id — what requests must put in <c>"model"</c>.</summary>
    public string ModelId { get; } = "smollm2-135m-instruct";

    public async Task InitializeAsync()
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF",
            "bartowski", "SmolLM2-135M-Instruct-GGUF",
            "SmolLM2-135M-Instruct-Q8_0.gguf");

        if (!loc.Found)
        {
            SkipReason = loc.SkipMessage("SmolLM2-135M-Instruct Q8_0 (embeddings HTTP fixture)");
            return;
        }

        var options = new ServerOptions
        {
            Model = loc.Path!,
            Device = "cpu",
            ModelId = ModelId,
        };

        ServerState state;
        try
        {
            state = ServerStartup.LoadModel(loc.Path!, options);
        }
        catch (IOException ex)
        {
            SkipReason = $"Could not load {loc.Path}: {ex.Message}";
            return;
        }

        HiddenSize = state.Config!.HiddenSize;
        MaxSequenceLength = state.Config.MaxSequenceLength;

        var app = ServerStartup.BuildApp(state, ["--urls", "http://127.0.0.1:0"]);
        await app.StartAsync();
        _app = app;

        string address = app.Services.GetRequiredService<IServer>()
            .Features.Get<IServerAddressesFeature>()!
            .Addresses.First();

        Client = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromMinutes(2) };
    }

    public async Task DisposeAsync()
    {
        Client?.Dispose();
        if (_app is not null)
        {
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }
}

/// <summary>Collection definition so the server is booted once for the whole HTTP suite.</summary>
[CollectionDefinition("EmbeddingsHttp")]
public sealed class EmbeddingsHttpCollection : ICollectionFixture<EmbeddingsServerFixture>;
