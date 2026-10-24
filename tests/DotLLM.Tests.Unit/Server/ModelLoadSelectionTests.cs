using System.Text.Json;
using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Regression for #728: the load modal said "All 32 layers on GPU" but the server loaded a different file
/// (42 layers) and split it 32/10. Guards exact-file selection and the "all layers" sentinel.
/// </summary>
public sealed class ModelLoadSelectionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-728-" + Guid.NewGuid().ToString("N"));

    public ModelLoadSelectionTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static ServerState StateWithLoaded(string loadedPath) => new()
    {
        Options = new ServerOptions { Model = "test" },
        LoadedModelPath = loadedPath,
    };

    [Theory]
    [InlineData(-1, "gpu", 42, 42)]     // sentinel: all, as the loader counts. Old Clamp(-1,0,N) gave 0 (CPU).
    [InlineData(null, "gpu", 42, 42)]
    [InlineData(null, "cpu", 42, 0)]
    [InlineData(32, "gpu", 42, 32)]
    [InlineData(99, "gpu", 42, 42)]
    [InlineData(0, "gpu", 42, 0)]
    public void ResolveGpuLayers_UsesLoaderLayerCount(int? requested, string device, int numLayers, int expected) =>
        Assert.Equal(expected, ServerStartup.ResolveGpuLayers(requested, device, numLayers));

    [Fact]
    public void AllSentinel_OverridesProfileAndStartupCounts()
    {
        // A persisted profile / startup --gpu-layers 32 must not leak into an explicit "All".
        Assert.Null(ModelManagementEndpoint.ResolveRequestedGpuLayers(ServerStartup.AllGpuLayers, profile: 32, startup: 32));
        Assert.Equal(12, ModelManagementEndpoint.ResolveRequestedGpuLayers(12, 32, 32));
        Assert.Equal(32, ModelManagementEndpoint.ResolveRequestedGpuLayers(null, 32, 8));
        Assert.Equal(8, ModelManagementEndpoint.ResolveRequestedGpuLayers(null, null, 8));
    }

    [Fact]
    public void ResolveExactPath_ReturnsChosenFile_NotLargestSibling()
    {
        // Repo-dir resolution with no quant suffix picks the LARGEST .gguf; the exact path must win.
        string chosen = Path.Combine(_dir, "gemma-e4b.gguf");
        string bigger = Path.Combine(_dir, "gemma-e4b-big.gguf");
        File.WriteAllBytes(chosen, new byte[16]);
        File.WriteAllBytes(bigger, new byte[4096]);
        var state = StateWithLoaded(Path.Combine(_dir, "loaded.gguf"));

        Assert.Equal(Path.GetFullPath(chosen), ModelManagementEndpoint.ResolveExactPath(chosen, state));
    }

    [Fact]
    public void ResolveExactPath_RejectsMissingNonGgufAndOutsidePaths()
    {
        var state = StateWithLoaded(Path.Combine(_dir, "loaded.gguf"));
        File.WriteAllBytes(Path.Combine(_dir, "x.bin"), new byte[4]);
        string outside = Path.Combine(Path.GetTempPath(), "dotllm-728-outside.gguf");
        File.WriteAllBytes(outside, new byte[4]);
        try
        {
            Assert.Null(ModelManagementEndpoint.ResolveExactPath(Path.Combine(_dir, "missing.gguf"), state));
            Assert.Null(ModelManagementEndpoint.ResolveExactPath(Path.Combine(_dir, "x.bin"), state));
            Assert.Null(ModelManagementEndpoint.ResolveExactPath(outside, state));
        }
        finally { File.Delete(outside); }
    }

    [Fact]
    public void LoadRequest_RoundTripsPathAndAllSentinel()
    {
        var req = JsonSerializer.Deserialize(
            "{\"model\":\"o/r\",\"path\":\"C:\\\\m\\\\gemma-e4b.gguf\",\"gpu_layers\":-1}",
            ServerJsonContext.Default.ModelLoadRequest)!;
        Assert.Equal("C:\\m\\gemma-e4b.gguf", req.ModelPath);
        Assert.Equal(-1, req.GpuLayers);
    }

    [Fact]
    public void Inspect_AcceptsEveryListedModel_SoSliderTracksSelection()
    {
        // Models the picker lists from the HF hub cache / ollama store live outside ~/.dotllm/models. If inspect refuses them the slider
        // never updates (stays at its HTML default 32) whichever model is selected.
        string a = Path.Combine(_dir, "hub", "a-32layers.gguf");
        string b = Path.Combine(_dir, "ollama", "sha256-b42layers");
        var listed = new[]
        {
            new LocalModel("o/a", "a-32layers.gguf", a, 1, DateTime.UtcNow),
            new LocalModel("o/b", "sha256-b42layers", b, 1, DateTime.UtcNow),
        };
        var state = StateWithLoaded("");

        Assert.False(ModelInspectEndpoint.IsAllowedModelPath(Path.GetFullPath(a), state)); // refused by the directory rules alone
        Assert.True(ModelInspectEndpoint.IsListedModelPath(Path.GetFullPath(a), listed));
        Assert.True(ModelInspectEndpoint.IsListedModelPath(Path.GetFullPath(b), listed));
        Assert.False(ModelInspectEndpoint.IsListedModelPath(Path.GetFullPath(Path.Combine(_dir, "other.gguf")), listed));
    }
}
