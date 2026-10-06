using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>Issue #729 inspect payload tests.</summary>
public class ModelInspectPartialOffloadTests
{
    private static DotLLM.Core.Models.ModelConfig ConfigFor(DotLLM.Core.Configuration.Architecture arch) => new()
    {
        Architecture = arch, VocabSize = 32, HiddenSize = 8, IntermediateSize = 16, NumLayers = 52,
        NumAttentionHeads = 2, NumKvHeads = 2, HeadDim = 4, MaxSequenceLength = 16, NormEpsilon = 1e-5f,
    };

    /// <summary>Issue #729: the inspect payload tells the UI whether the GPU-layers slider is meaningful.</summary>
    [Theory]
    [InlineData(DotLLM.Core.Configuration.Architecture.NemotronH, false)]
    [InlineData(DotLLM.Core.Configuration.Architecture.Llama, true)]
    public void BuildResponse_ExposesPartialOffloadSupport_AndSurvivesJsonRoundTrip(
        DotLLM.Core.Configuration.Architecture arch, bool expected)
    {
        var response = ModelInspectEndpoint.BuildResponse(ConfigFor(arch), 123);
        Assert.Equal(expected, response.SupportsPartialOffload);

        string json = System.Text.Json.JsonSerializer.Serialize(response, ServerJsonContext.Default.ModelInspectResponse);
        Assert.Contains($"\"supports_partial_offload\":{(expected ? "true" : "false")}", json, StringComparison.Ordinal);
    }
}

/// <summary>
/// Tests for <see cref="ModelInspectEndpoint"/> path traversal protection.
/// Validates that only paths within allowed model directories are accepted.
/// </summary>
public class ModelInspectEndpointTests
{
    private static ServerState CreateState(string loadedModelPath = "")
    {
        return new ServerState
        {
            Options = new ServerOptions { Model = "test" },
            LoadedModelPath = loadedModelPath,
        };
    }

    [Fact]
    public void IsAllowedModelPath_RejectsAbsolutePathOutsideModelDirs()
    {
        using var state = CreateState();

        // Path outside of any allowed directory
        string outsidePath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "evil.gguf"));

        Assert.False(ModelInspectEndpoint.IsAllowedModelPath(outsidePath, state));
    }

    [Fact]
    public void IsAllowedModelPath_RejectsPathTraversal()
    {
        using var state = CreateState();

        // Attempt path traversal relative to models directory
        string modelsDir = HuggingFaceDownloader.DefaultModelsDirectory;
        string traversalPath = Path.GetFullPath(Path.Combine(modelsDir, "..", "..", "etc", "passwd"));

        Assert.False(ModelInspectEndpoint.IsAllowedModelPath(traversalPath, state));
    }

    [Fact]
    public void IsAllowedModelPath_AcceptsPathInModelsDirectory()
    {
        using var state = CreateState();

        string modelsDir = HuggingFaceDownloader.DefaultModelsDirectory;
        string validPath = Path.GetFullPath(Path.Combine(modelsDir, "owner", "repo", "model.gguf"));

        Assert.True(ModelInspectEndpoint.IsAllowedModelPath(validPath, state));
    }

    [Fact]
    public void IsAllowedModelPath_AcceptsPathInLoadedModelDirectory()
    {
        string loadedModelDir = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "my-models"));
        string loadedModelPath = Path.Combine(loadedModelDir, "loaded.gguf");
        using var state = CreateState(loadedModelPath);

        // A different file in the same directory should be allowed
        string siblingPath = Path.GetFullPath(Path.Combine(loadedModelDir, "other.gguf"));

        Assert.True(ModelInspectEndpoint.IsAllowedModelPath(siblingPath, state));
    }

    [Fact]
    public void IsAllowedModelPath_RejectsPathOutsideLoadedModelDirectory()
    {
        string loadedModelDir = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "my-models"));
        string loadedModelPath = Path.Combine(loadedModelDir, "loaded.gguf");
        using var state = CreateState(loadedModelPath);

        // Traversal from loaded model directory
        string traversalPath = Path.GetFullPath(Path.Combine(loadedModelDir, "..", "secret.gguf"));

        Assert.False(ModelInspectEndpoint.IsAllowedModelPath(traversalPath, state));
    }

    [Fact]
    public void IsAllowedModelPath_RejectsWhenNoModelLoaded_OutsideModelsDir()
    {
        using var state = CreateState(); // No loaded model

        string outsidePath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "probe.gguf"));

        Assert.False(ModelInspectEndpoint.IsAllowedModelPath(outsidePath, state));
    }
}
