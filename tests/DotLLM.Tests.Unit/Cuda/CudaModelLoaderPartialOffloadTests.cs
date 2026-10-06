using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>Issue #729, end to end on the dispatcher shared by serve / run / chat.</summary>
public sealed class CudaModelLoaderPartialOffloadTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-partial-offload-{Guid.NewGuid():N}");

    public CudaModelLoaderPartialOffloadTests() => Directory.CreateDirectory(_scratch);

    public void Dispose() { try { Directory.Delete(_scratch, recursive: true); } catch { } }

    /// <summary>
    /// Qwen3MoeHybrid (Gated-DeltaNet layers, no attn_output) stands in for Nemotron-H: same
    /// "dedicated loader only" category. Before the fix a partial request threw NotSupportedException
    /// from HybridTransformerModel; now it warns and falls back (all-GPU, else CPU).
    /// </summary>
    [SkippableFact]
    public void PartialRequest_OnUnsupportedArchitecture_WarnsAndFallsBack_InsteadOfThrowing()
    {
        Skip.If(CudaDevice.IsAvailable(), "CUDA present: the fallback would load on the GPU; this test is CPU-only (no GPU work without gpu-lock).");

        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "tiny.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        Assert.True(config.NumLayers >= 2, "fixture must have >=2 layers so 1 is a genuine partial request");

        var warnings = new List<string>();
        var (model, kv) = CudaModelLoader.CreateForGpuLayers(
            gguf, config, requestedGpuLayers: 1, deviceId: 0, new ThreadingConfig(1, 1), warnings.Add);
        using var _m = model;

        Assert.IsNotType<HybridTransformerModel>(model);
        Assert.Null(kv); // CPU fallback model
        Assert.Contains(warnings, w => w.Contains("not supported", StringComparison.Ordinal));
        Assert.Contains(warnings, w => w.Contains("using the CPU", StringComparison.Ordinal));
    }

    [Fact]
    public void ZeroGpuLayers_LoadsOnCpu_WithoutWarnings()
    {
        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "tiny.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        var warnings = new List<string>();
        var (model, kv) = CudaModelLoader.CreateForGpuLayers(gguf, config, 0, 0, new ThreadingConfig(1, 1), warnings.Add);
        using var _m = model;

        Assert.Null(kv);
        Assert.Empty(warnings);
    }
}
