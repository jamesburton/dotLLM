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
    /// "dedicated loader only" category. Before #729 a partial request threw an opaque NotSupportedException
    /// from HybridTransformerModel; a first fix fell back to CPU silently. Policy now: all-GPU if it fits,
    /// otherwise a clear error -- NEVER a CPU model.
    /// </summary>
    [SkippableFact]
    public void PartialRequest_OnUnsupportedArchitecture_WhenGpuCannotLoad_FailsActionably_NoCpuFallback()
    {
        Skip.If(CudaDevice.IsAvailable(), "CUDA present: the all-GPU load could succeed; this test needs a CPU-only host (no GPU work without gpu-lock).");

        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "tiny.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        Assert.True(config.NumLayers >= 2, "fixture must have >=2 layers so 1 is a genuine partial request");

        var warnings = new List<string>();
        var ex = Assert.Throws<InvalidOperationException>(() => CudaModelLoader.CreateForGpuLayers(
            gguf, config, requestedGpuLayers: 1, deviceId: 0, new ThreadingConfig(1, 1), warnings.Add));

        Assert.Contains($"1/{config.NumLayers}", ex.Message, StringComparison.Ordinal);
        Assert.Contains("--device cpu", ex.Message, StringComparison.Ordinal);
        Assert.Contains("/issues/735", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Nothing was run on the CPU", ex.Message, StringComparison.Ordinal);
        Assert.NotNull(ex.InnerException);
        Assert.Contains(warnings, w => w.Contains("not supported", StringComparison.Ordinal));
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
