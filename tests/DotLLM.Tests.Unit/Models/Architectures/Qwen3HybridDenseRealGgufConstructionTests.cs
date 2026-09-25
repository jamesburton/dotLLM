using DotLLM.Core.Configuration;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Real-weight construction smoke test for <see cref="Qwen3HybridDenseTransformerModel"/>
/// against PrismML's actual <c>Ternary-Bonsai-27B-Q2_0.gguf</c> (issue #157). Confirms the
/// full loader path — GGUF parsing, config extraction (<c>qwen35</c> → dense GDN/attention
/// hybrid layout), per-layer weight loading (GDN, full-attention, and dense FFN tensor
/// lookups) — completes without throwing on the real 27B checkpoint. Does NOT run a forward
/// pass (that's covered separately, gated on real hardware for the CUDA path).
/// </summary>
public sealed class Qwen3HybridDenseRealGgufConstructionTests
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    [SkippableFact]
    public void LoadFromGguf_RealBonsai27B_ConstructsWithoutThrowing()
    {
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        Assert.Equal(64, config.NumLayers);
        Assert.Equal(5120, config.HiddenSize);
        Assert.Equal(17408, config.IntermediateSize);
        Assert.NotNull(config.GdnConfig);
        Assert.NotNull(config.HybridLayout);
        Assert.Null(config.Moe);

        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(
            gguf, config, ThreadingConfig.SingleThreaded);

        Assert.Equal(config, model.Config);
        // full_attention_interval=4 over 64 layers -> exactly 16 full-attention layers,
        // 48 GDN layers.
        Assert.Equal(16, model.AttentionLayerCount);
    }

    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }
}
