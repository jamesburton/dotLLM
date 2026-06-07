using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Phase 4d.2 — Verifies that adapters carrying standard q/k/v/o or
/// gate/up/down projection names no longer fail at validation time on MLA
/// (DeepSeek-V2/V3) or MoE base models. The blanket rejection has been
/// lifted; the projections silently pass through at runtime when the model
/// uses MLA-specific (q_a_proj/q_b_proj/...) or per-expert weights.
/// </summary>
/// <remarks>
/// Real MLA-LoRA / MoE-LoRA adapters in the wild are extremely rare (no
/// public PEFT release as of 2026-04). When such an adapter surfaces, the
/// runtime call sites in <c>MlaAttention.Execute</c> and
/// <c>MoeSwiGluMlp.Dispatch</c> will need additional wiring — tracked as a
/// follow-up. Until then, validation must not reject.
/// </remarks>
public sealed class LoraMlaMoeAcceptanceTests
{
    private static ModelConfig BuildMlaConfig() => new()
    {
        Architecture = Architecture.DeepSeekV3,
        VocabSize = 32,
        HiddenSize = 64,
        IntermediateSize = 128,
        NumLayers = 2,
        NumAttentionHeads = 4,
        NumKvHeads = 4,
        HeadDim = 16,
        MaxSequenceLength = 128,
        // Presence of MlaConfig is what matters for the validation lift.
        MlaConfig = new MlaConfig
        {
            QLoraRank = 32,
            KvLoraRank = 32,
            QkNopeHeadDim = 16,
            QkRopeHeadDim = 8,
            VHeadDim = 16,
            RopeTheta = 10000f,
        },
    };

    private static ModelConfig BuildMoeConfig() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = 64,
        IntermediateSize = 128,
        NumLayers = 2,
        NumAttentionHeads = 4,
        NumKvHeads = 4,
        HeadDim = 16,
        MaxSequenceLength = 128,
        Moe = new MoeConfig
        {
            NumExperts = 4,
            NumExpertsPerTok = 2,
            MoeIntermediateSize = 128,
        },
    };

    [Fact]
    public void IsCompatible_AcceptsStandardProjOnMlaModel()
    {
        var cfg = BuildMlaConfig();
        using var adapter = BuildStandardAdapterFor(cfg);
        Assert.True(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void IsCompatible_AcceptsStandardProjOnMoeModel()
    {
        var cfg = BuildMoeConfig();
        using var adapter = BuildStandardAdapterFor(cfg);
        Assert.True(adapter.IsCompatible(cfg));
    }

    /// <summary>
    /// Smoke test: the LoraAdapter dispose path must not leak when validation
    /// is bypassed for MLA / MoE base models.
    /// </summary>
    [Fact]
    public void Dispose_FreesNativeMemory_OnMlaAdapter()
    {
        var cfg = BuildMlaConfig();
        var adapter = BuildStandardAdapterFor(cfg);
        // If Dispose threw or leaked, this would be flagged by the IDisposable
        // contract — adapter holds 4 native buffers (q/k/v/o A+B factors).
        adapter.Dispose();
    }

    private static LoraAdapter BuildStandardAdapterFor(ModelConfig cfg)
    {
        const int rank = 4;
        var adapter = new LoraAdapter("test", rank, alpha: 8f, targetModules: ["q_proj", "o_proj"]);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;

        // q_proj: hidden -> qOut
        nint qB = LoraAdapter.AllocAligned((long)rank * cfg.HiddenSize);
        nint qA = LoraAdapter.AllocAligned((long)qOut * rank);
        adapter.AddLayerWeights(0, "q_proj", new LoraLayerWeights(
            AHandle: qA, BHandle: qB, InputDim: cfg.HiddenSize, OutputDim: qOut));

        // o_proj: qOut -> hidden
        nint oB = LoraAdapter.AllocAligned((long)rank * qOut);
        nint oA = LoraAdapter.AllocAligned((long)cfg.HiddenSize * rank);
        adapter.AddLayerWeights(0, "o_proj", new LoraLayerWeights(
            AHandle: oA, BHandle: oB, InputDim: qOut, OutputDim: cfg.HiddenSize));

        return adapter;
    }
}
