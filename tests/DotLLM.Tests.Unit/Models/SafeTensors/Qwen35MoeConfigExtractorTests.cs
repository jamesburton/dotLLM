using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="Qwen35MoeConfigExtractor"/> (issue #266 Phase B)
/// against a shape mirroring the real
/// <c>SyzygyResearch/Mach-1-Additive-35B</c> / <c>Qwen/Qwen3.6-35B-A3B</c>
/// <c>config.json</c> (field names and values verified against the real
/// fixture's <c>text_config</c>), truncated to 4 layers — enough to exercise
/// one full <c>full_attention_interval=4</c> period (3 GDN + 1 full-attention,
/// matching layers 0-3 of the real 40-layer checkpoint exactly).
/// </summary>
public sealed class Qwen35MoeConfigExtractorTests
{
    private const string RealShapeJson = """
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "attn_output_gate": true,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": false,
            "vocab_size": 248320
        },
        "tie_word_embeddings": false,
        "vision_config": {
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "out_hidden_size": 2048
        }
    }
    """;

    private static JsonElement Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }

    [Fact]
    public void RealShape_ResolvesArchitectureAndCoreFields()
    {
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.Equal(Architecture.Qwen3MoeHybrid, cfg.Architecture);
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.Equal(4, cfg.NumLayers);
        Assert.Equal(16, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(256, cfg.HeadDim);
        Assert.Equal(248320, cfg.VocabSize);
        Assert.Equal(262144, cfg.MaxSequenceLength);
        Assert.Equal(1e-06f, cfg.NormEpsilon);
        Assert.False(cfg.TiedEmbeddings);
        Assert.Equal(AttentionType.GQA, cfg.AttentionType);
    }

    [Fact]
    public void RealShape_VisionConfigIsSkipped_TextTowerHoisted()
    {
        // The extractor must read hidden_size=2048 (text_config) not 1152 (vision_config) —
        // proves the vision_config sibling is correctly ignored, not silently merged.
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.NotEqual(1152, cfg.HiddenSize);
    }

    [Fact]
    public void RealShape_BuildsGdnConfigFromLinearAttnKeys()
    {
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.GdnConfig);
        GatedDeltaNetConfig gdn = cfg.GdnConfig!.Value;
        Assert.Equal(4, gdn.FullAttnInterval);
        Assert.Equal(32, gdn.NVHead);
        Assert.Equal(16, gdn.NKHead);
        Assert.Equal(128, gdn.DState);
        Assert.Equal(4, gdn.DConv);
        // DInner = 2*NKHead*DState + NVHead*DState = 2*16*128 + 32*128 = 4096 + 4096 = 8192,
        // matching the real fixture's in_proj_qkv.weight output width exactly.
        Assert.Equal(8192, gdn.DInner);
    }

    [Fact]
    public void RealShape_BuildsHybridLayoutFromLayerTypes_3Gdn1Attention()
    {
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.HybridLayout);
        HybridLayerLayout layout = cfg.HybridLayout!;
        Assert.Equal(HybridLayerKind.GatedDeltaNet, layout.LayerKind[0]);
        Assert.Equal(HybridLayerKind.GatedDeltaNet, layout.LayerKind[1]);
        Assert.Equal(HybridLayerKind.GatedDeltaNet, layout.LayerKind[2]);
        Assert.Equal(HybridLayerKind.Attention, layout.LayerKind[3]);
        Assert.Equal(0, layout.HeadCountKv[0]);
        Assert.Equal(2, layout.HeadCountKv[3]);
    }

    [Fact]
    public void RealShape_BuildsMoeConfigWithImplicitSingleSharedExpertAndGate()
    {
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.Moe);
        MoeConfig moe = cfg.Moe!;
        Assert.Equal(256, moe.NumExperts);
        Assert.Equal(8, moe.NumExpertsPerTok);
        Assert.Equal(512, moe.MoeIntermediateSize);
        Assert.True(moe.NormTopKProb);
        Assert.Equal(512, moe.SharedExpertIntermediateSize);
        Assert.Equal(1, moe.NumSharedExperts);
        Assert.True(moe.HasSharedExpertGate);
    }

    [Fact]
    public void RealShape_PartialRotaryBakedIntoRopeDimensionCount()
    {
        // head_dim=256, partial_rotary_factor=0.25 -> rope_dim=64 baked directly into
        // RoPEConfig.DimensionCount (NOT ModelConfig.PartialRotaryFactor, which the GDN
        // hybrid forward path does not consult — mirrors LoadFromGguf's convention).
        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.RoPEConfig);
        Assert.Equal(64, cfg.RoPEConfig!.Value.DimensionCount);
        Assert.Equal(10_000_000.0f, cfg.RoPEConfig!.Value.Theta);
        Assert.Null(cfg.PartialRotaryFactor);
    }

    [Fact]
    public void FlatTextConfig_TreatsRootAsTextTower()
    {
        // model_type=qwen3_5_moe_text at the root (no text_config wrapper) — the
        // extractor's fallback path for a synthetic/flat fixture.
        const string flat = """
        {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 1000,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64
        }
        """;

        ModelConfig cfg = Qwen35MoeConfigExtractor.Extract(Parse(flat));
        Assert.Equal(256, cfg.HiddenSize);
        Assert.Equal(4, cfg.NumLayers);
        Assert.NotNull(cfg.GdnConfig);
        Assert.NotNull(cfg.Moe);
        Assert.Null(cfg.Moe!.SharedExpertIntermediateSize);
        Assert.False(cfg.Moe.HasSharedExpertGate);
    }

    [Fact]
    public void WrongModelType_Throws()
    {
        const string wrong = """{ "model_type": "llama", "hidden_size": 4096 }""";
        Assert.Throws<InvalidDataException>(() => Qwen35MoeConfigExtractor.Extract(Parse(wrong)));
    }

    [Fact]
    public void MissingLinearKeyValueHeadDimMismatch_Throws()
    {
        const string mismatched = """
        {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "vocab_size": 1000,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64
        }
        """;
        Assert.Throws<InvalidDataException>(() => Qwen35MoeConfigExtractor.Extract(Parse(mismatched)));
    }
}
