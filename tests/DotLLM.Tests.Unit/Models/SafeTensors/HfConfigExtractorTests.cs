using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="HfConfigExtractor"/> — the HuggingFace
/// <c>config.json</c> → <see cref="Core.Models.ModelConfig"/> parser.
/// </summary>
public sealed class HfConfigExtractorTests
{
    [Fact]
    public void Llama_MinimalConfig_PopulatesCoreFields()
    {
        const string json = """
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Llama, cfg.Architecture);
        Assert.Equal(128, cfg.HiddenSize);
        Assert.Equal(2, cfg.NumLayers);
        Assert.Equal(4, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(256, cfg.IntermediateSize);
        Assert.Equal(1000, cfg.VocabSize);
        Assert.Equal(512, cfg.MaxSequenceLength);
        Assert.Equal(32, cfg.HeadDim); // 128 / 4
        Assert.Equal(1e-5f, cfg.NormEpsilon);
        Assert.Equal(PositionEncodingType.RoPE, cfg.PositionEncodingType);
        Assert.NotNull(cfg.RoPEConfig);
        Assert.Equal(500000.0f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig.Value.Type);
        Assert.False(cfg.TiedEmbeddings);
    }

    [Fact]
    public void Mistral_UsesNormRoPE()
    {
        const string json = """
        {
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 500, "max_position_embeddings": 256,
            "sliding_window": 64
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Mistral, cfg.Architecture);
        Assert.Equal(4, cfg.NumKvHeads); // defaults to num_attention_heads
        Assert.Equal(64, cfg.SlidingWindowSize);
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);
    }

    [Fact]
    public void Phi_UsesNeoXRoPE_AndTiesByDefault()
    {
        const string json = """
        {
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 192, "vocab_size": 500, "max_position_embeddings": 256
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Phi, cfg.Architecture);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.True(cfg.TiedEmbeddings);
    }

    [Fact]
    public void Qwen_UsesNeoXRoPE_AndExplicitHeadDim()
    {
        const string json = """
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256, "vocab_size": 500, "max_position_embeddings": 256,
            "head_dim": 48,
            "tie_word_embeddings": false
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Qwen, cfg.Architecture);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.Equal(48, cfg.HeadDim);
        Assert.False(cfg.TiedEmbeddings);
    }

    [Fact]
    public void NullNumKvHeads_FallsBackToAttentionHeads()
    {
        // HF checkpoints sometimes emit `"num_key_value_heads": null` to mean
        // "use num_attention_heads". JSON null must not crash the parser.
        const string json = """
        {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": null,
            "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(4, cfg.NumKvHeads);
    }

    [Fact]
    public void UnsupportedArchitecture_Throws()
    {
        const string json = """
        {"architectures": ["BertForMaskedLM"], "model_type": "bert",
         "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128}
        """;
        var ex = Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(json));
        Assert.Contains("Unsupported HF architecture", ex.Message);
    }

    /// <summary>
    /// SmolLM3-3B detection on the real-world HF config (2026-05 snapshot).
    /// <c>architectures[0]=SmolLM3ForCausalLM</c>, <c>model_type=smollm3</c>,
    /// GQA-4, NeoX RoPE, NoPE on every 4th layer (indices 3, 7, 11, ... in
    /// the 36-layer SKU).
    /// </summary>
    [Fact]
    public void SmolLM3_3B_DetectsArchAndParsesNoPeLayers()
    {
        // Authoritative copy of HuggingFaceTB/SmolLM3-3B/config.json's
        // shape (vocab/hidden/heads), with the canonical 36-element
        // no_rope_layers pattern (1,1,1,0) × 9. The roadmap step 56
        // acceptance test specifies the resulting NoPE index set
        // {3, 7, 11, 15, 19, 23, 27, 31, 35}.
        const string json = """
        {
            "architectures": ["SmolLM3ForCausalLM"],
            "model_type": "smollm3",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 11008,
            "vocab_size": 128256,
            "max_position_embeddings": 65536,
            "rope_theta": 5000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "no_rope_layer_interval": 4,
            "no_rope_layers": [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0]
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.SmolLM3, cfg.Architecture);
        Assert.Equal(36, cfg.NumLayers);
        Assert.Equal(16, cfg.NumAttentionHeads);
        Assert.Equal(4, cfg.NumKvHeads);
        Assert.Equal(128, cfg.HeadDim);
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.Equal(11008, cfg.IntermediateSize);
        Assert.Equal(128256, cfg.VocabSize);
        Assert.Equal(65536, cfg.MaxSequenceLength);
        Assert.Equal(5_000_000.0f, cfg.RoPEConfig!.Value.Theta);
        // HF transformers' SmolLM3 inherits rotate_half from Llama.
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig.Value.Type);
        Assert.True(cfg.TiedEmbeddings);

        // NoPE mask: HF stores 1 = apply RoPE, 0 = skip RoPE. Extractor
        // inverts to the indices that SKIP RoPE.
        Assert.NotNull(cfg.NoRopeLayers);
        Assert.Equal(
            new[] { 3, 7, 11, 15, 19, 23, 27, 31, 35 },
            cfg.NoRopeLayers!.ToArray());
        Assert.True(cfg.IsNoRopeLayer(3));
        Assert.True(cfg.IsNoRopeLayer(35));
        Assert.False(cfg.IsNoRopeLayer(0));
        Assert.False(cfg.IsNoRopeLayer(1));
        Assert.False(cfg.IsNoRopeLayer(2));
    }

    /// <summary>
    /// SmolLM3 without <c>no_rope_layers</c> (e.g. a hypothetical "every
    /// layer keeps RoPE" SKU) must leave <see cref="Core.Models.ModelConfig.NoRopeLayers"/>
    /// null so the forward path skips the gating altogether (zero cost
    /// when feature is absent).
    /// </summary>
    [Fact]
    public void SmolLM3_NoNoRopeLayers_FieldIsNull()
    {
        const string json = """
        {
            "architectures": ["SmolLM3ForCausalLM"],
            "model_type": "smollm3",
            "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
            "num_key_value_heads": 2, "intermediate_size": 128,
            "vocab_size": 100, "max_position_embeddings": 256
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.SmolLM3, cfg.Architecture);
        Assert.Null(cfg.NoRopeLayers);
        Assert.False(cfg.IsNoRopeLayer(0));
        Assert.False(cfg.IsNoRopeLayer(3));
    }

    /// <summary>
    /// SmolLM3 with a <c>rope_scaling</c> YaRN block (the 128k long-context
    /// SKU) populates <see cref="Core.PositionEncoding.RoPEConfig.ScalingType"/>,
    /// the factor, <c>original_max_position_embeddings</c>, and the YaRN
    /// beta_fast/beta_slow defaults. The roadmap step 56 acceptance test
    /// specifies this pathway: with <c>original_max_position_embeddings=4096</c>
    /// and a position beyond it the RoPE frequency must be YaRN-scaled.
    /// </summary>
    [Fact]
    public void SmolLM3_YarnRopeScaling_PopulatesRopeConfig()
    {
        const string json = """
        {
            "architectures": ["SmolLM3ForCausalLM"],
            "model_type": "smollm3",
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "intermediate_size": 128,
            "vocab_size": 100, "max_position_embeddings": 131072,
            "rope_theta": 5000000.0,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096
            },
            "no_rope_layers": [1,0]
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.SmolLM3, cfg.Architecture);
        Assert.Equal(RoPEScalingType.YaRN, cfg.RoPEConfig!.Value.ScalingType);
        Assert.Equal(32.0f, cfg.RoPEConfig.Value.ScalingFactor);
        Assert.Equal(4096, cfg.RoPEConfig.Value.OrigMaxSeqLen);
        Assert.Equal(32.0f, cfg.RoPEConfig.Value.BetaFast); // default
        Assert.Equal(1.0f, cfg.RoPEConfig.Value.BetaSlow); // default
        // no_rope_layers=[1,0] -> only layer index 1 skips RoPE.
        Assert.Equal(new[] { 1 }, cfg.NoRopeLayers!.ToArray());
    }

    // ───────────────────── Gemma 3 ─────────────────────

    /// <summary>
    /// Text-only Gemma 3 checkpoint: <c>model_type=gemma3_text</c>,
    /// <c>architectures[0]=Gemma3TextForCausalLM</c>. Verifies the activation flips to
    /// <see cref="ActivationFunction.GELUTanh"/>, the four-norm Gemma layout knobs land,
    /// soft-cap fields propagate, and the per-layer attention-type list follows the
    /// <c>sliding_window_pattern</c> formula <c>(i + 1) % pattern == 0 ⇒ full</c>.
    /// </summary>
    [Fact]
    public void Gemma3_TextOnly_PopulatesGemmaFields()
    {
        const string json = """
        {
            "architectures": ["Gemma3TextForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": 64,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 128,
            "vocab_size": 256,
            "max_position_embeddings": 1024,
            "head_dim": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 512,
            "sliding_window_pattern": 3,
            "query_pre_attn_scalar": 256,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Gemma3, cfg.Architecture);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(NormType.RMSNorm, cfg.NormType);
        Assert.Equal(512, cfg.SlidingWindowSize);
        Assert.Equal(50f, cfg.AttnLogitSoftcap);
        Assert.Equal(30f, cfg.FinalLogitSoftcap);
        Assert.Equal(256f, cfg.QueryPreAttnScalar);
        Assert.True(cfg.TiedEmbeddings, "Gemma 3 default ties word embeddings.");

        // Per-layer attention pattern with sliding_window_pattern=3 on 6 layers:
        // (i+1) % 3 == 0 ⇒ full, else sliding.
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        var perLayer = cfg.PerLayerSlidingWindow!;
        Assert.Equal(6, perLayer.Count);
        Assert.Equal(512, perLayer[0]);
        Assert.Equal(512, perLayer[1]);
        Assert.Null(perLayer[2]);   // full attention every 3rd layer
        Assert.Equal(512, perLayer[3]);
        Assert.Equal(512, perLayer[4]);
        Assert.Null(perLayer[5]);
    }

    /// <summary>
    /// Multimodal Gemma 3 checkpoint (<c>model_type=gemma3</c>,
    /// <c>architectures[0]=Gemma3ForConditionalGeneration</c>) embeds the text-tower
    /// config under a <c>text_config</c> sub-object. The extractor must hoist that
    /// sub-object so subsequent field lookups see the text-tower shape, and the
    /// architecture still resolves to <see cref="Architecture.Gemma3"/>.
    /// </summary>
    [Fact]
    public void Gemma3_Multimodal_HoistsTextConfig()
    {
        const string json = """
        {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 128,
                "vocab_size": 1000,
                "max_position_embeddings": 1024,
                "head_dim": 32,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window": 1024,
                "sliding_window_pattern": 2,
                "query_pre_attn_scalar": 168,
                "attn_logit_softcapping": null,
                "final_logit_softcapping": null
            },
            "vision_config": { "model_type": "siglip_vision_model" }
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Gemma3, cfg.Architecture);
        Assert.Equal(32, cfg.HiddenSize);
        Assert.Equal(2, cfg.NumLayers);
        Assert.Equal(1000, cfg.VocabSize);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(1024, cfg.SlidingWindowSize);
        Assert.Null(cfg.AttnLogitSoftcap);     // null in JSON ⇒ null in config
        Assert.Null(cfg.FinalLogitSoftcap);
        Assert.Equal(168f, cfg.QueryPreAttnScalar);

        // sliding_window_pattern=2 on 2 layers → [sliding, full]
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        Assert.Equal(2, cfg.PerLayerSlidingWindow!.Count);
        Assert.Equal(1024, cfg.PerLayerSlidingWindow[0]);
        Assert.Null(cfg.PerLayerSlidingWindow[1]);
    }

    /// <summary>
    /// Explicit <c>layer_types</c> array overrides the
    /// <c>sliding_window_pattern</c> formula. Supports HF's newer convention where the
    /// per-layer pattern is shipped verbatim.
    /// </summary>
    [Fact]
    public void Gemma3_LayerTypesArray_OverridesSlidingWindowPattern()
    {
        const string json = """
        {
            "architectures": ["Gemma3TextForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 64, "vocab_size": 100,
            "max_position_embeddings": 512,
            "sliding_window": 128,
            "sliding_window_pattern": 6,
            "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        Assert.Equal(4, cfg.PerLayerSlidingWindow!.Count);
        Assert.Null(cfg.PerLayerSlidingWindow[0]);    // full
        Assert.Equal(128, cfg.PerLayerSlidingWindow[1]); // sliding
        Assert.Null(cfg.PerLayerSlidingWindow[2]);
        Assert.Equal(128, cfg.PerLayerSlidingWindow[3]);
    }
}
