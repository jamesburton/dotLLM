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
    /// Mixtral config is detected from <c>architectures[0] = "MixtralForCausalLM"</c>
    /// AND populates <see cref="Core.Models.MoeConfig"/> from
    /// <c>num_local_experts</c> / <c>num_experts_per_tok</c>. Copy of the
    /// <c>yujiepan/mixtral-tiny-random</c> config (2026-04).
    /// </summary>
    [Fact]
    public void Mixtral_TinyRandom_PopulatesMoeConfig()
    {
        const string json = """
        {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-5,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": false,
            "sliding_window": null
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Mixtral, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(8, cfg.Moe!.NumExperts);
        Assert.Equal(2, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(8, cfg.Moe.MoeIntermediateSize); // defaults to intermediate_size
        // Attention path stays GQA/RoPE — nothing Mixtral-specific there.
        Assert.Equal(4, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(1, cfg.HeadDim); // 4 / 4
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);
    }

    /// <summary>
    /// When <c>moe_intermediate_size</c> is declared explicitly (Phi-3.5-MoE
    /// convention), <see cref="Core.Models.MoeConfig.MoeIntermediateSize"/>
    /// should reflect that value, not the top-level <c>intermediate_size</c>.
    /// </summary>
    [Fact]
    public void Mixtral_OverrideMoeIntermediateSize_UsedOverTopLevel()
    {
        const string json = """
        {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 4, "intermediate_size": 64, "moe_intermediate_size": 32,
            "vocab_size": 100, "max_position_embeddings": 128,
            "num_local_experts": 4, "num_experts_per_tok": 2
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(32, cfg.Moe!.MoeIntermediateSize);
        Assert.Equal(64, cfg.IntermediateSize);
    }

    /// <summary>
    /// Non-MoE configs must leave <c>ModelConfig.Moe</c> null — the dense
    /// FFN path keys off that.
    /// </summary>
    [Fact]
    public void DenseLlama_NoMoeConfig()
    {
        const string json = """
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
         "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128}
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Null(cfg.Moe);
    }

    /// <summary>
    /// Declaring experts without a top-k should throw — misconfigured MoE is
    /// never silently ignored.
    /// </summary>
    [Fact]
    public void Mixtral_MissingNumExpertsPerTok_Throws()
    {
        const string json = """
        {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
         "hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 8, "vocab_size": 100, "max_position_embeddings": 128,
         "num_local_experts": 8}
        """;
        var ex = Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(json));
        Assert.Contains("num_experts_per_tok", ex.Message);
    }
}
