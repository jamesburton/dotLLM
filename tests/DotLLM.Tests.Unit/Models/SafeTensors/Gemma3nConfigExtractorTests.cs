using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="Gemma3nConfigExtractor"/> against a shape mirroring
/// the real <c>google/gemma-3n-E4B-it</c> / <c>unsloth/gemma-3n-E4B-it</c>
/// <c>config.json</c> (field names and values verified against the published
/// checkpoint's <c>text_config</c>), scaled down to 5 layers (the real 35-layer
/// pattern's first "full-attention-every-5th" period: 4 sliding + 1 full).
/// </summary>
public sealed class Gemma3nConfigExtractorTests
{
    private const string RealShapeJson = """
    {
        "architectures": ["Gemma3nForConditionalGeneration"],
        "model_type": "gemma3n",
        "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": 2048,
            "num_hidden_layers": 5,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": [16384, 16384, 16384, 16384, 16384],
            "vocab_size": 262400,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "sliding_window": 512,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "num_kv_shared_layers": 2,
            "altup_active_idx": 0,
            "altup_coef_clip": 120.0,
            "altup_correct_scale": true,
            "altup_num_inputs": 4,
            "laurel_rank": 64,
            "activation_sparsity_pattern": [0.95, 0.95, 0.0, 0.0, 0.0]
        }
    }
    """;

    private static JsonElement Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }

    [Fact]
    public void RealShape_ResolvesGemma3nArchitectureAndCoreFields()
    {
        var cfg = Gemma3nConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.Equal(Architecture.Gemma3n, cfg.Architecture);
        Assert.True(cfg.IsGemmaArchitecture);
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.Equal(5, cfg.NumLayers);
        Assert.Equal(8, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(256, cfg.HeadDim);
        Assert.Equal(16384, cfg.IntermediateSize);
        Assert.Equal(262400, cfg.VocabSize);
        Assert.Equal(30.0f, cfg.FinalLogitSoftcap);
        Assert.NotNull(cfg.EmbeddingScale);
        Assert.Equal(MathF.Sqrt(2048), cfg.EmbeddingScale);
    }

    [Fact]
    public void RealShape_DualRopeNoGlobalHeadDimOverride()
    {
        // Gemma-3n, unlike Gemma-4, does NOT ship a distinct global head-dim /
        // global KV-head count / partial-rotary factor: only the RoPE base differs
        // between sliding (rope_local_base_freq) and full (rope_theta) layers.
        var cfg = Gemma3nConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.Equal(10_000.0f, cfg.RoPEConfig!.Value.Theta);
        Assert.NotNull(cfg.GlobalRoPEConfig);
        Assert.Equal(1_000_000.0f, cfg.GlobalRoPEConfig!.Value.Theta);
        Assert.Null(cfg.PartialRotaryFactor);
        Assert.Null(cfg.NumGlobalKvHeads);
        Assert.Null(cfg.GlobalHeadDim);
    }

    [Fact]
    public void RealShape_PopulatesPerLayerEmbeddingAndSharedKv()
    {
        var cfg = Gemma3nConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.PerLayerEmbedding);
        Assert.Equal(256, cfg.PerLayerEmbedding!.PerLayerDim);
        Assert.Equal(262144, cfg.PerLayerEmbedding.VocabSize);

        Assert.Equal(2, cfg.NumSharedKvLayers);
        Assert.True(cfg.LayerHasOwnKv(0));
        Assert.True(cfg.LayerHasOwnKv(2));
        Assert.False(cfg.LayerHasOwnKv(3)); // trailing 2 of 5 layers share KV
        Assert.False(cfg.LayerHasOwnKv(4));
    }

    [Fact]
    public void RealShape_PopulatesGemma3nAltUpLaurelSparsity()
    {
        var cfg = Gemma3nConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.Gemma3n);
        var g3n = cfg.Gemma3n!;
        Assert.Equal(4, g3n.NumInputs);
        Assert.Equal(0, g3n.ActiveIdx);
        Assert.Equal(120.0f, g3n.CoefClip);
        Assert.True(g3n.CorrectOutputScale);
        Assert.Equal(64, g3n.LaurelRank);

        Assert.Equal(5, g3n.ActivationSparsityPattern.Count);
        Assert.Equal(0.95f, g3n.ActivationSparsityPattern[0]);
        Assert.Equal(0.95f, g3n.ActivationSparsityPattern[1]);
        Assert.Equal(0.0f, g3n.ActivationSparsityPattern[2]);
    }

    [Fact]
    public void RealShape_PerLayerSlidingWindowMatchesLayerTypes()
    {
        var cfg = Gemma3nConfigExtractor.Extract(Parse(RealShapeJson));

        Assert.NotNull(cfg.PerLayerSlidingWindow);
        Assert.Equal(512, cfg.PerLayerSlidingWindow![0]);
        Assert.Equal(512, cfg.PerLayerSlidingWindow[3]);
        Assert.Null(cfg.PerLayerSlidingWindow[4]); // full attention (5th layer)
    }

    [Fact]
    public void HeterogeneousIntermediateSizeArray_Throws()
    {
        const string json = """
        {
            "model_type": "gemma3n_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": [128, 256],
            "vocab_size": 32
        }
        """;

        Assert.Throws<InvalidDataException>(() => Gemma3nConfigExtractor.Extract(Parse(json)));
    }

    [Fact]
    public void WrongModelType_Throws()
    {
        const string json = """{ "model_type": "gemma4_text" }""";
        Assert.Throws<InvalidDataException>(() => Gemma3nConfigExtractor.Extract(Parse(json)));
    }
}
