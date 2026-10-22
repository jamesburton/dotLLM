using DotLLM.Core.Configuration;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// BitNet detection + config mapping for the HF <c>config.json</c> parser: <c>model_type=bitnet</c>
/// / <c>architectures=[BitNetForCausalLM]</c> → <see cref="Architecture.BitNet"/> with the
/// squared-ReLU FFN activation, mirroring the GGUF path.
/// </summary>
public sealed class HfConfigExtractorBitNetTests
{
    private const string BitNetConfig = """
    {
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": 2560,
        "num_hidden_layers": 30,
        "num_attention_heads": 20,
        "num_key_value_heads": 5,
        "intermediate_size": 6912,
        "vocab_size": 128256,
        "max_position_embeddings": 4096,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "hidden_act": "relu2",
        "tie_word_embeddings": true
    }
    """;

    [Fact]
    public void BitNet_ResolvesArchitectureAndReluSquaredActivation()
    {
        var cfg = HfConfigExtractor.Extract(BitNetConfig);

        Assert.Equal(Architecture.BitNet, cfg.Architecture);
        Assert.Equal(ActivationFunction.ReluSquared, cfg.ActivationFunction);
        Assert.Equal(2560, cfg.HiddenSize);
        Assert.Equal(30, cfg.NumLayers);
        Assert.Equal(20, cfg.NumAttentionHeads);
        Assert.Equal(5, cfg.NumKvHeads);
        Assert.Equal(6912, cfg.IntermediateSize);
        Assert.Equal(128256, cfg.VocabSize);
        Assert.True(cfg.TiedEmbeddings);
    }

    [Fact]
    public void BitNet_DetectedByModelTypeAlone()
    {
        const string json = """
        {
            "model_type": "bitnet",
            "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 512, "vocab_size": 100, "hidden_act": "relu2"
        }
        """;

        Assert.Equal(Architecture.BitNet, HfConfigExtractor.Extract(json).Architecture);
    }
}
