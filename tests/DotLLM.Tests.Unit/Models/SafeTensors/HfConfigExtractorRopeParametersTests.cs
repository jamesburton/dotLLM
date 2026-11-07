using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.SafeTensors;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Issue #1667 (TwIL-8B S0): transformers 5.x nests <c>rope_theta</c> (and the scaling block) under <c>rope_parameters</c>.
/// The extractor used to read only top-level <c>rope_theta</c> (default 1e4), so the TwIL-8B safetensors would have loaded with
/// theta 1e4 instead of 1e7 and produced wrong text with no error.
/// </summary>
public sealed class HfConfigExtractorRopeParametersTests
{
    /// <summary>The real webAI-Official/TwIL-8B config.json (transformers 5.5.3), verbatim apart from whitespace.</summary>
    private const string TwilConfig = """
        {"architectures":["GraniteForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"attention_multiplier":0.0078125,
         "bos_token_id":100283,"dtype":"bfloat16","embedding_multiplier":1.0,"eos_token_id":100257,"hidden_act":"silu","hidden_size":4096,
         "initializer_range":0.1,"intermediate_size":12800,"logits_scaling":1.0,"max_position_embeddings":131072,"mlp_bias":false,
         "model_type":"granite","num_attention_heads":32,"num_hidden_layers":40,"num_key_value_heads":8,"pad_token_id":100257,
         "residual_multiplier":1.0,"rms_norm_eps":1e-05,"rope_parameters":{"rope_theta":10000000,"rope_type":"default"},
         "tie_word_embeddings":false,"transformers_version":"5.5.3","use_cache":true,"vocab_size":100352}
        """;

    private static string Llama(string ropeFragment) => $$"""
        {"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,
         "num_key_value_heads":2,"intermediate_size":128,"vocab_size":100,"max_position_embeddings":4096,{{ropeFragment}}"rms_norm_eps":1e-5}
        """;

    [Fact]
    public void TwilConfig_ReadsNestedRopeTheta_1e7()
    {
        ModelConfig c = HfConfigExtractor.Extract(TwilConfig);
        Assert.Equal(Architecture.Granite, c.Architecture);
        Assert.Equal(1e7f, c.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEScalingType.None, c.RoPEConfig!.Value.ScalingType); // rope_type=default
        Assert.Equal(40, c.NumLayers);
        Assert.Equal(128, c.HeadDim);
    }

    [Fact]
    public void NestedTheta_DiffersFromDefault_SoDroppingItIsObservable()
    {
        // Discriminates: two configs identical except for the nested theta must not extract the same RoPE base.
        float a = HfConfigExtractor.Extract(Llama("\"rope_parameters\":{\"rope_theta\":10000000,\"rope_type\":\"default\"},")).RoPEConfig!.Value.Theta;
        float b = HfConfigExtractor.Extract(Llama("\"rope_parameters\":{\"rope_theta\":500000,\"rope_type\":\"default\"},")).RoPEConfig!.Value.Theta;
        Assert.Equal(1e7f, a);
        Assert.Equal(5e5f, b);
    }

    [Fact]
    public void LegacyTopLevelTheta_StillRead_AndWinsOverNested()
    {
        Assert.Equal(1e6f, HfConfigExtractor.Extract(Llama("\"rope_theta\":1000000,")).RoPEConfig!.Value.Theta);
        Assert.Equal(1e6f, HfConfigExtractor.Extract(Llama("\"rope_theta\":1000000,\"rope_parameters\":{\"rope_theta\":7},")).RoPEConfig!.Value.Theta);
    }

    [Fact]
    public void NoRopeAtAll_KeepsDefault1e4()
        => Assert.Equal(10000f, HfConfigExtractor.Extract(Llama("")).RoPEConfig!.Value.Theta);

    [Fact]
    public void NestedYarnScaling_IsRead_LikeLegacyRopeScaling()
    {
        const string nested = "\"rope_parameters\":{\"rope_theta\":1000000,\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":1024},";
        const string legacy = "\"rope_theta\":1000000,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":1024},";
        RoPEConfig n = HfConfigExtractor.Extract(Llama(nested)).RoPEConfig!.Value;
        RoPEConfig l = HfConfigExtractor.Extract(Llama(legacy)).RoPEConfig!.Value;
        Assert.Equal(RoPEScalingType.YaRN, n.ScalingType);
        Assert.Equal(4.0f, n.ScalingFactor);
        Assert.Equal(1024, n.OrigMaxSeqLen);
        Assert.Equal(l, n);
    }
}
