using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Olala (#1970) config contract: the keys that carry numerics (<c>attn_logit_softcapping</c>, <c>gate_bias</c>) are required rather than
/// defaulted to 0, and an HF Olala <c>config.json</c> is recognised and refused with an actionable message. Synthetic metadata, no files.
/// </summary>
public sealed class OlalaConfigRefusalTests
{
    private static Dictionary<string, GgufMetadataValue> Valid()
    {
        GgufMetadataValue U(uint v) => new(GgufValueType.UInt32, v);
        var d = new Dictionary<string, GgufMetadataValue>
        {
            ["general.architecture"] = new(GgufValueType.String, "olala"),
            ["olala.vocab_size"] = U(96),
            ["olala.embedding_length"] = U(64), ["olala.block_count"] = U(2),
            ["olala.attention.head_count"] = U(8), ["olala.attention.head_count_kv"] = U(2),
            ["olala.attention.key_length"] = U(16), ["olala.attention.value_length"] = U(16),
            ["olala.attention.layer_norm_rms_epsilon"] = new(GgufValueType.Float32, 1e-6f),
            ["olala.attention.sliding_window"] = U(32),
            ["olala.attn_logit_softcapping"] = new(GgufValueType.Float32, 150f),
            ["olala.gate_bias"] = new(GgufValueType.Float32, 1.15f),
            ["olala.layer_types"] = new(GgufValueType.String, "MV"),
            ["olala.geodesic_update"] = new(GgufValueType.Bool, true),
            ["olala.final_norm"] = new(GgufValueType.Bool, false),
            ["olala.ssm.group_count"] = U(1), ["olala.ssm.inner_size"] = U(128), ["olala.ssm.state_size"] = U(16),
            ["olala.mamba_headdim"] = U(16), ["olala.mamba_mimo_dim"] = U(2), ["olala.num_rope_angles"] = U(4),
            ["olala.num_signal_heads"] = U(6), ["olala.num_noise_heads"] = U(2), ["olala.tpa_rank"] = U(2),
            ["olala.expert_count"] = U(8), ["olala.expert_used_count"] = U(2), ["olala.expert_feed_forward_length"] = U(16),
        };
        return d;
    }

    [Fact]
    public void ValidMetadata_Extracts()
    {
        var cfg = GgufModelConfigExtractor.Extract(new GgufMetadata(Valid()));
        Assert.Equal(Architecture.Olala, cfg.Architecture);
        Assert.Equal(150f, cfg.AttnLogitSoftcap);
        Assert.Equal(1.15f, cfg.Olala!.GateBias);
    }

    [Theory]
    [InlineData("olala.attn_logit_softcapping")]
    [InlineData("olala.gate_bias")]
    public void MissingNumericsKey_IsRefused_NotDefaultedToZero(string key)
    {
        var d = Valid();
        Assert.True(d.Remove(key));
        var ex = Assert.Throws<InvalidDataException>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));
        Assert.Contains(key, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"architectures":["OlalaForCausalLM"],"model_type":"olala","hidden_size":64}""")]
    [InlineData("""{"model_type":"olala","hidden_size":64}""")]
    public void HfOlalaConfig_IsRecognised_AndRefusedExplicitly(string json)
    {
        var ex = Assert.Throws<NotSupportedException>(() => HfConfigExtractor.Extract(json));
        Assert.Contains("olala", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("GGUF", ex.Message, StringComparison.Ordinal);
    }
}
