using DotLLM.Core.Configuration;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Config-level bridge test for the identity-MoTE export (trackM-mote →
/// <c>scripts/lora/mote_export.py</c>). A depth-expanded BitNet whose inserted
/// layers are per-layer top-1 MoE blocks is expressed entirely with dotLLM's
/// EXISTING <see cref="Core.Models.MoeConfig"/> knobs:
/// <list type="bullet">
///   <item><c>model_type=bitnet</c> → <see cref="Architecture.BitNet"/> (ternary I2_S linears reused).</item>
///   <item><c>num_experts=K+1</c>, <c>num_experts_per_tok=1</c> → routed top-1.</item>
///   <item><c>decoder_sparse_step=1</c> + <c>mlp_only_layers=[all original layer indices]</c>
///         → only the inserted layers resolve as MoE via
///         <see cref="Core.Models.MoeConfig.IsMoeLayer(int)"/>.</item>
/// </list>
/// This test proves the exported <c>config.json</c> is consumed correctly by the
/// current extractor with NO code changes — the config surface already fits.
/// The remaining dotLLM build (BitNet-MoE weight loader + relu2/ffn_sub_norm/router-bias
/// forward) is tracked by <see cref="MoteBitNetMoeLoaderScaffoldTests"/>.
/// </summary>
public sealed class HfConfigExtractorMoteTests
{
    // Mirrors mote_export.build_dotllm_config output for a 6-layer expanded model
    // with identity blocks inserted at final indices 1, 3, 5 (originals 0, 2, 4).
    private const string MoteConfig = """
    {
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "relu2",
        "tie_word_embeddings": false,
        "num_experts": 3,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 128,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0, 2, 4]
    }
    """;

    [Fact]
    public void Mote_ResolvesBitNetWithReluSquared()
    {
        var cfg = HfConfigExtractor.Extract(MoteConfig);

        Assert.Equal(Architecture.BitNet, cfg.Architecture);
        Assert.Equal(ActivationFunction.ReluSquared, cfg.ActivationFunction);
        Assert.Equal(6, cfg.NumLayers);
    }

    [Fact]
    public void Mote_SurfacesTopOneMoeConfig()
    {
        var cfg = HfConfigExtractor.Extract(MoteConfig);

        Assert.NotNull(cfg.Moe);
        Assert.Equal(3, cfg.Moe!.NumExperts);        // skip expert 0 + 2 capability experts
        Assert.Equal(1, cfg.Moe.NumExpertsPerTok);   // supervised top-1
        Assert.Equal(128, cfg.Moe.MoeIntermediateSize);
        Assert.True(cfg.Moe.NormTopKProb);
    }

    [Fact]
    public void Mote_OnlyInsertedLayersAreMoe()
    {
        var cfg = HfConfigExtractor.Extract(MoteConfig);
        var moe = cfg.Moe!;

        // Inserted (identity-MoTE) layers → MoE.
        Assert.True(moe.IsMoeLayer(1));
        Assert.True(moe.IsMoeLayer(3));
        Assert.True(moe.IsMoeLayer(5));
        // Original BitNet layers → dense (force-dense via mlp_only_layers).
        Assert.False(moe.IsMoeLayer(0));
        Assert.False(moe.IsMoeLayer(2));
        Assert.False(moe.IsMoeLayer(4));
    }
}
