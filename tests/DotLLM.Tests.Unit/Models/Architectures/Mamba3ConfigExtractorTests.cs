using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Unit tests for <see cref="Mamba3ConfigExtractor"/>.
/// </summary>
/// <remarks>
/// Payload below is a byte-for-byte copy of the
/// <c>ib-ssm/mamba3-370M-10BT</c> HF config.json at commit
/// <c>02943831ad63d36783f41fa872f08cc8631538ee</c> (snapshot 2026-04-19).
/// Embedded rather than file-loaded so these remain true unit tests —
/// the integration project owns the filesystem copy.
/// </remarks>
public class Mamba3ConfigExtractorTests
{
    private const string IbSsm370MConfigJson = """
    {
      "A_floor": 0.0001,
      "architectures": [
        "Mamba3ForCausalLM"
      ],
      "bos_token_id": 1,
      "chunk_size": 64,
      "dt_init_floor": 0.0001,
      "dt_max": 0.1,
      "dt_min": 0.001,
      "dtype": "bfloat16",
      "eos_token_id": 2,
      "expand": 2,
      "fuse_cross_entropy": true,
      "fuse_linear_cross_entropy": false,
      "fuse_norm": true,
      "head_dim": 64,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "is_mimo": false,
      "is_outproj_norm": false,
      "mimo_rank": 4,
      "model_type": "mamba3",
      "n_groups": 1,
      "norm_eps": 1e-05,
      "num_heads": 32,
      "num_hidden_layers": 48,
      "pad_token_id": 0,
      "rescale_prenorm_residual": true,
      "residual_in_fp32": true,
      "rmsnorm": true,
      "rope_fraction": 0.5,
      "state_size": 128,
      "tie_word_embeddings": false,
      "transformers_version": "5.5.0",
      "use_bias": false,
      "use_cache": false,
      "use_l2warp": false,
      "vocab_size": 32000
    }
    """;

    [Fact]
    public void Extract_IbSsm370M_ArchitectureAndBasicSizes()
    {
        var config = Mamba3ConfigExtractor.Extract(IbSsm370MConfigJson);

        Assert.Equal(Architecture.Mamba3, config.Architecture);
        Assert.Equal(32000, config.VocabSize);
        Assert.Equal(1024, config.HiddenSize);
        Assert.Equal(48, config.NumLayers);
        Assert.Equal(32, config.NumAttentionHeads);
        Assert.Equal(64, config.HeadDim);
        Assert.Equal(1e-5f, config.NormEpsilon);
        Assert.False(config.TiedEmbeddings);
        // Mamba-3 has no attention and no conventional FFN on this checkpoint.
        Assert.Equal(0, config.IntermediateSize);
        Assert.Equal(PositionEncodingType.None, config.PositionEncodingType);
        Assert.Null(config.RoPEConfig);
        Assert.Null(config.SsmConfig);
        Assert.Null(config.HybridLayout);
    }

    [Fact]
    public void Extract_IbSsm370M_Mamba3ConfigPopulated()
    {
        var config = Mamba3ConfigExtractor.Extract(IbSsm370MConfigJson);

        Assert.NotNull(config.Mamba3Config);
        var m3 = config.Mamba3Config!;

        Assert.Equal(128, m3.StateSize);
        Assert.Equal(32, m3.NumHeads);
        Assert.Equal(64, m3.HeadDim);
        Assert.Equal(2, m3.Expand);
        Assert.Equal(1, m3.NumGroups);
        Assert.Equal(64, m3.ChunkSize);
        Assert.False(m3.IsMimo);
        Assert.Equal(4, m3.MimoRank);
        Assert.Equal(1e-4f, m3.AFloor);
        Assert.Equal(1e-4f, m3.DtInitFloor);
        Assert.Equal(1e-3f, m3.DtMin);
        Assert.Equal(0.1f, m3.DtMax);
        Assert.False(m3.UseL2Warp);
        Assert.Equal(0.5f, m3.RopeFraction);
        Assert.False(m3.IsOutProjNorm);
        Assert.True(m3.RescalePrenormResidual);
        Assert.True(m3.ResidualInFp32);

        // Derived quantities.
        Assert.Equal(32 * 64, m3.DInner);   // 2048
        Assert.Equal(128, m3.BcDim);        // SISO → d_state·G·R with G=R=1
        Assert.Equal(32, m3.NumRopeAngles); // int(128 * 0.5) / 2
        // d_in_proj = 2*2048 + 2*128 + 3*32 + 32 = 4480 — matches HF in_proj row count.
        Assert.Equal(4480, m3.InputProjectionDim);
    }

    [Fact]
    public void Extract_NumHeadsTimesHeadDim_MustEqualExpandTimesHiddenSize()
    {
        // Force an inconsistency: expand=2, hidden_size=1024 → d_inner=2048,
        // but num_heads=32 * head_dim=32 = 1024 ≠ 2048.
        string bad = IbSsm370MConfigJson.Replace("\"head_dim\": 64", "\"head_dim\": 32");
        var ex = Assert.Throws<InvalidDataException>(() => Mamba3ConfigExtractor.Extract(bad));
        Assert.Contains("num_heads*head_dim", ex.Message);
    }

    [Fact]
    public void Extract_StateSizeMustBeEven()
    {
        string bad = IbSsm370MConfigJson.Replace("\"state_size\": 128", "\"state_size\": 127");
        var ex = Assert.Throws<InvalidDataException>(() => Mamba3ConfigExtractor.Extract(bad));
        Assert.Contains("state_size", ex.Message);
    }

    [Fact]
    public void Extract_RejectsNonMamba3ModelType()
    {
        string bad = IbSsm370MConfigJson.Replace("\"model_type\": \"mamba3\"", "\"model_type\": \"llama\"");
        var ex = Assert.Throws<InvalidDataException>(() => Mamba3ConfigExtractor.Extract(bad));
        Assert.Contains("mamba3", ex.Message);
    }

    [Fact]
    public void Extract_RopeFractionOutOfRange_Throws()
    {
        string bad = IbSsm370MConfigJson.Replace("\"rope_fraction\": 0.5", "\"rope_fraction\": 1.5");
        var ex = Assert.Throws<InvalidDataException>(() => Mamba3ConfigExtractor.Extract(bad));
        Assert.Contains("rope_fraction", ex.Message);
    }

    [Fact]
    public void Extract_AcceptsJsonElementOverload()
    {
        using var doc = JsonDocument.Parse(IbSsm370MConfigJson);
        var config = Mamba3ConfigExtractor.Extract(doc.RootElement);
        Assert.Equal(Architecture.Mamba3, config.Architecture);
    }

    [Fact]
    public void Architecture_Mamba3_EnumVariantIsDiscoverable()
    {
        // Cross-check: the enum was extended in this change.
        Assert.Contains(Architecture.Mamba3, Enum.GetValues<Architecture>());
        Assert.Equal("Mamba3", Architecture.Mamba3.ToString());
    }
}
