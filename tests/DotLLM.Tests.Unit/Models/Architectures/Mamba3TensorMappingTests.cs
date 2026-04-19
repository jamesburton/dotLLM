using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Unit tests for <see cref="Mamba3TensorMapping"/>. Cross-references the
/// mapping against the HF tensor-name signature captured from
/// <c>ib-ssm/mamba3-370M-10BT</c> commit
/// <c>02943831ad63d36783f41fa872f08cc8631538ee</c> via a safetensors header
/// Range request on 2026-04-19 (see
/// <c>tests/DotLLM.Tests.Integration/Fixtures/Mamba3/ib_ssm_config.README.md</c>).
/// </summary>
public class Mamba3TensorMappingTests
{
    /// <summary>
    /// Exact per-layer tensor name suffixes from the 2026-04-19 header probe
    /// of <c>ib-ssm/mamba3-370M-10BT</c>. Sorted to match HF's
    /// alphabetical ordering so drift is obvious.
    /// </summary>
    private static readonly string[] ExpectedPerLayerSuffixes =
    [
        "mixer.B_bias",
        "mixer.B_norm.weight",
        "mixer.C_bias",
        "mixer.C_norm.weight",
        "mixer.D",
        "mixer.dt_bias",
        "mixer.in_proj.weight",
        "mixer.out_proj.weight",
        "norm.weight",
    ];

    [Fact]
    public void Globals_MatchHfNames()
    {
        Assert.Equal("backbone.embeddings.weight", Mamba3TensorMapping.TokenEmbedding);
        Assert.Equal("backbone.norm_f.weight", Mamba3TensorMapping.FinalNorm);
        Assert.Equal("lm_head.weight", Mamba3TensorMapping.LmHead);
    }

    [Fact]
    public void LayerAccessors_FormatWithBareIndex()
    {
        // No zero-padding — HF uses 'backbone.layers.0.*', not 'backbone.layers.00.*'.
        Assert.Equal("backbone.layers.0.norm.weight", Mamba3TensorMapping.LayerNorm(0));
        Assert.Equal("backbone.layers.47.mixer.in_proj.weight", Mamba3TensorMapping.InProj(47));
        Assert.Equal("backbone.layers.3.mixer.B_norm.weight", Mamba3TensorMapping.BNorm(3));
        Assert.Equal("backbone.layers.12.mixer.C_norm.weight", Mamba3TensorMapping.CNorm(12));
        Assert.Equal("backbone.layers.0.mixer.B_bias", Mamba3TensorMapping.BBias(0));
        Assert.Equal("backbone.layers.0.mixer.C_bias", Mamba3TensorMapping.CBias(0));
        Assert.Equal("backbone.layers.5.mixer.D", Mamba3TensorMapping.D(5));
        Assert.Equal("backbone.layers.5.mixer.dt_bias", Mamba3TensorMapping.DtBias(5));
        Assert.Equal("backbone.layers.9.mixer.out_proj.weight", Mamba3TensorMapping.OutProj(9));
    }

    [Fact]
    public void ExpectedTensorNames_CountsMatch()
    {
        // 3 globals + 9 per-layer × 48 = 435 — one less than the 436 entries
        // the HF header reports because safetensors wraps a '__metadata__'
        // entry that is not a tensor.
        var names = Mamba3TensorMapping.ExpectedTensorNames(48);
        Assert.Equal(3 + 9 * 48, names.Count);
        Assert.Equal(Mamba3TensorMapping.PerLayerTensorCount, 9);
    }

    [Fact]
    public void ExpectedTensorNames_AreUnique()
    {
        var names = Mamba3TensorMapping.ExpectedTensorNames(48);
        Assert.Equal(names.Count, names.Distinct().Count());
    }

    [Fact]
    public void ExpectedTensorNames_CoverOneFullLayerPlusGlobals()
    {
        var names = Mamba3TensorMapping.ExpectedTensorNames(numLayers: 1);

        Assert.Equal(
            [
                Mamba3TensorMapping.TokenEmbedding,
                Mamba3TensorMapping.FinalNorm,
                Mamba3TensorMapping.LmHead,
                Mamba3TensorMapping.LayerNorm(0),
                Mamba3TensorMapping.InProj(0),
                Mamba3TensorMapping.OutProj(0),
                Mamba3TensorMapping.BNorm(0),
                Mamba3TensorMapping.CNorm(0),
                Mamba3TensorMapping.BBias(0),
                Mamba3TensorMapping.CBias(0),
                Mamba3TensorMapping.D(0),
                Mamba3TensorMapping.DtBias(0),
            ],
            names);
    }

    [Fact]
    public void ExpectedLayerNames_MatchHfHeaderProbeSuffixes()
    {
        // Cross-reference against the 2026-04-19 header probe: every
        // per-layer name we emit for layer 0 must, after stripping the
        // 'backbone.layers.0.' prefix, appear verbatim in the HF signature.
        // Conversely, every HF signature suffix must be emitted by exactly
        // one mapping accessor. This catches name drift in either direction.
        var names = Mamba3TensorMapping.ExpectedTensorNames(numLayers: 1);
        var layerNames = names
            .Where(n => n.StartsWith("backbone.layers.0.", StringComparison.Ordinal))
            .Select(n => n["backbone.layers.0.".Length..])
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(ExpectedPerLayerSuffixes, layerNames);
    }

    [Fact]
    public void ReferenceKeys_NoteDivergencesFromHf()
    {
        // Sanity check that the reference keys we carry for Stage D3 are
        // spelled differently from the HF ones (proof we actually
        // documented the drift rather than aliasing them silently).
        Assert.NotEqual(Mamba3TensorMapping.TokenEmbedding,
                        Mamba3TensorMapping.ReferenceKeys.TokenEmbedding);
        Assert.Equal("backbone.embedding.weight",
                     Mamba3TensorMapping.ReferenceKeys.TokenEmbedding);
        // A_log is only in the reference.
        Assert.Equal("backbone.layers.7.mixer.A_log",
                     Mamba3TensorMapping.ReferenceKeys.ALog(7));
    }

    [Fact]
    public void ExpectedTensorNames_NegativeLayerCount_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Mamba3TensorMapping.ExpectedTensorNames(-1));
    }
}
