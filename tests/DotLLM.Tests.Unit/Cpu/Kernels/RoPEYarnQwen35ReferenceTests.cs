using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #1101: YaRN for the Qwen3.5/3.8 hybrid hosts (Bonsai-2 built from Qwen3.8) checked against an independent
/// reference — HF transformers' own <c>_compute_yarn_parameters</c>, run in torch float32
/// (<c>.docs/ctx128k/yarn/hf_yarn_reference.py</c>, copied verbatim from transformers main on 2026-10-22).
/// </summary>
/// <remarks>
/// <para>
/// The ramp runs over the ROTARY dim: HF sets <c>dim = int(head_dim * partial_rotary_factor)</c>, 64 for Bonsai-2's
/// 256-wide heads. Ramping over the head width instead moves the correction range from (14, 22) to (57, 89→255) and
/// changes every inverse frequency past index 13, which <see cref="ResolveDenseYarn_WrongRampDim_WouldNotMatch"/> pins.
/// </para>
/// <para>
/// HF's default <c>attention_factor</c> is <c>0.1·ln(factor) + 1</c> (factor 4: 1.1386294), applied to cos AND sin —
/// the llama.cpp <c>ggml_rope_ext</c> convention gpt-oss already used. SmolLM3 and Llama keep the plain
/// <see cref="RoPEConfig.AttnFactor"/> convention they were wired with.
/// </para>
/// <para>
/// Tolerance: dotLLM evaluates <c>MathF.Pow</c> where torch evaluates <c>base ** (arange / dim)</c>; both are float32,
/// so the two may differ by an ulp or two. The bound below is 4 float32 ulps relative, far tighter than any ramp or
/// mscale error (the smallest such error is a whole factor of 4 on one frequency).
/// </para>
/// </remarks>
public sealed class RoPEYarnQwen35ReferenceTests
{
    private const float MaxRelUlps = 4 * 1.1920929e-7f;

    /// <summary>HF <c>attention_factor</c> for factor 4 (float32 of 1.138629436111989).</summary>
    internal const float HfAttentionFactorFactor4 = 1.1386294f;

    /// <summary>
    /// HF inverse frequencies for Bonsai-2 / Qwen3.8: head 256, partial 0.25 (rotary 64), base 1e7, factor 4,
    /// original 262,144, beta 32/1, truncate. Correction range (14, 22).
    /// </summary>
    internal static readonly float[] HfInvFreqBonsai2 =
    [
        1.0f, 0.6042963862419128f, 0.36517414450645447f, 0.22067339718341827f,
        0.1333521455526352f, 0.08058421313762665f, 0.04869675263762474f, 0.02942727319896221f,
        0.017782794311642647f, 0.010746078565716743f, 0.006493816617876291f, 0.003924189601093531f,
        0.0023713738191872835f, 0.0014330126577988267f, 0.0008659643353894353f, 0.00047423981595784426f,
        0.00025693507632240653f, 0.00013734975073020905f, 7.217386882985011e-05f, 3.707224823301658e-05f,
        1.844922371674329e-05f, 8.759770025790203e-06f, 3.849816039291909e-06f, 2.3264301489689387e-06f,
        1.405853367941745e-06f, 8.495520660289912e-07f, 5.133812237545499e-07f, 3.102344408034696e-07f,
        1.8747356023141037e-07f, 1.1328959459433463e-07f, 6.846049416253663e-08f, 4.1370427084075345e-08f,
    ];

    /// <summary>
    /// HF inverse frequencies for the discriminating synthetic fixture the cross-backend parity tests use: the same
    /// rotary shape, base 1e4, factor 4, original 512. Correction range (3, 16): the ramp spans 13 of the 32 pairs.
    /// </summary>
    internal static readonly float[] HfInvFreqFixtureBase1e4Orig512 =
    [
        1.0f, 0.7498942017555237f, 0.5623413324356079f, 0.4216965138912201f,
        0.297983855009079f, 0.20977537333965302f, 0.14705002307891846f, 0.10257857292890549f,
        0.07115384191274643f, 0.049031540751457214f, 0.03352418914437294f, 0.022706732153892517f,
        0.015203258022665977f, 0.010032733902335167f, 0.006497559137642384f, 0.0041031427681446075f,
        0.0024999999441206455f, 0.0018747355788946152f, 0.0014058533124625683f, 0.0010542412055656314f,
        0.0007905694656074047f, 0.0005928434547968209f, 0.00044456985779106617f, 0.00033338036155328155f,
        0.0002500000118743628f, 0.00018747354624792933f, 0.00014058532542549074f, 0.00010542412201175466f,
        7.905694656074047e-05f, 5.928434620727785e-05f, 4.4456985051510856e-05f, 3.333803761051968e-05f,
    ];

    /// <summary>Bonsai-2's RoPE with Qwen's opt-in YaRN (<c>--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144</c>).</summary>
    internal static RoPEConfig Bonsai2Yarn() => new(
        Theta: 1e7f, DimensionCount: 64, Type: RoPEType.NeoX,
        ScalingType: RoPEScalingType.YaRN, ScalingFactor: 4.0f, OrigMaxSeqLen: 262144);

    [Fact]
    public void ComputeYarnInverseFrequencies_Bonsai2_MatchesHfReference()
    {
        var inv = new float[32];
        RoPE.ComputeYarnInverseFrequencies(64, 1e7f, 4.0f, 262144, 32.0f, 1.0f, inv);
        AssertMatches(HfInvFreqBonsai2, inv);
    }

    [Fact]
    public void ComputeYarnInverseFrequencies_ParityFixture_MatchesHfReference()
    {
        var inv = new float[32];
        RoPE.ComputeYarnInverseFrequencies(64, 1e4f, 4.0f, 512, 32.0f, 1.0f, inv);
        AssertMatches(HfInvFreqFixtureBase1e4Orig512, inv);
    }

    /// <summary>The helper every hybrid host uses: rotary-dim ramp and the HF attention factor for Qwen3.5/3.8.</summary>
    [Fact]
    public void ResolveDenseYarn_Qwen3HybridDense_MatchesHfReference()
    {
        var (inv, mscale) = RoPE.ResolveDenseYarn(Bonsai2Yarn(), Architecture.Qwen3HybridDense, ropeDim: 64);
        Assert.NotNull(inv);
        AssertMatches(HfInvFreqBonsai2, inv!);
        Assert.Equal(HfAttentionFactorFactor4, mscale);
    }

    /// <summary>Mutant guard: a host that passed the 256-wide head dim to the ramp would not match HF.</summary>
    [Fact]
    public void ResolveDenseYarn_WrongRampDim_WouldNotMatch()
    {
        var wrong = new float[128];
        RoPE.ComputeYarnInverseFrequencies(256, 1e7f, 4.0f, 262144, 32.0f, 1.0f, wrong);

        // With the head width as the dim, both the exponent (2i/256) and the correction range move, so no pair past 0 agrees.
        int agreeing = 0;
        for (int i = 1; i < 32; i++)
            if (MathF.Abs(wrong[i] - HfInvFreqBonsai2[i]) <= MaxRelUlps * HfInvFreqBonsai2[i]) agreeing++;
        Assert.Equal(0, agreeing);
    }

    [Fact]
    public void ResolveDenseYarn_YarnOff_ReturnsNullAndUnitMscale()
    {
        var plain = new RoPEConfig(Theta: 1e7f, DimensionCount: 64, Type: RoPEType.NeoX);
        var (inv, mscale) = RoPE.ResolveDenseYarn(plain, Architecture.Qwen3HybridDense, 64);
        Assert.Null(inv);
        Assert.Equal(1.0f, mscale);
        Assert.Equal(0, plain.YarnExtendedContextLength);
    }

    /// <summary>Per-architecture mscale convention (issue #1101): the Qwen3.5/3.8 family joins gpt-oss; SmolLM3/Llama unchanged.</summary>
    [Theory]
    [InlineData(Architecture.Qwen3HybridDense, true)]
    [InlineData(Architecture.Qwen3MoeHybrid, true)]
    [InlineData(Architecture.Qwen4Exp, true)]
    [InlineData(Architecture.GptOss, true)]
    [InlineData(Architecture.SmolLM3, false)]
    [InlineData(Architecture.Llama, false)]
    public void ComputeYarnMscaleMultiplier_PerArchitecture(Architecture architecture, bool logFactor)
    {
        float mscale = Bonsai2Yarn().ComputeYarnMscaleMultiplier(architecture);
        Assert.Equal(logFactor ? HfAttentionFactorFactor4 : 1.0f, mscale);
    }

    [Fact]
    public void YarnExtendedContextLength_Bonsai2_IsFactorTimesOriginal()
        => Assert.Equal(1_048_576, Bonsai2Yarn().YarnExtendedContextLength);

    /// <summary>
    /// The CPU host's position-indexed tables (the cross-backend oracle) equal <c>cos/sin(pos · invFreq_HF) · mscale_HF</c>.
    /// </summary>
    [Fact]
    public void BuildFrequencyTables_Bonsai2_MatchesHfRotation()
    {
        var (cos, sin) = RoPE.BuildFrequencyTables(Bonsai2Yarn(), Architecture.Qwen3HybridDense, maxSeqLen: 64, ropeDim: 64);
        for (int pos = 0; pos < 64; pos += 7)
        {
            for (int i = 0; i < 32; i++)
            {
                float angle = pos * HfInvFreqBonsai2[i];
                Assert.Equal(MathF.Cos(angle) * HfAttentionFactorFactor4, cos[pos * 32 + i], 6);
                Assert.Equal(MathF.Sin(angle) * HfAttentionFactorFactor4, sin[pos * 32 + i], 6);
            }
        }
    }

    private static void AssertMatches(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float bar = MaxRelUlps * MathF.Abs(expected[i]);
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= bar,
                $"invFreq[{i}]: dotLLM {actual[i]:R} vs HF {expected[i]:R} (|diff| {MathF.Abs(expected[i] - actual[i]):E3} > {bar:E3})");
        }
    }
}
