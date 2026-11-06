using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>#1623: the <see cref="CpuFeatures"/> flag algebra and the <c>DOTLLM_CPU_ISA</c> cap.</summary>
public sealed class CpuFeaturesTests
{
    [Theory]
    [InlineData(null, CpuIsaTier.Avx512, false)]
    [InlineData("", CpuIsaTier.Avx512, false)]
    [InlineData("auto", CpuIsaTier.Avx512, false)]
    [InlineData("BEST", CpuIsaTier.Avx512, false)]
    [InlineData("avx512", CpuIsaTier.Avx512, false)]
    [InlineData("avx2", CpuIsaTier.Avx2, false)]
    [InlineData(" AVX2 ", CpuIsaTier.Avx2, false)]
    [InlineData("v128", CpuIsaTier.Vector128, false)]
    [InlineData("sse", CpuIsaTier.Vector128, false)]
    [InlineData("scalar", CpuIsaTier.Scalar, false)]
    [InlineData("none", CpuIsaTier.Scalar, false)]
    [InlineData("avx3", CpuIsaTier.Avx512, true)]     // typo: no cap, reported, never silently a slower tier
    [InlineData("scaler", CpuIsaTier.Avx512, true)]
    public void ParseCap_Table(string? value, CpuIsaTier expected, bool invalid)
    {
        Assert.Equal(expected, CpuFeatures.ParseCap(value, out string? bad));
        Assert.Equal(invalid, bad is not null);
    }

    /// <summary>The flags form a lattice: a flag never holds without the flags it is built on, and never exceeds the hardware.</summary>
    [Fact]
    public void Flags_AreAConsistentLattice_AndNeverExceedHardware()
    {
        Assert.True(!CpuFeatures.Avx512Bw || CpuFeatures.Avx512F);
        Assert.True(!CpuFeatures.Avx512F || CpuFeatures.Avx2);
        Assert.True(!CpuFeatures.AvxVnni || CpuFeatures.Avx2);
        Assert.True(!CpuFeatures.Avx2Fma || (CpuFeatures.Avx2 && CpuFeatures.Fma));
        Assert.True(!CpuFeatures.Avx2FmaVnni || (CpuFeatures.Avx2Fma && CpuFeatures.AvxVnni));
        Assert.True(!CpuFeatures.Sse41 || CpuFeatures.Ssse3);
        Assert.True(!CpuFeatures.Ssse3 || (CpuFeatures.Sse2 && CpuFeatures.Sse3));

        Assert.True(!CpuFeatures.Avx2 || (Avx.IsSupported && Avx2.IsSupported));
        Assert.True(!CpuFeatures.Fma || Fma.IsSupported);
        Assert.True(!CpuFeatures.AvxVnni || AvxVnni.IsSupported);
        Assert.True(!CpuFeatures.Avx512F || Avx512F.IsSupported);
        Assert.True(!CpuFeatures.Avx512Bw || Avx512BW.IsSupported);
        Assert.True(!CpuFeatures.Ssse3 || Ssse3.IsSupported);
        Assert.True(!CpuFeatures.Sse41 || Sse41.IsSupported);
    }

    /// <summary>Each flag is on exactly when the tier cap reaches it and the hardware has it (no cap -> equals IsSupported).</summary>
    [Fact]
    public void Flags_FollowTheCap()
    {
        CpuIsaTier cap = CpuFeatures.Cap;
        Assert.Equal(cap >= CpuIsaTier.Vector128 && Ssse3.IsSupported && Sse2.IsSupported && Sse3.IsSupported, CpuFeatures.Ssse3);
        Assert.Equal(cap >= CpuIsaTier.Avx2 && Avx.IsSupported && Avx2.IsSupported, CpuFeatures.Avx2);
        Assert.Equal(cap >= CpuIsaTier.Avx2 && Avx.IsSupported && Fma.IsSupported, CpuFeatures.Fma);
        Assert.Equal(CpuFeatures.Avx2 && AvxVnni.IsSupported, CpuFeatures.AvxVnni);
        Assert.Equal(cap >= CpuIsaTier.Avx512 && CpuFeatures.Avx2 && Avx512F.IsSupported, CpuFeatures.Avx512F);
        Assert.Equal(CpuFeatures.Avx512F && Avx512BW.IsSupported, CpuFeatures.Avx512Bw);
        Assert.Equal(CpuFeatures.HardwareTier < cap ? CpuFeatures.HardwareTier : cap, CpuFeatures.Tier);
    }

    /// <summary>
    /// <c>PortableMath.Exp</c> predicts what the BCL does, so the <c>DOTLLM_CPU_ISA</c> cap must not change it: on a runtime that
    /// accelerates <c>Vector128</c> a long span is exactly <c>TensorPrimitives.Exp</c> under every cap. Gating it on the capped
    /// <see cref="CpuFeatures.Vector128Accelerated"/> swapped in the double-evaluated exp under <c>scalar</c>, which differs by an ulp on
    /// some of these inputs and moved the quantized Flint golden by 0.027 (all wide ISAs off + scalar cap).
    /// </summary>
    [SkippableFact]
    public void PortableMathExp_LongSpans_AreTensorPrimitivesExp_UnderEveryCap()
    {
        Skip.IfNot(System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated, "runtime does not accelerate Vector128: both sides take expf");
        var x = new float[4099];
        for (int i = 0; i < x.Length; i++) x[i] = -20f + 40f * i / x.Length + 1e-3f * (i % 7);
        var want = new float[x.Length];
        var got = new float[x.Length];
        System.Numerics.Tensors.TensorPrimitives.Exp(x, want);
        DotLLM.Cpu.Kernels.PortableMath.Exp(x, got);
        for (int i = 0; i < x.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                $"exp({x[i]:R}): TensorPrimitives {want[i]:R} vs PortableMath {got[i]:R} [{CpuFeatures.Describe()}]");
    }

    /// <summary>
    /// When the process was started with <c>DOTLLM_CPU_ISA</c> (the CI tier legs), the cap is in force: every flag above the cap is off.
    /// Without the variable the cap is "none" and this asserts only that the parse agrees, so it passes on every configuration.
    /// </summary>
    [Fact]
    public void EnvOverride_WhenSet_CapsEveryHigherFlag()
    {
        CpuIsaTier cap = CpuFeatures.ParseCap(Environment.GetEnvironmentVariable(CpuFeatures.EnvVar), out _);
        Assert.Equal(cap, CpuFeatures.Cap);
        Assert.True(CpuFeatures.Tier <= cap);
        if (cap < CpuIsaTier.Avx512) Assert.False(CpuFeatures.Avx512F || CpuFeatures.Avx512Bw || CpuFeatures.Vector512Accelerated);
        if (cap < CpuIsaTier.Avx2) Assert.False(CpuFeatures.Avx2 || CpuFeatures.Fma || CpuFeatures.AvxVnni || CpuFeatures.Vector256Accelerated);
        if (cap < CpuIsaTier.Vector128) Assert.False(CpuFeatures.Sse2 || CpuFeatures.Ssse3 || CpuFeatures.Sse41 || CpuFeatures.Vector128Accelerated);
    }
}
