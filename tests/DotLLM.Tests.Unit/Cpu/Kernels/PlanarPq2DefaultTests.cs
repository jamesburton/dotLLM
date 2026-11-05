using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1620: the planar PQ2_0 kernels are ON by default (AVX2) with <c>DOTLLM_PQ2_PLANAR=0</c> as the off switch, the blocked kernel
/// covers n &lt;= 16 by default (covers the default 9-row prompt-lookup verify; above 16 it lost to production on Bonsai-2 2B), and
/// <c>DOTLLM_PQ2_W2A8=0</c> (the unquantized-activation reference tier) still turns planar off. Pure parsing, no process state.
/// </summary>
public sealed class PlanarPq2DefaultTests
{
    [Theory]
    [InlineData(null, null, true)]      // the default flip
    [InlineData("1", null, true)]
    [InlineData("on", null, true)]
    [InlineData("0", null, false)]      // off switch
    [InlineData("false", null, false)]
    [InlineData("off", null, false)]
    [InlineData(null, "0", false)]      // float-activation reference tier must not be shadowed by planar
    [InlineData(null, "off", false)]
    [InlineData("1", "0", false)]
    [InlineData(null, "1", true)]
    public void PlanarDefault_IsOnUnlessSwitchedOff(string? planar, string? w2a8, bool expected) =>
        Assert.Equal(expected, MatMul.ResolvePlanarDefault(planar, w2a8));

    [Theory]
    [InlineData(null, 16)]              // default cut (measured, docs/TERNARY_ADAPTER.md)
    [InlineData("", 16)]
    [InlineData("junk", 16)]
    [InlineData("4", 4)]
    [InlineData("100000", 100000)]
    [InlineData("1", 1)]                // decode GEMV only
    public void PlanarMaxN_DefaultsTo16(string? env, int expected) =>
        Assert.Equal(expected, MatMul.ResolvePlanarMaxN(env));
}

/// <summary>The process-wide initial state is wired to the resolvers (not only the resolvers being right). Runs in the toggle collection so no
/// test that flips the toggle overlaps it; those tests restore the state in <c>finally</c>.</summary>
[Collection("PQ2PlanarToggle")]
public sealed class PlanarPq2DefaultWiringTests
{
    [Fact]
    public void ProcessDefault_FollowsResolvers()
    {
        bool expected = global::System.Runtime.Intrinsics.X86.Avx2.IsSupported && MatMul.ResolvePlanarDefault(
            Environment.GetEnvironmentVariable("DOTLLM_PQ2_PLANAR"), Environment.GetEnvironmentVariable("DOTLLM_PQ2_W2A8"));
        Assert.Equal(expected, MatMul.PQ2_0PlanarEnabled);
        Assert.Equal(MatMul.ResolvePlanarMaxN(Environment.GetEnvironmentVariable("DOTLLM_PQ2_PLANAR_NMAX")), MatMul.PQ2_0PlanarMaxN);
    }
}
