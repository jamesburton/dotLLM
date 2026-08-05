using DotLLM.Core.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Tests for the quantization gate's comparison metric (#256).
/// </summary>
/// <remarks>
/// The cases below are real measurements, not invented numbers. The #254 pair is the defect this
/// gate exists to catch; the Q2_K pair is a legitimately-destroyed <c>--pure</c> fixture that must
/// NOT trip it. Together they fix both ends of the threshold.
/// </remarks>
public sealed class QuantGateComparisonTests
{
    // CPU Q4_0 21.0627 (NLL 3.047) vs the pre-fix CUDA Q4_0 2.0253e11 (NLL 26.034). Issue #254.
    private static readonly PerplexityResult Q4_0Cpu = new(21.0627, 3.047, 1000, 1);
    private static readonly PerplexityResult Q4_0CudaBroken = new(202530958818.744, 26.034159, 1000, 1);

    // Q2_K --pure on Llama-3.2-1B: CPU vs Vulkan, both correct, model destroyed by design.
    private static readonly PerplexityResult Q2KCpu = new(7137435.5839, 15.780, 1000, 1);
    private static readonly PerplexityResult Q2KVulkan = new(7001965.5753, 15.761701, 1000, 1);

    [Theory]
    [InlineData(null, QuantGateMetric.Both)]
    [InlineData("", QuantGateMetric.Both)]
    [InlineData("both", QuantGateMetric.Both)]
    [InlineData("BOTH", QuantGateMetric.Both)]
    [InlineData("nats", QuantGateMetric.Nats)]
    [InlineData("perplexity", QuantGateMetric.Perplexity)]
    public void ResolveMode_MapsKnownValues(string? raw, QuantGateMetric expected)
        => Assert.Equal(expected, QuantGateComparison.ResolveMode(raw));

    /// <summary>
    /// An unknown mode must throw, never silently fall back. A harness that quietly substitutes a
    /// default is how twelve broken cells once read as a completed sweep.
    /// </summary>
    [Fact]
    public void ResolveMode_UnknownValue_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => QuantGateComparison.ResolveMode("nat"));
        Assert.Contains("nat", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(QuantGateMetric.Nats)]
    [InlineData(QuantGateMetric.Perplexity)]
    [InlineData(QuantGateMetric.Both)]
    public void Compare_IdenticalResults_Passes(QuantGateMetric metric)
    {
        var v = QuantGateComparison.Compare(Q4_0Cpu, Q4_0Cpu, metric, QuantGateThresholds.Default);
        Assert.True(v.Passed);
        Assert.Equal(0, v.NatsDelta, 12);
        Assert.Equal(0, v.PerplexityRelative, 12);
    }

    /// <summary>The #254 defect must fail under every metric, not just the favoured one.</summary>
    [Theory]
    [InlineData(QuantGateMetric.Nats)]
    [InlineData(QuantGateMetric.Perplexity)]
    [InlineData(QuantGateMetric.Both)]
    public void Compare_Issue254Garbage_Fails(QuantGateMetric metric)
    {
        var v = QuantGateComparison.Compare(Q4_0Cpu, Q4_0CudaBroken, metric, QuantGateThresholds.Default);
        Assert.False(v.Passed);
        Assert.True(v.NatsDelta > 22.0, $"expected a huge nats gap, got {v.NatsDelta}");
    }

    /// <summary>
    /// A destroyed-but-correct fixture must pass. This is the case that motivates nats being the
    /// default arm: 0.018 nats is 2.7x inside its bound, while the same pair is 1.93% in
    /// perplexity — inside a 2% bound, but only just, because perplexity is exp(NLL) and these
    /// values are ~7e6.
    /// </summary>
    [Fact]
    public void Compare_DestroyedButAgreeingFixture_PassesComfortablyInNats()
    {
        var nats = QuantGateComparison.Compare(Q2KCpu, Q2KVulkan, QuantGateMetric.Nats, QuantGateThresholds.Default);
        Assert.True(nats.Passed);
        Assert.True(nats.NatsDelta < 0.02, $"expected < 0.02 nats, got {nats.NatsDelta}");

        var ppl = QuantGateComparison.Compare(Q2KCpu, Q2KVulkan, QuantGateMetric.Perplexity, QuantGateThresholds.Default);
        Assert.True(ppl.Passed);
        Assert.InRange(ppl.PerplexityRelative, 0.015, 0.02);
    }

    /// <summary>Both mode fails if either arm fails, even when the other passes.</summary>
    [Fact]
    public void Compare_Both_FailsIfEitherArmFails()
    {
        // Same NLL to within the nats bound, but perplexity deliberately far apart.
        var a = new PerplexityResult(100.0, 4.605, 1000, 1);
        var b = new PerplexityResult(104.0, 4.644, 1000, 1);

        Assert.True(QuantGateComparison.Compare(a, b, QuantGateMetric.Nats, QuantGateThresholds.Default).Passed);
        Assert.False(QuantGateComparison.Compare(a, b, QuantGateMetric.Perplexity, QuantGateThresholds.Default).Passed);
        Assert.False(QuantGateComparison.Compare(a, b, QuantGateMetric.Both, QuantGateThresholds.Default).Passed);
    }

    /// <summary>
    /// A non-finite score is a failure, never a pass. NaN comparisons are false by default, so an
    /// unguarded threshold check would let a kernel emitting NaN slip through as "within bounds".
    /// </summary>
    [Theory]
    [InlineData(double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    public void Compare_NonFiniteScore_Fails(double ppl, double nll)
    {
        var broken = new PerplexityResult(ppl, nll, 1000, 1);
        var v = QuantGateComparison.Compare(Q4_0Cpu, broken, QuantGateMetric.Both, QuantGateThresholds.Default);
        Assert.False(v.Passed);
        Assert.Contains("finite", v.Detail, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The verdict must name both measured quantities so a failure is actionable.</summary>
    [Fact]
    public void Compare_Detail_ReportsBothQuantities()
    {
        var v = QuantGateComparison.Compare(Q2KCpu, Q2KVulkan, QuantGateMetric.Both, QuantGateThresholds.Default);
        Assert.Contains("nats", v.Detail, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("%", v.Detail, StringComparison.Ordinal);
    }
}
