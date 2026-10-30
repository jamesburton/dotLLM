using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Policy of <see cref="MtpAdaptiveGate"/>: on Tev1-4B (Vulkan) MTP ran at 18-20 tok/s against 35-37 plain, so a fixed
/// "MTP on" default halves throughput there while it wins on large models. The gate must never engage MTP for short
/// requests, must explore each arm once, then follow whichever measured faster, and must re-probe the loser.
/// </summary>
public sealed class MtpAdaptiveGateTests
{
    private const long Small = 3L << 30;   // 3 GiB  -> plain first
    private const long Large = 7L << 30;   // 7 GiB  -> MTP first

    [Fact]
    public void ShortRequests_NeverUseMtp()
    {
        var g = new MtpAdaptiveGate(Large);
        for (int i = 0; i < 5; i++)
            Assert.False(g.ShouldUseMtp(MtpAdaptiveGate.MinMaxTokens - 1));
        Assert.True(g.ShouldUseMtp(MtpAdaptiveGate.MinMaxTokens));
    }

    [Fact]
    public void FirstArm_FollowsTheModelSizePrior_ThenExploresTheOther()
    {
        var big = new MtpAdaptiveGate(Large);
        Assert.True(big.ShouldUseMtp(64));
        big.Record(usedMtp: true, generatedTokens: 65, decodeMs: 650);      // 10 ms/token
        Assert.False(big.ShouldUseMtp(64));                                  // now explore plain once

        var small = new MtpAdaptiveGate(Small);
        Assert.False(small.ShouldUseMtp(64));
        small.Record(usedMtp: false, generatedTokens: 65, decodeMs: 1600);  // 25 ms/token
        Assert.True(small.ShouldUseMtp(64));                                 // explore MTP once
    }

    [Fact]
    public void AfterBothArmsSampled_ChoosesTheFasterOne()
    {
        // Tev1-4B-like: plain 27 ms/token, MTP 55 ms/token  -> plain wins.
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 27 * 64);
        g.Record(true, 65, 55 * 64);
        Assert.False(g.ShouldUseMtp(64));

        // Bonsai-27B-like: plain 45 ms/token, MTP 22 ms/token -> MTP wins.
        var h = new MtpAdaptiveGate(Large);
        h.Record(false, 65, 45 * 64);
        h.Record(true, 65, 22 * 64);
        Assert.True(h.ShouldUseMtp(64));
    }

    [Fact]
    public void TheLosingArm_IsReprobedPeriodically()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 27 * 64);
        g.Record(true, 65, 55 * 64);

        // 55 vs 27 ms/token is a clear loss, so the interval doubles after every probe (40, 80, 160 ...; see MtpCostModelTests for the near-tie case).
        int mtpPicks = 0;
        for (int i = 0; i < MtpAdaptiveGate.ReprobeEvery * 7; i++)
            if (g.ShouldUseMtp(64)) mtpPicks++;
        Assert.Equal(3, mtpPicks);
    }

    [Fact]
    public void TinyOrEmptySamples_AreIgnored()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, generatedTokens: 1, decodeMs: 25);        // 1-token classifier answer
        g.Record(false, generatedTokens: MtpAdaptiveGate.MinSampleTokens, decodeMs: 100);   // 5 decode tokens
        g.Record(true, generatedTokens: 64, decodeMs: 0);          // no timing
        Assert.Equal(0, g.PlainMsPerToken);
        Assert.Equal(0, g.MtpMsPerToken);
    }

    [Fact]
    public void Measurements_AreSmoothed_NotLastWriterWins()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 20 * 64);
        g.Record(false, 65, 40 * 64);
        Assert.InRange(g.PlainMsPerToken, 20.5, 39.5);
    }

    [Fact]
    public void LikelyUsesMtpNext_IsTrueWhileAnArmIsUnexplored()
    {
        // A plain first turn (the size prior) must keep the head in step: the next request explores MTP.
        var g = new MtpAdaptiveGate(Small);
        Assert.True(g.LikelyUsesMtpNext());
        Assert.False(g.ShouldUseMtp(64));
        g.Record(false, 65, 27 * 64);
        Assert.True(g.LikelyUsesMtpNext());
        Assert.True(g.ShouldUseMtp(64));   // and the forecast was right

        var mtpOnly = new MtpAdaptiveGate(Large);
        mtpOnly.Record(true, 65, 55 * 64);
        Assert.True(mtpOnly.LikelyUsesMtpNext());   // plain unmeasured: MTP may still win
    }

    [Fact]
    public void LikelyUsesMtpNext_FollowsTheWinner_AndForeseesTheReprobe()
    {
        var mtpWins = new MtpAdaptiveGate(Large);
        mtpWins.Record(false, 65, 45 * 64);
        mtpWins.Record(true, 65, 22 * 64);
        Assert.True(mtpWins.LikelyUsesMtpNext());

        // Plain wins: no MTP forecast until the decision that re-probes MTP is the next one.
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 27 * 64);
        g.Record(true, 65, 55 * 64);
        for (int i = 1; i < MtpAdaptiveGate.ReprobeEvery; i++)
        {
            Assert.False(g.LikelyUsesMtpNext(), $"decision {i}");
            Assert.False(g.ShouldUseMtp(64));
        }
        Assert.True(g.LikelyUsesMtpNext());
        Assert.True(g.ShouldUseMtp(64));   // the re-probe
    }

    [Fact]
    public void LikelyUsesMtpNext_DoesNotCountAsADecision()
    {
        // Asking must not move the re-probe schedule: the engine asks once per plain snapshot turn.
        var asked = new MtpAdaptiveGate(Small);
        var silent = new MtpAdaptiveGate(Small);
        foreach (var g in new[] { asked, silent })
        {
            g.Record(false, 65, 27 * 64);
            g.Record(true, 65, 55 * 64);
        }
        for (int i = 0; i < 3 * MtpAdaptiveGate.ReprobeEvery; i++)
        {
            for (int k = 0; k < 5; k++) asked.LikelyUsesMtpNext();
            Assert.Equal(silent.ShouldUseMtp(64), asked.ShouldUseMtp(64));
        }
        Assert.Equal(silent.Describe(), asked.Describe());
    }

    /// <summary>
    /// #1223: the CUDA hybrid-dense model's <c>ComputeMemoryBytes</c> leaves out its device weights, so a 7+ GiB Bonsai-2 fell on the plain-first
    /// side of the size prior and a one-shot run never drafted. The prior must put it MTP-first, like the Vulkan MTP models.
    /// </summary>
    [Fact]
    public void CudaHybridDensePrior_TriesMtpFirst()
    {
        // The property is a constant, so an uninitialised instance (no GPU needed) is enough to read it.
        var model = (DotLLM.Core.Models.IModel)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(
            typeof(DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel));
        Assert.True(model.MtpGatePriorBytes >= MtpAdaptiveGate.MtpPriorMinModelBytes);
        Assert.True(new MtpAdaptiveGate(model.MtpGatePriorBytes).ShouldUseMtp(128));
    }

    /// <summary>
    /// One fresh 128-token request driven through the gate and the round controller with injected timings (#1223). MTP-first is only safe
    /// because the request measures itself: where drafting wins (GPU verify ~1.3x a decode step) it must keep drafting, and where it loses
    /// (CPU, an 8-row verify ~7x a step) it must drop to plain inside that same request.
    /// </summary>
    [Theory]
    [InlineData(1.3, true)]    // bandwidth-bound GPU trunk: a 4-row verify costs ~1.3 decode steps
    [InlineData(7.0, false)]   // compute-bound CPU trunk: the verify costs ~7 decode steps
    public void FreshRequest_KeepsDraftingOnlyWhereItMeasuresFaster(double roundCostInSteps, bool expectDrafting)
    {
        const double StepMs = 50;   // plain decode step
        const int K = 3;
        var gate = new MtpAdaptiveGate(long.MaxValue);
        Assert.True(gate.ShouldUseMtp(128));

        var ctl = new MtpRoundController(gate.CostModel, K, calibratePlain: gate.NeedsPlainReference(127), gate: gate);
        int generated = 1, drafted = 0;
        bool fellBack = false;
        while (generated < 128)
        {
            int k = ctl.NextK(128 - generated);
            if (k == 0)
            {
                ctl.Record(0, 1, StepMs);   // zero-draft calibration round
                generated++;
                continue;
            }
            drafted += k;
            int emitted = k + 1;   // every draft accepted: the most favourable case for MTP
            ctl.Record(k, emitted, StepMs * roundCostInSteps * (k + 1) / (K + 1));
            generated += emitted;
            if (ctl.ShouldFallBack(gate.PlainMsPerToken)) { fellBack = true; break; }
        }

        Assert.True(drafted > 0, "an MTP-first request must try drafting");
        Assert.Equal(expectDrafting, !fellBack);
        Assert.Equal(StepMs, gate.PlainMsPerToken, 3);   // the calibration became the gate's plain reference
    }
}
