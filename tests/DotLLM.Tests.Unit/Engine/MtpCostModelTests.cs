using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// <see cref="MtpCostModel"/> / <see cref="MtpRoundController"/> (#900): the per-K throughput model that keeps MTP from losing to plain decode.
/// Numbers mimic the measured qwen4exp rounds on Vulkan (plain 40-45 ms/token; a round costs a fixed part plus ~20 ms per draft row).
/// </summary>
public sealed class MtpCostModelTests
{
    /// <summary>Runs <paramref name="rounds"/> simulated rounds and returns the Ks the controller chose.</summary>
    private static List<int> Simulate(MtpRoundController ctl, int rounds, double fixedMs, double perRowMs, double acceptance, int seed = 1)
    {
        var rng = new Random(seed);
        var ks = new List<int>();
        for (int r = 0; r < rounds; r++)
        {
            int k = ctl.NextK(10_000);
            ks.Add(k);
            int accepted = 0;
            while (accepted < k && rng.NextDouble() < acceptance) accepted++;
            ctl.Record(k, accepted + 1, fixedMs + perRowMs * k);
        }
        return ks;
    }

    [Fact]
    public void Calibration_RunsZeroDraftRoundsFirst_DropsTheColdOne_AndDoesNotCountAsMtpRounds()
    {
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4, calibratePlain: true);
        for (int i = 0; i < MtpRoundController.CalibrationRounds; i++)
        {
            Assert.True(ctl.Calibrating);
            Assert.Equal(0, ctl.NextK(1000));
            ctl.Record(0, 1, i == 0 ? 90.0 : 40.0);   // the first (cold) round is slower and must not enter the average
        }
        Assert.False(ctl.Calibrating);
        Assert.Equal(40.0, ctl.CalibratedPlainMsPerToken, 6);
        Assert.Equal(0, model.TotalRounds);           // calibration never pollutes the MTP cost model
        Assert.InRange(ctl.NextK(1000), 1, 4);
    }

    [Fact]
    public void Calibration_GivesAFreshRequestItsFallBackReference()
    {
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 3, calibratePlain: true);
        for (int i = 0; i < MtpRoundController.CalibrationRounds; i++) { ctl.NextK(1000); ctl.Record(0, 1, 10.0); }   // plain: 10 ms / token
        // MTP rounds at 60 ms for 1 token (all drafts rejected): 6x slower than plain.
        for (int i = 0; i < MtpRoundController.MinRoundsForFallback; i++)
        {
            Assert.False(ctl.ShouldFallBack(plainMsPerToken: 0), "must not judge before enough MTP rounds");
            int k = ctl.NextK(1000);
            ctl.Record(k, 1, 60.0);
        }
        Assert.True(ctl.ShouldFallBack(plainMsPerToken: 0), "the request's own calibration is the reference when the gate has none");
        Assert.False(new MtpRoundController(model, 3).ShouldFallBack(0), "without calibration or a gate sample there is no reference");
    }

    [Fact]
    public void Controller_WarmsUpAtTheConfiguredK_ThenFollowsTheModel()
    {
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4);
        var ks = Simulate(ctl, 40, fixedMs: 30, perRowMs: 20, acceptance: 0.75);
        Assert.All(ks.Take(MtpRoundController.WarmupRounds), k => Assert.Equal(4, k));
        Assert.All(ks, k => Assert.InRange(k, 1, 4));
    }

    [Fact]
    public void WinningRegime_StaysOnMtp_AtTheMeasuredBestK()
    {
        // UD-IQ4_XS-like: plain 43.5 ms; round = 55 + 16*K ms, acceptance 0.75: K=1 40.6, K=2 37.7, K=3 37.6, K=4 39.0 ms/token -> MTP wins at K 2-3.
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4);
        var ks = Simulate(ctl, 150, 55, 16, 0.75);
        Assert.False(ctl.ShouldFallBack(43.5));
        var (k, perTok) = model.BestK(4);
        Assert.True(perTok < 43.5, $"best {perTok:F1} ms/token vs plain 43.5");
        Assert.InRange(k, 2, 4);
        Assert.True(ks.Skip(30).Count(x => x >= 2) > 90, "the controller should sit at the 2-4 plateau");
    }

    [Fact]
    public void LosingAtK4_ButWinningAtShortK_ShrinksTheDraftInsteadOfFallingBack()
    {
        // IQ2_XS-like: plain 40 ms; K=4 round = 40 + 4*22 = 128 ms for 2.77 tokens = 46 ms/token (loses), K=2 = 84 ms for 2.2 tokens = 38 ms/token (wins).
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4);
        var ks = Simulate(ctl, 150, 40, 22, 0.7);
        Assert.False(ctl.ShouldFallBack(41.0));
        var (k, perTok) = model.BestK(4);
        Assert.InRange(k, 1, 3);
        Assert.True(perTok < 41.0, $"best {perTok:F1} ms/token");
        Assert.True(ks.Skip(40).Count(x => x <= 3) > 70, "the controller must have moved off K=4");
    }

    [Fact]
    public void HopelessRegime_FallsBackToPlain_ButNotBeforeEnoughRounds()
    {
        // fixed 60 ms + 25 ms/row, acceptance 0.5: no K beats a 40 ms plain step.
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4);
        for (int r = 0; r < MtpRoundController.MinRoundsForFallback - 1; r++)
        {
            int k = ctl.NextK(1000);
            ctl.Record(k, 1, 60 + 25 * k);
            Assert.False(ctl.ShouldFallBack(40), "too early to trust the model");
        }
        for (int r = 0; r < 10; r++)
        {
            int k = ctl.NextK(1000);
            ctl.Record(k, 1, 60 + 25 * k);
        }
        Assert.True(ctl.ShouldFallBack(40));
    }

    [Fact]
    public void FallBack_NeedsAPlainReference()
    {
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4);
        Simulate(ctl, 30, 200, 100, 0.2);
        Assert.False(ctl.ShouldFallBack(0));
    }

    [Fact]
    public void NextK_NeverExceedsRemainingTokensOrTheConfiguredK()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4);
        Assert.Equal(2, ctl.NextK(2));
        Assert.Equal(1, ctl.NextK(1));
        Assert.True(ctl.NextK(50) <= 4);
        var big = new MtpRoundController(new MtpCostModel(), 99);
        Assert.True(big.NextK(1000) <= MtpCostModel.MaxK);
    }

    [Fact]
    public void ExpectedTokens_FollowsTheAcceptanceChain()
    {
        var m = new MtpCostModel();
        // Position 1 always accepted, position 2 never: expected tokens for K=4 is 1 + 1 + 0 + ... = 2.
        for (int i = 0; i < 40; i++) m.RecordRound(4, 1, 100);
        Assert.InRange(m.ExpectedTokens(4), 1.99, 2.01);
        Assert.InRange(m.ExpectedTokens(1), 1.99, 2.01);
        Assert.InRange(m.Acceptance(1), 0.99, 1.0);   // accepted = 1: position 1 held ...
        Assert.InRange(m.Acceptance(2), 0.0, 0.01);   // ... and position 2, reached only because 1 held, failed
    }

    [Fact]
    public void FirstRoundIsColdAndItsTimeIsDropped()
    {
        var m = new MtpCostModel();
        m.RecordRound(4, 3, 400);    // cold round
        Assert.Equal(0, m.PredictRoundMs(4));
        m.RecordRound(4, 3, 100);
        Assert.Equal(100, m.PredictRoundMs(4), 1);
    }

    [Fact]
    public void UnmeasuredK_IsInterpolatedFromTheNeighbours()
    {
        var m = new MtpCostModel();
        m.RecordRound(2, 1, 1);               // cold, dropped
        for (int i = 0; i < 20; i++) { m.RecordRound(2, 1, 80); m.RecordRound(4, 2, 120); }
        Assert.InRange(m.PredictRoundMs(3), 99, 101);
        Assert.InRange(m.PredictRoundMs(1), 59, 61);   // extrapolated below with the same slope
        Assert.InRange(m.PredictRoundMs(5), 139, 141);
    }

    [Fact]
    public void Gate_BacksOffTheReprobeOfAClearlyLosingArm_ButNotOfANearTie()
    {
        var clear = new MtpAdaptiveGate(3L << 30);
        clear.Record(false, 65, 27 * 64);
        clear.Record(true, 65, 55 * 64);
        int picks = 0;
        int e = MtpAdaptiveGate.ReprobeEvery;
        for (int i = 0; i < e + 2 * e + 4 * e; i++) if (clear.ShouldUseMtp(64)) picks++;
        Assert.Equal(3, picks);   // after e, then 2e more, then 4e more decisions

        var tie = new MtpAdaptiveGate(3L << 30);
        tie.Record(false, 65, 27 * 64);
        tie.Record(true, 65, 28 * 64);
        picks = 0;
        for (int i = 0; i < MtpAdaptiveGate.ReprobeEvery * 3; i++) if (tie.ShouldUseMtp(64)) picks++;
        Assert.Equal(3, picks);
    }

    [Fact]
    public void Gate_Reset_ForgetsTheColdWarmupVerdict()
    {
        // A cold warm-up pair (MTP slow, plain fast) settles the gate on plain; the server resets it so the first real request tries MTP again.
        var g = new MtpAdaptiveGate(long.MaxValue);
        Assert.True(g.ShouldUseMtp(64));
        g.Record(true, 17, 15 * 120.0);
        g.Record(false, 17, 15 * 45.0);
        Assert.False(g.ShouldUseMtp(64));
        g.CostModel.RecordRound(4, 3, 150);
        g.Reset();
        Assert.Equal(0, g.PlainMsPerToken);
        Assert.Equal(0, g.MtpMsPerToken);
        Assert.Equal(0, g.CostModel.TotalRounds);
        Assert.True(g.ShouldUseMtp(64));
    }

    [Fact]
    public void Gate_RecordsSegmentsSeparately()
    {
        var g = new MtpAdaptiveGate(7L << 30);
        g.RecordSegment(true, 20, 1000);
        g.RecordSegment(false, 30, 1200);
        Assert.Equal(50, g.MtpMsPerToken, 1);
        Assert.Equal(40, g.PlainMsPerToken, 1);
        g.RecordSegment(false, MtpAdaptiveGate.MinSampleTokens - 1, 100);   // too short: ignored
        Assert.Equal(40, g.PlainMsPerToken, 1);
    }
}
