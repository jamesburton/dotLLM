using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The MTP upward depth probe (#1335): after a streak of fully accepted rounds at the top K the controller may try one K more (up to the model's
/// declared limit), keeps it only while its measured ms per generated token is lower, and never moves on weak acceptance or when the model declares no limit.
/// </summary>
public sealed class MtpUpwardProbeTests
{
    private static List<int> Run(MtpRoundController ctl, int rounds, double fixedMs, double perRowMs, double acceptance, int seed = 7)
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
    public void FullAcceptanceAndCheapRows_ClimbToTheLimit_AndStayThere()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        var ks = Run(ctl, 120, fixedMs: 60, perRowMs: 8, acceptance: 1.0);
        Assert.Equal(7, ks.Max());
        Assert.All(ks.Skip(100).Where((_, i) => i % 16 != 0), k => Assert.True(k >= 6));   // exploration rounds aside, the tail sits at the top
        Assert.Equal(7, ctl.Ceiling);
    }

    [Fact]
    public void ClimbsOneStepPerStreak_NotAtOnce()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        var ks = Run(ctl, MtpRoundController.UpwardProbeStreak - 1, 60, 8, 1.0);
        Assert.Equal(4, ks.Max());   // fewer full rounds than the streak: still the configured K
        Assert.Equal(4, ctl.Ceiling);
        Run(ctl, 1, 60, 8, 1.0);     // the streak completes: exactly one step up
        Assert.Equal(5, ctl.Ceiling);
    }

    [Fact]
    public void WeakAcceptance_NeverProbesUpward()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        var ks = Run(ctl, 300, fixedMs: 60, perRowMs: 8, acceptance: 0.5);
        Assert.True(ks.Max() <= 4, "streaks of 6 full rounds at K=4 are 0.5^24 events");
        Assert.Equal(4, ctl.Ceiling);
    }

    [Fact]
    public void NoDeclaredLimit_NeverExceedsTheConfiguredK()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4);   // maxProbeK 0 = off
        Assert.Equal(4, Run(ctl, 120, 60, 8, 1.0).Max());
        Assert.Equal(4, new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 4).Ceiling);
    }

    [Fact]
    public void ExpensiveRows_ProbeIsMeasuredThenAbandoned()
    {
        // Every draft is accepted, but a row costs 60 ms against a 10 ms fixed part: ms/token rises with K (K=4 50, K=5 51.7, K=6 52.9), so a probe must not stay.
        var model = new MtpCostModel();
        var ctl = new MtpRoundController(model, 4, maxProbeK: 7);
        var ks = Run(ctl, 200, fixedMs: 10, perRowMs: 60, acceptance: 1.0);
        Assert.True(ks.Max() >= 5, "the streak must have triggered a probe");
        Assert.True(ks.Skip(150).Count(k => k > 4) <= 12, "after settling, only exploration rounds may run above the configured K");
    }

    [Fact]
    public void AlmostFullRounds_DoNotCountTowardTheStreak()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        for (int i = 0; i < 40; i++) ctl.Record(4, 4, 92);   // emitted = k means k-1 accepted: one short of full
        Assert.Equal(4, ctl.Ceiling);
    }

    [Fact]
    public void AStreakIsConsecutive_AnyShortRoundResetsIt()
    {
        var ctl = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < MtpRoundController.UpwardProbeStreak - 1; j++) ctl.Record(4, 5, 92);   // 5 full rounds ...
            ctl.Record(4, 2, 92);                                                                       // ... then a short one
        }
        Assert.Equal(4, ctl.Ceiling);
    }

    [Fact]
    public void AHigherKThatMeasuredWorse_DoesNotCarryOver()
    {
        var model = new MtpCostModel();
        for (int i = 0; i < 6; i++) { model.RecordRound(4, 4, 10 + 60 * 4); model.RecordRound(5, 5, 10 + 60 * 5); }   // all accepted, K=5 costs more per token
        Assert.True(model.MsPerToken(5) > model.MsPerToken(4));
        Assert.Equal(4, new MtpRoundController(model, 4, maxProbeK: 7).Ceiling);
    }

    [Fact]
    public void AKnownGoodHigherK_SurvivesIntoTheNextRequest()
    {
        var model = new MtpCostModel();
        Run(new MtpRoundController(model, 4, maxProbeK: 7), 120, 60, 8, 1.0);
        var next = new MtpRoundController(model, 4, maxProbeK: 7);
        Assert.True(next.Ceiling > 4, "the cost model measured K>4 as cheaper per token; a new request starts from it");
        var fresh = new MtpRoundController(new MtpCostModel(), 4, maxProbeK: 7);
        Assert.Equal(4, fresh.Ceiling);
    }
}
