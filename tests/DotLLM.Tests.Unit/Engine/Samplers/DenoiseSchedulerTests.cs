using DotLLM.Core.Models;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class DenoiseSchedulerTests
{
    private static DiffusionConfig Config(
        int maxSteps = 48,
        float tMax = 0.8f,
        float tMin = 0.4f,
        float confidence = 0.005f,
        int stability = 1,
        float entropyBound = 0.1f)
        => new()
        {
            CanvasLength = 256,
            MaxDenoisingSteps = maxSteps,
            TemperatureMax = tMax,
            TemperatureMin = tMin,
            ConfidenceThreshold = confidence,
            StabilityThreshold = stability,
            EntropyBound = entropyBound,
            MaskTokenId = 99,
        };

    [Fact]
    public void Temperature_AtStepZero_IsTemperatureMax()
    {
        var scheduler = new DenoiseScheduler(Config());
        Assert.Equal(0.8f, scheduler.TemperatureForStep(0), precision: 5);
    }

    [Fact]
    public void Temperature_AtLastStep_IsTemperatureMin()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 48));
        Assert.Equal(0.4f, scheduler.TemperatureForStep(47), precision: 5);
    }

    [Fact]
    public void Temperature_AtMidStep_IsLinearMidpoint()
    {
        // maxSteps = 49 -> last step = 48, midpoint step 24 -> exactly halfway = 0.6.
        var scheduler = new DenoiseScheduler(Config(maxSteps: 49));
        Assert.Equal(0.6f, scheduler.TemperatureForStep(24), precision: 5);
    }

    [Fact]
    public void Temperature_FollowsLinearFormula()
    {
        const int maxSteps = 48;
        var scheduler = new DenoiseScheduler(Config(maxSteps: maxSteps));
        for (int step = 0; step < maxSteps; step++)
        {
            float expected = 0.8f + (0.4f - 0.8f) * step / (maxSteps - 1);
            Assert.Equal(expected, scheduler.TemperatureForStep(step), precision: 5);
        }
    }

    [Fact]
    public void Temperature_BeyondBudget_ClampedToMin()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 48));
        Assert.Equal(0.4f, scheduler.TemperatureForStep(100), precision: 5);
    }

    [Fact]
    public void Temperature_SingleStep_IsTemperatureMin()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 1));
        Assert.Equal(0.4f, scheduler.TemperatureForStep(0), precision: 5);
    }

    [Fact]
    public void Temperature_MonotonicallyNonIncreasing()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 48));
        float prev = scheduler.TemperatureForStep(0);
        for (int step = 1; step < 48; step++)
        {
            float cur = scheduler.TemperatureForStep(step);
            Assert.True(cur <= prev + 1e-6f, $"step {step}: {cur} > {prev}");
            prev = cur;
        }
    }

    [Fact]
    public void Budget_ZeroMasked_IsZero()
    {
        var scheduler = new DenoiseScheduler(Config());
        Assert.Equal(0, scheduler.UnmaskBudgetForStep(0, maskedCount: 0));
    }

    [Fact]
    public void Budget_LastStep_CoversAllRemaining()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 10));
        // Final step (index 9) — remainingSteps = 1 — must unmask everything left.
        Assert.Equal(7, scheduler.UnmaskBudgetForStep(step: 9, maskedCount: 7));
    }

    [Fact]
    public void Budget_DefaultProportional_FinishesWithinStepBudget()
    {
        const int maxSteps = 48;
        var scheduler = new DenoiseScheduler(Config(maxSteps: maxSteps));
        int masked = 256;
        int steps = 0;
        while (masked > 0 && steps < maxSteps)
        {
            int budget = scheduler.UnmaskBudgetForStep(steps, masked);
            Assert.InRange(budget, 1, masked);
            masked -= budget;
            steps++;
        }
        Assert.Equal(0, masked);
        Assert.True(steps <= maxSteps);
    }

    [Fact]
    public void Budget_CustomPolicy_IsUsedAndClamped()
    {
        // Policy asks for a huge budget; scheduler clamps to maskedCount.
        var scheduler = new DenoiseScheduler(Config(), budgetPolicy: (_, _, _) => 1_000);
        Assert.Equal(5, scheduler.UnmaskBudgetForStep(0, maskedCount: 5));
    }

    [Fact]
    public void Budget_CustomPolicy_BelowOneClampedToOne()
    {
        var scheduler = new DenoiseScheduler(Config(), budgetPolicy: (_, _, _) => 0);
        Assert.Equal(1, scheduler.UnmaskBudgetForStep(0, maskedCount: 5));
    }

    [Fact]
    public void ShouldStop_AllUnmasked_ReturnsComplete()
    {
        var scheduler = new DenoiseScheduler(Config());
        var result = scheduler.ShouldStop(step: 3, remainingMasked: 0, averageEntropy: 5f, canvasChanged: true);
        Assert.Equal(DenoiseStopResult.Complete, result);
    }

    [Fact]
    public void ShouldStop_LowEntropy_ReturnsConfidence()
    {
        var scheduler = new DenoiseScheduler(Config(confidence: 0.005f));
        var result = scheduler.ShouldStop(step: 3, remainingMasked: 10, averageEntropy: 0.001f, canvasChanged: true);
        Assert.Equal(DenoiseStopResult.Confidence, result);
    }

    [Fact]
    public void ShouldStop_StableSteps_ReturnsStability()
    {
        var scheduler = new DenoiseScheduler(Config(stability: 1));
        // One stable (no-change) step trips the threshold of 1.
        var result = scheduler.ShouldStop(step: 3, remainingMasked: 10, averageEntropy: 5f, canvasChanged: false);
        Assert.Equal(DenoiseStopResult.Stability, result);
    }

    [Fact]
    public void ShouldStop_StabilityThreshold_RequiresConsecutive()
    {
        var scheduler = new DenoiseScheduler(Config(stability: 2));
        // First stable step: streak = 1 < 2 -> continue.
        Assert.Equal(DenoiseStopResult.Continue,
            scheduler.ShouldStop(0, remainingMasked: 10, averageEntropy: 5f, canvasChanged: false));
        // Second stable step: streak = 2 -> stop.
        Assert.Equal(DenoiseStopResult.Stability,
            scheduler.ShouldStop(1, remainingMasked: 10, averageEntropy: 5f, canvasChanged: false));
    }

    [Fact]
    public void ShouldStop_ChangeResetsStableStreak()
    {
        var scheduler = new DenoiseScheduler(Config(stability: 2));
        Assert.Equal(DenoiseStopResult.Continue,
            scheduler.ShouldStop(0, 10, 5f, canvasChanged: false)); // streak 1
        Assert.Equal(DenoiseStopResult.Continue,
            scheduler.ShouldStop(1, 10, 5f, canvasChanged: true));  // reset to 0
        Assert.Equal(DenoiseStopResult.Continue,
            scheduler.ShouldStop(2, 10, 5f, canvasChanged: false)); // streak 1 again
    }

    [Fact]
    public void ShouldStop_HardCap_AtLastStep()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 48));
        // Step index 47 is the last allowed step (zero-based); high entropy + change would
        // otherwise continue, but the hard cap fires.
        var result = scheduler.ShouldStop(step: 47, remainingMasked: 10, averageEntropy: 5f, canvasChanged: true);
        Assert.Equal(DenoiseStopResult.MaxSteps, result);
    }

    [Fact]
    public void ShouldStop_NoCondition_ReturnsContinue()
    {
        var scheduler = new DenoiseScheduler(Config());
        var result = scheduler.ShouldStop(step: 3, remainingMasked: 10, averageEntropy: 5f, canvasChanged: true);
        Assert.Equal(DenoiseStopResult.Continue, result);
    }

    [Fact]
    public void Reset_ClearsStableStreak()
    {
        var scheduler = new DenoiseScheduler(Config(stability: 2));
        scheduler.ShouldStop(0, 10, 5f, canvasChanged: false); // streak 1
        scheduler.Reset();
        // After reset, a single stable step should not trip a threshold of 2.
        Assert.Equal(DenoiseStopResult.Continue,
            scheduler.ShouldStop(0, 10, 5f, canvasChanged: false));
    }

    [Fact]
    public void Constructor_InvalidMaxSteps_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenoiseScheduler(Config(maxSteps: 0)));
    }

    [Fact]
    public void CreateStepContext_CarriesScheduleValues()
    {
        var scheduler = new DenoiseScheduler(Config(maxSteps: 48));
        var ctx = scheduler.CreateStepContext(step: 0, maskedCount: 48, logitSoftCap: 30f);
        Assert.Equal(0.8f, ctx.Temperature, precision: 5);
        Assert.Equal(0.1f, ctx.EntropyBound, precision: 5);
        Assert.Equal(30f, ctx.LogitSoftCap, precision: 5);
        Assert.InRange(ctx.UnmaskBudget, 1, 48);
    }

    [Fact]
    public void RemaskOnLowConfidence_DefaultsToAbsorbing()
    {
        Assert.False(new DenoiseScheduler(Config()).RemaskOnLowConfidence);
        Assert.True(new DenoiseScheduler(Config(), remaskOnLowConfidence: true).RemaskOnLowConfidence);
    }
}
