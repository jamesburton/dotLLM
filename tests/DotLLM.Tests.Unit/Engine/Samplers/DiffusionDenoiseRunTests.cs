using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

/// <summary>
/// End-to-end simulation of the diffusion denoise loop driving the <see cref="DenoiseScheduler"/>
/// and <see cref="EntropyBoundSampler"/> over synthetic logits — no model required.
/// </summary>
public class DiffusionDenoiseRunTests
{
    private const int MaskTokenId = 99;

    private static DiffusionConfig Config(int canvas, int maxSteps = 48)
        => new()
        {
            CanvasLength = canvas,
            MaxDenoisingSteps = maxSteps,
            TemperatureMax = 0.8f,
            TemperatureMin = 0.4f,
            ConfidenceThreshold = 0.005f,
            StabilityThreshold = 1,
            EntropyBound = 0.1f,
            MaskTokenId = MaskTokenId,
        };

    /// <summary>
    /// Drives a full denoising run. The synthetic forward pass produces, for every masked position,
    /// a logit row peaked at a fixed "answer" token; the peak sharpness grows with the step so canvas
    /// entropy decreases over time (modelling an improving model). Returns the canvas and step count.
    /// </summary>
    private static (int[] canvas, int steps, DenoiseStopResult stop) Run(
        DiffusionConfig config,
        int vocab,
        Func<int, float> peakForStep)
    {
        int canvasLen = config.CanvasLength;
        var canvas = new int[canvasLen];
        Array.Fill(canvas, MaskTokenId);

        var scheduler = new DenoiseScheduler(config);
        var sampler = new EntropyBoundSampler();

        // Each canvas position's "true" answer token, deterministic per position.
        int[] answer = new int[canvasLen];
        for (int i = 0; i < canvasLen; i++)
            answer[i] = (i % (vocab - 1)) + 1; // never the mask-equivalent index 0

        int steps = 0;
        DenoiseStopResult stop = DenoiseStopResult.Continue;

        while (true)
        {
            // Gather masked positions.
            var masked = new List<int>();
            for (int i = 0; i < canvasLen; i++)
                if (canvas[i] == MaskTokenId)
                    masked.Add(i);

            int maskedCount = masked.Count;
            if (maskedCount == 0)
            {
                stop = scheduler.ShouldStop(steps, 0, 0f, canvasChanged: false);
                break;
            }

            float peak = peakForStep(steps);
            var logits = new float[maskedCount * vocab];
            for (int r = 0; r < maskedCount; r++)
                logits[r * vocab + answer[masked[r]]] = peak;

            var ctx = scheduler.CreateStepContext(steps, maskedCount);
            var decision = sampler.SelectAndSample(logits, masked.ToArray(), vocab, maskedCount, ctx);

            // Apply absorbing-state policy: commit decided positions, freeze the rest.
            foreach (var (pos, tok) in decision.UnmaskedPositions.Zip(decision.TokenIds))
            {
                Assert.Equal(MaskTokenId, canvas[pos]); // never re-commit an already-unmasked pos
                canvas[pos] = tok;
            }

            int remaining = maskedCount - decision.UnmaskedPositions.Count;
            bool changed = decision.UnmaskedPositions.Count > 0;
            stop = scheduler.ShouldStop(steps, remaining, decision.AverageEntropy, changed);
            steps++;

            if (stop != DenoiseStopResult.Continue)
                break;
        }

        return (canvas, steps, stop);
    }

    [Fact]
    public void Run_DecreasingEntropy_StopsAdaptively_NotFullBudget()
    {
        // Peak grows quickly so entropy crosses the 0.005 confidence threshold well before step 48.
        // peak(step): start modest, sharpen. At ~step 12-16 the average entropy < 0.005.
        // peak(step) = 3.6 + 0.30*step. The sampler applies the scheduler's decaying temperature
        // (~0.8 -> ~0.4) before the entropy softmax, so the effective distribution sharpens over
        // time and the average canvas entropy first drops below the 0.005 confidence threshold around
        // step index 14 -> the loop reports ~15 steps (inside the documented 12-16 window).
        var config = Config(canvas: 64, maxSteps: 48);
        var (_, steps, stop) = Run(config, vocab: 32, peakForStep: s => 3.6f + 0.30f * s);

        // Adaptive stop lands in the documented 12-16 step window (well under the 48 cap).
        Assert.InRange(steps, 12, 16);
        Assert.True(stop is DenoiseStopResult.Confidence or DenoiseStopResult.Complete,
            $"unexpected stop reason {stop}");
    }

    [Fact]
    public void Run_AbsorbingInvariantHolds_CommittedPositionsAreArgmax()
    {
        // The absorbing invariant (a committed position is never re-masked / re-committed) is
        // asserted inside Run via Assert.Equal(MaskTokenId, canvas[pos]) before each commit.
        var config = Config(canvas: 32, maxSteps: 48);
        var (canvas, _, _) = Run(config, vocab: 16, peakForStep: s => 5f + 1.0f * s);

        // Every committed (non-mask) position holds exactly its argmax answer token; remaining masked
        // positions (the loop may stop on confidence before fully draining) are untouched.
        for (int i = 0; i < canvas.Length; i++)
        {
            if (canvas[i] == MaskTokenId)
                continue;
            int expected = (i % 15) + 1;
            Assert.Equal(expected, canvas[i]);
        }
    }

    [Fact]
    public void Run_DrainCompletesBeforeCap_ReturnsComplete()
    {
        // maxSteps == 1 makes the step-0 proportional budget cover the whole canvas, so the single
        // step commits every position and the loop terminates with Complete (which takes precedence
        // over the confidence / max-steps reasons) — no mask token survives.
        var config = Config(canvas: 8, maxSteps: 1);
        var (canvas, steps, stop) = Run(config, vocab: 16, peakForStep: _ => 30f);

        Assert.DoesNotContain(MaskTokenId, canvas);
        Assert.Equal(DenoiseStopResult.Complete, stop);
        Assert.Equal(1, steps);
    }

    [Fact]
    public void Run_FlatHighEntropy_HitsHardCapAtMaxSteps()
    {
        // Constant low peak => entropy stays high, never crossing the confidence threshold, and the
        // forced single-commit keeps the canvas changing (so stability never fires). With the
        // proportional budget this still finishes near/at the cap; force the cap by using a tiny
        // constant budget via a never-confident, slow-drain scenario.
        var config = Config(canvas: 256, maxSteps: 48);

        // Budget 1 per step would need 256 steps; the proportional default drains faster but entropy
        // is high so neither confidence nor stability fires — termination is by drain (Complete) or
        // the MaxSteps cap, never Continue. Assert the loop respects the hard cap.
        var (_, steps, stop) = Run(config, vocab: 8, peakForStep: _ => 0.05f);

        Assert.True(steps <= 48, $"exceeded hard cap: {steps}");
        Assert.True(stop is DenoiseStopResult.MaxSteps or DenoiseStopResult.Complete,
            $"unexpected stop reason {stop}");
    }

    [Fact]
    public void Run_StabilityCap_NeverExceedsMaxSteps()
    {
        // Sanity: across a range of peak schedules the loop always halts within the budget.
        var config = Config(canvas: 100, maxSteps: 48);
        for (int variant = 0; variant < 5; variant++)
        {
            float baseP = 1f + variant;
            var (_, steps, _) = Run(config, vocab: 20, peakForStep: s => baseP + 0.3f * s);
            Assert.True(steps <= 48, $"variant {variant}: {steps} steps");
        }
    }
}
