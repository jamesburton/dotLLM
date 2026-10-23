using DotLLM.Core.Models;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Policy layer between the diffusion forward pass and the canvas-level unmasking sampler. It owns
/// the per-step <b>temperature schedule</b>, the per-step <b>unmask budget</b>, the adaptive
/// <b>early-stop</b> decision, and the <b>absorbing-state remask</b> policy. It does not run the
/// forward pass or mutate the canvas itself — the caller drives the loop, asking the scheduler for a
/// step plan, sampling, then reporting the outcome.
/// </summary>
/// <remarks>
/// <para>
/// <b>Temperature schedule.</b> Linear decay from <see cref="DiffusionConfig.TemperatureMax"/>
/// (0.8) at step 0 to <see cref="DiffusionConfig.TemperatureMin"/> (0.4) at the last step:
/// <code>t(step) = tMax + (tMin - tMax) * step / (maxSteps - 1)</code>
/// so <c>t(0) = tMax</c>, <c>t(maxSteps-1) = tMin</c>, clamped to <c>[tMin, tMax]</c> for any
/// step beyond the budget. With <c>maxSteps == 1</c> the schedule is constant <c>tMin</c>.
/// </para>
/// <para>
/// <b>Unmask budget.</b> The default is the <i>proportional / ceil-divide</i> schedule
/// <code>budget(step) = ceil(masked / max(1, remainingSteps))</code>
/// where <c>remainingSteps = maxSteps - step</c>. This spreads the still-masked positions evenly
/// across the remaining step budget so an un-stopped run finishes exactly within
/// <see cref="DiffusionConfig.MaxDenoisingSteps"/> steps (the last step always has budget ≥ masked).
/// Override via the <c>budgetPolicy</c> constructor argument to plug a different schedule (e.g. a
/// cosine ramp) without touching the stop logic.
/// </para>
/// <para>
/// <b>Early stop.</b> Mirrors the <see cref="IStopCondition"/> pattern with a diffusion-specific
/// signal: after each step the scheduler is told the average canvas entropy and whether the canvas
/// changed. It stops when (a) all positions are unmasked, OR (b) the average entropy drops below
/// <see cref="DiffusionConfig.ConfidenceThreshold"/> (0.005), OR (c)
/// <see cref="DiffusionConfig.StabilityThreshold"/> (1) consecutive steps made no canvas change,
/// OR (d) the hard cap <see cref="DiffusionConfig.MaxDenoisingSteps"/> is reached.
/// </para>
/// <para>
/// <b>Absorbing-state remask.</b> Default policy is absorbing: positions not selected this step
/// stay <see cref="DiffusionConfig.MaskTokenId"/>, and once a position is unmasked it is frozen and
/// never re-masked. Enable <c>remaskOnLowConfidence</c> to opt into a re-mask-on-low-confidence
/// policy (reserved hook; the default absorbing path is what DiffusionGemma uses).
/// </para>
/// <para>
/// This type is stateless across runs but carries per-run progress (consecutive-stable counter,
/// step index); create one per generation or call <see cref="Reset"/> between runs.
/// </para>
/// </remarks>
public sealed class DenoiseScheduler
{
    private readonly DiffusionConfig _config;
    private readonly Func<int, int, int, int> _budgetPolicy;
    private readonly bool _remaskOnLowConfidence;

    private int _stableStreak;

    /// <summary>
    /// Creates a scheduler driven by the supplied diffusion configuration.
    /// </summary>
    /// <param name="config">Diffusion decode configuration (schedule bounds, thresholds, mask id).</param>
    /// <param name="budgetPolicy">
    /// Optional per-step unmask-budget override. Receives <c>(step, maskedCount, maxSteps)</c> and
    /// returns the number of positions to unmask this step (clamped to <c>[1, maskedCount]</c> by the
    /// scheduler). Null uses the default proportional ceil-divide schedule.
    /// </param>
    /// <param name="remaskOnLowConfidence">
    /// When true, opt into a remask-on-low-confidence policy instead of the default absorbing state.
    /// </param>
    public DenoiseScheduler(
        DiffusionConfig config,
        Func<int, int, int, int>? budgetPolicy = null,
        bool remaskOnLowConfidence = false)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.MaxDenoisingSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(config), config.MaxDenoisingSteps, "MaxDenoisingSteps must be at least 1.");

        _config = config;
        _budgetPolicy = budgetPolicy ?? DefaultBudget;
        _remaskOnLowConfidence = remaskOnLowConfidence;
    }

    /// <summary>Whether the remask-on-low-confidence policy is enabled (default false = absorbing).</summary>
    public bool RemaskOnLowConfidence => _remaskOnLowConfidence;

    /// <summary>Resets per-run progress (stable-step streak) so the scheduler can drive a fresh canvas.</summary>
    public void Reset() => _stableStreak = 0;

    /// <summary>
    /// Sampling temperature for the given (zero-based) denoising step under the linear
    /// <c>tMax → tMin</c> schedule. Clamped to <c>[tMin, tMax]</c> for steps outside the budget.
    /// </summary>
    /// <param name="step">Zero-based step index.</param>
    public float TemperatureForStep(int step)
    {
        float tMax = _config.TemperatureMax;
        float tMin = _config.TemperatureMin;
        int maxSteps = _config.MaxDenoisingSteps;

        if (maxSteps <= 1)
            return tMin;

        float fraction = (float)step / (maxSteps - 1);
        float t = tMax + (tMin - tMax) * fraction;

        // Clamp into [tMin, tMax] regardless of schedule direction.
        float lo = MathF.Min(tMin, tMax);
        float hi = MathF.Max(tMin, tMax);
        return Math.Clamp(t, lo, hi);
    }

    /// <summary>
    /// Number of masked positions to unmask on the given step, clamped to <c>[1, maskedCount]</c>.
    /// Returns 0 only when <paramref name="maskedCount"/> is 0.
    /// </summary>
    /// <param name="step">Zero-based step index.</param>
    /// <param name="maskedCount">Number of positions still masked at the start of the step.</param>
    public int UnmaskBudgetForStep(int step, int maskedCount)
    {
        if (maskedCount <= 0)
            return 0;
        int budget = _budgetPolicy(step, maskedCount, _config.MaxDenoisingSteps);
        return Math.Clamp(budget, 1, maskedCount);
    }

    /// <summary>
    /// Builds the step context (temperature, budget, entropy bound, soft-cap) to hand to the
    /// canvas-level unmasking sampler for the given step.
    /// </summary>
    /// <param name="step">Zero-based step index.</param>
    /// <param name="maskedCount">Number of positions still masked at the start of the step.</param>
    /// <param name="logitSoftCap">Optional Gemma final-logit soft-cap; 0 disables.</param>
    public DiffusionStepContext CreateStepContext(int step, int maskedCount, float logitSoftCap = 0f)
        => new(
            Temperature: TemperatureForStep(step),
            UnmaskBudget: UnmaskBudgetForStep(step, maskedCount),
            EntropyBound: _config.EntropyBound,
            LogitSoftCap: logitSoftCap);

    /// <summary>
    /// Decides whether the denoising loop should stop after a completed step. Mirrors the
    /// <see cref="IStopCondition"/> contract: pure decision over the step outcome, no side effects on
    /// the canvas. Updates the internal consecutive-stable-step counter.
    /// </summary>
    /// <param name="step">Zero-based index of the step that just completed.</param>
    /// <param name="remainingMasked">Masked positions left <i>after</i> applying this step.</param>
    /// <param name="averageEntropy">Average canvas entropy reported by the sampler for this step.</param>
    /// <param name="canvasChanged">Whether this step changed the canvas (committed ≥ 1 position).</param>
    /// <returns>The reason to stop, or <see cref="DenoiseStopResult.Continue"/>.</returns>
    public DenoiseStopResult ShouldStop(int step, int remainingMasked, float averageEntropy, bool canvasChanged)
    {
        // Track consecutive stable (no-change) steps.
        if (canvasChanged)
            _stableStreak = 0;
        else
            _stableStreak++;

        if (remainingMasked <= 0)
            return DenoiseStopResult.Complete;

        // Hard cap: step is zero-based, so the final allowed step is maxSteps - 1.
        if (step >= _config.MaxDenoisingSteps - 1)
            return DenoiseStopResult.MaxSteps;

        if (averageEntropy < _config.ConfidenceThreshold)
            return DenoiseStopResult.Confidence;

        if (_stableStreak >= _config.StabilityThreshold)
            return DenoiseStopResult.Stability;

        return DenoiseStopResult.Continue;
    }

    /// <summary>
    /// Default proportional unmask budget: ceil-divide the masked positions over the remaining steps
    /// so the canvas finishes within the step budget. Always ≥ 1 while masked positions remain.
    /// </summary>
    private static int DefaultBudget(int step, int maskedCount, int maxSteps)
    {
        int remainingSteps = maxSteps - step;
        if (remainingSteps < 1)
            remainingSteps = 1;
        return (maskedCount + remainingSteps - 1) / remainingSteps;
    }
}
