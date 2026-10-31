namespace DotLLM.Core.Sampling;

/// <summary>
/// Per-step inputs handed to an <see cref="IDiffusionUnmaskSampler"/> by the denoising scheduler.
/// </summary>
/// <param name="Temperature">
/// Sampling temperature for this step, produced by the scheduler's linear schedule
/// (decaying from <c>t_max</c> toward <c>t_min</c> across the step budget).
/// </param>
/// <param name="UnmaskBudget">
/// Maximum number of masked positions the sampler may commit this step. The sampler may commit
/// fewer if the entropy bound is exhausted first. Always at least 1 while masked positions remain.
/// </param>
/// <param name="EntropyBound">
/// Upper bound on the cumulative predictive entropy admitted into a single step
/// (<c>EntropyBoundSamplerConfig.entropy_bound</c>). The sampler unmasks positions in ascending
/// entropy order while their running entropy total stays under this bound.
/// </param>
/// <param name="LogitSoftCap">
/// Optional Gemma-style logit soft-cap (<c>final_logit_softcapping</c>) applied before the
/// entropy softmax: <c>cap * tanh(logit / cap)</c>. <c>0</c> or negative disables capping.
/// </param>
/// <param name="TokenIndexBase">
/// Absolute sequence index of canvas position 0 (prompt tokens plus every earlier canvas). A token committed at canvas
/// position <c>p</c> has generation index <c>TokenIndexBase + p</c>, the counter a position-seeded Gumbel-max draw (#1358) is
/// keyed on, so every committed position gets independent, reproducible noise (#1433).
/// </param>
public readonly record struct DiffusionStepContext(
    float Temperature,
    int UnmaskBudget,
    float EntropyBound,
    float LogitSoftCap = 0f,
    int TokenIndexBase = 0);
