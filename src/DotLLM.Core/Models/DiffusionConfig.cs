namespace DotLLM.Core.Models;

/// <summary>
/// Diffusion-decoding configuration for masked-canvas text-diffusion models
/// (DiffusionGemma). Present on a <see cref="ModelConfig"/> iff the checkpoint
/// is a diffusion wrapper around a Gemma-4 text tower. Null for every
/// autoregressive architecture — those models never consult this record.
/// </summary>
/// <remarks>
/// <para>
/// <b>Decode model.</b> Diffusion text generation refines a fixed-length
/// "canvas" of <see cref="CanvasLength"/> positions over up to
/// <see cref="MaxDenoisingSteps"/> denoising steps. Every position starts as
/// the <see cref="MaskTokenId"/> mask token; each step unmasks a subset of the
/// most-confident positions and (optionally) re-masks low-confidence ones until
/// the canvas stabilises or the step budget is exhausted.
/// </para>
/// <para>
/// <b>Defaults.</b> All numeric fields carry the verified DiffusionGemma
/// defaults so a checkpoint whose <c>generation_config.json</c> omits a field
/// still produces the reference schedule. <see cref="MaskTokenId"/> has no
/// default — it is resolved from the tokenizer metadata at load time and must
/// be supplied explicitly (a diffusion model cannot decode without it).
/// </para>
/// </remarks>
public sealed record DiffusionConfig
{
    /// <summary>
    /// Number of token positions in the diffusion canvas — the fixed length of
    /// the sequence being denoised in one generation pass
    /// (DiffusionGemma <c>config.json</c> <c>canvas_length</c>, default 256).
    /// </summary>
    public int CanvasLength { get; init; } = 256;

    /// <summary>
    /// Maximum number of denoising (unmask/re-mask) iterations per generation
    /// pass (DiffusionGemma <c>generation_config.json</c>
    /// <c>max_denoising_steps</c>, default 48). The loop also terminates early
    /// once the canvas stabilises (see <see cref="StabilityThreshold"/>).
    /// </summary>
    public int MaxDenoisingSteps { get; init; } = 48;

    /// <summary>
    /// Per-position entropy bound for the entropy-bounded unmasking sampler
    /// (<c>EntropyBoundSamplerConfig.entropy_bound</c>, default 0.1). Positions
    /// whose predictive entropy exceeds this bound are not committed in the
    /// current step.
    /// </summary>
    public float EntropyBound { get; init; } = 0.1f;

    /// <summary>
    /// Confidence/entropy early-stop threshold
    /// (<c>EntropyBoundSamplerConfig.confidence_threshold</c>, default 0.005).
    /// Used to decide when a position is confident enough to commit and to gate
    /// early termination of the denoising loop.
    /// </summary>
    public float ConfidenceThreshold { get; init; } = 0.005f;

    /// <summary>
    /// Number of consecutive stable steps (no canvas change) required before
    /// the denoising loop terminates early
    /// (<c>EntropyBoundSamplerConfig.stability_threshold</c>, default 1).
    /// </summary>
    public int StabilityThreshold { get; init; } = 1;

    /// <summary>
    /// Upper bound of the linear sampling-temperature schedule applied across
    /// denoising steps (<c>t_max</c>, default 0.8). The temperature decays
    /// linearly from <see cref="TemperatureMax"/> at the first step toward
    /// <see cref="TemperatureMin"/> at the last.
    /// </summary>
    public float TemperatureMax { get; init; } = 0.8f;

    /// <summary>
    /// Lower bound of the linear sampling-temperature schedule applied across
    /// denoising steps (<c>t_min</c>, default 0.4).
    /// </summary>
    public float TemperatureMin { get; init; } = 0.4f;

    /// <summary>
    /// Vocabulary id of the mask token used to initialise and re-mask canvas
    /// positions. <b>Required.</b> This id is NOT present in
    /// <c>config.json</c>/<c>generation_config.json</c>; it is resolved from the
    /// tokenizer metadata (the <c>[MASK]</c> special token in
    /// <c>special_tokens_map.json</c> / <c>tokenizer_config.json</c> /
    /// <c>tokenizer.json</c> added tokens) at load time and never hardcoded.
    /// Loading a diffusion model fails loudly when this cannot be resolved.
    /// </summary>
    public required int MaskTokenId { get; init; }
}
