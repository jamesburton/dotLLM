namespace DotLLM.Core.Lora;

/// <summary>
/// Which sequence region a <see cref="LoraLayerWeights"/> entry applies to.
/// </summary>
/// <remarks>
/// <para>
/// Autoregressive models and single-region adapters use <see cref="Any"/> — the
/// same delta applies uniformly to every row. DiffusionGemma-family real PEFT
/// adapters are trained against the HF model's <c>decoder.layers.*</c> and
/// <c>encoder.language_model.layers.*</c> module trees, which reference the SAME
/// underlying backbone weight but get INDEPENDENT LoRA deltas — the encoder copy
/// is exercised only by the diffusion prompt (encoder) rows, the decoder copy only
/// by the canvas (decoder) rows, of the same unified <c>[prompt|canvas]</c> forward
/// pass. See <c>docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md</c> ("Region per-layer
/// scalar") for the same prompt/canvas split applied to the backbone's own
/// per-layer output scalar — this is the same split, for LoRA.
/// </para>
/// </remarks>
public enum LoraRegion
{
    /// <summary>No region distinction — applies uniformly to every row.</summary>
    Any = 0,

    /// <summary>Applies only to prompt (encoder) rows of a diffusion Hybrid forward.</summary>
    Encoder = 1,

    /// <summary>Applies only to canvas (decoder) rows of a diffusion Hybrid forward.</summary>
    Decoder = 2,
}
