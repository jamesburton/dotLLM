namespace DotLLM.Engine.Samplers;

/// <summary>
/// Outcome of a <see cref="DenoiseScheduler.ShouldStop"/> check, mirroring the autoregressive
/// <see cref="DotLLM.Core.Sampling.StopResult"/> pattern but naming the diffusion-specific reason a
/// denoising loop terminated.
/// </summary>
public enum DenoiseStopResult
{
    /// <summary>Keep denoising — none of the stop conditions fired.</summary>
    Continue,

    /// <summary>Every canvas position is unmasked; nothing left to denoise.</summary>
    Complete,

    /// <summary>Average canvas entropy fell below the confidence threshold.</summary>
    Confidence,

    /// <summary>The required number of consecutive stable (no-change) steps was reached.</summary>
    Stability,

    /// <summary>The hard <c>MaxDenoisingSteps</c> cap was reached.</summary>
    MaxSteps,
}
