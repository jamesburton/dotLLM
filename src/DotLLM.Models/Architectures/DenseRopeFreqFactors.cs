using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Single source of truth for whether a GGUF <c>rope_freqs.weight</c> tensor (llama.cpp
/// <c>freq_factors</c>) must be folded into the MAIN (per-layer-uniform) RoPE frequencies of a
/// dense model — Llama-3.1/3.2/3.3 (llama3 NTK scaling is baked into this tensor by the
/// converter), Ministral/Mistral-3 and friends. llama.cpp applies the factors to every layer from
/// position 0 (<c>model.get_rope_factors</c> → <c>ggml_rope_ext(..., freq_factors, ...)</c>), dividing
/// pair <c>i</c>'s inverse frequency by <c>factor[i]</c>. Before issue #743 the tensor was loaded but
/// consumed only by Gemma-4's GLOBAL table, so every Llama-3.x GGUF ran with unscaled frequencies.
/// </summary>
public static class DenseRopeFreqFactors
{
    /// <summary>
    /// Returns the per-pair divisors to fold into the main RoPE table, or <c>null</c> when the model
    /// has none or the tensor is consumed elsewhere (models with a separate global rotary table,
    /// e.g. Gemma-4, keep their local layers factor-free).
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="factors">The loaded <c>rope_freqs.weight</c> values, or <c>null</c>.</param>
    /// <param name="ropeDim">Rotated dimension count of the main table.</param>
    /// <exception cref="NotSupportedException">The tensor is combined with dense YaRN scaling (not implemented), or has the wrong length.</exception>
    public static float[]? Select(ModelConfig config, float[]? factors, int ropeDim)
    {
        if (factors is null || config.MlaConfig is not null || config.GlobalRoPEConfig is not null)
            return null;
        if (config.Architecture is Architecture.Gemma3 or Architecture.Gemma4)
            return null;
        if (factors.Length < ropeDim / 2)
            throw new NotSupportedException(
                $"rope_freqs.weight has {factors.Length} entries but the model rotates {ropeDim} dims " +
                $"({ropeDim / 2} pairs). Refusing to guess (issue #743).");
        if (config.RoPEConfig is { IsDenseYarnActive: true })
            throw new NotSupportedException(
                "A model with both rope_freqs.weight and YaRN rope scaling is not supported: the combination " +
                "must compose factors with the YaRN ramp and is unimplemented (issue #743). Refusing to load " +
                "rather than silently ignore one of them.");
        return factors;
    }
}
