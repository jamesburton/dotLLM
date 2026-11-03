using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.ChunkReuse;

/// <summary>
/// The RoPE parameters <see cref="RopeShift"/> needs to re-rotate stored keys: rotated dims per head, base frequency and
/// element pairing. Only plain RoPE is representable, the variant used by the proof model (Qwen3-4B-Instruct-2507: NeoX,
/// theta 5e6, full-head rotation, no scaling, no <c>rope_freqs.weight</c>).
/// </summary>
/// <param name="RopeDim">Leading dims of each head that rotate (even, &lt;= head dim).</param>
/// <param name="Theta">RoPE base frequency.</param>
/// <param name="Type">Element pairing: <see cref="RoPEType.Norm"/> (interleaved) or <see cref="RoPEType.NeoX"/> (rotate_half).</param>
public readonly record struct RopeParameters(int RopeDim, float Theta, RoPEType Type)
{
    /// <summary>
    /// Derives the parameters from <paramref name="config"/> exactly as the CPU <c>TransformerModel</c> builds its main RoPE table
    /// (<c>RopeDim = RoPEConfig.DimensionCount</c>, or the head dim when 0; <c>Theta = RoPEConfig.Theta</c>).
    /// </summary>
    /// <remarks>
    /// A shift by <c>Δ</c> is the pair rotation by <c>Δ·freq_i</c>; it composes exactly (<c>R(Δ)·R(p) = R(p+Δ)</c>) only when the
    /// model's per-position transform is a pure rotation with position-independent frequencies. Every variant this method cannot
    /// prove that for is refused by name rather than shifted with a guessed table. Not detectable here: a GGUF
    /// <c>rope_freqs.weight</c> tensor (Llama-3.1-style factors) is not part of <see cref="ModelConfig"/>, so a model carrying one
    /// would be shifted with unscaled frequencies.
    /// </remarks>
    /// <param name="config">Model configuration.</param>
    /// <returns>The plain-RoPE parameters.</returns>
    /// <exception cref="NotSupportedException">The configuration uses a RoPE variant this shift does not reproduce.</exception>
    public static RopeParameters FromConfig(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.RoPEConfig is not { } rope)
            throw new NotSupportedException(
                $"Model architecture {config.Architecture} has no RoPEConfig; chunk KV re-rotation needs an explicit RoPE configuration.");
        if (config.MlaConfig is not null)
            throw new NotSupportedException("RoPE variant 'MLA decoupled rope' (DeepSeek-V2/V3) is not supported by RopeShift.");
        if (config.GlobalRoPEConfig is not null)
            throw new NotSupportedException(
                "RoPE variant 'per-attention-type RoPE' (separate global table, Gemma-3/4) is not supported by RopeShift.");
        if (config.GlobalHeadDim is int g && g != config.HeadDim)
            throw new NotSupportedException("RoPE variant 'distinct global head dim' (Gemma-4) is not supported by RopeShift.");

        switch (rope.ScalingType)
        {
            case RoPEScalingType.None:
                break;
            case RoPEScalingType.YaRN when !rope.IsDenseYarnActive:
                break; // factor <= 1 or no original length: the model builds the plain table.
            default:
                // YaRN: the ramped frequencies are position-independent, so a rotation by Δ·invFreq_i would compose, but the
                // model's table also scales cos/sin by mscale; reproducing it needs the ramp + mscale convention verified
                // against a forward, which is not done yet. Linear/NTK/DynamicNTK/Su: the CPU dense forward does not apply them
                // to the main table today, so the config and the model disagree; refuse rather than pick one.
                throw new NotSupportedException(
                    $"RoPE scaling variant '{rope.ScalingType}' (factor {rope.ScalingFactor}) is not supported by RopeShift; only plain RoPE is.");
        }

        int ropeDim = rope.DimensionCount > 0 ? rope.DimensionCount : config.HeadDim;
        if (ropeDim % 2 != 0 || ropeDim > config.HeadDim)
            throw new NotSupportedException($"RoPE dimension count {ropeDim} must be even and <= head dim {config.HeadDim}.");
        return new RopeParameters(ropeDim, rope.Theta, rope.Type);
    }
}
