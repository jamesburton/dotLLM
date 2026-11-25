using DotLLM.Core.Models;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-state layout of the CUDA Olala forward (#1984), as pure functions of the model geometry so a unit test can check the
/// precision policy without a GPU. The allocator in <see cref="CudaOlalaTransformerModel"/> sizes every buffer from these values, and
/// every recurrent buffer's element width comes from <see cref="OlalaStatePolicy"/> (F32 by James's 2026-11-25 ruling): narrowing
/// one here narrows the allocation, which the guard test then sees.
/// </summary>
internal static class CudaOlalaStateLayout
{
    /// <summary>Bytes per stored activation element: BF16. Olala activations reach ~1e9..1e14, FP16 (65,504) is not an option.</summary>
    public const int ActivationBytes = 2;

    /// <summary>Recurrent state of one Mamba-3 MIMO (<c>M</c>) layer.</summary>
    /// <param name="SsmBytes">SSM matrix <c>[H][P][N]</c>.</param>
    /// <param name="AngleBytes">Rotary phase <c>[H][A]</c>.</param>
    /// <param name="TrapKBytes">Previous token's rotated K <c>[R][H][N]</c>.</param>
    /// <param name="TrapVBytes">Previous token's raw x <c>[H][P]</c>.</param>
    public readonly record struct SsmLayer(long SsmBytes, long AngleBytes, long TrapKBytes, long TrapVBytes)
    {
        /// <summary>Total bytes of the layer.</summary>
        public long Total => SsmBytes + AngleBytes + TrapKBytes + TrapVBytes;
    }

    /// <summary>State of one differential-TPA (<c>V</c>) layer.</summary>
    /// <param name="LastKBytes">Token-shift carry: previous token's raw K <c>[kv heads * head_dim]</c>.</param>
    /// <param name="LastVBytes">Token-shift carry: previous token's raw V.</param>
    /// <param name="HistoryBytesPerToken">BF16 K plus V history per token.</param>
    public readonly record struct AttnLayer(long LastKBytes, long LastVBytes, long HistoryBytesPerToken);

    /// <summary>Per-<c>M</c>-layer state sizes.</summary>
    public static SsmLayer Ssm(OlalaConfig o)
    {
        long h = o.NumSsmHeads;
        return new SsmLayer(
            h * o.MambaHeadDim * o.StateSize * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Ssm),
            h * o.NumRopeAngles * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Angle),
            (long)o.MimoRank * h * o.StateSize * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Trapezoid),
            h * o.MambaHeadDim * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Trapezoid));
    }

    /// <summary>Per-<c>V</c>-layer state sizes.</summary>
    public static AttnLayer Attn(OlalaConfig o, ModelConfig config)
    {
        long kvWidth = (long)o.NumNoiseHeads * config.HeadDim;
        long shift = kvWidth * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.TokenShift);
        return new AttnLayer(shift, shift, 2 * kvWidth * ActivationBytes);
    }

    /// <summary>Resident bytes of the whole sequence state at <paramref name="tokens"/> of KV history.</summary>
    public static long TotalBytes(OlalaConfig o, ModelConfig config, long tokens)
        => Ssm(o).Total * o.SsmLayerCount
         + (Attn(o, config).LastKBytes + Attn(o, config).LastVBytes + Attn(o, config).HistoryBytesPerToken * tokens) * o.AttentionLayerCount;
}
