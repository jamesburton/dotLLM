namespace DotLLM.Core.Models;

/// <summary>
/// Storage type of one component of the Olala recurrent state. Only <see cref="F32"/> exists: the policy is an explicit,
/// tracked exception to the "no FP32 storage" rule (CLAUDE.md precision rule 1, user sign-off 2026-11-25), and a narrower
/// type needs the evidence of issue #1971 first.
/// </summary>
public enum OlalaStateStorage
{
    /// <summary>IEEE single precision (4 bytes per element).</summary>
    F32,
}

/// <summary>
/// Per-architecture recurrent-state precision policy of Olala. It is deliberately a compile-time constant of this architecture:
/// no environment variable, global setting or KV-cache option narrows it, and no other architecture reads it.
/// </summary>
/// <remarks>
/// Scope: the user's 2026-11-25 sign-off covers the SSM matrix and the angle state; an explicit extension the same day (user ruling)
/// puts the trapezoid carry and the token-shift last K/V under F32 too, until #1971 shows a narrower type is safe. HF keeps the
/// trapezoid and token-shift state in the model dtype, so this is stricter than the reference.
/// Evidence for F32: OVHcloud serve Olala with vLLM <c>--mamba-ssm-cache-dtype float32</c>; the OVH llama.cpp fork keeps the state
/// F32 by default and documents that a reduced-precision state "degrades long generations into repetition loops". Covered buffers
/// (per M layer, 7A1B geometry: 48 heads, headdim 64, d_state 128, MIMO rank 4, 32 rope angles):
/// the SSM matrix <c>h</c> (48x64x128 = 1.5 MiB), the rotary angle state (48x32 = 6 KiB), the trapezoid boundary K state
/// (4x48x128 = 96 KiB) and V state (48x64 = 12 KiB); per V layer the token-shift last raw K/V (12x128x2 = 12 KiB).
/// Tracking and the evidence plan for a smaller dtype: issue #1971. Listed in <c>docs/PRECISION.md</c> (Keep-FP32 by policy).
/// </remarks>
public static class OlalaStatePolicy
{
    /// <summary>SSM matrix <c>h[head][p][n]</c>.</summary>
    public const OlalaStateStorage Ssm = OlalaStateStorage.F32;

    /// <summary>Wrapped rotary phase state <c>[head][angle]</c>.</summary>
    public const OlalaStateStorage Angle = OlalaStateStorage.F32;

    /// <summary>Previous-token trapezoid boundary state: rotated K <c>[rank][head][n]</c> and raw V (= x) <c>[head][p]</c>.</summary>
    public const OlalaStateStorage Trapezoid = OlalaStateStorage.F32;

    /// <summary>Previous-token raw K/V of the attention token shift.</summary>
    public const OlalaStateStorage TokenShift = OlalaStateStorage.F32;

    /// <summary>Bytes per stored element for <paramref name="storage"/>.</summary>
    public static int ElementBytes(OlalaStateStorage storage) => storage switch
    {
        OlalaStateStorage.F32 => sizeof(float),
        _ => throw new ArgumentOutOfRangeException(nameof(storage)),
    };
}

/// <summary>
/// Hyper-parameters of the OVHcloud Olala architecture (GGUF <c>general.architecture = olala</c>) on top of the generic
/// <see cref="ModelConfig"/> fields. See <c>docs/OLALA.md</c> for the layer-by-layer semantics.
/// </summary>
public sealed record OlalaConfig
{
    /// <summary>One letter per block: <c>M</c> = Mamba-3 MIMO mixer, <c>V</c> = differential TPA attention (e.g. <c>MMMMV...</c>).</summary>
    public required string LayerTypes { get; init; }

    /// <summary>Mamba inner width (<c>olala.ssm.inner_size</c>, 2 x hidden = 3072).</summary>
    public required int SsmInner { get; init; }

    /// <summary>SSM state size N (<c>olala.ssm.state_size</c> = 128).</summary>
    public required int StateSize { get; init; }

    /// <summary>B/C group count G (<c>olala.ssm.group_count</c>); the only supported value is 1.</summary>
    public required int GroupCount { get; init; }

    /// <summary>Mamba head dimension P (<c>olala.mamba_headdim</c> = 64).</summary>
    public required int MambaHeadDim { get; init; }

    /// <summary>MIMO rank R (<c>olala.mamba_mimo_dim</c> = 4).</summary>
    public required int MimoRank { get; init; }

    /// <summary>Number of rotary angle pairs of the data-dependent RoPE (<c>olala.num_rope_angles</c> = 32).</summary>
    public required int NumRopeAngles { get; init; }

    /// <summary>Signal attention heads of a V layer (36); the layer's output width is <c>NumSignalHeads x head_dim</c>.</summary>
    public required int NumSignalHeads { get; init; }

    /// <summary>Noise attention heads of a V layer (12) = the KV head count.</summary>
    public required int NumNoiseHeads { get; init; }

    /// <summary>TPA rank (4): K and V of a V layer are <c>A (heads x rank) @ B (rank x head_dim) / rank</c>.</summary>
    public required int TpaRank { get; init; }

    /// <summary>Bias added to the V-layer output gate before the SiLU (<c>olala.gate_bias</c>, 1.15 when zero-centred gate).</summary>
    public required float GateBias { get; init; }

    /// <summary>Routed expert count (256).</summary>
    public required int ExpertCount { get; init; }

    /// <summary>Experts used per token (6).</summary>
    public required int ExpertUsedCount { get; init; }

    /// <summary>Routed expert hidden width (768).</summary>
    public required int ExpertFfLength { get; init; }

    /// <summary>Shared expert hidden width (1536; 0 = none).</summary>
    public required int SharedExpertFfLength { get; init; }

    /// <summary>Routed probabilities are normalised to sum 1 then multiplied by this (2.45).</summary>
    public required float ExpertWeightsScale { get; init; }

    /// <summary>Latent width the routed experts operate in (384; 0 = hidden).</summary>
    public required int MoeLatentSize { get; init; }

    /// <summary>Number of SSM heads: <c>SsmInner / MambaHeadDim</c> (48).</summary>
    public int NumSsmHeads => SsmInner / MambaHeadDim;

    /// <summary>Per-head width of the interleaved <c>ssm_in</c> projection: <c>[z(P), x(P), dt, A, trap]</c>.</summary>
    public int SsmInPerHead => 2 * MambaHeadDim + 3;

    /// <summary>True when block <paramref name="layer"/> is a Mamba-3 MIMO (<c>M</c>) layer.</summary>
    public bool IsSsmLayer(int layer) => LayerTypes[layer] == 'M';

    /// <summary>Number of <c>M</c> layers.</summary>
    public int SsmLayerCount => LayerTypes.Count(c => c == 'M');

    /// <summary>Number of <c>V</c> layers.</summary>
    public int AttentionLayerCount => LayerTypes.Count(c => c == 'V');

    /// <summary>Actionable refusal for loaders that cannot run <see cref="DotLLM.Core.Configuration.Architecture.Olala"/> yet.</summary>
    /// <param name="backend">Backend name shown to the user.</param>
    /// <returns>The refusal message.</returns>
    public static string UnsupportedMessage(string backend)
        => $"The olala architecture (OVHcloud Olala: Mamba-3 MIMO + differential TPA + latent MoE) has CPU and Vulkan forwards only; the {backend} backend is not implemented yet "
         + "(epic https://github.com/jamesburton/dotLLM/issues/1969). Loading it through a generic loader would silently run a different model, so it is refused. Use the CPU or Vulkan backend.";
}
