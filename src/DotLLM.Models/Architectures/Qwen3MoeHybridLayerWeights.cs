using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-layer weight references for a Qwen3MoeHybrid block.
/// Every layer has a pre-mixing RMSNorm, a post-mixing RMSNorm, and a sparse MoE FFN.
/// The token-mixing path is either <see cref="Gdn"/> (Gated DeltaNet recurrence)
/// or <see cref="FullAttn"/> (full GQA attention) — exactly one is non-null.
/// </summary>
internal sealed class Qwen3MoeLayerWeights
{
    /// <summary>Pre-token-mixing RMSNorm [hiddenSize] — always present.</summary>
    public required float[] AttnNormWeight { get; init; }

    /// <summary>Post-token-mixing RMSNorm [hiddenSize] — always present (before MoE FFN).</summary>
    public required float[] PostAttnNormWeight { get; init; }

    /// <summary>GDN recurrence weights — non-null for GDN layers.</summary>
    public GdnTokenMixingWeights? Gdn { get; init; }

    /// <summary>Full GQA attention weights — non-null for full-attention layers.</summary>
    public Qwen3FullAttnWeights? FullAttn { get; init; }

    /// <summary>Sparse MoE FFN weights — always present.</summary>
    public required MoeLayerWeights Moe { get; init; }
}

/// <summary>
/// Gated DeltaNet (GDN) token-mixing weights for a single Qwen3MoeHybrid GDN layer.
/// </summary>
/// <remarks>
/// GGUF tensor names (prefix = <c>blk.N</c>):
/// <list type="bullet">
///   <item><c>attn_qkv.weight</c> — fused Q/K/V projection; in GDN context Q and K share
///     dimension <c>NKHead*DState</c>, V has dimension <c>NVHead*DState</c>, giving total
///     output width <c>2*NKHead*DState + NVHead*DState</c>.</item>
///   <item><c>attn_gate.weight</c> — output gate z; final output is
///     <c>RMSNorm(gdn_output) * silu(z)</c>.</item>
///   <item><c>ssm_a</c> — log-space decay magnitude A per head [NVHead], F32. No .weight suffix.</item>
///   <item><c>ssm_alpha.weight</c> — decay projection α [n_embd → NVHead].</item>
///   <item><c>ssm_beta.weight</c> — write-gate β projection [n_embd → NVHead].</item>
///   <item><c>ssm_conv1d.weight</c> — causal 1-D conv on K [DConv, NKHead*DState], F32.</item>
///   <item><c>ssm_dt.bias</c> — delta-time bias [NVHead], F32.</item>
///   <item><c>ssm_norm.weight</c> — SSM output RMSNorm gains [DState], F32 (broadcast across all NVHead heads).</item>
///   <item><c>ssm_out.weight</c> — output projection [NVHead*DState → n_embd].</item>
/// </list>
/// </remarks>
internal sealed class GdnTokenMixingWeights
{
    // ── Fused Q/K/V projection ────────────────────────────────────────────────

    /// <summary><c>attn_qkv.weight</c> [n_embd, 2*NKHead*DState + NVHead*DState]. Quantized.</summary>
    public required nint QkvWeight { get; init; }
    public required QuantizationType QkvQuantType { get; init; }
    public required int QkvInputDim { get; init; }
    public required int QkvOutputDim { get; init; }

    // ── Post-recurrence output gate ───────────────────────────────────────────

    /// <summary>
    /// <c>attn_gate.weight</c> [n_embd, NVHead*DState]. Quantized.
    /// Gate z applied as <c>RMSNorm(gdn_out) * silu(z)</c> before <c>ssm_out</c>.
    /// </summary>
    public required nint GateWeight { get; init; }
    public required QuantizationType GateQuantType { get; init; }
    public required int GateInputDim { get; init; }
    public required int GateOutputDim { get; init; }

    // ── GDN recurrence scalars ────────────────────────────────────────────────

    /// <summary><c>ssm_a</c> — log-space per-head decay magnitude [NVHead]. F32.</summary>
    public required float[] A { get; init; }

    // ── Decay and write-gate projections ─────────────────────────────────────

    /// <summary><c>ssm_alpha.weight</c> — decay projection [n_embd → NVHead]. Quantized.</summary>
    public required nint AlphaWeight { get; init; }
    public required QuantizationType AlphaQuantType { get; init; }
    public required int AlphaInputDim { get; init; }
    public required int AlphaOutputDim { get; init; }

    /// <summary><c>ssm_beta.weight</c> — write-gate projection [n_embd → NVHead]. Quantized.</summary>
    public required nint BetaWeight { get; init; }
    public required QuantizationType BetaQuantType { get; init; }
    public required int BetaInputDim { get; init; }
    public required int BetaOutputDim { get; init; }

    // ── Small F32 tensors ─────────────────────────────────────────────────────

    /// <summary><c>ssm_conv1d.weight</c> [DConv, (2*NKHead+NVHead)*DState]. Dequantized to F32 at load time.</summary>
    public required float[] Conv1dWeight { get; init; }

    /// <summary>
    /// Conv1d bias — zeros, shape [convDim = (2*NKHead+NVHead)*DState].
    /// GDN has no GGUF conv bias tensor; this zero buffer satisfies
    /// <c>Conv1dCausal.Execute</c>'s <c>bias.Length >= channels</c> precondition.
    /// </summary>
    public required float[] Conv1dBias { get; init; }

    /// <summary><c>ssm_dt.bias</c> [NVHead]. F32.</summary>
    public required float[] DtBias { get; init; }

    /// <summary><c>ssm_norm.weight</c> [DState] — SSM output RMSNorm gains, broadcast across all NVHead heads. F32.</summary>
    public required float[] SsmNormWeight { get; init; }

    // ── Output projection ─────────────────────────────────────────────────────

    /// <summary><c>ssm_out.weight</c> [NVHead*DState, n_embd] — maps GDN output to residual. Quantized.</summary>
    public required nint OutWeight { get; init; }
    public required QuantizationType OutQuantType { get; init; }
    public required int OutInputDim { get; init; }
    public required int OutOutputDim { get; init; }
}

/// <summary>
/// Full GQA attention weights for a Qwen3MoeHybrid attention layer.
/// </summary>
/// <remarks>
/// <para>
/// GGUF tensor names (prefix = <c>blk.N</c>):
/// </para>
/// <list type="bullet">
///   <item><c>attn_q.weight</c> — fused Q+Gate projection [n_embd, 2*nQ*headDim].
///     Output is interleaved per head: <c>[Q_h0(headDim), Gate_h0(headDim), Q_h1(headDim), Gate_h1(headDim), ...]</c>.
///     Verified against llama.cpp <c>src/models/qwen35moe.cpp</c> (<c>create_tensor_qkv</c> with
///     <c>n_embd_q = n_embd_head_k * n_head * 2</c>).</item>
///   <item><c>attn_k.weight</c> — K projection [n_embd, nKV*headDim].</item>
///   <item><c>attn_v.weight</c> — V projection [n_embd, nKV*headDim].</item>
///   <item><c>attn_output.weight</c> — output projection W_o [nQ*headDim, n_embd].</item>
///   <item><c>attn_q_norm.weight</c> — per-head Q normalisation [headDim] (Qwen3 QK-norm).</item>
///   <item><c>attn_k_norm.weight</c> — per-head K normalisation [headDim].</item>
/// </list>
/// <para>
/// Forward semantics (verified against llama.cpp <c>build_layer_attn</c> in qwen35moe.cpp):
/// </para>
/// <code>
/// QG_full = attn_q @ x         // [seqLen, 2 * nQ * headDim]
/// Q       = slice(QG_full, even half-dim per head)
/// gate    = slice(QG_full, odd  half-dim per head)
/// Q       = RmsNorm(Q, attn_q_norm)
/// K       = RmsNorm(attn_k @ x, attn_k_norm)
/// V       = attn_v @ x
/// (Q, K)  = MultiRope(Q, K, positions, rope_sections)
/// attn    = Attention(Q, K, V)
/// attn    = attn * sigmoid(gate)
/// out     = attn_output @ attn
/// </code>
/// </remarks>
internal sealed class Qwen3FullAttnWeights
{
    // ── Fused Q+Gate projection ───────────────────────────────────────────────

    /// <summary>
    /// <c>attn_q.weight</c> [n_embd, 2*nQ*headDim]. Quantized.
    /// Output layout is interleaved per head: <c>[Q_h0, Gate_h0, Q_h1, Gate_h1, ...]</c>
    /// with a per-head stride of <c>2*headDim</c>.
    /// </summary>
    public required nint QWeight { get; init; }
    public required QuantizationType QQuantType { get; init; }
    public required int QInputDim { get; init; }
    public required int QOutputDim { get; init; }

    /// <summary><c>attn_k.weight</c> [n_embd, nKV*headDim]. Quantized.</summary>
    public required nint KWeight { get; init; }
    public required QuantizationType KQuantType { get; init; }
    public required int KInputDim { get; init; }
    public required int KOutputDim { get; init; }

    /// <summary><c>attn_v.weight</c> [n_embd, nKV*headDim]. Quantized.</summary>
    public required nint VWeight { get; init; }
    public required QuantizationType VQuantType { get; init; }
    public required int VInputDim { get; init; }
    public required int VOutputDim { get; init; }

    // ── Output projection ─────────────────────────────────────────────────────

    /// <summary><c>attn_output.weight</c> W_o [nQ*headDim, n_embd]. Quantized.</summary>
    public required nint OWeight { get; init; }
    public required QuantizationType OQuantType { get; init; }
    public required int OInputDim { get; init; }
    public required int OOutputDim { get; init; }

    // ── Attention shape metadata ──────────────────────────────────────────────

    /// <summary>KV-head count for this layer (from <see cref="DotLLM.Core.Models.HybridLayerLayout.HeadCountKv"/>).</summary>
    public required int NumKvHeads { get; init; }

    // ── QK-norm (always present on qwen35moe full-attn) ───────────────────────

    /// <summary><c>attn_q_norm.weight</c> [headDim]. Required by llama.cpp qwen35moe loader.</summary>
    public required float[] QNormWeight { get; init; }

    /// <summary><c>attn_k_norm.weight</c> [headDim]. Required by llama.cpp qwen35moe loader.</summary>
    public required float[] KNormWeight { get; init; }
}
