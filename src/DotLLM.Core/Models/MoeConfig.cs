namespace DotLLM.Core.Models;

/// <summary>
/// Dense-routing top-k Mixture-of-Experts configuration. Present on a
/// <see cref="ModelConfig"/> iff the model's FFN is replaced by an MoE block
/// (Mixtral, Qwen*-MoE without shared experts, Phi-3.5-MoE, ...).
/// </summary>
/// <remarks>
/// <para>
/// <b>Semantics (Mixtral convention).</b> For each token the router projects
/// <c>hidden[hidden_size]</c> through <c>gate.weight[num_experts, hidden_size]</c>
/// to produce <c>num_experts</c> logits. Softmax is applied over the full
/// expert set, then the <see cref="NumExpertsPerTok"/> largest entries are
/// gathered. The gathered probabilities are re-normalised by dividing by
/// their own sum (not a second softmax) so the top-k gating weights sum to
/// 1.0 per token. Each selected expert runs an independent SwiGLU MLP over
/// the token's hidden state and its output is scaled by the gating weight
/// and summed.
/// </para>
/// <para>
/// <b>Out of scope for this config.</b> Shared experts (DeepSeek-V3,
/// Qwen1.5-MoE), router aux-loss (training-only), expert parallelism, and
/// fused GroupedGEMM kernels. Those are handled elsewhere in the roadmap.
/// </para>
/// <para>
/// <b>Expert MLP shape.</b> Each expert is a SwiGLU MLP with the same
/// <c>gate_proj</c>/<c>up_proj</c>/<c>down_proj</c> topology as dense Llama
/// — dims <c>[moe_intermediate_size, hidden_size]</c>,
/// <c>[moe_intermediate_size, hidden_size]</c>, and
/// <c>[hidden_size, moe_intermediate_size]</c> respectively. Mixtral reuses
/// the top-level <see cref="ModelConfig.IntermediateSize"/> for the MoE
/// expert width; Phi-3.5-MoE exposes a separate <c>moe_intermediate_size</c>
/// that is surfaced via <see cref="MoeIntermediateSize"/>.
/// </para>
/// </remarks>
public sealed record MoeConfig
{
    /// <summary>
    /// Total number of experts per MoE layer (HF <c>num_local_experts</c> or
    /// <c>num_experts</c>). Typically 8 for Mixtral-8x7B, 16/64/... for others.
    /// Must be &gt; 0 and &gt;= <see cref="NumExpertsPerTok"/>.
    /// </summary>
    public required int NumExperts { get; init; }

    /// <summary>
    /// Number of experts activated per token (HF <c>num_experts_per_tok</c>,
    /// also known as top-k). Typically 2 for Mixtral / Qwen-MoE / Phi-3.5-MoE.
    /// Must satisfy <c>1 &lt;= NumExpertsPerTok &lt;= NumExperts</c>.
    /// </summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>
    /// FFN intermediate width per expert. Mixtral reuses the top-level
    /// <see cref="ModelConfig.IntermediateSize"/> for its experts, while
    /// Phi-3.5-MoE exposes a separate <c>moe_intermediate_size</c>. When the
    /// HF config declares both (<c>intermediate_size</c> ≠
    /// <c>moe_intermediate_size</c>) this carries the per-expert value; when
    /// only <c>intermediate_size</c> exists it mirrors that. Callers SHOULD
    /// use this value, not <see cref="ModelConfig.IntermediateSize"/>, when
    /// allocating MoE expert scratch.
    /// </summary>
    public required int MoeIntermediateSize { get; init; }
}
