namespace DotLLM.Core.Models;

/// <summary>
/// Ripple (<c>ricursive</c>: llama.cpp GGUF arch <c>ricursive</c>, HF <c>RicursiveForCausalLM</c>) switches that the generic
/// <see cref="ModelConfig"/> fields cannot express. Non-null iff <see cref="ModelConfig.Architecture"/> is
/// <see cref="DotLLM.Core.Configuration.Architecture.Ricursive"/>.
/// </summary>
/// <remarks>
/// <para>
/// The rest of the architecture is carried by generic fields: the hybrid layer pattern by
/// <see cref="ModelConfig.PerLayerSlidingWindow"/> (window on sliding layers, <see langword="null"/> on full-attention layers) and
/// <see cref="ModelConfig.SlidingWindowSize"/> (1024 keys <b>including</b> the current token: HF masks
/// <c>key &gt;= query - (window - 1)</c>); NoPE on the full-attention layers by <see cref="ModelConfig.NoRopeLayers"/> (derived from the
/// pattern: full = NoPE, there is no separate key); NeoX RoPE over the full head dim by <see cref="ModelConfig.RoPEConfig"/>; the
/// embedding multiplier by <see cref="ModelConfig.EmbeddingScale"/>; per-head Q/K RMS-norm by <see cref="ModelConfig.UseQkNorm"/> (always
/// <see langword="true"/>). Every block has four plain RMSNorms (<c>w * x / rms</c>, no Gemma <c>1 + w</c> offset): input, post-attention
/// (inside the residual), pre-FFN and post-FFN — the GGUF names are <c>attn_norm</c>, <c>post_attention_norm</c>, <c>ffn_norm</c>,
/// <c>post_ffw_norm</c>.
/// </para>
/// <para>
/// The per-layer value-residual mix logits (<c>lambda_l</c>, one F32 scalar per layer, sigmoid applied at run time) are weights, not
/// config: GGUF <c>blk.N.attn_vres_lambda.weight</c> (AtomicChat, llama.cpp PR #31692) or <c>blk.N.attn_value_res.weight</c>
/// (prometheusAIR), HF <c>model.layers.N.self_attn.value_res_lambda</c>. GGUF carries no key for the value residual or the gate: like
/// llama.cpp they are discovered from tensor presence (all layers or none) by <c>RicursiveTensorCensus</c>, so
/// <see cref="UseValueResidual"/> / <see cref="UseGatedAttention"/> are <see langword="null"/> on a GGUF-derived config.
/// </para>
/// </remarks>
public sealed record RicursiveConfig
{
    /// <summary>
    /// Elementwise sigmoid output gate (<c>g_proj</c> / <c>attn_gate</c>, hidden → heads*head_dim, input = the normed block input) applied
    /// to the attention output before <c>o_proj</c>. HF <c>use_gated_attention</c> (default true); <see langword="null"/> = not declared
    /// (GGUF): discovered from <c>blk.N.attn_gate.weight</c>.
    /// </summary>
    public bool? UseGatedAttention { get; init; }

    /// <summary>HF <c>gate_full_precision</c>: the gate logit is promoted to F32 before the sigmoid (llama.cpp always does this; GGUF has no key, so true).</summary>
    public required bool GateFullPrecision { get; init; }

    /// <summary>
    /// Value residual: V reads <c>lerp(x0, a, sigmoid(lambda_l))</c> where <c>x0</c> is the scaled token embedding (not normed) and
    /// <c>a</c> the normed block input. HF <c>use_valres</c> (vendor default false); <see langword="null"/> = not declared (GGUF): discovered
    /// from the presence of either value-residual tensor name.
    /// </summary>
    public bool? UseValueResidual { get; init; }

    /// <summary>
    /// Per-layer SwiGLU pre-activation clamp <c>c</c> (length = <see cref="ModelConfig.NumLayers"/>): the FFN computes
    /// <c>down(silu(min(gate, c)) * clamp(up, -c, c))</c>. <c>0</c> = no clamp on that layer (the reference tests <c>if clamp_threshold:</c>).
    /// </summary>
    public required IReadOnlyList<float> FfnClamp { get; init; }

    /// <summary>
    /// Whether the clamp was declared by the source (GGUF <c>{arch}.swiglu_clamp</c> scalar or per-layer array; HF a non-null
    /// <c>ffn_clamp_threshold</c>). <see langword="false"/> = the explicit "no clamp" variant (the converters write no key when the HF
    /// value is null) — kept distinct from a declared <c>0</c> so a dropped key can be told apart in diagnostics.
    /// </summary>
    public required bool FfnClampDeclared { get; init; }

    /// <summary>
    /// <see langword="true"/> when the source declares no embedding scale (GGUF <c>{arch}.embedding_scale</c> absent; HF
    /// <c>embed_scale: null</c>): the vendor convention for checkpoints whose scale is folded into <c>embed_tokens.weight</c>, so
    /// <see cref="ModelConfig.EmbeddingScale"/> is <see langword="null"/>. A present scale must be finite and positive (0 is rejected,
    /// as llama.cpp PR #31692's converter does, rather than silently meaning "folded").
    /// </summary>
    public required bool EmbeddingScaleFolded { get; init; }

    /// <summary>Vendor default layer pattern when none is given: every <c>DefaultFullAttentionPeriod</c>-th layer (index % 5 == 4) is full attention.</summary>
    public const int DefaultFullAttentionPeriod = 5;

    /// <summary>The two GGUF names of the per-layer value-residual logit seen in the wild (llama.cpp PR #31692 / AtomicChat first).</summary>
    public static readonly IReadOnlyList<string> ValueResidualTensorNames = ["attn_vres_lambda", "attn_value_res"];

    /// <summary>The clamp of <paramref name="layer"/> (0 = none).</summary>
    /// <param name="layer">Layer index.</param>
    /// <returns>The clamp threshold.</returns>
    public float FfnClampOf(int layer) => (uint)layer < (uint)FfnClamp.Count ? FfnClamp[layer] : 0f;

    /// <summary>Actionable refusal for loaders that cannot run <see cref="DotLLM.Core.Configuration.Architecture.Ricursive"/> yet.</summary>
    /// <param name="backend">Backend name shown to the user.</param>
    /// <returns>The refusal message.</returns>
    public static string UnsupportedMessage(string backend)
        => $"The ricursive architecture (GGUF 'ricursive', HF RicursiveForCausalLM: ricursiveAI Ripple-65B-Preview) is recognised but has no "
         + $"forward pass yet (backend: {backend}) - only config extraction, the tensor census, the tokenizer and the chat template exist (issue #1643). "
         + "Loading it through a generic loader would drop the value residual, the separate sigmoid output gate, the sandwich norms, the "
         + "NoPE full-attention layers and the SwiGLU clamp and produce garbage silently, so it is refused. "
         + "CPU forward is tracked in https://github.com/jamesburton/dotLLM/issues/1644, Vulkan in #1645.";
}
