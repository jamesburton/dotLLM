namespace DotLLM.Core.Models;

/// <summary>
/// Gemma-3n-specific architecture parameters: AltUp (Alternating Updates) and
/// Laurel (Learned Augmented Residual Layer) block configuration, plus the
/// per-layer activation-sparsity pattern. Non-null iff
/// <see cref="ModelConfig.Architecture"/> is
/// <see cref="DotLLM.Core.Configuration.Architecture.Gemma3n"/>. Every other
/// architecture leaves <see cref="ModelConfig.Gemma3n"/> null and is
/// unaffected. Mirrors HF <c>Gemma3nTextConfig</c> (see
/// <c>configuration_gemma3n.py</c>).
/// </summary>
/// <remarks>
/// Gemma-3n's Per-Layer Embeddings (PLE — <c>hidden_size_per_layer_input</c> /
/// <c>vocab_size_per_layer_input</c>) and trailing KV-shared layers
/// (<c>num_kv_shared_layers</c>) reuse the SAME <see cref="PerLayerEmbeddingConfig"/>
/// and <see cref="ModelConfig.NumSharedKvLayers"/> slots already built for the
/// dense Gemma-4 text tower (PLE originated in Gemma 3n) — they are not
/// duplicated on this record.
/// </remarks>
public sealed record Gemma3nConfig
{
    /// <summary>
    /// Number of parallel hidden-state streams AltUp maintains per token
    /// (<c>altup_num_inputs</c>; 4 on every released Gemma-3n SKU).
    /// </summary>
    public required int NumInputs { get; init; }

    /// <summary>
    /// Index of the "active" stream — the one attention/Laurel/MLP actually
    /// operate on each layer, and the one whose corrected value seeds the next
    /// layer's <c>predict</c> router (<c>altup_active_idx</c>; always 0 on
    /// released checkpoints).
    /// </summary>
    public required int ActiveIdx { get; init; }

    /// <summary>
    /// Training-time clip applied to the AltUp prediction/correction
    /// coefficient weights (<c>altup_coef_clip</c>, e.g. 120.0). Inference-only
    /// (this codebase never trains), so it has no effect on the forward pass —
    /// carried for completeness / parity with the HF config.
    /// </summary>
    public float? CoefClip { get; init; }

    /// <summary>
    /// When true, the AltUp-corrected active stream is multiplied by the
    /// learned <c>correct_output_scale</c> vector before being used as the
    /// Per-Layer-Embeddings gate input (<c>altup_correct_scale</c>; true on
    /// every released Gemma-3n SKU). The unscaled corrected stream is still
    /// what feeds the next layer's <c>predict</c> — only the local copy used
    /// for the PLE gate is scaled.
    /// </summary>
    public required bool CorrectOutputScale { get; init; }

    /// <summary>
    /// Low-rank bottleneck width for the Laurel (Learned Augmented Residual
    /// Layer) block's <c>linear_left</c> / <c>linear_right</c> pair
    /// (<c>laurel_rank</c>; 64 on every released Gemma-3n SKU).
    /// </summary>
    public required int LaurelRank { get; init; }

    /// <summary>
    /// Per-layer FFN activation-sparsity target (length
    /// <see cref="ModelConfig.NumLayers"/>). Layer <c>i</c> applies the Gaussian
    /// top-k gate (<c>relu(gate_proj - (mean + std·Φ⁻¹(p)))</c>) to its FFN gate
    /// before the activation function when <c>ActivationSparsityPattern[i] &gt;
    /// 0</c>; a value of exactly 0 skips the gate entirely (byte-identical to
    /// the plain GeGLU path). The real E4B/E2B ship 0.95 on the first 10 layers
    /// and 0.0 on every layer after.
    /// </summary>
    public required IReadOnlyList<float> ActivationSparsityPattern { get; init; }
}
