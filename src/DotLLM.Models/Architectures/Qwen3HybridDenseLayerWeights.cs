using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-layer weight references for a Qwen3HybridDense block (e.g. PrismML's Bonsai-27B).
/// Every layer has a pre-mixing RMSNorm, a post-mixing RMSNorm, and a dense SwiGLU FFN.
/// The token-mixing path is either <see cref="Gdn"/> (Gated DeltaNet recurrence) or
/// <see cref="FullAttn"/> (full GQA attention) — exactly one is non-null, identical to
/// <see cref="Qwen3MoeLayerWeights"/>. The only structural difference from the MoE hybrid is
/// the FFN sublayer: dense gate/up/down (this type) instead of sparse MoE routing.
/// </summary>
internal sealed class Qwen3HybridDenseLayerWeights
{
    /// <summary>Pre-token-mixing RMSNorm [hiddenSize] — always present.</summary>
    public required float[] AttnNormWeight { get; init; }

    /// <summary>Post-token-mixing RMSNorm [hiddenSize] — always present (before FFN).</summary>
    public required float[] PostAttnNormWeight { get; init; }

    /// <summary>GDN recurrence weights — non-null for GDN layers.</summary>
    public GdnTokenMixingWeights? Gdn { get; init; }

    /// <summary>Full GQA attention weights — non-null for full-attention layers.</summary>
    public Qwen3FullAttnWeights? FullAttn { get; init; }

    // ── Dense SwiGLU FFN (replaces Qwen3MoeLayerWeights.Moe) ──────────────────
    // GGUF tensor names (prefix = blk.N): ffn_gate.weight, ffn_up.weight, ffn_down.weight —
    // standard dense naming, confirmed against the real Ternary-Bonsai-27B-Q2_0.gguf (no
    // "_exps" suffix, no expert_count metadata). No ffn sub-norm tensor present on Bonsai.

    /// <summary><c>ffn_gate.weight</c> [n_embd, intermediateSize]. Quantized.</summary>
    public required nint GateWeight { get; init; }
    public required QuantizationType GateQuantType { get; init; }
    public required int GateInputDim { get; init; }
    public required int GateOutputDim { get; init; }

    /// <summary><c>ffn_up.weight</c> [n_embd, intermediateSize]. Quantized.</summary>
    public required nint UpWeight { get; init; }
    public required QuantizationType UpQuantType { get; init; }
    public required int UpInputDim { get; init; }
    public required int UpOutputDim { get; init; }

    /// <summary><c>ffn_down.weight</c> [intermediateSize, n_embd]. Quantized.</summary>
    public required nint DownWeight { get; init; }
    public required QuantizationType DownQuantType { get; init; }
    public required int DownInputDim { get; init; }
    public required int DownOutputDim { get; init; }
}
