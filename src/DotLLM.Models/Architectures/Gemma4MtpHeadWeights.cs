using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Weights for the Gemma 4 MTP drafter model (<c>gemma4-assistant</c> architecture, issue #959).
/// A standalone 4-layer mini-transformer (78M params for E4B) that drafts speculative tokens
/// via Q-only cross-attention into the trunk's KV-cache. Ships as a separate GGUF
/// (<c>mtp-*.gguf</c>, ~57 MB for E4B Q4_0) beside the trunk checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// <b>Structural differences from Qwen MTP (<see cref="MtpHeadWeights"/>):</b> the Qwen MTP
/// head is a single nextn decoder block appended to the trunk with <c>enorm/hnorm/eh_proj</c>
/// glue; the Gemma 4 drafter is a complete standalone model with its own token embeddings,
/// 4 decoder layers (all Q-only, shared-KV), and <c>pre_projection/post_projection</c> glue.
/// </para>
/// <para>
/// <b>Forward flow (per draft step):</b>
/// <c>pre_projection(concat(trunkEmbed(token) * sqrt(trunk_hidden_size), trunk_hidden))</c>
/// → 4 decoder layers (Q-only GQA cross-attention into trunk KV + GeGLU FFN) →
/// <c>output_norm</c> → drafter's own <c>token_embd</c> [256→vocab] as LM head for logits.
/// The <c>post_projection</c> output (256 → 2560) feeds back as the next draft step's
/// hidden-state input. The token embedding for the INPUT uses the TRUNK's <c>tok_embd</c>
/// (2560-dim), not the drafter's (256-dim).
/// </para>
/// <para>
/// <b>Constant position IDs:</b> all draft steps use the same RoPE position — the position
/// of the last target-model token — unlike Qwen's advancing-position design.
/// </para>
/// </remarks>
internal sealed class Gemma4MtpHeadWeights
{
    /// <summary>Number of decoder layers in the drafter (typically 4).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Drafter's internal hidden size (typically 256).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Trunk's hidden size — the output dimension of <see cref="PostProjectionWeight"/> (typically 2560 for E4B).</summary>
    public required int TrunkHiddenSize { get; init; }

    /// <summary>FFN intermediate size (typically 2048).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of Q attention heads (typically 4).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of KV heads the drafter reads from the trunk's cache (typically 2).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Vocabulary size (typically 262144, matching the trunk).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Maximum draft tokens per speculation round.</summary>
    public required int NextnPredictLayers { get; init; }

    /// <summary>
    /// <c>nextn.pre_projection.weight</c> GGUF shape [trunkHidden * 2, drafterHidden].
    /// Maps 5120→256: inputDim = trunkHidden*2, outputDim = drafterHidden (GGUF shape[0] = inputDim).
    /// </summary>
    public required nint PreProjectionWeight { get; init; }
    public required QuantizationType PreProjectionQuantType { get; init; }

    /// <summary>
    /// <c>nextn.post_projection.weight</c> GGUF shape [drafterHidden, trunkHidden].
    /// Maps 256→2560: inputDim = drafterHidden, outputDim = trunkHidden (GGUF shape[0] = inputDim).
    /// </summary>
    public required nint PostProjectionWeight { get; init; }
    public required QuantizationType PostProjectionQuantType { get; init; }

    /// <summary><c>token_embd.weight</c> GGUF shape [drafterHidden, vocabSize]. Quantized. Drafter's own embeddings.</summary>
    public required nint TokenEmbdWeight { get; init; }
    public required QuantizationType TokenEmbdQuantType { get; init; }

    /// <summary><c>output_norm.weight</c> [drafterHidden]. Drafter's final RMSNorm before the LM head.</summary>
    public required float[] OutputNormWeight { get; init; }

    /// <summary>Optional <c>rope_freqs.weight</c> [ropeDim]. Proportional-rope frequency factors.</summary>
    public float[]? RopeFreqsWeight { get; init; }

    /// <summary>Per-layer decoder weights (Q-only attention + GeGLU FFN + norms).</summary>
    public required Gemma4MtpLayerWeights[] Layers { get; init; }
}

/// <summary>
/// Per-layer weights for a Gemma 4 MTP drafter decoder layer. Q-only attention (no K/V weights)
/// with GeGLU FFN and Gemma-style four-norm residual + learned output scale.
/// </summary>
internal sealed class Gemma4MtpLayerWeights
{
    /// <summary><c>blk.{n}.attn_norm.weight</c> [drafterHidden]. Pre-attention RMSNorm.</summary>
    public required float[] AttnNormWeight { get; init; }

    /// <summary><c>blk.{n}.attn_q.weight</c> [drafterHidden, numHeads * headDim]. Quantized.</summary>
    public required nint AttnQWeight { get; init; }
    public required QuantizationType AttnQQuantType { get; init; }
    public required int AttnQOutputDim { get; init; }

    /// <summary><c>blk.{n}.attn_q_norm.weight</c> [numHeads * headDim]. Per-head Q RMSNorm.</summary>
    public required float[] AttnQNormWeight { get; init; }

    /// <summary><c>blk.{n}.attn_output.weight</c> [numHeads * headDim, drafterHidden]. Quantized.</summary>
    public required nint AttnOutputWeight { get; init; }
    public required QuantizationType AttnOutputQuantType { get; init; }

    /// <summary><c>blk.{n}.post_attention_norm.weight</c> [drafterHidden]. Post-attention RMSNorm.</summary>
    public required float[] PostAttentionNormWeight { get; init; }

    /// <summary><c>blk.{n}.ffn_norm.weight</c> [drafterHidden]. Pre-FFN RMSNorm.</summary>
    public required float[] FfnNormWeight { get; init; }

    /// <summary><c>blk.{n}.ffn_gate.weight</c> [drafterHidden, intermediateSize]. Quantized.</summary>
    public required nint FfnGateWeight { get; init; }
    public required QuantizationType FfnGateQuantType { get; init; }

    /// <summary><c>blk.{n}.ffn_up.weight</c> [drafterHidden, intermediateSize]. Quantized.</summary>
    public required nint FfnUpWeight { get; init; }
    public required QuantizationType FfnUpQuantType { get; init; }

    /// <summary><c>blk.{n}.ffn_down.weight</c> [intermediateSize, drafterHidden]. Quantized.</summary>
    public required nint FfnDownWeight { get; init; }
    public required QuantizationType FfnDownQuantType { get; init; }

    /// <summary><c>blk.{n}.post_ffw_norm.weight</c> [drafterHidden]. Post-FFN RMSNorm.</summary>
    public required float[] PostFfwNormWeight { get; init; }

    /// <summary><c>blk.{n}.layer_output_scale.weight</c> [1]. Learned residual scale scalar.</summary>
    public required float LayerOutputScale { get; init; }

    /// <summary>Whether this layer uses sliding-window attention (vs full attention).</summary>
    public required bool IsSlidingWindow { get; init; }

    /// <summary>Head dimension for this layer's Q projection (256 for SWA, 512 for full).</summary>
    public required int HeadDim { get; init; }
}
