using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Weights for a Multi-Token Prediction (MTP / "NextN") head — the trailing extra decoder block
/// used for self-speculative decoding (issue #253). Confirmed against llama.cpp PR
/// ggml-org/llama.cpp#22673 (<c>src/models/qwen35.cpp</c>'s <c>graph_mtp</c> constructor and
/// <c>load_block_mtp</c> loader): the MTP block for Qwen3.5/3.6 is structurally a full-attention
/// Qwen3HybridDense decoder layer — same <c>attn_norm</c>/<c>post_attention_norm</c>, gated QKV
/// (<see cref="Qwen3HybridDenseLayerWeights.FullAttn"/>), and dense SwiGLU FFN as any other
/// full-attention trunk layer — with four extra "nextn" tensors wrapped around it:
/// <list type="bullet">
///   <item><description><c>blk.{n}.nextn.hnorm</c> — RMSNorm applied to the trunk's incoming hidden state.</description></item>
///   <item><description><c>blk.{n}.nextn.enorm</c> — RMSNorm applied to the embedding of the token being predicted from.</description></item>
///   <item><description><c>blk.{n}.nextn.eh_proj</c> — projects <c>concat(enorm(embed), hnorm(hidden))</c> (2·hiddenSize) back down to hiddenSize.</description></item>
///   <item><description><c>blk.{n}.nextn.shared_head_norm</c> / <c>blk.{n}.nextn.shared_head_head</c> — optional
///     head-local final-norm + LM-head; when absent the model's own <c>output_norm</c>/<c>output.weight</c> (or tied
///     <c>token_embd.weight</c>) are reused instead (llama.cpp: <c>head_norm_w = shared_head_norm ?: output_norm</c>).</description></item>
///   <item><description><c>blk.{n}.nextn.embed_tokens</c> — optional MTP-local token embedding; when absent the
///     model's own <c>token_embd.weight</c> is reused.</description></item>
/// </list>
/// where <c>n</c> is the raw GGUF block index of the MTP layer — <c>trunkLayerCount</c> (0-based),
/// since llama.cpp's <c>convert_hf_to_gguf.py</c> sets <c>block_count = num_hidden_layers +
/// mtp_num_hidden_layers</c> and appends the MTP block(s) after the trunk.
/// </summary>
internal sealed class MtpHeadWeights
{
    /// <summary>
    /// The MTP block's own decoder-layer weights (attn norms, gated full attention, dense FFN) —
    /// structurally identical to any other full-attention <see cref="Qwen3HybridDenseLayerWeights"/>.
    /// </summary>
    public required Qwen3HybridDenseLayerWeights Layer { get; init; }

    /// <summary><c>nextn.eh_proj.weight</c> [2·hiddenSize, hiddenSize]. Quantized.</summary>
    public required nint EhProjWeight { get; init; }
    public required QuantizationType EhProjQuantType { get; init; }
    public required int EhProjInputDim { get; init; }
    public required int EhProjOutputDim { get; init; }

    /// <summary><c>nextn.enorm.weight</c> [hiddenSize] — RMSNorm applied to the predicted token's embedding.</summary>
    public required float[] EnormWeight { get; init; }

    /// <summary><c>nextn.hnorm.weight</c> [hiddenSize] — RMSNorm applied to the incoming trunk hidden state.</summary>
    public required float[] HnormWeight { get; init; }

    /// <summary>Optional <c>nextn.embed_tokens.weight</c> [hiddenSize, vocabSize]. Null ⇒ reuse the trunk's <c>token_embd.weight</c>.</summary>
    public nint? EmbedTokensWeight { get; init; }
    public QuantizationType EmbedTokensQuantType { get; init; }

    /// <summary>Optional <c>nextn.shared_head_head.weight</c> [hiddenSize, vocabSize]. Null ⇒ reuse the trunk's LM head.</summary>
    public nint? SharedHeadHeadWeight { get; init; }
    public QuantizationType SharedHeadHeadQuantType { get; init; }

    /// <summary>Optional <c>nextn.shared_head_norm.weight</c> [hiddenSize]. Null ⇒ reuse the trunk's <c>output_norm.weight</c>.</summary>
    public float[]? SharedHeadNormWeight { get; init; }
}
