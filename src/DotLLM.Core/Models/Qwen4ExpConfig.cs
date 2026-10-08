namespace DotLLM.Core.Models;

/// <summary>
/// Qwen4-Exp (Qwen3.8-Flash-Next; llama.cpp GGUF arch <c>qwen4exp</c>, HF <c>qwen4_exp</c>) parameters that are
/// not expressible in the generic <see cref="ModelConfig"/> fields: the multi-stream gated residual
/// ("hyper-connection", HC), the QSA indexer (block-pooled sparse attention), the n-gram hash embedding ("PLE")
/// and the rope section split. Non-null iff <see cref="ModelConfig.Architecture"/> is
/// <see cref="DotLLM.Core.Configuration.Architecture.Qwen4Exp"/>.
/// </summary>
/// <remarks>
/// <para>
/// The Gated-DeltaNet geometry lives in <see cref="ModelConfig.GdnConfig"/>, the 512-expert softmax MoE with a
/// sigmoid-gated shared expert in <see cref="ModelConfig.Moe"/>, and the 3:1 GDN/attention layout in
/// <see cref="ModelConfig.HybridLayout"/> — all reused from the Qwen3.5/3.6 hybrid. The trailing MTP block (when the
/// file carries one) is counted by <see cref="ModelConfig.NextnPredictLayers"/> and excluded from
/// <see cref="ModelConfig.NumLayers"/>.
/// </para>
/// <para>
/// Every field is read under the exact key llama.cpp's loader reads (<c>src/models/qwen4exp.cpp</c>
/// <c>load_arch_hparams</c>; keys in <c>llama-arch.cpp</c>) and was checked against the real
/// <c>unsloth/Qwen3.8-Flash-Next-GGUF</c> UD-Q4_K_XL header. See <c>docs/MODEL_CONFIG.md</c>.
/// </para>
/// </remarks>
public sealed record Qwen4ExpConfig
{
    /// <summary>Number of parallel residual streams (<c>{arch}.hyper_connection.count</c>; 4 on the released model).</summary>
    public required int HyperConnectionCount { get; init; }

    /// <summary>Low-rank bottleneck of the HC mixers (<c>{arch}.hyper_connection.low_rank</c>; 320).</summary>
    public required int HyperConnectionLowRank { get; init; }

    /// <summary>QSA indexer query heads (<c>{arch}.attention.indexer.head_count</c>; 4).</summary>
    public required int IndexerHeadCount { get; init; }

    /// <summary>QSA indexer per-head key/query width (<c>{arch}.attention.indexer.key_length</c>; 128).</summary>
    public required int IndexerKeyLength { get; init; }

    /// <summary>
    /// QSA token budget (<c>{arch}.attention.indexer.top_k</c>; 2048). This is a count of TOKENS, i.e.
    /// <c>IndexerTopK / IndexerBlockSize</c> pooled blocks are selected; below the budget QSA equals dense attention.
    /// </summary>
    public required int IndexerTopK { get; init; }

    /// <summary>
    /// Per-block compress ratio (<c>{arch}.attention.compress_ratios</c>), one entry per block INCLUDING the MTP block:
    /// 0 for a linear-attention (GDN) layer, the shared pooling ratio (4) for a QSA layer. Empty when the key is absent.
    /// </summary>
    public required IReadOnlyList<int> CompressRatios { get; init; }

    /// <summary>
    /// The single pooling block size shared by every QSA layer (llama.cpp <c>indexer_kpool</c>): the common non-zero value in
    /// <see cref="CompressRatios"/>, or 0 if no layer has one. Always &gt; 1 and a divisor of <see cref="IndexerTopK"/> when non-zero.
    /// </summary>
    public required int IndexerBlockSize { get; init; }

    /// <summary>
    /// Multimodal rope section split (<c>{arch}.rope.dimension_sections</c>; <c>[11, 11, 10, 0]</c>, interleaved
    /// IMROPE). Text-only inference collapses to plain rope; kept for the vision path.
    /// </summary>
    public required IReadOnlyList<int> RopeSections { get; init; }

    /// <summary>The n-gram hash embedding branch, or null when the checkpoint has none.</summary>
    public Qwen4ExpPleConfig? Ple { get; init; }

    /// <summary>
    /// The refusal text the GPU backends use while their forward pass is unimplemented (the CPU backend has the reference
    /// forward), so a user never reaches the misleading "blk.0.attn_output.weight not present" of a generic loader.
    /// </summary>
    /// <param name="backend">Backend name for the message ("CPU", "Vulkan", "CUDA").</param>
    public static string UnsupportedMessage(string backend) =>
        $"Architecture Qwen4Exp (GGUF 'qwen4exp', Qwen3.8-Flash-Next) is recognised — its config and metadata parse — but its forward pass " +
        $"is not implemented on the {backend} backend yet (4-stream gated residual, QSA indexer attention and the n-gram embedding have no " +
        "kernels there; the CPU backend has the reference forward). Tracked in the Qwen4Exp epic, issue #814.";
}

/// <summary>
/// The n-gram hash embedding ("PLE") branch of <see cref="Qwen4ExpConfig"/>. Row indices depend only on token ids:
/// <c>mixed = (t[p]*m0) XOR (t[p-1]*m1) [XOR (t[p-2]*m2)]</c> in wrapping int64, then
/// <c>row_h = mixed mod HeadVocabSizes[h] + HeadOffsets[h]</c>. The multipliers, offsets and vocab sizes are stored
/// as <b>exact uint64</b> (the converter reads them bypassing float casting) — never round-trip them through
/// <see cref="double"/> or <see cref="float"/>.
/// </summary>
public sealed record Qwen4ExpPleConfig
{
    /// <summary>Zero-based block indices that carry a PLE module (<c>{arch}.ple.layers</c>; <c>[1]</c>). Always linear-attention layers.</summary>
    public required IReadOnlyList<int> Layers { get; init; }

    /// <summary>N-gram order (<c>{arch}.ple.ngram_size</c>; 3).</summary>
    public required int NgramSize { get; init; }

    /// <summary>Hash heads per n-gram order (<c>{arch}.ple.heads_per_ngram</c>; 8).</summary>
    public required int HeadsPerNgram { get; init; }

    /// <summary>Depthwise conv kernel of the PLE gate path (<c>{arch}.ple.conv_kernel</c>; 4).</summary>
    public required int ConvKernel { get; init; }

    /// <summary>Token id that resets the n-gram history (<c>{arch}.ple.eos_token_id</c>).</summary>
    public required int EosTokenId { get; init; }

    /// <summary>
    /// Stand-in token id used for image tokens (<c>{arch}.ple.image_token_id</c>). Null when the key is absent, in which
    /// case llama.cpp falls back to <see cref="EosTokenId"/> (files written before the key existed).
    /// </summary>
    public int? ImageTokenId { get; init; }

    /// <summary>Width of one hash-table row (<c>{arch}.embedding_length_per_layer_input</c>; 160).</summary>
    public required int RowDim { get; init; }

    /// <summary>Per-order hash multipliers (<c>{arch}.ple.layer_multipliers</c>), exact uint64; at least <see cref="NgramSize"/> entries.</summary>
    public required IReadOnlyList<ulong> LayerMultipliers { get; init; }

    /// <summary>First table row of each head's range (<c>{arch}.ple.head_offsets</c>), exact uint64; at least <see cref="NumHeads"/> entries.</summary>
    public required IReadOnlyList<ulong> HeadOffsets { get; init; }

    /// <summary>Row count (hash modulus) of each head's range (<c>{arch}.ple.head_vocab_sizes</c>), exact uint64; at least <see cref="NumHeads"/> entries.</summary>
    public required IReadOnlyList<ulong> HeadVocabSizes { get; init; }

    /// <summary>Total hash heads: <c>(NgramSize - 1) * HeadsPerNgram</c> (order 1 is the plain token embedding; 16 on the released model).</summary>
    public int NumHeads => (NgramSize - 1) * HeadsPerNgram;

    /// <summary>Minimum row count the <c>per_layer_token_embd.weight</c> table must have: <c>max(offset + vocab)</c> over the heads.</summary>
    public ulong MinTableRows
    {
        get
        {
            ulong rows = 0;
            for (int h = 0; h < NumHeads; h++)
                rows = Math.Max(rows, checked(HeadOffsets[h] + HeadVocabSizes[h]));
            return rows;
        }
    }
}
