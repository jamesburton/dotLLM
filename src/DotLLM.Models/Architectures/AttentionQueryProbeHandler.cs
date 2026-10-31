using DotLLM.Core.Attention;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Callback for <see cref="TransformerModel.AttentionQueryProbe"/>: one call per layer of a cached F32 forward, after the
/// layer's K/V rows were written to <paramref name="kvCache"/> and attention ran. Read-only: the spans alias the model's
/// scratch and are valid only during the call.
/// </summary>
/// <param name="layer">Global layer index.</param>
/// <param name="q">Post-RoPE queries, <c>[seqLen, numHeads × headDim]</c>.</param>
/// <param name="attnOut">Attention output before the output projection, <c>[seqLen, numHeads × headDim]</c>.</param>
/// <param name="positions">Absolute position of each query row.</param>
/// <param name="numHeads">Query heads.</param>
/// <param name="numKvHeads">KV heads (GQA groups of <c>numHeads / numKvHeads</c>).</param>
/// <param name="headDim">Head dimension.</param>
/// <param name="scale">Attention score scale the layer used.</param>
/// <param name="kvCache">The cache the layer attended over (rows <c>[0, CurrentLength)</c>).</param>
internal delegate void AttentionQueryProbeHandler(int layer, ReadOnlySpan<float> q, ReadOnlySpan<float> attnOut,
    ReadOnlySpan<int> positions, int numHeads, int numKvHeads, int headDim, float scale, IKvCache kvCache);
