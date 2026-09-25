namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Gemma-3n Laurel block (Learned Augmented Residual Layer) CPU kernel: a
/// low-rank bottleneck applied in parallel with attention on the
/// post-<c>input_layernorm</c> hidden state. Verified against HF
/// <c>transformers</c> <c>modeling_gemma3n.py</c>
/// <c>Gemma3nTextLaurelBlock.forward</c>:
/// <code>
/// laurel = linear_right(linear_left(x))
/// laurel = post_laurel_norm(laurel)
/// return x + laurel
/// </code>
/// The caller (<c>Gemma3nTextDecoderLayer.forward</c>) does NOT use this
/// method's residual-added return directly — it takes only the
/// <c>post_laurel_norm</c> output (<see cref="ComputeDelta"/>) and combines it
/// with the attention residual via <c>(attn_gated + laurel_output) /
/// sqrt(2)</c>, so this kernel exposes the un-added delta rather than
/// replicating the (unused, in that call site) internal residual add.
/// </summary>
public static unsafe class Gemma3nLaurel
{
    /// <summary>
    /// Computes the Laurel delta (<c>post_laurel_norm(linear_right(linear_left(x)))</c>,
    /// WITHOUT the residual add) for every token.
    /// </summary>
    /// <param name="x">Post-<c>input_layernorm</c> hidden state (the same
    /// normalised value fed to attention Q/K/V), <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="linearLeftWeight"><c>laurel.linear_left.weight</c>
    /// <c>[laurelRank, hiddenSize]</c> row-major F32 (no bias).</param>
    /// <param name="linearRightWeight"><c>laurel.linear_right.weight</c>
    /// <c>[hiddenSize, laurelRank]</c> row-major F32 (no bias).</param>
    /// <param name="postLaurelNormWeight"><c>laurel.post_laurel_norm.weight</c>
    /// <c>[hiddenSize]</c> ((1+w) already absorbed).</param>
    /// <param name="rankScratch">Scratch <c>[seqLen, laurelRank]</c>.</param>
    /// <param name="output">Destination <c>[seqLen, hiddenSize]</c>. May alias
    /// nothing else live (written after both matmuls).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="laurelRank">Laurel low-rank bottleneck width.</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    public static void ComputeDelta(
        float* x, float* linearLeftWeight, float* linearRightWeight,
        ReadOnlySpan<float> postLaurelNormWeight,
        float* rankScratch, float* output,
        int seqLen, int hiddenSize, int laurelRank, float eps)
    {
        // rank = x · linear_left^T -> [seqLen, laurelRank]
        MatMul.GemmF32(linearLeftWeight, x, rankScratch, laurelRank, hiddenSize, seqLen);
        // output = rank · linear_right^T -> [seqLen, hiddenSize]
        MatMul.GemmF32(linearRightWeight, rankScratch, output, hiddenSize, laurelRank, seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(output + (long)t * hiddenSize, hiddenSize);
            RmsNorm.Execute(row, postLaurelNormWeight, eps, row);
        }
    }
}
