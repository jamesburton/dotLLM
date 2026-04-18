using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Rank-<c>R</c> projection helpers used by Mamba-3's MIMO (multi-input
/// multi-output) SSM variant (Lahoti et al., ICLR 2026, Appendix D).
/// The input-side <see cref="ExpandInput"/> expands <c>x</c> into a rank-R
/// factorization before the scan; the output-side <see cref="ContractOutput"/>
/// contracts the rank axis away after the scan.
/// </summary>
/// <remarks>
/// <para>
/// <b>Shape conventions (matched to <c>VikramKarLex/mamba3-minimal</c>).</b>
/// In the reference, MIMO factor tensors are stored <b>per-head-per-channel</b>:
/// </para>
/// <list type="bullet">
///   <item><description><c>mimo_x_proj</c>: <c>(n_head, head_dim, rank)</c></description></item>
///   <item><description><c>mimo_down</c>:   <c>(n_head, head_dim, rank)</c></description></item>
///   <item><description><c>x</c> entering the SSM: <c>(T, n_head, head_dim)</c></description></item>
///   <item><description><c>x_mimo</c> after expansion: <c>(T, n_head, head_dim, rank)</c></description></item>
/// </list>
/// <para>
/// The task brief suggested a flatter <c>[R, d_inner]</c> / <c>[R]</c> layout;
/// we follow the reference's three-axis tensors because that is what
/// <c>state-spaces/mamba</c> and the endorsed minimal port actually use —
/// the per-element rank factor is not uniform over the head or the channel
/// axis.
/// </para>
/// <para>
/// <b>Forward expansion</b> (pre-scan) is a pure outer-product broadcast:
/// </para>
/// <code>
/// x_mimo[t, h, p, r] = x[t, h, p] * mimo_x_proj[h, p, r]
/// </code>
/// <para>
/// <b>Backward contraction</b> (post-scan) is a rank-R reduction:
/// </para>
/// <code>
/// y[t, h, p] = sum_{r=0..R-1} y_mimo[t, h, p, r] * mimo_down[h, p, r]
/// </code>
/// <para>
/// <b>Alias safety.</b> Input and output spans have different shapes (the
/// rank-<c>R</c> axis is added / removed), so in-place aliasing is
/// structurally impossible — the output has <c>R×</c> the length of the
/// input for <see cref="ExpandInput"/>, and the reverse for
/// <see cref="ContractOutput"/>. Callers must still supply distinct buffers.
/// </para>
/// </remarks>
public static class Mamba3MimoProject
{
    /// <summary>
    /// Rank-<c>R</c> input expansion: elementwise outer product of <c>x</c> and
    /// <c>mimo_x_proj</c> along the rank axis.
    /// </summary>
    /// <param name="x">
    /// SSM input, shape <c>[T, n_head, head_dim]</c> row-major, length
    /// <c>T * n_head * head_dim</c>.
    /// </param>
    /// <param name="mimoXProj">
    /// Per-head-per-channel rank weights, shape <c>[n_head, head_dim, rank]</c>
    /// row-major, length <c>n_head * head_dim * rank</c>.
    /// </param>
    /// <param name="xMimo">
    /// Destination, shape <c>[T, n_head, head_dim, rank]</c> row-major, length
    /// <c>T * n_head * head_dim * rank</c>. Written.
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head (<c>d_inner / n_head</c>).</param>
    /// <param name="rank">MIMO rank (typically 2–4).</param>
    [SkipLocalsInit]
    public static void ExpandInput(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> mimoXProj,
        Span<float> xMimo,
        int seqLen,
        int nHead,
        int headDim,
        int rank)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        long xLen = (long)seqLen * nHead * headDim;
        long projLen = (long)nHead * headDim * rank;
        long outLen = (long)seqLen * nHead * headDim * rank;

        if (x.Length < xLen)
            throw new ArgumentException($"x length {x.Length} < T*n_head*head_dim = {xLen}.", nameof(x));
        if (mimoXProj.Length < projLen)
            throw new ArgumentException(
                $"mimoXProj length {mimoXProj.Length} < n_head*head_dim*rank = {projLen}.",
                nameof(mimoXProj));
        if (xMimo.Length < outLen)
            throw new ArgumentException(
                $"xMimo length {xMimo.Length} < T*n_head*head_dim*rank = {outLen}.",
                nameof(xMimo));

        if (seqLen == 0) return;

        ExpandInputScalar(x, mimoXProj, xMimo, seqLen, nHead, headDim, rank);
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="ExpandInput"/>. Kept
    /// <c>internal</c> for unit-test pinning against future SIMD variants.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExpandInputScalar(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> mimoXProj,
        Span<float> xMimo,
        int seqLen,
        int nHead,
        int headDim,
        int rank)
    {
        // Layout reminder (row-major, innermost rightmost):
        //   x[t, h, p]         = x[((t*nHead + h)*headDim + p)]
        //   mimo[h, p, r]      = mimoXProj[((h)*headDim + p)*rank + r]
        //   xMimo[t, h, p, r]  = xMimo[(((t*nHead + h)*headDim + p)*rank + r)]
        // The per-(h, p) rank-vector is contiguous in both the weight and the
        // output, so pulling each (h, p) slice of mimoXProj and scaling it by
        // x[t, h, p] gives a contiguous write.
        int hpCount = nHead * headDim; // number of (h, p) pairs

        for (int t = 0; t < seqLen; t++)
        {
            int xTokBase = t * hpCount;
            int outTokBase = xTokBase * rank;

            for (int hp = 0; hp < hpCount; hp++)
            {
                float scale = x[xTokBase + hp];
                int weightBase = hp * rank;
                int outBase = outTokBase + hp * rank;

                // Inner rank loop. R is typically 2–4, so a plain scalar loop
                // is at least as good as a vectorized one after JIT unrolling.
                for (int r = 0; r < rank; r++)
                {
                    xMimo[outBase + r] = scale * mimoXProj[weightBase + r];
                }
            }
        }
    }

    /// <summary>
    /// Rank-<c>R</c> output contraction: per-(t, h, p), weighted sum over the
    /// rank axis using <c>mimo_down</c>.
    /// </summary>
    /// <param name="yMimo">
    /// Rank-expanded SSM output, shape <c>[T, n_head, head_dim, rank]</c>
    /// row-major, length <c>T * n_head * head_dim * rank</c>.
    /// </param>
    /// <param name="mimoDown">
    /// Per-head-per-channel down-projection weights, shape
    /// <c>[n_head, head_dim, rank]</c> row-major, length
    /// <c>n_head * head_dim * rank</c>.
    /// </param>
    /// <param name="y">
    /// Destination, shape <c>[T, n_head, head_dim]</c> row-major, length
    /// <c>T * n_head * head_dim</c>. Written (not accumulated).
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="rank">MIMO rank (typically 2–4).</param>
    [SkipLocalsInit]
    public static void ContractOutput(
        ReadOnlySpan<float> yMimo,
        ReadOnlySpan<float> mimoDown,
        Span<float> y,
        int seqLen,
        int nHead,
        int headDim,
        int rank)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        long inLen = (long)seqLen * nHead * headDim * rank;
        long downLen = (long)nHead * headDim * rank;
        long yLen = (long)seqLen * nHead * headDim;

        if (yMimo.Length < inLen)
            throw new ArgumentException(
                $"yMimo length {yMimo.Length} < T*n_head*head_dim*rank = {inLen}.",
                nameof(yMimo));
        if (mimoDown.Length < downLen)
            throw new ArgumentException(
                $"mimoDown length {mimoDown.Length} < n_head*head_dim*rank = {downLen}.",
                nameof(mimoDown));
        if (y.Length < yLen)
            throw new ArgumentException(
                $"y length {y.Length} < T*n_head*head_dim = {yLen}.", nameof(y));

        if (seqLen == 0) return;

        ContractOutputScalar(yMimo, mimoDown, y, seqLen, nHead, headDim, rank);
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="ContractOutput"/>. Kept
    /// <c>internal</c> for unit-test pinning.
    /// </summary>
    [SkipLocalsInit]
    internal static void ContractOutputScalar(
        ReadOnlySpan<float> yMimo,
        ReadOnlySpan<float> mimoDown,
        Span<float> y,
        int seqLen,
        int nHead,
        int headDim,
        int rank)
    {
        int hpCount = nHead * headDim;

        for (int t = 0; t < seqLen; t++)
        {
            int yTokBase = t * hpCount;
            int inTokBase = yTokBase * rank;

            for (int hp = 0; hp < hpCount; hp++)
            {
                int weightBase = hp * rank;
                int inBase = inTokBase + hp * rank;

                float sum = 0f;
                for (int r = 0; r < rank; r++)
                {
                    sum += yMimo[inBase + r] * mimoDown[weightBase + r];
                }
                y[yTokBase + hp] = sum;
            }
        }
    }
}
