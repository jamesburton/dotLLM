using System.Globalization;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #446 / #470 — the PQ2_0 matmul small-<c>n</c> dispatch policy, shared by every Vulkan
/// model that carries PQ2_0 weights so the three call sites cannot drift apart.
/// </summary>
/// <remarks>
/// <para>
/// <b>Three kernels, chosen by token count.</b> <c>n = 1</c> is the decode GEMV.
/// <c>2 &lt;= n &lt;= </c><see cref="MultiColumnMaxN"/> (default 8) is the multi-column GEMV
/// (#470), which decodes each weight byte once for all <c>n</c> columns. Above that is the
/// 128x128 coopmat GEMM. The per-token GEMV loop from #446 is now only a fallback, used when the
/// multi-column path is switched off.
/// </para>
/// <para>
/// <b>Why the GEMM loses at small n.</b> Its N-tile is 128 wide and does a full tile's PQ2_0
/// unpack per K-step however few columns are live (#440's RGP capture: VALU-bound on the unpack,
/// matrix pipe 99.5% idle), so it costs about 4-6 single-column GEMVs even at <c>n = 2</c>.
/// </para>
/// <para>
/// <b>Measured on gfx1151</b> (<c>Bench_PQ2_0MultiColumnGemv</c>, Bonsai 27B shapes, cost relative
/// to ONE single-column GEMV, 3 arms rotated each pass, medians):
/// </para>
/// <code>
///                        n=2    n=3    n=4    n=6    n=8
/// lm_head   multi-col  1.14x  1.34x  1.86x  2.68x  3.51x    loop 8.19x, GEMM 4.38x at n=8
///           GEMM       4.28x  4.35x  4.33x  4.32x  4.38x
/// ffn_down  multi-col  1.21x  1.60x  2.16x  3.32x  4.52x    loop 7.83x, GEMM 6.20x at n=8
/// attn_out  multi-col  1.18x  1.32x  1.58x  2.62x  5.08x    loop 6.76x, GEMM 5.45x at n=8
/// </code>
/// <para>
/// The multi-column kernel is below the GEMM at every <c>n &lt;= 8</c> on every shape measured,
/// and below the loop everywhere from <c>n = 2</c>. Its closest margin is <c>attn_output</c> at
/// <c>n = 8</c> (5.08x vs 5.45x). Past 8 the GEMM is flat while two multi-column passes would cost
/// more than 7x, so 8 is also where the GEMM should take over.
/// </para>
/// <para>
/// <b>Escape hatches</b>, read once on first use because this sits on the record path:
/// <c>DOTLLM_VK_PQ2_0_MULTICOL_MAX_N</c> caps the multi-column range (<c>0</c> disables it);
/// <c>DOTLLM_VK_PQ2_0_GEMV_LOOP_MAX_N</c> is the #446 loop threshold that applies beyond it
/// (default 4, <c>0</c> disables the loop). Both at <c>0</c> restores the pre-#446
/// <c>n == 1 ? GEMV : GEMM</c> dispatch.
/// </para>
/// <para>
/// <b>Why the loop needs no barriers.</b> Token <c>t</c> reads <c>x[t·K ..]</c> and writes
/// <c>y[t·M ..]</c>; the ranges are disjoint across tokens and the weight buffer is read-only, so
/// there is no hazard between successive dispatches and they are free to overlap.
/// </para>
/// </remarks>
public static class PQ2_0SmallNDispatch
{
    /// <summary>
    /// Largest <c>n</c> for which the per-token GEMV loop is preferred to the batched GEMM, when
    /// the multi-column kernel does not cover <c>n</c>.
    /// </summary>
    public const int DefaultGemvLoopMaxN = 4;

    /// <summary>Environment variable overriding <see cref="DefaultGemvLoopMaxN"/>; <c>0</c> disables the loop.</summary>
    public const string ThresholdEnvVar = "DOTLLM_VK_PQ2_0_GEMV_LOOP_MAX_N";

    /// <summary>Largest <c>n</c> sent to the multi-column GEMV by default.</summary>
    public const int DefaultMultiColumnMaxN = MatMulPQ2_0GemvF32Kernel.MaxColumns;

    /// <summary>Environment variable overriding <see cref="DefaultMultiColumnMaxN"/>; <c>0</c> disables the multi-column path.</summary>
    public const string MultiColumnEnvVar = "DOTLLM_VK_PQ2_0_MULTICOL_MAX_N";

    private static readonly int Threshold = ReadEnv(ThresholdEnvVar, DefaultGemvLoopMaxN, int.MaxValue);
    private static readonly int MultiColumnFromEnv =
        ReadEnv(MultiColumnEnvVar, DefaultMultiColumnMaxN, MatMulPQ2_0GemvF32Kernel.MaxColumns);

    /// <summary>
    /// The effective loop threshold: the GEMV loop is recorded for <c>n &lt;= </c> this when the
    /// multi-column kernel does not take <c>n</c>. <c>0</c> disables the loop.
    /// </summary>
    public static int GemvLoopMaxN => Threshold;

    /// <summary>
    /// The effective multi-column range: <c>2 &lt;= n &lt;= </c> this goes to
    /// <see cref="MatMulPQ2_0GemvF32Kernel.RecordColumns"/>. <c>0</c> or <c>1</c> disables it.
    /// </summary>
    public static int MultiColumnMaxN => MultiColumnMaxNOverride ?? MultiColumnFromEnv;

    /// <summary>
    /// In-process override of <see cref="MultiColumnMaxN"/>, so a benchmark can A/B the
    /// multi-column path against the pre-#470 dispatch in one session. Not for production use:
    /// it is read at record time, so changing it mid-forward mixes the two paths.
    /// </summary>
    internal static int? MultiColumnMaxNOverride { get; set; }

    private static int ReadEnv(string name, int fallback, int max)
        => int.TryParse(Environment.GetEnvironmentVariable(name),
               NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0
           ? Math.Min(v, max)
           : fallback;

    /// <summary>
    /// Records <c>C[n, m] = B[n, k] @ W_pq2_0[m, k]^T</c> through whichever kernel is fastest at
    /// this <paramref name="n"/>.
    /// </summary>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="gemv">Decode-path PQ2_0 GEMV, including its multi-column variants.</param>
    /// <param name="gemm">Prefill-path PQ2_0 GEMM.</param>
    /// <param name="weights">Packed PQ2_0 weights.</param>
    /// <param name="input">Activations, <c>[n, k]</c> row-major.</param>
    /// <param name="output">Output, <c>[n, m]</c> row-major.</param>
    /// <param name="m">Output rows per token.</param>
    /// <param name="k">Inner dimension.</param>
    /// <param name="n">Token count.</param>
    public static void Record(
        nint cmdBuf,
        MatMulPQ2_0GemvF32Kernel gemv, MatMulPQ2_0GemmF32Kernel gemm,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int m, int k, int n)
    {
        ArgumentNullException.ThrowIfNull(gemv);
        ArgumentNullException.ThrowIfNull(gemm);
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        if (n == 1)
        {
            gemv.Record(cmdBuf, weights, input, output, m, k);
            return;
        }

        if (n <= MultiColumnMaxN)
        {
            gemv.RecordColumns(cmdBuf, weights, input, output, m, k, n);
            return;
        }

        if (n <= Threshold)
        {
            for (int t = 0; t < n; t++)
                gemv.Record(cmdBuf, weights, input, output, m, k,
                    xOffsetElements: t * k, yOffsetElements: t * m);
            return;
        }

        gemm.Record(cmdBuf, weights, input, output, m, k, n);
    }
}
