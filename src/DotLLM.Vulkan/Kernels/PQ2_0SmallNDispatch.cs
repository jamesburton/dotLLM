using System.Globalization;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #446 — the PQ2_0 matmul small-<c>n</c> dispatch policy, shared by every Vulkan model
/// that carries PQ2_0 weights so the three call sites cannot drift apart.
/// </summary>
/// <remarks>
/// <para>
/// <b>The bug this fixes.</b> Every PQ2_0 dispatch site read
/// <c>seqLen == 1 ? GEMV : GEMM</c>, so a batch of two tokens went straight to a GEMM whose
/// N-tile is 128 wide. That tile does a fixed amount of PQ2_0 unpack per K-step regardless of how
/// few token columns are live (issue #440's RGP capture measured the shipping kernel VALU-bound
/// on exactly that unpack, with the matrix pipe 99.5% idle), so the GEMM is near-flat in
/// <c>n</c> while a per-token GEMV loop is linear. Below the crossing point the single dispatch
/// is simply the slower option.
/// </para>
/// <para>
/// <b>Where the crossing point is.</b> Measured on gfx1151 (Radeon 8060S) against the shipping
/// <c>matmul_pq2_0_f32_gemm_ladder_128x128x4</c> tile, interleaved and order-reversed, median of
/// per-pass ratios — GEMM-relative, so <c>&lt; 1.00x</c> means the loop wins:
/// </para>
/// <code>
/// n                 1      2      3      4      6      8     16     64
/// lm_head       0.24x  0.47x  0.68x  0.91x  1.36x  1.77x  3.37x 10.30x   -> crosses ~4.4
/// ffn_gate/up   0.17x  0.31x  0.47x  0.64x  0.94x  1.22x  2.33x  6.05x   -> crosses ~6.5
/// </code>
/// <para>
/// <see cref="DefaultGemvLoopMaxN"/> is 4: the lower of the two crossings, rounded down, so the
/// loop is never selected on a shape where the GEMM has already won. It is deliberately not 6 —
/// that would hand <c>lm_head</c> at <c>n = 6</c> to the loop, where the GEMM is 1.36x ahead.
/// </para>
/// <para>
/// <b>Escape hatch.</b> <c>DOTLLM_VK_PQ2_0_GEMV_LOOP_MAX_N</c> overrides the threshold;
/// <c>0</c> disables the loop entirely and restores the pre-#446 dispatch. Read once, on first
/// use — this sits on the record path and must not re-read the environment per matmul.
/// </para>
/// <para>
/// <b>Why the loop needs no barriers.</b> Token <c>t</c> reads <c>x[t·K ..]</c> and writes
/// <c>y[t·M ..]</c>; the ranges are disjoint across tokens and the weight buffer is read-only, so
/// there is no hazard between successive dispatches and they are free to overlap. The three bound
/// handles are identical for every token, so the handle-keyed descriptor cache hits and the whole
/// per-token cost is two push-constant words.
/// </para>
/// </remarks>
public static class PQ2_0SmallNDispatch
{
    /// <summary>
    /// Largest <c>n</c> for which the per-token GEMV loop is preferred to the batched GEMM.
    /// </summary>
    public const int DefaultGemvLoopMaxN = 4;

    /// <summary>Environment variable overriding <see cref="DefaultGemvLoopMaxN"/>; <c>0</c> disables the loop.</summary>
    public const string ThresholdEnvVar = "DOTLLM_VK_PQ2_0_GEMV_LOOP_MAX_N";

    private static readonly int Threshold = ReadThreshold();

    /// <summary>
    /// The effective threshold: the GEMV loop is recorded for <c>n &lt;= </c> this, the GEMM
    /// above it. <c>0</c> means the loop is disabled except at the unconditional <c>n == 1</c>
    /// decode shape.
    /// </summary>
    public static int GemvLoopMaxN => Threshold;

    private static int ReadThreshold()
        => int.TryParse(Environment.GetEnvironmentVariable(ThresholdEnvVar),
               NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0
           ? v
           : DefaultGemvLoopMaxN;

    /// <summary>
    /// Records <c>C[n, m] = B[n, k] @ W_pq2_0[m, k]^T</c> through whichever of the two kernels is
    /// faster at this <paramref name="n"/>.
    /// </summary>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="gemv">Decode-path PQ2_0 GEMV.</param>
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

        if (n == 1 || n <= Threshold)
        {
            for (int t = 0; t < n; t++)
                gemv.Record(cmdBuf, weights, input, output, m, k,
                    xOffsetElements: t * k, yOffsetElements: t * m);
            return;
        }

        gemm.Record(cmdBuf, weights, input, output, m, k, n);
    }
}
