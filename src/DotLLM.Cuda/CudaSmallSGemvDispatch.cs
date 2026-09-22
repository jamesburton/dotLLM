using System.Globalization;

namespace DotLLM.Cuda;

/// <summary>
/// Issue #482 — the CUDA small-<c>S</c> projection policy: which token counts go to the
/// multi-column PQ2_0 GEMV (<see cref="CudaKernels.LaunchPQ2_0GemvMulti"/>) instead of the
/// dequant-to-F16 + cuBLAS path. The CUDA twin of the Vulkan <c>PQ2_0SmallNDispatch</c> (#470).
/// </summary>
/// <remarks>
/// <para>
/// <b>The cliff this removes.</b> With <c>seqLen &gt; 1</c>, every PQ2_0 projection used to be
/// dequantized in full to F16 and handed to HGEMM, so a 2-token MTP verify on Bonsai 2 27B cost
/// 6.4x a decode step (about 330 ms of whole-model dequant per forward, RTX 3060).
/// <c>2 &lt;= seqLen &lt;= </c><see cref="MaxColumns"/> now reads the packed weights once for all
/// columns; above that the dequant+HGEMM prefill path is unchanged.
/// </para>
/// <para>
/// <b>Escape hatches</b>, read once on first use because this sits on the per-projection path:
/// <c>DOTLLM_CUDA_SMALL_S_MAX</c> caps the multi-column range (default 8, clamped to 8; <c>0</c> or
/// <c>1</c> restores the pre-#482 dispatch). Single-token decode also goes through the S=1
/// instantiation of the same kernel by default: measured on the T5500 (RTX 3060, Bonsai 2 27B,
/// same-session order-reversed A/B), plain decode went from 16.9 to 18.2 tok/s, and greedy output still
/// matches the CPU oracle. <c>DOTLLM_CUDA_PQ2_0_S1_MULTI=0</c> restores <c>pq2_0_gemv_f32io</c> and the
/// fused gate+up / K+V pairs.
/// </para>
/// <para>
/// <b>Issue #485.</b> The dp4a path is the default and takes precedence over both of the above for
/// <c>seqLen == 1</c> and <c>2..</c><see cref="MaxColumns"/>. The activations are quantized to int8
/// (the CPU W2A8 tier's per-32 Q8_0 rounding), and the GEMV runs on <c>__dp4a</c>
/// (<see cref="CudaKernels.LaunchPQ2_0GemvDp4a"/>). Measured on the T5500 (RTX 3060, Bonsai 2 27B,
/// same-session order-reversed A/B): plain decode went from 18.3–18.6 to 20.1 tok/s, and MTP K=3 from
/// 22.7–23.7 to 29.1–29.8 tok/s. Greedy output matches the CPU oracle, which itself runs W2A8 on
/// AVX2/SSSE3. This is a numerics change, the same one the CPU tiers make: <c>DOTLLM_CUDA_PQ2_0_DP4A=0</c>
/// restores the F16-activation kernels.
/// </para>
/// </remarks>
public static class CudaSmallSGemvDispatch
{
    /// <summary>Largest token count sent to the multi-column GEMV by default.</summary>
    public const int DefaultMaxColumns = CudaKernels.Pq2_0GemvMultiMaxColumns;

    /// <summary>Environment variable overriding <see cref="DefaultMaxColumns"/>; <c>0</c> disables the path.</summary>
    public const string MaxColumnsEnvVar = "DOTLLM_CUDA_SMALL_S_MAX";

    /// <summary>Environment variable that, when <c>0</c>, sends seqLen == 1 PQ2_0 decode back to the single-column kernel (default: multi-column S=1).</summary>
    public const string SingleColumnEnvVar = "DOTLLM_CUDA_PQ2_0_S1_MULTI";

    private static readonly int MaxColumnsFromEnv = ReadMaxColumns();
    private static readonly bool SingleColumnFromEnv =
        Environment.GetEnvironmentVariable(SingleColumnEnvVar) != "0";

    /// <summary>
    /// Effective multi-column range: <c>2 &lt;= seqLen &lt;= </c> this goes to the multi-column
    /// kernel. <c>0</c> or <c>1</c> means the path is off.
    /// </summary>
    public static int MaxColumns => MaxColumnsOverride ?? MaxColumnsFromEnv;

    /// <summary>Whether single-token PQ2_0 decode uses the multi-column kernel's S=1 variant (default on).</summary>
    public static bool UseMultiForSingleColumn => SingleColumnOverride ?? SingleColumnFromEnv;

    /// <summary>
    /// In-process override of <see cref="MaxColumns"/> so a test or bench can A/B against the
    /// pre-#482 dispatch in one session. Not for production use; read per projection, so changing
    /// it mid-forward mixes the two paths.
    /// </summary>
    internal static int? MaxColumnsOverride { get; set; }

    /// <summary>In-process override of <see cref="UseMultiForSingleColumn"/> (tests/benches only).</summary>
    internal static bool? SingleColumnOverride { get; set; }

    /// <summary>
    /// Whether a PQ2_0 projection over <paramref name="seqLen"/> token rows should take the
    /// multi-column kernel (given that the kernel is loaded).
    /// </summary>
    /// <param name="seqLen">Token rows in the projection.</param>
    public static bool Covers(int seqLen)
        => seqLen == 1 ? UseMultiForSingleColumn : seqLen >= 2 && seqLen <= MaxColumns;

    /// <summary>
    /// Environment variable that, when <c>0</c>, stops PQ2_0 projections of <c>seqLen == 1</c> and
    /// <c>2 &lt;= seqLen &lt;= </c><see cref="MaxColumns"/> from using the int8-activation dp4a GEMV
    /// (issue #485, the default) and sends them to the #482 / single-column F16-activation kernels.
    /// </summary>
    public const string Dp4aEnvVar = "DOTLLM_CUDA_PQ2_0_DP4A";

    private static readonly bool Dp4aFromEnv = Environment.GetEnvironmentVariable(Dp4aEnvVar) != "0";

    /// <summary>Whether the dp4a (W2A8) PQ2_0 GEMV is enabled (<see cref="Dp4aEnvVar"/>).</summary>
    public static bool UseDp4a => Dp4aOverride ?? Dp4aFromEnv;

    /// <summary>In-process override of <see cref="UseDp4a"/> (tests/benches only; read per projection).</summary>
    internal static bool? Dp4aOverride { get; set; }

    /// <summary>
    /// Whether projections reading the same input share one dp4a activation quantization (default
    /// true). Only <see cref="ShareDp4aInputOverride"/> turns it off — a test A/B whose results must be
    /// bit-identical, proving the shared scratch is never stale.
    /// </summary>
    public static bool ShareDp4aInputs => ShareDp4aInputOverride ?? true;

    /// <summary>In-process override of <see cref="ShareDp4aInputs"/> (tests only).</summary>
    internal static bool? ShareDp4aInputOverride { get; set; }

    /// <summary>
    /// Whether a PQ2_0 projection over <paramref name="seqLen"/> token rows should take the dp4a
    /// GEMV (given that its kernels are loaded): <c>seqLen == 1</c>, or the multi-column range
    /// <c>2..</c><see cref="MaxColumns"/>, when <see cref="UseDp4a"/> is on.
    /// </summary>
    /// <param name="seqLen">Token rows in the projection.</param>
    public static bool CoversDp4a(int seqLen)
        => UseDp4a && (seqLen == 1 || (seqLen >= 2 && seqLen <= MaxColumns));

    /// <summary>
    /// Environment variable that, when set to <c>1</c>, routes PQ2_0 projections WIDER than
    /// <see cref="MaxColumns"/> (prefill, and perplexity's whole-window forward) to the packed dp4a
    /// GEMM (issue #490) instead of dequantizing the matrix to F16 for cuBLAS. OPT-IN until measured:
    /// like #485 this is a numerics change (int8 activations, the CPU W2A8 tier), and unlike #485 it
    /// is on the path perplexity scores.
    /// </summary>
    public const string MmqEnvVar = "DOTLLM_CUDA_PQ2_0_MMQ";

    private static readonly bool MmqFromEnv = Environment.GetEnvironmentVariable(MmqEnvVar) == "1";

    /// <summary>Whether the packed PQ2_0 prefill GEMM is enabled (<see cref="MmqEnvVar"/>, default off).</summary>
    public static bool UseMmq => MmqOverride ?? MmqFromEnv;

    /// <summary>In-process override of <see cref="UseMmq"/> (tests/benches only; read per projection).</summary>
    internal static bool? MmqOverride { get; set; }

    /// <summary>
    /// Whether a PQ2_0 projection over <paramref name="seqLen"/> token rows should take the packed
    /// prefill GEMM (given that its kernels are loaded): every width the GEMV does not already cover.
    /// </summary>
    /// <remarks>
    /// The floor is <c>max(1, MaxColumns)</c>, not <see cref="MaxColumns"/>: with
    /// <c>DOTLLM_CUDA_SMALL_S_MAX=0</c> the GEMV range is off and a plain <c>&gt; MaxColumns</c> would
    /// send single-token decode to a GEMM tile that idles 31 of its 32 columns.
    /// </remarks>
    /// <param name="seqLen">Token rows in the projection.</param>
    public static bool CoversMmq(int seqLen)
        => UseMmq && seqLen > Math.Max(1, MaxColumns);

    /// <summary>
    /// Environment variable forcing the PQ2_0 MMQ column tile (<c>16</c> or <c>32</c>); unset picks
    /// <see cref="MmqTileColumns"/>'s width for the token count.
    /// </summary>
    public const string MmqTileEnvVar = "DOTLLM_CUDA_PQ2_0_MMQ_BN";

    private static readonly int MmqTileFromEnv = ReadMmqTile();

    /// <summary>In-process override of the MMQ column tile, <c>16</c> or <c>32</c> (tests/benches only).</summary>
    internal static int? MmqTileOverride { get; set; }

    /// <summary>
    /// Column tile for a <paramref name="seqLen"/>-row MMQ projection: the narrow (16-column, 256-row)
    /// instantiation at or below 16 tokens, where the 32-wide tile would idle half its lanes, and the
    /// wide (32-column, 128-row) one above. <see cref="MmqTileEnvVar"/> pins either.
    /// </summary>
    /// <param name="seqLen">Token rows in the projection.</param>
    public static int MmqTileColumns(int seqLen)
    {
        int forced = MmqTileOverride ?? MmqTileFromEnv;
        if (forced != 0) return forced;
        return seqLen <= CudaKernels.Pq2_0MmqTileColumnsNarrow
            ? CudaKernels.Pq2_0MmqTileColumnsNarrow
            : CudaKernels.Pq2_0MmqTileColumnsWide;
    }

    private static int ReadMmqTile()
        => int.TryParse(Environment.GetEnvironmentVariable(MmqTileEnvVar),
               NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
           && (v == CudaKernels.Pq2_0MmqTileColumnsNarrow || v == CudaKernels.Pq2_0MmqTileColumnsWide)
           ? v
           : 0;

    private static int ReadMaxColumns()
        => int.TryParse(Environment.GetEnvironmentVariable(MaxColumnsEnvVar),
               NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0
           ? Math.Min(v, CudaKernels.Pq2_0GemvMultiMaxColumns)
           : DefaultMaxColumns;
}
