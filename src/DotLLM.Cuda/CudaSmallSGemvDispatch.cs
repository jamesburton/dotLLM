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
/// <c>1</c> restores the pre-#482 dispatch). <c>DOTLLM_CUDA_PQ2_0_S1_MULTI=1</c> additionally routes
/// single-token decode through the S=1 instantiation of the same kernel — an A/B switch only; the
/// shipped decode path stays on <c>pq2_0_gemv_f32io</c> until a measurement says otherwise.
/// </para>
/// </remarks>
public static class CudaSmallSGemvDispatch
{
    /// <summary>Largest token count sent to the multi-column GEMV by default.</summary>
    public const int DefaultMaxColumns = CudaKernels.Pq2_0GemvMultiMaxColumns;

    /// <summary>Environment variable overriding <see cref="DefaultMaxColumns"/>; <c>0</c> disables the path.</summary>
    public const string MaxColumnsEnvVar = "DOTLLM_CUDA_SMALL_S_MAX";

    /// <summary>Environment variable that, when <c>1</c>, routes seqLen == 1 PQ2_0 decode through the multi-column kernel's S=1 variant.</summary>
    public const string SingleColumnEnvVar = "DOTLLM_CUDA_PQ2_0_S1_MULTI";

    private static readonly int MaxColumnsFromEnv = ReadMaxColumns();
    private static readonly bool SingleColumnFromEnv =
        Environment.GetEnvironmentVariable(SingleColumnEnvVar) == "1";

    /// <summary>
    /// Effective multi-column range: <c>2 &lt;= seqLen &lt;= </c> this goes to the multi-column
    /// kernel. <c>0</c> or <c>1</c> means the path is off.
    /// </summary>
    public static int MaxColumns => MaxColumnsOverride ?? MaxColumnsFromEnv;

    /// <summary>Whether single-token PQ2_0 decode uses the multi-column kernel's S=1 variant (A/B only).</summary>
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
    /// Environment variable that, when <c>1</c>, routes PQ2_0 projections of <c>seqLen == 1</c> and
    /// <c>2 &lt;= seqLen &lt;= </c><see cref="MaxColumns"/> through the int8-activation dp4a GEMV
    /// (issue #485) instead of the #482 / single-column F16-activation kernels. Opt-in until measured.
    /// </summary>
    public const string Dp4aEnvVar = "DOTLLM_CUDA_PQ2_0_DP4A";

    private static readonly bool Dp4aFromEnv = Environment.GetEnvironmentVariable(Dp4aEnvVar) == "1";

    /// <summary>Whether the dp4a (W2A8) PQ2_0 GEMV is enabled (<see cref="Dp4aEnvVar"/>).</summary>
    public static bool UseDp4a => Dp4aOverride ?? Dp4aFromEnv;

    /// <summary>In-process override of <see cref="UseDp4a"/> (tests/benches only; read per projection).</summary>
    internal static bool? Dp4aOverride { get; set; }

    /// <summary>
    /// Whether a PQ2_0 projection over <paramref name="seqLen"/> token rows should take the dp4a
    /// GEMV (given that its kernels are loaded): <c>seqLen == 1</c>, or the multi-column range
    /// <c>2..</c><see cref="MaxColumns"/>, when <see cref="UseDp4a"/> is on.
    /// </summary>
    /// <param name="seqLen">Token rows in the projection.</param>
    public static bool CoversDp4a(int seqLen)
        => UseDp4a && (seqLen == 1 || (seqLen >= 2 && seqLen <= MaxColumns));

    private static int ReadMaxColumns()
        => int.TryParse(Environment.GetEnvironmentVariable(MaxColumnsEnvVar),
               NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0
           ? Math.Min(v, CudaKernels.Pq2_0GemvMultiMaxColumns)
           : DefaultMaxColumns;
}
