namespace DotLLM.Cuda;

/// <summary>
/// G-flash long-context prefill-attention path: a single hand-fused mma.sync
/// flash-attention kernel that computes QK, the causal softmax, and PV entirely in
/// shared memory / registers, never materialising the <c>numHeads · s²</c> scores to
/// global memory (unlike the <see cref="CudaG3Attention"/> cuBLAS+softmax path, which
/// round-trips them through HBM). Because it sweeps each query tile's KV axis only up to
/// its own diagonal, it also does roughly half the matmul FLOPs (the causal triangle).
/// </summary>
/// <remarks>
/// <para>
/// <b>Where it sits in the dispatch.</b> The tuned kernel (GQA-group K/V sharing, one warp
/// per query head, <c>group</c> warps/block) beats G3 across the whole measured range on an
/// RTX 3060 (CC 8.6): end-to-end attention-category G3/flash 2.14× @ s=512, 1.83× @ 1024,
/// 1.81× @ 2048, and kernel-level 1.94× @ 4096 — combining the causal-triangle FLOP halving
/// with score fusion (no <c>numHeads · s²</c> HBM round-trip). The crossover is set to 512,
/// the lowest length with a clean measurement; below that, per-block launch overhead may
/// favour G3, so shorter prefills keep G3. See <see cref="CrossoverSeqLen"/> for the
/// threshold and its override.
/// </para>
/// <para>
/// <b>Eligibility (see <see cref="CanUse"/>).</b> The kernel is a prototype for the
/// Llama-3.2-1B head shape: <c>headDim == 64</c>, causal, <c>positionOffset == 0</c>,
/// FP16 in/out, square scores (<c>seqKv == seqLen</c>), global attention
/// (<c>slidingWindow ≤ 0</c>), and GQA that divides evenly. Anything outside that — most
/// importantly <b>any model whose headDim is not 64</b> — falls through to G3, then to
/// <c>attention_f16</c>. The default is gated to GeForce Ampere; the kernel emits
/// Ampere-only mma.sync PTX (built at compute_86) and its PTX never loads on Turing.
/// </para>
/// </remarks>
internal sealed class CudaFlashAttention
{
    /// <summary>Head dimension the prototype kernel is specialised for.</summary>
    private const int SupportedHeadDim = 64;

    /// <summary>
    /// Max GQA group (query heads per kv head) the tuned kernel supports: it launches
    /// <c>group</c> warps per block (<c>group · 32</c> threads) and statically sizes per-warp
    /// shared memory for this many warps. Larger groups fall through to G3. Llama-3.2-1B is
    /// group 4.
    /// <para>
    /// <b>MUST equal <c>MAX_GROUP_WARPS</c> in <c>native/kernels/attention_flash_mma.cu</c>.</b>
    /// The kernel is compiled <c>__launch_bounds__(MAX_GROUP_WARPS · 32)</c> and indexes its
    /// per-warp shared arrays by warp id; if this cap exceeds the kernel's, an admitted
    /// large-group call launches more threads than launch-bounds allows and indexes shared
    /// memory out of bounds. Keep the two constants in lockstep.
    /// </para>
    /// </summary>
    private const int MaxGroupWarps = 4;

    /// <summary>
    /// Default sequence-length crossover at/above which flash is dispatched ahead of G3.
    /// The tuned kernel (GQA-group K/V sharing, <c>group</c> warps/block) beats G3 across the
    /// whole measured range on a 3060 — end-to-end attention-category G3/flash 2.14× @ s=512,
    /// 1.83× @ 1024, 1.81× @ 2048; kernel-level 1.94× @ 4096 — so the crossover is the lowest
    /// length with a clean measurement (512). Below 512 there is no data and per-block launch
    /// overhead may favour G3, so 512 is kept as a conservative floor. Overridable via
    /// <c>DOTLLM_CUDA_FLASH_ATTN_MINSEQ</c>.
    /// </summary>
    private const int DefaultCrossoverSeqLen = 512;

    private static readonly string? FlashAttnEnv =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_ATTN");

    /// <summary>
    /// Effective toggle. Honours an explicit env override immediately ("1"/"0"); otherwise
    /// stays off until <see cref="ConfigureDefault"/> sets the device-gated default at model
    /// load. Mutable so a benchmark can interleave OFF/ON reps within one warmed process
    /// (consumer GPUs drift clocks ~2× across separate runs, so separate-process minima are
    /// not comparable).
    /// </summary>
    internal static bool Enabled = FlashAttnEnv == "1";

    /// <summary>
    /// Sequence-length crossover. Read once from <c>DOTLLM_CUDA_FLASH_ATTN_MINSEQ</c>
    /// (a positive integer) or the measured default.
    /// </summary>
    internal static readonly int CrossoverSeqLen =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_ATTN_MINSEQ"), out int v) && v > 0
            ? v
            : DefaultCrossoverSeqLen;

    /// <summary>
    /// Count of flash-kernel launches since process start (one per eligible attention call,
    /// i.e. per layer per forward). Lets a test prove the flash branch actually fired in the
    /// wired <c>Forward</c> path — without it an end-to-end "parity" pass is vacuous if
    /// <see cref="CanUse"/> silently fell through to G3 on both arms. Diagnostics only.
    /// </summary>
    internal static long DispatchCount;

    private readonly CudaKernels _kernels;

    internal CudaFlashAttention(CudaKernels kernels) => _kernels = kernels;

    /// <summary>
    /// Sets the flash-attention default from device eligibility, unless
    /// <c>DOTLLM_CUDA_FLASH_ATTN</c> overrides it ("1" force on, "0" force off). Gating: on
    /// for GeForce Ampere (the parts that benefit and whose driver JITs the sm_86 PTX);
    /// off elsewhere (Turing cannot run the mma.sync PTX, and other parts use G3/eager).
    /// </summary>
    /// <param name="deviceEligible">True when the device benefits — GeForce Ampere.</param>
    internal static void ConfigureDefault(bool deviceEligible) =>
        Enabled = FlashAttnEnv switch
        {
            "1" => true,
            "0" => false,
            _ => deviceEligible,
        };

    /// <summary>
    /// True when the G-flash kernel may be used for this attention call: the toggle is on,
    /// the kernel is loaded, the sequence is long enough to clear the G3 crossover, the head
    /// shape matches the prototype (<c>headDim == 64</c>, GQA divides), and the call is a
    /// pure square-causal global-attention prefill (no position offset, no prefix-cache
    /// reuse, no sliding window). Otherwise the caller falls through to G3 / eager.
    /// </summary>
    internal bool CanUse(int seqLen, int seqKv, int positionOffset, int slidingWindow,
                         int numHeads, int numKvHeads, int headDim)
        => Enabled
        && _kernels.HasAttentionFlashMma
        && headDim == SupportedHeadDim
        && seqLen >= CrossoverSeqLen
        && seqKv == seqLen
        && positionOffset == 0
        && slidingWindow <= 0
        && numKvHeads > 0
        && (numHeads % numKvHeads) == 0
        && (numHeads / numKvHeads) <= MaxGroupWarps;

    /// <summary>
    /// Runs the G-flash kernel for one prefill attention call, writing the result into
    /// <paramref name="output"/> in the row-major <c>[seq, numHeads, headDim]</c> layout
    /// (matching <c>attention_f16</c> and G3). Q/K/V are the same RoPE'd FP16 buffers the
    /// eager path consumes. Caller must have confirmed <see cref="CanUse"/>.
    /// </summary>
    internal void Run(nint q, nint k, nint v, nint output,
                      int seqLen, int numHeads, int numKvHeads, int headDim, nint stream)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        DispatchCount++;
        _kernels.LaunchAttentionFlashMma(q, k, v, output,
            seqLen, numHeads, numKvHeads, headDim, scale, stream);
    }
}
