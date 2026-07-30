namespace DotLLM.Cuda;

/// <summary>
/// Tensor-core (mma.sync) FP16 decode-attention gate/dispatch wrapper, composed with the
/// #197/#198 GQA-group + split-KV grid design — issue #199 v2.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why v2, not v1.</b> v1 (the <c>CudaAttentionMmaDecode</c> class, branch
/// <c>issue/199-tensor-core-decode-attention</c>, not merged, not present on this branch)
/// built a decode-only tensor-core
/// kernel with genuine HMMA/LDSM SASS but scoped to one warp per block, grid=numHeads — real
/// wall-clock A/B found it 4-5x SLOWER than the F32 baseline at every realistic Bonsai depth,
/// root-caused to ~4% occupancy (worse than the baseline's own already-diagnosed ~16.5%). This
/// wrapper drives <see cref="CudaKernels.LaunchAttentionMmaDecodeGqaSplit"/>, which packs the
/// GQA group into the mma tile's M dimension (free — same instruction count as v1, up to
/// MAX_GQA_GROUP=8x more useful throughput) and grids <c>(numKvHeads, kvSplit)</c> like
/// <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> — see
/// <c>attention_flash_mma_decode_gqa_split.cu</c>'s header for the full design.
/// </para>
/// <para>
/// <b>Scope (see <see cref="CanUse"/>).</b> Decode only (<c>seqQ == 1</c>), <c>headDim ==
/// 256</c> (the shape the kernel is compiled for), GQA that divides evenly with
/// <c>group &lt;= CudaKernels.MaxGqaGroup</c> (8 — required by the M-dim-packing scheme, not
/// just a register-array cap the way it is for the FP32 sibling kernel), no sliding window
/// (decode always attends to the full live KV prefix by construction — no masking logic in
/// this kernel, same as v1).
/// </para>
/// <para>
/// <b>Cooperative launch.</b> Unlike v1 (a plain <c>cuLaunchKernel</c>), this kernel uses
/// <c>cuLaunchCooperativeKernel</c> + <c>grid.sync()</c> to combine cross-split partials, the
/// same mechanism <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> already uses. Callers
/// MUST query <see cref="ComputeSafeKvSplit"/> (which internally clamps to
/// <see cref="CudaKernels.MaxSafeAttentionMmaDecodeGqaSplit"/>'s co-residency ceiling) and
/// allocate <c>partialMax</c>/<c>partialSum</c>/<c>partialOut</c>-shaped scratch sized for the
/// returned <c>kvSplit</c> before calling <see cref="Run"/> — exceeding the co-residency
/// ceiling is a hard CUDA error, not a soft perf regression.
/// </para>
/// <para>
/// <b>Precision.</b> Inherits v1's hard-won precision groundwork (FP16 Q/K/V, FP32 mma
/// accumulator, fast_exp_neg-for-P / precise-expf-for-cross-tile-correction split) plus
/// <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/>'s already-characterized cross-split
/// reassociation tolerance at <c>kvSplit&gt;1</c> — re-verified for this kernel's new
/// multi-warp PV split and packed-M-dim layout in
/// <c>CudaAttentionMmaDecodeGqaSplitTests.cs</c>, not just assumed. Ships opt-in
/// (<c>DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT=1</c>, default OFF), per this project's #180/#183
/// precedent for new precision/reassociation axes.
/// </para>
/// </remarks>
internal sealed class CudaAttentionMmaDecodeGqaSplit
{
    /// <summary>Head dimension the kernel is compiled for.</summary>
    internal const int SupportedHeadDim = CudaKernels.AttentionMmaDecodeGqaSplitHeadDim;

    /// <summary>Max GQA group the M-dim-packing scheme supports — MUST equal
    /// <c>MAX_GQA_GROUP</c> in <c>attention_flash_mma_decode_gqa_split.cu</c> (also equals
    /// <see cref="CudaKernels.MaxGqaGroup"/>, the project-wide convention).</summary>
    internal const int MaxGroup = CudaKernels.MaxGqaGroup;

    private static readonly string? MmaDecodeGqaSplitEnv =
        Environment.GetEnvironmentVariable("DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT");

    /// <summary>
    /// Effective toggle. Honours an explicit env override immediately ("1"/"0"); mutable so a
    /// benchmark can interleave OFF/ON reps within one warmed process, same rationale as v1's
    /// <c>CudaAttentionMmaDecode</c> and <see cref="CudaFlashAttention"/>'s toggles.
    /// </summary>
    internal static bool Enabled = MmaDecodeGqaSplitEnv == "1";

    /// <summary>
    /// Count of kernel launches since process start — lets a test/benchmark prove the branch
    /// actually fired rather than silently falling through to a different kernel.
    /// </summary>
    internal static long DispatchCount;

    private readonly CudaKernels _kernels;

    internal CudaAttentionMmaDecodeGqaSplit(CudaKernels kernels) => _kernels = kernels;

    /// <summary>
    /// True when the composed tensor-core decode kernel may be used for this attention call:
    /// the toggle is on, the kernel is loaded, this is a genuine decode step (<c>seqQ == 1</c>),
    /// the head shape matches the compiled kernel (<c>headDim == 256</c>), GQA divides evenly
    /// with <c>group &lt;= MaxGroup</c>, and there is no sliding window.
    /// </summary>
    internal bool CanUse(int seqQ, int seqKv, int slidingWindow, int numHeads, int numKvHeads, int headDim)
        => Enabled
        && _kernels.HasAttentionMmaDecodeGqaSplit
        && seqQ == 1
        && seqKv > 0
        && headDim == SupportedHeadDim
        && slidingWindow <= 0
        && CudaKernels.IsGqaGroupShapeSupported(numHeads, numKvHeads);

    /// <summary>
    /// Computes a safe <c>kvSplit</c> for this (numKvHeads, group, seqKv) shape on THIS GPU,
    /// reusing <see cref="CudaKernels.ComputeAttentionKvSplit"/>'s occupancy-target heuristic
    /// (the same one <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> uses — the two
    /// kernels grid identically, so the heuristic's <c>baseBlocks=numKvHeads</c> framing
    /// applies unchanged) clamped to this kernel's own co-residency ceiling
    /// (<see cref="CudaKernels.MaxSafeAttentionMmaDecodeGqaSplit"/>, queried against its own
    /// function pointer/register footprint, not the FP32 sibling's). Returns 0 if the shape is
    /// not safe to launch at all on this device (caller must fall back to a different kernel).
    /// </summary>
    internal int ComputeSafeKvSplit(int numKvHeads, int group, int seqKv)
    {
        int maxSafe = _kernels.MaxSafeAttentionMmaDecodeGqaSplit(numKvHeads, SupportedHeadDim, group);
        if (maxSafe <= 0) return 0;
        return CudaKernels.ComputeAttentionKvSplit(seqKv, numKvHeads, maxSafe);
    }

    /// <summary>
    /// Runs the composed tensor-core decode kernel. <paramref name="q"/>/<paramref name="k"/>/
    /// <paramref name="v"/> must already be FP16 (caller converts Q; K/V come straight from the
    /// model's FP16 KV cache). <paramref name="output"/> is written F32 directly.
    /// <paramref name="partialMax"/>/<paramref name="partialSum"/>/<paramref name="partialOut"/>
    /// must be sized <c>[numHeads, kvSplit]</c> / <c>[numHeads, kvSplit]</c> /
    /// <c>[numHeads, kvSplit, headDim]</c> floats respectively — same layout
    /// <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> uses. Caller must have confirmed
    /// <see cref="CanUse"/> and obtained <paramref name="kvSplit"/> via
    /// <see cref="ComputeSafeKvSplit"/> (or otherwise guaranteed it does not exceed the
    /// co-residency ceiling — exceeding it is a hard CUDA error).
    /// </summary>
    internal void Run(nint q, nint k, nint v, nint output,
                      int seqKv, int numHeads, int numKvHeads, int kvSplit,
                      nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        DispatchCount++;
        _kernels.LaunchAttentionMmaDecodeGqaSplit(q, k, v, output, seqKv, numHeads, numKvHeads,
            kvSplit, partialMax, partialSum, partialOut, stream);
    }
}
