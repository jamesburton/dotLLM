using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// G3 tensor-core prefill-attention path: two cuBLAS strided-batched GEMMs around a
/// coalesced causal-softmax kernel, replacing the FP32 CUDA-core <c>attention_f16</c>
/// kernel for pure-prefill attention on throttle-prone GeForce Ampere parts.
/// </summary>
/// <remarks>
/// <para>
/// Pipeline per forward (for one layer's attention):
/// </para>
/// <list type="number">
///   <item>QK^T → scores: a strided-batched GEMM per kv-head over its <c>group</c> query
///   heads (GQA broadcast via KV stride 0). FP16 inputs + tensor cores + COMPUTE_32F, but
///   the C output is kept in <b>FP32</b> (per-query-head column-major <c>[s × s]</c>,
///   leading dim <c>s</c>); the 1/√headDim scale is folded into alpha. Keeping the
///   wide-range pre-softmax scores in FP32 is required to hold end-to-end logit parity at
///   the 5e-3 bar — rounding them to FP16 before <c>exp()</c> diverges past tolerance at
///   realistic activation magnitudes (verified by a magnitude sweep on the isolated
///   parity test).</item>
///   <item>Causal softmax (coalesced one-thread-per-row, FP32 in → FP16 out): max/exp/
///   normalize over keys <c>0..tq</c> in FP32, writing normalized FP16 probs to a second
///   buffer; masked tail zeroed so the PV sum over the full key axis ignores it. Probs
///   live in [0,1] where FP16 rel error is ~5e-4, so PV stays on tensor cores cheaply.</item>
///   <item>P·V → output, written <b>directly into the row-major</b>
///   <c>[seq, numHeads, headDim]</c> layout the rest of the forward expects (the same
///   layout <c>attention_f16</c> produces). This is achieved by computing the transposed
///   product (<c>m=headDim, n=s</c>) so cuBLAS's native column-major store lands element
///   <c>(tq,hq,d)</c> at <c>tq·numHeads·headDim + hq·headDim + d</c> — no repack kernel,
///   no second output buffer.</item>
/// </list>
/// <para>
/// <b>Gating (see <see cref="CanUse"/>).</b> The softmax kernel hard-codes
/// <c>causal_len = tq+1</c> with no position offset and no sliding window, and the GEMMs
/// assume <c>seqKv == seqLen</c> (square scores). So the path is restricted to pure
/// prefill from an empty/aligned cache (<c>positionOffset == 0</c>, <c>seqKv == seqLen</c>)
/// with global attention (<c>slidingWindow &lt;= 0</c>) and <c>seqLen &gt; 1</c>. Anything
/// outside that — decode, prefix-cache reuse, sliding-window models — falls back to
/// <c>attention_f16</c>, which honours those cases.
/// </para>
/// <para>
/// The dense-square QK/PV process the full <c>s × s</c> even though prefill is causal
/// (~half the entries are masked). Block-triangular / causal-aware GEMM is a documented
/// follow-up: it fights the strided-batched-cuBLAS model (fragmenting into many small
/// per-diagonal GEMMs would destroy throughput) and is out of scope for a clean
/// prefill-only wire. The softmax already zeroes the upper triangle, so the dense path
/// is numerically correct today.
/// </para>
/// </remarks>
internal sealed class CudaG3Attention : IDisposable
{
    private const int Fp16Size = 2;
    private const int Fp32Size = 4;

    private static readonly string? G3AttnEnv =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_G3_ATTN");

    // Effective flag. Honours an explicit env override immediately; otherwise stays off
    // until ConfigureDefault sets the device-gated default at model load. Mutable so a
    // benchmark can interleave OFF/ON reps within one warmed process (consumer GPUs drift
    // clocks ~2× across separate runs, so separate-process minima are not comparable).
    internal static bool Enabled = G3AttnEnv == "1";

    private readonly CudaKernels _kernels;

    // Grow-on-demand scratch for the [numHeads × s × s] attention scores/probs, reused
    // across layers and forwards. The QK scores are kept in FP32 (precision: rounding
    // wide-range pre-softmax scores to FP16 breaks end-to-end logit parity at the 5e-3
    // bar); the softmax writes normalized FP16 probs for the PV GEMM. Both sized to the
    // largest (numHeads·s·s) seen so far.
    private nint _scoresF32;
    private nint _probsF16;
    private long _planeElems;
    private bool _disposed;

    internal CudaG3Attention(CudaKernels kernels) => _kernels = kernels;

    /// <summary>
    /// Sets the G3-attention default from device eligibility, unless
    /// <c>DOTLLM_CUDA_G3_ATTN</c> overrides it ("1" force on, "0" force off). Gating: on for
    /// GeForce Ampere (the parts that throttle FP32-accumulate CUDA-core attention), off elsewhere.
    /// </summary>
    /// <param name="deviceEligible">True when the device benefits — GeForce Ampere.</param>
    internal static void ConfigureDefault(bool deviceEligible) =>
        Enabled = G3AttnEnv switch
        {
            "1" => true,
            "0" => false,
            _ => deviceEligible,
        };

    /// <summary>
    /// True when the G3 prefill-attention path may be used for this attention call: the
    /// toggle is on, the causal-softmax kernel is loaded, GQA divides evenly, and the call
    /// is a pure square-causal prefill with global attention (no position offset, no
    /// prefix-cache reuse, no sliding window). Otherwise the caller keeps
    /// <c>attention_f16</c>.
    /// </summary>
    internal bool CanUse(int seqLen, int seqKv, int positionOffset, int slidingWindow,
                         int numHeads, int numKvHeads)
        => Enabled
        && _kernels.HasAttentionSoftmaxCausalCoalescedF32In
        && seqLen > 1
        && seqKv == seqLen
        && positionOffset == 0
        && slidingWindow <= 0
        && numKvHeads > 0
        && (numHeads % numKvHeads) == 0;

    /// <summary>
    /// Runs the G3 path (QK GEMM → causal softmax → PV GEMM) for one prefill attention
    /// call, writing the result into <paramref name="output"/> in the row-major
    /// <c>[seq, numHeads, headDim]</c> layout (matching <c>attention_f16</c>). Q/K/V are
    /// the same RoPE'd FP16 buffers the eager path consumes: Q is
    /// <c>[seq, numHeads, headDim]</c> (stride <c>numHeads·headDim</c>) and K/V are
    /// <c>[seq, numKvHeads, headDim]</c> (stride <c>numKvHeads·headDim</c>, e.g. the
    /// KV-cache rows). Caller must have confirmed <see cref="CanUse"/>.
    /// </summary>
    internal unsafe void Run(nint cublasHandle, nint q, nint k, nint v, nint output,
                             int seqLen, int numHeads, int numKvHeads, int headDim,
                             nint stream)
    {
        int s = seqLen;
        int group = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);
        float one = 1.0f, zero = 0.0f, sc = scale;

        EnsureScratch(SquareScratchElems(s, numHeads));

        // ── QK^T → FP32 scores (col-major [s × s] per query head, ldc=s; scale in alpha) ──
        // FP16 inputs + tensor cores + COMPUTE_32F, but the C output is CUDA_R_32F so the
        // wide-range pre-softmax scores are NOT rounded to FP16 (the dominant error term).
        for (int h = 0; h < numKvHeads; h++)
        {
            nint qBase = q + (nint)((long)h * group * headDim * Fp16Size);
            nint kBase = k + (nint)((long)h * headDim * Fp16Size);
            nint scBase = _scoresF32 + (nint)((long)h * group * s * s * Fp32Size);

            CublasApi.cublasGemmStridedBatchedEx(cublasHandle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                s, s, headDim,
                (nint)(&sc),
                qBase, CublasApi.CUDA_R_16F, qStride, headDim,
                kBase, CublasApi.CUDA_R_16F, kvStride, 0,
                (nint)(&zero),
                scBase, CublasApi.CUDA_R_32F, s, (long)s * s,
                group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT)
                .ThrowOnCublasError();
        }

        // ── Causal softmax (FP32 scores → FP16 probs), all numHeads planes ──
        _kernels.LaunchAttentionSoftmaxCausalF32In(_scoresF32, _probsF16, s, numHeads, stream);

        // ── P·V → output, transposed so the col-major store lands row-major ──
        // C = out^T[d, tq] = Σ_tk V[tk, d] · probs[tq, tk]; m=headDim, n=s, k=s.
        // ldc = numHeads·headDim, strideC = headDim → query head hq=h·group+g lands at
        // tq·numHeads·headDim + hq·headDim + d  (== attention_f16 row-major output).
        for (int h = 0; h < numKvHeads; h++)
        {
            nint vBase = v + (nint)((long)h * headDim * Fp16Size);
            nint pBase = _probsF16 + (nint)((long)h * group * s * s * Fp16Size);
            nint oBase = output + (nint)((long)h * group * headDim * Fp16Size);

            CublasApi.cublasGemmStridedBatchedEx(cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T,
                headDim, s, s,
                (nint)(&one),
                vBase, CublasApi.CUDA_R_16F, kvStride, 0,
                pBase, CublasApi.CUDA_R_16F, s, (long)s * s,
                (nint)(&zero),
                oBase, CublasApi.CUDA_R_16F, numHeads * headDim, headDim,
                group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT)
                .ThrowOnCublasError();
        }
    }

    /// <summary>
    /// Least query rows for the offset path (#1594). Below it (speculative / MTP verify rows, single-token decode) the
    /// fused <c>attention_f16</c> is cheaper than two cuBLAS GEMMs plus a softmax, and those paths keep their numerics.
    /// </summary>
    internal const int OffsetMinQueryRows = 32;

    /// <summary>
    /// Kill switch for the offset path (#1594): <c>DOTLLM_CUDA_G3_OFFSET=0</c> sends prefills over a cached prefix back to
    /// <c>attention_f16</c>. Mutable so a test can compare both in one process.
    /// </summary>
    internal static bool OffsetEnabled = Environment.GetEnvironmentVariable("DOTLLM_CUDA_G3_OFFSET") != "0";

    /// <summary>
    /// Score elements (query heads x query rows x keys) one offset-path chunk may hold (#1594). The suffix is processed in
    /// query-row chunks under this bound, so a long cached prefix never sizes the scratch to heads x suffix x context
    /// (6 bytes per element: FP32 scores + FP16 probs; 64M elements = 384 MiB).
    /// </summary>
    internal static long OffsetMaxScoreElems { get; set; } = DefaultOffsetMaxScoreElems;

    /// <summary>Default <see cref="OffsetMaxScoreElems"/>.</summary>
    internal const long DefaultOffsetMaxScoreElems = 64L << 20;

    /// <summary>
    /// True when the G3 offset path may run: queries <c>[positionOffset, positionOffset + seqLen)</c> attending the whole
    /// cached history (<c>seqKv == positionOffset + seqLen</c>), at least <see cref="OffsetMinQueryRows"/> rows, global
    /// attention, GQA dividing evenly, and the offset softmax loaded. This is the prefill of a prefix-cache hit (#1594):
    /// before it, those prefills fell to <c>attention_f16</c>, which ran a 647-token suffix over a 4K-token TwIL-8B prefix
    /// 2.5x slower than G3 prefilled the whole 4.7K-token prompt cold.
    /// </summary>
    internal bool CanUseOffset(int seqLen, int seqKv, int positionOffset, int slidingWindow, int numHeads, int numKvHeads)
        => Enabled && OffsetEnabled
        && _kernels.HasAttentionSoftmaxCausalOffsetF32In
        && seqLen >= OffsetMinQueryRows
        && positionOffset > 0
        && seqKv == positionOffset + seqLen
        && slidingWindow <= 0
        && numKvHeads > 0
        && (numHeads % numKvHeads) == 0;

    /// <summary>Calls of <see cref="RunOffset"/>, process-wide (test observable: proves the offset path ran).</summary>
    internal static long OffsetRunCount => Interlocked.Read(ref s_offsetRuns);
    private static long s_offsetRuns;

    /// <summary>
    /// G3 attention for queries that start <paramref name="positionOffset"/> rows into the cache: per query-row chunk,
    /// QK^T over the keys that chunk can see (<c>[chunk x (offset + chunkEnd)]</c>, FP32), the offset causal softmax, and P.V
    /// written row-major into <paramref name="output"/>. <paramref name="q"/> and <paramref name="output"/> hold the
    /// <paramref name="seqLen"/> new rows; <paramref name="k"/>/<paramref name="v"/> are the cache from row 0. Caller must have
    /// confirmed <see cref="CanUseOffset"/>.
    /// </summary>
    internal unsafe void RunOffset(nint cublasHandle, nint q, nint k, nint v, nint output,
                                   int seqLen, int positionOffset, int numHeads, int numKvHeads, int headDim, nint stream)
    {
        int group = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);
        float one = 1.0f, zero = 0.0f, sc = scale;

        Interlocked.Increment(ref s_offsetRuns);
        long perRow = (long)numHeads * (positionOffset + seqLen);
        int chunk = (int)Math.Clamp(OffsetMaxScoreElems / Math.Max(1, perRow), 1, seqLen);
        EnsureScratch(OffsetScratchElems(seqLen, positionOffset, numHeads));

        for (int c0 = 0; c0 < seqLen; c0 += chunk)
        {
            int sq = Math.Min(chunk, seqLen - c0);
            int off = positionOffset + c0;
            int skv = off + sq;   // keys this chunk can see
            nint qc = q + (nint)((long)c0 * qStride * Fp16Size);
            nint oc = output + (nint)((long)c0 * qStride * Fp16Size);

            // QK^T -> FP32 scores, col-major [sq x skv] per query head (ldc = sq).
            for (int h = 0; h < numKvHeads; h++)
            {
                nint qBase = qc + (nint)((long)h * group * headDim * Fp16Size);
                nint kBase = k + (nint)((long)h * headDim * Fp16Size);
                nint scBase = _scoresF32 + (nint)((long)h * group * sq * skv * Fp32Size);
                CublasApi.cublasGemmStridedBatchedEx(cublasHandle,
                    CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    sq, skv, headDim,
                    (nint)(&sc),
                    qBase, CublasApi.CUDA_R_16F, qStride, headDim,
                    kBase, CublasApi.CUDA_R_16F, kvStride, 0,
                    (nint)(&zero),
                    scBase, CublasApi.CUDA_R_32F, sq, (long)sq * skv,
                    group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT)
                    .ThrowOnCublasError();
            }

            _kernels.LaunchAttentionSoftmaxCausalOffsetF32In(_scoresF32, _probsF16, sq, skv, off, numHeads, stream);

            // P.V: out^T[d, tq] = sum_tk V[tk, d] * probs[tq, tk]; m = headDim, n = sq, k = skv.
            for (int h = 0; h < numKvHeads; h++)
            {
                nint vBase = v + (nint)((long)h * headDim * Fp16Size);
                nint pBase = _probsF16 + (nint)((long)h * group * sq * skv * Fp16Size);
                nint oBase = oc + (nint)((long)h * group * headDim * Fp16Size);
                CublasApi.cublasGemmStridedBatchedEx(cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T,
                    headDim, sq, skv,
                    (nint)(&one),
                    vBase, CublasApi.CUDA_R_16F, kvStride, 0,
                    pBase, CublasApi.CUDA_R_16F, sq, (long)sq * skv,
                    (nint)(&zero),
                    oBase, CublasApi.CUDA_R_16F, numHeads * headDim, headDim,
                    group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT)
                    .ThrowOnCublasError();
            }
        }
    }

    /// <summary>Bytes of score scratch per element: FP32 scores + FP16 probs.</summary>
    internal const int ScratchBytesPerElement = Fp32Size + Fp16Size;

    /// <summary>
    /// Plane size above which the scratch is released at the end of the forward that grew it (#1848, <see cref="ReleaseIfLarge"/>):
    /// 64M elements = 384 MiB, the offset path's chunk bound. Smaller planes stay for reuse.
    /// </summary>
    internal const long RetainedMaxElems = DefaultOffsetMaxScoreElems;

    /// <summary>Times a G3 call was declined because its scratch did not fit in free VRAM (#1848; test observable).</summary>
    internal static long ScratchRefusalCount => Interlocked.Read(ref s_scratchRefusals);
    private static long s_scratchRefusals;
    private static int s_refusalLogged;

    /// <summary>Score elements <see cref="Run"/> needs for a square-causal prefill of <paramref name="seqLen"/> rows.</summary>
    internal static long SquareScratchElems(int seqLen, int numHeads) => (long)numHeads * seqLen * seqLen;

    /// <summary>Score elements <see cref="RunOffset"/> needs (one query-row chunk under <see cref="OffsetMaxScoreElems"/>).</summary>
    internal static long OffsetScratchElems(int seqLen, int positionOffset, int numHeads)
    {
        long perRow = (long)numHeads * (positionOffset + seqLen);
        int chunk = (int)Math.Clamp(OffsetMaxScoreElems / Math.Max(1, perRow), 1, seqLen);
        return (long)numHeads * chunk * (positionOffset + seqLen);
    }

    /// <summary>
    /// Makes the score scratch hold <paramref name="elems"/> elements, or declines (#1848). The scratch is quadratic in the prompt
    /// (numHeads x s^2 x 6 B: 3 GiB at 4K tokens with 32 heads), so no per-token context fit can count it: it is admitted against
    /// free VRAM here (the plane it replaces counts as free) and, when it does not fit, the caller runs <c>attention_f16</c>, which
    /// needs no scratch, instead of letting WDDM spill it into shared memory. Call after <see cref="CanUse"/> /
    /// <see cref="CanUseOffset"/> and before <see cref="Run"/> / <see cref="RunOffset"/>.
    /// </summary>
    /// <param name="elems">Score elements needed.</param>
    /// <returns>Whether the scratch is ready.</returns>
    internal bool TryReserve(long elems)
    {
        if (elems <= _planeElems) return true;
        if (!CudaAdmission.TryFit(elems * ScratchBytesPerElement, out long free, releasedBytes: _planeElems * ScratchBytesPerElement))
        {
            Interlocked.Increment(ref s_scratchRefusals);
            if (Interlocked.Exchange(ref s_refusalLogged, 1) == 0)
                Console.Error.WriteLine(
                    $"[dotllm] G3 prefill attention needs {(elems * ScratchBytesPerElement) >> 20} MiB of score scratch and only " +
                    $"{free >> 20} MiB of VRAM is free ({CudaAdmission.ReserveBytes >> 20} MiB reserve): running attention_f16 for " +
                    "this prefill instead (slower on GeForce Ampere, but no shared-memory spill). --prefill-chunk bounds it. Logged once.");
            return false;
        }
        EnsureScratch(elems);
        return true;
    }

    /// <summary>
    /// Frees the score scratch when it is larger than <see cref="RetainedMaxElems"/> (#1848). A long cold prompt grows it to GiBs;
    /// kept, it would hold VRAM the context fit gave to the KV cache. Call at the end of a prefill forward.
    /// </summary>
    internal void ReleaseIfLarge()
    {
        if (_planeElems <= RetainedMaxElems) return;
        if (_scoresF32 != 0) { CudaDriverApi.cuMemFree_v2(_scoresF32); _scoresF32 = 0; }
        if (_probsF16 != 0) { CudaDriverApi.cuMemFree_v2(_probsF16); _probsF16 = 0; }
        _planeElems = 0;
    }

    /// <summary>Score elements the scratch holds now (tests).</summary>
    internal long PlaneElems => _planeElems;

    /// <summary>
    /// Whether any G3 prefill can run on this model and device (#1848): the toggle is on and the causal softmax is loaded. When it
    /// can, a plane of up to <see cref="RetainedMaxElems"/> stays allocated between prefills and the context fit must charge it.
    /// </summary>
    internal bool MayRun => Enabled && _kernels.HasAttentionSoftmaxCausalCoalescedF32In;

    /// <summary>Frees the score scratch whatever its size (#1848): a KV cache that was refused gets its room back first.</summary>
    /// <returns>The bytes released.</returns>
    internal long Release()
    {
        long bytes = _planeElems * ScratchBytesPerElement;
        if (_scoresF32 != 0) { CudaDriverApi.cuMemFree_v2(_scoresF32); _scoresF32 = 0; }
        if (_probsF16 != 0) { CudaDriverApi.cuMemFree_v2(_probsF16); _probsF16 = 0; }
        _planeElems = 0;
        return bytes;
    }

    private void EnsureScratch(long elems)
    {
        if (elems <= _planeElems) return;
        // Release and forget the old pair first: if an allocation below throws, the object must not keep a freed pointer or a
        // plane size it no longer has (the next call would reuse a dangling buffer).
        if (_scoresF32 != 0) { CudaDriverApi.cuMemFree_v2(_scoresF32); _scoresF32 = 0; }
        if (_probsF16 != 0) { CudaDriverApi.cuMemFree_v2(_probsF16); _probsF16 = 0; }
        _planeElems = 0;
        CudaDriverApi.cuMemAlloc_v2(out nint scores, (nuint)(elems * Fp32Size)).ThrowOnError();
        _scoresF32 = scores;
        CudaDriverApi.cuMemAlloc_v2(out nint probs, (nuint)(elems * Fp16Size)).ThrowOnError();
        _probsF16 = probs;
        _planeElems = elems;   // only once both exist; a throw above leaves a zero size, so the next call reallocates
    }

    public void Dispose()
    {
        if (_disposed) return;
        if (_scoresF32 != 0) { CudaDriverApi.cuMemFree_v2(_scoresF32); _scoresF32 = 0; }
        if (_probsF16 != 0) { CudaDriverApi.cuMemFree_v2(_probsF16); _probsF16 = 0; }
        _planeElems = 0;
        _disposed = true;
    }
}
