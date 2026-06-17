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
/// (~half the entries are masked). Block-triangular / causal-aware GEMM was investigated
/// as a follow-up (issue #72) and is a measured <b>NO-GO</b> for this cuBLAS path: a
/// triangular GEMM-floor probe on the production head shape (RTX 3060, Llama-3.2-1B
/// 32/8/64, interleaved min-of-N) showed per-query-block GEMMs give near-zero GEMM-half
/// saving despite ~half the FLOPs — per-block cuBLAS efficiency loss + launch overhead
/// from the smaller growing-K GEMMs swamp the cut (tri/dense GEMM-half ratio 0.92–1.14×
/// at the best block size; worse for smaller blocks). And block-triangular cannot touch
/// the other half of the G3 cost — the full <c>numHeads·s²</c> score round-trip through
/// global memory — so the projected end-to-end attention speedup is ≈0.96× at s=512/1024
/// and only ≈1.06× at s=2048, below the bar. The fused <c>mma.sync</c> flash kernel
/// (issue #70) is the right vehicle for causal-FLOP savings: it cuts both the GEMM half
/// AND the score round-trip by keeping scores in shared/registers. The softmax already
/// zeroes the upper triangle, so the dense path is numerically correct today.
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

        EnsureScratch((long)numHeads * s * s);

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

    private void EnsureScratch(long elems)
    {
        if (elems <= _planeElems) return;
        if (_scoresF32 != 0) CudaDriverApi.cuMemFree_v2(_scoresF32);
        if (_probsF16 != 0) CudaDriverApi.cuMemFree_v2(_probsF16);
        CudaDriverApi.cuMemAlloc_v2(out _scoresF32, (nuint)(elems * Fp32Size)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _probsF16, (nuint)(elems * Fp16Size)).ThrowOnError();
        _planeElems = elems;
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
