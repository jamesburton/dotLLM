using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for the FP32 scaled-dot-product attention kernel.
/// Exercises both decode (1 query token) and prefill (multi-token) configurations via
/// <see cref="CudaKernels.LaunchAttentionF32"/>, comparing against
/// <see cref="Attention.Execute"/>.
/// </summary>
/// <remarks>
/// <para>
/// Both sides use precise exp since #501 (the CUDA kernel's <c>fast_exp_neg</c> and the CPU's
/// Schraudolph path in <see cref="FastMath"/> were removed together). Before #501 both used the
/// matching approximation, and the tolerance below was set for matched-approximation drift; it
/// now covers reduction-order drift alone and is a candidate for tightening.
/// </para>
/// <para>
/// <b>Historical trap-the-bug verification, no longer reproducible (#501).</b> It read: if the
/// CUDA side were swapped to precise <c>expf</c> while the CPU stays on the fast polynomial,
/// max-abs-diff jumps above ~5e-3 — empirically verified by temporarily switching the CPU oracle
/// to <see cref="Attention.ExecuteScalar"/> (precise softmax), which failed the assertion with
/// maxAbsDiff &gt; 5e-3. #501 made precise exp the default on both sides, so that mutation is now
/// a no-op. A replacement discriminator, if one is wanted, is to run with
/// <c>DOTLLM_FAST_EXP=1</c>, which restores the bit trick on the CPU only (CUDA has no such
/// lever — its PTX is precompiled) and should reopen the same ~5e-3 gap.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class AttentionF32ParityTests : IDisposable
{
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;

    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    [SkippableFact]
    public unsafe void AttentionF32_SingleToken_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        // Decode-step shape: 1 query, several KV positions in the cache.
        int seqQ = 1, seqKv = 4;
        int posOffset = seqKv - 1;
        var rng = new Random(42);

        // Scale 1.0 puts softmax pre-scores into a high-magnitude range where the
        // fast_exp polynomial diverges measurably from precise expf — this is the
        // regime that lets the trap-the-bug verification flag a precision swap.
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 1.0f);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 1.0f);
        float[] v = CudaKernelTestHarness.RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 1.0f);

        float[] cpuOutput = new float[seqQ * NumHeads * HeadDim];
        // STALE SINCE #501 — kept for provenance, do not act on it as written. It used to
        // read: "swapping this to Attention.ExecuteScalar (precise expf via
        // Softmax.ExecuteScalar) trips the assertion at maxAbs ~1.5e-2 on 57/128 elements,
        // well outside the 5e-3 tolerance; the current code path matches CUDA's
        // fast_exp_neg." That was a real measurement of the Schraudolph-vs-precise gap.
        // #501 removed the approximation from BOTH sides, so Attention.Execute and
        // Attention.ExecuteScalar now use the same precise exp as the CUDA kernel and the
        // swap no longer discriminates. The 5e-3 tolerance below is therefore calibrated
        // around a difference that no longer exists — re-run on CUDA hardware and
        // re-baseline it downward. Left unwidened and untightened here on purpose.
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, posOffset);

        // Originally: FastExp polynomial bound + reduction-order differences leave us a few
        // e-3 of headroom. Since #501 only the reduction-order term remains (see above).
        CudaKernelTestHarness.AssertClose("AttentionF32-decode", cpuOutput, gpuOutput,
                                          absoluteTolerance: 5e-3f, relativeTolerance: 5e-3f);
    }

    [SkippableFact]
    public unsafe void AttentionF32_Prefill_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        // Prefill: seqQ == seqKv > 1 — exercises the upper-triangle causal mask.
        int seqQ = 4, seqKv = 4;
        int posOffset = 0;
        var rng = new Random(43);

        // Scale 1.0 — same precision-sensitive regime as the decode test, exercised
        // across the upper-triangular causal mask of multi-token prefill.
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 1.0f);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 1.0f);
        float[] v = CudaKernelTestHarness.RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 1.0f);

        float[] cpuOutput = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, posOffset);

        CudaKernelTestHarness.AssertClose("AttentionF32-prefill", cpuOutput, gpuOutput,
                                          absoluteTolerance: 5e-3f, relativeTolerance: 5e-3f);
    }

    private float[] RunGpuAttention(float[] q, float[] k, float[] v,
                                     int seqQ, int seqKv, int posOffset)
    {
        nint devQ = _harness.Upload(q);
        nint devK = _harness.Upload(k);
        nint devV = _harness.Upload(v);
        nint devOut = _harness.Allocate((long)q.Length * sizeof(float));

        _harness.Kernels.LaunchAttentionF32(devQ, devK, devV, devOut,
            seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, slidingWindow: 0,
            _harness.StreamHandle);
        _harness.Synchronize();

        return _harness.DownloadFloats(devOut, q.Length);
    }
}
