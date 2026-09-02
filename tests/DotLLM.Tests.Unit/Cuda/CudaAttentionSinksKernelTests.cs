using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Tests.Unit.Cuda.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Kernel-level parity tests for gpt-oss per-head attention sinks (issue #365) in
/// <see cref="CudaKernels.LaunchAttentionF32"/>'s trailing <c>sinks</c> parameter, against the CPU
/// reference <see cref="Attention.SoftmaxRowWithSink"/> convention (llama.cpp
/// <c>ggml_soft_max_add_sinks</c>: the sink is a per-head learned scalar that joins the softmax
/// denominator as a virtual final tile contributing no value vector).
/// </summary>
/// <remarks>
/// <para>
/// <b>Test 1 tolerance (no-regression gate):</b> <c>sinks=0</c> (nullptr) must reproduce the
/// pre-#365 CPU reference exactly at the SAME tolerance already established by
/// <c>AttentionF32ParityTests</c> (<c>abs=5e-3, rel=5e-3</c> — both sides use the Schraudolph
/// fast-exp approximation, so only matched-approximation / reduction-order drift is expected).
/// Observed on the last local GPU run: maxAbs ≈ 2.4E-04, maxRel ≈ 1.9E-04 — comfortably inside
/// tolerance, confirming <c>sinks=0</c> is a true no-op.
/// </para>
/// <para>
/// <b>Test 2 tolerance (sink-bearing parity):</b> the CPU sink path
/// (<see cref="Attention.SoftmaxRowWithSink"/>) deliberately switches to exact
/// <c>TensorPrimitives.Exp</c>/<c>MathF.Exp</c> (so masked <c>-inf</c> entries map to exactly 0),
/// while the CUDA epilogue still uses <c>fast_exp_neg</c> (the Schraudolph bit-trick). That
/// approximation-mismatch is the same one <c>attention_f32.cu</c>'s file header documents as
/// worth ~1% / ~5e-3 abs on plain (no-sink) attention output when the two sides' softmax
/// implementations disagree. Per Task 1's hand-off, the plan's guessed "~1e-6 expected" tolerance
/// is not achievable here and was not fought — the tolerance below is calibrated from what was
/// actually observed on the first real run: <c>maxAbs=1.1829E-02</c> (idx 78, expected 0.190795,
/// actual 0.202624; <c>maxRel</c> is dominated by a handful of near-zero-expected elements and is
/// not a useful signal here — the mixed-tolerance comparator already treats those as "abs decides"
/// cases). Tolerance is set to <c>abs=1.5e-2, rel=1.5e-2</c> (~27% headroom over the observed
/// 1.18e-2 peak, in the same spirit as this file's existing 5e-3-over-~1.5e-3-ish margin
/// convention) — tight enough that the required mutation check (below) still fails by nearly two
/// orders of magnitude.
/// </para>
/// <para>
/// <b>Mutation check (required, #384/#385/#366 precedent):</b> temporarily changed
/// <c>native/kernels/attention_f32.cu</c>'s sink lookup from <c>sinks[hq]</c> to <c>sinks[hkv]</c>
/// (the exact wrong-head-index bug class this GQA-repeat-2 fixture — heads=4, kvHeads=2 — exists
/// to catch), rebuilt <c>attention_f32.ptx</c> with the pinned toolkit
/// (<c>E:\CUDA_v12.8.1\bin\nvcc.exe -ptx -arch=compute_75 -std=c++17
/// -allow-unsupported-compiler</c>, matching Task 1's exact command). NOTE: the first rebuild
/// attempt produced a false pass — <c>CudaKernelTestHarness.FindPtxDir()</c> prefers a PTX copy
/// already sitting in the test binary's output directory (<c>bin/.../ptx/</c>, an MSBuild
/// <c>Content</c> item with <c>CopyToOutputDirectory=PreserveNewest</c>) over the freshly rebuilt
/// <c>native/ptx/</c> copy; re-running <c>dotnet build</c> on the test project (which re-triggers
/// the newer-wins copy) was required before the mutated PTX actually reached the test. After that,
/// <see cref="WithSinks_MatchesCpuSoftmaxRowWithSink"/> FAILED as expected: 130/320 elements
/// exceeded tolerance, maxAbs=3.8763E-01 @idx=126 (expected -0.019618, actual -0.407252) —
/// roughly 26x past the 1.5e-2 tolerance — because <c>hkv = hq / 2</c> collapses two distinct
/// per-head sinks onto one shared value per KV-head group (heads 0/1 both read
/// <c>sinks[0]</c>=-20.0 instead of their true -20.0/-1.0, heads 2/3 both read <c>sinks[1]</c>=-1.0
/// instead of their true 0.4/5.0 — head 3 in particular loses its dominant +5.0 sink entirely).
/// Reverted via <c>git restore native/kernels/attention_f32.cu native/ptx/attention_f32.ptx</c>,
/// confirmed <c>git diff --stat native/</c> empty (byte-identical revert: rebuilt PTX file size
/// 115923 bytes matches the pre-mutation file exactly, vs. 115961 bytes for the mutated PTX),
/// rebuilt the test project so the bin-output PTX copy was refreshed back to the original, and
/// confirmed the test passing again — both tests green, same pass-case numbers as the original run.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class CudaAttentionSinksKernelTests : IDisposable
{
    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    /// <summary>
    /// <c>sinks=0</c> (nullptr) must be bit-near-identical to the CPU reference computed WITHOUT
    /// sinks — the no-regression gate for the #365 epilogue addition. Uses the same 5e-3 tolerance
    /// as the pre-existing <c>AttentionF32ParityTests</c> (same fast-exp-vs-fast-exp comparison,
    /// no new approximation mismatch introduced by this path).
    /// </summary>
    [SkippableFact]
    public void NullSinks_BitIdenticalToBaseline()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4, numKvHeads = 2, headDim = 16;
        int seqQ = 5, seqKv = 9;
        int positionOffset = seqKv - seqQ;
        var rng = new Random(365_1);

        float[] q = CudaKernelTestHarness.RandomF32(rng, seqQ * numHeads * headDim, scale: 1.0f);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqKv * numKvHeads * headDim, scale: 1.0f);
        float[] v = CudaKernelTestHarness.RandomF32(rng, seqKv * numKvHeads * headDim, scale: 1.0f);

        float[] cpuOutput = new float[seqQ * numHeads * headDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                             positionOffset, sinksDevPtr: 0);

        CudaKernelTestHarness.AssertClose("AttentionSinks-null", cpuOutput, gpuOutput,
                                          absoluteTolerance: 5e-3f, relativeTolerance: 5e-3f);
    }

    /// <summary>
    /// Sink-bearing parity against <see cref="Attention.SoftmaxRowWithSink"/> on a GQA
    /// (repeat-factor-2) fixture. Per-head sinks are DISTINCT so a wrong-head-index bug
    /// discriminates, and span both regimes the epilogue's <c>m = max(running_max, sink)</c>
    /// branch must handle correctly in one fixture:
    /// <list type="bullet">
    /// <item>head 3: sink = +5.0 — a DETERMINISTIC upper bound on any attainable raw score. With
    /// headDim=16 and Q/K sampled uniformly in [-1, 1], the maximum possible pre-softmax score is
    /// <c>headDim * scale^2 / sqrt(headDim) = sqrt(headDim) = 4.0</c> (every per-dim product at its
    /// extreme, same sign) — sink=5.0 exceeds that unconditionally, forcing <c>m = sink</c> for
    /// every row of this head regardless of what the random data happens to produce.</item>
    /// <item>head 0: sink = -20.0 — well below the symmetric -4.0 floor, so the sink is provably
    /// negligible (its <c>exp(sink - m)</c> term is astronomically small relative to the row's
    /// other terms) and should have no measurable effect on head 0's output.</item>
    /// <item>heads 1, 2: sink = -1.0, 0.4 — mid-range values that sit inside the possible score
    /// band, neither dominating nor negligible; exercises the ordinary (non-degenerate) blend.</item>
    /// </list>
    /// GQA shape (heads=4, kvHeads=2, repeat factor 2) means <c>hkv = hq / 2</c>: heads 0/1 share
    /// kvHead 0, heads 2/3 share kvHead 1. Sinks index by <c>hq</c> (query head), never <c>hkv</c>
    /// — see the mutation-check evidence in this file's remarks for proof this fixture catches
    /// that exact collision.
    /// </summary>
    [SkippableFact]
    public void WithSinks_MatchesCpuSoftmaxRowWithSink()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4, numKvHeads = 2, headDim = 16;
        int seqQ = 5, seqKv = 9;
        int positionOffset = seqKv - seqQ;
        var rng = new Random(365_2);

        float[] q = CudaKernelTestHarness.RandomF32(rng, seqQ * numHeads * headDim, scale: 1.0f);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqKv * numKvHeads * headDim, scale: 1.0f);
        float[] v = CudaKernelTestHarness.RandomF32(rng, seqKv * numKvHeads * headDim, scale: 1.0f);

        // Distinct per head; see remarks for the regime each value exercises.
        float[] sinks = [-20.0f, -1.0f, 0.4f, 5.0f];

        float[] cpuOutput = new float[seqQ * numHeads * headDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, numHeads, numKvHeads, headDim,
                           positionOffset, sinks: sinks);

        nint devSinks = _harness.Upload(sinks);
        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                             positionOffset, sinksDevPtr: devSinks);

        // Calibrated from observed (see remarks): pass-case maxAbs = 1.1829e-2, comfortably under
        // this tolerance; the required mutation check (sinks[hkv] instead of sinks[hq]) diverges
        // to maxAbs = 3.8763e-1 (~26x this tolerance), confirming discrimination.
        CudaKernelTestHarness.AssertClose("AttentionSinks-gqa", cpuOutput, gpuOutput,
                                          absoluteTolerance: 1.5e-2f, relativeTolerance: 1.5e-2f);
    }

    private float[] RunGpuAttention(float[] q, float[] k, float[] v,
                                     int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                     int positionOffset, nint sinksDevPtr)
    {
        nint devQ = _harness.Upload(q);
        nint devK = _harness.Upload(k);
        nint devV = _harness.Upload(v);
        nint devOut = _harness.Allocate((long)q.Length * sizeof(float));

        _harness.Kernels.LaunchAttentionF32(devQ, devK, devV, devOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow: 0,
            _harness.StreamHandle, sinks: sinksDevPtr);
        _harness.Synchronize();

        return _harness.DownloadFloats(devOut, q.Length);
    }
}
