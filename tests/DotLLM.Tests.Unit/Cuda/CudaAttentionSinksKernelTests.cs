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
/// Observed on the last local GPU run: maxAbs=9.8068E-07, maxRel=1.2621E-04 (idx 103, expected
/// 0.007770, actual 0.007769) — effectively bit-identical, as expected since <c>sinks=0</c>
/// introduces no new approximation and both sides still use the fast-exp path identically.
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

    // ─────────────────────────────────────────────────────────────────────────────────────────
    // attention_f16 (native/kernels/attention.cu) — Task 4.
    //
    // WHY these exist in addition to the attention_f32 pair above: the F32 kernel is NOT on
    // gpt-oss's forward path. CudaTransformerModel's eager attention dispatch is FP16 end to end
    // (FP16 Q/K/V projections, FP16 KV cache), and its `else` branch calls LaunchAttention →
    // attention_f16; LaunchAttentionF32 is reached only from ForwardHighPrecision (gated on
    // IQ-family quantization — gpt-oss ships MXFP4, not IQ) and a Gemma-4-specific body. So the
    // sink epilogue had to be ported into attention_f16 as well, and that port needs its own
    // discriminating coverage.
    //
    // The two tests below mirror the F32 pair exactly (same shapes, same GQA repeat-2 collision
    // fixture, same per-head sink regimes) so a divergence between the two kernels shows up as one
    // failing and the other passing.
    // ─────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// F16 counterpart of <see cref="NullSinks_BitIdenticalToBaseline"/>: <c>sinks=0</c> (nullptr)
    /// must reproduce the CPU reference computed WITHOUT sinks. The no-regression gate for adding
    /// the epilogue to <c>attention_f16</c>.
    /// <para>
    /// <b>Why this tolerance (2e-2) is LOOSER than the sink-bearing test's (5e-3) — the inverse of
    /// the F32 pair above, and not a mistake.</b> The gap being measured here is
    /// <c>attention.cu</c>'s precise <c>expf</c> against the CPU's <i>non-sink</i> tiled softmax,
    /// which uses the Schraudolph approximation (<c>FastMath.FastExp</c> /
    /// <c>FastMath.ExpSumAndStore</c>, <c>Attention.cs:591,599</c>). That is the ~1%-scale
    /// backend disagreement <c>attention_f32.cu</c>'s file header documents — and it is entirely
    /// PRE-EXISTING, nothing to do with #365. The sink-bearing test is tighter precisely because
    /// the CPU sink path switches to exact <c>TensorPrimitives.Exp</c>/<c>MathF.Exp</c>
    /// (<c>Attention.cs:216-217</c>), which <i>matches</i> the kernel's <c>expf</c>.
    /// </para>
    /// <para>
    /// <b>Verified, not assumed.</b> This exact assertion was run against the PRE-#365
    /// <c>attention.ptx</c> (restored via <c>git checkout HEAD -- native/ptx/attention.ptx</c>,
    /// confirmed 11 params on the <c>attention_f16</c> entry vs 12 after, with a full
    /// <c>dotnet build</c> in between so the test binary's <c>bin/.../ptx/</c> copy actually
    /// refreshed — see this file's Task 3 stale-PTX note). It produced numerically IDENTICAL
    /// output: <c>92/320 elements, maxAbs=1.4376E-002 @idx=218 (expected=-0.045225,
    /// actual=-0.059601)</c>, the same figures to the last digit as the post-change run. That is a
    /// stronger no-regression result than the tolerance assertion itself: with <c>sinks=nullptr</c>
    /// the epilogue is skipped entirely and the kernel is bit-identical to its pre-#365 form.
    /// Tolerance set to 2e-2 (~39% headroom over the observed 1.4376e-2).
    /// </para>
    /// </summary>
    [SkippableFact]
    public void F16_NullSinks_MatchesCpuBaseline()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4, numKvHeads = 2, headDim = 16;
        int seqQ = 5, seqKv = 9;
        int positionOffset = seqKv - seqQ;
        var rng = new Random(365_3);

        var (q, k, v) = RandomF16RoundTripped(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);

        float[] cpuOutput = new float[seqQ * numHeads * headDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        float[] gpuOutput = RunGpuAttentionF16(q, k, v, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                                positionOffset, sinksDevPtr: 0);

        // 2e-2, not 5e-3: pre-existing CPU-fast-exp vs GPU-expf gap, observed maxAbs=1.4376e-2 both
        // before and after this change. See remarks for the pre-change-PTX control run.
        CudaKernelTestHarness.AssertClose("AttentionSinksF16-null", cpuOutput, gpuOutput,
                                          absoluteTolerance: 2e-2f, relativeTolerance: 2e-2f);
    }

    /// <summary>
    /// F16 counterpart of <see cref="WithSinks_MatchesCpuSoftmaxRowWithSink"/> — identical GQA
    /// repeat-2 fixture (heads=4, kvHeads=2, so <c>hkv = hq / 2</c> collides heads 0/1 and 2/3) and
    /// identical distinct per-head sink regimes (-20 negligible, -1 / 0.4 mid-band, +5 provably
    /// dominant given the ±4.0 attainable score bound at headDim=16) — so a <c>sinks[hkv]</c>
    /// mis-indexing in <c>attention.cu</c> is caught the same way Task 3's mutation check proved
    /// for <c>attention_f32.cu</c>.
    /// <para>
    /// Tolerance is the same 5e-3 as the null-sink gate above, i.e. TIGHTER than the F32 sink
    /// test's 1.5e-2. That is not an oversight: the F32 test's loose bound exists purely because
    /// <c>attention_f32.cu</c> uses the Schraudolph <c>fast_exp_neg</c> bit-trick while the CPU
    /// sink path uses exact <c>MathF.Exp</c>. <c>attention.cu</c> has no fast-exp helper and its
    /// epilogue uses precise <c>expf</c>, matching the CPU's exact exp — so that particular
    /// approximation mismatch simply is not present here, and only FP16 storage rounding remains.
    /// </para>
    /// </summary>
    [SkippableFact]
    public void F16_WithSinks_MatchesCpuSoftmaxRowWithSink()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4, numKvHeads = 2, headDim = 16;
        int seqQ = 5, seqKv = 9;
        int positionOffset = seqKv - seqQ;
        var rng = new Random(365_4);

        var (q, k, v) = RandomF16RoundTripped(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);

        // Same values, same per-head regimes, as the F32 sink test — see its remarks.
        float[] sinks = [-20.0f, -1.0f, 0.4f, 5.0f];

        float[] cpuOutput = new float[seqQ * numHeads * headDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, numHeads, numKvHeads, headDim,
                           positionOffset, sinks: sinks);

        nint devSinks = _harness.Upload(sinks);
        float[] gpuOutput = RunGpuAttentionF16(q, k, v, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                                positionOffset, sinksDevPtr: devSinks);

        CudaKernelTestHarness.AssertClose("AttentionSinksF16-gqa", cpuOutput, gpuOutput,
                                          absoluteTolerance: 5e-3f, relativeTolerance: 5e-3f);
    }

    /// <summary>
    /// <c>attention_f16_dyn</c> (via <see cref="CudaKernels.LaunchAttentionDyn"/>) must produce the
    /// same sink-bearing output as <c>attention_f16</c> on identical inputs.
    /// <para>
    /// <b>Why this test is not redundant with <see cref="F16_WithSinks_MatchesCpuSoftmaxRowWithSink"/>.</b>
    /// The two entry points share <c>attention_f16_body</c>, but they are separate
    /// <c>__global__</c> instantiations with separate PTX entries (both went 11 → 12 params in #365)
    /// and separate C# launchers marshalling the new trailing argument. The mutation check in this
    /// file's remarks exercised only the SCALAR instantiation. `_dyn` is the one graph-captured
    /// decode replays — i.e. the entry point gpt-oss actually hits at decode time — so an argument
    /// order slip or a missed plumb there would silently drop sinks on the default decode path while
    /// every other test in this file stayed green.
    /// </para>
    /// <para>
    /// Decode-shaped (<c>seqQ=1</c>, <c>seqKv=9</c>, <c>positionOffset=8</c>) because that is the only
    /// shape <c>_dyn</c> is ever launched with, and its <c>seq_kv</c> / <c>position_offset</c> come
    /// from device ints. Asserted <b>bit-exact</b> on the raw FP16 bits — same-body instantiations,
    /// so anything less would be a weaker claim than the data supports (precedent:
    /// <c>CudaAttentionF16PagedTests</c> asserts bit-exactness between two genuinely different
    /// kernels).
    /// </para>
    /// </summary>
    [SkippableFact]
    public void F16Dyn_WithSinks_MatchesScalarEntryPoint()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4, numKvHeads = 2, headDim = 16;
        const int seqQ = 1, seqKv = 9;
        const int positionOffset = seqKv - seqQ;
        var rng = new Random(365_5);

        var (q, k, v) = RandomF16RoundTripped(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);
        float[] sinks = [-20.0f, -1.0f, 0.4f, 5.0f];
        nint devSinks = _harness.Upload(sinks);

        nint devQ = _harness.Upload(ToHalf(q));
        nint devK = _harness.Upload(ToHalf(k));
        nint devV = _harness.Upload(ToHalf(v));

        int outElems = seqQ * numHeads * headDim;
        nint devOutScalar = _harness.Allocate((long)outElems * sizeof(ushort));
        nint devOutDyn = _harness.Allocate((long)outElems * sizeof(ushort));

        // Scalar entry point: seq_kv / position_offset passed by value.
        _harness.Kernels.LaunchAttention(devQ, devK, devV, devOutScalar,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow: 0,
            _harness.StreamHandle, sinks: devSinks);

        // Dyn entry point: the same two scalars read from device memory instead.
        nint devSeqKv = _harness.Upload(new[] { seqKv });
        nint devPosOffset = _harness.Upload(new[] { positionOffset });
        _harness.Kernels.LaunchAttentionDyn(devQ, devK, devV, devOutDyn,
            seqQ, devSeqKv, numHeads, numKvHeads, headDim, devPosOffset, slidingWindow: 0,
            _harness.StreamHandle, sinks: devSinks);

        _harness.Synchronize();

        Half[] scalar = _harness.DownloadHalves(devOutScalar, outElems);
        Half[] dyn = _harness.DownloadHalves(devOutDyn, outElems);

        for (int i = 0; i < outElems; i++)
        {
            Assert.False(float.IsNaN((float)dyn[i]) || float.IsInfinity((float)dyn[i]),
                $"NaN/Inf in attention_f16_dyn sink output at index {i}");
            Assert.Equal(scalar[i], dyn[i]); // bit-exact: same body, same inputs
        }

        // Guard against a vacuous pass: the sinks must actually have moved the output. Head 3's
        // sink (+5.0) provably dominates every attainable score at headDim=16 (see the F32 sink
        // test's remarks for that bound), so a nullptr run must differ somewhere.
        nint devOutNoSink = _harness.Allocate((long)outElems * sizeof(ushort));
        _harness.Kernels.LaunchAttentionDyn(devQ, devK, devV, devOutNoSink,
            seqQ, devSeqKv, numHeads, numKvHeads, headDim, devPosOffset, slidingWindow: 0,
            _harness.StreamHandle, sinks: 0);
        _harness.Synchronize();

        Half[] noSink = _harness.DownloadHalves(devOutNoSink, outElems);
        Assert.True(noSink.AsSpan().SequenceCompareTo(dyn.AsSpan()) != 0,
            "attention_f16_dyn produced identical output with and without sinks — the sinks "
            + "argument is not reaching the kernel.");
    }

    private static Half[] ToHalf(float[] src)
    {
        var h = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) h[i] = (Half)src[i];
        return h;
    }

    /// <summary>
    /// Random Q/K/V already round-tripped through <see cref="Half"/>, so the CPU reference consumes
    /// EXACTLY the values the FP16 kernel will read. Without this the comparison would also be
    /// measuring input quantization error, which has nothing to do with the sink epilogue under test.
    /// </summary>
    private static (float[] Q, float[] K, float[] V) RandomF16RoundTripped(
        Random rng, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        static float[] Gen(Random rng, int count)
        {
            float[] f = CudaKernelTestHarness.RandomF32(rng, count, scale: 1.0f);
            for (int i = 0; i < f.Length; i++) f[i] = (float)(Half)f[i];
            return f;
        }

        return (Gen(rng, seqQ * numHeads * headDim),
                Gen(rng, seqKv * numKvHeads * headDim),
                Gen(rng, seqKv * numKvHeads * headDim));
    }

    private float[] RunGpuAttentionF16(float[] q, float[] k, float[] v,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, nint sinksDevPtr)
    {
        nint devQ = _harness.Upload(ToHalf(q));
        nint devK = _harness.Upload(ToHalf(k));
        nint devV = _harness.Upload(ToHalf(v));
        nint devOut = _harness.Allocate((long)q.Length * sizeof(ushort));

        _harness.Kernels.LaunchAttention(devQ, devK, devV, devOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow: 0,
            _harness.StreamHandle, sinks: sinksDevPtr);
        _harness.Synchronize();

        Half[] outHalf = _harness.DownloadHalves(devOut, q.Length);
        float[] outF = new float[outHalf.Length];
        for (int i = 0; i < outHalf.Length; i++) outF[i] = (float)outHalf[i];
        return outF;
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
