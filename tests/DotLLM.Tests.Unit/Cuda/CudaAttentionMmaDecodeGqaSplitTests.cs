using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness coverage for the OPT-IN tensor-core (mma.sync) FP16 decode-attention kernel
/// composed with the #197/#198 GQA-group + split-KV grid design (issue #199 v2, this is a
/// clean-room v2 based on <c>dev</c>, NOT a port of v1's branch
/// <c>issue/199-tensor-core-decode-attention</c> — see <c>CudaAttentionMmaDecodeGqaSplit</c>'s
/// doc for why v1's single-warp/block scope regressed and what v2 changes structurally).
///
/// <para>
/// References, same three as v1 used: the CPU F32 oracle (<see cref="Attention.Execute(float*,
/// float*, float*, float*, int, int, int, int, int, int, ComputeThreadPool?, int?)"/>), the
/// shipping GPU F32 decode kernel (<see cref="CudaKernels.LaunchAttentionF32"/>), and — new for
/// v2, since this kernel composes with the GQA-group grid design — the sibling FP32 combined
/// kernel <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> (issues #197/#198), to confirm
/// this kernel does not systematically diverge from the grid design it was built to match.
/// </para>
///
/// <para>
/// <b>Tolerance: 5e-3 abs OR rel, same bar v1 established</b> (matching
/// <c>CudaTensorCoreAttentionParityTests</c>'s existing FP16-tensor-core precedent). v1's
/// bring-up found and fixed a real precision bug at this tolerance (Schraudolph fast-exp vs
/// precise expf for the cross-KV-tile online-softmax correction — see the .cu file's header);
/// this test suite re-verifies that fix still holds under v2's new multi-warp PV split and
/// packed-M-dimension layout rather than assuming it carries over unchanged.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaAttentionMmaDecodeGqaSplitTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    // Bonsai-27B's real qwen35moe shape (numKvHeads=4, headDim=256, group=6).
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;
    private const int Group = NumHeads / NumKvHeads;

    private readonly ITestOutputHelper _out;
    public CudaAttentionMmaDecodeGqaSplitTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    private static float[] RandomVec(Random rng, int n, float magnitude = 1.0f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * magnitude;
        return v;
    }

    private static ushort[] ToHalf(float[] f)
    {
        var h = new ushort[f.Length];
        for (int i = 0; i < f.Length; i++) h[i] = BitConverter.HalfToUInt16Bits((Half)f[i]);
        return h;
    }

    private static void UploadF32(nint dst, float[] host)
    {
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(host.Length * sizeof(float))).ThrowOnError();
    }

    private static void UploadF16(nint dst, ushort[] host)
    {
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(host.Length * sizeof(ushort))).ThrowOnError();
    }

    private static float[] DownloadF32(nint src, int elems)
    {
        var host = new float[elems];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(elems * sizeof(float))).ThrowOnError();
        return host;
    }

    /// <summary>
    /// Allocates scratch, uploads FP16 Q/K/V, runs the composed tensor-core decode kernel for
    /// one decode step, downloads F32 output, frees everything. <paramref name="kvSplit"/> is
    /// passed through unclamped (tests choose it explicitly, including forcing 1 to isolate the
    /// M-dim-packing axis from the cross-split combine axis).
    /// </summary>
    private static float[] RunMmaGqaSplit(CudaAttentionMmaDecodeGqaSplit kernel, CudaStream stream,
        ushort[] qF16, ushort[] kF16, ushort[] vF16, int seqKv, int numHeads, int numKvHeads, int kvSplit)
    {
        int qElems = numHeads * HeadDim;
        long kvElemsF16 = (long)kF16.Length;

        nint dQ = 0, dK = 0, dV = 0, dOut = 0, dPartialMax = 0, dPartialSum = 0, dPartialOut = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)(qF16.Length * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)(kvElemsF16 * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)(kvElemsF16 * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOut, (nuint)(qElems * sizeof(float))).ThrowOnError();
            UploadF16(dQ, qF16); UploadF16(dK, kF16); UploadF16(dV, vF16);

            long scalarBytes = (long)numHeads * kvSplit * sizeof(float);
            long outPartialBytes = scalarBytes * HeadDim;
            CudaDriverApi.cuMemAlloc_v2(out dPartialMax, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialSum, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialOut, (nuint)outPartialBytes).ThrowOnError();

            long before = CudaAttentionMmaDecodeGqaSplit.DispatchCount;
            kernel.Run(dQ, dK, dV, dOut, seqKv, numHeads, numKvHeads, kvSplit,
                dPartialMax, dPartialSum, dPartialOut, stream.Handle);
            stream.Synchronize();
            Assert.Equal(before + 1, CudaAttentionMmaDecodeGqaSplit.DispatchCount);

            return DownloadF32(dOut, qElems);
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOut != 0) CudaDriverApi.cuMemFree_v2(dOut);
            if (dPartialMax != 0) CudaDriverApi.cuMemFree_v2(dPartialMax);
            if (dPartialSum != 0) CudaDriverApi.cuMemFree_v2(dPartialSum);
            if (dPartialOut != 0) CudaDriverApi.cuMemFree_v2(dPartialOut);
        }
    }

    private (CudaContext ctx, CudaStream stream, CudaKernels kernels, CudaAttentionMmaDecodeGqaSplit kernel)?
        TrySetUp(out string? skipReason)
    {
        skipReason = null;
        if (!IsCudaDriverPresent()) { skipReason = "No CUDA GPU available"; return null; }
        string? ptxDir = FindPtxDir();
        if (ptxDir == null) { skipReason = "PTX files not found"; return null; }

        var ctx = CudaContext.Create(0);
        var stream = CudaStream.Create();
        var kernels = new CudaKernels(ptxDir);
        if (!kernels.HasAttentionMmaDecodeGqaSplit)
        {
            skipReason = "attention_flash_mma_decode_gqa_split not present in PTX (stale build / pre-Ampere GPU)";
            stream.Dispose(); kernels.Dispose(); ctx.Dispose();
            return null;
        }
        var kernel = new CudaAttentionMmaDecodeGqaSplit(kernels);
        CudaAttentionMmaDecodeGqaSplit.Enabled = true;
        return (ctx, stream, kernels, kernel);
    }

    /// <summary>
    /// Single-decode-step correctness at the real Bonsai-27B shape, vs both the CPU oracle and
    /// the F32 GPU baseline, at the realistic decode depths this whole #197/#198/#199
    /// investigation cares about, using the SAME occupancy-tuned <c>kvSplit</c> real callers
    /// would get via <see cref="CudaAttentionMmaDecodeGqaSplit.ComputeSafeKvSplit"/> (not a
    /// fixed value) — so this exercises the actual cross-split combine path whenever the
    /// heuristic picks kvSplit&gt;1 on this GPU, not just the trivial kvSplit==1 case.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    public void MmaGqaSplit_MatchesCpuReferenceAndF32Baseline_AtBonsaiShape(int seqKv)
    {
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        Assert.True(kernel.CanUse(seqQ: 1, seqKv, slidingWindow: 0, NumHeads, NumKvHeads, HeadDim),
            "CanUse should be true for the real Bonsai shape once enabled.");

        int kvSplit = kernel.ComputeSafeKvSplit(NumKvHeads, Group, seqKv);
        Skip.IfNot(kvSplit >= 1, "Cooperative launch not safe for this shape on this GPU");

        var rng = new Random(0x199B ^ seqKv);
        int qElems = NumHeads * HeadDim;
        int kvElems = NumKvHeads * HeadDim;
        int positionOffset = seqKv - 1;

        float[] q = RandomVec(rng, qElems, magnitude: 1.0f);
        float[] k = RandomVec(rng, seqKv * kvElems, magnitude: 1.0f);
        float[] v = RandomVec(rng, seqKv * kvElems, magnitude: 1.0f);

        float[] cpuOut = new float[qElems];
        fixed (float* pq = q, pk = k, pv = v, pOut = cpuOut)
            Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, NumHeads, NumKvHeads, HeadDim,
                positionOffset, pool: null, slidingWindowSize: null);

        nint dQf32 = 0, dK = 0, dV = 0, dOutF32Baseline = 0;
        try
        {
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQf32, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutF32Baseline, (nuint)qBytes).ThrowOnError();
            UploadF32(dQf32, q); UploadF32(dK, k); UploadF32(dV, v);

            kernels.LaunchAttentionF32(dQf32, dK, dV, dOutF32Baseline, seqQ: 1, seqKv, NumHeads, NumKvHeads, HeadDim,
                positionOffset, slidingWindow: 0, stream.Handle);
            stream.Synchronize();
            float[] f32Baseline = DownloadF32(dOutF32Baseline, qElems);

            float[] mmaOut = RunMmaGqaSplit(kernel, stream, ToHalf(q), ToHalf(k), ToHalf(v), seqKv, NumHeads, NumKvHeads, kvSplit);

            double maxAbsVsCpu = 0, maxRelVsCpu = 0, maxAbsVsF32 = 0, maxRelVsF32 = 0;
            int failVsCpu = 0, failVsF32 = 0;
            for (int i = 0; i < qElems; i++)
            {
                Assert.False(float.IsNaN(mmaOut[i]) || float.IsInfinity(mmaOut[i]), $"NaN/Inf at index {i}");

                double absCpu = Math.Abs((double)mmaOut[i] - cpuOut[i]);
                double relCpu = absCpu / (Math.Abs((double)cpuOut[i]) + 1e-6);
                if (absCpu > maxAbsVsCpu) maxAbsVsCpu = absCpu;
                if (relCpu > maxRelVsCpu) maxRelVsCpu = relCpu;
                if (!(absCpu <= AbsTol || relCpu <= RelTol)) failVsCpu++;

                double absF32 = Math.Abs((double)mmaOut[i] - f32Baseline[i]);
                double relF32 = absF32 / (Math.Abs((double)f32Baseline[i]) + 1e-6);
                if (absF32 > maxAbsVsF32) maxAbsVsF32 = absF32;
                if (relF32 > maxRelVsF32) maxRelVsF32 = relF32;
                if (!(absF32 <= AbsTol || relF32 <= RelTol)) failVsF32++;
            }

            _out.WriteLine($"seqKv={seqKv} kvSplit={kvSplit}: vsCPU maxAbs={maxAbsVsCpu:E3} maxRel={maxRelVsCpu:E3} fail={failVsCpu}/{qElems} | " +
                $"vsF32baseline maxAbs={maxAbsVsF32:E3} maxRel={maxRelVsF32:E3} fail={failVsF32}/{qElems} (tol abs {AbsTol} OR rel {RelTol})");

            Assert.True(failVsCpu == 0,
                $"mma-gqa-split vs CPU oracle: {failVsCpu}/{qElems} outside tolerance; maxAbs={maxAbsVsCpu:E3} maxRel={maxRelVsCpu:E3}");
            Assert.True(failVsF32 == 0,
                $"mma-gqa-split vs F32 GPU baseline: {failVsF32}/{qElems} outside tolerance; maxAbs={maxAbsVsF32:E3} maxRel={maxRelVsF32:E3}");
        }
        finally
        {
            if (dQf32 != 0) CudaDriverApi.cuMemFree_v2(dQf32);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOutF32Baseline != 0) CudaDriverApi.cuMemFree_v2(dOutF32Baseline);
        }
    }

    /// <summary>
    /// Three-way parity check: this kernel vs <c>attention_f32_gqa_split_kv</c> (issues
    /// #197/#198), forcing BOTH kernels to <c>kvSplit==1</c> so this isolates the "does the
    /// M-dim-packing / tensor-core axis agree with the register-blocked-FP32 axis at the same
    /// grid shape" question from any cross-split reassociation. Not expected to be bit-exact
    /// (different dtype AND different accumulation order — packed mma vs per-head scalar loop)
    /// but should agree within this suite's standard 5e-3 bar, same as the CPU/F32-baseline
    /// checks above.
    /// </summary>
    [SkippableTheory]
    [InlineData(256)]
    [InlineData(1024)]
    public void MmaGqaSplit_KvSplit1_AgreesWithF32GqaSplitKernel(int seqKv)
    {
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX");
        int maxSafeF32 = kernels.MaxSafeAttentionGqaSplit(NumKvHeads, HeadDim, Group);
        Skip.IfNot(maxSafeF32 >= 1, "F32 GQA-split cooperative launch not safe on this GPU");
        int maxSafeMma = kernels.MaxSafeAttentionMmaDecodeGqaSplit(NumKvHeads, HeadDim, Group);
        Skip.IfNot(maxSafeMma >= 1, "mma GQA-split cooperative launch not safe on this GPU");

        var rng = new Random(0x199C ^ seqKv);
        int qElems = NumHeads * HeadDim;
        int kvElems = NumKvHeads * HeadDim;
        int positionOffset = seqKv - 1;

        float[] q = RandomVec(rng, qElems);
        float[] k = RandomVec(rng, seqKv * kvElems);
        float[] v = RandomVec(rng, seqKv * kvElems);

        // F32 GQA-split kernel, kvSplit=1.
        nint dQ = 0, dK = 0, dV = 0, dOut = 0, dPM = 0, dPS = 0, dPO = 0;
        float[] f32GqaOut;
        try
        {
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOut, (nuint)qBytes).ThrowOnError();
            long scalarBytes = (long)NumHeads * 1 * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dPM, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPS, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPO, (nuint)(scalarBytes * HeadDim)).ThrowOnError();
            UploadF32(dQ, q); UploadF32(dK, k); UploadF32(dV, v);

            kernels.LaunchAttentionF32GqaSplit(dQ, dK, dV, dOut, seqKv, NumHeads, NumKvHeads, HeadDim,
                positionOffset, slidingWindow: 0, kvSplit: 1, dPM, dPS, dPO, stream.Handle);
            stream.Synchronize();
            f32GqaOut = DownloadF32(dOut, qElems);
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOut != 0) CudaDriverApi.cuMemFree_v2(dOut);
            if (dPM != 0) CudaDriverApi.cuMemFree_v2(dPM);
            if (dPS != 0) CudaDriverApi.cuMemFree_v2(dPS);
            if (dPO != 0) CudaDriverApi.cuMemFree_v2(dPO);
        }

        float[] mmaOut = RunMmaGqaSplit(kernel, stream, ToHalf(q), ToHalf(k), ToHalf(v), seqKv, NumHeads, NumKvHeads, kvSplit: 1);

        double maxAbs = 0, maxRel = 0; int fail = 0;
        for (int i = 0; i < qElems; i++)
        {
            double abs = Math.Abs((double)mmaOut[i] - f32GqaOut[i]);
            double rel = abs / (Math.Abs((double)f32GqaOut[i]) + 1e-6);
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
            if (!(abs <= AbsTol || rel <= RelTol)) fail++;
        }
        _out.WriteLine($"seqKv={seqKv}: mma-gqa-split vs attention_f32_gqa_split_kv (both kvSplit=1): maxAbs={maxAbs:E3} maxRel={maxRel:E3} fail={fail}/{qElems}");
        Assert.True(fail == 0, $"{fail}/{qElems} outside tolerance vs attention_f32_gqa_split_kv; maxAbs={maxAbs:E3} maxRel={maxRel:E3}");
    }

    /// <summary>
    /// Many consecutive decode steps (seqKv growing by one appended row per step, matching real
    /// generation) — same methodology and expectation as v1's equivalent test and
    /// <c>CudaAttentionF32SplitKvTests.AttentionF32SplitKv_ManyStepsDoesNotCompoundDrift</c>:
    /// attention recomputes from the exact, unperturbed KV cache each step, so max diff should
    /// stay roughly flat as steps/seqKv grow, not compound. Uses kvSplit=1 throughout (seqKv is
    /// small for most of this run, where cooperative-launch co-residency for large kvSplit may
    /// not be needed/safe) — the cross-split combine path is separately covered by the
    /// depth-swept parity test above.
    /// </summary>
    [SkippableFact]
    public void MmaGqaSplit_ManyStepsDoesNotCompoundDrift()
    {
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        const int steps = 300;
        const int startSeqKv = 8;
        int maxSeqKv = startSeqKv + steps;
        int qElems = NumHeads * HeadDim;
        int kvElems = NumKvHeads * HeadDim;

        var rng = new Random(0x199BEEF2);
        float[] kFull = RandomVec(rng, maxSeqKv * kvElems);
        float[] vFull = RandomVec(rng, maxSeqKv * kvElems);
        ushort[] kFullF16 = ToHalf(kFull);
        ushort[] vFullF16 = ToHalf(vFull);

        double firstMaxAbs = -1, lastMaxAbs = -1, worstMaxAbs = 0;
        for (int step = 0; step < steps; step++)
        {
            int seqKv = startSeqKv + step;
            float[] q = RandomVec(rng, qElems);

            float[] kSlice = kFull[..(seqKv * kvElems)];
            float[] vSlice = vFull[..(seqKv * kvElems)];
            float[] cpuOut = new float[qElems];
            fixed (float* pq = q, pk = kSlice, pv = vSlice, pOut = cpuOut)
                Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, NumHeads, NumKvHeads, HeadDim,
                    positionOffset: seqKv - 1, pool: null, slidingWindowSize: null);

            ushort[] kSliceF16 = kFullF16[..(seqKv * kvElems)];
            ushort[] vSliceF16 = vFullF16[..(seqKv * kvElems)];
            float[] gpuOut = RunMmaGqaSplit(kernel, stream, ToHalf(q), kSliceF16, vSliceF16, seqKv, NumHeads, NumKvHeads, kvSplit: 1);

            double maxAbs = 0;
            for (int i = 0; i < qElems; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs((double)gpuOut[i] - cpuOut[i]));

            if (step == 0) firstMaxAbs = maxAbs;
            if (step == steps - 1) lastMaxAbs = maxAbs;
            worstMaxAbs = Math.Max(worstMaxAbs, maxAbs);

            // Same loose-bound convention as v1: this test's target is COMPOUNDING, not
            // per-step tightness against the 5e-3 production bar (that is the depth-swept
            // parity test's job at realistic depths). Small seqKv falls in the same "few
            // competing keys amplify softmax sensitivity" transitional band v1 documented.
            const double stepTol = 2e-2;
            Assert.True(maxAbs <= stepTol, $"step {step} (seqKv={seqKv}): maxAbs {maxAbs:E3} exceeded {stepTol}");
        }

        _out.WriteLine($"{steps} steps, seqKv {startSeqKv}..{maxSeqKv - 1}: firstStepMaxAbs={firstMaxAbs:E3} " +
            $"lastStepMaxAbs={lastMaxAbs:E3} worstMaxAbs={worstMaxAbs:E3} " +
            "(flat-not-compounding expectation: last should not be orders of magnitude above first)");

        Assert.True(lastMaxAbs < firstMaxAbs * 20 + 1e-3,
            $"Drift appears to compound across steps: first={firstMaxAbs:E3} last={lastMaxAbs:E3} " +
            "(attention has no persistent approximate state between decode steps, so this would indicate a real bug).");
    }

    /// <summary>
    /// <see cref="CudaAttentionMmaDecodeGqaSplit.CanUse"/> gate sanity: rejects prefill
    /// (seqQ&gt;1), wrong headDim, sliding window, non-dividing GQA, and empty cache, so a
    /// caller can never accidentally launch the kernel outside the shape it was compiled/
    /// validated for.
    /// </summary>
    [SkippableFact]
    public void CanUse_RejectsShapesOutsideScope()
    {
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        Assert.True(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 0, NumHeads, NumKvHeads, HeadDim));
        Assert.False(kernel.CanUse(seqQ: 8, seqKv: 256, slidingWindow: 0, NumHeads, NumKvHeads, HeadDim), "prefill (seqQ>1) must not engage");
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 0, NumHeads, NumKvHeads, headDim: 64), "wrong headDim must not engage");
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 128, NumHeads, NumKvHeads, HeadDim), "sliding window must not engage");
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 0, numHeads: 25, NumKvHeads, HeadDim), "non-dividing GQA must not engage");
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 0, slidingWindow: 0, NumHeads, NumKvHeads, HeadDim), "empty cache must not engage");
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 0, numHeads: 72, numKvHeads: 8, HeadDim), "group=9 > MaxGroup=8 must not engage");

        CudaAttentionMmaDecodeGqaSplit.Enabled = false;
        Assert.False(kernel.CanUse(seqQ: 1, seqKv: 256, slidingWindow: 0, NumHeads, NumKvHeads, HeadDim), "disabled toggle must not engage");
    }

    /// <summary>
    /// Logs the real co-residency ceiling / chosen kvSplit this GPU gets for both the FP32
    /// GQA-split kernel and this composed tensor-core kernel at the real Bonsai shape --
    /// informative (not hard-asserted beyond "a positive split exists"), since the whole point
    /// of v2 is that this kernel's occupancy story should be at least as good as the sibling's.
    /// </summary>
    [SkippableFact]
    public void OccupancyCeiling_ReportedAtBonsaiShape()
    {
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        int seqKv = 512;
        int maxSafeMma = kernels.MaxSafeAttentionMmaDecodeGqaSplit(NumKvHeads, HeadDim, Group);
        int kvSplitMma = kernel.ComputeSafeKvSplit(NumKvHeads, Group, seqKv);

        string f32Line = "attention_f32_gqa_split_kv not present in PTX";
        if (kernels.HasAttentionF32GqaSplitKv)
        {
            int maxSafeF32 = kernels.MaxSafeAttentionGqaSplit(NumKvHeads, HeadDim, Group);
            int kvSplitF32 = CudaKernels.ComputeAttentionKvSplit(seqKv, NumKvHeads, maxSafeF32);
            f32Line = $"attention_f32_gqa_split_kv: maxSafeSplit={maxSafeF32} chosenKvSplit={kvSplitF32} grid=({NumKvHeads},{kvSplitF32})={NumKvHeads * kvSplitF32} blocks, blockDim=256";
        }

        _out.WriteLine($"Bonsai shape (numKvHeads={NumKvHeads}, headDim={HeadDim}, group={Group}), seqKv={seqKv}:");
        _out.WriteLine($"  {f32Line}");
        _out.WriteLine($"  attention_flash_mma_decode_gqa_split: maxSafeSplit={maxSafeMma} chosenKvSplit={kvSplitMma} grid=({NumKvHeads},{kvSplitMma})={NumKvHeads * kvSplitMma} blocks, blockDim={CudaKernels.AttentionMmaDecodeGqaSplitBlockSize}");

        Assert.True(maxSafeMma >= 1, "Composed tensor-core kernel should be launchable at the real Bonsai shape on an Ampere+ GPU.");
    }

    private static double TimeGpu(CudaStream stream, Action work)
    {
        CudaDriverApi.cuEventCreate(out nint start, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint stop, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
        try
        {
            CudaDriverApi.cuEventRecord(start, stream.Handle).ThrowOnError();
            work();
            CudaDriverApi.cuEventRecord(stop, stream.Handle).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(stop).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, start, stop).ThrowOnError();
            return ms;
        }
        finally
        {
            CudaDriverApi.cuEventDestroy_v2(start);
            CudaDriverApi.cuEventDestroy_v2(stop);
        }
    }

    /// <summary>
    /// THREE-WAY interleaved wall-clock A/B: <c>attention_f32</c> (shipping baseline) vs
    /// <c>attention_f32_gqa_split_kv</c> (issues #197/#198) vs this kernel (issue #199 v2), at
    /// the real Bonsai-27B shape, using the corrected methodology this whole investigation
    /// settled on (docs/CUDA.md's "2026-07-30 re-profile" section and v1's own README):
    /// interleaved reps within one warmed process (not blocked/separate runs — this project's
    /// documented lesson is that the 3060's clocks drift across separate blocked runs), min-of-
    /// reps, and depths that actually matter for decode (256/512/1024/2048). Opt-in via
    /// <c>DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT_BENCH=1</c>.
    /// </summary>
    [SkippableFact]
    public void TimingThreeWayVsF32BaselineAndGqaSplit()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT_BENCH=1 not set.");
        var setup = TrySetUp(out string? skip);
        Skip.If(setup == null, skip);
        using var ctx = setup!.Value.ctx;
        using var stream = setup.Value.stream;
        using var kernels = setup.Value.kernels;
        var kernel = setup.Value.kernel;

        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX");

        int[] depths = { 256, 512, 1024, 2048 };
        int reps = 30, warmup = 6;
        int qElems = NumHeads * HeadDim;
        int kvElems = NumKvHeads * HeadDim;

        _out.WriteLine("attention_f32 (baseline) vs attention_f32_gqa_split_kv (#197/#198) vs attention_flash_mma_decode_gqa_split (this issue, v2).");
        _out.WriteLine($"numHeads={NumHeads} numKvHeads={NumKvHeads} headDim={HeadDim} group={Group} reps={reps} (min ms, interleaved)");
        _out.WriteLine($"{"seqKv",6} | {"f32base",10} | {"f32gqa",10} | {"mmaGqaSplit",12} | {"f32/mma",8} | {"gqa/mma",8}");
        _out.WriteLine(new string('-', 70));

        foreach (int seqKv in depths)
        {
            var rng = new Random(seqKv);
            float[] q = RandomVec(rng, qElems);
            float[] k = RandomVec(rng, seqKv * kvElems);
            float[] v = RandomVec(rng, seqKv * kvElems);
            ushort[] qF16 = ToHalf(q), kF16 = ToHalf(k), vF16 = ToHalf(v);

            int group = Group;
            int maxSafeF32 = kernels.MaxSafeAttentionGqaSplit(NumKvHeads, HeadDim, group);
            int kvSplitF32 = maxSafeF32 >= 1 ? CudaKernels.ComputeAttentionKvSplit(seqKv, NumKvHeads, maxSafeF32) : 1;
            int kvSplitMma = kernel.ComputeSafeKvSplit(NumKvHeads, group, seqKv);
            if (maxSafeF32 < 1 || kvSplitMma < 1)
            {
                _out.WriteLine($"{seqKv,6} | SKIPPED (cooperative launch not safe for one or both split kernels on this GPU)");
                continue;
            }

            nint dQf32 = 0, dK = 0, dV = 0, dOutF32 = 0;
            nint dPMf32 = 0, dPSf32 = 0, dPOf32 = 0;
            nint dQf16 = 0, dKf16 = 0, dVf16 = 0, dOutMma = 0;
            nint dPMmma = 0, dPSmma = 0, dPOmma = 0;
            try
            {
                CudaDriverApi.cuMemAlloc_v2(out dQf32, (nuint)(qElems * sizeof(float))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)(seqKv * kvElems * sizeof(float))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)(seqKv * kvElems * sizeof(float))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dOutF32, (nuint)(qElems * sizeof(float))).ThrowOnError();
                UploadF32(dQf32, q); UploadF32(dK, k); UploadF32(dV, v);

                long f32ScalarBytes = (long)NumHeads * kvSplitF32 * sizeof(float);
                CudaDriverApi.cuMemAlloc_v2(out dPMf32, (nuint)f32ScalarBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dPSf32, (nuint)f32ScalarBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dPOf32, (nuint)(f32ScalarBytes * HeadDim)).ThrowOnError();

                CudaDriverApi.cuMemAlloc_v2(out dQf16, (nuint)(qElems * sizeof(ushort))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dKf16, (nuint)(seqKv * kvElems * sizeof(ushort))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dVf16, (nuint)(seqKv * kvElems * sizeof(ushort))).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dOutMma, (nuint)(qElems * sizeof(float))).ThrowOnError();
                UploadF16(dQf16, qF16); UploadF16(dKf16, kF16); UploadF16(dVf16, vF16);

                long mmaScalarBytes = (long)NumHeads * kvSplitMma * sizeof(float);
                CudaDriverApi.cuMemAlloc_v2(out dPMmma, (nuint)mmaScalarBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dPSmma, (nuint)mmaScalarBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dPOmma, (nuint)(mmaScalarBytes * HeadDim)).ThrowOnError();

                int positionOffset = seqKv - 1;
                void RunF32() => kernels.LaunchAttentionF32(dQf32, dK, dV, dOutF32, seqQ: 1, seqKv,
                    NumHeads, NumKvHeads, HeadDim, positionOffset, slidingWindow: 0, stream.Handle);
                void RunF32Gqa() => kernels.LaunchAttentionF32GqaSplit(dQf32, dK, dV, dOutF32, seqKv,
                    NumHeads, NumKvHeads, HeadDim, positionOffset, slidingWindow: 0, kvSplitF32,
                    dPMf32, dPSf32, dPOf32, stream.Handle);
                void RunMma() => kernel.Run(dQf16, dKf16, dVf16, dOutMma, seqKv, NumHeads, NumKvHeads,
                    kvSplitMma, dPMmma, dPSmma, dPOmma, stream.Handle);

                for (int w = 0; w < warmup; w++) { RunF32(); RunF32Gqa(); RunMma(); }
                stream.Synchronize();

                double f32Min = double.MaxValue, gqaMin = double.MaxValue, mmaMin = double.MaxValue;
                for (int r = 0; r < reps; r++)
                {
                    f32Min = Math.Min(f32Min, TimeGpu(stream, RunF32));
                    gqaMin = Math.Min(gqaMin, TimeGpu(stream, RunF32Gqa));
                    mmaMin = Math.Min(mmaMin, TimeGpu(stream, RunMma));
                }

                _out.WriteLine($"{seqKv,6} | {f32Min,7:F4}ms | {gqaMin,7:F4}ms | {mmaMin,9:F4}ms | {f32Min / mmaMin,6:F2}x | {gqaMin / mmaMin,6:F2}x");
            }
            finally
            {
                if (dQf32 != 0) CudaDriverApi.cuMemFree_v2(dQf32);
                if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
                if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
                if (dOutF32 != 0) CudaDriverApi.cuMemFree_v2(dOutF32);
                if (dPMf32 != 0) CudaDriverApi.cuMemFree_v2(dPMf32);
                if (dPSf32 != 0) CudaDriverApi.cuMemFree_v2(dPSf32);
                if (dPOf32 != 0) CudaDriverApi.cuMemFree_v2(dPOf32);
                if (dQf16 != 0) CudaDriverApi.cuMemFree_v2(dQf16);
                if (dKf16 != 0) CudaDriverApi.cuMemFree_v2(dKf16);
                if (dVf16 != 0) CudaDriverApi.cuMemFree_v2(dVf16);
                if (dOutMma != 0) CudaDriverApi.cuMemFree_v2(dOutMma);
                if (dPMmma != 0) CudaDriverApi.cuMemFree_v2(dPMmma);
                if (dPSmma != 0) CudaDriverApi.cuMemFree_v2(dPSmma);
                if (dPOmma != 0) CudaDriverApi.cuMemFree_v2(dPOmma);
            }
        }
        _out.WriteLine(new string('-', 70));
        _out.WriteLine("f32/mma and gqa/mma > 1 = mma-gqa-split (this issue) is faster.");
    }
}
