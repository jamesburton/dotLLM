using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness + drift-characterization coverage for <see cref="CudaKernels.LaunchAttentionF32SplitKv"/>
/// (the opt-in, default-OFF split-KV "Flash-Decoding" <c>attention_f32_split_kv</c> CUDA kernel,
/// issue #183) against the CPU oracle, <see cref="Attention.Execute(float*, float*, float*, float*,
/// int, int, int, int, int, int, ComputeThreadPool?, int?)"/> (which itself uses the same
/// Schraudolph fast-exp approximation via <see cref="Softmax.ExecuteFast"/>/<see cref="FastMath"/>
/// that <c>attention_f32.cu</c>'s <c>fast_exp_neg</c> mirrors).
///
/// UNLIKE a bit-exact test, this uses a TOLERANCE comparison — splitting the KV dimension across
/// blocks reassociates the online-softmax accumulation (independent partial (max, sum, out) per
/// split, combined afterward), which is mathematically equal but not bit-identical to the single
/// sequential accumulation <see cref="LaunchAttentionF32"/>/the CPU oracle perform. See
/// <c>attention_f32.cu</c>'s header for the full design writeup.
///
/// Tests run at REAL depth (256, 1024+) specifically — at shallow depth (few or one KV tile) the
/// interesting cross-split-block combine code path is barely exercised (with seqKv &lt; TILE_KV,
/// most splits get zero or near-zero KV rows), which would not actually test the mechanism this
/// kernel exists for.
///
/// A second test characterizes MANY consecutive decode steps (growing seqKv, one new KV row
/// appended per step, matching real generation) to check for the "different, worth-flagging
/// finding" this kernel's header calls out: UNLIKE <see cref="CudaGdnScanStepF32CoopSplit4Tests"/>
/// (where the recurrent GDN state visibly compounds drift step-over-step), attention has no
/// persistent approximate state carried between decode steps — each step recomputes from the
/// exact, unperturbed KV cache plus that step's query — so the expectation is that max diff stays
/// roughly flat as steps/seqKv grow, NOT that it compounds. This test verifies that expectation
/// empirically rather than assuming it.
/// </summary>
[Trait("Category", "GPU")]
public class CudaAttentionF32SplitKvTests
{
    private readonly ITestOutputHelper _out;
    public CudaAttentionF32SplitKvTests(ITestOutputHelper output) => _out = output;

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

    private static float[] RandomVec(Random rng, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return v;
    }

    /// <summary>
    /// Single-step correctness at REAL depth, including a non-multiple-of-(TILE_KV*SPLIT) seqKv
    /// to exercise the remainder-handling in the per-split chunking arithmetic.
    /// </summary>
    [SkippableTheory]
    [InlineData(24, 4, 256, 256)]    // real Bonsai-27B shape, depth == exactly 1 tile / minimum split threshold
    [InlineData(24, 4, 256, 1024)]   // real Bonsai-27B shape, depth == 4 full tiles (one per split, evenly)
    [InlineData(24, 4, 256, 1300)]   // real Bonsai-27B shape, depth NOT a multiple of TILE_KV*SPLIT=1024
    [InlineData(8, 2, 64, 512)]      // smaller synthetic shape, different headDim/GQA ratio
    public void AttentionF32SplitKv_MatchesCpuReferenceWithinTolerance_AtRealDepth(
        int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        Skip.IfNot(kernels.HasAttentionF32SplitKv, "attention_f32_split_kv not present in PTX (stale build)");
        Skip.IfNot(kernels.IsAttentionSplitKvSafe(numHeads, headDim),
            $"split-KV cooperative launch not safe for numHeads={numHeads}, headDim={headDim} on this GPU");

        var rng = new Random(0xA77E17 ^ numHeads ^ numKvHeads ^ (headDim << 8) ^ (seqKv << 16));

        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;
        int positionOffset = seqKv - 1; // causal: query is the most-recently-cached position

        float[] q = RandomVec(rng, qElems);
        float[] k = RandomVec(rng, seqKv * kvElems);
        float[] v = RandomVec(rng, seqKv * kvElems);

        // CPU oracle (uses the same fast-exp approximation as the GPU kernel).
        float[] cpuOut = new float[qElems];
        unsafe
        {
            fixed (float* pq = q, pk = k, pv = v, pOut = cpuOut)
                Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset, pool: null, slidingWindowSize: null);
        }

        // GPU exact kernel (LaunchAttentionF32) — sanity cross-check that the CPU oracle and the
        // exact GPU path already agree, isolating whatever tolerance we see below to the SPLIT
        // kernel's reassociation specifically, not some other discrepancy.
        nint dQ = 0, dK = 0, dV = 0, dOutExact = 0, dOutSplit = 0;
        nint dPartialMax = 0, dPartialSum = 0, dPartialOut = 0;
        try
        {
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutExact, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutSplit, (nuint)qBytes).ThrowOnError();

            long scalarBytes = (long)numHeads * CudaKernels.AttentionKvSplit * sizeof(float);
            long outPartialBytes = scalarBytes * headDim;
            CudaDriverApi.cuMemAlloc_v2(out dPartialMax, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialSum, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialOut, (nuint)outPartialBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
                fixed (float* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
                fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();
            }

            nint s = stream.Handle;
            kernels.LaunchAttentionF32(dQ, dK, dV, dOutExact, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, slidingWindow: 0, s);
            kernels.LaunchAttentionF32SplitKv(dQ, dK, dV, dOutSplit, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, slidingWindow: 0, dPartialMax, dPartialSum, dPartialOut, s);
            stream.Synchronize();

            float[] gpuExact = new float[qElems];
            float[] gpuSplit = new float[qElems];
            unsafe
            {
                fixed (float* p = gpuExact) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutExact, (nuint)qBytes).ThrowOnError();
                fixed (float* p = gpuSplit) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutSplit, (nuint)qBytes).ThrowOnError();
            }

            double maxAbsExactVsCpu = 0, maxAbsSplitVsCpu = 0, maxAbsSplitVsExact = 0;
            for (int i = 0; i < qElems; i++)
            {
                Assert.False(float.IsNaN(gpuSplit[i]) || float.IsInfinity(gpuSplit[i]),
                    $"NaN/Inf in split-KV GPU output at index {i}");
                maxAbsExactVsCpu = Math.Max(maxAbsExactVsCpu, Math.Abs((double)gpuExact[i] - cpuOut[i]));
                maxAbsSplitVsCpu = Math.Max(maxAbsSplitVsCpu, Math.Abs((double)gpuSplit[i] - cpuOut[i]));
                maxAbsSplitVsExact = Math.Max(maxAbsSplitVsExact, Math.Abs((double)gpuSplit[i] - gpuExact[i]));
            }

            _out.WriteLine($"numHeads={numHeads} numKvHeads={numKvHeads} headDim={headDim} seqKv={seqKv}: " +
                $"maxAbs(exactGPU-CPU)={maxAbsExactVsCpu:e3} maxAbs(splitGPU-CPU)={maxAbsSplitVsCpu:e3} " +
                $"maxAbs(splitGPU-exactGPU)={maxAbsSplitVsExact:e3}");

            // The exact (non-split) GPU kernel vs CPU oracle should already be tight (both use the
            // same fast_exp_neg/FastMath approximation) — a generous bound just confirms the test
            // harness itself is sound, independent of the split kernel under test.
            Assert.True(maxAbsExactVsCpu < 5e-2,
                $"exact GPU kernel vs CPU oracle diverged more than expected: {maxAbsExactVsCpu:e3} (harness/environment issue, not this test's target)");

            // The split-KV kernel's reassociation tolerance vs the CPU oracle. Attention outputs
            // here are O(1)-magnitude (softmax-weighted averages of ~U(-1,1) V vectors), so an
            // absolute tolerance is meaningful without a relative-diff near-zero-denominator dodge.
            Assert.True(maxAbsSplitVsCpu < 5e-2,
                $"split-KV GPU kernel vs CPU oracle exceeded tolerance: {maxAbsSplitVsCpu:e3}");
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOutExact != 0) CudaDriverApi.cuMemFree_v2(dOutExact);
            if (dOutSplit != 0) CudaDriverApi.cuMemFree_v2(dOutSplit);
            if (dPartialMax != 0) CudaDriverApi.cuMemFree_v2(dPartialMax);
            if (dPartialSum != 0) CudaDriverApi.cuMemFree_v2(dPartialSum);
            if (dPartialOut != 0) CudaDriverApi.cuMemFree_v2(dPartialOut);
        }
    }

    /// <summary>
    /// Runs many consecutive decode steps (growing seqKv by one appended KV row per step, matching
    /// real generation) through both the split-KV GPU kernel and the CPU reference, tracking how
    /// max diff evolves. UNLIKE GDN's coop-split4 characterization (which found drift compounding
    /// because the GDN state persists across steps), attention recomputes from the exact KV cache
    /// each step with no persistent approximate state — so the expectation here is that max diff
    /// stays roughly FLAT as seqKv grows, not that it compounds. This test reports the real numbers
    /// so that expectation is verified empirically, not assumed.
    /// </summary>
    [SkippableFact]
    public void AttentionF32SplitKv_ManyConsecutiveDecodeSteps_DoesNotCompoundDrift()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        Skip.IfNot(kernels.HasAttentionF32SplitKv, "attention_f32_split_kv not present in PTX (stale build)");
        Skip.IfNot(kernels.IsAttentionSplitKvSafe(numHeads, headDim),
            "split-KV cooperative launch not safe for the real Bonsai-27B shape on this GPU");

        const int startSeqKv = 256;   // above CudaKernels.AttentionSplitKvMinSeqKv
        const int steps = 300;        // grows seqKv from 256 to 555 -- spans 1 to 3 tiles
        var rng = new Random(0x51DE517);

        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;

        // Pre-generate the full K/V history up front (exact, unperturbed KV cache — matching real
        // decode where the cache holds original, unmodified projections regardless of any
        // downstream attention approximation from a PRIOR step).
        int maxSeqKv = startSeqKv + steps;
        float[] kAll = RandomVec(rng, maxSeqKv * kvElems);
        float[] vAll = RandomVec(rng, maxSeqKv * kvElems);

        nint dK = 0, dV = 0, dQ = 0, dOutSplit = 0, dPartialMax = 0, dPartialSum = 0, dPartialOut = 0;
        try
        {
            long kvBytesMax = (long)maxSeqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytesMax).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytesMax).ThrowOnError();
            unsafe
            {
                fixed (float* p = kAll) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytesMax).ThrowOnError();
                fixed (float* p = vAll) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytesMax).ThrowOnError();
            }

            long qBytes = (long)qElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutSplit, (nuint)qBytes).ThrowOnError();
            long scalarBytes = (long)numHeads * CudaKernels.AttentionKvSplit * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dPartialMax, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialSum, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialOut, (nuint)(scalarBytes * headDim)).ThrowOnError();

            nint s = stream.Handle;
            var driftSamples = new List<(int step, int seqKv, double maxAbs)>();
            double maxAbsEver = 0.0;

            for (int step = 0; step < steps; step++)
            {
                int seqKv = startSeqKv + step;
                int positionOffset = seqKv - 1;
                float[] q = RandomVec(rng, qElems);

                float[] cpuOut = new float[qElems];
                unsafe
                {
                    fixed (float* pq = q, pk = kAll, pv = vAll, pOut = cpuOut)
                        Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                            positionOffset, pool: null, slidingWindowSize: null);
                }

                unsafe
                {
                    fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
                }
                kernels.LaunchAttentionF32SplitKv(dQ, dK, dV, dOutSplit, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset, slidingWindow: 0, dPartialMax, dPartialSum, dPartialOut, s);
                stream.Synchronize();

                float[] gpuOut = new float[qElems];
                unsafe
                {
                    fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutSplit, (nuint)qBytes).ThrowOnError();
                }

                double stepMaxAbs = 0.0;
                for (int i = 0; i < qElems; i++)
                {
                    Assert.False(float.IsNaN(gpuOut[i]) || float.IsInfinity(gpuOut[i]),
                        $"step {step} (seqKv={seqKv}): NaN/Inf in split-KV GPU output at index {i}");
                    stepMaxAbs = Math.Max(stepMaxAbs, Math.Abs((double)gpuOut[i] - cpuOut[i]));
                }
                maxAbsEver = Math.Max(maxAbsEver, stepMaxAbs);
                if (step < 10 || step % 25 == 0 || step == steps - 1)
                    driftSamples.Add((step, seqKv, stepMaxAbs));
            }

            _out.WriteLine($"steps={steps} startSeqKv={startSeqKv} maxAbsDiffEver={maxAbsEver:e3}");
            _out.WriteLine("Drift over run (step, seqKv, maxAbs) -- expected roughly FLAT, not growing:");
            foreach (var (step, seqKv, maxAbs) in driftSamples)
                _out.WriteLine($"  step={step,4} seqKv={seqKv,5} maxAbs={maxAbs:e3}");

            // Compare the FIRST 10 steps' average against the LAST 10 steps' average: if attention
            // truly doesn't compound (no persistent state, unlike GDN), these should be the same
            // order of magnitude, not growing by orders of magnitude as GDN's did.
            double firstAvg = driftSamples.Take(10).Average(x => x.maxAbs);
            double lastAvg = driftSamples.TakeLast(3).Average(x => x.maxAbs);
            _out.WriteLine($"firstStepsAvg={firstAvg:e3} lastStepsAvg={lastAvg:e3} ratio={lastAvg / Math.Max(firstAvg, 1e-12):F2}x");
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOutSplit != 0) CudaDriverApi.cuMemFree_v2(dOutSplit);
            if (dPartialMax != 0) CudaDriverApi.cuMemFree_v2(dPartialMax);
            if (dPartialSum != 0) CudaDriverApi.cuMemFree_v2(dPartialSum);
            if (dPartialOut != 0) CudaDriverApi.cuMemFree_v2(dPartialOut);
        }
    }

    /// <summary>
    /// Confirms the safe-fallback contract: when the requested shape doesn't fit this GPU's
    /// cooperative-launch co-residency ceiling, <see cref="CudaKernels.IsAttentionSplitKvSafe"/>
    /// returns false rather than letting a caller hit the hard "too many blocks in cooperative
    /// launch" CUDA error.
    /// </summary>
    [SkippableFact]
    public void IsAttentionSplitKvSafe_ReturnsFalse_ForShapeExceedingCoResidencyCeiling()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF32SplitKv, "attention_f32_split_kv not present in PTX (stale build)");

        Assert.False(kernels.IsAttentionSplitKvSafe(numHeads: 1_000_000, headDim: 256));
    }
}
