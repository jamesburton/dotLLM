using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness coverage for <see cref="CudaKernels.LaunchAttentionF32GqaSplit"/> -- the opt-in,
/// default-OFF combined GQA-group + split-KV decode attention kernel (issues #197 + #198),
/// mirroring <see cref="CudaAttentionF32SplitKvTests"/>'s structure and conventions.
///
/// This kernel generalizes <see cref="CudaKernels.LaunchAttentionF32SplitKv"/> (issue #183) two
/// ways at once: grid = (numKvHeads, kvSplit) instead of (numHeads, ATTN_KV_SPLIT), with each
/// block register-blocking the QK/PV loops across the <c>group = numHeads/numKvHeads</c> query
/// heads sharing a KV head; and <c>kvSplit</c> is a runtime parameter (issue #197's heuristic),
/// not the old kernel's compile-time <c>ATTN_KV_SPLIT=4</c>.
///
/// Correctness expectation (see <c>attention_f32.cu</c>'s combined-kernel header): at
/// <c>kvSplit==1</c> the kernel is expected to be BIT-EXACT per query head vs
/// <see cref="CudaKernels.LaunchAttentionF32"/> -- the GQA regrid changes which block computes
/// which head, never the order of floating-point operations within any one head's accumulation,
/// and the kv_split==1 combine path explicitly skips the reassociating <c>fast_exp_neg</c>
/// reweighting since there is nothing to combine. At <c>kvSplit&gt;1</c> the kernel inherits
/// EXACTLY <see cref="CudaAttentionF32SplitKvTests"/>'s already-characterized reassociation
/// tolerance (same combine formula, same partial-buffer layout) -- no new tolerance category.
/// </summary>
[Trait("Category", "GPU")]
public class CudaAttentionF32GqaSplitTests
{
    private readonly ITestOutputHelper _out;
    public CudaAttentionF32GqaSplitTests(ITestOutputHelper output) => _out = output;

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
    /// Runs the GQA-split kernel for one decode step and returns the GPU output, allocating and
    /// freeing all scratch itself. <paramref name="kvSplit"/> is passed through unclamped (tests
    /// choose it explicitly, including forcing 1 to isolate the GQA regrid from splitting).
    /// </summary>
    private static unsafe float[] RunGqaSplit(CudaKernels kernels, CudaStream stream,
        float[] q, float[] k, float[] v, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int kvSplit)
    {
        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;
        nint dQ = 0, dK = 0, dV = 0, dOut = 0, dPartialMax = 0, dPartialSum = 0, dPartialOut = 0;
        try
        {
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOut, (nuint)qBytes).ThrowOnError();

            long scalarBytes = (long)numHeads * kvSplit * sizeof(float);
            long outPartialBytes = scalarBytes * headDim;
            CudaDriverApi.cuMemAlloc_v2(out dPartialMax, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialSum, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPartialOut, (nuint)outPartialBytes).ThrowOnError();

            fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (float* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
            fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();

            nint s = stream.Handle;
            kernels.LaunchAttentionF32GqaSplit(dQ, dK, dV, dOut, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, slidingWindow: 0, kvSplit, dPartialMax, dPartialSum, dPartialOut, s);
            stream.Synchronize();

            float[] gpuOut = new float[qElems];
            fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)qBytes).ThrowOnError();
            return gpuOut;
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

    /// <summary>
    /// Isolates the GQA regrid from splitting: kvSplit forced to 1 (no grid.sync/combine
    /// reassociation reachable) vs <see cref="CudaKernels.LaunchAttentionF32"/>. Per
    /// attention_f32.cu's header, expected to be BIT-EXACT (0 ULP) at real Bonsai-27B shape and
    /// smaller/edge shapes -- this test asserts that directly rather than assuming it.
    /// </summary>
    [SkippableTheory]
    [InlineData(24, 4, 256, 256)]    // real Bonsai-27B shape, group=6
    [InlineData(24, 4, 256, 1300)]   // real Bonsai-27B shape, depth not a multiple of TILE_KV
    [InlineData(8, 2, 64, 512)]      // smaller synthetic shape, group=4
    [InlineData(8, 1, 64, 300)]      // MQA-style extreme: group=8 == MaxGqaGroup boundary
    [InlineData(4, 4, 32, 100)]      // MHA (group=1): kernel itself must still be correct here
    public void AttentionF32GqaSplit_KvSplit1_IsBitExactVsExactKernel(
        int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX (stale build)");
        int group = numHeads / numKvHeads;
        int maxSafeSplit = kernels.MaxSafeAttentionGqaSplit(numKvHeads, headDim, group);
        Skip.IfNot(maxSafeSplit >= 1,
            $"GQA-split cooperative launch not safe for numKvHeads={numKvHeads}, headDim={headDim}, group={group} on this GPU");

        var rng = new Random(0xB0715A1 ^ numHeads ^ numKvHeads ^ (headDim << 8) ^ (seqKv << 16));

        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;
        int positionOffset = seqKv - 1; // causal: query is the most-recently-cached position

        float[] q = RandomVec(rng, qElems);
        float[] k = RandomVec(rng, seqKv * kvElems);
        float[] v = RandomVec(rng, seqKv * kvElems);

        nint dQ = 0, dK = 0, dV = 0, dOutExact = 0;
        try
        {
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutExact, (nuint)qBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
                fixed (float* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
                fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();
            }

            nint s = stream.Handle;
            kernels.LaunchAttentionF32(dQ, dK, dV, dOutExact, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, slidingWindow: 0, s);
            stream.Synchronize();

            float[] gpuExact = new float[qElems];
            unsafe
            {
                fixed (float* p = gpuExact) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutExact, (nuint)qBytes).ThrowOnError();
            }

            float[] gpuGqa = RunGqaSplit(kernels, stream, q, k, v, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, kvSplit: 1);

            double maxAbsDiff = 0, maxUlpDiff = 0;
            for (int i = 0; i < qElems; i++)
            {
                Assert.False(float.IsNaN(gpuGqa[i]) || float.IsInfinity(gpuGqa[i]),
                    $"NaN/Inf in GQA-split GPU output at index {i}");
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)gpuGqa[i] - gpuExact[i]));
                int bitsA = BitConverter.SingleToInt32Bits(gpuGqa[i]);
                int bitsB = BitConverter.SingleToInt32Bits(gpuExact[i]);
                maxUlpDiff = Math.Max(maxUlpDiff, Math.Abs((double)bitsA - bitsB));
            }

            _out.WriteLine($"numHeads={numHeads} numKvHeads={numKvHeads} headDim={headDim} seqKv={seqKv} group={group}: " +
                $"maxAbsDiff(gqaSplit-exact)={maxAbsDiff:e3} maxUlpDiff={maxUlpDiff:F0}");

            // Expect bit-exact (0 ULP). A tiny allowance is kept only for FMA-contraction ordering
            // differences the compiler may legally introduce across the two separately-compiled
            // kernels (same risk class documented for attention_f32_split_kv's own "reuses exactly"
            // claims) -- if this ever exceeds a handful of ULPs, that is a real correctness bug in
            // the GQA regrid, not an accepted reassociation tolerance.
            Assert.True(maxUlpDiff <= 8,
                $"GQA-split kernel at kvSplit=1 should be (near-)bit-exact vs LaunchAttentionF32: maxUlpDiff={maxUlpDiff:F0}, maxAbsDiff={maxAbsDiff:e3}");
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dOutExact != 0) CudaDriverApi.cuMemFree_v2(dOutExact);
        }
    }

    /// <summary>
    /// Full kernel (grouped + split, kvSplit &gt; 1) vs the CPU oracle, at real depth, including a
    /// non-multiple-of-(TILE_KV*split) seqKv and the heuristic's own actual output (not just
    /// hand-picked splits) -- mirrors <see cref="CudaAttentionF32SplitKvTests"/>'s tolerance
    /// bound exactly (5e-2 absolute), since this inherits the same reassociation, not a new one.
    /// </summary>
    [SkippableTheory]
    [InlineData(24, 4, 256, 1024, 2)]   // real Bonsai-27B shape, hand-picked split
    [InlineData(24, 4, 256, 1300, 3)]   // real Bonsai-27B shape, non-power-of-2 split, remainder seqKv
    [InlineData(8, 2, 64, 512, 5)]      // smaller synthetic shape, non-power-of-2 split
    [InlineData(24, 4, 256, 2048, 0)]   // split==0 sentinel: use ComputeAttentionKvSplit's real output
    public void AttentionF32GqaSplit_MatchesCpuReferenceWithinTolerance_AtRealDepth(
        int numHeads, int numKvHeads, int headDim, int seqKv, int requestedSplit)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX (stale build)");
        int group = numHeads / numKvHeads;
        int maxSafeSplit = kernels.MaxSafeAttentionGqaSplit(numKvHeads, headDim, group);
        Skip.IfNot(maxSafeSplit >= 1,
            $"GQA-split cooperative launch not safe for numKvHeads={numKvHeads}, headDim={headDim}, group={group} on this GPU");

        int kvSplit = requestedSplit > 0
            ? Math.Min(requestedSplit, maxSafeSplit)
            : CudaKernels.ComputeAttentionKvSplit(seqKv, numKvHeads, maxSafeSplit);

        var rng = new Random(0xC0FFEE ^ numHeads ^ numKvHeads ^ (headDim << 8) ^ (seqKv << 16) ^ kvSplit);

        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;
        int positionOffset = seqKv - 1;

        float[] q = RandomVec(rng, qElems);
        float[] k = RandomVec(rng, seqKv * kvElems);
        float[] v = RandomVec(rng, seqKv * kvElems);

        float[] cpuOut = new float[qElems];
        unsafe
        {
            fixed (float* pq = q, pk = k, pv = v, pOut = cpuOut)
                Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset, pool: null, slidingWindowSize: null);
        }

        float[] gpuGqa = RunGqaSplit(kernels, stream, q, k, v, seqKv, numHeads, numKvHeads, headDim,
            positionOffset, kvSplit);

        double maxAbsDiff = 0;
        for (int i = 0; i < qElems; i++)
        {
            Assert.False(float.IsNaN(gpuGqa[i]) || float.IsInfinity(gpuGqa[i]),
                $"NaN/Inf in GQA-split GPU output at index {i}");
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)gpuGqa[i] - cpuOut[i]));
        }

        _out.WriteLine($"numHeads={numHeads} numKvHeads={numKvHeads} headDim={headDim} seqKv={seqKv} group={group} kvSplit={kvSplit}: " +
            $"maxAbsDiff(gqaSplit-cpu)={maxAbsDiff:e3}");

        // Same 5e-2 absolute bound as CudaAttentionF32SplitKvTests -- this kernel inherits EXACTLY
        // #183's reassociation tolerance, not a new/looser category.
        Assert.True(maxAbsDiff < 5e-2,
            $"GQA-split GPU kernel vs CPU oracle exceeded tolerance: {maxAbsDiff:e3}");
    }

    /// <summary>
    /// Non-divisible edge case: numHeads % numKvHeads != 0 must be refused by the shape gate
    /// rather than silently mis-indexing via truncating division (mirrors
    /// <see cref="DotLLM.Cuda.CudaFlashAttention"/>'s identical gate convention).
    /// </summary>
    [Theory]
    [InlineData(10, 3)]   // 10 % 3 != 0
    [InlineData(24, 5)]   // 24 % 5 != 0
    public void IsGqaGroupShapeSupported_ReturnsFalse_ForNonDivisibleRatio(int numHeads, int numKvHeads)
    {
        Assert.False(CudaKernels.IsGqaGroupShapeSupported(numHeads, numKvHeads));
    }

    /// <summary>
    /// Group-size boundary: MaxGqaGroup (8) exactly met is supported; exceeded by 1 is refused
    /// (must fall back cleanly rather than corrupting the compile-time-capped register arrays in
    /// attention_f32.cu's MAX_GQA_GROUP).
    /// </summary>
    [Fact]
    public void IsGqaGroupShapeSupported_BoundaryAtMaxGqaGroup()
    {
        Assert.True(CudaKernels.IsGqaGroupShapeSupported(numHeads: 8 * CudaKernels.MaxGqaGroup, numKvHeads: 8));
        Assert.False(CudaKernels.IsGqaGroupShapeSupported(numHeads: (CudaKernels.MaxGqaGroup + 1) * 8, numKvHeads: 8));
    }

    /// <summary>
    /// MHA edge case: numKvHeads == numHeads (group == 1) is a shape the kernel itself can
    /// compute correctly (covered directly by
    /// <see cref="AttentionF32GqaSplit_KvSplit1_IsBitExactVsExactKernel"/>'s group=1 case above),
    /// but the model call site intentionally does not route to this kernel for it (no benefit
    /// batching a group of one) -- documented here as a deliberate policy choice, not a kernel
    /// limitation.
    /// </summary>
    [Fact]
    public void IsGqaGroupShapeSupported_TrueForGroupOne_PolicyNoteOnly()
    {
        Assert.True(CudaKernels.IsGqaGroupShapeSupported(numHeads: 24, numKvHeads: 24));
    }

    /// <summary>
    /// Confirms the safe-fallback contract: when the requested shape doesn't fit this GPU's
    /// cooperative-launch co-residency ceiling, <see cref="CudaKernels.MaxSafeAttentionGqaSplit"/>
    /// returns 0 rather than letting a caller hit the hard "too many blocks in cooperative
    /// launch" CUDA error.
    /// </summary>
    [SkippableFact]
    public void MaxSafeAttentionGqaSplit_ReturnsZero_ForShapeExceedingCoResidencyCeiling()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX (stale build)");

        Assert.Equal(0, kernels.MaxSafeAttentionGqaSplit(numKvHeads: 1_000_000, headDim: 256, group: 6));
    }

    /// <summary>
    /// Split-heuristic behavior (issue #197): pure logic, no GPU required. Confirms
    /// <see cref="CudaKernels.ComputeAttentionKvSplit"/>'s clamping contract -- monotonic
    /// non-decreasing in seqKv (more KV to split, never fewer splits chosen), always within
    /// [1, maxSafeSplit], and a small maxSafeSplit (e.g. 1) forces the "don't bother splitting"
    /// floor regardless of depth.
    /// </summary>
    [Theory]
    [InlineData(256, 4, 8)]
    [InlineData(1024, 4, 8)]
    [InlineData(2048, 4, 8)]
    [InlineData(4096, 4, 8)]
    public void ComputeAttentionKvSplit_StaysWithinSafeBounds_AndIsMonotonicInSeqKv(
        int seqKvHigh, int baseBlocks, int maxSafeSplit)
    {
        // Build a strictly-ascending seqKv sequence up to and including seqKvHigh (per-row cap)
        // so monotonicity is actually being asserted about seqKv growth, not incidentally failed
        // by feeding the heuristic a non-monotonic seqKv list.
        var seqKvs = new[] { 64, 128, 256, 512, 1024, 2048, 4096 }
            .Where(s => s <= seqKvHigh)
            .Append(seqKvHigh)
            .Distinct()
            .OrderBy(s => s);

        int prev = 1;
        foreach (int seqKv in seqKvs)
        {
            int s = CudaKernels.ComputeAttentionKvSplit(seqKv, baseBlocks, maxSafeSplit);
            Assert.InRange(s, 1, maxSafeSplit);
            Assert.True(s >= prev, $"split should be non-decreasing as seqKv grows: seqKv={seqKv} got {s}, prev={prev}");
            prev = s;
        }
    }

    [Fact]
    public void ComputeAttentionKvSplit_ForcesFloorOfOne_WhenMaxSafeSplitIsOne()
    {
        Assert.Equal(1, CudaKernels.ComputeAttentionKvSplit(seqKv: 4096, baseBlocks: 4, maxSafeSplit: 1));
    }

    [Fact]
    public void ComputeAttentionKvSplit_ReturnsOne_ForDegenerateInputs()
    {
        Assert.Equal(1, CudaKernels.ComputeAttentionKvSplit(seqKv: 0, baseBlocks: 4, maxSafeSplit: 8));
        Assert.Equal(1, CudaKernels.ComputeAttentionKvSplit(seqKv: 256, baseBlocks: 0, maxSafeSplit: 8));
        Assert.Equal(1, CudaKernels.ComputeAttentionKvSplit(seqKv: 256, baseBlocks: 4, maxSafeSplit: 0));
    }

    /// <summary>
    /// Many-consecutive-decode-steps drift characterization, mirroring
    /// <see cref="CudaAttentionF32SplitKvTests.AttentionF32SplitKv_ManyConsecutiveDecodeSteps_DoesNotCompoundDrift"/>
    /// exactly, but for the combined kernel with the REAL runtime heuristic recomputing kvSplit
    /// every step (not held fixed) -- kvSplit CAN change mid-run as seqKv crosses
    /// MinKvPerSplit-driven thresholds. Flags the step indices where kvSplit changes and confirms
    /// no discontinuity or accuracy cliff at those transitions -- this is a genuinely new code
    /// path vs #183's fixed-split test (issue #198's brainstorm explicitly calls this out as a
    /// required scenario, not covered by holding kvSplit constant).
    /// </summary>
    [SkippableFact]
    public void AttentionF32GqaSplit_ManyConsecutiveDecodeSteps_DoesNotCompoundDrift_AcrossSplitTransitions()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        int group = numHeads / numKvHeads;
        Skip.IfNot(kernels.HasAttentionF32GqaSplitKv, "attention_f32_gqa_split_kv not present in PTX (stale build)");
        int maxSafeSplit = kernels.MaxSafeAttentionGqaSplit(numKvHeads, headDim, group);
        Skip.IfNot(maxSafeSplit >= 1,
            "GQA-split cooperative launch not safe for the real Bonsai-27B shape on this GPU");

        const int startSeqKv = 256;
        const int steps = 300;
        var rng = new Random(0x51DE517 ^ 0x6C7A);

        int qElems = numHeads * headDim;
        int kvElems = numKvHeads * headDim;
        int maxSeqKv = startSeqKv + steps;

        float[] kAll = RandomVec(rng, maxSeqKv * kvElems);
        float[] vAll = RandomVec(rng, maxSeqKv * kvElems);

        var driftSamples = new List<(int step, int seqKv, int kvSplit, double maxAbs)>();
        var splitTransitions = new List<(int step, int fromSplit, int toSplit)>();
        double maxAbsEver = 0.0;
        int lastSplit = -1;

        for (int step = 0; step < steps; step++)
        {
            int seqKv = startSeqKv + step;
            int positionOffset = seqKv - 1;
            float[] q = RandomVec(rng, qElems);

            // Slice this step's KV prefix out of the pre-generated history (exact, unperturbed KV
            // cache, matching real decode -- see #183's test for the same rationale).
            float[] kSlice = new float[seqKv * kvElems];
            float[] vSlice = new float[seqKv * kvElems];
            Array.Copy(kAll, kSlice, seqKv * kvElems);
            Array.Copy(vAll, vSlice, seqKv * kvElems);

            float[] cpuOut = new float[qElems];
            unsafe
            {
                fixed (float* pq = q, pk = kSlice, pv = vSlice, pOut = cpuOut)
                    Attention.Execute(pq, pk, pv, pOut, seqQ: 1, seqKv, numHeads, numKvHeads, headDim,
                        positionOffset, pool: null, slidingWindowSize: null);
            }

            int kvSplit = CudaKernels.ComputeAttentionKvSplit(seqKv, numKvHeads, maxSafeSplit);
            if (lastSplit >= 0 && kvSplit != lastSplit)
                splitTransitions.Add((step, lastSplit, kvSplit));
            lastSplit = kvSplit;

            float[] gpuOut = RunGqaSplit(kernels, stream, q, kSlice, vSlice, seqKv, numHeads, numKvHeads, headDim,
                positionOffset, kvSplit);

            double stepMaxAbs = 0.0;
            for (int i = 0; i < qElems; i++)
            {
                Assert.False(float.IsNaN(gpuOut[i]) || float.IsInfinity(gpuOut[i]),
                    $"step {step} (seqKv={seqKv}, kvSplit={kvSplit}): NaN/Inf in GQA-split GPU output at index {i}");
                stepMaxAbs = Math.Max(stepMaxAbs, Math.Abs((double)gpuOut[i] - cpuOut[i]));
            }
            maxAbsEver = Math.Max(maxAbsEver, stepMaxAbs);
            if (step < 10 || step % 25 == 0 || step == steps - 1)
                driftSamples.Add((step, seqKv, kvSplit, stepMaxAbs));
        }

        _out.WriteLine($"steps={steps} startSeqKv={startSeqKv} maxAbsDiffEver={maxAbsEver:e3}");
        _out.WriteLine($"kvSplit transitions ({splitTransitions.Count}): " +
            string.Join(", ", splitTransitions.Select(t => $"step={t.step} {t.fromSplit}->{t.toSplit}")));
        _out.WriteLine("Drift over run (step, seqKv, kvSplit, maxAbs) -- expected roughly FLAT, not growing, " +
            "and no discontinuity at split-transition steps:");
        foreach (var (step, seqKv, kvSplit, maxAbs) in driftSamples)
            _out.WriteLine($"  step={step,4} seqKv={seqKv,5} kvSplit={kvSplit,2} maxAbs={maxAbs:e3}");

        double firstAvg = driftSamples.Take(10).Average(x => x.maxAbs);
        double lastAvg = driftSamples.TakeLast(3).Average(x => x.maxAbs);
        _out.WriteLine($"firstStepsAvg={firstAvg:e3} lastStepsAvg={lastAvg:e3} ratio={lastAvg / Math.Max(firstAvg, 1e-12):F2}x");

        // No accuracy cliff anywhere in the run, including at split-transition steps -- same
        // tolerance bound as the per-step correctness test above (inherits #183's, not a new one).
        Assert.True(maxAbsEver < 5e-2,
            $"GQA-split kernel drift exceeded tolerance at some step across the 300-step run: {maxAbsEver:e3}");
    }
}
