using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Three-way parity gate for the batched I2_S GEMM kernel (issue #250):
/// <see cref="CudaKernels.LaunchI2_SGemmF32In"/>, which decodes each expert's gate/up/down weight
/// row ONCE and reuses it across every token routed to that expert during prefill, instead of the
/// original per-row-GEMV-call loop (issue #246 scope note) that re-decoded the whole weight matrix
/// once PER TOKEN.
/// </summary>
/// <remarks>
/// <para>
/// Every fixture compares THREE independently-computed outputs for the SAME synthetic weights,
/// router, and inputs:
/// <list type="number">
///   <item>The CPU oracle, <see cref="MoeSwiGluMlp.ExecuteBitNetMoe"/>.</item>
///   <item>The GPU forward with <c>forceUseI2SBatchedGemm: false</c> — the original per-row-GEMV-call
///     loop (<see cref="CudaMoeFfnBitNetI2STests"/>'s existing coverage).</item>
///   <item>The GPU forward with <c>forceUseI2SBatchedGemm: true</c> — the new batched kernel.</item>
/// </list>
/// This catches a bug that happens to agree with only one of the two references (e.g. a batched
/// kernel that reproduces the CPU oracle's output by coincidence on a degenerate shape but diverges
/// from the already-proven per-row-loop GPU path, or vice versa) — a two-way check against either
/// reference alone would miss that class of bug.
/// </para>
/// <para>
/// Weight packing / upload helpers mirror <see cref="CudaMoeFfnBitNetI2STests"/> exactly (duplicated
/// locally rather than shared — both files are self-contained fixtures, matching this test
/// directory's existing convention of not cross-referencing between test classes).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMoeFfnBitNetI2SBatchedGemmTests : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMoeFfnBitNetI2SBatchedGemmTests()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        _cublas = CudaCublasHandle.Create();
        _cublas.SetStream(_stream);
        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _cublas?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

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

    [SkippableFact]
    public void BatchedGemm_Prefill_MatchesCpuOracle_AndPerRowLoop()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");
        Skip.IfNot(_kernels!.HasI2SBatchedGemm, "Batched I2_S GEMM kernel not available (recompile i2_s_gemv.cu)");

        Run(seqLen: 6, numExperts: 3, topK: 2, hidden: 256, intermediate: 384,
            normTopKProb: true, withGateBias: true, seed: 250);
    }

    [SkippableFact]
    public void BatchedGemm_LargeBatchPerExpert_MatchesCpuOracle_AndPerRowLoop()
    {
        // seqLen large enough (with few experts) that at least one expert accumulates a batch
        // larger than the kernel's row-cache block width (I2sGemmMaxRowsPerBlock = 8), exercising
        // multiple row-blocks each looping the full token batch.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");
        Skip.IfNot(_kernels!.HasI2SBatchedGemm, "Batched I2_S GEMM kernel not available (recompile i2_s_gemv.cu)");

        Run(seqLen: 24, numExperts: 2, topK: 1, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: false, seed: 2501, requireMaxBatchAtLeast: 9);
    }

    [SkippableFact]
    public void BatchedGemm_LargerHiddenDims_MatchesCpuOracle_AndPerRowLoop()
    {
        // More representative (if still scaled-down) hidden/intermediate sizes than the other
        // fixtures — checks the shared-memory-budget arithmetic in LaunchI2_SGemmF32In at a k large
        // enough to matter, not just the smallest 128/256 shapes.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");
        Skip.IfNot(_kernels!.HasI2SBatchedGemm, "Batched I2_S GEMM kernel not available (recompile i2_s_gemv.cu)");

        Run(seqLen: 12, numExperts: 3, topK: 2, hidden: 1024, intermediate: 1536,
            normTopKProb: true, withGateBias: true, seed: 2502);
    }

    [SkippableFact]
    public void BatchedGemm_Decode_DegradesToGemv_MatchesCpuOracle_AndPerRowLoop()
    {
        // seqLen=1: LaunchI2_SGemmF32In's numTokens==1 fast path degrades to a plain GEMV call.
        // Regression check that the degrade doesn't silently diverge from either reference.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");
        Skip.IfNot(_kernels!.HasI2SBatchedGemm, "Batched I2_S GEMM kernel not available (recompile i2_s_gemv.cu)");

        Run(seqLen: 1, numExperts: 4, topK: 2, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: true, seed: 2503);
    }

    private unsafe void Run(int seqLen, int numExperts, int topK, int hidden, int intermediate,
        bool normTopKProb, bool withGateBias, int seed,
        int requireMaxBatchAtLeast = 0,
        float tolerance = 1e-3f)
    {
        var rng = new Random(seed);
        const float Eps = 1e-5f;

        float[] hiddenAct = RandomVec(rng, seqLen * hidden, 0.5f);
        float[] gate = RandomVec(rng, numExperts * hidden, 0.3f);
        float[] gateBias = withGateBias ? RandomVec(rng, numExperts, 0.5f) : Array.Empty<float>();

        long gateUpRowBytes = (long)intermediate * hidden / 4;
        long downRowBytes = (long)hidden * intermediate / 4;

        byte* gateBank = AllocZeroed(gateUpRowBytes * numExperts);
        byte* upBank = AllocZeroed(gateUpRowBytes * numExperts);
        byte* downBank = AllocZeroed(downRowBytes * numExperts);
        float[] gateScales = new float[numExperts];
        float[] upScales = new float[numExperts];
        float[] downScales = new float[numExperts];
        var ffnSubNorm = new float[numExperts][];

        var allocs = new List<nint>();
        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                gateScales[e] = 0.02f + 0.01f * e;
                upScales[e] = 0.03f + 0.01f * e;
                downScales[e] = 0.04f + 0.01f * e;
                ffnSubNorm[e] = RandomVec(rng, intermediate, 0.2f);
                for (int i = 0; i < intermediate; i++) ffnSubNorm[e][i] += 1.0f;

                sbyte[] gT = RandomTernary(rng, intermediate * hidden);
                sbyte[] uT = RandomTernary(rng, intermediate * hidden);
                sbyte[] dT = RandomTernary(rng, hidden * intermediate);

                PackPayload(gT, gateBank + e * gateUpRowBytes);
                PackPayload(uT, upBank + e * gateUpRowBytes);
                PackPayload(dT, downBank + e * downRowBytes);
            }

            // ── Reference 1: CPU oracle ──
            float[] cpuOut = new float[seqLen * hidden];
            fixed (float* gatePtr = gate, biasPtr = gateBias, hiddenPtr = hiddenAct, outPtr = cpuOut)
            fixed (float* gsc = gateScales, usc = upScales, dsc = downScales)
            {
                MoeSwiGluMlp.ExecuteBitNetMoe(
                    hidden: new ReadOnlySpan<float>(hiddenPtr, seqLen * hidden),
                    gateWeights: new ReadOnlySpan<float>(gatePtr, numExperts * hidden),
                    gateBias: withGateBias ? new ReadOnlySpan<float>(biasPtr, numExperts) : ReadOnlySpan<float>.Empty,
                    gateBank, gateUpRowBytes, new ReadOnlySpan<float>(gsc, numExperts),
                    upBank, gateUpRowBytes, new ReadOnlySpan<float>(usc, numExperts),
                    downBank, downRowBytes, new ReadOnlySpan<float>(dsc, numExperts),
                    ffnSubNorm,
                    output: new Span<float>(outPtr, seqLen * hidden),
                    numExperts, topK, hidden, intermediate, seqLen,
                    normTopKProb, rmsEps: Eps, threadPool: null);
            }

            if (requireMaxBatchAtLeast > 0)
            {
                int maxBatch = MaxAssignmentsPerExpert(
                    hiddenAct, gate, gateBias, withGateBias, numExperts, topK, hidden, seqLen, normTopKProb);
                Assert.True(maxBatch >= requireMaxBatchAtLeast,
                    $"expected at least one expert to receive >= {requireMaxBatchAtLeast} routed tokens " +
                    $"(got max {maxBatch}) — fixture doesn't exercise the intended multi-row-block path");
            }

            // ── Upload device-resident weights/inputs once, shared by both GPU forward calls ──
            nint dHidden = AllocAndUploadF32(hiddenAct, allocs);
            nint dRouter = AllocAndUploadF32(gate, allocs);
            nint dGateBias = withGateBias ? AllocAndUploadF32(gateBias, allocs) : (nint)0;

            var dGateProj = new nint[numExperts];
            var dUpProj = new nint[numExperts];
            var dDownProj = new nint[numExperts];
            var dFfnSubNorm = new nint[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                dGateProj[e] = UploadI2SWithTailScale(gateBank + e * gateUpRowBytes, gateUpRowBytes, gateScales[e], allocs);
                dUpProj[e] = UploadI2SWithTailScale(upBank + e * gateUpRowBytes, gateUpRowBytes, upScales[e], allocs);
                dDownProj[e] = UploadI2SWithTailScale(downBank + e * downRowBytes, downRowBytes, downScales[e], allocs);
                dFfnSubNorm[e] = AllocAndUploadF32(ffnSubNorm[e], allocs);
            }

            var weights = new CudaMoeLayerWeights(
                numExperts: numExperts,
                numExpertsPerTok: topK,
                hiddenSize: hidden,
                moeIntermediateSize: intermediate,
                normTopKProb: normTopKProb,
                router: dRouter,
                gateProj: dGateProj, upProj: dUpProj, downProj: dDownProj,
                numSharedExperts: 0, sharedIntermediateSize: 0,
                sharedGateProj: null, sharedUpProj: null, sharedDownProj: null,
                sharedExpertGate: 0,
                precision: MoePrecision.BitNetI2S,
                gateProjQuantType: QuantizationTypeI2S,
                upProjQuantType: QuantizationTypeI2S,
                downProjQuantType: QuantizationTypeI2S,
                sharedGateProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                sharedUpProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                sharedDownProjQuantType: DotLLM.Core.Configuration.QuantizationType.F32,
                expertFfnSubNormF32: dFfnSubNorm,
                gateBiasF32: dGateBias,
                rmsEps: Eps);

            // ── Reference 2: GPU per-row-GEMV-call loop (forceUseI2SBatchedGemm: false) ──
            float[] gpuPerRowOut;
            using (var scratchPerRow = new CudaMoeScratch())
            {
                nint dOutPerRow = AllocF32(seqLen * hidden, allocs);
                CudaMoeFfn.Forward(
                    hiddenF32: dHidden, outputF32: dOutPerRow,
                    seqLen: seqLen, weights: weights,
                    scratch: scratchPerRow, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle,
                    forceUseI2SBatchedGemm: false);
                _stream.Synchronize();

                gpuPerRowOut = new float[seqLen * hidden];
                fixed (float* p = gpuPerRowOut)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutPerRow,
                        (nuint)(gpuPerRowOut.Length * sizeof(float))).ThrowOnError();
            }

            // ── Test subject: GPU batched-GEMM kernel (forceUseI2SBatchedGemm: true) ──
            float[] gpuBatchedOut;
            using (var scratchBatched = new CudaMoeScratch())
            {
                nint dOutBatched = AllocF32(seqLen * hidden, allocs);
                CudaMoeFfn.Forward(
                    hiddenF32: dHidden, outputF32: dOutBatched,
                    seqLen: seqLen, weights: weights,
                    scratch: scratchBatched, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle,
                    forceUseI2SBatchedGemm: true);
                _stream.Synchronize();

                gpuBatchedOut = new float[seqLen * hidden];
                fixed (float* p = gpuBatchedOut)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutBatched,
                        (nuint)(gpuBatchedOut.Length * sizeof(float))).ThrowOnError();
            }

            AssertClose(cpuOut, gpuPerRowOut, tolerance, "CPU oracle", "GPU per-row-loop");
            AssertClose(cpuOut, gpuBatchedOut, tolerance, "CPU oracle", "GPU batched-GEMM");
            AssertClose(gpuPerRowOut, gpuBatchedOut, tolerance, "GPU per-row-loop", "GPU batched-GEMM");
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
            NativeMemory.Free(gateBank); NativeMemory.Free(upBank); NativeMemory.Free(downBank);
        }
    }

    private static void AssertClose(float[] a, float[] b, float tolerance, string aName, string bName)
    {
        int mismatches = 0;
        float maxDiff = 0f;
        int maxDiffIdx = -1;
        for (int i = 0; i < a.Length; i++)
        {
            float tol = tolerance + tolerance * MathF.Abs(a[i]);
            float diff = MathF.Abs(a[i] - b[i]);
            if (diff > tol)
            {
                mismatches++;
                if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
            }
        }
        Assert.True(mismatches == 0,
            $"{aName} vs {bName}: {mismatches}/{a.Length} elements outside tolerance " +
            $"(max diff {maxDiff} at idx {maxDiffIdx}: {aName}={(maxDiffIdx >= 0 ? a[maxDiffIdx] : 0)} " +
            $"{bName}={(maxDiffIdx >= 0 ? b[maxDiffIdx] : 0)}).");
    }

    /// <summary>Re-derives routing on CPU (mirrors <c>MoeSwiGluMlp.Route</c>) purely to report the
    /// largest per-expert assignment count, so a fixture can assert it actually stresses the
    /// multi-row-block path rather than silently degenerating to batch=1 everywhere.</summary>
    private static unsafe int MaxAssignmentsPerExpert(
        float[] hiddenAct, float[] gate, float[] gateBias, bool withGateBias,
        int numExperts, int topK, int hidden, int seqLen, bool normTopKProb)
    {
        int[] assignExpert = new int[seqLen * topK];
        float[] assignWeight = new float[seqLen * topK];
        int[] bc = new int[numExperts + 1];
        int[] bt = new int[seqLen * topK];
        int[] bs = new int[seqLen * topK];
        int[] uniq = new int[seqLen * topK];
        fixed (float* hp = hiddenAct, gp = gate, bp = gateBias)
        {
            MoeSwiGluMlp.Route(
                new ReadOnlySpan<float>(hp, seqLen * hidden),
                new ReadOnlySpan<float>(gp, numExperts * hidden),
                assignExpert, assignWeight, bc, bt, bs, uniq,
                numExperts, topK, hidden, seqLen, normTopKProb,
                withGateBias ? new ReadOnlySpan<float>(bp, numExperts) : ReadOnlySpan<float>.Empty);
        }
        int[] counts = new int[numExperts];
        foreach (int e in assignExpert)
            if ((uint)e < (uint)numExperts) counts[e]++;
        int max = 0;
        foreach (int c in counts) if (c > max) max = c;
        return max;
    }

    private const DotLLM.Core.Configuration.QuantizationType QuantizationTypeI2S =
        DotLLM.Core.Configuration.QuantizationType.I2_S;

    private static unsafe nint UploadI2SWithTailScale(byte* hostPayload, long payloadBytes, float scale, List<nint> allocs)
    {
        long totalBytes = payloadBytes + sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)totalBytes).ThrowOnError();
        allocs.Add(dev);
        CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)hostPayload, (nuint)payloadBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD_v2(dev + (nint)payloadBytes, (nint)(&scale), (nuint)sizeof(float)).ThrowOnError();
        return dev;
    }

    private static unsafe nint AllocAndUploadF32(float[] data, List<nint> allocs)
    {
        long bytes = (long)data.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)bytes).ThrowOnError();
        allocs.Add(dev);
        fixed (float* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    private static nint AllocF32(int elems, List<nint> allocs)
    {
        long bytes = (long)elems * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)bytes).ThrowOnError();
        allocs.Add(dev);
        return dev;
    }

    // ── I2_S packing (mirrors CudaMoeFfnBitNetI2STests / MoteBitNetMoeLoaderScaffoldTests) ──
    private static unsafe void PackPayload(sbyte[] ternary, byte* dest)
    {
        int n = ternary.Length;
        for (int e = 0; e < n; e++)
        {
            int block = e / 128, j = e % 128, groupIdx = j / 32, groupPos = j % 32;
            dest[block * 32 + groupPos] |= (byte)((ternary[e] + 1) << (6 - 2 * groupIdx));
        }
    }

    private static sbyte[] RandomTernary(Random rng, int n)
    {
        var v = new sbyte[n];
        for (int i = 0; i < n; i++) v[i] = (sbyte)(rng.Next(3) - 1);
        return v;
    }

    private static unsafe byte* AllocZeroed(long bytes) => (byte*)NativeMemory.AllocZeroed((nuint)bytes);

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }
}
