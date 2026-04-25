using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// MLA Phase B (latent KV cache + W_UK absorbed attention) GPU equivalence
/// tests. The Phase B path stores only the shared <c>c_kv</c> + <c>k_pe</c>
/// latents and recovers per-head K/V on the fly through W_UK / W_UV
/// absorption — production decode-efficiency unlock for DeepSeek-V2/V3.
/// </summary>
/// <remarks>
/// <para>
/// Validates <see cref="CudaMlaAttention.ForwardLatent"/> against the CPU
/// oracle <see cref="MlaAttention.ExecuteLatent"/> on synthetic fixtures of
/// varying shape, plus a cache-size assertion proving the ~7-14× footprint
/// reduction vs Phase A's <see cref="CudaMlaKvCache"/> on V2-Lite shapes.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMlaPhaseBTests : IDisposable
{
    private const float DefaultTolerance = 1e-3f;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMlaPhaseBTests()
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
    public void MlaForwardLatent_SingleToken_LoraQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaPhaseB && _kernels.HasMlaHelpers,
            "MLA Phase B PTX kernels not available");

        Run(seqLen: 1, hiddenSize: 8, numHeads: 1,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 6, kvLora: 5,
            positionOffset: 0, seed: 42);
    }

    [SkippableFact]
    public void MlaForwardLatent_Prefill_LoraQ_MultiHead_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaPhaseB && _kernels.HasMlaHelpers,
            "MLA Phase B PTX kernels not available");

        Run(seqLen: 4, hiddenSize: 12, numHeads: 3,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 8, kvLora: 6,
            positionOffset: 0, seed: 7);
    }

    [SkippableFact]
    public void MlaForwardLatent_Decode_MonolithicQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaPhaseB && _kernels.HasMlaHelpers,
            "MLA Phase B PTX kernels not available");

        Run(seqLen: 1, hiddenSize: 8, numHeads: 2,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 0, kvLora: 5,
            positionOffset: 3, seed: 123, prefillBeforeDecode: 3);
    }

    [SkippableFact]
    public void MlaForwardLatent_DeepSeekV2LiteShapes_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaPhaseB && _kernels.HasMlaHelpers,
            "MLA Phase B PTX kernels not available");

        // DeepSeek-V2-Lite production shapes: 16 heads, qkNope=128,
        // qkRope=64, vHead=128, qLora=0 (V2-Lite), kvLora=512. seqLen=2
        // keeps the test fast (CPU oracle is O(seqLen² * heads * (qkLora +
        // qkRope + vLora))). Tolerance loosened to 2e-2 for the same reason
        // as Phase A's V2-Lite test: 2048-wide o_proj on top of a 16x576-dim
        // chained reduction lets the cuBLAS Tensor-Core SGEMM order drift
        // up to ~1e-2 relative to scalar.
        Run(seqLen: 2, hiddenSize: 2048, numHeads: 16,
            qkNope: 128, qkRope: 64, vHead: 128,
            qLora: 0, kvLora: 512,
            positionOffset: 0, seed: 99, tolerance: 2e-2f);
    }

    /// <summary>
    /// Cache-size sanity check: Phase B latent cache must be substantially
    /// smaller than Phase A expanded cache for DeepSeek-V2-Lite shapes.
    /// Pure C# math — no GPU touched, no PTX needed (still tagged GPU because
    /// it's part of the Phase B test family).
    /// </summary>
    [SkippableFact]
    public void MlaLatentKvCache_VRamFootprint_ShrinksVsPhaseA()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        // V2-Lite shapes.
        const int numLayers = 27;     // V2-Lite layer count
        const int maxSeqLen = 4096;   // typical context
        const int numHeads = 16;
        const int qkNope = 128;
        const int qkRope = 64;
        const int vHead = 128;
        const int kvLora = 512;

        using var phaseA = new CudaMlaKvCache(
            numLayers: numLayers, maxSeqLen: maxSeqLen,
            numHeads: numHeads, qkNopeHeadDim: qkNope,
            vHeadDim: vHead, qkRopeHeadDim: qkRope);
        using var phaseB = new CudaMlaLatentKvCache(
            numLayers: numLayers, maxSeqLen: maxSeqLen,
            kvLoraRank: kvLora, qkRopeHeadDim: qkRope);

        long phaseABytes = phaseA.AllocatedBytes;
        long phaseBBytes = phaseB.AllocatedBytes;
        double ratio = (double)phaseABytes / phaseBBytes;

        // Per-token math:
        //   Phase A: numHeads*qkNope + numHeads*vHead + qkRope = 16*128 + 16*128 + 64 = 4160 F32
        //   Phase B: kvLora + qkRope                            = 512 + 64           = 576  F32
        //   ratio = 4160 / 576 ≈ 7.22×
        // Verify the actual allocation reflects this within a small margin
        // (the per-layer overhead from the buffer struct headers is tiny).
        Assert.True(ratio > 7.0,
            $"Phase B cache should be ~7.22× smaller than Phase A on V2-Lite shapes. "
          + $"Got ratio={ratio:F2} (Phase A {phaseABytes:N0} bytes, Phase B {phaseBBytes:N0} bytes).");
        Assert.True(ratio < 8.0,
            $"Ratio above expected upper bound. Phase A {phaseABytes:N0} bytes, "
          + $"Phase B {phaseBBytes:N0} bytes, ratio={ratio:F2}.");

        // Print actual numbers so the test trace makes the win visible.
        // (xUnit only surfaces this on failure or with -v option, but the
        // assert messages above also embed the values.)
        long phaseAMB = phaseABytes / (1024 * 1024);
        long phaseBMB = phaseBBytes / (1024 * 1024);
        Assert.True(phaseAMB > 0, $"Phase A: {phaseAMB} MiB, Phase B: {phaseBMB} MiB ({ratio:F2}× smaller)");
    }

    /// <summary>
    /// Bonus end-to-end equivalence test: run Phase A and Phase B on the same
    /// synthetic input + weights, compare the GPU outputs of both directly to
    /// each other. Both should produce the same logits within accumulated FP32
    /// reduction noise — the only difference is the order of the dot product
    /// (W_UK^T @ Q · c_kv vs Q · W_UK @ c_kv).
    /// </summary>
    [SkippableFact]
    public void MlaPhaseAvsPhaseB_GpuOutputs_AgreeWithinFp16Noise()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaHelpers
                   && _kernels.HasMlaPhaseB,
            "MLA Phase A + Phase B PTX kernels both required");

        // Small shape — bit-near-equivalence regime for both paths.
        RunPhaseAvsPhaseB(seqLen: 2, hiddenSize: 12, numHeads: 3,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 8, kvLora: 6,
            positionOffset: 0, seed: 17, tolerance: 1e-3f);
    }

    private unsafe void Run(int seqLen, int hiddenSize, int numHeads,
        int qkNope, int qkRope, int vHead, int qLora, int kvLora,
        int positionOffset, int seed,
        int prefillBeforeDecode = 0, float tolerance = DefaultTolerance)
    {
        var rng = new Random(seed);
        const float eps = 1e-6f;

        int qkHead = qkNope + qkRope;
        int qTotal = numHeads * qkHead;
        int kvAOut = kvLora + qkRope;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInput = numHeads * vHead;
        int maxSeq = positionOffset + seqLen + 8;
        int prefillLen = prefillBeforeDecode;

        float[] hidden = RandomArr(rng, (prefillLen + seqLen) * hiddenSize, 0.3f);

        float[] qAProj = qLora > 0 ? RandomArr(rng, qLora * hiddenSize, 0.1f) : Array.Empty<float>();
        float[] qANorm = qLora > 0 ? FillArr(rng, qLora, 1.0f, 0.05f) : Array.Empty<float>();
        float[] qBProj = qLora > 0 ? RandomArr(rng, qTotal * qLora, 0.1f) : Array.Empty<float>();
        float[] qProj = qLora == 0 ? RandomArr(rng, qTotal * hiddenSize, 0.1f) : Array.Empty<float>();

        float[] kvAProj = RandomArr(rng, kvAOut * hiddenSize, 0.1f);
        float[] kvANorm = FillArr(rng, kvLora, 1.0f, 0.05f);
        float[] kvBProj = RandomArr(rng, kvBOut * kvLora, 0.1f);
        float[] oProj = RandomArr(rng, hiddenSize * oInput, 0.1f);
        float[] attnNorm = FillArr(rng, hiddenSize, 1.0f, 0.05f);

        var (cosTab, sinTab) = PrecomputeRopeTables(maxSeq, qkRope, theta: 10000.0f);
        float softmaxScale = 1.0f / MathF.Sqrt(qkHead);

        float[] cpuPrefillOut = new float[prefillLen * hiddenSize];
        float[] cpuOut = new float[seqLen * hiddenSize];

        nint cpuLatent = 0, cpuKPe = 0;
        try
        {
            cpuLatent = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)((long)maxSeq * kvLora * sizeof(float)), 64);
            cpuKPe = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)((long)maxSeq * qkRope * sizeof(float)), 64);

            int cpuCachedLen = 0;
            if (prefillLen > 0)
            {
                float[] prefillNorm = new float[prefillLen * hiddenSize];
                ApplyRmsNorm(hidden.AsSpan(0, prefillLen * hiddenSize), attnNorm, eps,
                    prefillNorm, prefillLen, hiddenSize);
                MlaAttention.ExecuteLatent(
                    hidden: prefillNorm,
                    output: cpuPrefillOut,
                    seqLen: prefillLen,
                    positionOffset: 0,
                    hiddenSize: hiddenSize, numHeads: numHeads,
                    qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
                    qLoraRank: qLora, kvLoraRank: kvLora, rmsNormEps: eps,
                    ropeCosTable: cosTab, ropeSinTable: sinTab,
                    qAProj: qAProj, qALayernormWeight: qANorm,
                    qBProj: qBProj, qProj: qProj,
                    kvAProjWithMqa: kvAProj, kvALayernormWeight: kvANorm, kvBProj: kvBProj,
                    oProj: oProj,
                    cachedLatent: cpuLatent, cachedKPe: cpuKPe,
                    cachedLength: 0);
                cpuCachedLen = prefillLen;
            }

            float[] decodeNorm = new float[seqLen * hiddenSize];
            ApplyRmsNorm(hidden.AsSpan(prefillLen * hiddenSize, seqLen * hiddenSize),
                attnNorm, eps, decodeNorm, seqLen, hiddenSize);
            MlaAttention.ExecuteLatent(
                hidden: decodeNorm,
                output: cpuOut,
                seqLen: seqLen,
                positionOffset: positionOffset,
                hiddenSize: hiddenSize, numHeads: numHeads,
                qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
                qLoraRank: qLora, kvLoraRank: kvLora, rmsNormEps: eps,
                ropeCosTable: cosTab, ropeSinTable: sinTab,
                qAProj: qAProj, qALayernormWeight: qANorm,
                qBProj: qBProj, qProj: qProj,
                kvAProjWithMqa: kvAProj, kvALayernormWeight: kvANorm, kvBProj: kvBProj,
                oProj: oProj,
                cachedLatent: cpuLatent, cachedKPe: cpuKPe,
                cachedLength: cpuCachedLen);
        }
        finally
        {
            if (cpuLatent != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuLatent);
            if (cpuKPe != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuKPe);
        }

        var allocs = new List<nint>();
        try
        {
            nint dHidden = AllocAndUploadF32(hidden, allocs);
            nint dCos = AllocAndUploadF32(cosTab, allocs);
            nint dSin = AllocAndUploadF32(sinTab, allocs);

            nint dAttnNorm = AllocAndUploadF32(attnNorm, allocs);
            nint dQAProj = qLora > 0 ? AllocAndUploadF32(qAProj, allocs) : 0;
            nint dQANorm = qLora > 0 ? AllocAndUploadF32(qANorm, allocs) : 0;
            nint dQBProj = qLora > 0 ? AllocAndUploadF32(qBProj, allocs) : 0;
            nint dQProj = qLora == 0 ? AllocAndUploadF32(qProj, allocs) : 0;
            nint dKvAProj = AllocAndUploadF32(kvAProj, allocs);
            nint dKvANorm = AllocAndUploadF32(kvANorm, allocs);
            nint dKvBProj = AllocAndUploadF32(kvBProj, allocs);
            nint dOProj = AllocAndUploadF32(oProj, allocs);

            nint dPrefillOut = prefillLen > 0 ? AllocF32(prefillLen * hiddenSize, allocs) : 0;
            nint dOut = AllocF32(seqLen * hiddenSize, allocs);

            var layer = new CudaMlaLayerWeights(
                qAProj: dQAProj, qALayernormWeight: dQANorm, qBProj: dQBProj, qProj: dQProj,
                kvAProjWithMqa: dKvAProj, kvALayernormWeight: dKvANorm, kvBProj: dKvBProj,
                oProj: dOProj, attnNormWeight: dAttnNorm, ffnNormWeight: 0, oBias: 0,
                numHeads: numHeads, qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope,
                vHeadDim: vHead, qLoraRank: qLora, kvLoraRank: kvLora, hiddenSize: hiddenSize);

            using var kvCache = new CudaMlaLatentKvCache(
                numLayers: 1, maxSeqLen: maxSeq,
                kvLoraRank: kvLora, qkRopeHeadDim: qkRope);
            using var scratch = new CudaMlaLatentScratch();

            if (prefillLen > 0)
            {
                CudaMlaAttention.ForwardLatent(
                    hiddenF32: dHidden, outputF32: dPrefillOut,
                    seqLen: prefillLen, positionOffset: 0,
                    layer: layer, kvCache: kvCache, layerIndex: 0,
                    ropeCosF32: dCos, ropeSinF32: dSin,
                    rmsNormEps: eps, softmaxScale: softmaxScale,
                    scratch: scratch, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle);
                kvCache.Advance(0, prefillLen);
            }

            CudaMlaAttention.ForwardLatent(
                hiddenF32: dHidden + (nint)((long)prefillLen * hiddenSize * sizeof(float)),
                outputF32: dOut,
                seqLen: seqLen, positionOffset: positionOffset,
                layer: layer, kvCache: kvCache, layerIndex: 0,
                ropeCosF32: dCos, ropeSinF32: dSin,
                rmsNormEps: eps, softmaxScale: softmaxScale,
                scratch: scratch, cublasHandle: _cublas!.Handle,
                kernels: _kernels!, stream: _stream!.Handle);
            kvCache.Advance(0, seqLen);

            _stream.Synchronize();

            float[] gpuOut = new float[seqLen * hiddenSize];
            fixed (float* p = gpuOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut,
                    (nuint)(gpuOut.Length * sizeof(float))).ThrowOnError();

            int mismatches = 0;
            float maxDiff = 0f;
            int maxDiffIdx = -1;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > tolerance)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"MLA Phase B forward: {mismatches}/{cpuOut.Length} elements outside tolerance {tolerance} "
              + $"(max diff {maxDiff} at idx {maxDiffIdx}: cpu={(maxDiffIdx >= 0 ? cpuOut[maxDiffIdx] : 0)} "
              + $"gpu={(maxDiffIdx >= 0 ? gpuOut[maxDiffIdx] : 0)}).");
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
        }
    }

    private unsafe void RunPhaseAvsPhaseB(int seqLen, int hiddenSize, int numHeads,
        int qkNope, int qkRope, int vHead, int qLora, int kvLora,
        int positionOffset, int seed, float tolerance)
    {
        var rng = new Random(seed);
        const float eps = 1e-6f;

        int qkHead = qkNope + qkRope;
        int qTotal = numHeads * qkHead;
        int kvAOut = kvLora + qkRope;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInput = numHeads * vHead;
        int maxSeq = positionOffset + seqLen + 8;

        float[] hidden = RandomArr(rng, seqLen * hiddenSize, 0.3f);
        float[] qAProj = qLora > 0 ? RandomArr(rng, qLora * hiddenSize, 0.1f) : Array.Empty<float>();
        float[] qANorm = qLora > 0 ? FillArr(rng, qLora, 1.0f, 0.05f) : Array.Empty<float>();
        float[] qBProj = qLora > 0 ? RandomArr(rng, qTotal * qLora, 0.1f) : Array.Empty<float>();
        float[] qProj = qLora == 0 ? RandomArr(rng, qTotal * hiddenSize, 0.1f) : Array.Empty<float>();
        float[] kvAProj = RandomArr(rng, kvAOut * hiddenSize, 0.1f);
        float[] kvANorm = FillArr(rng, kvLora, 1.0f, 0.05f);
        float[] kvBProj = RandomArr(rng, kvBOut * kvLora, 0.1f);
        float[] oProj = RandomArr(rng, hiddenSize * oInput, 0.1f);
        float[] attnNorm = FillArr(rng, hiddenSize, 1.0f, 0.05f);

        var (cosTab, sinTab) = PrecomputeRopeTables(maxSeq, qkRope, theta: 10000.0f);
        float softmaxScale = 1.0f / MathF.Sqrt(qkHead);

        var allocs = new List<nint>();
        try
        {
            nint dHidden = AllocAndUploadF32(hidden, allocs);
            nint dCos = AllocAndUploadF32(cosTab, allocs);
            nint dSin = AllocAndUploadF32(sinTab, allocs);
            nint dAttnNorm = AllocAndUploadF32(attnNorm, allocs);
            nint dQAProj = qLora > 0 ? AllocAndUploadF32(qAProj, allocs) : 0;
            nint dQANorm = qLora > 0 ? AllocAndUploadF32(qANorm, allocs) : 0;
            nint dQBProj = qLora > 0 ? AllocAndUploadF32(qBProj, allocs) : 0;
            nint dQProj = qLora == 0 ? AllocAndUploadF32(qProj, allocs) : 0;
            nint dKvAProj = AllocAndUploadF32(kvAProj, allocs);
            nint dKvANorm = AllocAndUploadF32(kvANorm, allocs);
            nint dKvBProj = AllocAndUploadF32(kvBProj, allocs);
            nint dOProj = AllocAndUploadF32(oProj, allocs);

            nint dOutA = AllocF32(seqLen * hiddenSize, allocs);
            nint dOutB = AllocF32(seqLen * hiddenSize, allocs);

            var layer = new CudaMlaLayerWeights(
                qAProj: dQAProj, qALayernormWeight: dQANorm, qBProj: dQBProj, qProj: dQProj,
                kvAProjWithMqa: dKvAProj, kvALayernormWeight: dKvANorm, kvBProj: dKvBProj,
                oProj: dOProj, attnNormWeight: dAttnNorm, ffnNormWeight: 0, oBias: 0,
                numHeads: numHeads, qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope,
                vHeadDim: vHead, qLoraRank: qLora, kvLoraRank: kvLora, hiddenSize: hiddenSize);

            // Phase A
            using (var kvCacheA = new CudaMlaKvCache(
                numLayers: 1, maxSeqLen: maxSeq,
                numHeads: numHeads, qkNopeHeadDim: qkNope, vHeadDim: vHead,
                qkRopeHeadDim: qkRope))
            using (var scratchA = new CudaMlaScratch())
            {
                CudaMlaAttention.Forward(
                    hiddenF32: dHidden, outputF32: dOutA,
                    seqLen: seqLen, positionOffset: positionOffset,
                    layer: layer, kvCache: kvCacheA, layerIndex: 0,
                    ropeCosF32: dCos, ropeSinF32: dSin,
                    rmsNormEps: eps, softmaxScale: softmaxScale,
                    scratch: scratchA, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle);
                kvCacheA.Advance(0, seqLen);
            }

            // Phase B
            using (var kvCacheB = new CudaMlaLatentKvCache(
                numLayers: 1, maxSeqLen: maxSeq,
                kvLoraRank: kvLora, qkRopeHeadDim: qkRope))
            using (var scratchB = new CudaMlaLatentScratch())
            {
                CudaMlaAttention.ForwardLatent(
                    hiddenF32: dHidden, outputF32: dOutB,
                    seqLen: seqLen, positionOffset: positionOffset,
                    layer: layer, kvCache: kvCacheB, layerIndex: 0,
                    ropeCosF32: dCos, ropeSinF32: dSin,
                    rmsNormEps: eps, softmaxScale: softmaxScale,
                    scratch: scratchB, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle);
                kvCacheB.Advance(0, seqLen);
            }

            _stream!.Synchronize();

            float[] outA = new float[seqLen * hiddenSize];
            float[] outB = new float[seqLen * hiddenSize];
            fixed (float* pA = outA)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, dOutA,
                    (nuint)(outA.Length * sizeof(float))).ThrowOnError();
            fixed (float* pB = outB)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pB, dOutB,
                    (nuint)(outB.Length * sizeof(float))).ThrowOnError();

            int mismatches = 0;
            float maxDiff = 0f;
            int maxDiffIdx = -1;
            for (int i = 0; i < outA.Length; i++)
            {
                float diff = MathF.Abs(outA[i] - outB[i]);
                if (diff > tolerance)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"Phase A vs Phase B GPU outputs disagree: {mismatches}/{outA.Length} outside tolerance {tolerance} "
              + $"(max diff {maxDiff} at idx {maxDiffIdx}: A={(maxDiffIdx >= 0 ? outA[maxDiffIdx] : 0)} "
              + $"B={(maxDiffIdx >= 0 ? outB[maxDiffIdx] : 0)}).");
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
        }
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

    private static void ApplyRmsNorm(
        ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float eps,
        Span<float> output, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            var inRow = input.Slice(r * dim, dim);
            var outRow = output.Slice(r * dim, dim);
            float sum = 0f;
            for (int i = 0; i < dim; i++) sum += inRow[i] * inRow[i];
            float rms = MathF.Sqrt(sum / dim + eps);
            float inv = 1.0f / rms;
            for (int i = 0; i < dim; i++) outRow[i] = inRow[i] * inv * weight[i];
        }
    }

    private static (float[] cos, float[] sin) PrecomputeRopeTables(int maxSeq, int dim, float theta)
    {
        int half = dim / 2;
        float[] cos = new float[maxSeq * half];
        float[] sin = new float[maxSeq * half];
        for (int pos = 0; pos < maxSeq; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, 2.0f * i / dim);
                float angle = pos * freq;
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static float[] RandomArr(Random rng, int n, float scale)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return arr;
    }

    private static float[] FillArr(Random rng, int n, float center, float jitter)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = center + (float)((rng.NextDouble() * 2.0 - 1.0) * jitter);
        return arr;
    }
}
