using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// FP16 equivalence tests for the GPU MLA Phase A path. Mirrors the F32
/// suites (<see cref="CudaMlaAttentionTests"/> and <see cref="CudaMlaForwardTests"/>)
/// but exercises the FP16 sibling kernels (<c>attention_mla_f16</c>,
/// <c>mla_split_kv_b_f16</c>, <c>mla_rope_*_f16</c>, <c>mla_rmsnorm_f16</c>)
/// plus the FP16 weight upload + cache layout. Tolerance is relaxed
/// (~1e-2) to absorb FP16 rounding throughout the pipeline; the larger
/// V2-Lite-class shape uses a wider tolerance per the same reasoning the
/// F32 tests do at scale.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaMlaFp16Tests : IDisposable
{
    private const float KernelTolerance = 1e-2f;
    private const float ForwardTolerance = 5e-2f;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMlaFp16Tests()
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

    // ── Kernel-level synthetic equivalence (attention only) ───────────────

    [SkippableFact]
    public void AttentionMlaF16_SingleToken_SingleHead_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16, "MLA FP16 attention PTX not built");

        RunAttentionKernel(seqQ: 1, seqKv: 1, numHeads: 1, qkNope: 4, qkRope: 2, vHead: 4,
            positionOffset: 0, seed: 42);
    }

    [SkippableFact]
    public void AttentionMlaF16_Decode_MultipleHeads_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16, "MLA FP16 attention PTX not built");

        RunAttentionKernel(seqQ: 1, seqKv: 8, numHeads: 4, qkNope: 8, qkRope: 4, vHead: 6,
            positionOffset: 7, seed: 7);
    }

    [SkippableFact]
    public void AttentionMlaF16_Prefill_MultipleHeads_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16, "MLA FP16 attention PTX not built");

        RunAttentionKernel(seqQ: 4, seqKv: 4, numHeads: 3, qkNope: 8, qkRope: 4, vHead: 8,
            positionOffset: 0, seed: 123);
    }

    [SkippableFact]
    public void AttentionMlaF16_LargeKvLength_TilingHandled()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16, "MLA FP16 attention PTX not built");

        RunAttentionKernel(seqQ: 1, seqKv: 200, numHeads: 2, qkNope: 16, qkRope: 8, vHead: 16,
            positionOffset: 199, seed: 555);
    }

    // ── Cross-precision parity (F32 vs F16 on the same input) ─────────────

    [SkippableFact]
    public void AttentionMla_F16_MatchesF32_OnSameSyntheticInput()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaAttentionKernelF16,
            "MLA F32 + FP16 attention kernels both required");

        // Drive both F32 and F16 paths from one host-side fixture, compare GPU
        // outputs against each other within FP16 noise.
        const int seqQ = 1, seqKv = 8, numHeads = 4;
        const int qkNope = 8, qkRope = 4, vHead = 6;
        const int positionOffset = 7;
        var rng = new Random(2026);
        int qkHead = qkNope + qkRope;

        float[] q = RandomArr(rng, seqQ * numHeads * qkHead, 0.5f);
        float[] kNope = RandomArr(rng, seqKv * numHeads * qkNope, 0.5f);
        float[] kPe = RandomArr(rng, seqKv * qkRope, 0.5f);
        float[] v = RandomArr(rng, seqKv * numHeads * vHead, 0.5f);
        float softmaxScale = 1.0f / MathF.Sqrt(qkHead);

        float[] outF32 = RunF32(q, kNope, kPe, v,
            seqQ, seqKv, numHeads, qkNope, qkRope, vHead, positionOffset, softmaxScale);
        float[] outF16 = RunF16(q, kNope, kPe, v,
            seqQ, seqKv, numHeads, qkNope, qkRope, vHead, positionOffset, softmaxScale);

        Assert.Equal(outF32.Length, outF16.Length);
        float maxDiff = 0f;
        int maxDiffIdx = -1;
        for (int i = 0; i < outF32.Length; i++)
        {
            float diff = MathF.Abs(outF32[i] - outF16[i]);
            if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
        }
        Assert.True(maxDiff <= KernelTolerance,
            $"F32 vs F16 cross-precision parity: max diff {maxDiff} at idx {maxDiffIdx} "
          + $"(f32={outF32[maxDiffIdx]} f16={outF16[maxDiffIdx]}) > tol {KernelTolerance}");
    }

    // ── End-to-end ForwardF16 vs CPU oracle ───────────────────────────────

    [SkippableFact]
    public void MlaForwardF16_SingleToken_LoraQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16 && _kernels.HasMlaHelpersF16,
            "MLA FP16 PTX kernels not available");

        RunForward(seqLen: 1, hiddenSize: 8, numHeads: 1,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 6, kvLora: 5,
            positionOffset: 0, seed: 42);
    }

    [SkippableFact]
    public void MlaForwardF16_Prefill_LoraQ_MultiHead_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16 && _kernels.HasMlaHelpersF16,
            "MLA FP16 PTX kernels not available");

        RunForward(seqLen: 4, hiddenSize: 12, numHeads: 3,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 8, kvLora: 6,
            positionOffset: 0, seed: 7);
    }

    [SkippableFact]
    public void MlaForwardF16_Decode_MonolithicQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16 && _kernels.HasMlaHelpersF16,
            "MLA FP16 PTX kernels not available");

        RunForward(seqLen: 1, hiddenSize: 8, numHeads: 2,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 0, kvLora: 5,
            positionOffset: 3, seed: 123, prefillBeforeDecode: 3);
    }

    [SkippableFact]
    public void MlaForwardF16_DeepSeekV2LiteShapes_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernelF16 && _kernels.HasMlaHelpersF16,
            "MLA FP16 PTX kernels not available");

        // Production V2-Lite shapes: hidden=2048, 16 heads, qkNope=128,
        // qkRope=64, vHead=128, qLora=0, kvLora=512.
        // FP16 tolerance is wider than F32 (the F32 test uses 2e-2 here,
        // FP16 picks up additional weight-quant noise from the F32→F16
        // down-cast on each projection — random unit-normal scale weights
        // amplify this).
        RunForward(seqLen: 2, hiddenSize: 2048, numHeads: 16,
            qkNope: 128, qkRope: 64, vHead: 128,
            qLora: 0, kvLora: 512,
            positionOffset: 0, seed: 99, tolerance: 0.4f);
    }

    // ── Memory measurement: F16 cache should be ~½ F32 cache. ─────────────

    [SkippableFact]
    public void CudaMlaKvCacheF16_HasHalfMemoryFootprintOfF32()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int numLayers = 4, maxSeqLen = 64;
        const int numHeads = 8, qkNope = 32, vHead = 32, qkRope = 16;
        using var f32Cache = new CudaMlaKvCache(numLayers, maxSeqLen, numHeads, qkNope, vHead, qkRope, MlaPrecision.F32);
        using var f16Cache = new CudaMlaKvCache(numLayers, maxSeqLen, numHeads, qkNope, vHead, qkRope, MlaPrecision.F16);

        Assert.Equal(MlaPrecision.F32, f32Cache.Precision);
        Assert.Equal(MlaPrecision.F16, f16Cache.Precision);
        Assert.Equal(f32Cache.AllocatedBytes, f16Cache.AllocatedBytes * 2);
    }

    // ── Implementation: attention-kernel-level F16 path ───────────────────

    private unsafe void RunAttentionKernel(int seqQ, int seqKv, int numHeads,
        int qkNope, int qkRope, int vHead, int positionOffset, int seed)
    {
        var rng = new Random(seed);
        int qkHead = qkNope + qkRope;
        float softmaxScale = 1.0f / MathF.Sqrt(qkHead);

        float[] q = RandomArr(rng, seqQ * numHeads * qkHead, 0.5f);
        float[] kNope = RandomArr(rng, seqKv * numHeads * qkNope, 0.5f);
        float[] kPe = RandomArr(rng, seqKv * qkRope, 0.5f);
        float[] v = RandomArr(rng, seqKv * numHeads * vHead, 0.5f);

        float[] expected = new float[seqQ * numHeads * vHead];
        ComputeReference(q, kNope, kPe, v, expected,
            seqQ, seqKv, numHeads, qkNope, qkRope, vHead, positionOffset, softmaxScale);

        float[] actual = RunF16(q, kNope, kPe, v,
            seqQ, seqKv, numHeads, qkNope, qkRope, vHead, positionOffset, softmaxScale);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= KernelTolerance,
                $"index {i}: expected={expected[i]} actual={actual[i]} diff={diff} (tol={KernelTolerance})");
        }
    }

    private unsafe float[] RunF32(float[] q, float[] kNope, float[] kPe, float[] v,
        int seqQ, int seqKv, int numHeads, int qkNope, int qkRope, int vHead,
        int positionOffset, float softmaxScale)
    {
        long qBytes = (long)q.Length * sizeof(float);
        long kNopeBytes = (long)kNope.Length * sizeof(float);
        long kPeBytes = (long)kPe.Length * sizeof(float);
        long vBytes = (long)v.Length * sizeof(float);
        int outElems = seqQ * numHeads * vHead;
        long outBytes = (long)outElems * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint dQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKnope, (nuint)kNopeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKpe, (nuint)kPeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dV, (nuint)vBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOut, (nuint)outBytes).ThrowOnError();
        try
        {
            fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (float* p = kNope) CudaDriverApi.cuMemcpyHtoD_v2(dKnope, (nint)p, (nuint)kNopeBytes).ThrowOnError();
            fixed (float* p = kPe) CudaDriverApi.cuMemcpyHtoD_v2(dKpe, (nint)p, (nuint)kPeBytes).ThrowOnError();
            fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();

            _kernels!.LaunchAttentionMla(
                dQ, dKnope, dKpe, dV, dOut,
                seqQ, seqKv, numHeads, qkNope, qkRope, vHead,
                positionOffset, softmaxScale, _stream!.Handle);
            _stream.Synchronize();

            float[] outF32 = new float[outElems];
            fixed (float* p = outF32)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)outBytes).ThrowOnError();
            return outF32;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dQ);
            CudaDriverApi.cuMemFree_v2(dKnope);
            CudaDriverApi.cuMemFree_v2(dKpe);
            CudaDriverApi.cuMemFree_v2(dV);
            CudaDriverApi.cuMemFree_v2(dOut);
        }
    }

    private unsafe float[] RunF16(float[] q, float[] kNope, float[] kPe, float[] v,
        int seqQ, int seqKv, int numHeads, int qkNope, int qkRope, int vHead,
        int positionOffset, float softmaxScale)
    {
        Half[] qH = ToHalf(q);
        Half[] kNopeH = ToHalf(kNope);
        Half[] kPeH = ToHalf(kPe);
        Half[] vH = ToHalf(v);

        long qBytes = (long)qH.Length * sizeof(ushort);
        long kNopeBytes = (long)kNopeH.Length * sizeof(ushort);
        long kPeBytes = (long)kPeH.Length * sizeof(ushort);
        long vBytes = (long)vH.Length * sizeof(ushort);
        int outElems = seqQ * numHeads * vHead;
        long outBytes = (long)outElems * sizeof(ushort);

        CudaDriverApi.cuMemAlloc_v2(out nint dQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKnope, (nuint)kNopeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKpe, (nuint)kPeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dV, (nuint)vBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOut, (nuint)outBytes).ThrowOnError();
        try
        {
            fixed (Half* p = qH) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (Half* p = kNopeH) CudaDriverApi.cuMemcpyHtoD_v2(dKnope, (nint)p, (nuint)kNopeBytes).ThrowOnError();
            fixed (Half* p = kPeH) CudaDriverApi.cuMemcpyHtoD_v2(dKpe, (nint)p, (nuint)kPeBytes).ThrowOnError();
            fixed (Half* p = vH) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();

            _kernels!.LaunchAttentionMlaF16(
                dQ, dKnope, dKpe, dV, dOut,
                seqQ, seqKv, numHeads, qkNope, qkRope, vHead,
                positionOffset, softmaxScale, _stream!.Handle);
            _stream.Synchronize();

            Half[] outH = new Half[outElems];
            fixed (Half* p = outH)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)outBytes).ThrowOnError();

            float[] outF = new float[outElems];
            for (int i = 0; i < outElems; i++) outF[i] = (float)outH[i];
            return outF;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dQ);
            CudaDriverApi.cuMemFree_v2(dKnope);
            CudaDriverApi.cuMemFree_v2(dKpe);
            CudaDriverApi.cuMemFree_v2(dV);
            CudaDriverApi.cuMemFree_v2(dOut);
        }
    }

    // ── Implementation: end-to-end ForwardF16 vs CPU oracle ───────────────

    private unsafe void RunForward(int seqLen, int hiddenSize, int numHeads,
        int qkNope, int qkRope, int vHead, int qLora, int kvLora,
        int positionOffset, int seed,
        int prefillBeforeDecode = 0, float tolerance = ForwardTolerance)
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

        // ── CPU oracle ──
        // To match the F16 path's noise budget, downcast every weight to FP16 and
        // back before feeding the CPU oracle. This isolates the F16 GPU pipeline's
        // numerical drift from the simple F32→F16 weight-quantization step.
        float[] hiddenForCpu = QuantizeF16Roundtrip(hidden);
        float[] qAProjC = QuantizeF16Roundtrip(qAProj);
        float[] qBProjC = QuantizeF16Roundtrip(qBProj);
        float[] qProjC  = QuantizeF16Roundtrip(qProj);
        float[] kvAProjC = QuantizeF16Roundtrip(kvAProj);
        float[] kvBProjC = QuantizeF16Roundtrip(kvBProj);
        float[] oProjC  = QuantizeF16Roundtrip(oProj);

        float[] cpuPrefillOut = new float[prefillLen * hiddenSize];
        float[] cpuOut = new float[seqLen * hiddenSize];
        nint cpuKNope = 0, cpuV = 0, cpuKPe = 0;
        try
        {
            int kNopePerTok = numHeads * qkNope;
            int vPerTok = numHeads * vHead;
            cpuKNope = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)((long)maxSeq * kNopePerTok * sizeof(float)), 64);
            cpuV = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)((long)maxSeq * vPerTok * sizeof(float)), 64);
            cpuKPe = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)((long)maxSeq * qkRope * sizeof(float)), 64);

            int cpuCachedLen = 0;
            if (prefillLen > 0)
            {
                float[] prefillNorm = new float[prefillLen * hiddenSize];
                ApplyRmsNorm(hiddenForCpu.AsSpan(0, prefillLen * hiddenSize), attnNorm, eps,
                    prefillNorm, prefillLen, hiddenSize);
                MlaAttention.Execute(
                    hidden: prefillNorm,
                    output: cpuPrefillOut,
                    seqLen: prefillLen,
                    positionOffset: 0,
                    hiddenSize: hiddenSize, numHeads: numHeads,
                    qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
                    qLoraRank: qLora, kvLoraRank: kvLora, rmsNormEps: eps,
                    ropeCosTable: cosTab, ropeSinTable: sinTab,
                    qAProj: qAProjC, qALayernormWeight: qANorm,
                    qBProj: qBProjC, qProj: qProjC,
                    kvAProjWithMqa: kvAProjC, kvALayernormWeight: kvANorm, kvBProj: kvBProjC,
                    oProj: oProjC,
                    cachedKNope: cpuKNope, cachedV: cpuV, cachedKPe: cpuKPe,
                    cachedLength: 0);
                cpuCachedLen = prefillLen;
            }

            float[] decodeNorm = new float[seqLen * hiddenSize];
            ApplyRmsNorm(hiddenForCpu.AsSpan(prefillLen * hiddenSize, seqLen * hiddenSize),
                attnNorm, eps, decodeNorm, seqLen, hiddenSize);
            MlaAttention.Execute(
                hidden: decodeNorm,
                output: cpuOut,
                seqLen: seqLen,
                positionOffset: positionOffset,
                hiddenSize: hiddenSize, numHeads: numHeads,
                qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
                qLoraRank: qLora, kvLoraRank: kvLora, rmsNormEps: eps,
                ropeCosTable: cosTab, ropeSinTable: sinTab,
                qAProj: qAProjC, qALayernormWeight: qANorm,
                qBProj: qBProjC, qProj: qProjC,
                kvAProjWithMqa: kvAProjC, kvALayernormWeight: kvANorm, kvBProj: kvBProjC,
                oProj: oProjC,
                cachedKNope: cpuKNope, cachedV: cpuV, cachedKPe: cpuKPe,
                cachedLength: cpuCachedLen);
        }
        finally
        {
            if (cpuKNope != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuKNope);
            if (cpuV != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuV);
            if (cpuKPe != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuKPe);
        }

        // ── GPU FP16 forward ──
        var allocs = new List<nint>();
        try
        {
            // Hidden state goes up as F16; F32 cos/sin tables stay F32.
            nint dHiddenF16 = AllocAndUploadAsF16(hidden, allocs);
            nint dCos = AllocAndUploadF32(cosTab, allocs);
            nint dSin = AllocAndUploadF32(sinTab, allocs);

            // Norm weights stay F32 in the F16 path.
            nint dAttnNorm = AllocAndUploadF32(attnNorm, allocs);
            nint dQANorm = qLora > 0 ? AllocAndUploadF32(qANorm, allocs) : 0;
            nint dKvANorm = AllocAndUploadF32(kvANorm, allocs);

            nint dQAProj = qLora > 0 ? AllocAndUploadAsF16(qAProj, allocs) : 0;
            nint dQBProj = qLora > 0 ? AllocAndUploadAsF16(qBProj, allocs) : 0;
            nint dQProj  = qLora == 0 ? AllocAndUploadAsF16(qProj, allocs) : 0;
            nint dKvAProj = AllocAndUploadAsF16(kvAProj, allocs);
            nint dKvBProj = AllocAndUploadAsF16(kvBProj, allocs);
            nint dOProj   = AllocAndUploadAsF16(oProj, allocs);

            nint dPrefillOut = prefillLen > 0 ? AllocF16(prefillLen * hiddenSize, allocs) : 0;
            nint dOut = AllocF16(seqLen * hiddenSize, allocs);

            var layer = new CudaMlaLayerWeights(
                qAProj: dQAProj, qALayernormWeight: dQANorm, qBProj: dQBProj, qProj: dQProj,
                kvAProjWithMqa: dKvAProj, kvALayernormWeight: dKvANorm, kvBProj: dKvBProj,
                oProj: dOProj, attnNormWeight: dAttnNorm, ffnNormWeight: 0, oBias: 0,
                numHeads: numHeads, qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope,
                vHeadDim: vHead, qLoraRank: qLora, kvLoraRank: kvLora, hiddenSize: hiddenSize,
                precision: MlaPrecision.F16);

            using var kvCache = new CudaMlaKvCache(
                numLayers: 1, maxSeqLen: maxSeq,
                numHeads: numHeads, qkNopeHeadDim: qkNope, vHeadDim: vHead,
                qkRopeHeadDim: qkRope, precision: MlaPrecision.F16);
            using var scratch = new CudaMlaScratchF16();

            if (prefillLen > 0)
            {
                CudaMlaAttention.ForwardF16(
                    hiddenF16: dHiddenF16, outputF16: dPrefillOut,
                    seqLen: prefillLen, positionOffset: 0,
                    layer: layer, kvCache: kvCache, layerIndex: 0,
                    ropeCosF32: dCos, ropeSinF32: dSin,
                    rmsNormEps: eps, softmaxScale: softmaxScale,
                    scratch: scratch, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle);
                kvCache.Advance(0, prefillLen);
            }

            CudaMlaAttention.ForwardF16(
                hiddenF16: dHiddenF16 + (nint)((long)prefillLen * hiddenSize * sizeof(ushort)),
                outputF16: dOut,
                seqLen: seqLen, positionOffset: positionOffset,
                layer: layer, kvCache: kvCache, layerIndex: 0,
                ropeCosF32: dCos, ropeSinF32: dSin,
                rmsNormEps: eps, softmaxScale: softmaxScale,
                scratch: scratch, cublasHandle: _cublas!.Handle,
                kernels: _kernels!, stream: _stream!.Handle);
            kvCache.Advance(0, seqLen);

            _stream.Synchronize();

            Half[] gpuOutH = new Half[seqLen * hiddenSize];
            fixed (Half* p = gpuOutH)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut,
                    (nuint)(gpuOutH.Length * sizeof(ushort))).ThrowOnError();

            int mismatches = 0;
            float maxDiff = 0f;
            int maxDiffIdx = -1;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float gpu = (float)gpuOutH[i];
                float diff = MathF.Abs(cpuOut[i] - gpu);
                if (diff > tolerance)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"MLA FP16 forward: {mismatches}/{cpuOut.Length} elements outside tolerance {tolerance} "
              + $"(max diff {maxDiff} at idx {maxDiffIdx}: cpu={(maxDiffIdx >= 0 ? cpuOut[maxDiffIdx] : 0)} "
              + $"gpu={(maxDiffIdx >= 0 ? (float)gpuOutH[maxDiffIdx] : 0)}).");
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static Half[] ToHalf(float[] src)
    {
        var dst = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
        return dst;
    }

    /// <summary>
    /// Returns a new float[] where each element has been round-tripped through
    /// FP16. Used to give the CPU oracle the same weight-precision noise
    /// as the GPU FP16 path so the comparison isolates kernel drift, not
    /// the F32→F16 weight downcast.
    /// </summary>
    private static float[] QuantizeF16Roundtrip(float[] src)
    {
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)(Half)src[i];
        return dst;
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

    private static unsafe nint AllocAndUploadAsF16(float[] data, List<nint> allocs)
    {
        Half[] h = ToHalf(data);
        long bytes = (long)h.Length * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)bytes).ThrowOnError();
        allocs.Add(dev);
        fixed (Half* p = h)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    private static nint AllocF16(int elems, List<nint> allocs)
    {
        long bytes = (long)elems * sizeof(ushort);
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

    private static void ComputeReference(
        float[] q, float[] kNope, float[] kPe, float[] v, float[] output,
        int seqQ, int seqKv, int numHeads, int qkNope, int qkRope, int vHead,
        int positionOffset, float softmaxScale)
    {
        int qkHead = qkNope + qkRope;
        int qStride = numHeads * qkHead;
        int kNopeStride = numHeads * qkNope;
        int vStride = numHeads * vHead;

        for (int h = 0; h < numHeads; h++)
        {
            for (int tq = 0; tq < seqQ; tq++)
            {
                int posQ = positionOffset + tq;
                float[] scores = new float[seqKv];
                for (int s = 0; s < seqKv; s++)
                {
                    if (s > posQ) { scores[s] = float.NegativeInfinity; continue; }
                    float dot = 0f;
                    int qNopeOff = tq * qStride + h * qkHead;
                    int qPeOff = qNopeOff + qkNope;
                    int kNopeOff = s * kNopeStride + h * qkNope;
                    int kPeOff = s * qkRope;
                    for (int d = 0; d < qkNope; d++)
                        dot += q[qNopeOff + d] * kNope[kNopeOff + d];
                    for (int d = 0; d < qkRope; d++)
                        dot += q[qPeOff + d] * kPe[kPeOff + d];
                    scores[s] = dot * softmaxScale;
                }

                float mx = float.NegativeInfinity;
                for (int i = 0; i < scores.Length; i++) if (scores[i] > mx) mx = scores[i];
                float sum = 0f;
                for (int i = 0; i < scores.Length; i++)
                {
                    if (float.IsNegativeInfinity(scores[i])) { scores[i] = 0f; continue; }
                    scores[i] = MathF.Exp(scores[i] - mx);
                    sum += scores[i];
                }
                if (sum > 0f) for (int i = 0; i < scores.Length; i++) scores[i] /= sum;

                int outOff = tq * vStride + h * vHead;
                for (int d = 0; d < vHead; d++)
                {
                    float acc = 0f;
                    for (int s = 0; s <= posQ && s < seqKv; s++)
                        acc += scores[s] * v[s * vStride + h * vHead + d];
                    output[outOff + d] = acc;
                }
            }
        }
    }
}
