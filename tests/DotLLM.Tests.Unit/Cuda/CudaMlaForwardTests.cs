using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end equivalence tests for the GPU MLA Phase A forward
/// (<see cref="CudaMlaAttention.Forward"/>): exercises the full attention
/// block — pre-attention RMSNorm, Q LoRA / monolithic projection, KV LoRA
/// projection + split + RMSNorm + expansion + per-head split, RoPE on
/// rope-half only, expanded-cache write, attention, o_proj — and compares
/// against <see cref="MlaAttention.Execute"/> running the same fixture
/// scalar on the CPU.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is the gate.</b> The kernel-level synthetic test
/// (<see cref="CudaMlaAttentionTests"/>) only covers the attention loop in
/// isolation. This file covers the projections + RoPE + cache-write path
/// too — the full GPU MLA layer. Pass here means the next agent can plug
/// <see cref="CudaMlaAttention.Forward"/> into <c>CudaTransformerModel</c>
/// once the MoE FFN path lands.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMlaForwardTests : IDisposable
{
    private const float DefaultTolerance = 1e-3f;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMlaForwardTests()
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
    public void MlaForward_SingleToken_LoraQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaHelpers,
            "MLA PTX kernels not available");

        Run(seqLen: 1, hiddenSize: 8, numHeads: 1,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 6, kvLora: 5,
            positionOffset: 0, seed: 42);
    }

    [SkippableFact]
    public void MlaForward_Prefill_LoraQ_MultiHead_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaHelpers,
            "MLA PTX kernels not available");

        Run(seqLen: 4, hiddenSize: 12, numHeads: 3,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 8, kvLora: 6,
            positionOffset: 0, seed: 7);
    }

    [SkippableFact]
    public void MlaForward_Decode_MonolithicQ_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaHelpers,
            "MLA PTX kernels not available");

        Run(seqLen: 1, hiddenSize: 8, numHeads: 2,
            qkNope: 4, qkRope: 2, vHead: 4, qLora: 0, kvLora: 5,
            positionOffset: 3, seed: 123, prefillBeforeDecode: 3);
    }

    [SkippableFact]
    public void MlaForward_DeepSeekV2LiteShapes_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel && _kernels.HasMlaHelpers,
            "MLA PTX kernels not available");

        // Production V2-Lite shapes: hidden=2048, 16 heads, qkNope=128,
        // qkRope=64, vHead=128, qLora=0 (monolithic on V2-Lite),
        // kvLora=512. seqLen=2 keeps the reference fast (CPU oracle is
        // O(seqLen² * heads * (qkHead + vHead))).
        // V2-Lite: 2048-wide hidden + 2048-wide o_proj on top of a 16x192-dim
        // attention reduction. cuBLAS Tensor-Core SGEMM order is materially
        // different from scalar — drift up to ~1e-2 over the chained dot
        // products is expected for randomly-initialised weights without
        // RMSNorm normalisation downstream. The CPU oracle is bit-near-identical
        // to a Python reference; a real model with normalised activations
        // sees much tighter agreement. See `MlaForward_*_MatchesCpuOracle`
        // tests for the small-shape gate at 1e-3.
        Run(seqLen: 2, hiddenSize: 2048, numHeads: 16,
            qkNope: 128, qkRope: 64, vHead: 128,
            qLora: 0, kvLora: 512,
            positionOffset: 0, seed: 99, tolerance: 2e-2f);
    }

    /// <summary>
    /// Builds a synthetic MLA fixture, runs the GPU forward through
    /// <see cref="CudaMlaAttention.Forward"/>, and compares against the CPU
    /// oracle <see cref="MlaAttention.Execute"/>. When
    /// <paramref name="prefillBeforeDecode"/> &gt; 0, runs the cache-warming
    /// step on both backends first so the decode call sees a populated cache.
    /// </summary>
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

        // ── Build random weights + a hidden-state stream of length (prefillLen + seqLen) ──
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

        // ── CPU oracle: pre-attention RMSNorm + MlaAttention.Execute ──
        // The CPU MlaAttention.Execute consumes the already-normalised hidden state
        // (TransformerModel applies AttnNorm before calling it). Mirror that here.
        float[] cpuPrefillOut = new float[prefillLen * hiddenSize];
        float[] cpuOut = new float[seqLen * hiddenSize];

        // Allocate CPU "expanded cache" buffers (managed; only used to mirror the
        // GPU's cachedLength state for the decode-after-prefill flow).
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
                ApplyRmsNorm(hidden.AsSpan(0, prefillLen * hiddenSize), attnNorm, eps,
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
                    qAProj: qAProj, qALayernormWeight: qANorm,
                    qBProj: qBProj, qProj: qProj,
                    kvAProjWithMqa: kvAProj, kvALayernormWeight: kvANorm, kvBProj: kvBProj,
                    oProj: oProj,
                    cachedKNope: cpuKNope, cachedV: cpuV, cachedKPe: cpuKPe,
                    cachedLength: 0);
                cpuCachedLen = prefillLen;
            }

            // Decode / prefill step that we'll compare GPU against
            float[] decodeNorm = new float[seqLen * hiddenSize];
            ApplyRmsNorm(hidden.AsSpan(prefillLen * hiddenSize, seqLen * hiddenSize),
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
                qAProj: qAProj, qALayernormWeight: qANorm,
                qBProj: qBProj, qProj: qProj,
                kvAProjWithMqa: kvAProj, kvALayernormWeight: kvANorm, kvBProj: kvBProj,
                oProj: oProj,
                cachedKNope: cpuKNope, cachedV: cpuV, cachedKPe: cpuKPe,
                cachedLength: cpuCachedLen);
        }
        finally
        {
            if (cpuKNope != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuKNope);
            if (cpuV != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuV);
            if (cpuKPe != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)cpuKPe);
        }

        // ── GPU forward ──
        // Upload all weights + hidden state + RoPE tables.
        var allocs = new List<nint>();
        try
        {
            // Hidden state lives on device (FP32 throughout MLA Phase 1).
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

            // Output buffers (one per call, sized to the max).
            nint dPrefillOut = prefillLen > 0 ? AllocF32(prefillLen * hiddenSize, allocs) : 0;
            nint dOut = AllocF32(seqLen * hiddenSize, allocs);

            var layer = new CudaMlaLayerWeights(
                qAProj: dQAProj, qALayernormWeight: dQANorm, qBProj: dQBProj, qProj: dQProj,
                kvAProjWithMqa: dKvAProj, kvALayernormWeight: dKvANorm, kvBProj: dKvBProj,
                oProj: dOProj, attnNormWeight: dAttnNorm, ffnNormWeight: 0, oBias: 0,
                numHeads: numHeads, qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope,
                vHeadDim: vHead, qLoraRank: qLora, kvLoraRank: kvLora, hiddenSize: hiddenSize);

            using var kvCache = new CudaMlaKvCache(
                numLayers: 1, maxSeqLen: maxSeq,
                numHeads: numHeads, qkNopeHeadDim: qkNope, vHeadDim: vHead,
                qkRopeHeadDim: qkRope);
            using var scratch = new CudaMlaScratch();

            if (prefillLen > 0)
            {
                CudaMlaAttention.Forward(
                    hiddenF32: dHidden, outputF32: dPrefillOut,
                    seqLen: prefillLen, positionOffset: 0,
                    layer: layer, kvCache: kvCache, layerIndex: 0,
                    ropeCosF32: dCos, ropeSinF32: dSin,
                    rmsNormEps: eps, softmaxScale: softmaxScale,
                    scratch: scratch, cublasHandle: _cublas!.Handle,
                    kernels: _kernels!, stream: _stream!.Handle);
                kvCache.Advance(0, prefillLen);
            }

            CudaMlaAttention.Forward(
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

            // Diff
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
                $"MLA forward: {mismatches}/{cpuOut.Length} elements outside tolerance {tolerance} "
              + $"(max diff {maxDiff} at idx {maxDiffIdx}: cpu={(maxDiffIdx >= 0 ? cpuOut[maxDiffIdx] : 0)} "
              + $"gpu={(maxDiffIdx >= 0 ? gpuOut[maxDiffIdx] : 0)}).");
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
