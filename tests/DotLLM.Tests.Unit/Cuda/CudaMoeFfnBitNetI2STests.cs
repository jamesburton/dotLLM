using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-vs-GPU parity gate for the BitNet-ternary (I2_S) routed-expert MoE forward
/// (issue #246): <see cref="CudaMoeFfn.Forward"/>'s new <see cref="MoePrecision.BitNetI2S"/>
/// branch (<c>ForwardBitNetI2S</c>) against the CPU oracle
/// <see cref="MoeSwiGluMlp.ExecuteBitNetMoe"/> on the SAME synthetic ternary experts,
/// router, per-expert FFN Sub-LN, and (optional) router bias.
/// </summary>
/// <remarks>
/// <para>
/// Builds synthetic per-expert I2_S banks the same way
/// <c>MoteBitNetMoeLoaderScaffoldTests</c> does (payload-only pack via the shared
/// <c>PackPayload</c> bit layout), then uploads each expert's payload to the GPU with its
/// own trailing F32 scale appended — the exact self-contained-tensor buffer
/// <see cref="CudaKernels.LaunchI2_SGemvF32In"/> requires. See
/// <see cref="CudaMoeWeightsLoader.LoadLayerBitNetI2S"/>'s remarks for why this two-copy
/// upload replaces the originally-anticipated device-side repack kernel.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMoeFfnBitNetI2STests : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMoeFfnBitNetI2STests()
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
    public void BitNetMoe_Decode_Top1_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");

        Run(seqLen: 1, numExperts: 3, topK: 1, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: false, zeroSkipExpert: false, seed: 42);
    }

    [SkippableFact]
    public void BitNetMoe_Decode_Top2_WithGateBias_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");
        Skip.IfNot(_kernels!.HasMoeGateBiasAdd, "moe_gate_bias_add_f32 not available");

        Run(seqLen: 1, numExperts: 4, topK: 2, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: true, zeroSkipExpert: false, seed: 7);
    }

    [SkippableFact]
    public void BitNetMoe_Prefill_MultipleTokensPerExpert_MatchesCpuOracle()
    {
        // Exercises the per-row I2_S GEMV loop (issue #246 scope: no batched/grouped I2_S
        // GEMM this pass) — seqLen > 1 routes multiple rows to the same expert.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");

        Run(seqLen: 5, numExperts: 3, topK: 1, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: true, zeroSkipExpert: false, seed: 11);
    }

    [SkippableFact]
    public void BitNetMoe_SkipExpert_ZeroDownProj_ProducesExactZero()
    {
        // identity-MoTE skip expert: an all-zero down_proj must output exactly 0 for any
        // token routed to it — no special-casing needed on either the CPU or GPU path.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasBitNetMoeKernels, "BitNet-MoE CUDA kernels not available");

        Run(seqLen: 6, numExperts: 3, topK: 1, hidden: 128, intermediate: 256,
            normTopKProb: true, withGateBias: false, zeroSkipExpert: true, seed: 2026);
    }

    private unsafe void Run(int seqLen, int numExperts, int topK, int hidden, int intermediate,
        bool normTopKProb, bool withGateBias, bool zeroSkipExpert, int seed,
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
                for (int i = 0; i < intermediate; i++) ffnSubNorm[e][i] += 1.0f; // keep norm weight ~1

                sbyte[] gT = RandomTernary(rng, intermediate * hidden);
                sbyte[] uT = RandomTernary(rng, intermediate * hidden);
                sbyte[] dT = (zeroSkipExpert && e == 0)
                    ? new sbyte[hidden * intermediate]      // skip expert: all-zero down_proj
                    : RandomTernary(rng, hidden * intermediate);

                PackPayload(gT, gateBank + e * gateUpRowBytes);
                PackPayload(uT, upBank + e * gateUpRowBytes);
                PackPayload(dT, downBank + e * downRowBytes);
            }

            // ── CPU oracle: MoeSwiGluMlp.ExecuteBitNetMoe directly on the payload-only banks. ──
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

            // ── GPU forward: upload per-expert payload+tail-scale buffers, router, bias, sub-norm. ──
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

            nint dOut = AllocF32(seqLen * hidden, allocs);

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

            using var scratch = new CudaMoeScratch();

            CudaMoeFfn.Forward(
                hiddenF32: dHidden, outputF32: dOut,
                seqLen: seqLen,
                weights: weights,
                scratch: scratch, cublasHandle: _cublas!.Handle,
                kernels: _kernels!, stream: _stream!.Handle);
            _stream.Synchronize();

            float[] gpuOut = new float[seqLen * hidden];
            fixed (float* p = gpuOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut,
                    (nuint)(gpuOut.Length * sizeof(float))).ThrowOnError();

            int mismatches = 0;
            float maxDiff = 0f;
            int maxDiffIdx = -1;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float tol = tolerance + tolerance * MathF.Abs(cpuOut[i]);
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > tol)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"BitNet-MoE forward: {mismatches}/{cpuOut.Length} elements outside tolerance "
              + $"(max diff {maxDiff} at idx {maxDiffIdx}: cpu={(maxDiffIdx >= 0 ? cpuOut[maxDiffIdx] : 0)} "
              + $"gpu={(maxDiffIdx >= 0 ? gpuOut[maxDiffIdx] : 0)}).");

            if (zeroSkipExpert)
            {
                // Every element must be exactly zero for tokens routed to expert 0 — re-derive
                // routing on CPU to know which output rows to check (matches the scaffold test's
                // convention). Cheap enough at this size to just re-run Route().
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
                bool sawSkip = false;
                for (int t = 0; t < seqLen; t++)
                {
                    bool tokenOnlyRoutesToSkip = true;
                    for (int slot = 0; slot < topK; slot++)
                        if (assignExpert[t * topK + slot] != 0) tokenOnlyRoutesToSkip = false;
                    if (!tokenOnlyRoutesToSkip) continue;
                    sawSkip = true;
                    for (int j = 0; j < hidden; j++)
                    {
                        Assert.Equal(0f, cpuOut[t * hidden + j]);
                        Assert.Equal(0f, gpuOut[t * hidden + j]);
                    }
                }
                Assert.True(sawSkip, "expected at least one token routed ONLY to the skip expert (topK=1 fixture)");
            }
        }
        finally
        {
            foreach (var p in allocs)
                CudaDriverApi.cuMemFree_v2(p);
            NativeMemory.Free(gateBank); NativeMemory.Free(upBank); NativeMemory.Free(downBank);
        }
    }

    private const DotLLM.Core.Configuration.QuantizationType QuantizationTypeI2S =
        DotLLM.Core.Configuration.QuantizationType.I2_S;

    /// <summary>
    /// Mirrors <c>CudaMoeWeightsLoader.UploadI2SExpertWithTailScale</c>: uploads a payload-only
    /// I2_S expert slice plus its own trailing F32 scale into one fresh device buffer — the
    /// self-contained-tensor layout <see cref="CudaKernels.LaunchI2_SGemvF32In"/> requires.
    /// </summary>
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

    // ── I2_S packing (mirrors MoteBitNetMoeLoaderScaffoldTests / MoeIndexedMatmulI2STests) ──
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
