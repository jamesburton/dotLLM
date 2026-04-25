using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end equivalence tests for the GPU MoE SwiGLU FFN forward
/// (<see cref="CudaMoeFfn.Forward"/>): exercises routing softmax + top-k +
/// optional renorm, per-expert SwiGLU MLP (gate/up/swiglu/down), per-(token,
/// slot) weighted accumulation, and shared-expert path (with and without
/// sigmoid gate). Compares against
/// <see cref="MoeSwiGluMlp.ExecuteWithSharedExpert"/> running the same fixture
/// scalar on the CPU.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is the gate.</b> The CudaMoeFfn helper is the contract the
/// next agent uses to wire MoE into <c>CudaTransformerModel.Forward</c>.
/// Pass here means the MLA + MoE pair (both F32 Phase 1) is ready for
/// end-to-end DeepSeek-V2/V3 dispatch.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMoeFfnTests : IDisposable
{
    private const float DefaultTolerance = 1e-3f;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaCublasHandle? _cublas;
    private readonly CudaKernels? _kernels;

    public CudaMoeFfnTests()
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

    // ── Small synthetic fixture: 2 experts, top-2, no shared expert ──
    [SkippableFact]
    public void MoeFfn_Small_RoutedOnly_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        Run(seqLen: 1,
            numExperts: 2, topK: 2,
            hidden: 64, intermediate: 128,
            normTopKProb: true,
            numSharedExperts: 0, sharedIntermediate: 0, hasSharedGate: false,
            seed: 42);
    }

    [SkippableFact]
    public void MoeFfn_Small_Prefill_RoutedOnly_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        Run(seqLen: 4,
            numExperts: 2, topK: 2,
            hidden: 64, intermediate: 128,
            normTopKProb: true,
            numSharedExperts: 0, sharedIntermediate: 0, hasSharedGate: false,
            seed: 7);
    }

    [SkippableFact]
    public void MoeFfn_Small_NoRenorm_MatchesCpuOracle()
    {
        // Qwen1.5-MoE convention: norm_topk_prob = false. Routed weights stay
        // raw softmax probabilities.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        Run(seqLen: 2,
            numExperts: 4, topK: 2,
            hidden: 32, intermediate: 64,
            normTopKProb: false,
            numSharedExperts: 0, sharedIntermediate: 0, hasSharedGate: false,
            seed: 11);
    }

    [SkippableFact]
    public void MoeFfn_Small_DeepSeekShared_MatchesCpuOracle()
    {
        // DeepSeek shape: 2 routed experts top-2, 2 shared experts (no gate).
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        Run(seqLen: 2,
            numExperts: 4, topK: 2,
            hidden: 64, intermediate: 128,
            normTopKProb: true,
            numSharedExperts: 2, sharedIntermediate: 96, hasSharedGate: false,
            seed: 99);
    }

    [SkippableFact]
    public void MoeFfn_Small_QwenSharedWithGate_MatchesCpuOracle()
    {
        // Qwen1.5-MoE shape: 1 shared expert + sigmoid shared_expert_gate.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        Run(seqLen: 1,
            numExperts: 4, topK: 2,
            hidden: 32, intermediate: 64,
            normTopKProb: false,
            numSharedExperts: 1, sharedIntermediate: 96, hasSharedGate: true,
            seed: 21);
    }

    // ── DeepSeek-V2-Lite-shaped synthetic fixture ──
    // Real V2-Lite: hidden=2048, n_routed_experts=64, num_experts_per_tok=6,
    // moe_intermediate_size=1408, n_shared_experts=2 (each
    // moe_intermediate_size wide).
    //
    // For this Phase 1 test we use 16 routed experts (instead of 64) keeping
    // the production per-expert dims (1408×2048) and top-6 routing. 64
    // experts × 192 weight matrices × 11.5 MB each comfortably fits a fresh
    // CUDA context's free pool (~11 GB on RTX 3060) but consistently OOMs
    // the per-method context recycled across the 6 tests in this file —
    // WDDM does not always reclaim the prior context's pool by the time
    // xUnit creates the next fixture. 16 experts validates the same per-
    // expert math + routing fan-out at ~550 MB.
    [SkippableFact]
    public void MoeFfn_DeepSeekV2LiteShape_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMoeKernels, "MoE PTX kernels not available");

        // Tolerance widened: 16-expert × 1408-intermediate × 2048-hidden chain
        // amplifies F32 reduction-order drift. cuBLAS Tensor-Core SGEMM order
        // diverges from the scalar reference; per-element drift bounded by
        // ~1e-2 across the chain. Same trade-off as the V2-Lite-shaped MLA
        // test in CudaMlaForwardTests.
        Run(seqLen: 1,
            numExperts: 16, topK: 6,
            hidden: 2048, intermediate: 1408,
            normTopKProb: true,
            numSharedExperts: 2, sharedIntermediate: 1408, hasSharedGate: false,
            seed: 1234,
            tolerance: 2e-2f);
    }

    /// <summary>
    /// Builds a synthetic MoE fixture, runs the GPU forward through
    /// <see cref="CudaMoeFfn.Forward"/>, and compares against
    /// <see cref="MoeSwiGluMlp.ExecuteWithSharedExpert"/>.
    /// </summary>
    private unsafe void Run(int seqLen,
        int numExperts, int topK,
        int hidden, int intermediate,
        bool normTopKProb,
        int numSharedExperts, int sharedIntermediate, bool hasSharedGate,
        int seed,
        float tolerance = DefaultTolerance)
    {
        var rng = new Random(seed);

        // Random weights and inputs (small magnitude → stays in well-conditioned
        // softmax range and avoids exp overflow in the routing softmax).
        float[] hidden_in = RandomArr(rng, seqLen * hidden, 0.3f);
        float[] router = RandomArr(rng, numExperts * hidden, 0.05f);

        // Per-routed-expert weights.
        float[][] w1 = new float[numExperts][];
        float[][] w2 = new float[numExperts][];
        float[][] w3 = new float[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            w1[e] = RandomArr(rng, intermediate * hidden, 0.05f);
            w3[e] = RandomArr(rng, intermediate * hidden, 0.05f);
            w2[e] = RandomArr(rng, hidden * intermediate, 0.05f);
        }

        // Per-shared-expert weights.
        float[][] sw1 = new float[numSharedExperts][];
        float[][] sw2 = new float[numSharedExperts][];
        float[][] sw3 = new float[numSharedExperts][];
        for (int s = 0; s < numSharedExperts; s++)
        {
            sw1[s] = RandomArr(rng, sharedIntermediate * hidden, 0.05f);
            sw3[s] = RandomArr(rng, sharedIntermediate * hidden, 0.05f);
            sw2[s] = RandomArr(rng, hidden * sharedIntermediate, 0.05f);
        }
        float[] sharedGate = hasSharedGate ? RandomArr(rng, hidden, 0.05f) : Array.Empty<float>();

        // ── CPU oracle ──
        float[] cpuOut = new float[seqLen * hidden];
        var cpuW1Ptrs = new nint[numExperts];
        var cpuW2Ptrs = new nint[numExperts];
        var cpuW3Ptrs = new nint[numExperts];
        var cpuSw1Ptrs = new nint[numSharedExperts];
        var cpuSw2Ptrs = new nint[numSharedExperts];
        var cpuSw3Ptrs = new nint[numSharedExperts];
        var pins = new List<System.Runtime.InteropServices.GCHandle>();
        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                var h1 = System.Runtime.InteropServices.GCHandle.Alloc(w1[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                var h2 = System.Runtime.InteropServices.GCHandle.Alloc(w2[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                var h3 = System.Runtime.InteropServices.GCHandle.Alloc(w3[e], System.Runtime.InteropServices.GCHandleType.Pinned);
                cpuW1Ptrs[e] = h1.AddrOfPinnedObject();
                cpuW2Ptrs[e] = h2.AddrOfPinnedObject();
                cpuW3Ptrs[e] = h3.AddrOfPinnedObject();
                pins.Add(h1); pins.Add(h2); pins.Add(h3);
            }
            for (int s = 0; s < numSharedExperts; s++)
            {
                var h1 = System.Runtime.InteropServices.GCHandle.Alloc(sw1[s], System.Runtime.InteropServices.GCHandleType.Pinned);
                var h2 = System.Runtime.InteropServices.GCHandle.Alloc(sw2[s], System.Runtime.InteropServices.GCHandleType.Pinned);
                var h3 = System.Runtime.InteropServices.GCHandle.Alloc(sw3[s], System.Runtime.InteropServices.GCHandleType.Pinned);
                cpuSw1Ptrs[s] = h1.AddrOfPinnedObject();
                cpuSw2Ptrs[s] = h2.AddrOfPinnedObject();
                cpuSw3Ptrs[s] = h3.AddrOfPinnedObject();
                pins.Add(h1); pins.Add(h2); pins.Add(h3);
            }

            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden: hidden_in,
                gateWeights: router,
                expertsW1: cpuW1Ptrs,
                expertsW2: cpuW2Ptrs,
                expertsW3: cpuW3Ptrs,
                output: cpuOut,
                numExperts: numExperts,
                numExpertsPerTok: topK,
                hiddenSize: hidden,
                intermediateSize: intermediate,
                seqLen: seqLen,
                normTopKProb: normTopKProb,
                sharedGateProj: cpuSw1Ptrs,
                sharedUpProj: cpuSw3Ptrs,
                sharedDownProj: cpuSw2Ptrs,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: sharedGate);
        }
        finally
        {
            foreach (var h in pins) h.Free();
        }

        // ── GPU forward ──
        var allocs = new List<nint>();
        try
        {
            nint dHidden = AllocAndUploadF32(hidden_in, allocs);
            nint dRouter = AllocAndUploadF32(router, allocs);

            nint[] dW1 = new nint[numExperts];
            nint[] dW2 = new nint[numExperts];
            nint[] dW3 = new nint[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                dW1[e] = AllocAndUploadF32(w1[e], allocs);
                dW2[e] = AllocAndUploadF32(w2[e], allocs);
                dW3[e] = AllocAndUploadF32(w3[e], allocs);
            }
            nint[] dSw1 = new nint[numSharedExperts];
            nint[] dSw2 = new nint[numSharedExperts];
            nint[] dSw3 = new nint[numSharedExperts];
            for (int s = 0; s < numSharedExperts; s++)
            {
                dSw1[s] = AllocAndUploadF32(sw1[s], allocs);
                dSw2[s] = AllocAndUploadF32(sw2[s], allocs);
                dSw3[s] = AllocAndUploadF32(sw3[s], allocs);
            }
            nint dSharedGate = hasSharedGate ? AllocAndUploadF32(sharedGate, allocs) : (nint)0;
            nint dOut = AllocF32(seqLen * hidden, allocs);

            var weights = new CudaMoeLayerWeights(
                numExperts: numExperts,
                numExpertsPerTok: topK,
                hiddenSize: hidden,
                moeIntermediateSize: intermediate,
                normTopKProb: normTopKProb,
                router: dRouter,
                gateProj: dW1, upProj: dW3, downProj: dW2,
                numSharedExperts: numSharedExperts,
                sharedIntermediateSize: sharedIntermediate,
                sharedGateProj: numSharedExperts > 0 ? dSw1 : null,
                sharedUpProj: numSharedExperts > 0 ? dSw3 : null,
                sharedDownProj: numSharedExperts > 0 ? dSw2 : null,
                sharedExpertGate: dSharedGate);

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
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > tolerance)
                {
                    mismatches++;
                    if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
                }
            }
            Assert.True(mismatches == 0,
                $"MoE forward: {mismatches}/{cpuOut.Length} elements outside tolerance {tolerance} "
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

    private static float[] RandomArr(Random rng, int n, float scale)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return arr;
    }
}
