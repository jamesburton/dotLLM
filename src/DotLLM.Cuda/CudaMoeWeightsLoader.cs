using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda;

/// <summary>
/// Loader that uploads a per-layer <see cref="MoeLayerWeights"/> bundle (Mixtral /
/// Qwen-MoE / Phi-3.5-MoE / DeepSeek-V2/V3 routed-MoE FFN) to GPU as F32.
/// CPU-side weights are already F32 (the safetensors / GGUF MoE loader upcasts
/// the per-expert quantized blocks to F32 once at load time). We allocate one
/// device buffer per expert + one router buffer + per-shared-expert buffers and
/// synchronously copy.
/// </summary>
/// <remarks>
/// <para>
/// <b>Phase A simplification.</b> Every per-expert <c>w1</c>/<c>w2</c>/<c>w3</c>
/// projection and the router gate are uploaded as raw F32 row-major. The MoE
/// forward path (<see cref="CudaMoeFfn.Forward"/>) operates fully in F32 to
/// match the CPU oracle <c>MoeSwiGluMlp.Execute</c> byte-for-byte algorithmically.
/// FP16 / quantized paths are deferred to follow-ups.
/// </para>
/// <para>
/// <b>Lifetime.</b> All allocations are added to the supplied <c>allocs</c>
/// list, which the parent <see cref="CudaWeights"/> owns and frees on dispose.
/// </para>
/// </remarks>
internal static unsafe class CudaMoeWeightsLoader
{
    /// <summary>
    /// Uploads a single MoE layer's projections (router + routed experts +
    /// optional shared experts + optional shared-expert sigmoid gate) to F32
    /// device buffers.
    /// </summary>
    /// <param name="cpuLayer">Per-layer weight bundle from the safetensors / GGUF loader.</param>
    /// <param name="allocs">Allocation list to extend (caller owns + frees on dispose).</param>
    /// <returns>Populated <see cref="CudaMoeLayerWeights"/> with device pointers.</returns>
    public static CudaMoeLayerWeights LoadLayer(
        in TransformerLayerWeights cpuLayer, List<nint> allocs)
    {
        var moe = cpuLayer.Moe
            ?? throw new InvalidOperationException(
                "CudaMoeWeightsLoader.LoadLayer called with non-MoE layer.");

        int numExperts = moe.NumExperts;
        int hidden = moe.HiddenSize;
        int moeIntermediate = moe.IntermediateSize;
        long expertGateBytes = (long)moeIntermediate * hidden * sizeof(float);
        long expertDownBytes = (long)hidden * moeIntermediate * sizeof(float);

        // Router: [numExperts, hiddenSize] — managed F32 array on the CPU side.
        nint router = UploadF32Array(moe.Gate, allocs);

        // Per-routed-expert projections.
        var gateProj = new nint[numExperts];
        var upProj = new nint[numExperts];
        var downProj = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            gateProj[e] = UploadF32(moe.W1[e], (long)moeIntermediate * hidden, allocs);
            upProj[e] = UploadF32(moe.W3[e], (long)moeIntermediate * hidden, allocs);
            downProj[e] = UploadF32(moe.W2[e], (long)hidden * moeIntermediate, allocs);
        }

        // Shared experts (DeepSeek-V2/V3 + Qwen1.5-MoE).
        int numSharedExperts = moe.NumSharedExperts;
        int sharedIntermediate = moe.SharedIntermediateSize;
        nint[] sharedGate = Array.Empty<nint>();
        nint[] sharedUp = Array.Empty<nint>();
        nint[] sharedDown = Array.Empty<nint>();
        if (numSharedExperts > 0 && sharedIntermediate > 0)
        {
            sharedGate = new nint[numSharedExperts];
            sharedUp = new nint[numSharedExperts];
            sharedDown = new nint[numSharedExperts];
            for (int s = 0; s < numSharedExperts; s++)
            {
                sharedGate[s] = UploadF32(moe.SharedGateProj[s], (long)sharedIntermediate * hidden, allocs);
                sharedUp[s] = UploadF32(moe.SharedUpProj[s], (long)sharedIntermediate * hidden, allocs);
                sharedDown[s] = UploadF32(moe.SharedDownProj[s], (long)hidden * sharedIntermediate, allocs);
            }
        }

        // Optional Qwen1.5-MoE shared-expert sigmoid gate.
        nint sharedExpertGate = moe.SharedExpertGate is float[] gate
            ? UploadF32Array(gate, allocs)
            : (nint)0;

        return new CudaMoeLayerWeights(
            numExperts, moe.NumExpertsPerTok, hidden, moeIntermediate,
            moe.NormTopKProb,
            router,
            gateProj, upProj, downProj,
            numSharedExperts, sharedIntermediate,
            sharedGate, sharedUp, sharedDown,
            sharedExpertGate);
    }

    /// <summary>
    /// Loads a single MoE layer's projections to GPU as raw GGUF quantized
    /// bytes per expert (zero-copy upload from CPU mmap). Used when the source
    /// has populated raw quant views (<see cref="DotLLM.Models.Architectures.MoeLayerWeights.HasRawQuantView"/>)
    /// — i.e. GGUF-loaded DeepSeek-V2 weights. Router stays F32. The forward
    /// path (<see cref="CudaMoeFfn.Forward"/> quantized branch) dequantizes
    /// per call into reused F16 scratch then HGEMMs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Memory budget</b> at DeepSeek-V2-Lite-Q4_K_M scale:
    /// 64 routed experts × 3 projections × ~5 MB/expert ≈ 1.0 GB raw bytes
    /// per layer × 26 MoE layers = ~26 GB. Plus shared experts (×2 fused into
    /// one wider MLP at width 2816) ≈ ~10 MB/layer × 26 = ~260 MB. Plus router
    /// (F32, ~0.5 MB × 26 = ~13 MB). Total ~26.3 GB on GPU — STILL exceeds
    /// 12 GB RTX 3060. We rely on the existing CudaWeights tracker to surface
    /// the OOM with context, and the next perf milestone (grouped-GEMM with
    /// raw quant compaction) is the path to fitting full V2-Lite. For now the
    /// only fully-loadable target is V2-Lite Q3_K or smaller, OR partial loads
    /// via numGpuLayers / NumLayers patching.
    /// </para>
    /// </remarks>
    public static CudaMoeLayerWeights LoadLayerQuant(
        in TransformerLayerWeights cpuLayer, List<nint> allocs)
    {
        var moe = cpuLayer.Moe
            ?? throw new InvalidOperationException(
                "CudaMoeWeightsLoader.LoadLayerQuant called with non-MoE layer.");
        if (!moe.HasRawQuantView)
            throw new InvalidOperationException(
                "CudaMoeWeightsLoader.LoadLayerQuant requires the source MoeLayerWeights to carry " +
                "raw GGUF quant views (the GGUF MoE loader populates these alongside the F32 dequants).");

        int numExperts = moe.NumExperts;
        int hidden = moe.HiddenSize;
        int moeIntermediate = moe.IntermediateSize;

        // Router stays F32 (small — numExperts × hidden floats).
        nint router = UploadF32Array(moe.Gate, allocs);

        // Per-routed-expert: upload raw quant bytes from the GGUF mmap. Each
        // 3D fused tensor has a single base pointer; per-expert byte offset
        // is e * (M * RowByteSize(K, qt)). For w1/w3 (gate/up) M=intermediate,
        // K=hidden. For w2 (down) M=hidden, K=intermediate.
        var gateProj = new nint[numExperts];
        var upProj = new nint[numExperts];
        var downProj = new nint[numExperts];
        long w1RowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(moe.GateExpsKDim, moe.GateExpsRawQt);
        long w3RowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(moe.UpExpsKDim, moe.UpExpsRawQt);
        long w2RowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(moe.DownExpsKDim, moe.DownExpsRawQt);
        long w1ExpertBytes = moe.GateExpsMDim * w1RowBytes;
        long w3ExpertBytes = moe.UpExpsMDim * w3RowBytes;
        long w2ExpertBytes = moe.DownExpsMDim * w2RowBytes;
        for (int e = 0; e < numExperts; e++)
        {
            gateProj[e] = UploadRawBytes(moe.GateExpsRaw + (nint)(e * w1ExpertBytes), w1ExpertBytes, allocs);
            upProj[e] = UploadRawBytes(moe.UpExpsRaw + (nint)(e * w3ExpertBytes), w3ExpertBytes, allocs);
            downProj[e] = UploadRawBytes(moe.DownExpsRaw + (nint)(e * w2ExpertBytes), w2ExpertBytes, allocs);
        }

        // Shared expert (DeepSeek-V2/V3 fuses N shared into a single wider MLP).
        int numSharedExperts = moe.NumSharedExperts;
        int sharedIntermediate = moe.SharedIntermediateSize;
        nint[] sharedGate = Array.Empty<nint>();
        nint[] sharedUp = Array.Empty<nint>();
        nint[] sharedDown = Array.Empty<nint>();
        QuantizationType sgQt = QuantizationType.F32;
        QuantizationType suQt = QuantizationType.F32;
        QuantizationType sdQt = QuantizationType.F32;
        if (numSharedExperts > 0 && sharedIntermediate > 0 && moe.SharedGateRaw.Length > 0
            && moe.SharedGateRaw[0] != 0)
        {
            long sgRowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(hidden, moe.SharedGateRawQt);
            long suRowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(hidden, moe.SharedUpRawQt);
            long sdRowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(sharedIntermediate, moe.SharedDownRawQt);
            long sgBytes = sharedIntermediate * sgRowBytes;
            long suBytes = sharedIntermediate * suRowBytes;
            long sdBytes = hidden * sdRowBytes;

            sharedGate = [UploadRawBytes(moe.SharedGateRaw[0], sgBytes, allocs)];
            sharedUp = [UploadRawBytes(moe.SharedUpRaw[0], suBytes, allocs)];
            sharedDown = [UploadRawBytes(moe.SharedDownRaw[0], sdBytes, allocs)];
            sgQt = moe.SharedGateRawQt;
            suQt = moe.SharedUpRawQt;
            sdQt = moe.SharedDownRawQt;
            numSharedExperts = 1;  // Collapse to fused-1 to match array length.
        }

        nint sharedExpertGate = moe.SharedExpertGate is float[] gateArr
            ? UploadF32Array(gateArr, allocs)
            : (nint)0;

        return new CudaMoeLayerWeights(
            numExperts, moe.NumExpertsPerTok, hidden, moeIntermediate,
            moe.NormTopKProb,
            router,
            gateProj, upProj, downProj,
            numSharedExperts, sharedIntermediate,
            sharedGate, sharedUp, sharedDown,
            sharedExpertGate,
            precision: MoePrecision.Quantized,
            gateProjQuantType: moe.GateExpsRawQt,
            upProjQuantType: moe.UpExpsRawQt,
            downProjQuantType: moe.DownExpsRawQt,
            sharedGateProjQuantType: sgQt,
            sharedUpProjQuantType: suQt,
            sharedDownProjQuantType: sdQt);
    }

    private static unsafe nint UploadRawBytes(nint hostPtr, long bytes, List<nint> allocs)
    {
        if (hostPtr == 0)
            throw new InvalidOperationException("UploadRawBytes called with null host pointer.");
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    private static nint UploadF32(nint hostPtr, long elementCount, List<nint> allocs)
    {
        if (hostPtr == 0)
            throw new InvalidOperationException("UploadF32 called with null host pointer.");
        long bytes = elementCount * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    private static nint UploadF32Array(float[] data, List<nint> allocs)
    {
        long bytes = (long)data.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        fixed (float* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)bytes).ThrowOnError();
        return devPtr;
    }
}
