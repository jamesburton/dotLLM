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
