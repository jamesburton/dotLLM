using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU forward for the Gemma-4 MoE branch (F32). Mirrors
/// <see cref="CudaMoeFfn.Forward"/>'s routing + bucketing + per-expert MLP
/// structure but with the gemma4-specific deltas:
/// <list type="number">
///   <item><b>Custom router.</b> The router logits read a DIFFERENTLY-normed input
///     (<c>rms(attn_out)·RouterScale·(1/√H)</c>) than the experts
///     (<c>rms(attn_out)·pre_ffw_norm_2</c>). The caller supplies both already-normed
///     buffers; this helper runs the router GEMV on the router input and the expert
///     MLP on the expert input.</item>
///   <item><b>GeGLU experts.</b> tanh-approx GELU (<c>geglu_tanh_f32</c>) instead of
///     SwiGLU.</item>
///   <item><b>Renorm clamp.</b> Top-k weights renormalise to sum 1 with the gemma4
///     <c>6.103515625e-05</c> denominator clamp.</item>
///   <item><b>Per-expert down scale.</b> Already folded into the F32 down bank at
///     load (<see cref="CudaGemma4WeightsLoader"/>), so the weighted accumulation
///     applies only the routing weight.</item>
/// </list>
/// The experts live in a standard <see cref="CudaMoeLayerWeights"/> (F32 banks: gate
/// = <see cref="CudaMoeLayerWeights.GateProj"/>, up = <see cref="CudaMoeLayerWeights.UpProj"/>,
/// down = <see cref="CudaMoeLayerWeights.DownProj"/>). Output is fully overwritten.
/// </summary>
public static unsafe class CudaGemma4Ffn
{
    /// <summary>
    /// Runs the gemma4 MoE branch: router (on <paramref name="routerInF32"/>) →
    /// softmax → top-k → clamped renorm → per-expert GeGLU MLP (on
    /// <paramref name="expertInF32"/>) → weighted sum into <paramref name="outputF32"/>.
    /// </summary>
    public static void ForwardMoe(
        nint expertInF32, nint routerInF32, nint outputF32,
        int seqLen,
        CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream)
    {
        if (!kernels.HasMoeKernels)
            throw new InvalidOperationException(
                "MoE kernels not available. Compile native/kernels/moe_ffn.cu to PTX.");
        if (!kernels.HasGemma4Kernels)
            throw new InvalidOperationException(
                "Gemma4 kernels not available. Compile native/kernels/gemma4_f32.cu to PTX.");
        if (seqLen <= 0) return;

        int hidden = weights.HiddenSize;
        int E = weights.NumExperts;
        int K = weights.NumExpertsPerTok;
        int I = weights.MoeIntermediateSize;
        int totalAssign = seqLen * K;

        scratch.EnsureCapacity(seqLen, weights);

        // 1. Clear output.
        kernels.LaunchMoeZeroF32(outputF32, seqLen * hidden, stream);

        // 2. Router GEMV on the ROUTER input: logits[seqLen, E] = routerIn @ router^T.
        CudaGemm.LinearF32(
            cublasHandle, routerInF32, weights.Router, scratch.Logits,
            seqLen, hidden, E, stream);

        // 3. Softmax + top-k selection.
        kernels.LaunchMoeSoftmaxTopk(
            scratch.Logits, scratch.TopkIdx, scratch.TopkWeight,
            seqLen, E, K, stream);

        // 4. Renorm top-k weights to sum 1 with the gemma4 6.1e-5 clamp.
        kernels.LaunchMoeRenormTopkClampedF32(scratch.TopkWeight, seqLen, K, stream);

        // 5. Download top-k indices to host for per-expert bucketing (small).
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        int[] topkIdxHost = new int[totalAssign];
        fixed (int* p = topkIdxHost)
            CudaDriverApi.cuMemcpyDtoH_v2(
                (nint)p, scratch.TopkIdx,
                (nuint)((long)totalAssign * sizeof(int))).ThrowOnError();

        // 6. Bucket assignments per expert (host-side), matching CudaMoeFfn.
        Span<int> counts = stackalloc int[E];
        counts.Clear();
        for (int i = 0; i < totalAssign; i++)
        {
            int e = topkIdxHost[i];
            if ((uint)e < (uint)E) counts[e]++;
        }

        Span<int> offsets = stackalloc int[E + 1];
        int running = 0;
        for (int e = 0; e < E; e++) { offsets[e] = running; running += counts[e]; }
        offsets[E] = running;

        int[] bucketTokens = new int[totalAssign];
        int[] bucketSlots = new int[totalAssign];
        Span<int> cursor = stackalloc int[E];
        for (int e = 0; e < E; e++) cursor[e] = offsets[e];
        for (int t = 0; t < seqLen; t++)
        {
            for (int slot = 0; slot < K; slot++)
            {
                int e = topkIdxHost[t * K + slot];
                if ((uint)e >= (uint)E) continue;
                int pos = cursor[e]++;
                bucketTokens[pos] = t;
                bucketSlots[pos] = slot;
            }
        }

        // 7. Per-expert grouped path: gather → gate/up GEMM → GeGLU → down GEMM → axpy.
        for (int e = 0; e < E; e++)
        {
            int batch = counts[e];
            if (batch == 0) continue;
            int start = offsets[e];

            // Upload this expert's bucket token ids.
            fixed (int* tp = bucketTokens)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(
                    scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                    (nint)(tp + start),
                    (nuint)((long)batch * sizeof(int))).ThrowOnError();
            }

            // Gather the expert-input rows for this expert.
            kernels.LaunchMoeGatherTokenRowsF32(
                expertInF32, scratch.GatheredInput,
                scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                batch, hidden, stream);

            // gate[batch, I] = gathered[batch, hidden] × GateProj[e][I, hidden]^T
            // up  [batch, I] = gathered[batch, hidden] × UpProj[e][I, hidden]^T
            CudaGemm.LinearF32(cublasHandle, scratch.GatheredInput, weights.GateProj[e],
                scratch.GateBatch, batch, hidden, I, stream);
            CudaGemm.LinearF32(cublasHandle, scratch.GatheredInput, weights.UpProj[e],
                scratch.UpBatch, batch, hidden, I, stream);

            // GeGLU(gate, up) → silu[batch, I].
            kernels.LaunchGeGLUTanhF32(scratch.GateBatch, scratch.UpBatch, scratch.SiluBatch,
                I, batch, stream);

            // down[batch, hidden] = silu[batch, I] × DownProj[e][hidden, I]^T
            // (DownProj already carries the per-expert ffn_down_exps.scale.)
            CudaGemm.LinearF32(cublasHandle, scratch.SiluBatch, weights.DownProj[e],
                scratch.DownBatch, batch, I, hidden, stream);

            // Per-slot weighted accumulate into the output. Group by slot so the
            // axpy reads the matching routing weight (weight[tok*K + slot]).
            for (int slot = 0; slot < K; slot++)
            {
                int slotBatchCount = 0;
                for (int b = 0; b < batch; b++)
                    if (bucketSlots[start + b] == slot) slotBatchCount++;
                if (slotBatchCount == 0) continue;

                if (slotBatchCount == batch)
                {
                    kernels.LaunchMoeAxpyScaledRowF32(
                        outputF32, scratch.DownBatch,
                        scratch.TopkWeight,
                        scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                        batch, hidden, K, slot, stream);
                }
                else
                {
                    for (int b = 0; b < batch; b++)
                    {
                        if (bucketSlots[start + b] != slot) continue;
                        int tokenId = bucketTokens[start + b];
                        CudaDriverApi.cuMemcpyHtoD_v2(
                            scratch.SingleTokenScratch,
                            (nint)(&tokenId), sizeof(int)).ThrowOnError();
                        kernels.LaunchMoeAxpyScaledRowF32(
                            outputF32,
                            scratch.DownBatch + (nint)((long)b * hidden * sizeof(float)),
                            scratch.TopkWeight,
                            scratch.SingleTokenScratch,
                            1, hidden, K, slot, stream);
                    }
                }
            }
        }
    }
}
