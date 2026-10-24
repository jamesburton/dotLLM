using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU forward pass for one Mixture-of-Experts SwiGLU FFN layer — Phase 1
/// F32 path. Runs the equivalent of
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.ExecuteWithSharedExpert</c>:
/// per-token routing softmax + top-k pick → optional renorm → per-expert
/// SwiGLU MLP (gate/up/swiglu/down) accumulated into the output → unconditional
/// shared-expert SwiGLU MLP(s) (optionally sigmoid-gated) summed into the
/// output.
/// </summary>
/// <remarks>
/// <para>
/// <b>F32 throughout.</b> Inputs / outputs / weights are F32. Phase 1 keeps
/// the entire MoE pipeline in F32 to match the CPU oracle byte-for-byte
/// algorithmically (numerical drift comes only from the order of GPU
/// floating-point reductions). Quantized / FP16 paths are an explicit
/// follow-up.
/// </para>
/// <para>
/// <b>Routing dispatch.</b> The router GEMV runs on GPU and the resulting
/// top-k indices are downloaded to host (small: <c>seqLen × topK × 4 bytes</c>
/// — for decode that's typically &lt; 32 bytes). Host then iterates active
/// experts and issues the per-expert gather/GEMM/SwiGLU/GEMM/axpy launches.
/// This is the same bucketing strategy as the CPU oracle and is ~bit-identical
/// modulo cuBLAS reduction order.
/// </para>
/// <para>
/// <b>Caller scratch contract.</b> The forward needs scratch buffers for the
/// router logits, top-k indices/weights, per-expert bucket scratch, gathered
/// inputs, gate/up/silu/down per-expert batches, plus shared-expert scratch.
/// The caller passes these via <see cref="CudaMoeScratch"/> — sized once for
/// the maximum (<c>seqLen</c>, <c>numExperts</c>, <c>topK</c>) the model
/// handles, reused across layers and across calls.
/// </para>
/// <para>
/// <b>Output overwrite.</b> The output is fully overwritten — the kernel
/// zero-clears it before accumulating expert contributions. This matches
/// the CPU oracle's contract (the per-token <c>acc.Clear()</c> + final copy
/// to output).
/// </para>
/// </remarks>
public static unsafe class CudaMoeFfn
{
    /// <summary>
    /// Runtime gate for the batched I2_S GEMM kernel (issue #250) in <see cref="ForwardBitNetI2S"/>'s
    /// per-expert gate/up/down projections. Mirrors <c>MatMul.I2S.I2SGemmTile</c>'s
    /// <c>DOTLLM_I2S_TILE</c> convention on the CPU side: defaults to on (when the kernel is loaded —
    /// see <see cref="CudaKernels.HasI2SBatchedGemm"/>), overridable via
    /// <c>DOTLLM_I2S_MOE_BATCHED_GEMM=0</c> to force the original per-row-GEMV-call loop for A/B
    /// comparison (this is exactly how <c>CudaMoeFfnBitNetI2SBatchedGemmTests</c> exercises both
    /// paths against the CPU oracle and each other within one test process).
    /// </summary>
    private static readonly bool UseI2SBatchedGemm = ResolveUseI2SBatchedGemm();

    private static bool ResolveUseI2SBatchedGemm()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_I2S_MOE_BATCHED_GEMM");
        return env is not ("0" or "false" or "off");
    }

    /// <summary>
    /// Runs one MoE SwiGLU FFN layer's forward pass on the GPU.
    /// </summary>
    /// <param name="hiddenF32">Device pointer to F32 input <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="outputF32">Device pointer to F32 output <c>[seqLen, hiddenSize]</c>. Fully overwritten.</param>
    /// <param name="seqLen">Number of tokens this call processes (decode = 1).</param>
    /// <param name="weights">Per-layer MoE weights bundle.</param>
    /// <param name="scratch">Caller-owned scratch (see <see cref="CudaMoeScratch"/>).</param>
    /// <param name="cublasHandle">cuBLAS handle for F32 GEMM/GEMV.</param>
    /// <param name="kernels">Loaded PTX kernel module.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <param name="forceUseI2SBatchedGemm">
    /// Test-only override for the <see cref="MoePrecision.BitNetI2S"/> path's batched-GEMM-vs-
    /// per-row-loop choice (issue #250). <c>null</c> (the production default) defers to
    /// <see cref="UseI2SBatchedGemm"/> (env-var-gated, resolved once per process). Non-null forces
    /// one path within a single process regardless of the env var — <c>UseI2SBatchedGemm</c> is a
    /// <c>static readonly</c> resolved once at type-load, so it cannot be toggled per-call at
    /// runtime; this parameter exists so a single test can three-way-compare the CPU oracle against
    /// BOTH GPU paths without spawning separate processes. Ignored for non-BitNetI2S precisions.
    /// </param>
    public static void Forward(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream,
        bool? forceUseI2SBatchedGemm = null)
        => ForwardCore(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, forceUseI2SBatchedGemm, false, null, null);

    /// <summary>
    /// <see cref="Forward(nint, nint, int, CudaMoeLayerWeights, CudaMoeScratch, nint, CudaKernels, nint, bool?)"/> for a model whose MoE layers may keep their routed experts on the CPU (<c>--cpu-moe</c>,
    /// #1187): a layer with <see cref="CudaMoeLayerWeights.HostExperts"/> runs them through <paramref name="hostExperts"/>.
    /// </summary>
    internal static void ForwardWithHostExperts(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream, CudaMoeHostExperts? hostExperts,
        DotLLM.Models.Architectures.MoeLayerWeights? nextHostLayer = null)
        => ForwardCore(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, null, false, null, hostExperts,
            nextHostLayer);

    /// <summary>Runs the expert body after architecture-specific routing populated scratch TopkIdx/TopkWeight.</summary>
    internal static void ForwardWithPrecomputedRouting(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream,
        Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection = null)
        => ForwardCore(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, null, true, packedQ4Projection, null);

    private static void ForwardCore(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream,
        bool? forceUseI2SBatchedGemm, bool routingAlreadyComputed,
        Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection, CudaMoeHostExperts? hostExperts,
        DotLLM.Models.Architectures.MoeLayerWeights? nextHostLayer = null)
    {
        if (!kernels.HasMoeKernels)
            throw new InvalidOperationException(
                "MoE kernels not available. Compile native/kernels/moe_ffn.cu to PTX.");

        if (seqLen <= 0) return;

        // BitNet-ternary (I2_S) routed experts (issue #246): entirely different per-expert
        // body (relu²·GLU + per-expert FFN Sub-LN, ternary GEMV instead of cuBLAS SwiGLU
        // GEMM) — dispatched to a dedicated method rather than threading a third branch
        // through every step below (mirrors CudaGemma4Ffn's precedent of a sibling forward
        // method for a structurally-different MoE variant).
        if (weights.Precision == MoePrecision.BitNetI2S)
        {
            ForwardBitNetI2S(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream,
                forceUseI2SBatchedGemm);
            return;
        }

        int hidden = weights.HiddenSize;
        int E = weights.NumExperts;
        int K = weights.NumExpertsPerTok;
        int I = weights.MoeIntermediateSize;
        int totalAssign = seqLen * K;

        scratch.EnsureCapacity(seqLen, weights);

        // ── Step 1: clear output ──
        kernels.LaunchMoeZeroF32(outputF32, seqLen * hidden, stream);

        if (!routingAlreadyComputed)
        {
            // ── Step 2: routing GEMV — logits[seqLen, numExperts] = hidden @ router^T ──
            CudaGemm.LinearF32(
                cublasHandle, hiddenF32, weights.Router, scratch.Logits,
                seqLen, hidden, E, stream);

            // ── Step 2b: additive router bias (gpt-oss, issue #348) ──
            // MUST run before softmax/top-k — same ordering as ForwardBitNetI2S's identity-MoTE
            // router bias (bias shifts both the top-k argmax AND the softmax probabilities).
            // RouterBiasF32 (gpt-oss, #348) and GateBiasF32 (identity-MoTE/BitNet, #246) are both
            // additive router biases; Forward (unlike ForwardBitNetI2S) may see either depending on
            // which loader populated this layer, so check both.
            nint effectiveRouterBias = weights.RouterBiasF32 != 0 ? weights.RouterBiasF32 : weights.GateBiasF32;
            if (effectiveRouterBias != 0)
            {
                if (!kernels.HasMoeGateBiasAdd)
                    throw new InvalidOperationException(
                        "MoE layer has a router bias but moe_gate_bias_add_f32 is not available. " +
                        "Recompile native/kernels/moe_ffn.cu to PTX.");
                kernels.LaunchMoeGateBiasAddF32(scratch.Logits, effectiveRouterBias, seqLen, E, stream);
            }

            // ── Steps 3-4: per-token sigmoid (Flint, renorm folded in) or softmax + top-k, then optional renorm ──
            if (weights.SigmoidGating)
            {
                kernels.LaunchMoeSigmoidTopk(
                    scratch.Logits, scratch.TopkIdx, scratch.TopkWeight,
                    seqLen, E, K, weights.NormTopKProb, stream);
            }
            else
            {
                kernels.LaunchMoeSoftmaxTopk(
                    scratch.Logits, scratch.TopkIdx, scratch.TopkWeight,
                    seqLen, E, K, stream);
                if (weights.NormTopKProb)
                    kernels.LaunchMoeRenormTopk(scratch.TopkWeight, seqLen, K, stream);
            }

        }

        // ── Step 5: download top-k indices to host so we can dispatch
        // per-expert grouped batches. Per-token weights stay device-resident —
        // the per-expert axpy reads them from device memory by (token, slot)
        // index. ──
        // One round trip: the ids (and, for a CPU-expert layer, the CPU leg's inputs) are queued as async copies into pinned
        // memory and the stream is synchronised once (#1202; a sync copy each cost a link round trip on an eGPU).
        if (weights.HostExperts is { } hostInputs && hostExperts is not null)
            hostExperts.EnqueueDownload(hostInputs, hiddenF32, scratch.TopkWeight, seqLen, stream);
        CudaDriverApi.cuMemcpyDtoHAsync_v2(scratch.HostTopkIdx, scratch.TopkIdx,
            (nuint)((long)totalAssign * sizeof(int)), stream).ThrowOnError();
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        var topkIdxHost = new ReadOnlySpan<int>((void*)scratch.HostTopkIdx, totalAssign);

        // CPU-expert layer (#1187): queue the shared expert on the device first so it overlaps the CPU leg, then add the
        // routed sum. shared + routed is one addition per element, so the order matches the resident path's routed + shared.
        if (weights.HostExperts is { } hostMoe)
        {
            if (hostExperts is null)
                throw new InvalidOperationException("This MoE layer keeps its routed experts on the CPU, but no CPU leg was supplied.");
            if (TryRunHostLayerSplit(hiddenF32, outputF32, seqLen, weights, hostMoe, nextHostLayer, scratch, cublasHandle,
                    kernels, stream, hostExperts, topkIdxHost, packedQ4Projection))
                return;
            RunSharedExperts(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, packedQ4Projection);
            hostExperts.Run(hostMoe, topkIdxHost, seqLen, scratch.DownBatch, stream);
            kernels.LaunchMoeAxpyUnweightedF32(outputF32, scratch.DownBatch, seqLen, hidden, stream);
            return;
        }

        if (CanRunBucketed(seqLen, weights, kernels))
        {
            int[] rowOffsets = PrepareBuckets(hiddenF32, seqLen, weights, scratch, kernels, stream, topkIdxHost);
            RunBucketedExperts(0, E, weights.GateProj, weights.UpProj, weights.DownProj, rowOffsets,
                weights, scratch, cublasHandle, kernels, stream, packedQ4Projection);
            kernels.LaunchMoeCombineBucketedF32(outputF32, scratch.DownBatch, scratch.TopkWeight,
                scratch.TokenIndices + (nint)((long)totalAssign * sizeof(int)), seqLen, hidden, K, stream);
            RunSharedExperts(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, packedQ4Projection);
            return;
        }

        // ── Step 6: bucket assignments per expert (host-side) ──
        // counts[e] = number of (token, slot) assignments routed to expert e.
        Span<int> counts = stackalloc int[E];
        counts.Clear();
        for (int i = 0; i < totalAssign; i++)
        {
            int e = topkIdxHost[i];
            if ((uint)e < (uint)E) counts[e]++;
        }

        // Per-expert offset cursor + per-assignment (tokenIdx, slot) lookup.
        // For each expert e with batch B>0 we need:
        //   bucketTokens[start..start+B) = list of token ids in routing order
        //   bucketSlots[start..start+B)  = matching slot ids
        // The slot is needed only for the final axpy weight lookup —
        // weight[tok * K + slot]. We pack both into a single cursor walk.
        int activeExperts = 0;
        for (int e = 0; e < E; e++) if (counts[e] > 0) activeExperts++;

        Span<int> offsets = stackalloc int[E + 1];
        int running = 0;
        for (int e = 0; e < E; e++)
        {
            offsets[e] = running;
            running += counts[e];
        }
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

        // ── Step 7: per-expert grouped path ──
        // For each active expert e:
        //   1. Upload bucketTokens slice to device (via scratch.TokenIndices).
        //   2. Gather the token rows from hidden into scratch.GatheredInput
        //      [batch, hidden].
        //   3. GEMM gate[batch, I] = gathered[batch, hidden] × W1_e[I, hidden]^T
        //      GEMM up  [batch, I] = gathered[batch, hidden] × W3_e[I, hidden]^T
        //   4. SwiGLU element-wise on (gate, up) → silu[batch, I].
        //   5. GEMM down[batch, hidden] = silu[batch, I] × W2_e[hidden, I]^T
        //   6. For each row b in [0, batch), the per-(token, slot) axpy:
        //      output[bucketTokens[b], :] += topkWeight[bucketTokens[b], bucketSlots[b]] * down[b, :]
        // We'd ideally fuse axpy across all batch rows by walking each per
        // (token, slot) pair, but slots vary across the bucket. So we group
        // batches by slot — issuing one axpy launch per (expert, slot) pair.

        // ── Phase B fast path: grouped quantized GEMV across K_active experts ──
        // For decode (seqLen=1) every active expert has batch=1 and reads the same
        // single hidden row. We can collapse all K_active gate / up projections into
        // ONE kernel launch each (instead of per-expert) by passing the K_active
        // weight pointers as device-resident arrays.
        //
        // Eligibility: Quantized precision + seqLen=1 + hidden % 256 == 0 + a
        // grouped-GEMV kernel exists for the gate/up dtype (Q4_K only at v1).
        //
        // When eligible we precompute gate/up F32 outputs for all active experts
        // into _gateBatch / _upBatch (each laid out as [K_active, I]). The per-
        // expert loop below then skips the gate/up ProjectF32OrQuant calls and
        // just slices into the already-populated buffers.
        bool useGrouped = weights.Precision == MoePrecision.Quantized
            && seqLen == 1
            && (hidden % CudaKernels.MoeGroupedGemvKAlignment(weights.GateProjQuantType)) == 0
            && kernels.HasMoeGroupedGemv(weights.GateProjQuantType)
            && kernels.HasMoeGroupedGemv(weights.UpProjQuantType)
            && weights.GateProjQuantType == weights.UpProjQuantType
            && weights.GateExpsBiasF32 == 0 && weights.UpExpsBiasF32 == 0
            && activeExperts > 0;

        // Map global expert id e → local index in the [0, K_active) compacted array.
        // For inactive experts the map entry stays -1. Stack-allocated since E is
        // bounded by numExperts (≤ 256 for any model we ship today).
        Span<int> expertLocalIdx = stackalloc int[E];
        for (int i = 0; i < E; i++) expertLocalIdx[i] = -1;
        int kActive = 0;
        for (int e = 0; e < E; e++)
        {
            if (counts[e] == 0) continue;
            expertLocalIdx[e] = kActive++;
        }

        // Fully grouped Q4_1 decode body (#1172): one SwiGLU, one grouped down GEMV and one weighted combine over all
        // K_active experts instead of the per-expert upload / SwiGLU / down / axpy loop (~4 launches + a sync copy per
        // expert). Same math; the combine runs in the loop's expert order.
        bool groupedDown = useGrouped && weights.DownProjQuantType == QuantizationType.Q4_1
            && kernels.HasQ4_1Gemm && kernels.HasMoeGroupedDownQ4_1 && (I & 31) == 0
            && weights.DownExpsBiasF32 == 0 && !weights.UseSwiGluOai && packedQ4Projection is null;

        if (useGrouped && kActive > 0)
        {
            DispatchGroupedGateUp(
                hiddenF32, weights, scratch, kernels, stream,
                expertLocalIdx: expertLocalIdx, kActive: kActive,
                hidden: hidden, I: I,
                downSlots: groupedDown ? bucketSlots.AsSpan(0, totalAssign) : default,
                downOffsets: groupedDown ? offsets : default);
        }

        if (groupedDown && kActive > 0)
        {
            long perArrayBytes = (long)kActive * sizeof(long);
            kernels.LaunchSwiGLUF32(scratch.GateBatch, scratch.UpBatch, scratch.SiluBatch, I, kActive, stream);
            kernels.LaunchMoeGroupedGemvQ4_1F32PerExpertInput(
                scratch.SiluBatch, scratch.GroupedPtrArrays + (nint)(4 * perArrayBytes), scratch.DownBatch,
                hidden, I, kActive, stream);
            kernels.LaunchMoeWeightedRowsSum(
                outputF32, scratch.DownBatch, scratch.TopkWeight, scratch.GroupedPtrArrays + (nint)(5 * perArrayBytes),
                kActive, hidden, stream);
        }

        for (int e = 0; e < E && !groupedDown; e++)
        {
            int batch = counts[e];
            if (batch == 0) continue;
            int start = offsets[e];

            // For the Phase-B grouped path, expert e's gate/up F32 outputs
            // already live at offset [e_local * I, (e_local+1) * I) in
            // _gateBatch / _upBatch. The remaining work (swiglu / down / axpy)
            // is identical to the per-expert path; we just rebase the swiglu
            // input pointers and skip the per-expert gate/up projection calls.
            int eLocal = useGrouped ? expertLocalIdx[e] : 0;
            long gateOff = useGrouped ? (long)eLocal * I * sizeof(float) : 0;
            long upOff = useGrouped ? (long)eLocal * I * sizeof(float) : 0;

            if (!useGrouped)
            {
                // 1. Upload bucketTokens slice (only this expert's section).
                // Sync HtoD: source is pageable managed memory, and the next
                // loop iteration may overwrite or invalidate the bucketTokens
                // pointer through array bounds reuse. cuMemcpyHtoD_v2 returns
                // only after the copy is staged, so it's safe to keep the
                // source pointer scoped to the `fixed` block.
                fixed (int* tp = bucketTokens)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(
                        scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                        (nint)(tp + start),
                        (nuint)((long)batch * sizeof(int))).ThrowOnError();
                }

                // 2. Gather hidden rows.
                kernels.LaunchMoeGatherTokenRowsF32(
                    hiddenF32, scratch.GatheredInput,
                    scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                    batch, hidden, stream);

                // 3. GEMMs gate / up.
                //    LinearF32 contract: Y[m, n] = X[m, k] × W[n, k]^T.
                //    gate[batch, I] = gathered[batch, hidden] × W1[I, hidden]^T
                //    Quantized path: dequant the per-expert weight raw bytes →
                //    F16 scratch → F32 scratch → LinearF32. F32 path: direct.
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.GatheredInput, batch, K: hidden, M: I,
                    weightF32: weights.GateProj[e],
                    weightQuant: weights.GateProj[e], weightQt: weights.GateProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                    gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.GateBatch, packedQ4Projection: packedQ4Projection);
                if (weights.GateExpsBiasF32 != 0)
                    kernels.LaunchMoeGateBiasAddF32(
                        scratch.GateBatch, weights.GateExpsBiasF32 + (nint)((long)e * I * sizeof(float)),
                        seqLen: batch, numExperts: I, stream);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.GatheredInput, batch, K: hidden, M: I,
                    weightF32: weights.UpProj[e],
                    weightQuant: weights.UpProj[e], weightQt: weights.UpProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                    gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.UpBatch, packedQ4Projection: packedQ4Projection);
                if (weights.UpExpsBiasF32 != 0)
                    kernels.LaunchMoeGateBiasAddF32(
                        scratch.UpBatch, weights.UpExpsBiasF32 + (nint)((long)e * I * sizeof(float)),
                        seqLen: batch, numExperts: I, stream);
            }
            else
            {
                // Grouped path: still need TokenIndices populated for the axpy.
                // For seqLen=1, every entry of bucketTokens is 0 (the only token).
                // We upload once via SingleTokenScratch so the axpy launcher (which
                // reads tokenIdx[0]) sees a valid slot.
                fixed (int* tp = bucketTokens)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(
                        scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                        (nint)(tp + start),
                        (nuint)((long)batch * sizeof(int))).ThrowOnError();
                }
            }

            // 4. SwiGLU element-wise.
            //    Sources rebase into the K_active-laid-out gate/up buffers when
            //    grouped is active; otherwise they read the per-expert buffer at offset 0.
            if (weights.UseSwiGluOai)
            {
                if (!kernels.HasSwiGluOai)
                    throw new InvalidOperationException(
                        "MoE layer requires the gpt-oss clamped SwiGLU activation but " +
                        "swiglu_oai_f32 is not available. Recompile native/kernels/moe_ffn.cu to PTX.");
                kernels.LaunchSwiGLUOaiF32(
                    scratch.GateBatch + (nint)gateOff,
                    scratch.UpBatch + (nint)upOff,
                    scratch.SiluBatch,
                    I, batch,
                    DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiAlpha,
                    DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiLimit,
                    stream);
            }
            else
            {
                kernels.LaunchSwiGLUF32(
                    scratch.GateBatch + (nint)gateOff,
                    scratch.UpBatch + (nint)upOff,
                    scratch.SiluBatch,
                    I, batch, stream);
            }

            // 5. GEMM down.
            //    down[batch, hidden] = silu[batch, I] × W2[hidden, I]^T
            //    K=intermediate may not be 256-aligned (V2-Lite intermediate=1408)
            //    so down keeps using the dequant fallback. No grouped variant yet.
            //    The packed Q4_0 x Q8_1 projection (LFM2) stores each block sum d * sum(q) in F16, which overflows once a
            //    row's max|x| passes ~2047 (Flint's routed SwiGLU output reaches ~1.3e4). Mirror the CPU guard (#1158,
            //    llama.cpp build_amax_src1_f16): rows above 1024 are divided by a power of two and the output multiplied
            //    back, which is exact. The other quantized paths keep integer sums or F32 activations and need no guard.
            bool downRescale = packedQ4Projection is not null && weights.Precision == MoePrecision.Quantized
                && weights.DownProjQuantType == QuantizationType.Q4_0;
            if (downRescale)
                kernels.LaunchMoePow2RescaleRows(scratch.SiluBatch, scratch.DownRowScale, batch, I, stream);
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                scratch.SiluBatch, batch, K: I, M: hidden,
                weightF32: weights.DownProj[e],
                weightQuant: weights.DownProj[e], weightQt: weights.DownProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: scratch.DownBatch, packedQ4Projection: packedQ4Projection);
            if (downRescale)
                kernels.LaunchMoeScaleRows(scratch.DownBatch, scratch.DownRowScale, batch, hidden, stream);
            if (weights.DownExpsBiasF32 != 0)
                kernels.LaunchMoeGateBiasAddF32(
                    scratch.DownBatch, weights.DownExpsBiasF32 + (nint)((long)e * hidden * sizeof(float)),
                    seqLen: batch, numExperts: hidden, stream);

            // 6. Per-slot axpy (group by slot to amortise weight lookups). See
            //    DispatchExpertAxpySlots for the slot-grouping strategy.
            DispatchExpertAxpySlots(
                outputF32, scratch.DownBatch, scratch.TopkWeight,
                scratch.TokenIndices, scratch.SingleTokenScratch,
                bucketTokens, bucketSlots, start, batch, hidden, K,
                kernels, stream);
        }

        // ── Step 8: shared-expert path (DeepSeek / Qwen1.5-MoE) ──
        RunSharedExperts(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, packedQ4Projection);
    }

    /// <summary>
    /// Multi-token call of a CPU-expert layer with a GPU leg (#1202): experts <c>[0, split)</c> are streamed to the device
    /// and run there in waves while the CPU leg computes <c>[split, E)</c>; then the CPU part is added and the next CPU
    /// layer's first waves are prefetched. False when the layer should stay entirely on the CPU leg.
    /// </summary>
    private static bool TryRunHostLayerSplit(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        DotLLM.Models.Architectures.MoeLayerWeights hostMoe, DotLLM.Models.Architectures.MoeLayerWeights? nextHostLayer,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream, CudaMoeHostExperts hostExperts,
        ReadOnlySpan<int> topkIdxHost, Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection)
    {
        if (hostExperts.Stream is not { } gpu || !CanRunBucketed(seqLen, weights, kernels) || !kernels.HasMoeAddF16F32
            || weights.Precision != MoePrecision.Quantized || !gpu.EnsureRing())
            return false;

        int E = weights.NumExperts, K = weights.NumExpertsPerTok, total = seqLen * K, hidden = weights.HiddenSize;
        Span<int> all = stackalloc int[E + 1];
        all.Clear();
        foreach (int e in topkIdxHost[..total]) all[e + 1]++;
        for (int e = 0; e < E; e++) all[e + 1] += all[e];
        int split = gpu.PlanSplit(all, gpu.PrefetchedEnd(hostMoe));
        if (split == 0)
            return false;

        // The CPU leg's inputs came down with the top-k ids.
        bool cpuPart = all[E] > all[split];

        int[] offsets = PrepareBuckets(hiddenF32, seqLen, weights, scratch, kernels, stream, topkIdxHost, split);
        for (int begin = 0; begin < split; begin += gpu.WaveExperts)
        {
            int end = Math.Min(begin + gpu.WaveExperts, split);
            int slot = gpu.BeginWave(hostMoe, begin, end, offsets, stream);
            RunBucketedExperts(begin, end, gpu.Gate, gpu.Up, gpu.Down, offsets, weights, scratch, cublasHandle, kernels,
                stream, packedQ4Projection);
            gpu.EndWave(slot, stream);
        }
        kernels.LaunchMoeCombineBucketedF32(outputF32, scratch.DownBatch, scratch.TopkWeight,
            scratch.TokenIndices + (nint)((long)total * sizeof(int)), seqLen, hidden, K, stream);
        RunSharedExperts(hiddenF32, outputF32, seqLen, weights, scratch, cublasHandle, kernels, stream, packedQ4Projection);

        if (cpuPart)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var rows = hostExperts.Compute(hostMoe, topkIdxHost, seqLen, split, out int assignments);
            gpu.ObserveCpu(System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds, assignments);
            gpu.AddCpuPart(rows, outputF32, kernels, stream);
        }

        // Keep the link busy through the next layer's attention: its first waves go now (at most the ring, at most
        // this layer's GPU share).
        if (nextHostLayer is not null)
            gpu.Prefetch(nextHostLayer, split);
        return true;
    }

    /// <summary>Test hook: false runs multi-token calls through the per-expert loop (the bucketed body's reference).</summary>
    internal static bool BucketedEnabled { get; set; } = true;

    /// <summary>
    /// True when a multi-token call can run the bucketed expert body (<see cref="PrepareBuckets"/> +
    /// <see cref="RunBucketedExperts"/> + one combine) instead of the per-expert loop with its per-expert host copies.
    /// </summary>
    private static bool CanRunBucketed(int seqLen, CudaMoeLayerWeights weights, CudaKernels kernels)
        => seqLen > 1 && BucketedEnabled && kernels.HasMoeCombineBucketed && weights.NumExpertsPerTok <= CudaKernels.MoeCombineMaxTopK
            && weights.Precision is MoePrecision.Quantized or MoePrecision.F32;

    /// <summary>
    /// Buckets the call's (token, slot) assignments by expert (ascending id, tokens in order), uploads the row-to-token
    /// map and the assignment-to-row map in one copy, and gathers every assignment's input row into
    /// <see cref="CudaMoeScratch.GatheredInput"/>. The stream must be idle on the scratch (the caller synchronised it
    /// to read the top-k ids). Only experts below <c>expertLimit</c> get rows (the rest run on the CPU leg).
    /// </summary>
    /// <returns>Row offsets per expert, <c>[numExperts + 1]</c>: expert <c>e</c> owns rows <c>[offsets[e], offsets[e + 1])</c>.</returns>
    private static unsafe int[] PrepareBuckets(nint hiddenF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, CudaKernels kernels, nint stream, ReadOnlySpan<int> topkIdx, int expertLimit = int.MaxValue)
    {
        int E = weights.NumExperts, K = weights.NumExpertsPerTok, total = seqLen * K;
        int limit = Math.Min(E, expertLimit);
        var offsets = new int[E + 1];
        foreach (int e in topkIdx[..total])
            if ((uint)e < (uint)limit) offsets[e + 1]++;
        for (int e = 0; e < E; e++) offsets[e + 1] += offsets[e];

        // [rowToken (total) | assignmentRow (total)] in one upload; rows past offsets[E] stay unused.
        int[] map = System.Buffers.ArrayPool<int>.Shared.Rent(2 * total);
        try
        {
            Span<int> cursor = stackalloc int[E];
            offsets.AsSpan(0, E).CopyTo(cursor);
            for (int i = 0; i < total; i++)
            {
                int e = topkIdx[i];
                if ((uint)e >= (uint)limit) { map[total + i] = -1; continue; }
                int row = cursor[e]++;
                map[row] = i / K;
                map[total + i] = row;
            }
            fixed (int* mp = map)
                CudaDriverApi.cuMemcpyHtoD_v2(scratch.TokenIndices, (nint)mp, (nuint)(2L * total * sizeof(int))).ThrowOnError();
        }
        finally
        {
            System.Buffers.ArrayPool<int>.Shared.Return(map);
        }

        kernels.LaunchMoeGatherTokenRowsF32(hiddenF32, scratch.GatheredInput, scratch.TokenIndices, offsets[E],
            weights.HiddenSize, stream);
        return offsets;
    }

    /// <summary>
    /// Runs experts <c>[eBegin, eEnd)</c> of the bucketed body: gate / up per expert on its rows of
    /// <see cref="CudaMoeScratch.GatheredInput"/>, one SwiGLU over the range's rows, down per expert into
    /// <see cref="CudaMoeScratch.DownBatch"/>. The per-expert math is the per-expert loop's, so the rows are identical.
    /// </summary>
    /// <param name="eBegin">First expert id.</param>
    /// <param name="eEnd">One past the last expert id.</param>
    /// <param name="gate">Gate weights indexed by expert id (device pointers; only the range is read).</param>
    /// <param name="up">Up weights indexed by expert id.</param>
    /// <param name="down">Down weights indexed by expert id.</param>
    /// <param name="offsets">Row offsets from <see cref="PrepareBuckets"/>.</param>
    /// <param name="weights">Layer weights (dims, quant types, biases).</param>
    /// <param name="scratch">Scratch.</param>
    /// <param name="cublasHandle">cuBLAS handle.</param>
    /// <param name="kernels">Kernels.</param>
    /// <param name="stream">Stream.</param>
    /// <param name="packedQ4Projection">Packed Q4_0 projection (LFM2), or null.</param>
    private static void RunBucketedExperts(int eBegin, int eEnd, ReadOnlySpan<nint> gate, ReadOnlySpan<nint> up,
        ReadOnlySpan<nint> down, int[] offsets, CudaMoeLayerWeights weights, CudaMoeScratch scratch, nint cublasHandle,
        CudaKernels kernels, nint stream, Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection)
    {
        int hidden = weights.HiddenSize, I = weights.MoeIntermediateSize;
        int rowBegin = offsets[eBegin], rows = offsets[eEnd] - rowBegin;
        if (rows == 0) return;
        long iBytes = (long)I * sizeof(float), hBytes = (long)hidden * sizeof(float);

        for (int e = eBegin; e < eEnd; e++)
        {
            int start = offsets[e], batch = offsets[e + 1] - start;
            if (batch == 0) continue;
            nint x = scratch.GatheredInput + (nint)(start * hBytes);
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream, x, batch, K: hidden, M: I,
                weightF32: gate[e], weightQuant: gate[e], weightQt: weights.GateProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: scratch.GateBatch + (nint)(start * iBytes), packedQ4Projection: packedQ4Projection);
            if (weights.GateExpsBiasF32 != 0)
                kernels.LaunchMoeGateBiasAddF32(scratch.GateBatch + (nint)(start * iBytes),
                    weights.GateExpsBiasF32 + (nint)(e * iBytes), seqLen: batch, numExperts: I, stream);
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream, x, batch, K: hidden, M: I,
                weightF32: up[e], weightQuant: up[e], weightQt: weights.UpProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: scratch.UpBatch + (nint)(start * iBytes), packedQ4Projection: packedQ4Projection);
            if (weights.UpExpsBiasF32 != 0)
                kernels.LaunchMoeGateBiasAddF32(scratch.UpBatch + (nint)(start * iBytes),
                    weights.UpExpsBiasF32 + (nint)(e * iBytes), seqLen: batch, numExperts: I, stream);
        }

        nint g = scratch.GateBatch + (nint)(rowBegin * iBytes), u = scratch.UpBatch + (nint)(rowBegin * iBytes);
        nint silu = scratch.SiluBatch + (nint)(rowBegin * iBytes);
        if (weights.UseSwiGluOai)
        {
            if (!kernels.HasSwiGluOai)
                throw new InvalidOperationException(
                    "MoE layer requires the gpt-oss clamped SwiGLU activation but swiglu_oai_f32 is not available. " +
                    "Recompile native/kernels/moe_ffn.cu to PTX.");
            kernels.LaunchSwiGLUOaiF32(g, u, silu, I, rows,
                DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiAlpha, DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp.SwiGluOaiLimit, stream);
        }
        else
        {
            kernels.LaunchSwiGLUF32(g, u, silu, I, rows, stream);
        }

        // Same F16 block-sum guard as the per-expert loop (packed Q4_0 x Q8_1 down projection); it is per row, so one
        // launch covers the range.
        bool downRescale = packedQ4Projection is not null && weights.Precision == MoePrecision.Quantized
            && weights.DownProjQuantType == QuantizationType.Q4_0;
        nint rowScale = scratch.DownRowScale + (nint)((long)rowBegin * sizeof(float));
        if (downRescale)
            kernels.LaunchMoePow2RescaleRows(silu, rowScale, rows, I, stream);
        for (int e = eBegin; e < eEnd; e++)
        {
            int start = offsets[e], batch = offsets[e + 1] - start;
            if (batch == 0) continue;
            nint y = scratch.DownBatch + (nint)(start * hBytes);
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                scratch.SiluBatch + (nint)(start * iBytes), batch, K: I, M: hidden,
                weightF32: down[e], weightQuant: down[e], weightQt: weights.DownProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: y, packedQ4Projection: packedQ4Projection);
            if (weights.DownExpsBiasF32 != 0)
                kernels.LaunchMoeGateBiasAddF32(y, weights.DownExpsBiasF32 + (nint)(e * hBytes),
                    seqLen: batch, numExperts: hidden, stream);
        }
        if (downRescale)
            kernels.LaunchMoeScaleRows(scratch.DownBatch + (nint)(rowBegin * hBytes), rowScale, rows, hidden, stream);
    }

    /// <summary>Adds the layer's shared expert(s) (DeepSeek / Qwen1.5-MoE / Flint) to <paramref name="outputF32"/>.</summary>
    private static void RunSharedExperts(
        nint hiddenF32, nint outputF32, int seqLen, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream,
        Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection)
    {
        int hidden = weights.HiddenSize;
        if (weights.NumSharedExperts > 0 && weights.SharedIntermediateSize > 0)
        {
            int sI = weights.SharedIntermediateSize;
            // Compute sigmoid scale once per token if Qwen1.5-MoE shared_expert_gate.
            bool hasGate = weights.SharedExpertGate != 0;
            if (hasGate)
            {
                kernels.LaunchMoeSigmoidLogitF32(
                    hiddenF32, weights.SharedExpertGate, scratch.SharedScale,
                    seqLen, hidden, stream);
            }

            for (int s = 0; s < weights.NumSharedExperts; s++)
            {
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    hiddenF32, seqLen, K: hidden, M: sI,
                    weightF32: weights.SharedGateProj[s],
                    weightQuant: weights.SharedGateProj[s], weightQt: weights.SharedGateProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.SharedGateBatch, packedQ4Projection: packedQ4Projection);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    hiddenF32, seqLen, K: hidden, M: sI,
                    weightF32: weights.SharedUpProj[s],
                    weightQuant: weights.SharedUpProj[s], weightQt: weights.SharedUpProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.SharedUpBatch, packedQ4Projection: packedQ4Projection);
                kernels.LaunchSwiGLUF32(
                    scratch.SharedGateBatch, scratch.SharedUpBatch, scratch.SharedSiluBatch,
                    sI, seqLen, stream);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.SharedSiluBatch, seqLen, K: sI, M: hidden,
                    weightF32: weights.SharedDownProj[s],
                    weightQuant: weights.SharedDownProj[s], weightQt: weights.SharedDownProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.SharedDownBatch, packedQ4Projection: packedQ4Projection);

                if (hasGate)
                {
                    kernels.LaunchMoeAxpyScaledPerTokenF32(
                        outputF32, scratch.SharedDownBatch, scratch.SharedScale,
                        seqLen, hidden, stream);
                }
                else
                {
                    kernels.LaunchMoeAxpyUnweightedF32(
                        outputF32, scratch.SharedDownBatch, seqLen, hidden, stream);
                }
            }
        }
    }

    /// <summary>
    /// Per-(expert, slot) weighted accumulation into <paramref name="outputF32"/>, extracted
    /// from the routed-expert loop so both the F32/Quantized <see cref="Forward"/> path and
    /// <see cref="ForwardBitNetI2S"/> (issue #246) share one implementation. Groups by slot
    /// (typical <c>K</c> is small — 2..8) to amortise weight lookups: when every assignment in
    /// this expert's bucket shares one slot (the common case for decode, or sparse routing),
    /// one launch does the whole batch; otherwise falls back to per-row axpys for the mixed-
    /// slot case (rare, a correctness safety net rather than the expected hot path).
    /// </summary>
    private static unsafe void DispatchExpertAxpySlots(
        nint outputF32, nint downBatch, nint topkWeight,
        nint tokenIndices, nint singleTokenScratch,
        int[] bucketTokens, int[] bucketSlots,
        int start, int batch, int hidden, int K,
        CudaKernels kernels, nint stream)
    {
        for (int slot = 0; slot < K; slot++)
        {
            // Count + collect bucket rows for this (expert, slot).
            int slotBatchCount = 0;
            for (int b = 0; b < batch; b++)
                if (bucketSlots[start + b] == slot) slotBatchCount++;
            if (slotBatchCount == 0) continue;

            if (slotBatchCount == batch)
            {
                // All assignments in this expert's bucket share the same
                // slot — issue the axpy directly, no second-level
                // bucketing needed. Common for decode (seqLen=1) and
                // for sparse routing patterns.
                kernels.LaunchMoeAxpyScaledRowF32(
                    outputF32, downBatch,
                    topkWeight,
                    tokenIndices + (nint)((long)start * sizeof(int)),
                    batch, hidden, K, slot, stream);
            }
            else
            {
                // Mixed slots in this expert's bucket. Build a per-slot
                // sub-bucket of (down_row_index, token_id) pairs:
                //   subTokens[k] = bucketTokens[start + b]   (the dst token)
                //   subDownRows[k] = b                       (the row in DownBatch)
                // Then we need to gather down[subDownRows] into a sub-batch
                // before the axpy. To keep Phase 1 simple, we issue per-
                // row axpys for the mixed-slot case (very rare for K≥2 if
                // routing has any token spread, but a safety net for
                // adversarial inputs).
                for (int b = 0; b < batch; b++)
                {
                    if (bucketSlots[start + b] != slot) continue;
                    int tokenId = bucketTokens[start + b];
                    // Upload single (tokenId) ⇒ TokenIndices scratch slot.
                    // Use the synchronous HtoD here so the source stack
                    // local can be safely overwritten on the next loop
                    // iteration. Async HtoD from non-pinned host memory
                    // would also serialise via the driver staging buffer
                    // but the sync variant makes the contract obvious.
                    CudaDriverApi.cuMemcpyHtoD_v2(
                        singleTokenScratch,
                        (nint)(&tokenId), sizeof(int)).ThrowOnError();
                    // Issue an axpy for the single down row at index b.
                    kernels.LaunchMoeAxpyScaledRowF32(
                        outputF32,
                        downBatch + (nint)((long)b * hidden * sizeof(float)),
                        topkWeight,
                        singleTokenScratch,
                        1, hidden, K, slot, stream);
                }
            }
        }
    }

    /// <summary>
    /// BitNet-ternary (I2_S) routed-expert MoE forward (issue #246) — the GPU port of
    /// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.ExecuteBitNetMoe</c>. Shares the router GEMV +
    /// softmax/top-k + host-side bucketing strategy with <see cref="Forward"/> (steps 1-6
    /// below mirror it exactly, plus an additive router-bias step BitNet/identity-MoTE
    /// needs and the F32 path doesn't), but the per-expert body differs in three numerically
    /// load-bearing ways — see <see cref="MoePrecision.BitNetI2S"/>:
    /// <list type="number">
    ///   <item>Experts are ternary I2_S. Decode (seqLen=1) dispatches through
    ///     <see cref="CudaKernels.LaunchI2_SGemvF32In"/> — a single-row GEMV, since batch is always
    ///     1 per touched expert. Prefill (seqLen&gt;1, multiple tokens routed to the same expert)
    ///     originally issued one GEMV LAUNCH PER ROW (issue #246 scope note) — that re-decoded the
    ///     whole weight matrix once per token. Issue #250 added
    ///     <see cref="CudaKernels.LaunchI2_SGemmF32In"/>, a real batched GEMM that decodes each
    ///     weight row ONCE and reuses it across every routed token (the same unpack-once-reuse
    ///     strategy as the CPU GEMM oracle's <c>GemmI2_SRows</c>/<c>GemmI2_SW2A8Rows</c>); gated on
    ///     <see cref="CudaKernels.HasI2SBatchedGemm"/> with the original per-row loop kept as a
    ///     fallback for stale PTX.</item>
    ///   <item>The gate non-linearity is relu² (<see cref="CudaKernels.LaunchReLU2GLUF32"/>),
    ///     not SiLU.</item>
    ///   <item>A per-expert BitNet FFN Sub-LN (<see cref="CudaKernels.LaunchRmsNormF32"/> with
    ///     <see cref="CudaMoeLayerWeights.ExpertFfnSubNormF32"/>) is applied to the gated
    ///     intermediate before <c>down_proj</c>. RMSNorm is a per-row-independent op, so calling
    ///     the existing kernel once per touched-expert batch (with that expert's Sub-LN weight
    ///     and <c>rows = batch</c>) needs no new kernel — verified in-place-safe by reading
    ///     <c>rmsnorm_f32.cu</c>: pass 1 fully reduces sum-of-squares over the (still
    ///     untouched) input before pass 2 reads-then-writes each element exactly once.</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// <b>Design deviation from the issue's suggested per-expert scale-attach mechanism.</b>
    /// The issue anticipated a <c>pq2_0_repack.cu</c>-style device kernel to reconcile
    /// <see cref="CudaKernels.LaunchI2_SGemvF32In"/>'s single-trailing-scale-per-tensor
    /// contract with the CPU bundle's separate per-expert scale array. This turned out to need
    /// no device kernel at all — see <see cref="CudaMoeWeightsLoader.LoadLayerBitNetI2S"/>'s
    /// remarks for why a load-time two-copy upload (payload + scale into one pre-sized device
    /// buffer) suffices; PQ2_0's repack kernel solves a different problem (byte-level
    /// re-interleaving for coalescing), not "attach a scalar to a slice".
    /// </remarks>
    private static unsafe void ForwardBitNetI2S(
        nint hiddenF32, nint outputF32,
        int seqLen,
        CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream,
        bool? forceUseI2SBatchedGemm = null)
    {
        if (!kernels.HasBitNetMoeKernels)
            throw new InvalidOperationException(
                "BitNet-MoE CUDA kernels not available (I2_S GEMV / relu²GLU / RMSNorm F32 / "
              + "shared MoE orchestration kernels). Recompile native/kernels/{i2_s_gemv,"
              + "swiglu_f32,rmsnorm_f32,moe_ffn}.cu to PTX.");
        if (weights.GateBiasF32 != 0 && !kernels.HasMoeGateBiasAdd)
            throw new InvalidOperationException(
                "MoE layer has a router bias but moe_gate_bias_add_f32 is not available. "
              + "Recompile native/kernels/moe_ffn.cu to PTX.");

        int hidden = weights.HiddenSize;
        int E = weights.NumExperts;
        int K = weights.NumExpertsPerTok;
        int I = weights.MoeIntermediateSize;
        int totalAssign = seqLen * K;

        // I2_S's aligned GEMV kernels require k % 128 == 0 (see i2_s_gemv.cu) — the same
        // invariant the CPU oracle documents ("hiddenSize/intermediateSize a multiple of
        // 128"). No ragged fallback wired here (explicitly out of scope for this pass — see
        // LaunchI2_SGemvF32InRagged for the CPU-parity twin if a non-128-aligned BitNet-MoE
        // checkpoint ever surfaces).
        if ((hidden % 128) != 0 || (I % 128) != 0)
            throw new InvalidOperationException(
                $"BitNet-MoE CUDA forward requires hiddenSize ({hidden}) and moeIntermediateSize "
              + $"({I}) to be multiples of 128 (I2_S block size). Ragged-K is not wired for the "
              + "MoE path yet — see CudaKernels.LaunchI2_SGemvF32InRagged for the dense twin.");

        scratch.EnsureCapacity(seqLen, weights);

        // ── Step 1: clear output ──
        kernels.LaunchMoeZeroF32(outputF32, seqLen * hidden, stream);

        // ── Step 2: routing GEMV — logits[seqLen, numExperts] = hidden @ router^T ──
        // Router stays F32 (small: numExperts × hidden floats) — same cuBLAS LinearF32 path
        // as the F32/Quantized MoE forward.
        CudaGemm.LinearF32(
            cublasHandle, hiddenF32, weights.Router, scratch.Logits,
            seqLen, hidden, E, stream);

        // ── Step 2b: additive router bias (identity-MoTE / Qwen3 aux-loss-free routing) ──
        // MUST run before softmax/top-k — mirrors MoeSwiGluMlp.Route's CPU ordering exactly
        // (bias shifts both the top-k argmax AND the softmax probabilities feeding assignWeight).
        if (weights.GateBiasF32 != 0)
            kernels.LaunchMoeGateBiasAddF32(scratch.Logits, weights.GateBiasF32, seqLen, E, stream);

        // ── Step 3: per-token softmax + top-k selection → device buffers ──
        kernels.LaunchMoeSoftmaxTopk(
            scratch.Logits, scratch.TopkIdx, scratch.TopkWeight,
            seqLen, E, K, stream);

        // ── Step 4: optional renorm of top-k weights ──
        if (weights.NormTopKProb)
            kernels.LaunchMoeRenormTopk(scratch.TopkWeight, seqLen, K, stream);

        // ── Step 5: download top-k indices to host for per-expert bucketing (same strategy
        // as Forward — see its Step 5/6 comments for the full rationale). ──
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        int[] topkIdxHost = new int[totalAssign];
        fixed (int* p = topkIdxHost)
            CudaDriverApi.cuMemcpyDtoH_v2(
                (nint)p, scratch.TopkIdx,
                (nuint)((long)totalAssign * sizeof(int))).ThrowOnError();

        Span<int> counts = stackalloc int[E];
        counts.Clear();
        for (int i = 0; i < totalAssign; i++)
        {
            int e = topkIdxHost[i];
            if ((uint)e < (uint)E) counts[e]++;
        }

        Span<int> offsets = stackalloc int[E + 1];
        int running = 0;
        for (int e = 0; e < E; e++)
        {
            offsets[e] = running;
            running += counts[e];
        }
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

        // ── Step 6: per-expert body — down( ffn_sub_norm( relu²(gate(x)) * up(x) ) ) ──
        for (int e = 0; e < E; e++)
        {
            int batch = counts[e];
            if (batch == 0) continue;
            int start = offsets[e];

            // 1. Upload this expert's bucketTokens slice + gather hidden rows (identical to
            //    the F32/Quantized path's steps 1-2).
            fixed (int* tp = bucketTokens)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(
                    scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                    (nint)(tp + start),
                    (nuint)((long)batch * sizeof(int))).ThrowOnError();
            }
            kernels.LaunchMoeGatherTokenRowsF32(
                hiddenF32, scratch.GatheredInput,
                scratch.TokenIndices + (nint)((long)start * sizeof(int)),
                batch, hidden, stream);

            // 2. gate / up projections: ternary I2_S. Batched GEMM (issue #250) decodes each
            //    weight row ONCE and reuses it across all `batch` tokens routed to this expert —
            //    the original per-row-GEMV-call loop (issue #246 scope note) re-decoded the whole
            //    weight matrix once PER TOKEN, which only mattered for prefill (decode is batch=1
            //    either way, and LaunchI2_SGemmF32In itself degrades batch=1 to the GEMV call).
            long hiddenRowBytes = (long)hidden * sizeof(float);
            long iRowBytes = (long)I * sizeof(float);
            bool useBatchedGemm = (forceUseI2SBatchedGemm ?? UseI2SBatchedGemm) && kernels.HasI2SBatchedGemm;
            if (useBatchedGemm)
            {
                kernels.LaunchI2_SGemmF32In(
                    weights.GateProj[e], scratch.GatheredInput, scratch.GateBatch,
                    n: I, k: hidden, numTokens: batch, stream);
                kernels.LaunchI2_SGemmF32In(
                    weights.UpProj[e], scratch.GatheredInput, scratch.UpBatch,
                    n: I, k: hidden, numTokens: batch, stream);
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    nint xRow = scratch.GatheredInput + (nint)((long)b * hiddenRowBytes);
                    kernels.LaunchI2_SGemvF32In(
                        weights.GateProj[e], xRow, scratch.GateBatch + (nint)((long)b * iRowBytes),
                        n: I, k: hidden, stream);
                    kernels.LaunchI2_SGemvF32In(
                        weights.UpProj[e], xRow, scratch.UpBatch + (nint)((long)b * iRowBytes),
                        n: I, k: hidden, stream);
                }
            }

            // 3. inter = ffn_sub_norm_e( relu²(gate) * up ). relu²·GLU is elementwise over the
            //    whole [batch, I] block in one launch (same shape as LaunchSwiGLUF32); the
            //    per-expert Sub-LN is RMSNorm — per-row-independent, so one call over
            //    rows=batch with this expert's Sub-LN weight covers the whole bucket, in-place.
            kernels.LaunchReLU2GLUF32(scratch.GateBatch, scratch.UpBatch, scratch.SiluBatch, I, batch, stream);
            kernels.LaunchRmsNormF32(
                scratch.SiluBatch, weights.ExpertFfnSubNormF32[e], scratch.SiluBatch,
                I, weights.RmsEps, batch, stream);

            // 4. down projection: ternary I2_S (same batched-GEMM-vs-per-row-loop choice as
            //    gate/up above).
            if (useBatchedGemm)
            {
                kernels.LaunchI2_SGemmF32In(
                    weights.DownProj[e], scratch.SiluBatch, scratch.DownBatch,
                    n: hidden, k: I, numTokens: batch, stream);
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    kernels.LaunchI2_SGemvF32In(
                        weights.DownProj[e],
                        scratch.SiluBatch + (nint)((long)b * iRowBytes),
                        scratch.DownBatch + (nint)((long)b * hiddenRowBytes),
                        n: hidden, k: I, stream);
                }
            }

            // 5. Per-slot weighted axpy into output — shared with the F32/Quantized path.
            DispatchExpertAxpySlots(
                outputF32, scratch.DownBatch, scratch.TopkWeight,
                scratch.TokenIndices, scratch.SingleTokenScratch,
                bucketTokens, bucketSlots, start, batch, hidden, K,
                kernels, stream);
        }
    }

    /// <summary>
    /// Projection helper that branches on <paramref name="precision"/>:
    /// <list type="bullet">
    /// <item>F32 weights → cuBLAS LinearF32 directly.</item>
    /// <item>Quantized + decode batch=1 + K%256==0 + quantized GEMV kernel
    ///     available for the dtype → convert F32 input to F16 (K elements),
    ///     <see cref="CudaKernels.LaunchQuantizedGemv"/> producing F16 output
    ///     (M elements), convert F16 output to F32. **3 launches** total per
    ///     projection, all small per-element kernels — eliminates the full
    ///     M*K dequant materialisation.</item>
    /// <item>Quantized fallback (prefill, K not aligned, or no GEMV kernel) →
    ///     <see cref="CudaKernels.LaunchDequantToF16"/> + F16→F32 convert +
    ///     LinearF32. Same 3 launches but each does M*K work.</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// V2-Lite Q4_K_M numbers: gate_proj/up_proj have K=hidden=2048 stored as Q4_K
    /// (256-aligned ✓), take the GEMV fast path. down_proj has K=intermediate=1408
    /// stored as Q8_0 (block_size=32, 32-aligned ✓) — also takes the GEMV fast
    /// path now that the gate uses per-quant-type minimum K alignment via
    /// <see cref="CudaKernels.MinKAlignmentFor(QuantizationType)"/>. All three
    /// MoE projections benefit; only the (rare) prefill batch&gt;1 case falls
    /// through to the dequant-then-GEMM fallback.
    /// </remarks>
    private static void ProjectF32OrQuant(
        MoePrecision precision, nint cublasHandle, CudaKernels kernels, nint stream,
        nint inputF32, int batch, int K, int M,
        nint weightF32, nint weightQuant, QuantizationType weightQt,
        nint dequantF16, nint dequantF32, nint outputF32,
        nint gemvInputF16, nint gemvOutputF16,
        Action<nint, nint, nint, int, int, int, nint>? packedQ4Projection = null)
    {
        if (precision == MoePrecision.Quantized && weightQt == QuantizationType.Q4_0 && packedQ4Projection is not null)
        { packedQ4Projection(weightQuant, inputF32, outputF32, M, K, batch, stream); return; }
        if (precision == MoePrecision.Quantized && weightQt == QuantizationType.Q4_1 && kernels.HasQ4_1Gemm
            && (K & 31) == 0)
        {
            // Q4_1 (Flint, #1172): decode GEMV and multi-token GEMM both read the blocks in place on F32 activations, so
            // there is no dequant scratch (rule 2) and no F16 round trip of a large routed SwiGLU output.
            kernels.LaunchQuantizedGemmQ4_1F32(weightQuant, inputF32, outputF32, M, K, batch, stream);
            return;
        }
        if (precision == MoePrecision.Quantized)
        {
            // Fast path: direct quantized GEMV when batch=1, K satisfies the
            // per-quant-type alignment (block-32 quants need K%32==0; K-quants
            // need K%256==0), and a kernel exists for this quant type. Avoids
            // materialising the full M*K dequant scratch on every projection —
            // typically an order of magnitude smaller memory footprint per launch.
            int minKAlign = CudaKernels.MinKAlignmentFor(weightQt);
            bool gemvEligible = batch == 1
                && (K % minKAlign) == 0
                && kernels.HasQuantizedGemvKernel(weightQt);
            if (gemvEligible)
            {
                kernels.LaunchConvertF32ToF16(inputF32, gemvInputF16, K, stream);
                kernels.LaunchQuantizedGemv(weightQuant, weightQt,
                    gemvInputF16, gemvOutputF16, M, K, stream);
                kernels.LaunchConvertF16ToF32(gemvOutputF16, outputF32, M, stream);
                return;
            }

            // Fallback: dequant entire weight to F16 → convert to F32 → LinearF32.
            kernels.LaunchDequantToF16(weightQuant, weightQt, dequantF16, M * K, stream);
            kernels.LaunchConvertF16ToF32(dequantF16, dequantF32, M * K, stream);
            CudaGemm.LinearF32(cublasHandle, inputF32, dequantF32, outputF32,
                batch, K, M, stream);
        }
        else
        {
            CudaGemm.LinearF32(cublasHandle, inputF32, weightF32, outputF32,
                batch, K, M, stream);
        }
    }

    /// <summary>
    /// Phase B: dispatch grouped quantized GEMV for the gate and up projections
    /// across all <paramref name="kActive"/> active experts in just two kernel
    /// launches (instead of <c>2 × kActive</c> per-expert launches). Caller
    /// guarantees decode batch=1 (single shared input row), <c>hidden % 256 == 0</c>,
    /// and <see cref="CudaKernels.HasMoeGroupedGemv(QuantizationType)"/> is true
    /// for both projection dtypes. After this returns, expert <c>e</c>'s gate
    /// and up F32 outputs occupy <c>scratch.GateBatch[e_local * I .. (e_local+1) * I)</c>
    /// and <c>scratch.UpBatch[e_local * I .. (e_local+1) * I)</c> where
    /// <c>e_local = expertLocalIdx[e]</c>.
    /// </summary>
    private static void DispatchGroupedGateUp(
        nint hiddenF32, CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, CudaKernels kernels, nint stream,
        Span<int> expertLocalIdx, int kActive, int hidden, int I,
        ReadOnlySpan<int> downSlots = default, ReadOnlySpan<int> downOffsets = default)
    {
        int E = weights.NumExperts;
        bool withDown = !downOffsets.IsEmpty;

        // Build host-side ptr arrays (4 × kActive nints, all packed into one
        // contiguous block matching the device-side layout).
        //   slice 0: gate weights[0..kActive)
        //   slice 1: gate outputs[0..kActive)
        //   slice 2: up weights  [0..kActive)
        //   slice 3: up outputs  [0..kActive)
        //   slice 4: down weights[0..kActive)  (withDown only: the fully grouped Q4_1 body)
        //   slice 5: routing slot of each expert (withDown only; decode has one token, so offsets[e] is its entry)
        int slices = withDown ? 6 : 4;
        long* hostPtrs = stackalloc long[slices * kActive];

        // Walk experts in increasing global id (matches the per-expert loop
        // ordering), skipping inactive ones. eLocal = position in the compacted
        // [0, kActive) array. Per-expert F16 output offsets land at e_local * I
        // halfs into the staging buffers.
        int eLocal = 0;
        long iBytesF16 = (long)I * sizeof(ushort);
        for (int e = 0; e < E; e++)
        {
            if (expertLocalIdx[e] < 0) continue;
            hostPtrs[0 * kActive + eLocal] = (long)weights.GateProj[e];
            hostPtrs[1 * kActive + eLocal] = (long)(scratch.GroupedGateF16 + (nint)((long)eLocal * iBytesF16));
            hostPtrs[2 * kActive + eLocal] = (long)weights.UpProj[e];
            hostPtrs[3 * kActive + eLocal] = (long)(scratch.GroupedUpF16 + (nint)((long)eLocal * iBytesF16));
            if (withDown)
            {
                hostPtrs[4 * kActive + eLocal] = (long)weights.DownProj[e];
                hostPtrs[5 * kActive + eLocal] = downSlots[downOffsets[e]];
            }
            eLocal++;
        }

        // One synchronous HtoD copy of all 4 ptr arrays at once. Source is
        // stack-local — must complete before this method returns. Sync HtoD
        // (cuMemcpyHtoD_v2) on pageable memory is immediate.
        long ptrBytes = (long)slices * kActive * sizeof(long);
        CudaDriverApi.cuMemcpyHtoD_v2(
            scratch.GroupedPtrArrays, (nint)hostPtrs, (nuint)ptrBytes).ThrowOnError();

        // Convert the shared input hidden vector F32→F16 once.
        kernels.LaunchConvertF32ToF16(hiddenF32, scratch.GemvInputF16, hidden, stream);

        // Slice out the device-side ptr arrays.
        long perArrayBytes = (long)kActive * sizeof(long);
        nint gateWeightsDev = scratch.GroupedPtrArrays;
        nint gateOutputsDev = scratch.GroupedPtrArrays + (nint)perArrayBytes;
        nint upWeightsDev   = scratch.GroupedPtrArrays + (nint)(2 * perArrayBytes);
        nint upOutputsDev   = scratch.GroupedPtrArrays + (nint)(3 * perArrayBytes);

        // Launch grouped gate + grouped up GEMVs. Each kernel walks (M output
        // rows) × (kActive experts) blocks.
        kernels.LaunchMoeGroupedGemv(
            gateWeightsDev, gateOutputsDev,
            scratch.GemvInputF16, weights.GateProjQuantType,
            M: I, K: hidden, kActive: kActive, stream);
        kernels.LaunchMoeGroupedGemv(
            upWeightsDev, upOutputsDev,
            scratch.GemvInputF16, weights.UpProjQuantType,
            M: I, K: hidden, kActive: kActive, stream);

        // Convert F16 outputs (kActive × I halfs each) → F32 in one launch each
        // into the existing gate/up F32 batch buffers, which the per-expert loop
        // below treats as a [kActive, I] layout and slices via e_local * I offsets.
        long elemsAll = (long)kActive * I;
        kernels.LaunchConvertF16ToF32(scratch.GroupedGateF16, scratch.GateBatch, (int)elemsAll, stream);
        kernels.LaunchConvertF16ToF32(scratch.GroupedUpF16,   scratch.UpBatch,   (int)elemsAll, stream);
    }
}

/// <summary>
/// Caller-owned per-call scratch buffers for <see cref="CudaMoeFfn.Forward"/>.
/// Sized to (<c>seqLen</c>, <c>numExperts</c>, <c>topK</c>, <c>moeIntermediateSize</c>,
/// <c>sharedIntermediateSize</c>); reallocated with power-of-2 growth on demand.
/// Reused across layers and forward calls.
/// </summary>
public sealed unsafe class CudaMoeScratch : IDisposable
{
    private nint _logits;            // [seqLen, numExperts] F32
    private nint _topkIdx;           // [seqLen, topK]       int32
    private nint _topkWeight;        // [seqLen, topK]       F32
    private nint _tokenIndices;      // [seqLen * topK]      int32 (per-expert slice)
    private nint _singleTokenScratch;// [1]                  int32 (mixed-slot fallback)
    private nint _gatheredInput;     // [seqLen * topK, hidden] F32 worst case
    private nint _gateBatch;         // [seqLen * topK, I]   F32
    private nint _upBatch;           // [seqLen * topK, I]   F32
    private nint _siluBatch;         // [seqLen * topK, I]   F32
    private nint _downBatch;         // [seqLen * topK, hidden] F32
    private nint _downRowScale;      // [seqLen * topK]      F32 (power-of-two down-input factors, #1172)
    private nint _sharedGateBatch;   // [seqLen, sI]         F32
    private nint _sharedUpBatch;
    private nint _sharedSiluBatch;
    private nint _sharedDownBatch;   // [seqLen, hidden]     F32
    private nint _sharedScale;       // [seqLen]             F32

    // Quantized-path dequant scratches: F16 for the LaunchDequantToF16 output,
    // F32 for the cuBLAS-LinearF32-consumed weight after F16→F32 conversion.
    // Sized to the largest single projection (max(I*hidden, hidden*sI) elements).
    // Reused across all per-expert and per-shared-expert calls in a forward.
    private nint _dequantF16;         // F16 scratch (sizeof(ushort) per elem)
    private nint _dequantF32;         // F32 scratch (sizeof(float) per elem)
    private long _dequantElems;
    private bool _capQuantized;

    // Direct-quantized-GEMV path scratches (decode batch=1 with K%256==0):
    // _gemvInputF16 holds K halfs (input row converted F32→F16 once per
    // (expert, projection) — only when GEMV-eligible); _gemvOutputF16 holds
    // M halfs (LaunchQuantizedGemv output, then converted F16→F32 into the
    // existing F32 output buffer). Both small (~2-4 KB each at V2-Lite).
    private nint _gemvInputF16;       // K_max halfs
    private nint _gemvOutputF16;      // M_max halfs
    private long _gemvKMax;
    private long _gemvMMax;

    // Phase-B grouped-GEMV scratch (decode batch=1, all K_active experts in
    // one launch). Two F16 staging buffers — one per projection (gate, up) —
    // each holding K_active_max × I_max halfs. After the grouped kernel, the
    // F16 outputs are converted to F32 in one launch into the existing
    // _gateBatch / _upBatch buffers (which are already sized [K_active_max, I]).
    // Plus a small device-resident pointer-array buffer (4 × K_active_max
    // nints — gate-weights, gate-outputs, up-weights, up-outputs).
    private nint _groupedGateF16;     // [K_active_max, I_max] halfs
    private nint _groupedUpF16;       // [K_active_max, I_max] halfs
    private nint _groupedPtrArrays;   // 4 × K_active_max nints (8 bytes each)
    private nint _hostTopkIdx;        // pinned host [maxBatch] int32
    private long _groupedActiveMax;   // K_active_max actually allocated for
    private long _groupedIMax;        // I_max actually allocated for

    private int _capSeqLen, _capE, _capK, _capHidden, _capI, _capSI;
    private bool _capHasShared;

    /// <summary>Total allocated bytes across all scratch buffers.</summary>
    public long AllocatedBytes { get; private set; }

    internal nint Logits => _logits;
    internal nint TopkIdx => _topkIdx;
    internal nint TopkWeight => _topkWeight;
    internal nint TokenIndices => _tokenIndices;
    internal nint SingleTokenScratch => _singleTokenScratch;
    internal nint GatheredInput => _gatheredInput;
    internal nint GateBatch => _gateBatch;
    internal nint UpBatch => _upBatch;
    internal nint SiluBatch => _siluBatch;
    internal nint DownBatch => _downBatch;
    internal nint DownRowScale => _downRowScale;
    internal nint SharedGateBatch => _sharedGateBatch;
    internal nint SharedUpBatch => _sharedUpBatch;
    internal nint SharedSiluBatch => _sharedSiluBatch;
    internal nint SharedDownBatch => _sharedDownBatch;
    internal nint SharedScale => _sharedScale;
    internal nint DequantF16 => _dequantF16;
    internal nint DequantF32 => _dequantF32;
    internal nint GemvInputF16 => _gemvInputF16;
    internal nint GemvOutputF16 => _gemvOutputF16;
    internal nint GroupedGateF16 => _groupedGateF16;
    internal nint GroupedUpF16 => _groupedUpF16;
    internal nint GroupedPtrArrays => _groupedPtrArrays;

    /// <summary>Pinned host copy of the top-k ids <c>[seqLen * topK]</c> (async read-back, #1202).</summary>
    internal nint HostTopkIdx => _hostTopkIdx;
    internal long GroupedActiveMax => _groupedActiveMax;
    internal long GroupedIMax => _groupedIMax;

    /// <summary>Ensures all scratch buffers fit the requested workload.</summary>
    public void EnsureCapacity(int seqLen, CudaMoeLayerWeights weights)
    {
        bool hasShared = weights.NumSharedExperts > 0 && weights.SharedIntermediateSize > 0;
        if (seqLen <= _capSeqLen
            && weights.NumExperts == _capE
            && weights.NumExpertsPerTok == _capK
            && weights.HiddenSize == _capHidden
            && weights.MoeIntermediateSize == _capI
            && (!hasShared || (weights.SharedIntermediateSize == _capSI && hasShared == _capHasShared)))
            return;

        int newCap = Math.Max(seqLen, 1);
        if (newCap > _capSeqLen)
            newCap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)newCap);

        Free();

        _capSeqLen = newCap;
        _capE = weights.NumExperts;
        _capK = weights.NumExpertsPerTok;
        _capHidden = weights.HiddenSize;
        _capI = weights.MoeIntermediateSize;
        _capSI = weights.SharedIntermediateSize;
        _capHasShared = hasShared;

        long maxBatch = (long)newCap * _capK;

        _logits = AllocF32((long)newCap * _capE);
        _topkIdx = AllocInt32(maxBatch);
        _topkWeight = AllocF32(maxBatch);
        _tokenIndices = AllocInt32(2 * maxBatch);
        CudaDriverApi.cuMemHostAlloc(out _hostTopkIdx, (nuint)(maxBatch * sizeof(int)), 0).ThrowOnError();   // [row token | assignment row] for the bucketed body
        _singleTokenScratch = AllocInt32(1);
        _gatheredInput = AllocF32(maxBatch * _capHidden);
        _gateBatch = AllocF32(maxBatch * _capI);
        _upBatch = AllocF32(maxBatch * _capI);
        _siluBatch = AllocF32(maxBatch * _capI);
        _downBatch = AllocF32(maxBatch * _capHidden);
        _downRowScale = AllocF32(maxBatch);

        if (hasShared)
        {
            _sharedGateBatch = AllocF32((long)newCap * _capSI);
            _sharedUpBatch = AllocF32((long)newCap * _capSI);
            _sharedSiluBatch = AllocF32((long)newCap * _capSI);
            _sharedDownBatch = AllocF32((long)newCap * _capHidden);
            _sharedScale = AllocF32(newCap);
        }

        // Dequant scratch — quantized path only. Sized to the largest single
        // projection across routed (I × hidden) and shared (sI × hidden).
        // Two buffers: F16 dequant target + F32 conversion target. Both
        // reused across all per-expert calls in a forward.
        bool isQuant = weights.Precision == MoePrecision.Quantized;
        _capQuantized = isQuant;
        if (isQuant)
        {
            long routedMax = (long)_capI * _capHidden;
            long sharedMax = hasShared ? (long)_capSI * _capHidden : 0;
            long maxElems = Math.Max(routedMax, sharedMax);
            if (maxElems > _dequantElems)
            {
                FreeIf(ref _dequantF16);
                FreeIf(ref _dequantF32);
                long f16Bytes = maxElems * sizeof(ushort);
                long f32Bytes = maxElems * sizeof(float);
                CudaDriverApi.cuMemAlloc_v2(out _dequantF16, (nuint)f16Bytes).ThrowOnError();
                AllocatedBytes += f16Bytes;
                CudaDriverApi.cuMemAlloc_v2(out _dequantF32, (nuint)f32Bytes).ThrowOnError();
                AllocatedBytes += f32Bytes;
                _dequantElems = maxElems;
            }

            // Direct-quantized-GEMV path scratches. K_max = max input dim
            // across all projections (= hidden for gate/up, intermediate for
            // down). M_max = max output dim. ~4 KB each at V2-Lite scale.
            long kMax = Math.Max(_capHidden, _capI);
            long mMax = Math.Max(_capI, _capHidden);
            if (hasShared)
            {
                kMax = Math.Max(kMax, _capSI);
                mMax = Math.Max(mMax, _capSI);
            }
            if (kMax > _gemvKMax)
            {
                FreeIf(ref _gemvInputF16);
                long bytes = kMax * sizeof(ushort);
                CudaDriverApi.cuMemAlloc_v2(out _gemvInputF16, (nuint)bytes).ThrowOnError();
                AllocatedBytes += bytes;
                _gemvKMax = kMax;
            }
            if (mMax > _gemvMMax)
            {
                FreeIf(ref _gemvOutputF16);
                long bytes = mMax * sizeof(ushort);
                CudaDriverApi.cuMemAlloc_v2(out _gemvOutputF16, (nuint)bytes).ThrowOnError();
                AllocatedBytes += bytes;
                _gemvMMax = mMax;
            }

            // Phase-B grouped-GEMV scratch. K_active_max = numExpertsPerTok (decode
            // sees up to K active experts per token; we route per-token into K_active
            // slots). I_max = MoE intermediate (the per-projection output dim).
            long activeMax = _capK;
            long iMaxGrouped = _capI;
            long needGroupedF16 = activeMax * iMaxGrouped * sizeof(ushort);
            if (activeMax > _groupedActiveMax || iMaxGrouped > _groupedIMax)
            {
                FreeIf(ref _groupedGateF16);
                FreeIf(ref _groupedUpF16);
                FreeIf(ref _groupedPtrArrays);
                CudaDriverApi.cuMemAlloc_v2(out _groupedGateF16, (nuint)needGroupedF16).ThrowOnError();
                AllocatedBytes += needGroupedF16;
                CudaDriverApi.cuMemAlloc_v2(out _groupedUpF16, (nuint)needGroupedF16).ThrowOnError();
                AllocatedBytes += needGroupedF16;
                // 6 ptr arrays × K_active_max × sizeof(nint): gate W/out, up W/out, and for the fully grouped
                // Q4_1 decode body the down weights and each expert's routing slot (#1172).
                long ptrBytes = 6L * activeMax * sizeof(long);
                CudaDriverApi.cuMemAlloc_v2(out _groupedPtrArrays, (nuint)ptrBytes).ThrowOnError();
                AllocatedBytes += ptrBytes;
                _groupedActiveMax = activeMax;
                _groupedIMax = iMaxGrouped;
            }
        }
    }

    private nint AllocF32(long elems)
    {
        long bytes = elems * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return p;
    }

    private nint AllocInt32(long elems)
    {
        long bytes = elems * sizeof(int);
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return p;
    }

    private void Free()
    {
        FreeIf(ref _logits); FreeIf(ref _topkIdx); FreeIf(ref _topkWeight);
        FreeIf(ref _tokenIndices); FreeIf(ref _singleTokenScratch);
        FreeIf(ref _gatheredInput);
        FreeIf(ref _gateBatch); FreeIf(ref _upBatch);
        FreeIf(ref _siluBatch); FreeIf(ref _downBatch); FreeIf(ref _downRowScale);
        FreeIf(ref _sharedGateBatch); FreeIf(ref _sharedUpBatch);
        FreeIf(ref _sharedSiluBatch); FreeIf(ref _sharedDownBatch);
        FreeIf(ref _sharedScale);
        FreeIf(ref _dequantF16); FreeIf(ref _dequantF32);
        FreeIf(ref _gemvInputF16); FreeIf(ref _gemvOutputF16);
        FreeIf(ref _groupedGateF16); FreeIf(ref _groupedUpF16);
        FreeIf(ref _groupedPtrArrays);
        if (_hostTopkIdx != 0) { CudaDriverApi.cuMemFreeHost(_hostTopkIdx); _hostTopkIdx = 0; }
        _dequantElems = 0;
        _gemvKMax = 0; _gemvMMax = 0;
        _groupedActiveMax = 0; _groupedIMax = 0;
        AllocatedBytes = 0;
    }

    private static void FreeIf(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Frees every device buffer.</summary>
    public void Dispose()
    {
        Free();
        _capSeqLen = 0;
    }
}
