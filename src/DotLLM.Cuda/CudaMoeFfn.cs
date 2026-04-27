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
    public static void Forward(
        nint hiddenF32, nint outputF32,
        int seqLen,
        CudaMoeLayerWeights weights,
        CudaMoeScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream)
    {
        if (!kernels.HasMoeKernels)
            throw new InvalidOperationException(
                "MoE kernels not available. Compile native/kernels/moe_ffn.cu to PTX.");

        if (seqLen <= 0) return;

        int hidden = weights.HiddenSize;
        int E = weights.NumExperts;
        int K = weights.NumExpertsPerTok;
        int I = weights.MoeIntermediateSize;
        int totalAssign = seqLen * K;

        scratch.EnsureCapacity(seqLen, weights);

        // ── Step 1: clear output ──
        kernels.LaunchMoeZeroF32(outputF32, seqLen * hidden, stream);

        // ── Step 2: routing GEMV — logits[seqLen, numExperts] = hidden @ router^T ──
        CudaGemm.LinearF32(
            cublasHandle, hiddenF32, weights.Router, scratch.Logits,
            seqLen, hidden, E, stream);

        // ── Step 3: per-token softmax + top-k selection → device buffers ──
        kernels.LaunchMoeSoftmaxTopk(
            scratch.Logits, scratch.TopkIdx, scratch.TopkWeight,
            seqLen, E, K, stream);

        // ── Step 4: optional renorm of top-k weights ──
        if (weights.NormTopKProb)
            kernels.LaunchMoeRenormTopk(scratch.TopkWeight, seqLen, K, stream);

        // ── Step 5: download top-k indices to host so we can dispatch
        // per-expert grouped batches. Per-token weights stay device-resident —
        // the per-expert axpy reads them from device memory by (token, slot)
        // index. ──
        // Sync once here — necessary so the host can read topkIdx. Cheap for
        // decode (seqLen=1, K=2..8).
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        int[] topkIdxHost = new int[totalAssign];
        fixed (int* p = topkIdxHost)
            CudaDriverApi.cuMemcpyDtoH_v2(
                (nint)p, scratch.TopkIdx,
                (nuint)((long)totalAssign * sizeof(int))).ThrowOnError();

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
            && (hidden % 256) == 0
            && kernels.HasMoeGroupedGemv(weights.GateProjQuantType)
            && kernels.HasMoeGroupedGemv(weights.UpProjQuantType)
            && weights.GateProjQuantType == weights.UpProjQuantType
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

        if (useGrouped && kActive > 0)
        {
            DispatchGroupedGateUp(
                hiddenF32, weights, scratch, kernels, stream,
                expertLocalIdx: expertLocalIdx, kActive: kActive,
                hidden: hidden, I: I);
        }

        for (int e = 0; e < E; e++)
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
                    outputF32: scratch.GateBatch);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.GatheredInput, batch, K: hidden, M: I,
                    weightF32: weights.UpProj[e],
                    weightQuant: weights.UpProj[e], weightQt: weights.UpProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                    gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.UpBatch);
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
            kernels.LaunchSwiGLUF32(
                scratch.GateBatch + (nint)gateOff,
                scratch.UpBatch + (nint)upOff,
                scratch.SiluBatch,
                I, batch, stream);

            // 5. GEMM down.
            //    down[batch, hidden] = silu[batch, I] × W2[hidden, I]^T
            //    K=intermediate may not be 256-aligned (V2-Lite intermediate=1408)
            //    so down keeps using the dequant fallback. No grouped variant yet.
            ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                scratch.SiluBatch, batch, K: I, M: hidden,
                weightF32: weights.DownProj[e],
                weightQuant: weights.DownProj[e], weightQt: weights.DownProjQuantType,
                dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                outputF32: scratch.DownBatch);

            // 6. Per-slot axpy (group by slot to amortise weight lookups).
            //    The fast common case is K=2..8; we walk slots 0..K-1 and
            //    upload only the rows that belong to this (expert, slot)
            //    combo. For decode (seqLen=1, batch=1) this collapses to one
            //    launch.
            //
            //    Sub-bucket by slot. We reuse scratch.SlotBuckets as
            //    temporary host scratch.
            //
            //    Alternative implementation (simpler, slightly more launches):
            //    walk every batch row and issue one axpy per row. We pick the
            //    slot-grouped path because typical K is small, so the number
            //    of launches per expert is bounded by K rather than batch.

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
                        outputF32, scratch.DownBatch,
                        scratch.TopkWeight,
                        scratch.TokenIndices + (nint)((long)start * sizeof(int)),
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
                            scratch.SingleTokenScratch,
                            (nint)(&tokenId), sizeof(int)).ThrowOnError();
                        // Issue an axpy for the single down row at index b.
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

        // ── Step 8: shared-expert path (DeepSeek / Qwen1.5-MoE) ──
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
                    outputF32: scratch.SharedGateBatch);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    hiddenF32, seqLen, K: hidden, M: sI,
                    weightF32: weights.SharedUpProj[s],
                    weightQuant: weights.SharedUpProj[s], weightQt: weights.SharedUpProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.SharedUpBatch);
                kernels.LaunchSwiGLUF32(
                    scratch.SharedGateBatch, scratch.SharedUpBatch, scratch.SharedSiluBatch,
                    sI, seqLen, stream);
                ProjectF32OrQuant(weights.Precision, cublasHandle, kernels, stream,
                    scratch.SharedSiluBatch, seqLen, K: sI, M: hidden,
                    weightF32: weights.SharedDownProj[s],
                    weightQuant: weights.SharedDownProj[s], weightQt: weights.SharedDownProjQuantType,
                    dequantF16: scratch.DequantF16, dequantF32: scratch.DequantF32,
                gemvInputF16: scratch.GemvInputF16, gemvOutputF16: scratch.GemvOutputF16,
                    outputF32: scratch.SharedDownBatch);

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
        nint gemvInputF16, nint gemvOutputF16)
    {
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
                && CudaKernels.HasQuantizedGemv(weightQt);
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
        Span<int> expertLocalIdx, int kActive, int hidden, int I)
    {
        int E = weights.NumExperts;

        // Build host-side ptr arrays (4 × kActive nints, all packed into one
        // contiguous block matching the device-side layout).
        //   slice 0: gate weights[0..kActive)
        //   slice 1: gate outputs[0..kActive)
        //   slice 2: up weights  [0..kActive)
        //   slice 3: up outputs  [0..kActive)
        long* hostPtrs = stackalloc long[4 * kActive];

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
            eLocal++;
        }

        // One synchronous HtoD copy of all 4 ptr arrays at once. Source is
        // stack-local — must complete before this method returns. Sync HtoD
        // (cuMemcpyHtoD_v2) on pageable memory is immediate.
        long ptrBytes = 4L * kActive * sizeof(long);
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
        _tokenIndices = AllocInt32(maxBatch);
        _singleTokenScratch = AllocInt32(1);
        _gatheredInput = AllocF32(maxBatch * _capHidden);
        _gateBatch = AllocF32(maxBatch * _capI);
        _upBatch = AllocF32(maxBatch * _capI);
        _siluBatch = AllocF32(maxBatch * _capI);
        _downBatch = AllocF32(maxBatch * _capHidden);

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
                // 4 ptr arrays × K_active_max × sizeof(nint).
                long ptrBytes = 4L * activeMax * sizeof(long);
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
        FreeIf(ref _siluBatch); FreeIf(ref _downBatch);
        FreeIf(ref _sharedGateBatch); FreeIf(ref _sharedUpBatch);
        FreeIf(ref _sharedSiluBatch); FreeIf(ref _sharedDownBatch);
        FreeIf(ref _sharedScale);
        FreeIf(ref _dequantF16); FreeIf(ref _dequantF32);
        FreeIf(ref _gemvInputF16); FreeIf(ref _gemvOutputF16);
        FreeIf(ref _groupedGateF16); FreeIf(ref _groupedUpF16);
        FreeIf(ref _groupedPtrArrays);
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
