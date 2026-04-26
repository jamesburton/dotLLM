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

        if (weights.Precision == MoePrecision.Quantized)
            throw new NotImplementedException(
                "MoE quantized forward path (per-expert raw GGUF quant bytes + on-device dequant) " +
                "is not yet wired into CudaMoeFfn.Forward. Loader-side scaffolding " +
                "(CudaMoeWeightsLoader.LoadLayerQuant) is in place; the per-expert dequant-to-F16-scratch " +
                "+ HGEMM (or quantized GEMV) branch is the next step. See " +
                "docs/perf/DEEPSEEK_QUANTIZED_GPU_PATH.md (#10-ii). For now this path fails fast " +
                "to avoid silently feeding raw Q4_K bytes through cuBLAS LinearF32 (would produce " +
                "garbage logits).");

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
        //
        // This is correct but suboptimal for high-fan-out experts; the
        // grouped-GEMM optimisation is an explicit follow-up.

        for (int e = 0; e < E; e++)
        {
            int batch = counts[e];
            if (batch == 0) continue;
            int start = offsets[e];

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
            CudaGemm.LinearF32(
                cublasHandle, scratch.GatheredInput, weights.GateProj[e],
                scratch.GateBatch, batch, hidden, I, stream);
            CudaGemm.LinearF32(
                cublasHandle, scratch.GatheredInput, weights.UpProj[e],
                scratch.UpBatch, batch, hidden, I, stream);

            // 4. SwiGLU element-wise.
            kernels.LaunchSwiGLUF32(
                scratch.GateBatch, scratch.UpBatch, scratch.SiluBatch,
                I, batch, stream);

            // 5. GEMM down.
            //    down[batch, hidden] = silu[batch, I] × W2[hidden, I]^T
            CudaGemm.LinearF32(
                cublasHandle, scratch.SiluBatch, weights.DownProj[e],
                scratch.DownBatch, batch, I, hidden, stream);

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
                CudaGemm.LinearF32(
                    cublasHandle, hiddenF32, weights.SharedGateProj[s],
                    scratch.SharedGateBatch, seqLen, hidden, sI, stream);
                CudaGemm.LinearF32(
                    cublasHandle, hiddenF32, weights.SharedUpProj[s],
                    scratch.SharedUpBatch, seqLen, hidden, sI, stream);
                kernels.LaunchSwiGLUF32(
                    scratch.SharedGateBatch, scratch.SharedUpBatch, scratch.SharedSiluBatch,
                    sI, seqLen, stream);
                CudaGemm.LinearF32(
                    cublasHandle, scratch.SharedSiluBatch, weights.SharedDownProj[s],
                    scratch.SharedDownBatch, seqLen, sI, hidden, stream);

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
