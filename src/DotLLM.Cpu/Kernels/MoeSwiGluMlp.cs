using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Lora;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Dense-routing top-k Mixture-of-Experts SwiGLU FFN kernel. Drops into the
/// per-layer MLP slot in a Mixtral-convention transformer block: for each
/// token the router picks the top-k experts (softmax over all N experts,
/// gather top-k, renormalise by sum), each selected expert runs a SwiGLU
/// MLP on the token, and the outputs are weighted-summed back.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference semantics.</b> This matches the HuggingFace Mixtral reference
/// (<c>transformers/models/mixtral/modeling_mixtral.py::MixtralSparseMoeBlock.forward</c>):
/// </para>
/// <code>
///   gate_logits = hidden @ gate.T                # [T, E]
///   routing    = softmax(gate_logits, dim=-1)    # full softmax
///   w, idx     = topk(routing, k, dim=-1)        # top-k probs+indices
///   w          = w / w.sum(-1, keepdim=True)     # renormalise (NOT softmax)
///   out        = sum_{e in idx} w[e] * expert_e(hidden)
/// </code>
/// <para>
/// <b>Tiebreaker.</b> Partial selection of the top-k uses a stable max-scan:
/// when two experts tie on probability, the <i>lower-indexed</i> expert
/// wins. This is deterministic and matches PyTorch's <c>torch.topk</c>
/// behaviour on the forward-order CPU path.
/// </para>
/// <para>
/// <b>GroupedGEMM execution.</b> The kernel buckets tokens by their assigned
/// expert, runs one batched SwiGLU GEMM per expert (gather → GEMM → SwiGLU →
/// GEMM → scatter), and finally accumulates each token's output by walking
/// its top-k slots in order. The per-(token,slot) accumulation order is
/// identical to the original scalar path, so the output is bit-identical.
/// </para>
/// <para>
/// <b>Weight layout.</b> Expert weights are passed as three flat arrays of
/// <c>nint</c> — one entry per expert, each pointing at row-major F32
/// <c>[intermediate, hidden]</c> (<c>w1</c>/<c>w3</c>) or
/// <c>[hidden, intermediate]</c> (<c>w2</c>) weight matrices. The router
/// <c>gateWeights</c> is row-major F32 <c>[numExperts, hiddenSize]</c>.
/// </para>
/// </remarks>
public static unsafe class MoeSwiGluMlp
{
    /// <summary>
    /// Executes the MoE SwiGLU FFN for a batch of <paramref name="seqLen"/>
    /// tokens. Reads <paramref name="hidden"/> [seqLen × hiddenSize], writes
    /// into <paramref name="output"/> [seqLen × hiddenSize].
    /// </summary>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="gateWeights">F32 router weight [numExperts × hiddenSize] row-major.</param>
    /// <param name="expertsW1">Per-expert gate_proj pointers — F32 [intermediateSize × hiddenSize] row-major, <paramref name="numExperts"/> entries.</param>
    /// <param name="expertsW2">Per-expert down_proj pointers — F32 [hiddenSize × intermediateSize] row-major.</param>
    /// <param name="expertsW3">Per-expert up_proj pointers — F32 [intermediateSize × hiddenSize] row-major.</param>
    /// <param name="output">F32 output activations [seqLen × hiddenSize]. Fully overwritten.</param>
    /// <param name="numExperts">Total expert count per layer (E).</param>
    /// <param name="numExpertsPerTok">Top-k: number of experts activated per token.</param>
    /// <param name="hiddenSize">Hidden / residual dimension (H).</param>
    /// <param name="intermediateSize">Per-expert MLP intermediate dimension (I).</param>
    /// <param name="seqLen">Number of tokens in this batch (T).</param>
    /// <param name="loraAdapter">Optional active LoRA adapter for per-expert projection deltas.</param>
    /// <param name="loraLayer">Layer index used to resolve adapter weights.</param>
    [SkipLocalsInit]
    public static void Execute(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> gateWeights,
        ReadOnlySpan<nint> expertsW1,
        ReadOnlySpan<nint> expertsW2,
        ReadOnlySpan<nint> expertsW3,
        Span<float> output,
        int numExperts,
        int numExpertsPerTok,
        int hiddenSize,
        int intermediateSize,
        int seqLen,
        ILoraAdapter? loraAdapter = null,
        int loraLayer = -1)
    {
        // Default overload keeps the Mixtral contract: always renormalise top-k,
        // no shared expert. Qwen-MoE / DeepSeek callers go through
        // ExecuteWithSharedExpert.
        ExecuteCoreGrouped(
            hidden, gateWeights, expertsW1, expertsW2, expertsW3, output,
            numExperts, numExpertsPerTok, hiddenSize, intermediateSize, seqLen,
            normTopKProb: true,
            sharedGateProj: ReadOnlySpan<nint>.Empty,
            sharedUpProj: ReadOnlySpan<nint>.Empty,
            sharedDownProj: ReadOnlySpan<nint>.Empty,
            sharedIntermediateSize: 0, sharedExpertGate: default,
            loraAdapter, loraLayer);
    }

    /// <summary>
    /// Qwen-MoE / DeepSeek overload: computes routed top-k output + summed
    /// dense shared-expert output (optionally sigmoid-gated). Supports
    /// multiple shared experts (DeepSeek-V2/V3 <c>n_shared_experts &gt;= 1</c>):
    /// each runs a dense SwiGLU on the token and their outputs are summed
    /// before the (optional) per-token sigmoid scale is applied. Pass three
    /// empty pointer spans with <paramref name="sharedIntermediateSize"/> = 0
    /// to fall back to the pure routed path (equivalent to <see cref="Execute"/>).
    /// </summary>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="gateWeights">F32 router weight [numExperts × hiddenSize] row-major.</param>
    /// <param name="expertsW1">Per-expert gate_proj pointers — F32 [intermediateSize × hiddenSize] row-major.</param>
    /// <param name="expertsW2">Per-expert down_proj pointers — F32 [hiddenSize × intermediateSize] row-major.</param>
    /// <param name="expertsW3">Per-expert up_proj pointers — F32 [intermediateSize × hiddenSize] row-major.</param>
    /// <param name="output">F32 output activations [seqLen × hiddenSize]. Fully overwritten.</param>
    /// <param name="numExperts">Total expert count per layer (E).</param>
    /// <param name="numExpertsPerTok">Top-k: number of routed experts activated per token.</param>
    /// <param name="hiddenSize">Hidden / residual dimension (H).</param>
    /// <param name="intermediateSize">Per-routed-expert MLP intermediate dimension (I).</param>
    /// <param name="seqLen">Number of tokens in this batch (T).</param>
    /// <param name="normTopKProb">
    /// <c>true</c> → renormalise the selected top-k probabilities to sum to 1.0
    /// (Mixtral + Qwen3-MoE). <c>false</c> → use raw softmax values as gating
    /// weights (Qwen1.5-MoE default).
    /// </param>
    /// <param name="sharedGateProj">
    /// Per-shared-expert gate_proj pointers — F32 [sharedIntermediateSize × hiddenSize]
    /// row-major. Length = number of shared experts (1 for Qwen1.5-MoE; 1..N
    /// for DeepSeek-V2/V3). Empty span ⇒ no shared expert.
    /// </param>
    /// <param name="sharedUpProj">Per-shared-expert up_proj pointers, same length as <paramref name="sharedGateProj"/>.</param>
    /// <param name="sharedDownProj">Per-shared-expert down_proj pointers, same length as <paramref name="sharedGateProj"/>.</param>
    /// <param name="sharedIntermediateSize">Per-shared-expert intermediate width (0 to disable).</param>
    /// <param name="sharedExpertGate">
    /// Optional F32 [hiddenSize] sigmoid-gate weight. Length 0 → no sigmoid
    /// scaling (DeepSeek; Qwen-MoE variants without <c>shared_expert_gate</c>).
    /// </param>
    /// <param name="loraAdapter">Optional active LoRA adapter for routed per-expert projection deltas.</param>
    /// <param name="loraLayer">Layer index used to resolve adapter weights.</param>
    [SkipLocalsInit]
    public static void ExecuteWithSharedExpert(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> gateWeights,
        ReadOnlySpan<nint> expertsW1,
        ReadOnlySpan<nint> expertsW2,
        ReadOnlySpan<nint> expertsW3,
        Span<float> output,
        int numExperts,
        int numExpertsPerTok,
        int hiddenSize,
        int intermediateSize,
        int seqLen,
        bool normTopKProb,
        ReadOnlySpan<nint> sharedGateProj,
        ReadOnlySpan<nint> sharedUpProj,
        ReadOnlySpan<nint> sharedDownProj,
        int sharedIntermediateSize,
        ReadOnlySpan<float> sharedExpertGate,
        ILoraAdapter? loraAdapter = null,
        int loraLayer = -1)
    {
        ExecuteCoreGrouped(
            hidden, gateWeights, expertsW1, expertsW2, expertsW3, output,
            numExperts, numExpertsPerTok, hiddenSize, intermediateSize, seqLen,
            normTopKProb,
            sharedGateProj, sharedUpProj, sharedDownProj,
            sharedIntermediateSize, sharedExpertGate,
            loraAdapter, loraLayer);
    }

    /// <summary>
    /// GroupedGEMM MoE core: routes once, buckets tokens per expert, runs one
    /// batched SwiGLU GEMM per active expert, then scatters the per-(token,slot)
    /// "down" outputs back into a dense table. The final accumulation walks
    /// <c>(t, slot=0..k-1)</c> in order — same accumulation order as the
    /// scalar per-token loop, so the output is bit-identical.
    /// </summary>
    [SkipLocalsInit]
    private static void ExecuteCoreGrouped(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> gateWeights,
        ReadOnlySpan<nint> expertsW1,
        ReadOnlySpan<nint> expertsW2,
        ReadOnlySpan<nint> expertsW3,
        Span<float> output,
        int numExperts,
        int numExpertsPerTok,
        int hiddenSize,
        int intermediateSize,
        int seqLen,
        bool normTopKProb,
        ReadOnlySpan<nint> sharedGateProj,
        ReadOnlySpan<nint> sharedUpProj,
        ReadOnlySpan<nint> sharedDownProj,
        int sharedIntermediateSize,
        ReadOnlySpan<float> sharedExpertGate,
        ILoraAdapter? loraAdapter,
        int loraLayer)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (numExpertsPerTok <= 0 || numExpertsPerTok > numExperts)
            throw new ArgumentOutOfRangeException(nameof(numExpertsPerTok));
        if (hidden.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("hidden too small", nameof(hidden));
        if (output.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("output too small", nameof(output));
        if (gateWeights.Length < (long)numExperts * hiddenSize)
            throw new ArgumentException("gateWeights too small", nameof(gateWeights));
        if (expertsW1.Length != numExperts || expertsW2.Length != numExperts || expertsW3.Length != numExperts)
            throw new ArgumentException("Expert weight arrays must each have numExperts entries.");
        if (sharedGateProj.Length != sharedUpProj.Length || sharedGateProj.Length != sharedDownProj.Length)
            throw new ArgumentException("Shared-expert weight spans must all have the same length.");

        if (seqLen == 0) return;

        int numSharedExperts = sharedGateProj.Length;
        bool hasSharedExpert = sharedIntermediateSize > 0 && numSharedExperts > 0;
        bool hasSharedGate = hasSharedExpert && sharedExpertGate.Length >= hiddenSize;

        int totalAssignments = seqLen * numExpertsPerTok;

        // ────────────── Pooled scratch buffers ──────────────
        // All per-call allocations routed through ArrayPool — zero sustained
        // heap traffic. .AsSpan(0, actualCount) because Rent may return larger.

        // Router scratch (per-token): gate logits & full softmax.
        float[] gateLogitsBuf = ArrayPool<float>.Shared.Rent(seqLen * numExperts);
        float[] routingBuf = ArrayPool<float>.Shared.Rent(seqLen * numExperts);

        // Per-assignment (token × slot) routing results.
        int[] assignExpertBuf = ArrayPool<int>.Shared.Rent(totalAssignments);
        float[] assignWeightBuf = ArrayPool<float>.Shared.Rent(totalAssignments);

        // Per-expert buckets: for expert e, tokens are stored at indices
        // [expertOffset[e], expertOffset[e+1]). Each entry stores
        // (tokenIdx, slot) packed so the scatter can walk a single stream.
        int[] expertCountBuf = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] expertTokenBuf = ArrayPool<int>.Shared.Rent(totalAssignments);
        int[] expertSlotBuf = ArrayPool<int>.Shared.Rent(totalAssignments);

        // Gathered per-expert batched inputs + batched GEMM scratch. Sized
        // for the worst-case batch = totalAssignments (an expert can appear in
        // every slot of every token). Intermediate widths cover both routed
        // and shared paths so one rent serves both.
        int maxIntermediate = hasSharedExpert
            ? Math.Max(intermediateSize, sharedIntermediateSize)
            : intermediateSize;
        int maxRoutedBatch = totalAssignments;

        float[] batchInBuf = ArrayPool<float>.Shared.Rent(maxRoutedBatch * hiddenSize);
        float[] gateBatchBuf = ArrayPool<float>.Shared.Rent(maxRoutedBatch * maxIntermediate);
        float[] upBatchBuf = ArrayPool<float>.Shared.Rent(maxRoutedBatch * maxIntermediate);
        float[] siluBatchBuf = ArrayPool<float>.Shared.Rent(maxRoutedBatch * maxIntermediate);
        float[] downBatchBuf = ArrayPool<float>.Shared.Rent(maxRoutedBatch * hiddenSize);

        // Per-(token,slot) down-projected outputs, keyed by t*k + slot. This
        // is the table we index from during the final in-order accumulation.
        float[] scatterDownBuf = ArrayPool<float>.Shared.Rent(totalAssignments * hiddenSize);

        // Shared-expert scratch: batched across all seqLen tokens at once.
        int sharedBatch = hasSharedExpert ? seqLen : 0;
        float[]? sharedGateBatchBuf = null;
        float[]? sharedUpBatchBuf = null;
        float[]? sharedSiluBatchBuf = null;
        float[]? sharedDownBatchBuf = null;
        float[]? sharedScaleBuf = null;
        if (hasSharedExpert)
        {
            sharedGateBatchBuf = ArrayPool<float>.Shared.Rent(sharedBatch * sharedIntermediateSize);
            sharedUpBatchBuf = ArrayPool<float>.Shared.Rent(sharedBatch * sharedIntermediateSize);
            sharedSiluBatchBuf = ArrayPool<float>.Shared.Rent(sharedBatch * sharedIntermediateSize);
            sharedDownBatchBuf = ArrayPool<float>.Shared.Rent(sharedBatch * hiddenSize);
            sharedScaleBuf = ArrayPool<float>.Shared.Rent(sharedBatch);
        }

        // Per-token accumulator (same role as the scalar kernel's acc buffer).
        // Keeping 'output' untouched until the very end lets callers alias
        // hidden and output.
        float[] accBuf = ArrayPool<float>.Shared.Rent(hiddenSize);

        Span<int> topkIdx = stackalloc int[numExpertsPerTok];
        Span<float> topkProb = stackalloc float[numExpertsPerTok];

        try
        {
            Span<int> expertCount = expertCountBuf.AsSpan(0, numExperts + 1);
            expertCount.Clear();

            // ────────────── 1) Routing (per-token, unchanged) ──────────────
            // Gate-logit GEMV + softmax + top-k + optional renormalise. We
            // deliberately keep these as per-token calls to preserve the
            // exact reduction order of the scalar kernel (softmax, sort, etc.).
            fixed (float* hiddenPtr = hidden)
            fixed (float* gateWPtr = gateWeights)
            fixed (float* logitsAllPtr = gateLogitsBuf)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = hiddenPtr + t * hiddenSize;
                    float* logits = logitsAllPtr + t * numExperts;
                    MatMul.GemvF32(gateWPtr, x, logits, numExperts, hiddenSize);
                }
            }

            for (int t = 0; t < seqLen; t++)
            {
                var logitsSpan = gateLogitsBuf.AsSpan(t * numExperts, numExperts);
                var routingSpan = routingBuf.AsSpan(t * numExperts, numExperts);
                Softmax.Execute(logitsSpan, routingSpan);

                SelectTopK(routingSpan, topkIdx, topkProb);

                if (normTopKProb)
                {
                    float sum = 0f;
                    for (int i = 0; i < numExpertsPerTok; i++) sum += topkProb[i];
                    float invSum = sum > 0f ? 1.0f / sum : 0f;
                    for (int i = 0; i < numExpertsPerTok; i++) topkProb[i] *= invSum;
                }

                // Record assignments in (token, slot) order. Also bump the
                // per-expert occupancy count so we can build the histogram.
                for (int slot = 0; slot < numExpertsPerTok; slot++)
                {
                    int e = topkIdx[slot];
                    int assignIdx = t * numExpertsPerTok + slot;
                    assignExpertBuf[assignIdx] = e;
                    assignWeightBuf[assignIdx] = topkProb[slot];
                    expertCount[e]++;
                }
            }

            // ────────────── 2) Build per-expert token buckets ──────────────
            // Exclusive-prefix-sum turns the histogram into offsets into
            // expertTokenBuf / expertSlotBuf. expertCount[numExperts] holds
            // the running cursor during fill; we reset it to the starts after.
            int running = 0;
            for (int e = 0; e <= numExperts; e++)
            {
                int c = expertCount[e];
                expertCount[e] = running;
                running += c;
            }
            // expertCount is now the exclusive-scan offset array.
            // Use a temporary cursor array for the fill phase.
            // We overload expertCountBuf by keeping the scan in 'expertCount'
            // but we need per-expert write cursors; use a local stack alloc
            // for typical small E (<=256), else rent.
            int[]? cursorRented = null;
            Span<int> cursor = numExperts <= 256
                ? stackalloc int[numExperts]
                : (cursorRented = ArrayPool<int>.Shared.Rent(numExperts)).AsSpan(0, numExperts);
            for (int e = 0; e < numExperts; e++) cursor[e] = expertCount[e];

            try
            {
                for (int t = 0; t < seqLen; t++)
                {
                    for (int slot = 0; slot < numExpertsPerTok; slot++)
                    {
                        int assignIdx = t * numExpertsPerTok + slot;
                        int e = assignExpertBuf[assignIdx];
                        int pos = cursor[e]++;
                        expertTokenBuf[pos] = t;
                        expertSlotBuf[pos] = slot;
                    }
                }
            }
            finally
            {
                if (cursorRented is not null) ArrayPool<int>.Shared.Return(cursorRented);
            }

            // ────────────── 3) Per-expert batched SwiGLU ──────────────
            // For each expert with B>0 assignments: gather B token rows into
            // batchIn, run two [B,I] GEMMs for gate/up, SwiGLU fuse, then one
            // [B,H] GEMM for down. The resulting down rows are scattered into
            // scatterDown[(tokenIdx*k + slot) * H : +H] so the final
            // accumulation can walk (t, slot) in order.
            fixed (float* hiddenPtr = hidden)
            fixed (float* batchInPtr = batchInBuf)
            fixed (float* gateBatchPtr = gateBatchBuf)
            fixed (float* upBatchPtr = upBatchBuf)
            fixed (float* siluBatchPtr = siluBatchBuf)
            fixed (float* downBatchPtr = downBatchBuf)
            fixed (float* scatterDownPtr = scatterDownBuf)
            {
                for (int e = 0; e < numExperts; e++)
                {
                    int start = expertCount[e];
                    int end = expertCount[e + 1];
                    int batch = end - start;
                    if (batch == 0) continue;

                    // Gather hidden[tokenIdx] → batchIn[0..batch, :]
                    for (int b = 0; b < batch; b++)
                    {
                        int t = expertTokenBuf[start + b];
                        Buffer.MemoryCopy(
                            hiddenPtr + (long)t * hiddenSize,
                            batchInPtr + (long)b * hiddenSize,
                            hiddenSize * sizeof(float),
                            hiddenSize * sizeof(float));
                    }

                    float* w1 = (float*)expertsW1[e];
                    float* w2 = (float*)expertsW2[e];
                    float* w3 = (float*)expertsW3[e];

                    // GemmF32 computes C[N,M] = B[N,K] × A[M,K]^T.
                    //   gate[batch,I] = batchIn[batch,H] × w1[I,H]^T
                    MatMul.GemmF32(w1, batchInPtr, gateBatchPtr, intermediateSize, hiddenSize, batch);
                    MatMul.GemmF32(w3, batchInPtr, upBatchPtr, intermediateSize, hiddenSize, batch);
                    ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "gate_proj"),
                        batchInPtr, gateBatchPtr, batch, hiddenSize, intermediateSize);
                    ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "up_proj"),
                        batchInPtr, upBatchPtr, batch, hiddenSize, intermediateSize);

                    // Per-row SwiGLU (fuse identical to scalar path).
                    var gateBatchSpan = new Span<float>(gateBatchPtr, batch * intermediateSize);
                    var upBatchSpan = new Span<float>(upBatchPtr, batch * intermediateSize);
                    var siluBatchSpan = new Span<float>(siluBatchPtr, batch * intermediateSize);
                    for (int b = 0; b < batch; b++)
                    {
                        int off = b * intermediateSize;
                        FusedOps.SwiGLU(
                            gateBatchSpan.Slice(off, intermediateSize),
                            upBatchSpan.Slice(off, intermediateSize),
                            siluBatchSpan.Slice(off, intermediateSize));
                    }

                    //   down[batch,H] = silu[batch,I] × w2[H,I]^T
                    MatMul.GemmF32(w2, siluBatchPtr, downBatchPtr, hiddenSize, intermediateSize, batch);
                    ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "down_proj"),
                        siluBatchPtr, downBatchPtr, batch, intermediateSize, hiddenSize);

                    // Scatter each row of down to scatterDown[(t*k + slot)*H].
                    for (int b = 0; b < batch; b++)
                    {
                        int t = expertTokenBuf[start + b];
                        int slot = expertSlotBuf[start + b];
                        Buffer.MemoryCopy(
                            downBatchPtr + (long)b * hiddenSize,
                            scatterDownPtr + (long)(t * numExpertsPerTok + slot) * hiddenSize,
                            hiddenSize * sizeof(float),
                            hiddenSize * sizeof(float));
                    }
                }

            }

            // ────────────── 4) Shared-expert batched compute ──────────────
            // Hoisted out of the routed-expert fixed-scope so the shared-expert
            // down-batch pinning can span into the final accumulation loop.
            // Supports multiple shared experts (DeepSeek-V2/V3 n_shared_experts
            // >= 1): each runs a dense SwiGLU over hidden[T,H] and their
            // down[T,H] outputs are summed into sharedDownBatchBuf before the
            // (optional) per-token sigmoid gate is applied at accumulation
            // time. For a single shared expert (Qwen1.5-MoE convention) the
            // first-expert GEMM writes directly into sharedDownBatchBuf,
            // preserving the exact arithmetic of the pre-multi-shared kernel.
            if (hasSharedExpert)
            {
                fixed (float* hiddenPtr = hidden)
                fixed (float* sgBatch = sharedGateBatchBuf)
                fixed (float* suBatch = sharedUpBatchBuf)
                fixed (float* ssBatch = sharedSiluBatchBuf)
                fixed (float* sdBatch = sharedDownBatchBuf)
                fixed (float* sharedGatePtr = sharedExpertGate)
                {
                    for (int k = 0; k < numSharedExperts; k++)
                    {
                        float* sharedW1k = (float*)sharedGateProj[k];
                        float* sharedW3k = (float*)sharedUpProj[k];
                        float* sharedW2k = (float*)sharedDownProj[k];

                        //   gate_shared[T,sI] = hidden[T,H] × sharedW1[sI,H]^T
                        MatMul.GemmF32(sharedW1k, hiddenPtr, sgBatch,
                            sharedIntermediateSize, hiddenSize, seqLen);
                        MatMul.GemmF32(sharedW3k, hiddenPtr, suBatch,
                            sharedIntermediateSize, hiddenSize, seqLen);

                        var sharedGateSpanBatch = new Span<float>(sgBatch, seqLen * sharedIntermediateSize);
                        var sharedUpSpanBatch = new Span<float>(suBatch, seqLen * sharedIntermediateSize);
                        var sharedSiluSpanBatch = new Span<float>(ssBatch, seqLen * sharedIntermediateSize);
                        for (int t = 0; t < seqLen; t++)
                        {
                            int off = t * sharedIntermediateSize;
                            FusedOps.SwiGLU(
                                sharedGateSpanBatch.Slice(off, sharedIntermediateSize),
                                sharedUpSpanBatch.Slice(off, sharedIntermediateSize),
                                sharedSiluSpanBatch.Slice(off, sharedIntermediateSize));
                        }

                        if (k == 0)
                        {
                            //   down_shared[T,H] = silu[T,sI] × sharedW2[H,sI]^T
                            //   (first expert writes directly; preserves bit-
                            //    identity with the single-shared-expert path)
                            MatMul.GemmF32(sharedW2k, ssBatch, sdBatch,
                                hiddenSize, sharedIntermediateSize, seqLen);
                        }
                        else
                        {
                            // Accumulator path: compute this expert's down into
                            // siluBatchBuf (safe — we've consumed silu already)
                            // then TensorPrimitives-add into sdBatch.
                            // We keep it simple and reuse sharedSiluBatchBuf as
                            // temp storage since it's already sized for the
                            // larger of (sharedIntermediate, hiddenSize*seqLen).
                            // Actually it's sized for sharedIntermediate*seqLen
                            // which may be smaller than hiddenSize*seqLen — so
                            // rent a short-lived scratch here. numSharedExperts
                            // is tiny (1..4) so the branch is rare.
                            float[] tmp = ArrayPool<float>.Shared.Rent(seqLen * hiddenSize);
                            try
                            {
                                fixed (float* tmpPtr = tmp)
                                {
                                    MatMul.GemmF32(sharedW2k, ssBatch, tmpPtr,
                                        hiddenSize, sharedIntermediateSize, seqLen);
                                }
                                System.Numerics.Tensors.TensorPrimitives.Add(
                                    new ReadOnlySpan<float>(sdBatch, seqLen * hiddenSize),
                                    tmp.AsSpan(0, seqLen * hiddenSize),
                                    new Span<float>(sdBatch, seqLen * hiddenSize));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(tmp);
                            }
                        }
                    }

                    // Per-token sigmoid gate logit — same scalar loop as the
                    // original kernel so the resulting scale is bit-identical.
                    // Only meaningful for numSharedExperts==1 (Qwen1.5-MoE);
                    // DeepSeek (numSharedExperts>=1 with no gate) sets scale=1.
                    for (int t = 0; t < seqLen; t++)
                    {
                        float scale = 1.0f;
                        if (hasSharedGate)
                        {
                            float* x = hiddenPtr + t * hiddenSize;
                            float logit = 0f;
                            for (int j = 0; j < hiddenSize; j++)
                                logit += sharedGatePtr[j] * x[j];
                            scale = 1.0f / (1.0f + MathF.Exp(-logit));
                        }
                        sharedScaleBuf![t] = scale;
                    }
                }
            }

            // ────────────── 5) Final in-order accumulation ──────────────
            // For each token: acc = 0; for slot 0..k-1: acc += w * down; then
            // if shared: acc += sharedScale * sharedDown. This is the exact
            // sequence the scalar kernel ran, so every TensorPrimitives
            // .MultiplyAdd fires on the same operands in the same order.
            var acc = accBuf.AsSpan(0, hiddenSize);
            fixed (float* outPtr = output)
            fixed (float* scatterDownPtr = scatterDownBuf)
            {
                if (hasSharedExpert)
                {
                    fixed (float* sdPtr = sharedDownBatchBuf)
                    {
                        FinalAccumulate(
                            scatterDownPtr, sdPtr, sharedScaleBuf!,
                            assignWeightBuf, outPtr, acc,
                            seqLen, numExpertsPerTok, hiddenSize, hasShared: true);
                    }
                }
                else
                {
                    FinalAccumulate(
                        scatterDownPtr, null, null,
                        assignWeightBuf, outPtr, acc,
                        seqLen, numExpertsPerTok, hiddenSize, hasShared: false);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gateLogitsBuf);
            ArrayPool<float>.Shared.Return(routingBuf);
            ArrayPool<int>.Shared.Return(assignExpertBuf);
            ArrayPool<float>.Shared.Return(assignWeightBuf);
            ArrayPool<int>.Shared.Return(expertCountBuf);
            ArrayPool<int>.Shared.Return(expertTokenBuf);
            ArrayPool<int>.Shared.Return(expertSlotBuf);
            ArrayPool<float>.Shared.Return(batchInBuf);
            ArrayPool<float>.Shared.Return(gateBatchBuf);
            ArrayPool<float>.Shared.Return(upBatchBuf);
            ArrayPool<float>.Shared.Return(siluBatchBuf);
            ArrayPool<float>.Shared.Return(downBatchBuf);
            ArrayPool<float>.Shared.Return(scatterDownBuf);
            if (sharedGateBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedGateBatchBuf);
            if (sharedUpBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedUpBatchBuf);
            if (sharedSiluBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedSiluBatchBuf);
            if (sharedDownBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedDownBatchBuf);
            if (sharedScaleBuf is not null) ArrayPool<float>.Shared.Return(sharedScaleBuf);
            ArrayPool<float>.Shared.Return(accBuf);
        }
    }

    private static string ExpertProjectionName(int expert, string projection)
        => $"mlp.experts.{expert}.{projection}";

    private static void ApplyLoraDelta(
        ILoraAdapter? adapter,
        int layer,
        string projection,
        float* input,
        float* output,
        int seqLen,
        int inputDim,
        int outputDim)
    {
        if (adapter is null || layer < 0) return;
        var lora = adapter.GetLayerWeights(layer, projection);
        if (lora is not { } w) return;
        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' layer={layer} proj='{projection}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match MoE projection "
                + $"({inputDim}x{outputDim}).");

        float scale = adapter.Alpha / adapter.Rank;
        LoraDelta.Apply(input, (void*)w.BHandle, (void*)w.AHandle, output,
            seqLen, inputDim, outputDim, adapter.Rank, scale,
            w.WeightDType, w.WeightDType);
    }

    /// <summary>
    /// Per-token final accumulation: walks slots 0..k-1 in order and
    /// (optionally) applies the shared-expert contribution, matching the
    /// scalar kernel's accumulation sequence exactly for bit-identity.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FinalAccumulate(
        float* scatterDownPtr,
        float* sharedDownPtr,
        float[]? sharedScale,
        float[] assignWeight,
        float* outPtr,
        Span<float> acc,
        int seqLen,
        int numExpertsPerTok,
        int hiddenSize,
        bool hasShared)
    {
        for (int t = 0; t < seqLen; t++)
        {
            acc.Clear();
            for (int slot = 0; slot < numExpertsPerTok; slot++)
            {
                int assignIdx = t * numExpertsPerTok + slot;
                float w = assignWeight[assignIdx];
                if (w == 0f) continue;

                var downRow = new ReadOnlySpan<float>(
                    scatterDownPtr + (long)assignIdx * hiddenSize, hiddenSize);
                TensorPrimitives.MultiplyAdd(downRow, w, acc, acc);
            }

            if (hasShared)
            {
                float scale = sharedScale![t];
                var sdRow = new ReadOnlySpan<float>(
                    sharedDownPtr + (long)t * hiddenSize, hiddenSize);
                TensorPrimitives.MultiplyAdd(sdRow, scale, acc, acc);
            }

            acc.CopyTo(new Span<float>(outPtr + (long)t * hiddenSize, hiddenSize));
        }
    }

    /// <summary>
    /// Selects the top-k largest entries of <paramref name="probs"/> in
    /// descending order. Writes indices and probabilities into
    /// <paramref name="topkIdx"/> / <paramref name="topkProb"/>.
    /// Stable on ties: the lower original index wins, matching <c>torch.topk</c>'s
    /// forward-order CPU behaviour.
    /// </summary>
    [SkipLocalsInit]
    internal static void SelectTopK(
        ReadOnlySpan<float> probs, Span<int> topkIdx, Span<float> topkProb)
    {
        int k = topkIdx.Length;
        int n = probs.Length;

        // Repeated max-scan with masking by "already picked". For small k and
        // n (Mixtral-style 8..64 experts with k=2..4) this is faster and
        // allocation-free vs sorting.
        for (int slot = 0; slot < k; slot++)
        {
            int bestIdx = -1;
            float bestVal = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                // Skip indices already claimed — linear scan over k is fine.
                bool claimed = false;
                for (int p = 0; p < slot; p++)
                    if (topkIdx[p] == i) { claimed = true; break; }
                if (claimed) continue;

                float v = probs[i];
                // Strict > ensures lower index wins on ties (stable).
                if (v > bestVal)
                {
                    bestVal = v;
                    bestIdx = i;
                }
            }
            topkIdx[slot] = bestIdx;
            topkProb[slot] = bestVal;
        }
    }
}
