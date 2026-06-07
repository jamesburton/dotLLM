using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Threading;

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
/// <b>Two-phase routed path.</b> <see cref="Route"/> performs routing and
/// bucketing only — no expert GEMMs run yet. <see cref="ExecuteRoutedFromAssignments"/>
/// consumes those buckets and, for each <i>actually used</i> expert, runs a
/// quantized-direct or F32 SwiGLU GEMM. This avoids the per-forward dequant
/// of every routed expert (the dominant cost on Qwen3.6-A3B-style models
/// with 256 routed experts and top-k=8).
/// </para>
/// <para>
/// <b>Weight layout.</b> Routed expert weights are passed either as a single
/// strided base pointer + per-expert byte stride (production GGUF mmap path —
/// the format llama.cpp / GGUF uses for fused <c>ffn_*_exps</c> tensors) or as
/// a discontiguous <c>nint[]</c> of per-expert F32 row-major pointers
/// (synthetic test fixtures). The router <c>gateWeights</c> is row-major F32
/// <c>[numExperts, hiddenSize]</c>.
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
    /// Phase 1 of grouped MoE: routing + bucketing only. Computes top-k expert IDs per token,
    /// renormalises probabilities (when <paramref name="normTopKProb"/> is set), builds the
    /// per-expert occupancy histogram and (token, slot) bucket lists. No GEMMs run yet.
    /// Subsequent call sites use the returned assignments to dequant or directly-matmul only
    /// the touched experts.
    /// </summary>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="gateWeights">F32 router weight [numExperts × hiddenSize] row-major.</param>
    /// <param name="assignExpert">OUT: per-(token,slot) expert id [seqLen × numExpertsPerTok].</param>
    /// <param name="assignWeight">OUT: per-(token,slot) probability [seqLen × numExpertsPerTok].</param>
    /// <param name="bucketCursors">
    /// OUT: exclusive-scan offsets [numExperts+1] into <paramref name="bucketTokens"/> /
    /// <paramref name="bucketSlots"/>. Expert e's bucket lives in indices
    /// [bucketCursors[e], bucketCursors[e+1]).
    /// </param>
    /// <param name="bucketTokens">OUT: per-bucket-entry token index [seqLen × numExpertsPerTok].</param>
    /// <param name="bucketSlots">OUT: per-bucket-entry top-k slot index [seqLen × numExpertsPerTok].</param>
    /// <param name="uniqueExperts">
    /// OUT: list of distinct expert ids that received at least one token, in ascending order
    /// [≤ seqLen × numExpertsPerTok]. Caller-provided buffer; only the leading prefix is
    /// written. Use the returned count to bound iteration.
    /// </param>
    /// <param name="numExperts">Total expert count per layer (E).</param>
    /// <param name="numExpertsPerTok">Top-k: number of experts activated per token.</param>
    /// <param name="hiddenSize">Hidden / residual dimension (H).</param>
    /// <param name="seqLen">Number of tokens in this batch (T).</param>
    /// <param name="normTopKProb">When true, renormalise the selected top-k probabilities to sum to 1.0.</param>
    /// <returns>Number of unique experts actually used (the valid prefix length of <paramref name="uniqueExperts"/>).</returns>
    public static int Route(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> gateWeights,
        Span<int> assignExpert,
        Span<float> assignWeight,
        Span<int> bucketCursors,
        Span<int> bucketTokens,
        Span<int> bucketSlots,
        Span<int> uniqueExperts,
        int numExperts, int numExpertsPerTok,
        int hiddenSize, int seqLen,
        bool normTopKProb)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (numExpertsPerTok <= 0 || numExpertsPerTok > numExperts)
            throw new ArgumentOutOfRangeException(nameof(numExpertsPerTok));

        int totalAssignments = seqLen * numExpertsPerTok;
        if (assignExpert.Length < totalAssignments) throw new ArgumentException("assignExpert too small", nameof(assignExpert));
        if (assignWeight.Length < totalAssignments) throw new ArgumentException("assignWeight too small", nameof(assignWeight));
        if (bucketCursors.Length < numExperts + 1) throw new ArgumentException("bucketCursors too small", nameof(bucketCursors));
        if (bucketTokens.Length < totalAssignments) throw new ArgumentException("bucketTokens too small", nameof(bucketTokens));
        if (bucketSlots.Length < totalAssignments) throw new ArgumentException("bucketSlots too small", nameof(bucketSlots));

        // Rent gate-logit / softmax scratch — small, per-call, allocation-free via ArrayPool.
        float[] gateLogitsBuf = ArrayPool<float>.Shared.Rent(seqLen * numExperts);
        float[] routingBuf = ArrayPool<float>.Shared.Rent(seqLen * numExperts);

        try
        {
            bucketCursors.Slice(0, numExperts + 1).Clear();

            Span<int> topkIdx = stackalloc int[numExpertsPerTok];
            Span<float> topkProb = stackalloc float[numExpertsPerTok];

            // ── 1) Gate-logit GEMV per token ──────────────────────────────────
            // Keep per-token GEMV (not batched) to preserve scalar reduction order
            // of the historical kernel — softmax/top-k decisions then bit-match.
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

            // ── 2) Per-token softmax + top-k + (optional) renormalise ─────────
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

                for (int slot = 0; slot < numExpertsPerTok; slot++)
                {
                    int e = topkIdx[slot];
                    int assignIdx = t * numExpertsPerTok + slot;
                    assignExpert[assignIdx] = e;
                    assignWeight[assignIdx] = topkProb[slot];
                    bucketCursors[e]++; // histogram pass
                }
            }

            // ── 3) Exclusive prefix sum → bucket offsets ──────────────────────
            int running = 0;
            for (int e = 0; e <= numExperts; e++)
            {
                int c = bucketCursors[e];
                bucketCursors[e] = running;
                running += c;
            }

            // ── 4) Fill buckets with (token, slot) using per-expert write cursors ──
            int[]? cursorRented = null;
            Span<int> cursor = numExperts <= 256
                ? stackalloc int[numExperts]
                : (cursorRented = ArrayPool<int>.Shared.Rent(numExperts)).AsSpan(0, numExperts);
            for (int e = 0; e < numExperts; e++) cursor[e] = bucketCursors[e];

            try
            {
                for (int t = 0; t < seqLen; t++)
                {
                    for (int slot = 0; slot < numExpertsPerTok; slot++)
                    {
                        int assignIdx = t * numExpertsPerTok + slot;
                        int e = assignExpert[assignIdx];
                        int pos = cursor[e]++;
                        bucketTokens[pos] = t;
                        bucketSlots[pos] = slot;
                    }
                }
            }
            finally
            {
                if (cursorRented is not null) ArrayPool<int>.Shared.Return(cursorRented);
            }

            // ── 5) Materialise the ordered unique-expert list ─────────────────
            // Walk the histogram once; ascending order matches the existing
            // per-expert iteration order so accumulation stays bit-identical.
            int uniqueCount = 0;
            for (int e = 0; e < numExperts; e++)
            {
                int start = bucketCursors[e];
                int end = bucketCursors[e + 1];
                if (end - start > 0)
                {
                    if (uniqueCount < uniqueExperts.Length)
                        uniqueExperts[uniqueCount] = e;
                    uniqueCount++;
                }
            }
            return uniqueCount;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gateLogitsBuf);
            ArrayPool<float>.Shared.Return(routingBuf);
        }
    }

    /// <summary>
    /// Phase 2 of grouped MoE: per-expert SwiGLU on the buckets produced by <see cref="Route"/>.
    /// Supports quantized weight inputs natively — when <paramref name="gateExpsQt"/> /
    /// <paramref name="upExpsQt"/> / <paramref name="downExpsQt"/> are quant types, calls the
    /// matching <c>MatMul.GemmQX</c> kernel against the raw GGUF-fused-expert tensor
    /// (base pointer + per-expert byte stride). When the quant type is F32 and a per-expert
    /// F32 pointer array is supplied (<paramref name="gateExpsF32Ptrs"/> et al.), uses those
    /// discontiguous pointers and the standard <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>
    /// path — this is the synthetic-weights test fixture path.
    /// </summary>
    /// <remarks>
    /// Per-expert GEMMs scatter into disjoint <c>(token, slot)</c> rows of an internal
    /// scratch table; the final accumulation in <see cref="FinalAccumulate"/> walks
    /// <c>(t, slot=0..k-1)</c> in fixed order, preserving bit-identity with the historical
    /// kernel. When <paramref name="threadPool"/> is non-null, the outer per-expert loop is
    /// parallelised — inner GEMMs run single-threaded to avoid oversubscription.
    /// </remarks>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="gateExpsRawBase">Base pointer of the fused gate_exps tensor (M*K elements per expert).</param>
    /// <param name="gateExpsQt">Storage type of <paramref name="gateExpsRawBase"/>.</param>
    /// <param name="gateExpsRowBytes">Byte stride between consecutive experts in <paramref name="gateExpsRawBase"/>.</param>
    /// <param name="gateExpsF32Ptrs">
    /// Optional per-expert F32 pointer array; consulted only when <paramref name="gateExpsQt"/> is F32.
    /// Empty → use <paramref name="gateExpsRawBase"/> + stride.
    /// </param>
    /// <param name="upExpsRawBase">Base pointer of the fused up_exps tensor.</param>
    /// <param name="upExpsQt">Storage type of <paramref name="upExpsRawBase"/>.</param>
    /// <param name="upExpsRowBytes">Byte stride between consecutive experts in <paramref name="upExpsRawBase"/>.</param>
    /// <param name="upExpsF32Ptrs">Optional per-expert F32 pointer array; see <paramref name="gateExpsF32Ptrs"/>.</param>
    /// <param name="downExpsRawBase">Base pointer of the fused down_exps tensor.</param>
    /// <param name="downExpsQt">Storage type of <paramref name="downExpsRawBase"/>.</param>
    /// <param name="downExpsRowBytes">Byte stride between consecutive experts in <paramref name="downExpsRawBase"/>.</param>
    /// <param name="downExpsF32Ptrs">Optional per-expert F32 pointer array; see <paramref name="gateExpsF32Ptrs"/>.</param>
    /// <param name="assignExpert">Routing output: per-(token,slot) expert id.</param>
    /// <param name="assignWeight">Routing output: per-(token,slot) probability.</param>
    /// <param name="bucketCursors">Routing output: per-expert exclusive-scan offsets [numExperts+1].</param>
    /// <param name="bucketTokens">Routing output: per-bucket-entry token index.</param>
    /// <param name="bucketSlots">Routing output: per-bucket-entry top-k slot index.</param>
    /// <param name="uniqueExperts">Routing output: ordered list of touched expert ids.</param>
    /// <param name="uniqueExpertCount">Valid prefix length of <paramref name="uniqueExperts"/>.</param>
    /// <param name="output">F32 output activations [seqLen × hiddenSize]. Fully overwritten.</param>
    /// <param name="numExperts">Total expert count per layer (E).</param>
    /// <param name="numExpertsPerTok">Top-k: number of experts activated per token.</param>
    /// <param name="hiddenSize">Hidden / residual dimension (H).</param>
    /// <param name="intermediateSize">Per-expert MLP intermediate dimension (I).</param>
    /// <param name="seqLen">Number of tokens in this batch (T).</param>
    /// <param name="sharedGateProj">Per-shared-expert gate_proj pointers (F32 only here; shared-expert quant overlay is rec #3 from the perf analysis).</param>
    /// <param name="sharedUpProj">Per-shared-expert up_proj pointers.</param>
    /// <param name="sharedDownProj">Per-shared-expert down_proj pointers.</param>
    /// <param name="sharedIntermediateSize">Per-shared-expert intermediate width (0 to disable).</param>
    /// <param name="sharedExpertGate">Optional F32 [hiddenSize] sigmoid-gate weight.</param>
    /// <param name="loraAdapter">Optional active LoRA adapter for routed per-expert projection deltas.</param>
    /// <param name="loraLayer">Layer index used to resolve adapter weights.</param>
    /// <param name="threadPool">Optional thread pool — when non-null the outer per-expert loop is parallelised.</param>
    [SkipLocalsInit]
    public static void ExecuteRoutedFromAssignments(
        ReadOnlySpan<float> hidden,
        nint gateExpsRawBase, QuantizationType gateExpsQt, long gateExpsRowBytes, ReadOnlySpan<nint> gateExpsF32Ptrs,
        nint upExpsRawBase, QuantizationType upExpsQt, long upExpsRowBytes, ReadOnlySpan<nint> upExpsF32Ptrs,
        nint downExpsRawBase, QuantizationType downExpsQt, long downExpsRowBytes, ReadOnlySpan<nint> downExpsF32Ptrs,
        ReadOnlySpan<int> assignExpert, ReadOnlySpan<float> assignWeight,
        ReadOnlySpan<int> bucketCursors, ReadOnlySpan<int> bucketTokens, ReadOnlySpan<int> bucketSlots,
        ReadOnlySpan<int> uniqueExperts, int uniqueExpertCount,
        Span<float> output,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize, int seqLen,
        ReadOnlySpan<nint> sharedGateProj, ReadOnlySpan<nint> sharedUpProj, ReadOnlySpan<nint> sharedDownProj,
        int sharedIntermediateSize, ReadOnlySpan<float> sharedExpertGate,
        ILoraAdapter? loraAdapter, int loraLayer,
        ComputeThreadPool? threadPool = null)
    {
        if (hidden.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("hidden too small", nameof(hidden));
        if (output.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("output too small", nameof(output));
        if (sharedGateProj.Length != sharedUpProj.Length || sharedGateProj.Length != sharedDownProj.Length)
            throw new ArgumentException("Shared-expert weight spans must all have the same length.");

        if (seqLen == 0) return;

        int numSharedExperts = sharedGateProj.Length;
        bool hasSharedExpert = sharedIntermediateSize > 0 && numSharedExperts > 0;
        bool hasSharedGate = hasSharedExpert && sharedExpertGate.Length >= hiddenSize;

        int totalAssignments = seqLen * numExpertsPerTok;

        // ────────────── Pooled scratch buffers ──────────────
        // Per-(token,slot) down-projected outputs, keyed by t*k + slot. Final
        // accumulation indexes from this table.
        float[] scatterDownBuf = ArrayPool<float>.Shared.Rent(totalAssignments * hiddenSize);

        // Shared-expert scratch.
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

        float[] accBuf = ArrayPool<float>.Shared.Rent(hiddenSize);

        try
        {
            // ────────────── Per-expert batched SwiGLU (only TOUCHED experts) ──────────────
            // Outer loop walks uniqueExperts in ascending id order. Per-expert state
            // (gather/gate/up/silu/down scratch + optional Q8 input scratch) is owned by
            // the worker — when parallelised each thread rents its own. Per-expert writes
            // hit disjoint (token, slot) rows in scatterDown by construction.
            fixed (float* hiddenPtr = hidden)
            fixed (float* scatterDownPtr = scatterDownBuf)
            {
                // Snapshot pointers + sizes the worker captures.
                var workerCtx = new RoutedExpertWorkerCtx
                {
                    HiddenPtr = hiddenPtr,
                    ScatterDownPtr = scatterDownPtr,

                    GateExpsRawBase = gateExpsRawBase,
                    GateExpsQt = gateExpsQt,
                    GateExpsRowBytes = gateExpsRowBytes,
                    UpExpsRawBase = upExpsRawBase,
                    UpExpsQt = upExpsQt,
                    UpExpsRowBytes = upExpsRowBytes,
                    DownExpsRawBase = downExpsRawBase,
                    DownExpsQt = downExpsQt,
                    DownExpsRowBytes = downExpsRowBytes,

                    SeqLen = seqLen,
                    HiddenSize = hiddenSize,
                    IntermediateSize = intermediateSize,
                    NumExpertsPerTok = numExpertsPerTok,
                };

                if (threadPool is null || uniqueExpertCount < 2)
                {
                    // Serial path — bit-identity with the historical ascending-id loop.
                    for (int u = 0; u < uniqueExpertCount; u++)
                    {
                        int e = uniqueExperts[u];
                        ProcessRoutedExpert(
                            workerCtx, e,
                            bucketCursors, bucketTokens, bucketSlots,
                            gateExpsF32Ptrs, upExpsF32Ptrs, downExpsF32Ptrs,
                            loraAdapter, loraLayer, threadPool: null);
                    }
                }
                else
                {
                    // Outer Parallel.For over touched experts. Inner GEMMs pass pool=null
                    // to avoid oversubscription. The per-(t,slot) scatter targets are
                    // disjoint by construction, so no synchronisation is needed.
                    // Span<T> can't be captured by a closure, so copy the routing arrays
                    // into managed arrays the lambda can close over.
                    int[] bucketCursorsArr = bucketCursors.ToArray();
                    int[] bucketTokensArr = bucketTokens.ToArray();
                    int[] bucketSlotsArr = bucketSlots.ToArray();
                    int[] uniqueExpertsArr = uniqueExperts.Slice(0, uniqueExpertCount).ToArray();
                    nint[] gateF32Arr = gateExpsF32Ptrs.ToArray();
                    nint[] upF32Arr = upExpsF32Ptrs.ToArray();
                    nint[] downF32Arr = downExpsF32Ptrs.ToArray();
                    var ctxCopy = workerCtx;

                    System.Threading.Tasks.Parallel.For(0, uniqueExpertCount, u =>
                    {
                        int e = uniqueExpertsArr[u];
                        // Each Parallel.For iteration is independent — disjoint scatter rows.
                        ProcessRoutedExpert(
                            ctxCopy, e,
                            bucketCursorsArr, bucketTokensArr, bucketSlotsArr,
                            gateF32Arr, upF32Arr, downF32Arr,
                            loraAdapter, loraLayer, threadPool: null);
                    });
                }
            }

            // ────────────── Shared-expert batched compute (F32 only, unchanged) ──────────────
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
                            MatMul.GemmF32(sharedW2k, ssBatch, sdBatch,
                                hiddenSize, sharedIntermediateSize, seqLen);
                        }
                        else
                        {
                            float[] tmp = ArrayPool<float>.Shared.Rent(seqLen * hiddenSize);
                            try
                            {
                                fixed (float* tmpPtr = tmp)
                                {
                                    MatMul.GemmF32(sharedW2k, ssBatch, tmpPtr,
                                        hiddenSize, sharedIntermediateSize, seqLen);
                                }
                                TensorPrimitives.Add(
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

            // ────────────── Final in-order accumulation ──────────────
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
                            assignWeight, outPtr, acc,
                            seqLen, numExpertsPerTok, hiddenSize, hasShared: true);
                    }
                }
                else
                {
                    FinalAccumulate(
                        scatterDownPtr, null, null,
                        assignWeight, outPtr, acc,
                        seqLen, numExpertsPerTok, hiddenSize, hasShared: false);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scatterDownBuf);
            if (sharedGateBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedGateBatchBuf);
            if (sharedUpBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedUpBatchBuf);
            if (sharedSiluBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedSiluBatchBuf);
            if (sharedDownBatchBuf is not null) ArrayPool<float>.Shared.Return(sharedDownBatchBuf);
            if (sharedScaleBuf is not null) ArrayPool<float>.Shared.Return(sharedScaleBuf);
            ArrayPool<float>.Shared.Return(accBuf);
        }
    }

    /// <summary>
    /// Per-expert worker context — captured as a value type by the per-expert loop body
    /// so it can be passed to both the serial loop and the <see cref="System.Threading.Tasks.Parallel.For(int, int, Action{int})"/>
    /// closure without boxing.
    /// </summary>
    private struct RoutedExpertWorkerCtx
    {
        public float* HiddenPtr;
        public float* ScatterDownPtr;

        public nint GateExpsRawBase;
        public QuantizationType GateExpsQt;
        public long GateExpsRowBytes;
        public nint UpExpsRawBase;
        public QuantizationType UpExpsQt;
        public long UpExpsRowBytes;
        public nint DownExpsRawBase;
        public QuantizationType DownExpsQt;
        public long DownExpsRowBytes;

        public int SeqLen;
        public int HiddenSize;
        public int IntermediateSize;
        public int NumExpertsPerTok;
    }

    /// <summary>
    /// Process one routed expert: gather assigned token rows, run gate/up/down GEMMs
    /// (quantized-direct when possible, F32 fallback otherwise), apply SwiGLU, and scatter
    /// the per-(token,slot) down output into the shared scatter table. Pure function in
    /// the bucket arrays — safe to invoke concurrently across distinct expert ids.
    /// </summary>
    [SkipLocalsInit]
    private static void ProcessRoutedExpert(
        RoutedExpertWorkerCtx ctx, int e,
        ReadOnlySpan<int> bucketCursors, ReadOnlySpan<int> bucketTokens, ReadOnlySpan<int> bucketSlots,
        ReadOnlySpan<nint> gateExpsF32Ptrs, ReadOnlySpan<nint> upExpsF32Ptrs, ReadOnlySpan<nint> downExpsF32Ptrs,
        ILoraAdapter? loraAdapter, int loraLayer, ComputeThreadPool? threadPool)
    {
        int start = bucketCursors[e];
        int end = bucketCursors[e + 1];
        int batch = end - start;
        if (batch == 0) return;

        int hiddenSize = ctx.HiddenSize;
        int intermediateSize = ctx.IntermediateSize;
        int numExpertsPerTok = ctx.NumExpertsPerTok;

        // Rent per-call buffers from ArrayPool — worker-local, returned in finally.
        float[] batchInBuf = ArrayPool<float>.Shared.Rent(batch * hiddenSize);
        float[] gateBatchBuf = ArrayPool<float>.Shared.Rent(batch * intermediateSize);
        float[] upBatchBuf = ArrayPool<float>.Shared.Rent(batch * intermediateSize);
        float[] siluBatchBuf = ArrayPool<float>.Shared.Rent(batch * intermediateSize);
        float[] downBatchBuf = ArrayPool<float>.Shared.Rent(batch * hiddenSize);

        // Optional Q8 pre-quantized input scratch — sized for the largest pre-quant format
        // we might need across gate/up (k=hiddenSize) and down (k=intermediateSize).
        // Computed up-front so the pin lifetime covers all GEMMs that consume it.
        bool gateUpUsesF32Resolved = (ctx.GateExpsQt == QuantizationType.F32 && gateExpsF32Ptrs.Length > e)
                                  || (ctx.UpExpsQt == QuantizationType.F32 && upExpsF32Ptrs.Length > e);
        bool gateUpShareQt = !gateUpUsesF32Resolved && ctx.GateExpsQt == ctx.UpExpsQt;
        int prequantBytes = gateUpShareQt
            ? ComputePrequantBytes(ctx.GateExpsQt, hiddenSize, batch)
            : 0;
        // Always rent at least 1 byte so the fixed-pointer pin is well-defined; we only
        // populate when prequantBytes > 0.
        byte[] prequantInputBuf = ArrayPool<byte>.Shared.Rent(Math.Max(prequantBytes, 1));

        try
        {
            float* hiddenPtr = ctx.HiddenPtr;
            float* scatterDownPtr = ctx.ScatterDownPtr;

            fixed (float* batchInPtr = batchInBuf)
            fixed (float* gateBatchPtr = gateBatchBuf)
            fixed (float* upBatchPtr = upBatchBuf)
            fixed (float* siluBatchPtr = siluBatchBuf)
            fixed (float* downBatchPtr = downBatchBuf)
            fixed (byte* prequantPtr = prequantInputBuf)
            {
                // ── Gather assigned token rows into batchIn[0..batch, :] ──────
                for (int b = 0; b < batch; b++)
                {
                    int t = bucketTokens[start + b];
                    Buffer.MemoryCopy(
                        hiddenPtr + (long)t * hiddenSize,
                        batchInPtr + (long)b * hiddenSize,
                        hiddenSize * sizeof(float),
                        hiddenSize * sizeof(float));
                }

                // Resolve per-expert weight pointers. F32 uses explicit per-expert array;
                // quant types use base + e * row stride.
                bool gateUsesF32 = ctx.GateExpsQt == QuantizationType.F32 && gateExpsF32Ptrs.Length > e;
                bool upUsesF32 = ctx.UpExpsQt == QuantizationType.F32 && upExpsF32Ptrs.Length > e;
                bool downUsesF32 = ctx.DownExpsQt == QuantizationType.F32 && downExpsF32Ptrs.Length > e;

                nint gatePtr = gateUsesF32
                    ? gateExpsF32Ptrs[e]
                    : ctx.GateExpsRawBase + (nint)(e * ctx.GateExpsRowBytes);
                nint upPtr = upUsesF32
                    ? upExpsF32Ptrs[e]
                    : ctx.UpExpsRawBase + (nint)(e * ctx.UpExpsRowBytes);
                nint downPtr = downUsesF32
                    ? downExpsF32Ptrs[e]
                    : ctx.DownExpsRawBase + (nint)(e * ctx.DownExpsRowBytes);

                // Pre-quantize once per K dimension when gate and up share the same
                // K dim (hiddenSize) and Q8 staging format. Stays pinned via the
                // outer fixed block above so the GEMM kernels can dereference it.
                byte* preQuantGateUp = null;
                if (prequantBytes > 0)
                {
                    QuantizeBatchForWeightQt(ctx.GateExpsQt, batchInPtr, prequantPtr, hiddenSize, batch);
                    preQuantGateUp = prequantPtr;
                }

                // ── gate = batchIn × W1 (gate_proj) ───────────────────────────
                DispatchGemm(
                    ctx.GateExpsQt, gatePtr, batchInPtr, gateBatchPtr,
                    intermediateSize, hiddenSize, batch,
                    preQuantGateUp, threadPool);

                // ── up = batchIn × W3 (up_proj) ───────────────────────────────
                DispatchGemm(
                    ctx.UpExpsQt, upPtr, batchInPtr, upBatchPtr,
                    intermediateSize, hiddenSize, batch,
                    preQuantGateUp, threadPool);

                ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "gate_proj"),
                    batchInPtr, gateBatchPtr, batch, hiddenSize, intermediateSize);
                ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "up_proj"),
                    batchInPtr, upBatchPtr, batch, hiddenSize, intermediateSize);

                // ── Per-row SwiGLU fuse ──────────────────────────────────────
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

                // ── down = silu × W2 (down_proj) ─────────────────────────────
                // K=intermediateSize differs from gate/up (K=hiddenSize) so the pre-quantized
                // input is not reusable here — let the kernel quantize internally. This is one
                // tiled QuantizeF32ToQ8_K over (batch × intermediate), small next to the matmul.
                DispatchGemm(
                    ctx.DownExpsQt, downPtr, siluBatchPtr, downBatchPtr,
                    hiddenSize, intermediateSize, batch,
                    preQuantizedInput: null, threadPool);

                ApplyLoraDelta(loraAdapter, loraLayer, ExpertProjectionName(e, "down_proj"),
                    siluBatchPtr, downBatchPtr, batch, intermediateSize, hiddenSize);

                // ── Scatter down rows to scatterDown[(t*k+slot)*H] ───────────
                for (int b = 0; b < batch; b++)
                {
                    int t = bucketTokens[start + b];
                    int slot = bucketSlots[start + b];
                    Buffer.MemoryCopy(
                        downBatchPtr + (long)b * hiddenSize,
                        scatterDownPtr + (long)(t * numExpertsPerTok + slot) * hiddenSize,
                        hiddenSize * sizeof(float),
                        hiddenSize * sizeof(float));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(batchInBuf);
            ArrayPool<float>.Shared.Return(gateBatchBuf);
            ArrayPool<float>.Shared.Return(upBatchBuf);
            ArrayPool<float>.Shared.Return(siluBatchBuf);
            ArrayPool<float>.Shared.Return(downBatchBuf);
            ArrayPool<byte>.Shared.Return(prequantInputBuf);
        }
    }

    /// <summary>
    /// Total byte size of a pre-quantized batch of <paramref name="batch"/> rows
    /// of <paramref name="k"/> elements for the Q8 staging type that pairs with
    /// the given weight quant type. Returns 0 when the weight quant type has no
    /// pre-quant fast path (F32 / F16 / IQ families fall through GemmDequantFallback).
    /// </summary>
    private static int ComputePrequantBytes(QuantizationType wQt, int k, int batch)
    {
        switch (wQt)
        {
            case QuantizationType.Q8_0:
                // Q8_0 weight × Q8_0 input: 34 bytes per 32-element block.
                if (k % 32 != 0) return 0;
                return batch * (k / 32) * 34;
            case QuantizationType.Q5_0:
                // Q5_0 weight × Q8_1 input: 36 bytes per 32-element block.
                if (k % 32 != 0) return 0;
                return batch * (k / 32) * 36;
            case QuantizationType.Q4_K:
            case QuantizationType.Q5_K:
            case QuantizationType.Q6_K:
                // K-quant weight × Q8_K input: 292 bytes per 256-element super-block.
                if (k % 256 != 0) return 0;
                return batch * (k / 256) * MatMul.Q8_K_BlockBytes;
            default:
                return 0;
        }
    }

    /// <summary>
    /// Quantize <paramref name="batch"/> input rows of <paramref name="k"/> F32 elements
    /// into the Q8 staging format matching <paramref name="wQt"/>, writing per-row Q8
    /// blocks back-to-back into <paramref name="dest"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void QuantizeBatchForWeightQt(QuantizationType wQt, float* src, byte* dest, int k, int batch)
    {
        switch (wQt)
        {
            case QuantizationType.Q8_0:
                {
                    int rowBytes = (k / 32) * 34;
                    for (int b = 0; b < batch; b++)
                        MatMul.QuantizeF32ToQ8_0(src + b * k, dest + b * rowBytes, k);
                    break;
                }
            case QuantizationType.Q5_0:
                {
                    int rowBytes = (k / 32) * 36;
                    for (int b = 0; b < batch; b++)
                        MatMul.QuantizeF32ToQ8_1(src + b * k, dest + b * rowBytes, k);
                    break;
                }
            case QuantizationType.Q4_K:
            case QuantizationType.Q5_K:
            case QuantizationType.Q6_K:
                {
                    int rowBytes = (k / 256) * MatMul.Q8_K_BlockBytes;
                    for (int b = 0; b < batch; b++)
                        MatMul.QuantizeF32ToQ8_K(src + b * k, dest + b * rowBytes, k);
                    break;
                }
        }
    }

    /// <summary>
    /// Dispatch one weight-quant-typed GEMM <c>C[N,M] = B[N,K] × A[M,K]^T</c>. Routes to
    /// the matching <c>MatMul.GemmQX</c> kernel for supported quant types, falls back to
    /// per-row dequant + <see cref="TensorPrimitives.Dot"/> for everything else.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DispatchGemm(
        QuantizationType qt, nint weights, float* b, float* c,
        int m, int k, int n, byte* preQuantizedInput, ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.F32:
                MatMul.GemmF32((float*)weights, b, c, m, k, n);
                return;
            case QuantizationType.F16:
                MatMul.GemmF16(weights, b, c, m, k, n);
                return;
            case QuantizationType.Q8_0:
                if (pool is not null)
                    MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                else
                    MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q5_0:
                if (pool is not null)
                    MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                else
                    MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q4_K:
                if (pool is not null)
                    MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                else
                    MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q5_K:
                if (pool is not null)
                    MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                else
                    MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q6_K:
                if (pool is not null)
                    MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                else
                    MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            default:
                // Fallback for quant types without a direct kernel (Q4_0, Q4_1, Q5_1, IQ*, BF16).
                // Dequant per row then F32 dot. Matches GemmDequantFallback in the model.
                GemmDequantFallback(weights, qt, b, c, m, k, n);
                return;
        }
    }

    /// <summary>
    /// Per-row dequant + F32 dot fallback for quant types without a direct GEMM kernel.
    /// </summary>
    private static void GemmDequantFallback(
        nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int t = 0; t < n; t++)
            {
                var xSpan = new ReadOnlySpan<float>(b + t * k, k);
                for (int i = 0; i < m; i++)
                {
                    Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                    c[t * m + i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// GroupedGEMM MoE core: routes once (via <see cref="Route"/>), buckets tokens per expert,
    /// then dispatches to <see cref="ExecuteRoutedFromAssignments"/> using a per-expert F32
    /// pointer array (the public <c>Execute</c> / <c>ExecuteWithSharedExpert</c> entry points
    /// always pass discontiguous F32 expert weights). Bit-identical to the historical
    /// scalar/grouped kernel.
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

        if (seqLen == 0) return;

        int totalAssignments = seqLen * numExpertsPerTok;

        // Phase 1: routing + bucketing — small scratch arrays rented per call.
        int[] assignExpertBuf = ArrayPool<int>.Shared.Rent(totalAssignments);
        float[] assignWeightBuf = ArrayPool<float>.Shared.Rent(totalAssignments);
        int[] bucketCursorsBuf = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] bucketTokensBuf = ArrayPool<int>.Shared.Rent(totalAssignments);
        int[] bucketSlotsBuf = ArrayPool<int>.Shared.Rent(totalAssignments);
        int[] uniqueExpertsBuf = ArrayPool<int>.Shared.Rent(Math.Min(numExperts, totalAssignments));

        try
        {
            int uniqueCount = Route(
                hidden, gateWeights,
                assignExpertBuf.AsSpan(0, totalAssignments),
                assignWeightBuf.AsSpan(0, totalAssignments),
                bucketCursorsBuf.AsSpan(0, numExperts + 1),
                bucketTokensBuf.AsSpan(0, totalAssignments),
                bucketSlotsBuf.AsSpan(0, totalAssignments),
                uniqueExpertsBuf.AsSpan(),
                numExperts, numExpertsPerTok, hiddenSize, seqLen,
                normTopKProb);

            // Phase 2: per-expert SwiGLU + accumulation. F32 per-expert pointer array
            // is passed via the F32 override params; the strided base is unused for F32.
            ExecuteRoutedFromAssignments(
                hidden,
                gateExpsRawBase: 0, QuantizationType.F32, gateExpsRowBytes: 0, expertsW1,
                upExpsRawBase: 0, QuantizationType.F32, upExpsRowBytes: 0, expertsW3,
                downExpsRawBase: 0, QuantizationType.F32, downExpsRowBytes: 0, expertsW2,
                assignExpertBuf.AsSpan(0, totalAssignments),
                assignWeightBuf.AsSpan(0, totalAssignments),
                bucketCursorsBuf.AsSpan(0, numExperts + 1),
                bucketTokensBuf.AsSpan(0, totalAssignments),
                bucketSlotsBuf.AsSpan(0, totalAssignments),
                uniqueExpertsBuf.AsSpan(0, Math.Min(uniqueCount, uniqueExpertsBuf.Length)),
                uniqueCount,
                output,
                numExperts, numExpertsPerTok, hiddenSize, intermediateSize, seqLen,
                sharedGateProj, sharedUpProj, sharedDownProj,
                sharedIntermediateSize, sharedExpertGate,
                loraAdapter, loraLayer,
                threadPool: null);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(assignExpertBuf);
            ArrayPool<float>.Shared.Return(assignWeightBuf);
            ArrayPool<int>.Shared.Return(bucketCursorsBuf);
            ArrayPool<int>.Shared.Return(bucketTokensBuf);
            ArrayPool<int>.Shared.Return(bucketSlotsBuf);
            ArrayPool<int>.Shared.Return(uniqueExpertsBuf);
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
        // Phase 4d.6 — opt into the outer-product stage-2 fast path when
        // available (rank=16 + AVX-512). EnsureATransposedF32 caches the
        // transposed-A on the adapter so subsequent calls hit the fast path
        // with no extra work; per-expert MoE projections each get their own
        // cached buffer, indexed by the synthetic projection name.
        nint aTransposedHandle = LoraStage2.EnsureATransposedF32(
            adapter as LoraAdapter, layer, projection, in w, adapter.Rank);
        LoraDelta.Apply(input, (void*)w.BHandle, (void*)w.AHandle, output,
            seqLen, inputDim, outputDim, adapter.Rank, scale,
            w.WeightDType, w.WeightDType, aTransposedHandle);
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
        ReadOnlySpan<float> assignWeight,
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
