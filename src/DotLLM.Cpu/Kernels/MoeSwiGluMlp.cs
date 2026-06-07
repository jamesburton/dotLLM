using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

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
/// <b>Scalar-first PoC.</b> The per-expert GEMV uses the single-threaded
/// F32 <c>MatMul.GemvF32</c> overload directly — no per-expert quantisation,
/// no fused GroupedGEMM. That is fine for validation; a fused kernel is a
/// follow-up when a real Mixtral-scale model is wired up.
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
        int seqLen)
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

        // Scratch buffers — rented from the pool so per-call allocations are free.
        // The per-token 'acc' buffer is the MoE output for that token; it
        // accumulates expert contributions without touching 'output' until the
        // end of the token, which keeps this kernel safe to call with
        // <c>hidden</c> and <c>output</c> aliasing.
        float[] gateLogitsBuf = ArrayPool<float>.Shared.Rent(numExperts);
        float[] routingBuf = ArrayPool<float>.Shared.Rent(numExperts);
        float[] gateBuf = ArrayPool<float>.Shared.Rent(intermediateSize);
        float[] upBuf = ArrayPool<float>.Shared.Rent(intermediateSize);
        float[] siluBuf = ArrayPool<float>.Shared.Rent(intermediateSize);
        float[] downBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        float[] accBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        Span<int> topkIdx = stackalloc int[numExpertsPerTok];
        Span<float> topkProb = stackalloc float[numExpertsPerTok];

        try
        {
            var gateLogits = gateLogitsBuf.AsSpan(0, numExperts);
            var routing = routingBuf.AsSpan(0, numExperts);
            var gate = gateBuf.AsSpan(0, intermediateSize);
            var up = upBuf.AsSpan(0, intermediateSize);
            var silu = siluBuf.AsSpan(0, intermediateSize);
            var down = downBuf.AsSpan(0, hiddenSize);
            var acc = accBuf.AsSpan(0, hiddenSize);

            fixed (float* hiddenPtr = hidden)
            fixed (float* gateWPtr = gateWeights)
            fixed (float* outPtr = output)
            fixed (float* gateBufPtr = gate)
            fixed (float* upBufPtr = up)
            fixed (float* siluBufPtr = silu)
            fixed (float* downBufPtr = down)
            fixed (float* logitsPtr = gateLogits)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = hiddenPtr + t * hiddenSize;
                    float* y = outPtr + t * hiddenSize;

                    // 1) Router: gate_logits[e] = gate.weight[e, :] . x
                    //    gate.weight is [E, H] row-major, so this is a plain GEMV.
                    MatMul.GemvF32(gateWPtr, x, logitsPtr, numExperts, hiddenSize);

                    // 2) Full softmax over E experts.
                    Softmax.Execute(gateLogits, routing);

                    // 3) Top-k selection: partial max-scan. numExperts is small
                    //    (8-64 in practice), so O(E*k) is fine and avoids a
                    //    temporary sort allocation.
                    SelectTopK(routing, topkIdx, topkProb);

                    // 4) Renormalise the top-k probabilities by sum (Mixtral
                    //    convention — NOT a second softmax).
                    float sum = 0f;
                    for (int i = 0; i < numExpertsPerTok; i++) sum += topkProb[i];
                    float invSum = sum > 0f ? 1.0f / sum : 0f;
                    for (int i = 0; i < numExpertsPerTok; i++) topkProb[i] *= invSum;

                    // 5) Accumulate weighted expert outputs into 'acc'. Starts
                    //    zeroed; aliasing 'hidden' with 'output' is safe because
                    //    we only write to 'output' at the end of each token,
                    //    after all reads from 'x' are complete.
                    acc.Clear();
                    for (int i = 0; i < numExpertsPerTok; i++)
                    {
                        int eIdx = topkIdx[i];
                        float w = topkProb[i];
                        if (w == 0f) continue;

                        float* w1 = (float*)expertsW1[eIdx];
                        float* w2 = (float*)expertsW2[eIdx];
                        float* w3 = (float*)expertsW3[eIdx];

                        // gate = w1 @ x   [I]
                        // up   = w3 @ x   [I]
                        MatMul.GemvF32(w1, x, gateBufPtr, intermediateSize, hiddenSize);
                        MatMul.GemvF32(w3, x, upBufPtr, intermediateSize, hiddenSize);

                        // silu = SwiGLU(gate, up) = sigmoid(gate) * gate * up
                        FusedOps.SwiGLU(gate, up, silu);

                        // down = w2 @ silu    [H]
                        MatMul.GemvF32(w2, siluBufPtr, downBufPtr, hiddenSize, intermediateSize);

                        // acc += w * down
                        TensorPrimitives.MultiplyAdd(down, w, acc, acc);
                    }

                    // 6) Write accumulated output for this token.
                    acc.CopyTo(new Span<float>(y, hiddenSize));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gateLogitsBuf);
            ArrayPool<float>.Shared.Return(routingBuf);
            ArrayPool<float>.Shared.Return(gateBuf);
            ArrayPool<float>.Shared.Return(upBuf);
            ArrayPool<float>.Shared.Return(siluBuf);
            ArrayPool<float>.Shared.Return(downBuf);
            ArrayPool<float>.Shared.Return(accBuf);
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
