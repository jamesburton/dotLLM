using System.Buffers;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BitNet-ternary (I2_S) Mixture-of-Experts FFN forward — the routed-expert body a
/// depth-expanded identity-MoTE / BitNet-MoE checkpoint needs. Differs from the SwiGLU
/// MoE path (<see cref="Execute"/>) in three numerically load-bearing ways:
/// <list type="number">
///   <item>Experts are <b>ternary I2_S</b>, dispatched through
///     <see cref="MatMul.MoeIndexedMatmulI2_S"/> (per-expert packed-trit banks + a
///     per-expert absmean α), not F32 SwiGLU GEMMs.</item>
///   <item>The gate non-linearity is <b>relu²</b> (<see cref="FusedOps.ReLU2GLU"/>), not SiLU.</item>
///   <item>A <b>per-expert BitNet FFN Sub-LN</b> (RMSNorm with <c>ffn_sub_norm.weight</c>) is
///     applied to the gated intermediate before <c>down_proj</c>.</item>
/// </list>
/// The full expert body is <c>down( ffn_sub_norm( relu²(gate(x)) * up(x) ) )</c>. Routing is
/// the shared <see cref="Route"/> (softmax → top-k → optional renormalise), extended with an
/// additive router bias.
///
/// <para><b>Skip expert.</b> An identity-MoTE skip expert is just an ordinary expert whose
/// <c>down_proj</c> is all-zero — it packs to all-zero trits and outputs exactly 0. No
/// special-casing here.</para>
///
/// <para><b>Extension point (Q4_K / Q8_0 experts).</b> The per-expert base+stride bank layout
/// and per-expert scale vectors are quant-type-agnostic; slotting in a Q4_K/Q8_0 expert format
/// means adding a <c>MoeIndexedMatmul{Q4_K,Q8_0}</c> kernel and dispatching on the layer's
/// routed-expert quant type at the three GEMM call sites below.</para>
/// </summary>
public static unsafe partial class MoeSwiGluMlp
{
    /// <summary>
    /// Executes the BitNet-ternary MoE FFN for <paramref name="seqLen"/> tokens.
    /// Reads <paramref name="hidden"/> [seqLen × hiddenSize], writes <paramref name="output"/>
    /// [seqLen × hiddenSize] (fully overwritten).
    /// </summary>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="gateWeights">F32 router weight [numExperts × hiddenSize] row-major.</param>
    /// <param name="gateBias">Optional F32 router bias [numExperts]; empty for none.</param>
    /// <param name="gateBank">Packed-trit <c>gate_proj</c> banks (payload only). Expert e at <c>gateBank + e*gateRowBytes</c>.</param>
    /// <param name="gateRowBytes">Byte stride between <c>gate_proj</c> expert banks (= intermediateSize·hiddenSize/4).</param>
    /// <param name="gateScales">Per-expert absmean α for <c>gate_proj</c> [numExperts].</param>
    /// <param name="upBank">Packed-trit <c>up_proj</c> banks.</param>
    /// <param name="upRowBytes">Byte stride between <c>up_proj</c> expert banks.</param>
    /// <param name="upScales">Per-expert absmean α for <c>up_proj</c> [numExperts].</param>
    /// <param name="downBank">Packed-trit <c>down_proj</c> banks.</param>
    /// <param name="downRowBytes">Byte stride between <c>down_proj</c> expert banks (= hiddenSize·intermediateSize/4).</param>
    /// <param name="downScales">Per-expert absmean α for <c>down_proj</c> [numExperts].</param>
    /// <param name="expertFfnSubNorm">Per-expert FFN Sub-LN weight [numExperts][intermediateSize].</param>
    /// <param name="output">F32 output activations [seqLen × hiddenSize]. Fully overwritten.</param>
    /// <param name="numExperts">Total expert count per layer (E).</param>
    /// <param name="numExpertsPerTok">Top-k: number of experts activated per token.</param>
    /// <param name="hiddenSize">Hidden / residual dimension (H); a multiple of 128.</param>
    /// <param name="intermediateSize">Per-expert MLP intermediate dimension (I); a multiple of 128.</param>
    /// <param name="seqLen">Number of tokens in this batch (T).</param>
    /// <param name="normTopKProb">Renormalise the selected top-k probabilities to sum to 1.0.</param>
    /// <param name="rmsEps">RMSNorm epsilon for the per-expert FFN Sub-LN.</param>
    /// <param name="threadPool">Optional thread pool — forwarded to the indexed GEMMs.</param>
    [SkipLocalsInit]
    public static void ExecuteBitNetMoe(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> gateWeights,
        ReadOnlySpan<float> gateBias,
        byte* gateBank, long gateRowBytes, ReadOnlySpan<float> gateScales,
        byte* upBank, long upRowBytes, ReadOnlySpan<float> upScales,
        byte* downBank, long downRowBytes, ReadOnlySpan<float> downScales,
        float[][] expertFfnSubNorm,
        Span<float> output,
        int numExperts, int numExpertsPerTok,
        int hiddenSize, int intermediateSize, int seqLen,
        bool normTopKProb, float rmsEps,
        ComputeThreadPool? threadPool = null)
    {
        if (hidden.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("hidden too small", nameof(hidden));
        if (output.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException("output too small", nameof(output));
        if (seqLen == 0) return;

        int n = seqLen * numExpertsPerTok;   // (token, slot) assignment rows

        // ── Routing scratch (shared SwiGLU router: softmax → top-k → renorm + bias) ──
        int[] assignExpertBuf = ArrayPool<int>.Shared.Rent(n);
        float[] assignWeightBuf = ArrayPool<float>.Shared.Rent(n);
        int[] bucketCursorsBuf = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] bucketTokensBuf = ArrayPool<int>.Shared.Rent(n);
        int[] bucketSlotsBuf = ArrayPool<int>.Shared.Rent(n);
        int[] uniqueExpertsBuf = ArrayPool<int>.Shared.Rent(n);

        // ── Per-assignment compute scratch ──
        float[] batchInBuf = ArrayPool<float>.Shared.Rent(n * hiddenSize);
        float[] gateOutBuf = ArrayPool<float>.Shared.Rent(n * intermediateSize);
        float[] upOutBuf = ArrayPool<float>.Shared.Rent(n * intermediateSize);
        float[] interBuf = ArrayPool<float>.Shared.Rent(n * intermediateSize);
        float[] downOutBuf = ArrayPool<float>.Shared.Rent(n * hiddenSize);
        int[] rowExpertIdsBuf = ArrayPool<int>.Shared.Rent(n);

        try
        {
            var assignExpert = assignExpertBuf.AsSpan(0, n);
            var assignWeight = assignWeightBuf.AsSpan(0, n);

            Route(
                hidden, gateWeights,
                assignExpert, assignWeight,
                bucketCursorsBuf.AsSpan(0, numExperts + 1),
                bucketTokensBuf.AsSpan(0, n),
                bucketSlotsBuf.AsSpan(0, n),
                uniqueExpertsBuf.AsSpan(0, n),
                numExperts, numExpertsPerTok,
                hiddenSize, seqLen,
                normTopKProb,
                gateBias);

            // Gather per-assignment input row (all top-k slots of token t share hidden[t])
            // and the routed expert id. assignExpert is laid out [t*k + slot] == row index.
            var rowExpertIds = rowExpertIdsBuf.AsSpan(0, n);
            fixed (float* hiddenPtr = hidden)
            fixed (float* batchInPtr = batchInBuf)
            {
                for (int a = 0; a < n; a++)
                {
                    int t = a / numExpertsPerTok;
                    rowExpertIds[a] = assignExpert[a];
                    Buffer.MemoryCopy(
                        hiddenPtr + (long)t * hiddenSize,
                        batchInPtr + (long)a * hiddenSize,
                        (long)hiddenSize * sizeof(float),
                        (long)hiddenSize * sizeof(float));
                }
            }

            fixed (float* batchInPtr = batchInBuf)
            fixed (float* gateOutPtr = gateOutBuf)
            fixed (float* upOutPtr = upOutBuf)
            fixed (float* interPtr = interBuf)
            fixed (float* downOutPtr = downOutBuf)
            {
                // gate = x · W_gate^T   [n × I]
                MatMul.MoeIndexedMatmulI2_S(
                    gateBank, gateRowBytes, gateScales,
                    batchInPtr, gateOutPtr, intermediateSize, hiddenSize, n,
                    rowExpertIds, threadPool);

                // up = x · W_up^T       [n × I]
                MatMul.MoeIndexedMatmulI2_S(
                    upBank, upRowBytes, upScales,
                    batchInPtr, upOutPtr, intermediateSize, hiddenSize, n,
                    rowExpertIds, threadPool);

                // inter = ffn_sub_norm_e( relu²(gate) * up )   per-expert Sub-LN
                for (int a = 0; a < n; a++)
                {
                    var gateSpan = new ReadOnlySpan<float>(gateOutPtr + (long)a * intermediateSize, intermediateSize);
                    var upSpan = new ReadOnlySpan<float>(upOutPtr + (long)a * intermediateSize, intermediateSize);
                    var interSpan = new Span<float>(interPtr + (long)a * intermediateSize, intermediateSize);
                    FusedOps.ReLU2GLU(gateSpan, upSpan, interSpan);
                    RmsNorm.Execute(interSpan, expertFfnSubNorm[rowExpertIds[a]], rmsEps, interSpan);
                }

                // down = inter · W_down^T   [n × H]
                MatMul.MoeIndexedMatmulI2_S(
                    downBank, downRowBytes, downScales,
                    interPtr, downOutPtr, hiddenSize, intermediateSize, n,
                    rowExpertIds, threadPool);

                // Weighted top-k accumulation → output[t] = Σ_slot w * down_out[t*k+slot].
                fixed (float* outPtr = output)
                {
                    for (int t = 0; t < seqLen; t++)
                    {
                        float* dst = outPtr + (long)t * hiddenSize;
                        for (int j = 0; j < hiddenSize; j++) dst[j] = 0f;
                        for (int slot = 0; slot < numExpertsPerTok; slot++)
                        {
                            int a = t * numExpertsPerTok + slot;
                            float w = assignWeight[a];
                            float* src = downOutPtr + (long)a * hiddenSize;
                            for (int j = 0; j < hiddenSize; j++) dst[j] += w * src[j];
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(assignExpertBuf);
            ArrayPool<float>.Shared.Return(assignWeightBuf);
            ArrayPool<int>.Shared.Return(bucketCursorsBuf);
            ArrayPool<int>.Shared.Return(bucketTokensBuf);
            ArrayPool<int>.Shared.Return(bucketSlotsBuf);
            ArrayPool<int>.Shared.Return(uniqueExpertsBuf);
            ArrayPool<float>.Shared.Return(batchInBuf);
            ArrayPool<float>.Shared.Return(gateOutBuf);
            ArrayPool<float>.Shared.Return(upOutBuf);
            ArrayPool<float>.Shared.Return(interBuf);
            ArrayPool<float>.Shared.Return(downOutBuf);
            ArrayPool<int>.Shared.Return(rowExpertIdsBuf);
        }
    }
}
