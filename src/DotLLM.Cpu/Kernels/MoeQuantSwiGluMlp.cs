using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Quantized-expert dense-routing top-k MoE FFN kernel (gpt-oss convention).
/// Unlike <see cref="MoeSwiGluMlp"/> (which consumes per-expert F32 pointer
/// banks), this kernel reads the routed-expert weights directly from the
/// GGUF-mmap'd 3D-stacked tensors in their on-disk quantization (MXFP4 for
/// gpt-oss; Q8_0 / F32 also supported) — no F32 host inflation.
/// </summary>
/// <remarks>
/// <para>Reference semantics (llama.cpp <c>llm_build_openai_moe_iswa</c> +
/// <c>build_moe_ffn</c> with <c>LLM_FFN_SWIGLU_OAI_MOE</c> and
/// <c>LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT</c>):</para>
/// <code>
///   logits = hidden @ router.T + router_bias        # [E]
///   idx    = topk(logits, k)                        # on RAW logits
///   w      = softmax(logits[idx])                   # softmax AFTER top-k
///   per expert e in idx:
///     gate = W_gate[e] @ x + b_gate[e]
///     up   = W_up[e]   @ x + b_up[e]
///     act  = min(gate, limit) * sigmoid(alpha * min(gate, limit))
///            * (clamp(up, -limit, limit) + 1)       # swiglu_oai
///     out += w[e] * (W_down[e] @ act + b_down[e])
/// </code>
/// <para>The optional plain-SwiGLU mode (<c>useSwiGluOai=false</c>,
/// <c>softmaxAfterTopK=false</c>) reproduces the Mixtral gating
/// (softmax-then-topk-then-renormalise) for reuse by other quantized-expert
/// architectures.</para>
/// </remarks>
public static unsafe class MoeQuantSwiGluMlp
{
    /// <summary>gpt-oss swiglu_oai gate scaling (llama.cpp constant).</summary>
    public const float SwiGluOaiAlpha = 1.702f;

    /// <summary>gpt-oss swiglu_oai clamp limit (llama.cpp constant).</summary>
    public const float SwiGluOaiLimit = 7.0f;

    /// <summary>
    /// Executes the quantized-expert MoE FFN over <paramref name="seqLen"/> tokens.
    /// <paramref name="hidden"/> and <paramref name="output"/> may alias.
    /// </summary>
    /// <param name="hidden">F32 input activations [seqLen × hiddenSize].</param>
    /// <param name="output">F32 output activations [seqLen × hiddenSize]. Fully overwritten.</param>
    /// <param name="seqLen">Token count.</param>
    /// <param name="routerWeight">F32 router weight [numExperts × hiddenSize] row-major.</param>
    /// <param name="routerBias">Optional router bias [numExperts]. Null = no bias.</param>
    /// <param name="gateExpsBase">Base pointer of the 3D <c>ffn_gate_exps</c> tensor (expert e at offset e × I × RowByteSize(H, qt)).</param>
    /// <param name="gateQt">Quantization of the gate bank.</param>
    /// <param name="upExpsBase">Base pointer of <c>ffn_up_exps</c>.</param>
    /// <param name="upQt">Quantization of the up bank.</param>
    /// <param name="downExpsBase">Base pointer of <c>ffn_down_exps</c> (expert e at offset e × H × RowByteSize(I, qt)).</param>
    /// <param name="downQt">Quantization of the down bank.</param>
    /// <param name="gateBias">Optional per-expert gate bias, flat [numExperts × intermediateSize].</param>
    /// <param name="upBias">Optional per-expert up bias, flat [numExperts × intermediateSize].</param>
    /// <param name="downBias">Optional per-expert down bias, flat [numExperts × hiddenSize].</param>
    /// <param name="numExperts">Total routed expert count (E).</param>
    /// <param name="numExpertsPerTok">Top-k active experts per token.</param>
    /// <param name="hiddenSize">Hidden / residual dim (H).</param>
    /// <param name="intermediateSize">Per-expert MLP intermediate dim (I).</param>
    /// <param name="softmaxAfterTopK">True = gpt-oss gating (top-k on raw logits, softmax over selected). False = Mixtral gating (softmax first, renormalised top-k).</param>
    /// <param name="useSwiGluOai">True = clamped swiglu_oai activation; false = plain SwiGLU.</param>
    /// <param name="pool">Optional thread pool for row-parallel expert GEMVs.</param>
    [SkipLocalsInit]
    public static void Execute(
        float* hidden, float* output, int seqLen,
        ReadOnlySpan<float> routerWeight, ReadOnlySpan<float> routerBias,
        nint gateExpsBase, QuantizationType gateQt,
        nint upExpsBase, QuantizationType upQt,
        nint downExpsBase, QuantizationType downQt,
        ReadOnlySpan<float> gateBias, ReadOnlySpan<float> upBias, ReadOnlySpan<float> downBias,
        int numExperts, int numExpertsPerTok,
        int hiddenSize, int intermediateSize,
        bool softmaxAfterTopK, bool useSwiGluOai,
        ComputeThreadPool? pool)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (numExpertsPerTok <= 0 || numExpertsPerTok > numExperts)
            throw new ArgumentOutOfRangeException(nameof(numExpertsPerTok));
        if (routerWeight.Length < (long)numExperts * hiddenSize)
            throw new ArgumentException("routerWeight too small", nameof(routerWeight));

        long gateExpertBytes = (long)intermediateSize * Dequantize.RowByteSize(hiddenSize, gateQt);
        long upExpertBytes = (long)intermediateSize * Dequantize.RowByteSize(hiddenSize, upQt);
        long downExpertBytes = (long)hiddenSize * Dequantize.RowByteSize(intermediateSize, downQt);

        float[] logitsBuf = ArrayPool<float>.Shared.Rent(numExperts);
        float[] gateBuf = ArrayPool<float>.Shared.Rent(intermediateSize);
        float[] upBuf = ArrayPool<float>.Shared.Rent(intermediateSize);
        float[] downBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        float[] accBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        float[] xBuf = ArrayPool<float>.Shared.Rent(hiddenSize);

        Span<int> topkIdx = stackalloc int[numExpertsPerTok];
        Span<float> topkVal = stackalloc float[numExpertsPerTok];

        try
        {
            fixed (float* routerPtr = routerWeight)
            fixed (float* logitsPtr = logitsBuf)
            fixed (float* gatePtr = gateBuf)
            fixed (float* upPtr = upBuf)
            fixed (float* downPtr = downBuf)
            fixed (float* xPtr = xBuf)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    // Copy the token's hidden row so callers may alias hidden/output.
                    new ReadOnlySpan<float>(hidden + (long)t * hiddenSize, hiddenSize)
                        .CopyTo(xBuf.AsSpan(0, hiddenSize));

                    // ── 1) Router logits (+ bias) ─────────────────────────
                    MatMul.GemvF32(routerPtr, xPtr, logitsPtr, numExperts, hiddenSize);
                    if (!routerBias.IsEmpty)
                    {
                        var l = logitsBuf.AsSpan(0, numExperts);
                        TensorPrimitives.Add((ReadOnlySpan<float>)l, routerBias, l);
                    }

                    // ── 2) Expert selection + gating weights ─────────────
                    if (softmaxAfterTopK)
                    {
                        // gpt-oss: top-k on RAW logits, then softmax over the k logits.
                        MoeSwiGluMlp.SelectTopK(logitsBuf.AsSpan(0, numExperts), topkIdx, topkVal);
                        float m = float.NegativeInfinity;
                        for (int i = 0; i < numExpertsPerTok; i++) m = MathF.Max(m, topkVal[i]);
                        float sum = 0f;
                        for (int i = 0; i < numExpertsPerTok; i++)
                        {
                            topkVal[i] = MathF.Exp(topkVal[i] - m);
                            sum += topkVal[i];
                        }
                        float inv = 1f / sum;
                        for (int i = 0; i < numExpertsPerTok; i++) topkVal[i] *= inv;
                    }
                    else
                    {
                        // Mixtral: softmax over all logits, top-k, renormalise.
                        var l = logitsBuf.AsSpan(0, numExperts);
                        Softmax.Execute(l, l);
                        MoeSwiGluMlp.SelectTopK(l, topkIdx, topkVal);
                        float sum = 0f;
                        for (int i = 0; i < numExpertsPerTok; i++) sum += topkVal[i];
                        float inv = sum > 0f ? 1f / sum : 0f;
                        for (int i = 0; i < numExpertsPerTok; i++) topkVal[i] *= inv;
                    }

                    // ── 3) Per-expert quantized SwiGLU MLP ───────────────
                    var acc = accBuf.AsSpan(0, hiddenSize);
                    acc.Clear();

                    for (int slot = 0; slot < numExpertsPerTok; slot++)
                    {
                        int e = topkIdx[slot];
                        float w = topkVal[slot];

                        GemvQuant(gateExpsBase + (nint)(e * gateExpertBytes), gateQt,
                                  xPtr, gatePtr, intermediateSize, hiddenSize, pool);
                        GemvQuant(upExpsBase + (nint)(e * upExpertBytes), upQt,
                                  xPtr, upPtr, intermediateSize, hiddenSize, pool);

                        var gateSpan = gateBuf.AsSpan(0, intermediateSize);
                        var upSpan = upBuf.AsSpan(0, intermediateSize);
                        if (!gateBias.IsEmpty)
                            TensorPrimitives.Add((ReadOnlySpan<float>)gateSpan,
                                gateBias.Slice(e * intermediateSize, intermediateSize), gateSpan);
                        if (!upBias.IsEmpty)
                            TensorPrimitives.Add((ReadOnlySpan<float>)upSpan,
                                upBias.Slice(e * intermediateSize, intermediateSize), upSpan);

                        if (useSwiGluOai)
                            SwiGluOai(gateSpan, upSpan, gateSpan);
                        else
                            FusedOps.SwiGLU(gateSpan, upSpan, gateSpan);

                        GemvQuant(downExpsBase + (nint)(e * downExpertBytes), downQt,
                                  gatePtr, downPtr, hiddenSize, intermediateSize, pool);

                        var downSpan = downBuf.AsSpan(0, hiddenSize);
                        if (!downBias.IsEmpty)
                            TensorPrimitives.Add((ReadOnlySpan<float>)downSpan,
                                downBias.Slice(e * hiddenSize, hiddenSize), downSpan);

                        TensorPrimitives.MultiplyAdd((ReadOnlySpan<float>)downSpan, w, acc, acc);
                    }

                    acc.CopyTo(new Span<float>(output + (long)t * hiddenSize, hiddenSize));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(logitsBuf);
            ArrayPool<float>.Shared.Return(gateBuf);
            ArrayPool<float>.Shared.Return(upBuf);
            ArrayPool<float>.Shared.Return(downBuf);
            ArrayPool<float>.Shared.Return(accBuf);
            ArrayPool<float>.Shared.Return(xBuf);
        }
    }

    /// <summary>
    /// gpt-oss clamped SwiGLU (llama.cpp <c>ggml_swiglu_oai</c>): with
    /// <c>x = min(gate, limit)</c> and <c>y = clamp(up, -limit, limit)</c>,
    /// <c>out = x / (1 + exp(-alpha * x)) * (y + 1)</c>. In-place safe when
    /// <paramref name="output"/> aliases <paramref name="gate"/>.
    /// </summary>
    [SkipLocalsInit]
    public static void SwiGluOai(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output,
                                 float alpha = SwiGluOaiAlpha, float limit = SwiGluOaiLimit)
    {
        for (int i = 0; i < output.Length; i++)
        {
            float x = MathF.Min(gate[i], limit);
            float y = Math.Clamp(up[i], -limit, limit);
            float glu = x / (1f + MathF.Exp(alpha * -x));
            output[i] = glu * (y + 1f);
        }
    }

    /// <summary>
    /// Dispatches a quantized GEMV (<c>y[m] = W[m,k] · x</c>) by weight
    /// quantization type. Falls back to a row-wise dequant + dot for formats
    /// without a fused kernel.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void GemvQuant(nint weights, QuantizationType qt, float* x, float* y,
                                   int m, int k, ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.MXFP4:
                MatMul.GemvMxfp4((byte*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.Q8_0:
                MatMul.GemvQ8_0((byte*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.Q4_K:
                MatMul.GemvQ4_K((byte*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.Q5_K:
                MatMul.GemvQ5_K((byte*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.Q6_K:
                MatMul.GemvQ6_K((byte*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.F32:
                MatMul.GemvF32((float*)weights, x, y, m, k, pool);
                break;
            case QuantizationType.F16:
                MatMul.GemvF16(weights, x, y, m, k, pool);
                break;
            default:
                GemvDequantRows(weights, qt, x, y, m, k);
                break;
        }
    }

    /// <summary>
    /// Dequantize-per-row GEMV fallback. Shared with the fused decode dispatcher so both
    /// last-resort paths behave identically.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GemvDequantRows(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
        => MatMul.GemvDequantRows((byte*)weights, qt, x, y, m, k);
}
