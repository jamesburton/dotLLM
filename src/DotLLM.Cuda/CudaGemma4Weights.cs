using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda;

/// <summary>
/// Per-layer GPU weights for a Gemma-4 (DiffusionGemma autoregressive) MoE layer.
/// Holds the gemma4-specific extras the generic <see cref="CudaLayerWeights"/> and
/// <see cref="CudaMoeLayerWeights"/> do not carry:
/// the five FFN norms, the custom-router channel scale (with <c>1/sqrt(hidden)</c>
/// pre-folded), the per-layer output scale, and the V-from-K flag.
/// </summary>
/// <remarks>
/// <para>
/// <b>Norms are F32 device buffers.</b> Every gemma4 norm weight is uploaded as
/// F32 (the gemma4 forward runs the F32 high-precision path so the RMSNorm kernel
/// takes an F32 weight). NO <c>+1</c> offset — gemma4 overrides Gemma-3's
/// <c>(1+w)</c> convention; the GGUF already stores final weights.
/// </para>
/// <para>
/// <b>Experts.</b> The per-expert gate/up/down projections live in the sibling
/// <see cref="CudaMoeLayerWeights"/> (one entry per gemma4 layer in
/// <see cref="CudaWeights.MoeLayers"/>). The gemma4 loader host-dequantises the
/// fused <c>ffn_gate_up_exps</c> bank into separate F32 gate/up per-expert banks
/// and the <c>ffn_down_exps</c> bank into an F32 down bank with the per-expert
/// <c>ffn_down_exps.scale</c> folded in — so <see cref="CudaMoeFfn"/>'s F32 routed
/// path runs unchanged (GeGLU substituted for SwiGLU by the gemma4 FFN helper).
/// The router stays F32 and is NOT scaled by <c>1/sqrt(hidden)</c> (that fold lives
/// in <see cref="RouterScaleDevice"/>, applied as the router-input RMSNorm gamma).
/// </para>
/// </remarks>
internal sealed class CudaGemma4LayerWeights
{
    /// <summary>Pre-attention RMSNorm <c>attn_norm</c> [hidden] (F32 device).</summary>
    public required nint AttnNorm;
    /// <summary>Dense-FFN pre-norm <c>ffn_norm</c> [hidden] (F32 device).</summary>
    public required nint FfnNorm;
    /// <summary>Post-attention RMSNorm <c>post_attention_norm</c> [hidden] (F32 device).</summary>
    public required nint PostAttnNorm;
    /// <summary>Per-head Q-norm <c>attn_q_norm</c> [headDim] (F32 device).</summary>
    public required nint QNorm;
    /// <summary>Per-head K-norm <c>attn_k_norm</c> [headDim] (F32 device).</summary>
    public required nint KNorm;
    /// <summary>MoE-branch pre-norm <c>pre_ffw_norm_2</c> [hidden] (F32 device).</summary>
    public required nint PreFfwNorm2;
    /// <summary>Dense-branch post-norm <c>post_ffw_norm_1</c> [hidden] (F32 device).</summary>
    public required nint PostFfwNorm1;
    /// <summary>MoE-branch post-norm <c>post_ffw_norm_2</c> [hidden] (F32 device).</summary>
    public required nint PostFfwNorm2;
    /// <summary>Combined post-norm <c>post_ffw_norm</c> [hidden] (F32 device) — wraps (dense + MoE).</summary>
    public required nint PostFfwNorm;
    /// <summary>
    /// Custom-router channel scale <c>ffn_gate_inp.scale</c> [hidden] (F32 device),
    /// with <c>1/sqrt(hidden)</c> pre-folded. Used as the gamma for the router-input
    /// RMSNorm so the router logits read <c>ffn_gate_inp · rms(attn_out)·RouterScale·(1/√H)</c>.
    /// </summary>
    public required nint RouterScaleDevice;
    /// <summary>Per-layer output scale <c>layer_output_scale</c> — scalar, LAST per-layer op.</summary>
    public required float LayerOutputScale;
    /// <summary>True on a V-less (global / full-attention) layer where V branches off the raw K projection.</summary>
    public required bool VFromK;
}

/// <summary>
/// Loads the per-layer gemma4 extras to GPU and host-dequantises the gemma4 MoE
/// expert banks into the F32 layout <see cref="CudaMoeFfn"/> consumes.
/// </summary>
internal static unsafe class CudaGemma4WeightsLoader
{
    /// <summary>
    /// Builds the gemma4 per-layer extras (F32 norms + folded router scale) and the
    /// companion F32 <see cref="CudaMoeLayerWeights"/> with the fused gate_up bank
    /// split into separate gate/up per-expert banks and the per-expert down scale
    /// folded into the down bank.
    /// </summary>
    public static (CudaGemma4LayerWeights Extras, CudaMoeLayerWeights Moe) LoadLayer(
        in TransformerLayerWeights cpuLayer, ModelConfig config, List<nint> allocs)
    {
        var g4 = cpuLayer.Gemma4
            ?? throw new InvalidOperationException("CudaGemma4WeightsLoader.LoadLayer called without Gemma4 extras.");
        var moe = cpuLayer.Moe
            ?? throw new InvalidOperationException("CudaGemma4WeightsLoader.LoadLayer called without Moe config.");
        if (!moe.HasRawQuantView)
            throw new InvalidOperationException(
                "Gemma4 MoE layer must carry raw GGUF quant views for the fused gate_up / down banks.");

        int hidden = moe.HiddenSize;
        int interm = moe.IntermediateSize;     // expert FF width Ie
        int numExperts = moe.NumExperts;
        float invSqrtH = 1.0f / MathF.Sqrt(hidden);

        // ── Router scale with 1/sqrt(hidden) folded in ──
        float[] routerScaleScaled = new float[g4.RouterScale!.Length];
        for (int j = 0; j < routerScaleScaled.Length; j++)
            routerScaleScaled[j] = g4.RouterScale[j] * invSqrtH;
        nint routerScaleDev = UploadF32Array(routerScaleScaled, allocs);

        var extras = new CudaGemma4LayerWeights
        {
            AttnNorm = UploadF32Array(cpuLayer.AttnNormWeight, allocs),
            FfnNorm = UploadF32Array(cpuLayer.FfnNormWeight, allocs),
            PostAttnNorm = UploadF32Array(
                cpuLayer.PostAttnNormWeight
                    ?? throw new InvalidOperationException("Gemma4 layer missing post_attention_norm."),
                allocs),
            QNorm = cpuLayer.QNormWeight is { } qn ? UploadF32Array(qn, allocs)
                : throw new InvalidOperationException("Gemma4 layer missing attn_q_norm."),
            KNorm = cpuLayer.KNormWeight is { } kn ? UploadF32Array(kn, allocs)
                : throw new InvalidOperationException("Gemma4 layer missing attn_k_norm."),
            PreFfwNorm2 = UploadF32Array(g4.PreFfwNorm2!, allocs),
            PostFfwNorm1 = UploadF32Array(g4.PostFfwNorm1!, allocs),
            PostFfwNorm2 = UploadF32Array(g4.PostFfwNorm2!, allocs),
            PostFfwNorm = UploadF32Array(g4.PostFfwNorm, allocs),
            RouterScaleDevice = routerScaleDev,
            LayerOutputScale = g4.LayerOutputScale,
            VFromK = g4.VFromK,
        };

        // ── Router gate: ffn_gate_inp.weight, already F32 [numExperts, hidden]. ──
        nint router = UploadF32Array(moe.Gate, allocs);

        // ── Host-dequant the fused gate_up bank into separate F32 gate/up per-expert
        //    banks. Per expert the slab is [2*Ie, hidden]; gate = rows [0, Ie), up =
        //    rows [Ie, 2*Ie). gateExpsRaw points at the gate base (= slab base); the
        //    per-expert slab stride is GateUpExpsRowBytes. ──
        long gateUpStride = g4.GateUpExpsRowBytes;
        long slabElems = (long)(2 * interm) * hidden;
        long gateElems = (long)interm * hidden;

        var gateProj = new nint[numExperts];
        var upProj = new nint[numExperts];
        var downProj = new nint[numExperts];

        float[] slabF32 = new float[slabElems];
        float[] downF32 = new float[(long)hidden * interm];
        for (int e = 0; e < numExperts; e++)
        {
            // Dequant the whole [2*Ie, hidden] slab (gate rows then up rows).
            Dequantize.ToFloat32(
                moe.GateExpsRaw + (nint)((long)e * gateUpStride),
                slabElems, moe.GateExpsRawQt, slabF32);
            // Gate = first Ie*hidden floats; Up = next Ie*hidden floats.
            gateProj[e] = UploadF32Span(slabF32.AsSpan(0, (int)gateElems), allocs);
            upProj[e] = UploadF32Span(slabF32.AsSpan((int)gateElems, (int)gateElems), allocs);

            // Down bank [hidden, Ie] per expert, scaled by ffn_down_exps.scale[e].
            long downStride = g4.DownExpsRowBytes;
            Dequantize.ToFloat32(
                moe.DownExpsRaw + (nint)((long)e * downStride),
                (long)hidden * interm, moe.DownExpsRawQt, downF32);
            float ds = g4.DownExpertScale![e];
            for (int i = 0; i < downF32.Length; i++) downF32[i] *= ds;
            downProj[e] = UploadF32Span(downF32, allocs);
        }

        var moeWeights = new CudaMoeLayerWeights(
            numExperts, moe.NumExpertsPerTok, hidden, interm,
            // gemma4 router renorm: top-k weights are renormalised to sum 1
            // (with the 6.1e-5 clamp handled by the gemma4 FFN helper).
            normTopKProb: true,
            router,
            gateProj, upProj, downProj,
            numSharedExperts: 0, sharedIntermediateSize: 0,
            sharedGateProj: null, sharedUpProj: null, sharedDownProj: null,
            sharedExpertGate: 0);

        return (extras, moeWeights);
    }

    private static nint UploadF32Array(float[] data, List<nint> allocs)
        => UploadF32Span(data, allocs);

    private static nint UploadF32Span(ReadOnlySpan<float> data, List<nint> allocs)
    {
        long bytes = (long)data.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        fixed (float* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)bytes).ThrowOnError();
        return devPtr;
    }
}
