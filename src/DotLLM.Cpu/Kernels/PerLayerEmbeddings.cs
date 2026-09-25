using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Gemma-4 Per-Layer Embeddings (PLE) CPU kernel. PLE feeds an auxiliary gated
/// residual into every decoder layer, combining a token-identity lookup with a
/// context-aware projection of the (scaled) input embedding. Verified against HF
/// <c>transformers</c> <c>modeling_gemma4.py</c> (<c>get_per_layer_inputs</c> /
/// <c>project_per_layer_inputs</c> and the per-layer injection in
/// <c>Gemma4TextDecoderLayer.forward</c>).
/// </summary>
/// <remarks>
/// All weights are consumed as row-major F32 (<c>[out, in]</c> like an HF
/// <c>nn.Linear</c>); <c>MatMul.GemmF32</c> computes <c>C[N,M]=B[N,K]·A[M,K]^T</c>.
/// The two Gemma RMSNorm weights (<c>per_layer_projection_norm</c>,
/// <c>post_per_layer_input_norm</c>) must already have the Gemma <c>(1+w)</c> offset
/// absorbed by the loader, matching every other Gemma norm in the model.
/// </remarks>
public static unsafe class PerLayerEmbeddings
{
    private const float GeluTanhSqrt2OverPi = 0.7978845608028654f;
    private const float GeluTanhCubicCoeff = 0.044715f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float GeluTanh(float x)
    {
        float inner = GeluTanhSqrt2OverPi * (x + GeluTanhCubicCoeff * x * x * x);
        return 0.5f * x * (1.0f + MathF.Tanh(inner));
    }

    /// <summary>
    /// Builds the per-layer input tensor <c>[seq, numLayers*pleDim]</c> (row-major,
    /// layer-major within a token), combining the token-identity lookup with the
    /// context projection:
    /// <code>
    /// plp = per_layer_model_projection(inputsEmbeds) * hidden^-0.5   // [seq, L*P]
    /// plp = per_layer_projection_norm(reshape(plp)[.., l, :])        // RMSNorm per (token,layer) over P
    /// out = (plp + tokenIdentity) * rsqrt(2)
    /// </code>
    /// </summary>
    /// <param name="tokenIdentity">Gathered + √pleDim-scaled PLE embedding rows <c>[seq, L*P]</c>.</param>
    /// <param name="inputsEmbeds">Scaled main token embedding <c>[seq, hidden]</c>.</param>
    /// <param name="projWeight">per_layer_model_projection <c>[L*P, hidden]</c> F32.</param>
    /// <param name="projNormWeight">per_layer_projection_norm <c>[pleDim]</c> ((1+w) absorbed).</param>
    /// <param name="projScratch">Scratch <c>[seq, L*P]</c> for the projection output.</param>
    /// <param name="output">Destination <c>[seq, L*P]</c>.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numLayers">Number of decoder layers (L).</param>
    /// <param name="pleDim">Per-layer embedding dimension (P).</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    public static void ComputeInputs(
        float* tokenIdentity,
        float* inputsEmbeds,
        float* projWeight,
        ReadOnlySpan<float> projNormWeight,
        float* projScratch,
        float* output,
        int seqLen, int hiddenSize, int numLayers, int pleDim, float eps)
    {
        int lp = numLayers * pleDim;
        float projScale = 1.0f / MathF.Sqrt(hiddenSize);
        float combineScale = 1.0f / MathF.Sqrt(2.0f);

        // Context projection: [seq, L*P] = inputsEmbeds[seq,hidden] · projWeight[L*P,hidden]^T
        MatMul.GemmF32(projWeight, inputsEmbeds, projScratch, lp, hiddenSize, seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            for (int l = 0; l < numLayers; l++)
            {
                int off = t * lp + l * pleDim;
                var p = new Span<float>(projScratch + off, pleDim);
                // Global scale before the per-block RMSNorm (mirrors HF ordering).
                TensorPrimitives.Multiply(p, projScale, p);
                RmsNorm.Execute(p, projNormWeight, eps, p);

                float* id = tokenIdentity + off;
                float* o = output + off;
                for (int i = 0; i < pleDim; i++)
                    o[i] = (p[i] + id[i]) * combineScale;
            }
        }
    }

    /// <summary>
    /// Injects the per-layer gated residual into <paramref name="hidden"/> (the layer
    /// output, AFTER the MLP post-norm + residual add), in place:
    /// <code>
    /// r = hidden
    /// h = per_layer_input_gate(hidden)            // [seq, pleDim]
    /// h = gelu_tanh(h) * perLayerInputs[:, layer, :]
    /// h = per_layer_projection(h)                 // [seq, hidden]
    /// h = post_per_layer_input_norm(h)
    /// hidden = r + h
    /// </code>
    /// </summary>
    /// <param name="hidden">Layer output <c>[seq, hidden]</c>, updated in place.</param>
    /// <param name="perLayerInputs">Output of <see cref="ComputeInputs"/> <c>[seq, L*P]</c>.</param>
    /// <param name="gateWeight">per_layer_input_gate <c>[pleDim, hidden]</c> F32.</param>
    /// <param name="projWeight">per_layer_projection <c>[hidden, pleDim]</c> F32.</param>
    /// <param name="postNormWeight">post_per_layer_input_norm <c>[hidden]</c> ((1+w) absorbed).</param>
    /// <param name="gateScratch">Scratch <c>[seq, pleDim]</c>.</param>
    /// <param name="projScratch">Scratch <c>[seq, hidden]</c>.</param>
    /// <param name="layerIdx">Decoder layer index (selects the per-layer slice).</param>
    /// <param name="numLayers">Number of decoder layers (row stride of perLayerInputs).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="pleDim">Per-layer embedding dimension.</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    public static void InjectLayer(
        float* hidden,
        float* perLayerInputs,
        int layerIdx, int numLayers,
        float* gateWeight,
        float* projWeight,
        ReadOnlySpan<float> postNormWeight,
        float* gateScratch,
        float* projScratch,
        int seqLen, int hiddenSize, int pleDim, float eps)
    {
        int lp = numLayers * pleDim;

        // gate = hidden · gateWeight^T → [seq, pleDim]
        MatMul.GemmF32(gateWeight, hidden, gateScratch, pleDim, hiddenSize, seqLen);

        // gelu_tanh(gate) * per_layer_input[:, layerIdx, :]
        for (int t = 0; t < seqLen; t++)
        {
            float* g = gateScratch + t * pleDim;
            float* e = perLayerInputs + t * lp + layerIdx * pleDim;
            for (int i = 0; i < pleDim; i++)
                g[i] = GeluTanh(g[i]) * e[i];
        }

        // proj = gate · projWeight^T → [seq, hidden]
        MatMul.GemmF32(projWeight, gateScratch, projScratch, hiddenSize, pleDim, seqLen);

        // post-norm + residual add into hidden (in place)
        for (int t = 0; t < seqLen; t++)
        {
            var p = new Span<float>(projScratch + t * hiddenSize, hiddenSize);
            RmsNorm.Execute(p, postNormWeight, eps, p);
            float* h = hidden + t * hiddenSize;
            for (int i = 0; i < hiddenSize; i++)
                h[i] += p[i];
        }
    }
}
