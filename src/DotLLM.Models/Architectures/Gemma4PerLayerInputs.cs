using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Host-side Gemma-4 Per-Layer-Embedding (PLE) input construction shared by every backend.
/// The PLE input tensor depends only on the token ids (token-identity gather from the huge
/// <c>per_layer_token_embd</c> table + a projection of the scaled main embedding), so GPU backends build it
/// here on the host with the exact CPU-oracle code and upload the result; only the cheap per-layer gated
/// injection runs on the device. Issue #734.
/// </summary>
internal static unsafe class Gemma4PerLayerInputs
{
    /// <summary>
    /// Gathers <c>per_layer_token_embd[token]</c> rows (<paramref name="rowWidth"/> = numLayers*pleDim) into
    /// <paramref name="dest"/> <c>[seq, rowWidth]</c>, dequantizing per token and multiplying by <paramref name="scale"/>
    /// (= sqrt(pleDim), the ScaledWordEmbedding factor).
    /// </summary>
    internal static void GatherIdentity(
        PerLayerEmbeddingWeights ple, ReadOnlySpan<int> tokenIds, float* dest, int rowWidth, float scale)
    {
        nint tablePtr = ple.EmbedTokensPerLayer;
        var qt = ple.EmbedTokensPerLayerQt;
        int pleVocab = ple.VocabSize;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)pleVocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"PLE token ID {tokenId} at position {t} is out of range [0, {pleVocab}).");

            var destSpan = new Span<float>(dest + (long)t * rowWidth, rowWidth);
            if (qt == QuantizationType.F32)
            {
                float* src = (float*)tablePtr + (long)tokenId * rowWidth;
                new ReadOnlySpan<float>(src, rowWidth).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)tablePtr + (long)tokenId * rowWidth;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, rowWidth), destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(rowWidth, qt);
                nint rowPtr = tablePtr + (nint)((long)tokenId * rowBytes);
                Dequantize.ToFloat32(rowPtr, rowWidth, qt, destSpan);
            }

            TensorPrimitives.Multiply(destSpan, scale, destSpan);
        }
    }

    /// <summary>
    /// Computes the PLE per-layer inputs for <paramref name="tokenIds"/> exactly as the CPU forward does, returned
    /// LAYER-MAJOR <c>[numLayers, seq, pleDim]</c> so a device backend can copy one contiguous
    /// <c>[seq, pleDim]</c> slice per layer.
    /// </summary>
    internal static float[] ComputeLayerMajor(TransformerWeights weights, ModelConfig config, ReadOnlySpan<int> tokenIds)
    {
        var ple = weights.PerLayerEmbedding
            ?? throw new InvalidOperationException("Model has no Per-Layer Embedding weights.");
        int seq = tokenIds.Length;
        int hidden = config.HiddenSize;
        int pleDim = ple.PerLayerDim;
        int layers = config.NumLayers;
        int lp = layers * pleDim;
        float eps = config.NormEpsilon;
        float embScale = config.EmbeddingScale ?? 1.0f;

        float* embeds = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * (long)seq * hidden), 64);
        float* identity = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * (long)seq * lp), 64);
        float* inputs = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * (long)seq * lp), 64);
        try
        {
            nint embPtr = weights.TokenEmbedWeight;
            var eqt = weights.TokenEmbedQuantType;
            for (int t = 0; t < seq; t++)
            {
                int id = tokenIds[t];
                if ((uint)id >= (uint)config.VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds),
                        $"Token ID {id} at position {t} is out of range [0, {config.VocabSize}).");
                var row = new Span<float>(embeds + (long)t * hidden, hidden);
                if (eqt == QuantizationType.F32)
                    new ReadOnlySpan<float>((float*)embPtr + (long)id * hidden, hidden).CopyTo(row);
                else if (eqt == QuantizationType.F16)
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)embPtr + (long)id * hidden, hidden), row);
                else
                    Dequantize.ToFloat32(embPtr + (nint)((long)id * Dequantize.RowByteSize(hidden, eqt)), hidden, eqt, row);
                TensorPrimitives.Multiply(row, embScale, row);
            }

            GatherIdentity(ple, tokenIds, identity, lp, MathF.Sqrt(pleDim));
            PerLayerEmbeddings.ComputeInputs(
                tokenIdentity: identity, inputsEmbeds: embeds,
                projWeight: (float*)ple.ModelProjection, projNormWeight: ple.ProjectionNorm,
                projScratch: inputs, output: inputs,
                seqLen: seq, hiddenSize: hidden, numLayers: layers, pleDim: pleDim, eps: eps);

            var result = new float[(long)layers * seq * pleDim];
            for (int l = 0; l < layers; l++)
                for (int t = 0; t < seq; t++)
                    new ReadOnlySpan<float>(inputs + (long)t * lp + (long)l * pleDim, pleDim)
                        .CopyTo(result.AsSpan((int)(((long)l * seq + t) * pleDim), pleDim));
            return result;
        }
        finally
        {
            NativeMemory.AlignedFree(embeds);
            NativeMemory.AlignedFree(identity);
            NativeMemory.AlignedFree(inputs);
        }
    }

    /// <summary>
    /// Maps the proportional-rope factors (<c>rope_freqs.weight</c>) of a Gemma-4 E2B/E4B checkpoint onto the
    /// partial-rotary form every backend's RoPE kernel already supports. The released checkpoints carry factors
    /// that are exactly 1.0 for the leading pairs and a huge divisor (1e30: angle ~ 0, i.e. identity) for the rest,
    /// which is equivalent to rotating only the leading <c>2*n</c> dims with the frequency denominator and the
    /// rotate-half pairing taken over the FULL head dim. Throws when the factors are anything else (a general
    /// per-pair factor table would need a dedicated kernel variant).
    /// </summary>
    /// <returns>The number of leading rotated PAIRS (<c>n</c>).</returns>
    internal static int ResolveProportionalRopePairs(float[] factors, int fullHeadDim)
    {
        int n = 0;
        while (n < factors.Length && factors[n] == 1.0f) n++;
        for (int i = n; i < factors.Length; i++)
            if (!(factors[i] >= 1e10f))
                throw new NotSupportedException(
                    "rope_freqs.weight holds general per-pair factors (not the {1.0 x n, >=1e10 x rest} proportional form "
                    + "of the released Gemma-4 E2B/E4B checkpoints); this backend maps proportional rope onto partial rotary only.");
        if (n <= 0 || 2 * n > fullHeadDim)
            throw new NotSupportedException($"rope_freqs.weight leading-1.0 count {n} is invalid for head dim {fullHeadDim}.");
        return n;
    }
}
