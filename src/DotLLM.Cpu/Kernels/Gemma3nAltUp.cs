using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Gemma-3n AltUp (Alternating Updates) CPU kernel. AltUp maintains
/// <c>numInputs</c> parallel hidden-state streams per token; each decoder layer
/// <c>predict</c>s a per-stream update from a router over the active stream,
/// runs attention/Laurel/MLP on the active stream only, then <c>correct</c>s
/// every stream from that result. Verified against HF <c>transformers</c>
/// <c>modeling_gemma3n.py</c> <c>Gemma3nTextAltUp.predict</c> /
/// <c>.correct</c> / <c>.compute_router_modalities</c> and the model-level
/// input/output stream construction in <c>Gemma3nTextModel.forward</c>
/// (<c>get_per_layer_inputs</c> is PLE, not AltUp — see
/// <see cref="PerLayerEmbeddings"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Layout.</b> Every stream is a contiguous <c>[seqLen, hiddenSize]</c> F32
/// buffer; a "stream set" is <c>numInputs</c> such buffers addressed via a
/// <c>float**</c> (an array of pointers, index 0..numInputs-1). All per-token
/// router/coefficient math operates on a tiny <c>numInputs</c>-length vector
/// (4 on every released Gemma-3n SKU) so it is implemented as plain scalar
/// loops — the dominant cost (attention + MLP) lives entirely outside this
/// kernel, on the single active-stream buffer the caller extracts per layer.
/// </para>
/// <para>
/// <b>Coefficient algebra.</b> HF's <c>predict</c> reshapes the
/// <c>prediction_coefs</c> Linear(N, N²) output to <c>[N,N]</c> and applies two
/// permutes around a batched matmul; working through the index algebra (see
/// the PR notes) collapses this to the direct form used below:
/// <c>predictions[n] = streams[n] + Σ_i coef[n,i]·streams[i]</c>, where
/// <c>coef[n,i]</c> is simply row <c>n</c>, column <c>i</c> of the Linear
/// output reshaped <c>[N,N]</c> — no transpose needed once expanded
/// algebraically. <c>correct</c> needs no such reduction: HF's
/// <c>correction_coefs</c> is a Linear(N,N) (not N²), and its permute/unsqueeze
/// only broadcasts a per-stream scalar over the hidden dimension.
/// </para>
/// </remarks>
public static unsafe class Gemma3nAltUp
{
    /// <summary>
    /// Computes <c>tanh(modality_router(router_norm(x) * hidden^-1))</c> for every
    /// token — HF <c>Gemma3nTextAltUp.compute_router_modalities</c>. Note the scale
    /// is <c>1/hiddenSize</c> (NOT <c>1/sqrt(hiddenSize)</c>).
    /// </summary>
    /// <param name="x">Input stream <c>[seqLen, hiddenSize]</c> (the active stream,
    /// or the "activated" post-MLP value in <c>correct</c>).</param>
    /// <param name="routerNormWeight">Gemma3nRMSNorm weight <c>[hiddenSize]</c>
    /// ((1+w) already absorbed).</param>
    /// <param name="modalityRouterWeight"><c>modality_router.weight</c>
    /// <c>[numInputs, hiddenSize]</c> row-major (no bias).</param>
    /// <param name="modalities">Destination <c>[seqLen, numInputs]</c>.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numInputs">AltUp stream count.</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    /// <param name="normScratch">Scratch <c>[hiddenSize]</c>.</param>
    [SkipLocalsInit]
    public static void ComputeRouterModalities(
        float* x, ReadOnlySpan<float> routerNormWeight, float* modalityRouterWeight,
        float* modalities, int seqLen, int hiddenSize, int numInputs, float eps,
        float* normScratch)
    {
        float routerInputScale = 1.0f / hiddenSize;
        for (int t = 0; t < seqLen; t++)
        {
            var xRow = new ReadOnlySpan<float>(x + (long)t * hiddenSize, hiddenSize);
            var normed = new Span<float>(normScratch, hiddenSize);
            RmsNorm.Execute(xRow, routerNormWeight, eps, normed);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)normed, routerInputScale, normed);

            float* mRow = modalities + (long)t * numInputs;
            for (int n = 0; n < numInputs; n++)
            {
                float* w = modalityRouterWeight + (long)n * hiddenSize;
                float dot = TensorPrimitives.Dot(normed, new ReadOnlySpan<float>(w, hiddenSize));
                mRow[n] = MathF.Tanh(dot);
            }
        }
    }

    /// <summary>
    /// AltUp <c>predict</c>: derives a per-stream prediction from the current
    /// stream set. <c>predictions[n] = streams[n] + Σ_i coef[n,i]·streams[i]</c>,
    /// <c>coef = reshape(prediction_coefs(modalities), [N,N])</c> (row <c>n</c> =
    /// output stream, column <c>i</c> = input stream), <c>modalities =
    /// compute_router_modalities(streams[activeIdx])</c>.
    /// </summary>
    /// <param name="streams">Input stream set, <paramref name="numInputs"/> pointers
    /// each <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="predictionCoefsWeight"><c>prediction_coefs.weight</c>
    /// <c>[numInputs*numInputs, numInputs]</c> row-major (no bias).</param>
    /// <param name="routerNormWeight"><c>altup.router_norm.weight</c> ((1+w) absorbed).</param>
    /// <param name="modalityRouterWeight"><c>altup.modality_router.weight</c>
    /// <c>[numInputs, hiddenSize]</c> row-major (no bias).</param>
    /// <param name="predictions">Output stream set (may alias a distinct buffer set
    /// from <paramref name="streams"/>; must not alias it in place since every
    /// output stream reads every input stream).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numInputs">AltUp stream count.</param>
    /// <param name="activeIdx">Index of the active stream.</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    /// <param name="modalityScratch">Scratch <c>[seqLen, numInputs]</c>.</param>
    /// <param name="coefScratch">Scratch <c>[numInputs*numInputs]</c> (per-token).</param>
    /// <param name="normScratch">Scratch <c>[hiddenSize]</c>.</param>
    public static void Predict(
        float** streams, float* predictionCoefsWeight,
        ReadOnlySpan<float> routerNormWeight, float* modalityRouterWeight,
        float** predictions,
        int seqLen, int hiddenSize, int numInputs, int activeIdx, float eps,
        float* modalityScratch, float* coefScratch, float* normScratch)
    {
        ComputeRouterModalities(streams[activeIdx], routerNormWeight, modalityRouterWeight,
            modalityScratch, seqLen, hiddenSize, numInputs, eps, normScratch);

        for (int t = 0; t < seqLen; t++)
        {
            float* modalitiesRow = modalityScratch + (long)t * numInputs;

            // coef[n,i] = row n of prediction_coefs(modalities) reshaped [N,N].
            for (int n = 0; n < numInputs; n++)
            {
                float* w = predictionCoefsWeight + (long)n * numInputs * numInputs;
                for (int i = 0; i < numInputs; i++)
                {
                    float* wRow = w + (long)i * numInputs;
                    float dot = 0f;
                    for (int p = 0; p < numInputs; p++)
                        dot += wRow[p] * modalitiesRow[p];
                    coefScratch[n * numInputs + i] = dot;
                }
            }

            for (int n = 0; n < numInputs; n++)
            {
                float* outRow = predictions[n] + (long)t * hiddenSize;
                float* baseRow = streams[n] + (long)t * hiddenSize;
                new ReadOnlySpan<float>(baseRow, hiddenSize).CopyTo(new Span<float>(outRow, hiddenSize));

                for (int i = 0; i < numInputs; i++)
                {
                    float c = coefScratch[n * numInputs + i];
                    if (c == 0f) continue;
                    float* inRow = streams[i] + (long)t * hiddenSize;
                    for (int h = 0; h < hiddenSize; h++)
                        outRow[h] += c * inRow[h];
                }
            }
        }
    }

    /// <summary>
    /// AltUp <c>correct</c>: propagates the "activated" post-MLP value of the
    /// active stream back across every stream. <c>innovation = activated -
    /// predictions[activeIdx]</c>; <c>corrected[n] = predictions[n] + innovation ×
    /// (correction_coefs(modalities)[n] + 1)</c>, <c>modalities =
    /// compute_router_modalities(activated)</c>.
    /// </summary>
    /// <param name="predictions">This layer's <see cref="Predict"/> output.</param>
    /// <param name="activated">Post-attention/Laurel/MLP value of the active
    /// stream, <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="correctionCoefsWeight"><c>correction_coefs.weight</c>
    /// <c>[numInputs, numInputs]</c> row-major (no bias).</param>
    /// <param name="routerNormWeight"><c>altup.router_norm.weight</c> ((1+w) absorbed).</param>
    /// <param name="modalityRouterWeight"><c>altup.modality_router.weight</c>
    /// <c>[numInputs, hiddenSize]</c> row-major (no bias).</param>
    /// <param name="corrected">Output stream set (may alias
    /// <paramref name="predictions"/> — written after <paramref name="activated"/>
    /// and <paramref name="predictions"/>[activeIdx] are both read per token).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numInputs">AltUp stream count.</param>
    /// <param name="activeIdx">Index of the active stream.</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    /// <param name="modalityScratch">Scratch <c>[seqLen, numInputs]</c>.</param>
    /// <param name="innovationScratch">Scratch <c>[hiddenSize]</c>.</param>
    /// <param name="normScratch">Scratch <c>[hiddenSize]</c>.</param>
    public static void Correct(
        float** predictions, float* activated, float* correctionCoefsWeight,
        ReadOnlySpan<float> routerNormWeight, float* modalityRouterWeight,
        float** corrected,
        int seqLen, int hiddenSize, int numInputs, int activeIdx, float eps,
        float* modalityScratch, float* innovationScratch, float* normScratch)
    {
        ComputeRouterModalities(activated, routerNormWeight, modalityRouterWeight,
            modalityScratch, seqLen, hiddenSize, numInputs, eps, normScratch);

        for (int t = 0; t < seqLen; t++)
        {
            float* actRow = activated + (long)t * hiddenSize;
            float* predActiveRow = predictions[activeIdx] + (long)t * hiddenSize;
            float* innovation = innovationScratch;
            for (int h = 0; h < hiddenSize; h++)
                innovation[h] = actRow[h] - predActiveRow[h];

            float* modalitiesRow = modalityScratch + (long)t * numInputs;
            for (int n = 0; n < numInputs; n++)
            {
                float* w = correctionCoefsWeight + (long)n * numInputs;
                float dot = 0f;
                for (int p = 0; p < numInputs; p++)
                    dot += w[p] * modalitiesRow[p];
                float coef = dot + 1.0f;

                float* outRow = corrected[n] + (long)t * hiddenSize;
                float* predRow = predictions[n] + (long)t * hiddenSize;
                for (int h = 0; h < hiddenSize; h++)
                    outRow[h] = predRow[h] + coef * innovation[h];
            }
        }
    }

    /// <summary>
    /// Builds the initial <paramref name="numInputs"/>-stream stack from the single
    /// scaled embedding (HF <c>Gemma3nTextModel.forward</c>, pre-layer-loop):
    /// stream 0 is the embedding unchanged; streams 1..N-1 are
    /// <c>altup_projections[i-1](embedding)</c>, each magnitude-matched per token to
    /// the embedding's RMS magnitude.
    /// </summary>
    /// <param name="mainStream">Scaled token embedding <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="streams">Output stream set, <paramref name="numInputs"/> buffers
    /// each <c>[seqLen, hiddenSize]</c>. <c>streams[0]</c> must not alias
    /// <paramref name="mainStream"/> (it is overwritten with a copy).</param>
    /// <param name="altupProjWeights"><paramref name="numInputs"/>-1 pointers, each
    /// <c>altup_projections.{i}.weight</c> <c>[hiddenSize, hiddenSize]</c> row-major
    /// F32 (no bias).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numInputs">AltUp stream count.</param>
    public static void BuildInputStreams(
        float* mainStream, float** streams, float** altupProjWeights,
        int seqLen, int hiddenSize, int numInputs)
    {
        new ReadOnlySpan<float>(mainStream, seqLen * hiddenSize)
            .CopyTo(new Span<float>(streams[0], seqLen * hiddenSize));

        for (int i = 1; i < numInputs; i++)
        {
            MatMul.GemmF32(altupProjWeights[i - 1], mainStream, streams[i], hiddenSize, hiddenSize, seqLen);
            MatchMagnitudePerToken(mainStream, streams[i], seqLen, hiddenSize);
        }
    }

    /// <summary>
    /// Collapses the final <paramref name="numInputs"/>-stream stack to a single
    /// hidden vector (HF <c>Gemma3nTextModel.forward</c>, post-layer-loop): streams
    /// 1..N-1 are unembed-projected + magnitude-matched to stream 0, then all N
    /// streams (including the untouched stream 0) are averaged.
    /// </summary>
    /// <param name="streams">Final stream set after the last decoder layer.</param>
    /// <param name="altupUnembedWeights"><paramref name="numInputs"/>-1 pointers,
    /// each <c>altup_unembed_projections.{i}.weight</c>
    /// <c>[hiddenSize, hiddenSize]</c> row-major F32 (no bias).</param>
    /// <param name="output">Destination <c>[seqLen, hiddenSize]</c>. May alias
    /// <c>streams[0]</c>.</param>
    /// <param name="scratch">Scratch, <paramref name="numInputs"/>-1 buffers each
    /// <c>[seqLen, hiddenSize]</c> (the projected+matched streams 1..N-1).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numInputs">AltUp stream count.</param>
    public static void ReduceOutputStreams(
        float** streams, float** altupUnembedWeights, float* output, float** scratch,
        int seqLen, int hiddenSize, int numInputs)
    {
        for (int i = 1; i < numInputs; i++)
        {
            MatMul.GemmF32(altupUnembedWeights[i - 1], streams[i], scratch[i - 1], hiddenSize, hiddenSize, seqLen);
            MatchMagnitudePerToken(streams[0], scratch[i - 1], seqLen, hiddenSize);
        }

        float invN = 1.0f / numInputs;
        for (int t = 0; t < seqLen; t++)
        {
            float* outRow = output + (long)t * hiddenSize;
            float* s0 = streams[0] + (long)t * hiddenSize;
            for (int h = 0; h < hiddenSize; h++)
                outRow[h] = s0[h];
            for (int i = 1; i < numInputs; i++)
            {
                float* sRow = scratch[i - 1] + (long)t * hiddenSize;
                for (int h = 0; h < hiddenSize; h++)
                    outRow[h] += sRow[h];
            }
            for (int h = 0; h < hiddenSize; h++)
                outRow[h] *= invN;
        }
    }

    /// <summary>
    /// Per-token magnitude match: <c>proj *= sqrt(mean(target^2)) /
    /// max(sqrt(mean(proj^2)), sqrt(1e-5))</c> — HF clamps the new magnitude's
    /// square (not the sqrt) to a 1e-5 floor before the sqrt
    /// (<c>torch.sqrt(torch.maximum(new_magnitude, eps))</c> where
    /// <c>new_magnitude</c> is already the mean-square, i.e. pre-sqrt).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatchMagnitudePerToken(float* target, float* proj, int seqLen, int hiddenSize)
    {
        const float floor = 1e-5f;
        for (int t = 0; t < seqLen; t++)
        {
            float* tRow = target + (long)t * hiddenSize;
            float* pRow = proj + (long)t * hiddenSize;
            float targetMag = MathF.Sqrt(TensorPrimitives.SumOfSquares(new ReadOnlySpan<float>(tRow, hiddenSize)) / hiddenSize);
            float projMeanSq = TensorPrimitives.SumOfSquares(new ReadOnlySpan<float>(pRow, hiddenSize)) / hiddenSize;
            float newMag = MathF.Sqrt(MathF.Max(projMeanSq, floor));
            float scale = targetMag / newMag;
            for (int h = 0; h < hiddenSize; h++)
                pRow[h] *= scale;
        }
    }
}
