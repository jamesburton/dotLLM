using System.Buffers;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Computes language-model perplexity: the exponential of the mean per-token
/// negative log-likelihood (cross-entropy) of a tokenized corpus under a model.
/// </summary>
/// <remarks>
/// <para><b>Shift / causal-LM convention.</b> A causal LM at position <c>t</c> predicts
/// the token at position <c>t+1</c>. Given a row of logits for each input position, the
/// score for target token <c>tokenIds[t+1]</c> is read from <c>logits[t]</c>. The very
/// first token (<c>tokenIds[0]</c>) is never scored — it has no preceding context inside
/// the window — so a window of <c>N</c> tokens contributes <c>N-1</c> scored predictions.</para>
/// <para><b>Numerical stability.</b> The per-row log-softmax denominator is computed with
/// the standard max-subtraction trick (<c>log Σ exp(x_j - max)</c>) and the running NLL is
/// accumulated in <see cref="double"/> to avoid single-precision drift over long corpora.
/// Only the target token's log-probability is needed per row, so no vocab-sized buffer is
/// materialized — this is both more accurate and cheaper than building a full log-softmax
/// vector per position.</para>
/// <para><b>Chunking.</b> A corpus longer than the model's context window is processed in
/// non-overlapping windows, each forwarded with positions reset to 0. The first token of every
/// window is therefore unscored (it has no in-window predecessor), and no context carries across
/// window boundaries — this is the standard non-overlapping, context-reset estimator. It is NOT
/// directly comparable to llama.cpp's <c>perplexity</c>, which slides a stride over a larger
/// window and scores only the second half; a sliding-window stride would reduce boundary bias
/// but is intentionally out of scope here.</para>
/// </remarks>
public static class PerplexityEvaluator
{
    /// <summary>Result of a perplexity evaluation.</summary>
    /// <param name="Perplexity">exp(mean NLL) — the perplexity.</param>
    /// <param name="MeanNll">Mean per-token negative log-likelihood (natural log, nats).</param>
    /// <param name="ScoredTokenCount">Number of tokens that contributed a prediction (targets scored).</param>
    /// <param name="TotalTokenCount">Total tokens in the corpus (including unscored first-of-window tokens).</param>
    public readonly record struct PerplexityResult(
        double Perplexity,
        double MeanNll,
        int ScoredTokenCount,
        int TotalTokenCount);

    /// <summary>
    /// Computes the summed negative log-likelihood over a single window of per-position logits.
    /// </summary>
    /// <param name="logits">Row-major logits, <paramref name="rows"/> × <paramref name="vocabSize"/>.
    /// Row <c>t</c> holds the model's logits after seeing <c>tokenIds[t]</c>.</param>
    /// <param name="rows">Number of logit rows (must equal the window's token count).</param>
    /// <param name="vocabSize">Vocabulary size (length of each row).</param>
    /// <param name="tokenIds">Token IDs for this window. Length must equal <paramref name="rows"/>.</param>
    /// <returns>A tuple of the summed NLL (nats) and the number of scored targets
    /// (<c>rows - 1</c> when <paramref name="rows"/> &gt;= 1, otherwise 0).</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are inconsistent.</exception>
    public static (double SumNll, int Scored) ComputeWindowNll(
        ReadOnlySpan<float> logits, int rows, int vocabSize, ReadOnlySpan<int> tokenIds)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (tokenIds.Length != rows)
            throw new ArgumentException(
                $"tokenIds length ({tokenIds.Length}) must equal rows ({rows}).", nameof(tokenIds));
        if ((long)rows * vocabSize > logits.Length)
            throw new ArgumentException(
                $"logits span ({logits.Length}) too small for {rows}×{vocabSize}.", nameof(logits));

        double sumNll = 0;
        int scored = 0;

        // Predict tokenIds[t+1] from logits[t]; tokenIds[0] has no preceding context (unscored).
        for (int t = 0; t < rows - 1; t++)
        {
            ReadOnlySpan<float> row = logits.Slice(t * vocabSize, vocabSize);
            int target = tokenIds[t + 1];
            if ((uint)target >= (uint)vocabSize)
                throw new ArgumentException(
                    $"Target token id {target} out of range [0,{vocabSize}).", nameof(tokenIds));

            // Stable log-softmax denominator in double: log Σ exp(x_j - max).
            float max = row[0];
            for (int j = 1; j < vocabSize; j++)
                if (row[j] > max) max = row[j];

            double sumExp = 0;
            for (int j = 0; j < vocabSize; j++)
                sumExp += Math.Exp(row[j] - max);

            double logProb = (row[target] - max) - Math.Log(sumExp);
            sumNll += -logProb;
            scored++;
        }

        return (sumNll, scored);
    }

    /// <summary>
    /// Computes perplexity for a tokenized corpus by running the model forward over
    /// non-overlapping context windows and averaging the per-token cross-entropy.
    /// </summary>
    /// <param name="model">A loaded model. Must return per-position logits of shape
    /// <c>[windowLength, vocabSize]</c> from <see cref="IModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/>
    /// — i.e. the CPU backend. GPU/hybrid models return only the last row and will fail the row-count guard.</param>
    /// <param name="tokenIds">The full tokenized corpus.</param>
    /// <param name="maxWindow">Maximum window length. Clamped to the model's context length.
    /// When &lt;= 0 the model's <see cref="ModelConfig.MaxSequenceLength"/> is used.</param>
    /// <param name="onWindow">Optional progress callback invoked after each window with
    /// (windowsDone, totalWindows, scoredSoFar).</param>
    /// <returns>The perplexity result.</returns>
    /// <exception cref="ArgumentException">Thrown when there are too few tokens to score, or
    /// the model returns the wrong logit shape.</exception>
    public static PerplexityResult Evaluate(
        IModel model,
        ReadOnlySpan<int> tokenIds,
        int maxWindow = 0,
        Action<int, int, int>? onWindow = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        int total = tokenIds.Length;
        int vocabSize = model.Config.VocabSize;

        if (total < 2)
            throw new ArgumentException(
                $"Need at least 2 tokens to score perplexity (got {total}).", nameof(tokenIds));

        int contextLimit = model.Config.MaxSequenceLength;
        int window = maxWindow > 0 ? Math.Min(maxWindow, contextLimit) : contextLimit;
        if (window < 2)
            window = 2;

        // Pre-count windows for progress reporting.
        int totalWindows = (total + window - 1) / window;

        double sumNll = 0;
        int scored = 0;
        int[] positions = ArrayPool<int>.Shared.Rent(window);
        try
        {
            int windowIndex = 0;
            for (int start = 0; start < total; start += window)
            {
                int len = Math.Min(window, total - start);

                // A trailing 1-token window contributes no scored targets — skip it.
                if (len < 2)
                {
                    windowIndex++;
                    onWindow?.Invoke(windowIndex, totalWindows, scored);
                    break;
                }

                ReadOnlySpan<int> windowTokens = tokenIds.Slice(start, len);
                Span<int> pos = positions.AsSpan(0, len);
                for (int i = 0; i < len; i++)
                    pos[i] = i;

                using ITensor logits = model.Forward(windowTokens, pos, deviceId: -1);

                // Guard: per-token perplexity requires all rows. CPU returns [len, vocab];
                // GPU/hybrid return [1, vocab] (last token only) — turn that into a loud failure
                // instead of a silently-wrong number.
                if (logits.Shape.Rank != 2 || logits.Shape[0] != len || logits.Shape[1] != vocabSize)
                    throw new ArgumentException(
                        $"Model returned logits of shape [{string.Join(",", ShapeDims(logits.Shape))}] "
                        + $"but perplexity needs [{len},{vocabSize}] (all positions). "
                        + "Use the CPU backend — GPU/hybrid models only return the last row.");

                (double winNll, int winScored) = ComputeWindowNllFromTensor(logits, len, vocabSize, windowTokens);
                sumNll += winNll;
                scored += winScored;

                windowIndex++;
                onWindow?.Invoke(windowIndex, totalWindows, scored);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positions);
        }

        if (scored == 0)
            throw new ArgumentException(
                "No tokens were scored — corpus too short for the model context window.", nameof(tokenIds));

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, total);
    }

    private static unsafe (double SumNll, int Scored) ComputeWindowNllFromTensor(
        ITensor logits, int rows, int vocabSize, ReadOnlySpan<int> tokenIds)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocabSize);
        return ComputeWindowNll(span, rows, vocabSize, tokenIds);
    }

    private static int[] ShapeDims(TensorShape shape)
    {
        var dims = new int[shape.Rank];
        for (int i = 0; i < shape.Rank; i++)
            dims[i] = shape[i];
        return dims;
    }
}
