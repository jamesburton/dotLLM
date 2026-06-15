namespace DotLLM.Engine.Diffusion;

/// <summary>
/// Result of a masked-diffusion denoising run.
/// </summary>
/// <param name="Tokens">The final denoised token ids (length == canvas length; no mask tokens remain).</param>
/// <param name="CommitStep">
/// Per-position step index at which each token was committed (unmasked). Useful for diagnostics/tests:
/// higher-confidence positions commit at earlier steps.
/// </param>
public readonly record struct DenoiseResult(int[] Tokens, int[] CommitStep);

/// <summary>
/// Host-side denoising loop for <b>absorbing-state masked diffusion</b> language models (e.g. DiffuGPT,
/// open-dcoder, LLaDA, MDLM). This is the B1 mechanism for the diffusion-LM spike — it is fully
/// backend-agnostic: the caller supplies a forward delegate that runs the model (with
/// <see cref="DotLLM.Cpu.Kernels"/> bidirectional attention) over the current canvas and returns logits.
///
/// <para>
/// Inference: the canvas starts fully masked. Each step the model predicts a distribution for every
/// position; the highest-confidence currently-masked positions are committed (unmasked) according to a
/// MaskGIT-style cosine schedule, so that after <c>steps</c> iterations every position is committed.
/// Committed tokens are held fixed and become context for subsequent steps. This is the discrete-diffusion
/// analogue of the autoregressive decode loop — but over the full sequence in parallel, with no causal KV
/// cache for the in-progress canvas.
/// </para>
/// </summary>
public static class MaskedDiffusionDenoiser
{
    /// <summary>
    /// Runs the masked-diffusion denoising loop and returns the final tokens.
    /// </summary>
    /// <param name="length">Canvas length (number of token positions to generate).</param>
    /// <param name="maskTokenId">The absorbing/mask token id placed in not-yet-committed positions.</param>
    /// <param name="vocabSize">Vocabulary size (logits row width).</param>
    /// <param name="steps">Number of denoising iterations (e.g. 16–32).</param>
    /// <param name="forward">
    /// Model forward: given the current canvas token ids (length <paramref name="length"/>, with
    /// <paramref name="maskTokenId"/> in uncommitted positions), returns logits laid out row-major as
    /// <c>[length × vocabSize]</c>. Called once per step.
    /// </param>
    /// <returns>The denoised tokens and the per-position commit step.</returns>
    /// <exception cref="ArgumentOutOfRangeException">If sizes are non-positive.</exception>
    /// <exception cref="ArgumentException">If <paramref name="forward"/> returns the wrong logit length.</exception>
    public static DenoiseResult Denoise(int length, int maskTokenId, int vocabSize, int steps,
                                        Func<int[], float[]> forward)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(steps);
        ArgumentNullException.ThrowIfNull(forward);

        int[] tokens = new int[length];
        Array.Fill(tokens, maskTokenId);
        bool[] committed = new bool[length];
        int[] commitStep = new int[length];
        Array.Fill(commitStep, -1);

        // Scratch for per-position best prediction this step.
        int[] bestToken = new int[length];
        float[] bestConf = new float[length];

        for (int t = 0; t < steps; t++)
        {
            float[] logits = forward(tokens);
            if (logits.Length != (long)length * vocabSize)
                throw new ArgumentException(
                    $"forward returned {logits.Length} logits; expected {(long)length * vocabSize} ({length}×{vocabSize}).",
                    nameof(forward));

            int remaining = 0;
            for (int p = 0; p < length; p++)
            {
                if (committed[p]) continue;
                remaining++;
                ArgmaxWithConfidence(logits.AsSpan(p * vocabSize, vocabSize), out bestToken[p], out bestConf[p]);
            }
            if (remaining == 0) break;

            // MaskGIT cosine schedule: target number STILL masked after this step. Last step → 0.
            int targetMasked = t == steps - 1
                ? 0
                : (int)MathF.Floor(length * MathF.Cos(MathF.PI / 2f * (t + 1) / steps));
            int toCommit = Math.Clamp(remaining - targetMasked, 1, remaining);

            CommitHighestConfidence(tokens, committed, commitStep, bestToken, bestConf, toCommit, t);
        }

        return new DenoiseResult(tokens, commitStep);
    }

    /// <summary>
    /// Commits the <paramref name="toCommit"/> highest-confidence currently-masked positions: writes their
    /// predicted token, marks them committed, and records the step. A partial selection sort over masked
    /// positions (toCommit is small relative to length in practice).
    /// </summary>
    private static void CommitHighestConfidence(int[] tokens, bool[] committed, int[] commitStep,
                                                int[] bestToken, float[] bestConf, int toCommit, int step)
    {
        for (int n = 0; n < toCommit; n++)
        {
            int best = -1;
            float bestC = float.NegativeInfinity;
            for (int p = 0; p < tokens.Length; p++)
            {
                if (committed[p]) continue;
                if (bestConf[p] > bestC) { bestC = bestConf[p]; best = p; }
            }
            if (best < 0) break;
            tokens[best] = bestToken[best];
            committed[best] = true;
            commitStep[best] = step;
        }
    }

    /// <summary>
    /// Returns the argmax token and its softmax probability (confidence) for one logits row.
    /// </summary>
    private static void ArgmaxWithConfidence(ReadOnlySpan<float> row, out int token, out float confidence)
    {
        int argmax = 0;
        float max = row[0];
        for (int i = 1; i < row.Length; i++)
            if (row[i] > max) { max = row[i]; argmax = i; }

        // softmax probability of the argmax = 1 / Σ exp(logit_i - max).
        float sum = 0f;
        for (int i = 0; i < row.Length; i++)
            sum += MathF.Exp(row[i] - max);

        token = argmax;
        confidence = sum > 0f ? 1f / sum : 1f;
    }
}
