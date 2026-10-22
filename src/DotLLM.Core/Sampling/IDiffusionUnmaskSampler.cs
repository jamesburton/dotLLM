namespace DotLLM.Core.Sampling;

/// <summary>
/// Canvas-level unmasking sampler for masked text-diffusion decoding (DiffusionGemma).
/// </summary>
/// <remarks>
/// <para>
/// Where <see cref="ISamplerStep"/> transforms the logits of a <i>single</i> autoregressive
/// position, a diffusion decode step must decide, across <i>all currently-masked canvas
/// positions at once</i>, which subset to commit (unmask) this step and which token each
/// commits to. This canvas-level contract therefore takes the full matrix of per-position
/// logits and returns an <see cref="UnmaskDecision"/>.
/// </para>
/// <para>
/// Implementations are pure functions of their inputs (given a fixed RNG / greedy mode) so a
/// denoising run is reproducible: the same logits, temperature and budget always yield the
/// same decision. The per-position entropy carried on the result lets the driving scheduler
/// read average canvas confidence without recomputing softmaxes.
/// </para>
/// </remarks>
public interface IDiffusionUnmaskSampler
{
    /// <summary>
    /// Selects which masked positions to unmask this step and samples their tokens.
    /// </summary>
    /// <param name="logits">
    /// Row-major logits for the currently-masked positions: <paramref name="maskedCount"/> rows
    /// of <paramref name="vocabSize"/> columns. Row <c>r</c> spans
    /// <c>[r*vocabSize, (r+1)*vocabSize)</c> and corresponds to <c>maskedPositions[r]</c>.
    /// Not modified.
    /// </param>
    /// <param name="maskedPositions">
    /// Canvas indices of the masked positions, parallel to the rows of <paramref name="logits"/>.
    /// </param>
    /// <param name="vocabSize">Number of vocabulary columns per row.</param>
    /// <param name="maskedCount">Number of masked rows (== <paramref name="maskedPositions"/> length).</param>
    /// <param name="context">Temperature, unmask budget and entropy bound for this step.</param>
    /// <returns>The positions to unmask, their token ids, and per-row entropy.</returns>
    UnmaskDecision SelectAndSample(
        ReadOnlySpan<float> logits,
        ReadOnlySpan<int> maskedPositions,
        int vocabSize,
        int maskedCount,
        DiffusionStepContext context);
}
