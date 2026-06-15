using DotLLM.Engine.Diffusion;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Diffusion;

/// <summary>
/// Tests for the masked-diffusion denoising loop (B1 mechanism of the diffusion-LM spike). Uses a mock
/// forward whose per-position confidence is fully controlled, so the schedule and commit ordering are
/// deterministic and checkable without a real model.
/// </summary>
public sealed class MaskedDiffusionDenoiserTests
{
    [Fact]
    public void Denoise_CommitsAllPositions_HighestConfidenceFirst()
    {
        const int length = 8, vocab = 10, mask = 9, steps = 4;
        // Mock: position p predicts token p (all distinct, none == mask), with peak logit (p+1) so
        // confidence strictly increases with p (position 7 most confident, position 0 least).
        int calls = 0;
        float[] Forward(int[] canvas)
        {
            calls++;
            float[] logits = new float[length * vocab];
            for (int p = 0; p < length; p++)
                logits[p * vocab + p] = p + 1; // peak at token==p; rest zero
            return logits;
        }

        var result = MaskedDiffusionDenoiser.Denoise(length, mask, vocab, steps, Forward);

        // All positions denoised to their predicted token; no mask tokens remain.
        for (int p = 0; p < length; p++)
        {
            Assert.Equal(p, result.Tokens[p]);
            Assert.NotEqual(mask, result.Tokens[p]);
            Assert.True(result.CommitStep[p] >= 0, $"position {p} never committed");
        }

        // Commit ordering respects confidence: more-confident (higher p) commit no later than less-confident.
        for (int p = 0; p < length; p++)
            for (int q = p + 1; q < length; q++) // peak[q] > peak[p]
                Assert.True(result.CommitStep[q] <= result.CommitStep[p],
                    $"higher-confidence pos {q} (step {result.CommitStep[q]}) committed later than pos {p} (step {result.CommitStep[p]})");

        // The most confident commits first, the least confident last.
        Assert.Equal(0, result.CommitStep[length - 1]);
        Assert.Equal(steps - 1, result.CommitStep[0]);

        // Discriminating: the schedule is genuinely iterative (partial commits), not all-at-once.
        Assert.Contains(result.CommitStep, s => s > 0);
        Assert.True(calls <= steps, $"forward called {calls} times, expected <= {steps}");
    }

    [Fact]
    public void Denoise_SingleStep_CommitsEverythingAtOnce()
    {
        const int length = 5, vocab = 7, mask = 6;
        float[] Forward(int[] canvas)
        {
            float[] logits = new float[length * vocab];
            for (int p = 0; p < length; p++)
                logits[p * vocab + (p % vocab)] = 5f;
            return logits;
        }

        var result = MaskedDiffusionDenoiser.Denoise(length, mask, vocab, steps: 1, Forward);

        for (int p = 0; p < length; p++)
        {
            Assert.Equal(p % vocab, result.Tokens[p]);
            Assert.Equal(0, result.CommitStep[p]); // single step → all committed at step 0
        }
    }

    [Fact]
    public void Denoise_RejectsWrongLogitLength()
    {
        Assert.Throws<ArgumentException>(() =>
            MaskedDiffusionDenoiser.Denoise(length: 4, maskTokenId: 0, vocabSize: 10, steps: 2,
                forward: _ => new float[3])); // wrong size
    }
}
