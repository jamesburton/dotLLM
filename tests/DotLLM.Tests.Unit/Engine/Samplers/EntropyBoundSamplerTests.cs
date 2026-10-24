using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class EntropyBoundSamplerTests
{
    private readonly EntropyBoundSampler _sampler = new();

    /// <summary>Builds a row whose peak is at <paramref name="argmax"/> with the given peak logit.</summary>
    private static float[] PeakedRow(int vocab, int argmax, float peak)
    {
        var row = new float[vocab];
        row[argmax] = peak;
        return row;
    }

    private static DiffusionStepContext Ctx(float temp = 0.001f, int budget = 100, float bound = 0.1f, float softCap = 0f)
        => new(temp, budget, bound, softCap);

    [Fact]
    public void EmptyCanvas_ReturnsEmptyDecision()
    {
        var decision = _sampler.SelectAndSample([], [], vocabSize: 4, maskedCount: 0, Ctx());
        Assert.Empty(decision.UnmaskedPositions);
        Assert.Empty(decision.TokenIds);
        Assert.Empty(decision.PerPositionEntropy);
        Assert.Equal(0f, decision.AverageEntropy);
    }

    [Fact]
    public void PerPositionEntropy_Exposed_ParallelToRows()
    {
        const int vocab = 4;
        // Row 0: sharply peaked (low entropy). Row 1: uniform (max entropy = ln 4).
        float[] logits =
        [
            10f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
        ];
        int[] positions = [5, 9];

        var decision = _sampler.SelectAndSample(logits, positions, vocab, maskedCount: 2, Ctx(temp: 1.0f));

        Assert.Equal(2, decision.PerPositionEntropy.Count);
        Assert.True(decision.PerPositionEntropy[0] < decision.PerPositionEntropy[1]);
        // Uniform over 4 classes -> entropy = ln 4 ≈ 1.386.
        Assert.Equal(Math.Log(4), decision.PerPositionEntropy[1], precision: 3);
    }

    [Fact]
    public void LowestEntropyPosition_SelectedFirst()
    {
        const int vocab = 4;
        // Row 0 mild peak, Row 1 strong peak (lowest entropy), Row 2 near-uniform.
        float[] logits =
        [
            2f, 0f, 0f, 0f,
            12f, 0f, 0f, 0f,
            0.1f, 0f, 0f, 0f,
        ];
        int[] positions = [0, 1, 2];

        // Tight bound + budget 1 => only the single most-confident row commits.
        var decision = _sampler.SelectAndSample(logits, positions, vocab, 3, Ctx(temp: 1.0f, budget: 1, bound: 0.01f));

        Assert.Single(decision.UnmaskedPositions);
        Assert.Equal(1, decision.UnmaskedPositions[0]); // position 1 = strongest peak
    }

    [Fact]
    public void Selection_RespectsEntropyBound()
    {
        const int vocab = 4;
        // Three identical low-entropy rows. With temp=1 each row's entropy is small but nonzero.
        float[] logits = new float[3 * vocab];
        for (int r = 0; r < 3; r++)
            logits[r * vocab] = 8f; // peak at index 0 each row
        int[] positions = [0, 1, 2];

        // Each row's entropy is ~0.00905 nats. The first is always admitted (running 0.00905); the
        // second would push the total to ~0.0181. Bound 0.015 therefore admits exactly 1.
        var single = _sampler.SelectAndSample(logits, positions, vocab, 3, Ctx(temp: 1.0f, budget: 100, bound: 0.015f));
        Assert.Single(single.UnmaskedPositions);

        // Bound 0.022 admits 2 (running 0.0181 ≤ 0.022) but not the third (~0.0271 > 0.022).
        var two = _sampler.SelectAndSample(logits, positions, vocab, 3, Ctx(temp: 1.0f, budget: 100, bound: 0.022f));
        Assert.Equal(2, two.UnmaskedPositions.Count);

        // A looser bound admits all three.
        var all = _sampler.SelectAndSample(logits, positions, vocab, 3, Ctx(temp: 1.0f, budget: 100, bound: 10f));
        Assert.Equal(3, all.UnmaskedPositions.Count);
    }

    [Fact]
    public void Selection_RespectsBudgetCap()
    {
        const int vocab = 4;
        float[] logits = new float[4 * vocab];
        for (int r = 0; r < 4; r++)
            logits[r * vocab] = 20f; // all extremely confident (entropy ~0)
        int[] positions = [0, 1, 2, 3];

        // Generous bound, but budget caps at 2.
        var decision = _sampler.SelectAndSample(logits, positions, vocab, 4, Ctx(temp: 1.0f, budget: 2, bound: 10f));
        Assert.Equal(2, decision.UnmaskedPositions.Count);
    }

    [Fact]
    public void AlwaysCommitsAtLeastOne_EvenWhenAllExceedBound()
    {
        const int vocab = 4;
        // Two uniform rows: each entropy = ln 4 ≈ 1.386, far above a 0.1 bound.
        float[] logits = new float[2 * vocab]; // all zeros => uniform
        int[] positions = [0, 1];

        var decision = _sampler.SelectAndSample(logits, positions, vocab, 2, Ctx(temp: 1.0f, budget: 100, bound: 0.1f));
        Assert.Single(decision.UnmaskedPositions); // forced single commit, no stall
    }

    [Fact]
    public void Deterministic_AtLowTemperature_PicksArgmax()
    {
        const int vocab = 5;
        float[] logits =
        [
            1f, 5f, 2f, 0f, 0f, // argmax = 1
            0f, 0f, 9f, 0f, 0f, // argmax = 2
        ];
        int[] positions = [0, 1];

        var a = _sampler.SelectAndSample(logits, positions, vocab, 2, Ctx(temp: 0.001f, budget: 100, bound: 100f));
        var b = _sampler.SelectAndSample(logits, positions, vocab, 2, Ctx(temp: 0.001f, budget: 100, bound: 100f));

        Assert.Equal(a.UnmaskedPositions, b.UnmaskedPositions);
        Assert.Equal(a.TokenIds, b.TokenIds);

        // Tokens must be the per-row argmax regardless of selection order.
        for (int i = 0; i < a.UnmaskedPositions.Count; i++)
        {
            int pos = a.UnmaskedPositions[i];
            int expectedToken = pos == 0 ? 1 : 2;
            Assert.Equal(expectedToken, a.TokenIds[i]);
        }
    }

    [Fact]
    public void SoftCap_ReducesEntropyDifference_StillRanksConfidentFirst()
    {
        const int vocab = 4;
        float[] logits =
        [
            50f, 0f, 0f, 0f, // huge logit, soft-capped
            3f, 0f, 0f, 0f,
        ];
        int[] positions = [0, 1];

        var decision = _sampler.SelectAndSample(logits, positions, vocab, 2, Ctx(temp: 1.0f, budget: 1, bound: 0.001f, softCap: 30f));
        // Row 0 still the most confident after capping.
        Assert.Single(decision.UnmaskedPositions);
        Assert.Equal(0, decision.UnmaskedPositions[0]);
        Assert.Equal(0, decision.TokenIds[0]);
    }

    [Fact]
    public void AverageEntropy_IsMeanOfPerPosition()
    {
        const int vocab = 4;
        float[] logits =
        [
            10f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
        ];
        int[] positions = [0, 1];

        var decision = _sampler.SelectAndSample(logits, positions, vocab, 2, Ctx(temp: 1.0f));
        float expected = (decision.PerPositionEntropy[0] + decision.PerPositionEntropy[1]) / 2f;
        Assert.Equal(expected, decision.AverageEntropy, precision: 5);
    }

    [Fact]
    public void MismatchedLengths_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _sampler.SelectAndSample(new float[8], new int[3], vocabSize: 4, maskedCount: 2, Ctx()));
    }
}
