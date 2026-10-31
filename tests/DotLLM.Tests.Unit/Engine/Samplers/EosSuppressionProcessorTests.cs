using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

/// <summary>#1345: <c>ignore_eos</c> / <c>min_tokens</c> as an end-of-generation logit mask.</summary>
public sealed class EosSuppressionProcessorTests
{
    private static readonly ProcessorContext Ctx = new(RepetitionPenalty: 1f, RepetitionPenaltyWindow: 0, SequenceId: 0);

    [Fact]
    public void IgnoreEos_MasksEveryEndOfGenerationId_Always()
    {
        var p = new EosSuppressionProcessor([2, 7], ignoreEos: true, minTokens: 0);
        float[] logits = [0f, 1f, 5f, 3f, 0f, 0f, 0f, 9f];
        p.Process(logits, new List<int> { 1, 1, 1, 1, 1, 1 }, Ctx);
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[7]));
        Assert.Equal(3f, logits[3]);
        Assert.Equal(1f, logits[1]);
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(3, true)]
    [InlineData(4, false)]   // 4 tokens exist: min_tokens=4 is satisfied
    [InlineData(10, false)]
    public void MinTokens_MasksOnlyUntilEnoughTokensExist(int generated, bool masked)
    {
        var p = new EosSuppressionProcessor([2], ignoreEos: false, minTokens: 4);
        float[] logits = [0f, 0f, 5f, 0f];
        p.Process(logits, Enumerable.Repeat(1, generated).ToList(), Ctx);
        Assert.Equal(masked, float.IsNegativeInfinity(logits[2]));
    }

    [Fact]
    public void OutOfRangeIds_AreIgnored()
    {
        var p = new EosSuppressionProcessor([99], ignoreEos: true, minTokens: 0);
        float[] logits = [1f, 2f];
        p.Process(logits, new List<int>(), Ctx);
        Assert.Equal([1f, 2f], logits);
    }

    [Fact]
    public void GreedyPipeline_NeverPicksMaskedToken_ThenRestoresItAfterMinTokens()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f,
            LogitProcessors = [new EosSuppressionProcessor([2], ignoreEos: false, minTokens: 2)],
        };
        var pipeline = new SamplerPipeline(options);
        Assert.True(pipeline.HasProcessors);   // keeps the host-side paths; device/raw-argmax shortcuts are gated on this

        float[] a = [0f, 1f, 9f, 0f];
        Assert.Equal(1, pipeline.Sample(a, new List<int> { 3 }));        // 1 token: EOS (2) masked, runner-up wins
        float[] b = [0f, 1f, 9f, 0f];
        Assert.Equal(2, pipeline.Sample(b, new List<int> { 3, 3 }));     // 2 tokens: EOS allowed again
    }
}
