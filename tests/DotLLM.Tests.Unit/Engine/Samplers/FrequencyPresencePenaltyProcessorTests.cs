using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class FrequencyPresencePenaltyProcessorTests
{
    private readonly FrequencyPresencePenaltyProcessor _processor = new();

    [Fact]
    public void Process_FrequencyPenalty_ScalesWithOccurrenceCount()
    {
        float[] logits = [10.0f, 10.0f, 10.0f];
        var previousTokens = new List<int> { 1, 1, 1, 2 }; // token 1 appears 3x, token 2 appears 1x
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            FrequencyPenalty: 1.0f);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(10.0f, logits[0]); // unseen, untouched
        Assert.Equal(7.0f, logits[1]);  // 10 - 1*3
        Assert.Equal(9.0f, logits[2]);  // 10 - 1*1
    }

    [Fact]
    public void Process_PresencePenalty_AppliedOnceRegardlessOfCount()
    {
        float[] logits = [10.0f, 10.0f];
        var previousTokens = new List<int> { 0, 0, 0, 0 }; // token 0 appears 4x
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            PresencePenalty: 2.0f);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(8.0f, logits[0]);  // 10 - 2 (once, not 4x)
        Assert.Equal(10.0f, logits[1]); // unseen
    }

    [Fact]
    public void Process_BothPenaltiesZero_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var previousTokens = new List<int> { 0, 1, 2 };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_EmptyHistory_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            FrequencyPenalty: 1.0f, PresencePenalty: 1.0f);

        _processor.Process(logits, [], context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_CombinedFrequencyAndPresence()
    {
        float[] logits = [10.0f];
        var previousTokens = new List<int> { 0, 0 }; // token 0 appears 2x
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            FrequencyPenalty: 0.5f, PresencePenalty: 1.0f);

        _processor.Process(logits, previousTokens, context);

        // 10 - (0.5 * 2) - 1.0 = 8.0
        Assert.Equal(8.0f, logits[0]);
    }
}
