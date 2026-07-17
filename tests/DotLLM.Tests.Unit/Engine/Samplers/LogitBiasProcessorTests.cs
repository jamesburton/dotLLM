using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class LogitBiasProcessorTests
{
    private readonly LogitBiasProcessor _processor = new();

    [Fact]
    public void Process_AppliesAdditiveBiasToMappedTokens()
    {
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f];
        var bias = new Dictionary<int, float> { [1] = 10.0f, [3] = -5.0f };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            LogitBias: bias);

        _processor.Process(logits, [], context);

        Assert.Equal(1.0f, logits[0]);
        Assert.Equal(12.0f, logits[1]);
        Assert.Equal(3.0f, logits[2]);
        Assert.Equal(-1.0f, logits[3]);
    }

    [Fact]
    public void Process_NullBias_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, [], context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_EmptyBias_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            LogitBias: new Dictionary<int, float>());

        _processor.Process(logits, [], context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_OutOfRangeTokenId_Ignored()
    {
        float[] logits = [1.0f, 2.0f];
        var bias = new Dictionary<int, float> { [99] = 10.0f };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            LogitBias: bias);

        // Should not throw
        _processor.Process(logits, [], context);

        Assert.Equal(1.0f, logits[0]);
        Assert.Equal(2.0f, logits[1]);
    }
}
