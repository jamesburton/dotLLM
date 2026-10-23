using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class DryProcessorTests
{
    private readonly DryProcessor _processor = new();

    [Fact]
    public void Process_RepeatedBigram_PenalizesTokenThatFollowedItLastTime()
    {
        // History: A B C A B  (tokens 1,2,3,1,2). Tail bigram "A B" (positions 3,4) already
        // occurred once before (positions 0,1), and was followed by C (position 2).
        // DRY should penalize token C for the next position.
        float[] logits = [0f, 0f, 0f, 0f]; // index 0 unused, 1=A, 2=B, 3=C
        var previousTokens = new List<int> { 1, 2, 3, 1, 2 };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 1.0f, DryBase: 1.75f, DryAllowedLength: 2);

        _processor.Process(logits, previousTokens, context);

        Assert.True(logits[3] < 0f, "Token that previously followed the repeated bigram should be penalized.");
        Assert.Equal(0f, logits[1]);
        Assert.Equal(0f, logits[2]);
    }

    [Fact]
    public void Process_LongerMatch_PenalizedMoreThanShorterMatch()
    {
        // Two candidate histories: one with a 2-token match, one with a 3-token match, same
        // allowed length and base — the longer match must produce a strictly larger penalty.
        var context2 = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 1.0f, DryBase: 1.75f, DryAllowedLength: 1);

        // "A B" -> C  (match length 2 at the repeat)
        float[] shortLogits = [0f, 0f, 0f, 0f];
        _processor.Process(shortLogits, [1, 2, 3, 1, 2], context2);

        // "A B C" -> D (match length 3 at the repeat)
        float[] longLogits = [0f, 0f, 0f, 0f, 0f];
        _processor.Process(longLogits, [1, 2, 3, 4, 1, 2, 3], context2);

        float shortPenalty = -shortLogits[3];
        float longPenalty = -longLogits[4];
        Assert.True(longPenalty > shortPenalty,
            $"Longer match should be penalized more (short={shortPenalty}, long={longPenalty}).");
    }

    [Fact]
    public void Process_MatchBelowAllowedLength_NoPenalty()
    {
        // Only a 1-token match (just the tail token itself); default allowed length is 2.
        float[] logits = [0f, 0f, 0f];
        float[] original = [0f, 0f, 0f];
        var previousTokens = new List<int> { 1, 2, 3, 2 }; // tail=2, earlier occurrence at index 1, no extension
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 1.0f, DryBase: 1.75f, DryAllowedLength: 2);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_MultiplierZero_Skips()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        float[] original = [0f, 0f, 0f, 0f];
        var previousTokens = new List<int> { 1, 2, 3, 1, 2 };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 0f);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_ShortHistory_Skips()
    {
        float[] logits = [0f, 0f];
        float[] original = [0f, 0f];
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 1.0f);

        _processor.Process(logits, [1], context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_SequenceBreaker_StopsMatchExtension()
    {
        // History: A 99 B A 99 B  (A=1, breaker=99, B=2). Without a breaker, the anchor pair at
        // offset=1 (h[1] vs h[4]) would extend the match to length 2 and trigger a penalty at
        // allowedLength=2. With 99 registered as a breaker token, extension stops immediately
        // (matchLength stays at 1), so no candidate reaches the threshold and logits are untouched.
        int[] history = [1, 99, 2, 1, 99, 2];
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0,
            DryMultiplier: 1.0f, DryBase: 1.75f, DryAllowedLength: 2);

        float[] withoutBreakers = [0f, 0f, 0f];
        new DryProcessor().Process(withoutBreakers, history, context);
        Assert.True(withoutBreakers[1] < 0f, "Without breakers, the bigram match should be penalized.");

        float[] withBreakers = [0f, 0f, 0f];
        new DryProcessor(new HashSet<int> { 99 }).Process(withBreakers, history, context);
        Assert.Equal(0f, withBreakers[1]);
        Assert.Equal(0f, withBreakers[0]);
        Assert.Equal(0f, withBreakers[2]);
    }
}
