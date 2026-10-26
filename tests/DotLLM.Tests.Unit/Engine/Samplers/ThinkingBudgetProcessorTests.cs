using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

/// <summary>
/// #1292: the thinking budget follows llama.cpp's <c>common/reasoning-budget.cpp</c>: count thinking tokens, force
/// message + close tag once the budget is spent, leave a block that closes on its own alone, wait for the open tag when
/// the prompt did not open one, and close at once with budget 0.
/// </summary>
public sealed class ThinkingBudgetProcessorTests
{
    private const int Open = 100, Close = 101, Msg1 = 50, Msg2 = 51, Vocab = 128;

    private static readonly ProcessorContext Ctx = new(RepetitionPenalty: 1f, RepetitionPenaltyWindow: 0, SequenceId: 0);

    /// <summary>Tags and message map to fixed ids; every other token decodes to a complete character.</summary>
    private sealed class Tok : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => 2;

        public int[] Encode(string text) => text switch
        {
            "<think>" => [Open],
            "</think>" => [Close],
            "Done." => [Msg1, Msg2],
            "" => [],
            _ => throw new ArgumentException(text),
        };

        public string Decode(ReadOnlySpan<int> tokenIds) => string.Concat(tokenIds.ToArray().Select(DecodeToken));
        public string DecodeToken(int tokenId) => tokenId == 7 ? "�" : tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => 1;
    }

    private static ThinkingBudgetProcessor Make(int budget, bool startsInside = true, string? message = "Done.")
        => new(new ThinkingBudget(budget, "<think>", "</think>", startsInside, message), new Tok());

    /// <summary>The token the processor forces after <paramref name="history"/>, or null when it leaves logits alone.</summary>
    private static int? ForcedAfter(ThinkingBudgetProcessor p, List<int> history)
    {
        var logits = new float[Vocab];
        p.Process(logits, history, Ctx);
        int finite = logits.Count(float.IsFinite);
        return finite == Vocab ? null : Array.FindIndex(logits, float.IsFinite);
    }

    [Fact]
    public void ForcesMessageThenCloseTag_AfterTheBudget_ThenLetsTheAnswerRun()
    {
        var p = Make(budget: 3);
        var h = new List<int>();

        Assert.Null(ForcedAfter(p, h));
        h.AddRange([10, 11]);
        Assert.Null(ForcedAfter(p, h)); // 2 of 3 used
        h.Add(12);
        Assert.Equal(Msg1, ForcedAfter(p, h)); // budget spent: message, then close tag
        h.Add(Msg1);
        Assert.Equal(Msg2, ForcedAfter(p, h));
        h.Add(Msg2);
        Assert.Equal(Close, ForcedAfter(p, h));
        h.Add(Close);
        Assert.Null(ForcedAfter(p, h)); // answer: untouched
        Assert.True(p.Forced);
    }

    [Fact]
    public void BlockThatClosesWithinTheBudget_IsNotTouched()
    {
        var p = Make(budget: 3);
        var h = new List<int> { 10, Close, 20, 21, 22, 23 };

        Assert.Null(ForcedAfter(p, h));
        Assert.False(p.Forced);
    }

    [Fact]
    public void WithoutAPromptOpenedBlock_CountingStartsAtTheModelsOpenTag()
    {
        var p = Make(budget: 2, startsInside: false);
        var h = new List<int> { 10, 11, 12, 13 }; // no think block: an answer, never forced

        Assert.Null(ForcedAfter(p, h));
        h.AddRange([Open, 20, 21]);
        Assert.Equal(Msg1, ForcedAfter(p, h));
    }

    [Fact]
    public void BudgetZero_ClosesAtOnce_AndNoMessageForcesTheTagAlone()
    {
        var p = Make(budget: 0, message: null);

        Assert.Equal(Close, ForcedAfter(p, []));
    }

    [Fact]
    public void BudgetEndingMidCharacter_WaitsForTheCharacterToComplete()
    {
        var p = Make(budget: 2);
        var h = new List<int> { 10, 7 }; // token 7 is half a UTF-8 character

        Assert.Null(ForcedAfter(p, h));
        h.Add(11);
        Assert.Equal(Msg1, ForcedAfter(p, h));
    }

    [Fact]
    public void Rollback_ReplaysFromTheStart()
    {
        var p = Make(budget: 2);
        Assert.Equal(Msg1, ForcedAfter(p, [10, 11]));

        // A rejected draft shrank the history: the processor re-derives its state.
        Assert.Null(ForcedAfter(p, [10]));
    }
}
