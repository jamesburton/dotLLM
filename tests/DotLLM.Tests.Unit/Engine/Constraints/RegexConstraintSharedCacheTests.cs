using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

/// <summary>
/// <see cref="RegexConstraint.Create"/> shares the compiled DFA and the per-state token masks across requests for one
/// (tokenizer, pattern). Rebuilding a mask walks the whole vocabulary (~10 ms at 248k tokens), which is more than a
/// one-token classifier answer's decode pass, so a per-request cache made constrained requests no faster than
/// unconstrained ones.
/// </summary>
public sealed class RegexConstraintSharedCacheTests
{
    private sealed class CountingTokenizer : ITokenizer
    {
        public int DecodeTokenCalls;
        public int VocabSize => 16;
        public int BosTokenId => 1;
        public int EosTokenId => 15;
        public int[] Encode(string text) => [];
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId)
        {
            Interlocked.Increment(ref DecodeTokenCalls);
            return tokenId.ToString(CultureInfo.InvariantCulture);
        }
        public int CountTokens(string text) => 0;
    }

    [Fact]
    public void SecondConstraintForTheSamePattern_ReusesTheBuiltMask()
    {
        var tok = new CountingTokenizer();
        var first = RegexConstraint.Create(tok, "12");
        var maskA = first.GetAllowedTokens();
        int callsAfterFirst = tok.DecodeTokenCalls;
        Assert.True(callsAfterFirst > 0);

        var second = RegexConstraint.Create(tok, "12");
        var maskB = second.GetAllowedTokens();

        Assert.Equal(callsAfterFirst, tok.DecodeTokenCalls);   // no vocabulary walk the second time
        Assert.True(Unsafe.AreSame(
            ref MemoryMarshal.GetReference(maskA.AsSpan()), ref MemoryMarshal.GetReference(maskB.AsSpan())),
            "the second constraint must receive the very same cached mask");
    }

    [Fact]
    public void ConstraintsKeepIndependentStateWhileSharingMasks()
    {
        var tok = new CountingTokenizer();
        var a = RegexConstraint.Create(tok, "12");
        var b = RegexConstraint.Create(tok, "12");

        a.Advance(12);                       // token "12" completes the pattern
        Assert.True(a.IsComplete());
        Assert.False(b.IsComplete());        // b has not advanced
    }

    [Fact]
    public void DifferentPatterns_DoNotShareMasks()
    {
        var tok = new CountingTokenizer();
        var a = RegexConstraint.Create(tok, "12").GetAllowedTokens();
        var b = RegexConstraint.Create(tok, "13").GetAllowedTokens();
        Assert.True(a.IsAllowed(12));
        Assert.False(a.IsAllowed(13));
        Assert.True(b.IsAllowed(13));
        Assert.False(b.IsAllowed(12));
    }
}
