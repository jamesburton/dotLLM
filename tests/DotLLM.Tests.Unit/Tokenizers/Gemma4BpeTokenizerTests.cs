using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>
/// Unit tests for the Gemma-4 SPM-style merge-ranked BPE variant
/// (<see cref="BpeTokenizer.CreateGemma4"/> / <c>Gemma4SpmBpeEncoding</c>) using a
/// small synthetic vocabulary. Covers the llama.cpp <c>LLAMA_VOCAB_PRE_TYPE_GEMMA4</c>
/// semantics: ▁ whitespace escaping, rank-ordered merges (not SentencePiece scores),
/// newline-only pre-splitting with the whole-run vocab shortcut, &lt;0xNN&gt; byte
/// fallback, and special-token pre-splitting.
/// </summary>
public class Gemma4BpeTokenizerTests
{
    private const string S = "▁"; // SentencePiece space marker

    // Synthetic vocab. Indices are load-bearing for the assertions below.
    private static readonly string[] Tokens =
    [
        /* 0*/ "<pad>",
        /* 1*/ "<eos>",
        /* 2*/ "<bos>",
        /* 3*/ "<unk>",
        /* 4*/ "<0xC3>",
        /* 5*/ "<0xA9>",
        /* 6*/ S,
        /* 7*/ "a",
        /* 8*/ "b",
        /* 9*/ "c",
        /*10*/ "ab",
        /*11*/ "bc",
        /*12*/ "abc",
        /*13*/ S + "a",
        /*14*/ "\n",
        /*15*/ "\n\n",
        /*16*/ "<sp>",
    ];

    private static readonly int[] TokenTypes =
    [
        3, 3, 3, 3, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
    ];

    private static BpeTokenizer Create(params string[] merges) =>
        BpeTokenizer.CreateGemma4(Tokens, merges, TokenTypes, bosId: 2, eosId: 1);

    // -------------------------------------------------------------------------
    // Whitespace escaping (llama_escape_whitespace: ' ' → ▁)
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_SpaceBecomesSpaceMarker()
    {
        var tok = Create(S + " a"); // merge (▁, a) → ▁a
        Assert.Equal(new[] { 7, 13 }, tok.Encode("a a"));
    }

    [Fact]
    public void Encode_NoSpacePrefixIsPrepended()
    {
        // gemma-4 sets add_space_prefix = false: "a" must be the bare 'a' token.
        var tok = Create();
        Assert.Equal(new[] { 7 }, tok.Encode("a"));
    }

    [Fact]
    public void Encode_LoneSpaceIsMarkerToken()
    {
        var tok = Create();
        Assert.Equal(new[] { 6 }, tok.Encode(" "));
    }

    // -------------------------------------------------------------------------
    // Merge-rank priority (NOT score-based longest match)
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_MergesApplyInRankOrder()
    {
        // Rank 0 = (b,c), rank 1 = (a,b), rank 2 = (a,bc).
        // "abc": (b,c) fires first → [a, bc]; then (a,bc) → [abc].
        var tok = Create("b c", "a b", "a bc");
        Assert.Equal(new[] { 12 }, tok.Encode("abc"));
    }

    [Fact]
    public void Encode_NoMergeEntry_NoMergeApplied()
    {
        // 'ab' exists in the vocab but there is no (a,b) merge entry —
        // rank-based BPE must NOT merge (longest-match would have).
        var tok = Create();
        Assert.Equal(new[] { 7, 8 }, tok.Encode("ab"));
    }

    [Fact]
    public void Encode_TieBrokenByLeftmostPosition()
    {
        // Single merge (a,b) applies leftmost-first: "abab" → [ab, ab].
        var tok = Create("a b");
        Assert.Equal(new[] { 10, 10 }, tok.Encode("abab"));
    }

    // -------------------------------------------------------------------------
    // Newline pre-splitting ([^\n]+|[\n]+) and whole-run shortcut
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_NewlineRunInVocab_EmittedAsSingleToken()
    {
        var tok = Create();
        Assert.Equal(new[] { 15 }, tok.Encode("\n\n"));
    }

    [Fact]
    public void Encode_NewlineRunNotInVocab_FallsBackToMerges()
    {
        // "\n\n\n" is not a vocab token; per-char BPE with merge (\n,\n) gives [\n\n, \n].
        var tok = Create("\n \n");
        Assert.Equal(new[] { 15, 14 }, tok.Encode("\n\n\n"));
    }

    [Fact]
    public void Encode_MergesDoNotCrossNewlineBoundary()
    {
        // Even with an (a,\n) merge entry, the [^\n]+|[\n]+ pre-split keeps
        // 'a' and '\n' in separate runs, so the merge can never fire.
        var tok = Create("a \n");
        Assert.Equal(new[] { 7, 14, 8 }, tok.Encode("a\nb"));
    }

    // -------------------------------------------------------------------------
    // Byte fallback
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_UnknownCodePoint_FallsBackToByteTokens()
    {
        // 'é' (U+00E9) is not in the vocab → UTF-8 bytes C3 A9 → <0xC3>, <0xA9>.
        var tok = Create();
        Assert.Equal(new[] { 4, 5 }, tok.Encode("é"));
    }

    // -------------------------------------------------------------------------
    // Special tokens & empty input
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_SpecialToken_PreSplitToSingleId()
    {
        var tok = Create();
        Assert.Equal(new[] { 7, 16, 7 }, tok.Encode("a<sp>a"));
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        var tok = Create();
        Assert.Empty(tok.Encode(""));
    }

    // -------------------------------------------------------------------------
    // Decode
    // -------------------------------------------------------------------------

    [Fact]
    public void Decode_SpaceMarkerBecomesSpace()
    {
        var tok = Create();
        Assert.Equal("a a", tok.Decode(new[] { 7, 13 }));
    }

    [Fact]
    public void Decode_ByteTokensCombineToUtf8()
    {
        var tok = Create();
        Assert.Equal("é", tok.Decode(new[] { 4, 5 }));
    }

    [Fact]
    public void Roundtrip_TextWithSpacesAndNewlines()
    {
        var tok = Create(S + " a", "a b");
        const string text = "ab a\n\nab";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }
}
