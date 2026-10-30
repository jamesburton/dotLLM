using System.Globalization;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The recurrent-prefix snapshot boundary for a chat request sits just before the chat template's generation prompt
/// (issue #1093): <see cref="RecurrentSnapshotBoundary.FromGenerationSuffix"/> finds it, and
/// <see cref="RecurrentSnapshotBoundary.Choose"/> combines it with the boundary each resolve path computed before.
/// </summary>
public sealed class RecurrentSnapshotBoundaryTests
{
    private static readonly CsvTokenizer Tok = new();

    private static int[] Ids(int from, int count) => [.. Enumerable.Range(from, count)];

    [Fact]
    public void FromGenerationSuffix_ReturnsLengthMinusSuffix_WhenTailMatches()
        => Assert.Equal(18, RecurrentSnapshotBoundary.FromGenerationSuffix(Ids(1, 20), Tok, "19,20"));

    [Fact]
    public void FromGenerationSuffix_ReturnsNull_WhenTailDiffers()
    {
        // The BPE-merge case: the generation prompt merged with the preceding text, so its own tokens are not the tail.
        int[] prompt = [.. Ids(1, 18), 19, 99];

        Assert.Null(RecurrentSnapshotBoundary.FromGenerationSuffix(prompt, Tok, "19,20"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void FromGenerationSuffix_ReturnsNull_ForNullOrEmptySuffix(string? suffix)
        => Assert.Null(RecurrentSnapshotBoundary.FromGenerationSuffix(Ids(1, 20), Tok, suffix));

    [Fact]
    public void FromGenerationSuffix_ReturnsNull_WhenTheSuffixIsTheWholePrompt()
        => Assert.Null(RecurrentSnapshotBoundary.FromGenerationSuffix(Ids(19, 2), Tok, "19,20"));

    [Fact]
    public void FromGenerationSuffix_ReturnsNull_WhenTheSuffixIsLongerThanThePrompt()
        => Assert.Null(RecurrentSnapshotBoundary.FromGenerationSuffix(Ids(20, 1), Tok, "19,20"));

    [Fact]
    public void Choose_TakesTheLaterBoundary()
    {
        Assert.Equal(30, RecurrentSnapshotBoundary.Choose(30, 25, 40));
        Assert.Equal(25, RecurrentSnapshotBoundary.Choose(10, 25, 40));
        Assert.Equal(10, RecurrentSnapshotBoundary.Choose(10, null, 40));
    }

    [Fact]
    public void Choose_CapsAtPromptLenMinusOne()
    {
        Assert.Equal(39, RecurrentSnapshotBoundary.Choose(45, null, 40));
        Assert.Equal(39, RecurrentSnapshotBoundary.Choose(10, 45, 40));
    }

    [Fact]
    public void ForContinuation_KeepsTheSuffixBoundary_OnlyWhenThePromptContinuesThePreviousOne()
    {
        // A chat turn shares the whole previous prompt up to its generation prompt; a classifier request shares only the
        // system prompt, and snapshotting it at its own generation prompt would never hit.
        Assert.Equal(62, RecurrentSnapshotBoundary.ForContinuation(62, shared: 22, previousSuffixBoundary: 22));
        Assert.Null(RecurrentSnapshotBoundary.ForContinuation(62, shared: 20, previousSuffixBoundary: 22));
        Assert.Null(RecurrentSnapshotBoundary.ForContinuation(62, shared: 22, previousSuffixBoundary: null));
        Assert.Null(RecurrentSnapshotBoundary.ForContinuation(null, shared: 22, previousSuffixBoundary: 22));
    }

    private sealed class CsvTokenizer : ITokenizer
    {
        public int VocabSize => 128;
        public int BosTokenId => 1;
        public int EosTokenId => 127;
        public int[] Encode(string text)
            => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => Encode(text).Length;
    }
}
