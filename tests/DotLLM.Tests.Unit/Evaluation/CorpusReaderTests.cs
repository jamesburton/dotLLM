using DotLLM.Engine.Evaluation;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class CorpusReaderTests
{
    // One token per whitespace-separated word; ids are word lengths, so order is checkable.
    private sealed class WordTokenizer : ITokenizer
    {
        public int[] Encode(string text) =>
            text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => w.Length).ToArray();

        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string DecodeToken(int tokenId) => throw new NotSupportedException();
        public int CountTokens(string text) => Encode(text).Length;
        public int VocabSize => 1024;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
    }

    [Fact]
    public void StreamTokens_ProducesTokensInOrder()
    {
        using var reader = new StringReader("a bb ccc dddd");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer()).ToArray();
        Assert.Equal([1, 2, 3, 4], tokens);
    }

    [Fact]
    public void StreamTokens_HonoursMaxTokens()
    {
        using var reader = new StringReader("a bb ccc dddd eeeee");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 3).ToArray();
        Assert.Equal([1, 2, 3], tokens);
    }

    [Fact]
    public void StreamTokens_DoesNotSplitTokensAcrossChunkBoundaries()
    {
        // A tiny chunk size forces the boundary case: "ccc" must not become "c" + "cc".
        using var reader = new StringReader("a bb ccc dddd eeeee ffffff");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 0, charChunkSize: 4).ToArray();
        Assert.Equal([1, 2, 3, 4, 5, 6], tokens);
    }

    // ──────────────────── BOS prepending (issue #516) ────────────────────

    /// <remarks>
    /// llama.cpp prepends BOS to the whole stream before chunking when the vocab asks for it.
    /// dotLLM did not, so on Llama-3.2 every chunk boundary sat one token off llama.cpp's and the
    /// two engines scored different text — issue #515, and the reason this parameter exists.
    /// </remarks>
    [Fact]
    public void StreamTokens_WithBosTokenId_PrependsItOnce()
    {
        using var reader = new StringReader("a bb ccc");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), bosTokenId: 7).ToArray();
        Assert.Equal([7, 1, 2, 3], tokens);
    }

    [Fact]
    public void StreamTokens_WithoutBosTokenId_PrependsNothing()
    {
        using var reader = new StringReader("a bb ccc");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), bosTokenId: -1).ToArray();
        Assert.Equal([1, 2, 3], tokens);
    }

    /// <remarks>
    /// BOS counts toward the cap, exactly as it does in llama.cpp where it is simply the first
    /// token of the stream. If it did not, <c>--max-tokens 32768</c> would cover a different span
    /// of corpus than llama.cpp's first 64 chunks — reintroducing the offset this fixes.
    /// </remarks>
    [Fact]
    public void StreamTokens_BosCountsTowardMaxTokens()
    {
        using var reader = new StringReader("a bb ccc dddd");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 3, bosTokenId: 7).ToArray();
        Assert.Equal([7, 1, 2], tokens);
    }

    [Fact]
    public void StreamTokens_BosAloneWhenMaxTokensIsOne()
    {
        using var reader = new StringReader("a bb ccc");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 1, bosTokenId: 7).ToArray();
        Assert.Equal([7], tokens);
    }
}
