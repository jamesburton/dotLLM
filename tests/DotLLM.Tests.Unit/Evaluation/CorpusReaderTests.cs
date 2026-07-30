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
}
