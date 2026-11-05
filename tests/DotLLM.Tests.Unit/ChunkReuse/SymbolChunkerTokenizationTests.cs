using DotLLM.ChunkReuse.CodeCorpus;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The Roslyn <see cref="SymbolChunker"/> tokenises its chunks with <see cref="ChunkTokenization"/> (Phase I Task 5b, contract
/// ruling 21). Split out of <see cref="ChunkTokenizationTests"/> (which holds the vocabulary and helpers) because the symbol
/// chunker lives in DotLLM.ChunkReuse.CSharp.
/// </summary>
public sealed class SymbolChunkerTokenizationTests
{
    private static DotLLM.Tokenizers.Bpe.BpeTokenizer Tok => ChunkTokenizationTests.Tok;
    private const string CodeWithThink = ChunkTokenizationTests.CodeWithThink;

    private static void AssertChunksUseChunkTokenization(SourceFile file, IReadOnlyList<Chunk> chunks)
        => ChunkTokenizationTests.AssertChunksUseChunkTokenization(file, chunks);

    [Fact]
    public void SymbolChunker_UsesChunkTokenization()
    {
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        AssertChunksUseChunkTokenization(file, new SymbolChunker(Tok, 512).Chunk(file));
    }

    [Fact]
    public void SymbolChunker_PackedPieces_UseChunkTokenization()
    {
        // A member over the limit whose lines all fit: re-split by Pack at statements / line starts (not sliced).
        const string code = """
            namespace Demo;

            public static class R
            {
                public static int M()
                {
                    // <think> a
                    // </think> b
                    int x = 1;
                    // <think> c
                    return x;
                }
            }

            """;
        var file = new SourceFile("R.cs", code.ReplaceLineEndings("\n"));
        Assert.All(file.Text.Split('\n'), l => Assert.True(ChunkTokenization.Encode(Tok, l + "\n").Length < 30));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(Tok, 30).Chunk(file);
        Assert.True(chunks.Count >= 3);
        Assert.All(chunks, c => Assert.True(c.WireEligible, "no slices here"));
        AssertChunksUseChunkTokenization(file, chunks);
    }

    [Fact]
    public void SymbolChunker_OverlongLineSlices_UseChunkTokenization()
    {
        // A line over the limit is cut into id slices of the whole line's ids: those ids are the marked encoding too.
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(Tok, 40).Chunk(file);
        string[] lines = file.Text.Split('\n');
        int lineNo = 1 + Array.FindIndex(lines, l => l.Contains("Strip", StringComparison.Ordinal));
        string line = lines[lineNo - 1] + "\n";
        List<Chunk> pieces = chunks.Where(c => c.StartLine == lineNo && c.EndLine == lineNo).ToList();
        Assert.True(pieces.Count >= 2);
        Assert.All(pieces, c => Assert.False(c.WireEligible));
        Assert.Equal(ChunkTokenization.Encode(Tok, line), pieces.SelectMany(c => c.TokenIds));
    }
}
