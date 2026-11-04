using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase I Task 19 (e2e findings F-A1 / F-A2): chunk edges must be token-stable, i.e. the tokenisation of two adjacent chunks
/// shown back to back equals the concatenation of their own tokenisations, for both chunkers. <see cref="NewlineRunTokenizer"/>
/// merges newline runs into one token, like the real BPEs do.
/// </summary>
public sealed class ChunkBoundaryTests
{
    // LF whatever the checkout's line endings are: NewlineRunTokenizer merges '\n' runs only.
    private static readonly string Source = """
        namespace Demo;

        /// <summary>A holder.</summary>
        public sealed class C
        {
            public int A() => 1;

            /// <summary>Two.</summary>
            public int B() => 2;


            public int D() => 3;

            public int E() => 4;
        }

        """.ReplaceLineEndings("\n");

    private static string Text(SourceFile file, Chunk c) => file.Text[c.StartOffset..c.EndOffset];

    private static void AssertEdgesStable(ITokenizer tok, SourceFile file, IReadOnlyList<Chunk> chunks, int limit)
    {
        Assert.True(chunks.Count >= 2);
        Assert.Equal(file.Text, string.Concat(chunks.Select(c => Text(file, c))));
        foreach (Chunk c in chunks)
        {
            Assert.True(c.TokenIds.Length <= limit, $"{c.Id} has {c.TokenIds.Length} tokens");
            Assert.Equal(c.TokenIds, ChunkTokenization.Encode(tok, Text(file, c)));
        }
        for (int k = 0; k + 1 < chunks.Count; k++)
        {
            string a = Text(file, chunks[k]), b = Text(file, chunks[k + 1]);
            Assert.True(ChunkBoundary.IsStable(tok, a, b),
                $"edge {k}/{k + 1} merges: ...{a[^Math.Min(a.Length, 8)..].Replace("\n", "\\n")} | {b[..Math.Min(b.Length, 8)].Replace("\n", "\\n")}...");
        }
    }

    [Fact]
    public void IsStable_DetectsANewlineMerge_AndTheJoiner()
    {
        var tok = new NewlineRunTokenizer();
        Assert.False(ChunkBoundary.IsStable(tok, "ab\n", "\ncd"));
        Assert.True(ChunkBoundary.IsStable(tok, "ab\n", "cd"));
        Assert.True(ChunkBoundary.IsStable(tok, "ab", "\ncd"));
        Assert.False(ChunkBoundary.IsStable(tok, "ab\n", "cd", "\n"));   // a trailing newline merges with an inserted "\n"
        Assert.True(ChunkBoundary.IsStable(tok, "ab", "cd", "\n"));
    }

    [Fact]
    public void IsStableAt_WindowAgreesWithTheExactCheck()
    {
        var tok = new NewlineRunTokenizer();
        string text = new string('x', 300) + "\n" + "\n" + new string('y', 300);
        Assert.False(ChunkBoundary.IsStableAt(tok, text, 301));
        Assert.True(ChunkBoundary.IsStableAt(tok, text, 300));
        Assert.True(ChunkBoundary.IsStableAt(tok, text, 0));
        Assert.True(ChunkBoundary.IsStableAt(tok, text, text.Length));
    }

    [Theory]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(11)]
    [InlineData(14)]
    [InlineData(20)]
    [InlineData(31)]
    [InlineData(10)]
    [InlineData(12)]
    [InlineData(17)]
    [InlineData(23)]
    [InlineData(27)]
    [InlineData(36)]
    public void LinesChunker_OverlongLine_CutsAreTokenStable(int n)
    {
        // One long line of words: the greedy character cut would land inside a word ("fo|obar" is not "foobar" shown back to back).
        var tok = new GluingTokenizer();
        string[] words = ["alpha", "b", "gamma_delta", "e", "zeta7", "theta", "io", "kappa_lambda_mu", "nu"];
        var file = new SourceFile("f.cs", string.Join(' ', Enumerable.Range(0, 60).Select(i => words[i * 7 % words.Length] + i)));
        AssertEdgesStable(tok, file, new LinesChunker(tok, n).Chunk(file), n);
    }

    [Theory]
    [InlineData(9)]
    [InlineData(13)]
    [InlineData(17)]
    [InlineData(22)]
    [InlineData(30)]
    public void LinesChunker_LineEndThatMergesWithTheNextLine_IsNotACut(int n)
    {
        // "\nx" is one token: a cut after a line whose successor starts with 'x' would merge across the edge.
        var tok = new NewlineXTokenizer();
        var file = new SourceFile("f.cs", string.Concat(Enumerable.Range(0, 30).Select(i => (i % 3 == 0 ? "xline" : "yline") + i + "\n")));
        AssertEdgesStable(tok, file, new LinesChunker(tok, n).Chunk(file), n);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(12)]
    [InlineData(15)]
    [InlineData(20)]
    public void LinesChunker_NoStableLineEnd_FallsBackToAStableCharacterCut(int n)
    {
        // Every line starts with 'x', so every line end merges with its successor ("\nx"): the cut must be a character cut.
        var tok = new NewlineXTokenizer();
        var file = new SourceFile("f.cs", string.Concat(Enumerable.Range(0, 20).Select(i => "xline" + i + "\n")));
        AssertEdgesStable(tok, file, new LinesChunker(tok, n).Chunk(file), n);
    }

    [Theory]
    [InlineData(6)]
    [InlineData(9)]
    [InlineData(14)]
    public void LinesChunker_LineEndBeforeABlankLine_IsNotACut(int n)
    {
        var tok = new NewlineRunTokenizer();
        var file = new SourceFile("f.cs", string.Concat(Enumerable.Range(0, 12).Select(i => $"line{i}\n\nx{i}\n\n\nyy{i}\n")));
        AssertEdgesStable(tok, file, new LinesChunker(tok, n).Chunk(file), n);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(40)]
    [InlineData(64)]
    [InlineData(512)]
    public void SymbolChunker_Edges_AreTokenStable(int n)
    {
        var tok = new NewlineRunTokenizer();
        var file = new SourceFile("f.cs", Source);
        IReadOnlyList<Chunk> chunks = new SymbolChunker(tok, n).Chunk(file);
        AssertEdgesStable(tok, file, chunks, n);
    }

    [Fact]
    public void SymbolChunker_BlankLineBeforeAMember_GoesToThePreviousChunk_NotIntoAMerge()
    {
        var tok = new NewlineRunTokenizer();
        var file = new SourceFile("f.cs", Source);
        IReadOnlyList<Chunk> chunks = new SymbolChunker(tok, 512).Chunk(file);
        AssertEdgesStable(tok, file, chunks, 512);
        // Members stay whole: no chunk after the first starts with a newline (the blank lines moved to the previous chunk).
        Assert.All(chunks.Skip(1), c => Assert.NotEqual('\n', Text(file, c)[0]));
    }

    [Fact]
    public void SymbolChunker_WhenNoLaterEdgeFits_MovesTheEdgeEarlier()
    {
        // The class line starts with 'c' (a "\nc" merge). The statements before it fill the limit, so neither moving the edge
        // later nor joining the class line to them fits: the edge moves to an earlier line start.
        var tok = new NewlineXTokenizer('c');
        var file = new SourceFile("t.cs", string.Concat(Enumerable.Range(1, 5).Select(i => $"var v{i} = {i};\n")) + "class Z { }\n");
        IReadOnlyList<Chunk> chunks = new SymbolChunker(tok, 62).Chunk(file);
        AssertEdgesStable(tok, file, chunks, 62);
        Assert.True(chunks[0].EndOffset < 60, "the edge moved into the statements");
    }

    [Fact]
    public void SymbolChunker_MovedEdge_NeverBreaksTheLimit()
    {
        var tok = new NewlineRunTokenizer();
        string text = "class K\n{\n" + string.Join("\n\n", Enumerable.Range(0, 8).Select(i => "    int M" + i + "() => " + i + ";")) + "\n}\n";
        var file = new SourceFile("g.cs", text);
        for (int n = 14; n < 60; n += 3)
        {
            IReadOnlyList<Chunk> chunks = new SymbolChunker(tok, n).Chunk(file);
            Assert.All(chunks, c => Assert.True(c.TokenIds.Length <= n || !c.WireEligible));
            Assert.Equal(text, string.Concat(chunks.Select(c => Text(file, c))));
            Assert.All(chunks.Where(c => c.WireEligible), c => Assert.Equal(c.TokenIds, ChunkTokenization.Encode(tok, Text(file, c))));
        }
    }

    [Fact]
    public void NoStableEdgeAnywhere_BothChunkers_StillCoverTheFile_WithinTheLimit_AndDeterministically()
    {
        var tok = new EndMarkerTokenizer();
        var file = new SourceFile("m.cs", string.Concat(Enumerable.Range(0, 10).Select(i => $"abc{i}\ndef{i}\n")));
        foreach (IChunker chunker in new IChunker[] { new LinesChunker(tok, 12), new SymbolChunker(tok, 12) })
        {
            IReadOnlyList<Chunk> a = chunker.Chunk(file), b = chunker.Chunk(file);
            Assert.Equal(file.Text, string.Concat(a.Select(c => Text(file, c))));
            Assert.All(a, c => Assert.True(c.TokenIds.Length <= 12));
            Assert.Equal(a.Select(c => c.Id), b.Select(c => c.Id));
        }
    }

    [Fact]
    public void WhereEveryEdgeIsAlreadyStable_NothingMoves()
    {
        var tok = new CharTokenizer();
        var file = new SourceFile("f.cs", "aa\nbb\ncc\ndd\nee\n");
        Assert.Equal(["aa\nbb\n", "cc\ndd\n", "ee\n"], new LinesChunker(tok, 7).Chunk(file).Select(c => Text(file, c)));
    }
}
