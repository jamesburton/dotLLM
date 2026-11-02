using System.Text;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// E4c: the text-first fixed-size chunker. The file text is cut first (at a line end when a whole line fits, otherwise at a
/// character boundary that splits neither a surrogate pair — i.e. a multi-byte UTF-8 sequence — nor a CRLF), and each piece is
/// tokenised alone, so a server that receives the piece as text reproduces exactly the ids the encoder used.
/// </summary>
public sealed class TextFirstFixedChunkerTests
{
    private static int[] Raw(ITokenizer tokenizer, string text) => tokenizer.EncodeRaw(text);

    private static string Fields(int count, string eol = "\n")
    {
        var sb = new StringBuilder();
        for (int i = 0; i < count; i++) sb.Append("    int field").Append(i).Append(" = ").Append(i * 7).Append(';').Append(eol);
        return sb.ToString();
    }

    private static void AssertCoversFile(SourceFile file, IReadOnlyList<Chunk> chunks)
    {
        Assert.Equal(file.Text, string.Concat(chunks.Select(c => file.Text[c.StartOffset..c.EndOffset])));
        Assert.Equal(0, chunks[0].StartOffset);
        Assert.Equal(file.Text.Length, chunks[^1].EndOffset);
        for (int k = 1; k < chunks.Count; k++) Assert.Equal(chunks[k - 1].EndOffset, chunks[k].StartOffset);
        foreach (Chunk c in chunks)
        {
            Assert.True(c.EndOffset > c.StartOffset);
            Assert.Equal(file.Path, c.FilePath);
            Assert.Equal(1 + file.Text[..c.StartOffset].Count(ch => ch == '\n'), c.StartLine);
            Assert.Equal(1 + file.Text[..(c.EndOffset - 1)].Count(ch => ch == '\n'), c.EndLine);
        }
    }

    [Fact]
    public void Name_And_Limit()
    {
        var chunker = new TextFirstFixedChunker(new CharTokenizer());
        Assert.Equal("fixed-text-512", chunker.Name);
        Assert.Equal(512, chunker.MaxChunkTokens);
        Assert.Equal("fixed-text-64", new TextFirstFixedChunker(new CharTokenizer(), 64).Name);
    }

    [Fact]
    public void EmptyFile_NoChunks_ShortFile_OneChunk()
    {
        var chunker = new TextFirstFixedChunker(new CharTokenizer(), 64);
        Assert.Empty(chunker.Chunk(new SourceFile("e.cs", "")));
        IReadOnlyList<Chunk> one = chunker.Chunk(new SourceFile("s.cs", "int x;\nint y;"));
        Assert.Single(one);
        Assert.Equal("int x;\nint y;".Select(ch => (int)ch), one[0].TokenIds);
        Assert.Equal((1, 2), (one[0].StartLine, one[0].EndLine));
    }

    [Fact]
    public void ARestOfExactlyNTokens_IsOneChunk()
    {
        // 33 chars with line ends inside: with n = 33 the whole file is one chunk (not cut back to an inner line end).
        string text = "int a = 1;\nint b = 2;\nint c = 3;\n";
        Assert.Equal(33, text.Length);
        var chunker = new TextFirstFixedChunker(new CharTokenizer(), 33);
        Assert.Single(chunker.Chunk(new SourceFile("x.cs", text)));
        IReadOnlyList<Chunk> two = new TextFirstFixedChunker(new CharTokenizer(), 32).Chunk(new SourceFile("x.cs", text));
        Assert.Equal(new[] { 22, 33 }, two.Select(c => c.EndOffset));
    }

    [Fact]
    public void ARestOfNPlusOneTokens_IsCutAfterN()
    {
        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(new CharTokenizer(), 4).Chunk(new SourceFile("r.cs", "abcde"));
        Assert.Equal(new[] { 4, 5 }, chunks.Select(c => c.EndOffset));
    }

    [Fact]
    public void NonMonotoneTokenCounts_ALineEndWhosePieceIsOverTheLimit_IsNotUsed()
    {
        // A bare '\n' costs 3 tokens, '\n' + letter only 1: "aaa\n" (6 tokens) is longer in tokens than "aaa\nzb" (5).
        var tok = new NewlineCostTokenizer();
        var file = new SourceFile("n.cs", "aaa\nzbbbbbbb");
        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, 5).Chunk(file);
        AssertCoversFile(file, chunks);
        Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, 5));
        Assert.All(chunks, c => Assert.Equal(Raw(tok, file.Text[c.StartOffset..c.EndOffset]), c.TokenIds));
        Assert.Equal(6, chunks[0].EndOffset);
    }

    /// <summary>One token per char, except '\n': merged with a following letter into one token, else 3 tokens.</summary>
    private sealed class NewlineCostTokenizer : ITokenizer
    {
        public int[] Encode(string text)
        {
            var ids = new List<int>();
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] != '\n') ids.Add(text[i]);
                else if (i + 1 < text.Length && char.IsLetter(text[i + 1])) ids.Add(0x10000 + text[++i]);
                else ids.AddRange([0x20000, 0x20001, 0x20001]);
            }
            return ids.ToArray();
        }

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new StringBuilder();
            foreach (int id in tokenIds)
                sb.Append(id switch { 0x20000 => "\n", 0x20001 => "", >= 0x10000 => "\n" + (char)(id - 0x10000), _ => ((char)id).ToString() });
            return sb.ToString();
        }

        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int BosTokenId => -1;
        public int EosTokenId => -1;
        public int VocabSize => 0x20002;
        public int CountTokens(string text) => Encode(text).Length;
    }

    [Theory]
    [InlineData(16)]
    [InlineData(40)]
    [InlineData(64)]
    [InlineData(100)]
    public void Chunks_ConcatenateToFile_EachPieceAloneAtMostN_AndIdsAreThePieceTokenisedAlone(int n)
    {
        var tok = new GluingTokenizer();
        var file = new SourceFile("f.cs", "namespace N;\n\npublic class C\n{\n" + Fields(30) + "    public int Sum() => field1 + field2;\n}\n");

        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, n).Chunk(file);

        Assert.True(chunks.Count > 1);
        AssertCoversFile(file, chunks);
        foreach (Chunk c in chunks)
        {
            string text = file.Text[c.StartOffset..c.EndOffset];
            Assert.InRange(c.TokenIds.Length, 1, n);
            Assert.Equal(Raw(tok, text), c.TokenIds);          // round trip: the wire re-tokenises the text alone
            Assert.Equal(text, tok.Decode(c.TokenIds));
        }
    }

    [Fact]
    public void Cuts_AtTheLastLineEndThatFits_Greedily()
    {
        var tok = new CharTokenizer();
        const int n = 64;
        var file = new SourceFile("g.cs", Fields(40));   // every line is 18-21 chars, so several fit in 64

        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, n).Chunk(file);

        AssertCoversFile(file, chunks);
        foreach (Chunk c in chunks.SkipLast(1))
        {
            Assert.Equal('\n', file.Text[c.EndOffset - 1]);                       // ends at a line end
            int nextEol = file.Text.IndexOf('\n', c.EndOffset);
            Assert.True(nextEol + 1 - c.StartOffset > n);                         // and the next whole line would not fit
        }
    }

    [Fact]
    public void LineEnd_IsPreferredOverAFullerCharacterCut()
    {
        var tok = new CharTokenizer();
        // A 10-char line, then a 30-char line: with n = 20 the first piece is the first line (10 chars), not 20 chars.
        string text = "abcdefghi\n" + new string('x', 29) + "\n";
        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, 20).Chunk(new SourceFile("p.cs", text));

        AssertCoversFile(new SourceFile("p.cs", text), chunks);
        Assert.Equal(10, chunks[0].EndOffset);
        Assert.Equal(new[] { 10, 30, 40 }, chunks.Select(c => c.EndOffset));      // the long line is cut at 20 chars
    }

    [Fact]
    public void OverlongLine_IsCutAtTheLastCharacterThatFits()
    {
        var tok = new CharTokenizer();
        string text = new string('y', 50) + "\nend\n";
        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, 16).Chunk(new SourceFile("o.cs", text));

        AssertCoversFile(new SourceFile("o.cs", text), chunks);
        Assert.Equal(new[] { 16, 32, 48, 55 }, chunks.Select(c => c.EndOffset));   // the rest fits whole
        Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, 16));
    }

    [Theory]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(9)]
    public void MultiByteCharacters_AreNeverSplit(int n)
    {
        var tok = new ByteTokenizer();   // one token per UTF-8 byte: a cut by count alone would land inside characters
        string text = "é€😀a😀€é\n" + string.Concat(Enumerable.Repeat("x😀é€", 6)) + "\nfin\n";
        var file = new SourceFile("u.cs", text);

        IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, n).Chunk(file);

        AssertCoversFile(file, chunks);
        Assert.Equal(Encoding.UTF8.GetBytes(text), chunks.SelectMany(c => Encoding.UTF8.GetBytes(file.Text[c.StartOffset..c.EndOffset])).ToArray());
        foreach (Chunk c in chunks)
        {
            string piece = file.Text[c.StartOffset..c.EndOffset];
            Assert.False(char.IsLowSurrogate(piece[0]));
            Assert.False(char.IsHighSurrogate(piece[^1]));
            Assert.InRange(c.TokenIds.Length, 1, n);
            Assert.Equal(Raw(tok, piece), c.TokenIds);
            Assert.DoesNotContain('�', tok.Decode(c.TokenIds));
        }
        // Greedy at character cuts too: the next code point would not have fitted.
        foreach (Chunk c in chunks.SkipLast(1).Where(c => file.Text[c.EndOffset - 1] != '\n'))
        {
            int next = char.IsHighSurrogate(file.Text[c.EndOffset]) ? 2 : 1;
            Assert.True(Encoding.UTF8.GetByteCount(file.Text[c.StartOffset..(c.EndOffset + next)]) > n);
        }
    }

    [Fact]
    public void Crlf_IsPreserved_AndNeverSplit()
    {
        var tok = new CharTokenizer();
        var file = new SourceFile("w.cs", Fields(20, "\r\n") + new string('z', 40) + "\r\n" + "tail");

        foreach (int n in new[] { 16, 23, 30, 64 })
        {
            IReadOnlyList<Chunk> chunks = new TextFirstFixedChunker(tok, n).Chunk(file);
            AssertCoversFile(file, chunks);
            Assert.Contains(chunks, c => file.Text[c.StartOffset..c.EndOffset].Contains("\r\n", StringComparison.Ordinal));
            foreach (Chunk c in chunks.Skip(1)) Assert.NotEqual('\n', file.Text[c.StartOffset]);   // no cut between \r and \n
            Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, n));
        }
        // A character cut that would fall between '\r' and '\n' backs off by one.
        IReadOnlyList<Chunk> small = new TextFirstFixedChunker(tok, 5).Chunk(new SourceFile("c.cs", "aaaa\r\nb"));
        Assert.Equal(new[] { 4, 7 }, small.Select(c => c.EndOffset));
    }

    [Fact]
    public void Deterministic()
    {
        var file = new SourceFile("d.cs", Fields(25) + "é€😀\n" + new string('q', 70) + "\n");
        IReadOnlyList<Chunk> a = new TextFirstFixedChunker(new ByteTokenizer(), 32).Chunk(file);
        IReadOnlyList<Chunk> b = new TextFirstFixedChunker(new ByteTokenizer(), 32).Chunk(file);
        Assert.Equal(a.Select(c => c.Id), b.Select(c => c.Id));
        Assert.Equal(a.SelectMany(c => c.TokenIds), b.SelectMany(c => c.TokenIds));
    }

    [Fact]
    public void ACharacterLongerThanTheLimit_Throws()
    {
        // Under a byte tokenizer '😀' is 4 tokens; a 3-token limit cannot hold it without splitting it.
        Assert.Throws<InvalidOperationException>(() => new TextFirstFixedChunker(new ByteTokenizer(), 3).Chunk(new SourceFile("t.cs", "ab😀")));
    }

    [Fact]
    public void IdSlicedChunks_CanFailTheTextRoundTrip_TextFirstChunksDoNot()
    {
        // The motivation (E4c): FixedSizeChunker slices the file's ids, so a slice can end inside a multi-byte character; its
        // decoded text (what a wire client sends) re-tokenises to different ids. The text-first chunks always round-trip.
        var tok = new ByteTokenizer();
        var file = new SourceFile("m.cs", "ab€cd€ef€gh€\n");
        IReadOnlyList<Chunk> sliced = new FixedSizeChunker(tok, 4).Chunk(file);
        Assert.Contains(sliced, c => !Raw(tok, tok.Decode(c.TokenIds)).SequenceEqual(c.TokenIds));

        IReadOnlyList<Chunk> textFirst = new TextFirstFixedChunker(tok, 4).Chunk(file);
        Assert.All(textFirst, c => Assert.Equal(Raw(tok, tok.Decode(c.TokenIds)), c.TokenIds));
    }

    [Fact]
    public void Corpus_UsesTheChunker_AndTheHashNamesIt()
    {
        var tok = new CharTokenizer();
        var files = new[] { new SourceFile("a.cs", Fields(10)), new SourceFile("b.cs", Fields(3)) };
        var corpus = new Corpus(files, new TextFirstFixedChunker(tok, 64), tok);
        var other = new Corpus(files, new FixedSizeChunker(tok, 64), tok);
        Assert.All(corpus.Chunks, c => Assert.Equal(Raw(tok, corpus.TextOf(c)), c.TokenIds));
        Assert.NotEqual(other.ChunkIdHash, corpus.ChunkIdHash);
    }
}
