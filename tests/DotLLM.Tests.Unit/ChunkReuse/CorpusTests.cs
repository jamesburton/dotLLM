using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Code corpus for chunk-KV reuse: the fixed-size and Roslyn symbol chunkers and the corpus. A character tokenizer (one
/// token per char) keeps token counts readable. The cross-file task generator, validity filter and model-backed scorer
/// (research, fork only) are in <c>CrossFileTaskTests</c>.
/// </summary>
public sealed class CorpusTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-corpus-" + Guid.NewGuid().ToString("N"));

    public CorpusTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Symbol chunker
    // ---------------------------------------------------------------------------------------------------------------

    private const string TwoClasses =
        "using System;\n" +                  // 1
        "\n" +                               // 2
        "namespace N;\n" +                   // 3
        "\n" +                               // 4
        "/// <summary>A.</summary>\n" +      // 5
        "public class A\n" +                 // 6
        "{\n" +                              // 7
        "    private int _x;\n" +            // 8
        "\n" +                               // 9
        "    public void M1()\n" +           // 10
        "    {\n" +                          // 11
        "        _x++;\n" +                  // 12
        "    }\n" +                          // 13
        "\n" +                               // 14
        "    // second\n" +                  // 15
        "    public int M2() => _x;\n" +     // 16
        "}\n" +                              // 17
        "\n" +                               // 18
        "public class B\n" +                 // 19
        "{\n" +                              // 20
        "    public void M3() { }\n" +       // 21
        "}\n";                               // 22

    [Fact]
    public void SymbolChunker_SplitsAtMethodAndClassBoundaries()
    {
        var chunker = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 4096);
        IReadOnlyList<Chunk> chunks = chunker.Chunk(new SourceFile("N/A.cs", TwoClasses));

        // Usings, the namespace line and A's header carry no member, so they glue forward to A's first member (the field);
        // each method starts a chunk with its leading blank line and comment; a type's closing brace stays with its last
        // member; class B starts a new chunk (its header glued to M3).
        Assert.Equal([(1, 8), (9, 13), (14, 17), (18, 22)], chunks.Select(c => (c.StartLine, c.EndLine)).ToArray());
        Assert.All(chunks, c => Assert.Equal("N/A.cs", c.FilePath));
        Assert.Contains("void M1()", Text(chunks[1]));
        Assert.Contains("int M2()", Text(chunks[2]));
        Assert.StartsWith("\npublic class B", Text(chunks[3]));
    }

    public static TheoryData<string> CoverageSamples => new()
    {
        "two-classes", "parse-error", "nested", "partial", "top-level", "properties", "oversized", "crlf-free-no-final-newline",
    };

    [Theory]
    [MemberData(nameof(CoverageSamples))]
    public void SymbolChunker_ChunksCoverFileWithoutOverlap(string sample)
    {
        string text = Sample(sample);
        var chunker = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 160);
        IReadOnlyList<Chunk> chunks = chunker.Chunk(new SourceFile("f.cs", text));

        AssertCoversWithoutOverlap(text, chunks);
        Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, 160));
    }

    [Fact]
    public void SymbolChunker_ConsecutiveFields_FormOneChunk_PropertiesSplit()
    {
        string text = Sample("properties");
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096).Chunk(new SourceFile("p.cs", text));

        AssertCoversWithoutOverlap(text, chunks);
        Chunk fields = Assert.Single(chunks, c => Text(c).Contains("private int _a;", StringComparison.Ordinal));
        Assert.Contains("private int _b;", Text(fields));
        Assert.DoesNotContain("public int A", Text(fields));
        Chunk a = Assert.Single(chunks, c => Text(c).Contains("public int A {", StringComparison.Ordinal));
        Assert.DoesNotContain("public int B", Text(a));
    }

    [Fact]
    public void SymbolChunker_FileWithParseErrors_StillCoversFile_AndSplitsIntactMembers()
    {
        string text = Sample("parse-error");
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096).Chunk(new SourceFile("e.cs", text));

        AssertCoversWithoutOverlap(text, chunks);
        // The intact method after the broken one still gets its own chunk.
        Assert.Contains(chunks, c => Text(c).Contains("public int Intact()", StringComparison.Ordinal)
                                     && !Text(c).Contains("Broken", StringComparison.Ordinal));
    }

    [Fact]
    public void SymbolChunker_NestedType_MembersAreSeparateChunks()
    {
        string text = Sample("nested");
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096).Chunk(new SourceFile("n.cs", text));

        AssertCoversWithoutOverlap(text, chunks);
        Chunk outerM = Assert.Single(chunks, c => Text(c).Contains("void OuterM()", StringComparison.Ordinal));
        Chunk innerA = Assert.Single(chunks, c => Text(c).Contains("void InnerA()", StringComparison.Ordinal));
        Chunk innerB = Assert.Single(chunks, c => Text(c).Contains("void InnerB()", StringComparison.Ordinal));
        Assert.Equal(3, new[] { outerM, innerA, innerB }.Distinct().Count());
        // The nested type's header glues to its first member, not to the outer method before it.
        Assert.Contains("class Inner", Text(innerA));
        Assert.DoesNotContain("class Inner", Text(outerM));
    }

    [Fact]
    public void SymbolChunker_PartialClass_EachPartChunkedInItsOwnFile()
    {
        var chunker = new SymbolChunker(new CharTokenizer(), 4096);
        string[] parts = Sample("partial").Split("//--split--\n");
        IReadOnlyList<Chunk> a = chunker.Chunk(new SourceFile("P1.cs", parts[0]));
        IReadOnlyList<Chunk> b = chunker.Chunk(new SourceFile("P2.cs", parts[1]));

        AssertCoversWithoutOverlap(parts[0], a);
        AssertCoversWithoutOverlap(parts[1], b);
        Assert.Equal(2, a.Count); // Compute, Helper
        Assert.Single(b);         // Use
        Assert.All(a, c => Assert.Equal("P1.cs", c.FilePath));
    }

    [Fact]
    public void SymbolChunker_TopLevelStatements_GroupedIntoOneChunk_TypesAfterThemSplit()
    {
        string text = Sample("top-level");
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096).Chunk(new SourceFile("Program.cs", text));

        AssertCoversWithoutOverlap(text, chunks);
        Assert.Equal(2, chunks.Count);
        Assert.Contains("Console.WriteLine(\"a\");", Text(chunks[0]));
        Assert.Contains("Console.WriteLine(\"c\");", Text(chunks[0]));
        Assert.Contains("static int Local(", Text(chunks[0]));
        Assert.Contains("class Tail", Text(chunks[1]));
    }

    [Fact]
    public void SymbolChunker_EmptyFile_NoChunks()
    {
        Assert.Empty(new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("empty.cs", "")));
        Assert.Empty(new FixedSizeChunker(new CharTokenizer(), 512).Chunk(new SourceFile("empty.cs", "")));
    }

    [Fact]
    public void SymbolChunker_SymbolLargerThanLimit_SplitAtStatementBoundaries_NoChunkOverLimit()
    {
        string text = Sample("oversized");
        var file = new SourceFile("big.cs", text);
        const int limit = 160;
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), limit).Chunk(file);

        AssertCoversWithoutOverlap(text, chunks);
        Assert.All(chunks, c => Assert.True(c.TokenIds.Length <= limit, $"chunk {c.StartLine}-{c.EndLine} has {c.TokenIds.Length} tokens"));
        string[] lines = text.Split('\n');
        // Every chunk inside Big() (lines with a statement) starts at a statement or member line, never mid-statement.
        Chunk[] bigChunks = chunks.Where(c => Text(c).Contains("Step", StringComparison.Ordinal)).ToArray();
        Assert.True(bigChunks.Length >= 3, "Big() was not split");
        foreach (Chunk c in bigChunks.Skip(1))
        {
            string first = lines[c.StartLine - 1].TrimStart();
            Assert.True(first.StartsWith("Step", StringComparison.Ordinal) || first.StartsWith("if", StringComparison.Ordinal)
                        || first.StartsWith("var", StringComparison.Ordinal),
                $"chunk at line {c.StartLine} starts mid-statement: '{first}'");
        }
        // The single line longer than the limit falls back to a token split; its pieces stay within the limit.
        Assert.Contains(chunks, c => c.StartLine == c.EndLine && c.TokenIds.Length == limit);
    }

    [Fact]
    public void SymbolChunker_ChunkOfExactlyLimitTokens_IsNotSplit_OneTokenLess_Is()
    {
        string text = "class C\n{\n    void M()\n    {\n        int a = 1;\n        int b = 2;\n    }\n}\n";
        int whole = text.Length; // one member: the whole file is one chunk
        Assert.Single(new SymbolChunker(new CharTokenizer(), whole).Chunk(new SourceFile("c.cs", text)));
        IReadOnlyList<Chunk> split = new SymbolChunker(new CharTokenizer(), whole - 1).Chunk(new SourceFile("c.cs", text));
        Assert.True(split.Count >= 2);
        AssertCoversWithoutOverlap(text, split);
    }

    [Fact]
    public void SymbolChunker_IsGenerated_DetectsMarkers()
    {
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/Foo.g.cs", "class A {}")));
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/Form1.Designer.cs", "class A {}")));
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/X.cs", "// <auto-generated>\n// x\n// </auto-generated>\nclass A {}")));
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/Y.cs",
            "[System.CodeDom.Compiler.GeneratedCode(\"tool\", \"1\")]\nclass A {}")));
        Assert.False(SymbolChunker.IsGenerated(new SourceFile("a/Z.cs", "// regular\nclass A { string s = \"auto\"; }")));
    }

    [Fact]
    public void Corpus_FromDirectory_ExcludesGeneratedTestsBinObj_NormalisesCrlf()
    {
        string root = Path.Combine(_scratch, "repo");
        Write(root, "src/Lib/A.cs", "class A\r\n{\r\n    void M() { }\r\n}\r\n");
        Write(root, "src/Lib/A.g.cs", "class G { }\n");
        Write(root, "src/Lib/obj/Debug/X.cs", "class X { }\n");
        Write(root, "src/Lib/bin/Y.cs", "class Y { }\n");
        Write(root, "tests/Lib.Tests/T.cs", "class T { }\n");
        Write(root, "src/Lib.Tests/U.cs", "class U { }\n");

        var corpus = Corpus.FromDirectory(root, new SymbolChunker(new CharTokenizer(), 512), new CharTokenizer());
        SourceFile only = Assert.Single(corpus.Files);
        Assert.Equal("src/Lib/A.cs", only.Path);
        Assert.DoesNotContain('\r', only.Text);
        Assert.Contains(corpus.Excluded, e => e.Path == "src/Lib/A.g.cs" && e.Reason == "generated");
        Assert.Contains(corpus.Excluded, e => e.Path == "tests/Lib.Tests/T.cs" && e.Reason == "test");
        Assert.Contains(corpus.Excluded, e => e.Path == "src/Lib.Tests/U.cs" && e.Reason == "test");
        Assert.Matches("^[0-9a-f]{64}$", corpus.ChunkIdHash);
    }

    [Fact]
    public void Corpus_UsesTheChunker_AndTheHashNamesIt()
    {
        // Moved from LinesChunkerTests (the corpus lives in DotLLM.ChunkReuse.CSharp).
        ITokenizer tok = new CharTokenizer();
        var files = new[] { new SourceFile("a.cs", LinesChunkerTests.Fields(10)), new SourceFile("b.cs", LinesChunkerTests.Fields(3)) };
        var corpus = new Corpus(files, new LinesChunker(tok, 64), tok);
        var other = new Corpus(files, new FixedSizeChunker(tok, 64), tok);
        Assert.All(corpus.Chunks, c => Assert.Equal(tok.EncodeRaw(corpus.TextOf(c)), c.TokenIds));
        Assert.NotEqual(other.ChunkIdHash, corpus.ChunkIdHash);
    }

    [Fact]
    public void SymbolChunker_IsGenerated_DetectsEverySuffix()
    {
        foreach (string name in new[] { "a/X.g.cs", "a/X.G.CS", "a/X.g.i.cs", "a/X.designer.cs", "a/X.generated.cs", "a/X.Generated.cs" })
            Assert.True(SymbolChunker.IsGenerated(new SourceFile(name, "class A {}")), name);
        Assert.False(SymbolChunker.IsGenerated(new SourceFile("a/Xgenerated.cs", "class A {}")));
    }

    /// <summary>A member that starts mid-line, after the tail of the previous member, does not split it (no cut inside a member).</summary>
    [Fact]
    public void SymbolChunker_MemberStartingMidLine_AfterAMemberTail_DoesNotSplit()
    {
        string text = "class C\n{\n    void A()\n    { } void B() { }\n}\n";
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("m.cs", text));
        Assert.Single(chunks);
        AssertCoversWithoutOverlap(text, chunks);
    }

    [Fact]
    public void SymbolChunker_Name_CarriesTheLimit()
    {
        Assert.Equal("symbol-512", new SymbolChunker(new CharTokenizer()).Name);
        Assert.Equal("symbol-100", new SymbolChunker(new CharTokenizer(), 100).Name);
    }

    /// <summary>A namespace that opens after the last member has a mark but no member: its text glues backward, so the file stays covered.</summary>
    [Fact]
    public void SymbolChunker_TrailingHeaderOnlyPiece_GluesBackward_FileStaysCovered()
    {
        string text = "class A\n{\n    void M() { }\n}\n\nnamespace B\n{\n}\n";
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("t.cs", text));
        AssertCoversWithoutOverlap(text, chunks);
        Assert.Equal(text.Length, chunks[^1].EndOffset);
    }

    /// <summary>Greedy packing takes a chunk of exactly the limit (the boundary is inclusive).</summary>
    [Fact]
    public void SymbolChunker_PackingFillsAChunkToExactlyTheLimit()
    {
        const string header = "class C\n{\n    void M()\n    {\n";
        const string statement = "        s00();\n";
        var sb = new System.Text.StringBuilder(header);
        for (int i = 0; i < 12; i++) sb.Append(statement.Replace("00", i.ToString("00", System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal));
        sb.Append("    }\n}\n");
        string text = sb.ToString();
        int limit = header.Length + 3 * statement.Length;
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), limit).Chunk(new SourceFile("p.cs", text));
        AssertCoversWithoutOverlap(text, chunks);
        Assert.Equal(limit, chunks[0].TokenIds.Length);
        Assert.Equal(text[..limit], Text(chunks[0]));
    }

    /// <summary>A single line over the limit is cut into slices of exactly the limit, at offsets that advance by the limit.</summary>
    [Fact]
    public void SymbolChunker_LongLine_SlicesAdvanceByTheLimit()
    {
        string line = "    string s = \"" + new string('x', 700) + "\";\n";
        string text = "class C\n{\n" + line + "}\n";
        const int limit = 100;
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), limit).Chunk(new SourceFile("l.cs", text));
        AssertCoversWithoutOverlap(text, chunks);
        Chunk[] slices = chunks.Where(c => !c.WireEligible).ToArray();
        Assert.True(slices.Length >= 7, $"{slices.Length} slices");
        foreach (Chunk c in slices[..^1])
            Assert.Equal(limit, c.EndOffset - c.StartOffset);
        for (int i = 1; i < slices.Length; i++)
            Assert.Equal(slices[i - 1].EndOffset, slices[i].StartOffset);
    }

    /// <summary>
    /// A tokenizer whose decode length disagrees with the text (replacement characters, a lossy byte-level BPE) still gives
    /// chunks that tile the file: the slice boundaries are clamped to stay within the line and never go back.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void SymbolChunker_LongLine_WithLossyDecode_StillTilesTheFile(int decodeFactor)
    {
        string line = "    string s = \"" + new string('x', 700) + "\";\n";
        string text = "class C\n{\n" + line + "}\n";
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new LossyDecodeTokenizer(decodeFactor), 100).Chunk(new SourceFile("l.cs", text));
        Assert.Equal(0, chunks[0].StartOffset);
        Assert.Equal(text.Length, chunks[^1].EndOffset);
        for (int i = 0; i < chunks.Count; i++)
        {
            Assert.True(chunks[i].StartOffset <= chunks[i].EndOffset, $"chunk {i}");
            if (i > 0) Assert.Equal(chunks[i - 1].EndOffset, chunks[i].StartOffset);
        }
    }

    /// <summary><see cref="CharTokenizer"/> whose decode returns <c>factor</c> characters per token.</summary>
    private sealed class LossyDecodeTokenizer(int factor) : ITokenizer
    {
        private readonly CharTokenizer _inner = new();
        public int[] Encode(string text) => _inner.Encode(text);
        public string Decode(ReadOnlySpan<int> tokenIds) => new('?', factor * tokenIds.Length);
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int BosTokenId => -1;
        public int EosTokenId => -1;
        public int VocabSize => 65536;
        public int CountTokens(string text) => Encode(text).Length;
    }

    [Fact]
    public void Corpus_FromDirectory_OrdersFilesAndExclusionsByOrdinalPath()
    {
        string root = Path.Combine(_scratch, "ordering");
        Write(root, "B.cs", "class B { }\n");
        Write(root, "a.cs", "class A { }\n");
        Write(root, "a.g.cs", "class GA { }\n");
        Write(root, "B.g.cs", "class GB { }\n");
        Write(root, "c.g.cs", "class GC { }\n");

        var corpus = Corpus.FromDirectory(root, new SymbolChunker(new CharTokenizer(), 512), new CharTokenizer());
        Assert.Equal(["B.cs", "a.cs"], corpus.Files.Select(f => f.Path));
        Assert.Equal(["B.g.cs", "a.g.cs", "c.g.cs"], corpus.Excluded.Select(e => e.Path));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Fixed-size chunker
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void FixedSizeChunker_EncodeStripsACharacter_ChunksTileAndMatchTheirIds()
    {
        // dotLLM's encoder strips U+FDD0 (ControlTokenGuard.Marker) on every encode, so the ids never represent it
        // (src/DotLLM.Tokenizers/ControlTokenGuard.cs in dotLLM; found on its own ControlTokenGuard.cs). Offsets derived from
        // decoded lengths then fell short: the last chunk ended before the text (ChunkIndex.Update refused the file) and
        // every chunk after the marker was shifted. Now the chunks tile the file and each chunk's text is exactly its ids'
        // text plus any stripped markers.
        const char marker = (char)0xFDD0;
        var tok = new StrippingTokenizer(marker);
        const char nl = '\n';
        string text = "class A {" + nl + "  const string S = \"" + marker + marker + "\";" + nl + "}" + nl + new string('x', 40) + nl + marker + "y" + nl;
        IReadOnlyList<Chunk> chunks = new FixedSizeChunker(tok, 16).Chunk(new SourceFile("a.cs", text));
        Assert.True(chunks.Count > 2);
        Assert.Equal(0, chunks[0].StartOffset);
        for (int i = 1; i < chunks.Count; i++) Assert.Equal(chunks[i - 1].EndOffset, chunks[i].StartOffset);
        Assert.Equal(text.Length, chunks[^1].EndOffset);
        foreach (Chunk c in chunks)
            Assert.Equal(tok.Decode(c.TokenIds), text[c.StartOffset..c.EndOffset].Replace(marker.ToString(), "", StringComparison.Ordinal));
        // PR 10 (chunk index) adds: ChunkIndex.Update refuses a non-tiling chunker; it must accept this file.

        // A marker at the very end has no decoded character to follow: only the last-chunk rule covers it.
        string trailing = text + marker;
        IReadOnlyList<Chunk> t = new FixedSizeChunker(tok, 16).Chunk(new SourceFile("b.cs", trailing));
        Assert.Equal(trailing.Length, t[^1].EndOffset);
        // PR 10 adds: ChunkIndex.Update accepts both files.
    }

    /// <summary><see cref="CharTokenizer"/> whose encode strips one character (as dotLLM's ControlTokenGuard does).</summary>
    private sealed class StrippingTokenizer(char stripped) : ITokenizer
    {
        private readonly CharTokenizer _inner = new();
        public int[] Encode(string text) => _inner.Encode(text.Replace(stripped.ToString(), "", StringComparison.Ordinal));
        public string Decode(ReadOnlySpan<int> tokenIds) => _inner.Decode(tokenIds);
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int BosTokenId => -1;
        public int EosTokenId => -1;
        public int VocabSize => 65536;
        public int CountTokens(string text) => Encode(text).Length;
    }

    [Fact]
    public void FixedSizeChunker_ProducesChunksOfAtMostN_Tokens()
    {
        var tok = new CharTokenizer();
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < 40; i++) sb.Append("    int field").Append(i).Append(" = ").Append(i * 7).Append(";\n");
        string text = sb.ToString();
        const int n = 64;

        IReadOnlyList<Chunk> chunks = new FixedSizeChunker(tok, n).Chunk(new SourceFile("f.cs", text));

        Assert.Equal(1, tok.EncodeCalls);                                     // the file is tokenised exactly once
        Assert.Equal((text.Length + n - 1) / n, chunks.Count);                // the tail is kept
        Assert.All(chunks.SkipLast(1), c => Assert.Equal(n, c.TokenIds.Length));
        Assert.InRange(chunks[^1].TokenIds.Length, 1, n);
        Assert.Equal(tok.Encode(text), chunks.SelectMany(c => c.TokenIds).ToArray());
        for (int k = 0; k < chunks.Count; k++)
        {
            int start = k * n, end = Math.Min(text.Length, start + n);
            Assert.Equal(1 + text[..start].Count(ch => ch == '\n'), chunks[k].StartLine);
            Assert.Equal(1 + text[..(end - 1)].Count(ch => ch == '\n'), chunks[k].EndLine);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------

    internal static string Text(Chunk c) => new CharTokenizer().Decode(c.TokenIds);

    private static void AssertCoversWithoutOverlap(string text, IReadOnlyList<Chunk> chunks)
    {
        Assert.NotEmpty(chunks);
        Assert.Equal(text, string.Concat(chunks.Select(Text)));
        Assert.Equal(1, chunks[0].StartLine);
        int lastLine = text.Length == 0 ? 0 : 1 + text[..^1].Count(ch => ch == '\n');
        Assert.Equal(lastLine, chunks[^1].EndLine);
        int consumed = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            Chunk c = chunks[i];
            Assert.True(c.StartLine <= c.EndLine);
            // Line numbers agree with the chunk's own text position in the file.
            Assert.Equal(1 + text[..consumed].Count(ch => ch == '\n'), c.StartLine);
            consumed += Text(c).Length;
            Assert.Equal(1 + text[..(consumed - 1)].Count(ch => ch == '\n'), c.EndLine);
            if (i > 0)
            {
                Chunk p = chunks[i - 1];
                bool tokenSplitLine = p.EndLine == c.StartLine && !Text(p).EndsWith('\n');
                Assert.True(c.StartLine == p.EndLine + 1 || tokenSplitLine, $"chunks {i - 1} and {i} overlap or leave a gap");
            }
        }
    }

    private static void Write(string root, string rel, string text)
    {
        string full = Path.Combine(root, rel.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, text);
    }

    internal static string Sample(string name) => name switch
    {
        "two-classes" => TwoClasses,
        "parse-error" =>
            "namespace E;\n\npublic class E\n{\n    public void Broken(\n    {\n        int x = ;\n    }\n\n" +
            "    public int Intact()\n    {\n        return 1;\n    }\n}\n",
        "nested" =>
            "namespace Q;\n\npublic class Outer\n{\n    public void OuterM() { }\n\n    private sealed class Inner\n    {\n" +
            "        public void InnerA() { }\n\n        public void InnerB() { }\n    }\n}\n",
        "partial" =>
            "namespace P;\n\npublic partial class P\n{\n    public int Compute(int x) => x * 2;\n\n" +
            "    private int Helper() => Compute(1);\n}\n" +
            "//--split--\n" +
            "namespace P;\n\npublic partial class P\n{\n    public int Use() => Compute(5);\n}\n",
        "top-level" =>
            "using System;\n\nConsole.WriteLine(\"a\");\nConsole.WriteLine(\"b\");\n\nstatic int Local(int v) => v;\n\n" +
            "Console.WriteLine(\"c\");\n\nclass Tail\n{\n    void M() { }\n}\n",
        "properties" =>
            "namespace R;\n\npublic record Rec(int A);\n\npublic class Props\n{\n    private int _a;\n    private int _b;\n\n" +
            "    public int A { get => _a; set => _a = value; }\n\n    public int B => _b;\n\n    public event System.Action? Changed;\n\n" +
            "    public int this[int i] => i;\n\n    public enum Kind { One, Two }\n}\n",
        "oversized" => Oversized(),
        "crlf-free-no-final-newline" => "class C\n{\n    void M() { }\n}",
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    private static string Oversized()
    {
        var sb = new System.Text.StringBuilder("namespace O;\n\npublic class Big\n{\n    public void Small() { }\n\n    public void Big()\n    {\n");
        for (int i = 0; i < 12; i++)
        {
            // Each call spans three lines, so a cut at an arbitrary line start would land mid-statement.
            sb.Append("        Step").Append(i).Append("(\n            ").Append(i).Append(",\n            ").Append(i + 1).Append(");\n");
            if (i % 4 == 3) sb.Append("        if (Flag)\n        {\n            Step(").Append(i).Append(");\n        }\n");
        }
        sb.Append("        var s = \"").Append(new string('x', 400)).Append("\";\n");
        sb.Append("    }\n}\n");
        return sb.ToString();
    }
}
