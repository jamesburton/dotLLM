using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Code corpus for chunk-KV reuse experiments: the fixed-size and Roslyn symbol chunkers, the cross-file task generator
/// (semantic resolution of a masked call site's callee to the chunk holding its definition) and the task validity filter.
/// A character tokenizer (one token per char) keeps token counts readable; the model-backed scorer runs on the synthetic
/// dense <c>qwen3</c> GGUF (no real weights).
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
        var index = new DotLLM.ChunkReuse.Store.ChunkIndex(ids => DotLLM.ChunkReuse.Store.ChunkKey.Create("fp", "tok", default, "cpu",
            DotLLM.ChunkReuse.Store.KvDtype.F32, ids));
        index.Update([new SourceFile("a.cs", text)], new FixedSizeChunker(tok, 16)); // throws on a non-tiling chunker
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
    // Cross-file task generator
    // ---------------------------------------------------------------------------------------------------------------

    private static readonly (string Path, string Text)[] TaskRepo =
    [
        ("Lib/MathUtil.cs",
            "namespace Lib;\n\npublic static class MathUtil\n{\n    public static int Add(int a, int b) => a + b;\n\n" +
            "    public static int Twice(int v)\n    {\n        return Add(v, v);\n    }\n}\n"),
        ("Lib/Text.cs",
            "namespace Lib;\n\npublic static class Text\n{\n    public static string Pad(string s, int width)\n    {\n" +
            "        return s.PadLeft(width);\n    }\n\n    public static string Upper(string s) => s.ToUpperInvariant();\n}\n"),
        ("App/Program.cs",
            "using Lib;\n\nnamespace App;\n\npublic class Calc\n{\n    public int Run(int seed)\n    {\n" +
            "        var x = MathUtil.Add(seed, 2);\n        var y = MathUtil.Twice(x);\n        return x + y;\n    }\n\n" +
            "    public string Show(int v)\n    {\n        string s = Text.Pad(v.ToString(), 8);\n        return Text.Upper(s);\n    }\n}\n"),
        ("App/More.cs",
            "using Lib;\n\nnamespace App;\n\npublic class More\n{\n    public int Go() => MathUtil.Add(40, 2);\n\n" +
            "    public string Name() => Text.Upper(\"name\");\n}\n"),
        ("Filler/F1.cs",
            "namespace Filler;\n\npublic class F1\n{\n    public int Alpha() => 1;\n\n    public int Beta() => 2;\n}\n"),
        ("Filler/F2.cs",
            "namespace Filler;\n\npublic class F2\n{\n    public int Gamma() => 3;\n\n    public int Delta() => 4;\n}\n"),
    ];

    private static Corpus BuildCorpus((string Path, string Text)[] files) =>
        new(files.Select(f => new SourceFile(f.Path, f.Text)), new SymbolChunker(new CharTokenizer(), 4096), new CharTokenizer());

    private static TaskGeneratorOptions SmallOptions => new()
    {
        ContextTokens = 100_000, ContinuationTokens = 24, PrefixTokens = 120, TasksPerContext = 1,
    };

    [Fact]
    public void TaskGenerator_RequiredChunkContainsDefinitionOfMaskedSymbol()
    {
        Corpus corpus = BuildCorpus(TaskRepo);
        IReadOnlyList<CodeTask> tasks = new CrossFileTaskGenerator(SmallOptions).Generate(corpus, count: 100, seed: 1);

        // Cross-file call sites: Add x2, Twice, Pad, Upper x2. The same-file call Add(v, v) inside Twice and the BCL calls
        // (PadLeft, ToString, ToUpperInvariant) are never tasks.
        Assert.True(tasks.Count == 6, Describe(tasks) + "\n" + new CrossFileTaskGenerator(SmallOptions).CountCallSites(corpus));
        var defs = new Dictionary<string, (string File, string Decl)>
        {
            ["Add"] = ("Lib/MathUtil.cs", "public static int Add(int a, int b)"),
            ["Twice"] = ("Lib/MathUtil.cs", "public static int Twice(int v)"),
            ["Pad"] = ("Lib/Text.cs", "public static string Pad(string s, int width)"),
            ["Upper"] = ("Lib/Text.cs", "public static string Upper(string s)"),
        };
        foreach (CodeTask t in tasks)
        {
            string callee = t.MaskedText[..t.MaskedText.IndexOf('(')];
            (string file, string decl) = defs[callee];
            Chunk req = Assert.Single(t.RequiredChunks);
            Assert.Equal(file, req.FilePath);
            Assert.Contains(decl, Text(req));
            Assert.NotEqual(t.CallSitePath, req.FilePath);
            // The other method of the same file is a different chunk: the required chunk is the definition, not the file.
            foreach (var (other, (ofile, odecl)) in defs)
                if (other != callee && ofile == file) Assert.DoesNotContain(odecl, Text(req));

            // Masked region: the prefix stops right before the callee name; the continuation starts with the call text.
            Assert.EndsWith(".", t.Prefix);
            Assert.StartsWith(t.MaskedText, t.ExpectedContinuation);
            string source = TaskRepo.Single(f => f.Path == t.CallSitePath).Text;
            string prefixBody = t.Prefix[(t.Prefix.IndexOf('\n') + 1)..];          // after the "// path" header line
            Assert.Contains(prefixBody + t.MaskedText, source);
            // File offsets of the query text and the mask.
            Assert.Equal(prefixBody, source[t.PrefixStartOffset..t.MaskStartOffset]);
            Assert.Equal(t.MaskedText, source.Substring(t.MaskStartOffset, t.MaskedText.Length));
            Assert.Equal(t.PrefixTokenIds.Length, t.MaskTokenOffset);
            Assert.Equal(t.MaskedText.Length, t.MaskTokenCount);                   // char tokenizer: one token per char
            Assert.Equal(t.ExpectedContinuation, new CharTokenizer().Decode(t.ContinuationTokenIds));
            Assert.True(t.ContinuationTokenIds.Length <= 24);

            // No context chunk comes from the call-site file (the answer would leak), none repeats the masked call.
            Assert.All(t.DistractorChunks, d => Assert.NotEqual(t.CallSitePath, d.FilePath));
            Assert.All(t.DistractorChunks, d => Assert.DoesNotContain(t.MaskedText, Text(d)));
            Assert.DoesNotContain(t.DistractorChunks, d => d.Id == req.Id);
        }
    }

    [Fact]
    public void TaskGenerator_MaskStartsOnANaturalTokenBoundary()
    {
        // The tokenizer glues '.' to a following letter (as Qwen's pre-tokenizer does with ".Add"). Splitting the query at the
        // callee name would give "." + "A" ids, which the joint text never tokenises to; the boundary must move before the dot.
        var tok = new DotMergingTokenizer();
        var corpus = new Corpus(TaskRepo.Select(f => new SourceFile(f.Path, f.Text)), new SymbolChunker(tok, 4096), tok);
        IReadOnlyList<CodeTask> tasks = new CrossFileTaskGenerator(SmallOptions).Generate(corpus, 100, seed: 1);

        Assert.Equal(6, tasks.Count);
        foreach (CodeTask t in tasks)
        {
            Assert.StartsWith(".", t.MaskedText);
            Assert.False(t.Prefix.EndsWith('.'), t.Prefix);
            string source = TaskRepo.Single(f => f.Path == t.CallSitePath).Text;
            Assert.Equal(t.Prefix[(t.Prefix.IndexOf('\n') + 1)..], source[t.PrefixStartOffset..t.MaskStartOffset]);
            Assert.Equal('.', source[t.MaskStartOffset]);
            int[] joint = tok.Encode(t.Prefix + t.ExpectedContinuation);
            Assert.Equal([.. t.PrefixTokenIds, .. t.ContinuationTokenIds], joint);
            Assert.Equal(t.MaskedText, tok.Decode(t.ContinuationTokenIds.AsSpan(0, t.MaskTokenCount))[..t.MaskedText.Length]);
        }
    }

    [Fact]
    public void TaskGenerator_SameSeed_SameTasks()
    {
        IReadOnlyList<CodeTask> a = new CrossFileTaskGenerator(SmallOptions).Generate(BuildCorpus(TaskRepo), 5, seed: 3);
        IReadOnlyList<CodeTask> b = new CrossFileTaskGenerator(SmallOptions).Generate(BuildCorpus(TaskRepo), 5, seed: 3);
        IReadOnlyList<CodeTask> c = new CrossFileTaskGenerator(SmallOptions).Generate(BuildCorpus(TaskRepo), 5, seed: 4);

        Assert.Equal(5, a.Count);
        Assert.Equal(Describe(a), Describe(b));
        Assert.NotEqual(Describe(a), Describe(c));
    }

    [Fact]
    public void TaskGenerator_DuplicateNamesInDifferentNamespaces_ResolvesSemantically()
    {
        (string, string)[] repo =
        [
            ("A/Util.cs", "namespace A;\n\npublic static class Util\n{\n    public static string Format(int v) => \"A\" + v;\n}\n"),
            ("B/Util.cs", "namespace B;\n\npublic static class Util\n{\n    public static string Format(int v) => \"B\" + v;\n}\n"),
            ("C/Use.cs", "using B;\n\nnamespace C;\n\npublic class Use\n{\n    public string F() => Util.Format(3);\n}\n"),
        ];
        IReadOnlyList<CodeTask> tasks = new CrossFileTaskGenerator(SmallOptions).Generate(BuildCorpus(repo), 10, seed: 0);

        CodeTask t = Assert.Single(tasks);
        Assert.Equal("Format(3)", t.MaskedText);
        Assert.Equal("B/Util.cs", Assert.Single(t.RequiredChunks).FilePath);
        // A's same-named definition would make the task ambiguous; it is not offered as a distractor.
        Assert.DoesNotContain(t.DistractorChunks, d => d.FilePath == "A/Util.cs");
    }

    [Fact]
    public void TaskGenerator_UnresolvedSymbolsAreSkipped()
    {
        (string, string)[] repo =
        [
            ("Lib/Known.cs", "namespace Lib;\n\npublic static class Known\n{\n    public static int Run(int v) => v;\n}\n"),
            // JsonConvert is not in the corpus or the BCL; Missing.Call does not exist; Ambig is ambiguous between two
            // namespaces brought in by usings; Over.Call(null) is an ambiguous overload (two candidate symbols, neither may be
            // guessed); the delegate call has no source definition. Only Known.Run is a task.
            ("Lib/Ambig1.cs", "namespace X1;\n\npublic static class Ambig\n{\n    public static int Go() => 1;\n}\n"),
            ("Lib/Ambig2.cs", "namespace X2;\n\npublic static class Ambig\n{\n    public static int Go() => 2;\n}\n"),
            ("Lib/Over.cs", "namespace Lib;\n\npublic static class Over\n{\n    public static int Call(string s) => 1;\n\n" +
                "    public static int Call(int[] a) => 2;\n}\n"),
            ("App/Use.cs",
                "using Lib;\nusing X1;\nusing X2;\n\nnamespace App;\n\npublic class Use\n{\n    public void G(System.Func<int> f)\n    {\n" +
                "        Newtonsoft.Json.JsonConvert.SerializeObject(1);\n        Missing.Call(2);\n        Ambig.Go();\n        Over.Call(null);\n" +
                "        f();\n        Known.Run(5);\n    }\n}\n"),
        ];
        Corpus corpus = BuildCorpus(repo);
        IReadOnlyList<CodeTask> tasks = new CrossFileTaskGenerator(SmallOptions).Generate(corpus, 10, seed: 0);

        CodeTask t = Assert.Single(tasks);
        Assert.Equal("Run(5)", t.MaskedText);
        Assert.Equal("Lib/Known.cs", Assert.Single(t.RequiredChunks).FilePath);
        CallSiteCounts counts = new CrossFileTaskGenerator(SmallOptions).CountCallSites(corpus);
        Assert.Equal(1, counts.CrossFile);
        Assert.Equal(4, counts.Unresolved); // JsonConvert, Missing, Ambig, Over (f() resolves to the BCL delegate Invoke)
    }

    [Fact]
    public void TaskGenerator_PartialClass_RequiredChunkIsInTheOtherPartsFile()
    {
        string[] parts = Sample("partial").Split("//--split--\n");
        IReadOnlyList<CodeTask> tasks = new CrossFileTaskGenerator(SmallOptions)
            .Generate(BuildCorpus([("P/P1.cs", parts[0]), ("P/P2.cs", parts[1])]), 10, seed: 0);

        CodeTask t = Assert.Single(tasks); // Use() calls Compute (other file); Helper's call to Compute is same-file
        Assert.Equal("Compute(5)", t.MaskedText);
        Chunk req = Assert.Single(t.RequiredChunks);
        Assert.Equal("P/P1.cs", req.FilePath);
        Assert.Contains("public int Compute(int x)", Text(req));
    }

    [Fact]
    public void TaskGenerator_PlansFor_KeepsTheNamedTasks_VariesDistractorsBySeed_AcrossChunkers()
    {
        // E4 re-samples contexts around a fixed task set: the same call sites (ids are seed- and chunker-independent), grouped
        // by required chunk, with seeded distractors and order.
        Corpus corpus = BuildCorpus(TaskRepo);
        var options = SmallOptions with { TasksPerContext = 4, ContextTokens = 260 };
        var gen = new CrossFileTaskGenerator(options);
        CodeTask[] all = gen.Generate(corpus, 100, seed: 1).ToArray();
        Assert.Equal(6, all.Length);
        string[] wanted = all.Where(t => !t.MaskedText.StartsWith("Pad", StringComparison.Ordinal)).Select(t => t.Id).Append("nope.cs:1:1").ToArray();

        IReadOnlyList<CodeContextPlan> a = gen.GeneratePlansFor(corpus, wanted, seed: 5);
        IReadOnlyList<CodeContextPlan> b = gen.GeneratePlansFor(corpus, wanted, seed: 5);
        Assert.Equal(wanted[..^1].Order(StringComparer.Ordinal), a.SelectMany(p => p.Tasks).Select(t => t.Id).Order(StringComparer.Ordinal));
        Assert.Equal(a.Select(p => string.Join('|', p.Chunks.Select(c => c.Id))), b.Select(p => string.Join('|', p.Chunks.Select(c => c.Id))));
        foreach (CodeContextPlan p in a)
        {
            string[] required = p.Tasks[0].RequiredChunks.Select(c => c.Id).ToArray();
            Assert.All(p.Tasks, t => Assert.Equal(required, t.RequiredChunks.Select(c => c.Id).ToArray()));
            Assert.All(required, r => Assert.Contains(p.Chunks, c => c.Id == r));
            Assert.True(p.ContextTokenIds.Length <= 260);
            Assert.True(p.Tasks.Count <= 4);
            var callFiles = p.Tasks.Select(t => t.CallSitePath).ToHashSet();
            Assert.DoesNotContain(p.Chunks, c => callFiles.Contains(c.FilePath));
            CodeTask same = all.Single(t => t.Id == p.Tasks[0].Id);
            Assert.Equal(same.PrefixTokenIds, p.Tasks[0].PrefixTokenIds);
            Assert.Equal(same.MaskedText, p.Tasks[0].MaskedText);
        }
        // One task per context: every named task gets its own plan, none repeated.
        IReadOnlyList<CodeContextPlan> single = new CrossFileTaskGenerator(options with { TasksPerContext = 1 }).GeneratePlansFor(corpus, wanted, seed: 5);
        Assert.Equal(5, single.Count);
        Assert.All(single, p => Assert.Single(p.Tasks));
        Assert.Equal(5, single.Select(p => p.Tasks[0].Id).Distinct().Count());
        // Add's two call sites share one plan (one required chunk).
        Assert.Contains(a, p => p.Tasks.Count == 2 && p.Tasks.All(t => t.MaskedText.StartsWith("Add(", StringComparison.Ordinal)));
        // Another seed keeps the tasks and required chunks but samples other distractors or another order.
        IReadOnlyList<CodeContextPlan> c = gen.GeneratePlansFor(corpus, wanted, seed: 6);
        Assert.Equal(a.Select(p => string.Join('|', p.Tasks.Select(t => t.Id))), c.Select(p => string.Join('|', p.Tasks.Select(t => t.Id))));
        Assert.NotEqual(a.Select(p => string.Join('|', p.Chunks.Select(x => x.Id))), c.Select(p => string.Join('|', p.Chunks.Select(x => x.Id))));

        // The fixed-size chunker finds the same call sites; their required chunk is the fixed chunk holding the declaration.
        var tok = new CharTokenizer();
        var fixedCorpus = new Corpus(TaskRepo.Select(f => new SourceFile(f.Path, f.Text)), new FixedSizeChunker(tok, 64), tok);
        IReadOnlyList<CodeContextPlan> f = new CrossFileTaskGenerator(options).GeneratePlansFor(fixedCorpus, wanted, seed: 5);
        Assert.Equal(wanted[..^1].Order(StringComparer.Ordinal), f.SelectMany(p => p.Tasks).Select(t => t.Id).Order(StringComparer.Ordinal));
        foreach (CodeTask t in f.SelectMany(p => p.Tasks))
        {
            Chunk req = Assert.Single(t.RequiredChunks);
            string name = t.MaskedText.TrimStart('.')[..t.MaskedText.TrimStart('.').IndexOf('(')];
            string text = TaskRepo.Single(x => x.Path == req.FilePath).Text;
            int decl = text.IndexOf(" " + name + "(", StringComparison.Ordinal) + 1;
            Assert.InRange(decl, req.StartOffset, req.EndOffset - 1);
            Assert.True(req.TokenIds.Length <= 64);
        }
    }

    [Fact]
    public void TaskGenerator_Plans_ShareOneRequiredChunk_ContextFitsBudget_AndExcludesCallSiteFiles()
    {
        Corpus corpus = BuildCorpus(TaskRepo);
        var options = SmallOptions with { TasksPerContext = 4, ContextTokens = 400 };
        IReadOnlyList<CodeContextPlan> plans = new CrossFileTaskGenerator(options).GeneratePlans(corpus, taskCount: 6, seed: 2);

        Assert.Equal(6, plans.Sum(p => p.Tasks.Count));
        foreach (CodeContextPlan p in plans)
        {
            Assert.True(p.ContextTokenIds.Length <= 400);
            Assert.Equal(p.Chunks.SelectMany(c => c.TokenIds).ToArray(), p.ContextTokenIds);
            string[] required = p.Tasks[0].RequiredChunks.Select(c => c.Id).ToArray();
            Assert.All(p.Tasks, t => Assert.Equal(required, t.RequiredChunks.Select(c => c.Id).ToArray()));
            Assert.All(p.Tasks, t => Assert.Equal(
                p.Chunks.Select(c => c.Id).Where(id => !required.Contains(id)).Order(StringComparer.Ordinal),
                t.DistractorChunks.Select(c => c.Id).Order(StringComparer.Ordinal)));
            var callFiles = p.Tasks.Select(t => t.CallSitePath).ToHashSet();
            Assert.DoesNotContain(p.Chunks, c => callFiles.Contains(c.FilePath));
            Assert.Equal(p.Chunks.Where(c => !required.Contains(c.Id)).SelectMany(c => c.TokenIds).ToArray(), p.ContextWithoutRequired);
        }
        // Add has two cross-file call sites in different files; with 4 tasks per context they share one plan.
        Assert.Contains(plans, p => p.Tasks.Count == 2 && p.Tasks.All(t => t.MaskedText.StartsWith("Add(", StringComparison.Ordinal)));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Validity filter
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void TaskValidator_KeepsOnlyTasksSolvedWithRequiredChunk_NotWithout_ByMargin()
    {
        var req = new Chunk("R.cs", 1, 1, [7, 7]);
        var d1 = new Chunk("D1.cs", 1, 1, [1]);
        var d2 = new Chunk("D2.cs", 1, 1, [2, 3]);
        double m = Math.Log(2);
        // (with: EM, mask NLL) / (without: EM, mask NLL), expected keep, expected reason
        var cases = new (bool EmW, double NllW, bool EmO, double NllO, bool Keep, string Reason)[]
        {
            (true, 1.0, false, 3.0, true, "kept"),
            (false, 1.0, false, 5.0, false, "reference-fails"),
            (true, 0.5, true, 4.0, false, "solved-without"),
            (true, 1.0, false, 1.3, false, "below-margin"),
            (true, 1.0, false, 1.0 + m, true, "kept"),
            (true, 2.0, false, 1.0, false, "below-margin"),
        };
        var tasks = cases.Select((c, i) => new CodeTask($"p{i}", "c", [req], [d1, d2])
        {
            Id = "t" + i, PrefixTokenIds = [10 + i], ContinuationTokenIds = [20, 21], MaskTokenCount = 2,
        }).ToList();
        var plan = new CodeContextPlan(0, [d2, req, d1], tasks);
        var scorer = new StubScorer(plan, cases.Select(c => (new TaskScore(c.NllW, c.NllW, c.EmW, 2, c.EmW ? 2 : 0),
                                                              new TaskScore(c.NllO, c.NllO, c.EmO, 2, c.EmO ? 2 : 0))).ToArray());

        IReadOnlyList<TaskValidation> v = new TaskValidator(scorer, marginNats: m).Validate(plan);

        Assert.Equal(cases.Select(c => c.Keep), v.Select(x => x.Keep));
        Assert.Equal(cases.Select(c => c.Reason), v.Select(x => x.Reason));
        Assert.Equal(2, scorer.Calls); // one WITH pass and one WITHOUT pass for the whole plan
        Assert.Equal(cases[0].NllO - cases[0].NllW, v[0].NllGain, 12);
    }

    [Fact]
    public void ModelTaskScorer_NllAndExactMatch_EqualSingleStepForwards()
    {
        string path = Path.Combine(_scratch, "qwen3.gguf");
        File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        var geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);

        int[] context = [3, 17, 5, 29, 11, 40, 8, 22, 1, 35];
        int[] prefixA = [12, 7, 44];
        int[] prefixB = [31, 26];
        // Query A's continuation is the model's own greedy continuation (exact match must be true); B's is arbitrary.
        int[] contA = Greedy(model, geometry, [.. context, .. prefixA], 5);
        int[] contB = [9, 14, 38, 0, 45];
        var scorer = new ModelTaskScorer(model, cap => new SimpleKvCache(geometry, cap), prefillBlock: 4);

        IReadOnlyList<TaskScore> scores = scorer.Score(context,
            [new TaskQuery(prefixA, contA, MaskTokens: 3), new TaskQuery(prefixB, contB, MaskTokens: 4)]);

        foreach (var (prefix, cont, mask, s) in new[] { (prefixA, contA, 3, scores[0]), (prefixB, contB, 4, scores[1]) })
        {
            double[] nll = SingleStepNll(model, geometry, [.. context, .. prefix], cont, out bool[] argmaxHit);
            Assert.Equal(nll.Take(mask).Sum(), s.MaskNll, 4);
            Assert.Equal(nll.Sum(), s.ContinuationNll, 4);
            Assert.Equal(argmaxHit.Take(mask).All(h => h), s.MaskExactMatch);
            Assert.Equal(argmaxHit.TakeWhile(h => h).Count(), s.GreedyMatchedTokens);
        }
        Assert.True(scores[0].MaskExactMatch);
        Assert.False(scores[1].MaskExactMatch);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------

    private static string Text(Chunk c) => new CharTokenizer().Decode(c.TokenIds);

    private static string Describe(IReadOnlyList<CodeTask> tasks) => string.Join("\n", tasks.Select(t =>
        $"{t.Id}|{t.Prefix}|{t.ExpectedContinuation}|{string.Join(',', t.RequiredChunks.Select(c => c.Id))}|" +
        $"{string.Join(',', t.DistractorChunks.Select(c => c.Id))}"));

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

    private static string Sample(string name) => name switch
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

    private static int[] Greedy(TransformerModel model, KvGeometry geometry, int[] ids, int n)
    {
        using var cache = new SimpleKvCache(geometry, SyntheticDenseGguf.Context);
        var result = new int[n];
        float[] last = Forward(model, cache, ids, 0);
        int pos = ids.Length;
        for (int i = 0; i < n; i++)
        {
            result[i] = ArgMax(last);
            last = Forward(model, cache, [result[i]], pos++);
        }
        return result;
    }

    private static double[] SingleStepNll(TransformerModel model, KvGeometry geometry, int[] ids, int[] cont, out bool[] argmaxHit)
    {
        using var cache = new SimpleKvCache(geometry, SyntheticDenseGguf.Context);
        var nll = new double[cont.Length];
        argmaxHit = new bool[cont.Length];
        float[] last = [];
        for (int i = 0; i < ids.Length; i++) last = Forward(model, cache, [ids[i]], i);
        int pos = ids.Length;
        for (int j = 0; j < cont.Length; j++)
        {
            double max = last.Max();
            double lse = max + Math.Log(last.Sum(v => Math.Exp(v - max)));
            nll[j] = lse - last[cont[j]];
            argmaxHit[j] = ArgMax(last) == cont[j];
            last = Forward(model, cache, [cont[j]], pos++);
        }
        return nll;
    }

    private static unsafe float[] Forward(TransformerModel model, SimpleKvCache cache, int[] ids, int firstPos)
    {
        int[] positions = Enumerable.Range(firstPos, ids.Length).ToArray();
        using var logits = model.Forward(ids, positions, -1, cache, lastTokenLogitsOnly: true);
        int vocab = logits.Shape[1];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>Char tokenizer that merges '.' + letter into one token (id 0x20000 + letter).</summary>
    private sealed class DotMergingTokenizer : ITokenizer
    {
        public int[] Encode(string text)
        {
            var ids = new List<int>(text.Length);
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == '.' && i + 1 < text.Length && char.IsLetter(text[i + 1])) ids.Add(0x20000 + text[++i]);
                else ids.Add(text[i]);
            }
            return ids.ToArray();
        }
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds)
            {
                if (id >= 0x20000) sb.Append('.').Append((char)(id - 0x20000));
                else sb.Append((char)id);
            }
            return sb.ToString();
        }
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int BosTokenId => -1;
        public int EosTokenId => -1;
        public int VocabSize => 0x30000;
        public int CountTokens(string text) => Encode(text).Length;
    }

    /// <summary>Returns planted scores: the first call must be the full context (WITH), the second the context without R.</summary>
    private sealed class StubScorer(CodeContextPlan plan, (TaskScore With, TaskScore Without)[] planted) : ITaskScorer
    {
        public int Calls { get; private set; }

        public IReadOnlyList<TaskScore> Score(ReadOnlySpan<int> context, IReadOnlyList<TaskQuery> queries)
        {
            Calls++;
            Assert.Equal(plan.Tasks.Count, queries.Count);
            for (int i = 0; i < queries.Count; i++)
            {
                Assert.Equal(plan.Tasks[i].PrefixTokenIds, queries[i].Prefix);
                Assert.Equal(plan.Tasks[i].ContinuationTokenIds, queries[i].Continuation);
                Assert.Equal(plan.Tasks[i].MaskTokenCount, queries[i].MaskTokens);
            }
            bool with = context.SequenceEqual(plan.ContextTokenIds);
            bool without = context.SequenceEqual(plan.ContextWithoutRequired);
            Assert.True(with ^ without, "scorer got neither the full context nor the context without the required chunks");
            return planted.Select(p => with ? p.With : p.Without).ToArray();
        }
    }
}
