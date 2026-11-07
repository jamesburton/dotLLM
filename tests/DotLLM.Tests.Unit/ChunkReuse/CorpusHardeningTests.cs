using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using DotLLM.ChunkReuse.CodeCorpus;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Hostile-input and mutation-killing tests for the symbol chunker and the code corpus (PR #1646 review): deep nesting must
/// not overflow the stack inside Roslyn, the directory loader must not follow links out of its root, and each rule the
/// review found unguarded (UTF-16 offsets, cut-point kinds, backward glue, hash inputs, generated/test detection) has a test
/// that fails when the rule is broken. Inputs are C# string literals with <c>\n</c> line ends and <c>\u</c> escapes for every
/// non-ASCII character, so neither checkout line endings nor file encoding can change them.
/// </summary>
public sealed class CorpusHardeningTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-corpus-hardening-" + Guid.NewGuid().ToString("N"));

    public CorpusHardeningTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } // removes links, never follows them
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static string Text(Chunk c) => CorpusTests.Text(c);

    // ---------------------------------------------------------------------------------------------------------------
    // (1) Deep nesting
    // ---------------------------------------------------------------------------------------------------------------

    private static string DeepParens(int depth) =>
        "class C\n{\n    int Shallow() => 0;\n\n    int M()\n    {\n        return " + new string('(', depth) + "1" + new string(')', depth)
        + ";\n    }\n}\n";

    /// <summary>
    /// ~2000 nested parentheses overflowed the stack inside Roslyn's parser on a default-size thread, killing the process
    /// (a StackOverflowException cannot be caught). A crash here aborts the test host, which the runner reports as a failure.
    /// Now the file is over <c>RoslynParsing.MaxNestingDepth</c>: not parsed, chunked by lines, with a warning.
    /// </summary>
    [Fact]
    public void SymbolChunker_5000NestedParens_DoesNotKillTheProcess()
    {
        string text = DeepParens(5000);
        var warnings = new List<string>();
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512, warnings.Add).Chunk(new SourceFile("deep.cs", text));
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, 512));
        Assert.Contains(warnings, w => w.Contains("deep.cs", StringComparison.Ordinal) && w.Contains("nesting", StringComparison.Ordinal));
    }

    [Fact]
    public void Corpus_Compilation_5000NestedParens_DoesNotKillTheProcess_FileLeftOut()
    {
        var files = new[] { new SourceFile("deep.cs", DeepParens(5000)), new SourceFile("ok.cs", "class Ok { int M() => 1; }\n") };
        (Microsoft.CodeAnalysis.CSharp.CSharpCompilation compilation, Dictionary<string, Microsoft.CodeAnalysis.SyntaxTree> trees) =
            Corpus.CreateCompilation(files, implicitUsings: false);
        Assert.Equal(2, trees.Count);
        Assert.NotNull(compilation.GetTypeByMetadataName("Ok"));
        Assert.Null(compilation.GetTypeByMetadataName("C")); // the deep file is an empty tree: nothing of it resolves
        Assert.Equal(0, trees["deep.cs"].Length);
    }

    /// <summary>
    /// Nesting that is not brackets (a chain of prefix operators) passes the bracket cap but still recurses in Roslyn's parser.
    /// Roslyn guards this path itself (on a default-size thread it gives up with an insufficient-stack error and the
    /// compilation loses type <c>C</c> — measured by running this test with the parse thread's stack size set to the default);
    /// on the large-stack thread the parse completes. The intact method still gets its own chunk (Roslyn ran).
    /// </summary>
    [Fact]
    public void SymbolChunker_DeepUnaryChain_UnderTheBracketCap_ParsedOnTheLargeStack()
    {
        string text = "class C\n{\n    int Shallow() => 0;\n\n    bool M() => " + new string('!', DeepUnaryDepth) + "true;\n}\n";
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("unary.cs", text));
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Assert.Contains(chunks, c => Text(c).Contains("int Shallow()", StringComparison.Ordinal)
                                     && !Text(c).Contains("bool M()", StringComparison.Ordinal));

        // The compilation is another matter: a tree this deep is measured after the parse and left out (an empty tree), because
        // binding anything that reaches into it would recurse per level. Its type is not in the compilation.
        var (compilation, _) = Corpus.CreateCompilation([new SourceFile("unary.cs", text)], implicitUsings: false);
        Assert.Null(compilation.GetTypeByMetadataName("C"));
    }

    [Fact]
    public void CreateCompilation_ModeratelyDeepUnaryChain_StaysInTheCompilation()
    {
        // Control for the test above: well under RoslynParsing.MaxSyntaxDepth the file keeps its symbols.
        string text = "class C\n{\n    bool M() => " + new string('!', 500) + "true;\n}\n";
        Assert.Contains('\n', text); // control: the escape reached the string as a newline
        Assert.True(RoslynParsing.SyntaxDepth(RoslynParsing.Parse(text, "u.cs").GetRoot()) < RoslynParsing.MaxSyntaxDepth);

        var (compilation, _) = Corpus.CreateCompilation([new SourceFile("u.cs", text)], implicitUsings: false);

        Assert.NotNull(compilation.GetTypeByMetadataName("C"));
    }

    /// <summary>Deep enough that Roslyn gives up on a default-size thread stack.</summary>
    internal const int DeepUnaryDepth = 50_000;

    // ---------------------------------------------------------------------------------------------------------------
    // (2) Links out of the root
    // ---------------------------------------------------------------------------------------------------------------

    [SkippableFact]
    public void Corpus_FromDirectory_DoesNotFollowADirectoryLinkOutOfTheRoot()
    {
        string root = Path.Combine(_scratch, "repo");
        string outside = Path.Combine(_scratch, "outside");
        WriteFile(root, "src/A.cs", "class A { }\n");
        WriteFile(outside, "Secret.cs", "class Secret { }\n");
        string link = Path.Combine(root, "linked");

        string? how = TryCreateDirectoryLink(link, outside);
        Skip.If(how is null, "the OS refused to create a directory symlink or junction");

        // The link really exists and resolves outside the root (else the exclusion below would prove nothing).
        Assert.True(Directory.Exists(link), $"{how} not visible");
        Assert.True(File.GetAttributes(link).HasFlag(FileAttributes.ReparsePoint), $"{how} is not a reparse point");
        Assert.True(File.Exists(Path.Combine(link, "Secret.cs")), $"{how} does not resolve to the outside directory");
        Assert.False(Path.GetFullPath(outside).StartsWith(Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase));

        var corpus = Corpus.FromDirectory(root, new SymbolChunker(new CharTokenizer(), 512), new CharTokenizer());
        Assert.Equal(["src/A.cs"], corpus.Files.Select(f => f.Path));
        Assert.DoesNotContain(corpus.Files, f => f.Text.Contains("Secret", StringComparison.Ordinal));
    }

    /// <summary>A directory symlink, else (Windows) a junction; null when the OS refuses both.</summary>
    private static string? TryCreateDirectoryLink(string link, string target)
    {
        try
        {
            Directory.CreateSymbolicLink(link, target);
            return "symlink";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
        }
        if (!OperatingSystem.IsWindows()) return null;
        try
        {
            using var p = Process.Start(new ProcessStartInfo("cmd.exe", $"/c mklink /J \"{link}\" \"{target}\"")
            {
                UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true,
            })!;
            p.StandardOutput.ReadToEnd();
            p.StandardError.ReadToEnd();
            p.WaitForExit();
            return p.ExitCode == 0 && Directory.Exists(link) ? "junction" : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    /// <summary>An unreadable directory is skipped instead of aborting the load (POSIX permissions; root reads anyway).</summary>
    [SkippableFact]
    public void Corpus_FromDirectory_SkipsAnInaccessibleDirectory()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX permission bits");
        string root = Path.Combine(_scratch, "perm");
        WriteFile(root, "src/A.cs", "class A { }\n");
        WriteFile(root, "locked/L.cs", "class L { }\n");
        string locked = Path.Combine(root, "locked");
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(locked, UnixFileMode.None);
        try
        {
            bool readable = true;
            try { _ = Directory.EnumerateFileSystemEntries(locked).ToArray(); }
            catch (UnauthorizedAccessException) { readable = false; }
            Skip.If(readable, "running as a user that ignores permissions (root)");

            var corpus = Corpus.FromDirectory(root, new SymbolChunker(new CharTokenizer(), 512), new CharTokenizer());
            Assert.Equal(["src/A.cs"], corpus.Files.Select(f => f.Path));
        }
        finally
        {
            if (!OperatingSystem.IsWindows())
                File.SetUnixFileMode(locked, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3a) Offsets are UTF-16 chars, not UTF-8 bytes
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// CJK (3 UTF-8 bytes, 1 UTF-16 char), an emoji (4 bytes, a surrogate pair) and a Latin-1 letter (2 bytes) before and
    /// inside chunks, plus a non-ASCII line long enough to be token-sliced. With the char tokenizer every chunk's offset
    /// slice must be exactly its decoded text — byte-counted offsets drift past the text.
    /// </summary>
    [Fact]
    public void SymbolChunker_NonAsciiAndNonBmpText_OffsetsAreUtf16()
    {
        const string cjk = "中文";
        const string emoji = "😀";
        string longLine = "    string L = \"" + string.Concat(Enumerable.Repeat("中" + emoji + "éx", 60)) + "\";\n";
        string text =
            "// " + cjk + " " + emoji + " header\n" +
            "class C\n{\n" +
            "    string A() => \"é" + emoji + cjk + "\";\n" +
            "\n" +
            "    // " + emoji + emoji + "\n" +
            "    int B() => 1;\n" +
            "\n" +
            longLine +
            "}\n";
        Assert.NotEqual(text.Length, Encoding.UTF8.GetByteCount(text)); // the fixture can tell chars from bytes

        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 64).Chunk(new SourceFile("u.cs", text));
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Assert.All(chunks, c => Assert.Equal(Text(c), text[c.StartOffset..c.EndOffset])); // token slices too
        Assert.Contains(chunks, c => !c.WireEligible); // the long line was sliced

        Chunk b = Assert.Single(chunks, c => Text(c).Contains("int B()", StringComparison.Ordinal));
        Assert.StartsWith("\n    // " + emoji + emoji + "\n    int B()", text[b.StartOffset..], StringComparison.Ordinal);
        Assert.Equal(text.IndexOf("\n\n    // " + emoji, StringComparison.Ordinal) + 1, b.StartOffset);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3b, 3c) Cut points when a member is over the limit
    // ---------------------------------------------------------------------------------------------------------------

    private static IReadOnlyList<Chunk> ChunkWithLimit(string text, int limit) =>
        new SymbolChunker(new CharTokenizer(), limit).Chunk(new SourceFile("cut.cs", text));

    /// <summary>
    /// A block belongs to its statement: no chunk may start at a line holding only <c>{</c>. The limit is chosen so the
    /// if-block's brace line is the furthest line start that fits — a cut there if blocks were cut points.
    /// </summary>
    [Fact]
    public void SymbolChunker_BlockIsNotACutPoint()
    {
        string text = "class C\n{\n    void M()\n    {\n        a();\n        if (f)\n        {\n            b0();\n            b1();\n" +
                      "        }\n        c();\n    }\n}\n";
        int brace = text.IndexOf("        {\n            b0", StringComparison.Ordinal);
        IReadOnlyList<Chunk> chunks = ChunkWithLimit(text, brace + 1);
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Assert.True(chunks.Count >= 2);
        Assert.All(chunks.Skip(1), c => Assert.NotEqual("{", LineAt(text, c.StartOffset).Trim()));
        Assert.Contains(chunks, c => c.StartOffset == text.IndexOf("        if (f)", StringComparison.Ordinal));
    }

    public static TheoryData<string, string> CutPointCases => new()
    {
        {
            "else",
            "class C\n{\n    void M()\n    {\n        if (f)\n        {\n            x();\n        }\n        else\n        {\n" +
            "            y();\n        }\n    }\n}\n"
        },
        {
            "catch",
            "class C\n{\n    void M()\n    {\n        try\n        {\n            x();\n        }\n        catch (System.Exception)\n" +
            "        {\n            y();\n        }\n        finally\n        {\n            z();\n        }\n    }\n}\n"
        },
        {
            "finally",
            "class C\n{\n    void M()\n    {\n        try\n        {\n            x();\n        }\n        catch (System.Exception)\n" +
            "        {\n            y();\n        }\n        finally\n        {\n            z();\n        }\n    }\n}\n"
        },
        {
            "set",
            "class C\n{\n    int P\n    {\n        get\n        {\n            return 1;\n        }\n        set\n        {\n" +
            "            _ = value;\n        }\n    }\n}\n"
        },
        {
            "case 2:",
            "class C\n{\n    void M(int v)\n    {\n        switch (v)\n        {\n            case 1:\n                x();\n" +
            "                break;\n            case 2:\n                y();\n                break;\n        }\n    }\n}\n"
        },
    };

    /// <summary>
    /// else, catch, finally, accessors and switch sections are cut points of an over-limit member. The limit is one token past
    /// the keyword's line start, so the keyword line is the furthest cut that fits: without that cut kind, the chunk ends at
    /// an earlier statement and no chunk starts on the keyword line.
    /// </summary>
    [Theory]
    [MemberData(nameof(CutPointCases))]
    public void SymbolChunker_ClauseAccessorAndSwitchSection_AreCutPoints(string keyword, string text)
    {
        int lineStart = LineStartOfKeyword(text, keyword);
        IReadOnlyList<Chunk> chunks = ChunkWithLimit(text, lineStart + 1);
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Assert.Contains(chunks, c => c.StartOffset == lineStart);
    }

    private static int LineStartOfKeyword(string text, string keyword)
    {
        int at = text.IndexOf("\n" + new string(' ', 8) + keyword, StringComparison.Ordinal);
        if (at < 0) at = text.IndexOf("\n" + new string(' ', 12) + keyword, StringComparison.Ordinal);
        Assert.True(at >= 0, keyword);
        return at + 1;
    }

    private static string LineAt(string text, int offset)
    {
        int end = text.IndexOf('\n', offset);
        return end < 0 ? text[offset..] : text[offset..end];
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3d) Trailing header-only piece glues backward
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void SymbolChunker_TrailingHeaderOnlyPiece_IsInsideThePreviousChunk()
    {
        string text = "class A\n{\n    void M() { }\n}\n\nnamespace B\n{\n}\n";
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("t.cs", text));
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        Chunk only = Assert.Single(chunks); // not a chunk of its own
        Assert.Contains("void M()", Text(only));
        Assert.EndsWith("namespace B\n{\n}\n", Text(only), StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3e, 3f) Chunk-id hash inputs, explicit little-endian
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>Returns fixed chunks for a file, under a chosen name.</summary>
    private sealed class ScriptedChunker(string name, params int[][] chunkIds) : IChunker
    {
        public string Name => name;
        public int MaxChunkTokens => int.MaxValue;

        public IReadOnlyList<Chunk> Chunk(SourceFile file)
        {
            var chunks = new List<Chunk>();
            int offset = 0;
            foreach (int[] ids in chunkIds)
            {
                chunks.Add(new Chunk(file.Path, 1, 1, ids) { StartOffset = offset, EndOffset = offset + ids.Length });
                offset += ids.Length;
            }
            return chunks;
        }
    }

    private static string Hash(IChunker chunker, string path = "abc") =>
        new Corpus([new SourceFile(path, "x")], chunker, new CharTokenizer()).ChunkIdHash;

    /// <summary>Same chunks under two chunker names hash differently; under one name, identically (the control).</summary>
    [Fact]
    public void ChunkIdHash_IncludesTheChunkerName()
    {
        int[][] ids = [[1, 2, 3], [4, 5]];
        Assert.Equal(Hash(new ScriptedChunker("symbol-512", ids)), Hash(new ScriptedChunker("symbol-512", ids)));
        Assert.NotEqual(Hash(new ScriptedChunker("symbol-512", ids)), Hash(new ScriptedChunker("lines-512", ids)));
    }

    /// <summary>
    /// Without the per-chunk token count, <c>[1] [2]</c> and <c>[1, "abc\0", 2]</c> of file <c>abc</c> feed the same bytes
    /// (the path's 4 bytes read as one little-endian id). The count separates them.
    /// </summary>
    [Fact]
    public void ChunkIdHash_IncludesTheTokenCount()
    {
        int pathAsId = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian("abc\0"u8); // 0x00636261
        string twoChunks = Hash(new ScriptedChunker("n", [1], [2]));
        string oneChunk = Hash(new ScriptedChunker("n", [1, pathAsId, 2]));
        Assert.NotEqual(twoChunks, oneChunk);
    }

    /// <summary>The exact hashed bytes, spelled out: pins the layout and its explicit little-endian ints.</summary>
    [Fact]
    public void ChunkIdHash_IsSha256OfTheDocumentedLittleEndianLayout()
    {
        byte[] expectedInput =
        [
            (byte)'n', 0,                         // chunker name, NUL
            (byte)'p', 0,                         // chunk path, NUL
            2, 0, 0, 0,                           // token count (LE int32)
            0x04, 0x03, 0x02, 0x01,               // id 0x01020304 (LE)
            0xFF, 0xFF, 0xFF, 0x7F,               // id int.MaxValue (LE)
        ];
        string expected = Convert.ToHexStringLower(SHA256.HashData(expectedInput));
        Assert.Equal(expected, Hash(new ScriptedChunker("n", [0x01020304, int.MaxValue]), path: "p"));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3g) Generated-code detection
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void IsGenerated_FindsAnAutoGeneratedMarkerAfterALongHeader_WithinTheWindow()
    {
        string header = string.Concat(Enumerable.Repeat("// Licensed under the MIT license. Lorem ipsum dolor sit amet.\n", 40));
        Assert.InRange(header.Length, 2000, 3900); // literal sizes: the documented window is 4000 characters
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/X.cs", header + "// <auto-generated/>\nclass A {}\n")));

        string farHeader = new string('/', 4000) + "\n";
        Assert.False(SymbolChunker.IsGenerated(new SourceFile("a/Y.cs", farHeader + "// <auto-generated/>\nclass A {}\n")));
    }

    [Fact]
    public void IsGenerated_GeneratedCodeAttribute_OnlyWhenItStartsALine()
    {
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/A.cs", "    [GeneratedCode(\"t\", \"1\")]\n    class A {}\n")));
        Assert.True(SymbolChunker.IsGenerated(new SourceFile("a/B.cs",
            "[global::System.CodeDom.Compiler.GeneratedCodeAttribute(\"t\", \"1\")]\nclass B {}\n")));
        Assert.False(SymbolChunker.IsGenerated(new SourceFile("a/C.cs", "// see [GeneratedCode(\"t\", \"1\")]\nclass C {}\n")));
        Assert.False(SymbolChunker.IsGenerated(new SourceFile("a/D.cs",
            "class D { string s = \"[GeneratedCode(\\\"t\\\", \\\"1\\\")]\"; }\n")));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // (3h) Test / build-output directory names
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Corpus_FromDirectory_TestAndBuildDirectoryNames()
    {
        string root = Path.Combine(_scratch, "names");
        WriteFile(root, "src/Lib/A.cs", "class A { }\n");
        WriteFile(root, "src/IntegrationTests/T.cs", "class T { }\n");   // *Tests
        WriteFile(root, "src/Contests/Contest.cs", "class Contest { }\n"); // ends in "tests", not a test project
        WriteFile(root, "src/Lib/Bin/B.cs", "class B { }\n");             // build output, any case
        WriteFile(root, "src/Lib/OBJ/O.cs", "class O { }\n");

        var corpus = Corpus.FromDirectory(root, new SymbolChunker(new CharTokenizer(), 512), new CharTokenizer());
        Assert.Equal(["src/Contests/Contest.cs", "src/Lib/A.cs"], corpus.Files.Select(f => f.Path));
        Assert.Contains(corpus.Excluded, e => e.Path == "src/IntegrationTests/T.cs" && e.Reason == "test");
        Assert.Contains(corpus.Excluded, e => e.Path == "src/Lib/Bin/B.cs" && e.Reason == "build-output");
        Assert.Contains(corpus.Excluded, e => e.Path == "src/Lib/OBJ/O.cs" && e.Reason == "build-output");
    }

    private static void WriteFile(string root, string rel, string text)
    {
        string full = Path.Combine(root, rel.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, text);
    }
}
