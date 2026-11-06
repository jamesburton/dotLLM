using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>Chunks cut at exact character offsets (one char token per character), to test boundaries the chunkers never produce.</summary>
internal static class TestChunks
{
    /// <summary>The chunk of <paramref name="file"/> covering [<paramref name="start"/>, <paramref name="end"/>).</summary>
    public static Chunk Piece(SourceFile file, int start, int end)
    {
        int Line(int offset) => 1 + file.Text.Take(offset).Count(c => c == '\n');
        int[] ids = new CharTokenizer().Encode(file.Text[start..end]);
        return new Chunk(file.Path, Line(start), Line(Math.Max(start, end - 1)), ids) { StartOffset = start, EndOffset = end };
    }

    public static Chunk[] Cut(SourceFile file, params int[] boundaries)
    {
        int[] all = [0, .. boundaries, file.Text.Length];
        return Enumerable.Range(0, all.Length - 1).Select(i => Piece(file, all[i], all[i + 1])).ToArray();
    }
}

/// <summary>
/// <see cref="ContextSelector"/> at exact offsets: a chunk that ends where a declaration starts (or starts where it ends) does
/// not overlap it, a name starting at a chunk's start is in that chunk and one starting at its end is not, and changed files on
/// another volume are reported with '/' separators.
/// </summary>
public sealed class ContextSelectorBoundaryTests
{
    private static ReferenceGraph GraphOf(SourceFile[] files, Chunk[] chunks) => new RoslynGraphBuilder(new CharTokenizer()).Build(files, chunks);

    [Fact]
    public void Symbol_ChunkTouchingTheDeclarationFromOutside_IsNotPartOfIt()
    {
        // Three declarations back to back: A [0,11), B [11,22), C [22,33). The chunks are cut exactly there.
        var file = new SourceFile("t/Abc.cs", "class A { }class B { }class C { }");
        Assert.Equal(33, file.Text.Length);
        Chunk[] chunks = TestChunks.Cut(file, 11, 22);
        Assert.Equal([(0, 11), (11, 22), (22, 33)], chunks.Select(c => (c.StartOffset, c.EndOffset)).ToArray());
        ReferenceGraph graph = GraphOf([file], chunks);

        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "B", Hops = 0 }, [file], graph);
        // Control: the locator found B exactly where the chunks were cut.
        SymbolDefinition def = Assert.Single(r.SymbolDefinitions);
        Assert.Equal((11, 22, 17), (def.StartOffset, def.EndOffset, def.NameOffset));
        // The chunk ending at B's start and the chunk starting at B's end are not part of B.
        SelectedChunk only = Assert.Single(r);
        Assert.Equal(chunks[1].Id, only.Chunk.Id);
        Assert.Equal(SelectionReason.Definition, only.Reason);

        // The same for the first and the last declaration.
        Assert.Equal([chunks[0].Id], ContextSelector.Select(new SelectionRequest { Symbol = "A", Hops = 0 }, [file], graph).Select(s => s.Chunk.Id));
        Assert.Equal([chunks[2].Id], ContextSelector.Select(new SelectionRequest { Symbol = "C", Hops = 0 }, [file], graph).Select(s => s.Chunk.Id));
    }

    [Fact]
    public void Symbol_NameStartingExactlyAtAChunkBoundary_BelongsToTheChunkThatStartsThere()
    {
        // "public class B { }": the name B is character 13; the chunks are cut right before it.
        var file = new SourceFile("t/B.cs", "public class B { }");
        Chunk[] chunks = TestChunks.Cut(file, 13);
        ReferenceGraph graph = GraphOf([file], chunks);

        SelectionResult r = ContextSelector.Select(new SelectionRequest { Symbol = "B", Hops = 0 }, [file], graph);
        Assert.Equal(13, Assert.Single(r.SymbolDefinitions).NameOffset); // control
        Assert.Equal(2, r.Count);
        // The chunk ending where the name starts holds the declaration's head but not its name; the next one holds the name.
        Assert.Equal(SelectionReason.DefinitionBody, Assert.Single(r, s => s.Chunk.StartOffset == 0).Reason);
        Assert.Equal(SelectionReason.Definition, Assert.Single(r, s => s.Chunk.StartOffset == 13).Reason);
    }

    [Fact]
    public void ChangedSince_AFileOnAnotherVolume_IsReportedWithForwardSlashes()
    {
        var file = new SourceFile("src/A.cs", "class A { }\n");
        ReferenceGraph graph = GraphOf([file], TestChunks.Cut(file));
        string root = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "ctxsel-volume", "repo"));
        string other, expected;
        if (OperatingSystem.IsWindows())
        {
            char drive = char.ToUpperInvariant(root[0]) == 'X' ? 'Y' : 'X';
            other = drive + @":\elsewhere\sub\x.cs";
            expected = drive + ":/elsewhere/sub/x.cs";
            // Controls: the backslash spelling really reaches the code, and no relative path exists to another drive.
            Assert.Contains('\\', other);
            Assert.True(Path.IsPathFullyQualified(Path.GetRelativePath(root, other)));
        }
        else
        {
            // One filesystem root: the path is relative to the corpus root, '/' separated (there is no other volume).
            other = "/ctxsel-elsewhere/sub/x.cs";
            int depth = root.Split('/', StringSplitOptions.RemoveEmptyEntries).Length;
            expected = string.Join('/', Enumerable.Repeat("..", depth).Concat(["ctxsel-elsewhere", "sub", "x.cs"]));
            Assert.False(Path.IsPathFullyQualified(Path.GetRelativePath(root, other)));
        }

        SelectionResult r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, [file], graph, new FixedGit(other), root);
        Assert.Empty(r);
        Assert.Equal([expected], r.UnmatchedChangedFiles);
    }

    [SkippableFact]
    public void ChangedSince_RootReachedThroughChainedLinks_MapsTheFinalTarget()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ctxsel-chain-" + Guid.NewGuid().ToString("N"));
        string real = Path.Combine(dir, "real"), second = Path.Combine(dir, "second"), first = Path.Combine(dir, "first");
        Directory.CreateDirectory(Path.Combine(real, "src"));
        try
        {
            // first -> second -> real: the immediate target of 'first' is itself a link.
            Skip.IfNot(TestLinks.TryCreateDirectoryLink(second, real), "Cannot create a directory junction or symlink here.");
            Skip.IfNot(TestLinks.TryCreateDirectoryLink(first, second), "Cannot create a directory junction or symlink here.");
            string immediate = new DirectoryInfo(first).LinkTarget!;
            Assert.NotNull(immediate); // control: 'first' is a link...
            Assert.NotEqual(Path.GetFullPath(real).TrimEnd(Path.DirectorySeparatorChar), Path.GetFullPath(immediate).TrimEnd(Path.DirectorySeparatorChar)); // ...to another link, not to the directory

            var file = new SourceFile("src/a.cs", "class A { }\n");
            ReferenceGraph graph = GraphOf([file], TestChunks.Cut(file));
            // git reports the path under the final target; the corpus root is the first link.
            SelectionResult r = ContextSelector.Select(new SelectionRequest { ChangedSince = "main" }, [file], graph,
                new FixedGit(Path.Combine(real, "src", "a.cs")), first);
            Assert.Equal(["src/a.cs"], r.Select(s => s.Chunk.FilePath));
            Assert.Empty(r.UnmatchedChangedFiles);
        }
        finally
        {
            TestLinks.DeleteTree(dir);
        }
    }

    private sealed class FixedGit(params string[] changed) : IGitInfo
    {
        public IReadOnlyList<string> ChangedFiles(string gitRef, bool sinceMergeBase = false) => changed;
    }
}
