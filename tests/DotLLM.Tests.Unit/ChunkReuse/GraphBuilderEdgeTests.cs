using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="RoslynGraphBuilder"/> and <see cref="ReferenceGraph"/> edge cases: names inside using directives, declarations in
/// files that have no chunks, and names that start exactly at a chunk's start or end.
/// </summary>
public sealed class GraphBuilderEdgeTests
{
    private static ReferenceGraph GraphOf(SourceFile[] files, Chunk[] chunks) => new RoslynGraphBuilder(new CharTokenizer()).Build(files, chunks);

    private static int Off(string text, string needle)
    {
        int at = text.IndexOf(needle, StringComparison.Ordinal);
        Assert.True(at >= 0, $"'{needle}' not found");
        return at;
    }

    [Fact]
    public void UsingDirective_GivesOnlyTheImportEdge_NotUseEdgesForTheNamesInsideIt()
    {
        var lib = new SourceFile("Lib/MathUtil.cs", "namespace Lib;\n\npublic static class MathUtil\n{\n    public static int Add(int a, int b) => a + b;\n}\n");
        var staticUse = new SourceFile("App/StaticUse.cs", "using static Lib.MathUtil;\n\nnamespace App;\n\npublic class StaticUse\n{\n    public int F() => Add(1, 2);\n}\n");
        var aliasUse = new SourceFile("App/AliasUse.cs", "using M = Lib.MathUtil;\n\nnamespace App;\n\npublic class AliasUse\n{\n    public int F() => M.Add(1, 2);\n}\n");
        SourceFile[] files = [lib, staticUse, aliasUse];
        var tok = new CharTokenizer();
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        ReferenceGraph graph = GraphOf(files, chunks);

        foreach ((SourceFile f, string directive) in new[] { (staticUse, "using static Lib.MathUtil;"), (aliasUse, "using M = Lib.MathUtil;") })
        {
            int end = Off(f.Text, directive) + directive.Length;
            IReadOnlyList<QueryEdge> inDirective = graph.QueryEdges(f.Path, 0, end);
            // The directive's name (its first character, 'Lib') imports MathUtil...
            QueryEdge import = Assert.Single(inDirective, e => e.Kind == SymbolEdgeKind.Import);
            Assert.Equal(Off(f.Text, "Lib.MathUtil"), import.UseOffset);
            // ...and the type name 'MathUtil' inside the directive is not also a use of it.
            Assert.DoesNotContain(inDirective, e => e.UseOffset == Off(f.Text, "MathUtil"));
            Assert.Single(inDirective);
            // Control: the same type name in code is a use (the alias file spells it M.Add, the static one calls Add).
            Assert.Contains(graph.QueryEdges(f.Path, end, f.Text.Length), e => e.Kind == SymbolEdgeKind.Call);
        }
    }

    [Fact]
    public void Declarations_InFilesWithoutChunks_AreSkipped_SoTheFirstChunkedDeclarationIsTheTarget()
    {
        // 'A/' sorts first but has no chunks: imports and definition edges must go to the first declaration that has a chunk.
        var noChunks1 = new SourceFile("A/Ns.cs", "namespace Shared;\n\npublic partial class Shape\n{\n}\n");
        var chunked1 = new SourceFile("B/Ns.cs", "namespace Shared;\n\npublic partial class Shape\n{\n    public int Side;\n}\n");
        var chunked2 = new SourceFile("C/Ns.cs", "namespace Shared;\n\npublic partial class Shape\n{\n    public int Area;\n}\n");
        var user = new SourceFile("Z/Use.cs", "using Shared;\n\nnamespace App;\n\npublic class U\n{\n}\n");
        SourceFile[] files = [noChunks1, chunked1, chunked2, user];
        var tok = new CharTokenizer();
        var chunker = new SymbolChunker(tok, 4096);
        Chunk[] chunks = [.. chunker.Chunk(chunked1), .. chunker.Chunk(chunked2), .. chunker.Chunk(user)];
        Assert.DoesNotContain(chunks, c => c.FilePath == noChunks1.Path); // control
        ReferenceGraph graph = GraphOf(files, chunks);

        // The import of 'Shared' goes to B/Ns.cs, the ordinal-first declaring file that has a chunk.
        QueryEdge import = Assert.Single(graph.QueryEdges(user.Path, 0, user.Text.Length), e => e.Kind == SymbolEdgeKind.Import);
        Assert.Equal("B/Ns.cs", chunks[import.DefinitionChunk].FilePath);
        // The later part's name links to the first part that has a chunk (B), not to the chunkless A.
        ChunkEdge def = Assert.Single(graph.Edges, e => e.Kind == SymbolEdgeKind.Definition);
        Assert.Equal(("C/Ns.cs", "B/Ns.cs"), (chunks[def.UseChunk].FilePath, chunks[def.DefinitionChunk].FilePath));
    }

    [Fact]
    public void Name_StartingExactlyAtAChunksStart_IsMappedIntoThatChunk()
    {
        var file = new SourceFile("t/Ab.cs", "class A { }\nclass B { A a; }\n");
        int use = Off(file.Text, "A a;");
        Chunk[] chunks = TestChunks.Cut(file, use);
        Assert.Equal(use, chunks[1].StartOffset); // control: the second chunk starts at the name
        ReferenceGraph graph = GraphOf([file], chunks);
        // Use: chunk 1, token 0 (the name is its first character); definition: chunk 0, token 6 (the 'A' of "class A").
        Assert.Equal([new ChunkEdge(1, 0, 0, Off(file.Text, "A {") , SymbolEdgeKind.Use)], graph.Edges);
    }

    [Fact]
    public void Name_StartingExactlyAtTheEndOfTheLastChunk_IsOutsideIt()
    {
        var file = new SourceFile("t/Ab.cs", "class A { }\nclass B { A a; }\n");
        int use = Off(file.Text, "A a;");
        // One chunk that ends where the use name starts; the rest of the file has no chunk.
        Chunk[] chunks = [TestChunks.Piece(file, 0, use)];
        Assert.Equal(use, chunks[0].EndOffset); // control
        ReferenceGraph graph = GraphOf([file], chunks);
        Assert.Empty(graph.Edges);

        // Control: one character further and the name is inside the chunk, giving the edge.
        Chunk[] covering = [TestChunks.Piece(file, 0, use + 1)];
        Assert.Single(GraphOf([file], covering).Edges);
    }
}
