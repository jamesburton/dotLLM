using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Roslyn reference graph over chunked code: semantic edges (call → callee declaration, symbol use → declaration, using
/// directive → namespace declaration, partial part → first part) mapped to chunk-local token indices, and
/// <see cref="ReferenceGraph.EdgesForContext(ChunkedContext)"/>, which keeps the edges that cross chunks inside a context, in
/// context coordinates. Character and byte tokenizers keep the expected token indices computable from the text.
/// </summary>
public sealed class ReferenceGraphTests
{
    // File order on purpose not ordinal: "Lib/MathUtil.cs" is the ordinal-first declarer of namespace Lib but comes last here.
    private static readonly (string Path, string Text)[] ThreeFiles =
    [
        ("App/Calc.cs",
            "using Lib;\nusing System.Text;\n\nnamespace App;\n\npublic class Calc\n{\n" +
            "    public int Run() => MathUtil.Add(1, 2) + MathUtil.One;\n\n" +
            "    public string Show() => new StringBuilder().Append(Run()).ToString();\n}\n"),
        ("Lib/Text.cs",
            "namespace Lib;\n\npublic static class Text\n{\n    public static string Pad(string s) => s + MathUtil.Zero;\n}\n"),
        ("Lib/MathUtil.cs",
            "namespace Lib;\n\npublic static class MathUtil\n{\n    public static int Add(int a, int b) => a + b;\n\n" +
            "    public const int Zero = 0;\n    public const int One = Zero + 1;\n}\n"),
    ];

    private static readonly (string Path, string Text)[] PartialFiles =
    [
        ("App/Draw.cs",
            "using Shapes;\n\nnamespace App;\n\npublic class Draw\n{\n" +
            "    public int Go() => new Shape().Area() + new Shape().Scale(2);\n}\n"),
        ("Shapes/P1.cs", "namespace Shapes;\n\npublic partial class Shape\n{\n    public int Area() => Side * Side;\n}\n"),
        ("Shapes/P2.cs",
            "namespace Shapes;\n\npublic partial class Shape\n{\n    public int Side { get; set; }\n\n    public partial int Scale(int k);\n}\n"),
        ("Shapes/P3.cs", "namespace Shapes;\n\npublic partial class Shape\n{\n    public partial int Scale(int k) => k * Side;\n}\n"),
    ];

    private static readonly (string Path, string Text)[] NestedFiles =
    [
        ("Q/Outer.cs",
            "namespace Q;\n\npublic class Outer\n{\n    public static int M() => 1;\n\n    public static class Inner\n    {\n" +
            "        public static int M() => 2;\n    }\n}\n"),
        ("App/UseQ.cs",
            "namespace App;\n\npublic class UseQ\n{\n    public int A() => Q.Outer.Inner.M();\n\n    public int B() => Q.Outer.M();\n}\n"),
    ];

    // ---------------------------------------------------------------------------------------------------------------
    // Plan tests
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Graph_CallInFileB_LinksToDefinitionInFileA()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(ThreeFiles, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray(); // placement differs from file order
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile a = File(files, "Lib/MathUtil.cs"), b = File(files, "App/Calc.cs");

        int callUse = At(order, tok, b, Off(b.Text, "Add("));
        Assert.Contains(new SymbolEdge(callUse, At(order, tok, a, Off(a.Text, "Add(")), SymbolEdgeKind.Call), edges);
        Assert.DoesNotContain(edges, e => e.UseTokenIndex == callUse && e.Kind != SymbolEdgeKind.Call);
        // The type name and a field are uses.
        Assert.Contains(new SymbolEdge(At(order, tok, b, Off(b.Text, "MathUtil.Add")), At(order, tok, a, Off(a.Text, "MathUtil\n")),
            SymbolEdgeKind.Use), edges);
        Assert.Contains(new SymbolEdge(At(order, tok, b, Off(b.Text, "One;")), At(order, tok, a, Off(a.Text, "One =")),
            SymbolEdgeKind.Use), edges);
        // A chunk placed twice is used at its first placement.
        Assert.Equal(edges, graph.EdgesForContext(Ctx([.. order, .. order])));
    }

    [Fact]
    public void Graph_UsingDirective_ProducesImportEdge()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(ThreeFiles, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile a = File(files, "Lib/MathUtil.cs"), b = File(files, "App/Calc.cs");

        int use = At(order, tok, b, Off(b.Text, "Lib;"));
        // One edge, to the namespace name in the ordinal-first declaring file (Lib/MathUtil.cs, not Lib/Text.cs).
        SymbolEdge import = Assert.Single(edges, e => e.UseTokenIndex == use);
        Assert.Equal(new SymbolEdge(use, At(order, tok, a, Off(a.Text, "Lib;")), SymbolEdgeKind.Import), import);
        // A namespace that only metadata declares gives no edge.
        int bcl = At(order, tok, b, Off(b.Text, "System.Text"));
        Assert.DoesNotContain(edges, e => e.UseTokenIndex >= bcl && e.UseTokenIndex < bcl + "System.Text".Length);
    }

    [Fact]
    public void Graph_EdgesForContext_ExcludesIntraChunkEdges()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(ThreeFiles, new SymbolChunker(tok, 4096), tok);
        SourceFile a = File(files, "Lib/MathUtil.cs");

        // `One = Zero + 1`: consecutive fields share a chunk, so the Zero use is an intra-chunk edge of the graph...
        int fields = Array.FindIndex(chunks, c => c.FilePath == a.Path && Text(a, c).Contains("Zero = 0", StringComparison.Ordinal));
        Chunk fc = chunks[fields];
        var intra = new ChunkEdge(fields, Off(a.Text, "Zero + 1") - fc.StartOffset, fields, Off(a.Text, "Zero = 0") - fc.StartOffset,
            SymbolEdgeKind.Use);
        Assert.Contains(intra, graph.Edges);

        // ...which no context reports; every reported edge crosses chunks.
        Chunk[] order = chunks.Reverse().ToArray();
        var ctx = Ctx(order);
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(ctx);
        Assert.NotEmpty(edges);
        Assert.All(edges, e => Assert.NotEqual(ctx.ChunkOf(e.UseTokenIndex), ctx.ChunkOf(e.DefinitionTokenIndex)));
        int zeroUse = At(order, tok, a, Off(a.Text, "Zero + 1"));
        Assert.DoesNotContain(edges, e => e.UseTokenIndex == zeroUse);

        // Edges whose other end is not in the context are excluded: without MathUtil.cs only Show → Run (two chunks of
        // Calc.cs) remains.
        Chunk[] partial = order.Where(c => c.FilePath != a.Path).ToArray();
        SourceFile b = File(files, "App/Calc.cs");
        Assert.Equal(new SymbolEdge(At(partial, tok, b, Off(b.Text, "Run()", 1)), At(partial, tok, b, Off(b.Text, "Run()")), SymbolEdgeKind.Call),
            Assert.Single(graph.EdgesForContext(Ctx(partial))));
    }

    [Fact]
    public void Graph_UnresolvedSymbol_ProducesNoEdgeAndNoException()
    {
        (string, string)[] repo =
        [
            ("Lib/Known.cs", "namespace Lib;\n\npublic static class Known\n{\n    public static int Run(int v) => v;\n}\n"),
            ("X1/A.cs", "namespace X1;\n\npublic static class Ambig\n{\n    public static int Go() => 1;\n}\n"),
            ("X2/A.cs", "namespace X2;\n\npublic static class Ambig\n{\n    public static int Go() => 2;\n}\n"),
            ("App/Use.cs",
                "using Lib;\nusing X1;\nusing X2;\nusing Nowhere;\n\nnamespace App;\n\npublic class Use\n{\n" +
                "    public int F() => Missing.Call(3) + Nope + Ambig.Go() + Known.Run(1) + Known.Absent();\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        ReferenceGraph? graph = null;
        Chunk[] chunks = [];
        SourceFile[] files = [];
        Assert.Null(Record.Exception(() => (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok)));

        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph!.EdgesForContext(Ctx(order));
        SourceFile use = File(files, "App/Use.cs"), known = File(files, "Lib/Known.cs");
        foreach (string unresolved in new[] { "Missing", "Call(", "Nope", "Ambig", "Go(", "Absent", "Nowhere" })
        {
            int at = At(order, tok, use, Off(use.Text, unresolved));
            Assert.DoesNotContain(edges, e => e.UseTokenIndex == at);
        }
        Assert.DoesNotContain(graph.Edges, e => e.DefinitionChunk != e.UseChunk && chunks[e.UseChunk].FilePath == use.Path
            && chunks[e.DefinitionChunk].FilePath.StartsWith('X') && e.Kind != SymbolEdgeKind.Import);
        // The resolvable call in the same expression still links.
        Assert.Contains(new SymbolEdge(At(order, tok, use, Off(use.Text, "Run(")), At(order, tok, known, Off(known.Text, "Run(")),
            SymbolEdgeKind.Call), edges);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Partial and nested types, determinism, content matching
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Graph_PartialClass_LinksEveryPart_AndDefinitionEdgesPointAtTheFirstPart()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(PartialFiles, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile draw = File(files, "App/Draw.cs"), p1 = File(files, "Shapes/P1.cs"), p2 = File(files, "Shapes/P2.cs"),
            p3 = File(files, "Shapes/P3.cs");
        int Name(SourceFile f) => At(order, tok, f, Off(f.Text, "Shape\n"));

        // A member of part 1 called from another file, a member of part 2 used from part 1.
        Assert.Contains(new SymbolEdge(At(order, tok, draw, Off(draw.Text, "Area(")), At(order, tok, p1, Off(p1.Text, "Area(")),
            SymbolEdgeKind.Call), edges);
        Assert.Contains(new SymbolEdge(At(order, tok, p1, Off(p1.Text, "Side")), At(order, tok, p2, Off(p2.Text, "Side")),
            SymbolEdgeKind.Use), edges);
        // The type name links to every part.
        int typeUse = At(order, tok, draw, Off(draw.Text, "Shape()"));
        Assert.Equal(new[] { Name(p1), Name(p2), Name(p3) }.Order(),
            edges.Where(e => e.UseTokenIndex == typeUse).Select(e => e.DefinitionTokenIndex).Order());
        Assert.All(edges.Where(e => e.UseTokenIndex == typeUse), e => Assert.Equal(SymbolEdgeKind.Use, e.Kind));
        // Later parts' names → the ordinal-first part; the first part has no Definition edge.
        Assert.Contains(new SymbolEdge(Name(p2), Name(p1), SymbolEdgeKind.Definition), edges);
        Assert.Contains(new SymbolEdge(Name(p3), Name(p1), SymbolEdgeKind.Definition), edges);
        Assert.DoesNotContain(edges, e => e.UseTokenIndex == Name(p1));
        // Partial method: the call reaches both the declaring and the implementing part; implementation → declaration.
        int scaleCall = At(order, tok, draw, Off(draw.Text, "Scale("));
        int scaleDecl = At(order, tok, p2, Off(p2.Text, "Scale(")), scaleImpl = At(order, tok, p3, Off(p3.Text, "Scale("));
        Assert.Equal(new[] { scaleDecl, scaleImpl }.Order(),
            edges.Where(e => e.UseTokenIndex == scaleCall && e.Kind == SymbolEdgeKind.Call).Select(e => e.DefinitionTokenIndex).Order());
        Assert.Contains(new SymbolEdge(scaleImpl, scaleDecl, SymbolEdgeKind.Definition), edges);
    }

    [Fact]
    public void Graph_NestedType_MemberBindsToTheInnerTypesDeclaration()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(NestedFiles, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile outer = File(files, "Q/Outer.cs"), use = File(files, "App/UseQ.cs");

        int outerM = At(order, tok, outer, Off(outer.Text, "M()")), innerM = At(order, tok, outer, Off(outer.Text, "M()", 1));
        int callA = At(order, tok, use, Off(use.Text, "M()")), callB = At(order, tok, use, Off(use.Text, "M()", 1));
        Assert.Equal(innerM, Assert.Single(edges, e => e.UseTokenIndex == callA).DefinitionTokenIndex);
        Assert.Equal(outerM, Assert.Single(edges, e => e.UseTokenIndex == callB).DefinitionTokenIndex);
        Assert.Contains(new SymbolEdge(At(order, tok, use, Off(use.Text, "Inner")), At(order, tok, outer, Off(outer.Text, "Inner")),
            SymbolEdgeKind.Use), edges);
        // Namespaces outside using directives give no edges.
        Assert.DoesNotContain(edges, e => e.UseTokenIndex == At(order, tok, use, Off(use.Text, "Q.Outer")));
    }

    [Fact]
    public void Graph_ObjectCreationAndAttribute_CallTheSourceConstructor_AndUseTheType()
    {
        (string, string)[] repo =
        [
            ("Lib/Tag.cs",
                "namespace Lib;\n\npublic sealed class TagAttribute : System.Attribute\n{\n    public TagAttribute(int v) { }\n}\n\n" +
                "public sealed class Box\n{\n    public Box(int v) { }\n}\n\npublic sealed class Plain\n{\n}\n"),
            ("App/Make.cs",
                "using Lib;\n\nnamespace App;\n\n[Tag(1)]\npublic class Make\n{\n    public Box New() => new Box(2);\n\n" +
                "    public Plain P() => new Plain();\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile lib = File(files, "Lib/Tag.cs"), make = File(files, "App/Make.cs");
        SymbolEdge[] From(int use) => edges.Where(e => e.UseTokenIndex == use).ToArray();

        Assert.Equal(new[]
        {
            new SymbolEdge(At(order, tok, make, Off(make.Text, "Tag(")), At(order, tok, lib, Off(lib.Text, "TagAttribute :")), SymbolEdgeKind.Use),
            new SymbolEdge(At(order, tok, make, Off(make.Text, "Tag(")), At(order, tok, lib, Off(lib.Text, "TagAttribute(")), SymbolEdgeKind.Call),
        }.OrderBy(e => e.DefinitionTokenIndex), From(At(order, tok, make, Off(make.Text, "Tag("))));
        int created = At(order, tok, make, Off(make.Text, "Box(2)"));
        Assert.Equal(new[]
        {
            new SymbolEdge(created, At(order, tok, lib, Off(lib.Text, "Box\n")), SymbolEdgeKind.Use),
            new SymbolEdge(created, At(order, tok, lib, Off(lib.Text, "Box(int")), SymbolEdgeKind.Call),
        }.OrderBy(e => e.DefinitionTokenIndex), From(created));
        // A return type is a plain use; an implicit constructor is no declaration.
        Assert.Equal(SymbolEdgeKind.Use, Assert.Single(From(At(order, tok, make, Off(make.Text, "Box New")))).Kind);
        Assert.Equal(SymbolEdgeKind.Use, Assert.Single(From(At(order, tok, make, Off(make.Text, "Plain()")))).Kind);
    }

    [Fact]
    public void Graph_PositionalRecordProperty_LinksToItsParameter_AndCorpusOverloadMatches()
    {
        (string, string)[] repo =
        [
            ("Lib/Rec.cs", "namespace Lib;\n\npublic record Rec(int Width);\n"),
            ("App/UseRec.cs", "using Lib;\n\nnamespace App;\n\npublic class UseRec\n{\n    public int W(Rec r) => r.Width + new Rec(1).Width;\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile rec = File(files, "Lib/Rec.cs"), use = File(files, "App/UseRec.cs");

        int width = At(order, tok, rec, Off(rec.Text, "Width"));
        Assert.Contains(new SymbolEdge(At(order, tok, use, Off(use.Text, "Width")), width, SymbolEdgeKind.Use), edges);
        Assert.Contains(new SymbolEdge(At(order, tok, use, Off(use.Text, "Width", 1)), width, SymbolEdgeKind.Use), edges);
        // The synthesized primary constructor is declared in source (the parameter list): `new Rec(1)` calls it.
        int created = At(order, tok, use, Off(use.Text, "Rec(1)"));
        Assert.Contains(edges, e => e.UseTokenIndex == created && e.Kind == SymbolEdgeKind.Use);
        Assert.Contains(edges, e => e.UseTokenIndex == created && e.Kind == SymbolEdgeKind.Call);

        // Build(Corpus) reuses the corpus compilation and gives the same graph.
        var corpus = new Corpus(files, new SymbolChunker(tok, 4096), tok);
        Assert.Equal(new RoslynGraphBuilder(tok).Build(corpus.Files, corpus.Chunks).Edges, new RoslynGraphBuilder(tok).Build(corpus).Edges);
    }

    [Fact]
    public void Graph_GluingTokenizer_NamesStartingMidToken_MapToTheTokenHoldingThem()
    {
        (string, string)[] repo =
        [
            .. ThreeFiles,
            ("Lib/Deep/Inner.cs", "namespace Lib.Deep;\n\npublic static class Inner\n{\n    public static int Z() => 0;\n}\n"),
            ("App/UseDeep.cs", "using Lib.Deep;\n\nnamespace App;\n\npublic class UseDeep\n{\n    public int Q() => Inner.Z();\n}\n"),
        ];
        ITokenizer tok = new GluingTokenizer();
        var (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile a = File(files, "Lib/MathUtil.cs"), b = File(files, "App/Calc.cs"), inner = File(files, "Lib/Deep/Inner.cs"),
            deep = File(files, "App/UseDeep.cs");
        int T(SourceFile f, string needle, int nth = 0) => TokenAt(order, tok, f, Off(f.Text, needle, nth));

        // ".Add" and " Add" are single tokens: both ends sit inside their token.
        Assert.Equal(".Add", tok.Decode([Ctx(order).TokenIds[T(b, "Add(")]]));
        Assert.Contains(new SymbolEdge(T(b, "Add("), T(a, "Add("), SymbolEdgeKind.Call), edges);
        Assert.Contains(new SymbolEdge(T(b, "MathUtil.Add"), T(a, "MathUtil\n"), SymbolEdgeKind.Use), edges);
        // `namespace Lib.Deep` also declares Lib, and Lib/Deep/Inner.cs is now the ordinal-first declaring file.
        Assert.Contains(new SymbolEdge(T(b, "Lib;"), T(inner, "Lib.Deep"), SymbolEdgeKind.Import), edges);
        // Dotted using: from " Lib" (start of the directive name) to ".Deep" (the namespace's last segment).
        Assert.Contains(new SymbolEdge(T(deep, "Lib.Deep"), T(inner, "Deep;"), SymbolEdgeKind.Import), edges);
        Assert.Contains(new SymbolEdge(T(deep, "Z("), T(inner, "Z("), SymbolEdgeKind.Call), edges);
    }

    [Fact]
    public void Graph_ConditionalCall_Event_AndPartialProperty()
    {
        (string, string)[] repo =
        [
            ("Lib/Svc.cs", "namespace Lib;\n\npublic class Svc\n{\n    public event System.Action? Changed;\n\n    public int Run() => 1;\n}\n"),
            ("Lib/Part1.cs", "namespace Lib;\n\npublic partial class Part\n{\n    public partial int Size { get; }\n}\n"),
            ("Lib/Part2.cs", "namespace Lib;\n\npublic partial class Part\n{\n    public partial int Size => 3;\n}\n"),
            ("App/Client.cs",
                "using Lib;\n\nnamespace App;\n\npublic class Client\n{\n    public int? Go(Svc? s, Part p)\n    {\n" +
                "        s!.Changed += () => { };\n        return s?.Run() + p.Size;\n    }\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Chunk[] order = chunks.Reverse().ToArray();
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(Ctx(order));
        SourceFile svc = File(files, "Lib/Svc.cs"), p1 = File(files, "Lib/Part1.cs"), p2 = File(files, "Lib/Part2.cs"),
            client = File(files, "App/Client.cs");

        // `s?.Run()` is a call (member binding), not a use.
        int run = At(order, tok, client, Off(client.Text, "Run("));
        Assert.Equal(new SymbolEdge(run, At(order, tok, svc, Off(svc.Text, "Run(")), SymbolEdgeKind.Call), Assert.Single(edges, e => e.UseTokenIndex == run));
        Assert.Contains(new SymbolEdge(At(order, tok, client, Off(client.Text, "Changed")), At(order, tok, svc, Off(svc.Text, "Changed")),
            SymbolEdgeKind.Use), edges);
        // Partial property: a use links both parts; the implementing part's name → the ordinal-first part.
        int size1 = At(order, tok, p1, Off(p1.Text, "Size")), size2 = At(order, tok, p2, Off(p2.Text, "Size"));
        int sizeUse = At(order, tok, client, Off(client.Text, "Size"));
        Assert.Equal(new[] { size1, size2 }.Order(), edges.Where(e => e.UseTokenIndex == sizeUse).Select(e => e.DefinitionTokenIndex).Order());
        Assert.Contains(new SymbolEdge(size2, size1, SymbolEdgeKind.Definition), edges);
    }

    [Fact]
    public void Graph_QueryEdges_ReturnFileOffsetsInTheRange_SoThePrefixExcludesTheMaskedCall()
    {
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(ThreeFiles, new SymbolChunker(tok, 4096), tok);
        SourceFile a = File(files, "Lib/MathUtil.cs"), b = File(files, "App/Calc.cs");
        (int Chunk, int Token) Def(SourceFile f, string needle)
        {
            int off = Off(f.Text, needle);
            int c = Array.FindIndex(chunks, k => k.FilePath == f.Path && off >= k.StartOffset && off < k.EndOffset);
            return (c, off - chunks[c].StartOffset);
        }

        // A task masking `Add(1, 2)`: the query is the file text before it.
        int maskStart = Off(b.Text, "Add("), maskEnd = maskStart + "Add(1, 2)".Length;
        IReadOnlyList<QueryEdge> prefix = graph.QueryEdges(b.Path, 0, maskStart);
        var (libC, libT) = Def(a, "Lib;");
        var (typeC, typeT) = Def(a, "MathUtil\n");
        var (addC, addT) = Def(a, "Add(");
        Assert.Equal(new[]
        {
            new QueryEdge(Off(b.Text, "Lib;"), libC, libT, SymbolEdgeKind.Import),
            new QueryEdge(Off(b.Text, "MathUtil.Add"), typeC, typeT, SymbolEdgeKind.Use),
        }, prefix);
        Assert.Equal(new[] { new QueryEdge(maskStart, addC, addT, SymbolEdgeKind.Call) }, graph.QueryEdges(b.Path, maskStart, maskEnd));
        // The range is half-open, and other files' edges never appear.
        Assert.Single(graph.QueryEdges(b.Path, 0, Off(b.Text, "MathUtil.Add")));
        SourceFile text = File(files, "Lib/Text.cs");
        Assert.Equal(new[] { Off(text.Text, "MathUtil.Zero"), Off(text.Text, "Zero;") },
            graph.QueryEdges(text.Path, 0, text.Text.Length).Select(e => e.UseOffset));
        // Sorted by offset whatever the chunk order of the graph.
        ReferenceGraph reversed = new RoslynGraphBuilder(tok).Build(files, chunks.Reverse().ToArray());
        IReadOnlyList<QueryEdge> all = reversed.QueryEdges(b.Path, 0, b.Text.Length);
        Assert.True(all.Count >= 4);
        Assert.Equal(all.OrderBy(e => e.UseOffset).ThenBy(e => e.DefinitionChunk).ThenBy(e => e.DefinitionToken).ThenBy(e => e.Kind), all);
    }

    [Fact]
    public void Graph_MappingMismatch_CountsAChunkCutInsideAMultiByteCharacter()
    {
        // "é" is two UTF-8 bytes at byte 12 and 13; 13-byte fixed chunks cut between them, so the second chunk starts with a
        // continuation byte and its decoded text is not the file slice. It holds the use of B.
        (string, string)[] repo = [("M.cs", "//xxxxxxxxxxé\nclass A { B b; }\nclass B { }\n")];
        ITokenizer tok = new ByteTokenizer();
        var (graph, chunks, _) = Build(repo, new FixedSizeChunker(tok, 13), tok);
        Assert.NotEmpty(graph.Edges);
        Assert.True(graph.MappingMismatchChunks >= 1);
        var (clean, _, _) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Assert.Equal(0, clean.MappingMismatchChunks);
    }

    [Fact]
    public void Graph_LocalsParametersAndVar_GiveNoEdges()
    {
        (string, string)[] repo =
        [
            ("Lib/Box.cs", "namespace Lib;\n\npublic sealed class Box\n{\n    public int Size;\n}\n"),
            ("App/Fill.cs",
                "using Lib;\n\nnamespace App;\n\npublic class Fill\n{\n    public int Run(int seed)\n    {\n        var box = new Box();\n" +
                "        int total = seed;\n        total += box.Size;\n        return total + seed;\n    }\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        // 16-token chunks put locals' declarations and uses in different chunks.
        var (graph, chunks, files) = Build(repo, new FixedSizeChunker(tok, 16), tok);
        string[] uses = graph.Edges.Select(e => IdentifierAt(files, chunks[e.UseChunk], tok, e.UseToken)).ToArray();
        Assert.Contains("Box", uses);
        Assert.Contains("Size", uses);
        Assert.DoesNotContain("var", uses);
        Assert.DoesNotContain("box", uses);
        Assert.DoesNotContain("total", uses);
        Assert.DoesNotContain("seed", uses);
    }

    [Fact]
    public void Graph_IsDeterministic_AndIndependentOfInputOrder()
    {
        ITokenizer tok = new CharTokenizer();
        (string, string)[] repo = [.. ThreeFiles, .. PartialFiles, .. NestedFiles];
        var (g1, chunks, _) = Build(repo, new SymbolChunker(tok, 4096), tok);
        var (g2, _, _) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Assert.Equal(g1.Edges, g2.Edges);
        var (g3, _, _) = Build(repo.Reverse().ToArray(), new SymbolChunker(tok, 4096), tok);
        // Graph edges sorted by (use chunk, use token, definition chunk, definition token, kind), no duplicates — also when the
        // chunk order is not the declaration order.
        foreach (ReferenceGraph g in new[] { g1, g3 })
            Assert.Equal(g.Edges.OrderBy(e => e.UseChunk).ThenBy(e => e.UseToken).ThenBy(e => e.DefinitionChunk).ThenBy(e => e.DefinitionToken)
                .ThenBy(e => e.Kind).Distinct(), g.Edges);
        var rng = new Random(7);
        Chunk[] order = chunks.ToArray();
        rng.Shuffle(order);
        var ctx = Ctx(order);
        IReadOnlyList<SymbolEdge> e1 = g1.EdgesForContext(ctx);
        Assert.True(e1.Count > 10);
        Assert.Equal(e1, g3.EdgesForContext(ctx));
        // Sorted by (use, definition, kind), no duplicates.
        var sorted = e1.OrderBy(e => e.UseTokenIndex).ThenBy(e => e.DefinitionTokenIndex).ThenBy(e => e.Kind).Distinct().ToArray();
        Assert.Equal(sorted, e1);
    }

    [Fact]
    public void Graph_EdgesForContext_AmbiguousChunkContent_OnlyTheExplicitOverloadResolvesIt()
    {
        // The two last members are textually identical chunks in different files.
        (string, string)[] repo =
        [
            ("Lib/MathUtil.cs", "namespace Lib;\n\npublic static class MathUtil\n{\n    public static int Add(int a, int b) => a + b;\n}\n"),
            ("App/A.cs", "using Lib;\n\nnamespace App;\n\npublic class A\n{\n    public int Pa() => 1;\n\n    public int Twice() => MathUtil.Add(1, 1);\n}\n"),
            ("App/B.cs", "using Lib;\n\nnamespace App;\n\npublic class B\n{\n    public int Pb() => 2;\n\n    public int Twice() => MathUtil.Add(1, 1);\n}\n"),
        ];
        ITokenizer tok = new CharTokenizer();
        var (graph, chunks, files) = Build(repo, new SymbolChunker(tok, 4096), tok);
        Chunk[] twice = chunks.Where(c => Text(File(files, c.FilePath), c).Contains("Twice", StringComparison.Ordinal)).ToArray();
        Assert.Equal(2, twice.Length);
        Assert.Equal(twice[0].TokenIds, twice[1].TokenIds);

        Chunk lib = chunks.Single(c => c.FilePath == "Lib/MathUtil.cs");
        Chunk[] order = [lib, twice[0]];
        var ctx = Ctx(order);
        Assert.Equal(new[] { Array.IndexOf(chunks, lib), -1 }, graph.MatchChunks(ctx));
        Assert.Empty(graph.EdgesForContext(ctx));

        IReadOnlyList<SymbolEdge> exact = graph.EdgesForContext(ctx, order);
        SourceFile a = File(files, "App/A.cs"), m = File(files, "Lib/MathUtil.cs");
        Assert.Contains(new SymbolEdge(At(order, tok, a, Off(a.Text, "Add(")), At(order, tok, m, Off(m.Text, "Add(")), SymbolEdgeKind.Call), exact);
        // The explicit chunk list must describe the context.
        Assert.Throws<ArgumentException>(() => graph.EdgesForContext(ctx, [lib]));
        Assert.Throws<ArgumentException>(() => graph.EdgesForContext(ctx, [twice[0], lib]));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Token-index correctness with real chunkers
    // ---------------------------------------------------------------------------------------------------------------

    public static TheoryData<string> ChunkerCases => new() { "symbol-char", "fixed-char-29", "fixed-char-64", "symbol-byte" };

    [Theory]
    [MemberData(nameof(ChunkerCases))]
    public void Graph_EdgeTokenIndices_PointAtTheIdentifiersFirstCharacter(string chunkerCase)
    {
        ITokenizer tok = chunkerCase.EndsWith("byte", StringComparison.Ordinal) ? new ByteTokenizer() : new CharTokenizer();
        IChunker chunker = chunkerCase.StartsWith("symbol", StringComparison.Ordinal)
            ? new SymbolChunker(tok, 4096)
            : new FixedSizeChunker(tok, int.Parse(chunkerCase[(chunkerCase.LastIndexOf('-') + 1)..], System.Globalization.CultureInfo.InvariantCulture));
        // Multi-byte characters before identifiers shift byte-token indices away from char offsets; identifiers starting with
        // one span several byte tokens.
        (string Path, string Text)[] repo =
        [
            .. ThreeFiles, .. PartialFiles, .. NestedFiles,
            ("Lib/Äpfel.cs", "namespace Lib;\n\npublic static class Äpfel\n{\n    public static int Öffne() => 1;\n}\n"),
            ("App/Obst.cs", "using Lib;\n\nnamespace App;\n\npublic class Obst\n{\n    public int Z() => Äpfel.Öffne();\n}\n"),
        ];
        repo = repo.Select(f => (f.Path, "// Größe — 日本語 ✓\n" + f.Text.Replace("{\n", "{\n    // café ✓\n", StringComparison.Ordinal))).ToArray();
        var (graph, chunks, files) = Build(repo, chunker, tok);

        // Graph level: chunk-local token indices map back to identifier starts with equal names.
        Assert.True(graph.Edges.Count > 20);
        foreach (ChunkEdge e in graph.Edges)
        {
            string useName = IdentifierAt(files, chunks[e.UseChunk], tok, e.UseToken);
            string defName = IdentifierAt(files, chunks[e.DefinitionChunk], tok, e.DefinitionToken);
            Assert.Equal(useName, defName);
        }

        // Context level, shuffled placement: same check through context coordinates.
        Chunk[] order = chunks.ToArray();
        new Random(11).Shuffle(order);
        var ctx = Ctx(order);
        int[] starts = ctx.ChunkStarts;
        IReadOnlyList<SymbolEdge> edges = graph.EdgesForContext(ctx);
        Assert.NotEmpty(edges);
        foreach (SymbolEdge e in edges)
        {
            int uc = ctx.ChunkOf(e.UseTokenIndex), dc = ctx.ChunkOf(e.DefinitionTokenIndex);
            Assert.NotEqual(uc, dc);
            Assert.Equal(IdentifierAt(files, order[uc], tok, e.UseTokenIndex - starts[uc]),
                IdentifierAt(files, order[dc], tok, e.DefinitionTokenIndex - starts[dc]));
        }

        // Known cross-file calls, at the exact indices.
        SourceFile a = File(files, "Lib/MathUtil.cs"), b = File(files, "App/Calc.cs");
        Assert.Contains(new SymbolEdge(At(order, tok, b, Off(b.Text, "Add(")), At(order, tok, a, Off(a.Text, "Add(")), SymbolEdgeKind.Call), edges);
        SourceFile apfel = File(files, "Lib/Äpfel.cs"), obst = File(files, "App/Obst.cs");
        Assert.Contains(new SymbolEdge(At(order, tok, obst, Off(obst.Text, "Öffne(")), At(order, tok, apfel, Off(apfel.Text, "Öffne(")),
            SymbolEdgeKind.Call), edges);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------

    private static (ReferenceGraph Graph, Chunk[] Chunks, SourceFile[] Files) Build((string Path, string Text)[] repo, IChunker chunker,
        ITokenizer tok)
    {
        SourceFile[] files = repo.Select(f => new SourceFile(f.Path, f.Text)).ToArray();
        Chunk[] chunks = files.SelectMany(chunker.Chunk).ToArray();
        return (new RoslynGraphBuilder(tok).Build(files, chunks), chunks, files);
    }

    private static ChunkedContext Ctx(IReadOnlyList<Chunk> order) => new(order.Select(c => c.TokenIds).ToArray(), []);

    private static SourceFile File(IEnumerable<SourceFile> files, string path) => files.Single(f => f.Path == path);

    private static string Text(SourceFile f, Chunk c) => f.Text[c.StartOffset..c.EndOffset];

    private static int Off(string text, string needle, int nth = 0)
    {
        int at = -1;
        for (int i = 0; i <= nth; i++)
        {
            at = text.IndexOf(needle, at + 1, StringComparison.Ordinal);
            Assert.True(at >= 0, $"'{needle}' #{nth} not found");
        }
        return at;
    }

    /// <summary>Context index of the token holding char <paramref name="offset"/> of <paramref name="file"/>.</summary>
    private static int At(IReadOnlyList<Chunk> order, ITokenizer tok, SourceFile file, int offset)
    {
        int start = 0;
        foreach (Chunk c in order)
        {
            if (c.FilePath == file.Path && offset >= c.StartOffset && offset < c.EndOffset)
                return start + tok.EncodeRaw(file.Text[c.StartOffset..offset]).Length;
            start += c.TokenIds.Length;
        }
        throw new InvalidOperationException($"offset {offset} of {file.Path} is not in the context");
    }

    /// <summary>Context index of the token whose decoded span contains char <paramref name="offset"/> of <paramref name="file"/>.</summary>
    private static int TokenAt(IReadOnlyList<Chunk> order, ITokenizer tok, SourceFile file, int offset)
    {
        int start = 0;
        foreach (Chunk c in order)
        {
            if (c.FilePath == file.Path && offset >= c.StartOffset && offset < c.EndOffset)
            {
                int pos = c.StartOffset;
                for (int t = 0; t < c.TokenIds.Length; t++)
                {
                    pos += tok.Decode(c.TokenIds.AsSpan(t, 1)).Length;
                    if (offset < pos) return start + t;
                }
            }
            start += c.TokenIds.Length;
        }
        throw new InvalidOperationException($"offset {offset} of {file.Path} is not in the context");
    }

    /// <summary>The identifier starting at chunk-local token <paramref name="token"/>; fails unless an identifier starts there.</summary>
    private static string IdentifierAt(SourceFile[] files, Chunk chunk, ITokenizer tok, int token)
    {
        string text = File(files, chunk.FilePath).Text;
        int at = chunk.StartOffset + tok.Decode(chunk.TokenIds.AsSpan(0, token)).Length;
        Assert.True(at < text.Length && (char.IsLetter(text[at]) || text[at] == '_'), $"no identifier at {chunk.FilePath}:{at}");
        Assert.True(at == 0 || !(char.IsLetterOrDigit(text[at - 1]) || text[at - 1] == '_'), $"mid-identifier at {chunk.FilePath}:{at}");
        int end = at;
        while (end < text.Length && (char.IsLetterOrDigit(text[end]) || text[end] == '_')) end++;
        return text[at..end];
    }
}
