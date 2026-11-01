using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>
/// Builds a <see cref="ReferenceGraph"/> with the Roslyn semantic model (one compilation over all files, BCL-only references,
/// as <see cref="Corpus"/>): symbols are resolved, never matched by name, and anything that does not resolve to a single
/// symbol declared in a chunked file gives no edge.
/// </summary>
/// <remarks>
/// <para>Edges (use end → definition end, both the first character of a name):</para>
/// <list type="bullet">
/// <item><b>Call</b>: an invoked method's name, an attribute's name, and a created type's name when the constructor is declared
/// in source → the method's declaration name (both parts of a partial method).</item>
/// <item><b>Use</b>: any other reference to a named type, field, property, event or method group → its declaration name(s)
/// (every part of a partial type). Locals, parameters, type parameters, range variables, labels and namespaces outside using
/// directives are not edges, and nor is <c>var</c>.</item>
/// <item><b>Import</b>: a using directive's name (first character of the name) → the namespace's name in its ordinal-first
/// declaring file; for <c>using static</c> and aliases of a type, the type's ordinal-first declaration. Namespaces that only
/// metadata declares give none.</item>
/// <item><b>Definition</b>: the name of each later declaration of a type, method or property declared in several places
/// (partial) → the ordinal-first declaration's name (by path, then position).</item>
/// </list>
/// <para>
/// Character → token mapping: each chunk's ids are decoded one token at a time, merging a token with the following ones while
/// the decoded text holds U+FFFD (a multi-byte character split across byte-level BPE tokens), so a character maps to the
/// first token of its group. When a chunk's decoded length differs from its character span (token-sliced chunks cut inside a
/// multi-byte character — see the caveat on <see cref="SymbolChunker"/> and <see cref="FixedSizeChunker"/>), offsets past
/// the end are clamped to the last token and the chunk is counted in <see cref="ReferenceGraph.MappingMismatchChunks"/>.
/// </para>
/// </remarks>
public sealed class RoslynGraphBuilder
{
    private const int MaxReplacementGroup = 8;
    private readonly ITokenizer _tokenizer;

    /// <summary>Creates the builder.</summary>
    /// <param name="tokenizer">The tokenizer the chunks were encoded with (used to decode chunk tokens to character offsets).</param>
    public RoslynGraphBuilder(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
    }

    /// <summary>Time spent by the last <c>Build</c> call: compilation, symbol binding (edges in characters), and token mapping.</summary>
    public (TimeSpan Compile, TimeSpan Bind, TimeSpan Map) LastTimings { get; private set; }

    /// <summary>Builds the graph of <paramref name="corpus"/>, reusing its compilation.</summary>
    public ReferenceGraph Build(Corpus corpus)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        CSharpCompilation compilation = corpus.Compilation;
        var trees = corpus.Files.ToDictionary(f => f.Path, f => corpus.TreeOf(f.Path), StringComparer.Ordinal);
        TimeSpan compile = sw.Elapsed;
        return Build(compilation, trees, corpus.Chunks, compile);
    }

    /// <summary>Builds the graph of <paramref name="chunks"/> of <paramref name="files"/>.</summary>
    /// <param name="files">The source files (unique paths); every chunk's file must be among them.</param>
    /// <param name="chunks">The chunks; graph chunk indices follow this order. Files without chunks contribute no edge ends.</param>
    /// <param name="implicitUsings">Add the SDK implicit global usings to the compilation (as <see cref="Corpus"/> does by default).</param>
    public ReferenceGraph Build(IReadOnlyList<SourceFile> files, IReadOnlyList<Chunk> chunks, bool implicitUsings = true)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(chunks);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        (CSharpCompilation compilation, Dictionary<string, SyntaxTree> trees) =
            Corpus.CreateCompilation(files.OrderBy(f => f.Path, StringComparer.Ordinal), implicitUsings);
        TimeSpan compile = sw.Elapsed;
        return Build(compilation, trees, chunks, compile);
    }

    private ReferenceGraph Build(CSharpCompilation compilation, Dictionary<string, SyntaxTree> trees, IReadOnlyList<Chunk> chunks,
        TimeSpan compile)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var index = new ChunkIndex(chunks, trees);
        var raw = new List<(Location Use, Location Def, SymbolEdgeKind Kind)>();
        foreach (string path in trees.Keys.Where(index.HasChunks).Order(StringComparer.Ordinal))
            CollectFile(compilation, trees[path], index, raw);
        TimeSpan bind = sw.Elapsed;

        sw.Restart();
        var edges = new List<ChunkEdge>(raw.Count);
        foreach ((Location use, Location def, SymbolEdgeKind kind) in raw)
        {
            if (index.TryMap(use, _tokenizer, out int uc, out int ut) && index.TryMap(def, _tokenizer, out int dc, out int dt))
                edges.Add(new ChunkEdge(uc, ut, dc, dt, kind));
        }
        ChunkEdge[] sorted = edges.Distinct().OrderBy(e => e.UseChunk).ThenBy(e => e.UseToken).ThenBy(e => e.DefinitionChunk)
            .ThenBy(e => e.DefinitionToken).ThenBy(e => e.Kind).ToArray();
        var graph = new ReferenceGraph(chunks, sorted, index.MismatchChunks);
        LastTimings = (compile, bind, sw.Elapsed);
        return graph;
    }

    private static void CollectFile(CSharpCompilation compilation, SyntaxTree tree, ChunkIndex index,
        List<(Location, Location, SymbolEdgeKind)> raw)
    {
        SemanticModel model = compilation.GetSemanticModel(tree);
        SyntaxNode root = tree.GetRoot();
        foreach (SyntaxNode node in root.DescendantNodes())
        {
            switch (node)
            {
                case UsingDirectiveSyntax u:
                    AddImport(model, u, index, raw);
                    break;
                case SimpleNameSyntax name when !name.Ancestors().Any(a => a is UsingDirectiveSyntax):
                    AddReference(model, name, index, raw);
                    break;
                case MemberDeclarationSyntax decl:
                    AddDefinition(model, decl, index, raw);
                    break;
            }
        }
    }

    private static void AddImport(SemanticModel model, UsingDirectiveSyntax u, ChunkIndex index, List<(Location, Location, SymbolEdgeKind)> raw)
    {
        if (u.NamespaceOrType is not { } target) return;
        ISymbol? symbol = model.GetSymbolInfo(target).Symbol;
        if (symbol is IAliasSymbol alias) symbol = alias.Target;
        if (symbol is not (INamespaceSymbol or INamedTypeSymbol)) return;
        Location? first = index.Ordered(Declarations(symbol)).FirstOrDefault();
        if (first is not null) raw.Add((target.GetLocation(), first, SymbolEdgeKind.Import));
    }

    private static void AddReference(SemanticModel model, SimpleNameSyntax name, ChunkIndex index, List<(Location, Location, SymbolEdgeKind)> raw)
    {
        if (name is IdentifierNameSyntax { IsVar: true }) return;
        ISymbol? symbol = model.GetSymbolInfo(name).Symbol;
        if (symbol is IAliasSymbol alias) symbol = alias.Target;
        if (symbol is null) return;
        Location use = name.Identifier.GetLocation();

        if (symbol is IMethodSymbol method)
        {
            // Attribute names bind to the constructor: a call of it, and a use of the attribute type.
            bool invoked = IsInvokedName(name) || name.Parent is AttributeSyntax || name.Parent?.Parent is AttributeSyntax;
            if (method.MethodKind == MethodKind.Constructor && !invoked) return;
            AddTo(raw, index, use, Declarations(method), invoked ? SymbolEdgeKind.Call : SymbolEdgeKind.Use);
            if (method.MethodKind == MethodKind.Constructor) AddTo(raw, index, use, Declarations(method.ContainingType), SymbolEdgeKind.Use);
            return;
        }
        if (symbol is not (INamedTypeSymbol or IFieldSymbol or IPropertySymbol or IEventSymbol)) return;
        if (symbol is INamedTypeSymbol { TypeKind: TypeKind.Error }) return;
        AddTo(raw, index, use, Declarations(symbol), SymbolEdgeKind.Use);

        // A created type's name also calls its constructor when that is declared in source.
        if (symbol is INamedTypeSymbol && CreationOf(name) is { } creation
            && model.GetSymbolInfo(creation).Symbol is IMethodSymbol { MethodKind: MethodKind.Constructor } ctor)
            AddTo(raw, index, use, Declarations(ctor), SymbolEdgeKind.Call);
    }

    private static void AddDefinition(SemanticModel model, MemberDeclarationSyntax decl, ChunkIndex index,
        List<(Location, Location, SymbolEdgeKind)> raw)
    {
        if (decl is not (BaseTypeDeclarationSyntax or MethodDeclarationSyntax or PropertyDeclarationSyntax)) return;
        if (model.GetDeclaredSymbol(decl) is not { } symbol) return;
        SyntaxToken id = decl switch
        {
            BaseTypeDeclarationSyntax t => t.Identifier,
            MethodDeclarationSyntax m => m.Identifier,
            PropertyDeclarationSyntax p => p.Identifier,
            _ => default,
        };
        Location? first = index.Ordered(Declarations(symbol)).FirstOrDefault();
        Location self = id.GetLocation();
        if (first is not null && (first.SourceSpan != self.SourceSpan || first.SourceTree?.FilePath != self.SourceTree?.FilePath))
            raw.Add((self, first, SymbolEdgeKind.Definition));
    }

    private static void AddTo(List<(Location, Location, SymbolEdgeKind)> raw, ChunkIndex index, Location use, IEnumerable<Location> defs,
        SymbolEdgeKind kind)
    {
        foreach (Location def in index.Ordered(defs)) raw.Add((use, def, kind));
    }

    /// <summary>Source declaration locations (name spans) of <paramref name="symbol"/> and its other partial parts; none when it is implicitly declared.</summary>
    private static IEnumerable<Location> Declarations(ISymbol symbol)
    {
        ISymbol s = symbol is IMethodSymbol m ? (m.ReducedFrom ?? m).OriginalDefinition : symbol.OriginalDefinition;
        // Synthesized members (a default constructor, record members) report their type's location: not a declaration.
        if (s.IsImplicitlyDeclared) return [];
        IEnumerable<Location> locations = s.Locations;
        switch (s)
        {
            case IMethodSymbol method:
                if (method.PartialImplementationPart is { } impl) locations = locations.Concat(impl.Locations);
                if (method.PartialDefinitionPart is { } def) locations = locations.Concat(def.Locations);
                break;
            case IPropertySymbol property:
                if (property.PartialImplementationPart is { } pimpl) locations = locations.Concat(pimpl.Locations);
                if (property.PartialDefinitionPart is { } pdef) locations = locations.Concat(pdef.Locations);
                break;
        }
        return locations.Where(l => l.IsInSource);
    }

    private static bool IsInvokedName(SimpleNameSyntax name)
    {
        SyntaxNode expr = name.Parent switch
        {
            MemberAccessExpressionSyntax ma when ma.Name == name => ma,
            MemberBindingExpressionSyntax mb when mb.Name == name => mb,
            _ => name,
        };
        return expr.Parent is InvocationExpressionSyntax inv && inv.Expression == expr;
    }

    /// <summary>The object creation whose type's rightmost name is <paramref name="name"/>, if any.</summary>
    private static ObjectCreationExpressionSyntax? CreationOf(SimpleNameSyntax name)
    {
        SyntaxNode type = name.Parent is QualifiedNameSyntax q && q.Right == name ? q
            : name.Parent is AliasQualifiedNameSyntax a && a.Name == name ? a
            : name;
        return type.Parent is ObjectCreationExpressionSyntax oc && oc.Type == type ? oc : null;
    }

    /// <summary>Chunks by file and character offset, with lazily computed per-token character starts.</summary>
    private sealed class ChunkIndex
    {
        private readonly IReadOnlyList<Chunk> _chunks;
        private readonly Dictionary<string, int[]> _byFile; // chunk indices sorted by start offset
        private readonly int[]?[] _tokenStarts;

        public ChunkIndex(IReadOnlyList<Chunk> chunks, Dictionary<string, SyntaxTree> trees)
        {
            _chunks = chunks;
            _tokenStarts = new int[]?[chunks.Count];
            _byFile = Enumerable.Range(0, chunks.Count).GroupBy(i => chunks[i].FilePath, StringComparer.Ordinal)
                .ToDictionary(g => g.Key, g => g.OrderBy(i => chunks[i].StartOffset).ToArray(), StringComparer.Ordinal);
            foreach (string path in _byFile.Keys)
                if (!trees.ContainsKey(path)) throw new ArgumentException($"Chunk file '{path}' is not among the files.", nameof(chunks));
        }

        public int MismatchChunks { get; private set; }

        public bool HasChunks(string path) => _byFile.ContainsKey(path);

        /// <summary>Locations in chunked files, ordered by (path, start), distinct.</summary>
        public IEnumerable<Location> Ordered(IEnumerable<Location> locations) => locations
            .Where(l => l.SourceTree is not null && HasChunks(l.SourceTree.FilePath))
            .DistinctBy(l => (l.SourceTree!.FilePath, l.SourceSpan.Start))
            .OrderBy(l => l.SourceTree!.FilePath, StringComparer.Ordinal).ThenBy(l => l.SourceSpan.Start);

        /// <summary>The chunk and chunk-local token holding the first character of <paramref name="location"/>.</summary>
        public bool TryMap(Location location, ITokenizer tokenizer, out int chunk, out int token)
        {
            chunk = token = -1;
            if (location.SourceTree is null || !_byFile.TryGetValue(location.SourceTree.FilePath, out int[]? list)) return false;
            int offset = location.SourceSpan.Start;
            int lo = 0, hi = list.Length - 1, found = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1;
                if (_chunks[list[mid]].StartOffset <= offset) { found = mid; lo = mid + 1; }
                else hi = mid - 1;
            }
            if (found < 0 || offset >= _chunks[list[found]].EndOffset) return false;
            chunk = list[found];
            int[] starts = _tokenStarts[chunk] ??= TokenStarts(_chunks[chunk], tokenizer);
            int local = offset - _chunks[chunk].StartOffset;
            int t = Array.BinarySearch(starts, local);
            if (t < 0) t = ~t - 1;
            while (t > 0 && starts[t - 1] == starts[t]) t--; // first token of a multi-byte group
            token = Math.Max(0, t);
            return true;
        }

        /// <summary>Chunk-local character start of each token (tokens of one multi-byte group share their group's start).</summary>
        private int[] TokenStarts(Chunk chunk, ITokenizer tokenizer)
        {
            int[] ids = chunk.TokenIds;
            var starts = new int[ids.Length];
            int pos = 0;
            for (int i = 0; i < ids.Length;)
            {
                int n = 1;
                string text = tokenizer.Decode(ids.AsSpan(i, 1));
                while (text.Contains('�') && n < MaxReplacementGroup && i + n < ids.Length)
                {
                    n++;
                    text = tokenizer.Decode(ids.AsSpan(i, n));
                }
                for (int k = 0; k < n; k++) starts[i + k] = pos;
                pos += text.Length;
                i += n;
            }
            if (pos != chunk.EndOffset - chunk.StartOffset) MismatchChunks++;
            return starts;
        }
    }
}
