using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>
/// Builds a <see cref="ReferenceGraph"/> with the Roslyn semantic model — either one compilation over all files, BCL-only
/// references (as <see cref="Corpus"/>), or one compilation per project (<see cref="ProjectCompilation"/>, from
/// the solution scanner in <c>DotLLM.ChunkReuse.Solution</c>): symbols are resolved, never matched by name, and anything that does not resolve to a single
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
/// first token of its group. When a chunk's decoded tokens do not spell its slice of the file (token-sliced chunks cut inside a
/// multi-byte character — see the caveat on <see cref="SymbolChunker"/> and <see cref="FixedSizeChunker"/>), offsets past
/// the end are clamped to the last token and the chunk is counted in <see cref="ReferenceGraph.MappingMismatchChunks"/>.
/// </para>
/// </remarks>
public sealed class RoslynGraphBuilder
{
    private const int MaxReplacementGroup = 8;
    private const char ReplacementChar = (char)0xFFFD; // U+FFFD, what a decoder gives for an incomplete UTF-8 sequence
    private readonly ITokenizer _tokenizer;

    /// <summary>Creates the builder.</summary>
    /// <param name="tokenizer">The tokenizer the chunks were encoded with (used to decode chunk tokens to character offsets).</param>
    public RoslynGraphBuilder(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// How file paths are compared where one file must not count twice: case-insensitive on Windows (where
    /// <c>Shared/A.cs</c> and <c>shared/a.cs</c> are one file), ordinal elsewhere.
    /// </summary>
    public static StringComparer PathComparer { get; } = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

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
        return Build(Single(compilation, trees), corpus.Chunks, compile);
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
            Corpus.CreateCompilation(files.OrderBy(f => f.Path, StringComparer.Ordinal), implicitUsings, RoslynParsing.MaxSyntaxDepth);
        TimeSpan compile = sw.Elapsed;
        return Build(Single(compilation, trees), chunks, compile);
    }

    /// <summary>
    /// Builds the graph of <paramref name="chunks"/> from one compilation per project (for instance the solution scanner's
    /// <c>ScanResult.Compilations</c>): each file is bound in its owning project's
    /// compilation, with that project's parse options, defines and references, so calls into referenced projects resolve
    /// where a single BCL-only compilation cannot. Edge definitions are the same as for the other overloads.
    /// </summary>
    /// <param name="projects">The projects; each corpus path is owned by one project only (compared with <see cref="PathComparer"/>:
    /// case-insensitive on Windows), and its tree must be in that project's compilation.</param>
    /// <param name="chunks">The chunks; every chunk's file must be owned by one of <paramref name="projects"/>, spelled exactly as
    /// the owner's key (chunk paths are matched ordinally; a different spelling throws).</param>
    /// <remarks><see cref="LastTimings"/>.Compile is zero: the compilations come built (lazily bound by Roslyn).</remarks>
    public ReferenceGraph Build(IReadOnlyList<ProjectCompilation> projects, IReadOnlyList<Chunk> chunks)
    {
        ArgumentNullException.ThrowIfNull(projects);
        ArgumentNullException.ThrowIfNull(chunks);
        var files = new Dictionary<string, (Compilation, SyntaxTree)>(PathComparer);
        foreach (ProjectCompilation p in projects)
        {
            ArgumentNullException.ThrowIfNull(p);
            foreach ((string path, SyntaxTree tree) in p.Trees)
            {
                if (!p.Compilation.ContainsSyntaxTree(tree))
                    throw new ArgumentException($"The tree of '{path}' is not in project '{p.Name}''s compilation.", nameof(projects));
                if (!files.TryAdd(path, (p.Compilation, tree)))
                    throw new ArgumentException($"File '{path}' is owned by more than one project ('{p.Name}').", nameof(projects));
            }
        }
        return Build(files, chunks, TimeSpan.Zero);
    }

    private static Dictionary<string, (Compilation, SyntaxTree)> Single(Compilation compilation, Dictionary<string, SyntaxTree> trees) =>
        trees.ToDictionary(kv => kv.Key, kv => (compilation, kv.Value), StringComparer.Ordinal);

    // files: corpus path → (owning compilation, its tree of the file).
    private ReferenceGraph Build(Dictionary<string, (Compilation Compilation, SyntaxTree Tree)> files, IReadOnlyList<Chunk> chunks,
        TimeSpan compile)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var index = new ChunkIndex(chunks, files.ToDictionary(kv => kv.Key, kv => kv.Value.Tree, StringComparer.Ordinal));
        var raw = new List<(Location Use, Location Def, SymbolEdgeKind Kind)>();
        // Binding recurses per nesting level of the source and a stack overflow cannot be caught: bind on the large-stack thread,
        // skip a file whose tree is deeper than the cap (the bracket cap does not see generic-angle, '.', ternary or unary nesting),
        // and treat Roslyn's own InsufficientExecutionStackException as "this file contributes no edge".
        RoslynParsing.RunWithLargeStack(() =>
        {
            foreach (string path in files.Keys.Where(index.HasChunks).Order(StringComparer.Ordinal))
            {
                SyntaxTree tree = files[path].Tree;
                int depth = RoslynParsing.SyntaxDepth(tree.GetRoot());
                if (depth > RoslynParsing.MaxSyntaxDepth)
                {
                    System.Diagnostics.Trace.TraceWarning($"{path}: syntax depth {depth} exceeds {RoslynParsing.MaxSyntaxDepth}; no edges from this file.");
                    continue;
                }
                var fileRaw = new List<(Location, Location, SymbolEdgeKind)>();
                // One file's binding may take minutes on a hostile shape: cancel it at the budget (Roslyn honours the token).
                using var budget = new CancellationTokenSource(FileBudget);
                try
                {
                    CollectFile(files[path].Compilation, tree, index, fileRaw, budget.Token);
                }
                catch (InsufficientExecutionStackException)
                {
                    System.Diagnostics.Trace.TraceWarning($"{path}: too deeply nested for Roslyn to bind; no edges from this file.");
                    continue;
                }
                catch (OperationCanceledException)
                {
                    System.Diagnostics.Trace.TraceWarning($"{path}: binding exceeded {FileBudget.TotalSeconds:F0} s; no edges from this file.");
                    continue;
                }
                raw.AddRange(fileRaw);
            }
            return 0;
        });
        TimeSpan bind = sw.Elapsed;

        sw.Restart();
        var edges = new List<(ChunkEdge Edge, int UseOffset)>(raw.Count);
        foreach ((Location use, Location def, SymbolEdgeKind kind) in raw)
        {
            if (index.TryMap(use, _tokenizer, out int uc, out int ut) && index.TryMap(def, _tokenizer, out int dc, out int dt))
                edges.Add((new ChunkEdge(uc, ut, dc, dt, kind), use.SourceSpan.Start));
        }
        // Distinct by edge (a token holding two names keeps the first name's offset), sorted.
        (ChunkEdge Edge, int UseOffset)[] sorted = edges.GroupBy(e => e.Edge).Select(g => (g.Key, g.Min(e => e.UseOffset)))
            .OrderBy(e => e.Key.UseChunk).ThenBy(e => e.Key.UseToken).ThenBy(e => e.Key.DefinitionChunk)
            .ThenBy(e => e.Key.DefinitionToken).ThenBy(e => e.Key.Kind).ToArray();
        var graph = new ReferenceGraph(chunks, sorted.Select(e => e.Edge).ToArray(), sorted.Select(e => e.UseOffset).ToArray(),
            index.MismatchChunks);
        LastTimings = (compile, bind, sw.Elapsed);
        return graph;
    }

    /// <summary>Time one file may take to bind before it contributes no edge (see <see cref="RoslynParsing.DefaultFileBudget"/>).</summary>
    internal TimeSpan FileBudget { get; init; } = RoslynParsing.DefaultFileBudget;

    private static void CollectFile(Compilation compilation, SyntaxTree tree, ChunkIndex index,
        List<(Location, Location, SymbolEdgeKind)> raw, CancellationToken ct)
    {
        SemanticModel model = compilation.GetSemanticModel(tree);
        SyntaxNode root = tree.GetRoot();
        foreach (SyntaxNode node in root.DescendantNodes())
        {
            ct.ThrowIfCancellationRequested();
            switch (node)
            {
                case UsingDirectiveSyntax u:
                    AddImport(model, u, index, raw, ct);
                    break;
                case SimpleNameSyntax name when !name.Ancestors().Any(a => a is UsingDirectiveSyntax):
                    AddReference(model, name, index, raw, ct);
                    break;
                case MemberDeclarationSyntax decl:
                    AddDefinition(model, decl, index, raw, ct);
                    break;
            }
        }
    }

    private static void AddImport(SemanticModel model, UsingDirectiveSyntax u, ChunkIndex index, List<(Location, Location, SymbolEdgeKind)> raw, CancellationToken ct)
    {
        if (u.NamespaceOrType is not { } target) return;
        ISymbol? symbol = model.GetSymbolInfo(target, ct).Symbol;
        if (symbol is IAliasSymbol alias) symbol = alias.Target;
        if (symbol is not (INamespaceSymbol or INamedTypeSymbol)) return;
        Location? first = index.Ordered(Declarations(symbol)).FirstOrDefault();
        if (first is not null) raw.Add((target.GetLocation(), first, SymbolEdgeKind.Import));
    }

    private static void AddReference(SemanticModel model, SimpleNameSyntax name, ChunkIndex index, List<(Location, Location, SymbolEdgeKind)> raw, CancellationToken ct)
    {
        if (name is IdentifierNameSyntax { IsVar: true }) return;
        ISymbol? symbol = model.GetSymbolInfo(name, ct).Symbol;
        if (symbol is IAliasSymbol alias) symbol = alias.Target;
        if (symbol is null) return;
        Location use = name.Identifier.GetLocation();

        if (symbol is IMethodSymbol method)
        {
            // Attribute names bind to the constructor: a call of it, and a use of the attribute type.
            bool invoked = IsInvokedName(name) || name.Parent is AttributeSyntax || name.Parent?.Parent is AttributeSyntax;
            // Defensive: a constructor symbol only binds to attribute names, which count as invoked above, so this is believed
            // unreachable (probed with ~95 name forms in the #1665 review). It is not covered by a test.
            if (method.MethodKind == MethodKind.Constructor && !invoked) return;
            AddTo(raw, index, use, Declarations(method), invoked ? SymbolEdgeKind.Call : SymbolEdgeKind.Use);
            if (method.MethodKind == MethodKind.Constructor) AddTo(raw, index, use, Declarations(method.ContainingType), SymbolEdgeKind.Use);
            return;
        }
        if (symbol is not (INamedTypeSymbol or IFieldSymbol or IPropertySymbol or IEventSymbol)) return;
        // Defensive, believed unreachable: unresolved names bind to a null symbol and error types have no source locations.
        if (symbol is INamedTypeSymbol { TypeKind: TypeKind.Error }) return;
        AddTo(raw, index, use, Declarations(symbol), SymbolEdgeKind.Use);

        // A created type's name also calls its constructor when that is declared in source.
        if (symbol is INamedTypeSymbol && CreationOf(name) is { } creation
            && model.GetSymbolInfo(creation, ct).Symbol is IMethodSymbol { MethodKind: MethodKind.Constructor } ctor)
            AddTo(raw, index, use, Declarations(ctor), SymbolEdgeKind.Call);
    }

    private static void AddDefinition(SemanticModel model, MemberDeclarationSyntax decl, ChunkIndex index,
        List<(Location, Location, SymbolEdgeKind)> raw, CancellationToken ct)
    {
        if (decl is not (BaseTypeDeclarationSyntax or MethodDeclarationSyntax or PropertyDeclarationSyntax)) return;
        if (model.GetDeclaredSymbol(decl, ct) is not { } symbol) return;
        SyntaxToken id = decl switch
        {
            BaseTypeDeclarationSyntax t => t.Identifier,
            MethodDeclarationSyntax m => m.Identifier,
            PropertyDeclarationSyntax p => p.Identifier,
            _ => default,
        };
        Location? first = index.Ordered(Declarations(symbol)).FirstOrDefault();
        Location self = id.GetLocation();
        if (first is not null && (first.SourceSpan != self.SourceSpan || index.PathOf(first) != index.PathOf(self)))
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
        private readonly Dictionary<string, SourceText> _texts = new(StringComparer.Ordinal);
        // Tree file path (as Roslyn locations carry it: absolute for workspace projects) → corpus path. Locations from any
        // compilation (a referencing project's view of another project's tree) map through the file path, not the tree object.
        private readonly Dictionary<string, string> _corpusPath = new(PathComparer);

        public ChunkIndex(IReadOnlyList<Chunk> chunks, Dictionary<string, SyntaxTree> trees)
        {
            _chunks = chunks;
            _tokenStarts = new int[]?[chunks.Count];
            foreach ((string path, SyntaxTree tree) in trees)
                if (!_corpusPath.TryAdd(tree.FilePath, path))
                    throw new ArgumentException($"Files '{_corpusPath[tree.FilePath]}' and '{path}' have the same tree path '{tree.FilePath}'.", nameof(trees));
            _byFile = Enumerable.Range(0, chunks.Count).GroupBy(i => chunks[i].FilePath, StringComparer.Ordinal)
                .ToDictionary(g => g.Key, g => g.OrderBy(i => chunks[i].StartOffset).ToArray(), StringComparer.Ordinal);
            foreach (string path in _byFile.Keys)
            {
                if (!trees.TryGetValue(path, out SyntaxTree? tree)) throw new ArgumentException($"Chunk file '{path}' is not among the files.", nameof(chunks));
                _texts[path] = tree.GetText();
            }
        }

        public int MismatchChunks { get; private set; }

        public bool HasChunks(string path) => _byFile.ContainsKey(path);

        /// <summary>Corpus path of <paramref name="location"/>'s file, or null when it is not a corpus file.</summary>
        public string? PathOf(Location location) =>
            location.SourceTree is { } t && _corpusPath.TryGetValue(t.FilePath, out string? path) ? path : null;

        /// <summary>Locations in chunked files, ordered by (corpus path, start), distinct.</summary>
        public IEnumerable<Location> Ordered(IEnumerable<Location> locations) => locations
            .Select(l => (Location: l, Path: PathOf(l)))
            .Where(l => l.Path is not null && HasChunks(l.Path))
            .DistinctBy(l => (l.Path, l.Location.SourceSpan.Start))
            .OrderBy(l => l.Path, StringComparer.Ordinal).ThenBy(l => l.Location.SourceSpan.Start)
            .Select(l => l.Location);

        /// <summary>The chunk and chunk-local token holding the first character of <paramref name="location"/>.</summary>
        public bool TryMap(Location location, ITokenizer tokenizer, out int chunk, out int token)
        {
            chunk = token = -1;
            if (PathOf(location) is not { } path || !_byFile.TryGetValue(path, out int[]? list)) return false;
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
            var decoded = new System.Text.StringBuilder();
            for (int i = 0; i < ids.Length;)
            {
                int n = 1;
                string text = tokenizer.Decode(ids.AsSpan(i, 1));
                while (text.Contains(ReplacementChar) && n < MaxReplacementGroup && i + n < ids.Length)
                {
                    n++;
                    text = tokenizer.Decode(ids.AsSpan(i, n));
                }
                for (int k = 0; k < n; k++) starts[i + k] = decoded.Length;
                decoded.Append(text);
                i += n;
            }
            // The decoded tokens must spell the chunk's own slice of the file; otherwise offsets are approximate.
            if (!_texts[chunk.FilePath].ToString(new TextSpan(chunk.StartOffset, chunk.EndOffset - chunk.StartOffset)).Equals(decoded.ToString(), StringComparison.Ordinal))
                MismatchChunks++;
            return starts;
        }
    }
}
