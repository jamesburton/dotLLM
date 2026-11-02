using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>
/// Chooses which chunks of a corpus go into a context (Phase S Task 6; the <c>chunkcache select</c> verb). The candidates are
/// the chunks of the <see cref="ReferenceGraph"/>, which must be built over the given files (chunk paths are file paths);
/// token counts are the chunks' own token ids, so no tokenizer is needed here.
/// </summary>
/// <remarks>
/// <para>
/// <b>Criteria</b> (union; see <see cref="SelectionRequest"/>):
/// <list type="bullet">
/// <item><see cref="SelectionRequest.Paths"/>: every chunk of each matching file (<see cref="PathGlob"/>).</item>
/// <item><see cref="SelectionRequest.Symbol"/>: the chunks overlapping each matching declaration (<see cref="SymbolLocator"/>);
/// the one holding its name is <see cref="SelectionReason.Definition"/>, the rest <see cref="SelectionReason.DefinitionBody"/>.
/// From those (hop 0) a breadth-first walk follows reference edges of <see cref="SelectionRequest.EdgeKinds"/> in both
/// directions for <see cref="SelectionRequest.Hops"/> hops: a chunk with an edge into the previous hop is a
/// <see cref="SelectionReason.Caller"/>, one that the previous hop has an edge into is a <see cref="SelectionReason.Callee"/>.
/// Edges inside one chunk are ignored. Excluded chunks are neither selected nor walked through.</item>
/// <item><see cref="SelectionRequest.ChangedSince"/>: every chunk of each file <see cref="IGitInfo.ChangedFiles"/> reports,
/// mapped to corpus paths against the root directory (case-insensitively on Windows).</item>
/// </list>
/// A chunk matched several ways appears once, with the highest-priority reason as <see cref="SelectedChunk.Reason"/> and all of
/// them in <see cref="SelectedChunk.Reasons"/>.
/// </para>
/// <para>
/// <b>Rank and trim order.</b> Candidates are ranked by <see cref="SelectedChunk.Priority"/> descending
/// (<see cref="SelectionPriority"/>: definition 1000 &gt; definition body 900 &gt; path 800 &gt; changed 700 &gt; neighbour at hop
/// h 600−h), then <see cref="SelectedChunk.Weight"/> descending (edges to the previous hop), then path (ordinal), then start
/// offset. Under <see cref="SelectionRequest.TokenBudget"/> the selection is the longest prefix of that ranking whose tokens fit:
/// the lowest-ranked candidates are trimmed first, and trimming is strict — once a candidate does not fit, nothing ranked
/// below it is taken, even if smaller (so a lower priority never displaces a higher one). A budget smaller than the first
/// candidate gives an empty selection, with everything in <see cref="SelectionResult.Trimmed"/>.
/// </para>
/// <para>
/// <b>Determinism.</b> The result depends only on the inputs' content, not on the order of the files or the graph's chunks.
/// </para>
/// </remarks>
public static class ContextSelector
{
    /// <summary>Largest <see cref="SelectionRequest.Hops"/>.</summary>
    public const int MaxHops = 100;

    private static readonly SymbolEdgeKind[] DefaultEdgeKinds = [SymbolEdgeKind.Call, SymbolEdgeKind.Use, SymbolEdgeKind.Definition];

    /// <summary>Selects chunks for a context.</summary>
    /// <param name="request">What to select.</param>
    /// <param name="files">The corpus files (relative paths, '/' separated, as given to the chunker; for a solution the scan's
    /// authored files after <c>ChunkIndex.RelativizeScan</c>). Used to find symbol declarations and to check the graph.</param>
    /// <param name="graph">Reference graph over chunks of <paramref name="files"/>.</param>
    /// <param name="git">Git access, required with <see cref="SelectionRequest.ChangedSince"/>.</param>
    /// <param name="rootDirectory">Absolute directory the file paths are relative to, required with <see cref="SelectionRequest.ChangedSince"/>.</param>
    /// <returns>The selection in rank order, with trimmed candidates and notes.</returns>
    /// <exception cref="ArgumentException">Invalid request (no criterion, bad glob, bad symbol syntax, bad revision, missing git or root), or a graph chunk whose file is not in <paramref name="files"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="SelectionRequest.Hops"/> or <see cref="SelectionRequest.TokenBudget"/> out of range.</exception>
    /// <exception cref="SymbolNotFoundException">No declaration of <see cref="SelectionRequest.Symbol"/>, or none the graph has a chunk of.</exception>
    /// <exception cref="InvalidOperationException"><paramref name="git"/> returned a relative path.</exception>
    public static SelectionResult Select(SelectionRequest request, IReadOnlyList<SourceFile> files, ReferenceGraph graph, IGitInfo? git = null,
        string? rootDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(graph);

        // ---- request validation (before any work or any git call) ----
        if (request.Paths.Count == 0 && request.Symbol is null && request.ChangedSince is null)
            throw new ArgumentException("The request selects nothing: give Paths, Symbol or ChangedSince.", nameof(request));
        ArgumentOutOfRangeException.ThrowIfNegative(request.Hops, nameof(request.Hops));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(request.Hops, MaxHops, nameof(request.Hops));
        if (request.TokenBudget is int b) ArgumentOutOfRangeException.ThrowIfNegative(b, nameof(request.TokenBudget));
        (PathGlob Glob, string Text)[] paths = request.Paths.Select(p => (PathGlob.Parse(p), p.Trim())).ToArray();
        PathGlob[] includes = paths.Where(g => !g.Glob.Negated).Select(g => g.Glob).ToArray();
        string[] includeTexts = paths.Where(g => !g.Glob.Negated).Select(g => g.Text).ToArray();
        PathGlob[] negations = paths.Where(g => g.Glob.Negated).Select(g => g.Glob).ToArray();
        PathGlob[] excludes = request.Exclude.Select(PathGlob.Parse).ToArray();
        if (excludes.Any(g => g.Negated))
            throw new ArgumentException("Exclude patterns cannot be negated ('!'); put the exception in Paths.", nameof(request));
        string[]? symbol = request.Symbol is null ? null : SymbolLocator.ParseQuery(request.Symbol);
        string? root = null;
        if (request.ChangedSince is not null)
        {
            GitCliInfo.ValidateRef(request.ChangedSince);
            if (git is null) throw new ArgumentException("ChangedSince needs an IGitInfo.", nameof(git));
            if (string.IsNullOrWhiteSpace(rootDirectory) || !Path.IsPathFullyQualified(rootDirectory))
                throw new ArgumentException("ChangedSince needs the absolute root directory the file paths are relative to.", nameof(rootDirectory));
            root = Path.GetFullPath(rootDirectory);
        }
        var kinds = new HashSet<SymbolEdgeKind>(request.EdgeKinds ?? DefaultEdgeKinds);

        // ---- the corpus: graph chunks by file ----
        StringComparer pathComparer = RoslynGraphBuilder.PathComparer;
        var fileSet = new HashSet<string>(files.Select(f => f.Path), pathComparer);
        IReadOnlyList<Chunk> chunks = graph.Chunks;
        var chunksByFile = new Dictionary<string, List<int>>(pathComparer);
        for (int c = 0; c < chunks.Count; c++)
        {
            if (!fileSet.Contains(chunks[c].FilePath))
                throw new ArgumentException($"The graph has a chunk of '{chunks[c].FilePath}', which is not among the files; build the graph over the same files.", nameof(graph));
            if (!chunksByFile.TryGetValue(chunks[c].FilePath, out List<int>? list)) chunksByFile[chunks[c].FilePath] = list = [];
            list.Add(c);
        }
        foreach (List<int> list in chunksByFile.Values) list.Sort((x, y) => chunks[x].StartOffset.CompareTo(chunks[y].StartOffset));
        string[] corpusPaths = chunksByFile.Keys.Order(StringComparer.Ordinal).ToArray();

        var excludedFile = new Dictionary<string, bool>(pathComparer);
        bool Excluded(string path)
        {
            if (!excludedFile.TryGetValue(path, out bool x)) excludedFile[path] = x = excludes.Any(g => g.IsMatch(path));
            return x;
        }

        var candidates = new Dictionary<int, Candidate>();
        void Add(int c, SelectionReason reason, SelectionReason reasons, int priority, int hop, int weight, string detail)
        {
            if (!candidates.TryGetValue(c, out Candidate? cur))
            {
                candidates[c] = new Candidate(reason, priority, weight, detail) { Reasons = reasons, Hop = hop };
                return;
            }
            cur.Reasons |= reasons;
            if (hop >= 0 && (cur.Hop < 0 || hop < cur.Hop)) cur.Hop = hop;
            if (priority > cur.Priority || (priority == cur.Priority && weight > cur.Weight))
                candidates[c] = new Candidate(reason, priority, weight, detail) { Reasons = cur.Reasons, Hop = cur.Hop };
        }

        var notes = new List<string>();

        // ---- paths ----
        var unmatchedPatterns = new List<string>();
        if (includes.Length > 0)
        {
            var matched = new bool[includes.Length];
            foreach (string path in corpusPaths)
            {
                if (Excluded(path) || negations.Any(g => g.IsMatch(path))) continue;
                int first = Array.FindIndex(includes, g => g.IsMatch(path));
                if (first < 0) continue;
                for (int i = first; i < includes.Length; i++) if (includes[i].IsMatch(path)) matched[i] = true;
                foreach (int c in chunksByFile[path]) Add(c, SelectionReason.Path, SelectionReason.Path, SelectionPriority.Path, -1, 0, $"path glob '{includeTexts[first]}'");
            }
            for (int i = 0; i < includes.Length; i++) if (!matched[i]) unmatchedPatterns.Add(includeTexts[i]);
        }

        // ---- changed since ----
        var unmatchedChanged = new List<string>();
        if (request.ChangedSince is not null)
        {
            foreach (string abs in git!.ChangedFiles(request.ChangedSince).Distinct(StringComparer.Ordinal))
            {
                if (!Path.IsPathFullyQualified(abs))
                    throw new InvalidOperationException($"IGitInfo returned a relative path '{abs}'; ChangedFiles must return absolute paths.");
                string rel = Path.GetRelativePath(root!, Path.GetFullPath(abs));
                rel = Path.IsPathFullyQualified(rel) ? rel.Replace('\\', '/') : string.Join('/', rel.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries));
                if (!chunksByFile.TryGetValue(rel, out List<int>? list))
                {
                    unmatchedChanged.Add(rel);
                    continue;
                }
                if (Excluded(rel)) continue;
                foreach (int c in list) Add(c, SelectionReason.Changed, SelectionReason.Changed, SelectionPriority.Changed, -1, 0, $"changed since {request.ChangedSince}");
            }
            unmatchedChanged.Sort(StringComparer.Ordinal);
        }

        // ---- symbol neighbourhood ----
        IReadOnlyList<SymbolDefinition> definitions = [];
        if (symbol is not null)
        {
            // Only files the graph has chunks of can hold a selectable definition; others give a clear error below.
            (IReadOnlyList<SymbolDefinition> found, IReadOnlyList<string> suggestions) = SymbolLocator.Find(files, symbol);
            definitions = found;
            if (found.Count == 0)
            {
                string hint = suggestions.Count == 0 ? "" : $" Did you mean: {string.Join(", ", suggestions.Take(10))}?";
                throw new SymbolNotFoundException(request.Symbol!,
                    $"Symbol '{request.Symbol}' was not found: no declaration in the {files.Count} files has a qualified name ending in '{string.Join('.', symbol)}' (case-sensitive).{hint}",
                    suggestions);
            }

            var seeds = new SortedSet<int>();
            bool anyInGraph = false;
            foreach (SymbolDefinition d in found)
            {
                if (!chunksByFile.TryGetValue(d.FilePath, out List<int>? list)) continue;
                int[] overlapping = list.Where(c => chunks[c].StartOffset < d.EndOffset && chunks[c].EndOffset > d.StartOffset).ToArray();
                if (overlapping.Length == 0) continue;
                anyInGraph = true;
                if (Excluded(d.FilePath))
                {
                    notes.Add($"definition {d.QualifiedName} ({d.FilePath}:{d.Line}) is excluded by Exclude");
                    continue;
                }
                foreach (int c in overlapping)
                {
                    bool name = chunks[c].StartOffset <= d.NameOffset && d.NameOffset < chunks[c].EndOffset;
                    if (name) Add(c, SelectionReason.Definition, SelectionReason.Definition, SelectionPriority.Definition, 0, 0, $"definition of {d.QualifiedName}");
                    else Add(c, SelectionReason.DefinitionBody, SelectionReason.DefinitionBody, SelectionPriority.DefinitionBody, 0, 0, $"body of {d.QualifiedName}");
                    seeds.Add(c);
                }
            }
            if (!anyInGraph)
            {
                string where = string.Join(", ", found.Select(d => $"{d.QualifiedName} in {d.FilePath}:{d.Line}"));
                throw new SymbolNotFoundException(request.Symbol!,
                    $"Symbol '{request.Symbol}' is declared ({where}) but the graph has no chunk covering it; build the graph over the chunks of those files.", []);
            }
            Walk(graph, seeds, request.Hops, kinds, c => !Excluded(chunks[c].FilePath), Add);
        }

        // ---- rank, trim ----
        int[] order = candidates.Keys.ToArray();
        Array.Sort(order, (x, y) =>
        {
            Candidate a = candidates[x], bb = candidates[y];
            int d = bb.Priority.CompareTo(a.Priority);
            if (d == 0) d = bb.Weight.CompareTo(a.Weight);
            if (d == 0) d = string.CompareOrdinal(chunks[x].FilePath, chunks[y].FilePath);
            if (d == 0) d = chunks[x].StartOffset.CompareTo(chunks[y].StartOffset);
            return d;
        });
        var selected = new List<SelectedChunk>();
        var trimmed = new List<SelectedChunk>();
        long used = 0;
        bool full = false;
        foreach (int c in order)
        {
            Candidate k = candidates[c];
            var s = new SelectedChunk(ChunkRef.From(chunks[c], request.ChunkKind), k.Reason, k.Priority, k.Detail)
            {
                Reasons = k.Reasons,
                Hop = k.Hop,
                Weight = k.Weight,
            };
            if (!full && (request.TokenBudget is not int budget || used + s.Tokens <= budget))
            {
                selected.Add(s);
                used += s.Tokens;
            }
            else
            {
                full = true;
                trimmed.Add(s);
            }
        }
        return new SelectionResult(selected, trimmed, request.TokenBudget, definitions, unmatchedPatterns, unmatchedChanged, notes);
    }

    private static void Walk(ReferenceGraph graph, SortedSet<int> seeds, int hops, HashSet<SymbolEdgeKind> kinds, Func<int, bool> eligible,
        Action<int, SelectionReason, SelectionReason, int, int, int, string> add)
    {
        if (hops == 0 || kinds.Count == 0) return;
        IReadOnlyList<ChunkEdge> edges = graph.Edges;
        // Reverse adjacency: definition chunk → use chunks (one entry per edge), in edge order (sorted, so deterministic).
        var callers = new Dictionary<int, List<int>>();
        foreach (ChunkEdge e in edges)
        {
            if (e.UseChunk == e.DefinitionChunk || !kinds.Contains(e.Kind)) continue;
            if (!callers.TryGetValue(e.DefinitionChunk, out List<int>? list)) callers[e.DefinitionChunk] = list = [];
            list.Add(e.UseChunk);
        }

        var visited = new HashSet<int>(seeds);
        int[] frontier = seeds.ToArray();
        for (int hop = 1; hop <= hops && frontier.Length > 0; hop++)
        {
            var found = new SortedDictionary<int, (int Caller, int Callee)>();
            foreach (int c in frontier)
            {
                foreach (ChunkEdge e in graph.EdgesFrom(c))
                {
                    int d = e.DefinitionChunk;
                    if (d == c || !kinds.Contains(e.Kind) || visited.Contains(d) || !eligible(d)) continue;
                    found[d] = found.TryGetValue(d, out var w) ? (w.Caller, w.Callee + 1) : (0, 1);
                }
                if (!callers.TryGetValue(c, out List<int>? users)) continue;
                foreach (int u in users)
                {
                    if (visited.Contains(u) || !eligible(u)) continue;
                    found[u] = found.TryGetValue(u, out var w) ? (w.Caller + 1, w.Callee) : (1, 0);
                }
            }
            foreach ((int c, (int caller, int callee)) in found)
            {
                SelectionReason reason = (caller > 0 ? SelectionReason.Caller : 0) | (callee > 0 ? SelectionReason.Callee : 0);
                // Primary reason: Caller when the chunk references the previous hop, else Callee; Reasons keeps both.
                SelectionReason primary = caller > 0 ? SelectionReason.Caller : SelectionReason.Callee;
                string detail = (caller > 0, callee > 0) switch
                {
                    (true, true) => $"references and is referenced by hop-{hop - 1} chunks (hop {hop}, {caller}+{callee} edges)",
                    (true, false) => $"references hop-{hop - 1} chunks (hop {hop}, {caller} edges)",
                    _ => $"referenced by hop-{hop - 1} chunks (hop {hop}, {callee} edges)",
                };
                add(c, primary, reason, SelectionPriority.Neighbour(hop), hop, caller + callee, detail);
                visited.Add(c);
            }
            frontier = found.Keys.ToArray();
        }
    }

    private sealed class Candidate(SelectionReason reason, int priority, int weight, string detail)
    {
        public SelectionReason Reason { get; } = reason;
        public int Priority { get; } = priority;
        public int Weight { get; } = weight;
        public string Detail { get; } = detail;
        public SelectionReason Reasons { get; set; }
        public int Hop { get; set; } = -1;
    }
}
