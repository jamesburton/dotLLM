using DotLLM.ChunkReuse.Graph;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// E4b: contexts built from a dependency neighbourhood instead of random distractors. A context holds the task's required
/// chunk(s), then distractors of which a <c>relatedFraction</c> share (by chunk count) comes from files within two
/// reference-graph hops of the calling file, the rest random; this is the setting where a structure-aware selector should be
/// strongest (in-context cross-chunk edges, definitions the query uses).
/// </summary>
/// <remarks>
/// <para><b>Neighbourhood</b> (<see cref="Neighbourhood"/>): directed use → definition edges at file level, of kinds
/// <see cref="SymbolEdgeKind.Use"/>, <see cref="SymbolEdgeKind.Call"/> and <see cref="SymbolEdgeKind.Definition"/>; imports are
/// not followed (a <c>using</c> points at whichever file declares the namespace first, not at a dependency). Hop 1: files the
/// calling file references anywhere in its text <b>except the masked call</b> (edges whose use lies in
/// [<see cref="CodeTask.MaskStartOffset"/>, + <see cref="CodeTask.MaskedText"/> length) are dropped, so the answer never decides
/// the neighbourhood). Hop 2: files the hop-1 files reference. Calling files are never part of it.</para>
/// <para><b>Filling</b>: candidates are related chunks (hop-1 files' chunks in a seeded order, then hop-2 files') and the other
/// eligible chunks (seeded order). Chunks are added one at a time, taking a related one while fewer than
/// <c>relatedFraction × (chosen + 1)</c> are related and a random one otherwise; a chunk that does not fit the remaining
/// budget is skipped, and when one pool has nothing left that fits the other fills the budget. Without exhaustion the related
/// count is therefore within one chunk of <c>relatedFraction × distractors</c>.</para>
/// <para><b>Leak filters</b> (as <see cref="CrossFileTaskGenerator"/>'s distractors, for both pools): no chunk of a calling
/// file, no chunk declaring a member or type with the callee's name, no chunk whose text contains a masked call's text.</para>
/// <para>The final chunk order is a seeded shuffle of required + distractors (as the generator's plans). Deterministic for a
/// corpus, graph, tasks, budget, fraction and seed.</para>
/// </remarks>
public static class RelatedContextBuilder
{
    /// <summary>Most hops from the calling file (2: the files it uses and the files those use).</summary>
    public const int Hops = 2;

    /// <summary>A related context for one task: see the class remarks.</summary>
    public static CodeContextPlan Build(Corpus corpus, CodeTask task, ReferenceGraph graph, int contextTokens, double relatedFraction, int seed)
    {
        ArgumentNullException.ThrowIfNull(task);
        return Build(corpus, [task], graph, contextTokens, relatedFraction, seed);
    }

    /// <summary>
    /// A related context for the tasks of one plan (call sites of the same definition, as
    /// <see cref="CrossFileTaskGenerator.GeneratePlansFor"/> groups them): the neighbourhood is the union over their calling
    /// files, and every calling file is excluded.
    /// </summary>
    /// <exception cref="ArgumentException">No tasks, tasks with different required chunks, or a budget smaller than the required chunks.</exception>
    public static CodeContextPlan Build(Corpus corpus, IReadOnlyList<CodeTask> tasks, ReferenceGraph graph, int contextTokens, double relatedFraction, int seed)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(tasks);
        ArgumentNullException.ThrowIfNull(graph);
        if (tasks.Count == 0) throw new ArgumentException("No tasks.", nameof(tasks));
        if (double.IsNaN(relatedFraction) || relatedFraction < 0 || relatedFraction > 1)
            throw new ArgumentOutOfRangeException(nameof(relatedFraction), relatedFraction, "Must be in [0, 1].");
        string key = RequiredKey(tasks[0]);
        if (tasks.Any(t => RequiredKey(t) != key))
            throw new ArgumentException("The tasks of one context must share their required chunk(s).", nameof(tasks));
        IReadOnlyList<Chunk> required = tasks[0].RequiredChunks;
        int requiredTokens = required.Sum(c => c.TokenIds.Length);
        if (requiredTokens > contextTokens)
            throw new ArgumentException($"The required chunks ({requiredTokens} tokens) exceed the context budget ({contextTokens}).", nameof(contextTokens));

        IReadOnlyDictionary<string, int> hood = Neighbourhood(graph, tasks);
        var callFiles = tasks.Select(t => t.CallSitePath).ToHashSet(StringComparer.Ordinal);
        var requiredIds = required.Select(c => c.Id).ToHashSet(StringComparer.Ordinal);
        string[] masked = tasks.Select(t => t.MaskedText.TrimStart('.', '?', ' ')).Distinct(StringComparer.Ordinal).ToArray();
        string[] names = masked.Select(LeadingIdentifier).Where(n => n.Length > 0).Distinct(StringComparer.Ordinal).ToArray();

        bool Eligible(Chunk c) =>
            !requiredIds.Contains(c.Id) && !callFiles.Contains(c.FilePath) && !names.Any(n => corpus.Declares(c, n))
            && !masked.Any(m => corpus.TextOf(c).Contains(m, StringComparison.Ordinal));

        var rng = new Random(seed);
        var related = new List<Chunk>();
        for (int hop = 1; hop <= Hops; hop++)
        {
            Chunk[] ring = corpus.Chunks.Where(c => hood.TryGetValue(c.FilePath, out int h) && h == hop && Eligible(c)).ToArray();
            rng.Shuffle(ring);
            related.AddRange(ring);
        }
        Chunk[] others = corpus.Chunks.Where(c => !hood.ContainsKey(c.FilePath) && Eligible(c)).ToArray();
        rng.Shuffle(others);

        var context = new List<Chunk>(required);
        int budget = contextTokens - requiredTokens, ri = 0, oi = 0, nRelated = 0, nOther = 0;
        // Next candidate of a pool that fits the remaining budget (skipped chunks never fit later: the budget only shrinks).
        Chunk? Next(IReadOnlyList<Chunk> pool, ref int i)
        {
            while (i < pool.Count && pool[i].TokenIds.Length > budget) i++;
            return i < pool.Count ? pool[i++] : null;
        }
        while (budget > 0)
        {
            bool wantRelated = nRelated < relatedFraction * (nRelated + nOther + 1);
            Chunk? c = wantRelated ? Next(related, ref ri) : Next(others, ref oi);
            bool isRelated = wantRelated;
            if (c is null)
            {
                c = wantRelated ? Next(others, ref oi) : Next(related, ref ri);
                isRelated = !wantRelated;
            }
            if (c is null) break;
            context.Add(c);
            budget -= c.TokenIds.Length;
            if (isRelated) nRelated++;
            else nOther++;
        }

        Chunk[] order = context.ToArray();
        rng.Shuffle(order);
        Chunk[] distractors = order.Where(c => !requiredIds.Contains(c.Id)).ToArray();
        return new CodeContextPlan(seed, order, tasks.Select(t => t with { DistractorChunks = distractors }).ToArray());
    }

    /// <summary>
    /// Files within <see cref="Hops"/> hops of the tasks' calling files, with their hop distance (1 or 2); see the class
    /// remarks. Calling files are not included.
    /// </summary>
    public static IReadOnlyDictionary<string, int> Neighbourhood(ReferenceGraph graph, IReadOnlyList<CodeTask> tasks)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(tasks);
        IReadOnlyList<Chunk> chunks = graph.Chunks;
        var callFiles = tasks.Select(t => t.CallSitePath).ToHashSet(StringComparer.Ordinal);
        var hood = new Dictionary<string, int>(StringComparer.Ordinal);

        // Hop 1: every non-import edge of the calling files except those whose use is a masked call.
        foreach (IGrouping<string, CodeTask> file in tasks.GroupBy(t => t.CallSitePath, StringComparer.Ordinal))
        {
            (int Start, int End)[] maskedSpans = file.Select(t => (t.MaskStartOffset, t.MaskStartOffset + t.MaskedText.Length)).ToArray();
            foreach (QueryEdge e in graph.QueryEdges(file.Key, 0, int.MaxValue))
            {
                if (e.Kind == SymbolEdgeKind.Import || maskedSpans.Any(s => e.UseOffset >= s.Start && e.UseOffset < s.End)) continue;
                string target = chunks[e.DefinitionChunk].FilePath;
                if (!callFiles.Contains(target)) hood.TryAdd(target, 1);
            }
        }

        // Further hops: every non-import edge of the previous ring's files.
        var ring = hood.Keys.ToHashSet(StringComparer.Ordinal);
        for (int hop = 2; hop <= Hops && ring.Count > 0; hop++)
        {
            var next = new HashSet<string>(StringComparer.Ordinal);
            for (int c = 0; c < chunks.Count; c++)
            {
                if (!ring.Contains(chunks[c].FilePath)) continue;
                foreach (ChunkEdge e in graph.EdgesFrom(c))
                {
                    if (e.Kind == SymbolEdgeKind.Import) continue;
                    string target = chunks[e.DefinitionChunk].FilePath;
                    if (!callFiles.Contains(target) && hood.TryAdd(target, hop)) next.Add(target);
                }
            }
            ring = next;
        }
        return hood;
    }

    private static string RequiredKey(CodeTask t) => string.Join('|', t.RequiredChunks.Select(c => c.Id).Order(StringComparer.Ordinal));

    /// <summary>The callee name at the start of a masked call's text (<c>Foo</c> of <c>Foo&lt;T&gt;(x)</c>).</summary>
    private static string LeadingIdentifier(string text)
    {
        int n = 0;
        if (n < text.Length && text[n] == '@') n++;
        while (n < text.Length && (char.IsLetterOrDigit(text[n]) || text[n] == '_')) n++;
        return text[..n].TrimStart('@');
    }
}
