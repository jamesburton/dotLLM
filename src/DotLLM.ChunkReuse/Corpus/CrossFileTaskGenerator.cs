using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// A cross-file completion task: the model reads a context (required + distractor chunks), then <see cref="Prefix"/> (the
/// query: the calling file's text up to a masked call site), and should continue with <see cref="ExpectedContinuation"/>,
/// whose first <see cref="MaskTokenCount"/> tokens are the masked callee name and arguments. The callee is defined in
/// <see cref="RequiredChunks"/>, in a different file.
/// </summary>
/// <param name="Prefix">Query text: a <c>// path</c> header line, then the calling file's lines up to the mask.</param>
/// <param name="ExpectedContinuation">The original text from the mask on (decoded <see cref="ContinuationTokenIds"/>).</param>
/// <param name="RequiredChunks">The chunk(s) holding the callee's declaration (one per declaring part).</param>
/// <param name="DistractorChunks">The other context chunks, in context order.</param>
public sealed record CodeTask(string Prefix, string ExpectedContinuation, IReadOnlyList<Chunk> RequiredChunks, IReadOnlyList<Chunk> DistractorChunks)
{
    /// <summary>Stable id of the call site: <c>path:line:column</c> of the mask start (seed-independent).</summary>
    public string Id { get; init; } = "";

    /// <summary><see cref="Prefix"/> as token ids (encoded once).</summary>
    public int[] PrefixTokenIds { get; init; } = [];

    /// <summary>The expected continuation's token ids (default 32).</summary>
    public int[] ContinuationTokenIds { get; init; } = [];

    /// <summary>Leading continuation tokens that cover the masked call text (the scored region).</summary>
    public int MaskTokenCount { get; init; }

    /// <summary>
    /// Token offset of the masked region relative to the start of the query (= prefix length). In a context of L tokens the
    /// masked region starts at absolute position L + <see cref="MaskTokenOffset"/>.
    /// </summary>
    public int MaskTokenOffset => PrefixTokenIds.Length;

    /// <summary>
    /// The masked source text: callee name through the closing parenthesis, preceded by whatever the tokenizer glues to the
    /// name (e.g. the <c>.</c> of a member access for Qwen's BPE), since the mask starts on a token boundary.
    /// </summary>
    public string MaskedText { get; init; } = "";

    /// <summary>Path of the calling file.</summary>
    public string CallSitePath { get; init; } = "";

    /// <summary>
    /// File character offset where the query's file text starts: <see cref="Prefix"/> is the <c>// path</c> header line followed by
    /// the calling file's text [<see cref="PrefixStartOffset"/>, <see cref="MaskStartOffset"/>).
    /// </summary>
    public int PrefixStartOffset { get; init; }

    /// <summary>File character offset of the masked text (<see cref="MaskedText"/> spans [this, this + its length)).</summary>
    public int MaskStartOffset { get; init; }

    /// <summary>1-based line of the mask start.</summary>
    public int CallSiteLine { get; init; }

    /// <summary>The resolved callee (Roslyn display string).</summary>
    public string Symbol { get; init; } = "";
}

/// <summary>
/// One assembled context and the tasks evaluated over it. All tasks share the same required chunk(s) (call sites of the
/// same definition chunk, from files outside the context), so one prefill of <see cref="ContextTokenIds"/> and one of
/// <see cref="ContextWithoutRequired"/> serve every task.
/// </summary>
/// <param name="Seed">Seed the plan came from.</param>
/// <param name="Chunks">Context chunks in context order.</param>
/// <param name="Tasks">Tasks over this context.</param>
public sealed record CodeContextPlan(int Seed, IReadOnlyList<Chunk> Chunks, IReadOnlyList<CodeTask> Tasks)
{
    /// <summary>Ids of every task's required chunks (the union).</summary>
    public IReadOnlySet<string> RequiredIds { get; } = Tasks.SelectMany(t => t.RequiredChunks).Select(c => c.Id).ToHashSet(StringComparer.Ordinal);

    /// <summary>The context: chunk token ids concatenated in order.</summary>
    public int[] ContextTokenIds => Chunks.SelectMany(c => c.TokenIds).ToArray();

    /// <summary>The context with the required chunks removed (the others keep their order).</summary>
    public int[] ContextWithoutRequired => Chunks.Where(c => !RequiredIds.Contains(c.Id)).SelectMany(c => c.TokenIds).ToArray();

    /// <summary>Token offset of each chunk in the context.</summary>
    public int[] ChunkStarts
    {
        get
        {
            var starts = new int[Chunks.Count];
            for (int i = 1; i < starts.Length; i++) starts[i] = starts[i - 1] + Chunks[i - 1].TokenIds.Length;
            return starts;
        }
    }
}

/// <summary>Task-generation knobs. Difficulty levers: <see cref="ContinuationTokens"/>, the distractor count (via
/// <see cref="ContextTokens"/> / <see cref="MaxDistractors"/>) and <see cref="PrefixTokens"/>.</summary>
public sealed record TaskGeneratorOptions
{
    /// <summary>Context budget in tokens (required + distractors). Default 2048.</summary>
    public int ContextTokens { get; init; } = 2048;

    /// <summary>Expected continuation length in tokens (the masked call must fit inside). Default 32.</summary>
    public int ContinuationTokens { get; init; } = 32;

    /// <summary>Query budget in tokens (the <c>// path</c> header plus whole lines before the mask). Default 256.</summary>
    public int PrefixTokens { get; init; } = 256;

    /// <summary>Most tasks per context (call sites of the same definition chunk). Default 4.</summary>
    public int TasksPerContext { get; init; } = 4;

    /// <summary>Most distractor chunks per context. Default: as many as fit <see cref="ContextTokens"/>.</summary>
    public int MaxDistractors { get; init; } = int.MaxValue;
}

/// <summary>Call-site census of a corpus (why invocations were or were not usable).</summary>
/// <param name="Invocations">Invocation expressions in non-generated corpus files (<c>nameof</c> excluded).</param>
/// <param name="Unresolved">No single symbol (missing reference, error type, ambiguity) — skipped, never guessed.</param>
/// <param name="External">Resolved to metadata (BCL), a delegate invoke, a lambda or a local function.</param>
/// <param name="SameFile">Callee declared in the calling file.</param>
/// <param name="DuplicateDefinition">Callee or its containing type is declared twice in the merged compilation (CS0101/CS0111).</param>
/// <param name="NotMaskable">Callee expression is not a simple or member name, or the masked call does not fit the continuation.</param>
/// <param name="CrossFile">Usable cross-file call sites.</param>
/// <param name="DefinitionChunks">Distinct required-chunk sets among them.</param>
public sealed record CallSiteCounts(int Invocations, int Unresolved, int External, int SameFile, int DuplicateDefinition,
    int NotMaskable, int CrossFile, int DefinitionChunks);

/// <summary>
/// Generates cross-file code-completion tasks from a corpus. A task masks a call site's callee name and arguments; the
/// callee is resolved with the Roslyn semantic model over the whole corpus (BCL-only references; unresolved or ambiguous
/// symbols are skipped), and the required chunk is the chunk containing the callee's declaration in a different file.
/// Distractors are other chunks, never from a calling file of the plan, never repeating a masked call's text and never
/// declaring a member of the callee's name. Fully deterministic for a corpus and a seed.
/// </summary>
public sealed class CrossFileTaskGenerator
{
    private readonly TaskGeneratorOptions _o;

    /// <summary>Creates a generator with <paramref name="options"/> (defaults when null).</summary>
    public CrossFileTaskGenerator(TaskGeneratorOptions? options = null)
    {
        _o = options ?? new TaskGeneratorOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_o.ContextTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_o.ContinuationTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_o.PrefixTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_o.TasksPerContext);
    }

    /// <summary>The options in use.</summary>
    public TaskGeneratorOptions Options => _o;

    /// <summary>Up to <paramref name="count"/> tasks (the tasks of <see cref="GeneratePlans"/>, in plan order).</summary>
    public IReadOnlyList<CodeTask> Generate(Corpus corpus, int count, int seed) =>
        GeneratePlans(corpus, count, seed).SelectMany(p => p.Tasks).ToArray();

    /// <summary>
    /// Plans holding up to <paramref name="taskCount"/> tasks in total. Definition chunks are visited in a seeded order; each
    /// becomes one plan with up to <see cref="TaskGeneratorOptions.TasksPerContext"/> of its call sites (seeded choice), its
    /// required chunk(s) and seeded distractors up to the context budget, shuffled into a seeded order.
    /// </summary>
    public IReadOnlyList<CodeContextPlan> GeneratePlans(Corpus corpus, int taskCount, int seed)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentOutOfRangeException.ThrowIfNegative(taskCount);
        var rng = new Random(seed);
        var groups = FindCallSites(corpus, out _).GroupBy(s => s.RequiredKey, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => g.ToArray()).ToArray();
        rng.Shuffle(groups);

        foreach (CallSite[] g in groups) rng.Shuffle(g);
        int[] cursor = new int[groups.Length];

        // Pass p gives each definition chunk (in the seeded order) its p-th plan, so every definition is used once before
        // any is used twice.
        var plans = new List<CodeContextPlan>();
        int remaining = taskCount;
        for (bool progress = true; progress && remaining > 0;)
        {
            progress = false;
            for (int gi = 0; gi < groups.Length && remaining > 0; gi++)
            {
                CallSite[] group = groups[gi];
                Chunk[] required = group[0].Required;
                if (cursor[gi] >= group.Length || required.Sum(c => c.TokenIds.Length) > _o.ContextTokens) continue;
                var chosen = new List<(CallSite Site, CodeTask Task)>();
                while (cursor[gi] < group.Length && chosen.Count < Math.Min(_o.TasksPerContext, remaining))
                {
                    CallSite s = group[cursor[gi]++];
                    CodeTask? t = BuildTask(corpus, s);
                    if (t is not null) chosen.Add((s, t));
                }
                if (chosen.Count == 0) continue;
                progress = true;
                plans.Add(BuildPlan(corpus, seed, rng, required, chosen));
                remaining -= chosen.Count;
            }
        }
        return plans;
    }

    /// <summary>
    /// Plans for the call sites named by <paramref name="taskIds"/> (<see cref="CodeTask.Id"/>, which depends neither on the seed
    /// nor on the chunker): the sites are grouped by required chunk(s) (ordinal order of the group key), each group is split into
    /// plans of up to <see cref="TaskGeneratorOptions.TasksPerContext"/> tasks (ordinal task-id order), and each plan gets
    /// seeded distractors and a seeded order exactly as <see cref="GeneratePlans"/> builds them. So one seed re-samples the
    /// contexts around a fixed task set. Ids that match no usable call site (or whose task does not fit the budgets) are
    /// skipped; compare the returned task ids to detect them.
    /// </summary>
    public IReadOnlyList<CodeContextPlan> GeneratePlansFor(Corpus corpus, IEnumerable<string> taskIds, int seed)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(taskIds);
        var wanted = taskIds.ToHashSet(StringComparer.Ordinal);
        var rng = new Random(seed);
        var found = new List<(CallSite Site, CodeTask Task)>();
        var lineIndex = new Dictionary<string, LineIndex>(StringComparer.Ordinal);
        foreach (CallSite s in FindCallSites(corpus, out _))
        {
            if (!lineIndex.TryGetValue(s.Path, out LineIndex? lines)) lineIndex[s.Path] = lines = new LineIndex(corpus.FileAt(s.Path).Text);
            if (!wanted.Contains($"{s.Path}:{lines.LineOf(s.MaskStart)}:{s.MaskStart - lines.LineStartOf(s.MaskStart) + 1}")) continue;
            CodeTask? t = BuildTask(corpus, s);
            if (t is not null) found.Add((s, t));
        }
        var plans = new List<CodeContextPlan>();
        foreach (var group in found.GroupBy(x => x.Site.RequiredKey, StringComparer.Ordinal).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            Chunk[] required = group.First().Site.Required;
            if (required.Sum(c => c.TokenIds.Length) > _o.ContextTokens) continue;
            var ordered = group.OrderBy(x => x.Task.Id, StringComparer.Ordinal).ToList();
            for (int i = 0; i < ordered.Count; i += _o.TasksPerContext)
                plans.Add(BuildPlan(corpus, seed, rng, required, ordered.GetRange(i, Math.Min(_o.TasksPerContext, ordered.Count - i))));
        }
        return plans;
    }

    /// <summary>Required chunk(s) + seeded distractors up to the budget, in a seeded order, with the chosen tasks.</summary>
    private CodeContextPlan BuildPlan(Corpus corpus, int seed, Random rng, Chunk[] required, List<(CallSite Site, CodeTask Task)> chosen)
    {
        int requiredTokens = required.Sum(c => c.TokenIds.Length);

        var callFiles = chosen.Select(c => c.Site.Path).ToHashSet(StringComparer.Ordinal);
        var requiredIds = required.Select(c => c.Id).ToHashSet(StringComparer.Ordinal);
        // Leak check on the call text itself (without the '.'/space the tokenizer glued to the name).
        string[] masked = chosen.Select(c => c.Task.MaskedText.TrimStart('.', '?', ' ')).Distinct(StringComparer.Ordinal).ToArray();
        string[] names = chosen.Select(c => c.Site.CalleeName).Distinct(StringComparer.Ordinal).ToArray();

        int[] pool = Enumerable.Range(0, corpus.Chunks.Count).ToArray();
        rng.Shuffle(pool);
        var context = new List<Chunk>(required);
        int budget = _o.ContextTokens - requiredTokens, distractors = 0;
        foreach (int i in pool)
        {
            if (distractors >= _o.MaxDistractors || budget <= 0) break;
            Chunk c = corpus.Chunks[i];
            if (c.TokenIds.Length > budget || requiredIds.Contains(c.Id) || callFiles.Contains(c.FilePath)) continue;
            if (names.Any(n => corpus.Declares(c, n))) continue;
            string text = corpus.TextOf(c);
            if (masked.Any(m => text.Contains(m, StringComparison.Ordinal))) continue;
            context.Add(c);
            budget -= c.TokenIds.Length;
            distractors++;
        }
        Chunk[] order = context.ToArray();
        rng.Shuffle(order);
        Chunk[] distractorChunks = order.Where(c => !requiredIds.Contains(c.Id)).ToArray();
        return new CodeContextPlan(seed, order, chosen.Select(c => c.Task with { DistractorChunks = distractorChunks }).ToArray());
    }

    /// <summary>Counts the corpus call sites by outcome (no tasks are built).</summary>
    public CallSiteCounts CountCallSites(Corpus corpus)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        List<CallSite> sites = FindCallSites(corpus, out CallSiteCounts counts);
        return counts with { CrossFile = sites.Count, DefinitionChunks = sites.Select(s => s.RequiredKey).Distinct(StringComparer.Ordinal).Count() };
    }

    private sealed record CallSite(string Path, int MaskStart, int MaskEnd, string CalleeName, string Symbol, Chunk[] Required)
    {
        public string RequiredKey { get; } = string.Join('|', Required.Select(c => c.Id));
    }

    private List<CallSite> FindCallSites(Corpus corpus, out CallSiteCounts counts)
    {
        CSharpCompilation compilation = corpus.Compilation;
        HashSet<ISymbol> duplicates = DuplicateDefinitions(compilation);
        int invocations = 0, unresolved = 0, external = 0, sameFile = 0, duplicate = 0, notMaskable = 0;
        var sites = new List<CallSite>();
        foreach (SourceFile file in corpus.Files)
        {
            if (SymbolChunker.IsGenerated(file)) continue;
            SyntaxTree tree = corpus.TreeOf(file.Path);
            SemanticModel model = compilation.GetSemanticModel(tree);
            foreach (InvocationExpressionSyntax inv in tree.GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                if (inv.Expression is IdentifierNameSyntax { Identifier.ValueText: "nameof" }) continue;
                invocations++;
                if (model.GetSymbolInfo(inv).Symbol is not IMethodSymbol called)
                {
                    unresolved++;
                    continue;
                }
                IMethodSymbol def = (called.ReducedFrom ?? called).OriginalDefinition;
                if (def.MethodKind is MethodKind.DelegateInvoke or MethodKind.LocalFunction or MethodKind.AnonymousFunction
                    or MethodKind.BuiltinOperator or MethodKind.FunctionPointerSignature)
                {
                    external++;
                    continue;
                }
                var parts = new List<IMethodSymbol> { def };
                if (def.PartialImplementationPart is { } impl) parts.Add(impl);
                Location[] locations = parts.SelectMany(p => p.Locations).Where(l => l.IsInSource && l.SourceTree is not null
                    && corpus.Contains(l.SourceTree.FilePath)).ToArray();
                if (locations.Length == 0)
                {
                    external++;
                    continue;
                }
                if (locations.Any(l => l.SourceTree!.FilePath == file.Path))
                {
                    sameFile++;
                    continue;
                }
                if (IsDuplicate(def, duplicates))
                {
                    duplicate++;
                    continue;
                }
                SyntaxToken name = inv.Expression switch
                {
                    MemberAccessExpressionSyntax ma => ma.Name.Identifier,
                    MemberBindingExpressionSyntax mb => mb.Name.Identifier,
                    SimpleNameSyntax sn => sn.Identifier,
                    _ => default,
                };
                Chunk?[] required = locations.Select(l => corpus.ChunkAt(l.SourceTree!.FilePath, l.SourceSpan.Start)).Distinct().ToArray();
                if (name.RawKind == 0 || required.Any(c => c is null))
                {
                    notMaskable++;
                    continue;
                }
                sites.Add(new CallSite(file.Path, name.SpanStart, inv.Span.End, name.ValueText,
                    def.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat), required.Select(c => c!).OrderBy(c => c.Id, StringComparer.Ordinal).ToArray()));
            }
        }
        counts = new CallSiteCounts(invocations, unresolved, external, sameFile, duplicate, notMaskable, sites.Count,
            sites.Select(s => s.RequiredKey).Distinct(StringComparer.Ordinal).Count());
        return sites;
    }

    /// <summary>Types and members declared more than once in the merged compilation (CS0101 duplicate type, CS0111 duplicate member).</summary>
    private static HashSet<ISymbol> DuplicateDefinitions(CSharpCompilation compilation)
    {
        var set = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
        foreach (Diagnostic d in compilation.GetDeclarationDiagnostics())
        {
            if (d.Id is not ("CS0101" or "CS0111") || d.Location.SourceTree is not { } tree) continue;
            SemanticModel model = compilation.GetSemanticModel(tree);
            SyntaxNode? node = tree.GetRoot().FindNode(d.Location.SourceSpan);
            for (SyntaxNode? n = node; n is not null; n = n.Parent)
            {
                if (n is MemberDeclarationSyntax && model.GetDeclaredSymbol(n) is { } s)
                {
                    set.Add(s.OriginalDefinition);
                    // CS0101: the type name exists twice; the merged symbol is what call sites bind to.
                    if (s is INamedTypeSymbol t) set.Add(t);
                    break;
                }
            }
        }
        return set;
    }

    private static bool IsDuplicate(IMethodSymbol def, HashSet<ISymbol> duplicates)
    {
        if (duplicates.Count == 0) return false;
        if (duplicates.Contains(def)) return true;
        for (INamedTypeSymbol? t = def.ContainingType; t is not null; t = t.ContainingType)
            if (duplicates.Contains(t.OriginalDefinition)) return true;
        return false;
    }

    /// <summary>
    /// Builds the task for one call site, or null when the masked call does not fit the continuation or the query budget.
    /// The query window (<c>// path</c> header + whole lines before the call + text after it) is tokenised once and split at
    /// the last token boundary at or before the callee name, so prefix ids ++ continuation ids are exactly the natural
    /// tokenisation of the joint text. With a byte-level BPE whose pre-tokenizer glues a leading <c>.</c> or space to the
    /// identifier (Qwen, GPT-2 style), the boundary therefore falls before the <c>.</c>/space, which joins the masked text.
    /// </summary>
    private CodeTask? BuildTask(Corpus corpus, CallSite s)
    {
        var tok = corpus.Tokenizer;
        string text = corpus.FileAt(s.Path).Text;
        int k = _o.ContinuationTokens;
        string header = $"// {s.Path}\n";
        var lines = new LineIndex(text);
        int lineStart = lines.LineStartOf(s.MaskStart);

        // Whole lines before the call, as many as fit the query budget.
        if (tok.EncodeRaw(header + text[lineStart..s.MaskStart]).Length > _o.PrefixTokens) return null;
        int from = lineStart;
        while (from > 0)
        {
            int prev = lines.LineStartOf(from - 1);
            if (tok.EncodeRaw(header + text[prev..s.MaskStart]).Length > _o.PrefixTokens) break;
            from = prev;
        }

        int contEnd = Math.Min(text.Length, s.MaskEnd + 16 * k);
        string window = header + text[from..contEnd];
        int maskInWindow = header.Length + (s.MaskStart - from);
        int[] joint = tok.EncodeRaw(window);
        // cut = number of tokens whose decoded text ends at or before the callee name.
        int cut = 0, cutChars = 0;
        for (int i = 1; i <= joint.Length; i++)
        {
            int len = tok.Decode(joint.AsSpan(0, i)).Length;
            if (len > maskInWindow) break;
            cut = i;
            cutChars = len;
        }
        int boundary = from + cutChars - header.Length; // file offset of the continuation start
        if (cut == 0 || boundary < from || boundary > s.MaskStart) return null;

        string maskedText = text[boundary..s.MaskEnd];
        int[] prefixIds = joint[..cut];
        int[] cont = joint.AsSpan(cut, Math.Min(k, joint.Length - cut)).ToArray();
        int maskTokens = 0;
        for (int m = 1; m <= cont.Length; m++)
        {
            if (tok.Decode(cont.AsSpan(0, m)).Length >= maskedText.Length)
            {
                maskTokens = m;
                break;
            }
        }
        if (maskTokens == 0) return null; // the masked call is longer than the continuation

        return new CodeTask(window[..cutChars], tok.Decode(cont), s.Required, [])
        {
            Id = $"{s.Path}:{lines.LineOf(s.MaskStart)}:{s.MaskStart - lineStart + 1}",
            PrefixTokenIds = prefixIds,
            ContinuationTokenIds = cont,
            MaskTokenCount = maskTokens,
            MaskedText = maskedText,
            CallSitePath = s.Path,
            PrefixStartOffset = from,
            MaskStartOffset = boundary,
            CallSiteLine = lines.LineOf(s.MaskStart),
            Symbol = s.Symbol,
        };
    }
}
