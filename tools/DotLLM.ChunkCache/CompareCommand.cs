using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;

namespace DotLLM.ChunkCache;

/// <summary>The context selection of a compare task (the <c>select</c> options).</summary>
internal sealed record TaskSelect
{
    public List<string>? Paths { get; init; }
    public List<string>? Exclude { get; init; }
    public string? Symbol { get; init; }
    public int? Hops { get; init; }
    public string? Since { get; init; }
    public bool MergeBase { get; init; }
    public int? Budget { get; init; }
    public int? MaxPerHop { get; init; }
}

/// <summary>A query taken from a corpus file's lines (1-based, inclusive).</summary>
internal sealed record TaskQueryFrom
{
    public string Path { get; init; } = "";
    public int StartLine { get; init; }
    public int EndLine { get; init; }
}

/// <summary>
/// A <c>compare</c> task file (JSON, snake_case): which chunks form the context, the query that follows it, and optionally the
/// expected continuation (then exact match is reported).
/// </summary>
internal sealed record CompareTask
{
    public string? Name { get; init; }
    public TaskSelect? Select { get; init; }
    public string? Query { get; init; }
    public TaskQueryFrom? QueryFrom { get; init; }
    public string? Continuation { get; init; }
    public int ContinuationTokens { get; init; } = 16;
}

/// <summary>One arm of one chunker in the compare table.</summary>
internal sealed record CompareRow
{
    public string Chunker { get; init; } = "";
    public string Arm { get; init; } = "";
    public double BudgetPct { get; init; }
    public int BudgetTokens { get; init; }
    public int ContextTokens { get; init; }
    public int TokensRecomputed { get; init; }
    public int ForwardCalls { get; init; }
    public double Kl { get; init; }
    public double KlMax { get; init; }
    public double Top1 { get; init; }
    public bool? ExactMatch { get; init; }
    public int? GraphPicks { get; init; }
    public double AssembleMs { get; init; }
    public double ScoreMs { get; init; }
    public double SelectMs { get; init; }
    public double RecomputeMs { get; init; }
    public double QueryMs { get; init; }
    public double ReuseMs { get; init; }
    public double FullMs { get; init; }
    public double Speedup { get; init; }
    public string RecomputePath { get; init; } = "";
    public string PrefillPath { get; init; } = "";
}

/// <summary>Per-chunker facts of a compare run.</summary>
internal sealed record CompareContext
{
    public string Chunker { get; init; } = "";
    public int ContextChunks { get; init; }
    public int ContextTokens { get; init; }
    public int QueryTokens { get; init; }
    public int ContinuationTokens { get; init; }
    public string ContinuationSource { get; init; } = "";
    public int StoreHits { get; init; }
    public int StoreMisses { get; init; }
    public long MissEncodedTokens { get; init; }
    public double FullMs { get; init; }
    public bool? ReferenceExactMatch { get; init; }
    public IReadOnlyList<string> Notes { get; init; } = [];
}

/// <summary>Result of <c>compare</c>.</summary>
internal sealed record CompareResult
{
    public string? Task { get; init; }
    public string PrefillPath { get; init; } = "";
    public string PrefillDescription { get; init; } = "";
    public string RecomputePath { get; init; } = "";
    public string RecomputeDescription { get; init; } = "";
    public IReadOnlyList<CompareContext> Contexts { get; init; } = [];
    public IReadOnlyList<CompareRow> Rows { get; init; } = [];
    public long ElapsedMs { get; init; }
}

/// <summary>
/// <c>chunkcache compare</c>: runs one task (a context selection plus a query) under several chunkers, recompute selectors and
/// budgets and prints a quality-versus-prefill table against full prefill: KL and top-1 agreement over the continuation rows
/// (teacher-forced), exact match when the task gives the expected continuation, tokens recomputed, forward calls, and wall time
/// of reuse (store read + assembly, scoring, recompute, query) against full prefill (context + query). Each timing names the path
/// that produced it.
/// </summary>
/// <remarks>
/// On this branch the only recompute path is the CPU per-run one (<see cref="CpuPerRunRecompute"/>): one forward per contiguous
/// run of selected positions, so scattered selections (chunk starts, query relevance) pay many trunk passes and reuse can be
/// slower than full prefill on CPU. The single-pass Vulkan path plugs in through <see cref="IRecomputeStrategy"/>. The store is
/// read only: misses are encoded in memory (and counted) but not written; run <c>build</c> first for warm timings.
/// </remarks>
internal static class CompareCommand
{
    private const string PrefillPath = "cpu-prefill";
    private static readonly string[] KnownSelectors = ["naive", "chunkstarts", "queryrelevance", "hybrid", "deviation", "random", "firstk"];

    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, Common.SourceOptions, new Dictionary<string, OptionKind>
    {
        ["task"] = OptionKind.Single,
        ["store"] = OptionKind.Single,
        ["selectors"] = OptionKind.List,
        ["budgets"] = OptionKind.List,
        ["chunkers"] = OptionKind.List,
        ["chunk-tokens"] = OptionKind.Single,
        ["seed"] = OptionKind.Single,
        ["block"] = OptionKind.Single,
    });

    public const string Help = """
        chunkcache compare --task FILE.json --model M.gguf --store DIR [options]

          For each chunker, selects the task's context, assembles it from the store (read only; misses encoded in memory),
          and compares naive concatenation and each selector at each budget (percent of the context recomputed) against full
          prefill: KL(full || reuse) and top-1 agreement over the continuation rows, exact match when the task gives the
          continuation, tokens recomputed, forward calls, and wall time of reuse vs full prefill. Each timing names its path;
          on CPU the recompute is per run (one forward per contiguous run of selected positions).

          --selectors a,b,..   naive (always), chunkstarts, queryrelevance, hybrid (graph picks then query relevance),
                               deviation, random, firstk (default chunkstarts,queryrelevance,hybrid)
          --budgets 5,10,20    percent of the context tokens recomputed (default 5,10,20)
          --chunkers fixed,symbol  (default symbol,fixed)      --chunk-tokens N (512)
          --solution X | --directory D   source (default: the one the store's index records for each chunker)
          --seed N (random selector)  --block N (prefill block, 512)  --threads N  --json

          Task file (JSON):
            {"name": "...", "select": {"paths": [..], "symbol": "Type.Member", "hops": 1, "exclude": [..], "since": "REF",
             "merge_base": false, "budget": 2048, "max_per_hop": null},
             "query": "text" | "query_from": {"path": "src/A.cs", "start_line": 10, "end_line": 30},
             "continuation": "expected text (optional)", "continuation_tokens": 16}
          Without "continuation", the reference continuation is full prefill's greedy output and exact match is not reported.
        """;

    public static int Run(CliContext ctx, CliArgs args)
    {
        var total = Stopwatch.StartNew();
        string taskPath = ctx.FullPath(args.Required("task"));
        string storeRoot = ctx.FullPath(args.Required("store"));
        string[] selectors = (args.List("selectors").Count > 0 ? args.List("selectors") : ["chunkstarts", "queryrelevance", "hybrid"])
            .Select(s => s == "query" ? "queryrelevance" : s).Distinct().ToArray();
        foreach (string s in selectors)
            if (!KnownSelectors.Contains(s)) throw new UsageException($"compare: unknown selector '{s}' ({string.Join('|', KnownSelectors)}).");
        double[] budgets = (args.List("budgets").Count > 0 ? args.List("budgets") : ["5", "10", "20"]).Select(b =>
            double.TryParse(b, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v > 0 && v <= 100 ? v
                : throw new UsageException($"compare: budget '{b}' is not a percentage in (0, 100].")).Distinct().ToArray();
        string[] chunkers = (args.List("chunkers").Count > 0 ? args.List("chunkers") : ["symbol", "fixed"]).Select(c => Common.CheckChunkerName(args, c)).Distinct().ToArray();
        int seed = args.Int("seed", 0);
        int block = args.Int("block", 512, 1);
        _ = args.IntOrNull("chunk-tokens", 1, 1 << 20);
        CompareTask task = ReadTask(taskPath);

        using ModelHandle model = Common.OpenModel(ctx, args);
        var tm = model.Model as TransformerModel ?? throw new NotSupportedException("compare needs the CPU TransformerModel.");
        StoreLayout layout = StoreLayout.Open(storeRoot);
        var store = new ReadOnlyChunkStore(layout.Store);
        IRecomputeStrategy strategy = new CpuPerRunRecompute();
        ctx.Info($"recompute path: {strategy.Path} ({strategy.Description}); full prefill path: {PrefillPath} (blocks of {block})");

        var rows = new List<CompareRow>();
        var contexts = new List<CompareContext>();
        var trees = new Dictionary<SourceSpec, SourceTree>();
        foreach (string chunkerName in chunkers)
        {
            IChunker chunker = Common.Chunker(chunkerName, args, model.Tokenizer);
            SourceSpec spec = Common.SourceFor(ctx, args, layout.Find(chunker.Name, model.Producer));
            if (!trees.TryGetValue(spec, out SourceTree? tree))
            {
                trees[spec] = tree = SourceTree.Load(ctx, spec, model.Tokenizer);
                ctx.Info(tree.Summary);
            }
            (CompareContext info, List<CompareRow> armRows) = RunChunker(ctx, model, tm, tree, chunker, task, store, strategy, selectors, budgets, seed, block);
            contexts.Add(info);
            rows.AddRange(armRows);
        }

        var result = new CompareResult
        {
            Task = task.Name ?? Path.GetFileNameWithoutExtension(taskPath), PrefillPath = PrefillPath,
            PrefillDescription = $"CPU forward over the context then the query, blocks of {block} tokens", RecomputePath = strategy.Path,
            RecomputeDescription = strategy.Description, Contexts = contexts, Rows = rows, ElapsedMs = total.ElapsedMilliseconds,
        };
        if (ctx.Json) ctx.WriteJson(result);
        PrintTable(ctx.Text, result); // with --json the table goes to stderr
        return ExitCodes.Ok;
    }

    internal static CompareTask ReadTask(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"Task file not found: {path}", path);
        var options = new JsonSerializerOptions(CliContext.JsonOptions) { UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
        CompareTask? task;
        try
        {
            task = JsonSerializer.Deserialize<CompareTask>(File.ReadAllText(path), options);
        }
        catch (JsonException ex)
        {
            throw new UsageException($"compare: task file {path}: {ex.Message}");
        }
        if (task is null) throw new UsageException($"compare: task file {path} is empty.");
        if (task.Select is null) throw new UsageException("compare: the task needs \"select\".");
        if ((task.Query is null) == (task.QueryFrom is null)) throw new UsageException("compare: the task needs exactly one of \"query\" and \"query_from\".");
        if (task.ContinuationTokens < 1) throw new UsageException("compare: \"continuation_tokens\" must be at least 1.");
        return task;
    }

    private static (CompareContext, List<CompareRow>) RunChunker(CliContext ctx, ModelHandle model, TransformerModel tm, SourceTree tree, IChunker chunker,
        CompareTask task, ReadOnlyChunkStore store, IRecomputeStrategy strategy, string[] selectors, double[] budgets, int seed, int block)
    {
        var sw = Stopwatch.StartNew();
        var notes = new List<string>();
        Chunk[] all = tree.Files.SelectMany(chunker.Chunk).ToArray();
        ReferenceGraph graph = tree.BuildGraph(model.Tokenizer, all);
        TaskSelect ts = task.Select!;
        var request = new SelectionRequest
        {
            Paths = ts.Paths ?? [], Exclude = ts.Exclude ?? [], Symbol = ts.Symbol, Hops = ts.Hops ?? 1, ChangedSince = ts.Since,
            ChangedSinceMergeBase = ts.MergeBase, TokenBudget = ts.Budget, MaxNeighboursPerHop = ts.MaxPerHop, ChunkKind = ChunkRef.KindOf(chunker),
        };
        SelectionResult selection = Common.Select(request, tree, graph);
        Chunk[] chunks = selection.InDocumentOrder().Select(s => graph.Chunks[graph.IndexOf(s.Chunk.Id)]).Where(c => c.TokenIds.Length > 0).ToArray();
        if (chunks.Length == 0) throw new InvalidOperationException($"{chunker.Name}: the task's selection is empty (budget {ts.Budget?.ToString(CultureInfo.InvariantCulture) ?? "none"}).");
        int[][] chunkIds = chunks.Select(c => c.TokenIds).ToArray();
        int[] contextIds = chunkIds.SelectMany(c => c).ToArray();
        int L = contextIds.Length;

        // Query (free text, or a corpus file's lines: then the graph knows its references).
        string? queryFile = null;
        int queryStart = 0, queryEnd = 0;
        string queryText;
        if (task.QueryFrom is { } qf)
        {
            string norm = ChunkIndex.NormalizePath(qf.Path);
            SourceFile file = tree.Files.FirstOrDefault(f => ChunkIndex.PathComparer.Equals(f.Path, norm))
                ?? throw new InvalidOperationException($"query_from: '{qf.Path}' is not a file of the tree.");
            (queryStart, queryEnd) = LineRange(file.Text, qf.StartLine, qf.EndLine);
            queryFile = file.Path;
            queryText = file.Text[queryStart..queryEnd];
        }
        else queryText = task.Query!;
        int[] query = model.Tokenizer.EncodeRaw(queryText);
        if (query.Length == 0) throw new InvalidOperationException("The query encodes to no tokens.");
        int[]? expected = task.Continuation is null ? null : model.Tokenizer.EncodeRaw(task.Continuation).Take(task.ContinuationTokens).ToArray();
        if (expected is { Length: 0 }) throw new InvalidOperationException("The continuation encodes to no tokens.");
        int n = expected?.Length ?? task.ContinuationTokens;
        int capacity = L + query.Length + n + 1;
        int vocab = model.Config.VocabSize;
        ctx.Info($"{chunker.Name}: context {chunks.Length} chunks / {L} tokens (selection {selection.Count} chunks, trimmed {selection.Trimmed.Count}); "
            + $"query {query.Length} tokens; continuation {n} tokens ({(expected is null ? "full prefill's greedy output" : "from the task")}) [{sw.Elapsed.TotalSeconds:F1} s]");

        // Reference: full prefill of context + query; first-token wall time; then the continuation rows.
        float[] refRows;
        int[] continuation;
        double fullMs;
        using (SimpleKvCache full = model.NewCache(capacity))
        {
            sw.Restart();
            Prefill(model, full, contextIds, 0, block);
            float[] first = PrefillLast(model, full, query, L, block);
            fullMs = sw.Elapsed.TotalMilliseconds;
            (refRows, continuation) = Continue(model, full, first, L + query.Length, expected, n, vocab);
        }
        bool? refEm = expected is null ? null : ExactMatch(refRows, expected, vocab);

        // Assemble the stale (naive-concatenation) cache from the store.
        using SimpleKvCache stale = model.NewCache(capacity);
        sw.Restart();
        StoredAssembly assembly = model.Assembler.Assemble(chunks, model.Model, store, stale);
        double assembleMs = sw.Elapsed.TotalMilliseconds;
        if (assembly.Stats.Misses > 0)
            notes.Add($"{assembly.Stats.Misses} of {chunks.Length} chunks were not in the store and were encoded in memory: the reuse timings include "
                + $"{assembly.Stats.EncodedTokens} tokens of encoding (run build first for warm timings)");
        var ctxBase = new ChunkedContext(chunkIds, []);
        var ctxQuery = new ChunkedContext(chunkIds, query);

        // Score sources (once per context; each arm that uses one is charged its time).
        double qrMs = 0, devMs = 0, graphMs = 0;
        QueryAttentionSource? qr = null;
        if (selectors.Contains("queryrelevance") || selectors.Contains("hybrid"))
        {
            qr = new QueryAttentionSource(tm, stale);
            sw.Restart();
            qr.Score(ctxQuery);
            qrMs = sw.Elapsed.TotalMilliseconds;
        }
        Layer1KvDeviationSource? dev = null;
        if (selectors.Contains("deviation"))
        {
            dev = new Layer1KvDeviationSource(tm, stale);
            sw.Restart();
            dev.Score(ctxBase);
            devMs = sw.Elapsed.TotalMilliseconds;
        }
        GraphSignals? signals = null;
        if (selectors.Contains("hybrid"))
        {
            sw.Restart();
            signals = GraphSignals.ForQuery(graph, ctxQuery, chunks, queryFile, queryStart, queryEnd);
            graphMs = sw.Elapsed.TotalMilliseconds;
            if (queryFile is null) notes.Add("hybrid: the query is free text, so it has no graph query edges; graph picks come from edges between context chunks only");
        }

        var rows = new List<CompareRow>
        {
            new()
            {
                Chunker = chunker.Name, Arm = "full", BudgetPct = 100, BudgetTokens = L, ContextTokens = L, TokensRecomputed = L, ForwardCalls = 0,
                Kl = 0, KlMax = 0, Top1 = 1, ExactMatch = refEm, FullMs = fullMs, ReuseMs = fullMs, Speedup = 1, RecomputePath = "-", PrefillPath = PrefillPath,
            },
        };
        using SimpleKvCache work = model.NewCache(capacity);
        CompareRow RunArm(string arm, double pct, int budget, int[] sel, double scoreMs, double selectMs, int? graphPicks)
        {
            CopyContext(stale, work, L);
            RecomputeOutcome rec = strategy.Recompute(model.Model, work, assembly.Entries, sel);
            var qs = Stopwatch.StartNew();
            float[] first = PrefillLast(model, work, query, L, block);
            double queryMs = qs.Elapsed.TotalMilliseconds;
            (float[] armRows, _) = Continue(model, work, first, L + query.Length, continuation, n, vocab);
            (double kl, double klMax, double top1) = Agreement(refRows, armRows, n, vocab);
            double reuse = assembleMs + scoreMs + selectMs + rec.Elapsed.TotalMilliseconds + queryMs;
            return new CompareRow
            {
                Chunker = chunker.Name, Arm = arm, BudgetPct = pct, BudgetTokens = budget, ContextTokens = L, TokensRecomputed = rec.TokensRecomputed,
                ForwardCalls = rec.ForwardCalls, Kl = kl, KlMax = klMax, Top1 = top1, ExactMatch = expected is null ? null : ExactMatch(armRows, expected, vocab),
                GraphPicks = graphPicks, AssembleMs = assembleMs, ScoreMs = scoreMs, SelectMs = selectMs, RecomputeMs = rec.Elapsed.TotalMilliseconds,
                QueryMs = queryMs, ReuseMs = reuse, FullMs = fullMs, Speedup = fullMs / Math.Max(reuse, 1e-9),
                RecomputePath = rec.ForwardCalls == 0 ? "-" : strategy.Path, PrefillPath = PrefillPath,
            };
        }

        rows.Add(RunArm("naive", 0, 0, [], 0, 0, null));
        foreach (string name in selectors.Where(s => s != "naive"))
        {
            (IRankedSelector selector, ChunkedContext c, double scoreMs) = name switch
            {
                "chunkstarts" => ((IRankedSelector)new ChunkStartsSelector(), ctxBase, 0.0),
                "queryrelevance" => (new QueryRelevanceSelector(qr!), ctxQuery, qrMs),
                "hybrid" => (new HybridSelector(signals!, new QueryRelevanceSelector(qr!)), ctxQuery, qrMs + graphMs),
                "deviation" => (new DeviationTopKSelector(dev!), ctxBase, devMs),
                "random" => (new RandomSelector(), ctxBase, 0.0),
                _ => (new FirstKSelector(16), ctxBase, 0.0),
            };
            foreach (double pct in budgets)
            {
                int budget = (int)Math.Round(L * pct / 100.0, MidpointRounding.AwayFromZero);
                var ss = Stopwatch.StartNew();
                int[] sel = selector.Select(c, budget, seed);
                double selectMs = ss.Elapsed.TotalMilliseconds;
                rows.Add(RunArm(name, pct, budget, sel, scoreMs, selectMs, name == "hybrid" ? Math.Min(budget, signals!.Count) : null));
            }
        }
        ctx.Info($"{chunker.Name}: {rows.Count} arms in {sw.Elapsed.TotalSeconds:F1} s");
        var info = new CompareContext
        {
            Chunker = chunker.Name, ContextChunks = chunks.Length, ContextTokens = L, QueryTokens = query.Length, ContinuationTokens = n,
            ContinuationSource = expected is null ? "reference-greedy" : "task", StoreHits = assembly.Stats.Hits, StoreMisses = assembly.Stats.Misses,
            MissEncodedTokens = assembly.Stats.EncodedTokens, FullMs = fullMs, ReferenceExactMatch = refEm, Notes = notes,
        };
        return (info, rows);
    }

    /// <summary>Character offsets [start, end) of lines <paramref name="first"/>..<paramref name="last"/> (1-based, inclusive).</summary>
    internal static (int Start, int End) LineRange(string text, int first, int last)
    {
        if (first < 1 || last < first) throw new InvalidOperationException($"query_from: bad line range {first}-{last}.");
        int line = 1, start = -1, i = 0;
        if (first == 1) start = 0;
        for (; i < text.Length; i++)
        {
            if (text[i] != '\n') continue;
            if (line == last) return (start, i + 1);
            line++;
            if (line == first) start = i + 1;
        }
        if (start < 0) throw new InvalidOperationException($"query_from: the file has {line} lines; line {first} does not exist.");
        return (start, text.Length);
    }

    private static unsafe void Prefill(ModelHandle model, SimpleKvCache cache, ReadOnlySpan<int> ids, int firstPosition, int block)
    {
        for (int off = 0; off < ids.Length; off += block)
        {
            int count = Math.Min(block, ids.Length - off);
            int[] positions = new int[count];
            for (int i = 0; i < count; i++) positions[i] = firstPosition + off + i;
            using ITensor _ = model.Model.Forward(ids.Slice(off, count), positions, -1, cache, lastTokenLogitsOnly: true);
        }
    }

    /// <summary>Prefills <paramref name="ids"/> and returns the logits of its last token (the first generated token's distribution).</summary>
    private static unsafe float[] PrefillLast(ModelHandle model, SimpleKvCache cache, int[] ids, int firstPosition, int block)
    {
        int head = ((ids.Length - 1) / block) * block;
        Prefill(model, cache, ids.AsSpan(0, head), firstPosition, block);
        return Forward(model, cache, ids.AsSpan(head), firstPosition + head, allRows: false);
    }

    private static unsafe float[] Forward(ModelHandle model, SimpleKvCache cache, ReadOnlySpan<int> ids, int firstPosition, bool allRows)
    {
        int[] positions = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) positions[i] = firstPosition + i;
        using ITensor logits = model.Model.Forward(ids, positions, -1, cache, lastTokenLogitsOnly: !allRows);
        int rows = logits.Shape[0], vocab = logits.Shape[1];
        int firstRow = allRows ? 0 : rows - 1, count = allRows ? rows : 1;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)firstRow * vocab, count * vocab).ToArray();
    }

    /// <summary>
    /// The <paramref name="n"/> continuation rows: row 0 is <paramref name="first"/>; the rest come from forwarding the
    /// continuation teacher-forced (<paramref name="forced"/>), or, when it is null, greedily from the rows themselves.
    /// </summary>
    private static (float[] Rows, int[] Tokens) Continue(ModelHandle model, SimpleKvCache cache, float[] first, int position, int[]? forced, int n, int vocab)
    {
        var rows = new float[(long)n * vocab];
        first.CopyTo(rows, 0);
        if (forced is not null)
        {
            if (n > 1) Forward(model, cache, forced.AsSpan(0, n - 1), position, allRows: true).CopyTo(rows, vocab);
            return (rows, forced);
        }
        int[] tokens = new int[n];
        for (int j = 0; j < n; j++)
        {
            tokens[j] = ArgMax(rows.AsSpan(j * vocab, vocab));
            if (j + 1 < n) Forward(model, cache, [tokens[j]], position + j, allRows: false).CopyTo(rows, (j + 1) * vocab);
        }
        return (rows, tokens);
    }

    /// <summary>Mean and max KL(reference || arm) over the rows, and the share of rows whose argmax agrees.</summary>
    internal static (double Kl, double KlMax, double Top1) Agreement(float[] reference, float[] arm, int n, int vocab)
    {
        double sum = 0, max = 0;
        int agree = 0;
        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<float> p = reference.AsSpan(j * vocab, vocab), q = arm.AsSpan(j * vocab, vocab);
            double kl = Kl(p, q);
            sum += kl;
            max = Math.Max(max, kl);
            if (ArgMax(p) == ArgMax(q)) agree++;
        }
        return (sum / n, max, (double)agree / n);
    }

    /// <summary>KL(softmax(p) || softmax(q)) in nats, clamped at 0 (rounding).</summary>
    internal static double Kl(ReadOnlySpan<float> p, ReadOnlySpan<float> q)
    {
        double lp = LogSumExp(p), lq = LogSumExp(q), kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double a = p[i] - lp;
            kl += Math.Exp(a) * (a - (q[i] - lq));
        }
        return Math.Max(kl, 0);
    }

    private static bool ExactMatch(float[] rows, int[] expected, int vocab)
    {
        for (int j = 0; j < expected.Length; j++)
            if (ArgMax(rows.AsSpan(j * vocab, vocab)) != expected[j]) return false;
        return true;
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private static double LogSumExp(ReadOnlySpan<float> x)
    {
        float max = float.NegativeInfinity;
        foreach (float v in x) if (v > max) max = v;
        double s = 0;
        foreach (float v in x) s += Math.Exp(v - max);
        return max + Math.Log(s);
    }

    /// <summary>Copies rows [0, L) of <paramref name="src"/> into <paramref name="dst"/> (exact F32 rows), after rolling it back to 0.</summary>
    private static unsafe void CopyContext(SimpleKvCache src, SimpleKvCache dst, int L)
    {
        dst.Rollback(0);
        int[] positions = Enumerable.Range(0, L).ToArray();
        for (int l = 0; l < src.NumLayers; l++)
        {
            int stride = src.KvStrideOf(l);
            TensorRef k = src.GetKeysRef(l), v = src.GetValuesRef(l);
            dst.Update(new TensorRef(L, stride, DType.Float32, -1, k.DataPointer), new TensorRef(L, stride, DType.Float32, -1, v.DataPointer), positions, l);
        }
    }

    private static void PrintTable(TextWriter o, CompareResult r)
    {
        var ci = CultureInfo.InvariantCulture;
        o.WriteLine($"compare {r.Task}");
        o.WriteLine($"  full prefill: path {r.PrefillPath} ({r.PrefillDescription})");
        o.WriteLine($"  reuse: store read + RoPE assembly, scoring, recompute on path {r.RecomputePath} ({r.RecomputeDescription}), then the query");
        foreach (CompareContext c in r.Contexts)
        {
            o.WriteLine($"  {c.Chunker}: {c.ContextChunks} chunks, {c.ContextTokens} context tokens, query {c.QueryTokens}, continuation {c.ContinuationTokens} ({c.ContinuationSource}); "
                + $"store hits {c.StoreHits}, misses {c.StoreMisses}; full prefill {c.FullMs:F0} ms" + (c.ReferenceExactMatch is bool em ? $"; reference EM {(em ? "yes" : "no")}" : ""));
            foreach (string n in c.Notes) o.WriteLine($"    note: {n}");
        }
        o.WriteLine();
        o.WriteLine(string.Format(ci, "| {0,-10} | {1,-14} | {2,6} | {3,10} | {4,5} | {5,9} | {6,5} | {7,3} | {8,9} | {9,9} | {10,7} | {11,-11} |",
            "chunker", "arm", "budget", "recomputed", "calls", "KL", "top-1", "EM", "reuse ms", "full ms", "speedup", "path"));
        o.WriteLine("|------------|----------------|--------|------------|-------|-----------|-------|-----|-----------|-----------|---------|-------------|");
        foreach (CompareRow x in r.Rows)
        {
            o.WriteLine(string.Format(ci, "| {0,-10} | {1,-14} | {2,5:0.#}% | {3,10} | {4,5} | {5,9:0.0000} | {6,5:0.00} | {7,3} | {8,9:F0} | {9,9:F0} | {10,6:0.00}x | {11,-11} |",
                x.Chunker, x.Arm, x.BudgetPct, x.TokensRecomputed, x.ForwardCalls, x.Kl, x.Top1, x.ExactMatch is bool e ? (e ? "yes" : "no") : "-",
                x.ReuseMs, x.FullMs, x.Speedup, x.Arm == "full" ? x.PrefillPath : x.RecomputePath));
        }
        o.WriteLine();
        o.WriteLine("  reuse ms = assembly (store read + RoPE) + scoring + selection + recompute + query prefill; full ms = context + query prefill.");
        o.WriteLine($"  On CPU the recompute is per run ({r.RecomputePath}): scattered selections pay one forward per run, so reuse can be slower than full prefill here.");
    }
}
