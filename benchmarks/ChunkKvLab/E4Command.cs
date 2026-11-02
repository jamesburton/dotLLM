using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;

namespace ChunkKvLab;

/// <summary>Options of the <c>e4</c> command.</summary>
internal sealed record E4Options(
    string ModelPath, string CorpusDir, string CorpusName, string Chunker, int ChunkTokens, string? ExpectHash, int Seed, int Count,
    TaskGeneratorOptions Generator, string Mode, string? KeptValidationCsv, string? PlansJsonl, int MaxContexts, int[] Budgets,
    string[] Arms, string[] LightArms, int[] LightBudgets, int Window, double MaxMinutes, int Threads, string OutCsv,
    long ChunkCacheTokens, int Shard, int Shards, string ContextKind = "random", double RelatedFraction = 0.5);

/// <summary>
/// E4 (Task 11): structure-aware recompute selection on the Task 8 cross-file tasks. For each context (a
/// <see cref="CodeContextPlan"/>) and each of its tasks: full prefill (reference), the context without the required chunk,
/// naive concatenation, and every selector arm at every budget (tokens recomputed = round(L × p / 100), equal across arms),
/// each evaluated teacher-forced on the task's query prefix + 32-token continuation. Primary metrics: NLL and greedy exact
/// match of the masked call (<see cref="CodeTask.MaskTokenOffset"/>, <see cref="CodeTask.MaskTokenCount"/>); secondary: KL and
/// top-1 against full over the 32 continuation rows (all after the query, the E4 analogue of E2's post-query rows).
/// Query-dependent arms (query relevance, graph, hybrids, oracle definition) select per task; static arms once per context.
/// Identical selections are evaluated once per task (memoised).
/// </summary>
/// <remarks>
/// Contexts: <c>--mode kept</c> takes the call sites the Task 8 validation kept (<c>--kept</c>). For symbol/seed 0 these are the
/// Task 8 plans themselves (verified against <c>--plans-jsonl</c>), restricted to their kept tasks; otherwise
/// <see cref="CrossFileTaskGenerator.GeneratePlansFor"/> re-samples distractors and order with the seed around the same call
/// sites (chunker-specific required chunk). With <c>--context-kind related</c> (E4b) the same kept call sites are grouped the same
/// way, but each context is rebuilt by <see cref="RelatedContextBuilder"/> (a <c>--related-fraction</c> share of the distractors
/// from files within two reference-graph hops of the calling files) and every cell is re-validated (the Task 8 plans are not
/// used). <c>--mode unfiltered</c> takes the first <c>--contexts</c> plans of
/// <see cref="CrossFileTaskGenerator.GeneratePlans"/> with all their tasks.
/// </remarks>
internal static class E4Command
{
    private static readonly string[] StaticArms = ["first_k", "chunk_starts", "deviation", "random"];
    private static readonly string[] TaskArms = ["query", "graph", "hybrid_query", "hybrid_dev", "oracle_def_fill"];

    private sealed record Eval(double MaskNll, double ContNll, bool MaskEm, int GreedyRun, double KlCont, double Top1Cont,
        double KlMask, double Top1Mask, double EvalMs);

    private sealed record Computed(int Calls, double RecomputeMs, Dictionary<int, Eval> ByTask);

    public static int Run(E4Options o)
    {
        var total = Stopwatch.StartNew();
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        var tm = lab.Model as TransformerModel ?? throw new NotSupportedException("E4 needs the CPU TransformerModel.");
        int vocab = lab.Config.VocabSize;
        Corpus corpus = TasksRunner.LoadCorpus(o.CorpusDir, o.Chunker, o.ChunkTokens, lab.Tokenizer, o.ExpectHash);
        var sw = Stopwatch.StartNew();
        ReferenceGraph graph = new RoslynGraphBuilder(lab.Tokenizer).Build(corpus);
        Console.WriteLine($"graph: {graph.Edges.Count} edges in {sw.Elapsed.TotalSeconds:F1} s (offline cost); mapping-mismatch chunks {graph.MappingMismatchChunks}");

        IReadOnlyList<CodeContextPlan> plans = SelectPlans(o, corpus, graph);
        var options = new GraphSignalOptions { DefinitionWindow = o.Window, ChunkStartTokens = 16 };
        Console.WriteLine($"arms {string.Join(',', o.Arms)}; budgets {string.Join(',', o.Budgets)} % (light arms {string.Join(',', o.LightArms)} at "
            + $"{string.Join(',', o.LightBudgets)} %); definition window {o.Window}; shard {o.Shard}/{o.Shards}");

        bool Has(string arm) => o.Arms.Contains(arm);
        int[] BudgetsOf(string arm) => o.LightArms.Contains(arm) ? o.LightBudgets : o.Budgets;

        const string header = "corpus,chunker,seed,mode,plan,context_id,cluster,task_id,task_index,context_tokens,chunks,required_first,prefix_tokens,mask_tokens,"
            + "arm,budget_pct,budget_tokens,graph_picks,query_edges,query_targets,query_targets_first,in_context_edges,forward_calls,score_ms,select_ms,"
            + "recompute_ms,eval_ms,memo,mask_nll,mask_em,cont_nll,greedy_run,kl_cont,top1_cont,kl_mask,top1_mask,context_kind,related_chunks,hop1_chunks";
        // Resumable: rows go to <out>.partial one whole context at a time; <out>.partial.done records each finished context and
        // the cumulative minutes (the time cap counts earlier runs). A finished job renames .partial to <out>.
        Partial? resume = Partial.Open(o.OutCsv, header, RunKey(o, corpus.ChunkIdHash));
        if (resume is null) return 0;
        using Partial csv = resume;
        double priorMinutes = csv.PriorMinutes;
        if (csv.Done.Count > 0)
            Console.WriteLine($"resuming {o.OutCsv}.partial: contexts done {string.Join(' ', csv.Done.Order())}; {priorMinutes:F1} min already spent");

        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config);
        var entryCache = new Dictionary<string, ChunkKvEntry>(StringComparer.Ordinal);
        long cachedTokens = 0;
        int done = 0;
        for (int p = 0; p < plans.Count; p++)
        {
            if (p % o.Shards != o.Shard) continue;
            if (csv.Done.Contains(p)) { done++; continue; }
            if (o.MaxMinutes > 0 && priorMinutes + total.Elapsed.TotalMinutes >= o.MaxMinutes)
            {
                Console.WriteLine($"time cap {o.MaxMinutes} min reached after {done} contexts of this shard");
                break;
            }
            var ctxSw = Stopwatch.StartNew();
            CodeContextPlan plan = plans[p];
            CodeTask[] tasks = plan.Tasks.ToArray();
            int[] ids = plan.ContextTokenIds;
            int L = ids.Length;
            int capacity = L + tasks.Max(t => t.PrefixTokenIds.Length + t.ContinuationTokenIds.Length) + 1;
            int[][] chunkIds = plan.Chunks.Select(c => c.TokenIds).ToArray();
            var ctxBase = new ChunkedContext(chunkIds, []);
            int first = ctxBase.ChunkLength(0);
            bool requiredFirst = plan.RequiredIds.Contains(plan.Chunks[0].Id);
            string contextId = $"{o.CorpusName}/{o.Chunker}/s{o.Seed}/{o.Mode}{(o.ContextKind == "related" ? "-related" : "")}/p{p}";
            // Distractors from the calling files' two-hop neighbourhood (reported for both context kinds).
            IReadOnlyDictionary<string, int> hood = RelatedContextBuilder.Neighbourhood(graph, plan.Tasks);
            Chunk[] distractorChunks = plan.Chunks.Where(c => !plan.RequiredIds.Contains(c.Id)).ToArray();
            int relatedChunks = distractorChunks.Count(c => hood.ContainsKey(c.FilePath));
            int hop1Chunks = distractorChunks.Count(c => hood.TryGetValue(c.FilePath, out int h) && h == 1);
            string cluster = $"{o.CorpusName}|{string.Join('+', tasks[0].RequiredChunks.Select(c => c.FilePath).Distinct(StringComparer.Ordinal))}";
            var graphInfo = new (int Edges, int Targets, int TargetsFirst, int InContext)[tasks.Length];

            void Emit(int ti, string arm, double pct, int budget, int graphPicks, int calls, double scoreMs, double selectMs, double recomputeMs,
                bool memo, Eval e)
            {
                CodeTask t = tasks[ti];
                var g = graphInfo[ti];
                csv.Row(string.Join(',', o.CorpusName, o.Chunker, o.Seed.ToString(ci), o.Mode, p.ToString(ci), contextId, Quote(cluster),
                    Quote(t.Id), ti.ToString(ci), L.ToString(ci), plan.Chunks.Count.ToString(ci), requiredFirst ? "1" : "0",
                    t.PrefixTokenIds.Length.ToString(ci), t.MaskTokenCount.ToString(ci), arm, pct.ToString("F2", ci), budget.ToString(ci),
                    graphPicks.ToString(ci), g.Edges.ToString(ci), g.Targets.ToString(ci), g.TargetsFirst.ToString(ci), g.InContext.ToString(ci),
                    calls.ToString(ci), scoreMs.ToString("F1", ci), selectMs.ToString("F3", ci), recomputeMs.ToString("F0", ci),
                    e.EvalMs.ToString("F0", ci), memo ? "1" : "0", e.MaskNll.ToString("G6", ci), e.MaskEm ? "1" : "0", e.ContNll.ToString("G6", ci),
                    e.GreedyRun.ToString(ci), e.KlCont.ToString("G6", ci), e.Top1Cont.ToString("G6", ci), e.KlMask.ToString("G6", ci),
                    e.Top1Mask.ToString("G6", ci), o.ContextKind, relatedChunks.ToString(ci), hop1Chunks.ToString(ci)));
            }

            // Reference: full prefill, then each task's query + continuation (rows kept for KL/top-1). ------------------------
            using SimpleKvCache full = lab.NewCache(capacity);
            sw.Restart();
            lab.Prefill(full, ids, 0, 512);
            double fullMs = sw.Elapsed.TotalMilliseconds;
            var refRows = new float[tasks.Length][];
            var refEval = new Eval[tasks.Length];
            for (int ti = 0; ti < tasks.Length; ti++)
            {
                var es = Stopwatch.StartNew();
                refRows[ti] = ForwardTask(lab, full, L, tasks[ti]);
                refEval[ti] = Score(refRows[ti], refRows[ti], tasks[ti], vocab, es.Elapsed.TotalMilliseconds);
            }

            // Graph signals per task (prefix only), for the CSV columns and the graph arms. -----------------------------------
            var taskCtx = new ChunkedContext[tasks.Length];
            var signals = new GraphSignals[tasks.Length];
            var signalMs = new double[tasks.Length];
            for (int ti = 0; ti < tasks.Length; ti++)
            {
                taskCtx[ti] = new ChunkedContext(chunkIds, tasks[ti].PrefixTokenIds);
                var gs = Stopwatch.StartNew();
                signals[ti] = GraphSignals.ForTask(graph, taskCtx[ti], plan.Chunks, tasks[ti], options);
                signalMs[ti] = gs.Elapsed.TotalMilliseconds;
                graphInfo[ti] = (signals[ti].QueryEdges, signals[ti].QueryTargets, signals[ti].QueryTargetsInFirstChunk, signals[ti].InContextEdges);
            }
            for (int ti = 0; ti < tasks.Length; ti++)
                Emit(ti, "full", 100, L, 0, 0, 0, 0, fullMs, false, refEval[ti]);

            // Without the required chunk (the lower bound the validity filter used). ------------------------------------------
            if (Has("drop_required"))
            {
                int[] without = plan.ContextWithoutRequired;
                using SimpleKvCache wc = lab.NewCache(without.Length + capacity - L);
                sw.Restart();
                lab.Prefill(wc, without, 0, 512);
                double ms = sw.Elapsed.TotalMilliseconds;
                for (int ti = 0; ti < tasks.Length; ti++)
                {
                    var es = Stopwatch.StartNew();
                    float[] rows = ForwardTask(lab, wc, without.Length, tasks[ti]);
                    Emit(ti, "drop_required", 0, 0, 0, 0, 0, 0, ms, false, Score(rows, refRows[ti], tasks[ti], vocab, es.Elapsed.TotalMilliseconds));
                }
            }

            // Naive concatenation (the stale cache every score source reads). -------------------------------------------------
            var entries = new List<ChunkKvEntry>(plan.Chunks.Count);
            sw.Restart();
            foreach (Chunk c in plan.Chunks)
            {
                if (!entryCache.TryGetValue(c.Id, out ChunkKvEntry? e))
                {
                    if (cachedTokens + c.TokenIds.Length > o.ChunkCacheTokens)
                    {
                        entryCache.Clear();
                        cachedTokens = 0;
                    }
                    entryCache[c.Id] = e = encoder.Encode(lab.Model, c.TokenIds);
                    cachedTokens += c.TokenIds.Length;
                }
                entries.Add(e);
            }
            double encodeMs = sw.Elapsed.TotalMilliseconds;
            using SimpleKvCache stale = lab.NewCache(capacity);
            assembler.Assemble(stale, entries, lab.Fingerprint);
            using SimpleKvCache work = lab.NewCache(capacity);

            var memo = new Dictionary<string, Computed>(StringComparer.Ordinal);
            Computed Run(int[] selection, IEnumerable<int> taskIndices)
            {
                string key = Key(selection);
                if (!memo.TryGetValue(key, out Computed? c)) memo[key] = c = new Computed(-1, 0, []);
                int[] missing = taskIndices.Where(ti => !c.ByTask.ContainsKey(ti)).ToArray();
                if (missing.Length == 0) return c;
                CopyContext(stale, work, L);
                var rs = Stopwatch.StartNew();
                SelectiveRecomputeResult rr = SelectiveRecomputer.Run(lab.Model, work, entries, selection);
                double recMs = rs.Elapsed.TotalMilliseconds;
                if (c.Calls < 0) memo[key] = c = c with { Calls = rr.ForwardCalls, RecomputeMs = recMs };
                foreach (int ti in missing)
                {
                    var es = Stopwatch.StartNew();
                    float[] rows = ForwardTask(lab, work, L, tasks[ti]);
                    c.ByTask[ti] = Score(rows, refRows[ti], tasks[ti], vocab, es.Elapsed.TotalMilliseconds);
                }
                return c;
            }
            int[] allTasks = Enumerable.Range(0, tasks.Length).ToArray();

            {
                Computed naive = Run([], allTasks);
                foreach (int ti in allTasks) Emit(ti, "naive", 0, 0, 0, 0, 0, 0, encodeMs, false, naive.ByTask[ti]);
            }

            // Score sources: layer-1 deviation once per context; query attention per task (its query = the task's prefix). ---
            Layer1KvDeviationSource? dev = null;
            double devMs = 0;
            if (Has("deviation") || Has("hybrid_dev"))
            {
                dev = new Layer1KvDeviationSource(tm, stale);
                sw.Restart();
                dev.Score(ctxBase);
                devMs = sw.Elapsed.TotalMilliseconds;
            }
            var qr = new QueryAttentionSource?[tasks.Length];
            var qrMs = new double[tasks.Length];
            if (Has("query") || Has("hybrid_query"))
                for (int ti = 0; ti < tasks.Length; ti++)
                {
                    qr[ti] = new QueryAttentionSource(tm, stale);
                    sw.Restart();
                    qr[ti]!.Score(taskCtx[ti]);
                    qrMs[ti] = sw.Elapsed.TotalMilliseconds;
                }

            int Budget(int pct) => (int)Math.Round(L * pct / 100.0, MidpointRounding.AwayFromZero);

            // Oracle: the whole required chunk re-encoded in context (its own budget; nothing if it is chunk 0). -------------
            if (Has("oracle_chunk"))
            {
                int[] sel = Enumerable.Range(0, L).Where(i => i >= first && plan.RequiredIds.Contains(plan.Chunks[ctxBase.ChunkOf(i)].Id)).ToArray();
                Computed c = Run(sel, allTasks);
                foreach (int ti in allTasks)
                    Emit(ti, "oracle_chunk", 100.0 * sel.Length / L, sel.Length, 0, c.Calls, 0, 0, c.RecomputeMs, false, c.ByTask[ti]);
            }

            // Static arms: one selection per context, evaluated on every task. -----------------------------------------------
            foreach (string arm in StaticArms.Where(Has))
            {
                IRankedSelector selector = arm switch
                {
                    "first_k" => new FirstKSelector(16),
                    "chunk_starts" => new ChunkStartsSelector(),
                    "deviation" => new DeviationTopKSelector(dev!),
                    _ => new RandomSelector(),
                };
                foreach (int pct in BudgetsOf(arm))
                {
                    int budget = Budget(pct);
                    sw.Restart();
                    int[] sel = selector.Select(ctxBase, budget, RandomSeed(o.Seed, p));
                    double selMs = sw.Elapsed.TotalMilliseconds;
                    bool cached = memo.ContainsKey(Key(sel)) && allTasks.All(ti => memo[Key(sel)].ByTask.ContainsKey(ti));
                    Computed c = Run(sel, allTasks);
                    foreach (int ti in allTasks)
                        Emit(ti, arm, pct, budget, 0, c.Calls, arm == "deviation" ? devMs : 0, selMs, c.RecomputeMs, cached, c.ByTask[ti]);
                }
            }

            // Query-dependent arms: one selection per task. ------------------------------------------------------------------
            if (Has("oracle_def") || Has("oracle_def_fill"))
                for (int ti = 0; ti < tasks.Length; ti++)
                {
                    int[] oracle = OracleDefinition(graph, ctxBase, plan.Chunks, tasks[ti], o.Window);
                    if (Has("oracle_def"))
                    {
                        int[] sel = oracle.Order().ToArray();
                        bool cached = memo.ContainsKey(Key(sel)) && memo[Key(sel)].ByTask.ContainsKey(ti);
                        Computed c = Run(sel, [ti]);
                        Emit(ti, "oracle_def", 100.0 * sel.Length / L, sel.Length, 0, c.Calls, 0, 0, c.RecomputeMs, cached, c.ByTask[ti]);
                    }
                    if (Has("oracle_def_fill"))
                    {
                        int[] rank = PicksThen(oracle, new ChunkStartsSelector().Rank(ctxBase, 0));
                        foreach (int pct in BudgetsOf("oracle_def_fill"))
                        {
                            int budget = Budget(pct);
                            int[] sel = rank[..budget].Order().ToArray();
                            bool cached = memo.ContainsKey(Key(sel)) && memo[Key(sel)].ByTask.ContainsKey(ti);
                            Computed c = Run(sel, [ti]);
                            Emit(ti, "oracle_def_fill", pct, budget, Math.Min(budget, oracle.Length), c.Calls, 0, 0, c.RecomputeMs, cached, c.ByTask[ti]);
                        }
                    }
                }
            foreach (string arm in new[] { "query", "graph", "hybrid_query", "hybrid_dev" }.Where(Has))
                for (int ti = 0; ti < tasks.Length; ti++)
                {
                    // hybrid_dev and graph rank on the query-free context instance (the deviation source memoises on it; graph
                    // signals check token ids only), query arms on the task's own context (its query tokens).
                    (IRankedSelector selector, ChunkedContext ctx, double scoreMs, Func<int, int> picks) = arm switch
                    {
                        "query" => ((IRankedSelector)new QueryRelevanceSelector(qr[ti]!), taskCtx[ti], qrMs[ti], (Func<int, int>)(_ => 0)),
                        "graph" => (new GraphSelector(signals[ti]), ctxBase, signalMs[ti], b => Math.Min(b, signals[ti].Count)),
                        "hybrid_query" => (new HybridSelector(signals[ti], new QueryRelevanceSelector(qr[ti]!)), taskCtx[ti], signalMs[ti] + qrMs[ti],
                            b => Math.Min(b, signals[ti].Count)),
                        _ => (new HybridSelector(signals[ti], new DeviationTopKSelector(dev!)), ctxBase, signalMs[ti] + devMs, b => Math.Min(b, signals[ti].Count)),
                    };
                    foreach (int pct in BudgetsOf(arm))
                    {
                        int budget = Budget(pct);
                        sw.Restart();
                        int[] sel = selector.Select(ctx, budget, 0);
                        double selMs = sw.Elapsed.TotalMilliseconds;
                        bool cached = memo.ContainsKey(Key(sel)) && memo[Key(sel)].ByTask.ContainsKey(ti);
                        Computed c = Run(sel, [ti]);
                        Emit(ti, arm, pct, budget, picks(budget), c.Calls, scoreMs, selMs, c.RecomputeMs, cached, c.ByTask[ti]);
                    }
                }

            done++;
            csv.CommitContext(p, priorMinutes + total.Elapsed.TotalMinutes);
            Console.WriteLine($"[{contextId} L {L} chunks {plan.Chunks.Count} tasks {tasks.Length}{(requiredFirst ? " required=chunk0" : "")}] "
                + $"{ctxSw.Elapsed.TotalSeconds:F0} s (full {fullMs / 1000:F1} s, encode {encodeMs / 1000:F1} s, dev {devMs / 1000:F1} s, "
                + $"query {qrMs.Sum() / 1000:F1} s, {memo.Count} distinct selections); query edges/targets "
                + string.Join(' ', graphInfo.Select(g => $"{g.Edges}/{g.Targets}")) + $"; in-context edges {graphInfo[0].InContext}; "
                + $"related distractors {relatedChunks}/{distractorChunks.Length} (hop 1: {hop1Chunks}); ref EM {refEval.Count(e => e.MaskEm)}/{tasks.Length}; "
                + $"at {priorMinutes + total.Elapsed.TotalMinutes:F1} min");
        }
        csv.Finish();
        Console.WriteLine($"e4 {o.CorpusName}/{o.Chunker}/s{o.Seed}/{o.Mode} shard {o.Shard}/{o.Shards}: {done} contexts in "
            + $"{priorMinutes + total.Elapsed.TotalMinutes:F1} min ({total.Elapsed.TotalMinutes:F1} in this run) -> {o.OutCsv}");
        return 0;
    }

    /// <summary>What must match for a .partial to be resumed: everything that changes the rows except threads and the time cap.</summary>
    private static string RunKey(E4Options o, string corpusHash) => string.Join('|', o.CorpusName, o.Chunker, o.ChunkTokens, corpusHash, o.Seed,
        o.Count, o.Mode, o.MaxContexts, string.Join(',', o.Budgets), string.Join(',', o.Arms), string.Join(',', o.LightArms),
        string.Join(',', o.LightBudgets), o.Window, o.Shard, o.Shards, o.ContextKind, o.RelatedFraction.ToString(CultureInfo.InvariantCulture),
        o.Generator.ContextTokens, o.Generator.ContinuationTokens, o.Generator.PrefixTokens, o.Generator.TasksPerContext);

    /// <summary>
    /// The resumable CSV: <c>&lt;out&gt;.partial</c> holds the rows of finished contexts only (a context's rows are appended and
    /// flushed to disk together, then its index and the cumulative minutes go to <c>&lt;out&gt;.partial.done</c>); on reopen, rows of
    /// contexts not in .done (a torn write) are dropped. Time spent in a context that did not finish is not counted.
    /// </summary>
    private sealed class Partial : IDisposable
    {
        private readonly string _out, _partial, _done;
        private readonly FileStream _stream;
        private readonly StreamWriter _writer;
        private readonly List<string> _pending = [];
        private bool _finished;

        private Partial(string outCsv, FileStream stream, HashSet<int> done, double priorMinutes)
        {
            _out = outCsv;
            _partial = outCsv + ".partial";
            _done = _partial + ".done";
            _stream = stream;
            _writer = new StreamWriter(stream);
            Done = done;
            PriorMinutes = priorMinutes;
        }

        public HashSet<int> Done { get; }

        public double PriorMinutes { get; }

        /// <summary>Opens (or resumes) the partial output; null when <paramref name="outCsv"/> is already complete.</summary>
        public static Partial? Open(string outCsv, string header, string key)
        {
            if (File.Exists(outCsv))
            {
                Console.WriteLine($"{outCsv} exists (job complete); nothing to do");
                return null;
            }
            string partial = outCsv + ".partial", donePath = partial + ".done";
            var done = new HashSet<int>();
            double minutes = 0;
            var keep = new List<string> { header };
            int columns = header.Split(',').Length;
            if (File.Exists(partial) && File.Exists(donePath))
            {
                string[] doneLines = File.ReadAllLines(donePath);
                if (doneLines.Length == 0 || !string.Equals(doneLines[0], "key " + key, StringComparison.Ordinal))
                    throw new InvalidOperationException($"{donePath} was written by a different configuration; move it and {partial} away to start over");
                foreach (string line in doneLines.Skip(1))
                {
                    string[] parts = line.Split(',');
                    if (parts.Length == 2 && int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int p)
                        && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double m))
                    {
                        done.Add(p);
                        minutes = Math.Max(minutes, m);
                    }
                }
                foreach (string line in File.ReadLines(partial).Skip(1))
                {
                    string[] f = line.Split(',');
                    if (f.Length >= columns && int.TryParse(f[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out int p) && done.Contains(p))
                        keep.Add(line);
                }
            }
            else File.WriteAllText(donePath, "key " + key + "\n");
            File.WriteAllLines(partial, keep);
            var stream = new FileStream(partial, FileMode.Append, FileAccess.Write, FileShare.Read);
            return new Partial(outCsv, stream, done, minutes);
        }

        public void Row(string line) => _pending.Add(line);

        public void CommitContext(int plan, double cumulativeMinutes)
        {
            foreach (string line in _pending) _writer.WriteLine(line);
            _pending.Clear();
            _writer.Flush();
            _stream.Flush(flushToDisk: true);
            using (var d = new FileStream(_done, FileMode.Append, FileAccess.Write, FileShare.Read))
            using (var w = new StreamWriter(d))
            {
                w.WriteLine(plan.ToString(CultureInfo.InvariantCulture) + "," + cumulativeMinutes.ToString("F3", CultureInfo.InvariantCulture));
                w.Flush();
                d.Flush(flushToDisk: true);
            }
            Done.Add(plan);
        }

        /// <summary>The job is over (all contexts, or the time cap): .partial becomes the output.</summary>
        public void Finish()
        {
            _writer.Flush();
            _stream.Flush(flushToDisk: true);
            _writer.Dispose();
            _finished = true;
            File.Move(_partial, _out);
            File.Delete(_done);
        }

        public void Dispose()
        {
            if (!_finished) _writer.Dispose();
        }
    }

    /// <summary>The plans (contexts + tasks) this run evaluates; see the class remarks.</summary>
    private static IReadOnlyList<CodeContextPlan> SelectPlans(E4Options o, Corpus corpus, ReferenceGraph graph)
    {
        if (o.ContextKind is not ("random" or "related")) throw new ArgumentException($"unknown --context-kind '{o.ContextKind}' (related|random)");
        if (o.ContextKind == "related" && o.Mode != "kept") throw new ArgumentException("--context-kind related needs --mode kept");
        var gen = new CrossFileTaskGenerator(o.Generator);
        var sw = Stopwatch.StartNew();
        List<CodeContextPlan> plans;
        if (o.Mode == "unfiltered")
        {
            plans = gen.GeneratePlans(corpus, o.Count, o.Seed).ToList();
            if (o.MaxContexts > 0) plans = plans.Take(o.MaxContexts).ToList();
        }
        else if (o.Mode == "kept")
        {
            Dictionary<int, List<string>> keptByPlan = E3Command.ReadKept(o.KeptValidationCsv ?? throw new ArgumentException("--mode kept needs --kept"));
            var kept = keptByPlan.Values.SelectMany(x => x).ToHashSet(StringComparer.Ordinal);
            if (o.ContextKind == "related")
            {
                // Same grouping of the kept call sites as the random contexts (by required chunk, up to --per-context tasks),
                // each context rebuilt from the calling files' neighbourhood with a seed derived from the run seed and the
                // context's first task id.
                plans = gen.GeneratePlansFor(corpus, kept, o.Seed)
                    .Select(pl => RelatedContextBuilder.Build(corpus, pl.Tasks, graph, o.Generator.ContextTokens, o.RelatedFraction,
                        PlanSeed(o.Seed, pl.Tasks.Select(t => t.Id).Min(StringComparer.Ordinal)!)))
                    .ToList();
            }
            else if (o.Chunker == "symbol" && o.Seed == 0 && o.PlansJsonl is not null)
            {
                // The Task 8 contexts themselves (validated there), restricted to their kept tasks.
                IReadOnlyList<CodeContextPlan> all = gen.GeneratePlans(corpus, o.Count, 0);
                E3Command.VerifyPlans(all, o.PlansJsonl);
                plans = all.Where(pl => pl.Tasks.Any(t => kept.Contains(t.Id)))
                    .Select(pl => new CodeContextPlan(pl.Seed, pl.Chunks, pl.Tasks.Where(t => kept.Contains(t.Id)).ToArray())).ToList();
            }
            else plans = gen.GeneratePlansFor(corpus, kept, o.Seed).ToList();
            var found = plans.SelectMany(pl => pl.Tasks).Select(t => t.Id).ToHashSet(StringComparer.Ordinal);
            Console.WriteLine($"kept task ids: {kept.Count}; found {found.Count}; missing {string.Join(' ', kept.Where(k => !found.Contains(k)))}");
            // A CPU-time subset: order the contexts by a hash of their smallest task id (independent of seed and chunker), so
            // every seed and chunker draws the same definitions when --contexts caps the count.
            plans = plans.OrderBy(pl => StableKey(pl.Tasks.Select(t => t.Id).Min(StringComparer.Ordinal)!), StringComparer.Ordinal).ToList();
            if (o.MaxContexts > 0) plans = plans.Take(o.MaxContexts).ToList();
            Console.WriteLine("contexts (first task id): " + string.Join(' ', plans.Select(pl => pl.Tasks.Select(t => t.Id).Min(StringComparer.Ordinal))));
        }
        else throw new ArgumentException($"unknown --mode '{o.Mode}' (kept|unfiltered)");
        Console.WriteLine($"plans: {plans.Count} contexts, {plans.Sum(pl => pl.Tasks.Count)} tasks ({o.Mode}, {o.ContextKind} contexts"
            + (o.ContextKind == "related" ? $" fraction {o.RelatedFraction.ToString(CultureInfo.InvariantCulture)}" : "") + $", seed {o.Seed}, {sw.Elapsed.TotalSeconds:F1} s); "
            + $"mean context {plans.Average(pl => pl.ContextTokenIds.Length):F0} tokens, {plans.Average(pl => pl.Chunks.Count):F1} chunks");
        return plans;
    }

    /// <summary>
    /// ORACLE (uses the answer; not runnable): the definition window of the masked call's callee — the definition end of the
    /// Call edge(s) whose use lies in the masked span — in the context, outside chunk 0.
    /// </summary>
    private static int[] OracleDefinition(ReferenceGraph graph, ChunkedContext ctx, IReadOnlyList<Chunk> chunks, CodeTask task, int window)
    {
        int[] starts = ctx.ChunkStarts;
        var picks = new List<int>();
        foreach (QueryEdge e in graph.QueryEdges(task.CallSitePath, task.MaskStartOffset, task.MaskStartOffset + task.MaskedText.Length))
        {
            if (e.Kind != SymbolEdgeKind.Call) continue;
            for (int k = 1; k < chunks.Count; k++)
            {
                if (graph.IndexOf(chunks[k].Id) != e.DefinitionChunk) continue;
                int end = Math.Min(starts[k] + chunks[k].TokenIds.Length, starts[k] + e.DefinitionToken + window);
                for (int t = starts[k] + e.DefinitionToken; t < end; t++)
                    if (!picks.Contains(t)) picks.Add(t);
                break;
            }
        }
        return picks.ToArray();
    }

    private static int[] PicksThen(int[] picks, int[] rest)
    {
        var taken = picks.ToHashSet();
        return picks.Concat(rest.Where(i => !taken.Contains(i))).ToArray();
    }

    private static string StableKey(string s) => Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(s)));

    private static int PlanSeed(int seed, string firstTaskId) =>
        unchecked(seed * 1_000_003 + int.Parse(StableKey(firstTaskId)[..7], NumberStyles.HexNumber, CultureInfo.InvariantCulture));

    private static int RandomSeed(int seed, int plan) => unchecked(seed * 1_000_003 + plan * 7_919 + 17);

    private static string Key(int[] selection)
    {
        byte[] bytes = new byte[selection.Length * sizeof(int)];
        Buffer.BlockCopy(selection, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes)) + ":" + selection.Length.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>Copies the context rows [0, L) of <paramref name="src"/> into <paramref name="dst"/> (exact F32 rows; no RoPE work).</summary>
    private static void CopyContext(SimpleKvCache src, SimpleKvCache dst, int L)
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

    /// <summary>
    /// Teacher-forced logits of <paramref name="task"/>'s continuation after the context at <c>[0, L)</c>: the prefix minus its
    /// last token is prefilled at <c>L</c>, then [last prefix token, continuation[..^1]] gives one row per continuation token.
    /// The cache is rolled back to <c>L</c>.
    /// </summary>
    private static float[] ForwardTask(LabModel lab, SimpleKvCache cache, int L, CodeTask task)
    {
        int[] prefix = task.PrefixTokenIds, cont = task.ContinuationTokenIds;
        try
        {
            lab.Prefill(cache, prefix.AsSpan(0, prefix.Length - 1), L, 512);
            int[] ids = [prefix[^1], .. cont.AsSpan(0, cont.Length - 1)];
            return lab.ForwardLogits(cache, ids, L + prefix.Length - 1, allRows: true);
        }
        finally
        {
            cache.Rollback(L);
        }
    }

    private static Eval Score(float[] rows, float[] reference, CodeTask task, int vocab, double evalMs)
    {
        int n = task.ContinuationTokenIds.Length, m = task.MaskTokenCount;
        double maskNll = 0, contNll = 0, klCont = 0, klMask = 0;
        int top1 = 0, top1Mask = 0, run = 0;
        bool em = true, running = true;
        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<float> row = rows.AsSpan(j * vocab, vocab), refRow = reference.AsSpan(j * vocab, vocab);
            int target = task.ContinuationTokenIds[j];
            double nll = LogSumExp(row) - row[target];
            int arg = Metrics.ArgMax(row);
            double kl = Metrics.KlDivergence(refRow, row);
            bool agree = arg == Metrics.ArgMax(refRow);
            contNll += nll;
            klCont += kl;
            if (agree) top1++;
            if (running && arg == target) run++;
            else running = false;
            if (j < m)
            {
                maskNll += nll;
                em &= arg == target;
                klMask += kl;
                if (agree) top1Mask++;
            }
        }
        return new Eval(maskNll, contNll, em, run, klCont / n, (double)top1 / n, klMask / m, (double)top1Mask / m, evalMs);
    }

    private static double LogSumExp(ReadOnlySpan<float> x)
    {
        float max = float.NegativeInfinity;
        foreach (float v in x) if (v > max) max = v;
        double s = 0;
        foreach (float v in x) s += Math.Exp(v - max);
        return max + Math.Log(s);
    }

    private static string Quote(string s) => "\"" + s.Replace("\"", "'", StringComparison.Ordinal) + "\"";
}
