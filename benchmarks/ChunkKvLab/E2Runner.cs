using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;

namespace ChunkKvLab;

/// <summary>E2 settings.</summary>
internal sealed record E2Options(
    string ModelPath, string CorpusDir, string Pattern, int Seeds, int[] Sizes, int[] BudgetPercents, string[] Selectors,
    string[] ContiguousSelectors, int ChunkTokens, int ContinuationTokens, int QueryTokens, int GreedyTokens, int Threads,
    string OutCsv, string? ExpectHash, bool NoiseFloor, int FirstSeed);

/// <summary>
/// E2: quality versus recompute budget for the baseline selectors. Per seed and size: full prefill (reference), a numeric
/// noise floor (the same ids prefilled in 128-token blocks instead of 512, and naive + recompute of every token), naive
/// concatenation, then each selector (optionally also as contiguous runs) at each budget: assemble the naive cache, recompute
/// the selection with <see cref="SelectiveRecomputer"/>, evaluate exactly as E1 (teacher-forced 64-token continuation, KL/top-1
/// against full, greedy 16 exact match). The query (= the first <c>QueryTokens</c> continuation tokens, the prompt tail known at
/// request time) is the only post-context text a selector may read; KL/top-1 are also reported over the rows after it.
/// </summary>
internal static class E2Runner
{
    private const string Header =
        "Method,Selector,Contiguous,BudgetPct,BudgetTokens,Seed,ContextTokens,Top1Agreement,MeanKl,Top1PostQuery,KlPostQuery,ExactMatch,"
        + "KlMax,OracleJaccard,ForwardCalls,ScoreMs,BuildMs,RecomputeMs";

    public static int Run(E2Options o)
    {
        var total = Stopwatch.StartNew();
        var ci = CultureInfo.InvariantCulture;
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        var tm = lab.Model as TransformerModel
            ?? throw new NotSupportedException($"E2 needs the CPU TransformerModel; got {lab.Model.GetType().Name}.");
        var sw = Stopwatch.StartNew();
        ContextBuilder corpus = ContextBuilder.FromDirectory(o.CorpusDir, o.Pattern, lab.Tokenizer, o.ChunkTokens);
        string hash = corpus.ChunkIdHash;
        Console.WriteLine($"corpus {o.CorpusDir} ({o.Pattern}): {corpus.DocumentCount} documents, {corpus.Chunks.Count} chunks of "
            + $"{o.ChunkTokens} tokens, tokenised once in {sw.Elapsed.TotalSeconds:F1} s; chunk-id sha256 {hash}");
        if (o.ExpectHash is not null && !string.Equals(o.ExpectHash, hash, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine($"chunk-id hash {hash} does not match the expected {o.ExpectHash}: wrong corpus snapshot");
            return 3;
        }
        Console.WriteLine($"selectors {string.Join(',', o.Selectors)}; contiguous runs (w=8) for {string.Join(',', o.ContiguousSelectors)}; "
            + $"budgets {string.Join(',', o.BudgetPercents)} %; query = first {o.QueryTokens} continuation tokens");

        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config);
        int vocab = lab.Config.VocabSize;
        using var csv = new StreamWriter(o.OutCsv) { AutoFlush = true };
        csv.WriteLine(Header);

        for (int seed = o.FirstSeed; seed < o.FirstSeed + o.Seeds; seed++)
        {
            var entries = new Dictionary<int, ChunkKvEntry>();
            foreach (int size in o.Sizes)
            {
                ContextPlan plan = corpus.Build(seed, size, o.ContinuationTokens);
                int L = plan.ContextIds.Length;
                int capacity = L + Math.Max(o.ContinuationTokens, o.GreedyTokens) + 1;
                var chunkIds = plan.RequestOrder.Select(c => corpus.Chunks[c].TokenIds).ToList();
                var ctx = new ChunkedContext(chunkIds, plan.Continuation.AsSpan(0, o.QueryTokens));
                Console.WriteLine($"[seed {seed} ctx {L}] request {string.Join(' ', plan.RequestOrder)}; "
                    + $"continuation {(plan.ContinuationIsSuccessor ? "document successor" : "unused chunk")}");

                void Emit(string method, string selector, bool contiguous, double pct, int budget, E1Runner.MethodResult r, E1Runner.MethodResult reference,
                    double jaccard, int calls, double scoreMs, double recomputeMs)
                {
                    int rows = r.Rows.Length / vocab;
                    var kls = new double[rows];
                    for (int i = 0; i < rows; i++)
                        kls[i] = Metrics.KlDivergence(reference.Rows.AsSpan(i * vocab, vocab), r.Rows.AsSpan(i * vocab, vocab));
                    int q = o.QueryTokens;
                    double klPost = kls.Skip(q).Average();
                    double top1Post = Metrics.Top1Agreement(reference.Rows.AsSpan(q * vocab), r.Rows.AsSpan(q * vocab), vocab);
                    double top1 = Metrics.Top1Agreement(reference.Rows, r.Rows, vocab);
                    double em = Metrics.ExactMatch(reference.Greedy, r.Greedy);
                    csv.WriteLine(string.Join(',', method, selector, contiguous ? "1" : "0", pct.ToString("F2", ci), budget.ToString(ci),
                        seed.ToString(ci), L.ToString(ci), top1.ToString("G6", ci), kls.Average().ToString("G6", ci), top1Post.ToString("G6", ci),
                        klPost.ToString("G6", ci), em.ToString("G6", ci), kls.Max().ToString("G6", ci),
                        double.IsNaN(jaccard) ? "" : jaccard.ToString("F3", ci), calls.ToString(ci), scoreMs.ToString("F0", ci),
                        r.BuildMs.ToString("F0", ci), recomputeMs.ToString("F0", ci)));
                    Console.WriteLine($"  {method,-28} b={budget,5} top1 {top1:F3} KL {kls.Average():G4} (post-q {klPost:G4}) EM {em:F3}"
                        + $"{(double.IsNaN(jaccard) ? "" : $"  J(oracle) {jaccard:F2}")}  calls {calls}  recompute {recomputeMs / 1000:F1} s  "
                        + $"[{total.Elapsed.TotalMinutes:F1} min]");
                }

                // full (reference) and the numeric noise floor ------------------------------------------------------------
                using SimpleKvCache refCache = lab.NewCache(capacity);
                sw.Restart();
                lab.Prefill(refCache, plan.ContextIds, 0, o.ChunkTokens);
                E1Runner.MethodResult reference = E1Runner.Evaluate(lab, refCache, L, plan.Continuation, o.GreedyTokens, sw.Elapsed.TotalMilliseconds);
                Emit("full", "", false, 100, L, reference, reference, double.NaN, 0, 0, reference.BuildMs);

                foreach (int c in plan.RequestOrder)
                {
                    if (entries.ContainsKey(c)) continue;
                    entries[c] = encoder.Encode(lab.Model, corpus.Chunks[c].TokenIds);
                }
                var list = plan.RequestOrder.Select(c => entries[c]).ToList();

                if (o.NoiseFloor)
                {
                    using (SimpleKvCache c128 = lab.NewCache(capacity))
                    {
                        sw.Restart();
                        lab.Prefill(c128, plan.ContextIds, 0, 128);
                        E1Runner.MethodResult r = E1Runner.Evaluate(lab, c128, L, plan.Continuation, o.GreedyTokens, sw.Elapsed.TotalMilliseconds);
                        Emit("noise_prefill_block128", "", false, 100, L, r, reference, double.NaN, 0, 0, r.BuildMs);
                    }
                    using (SimpleKvCache all = lab.NewCache(capacity))
                    {
                        assembler.Assemble(all, list, lab.Fingerprint);
                        sw.Restart();
                        // Every token: one Forward of the whole context over the assembled cache (a 4096-row prefill block).
                        SelectiveRecomputeResult rr = SelectiveRecomputer.Run(lab.Model, all, list, Enumerable.Range(0, L).ToArray());
                        double ms = sw.Elapsed.TotalMilliseconds;
                        E1Runner.MethodResult r = E1Runner.Evaluate(lab, all, L, plan.Continuation, o.GreedyTokens, 0);
                        Emit("noise_recompute_all", "", false, 100, L, r, reference, double.NaN, rr.ForwardCalls, 0, ms);
                    }
                }

                // naive concatenation (the stale cache every score source reads) --------------------------------------------
                using SimpleKvCache stale = lab.NewCache(capacity);
                assembler.Assemble(stale, list, lab.Fingerprint);
                E1Runner.MethodResult naive = E1Runner.Evaluate(lab, stale, L, plan.Continuation, o.GreedyTokens, 0);
                Emit("naive_concat", "", false, 0, 0, naive, reference, double.NaN, 0, 0, 0);

                // score sources (computed once per context, shared by every budget) -----------------------------------------
                var sources = new Dictionary<string, (ITokenScoreSource Source, double Ms)>();
                (ITokenScoreSource, double) Timed(ITokenScoreSource s)
                {
                    var t = Stopwatch.StartNew();
                    s.Score(ctx);
                    return (s, t.Elapsed.TotalMilliseconds);
                }
                bool Needs(params string[] names) => names.Any(n => o.Selectors.Contains(n));
                if (Needs("deviation", "combined")) sources["deviation"] = Timed(new Layer1KvDeviationSource(tm, stale));
                sources["oracle"] = Timed(new OracleKvDeviationSource(refCache, stale)); // also the Jaccard yardstick
                if (Needs("query")) sources["query"] = Timed(new QueryAttentionSource(tm, stale));
                Console.WriteLine("  score sources: " + string.Join(", ", sources.Select(kv => $"{kv.Key} {kv.Value.Ms / 1000:F1} s")));

                IRankedSelector Make(string name) => name switch
                {
                    "first_k" => new FirstKSelector(16),
                    "chunk_starts" => new ChunkStartsSelector(),
                    "deviation" => new DeviationTopKSelector(sources["deviation"].Source),
                    "deviation_oracle" => new DeviationTopKSelector(sources["oracle"].Source),
                    "random" => new RandomSelector(),
                    "query" => new QueryRelevanceSelector(sources["query"].Source),
                    "combined" => new CombinedSelector(sources["deviation"].Source, startTokens: 16),
                    _ => throw new ArgumentException($"unknown selector '{name}'"),
                };
                double ScoreMs(string name) => name switch
                {
                    "deviation" or "combined" => sources["deviation"].Ms,
                    "deviation_oracle" => sources["oracle"].Ms,
                    "query" => sources["query"].Ms,
                    _ => 0,
                };
                var oracleSel = new DeviationTopKSelector(sources["oracle"].Source);

                var runs = new List<(string Name, bool Contig, int Budget, double Pct)>();
                foreach (string name in o.Selectors)
                {
                    foreach (bool contig in new[] { false, true })
                    {
                        if (contig && !o.ContiguousSelectors.Contains(name)) continue;
                        foreach (int pct in o.BudgetPercents)
                            runs.Add((name, contig, (int)Math.Round(L * pct / 100.0, MidpointRounding.AwayFromZero), pct));
                        if (name == "first_k" && !contig)
                        {
                            int native = 16 * (plan.RequestOrder.Length - 1); // EPIC's own budget, k = 16
                            runs.Add((name, false, native, 100.0 * native / L));
                        }
                    }
                }

                foreach ((string name, bool contig, int budget, double pct) in runs)
                {
                    IRankedSelector sel = contig ? new ContiguousRuns(Make(name), window: 8) : Make(name);
                    int[] picked = sel.Select(ctx, budget, seed);
                    double jac = Jaccard(picked, oracleSel.Select(ctx, budget, seed));
                    using SimpleKvCache work = lab.NewCache(capacity);
                    sw.Restart();
                    assembler.Assemble(work, list, lab.Fingerprint);
                    double buildMs = sw.Elapsed.TotalMilliseconds;
                    sw.Restart();
                    SelectiveRecomputeResult rr = SelectiveRecomputer.Run(lab.Model, work, list, picked);
                    double recMs = sw.Elapsed.TotalMilliseconds;
                    E1Runner.MethodResult r = E1Runner.Evaluate(lab, work, L, plan.Continuation, o.GreedyTokens, buildMs);
                    string method = (contig ? name + "+runs" : name) + "@" + pct.ToString("F1", ci);
                    Emit(method, name, contig, pct, budget, r, reference, jac, rr.ForwardCalls, ScoreMs(name), recMs);
                }
                Console.WriteLine($"[seed {seed} ctx {L}] done at {total.Elapsed.TotalMinutes:F1} min");
            }
        }
        Console.WriteLine($"E2 finished in {total.Elapsed.TotalMinutes:F1} min -> {o.OutCsv}");
        return 0;
    }

    private static double Jaccard(int[] a, int[] b)
    {
        if (a.Length == 0 && b.Length == 0) return 1;
        var sa = a.ToHashSet();
        int inter = b.Count(sa.Contains);
        return (double)inter / (a.Length + b.Length - inter);
    }
}
