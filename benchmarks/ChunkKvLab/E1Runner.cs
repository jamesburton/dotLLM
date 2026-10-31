using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.Engine.KvCache;

namespace ChunkKvLab;

/// <summary>E1 settings.</summary>
internal sealed record E1Options(
    string ModelPath, string CorpusDir, string Pattern, int Seeds, int[] Sizes, int ChunkTokens, int ContinuationTokens,
    int GreedyTokens, int Threads, string OutCsv);

/// <summary>
/// E1: for each seed and context size, the full prefill (reference), naive concatenation of isolated chunk KV (RoPE-shifted,
/// no recompute) and exact prefix reuse are evaluated on the same held-out continuation.
/// </summary>
internal static class E1Runner
{
    private const string Full = "full";
    private const string Naive = "naive_concat";
    private const string Prefix = "prefix_reuse";

    public static int Run(E1Options o)
    {
        var total = Stopwatch.StartNew();
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        var sw = Stopwatch.StartNew();
        ContextBuilder corpus = ContextBuilder.FromDirectory(o.CorpusDir, o.Pattern, lab.Tokenizer, o.ChunkTokens);
        Console.WriteLine($"corpus {o.CorpusDir} ({o.Pattern}): {corpus.DocumentCount} documents, {corpus.Chunks.Count} chunks of "
            + $"{o.ChunkTokens} tokens, tokenised once in {sw.Elapsed.TotalSeconds:F1} s; chunk-id sha256 {corpus.ChunkIdHash}");
        Console.WriteLine($"KV F32 bytes/token {lab.KvBytesPerToken}; fingerprint {lab.Fingerprint[..16]}…");

        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config);
        int vocab = lab.Config.VocabSize;

        string detailsPath = Path.ChangeExtension(o.OutCsv, null) + "-details.csv";
        using var csv = new StreamWriter(o.OutCsv) { AutoFlush = true };
        using var details = new StreamWriter(detailsPath) { AutoFlush = true };
        csv.WriteLine(RunRow.CsvHeader);
        details.WriteLine("Method,Seed,ContextTokens,MatchedChunks,ContinuationIsSuccessor,BuildMs,FirstTokenMs,KlMax,KlMedian,KlFirst16,KlLast16,EncodeMsPerChunk,GreedyText");

        for (int seed = 0; seed < o.Seeds; seed++)
        {
            var entries = new Dictionary<int, ChunkKvEntry>(); // nested contexts per seed: encode each chunk once
            double encodeMs = 0;
            foreach (int size in o.Sizes)
            {
                ContextPlan plan = corpus.Build(seed, size, o.ContinuationTokens);
                int L = plan.ContextIds.Length;
                int capacity = L + Math.Max(o.ContinuationTokens, o.GreedyTokens) + 1;
                Console.WriteLine($"[seed {seed} ctx {L}] request {string.Join(' ', plan.RequestOrder)}; base {string.Join(' ', plan.BaseOrder)}; "
                    + $"matched {plan.MatchedChunks} chunks; continuation {(plan.ContinuationIsSuccessor ? "document successor" : "unused chunk")}");

                // full (reference) -------------------------------------------------------------------------------
                var refResult = default(MethodResult);
                using (SimpleKvCache cache = lab.NewCache(capacity))
                {
                    sw.Restart();
                    lab.Prefill(cache, plan.ContextIds, 0, o.ChunkTokens);
                    double buildMs = sw.Elapsed.TotalMilliseconds;
                    refResult = Evaluate(lab, cache, L, plan.Continuation, o.GreedyTokens, buildMs);
                    Report(csv, details, lab, Full, plan, refResult, refResult, vocab, L, 0, double.NaN);

                    // prefix_reuse: KV of the matched leading chunks equals the earlier (base) request's KV for the same tokens at the
                    // same positions, so the stored prefix is reproduced by truncating to it; the rest is prefilled.
                    int m = plan.MatchedChunks * o.ChunkTokens;
                    sw.Restart();
                    cache.Rollback(m);
                    lab.Prefill(cache, plan.ContextIds.AsSpan(m), m, o.ChunkTokens);
                    double prefixBuildMs = sw.Elapsed.TotalMilliseconds;
                    MethodResult pr = Evaluate(lab, cache, L, plan.Continuation, o.GreedyTokens, prefixBuildMs);
                    Report(csv, details, lab, Prefix, plan, pr, refResult, vocab, L - m, (long)m * lab.KvBytesPerToken, double.NaN);
                }

                // naive_concat ------------------------------------------------------------------------------------
                foreach (int c in plan.RequestOrder)
                {
                    if (entries.ContainsKey(c)) continue;
                    sw.Restart();
                    entries[c] = encoder.Encode(lab.Model, corpus.Chunks[c].TokenIds);
                    encodeMs += sw.Elapsed.TotalMilliseconds;
                }
                using (SimpleKvCache cache = lab.NewCache(capacity))
                {
                    var list = plan.RequestOrder.Select(c => entries[c]).ToList();
                    sw.Restart();
                    assembler.Assemble(cache, list, lab.Fingerprint);
                    double buildMs = sw.Elapsed.TotalMilliseconds;
                    MethodResult nr = Evaluate(lab, cache, L, plan.Continuation, o.GreedyTokens, buildMs);
                    Report(csv, details, lab, Naive, plan, nr, refResult, vocab, 0, (long)L * lab.KvBytesPerToken, encodeMs / entries.Count);
                }
                Console.WriteLine($"[seed {seed} ctx {L}] done at {total.Elapsed.TotalMinutes:F1} min; {entries.Count} chunks encoded, "
                    + $"{encodeMs / entries.Count:F0} ms/chunk");
            }
        }
        Console.WriteLine($"E1 finished in {total.Elapsed.TotalMinutes:F1} min -> {o.OutCsv}, {detailsPath}");
        return 0;
    }

    /// <summary>Teacher-forced continuation logits, greedy tokens and the first-token forward time for a cache holding the context.</summary>
    private sealed record MethodResult(float[] Rows, int[] Greedy, double BuildMs, double FirstTokenMs);

    private static MethodResult Evaluate(LabModel lab, SimpleKvCache cache, int L, int[] continuation, int greedyTokens, double buildMs)
    {
        if (cache.CurrentLength != L)
            throw new InvalidOperationException($"cache holds {cache.CurrentLength} rows, expected {L}");
        // Greedy first (its first forward is the TTFT tail): feed the first held-out token, then decode.
        var sw = Stopwatch.StartNew();
        float[] last = lab.ForwardLogits(cache, continuation.AsSpan(0, 1), L, allRows: false);
        double firstMs = sw.Elapsed.TotalMilliseconds;
        int[] greedy = new int[greedyTokens];
        greedy[0] = Metrics.ArgMax(last);
        for (int i = 1; i < greedyTokens; i++)
            greedy[i] = Metrics.ArgMax(lab.ForwardLogits(cache, greedy.AsSpan(i - 1, 1), L + i, allRows: false));

        cache.Rollback(L);
        float[] rows = lab.ForwardLogits(cache, continuation, L, allRows: true);
        cache.Rollback(L);
        return new MethodResult(rows, greedy, buildMs, firstMs);
    }

    private static void Report(StreamWriter csv, StreamWriter details, LabModel lab, string method, ContextPlan plan, MethodResult r,
        MethodResult reference, int vocab, long prefillTokens, long cacheBytes, double encodeMsPerChunk)
    {
        var ci = CultureInfo.InvariantCulture;
        int rows = r.Rows.Length / vocab;
        var kls = new double[rows];
        for (int i = 0; i < rows; i++)
            kls[i] = Metrics.KlDivergence(reference.Rows.AsSpan(i * vocab, vocab), r.Rows.AsSpan(i * vocab, vocab));
        var row = new RunRow(method, plan.Seed, plan.ContextIds.Length,
            Metrics.Top1Agreement(reference.Rows, r.Rows, vocab), Metrics.MeanKl(reference.Rows, r.Rows, vocab),
            Metrics.ExactMatch(reference.Greedy, r.Greedy), r.BuildMs + r.FirstTokenMs, prefillTokens, cacheBytes);
        csv.WriteLine(row.ToCsv());

        double[] sorted = (double[])kls.Clone();
        Array.Sort(sorted);
        int q = Math.Min(16, rows);
        string text = lab.Tokenizer.Decode(r.Greedy).Replace("\n", "\\n", StringComparison.Ordinal).Replace("\"", "'", StringComparison.Ordinal);
        details.WriteLine(string.Join(',', method, plan.Seed.ToString(ci), plan.ContextIds.Length.ToString(ci), plan.MatchedChunks.ToString(ci),
            plan.ContinuationIsSuccessor ? "1" : "0", r.BuildMs.ToString("F1", ci), r.FirstTokenMs.ToString("F1", ci),
            sorted[^1].ToString("G6", ci), sorted[rows / 2].ToString("G6", ci), kls.Take(q).Average().ToString("G6", ci),
            kls.Skip(rows - q).Average().ToString("G6", ci), double.IsNaN(encodeMsPerChunk) ? "" : encodeMsPerChunk.ToString("F1", ci),
            "\"" + text + "\""));
        Console.WriteLine($"  {method,-13} top1 {row.Top1Agreement:F3}  KL {row.MeanKl:G4} (max {sorted[^1]:G3})  EM {row.ExactMatch:F3}  "
            + $"TTFT {row.TtftMs:F0} ms (build {r.BuildMs:F0} + first {r.FirstTokenMs:F0})  prefill {prefillTokens}  reusedKV {cacheBytes / (1024.0 * 1024):F0} MiB");
    }
}
