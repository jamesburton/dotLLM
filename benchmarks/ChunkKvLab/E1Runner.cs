using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.Core.Attention;

namespace ChunkKvLab;

/// <summary>E1 settings.</summary>
/// <param name="ExpectHash">Required corpus chunk-id SHA-256 (pinned snapshot), or null to accept any corpus.</param>
/// <param name="CrossCheck">On Vulkan: also load the CPU model and run CPU naive concatenation on the same contexts, comparing its logits
/// and assembled KV with the Vulkan naive cache (G0 pair (a)).</param>
/// <param name="FirstSeed">First seed (seeds <c>FirstSeed .. FirstSeed + Seeds - 1</c>).</param>
internal sealed record E1Options(
    string ModelPath, string CorpusDir, string Pattern, int Seeds, int[] Sizes, int ChunkTokens, int ContinuationTokens,
    int GreedyTokens, int Threads, string OutCsv, string? ExpectHash = null, bool CrossCheck = false, int FirstSeed = 0);

/// <summary>
/// E1: for each seed and context size, the full prefill (reference), naive concatenation of isolated chunk KV (RoPE-shifted,
/// no recompute) and exact prefix reuse are evaluated on the same held-out continuation, on the selected backend
/// (<c>--backend cpu|vulkan</c>).
/// </summary>
/// <remarks>
/// Besides <c>results.csv</c> (<see cref="RunRow"/>) and <c>-details.csv</c>, it writes:
/// <list type="bullet">
/// <item><c>-timing.csv</c>: wall ms per step (full prefill, prefix suffix prefill, chunk encodes, assembly by host write and by
/// device copy with the backend's write / copy / shift split, first-token forward);</item>
/// <item><c>-kv.csv</c>: per-layer K/V row relative L2 (mean, p99, max) of <c>chunk0_vs_full</c> (the naive cache's first chunk, encoded
/// alone at position 0, against the full prefill's first rows), <c>dev_vs_host</c> (device-copy against host-write assembly) and, with
/// the cross-check, <c>cpu_vs_vk_naive</c> (CPU naive cache as reference);</item>
/// <item><c>-xcheck.csv</c>: logits comparisons that need no reference prefill: <c>prefix_vs_full</c> and <c>dev_vs_host</c> (bit
/// identity expected) and, with the cross-check, <c>cpu_vs_vk_naive</c> (KL(CPU ‖ Vulkan), top-1, greedy EM).</item>
/// </list>
/// Every output is written as <c>*.partial</c> and renamed only when the run finished, so a crash never leaves a complete-looking CSV.
/// </remarks>
internal static class E1Runner
{
    private const string Full = "full";
    private const string Naive = "naive_concat";
    private const string Prefix = "prefix_reuse";

    public static int Run(E1Options o)
    {
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        var total = Stopwatch.StartNew();
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        bool vulkan = lab.Backend.Tag == ChunkBackendTags.Vulkan;
        if (o.CrossCheck && !vulkan)
            throw new ArgumentException("--cross-check cpu needs --backend vulkan (it compares the CPU path against it).");
        Console.WriteLine($"backend {lab.Backend.Tag}, device {lab.DeviceName}");
        using LabModel? cpu = o.CrossCheck ? LabModel.Load(o.ModelPath, o.Threads, ChunkBackendTags.Cpu) : null;

        var sw = Stopwatch.StartNew();
        ContextBuilder corpus = ContextBuilder.FromDirectory(o.CorpusDir, o.Pattern, lab.Tokenizer, o.ChunkTokens);
        Console.WriteLine($"corpus {o.CorpusDir} ({o.Pattern}): {corpus.DocumentCount} documents, {corpus.Chunks.Count} chunks of "
            + $"{o.ChunkTokens} tokens, tokenised once in {sw.Elapsed.TotalSeconds:F1} s; chunk-id sha256 {corpus.ChunkIdHash}");
        if (o.ExpectHash is not null && !string.Equals(corpus.ChunkIdHash, o.ExpectHash, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine($"chunk-id hash {corpus.ChunkIdHash} does not match the expected {o.ExpectHash}: wrong corpus snapshot");
            return 3;
        }
        Console.WriteLine($"KV F32 bytes/token {lab.KvBytesPerToken}; fingerprint {lab.Fingerprint[..16]}…");

        var timed = new TimedBackend(lab.Backend);
        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config, timed);
        int vocab = lab.Config.VocabSize;

        string stem = Path.ChangeExtension(o.OutCsv, null);
        var partials = new List<(string Partial, string Final)>();
        StreamWriter Open(string path, string header)
        {
            partials.Add((path + ".partial", path));
            var w = new StreamWriter(path + ".partial") { AutoFlush = true };
            w.WriteLine(header);
            return w;
        }
        using var csv = Open(o.OutCsv, RunRow.CsvHeader);
        using var details = Open(stem + "-details.csv",
            "Method,Seed,ContextTokens,MatchedChunks,ContinuationIsSuccessor,BuildMs,FirstTokenMs,KlMax,KlMedian,KlFirst16,KlLast16,EncodeMsPerChunk,GreedyText");
        using var timing = Open(stem + "-timing.csv", "backend,seed,context_tokens,step,ms,tokens,calls");
        using var kvCsv = Open(stem + "-kv.csv", "seed,context_tokens,pair,layer,kind,n,rel_l2_mean,rel_l2_p99,rel_l2_max");
        using var xcheck = Open(stem + "-xcheck.csv", "seed,context_tokens,pair,rows,kl_mean,kl_max,top1,bit_identical,em_ref,em_other,greedy_agree");

        // Warm-up (pipelines, allocator) so the first timed forward is not a cold one: one chunk encoded and discarded.
        sw.Restart();
        _ = encoder.Encode(lab.Model, lab.Backend, corpus.Chunks[0].TokenIds);
        Console.WriteLine($"warm-up encode {sw.Elapsed.TotalMilliseconds:F0} ms");

        for (int seed = o.FirstSeed; seed < o.FirstSeed + o.Seeds; seed++)
        {
            var entries = new Dictionary<int, ChunkKvEntry>();    // nested contexts per seed: encode each chunk once
            var cpuEntries = new Dictionary<int, ChunkKvEntry>();
            double encodeMs = 0;
            foreach (int size in o.Sizes)
            {
                ContextPlan plan = corpus.Build(seed, size, o.ContinuationTokens);
                int L = plan.ContextIds.Length;
                int capacity = L + Math.Max(o.ContinuationTokens, o.GreedyTokens) + 1;
                var ctxSw = Stopwatch.StartNew();
                string Step(string s) => $"[seed {seed} ctx {L} +{ctxSw.Elapsed.TotalSeconds:F0} s] {s}";
                void Time(string step, double ms, long tokens, int calls = 1) => timing.WriteLine(string.Join(',', lab.Backend.Tag,
                    seed.ToString(ci), L.ToString(ci), step, ms.ToString("F1", ci), tokens.ToString(ci), calls.ToString(ci)));
                Console.WriteLine($"[seed {seed} ctx {L}] request {string.Join(' ', plan.RequestOrder)}; base {string.Join(' ', plan.BaseOrder)}; "
                    + $"matched {plan.MatchedChunks} chunks; continuation {(plan.ContinuationIsSuccessor ? "document successor" : "unused chunk")}");

                // full (reference) and prefix_reuse ------------------------------------------------------------------------
                MethodResult refResult;
                using IKvCache fullCache = lab.NewBackendCache(capacity);
                Console.WriteLine(Step("full prefill"));
                sw.Restart();
                lab.Prefill(fullCache, plan.ContextIds, 0, o.ChunkTokens);
                double buildMs = sw.Elapsed.TotalMilliseconds;
                Time("full_prefill", buildMs, L, (L + o.ChunkTokens - 1) / o.ChunkTokens);
                refResult = Evaluate(lab, fullCache, L, plan.Continuation, o.GreedyTokens, buildMs);
                Time("first_token_full", refResult.FirstTokenMs, 1);
                Report(csv, details, lab, Full, plan, refResult, refResult, vocab, L, 0, double.NaN);

                // naive_concat ---------------------------------------------------------------------------------------------
                Console.WriteLine(Step("encode chunks"));
                int newChunks = 0;
                foreach (int c in plan.RequestOrder)
                {
                    if (entries.ContainsKey(c)) continue;
                    timed.Reset();
                    sw.Restart();
                    entries[c] = encoder.Encode(lab.Model, timed, corpus.Chunks[c].TokenIds);
                    double ms = sw.Elapsed.TotalMilliseconds;
                    encodeMs += ms;
                    newChunks++;
                    Time("encode_chunk", ms, corpus.Chunks[c].TokenIds.Length);
                    Time("encode_chunk.create_cache", timed.CreateMs, corpus.Chunks[c].TokenIds.Length);
                    Time("encode_chunk.read_rows", timed.ReadMs, corpus.Chunks[c].TokenIds.Length, timed.ReadCalls);
                }
                var list = plan.RequestOrder.Select(c => entries[c]).ToList();
                using IKvCache naiveCache = lab.NewBackendCache(capacity);
                Console.WriteLine(Step("assemble (host write)"));
                timed.Reset();
                sw.Restart();
                assembler.Assemble(naiveCache, list, lab.Fingerprint);
                double naiveBuildMs = sw.Elapsed.TotalMilliseconds;
                Time("assemble_host", naiveBuildMs, L, list.Count);
                Time("assemble_host.write_rows", timed.WriteMs, L, timed.WriteCalls);
                Time("assemble_host.shift_keys", timed.ShiftMs, L, timed.ShiftCalls);

                // chunk 0 at position 0 (one 512-row forward) against the full prefill's first block (the same tokens and row count)
                AddKv(kvCsv, seed, L, "chunk0_vs_full",
                    KvDeviation.Compute(lab.Geometry.LayerCount, list[0].TokenIds.Length, lab.Geometry.KvStrideOf,
                        (l, n, k, v) => lab.Backend.ReadRows(fullCache, l, 0, n, k, v),
                        (l, n, k, v) => lab.Backend.ReadRows(naiveCache, l, 0, n, k, v)));

                MethodResult nr = Evaluate(lab, naiveCache, L, plan.Continuation, o.GreedyTokens, naiveBuildMs);
                Time("first_token_naive", nr.FirstTokenMs, 1);
                Report(csv, details, lab, Naive, plan, nr, refResult, vocab, 0, (long)L * lab.KvBytesPerToken,
                    newChunks > 0 ? encodeMs / entries.Count : double.NaN);

                // prefix_reuse: KV of the matched leading chunks equals the earlier (base) request's KV for the same tokens at the same
                // positions, so the stored prefix is reproduced by truncating the full cache to it; the rest is prefilled again (same
                // 512-row block boundaries, so on a deterministic backend the result is the full prefill bit for bit).
                int m = plan.MatchedChunks * o.ChunkTokens;
                Console.WriteLine(Step("prefix reuse"));
                sw.Restart();
                fullCache.Rollback(m);
                lab.Prefill(fullCache, plan.ContextIds.AsSpan(m), m, o.ChunkTokens);
                double prefixBuildMs = sw.Elapsed.TotalMilliseconds;
                Time("prefix_suffix_prefill", prefixBuildMs, L - m, (L - m + o.ChunkTokens - 1) / o.ChunkTokens);
                MethodResult pr = Evaluate(lab, fullCache, L, plan.Continuation, o.GreedyTokens, prefixBuildMs);
                Report(csv, details, lab, Prefix, plan, pr, refResult, vocab, L - m, (long)m * lab.KvBytesPerToken, double.NaN);
                XCheck(xcheck, seed, L, "prefix_vs_full", refResult, pr, vocab);
                fullCache.Dispose(); // idempotent; frees ~9.4 GB at 32K before the device-copy arm

                // naive by device copy: the same entries made resident (uploaded once), then assembled cache to cache.
                {
                    var resident = new Dictionary<int, ResidentChunkKvEntry>();
                    try
                    {
                        foreach (int c in plan.RequestOrder.Distinct())
                            resident[c] = Upload(lab, entries[c]);
                        using IKvCache devCache = lab.NewBackendCache(capacity);
                        Console.WriteLine(Step("assemble (device copy)"));
                        timed.Reset();
                        sw.Restart();
                        assembler.Assemble(devCache, plan.RequestOrder.Select(c => resident[c]).ToList(), lab.Fingerprint);
                        double devMs = sw.Elapsed.TotalMilliseconds;
                        Time("assemble_device", devMs, L, plan.RequestOrder.Length);
                        Time("assemble_device.copy_rows", timed.CopyMs, L, timed.CopyCalls);
                        Time("assemble_device.shift_keys", timed.ShiftMs, L, timed.ShiftCalls);
                        AddKv(kvCsv, seed, L, "dev_vs_host", FloorMetrics.Deviation(lab.Backend, naiveCache, lab.Backend, devCache, L, lab.Geometry));
                        MethodResult dr = Evaluate(lab, devCache, L, plan.Continuation, o.GreedyTokens, devMs);
                        XCheck(xcheck, seed, L, "dev_vs_host", nr, dr, vocab);
                    }
                    finally
                    {
                        foreach (ResidentChunkKvEntry r in resident.Values) r.Dispose();
                    }
                }

                // cross-check: CPU naive concatenation of the same chunks (G0 pair (a): CPU reference, Vulkan compared) -----------
                if (cpu is not null)
                {
                    Console.WriteLine(Step("cross-check: CPU encode + assembly"));
                    var cpuEncoder = new ChunkEncoder(cpu.Fingerprint);
                    foreach (int c in plan.RequestOrder)
                    {
                        if (cpuEntries.ContainsKey(c)) continue;
                        sw.Restart();
                        cpuEntries[c] = cpuEncoder.Encode(cpu.Model, cpu.Backend, corpus.Chunks[c].TokenIds);
                        Time("cpu_encode_chunk", sw.Elapsed.TotalMilliseconds, corpus.Chunks[c].TokenIds.Length);
                    }
                    using IKvCache cpuNaive = cpu.NewBackendCache(capacity);
                    sw.Restart();
                    new ChunkAssembler(cpu.Config, cpu.Backend).Assemble(cpuNaive, plan.RequestOrder.Select(c => cpuEntries[c]).ToList(), cpu.Fingerprint);
                    Time("cpu_assemble_host", sw.Elapsed.TotalMilliseconds, L, plan.RequestOrder.Length);
                    AddKv(kvCsv, seed, L, "cpu_vs_vk_naive", FloorMetrics.Deviation(cpu.Backend, cpuNaive, lab.Backend, naiveCache, L, lab.Geometry));
                    MethodResult cr = Evaluate(cpu, cpuNaive, L, plan.Continuation, o.GreedyTokens, 0);
                    XCheck(xcheck, seed, L, "cpu_vs_vk_naive", cr, nr, vocab);
                }

                Console.WriteLine(Step($"done at {total.Elapsed.TotalMinutes:F1} min; {entries.Count} chunks encoded, "
                    + $"{encodeMs / entries.Count:F0} ms/chunk"));
            }
            if (cpu is not null) cpuEntries.Clear();
        }

        foreach (StreamWriter w in new[] { csv, details, timing, kvCsv, xcheck }) w.Dispose();
        foreach ((string partial, string final) in partials) File.Move(partial, final, overwrite: true);
        Console.WriteLine($"E1 finished in {total.Elapsed.TotalMinutes:F1} min -> {o.OutCsv} (+ -details, -timing, -kv, -xcheck)");
        return 0;
    }

    /// <summary>Teacher-forced continuation logits, greedy tokens and the first-token forward time for a cache holding the context.</summary>
    internal sealed record MethodResult(float[] Rows, int[] Greedy, double BuildMs, double FirstTokenMs);

    internal static MethodResult Evaluate(LabModel lab, IKvCache cache, int L, int[] continuation, int greedyTokens, double buildMs)
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

    // A host entry made resident: written once into its own backend cache at 0..n-1 (the same bits a resident encode would hold).
    private static ResidentChunkKvEntry Upload(LabModel lab, ChunkKvEntry e)
    {
        IKvCache cache = lab.Backend.CreateCache(e.TokenIds.Length);
        try
        {
            for (int l = 0; l < e.KeysPerLayer.Length; l++)
                lab.Backend.WriteRows(cache, l, 0, e.KeysPerLayer[l], e.ValuesPerLayer[l]);
            return new ResidentChunkKvEntry(e.ModelFingerprint, e.TokenIds, cache, e.Backend);
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }

    private static void AddKv(StreamWriter w, int seed, int L, string pair, KvDeviation d)
    {
        var ci = CultureInfo.InvariantCulture;
        for (int l = 0; l < d.Layers; l++)
        {
            foreach ((string kind, float[] rows) in new[] { ("K", FloorMetrics.KeyRelativeRows(d, l)), ("V", FloorMetrics.ValueRelativeRows(d, l)) })
            {
                FloorMetrics.Summary s = FloorMetrics.Summarize(rows);
                w.WriteLine(string.Join(',', seed.ToString(ci), L.ToString(ci), pair, l.ToString(ci), kind, s.Count.ToString(ci),
                    s.Mean.ToString("G6", ci), s.P99.ToString("G6", ci), s.Max.ToString("G6", ci)));
            }
        }
    }

    // Logits of two evaluations of the same context and continuation: KL(reference ‖ other) per row, top-1 agreement, bit identity of
    // all logits, and the greedy runs (exact match against the full reference is in results.csv; here the two runs against each other).
    private static void XCheck(StreamWriter w, int seed, int L, string pair, MethodResult reference, MethodResult other, int vocab)
    {
        var ci = CultureInfo.InvariantCulture;
        int rows = reference.Rows.Length / vocab;
        double klMax = 0;
        for (int i = 0; i < rows; i++)
            klMax = Math.Max(klMax, Metrics.KlDivergence(reference.Rows.AsSpan(i * vocab, vocab), other.Rows.AsSpan(i * vocab, vocab)));
        bool identical = reference.Rows.AsSpan().SequenceEqual(other.Rows);
        w.WriteLine(string.Join(',', seed.ToString(ci), L.ToString(ci), pair, rows.ToString(ci),
            Metrics.MeanKl(reference.Rows, other.Rows, vocab).ToString("G6", ci), klMax.ToString("G6", ci),
            Metrics.Top1Agreement(reference.Rows, other.Rows, vocab).ToString("R", ci), identical ? "1" : "0",
            string.Join(' ', reference.Greedy), string.Join(' ', other.Greedy), Metrics.ExactMatch(reference.Greedy, other.Greedy).ToString("R", ci)));
        Console.WriteLine($"  check {pair,-16} KL {Metrics.MeanKl(reference.Rows, other.Rows, vocab):G4} (max {klMax:G3})  "
            + $"top1 {Metrics.Top1Agreement(reference.Rows, other.Rows, vocab):F3}  bit-identical {identical}");
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

    /// <summary>Forwards to a backend and accumulates wall time per operation (the assembly split into row writes, copies and shifts).</summary>
    private sealed class TimedBackend(IChunkBackend inner) : IChunkBackend
    {
        public double WriteMs { get; private set; }
        public int WriteCalls { get; private set; }
        public double CopyMs { get; private set; }
        public int CopyCalls { get; private set; }
        public double ShiftMs { get; private set; }
        public int ShiftCalls { get; private set; }
        public double ReadMs { get; private set; }
        public int ReadCalls { get; private set; }
        public double CreateMs { get; private set; }

        public void Reset() => (WriteMs, WriteCalls, CopyMs, CopyCalls, ShiftMs, ShiftCalls, ReadMs, ReadCalls, CreateMs) = (0, 0, 0, 0, 0, 0, 0, 0, 0);

        public string Tag => inner.Tag;
        public int ForwardDeviceId => inner.ForwardDeviceId;
        public void RequireCache(IKvCache cache) => inner.RequireCache(cache);

        public IKvCache CreateCache(int maxSeqLen)
        {
            long t = Stopwatch.GetTimestamp();
            IKvCache c = inner.CreateCache(maxSeqLen);
            CreateMs += Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            return c;
        }

        public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v)
        {
            long t = Stopwatch.GetTimestamp();
            inner.ReadRows(cache, layer, startPos, count, k, v);
            ReadMs += Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            ReadCalls++;
        }

        public void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
        {
            long t = Stopwatch.GetTimestamp();
            inner.WriteRows(cache, layer, startPos, k, v);
            WriteMs += Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            WriteCalls++;
        }

        public void CopyRows(IKvCache source, int layer, int srcStart, IKvCache destination, int dstStart, int count)
        {
            long t = Stopwatch.GetTimestamp();
            inner.CopyRows(source, layer, srcStart, destination, dstStart, count);
            CopyMs += Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            CopyCalls++;
        }

        public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope)
        {
            long t = Stopwatch.GetTimestamp();
            inner.ShiftKeys(cache, startPos, count, delta, rope);
            ShiftMs += Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            ShiftCalls++;
        }
    }
}
