using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #822 measurements on the REAL 28.8 GB IQ4_NL n-gram table. Opt-in (no-ops unless the env var is set): reads GBs from disk and,
/// for the oracle arm, runs the CPU oracle for minutes. Never touches a GPU.
/// <c>DOTLLM_BENCH_822_GGUF</c> = path of shard 1 of the UD-Q4_K_XL file (table-level microbenchmarks);
/// <c>DOTLLM_BENCH_822_ORACLE</c> = off|on (+ <c>DOTLLM_BENCH_822_OUT</c> = file to write the raw logits to) for the 512-token oracle arm.
/// </summary>
public sealed unsafe partial class Qwen4ExpPleTableRealBench(ITestOutputHelper output)
{
    [LibraryImport("psapi.dll", EntryPoint = "EmptyWorkingSet")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool EmptyWorkingSet(nint process);

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private static long[] Rows(Qwen4ExpPleConfigView c, int tokens, int seed)
    {
        var rng = new Random(seed);
        var ids = new int[tokens];
        for (int i = 0; i < tokens; i++) { do ids[i] = rng.Next(0, 248_000); while (ids[i] == c.Eos); }
        var rows = new long[tokens * c.NumHeads];
        Qwen4ExpPle.BuildRowIndices(ids, Enumerable.Repeat(c.Eos, c.Ngram - 1).ToArray(), c.Ngram, c.HeadsPerNgram, c.Eos, c.Mult, c.Offs, c.Vocab, rows);
        return rows;
    }

    private sealed record Qwen4ExpPleConfigView(int Eos, int Ngram, int HeadsPerNgram, int NumHeads, long[] Mult, long[] Offs, long[] Vocab);

    [Fact]
    public void TableLevel_ColdAndWarm_WaitFaultsResidency()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_GGUF");
        if (path is null) return;
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var ple = cfg.Qwen4Exp!.Ple!;
        var view = new Qwen4ExpPleConfigView(ple.EosTokenId, ple.NgramSize, ple.HeadsPerNgram, ple.NumHeads,
            ple.MultipliersOf(0).Select(v => unchecked((long)v)).ToArray(), ple.HeadOffsetsOf(0).Select(v => (long)v).ToArray(),
            ple.HeadVocabSizesOf(0).Select(v => (long)v).ToArray());
        var td = gguf.Tensors.Single(t => t.Name == "per_layer_token_embd.weight");
        nint tbl = gguf.TensorDataPointer(td);
        long tableRows = td.Shape[1]; int rowDim = ple.RowDim;
        var qt = td.QuantizationType;
        long rowBytes = Dequantize.RowByteSize(rowDim, qt);
        int seedBase = Environment.TickCount;
        output.WriteLine($"table {qt} [{rowDim}, {tableRows}] rowBytes {rowBytes} = {rowBytes * tableRows / 1e9:F1} GB; threads {Environment.ProcessorCount}; start {DateTime.Now:HH:mm:ss}");

        // 1. cold probe: plain gather of never-used random rows; us/row says cached (<~2) vs NVMe (tens-hundreds)
        {
            var rows = Rows(view, 128, seedBase + 1);   // 2048 rows
            var dst = new float[rows.Length * rowDim];
            long t = Stopwatch.GetTimestamp();
            Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rows, dst);
            double ms = Ms(Stopwatch.GetTimestamp() - t);
            long t2 = Stopwatch.GetTimestamp();
            Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rows, dst);
            output.WriteLine($"PROBE fresh 2048 rows demand-faulted: {ms:F1} ms = {ms * 1000 / rows.Length:F1} us/row; repeat (warm) {Ms(Stopwatch.GetTimestamp() - t2):F2} ms");
        }

        // 2. per chunk size: demand gather vs prefetch strategies, all on fresh (cold-if-uncached) ids, then warm repeat
        foreach (int tokens in new[] { 512, 1024 })
        {
            for (int rep = 0; rep < 2; rep++)
            {
                output.WriteLine($"-- chunk {tokens} tokens, rep {rep}");
                // demand (no prefetch)
                {
                    var rows = Rows(view, tokens, seedBase + 100 * tokens + rep * 7 + 1);
                    var dst = new float[rows.Length * rowDim];
                    long f0 = ProcessFaults.Current();
                    long t = Stopwatch.GetTimestamp();
                    Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rows, dst);
                    double cold = Ms(Stopwatch.GetTimestamp() - t);
                    long f1 = ProcessFaults.Current();
                    t = Stopwatch.GetTimestamp();
                    Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rows, dst);
                    output.WriteLine($"  demand       : gather {cold,8:F1} ms (faults {f1 - f0}, {cold * 1000 / rows.Length:F1} us/row) ; warm repeat {Ms(Stopwatch.GetTimestamp() - t):F2} ms");
                }
                int arm = 0;
                foreach (var (name, opt) in new (string, PleTableOptions)[]
                {
                    ("os-hint     ", new() { Hint = PleHintMode.Os }),
                    ("touch-8     ", new() { Hint = PleHintMode.Touch, TouchThreads = 8 }),
                    ("touch-16    ", new() { Hint = PleHintMode.Touch, TouchThreads = 16 }),
                    ("touch-32    ", new() { Hint = PleHintMode.Touch, TouchThreads = 32 }),
                    ("os+touch-16 ", new() { Hint = PleHintMode.OsThenTouch, TouchThreads = 16 }),
                    ("ram+touch-16", new() { Mode = PleTableMode.Ram, Hint = PleHintMode.Touch, TouchThreads = 16, CacheMegabytes = 64 }),
                })
                {
                    var rows = Rows(view, tokens, seedBase + 100 * tokens + rep * 7 + 10 + ++arm);
                    var dst = new float[rows.Length * rowDim];
                    using var pf = new PleRowPrefetcher(tbl, qt, tableRows, rowDim, opt with { CollectFaults = true });
                    long f0 = ProcessFaults.Current();
                    long t = Stopwatch.GetTimestamp();
                    var req = pf.Begin(rows);
                    double issue = Ms(Stopwatch.GetTimestamp() - t);
                    pf.Collect(req);                                   // lead time 0: the whole prefetch is exposed
                    double pfTotal = Ms(Stopwatch.GetTimestamp() - t);
                    long f1 = ProcessFaults.Current();
                    t = Stopwatch.GetTimestamp();
                    pf.Gather(null, rows, dst);
                    double gather = Ms(Stopwatch.GetTimestamp() - t);
                    var s = pf.Stats;
                    output.WriteLine($"  {name}: begin {issue:F2} ms, prefetch complete {pfTotal,8:F1} ms (faults {f1 - f0}), then gather {gather:F2} ms ; rows {s.UniqueRows} pages {s.UniquePages} ranges {s.Ranges} worker {s.WorkMicros / 1000.0:F1} ms hintFail {s.HintFailures}");
                }
            }
        }

        // 3. faults/s microbench: random fresh pages across the table, N threads
        foreach (int threads in new[] { 1, 4, 8, 16, 32, 64 })
        {
            const int pagesPerRun = 40_000;
            var rng = new Random(seedBase + 5000 + threads);
            long tableBytes = rowBytes * tableRows;
            var addrs = new long[pagesPerRun];
            for (int i = 0; i < addrs.Length; i++) addrs[i] = (long)tbl + (rng.NextInt64(tableBytes / 4096) * 4096);
            long sink = 0;
            long f0 = ProcessFaults.Current();
            long t = Stopwatch.GetTimestamp();
            Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, k =>
            {
                long acc = 0;
                for (int i = k; i < addrs.Length; i += threads) acc += Volatile.Read(ref *(byte*)addrs[i]);
                Interlocked.Add(ref sink, acc);
            });
            double ms = Ms(Stopwatch.GetTimestamp() - t);
            long df = ProcessFaults.Current() - f0;
            output.WriteLine($"FAULTS threads {threads,2}: {pagesPerRun} random fresh pages in {ms,8:F1} ms = {pagesPerRun / ms * 1000:F0} pages/s ({ms * 1000 / pagesPerRun * threads:F1} us/page/thread), process faults +{df}");
        }

        // 4. Windows mapped-file residency: prefetch 1K tokens, count WS-resident pages, trim the working set, re-gather
        {
            var rows = Rows(view, 1024, seedBase + 9000);
            var dst = new float[rows.Length * rowDim];
            using var pf = new PleRowPrefetcher(tbl, qt, tableRows, rowDim, new PleTableOptions { Hint = PleHintMode.Touch, TouchThreads = 16 });
            var sorted = rows.Distinct().OrderBy(r => r).ToArray();
            var ranges = PleRowPrefetcher.BuildPageRanges(sorted, rowBytes, (long)tbl, 12, out long pages);
            var addrs = ranges.SelectMany(r => Enumerable.Range(0, (int)r.Count).Select(k => (nint)((r.Page + k) << 12))).ToArray();
            long before = ProcessFaults.CountWorkingSetResident(addrs);
            pf.Collect(pf.Begin(rows));
            long after = ProcessFaults.CountWorkingSetResident(addrs);
            // neighbours: pages adjacent to the needed ones that were NOT requested - read-around shows up as extra residency here
            var neigh = ranges.Select(r => (nint)((r.Page + r.Count) << 12)).ToArray();
            long neighRes = ProcessFaults.CountWorkingSetResident(neigh);
            long ws0 = Process.GetCurrentProcess().WorkingSet64;
            var sw = Stopwatch.StartNew();
            bool trimmed = EmptyWorkingSet(Process.GetCurrentProcess().Handle);
            long afterTrim = ProcessFaults.CountWorkingSetResident(addrs);
            long f0 = ProcessFaults.Current();
            long t = Stopwatch.GetTimestamp();
            pf.Gather(null, rows, dst);
            double regather = Ms(Stopwatch.GetTimestamp() - t);
            long df = ProcessFaults.Current() - f0;
            output.WriteLine($"RESIDENCY 1K tokens: needed pages {pages}; WS-resident before prefetch {before}, after {after}; neighbour (unrequested) pages resident {neighRes}/{neigh.Length}");
            output.WriteLine($"RESIDENCY after EmptyWorkingSet({trimmed}): resident {afterTrim}; re-gather (standby soft-faults) {regather:F2} ms, faults +{df}; WS {ws0 / 1e6:F0} MB -> {Process.GetCurrentProcess().WorkingSet64 / 1e6:F0} MB");
        }
        output.WriteLine($"end {DateTime.Now:HH:mm:ss}");
    }

    /// <summary>Decode-sized chunks (1 token = 16 rows): cold demand gather vs OS-hinted prefetch completion vs warm.</summary>
    [Fact]
    public void DecodeLevel_OneTokenChunks()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_GGUF");
        if (path is null) return;
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var ple = cfg.Qwen4Exp!.Ple!;
        var view = new Qwen4ExpPleConfigView(ple.EosTokenId, ple.NgramSize, ple.HeadsPerNgram, ple.NumHeads,
            ple.MultipliersOf(0).Select(v => unchecked((long)v)).ToArray(), ple.HeadOffsetsOf(0).Select(v => (long)v).ToArray(),
            ple.HeadVocabSizesOf(0).Select(v => (long)v).ToArray());
        var td = gguf.Tensors.Single(t => t.Name == "per_layer_token_embd.weight");
        nint tbl = gguf.TensorDataPointer(td);
        long tableRows = td.Shape[1]; int rowDim = ple.RowDim; var qt = td.QuantizationType;
        int seed = Environment.TickCount;
        const int N = 60;
        var demand = new double[N]; var hint = new double[N]; var warm = new double[N]; var hintThenGather = new double[N];
        var dst = new float[16 * rowDim];
        using var pf = new PleRowPrefetcher(tbl, qt, tableRows, rowDim, new PleTableOptions { Hint = PleHintMode.OsThenTouch, TouchThreads = 2 });
        for (int i = 0; i < N; i++)
        {
            var rowsA = Rows(view, 1, seed + 2 * i);
            long t = Stopwatch.GetTimestamp();
            Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rowsA, dst);
            demand[i] = Ms(Stopwatch.GetTimestamp() - t) * 1000;
            t = Stopwatch.GetTimestamp();
            Qwen4ExpPle.GatherRows(tbl, qt, tableRows, rowDim, rowsA, dst);
            warm[i] = Ms(Stopwatch.GetTimestamp() - t) * 1000;
            var rowsB = Rows(view, 1, seed + 2 * i + 1);
            t = Stopwatch.GetTimestamp();
            var req = pf.Begin(rowsB);
            pf.Collect(req);
            hint[i] = Ms(Stopwatch.GetTimestamp() - t) * 1000;
            t = Stopwatch.GetTimestamp();
            pf.Gather(null, rowsB, dst);
            hintThenGather[i] = Ms(Stopwatch.GetTimestamp() - t) * 1000;
        }
        static string Med(double[] v) { var c = v.OrderBy(x => x).ToArray(); return $"median {c[c.Length / 2]:F0} us (p10 {c[c.Length / 10]:F0}, p90 {c[c.Length * 9 / 10]:F0})"; }
        output.WriteLine($"DECODE 1-token (16 rows), fresh ids, {N} trials:");
        output.WriteLine($"  cold demand gather : {Med(demand)}");
        output.WriteLine($"  warm repeat gather : {Med(warm)}");
        output.WriteLine($"  prefetch Begin..Collect (no lead): {Med(hint)}");
        output.WriteLine($"  gather after prefetch: {Med(hintThenGather)}");
    }

    /// <summary>One 512-token oracle forward on the real file with the table service off or on; writes the raw logits for a byte compare.</summary>
    [Fact]
    public void OracleArm_Logits512()
    {
        string? arm = Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_ORACLE");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_GGUF");
        string? outPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_OUT");
        if (arm is null || path is null || outPath is null) return;
        var opts = arm == "on" ? new PleTableOptions { Hint = PleHintMode.OsThenTouch, CollectFaults = true } : PleTableOptions.Disabled;
        PleTableOptions.Ambient = opts;
        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_BENCH_822_THREADS"), out int th) ? th : 0;
        var (m, g, c) = ModelLoader.LoadFromGguf(path, new ThreadingConfig(threads));
        using var _g = g; using var _m = (IDisposable)m;
        var q = (Qwen4ExpTransformerModel)m;
        var rng = new Random(822);
        const int T = 512;
        var ids = new int[T];
        for (int i = 0; i < T; i++) ids[i] = i % 97 == 50 ? c.Qwen4Exp!.Ple!.EosTokenId : rng.Next(1000, 200_000);
        using var st = q.CreateState();
        long ws0 = Process.GetCurrentProcess().WorkingSet64;
        long t = Stopwatch.GetTimestamp();
        using var logits = q.Forward(ids, Enumerable.Range(0, T).ToArray(), -1, st, null, lastTokenLogitsOnly: false);
        double sec = (Stopwatch.GetTimestamp() - t) / (double)Stopwatch.Frequency;
        int n = logits.Shape[0] * logits.Shape[1];
        using (var fs = new FileStream(outPath, FileMode.Create))
            fs.Write(new ReadOnlySpan<byte>((void*)logits.DataPointer, n * 4));
        output.WriteLine($"ORACLE arm={arm} threads={(threads == 0 ? Environment.ProcessorCount : threads)} T={T}: forward {sec:F1} s ({T / sec:F2} tok/s); WS {ws0 / 1e9:F1} -> {Process.GetCurrentProcess().WorkingSet64 / 1e9:F1} GB; stats {q.PleTableStats}");
    }
}
