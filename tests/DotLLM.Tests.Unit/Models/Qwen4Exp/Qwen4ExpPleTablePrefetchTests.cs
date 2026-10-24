using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// N-gram (PLE) table prefetch / offload service (#822). The contract under test is the correctness invariant: the service changes
/// WHEN pages become resident (and which copy of the same bytes is dequantised), never WHAT is gathered — bit for bit.
/// </summary>
public sealed unsafe class Qwen4ExpPleTablePrefetchTests(ITestOutputHelper output) : IDisposable
{
    private readonly List<IDisposable> _cleanup = [];
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-ple-pf-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        foreach (var d in _cleanup) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>A file-backed lazy mapping of random bytes: the same kind of memory the real table is (a view of a file).</summary>
    private sealed class MappedTable : IDisposable
    {
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _view;
        public nint Ptr { get; }
        public MappedTable(string path, long bytes, int seed)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
            {
                var rng = new Random(seed);
                var buf = new byte[1 << 20];
                for (long done = 0; done < bytes; done += buf.Length)
                {
                    rng.NextBytes(buf);
                    fs.Write(buf, 0, (int)Math.Min(buf.Length, bytes - done));
                }
            }
            _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            Ptr = (nint)p;
        }
        public void Dispose() { _view.SafeMemoryMappedViewHandle.ReleasePointer(); _view.Dispose(); _mmf.Dispose(); }
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{what}: element {i} differs ({expected[i]:R} vs {actual[i]:R})");
    }

    // ── page coalescing ─────────────────────────────────────────────────────

    [Fact]
    public void PageRanges_AreSortedUniqueCoalesced_AndCountStraddlingRows()
    {
        // 90-byte rows (IQ4_NL, rowDim 160) at a 4 KiB page: row 45 starts at byte 4050 and straddles pages 0|1.
        var rows = new long[] { 0, 1, 45, 46, 100, 1000, 1001 };
        var ranges = PleRowPrefetcher.BuildPageRanges(rows, rowBytes: 90, baseAddress: 0, pageShift: 12, out long pages);
        Assert.Equal(pages, ranges.Sum(r => r.Count));
        long prevEnd = -1;
        foreach (var (page, count) in ranges) { Assert.True(page > prevEnd, "ranges overlap or touch"); prevEnd = page + count - 1; }
        // rows 0,1 -> page 0; 45 -> pages 0..1 (merges with the first range); 46 -> page 1; 100 -> page 2; 1000 -> page 21; 1001 -> 21..22
        Assert.Equal([(0L, 3L), (21L, 2L)], ranges.ToArray());
        Assert.Equal(5, pages);
    }

    [Fact]
    public void PageRanges_UnalignedBase_ShiftsPages()
    {
        var ranges = PleRowPrefetcher.BuildPageRanges([0], rowBytes: 90, baseAddress: 4096 - 10, pageShift: 12, out long pages);
        Assert.Equal(2, pages);   // the row straddles the boundary because the table base is not page aligned
        Assert.Single(ranges);
    }

    // ── gather equivalence per format and arm ──────────────────────────────

    public static IEnumerable<object[]> FormatArms()
    {
        foreach (var qt in new[] { QuantizationType.IQ4_NL, QuantizationType.BF16, QuantizationType.Q8_0, QuantizationType.F16, QuantizationType.F32 })
        foreach (var hint in Enum.GetValues<PleHintMode>())
        foreach (var mode in new[] { PleTableMode.Mmap, PleTableMode.Ram })
            yield return new object[] { qt, hint, mode };
    }

    [Theory]
    [MemberData(nameof(FormatArms))]
    public void Gather_IsBitIdenticalToPlainGatherRows(QuantizationType qt, PleHintMode hint, PleTableMode mode)
    {
        const int rowDim = 160;
        const long tableRows = 50_000;
        long rowBytes = Dequantize.RowByteSize(rowDim, qt);
        Directory.CreateDirectory(_dir);
        var table = new MappedTable(Path.Combine(_dir, $"t-{qt}-{hint}-{mode}.bin"), tableRows * rowBytes, seed: 7);
        _cleanup.Add(table);
        var rng = new Random(11);
        var rows = new long[2048];   // with repeats, ascending and descending runs, first and last row
        for (int i = 0; i < rows.Length; i++) rows[i] = rng.Next(0, 3) == 0 ? rows[Math.Max(0, i - 1 - rng.Next(8))] : rng.NextInt64(tableRows);
        rows[0] = 0; rows[^1] = tableRows - 1;

        var expected = new float[rows.Length * rowDim];
        Qwen4ExpPle.GatherRows(table.Ptr, qt, tableRows, rowDim, rows, expected);

        var opts = new PleTableOptions { Mode = mode, Hint = hint, CacheMegabytes = mode == PleTableMode.Ram ? 1 : 0, TouchThreads = 3, CollectFaults = true };
        using var pf = new PleRowPrefetcher(table.Ptr, qt, tableRows, rowDim, opts);
        var req = pf.Begin(rows);
        Assert.Equal(hint != PleHintMode.Off, req is not null);
        var got = new float[expected.Length];
        pf.Gather(req, rows, got);
        AssertBitEqual(expected, got, $"{qt}/{hint}/{mode} first gather");

        var again = new float[expected.Length];   // second pass: cache hits (ram) / warm pages (mmap)
        pf.Gather(pf.Begin(rows), rows, again);
        AssertBitEqual(expected, again, $"{qt}/{hint}/{mode} second gather");

        var s = pf.Stats;
        Assert.Equal(2L * rows.Length, s.Rows);
        if (hint != PleHintMode.Off) { Assert.True(s.UniquePages > 0); Assert.True(s.Ranges > 0); Assert.True(s.UniqueRows <= 2 * rows.Length); }
        if (mode == PleTableMode.Ram) Assert.True(s.CacheHits > 0, "the second pass should be served from the row cache");
    }

    [Fact]
    public void RowCache_ClockEviction_StaysCorrectWhenFarSmallerThanTheWorkingSet()
    {
        const int rowDim = 160; const long tableRows = 20_000;
        var qt = QuantizationType.IQ4_NL;
        long rowBytes = Dequantize.RowByteSize(rowDim, qt);
        Directory.CreateDirectory(_dir);
        var table = new MappedTable(Path.Combine(_dir, "evict.bin"), tableRows * rowBytes, seed: 3);
        _cleanup.Add(table);
        // 1 MiB cache -> ~11.6k rows; use a tiny budget through a hand-sized cache: 2 KiB would round to 0, so use 1 MiB and a
        // working set of 3x its capacity so the clock hand wraps repeatedly.
        var opts = new PleTableOptions { Mode = PleTableMode.Ram, Hint = PleHintMode.OsThenTouch, CacheMegabytes = 1 };
        using var pf = new PleRowPrefetcher(table.Ptr, qt, tableRows, rowDim, opts);
        Assert.InRange(pf.CacheCapacityRows, 1, (int)tableRows);
        var rng = new Random(5);
        for (int round = 0; round < 12; round++)
        {
            var rows = Enumerable.Range(0, 3000).Select(_ => rng.NextInt64(tableRows)).ToArray();
            var expected = new float[rows.Length * rowDim];
            Qwen4ExpPle.GatherRows(table.Ptr, qt, tableRows, rowDim, rows, expected);
            var got = new float[expected.Length];
            pf.Gather(pf.Begin(rows), rows, got);
            AssertBitEqual(expected, got, $"round {round}");
        }
        Assert.True(pf.CachedRows <= pf.CacheCapacityRows);
        Assert.True(pf.Stats.CacheHits > 0);
    }

    [Fact]
    public void Gather_OutOfRangeRow_Throws_LikeThePlainPath()
    {
        const int rowDim = 160; const long tableRows = 100;
        Directory.CreateDirectory(_dir);
        var qt = QuantizationType.IQ4_NL;
        var table = new MappedTable(Path.Combine(_dir, "oob.bin"), tableRows * Dequantize.RowByteSize(rowDim, qt), seed: 1);
        _cleanup.Add(table);
        foreach (var mode in new[] { PleTableMode.Mmap, PleTableMode.Ram })
        {
            using var pf = new PleRowPrefetcher(table.Ptr, qt, tableRows, rowDim, new PleTableOptions { Mode = mode, CacheMegabytes = 1 });
            long[] rows = [3, tableRows, 7];
            var req = pf.Begin(rows);   // must not prefetch past the mapping (would be an access violation)
            Assert.Throws<ArgumentOutOfRangeException>(() => pf.Gather(req, rows, new float[rows.Length * rowDim]));
        }
    }

    [Fact]
    public void AbandonedRequest_DoesNotBreakTheNextGather_NorDispose()
    {
        const int rowDim = 160; const long tableRows = 10_000;
        Directory.CreateDirectory(_dir);
        var qt = QuantizationType.IQ4_NL;
        var table = new MappedTable(Path.Combine(_dir, "abandon.bin"), tableRows * Dequantize.RowByteSize(rowDim, qt), seed: 2);
        _cleanup.Add(table);
        var pf = new PleRowPrefetcher(table.Ptr, qt, tableRows, rowDim, new PleTableOptions { Mode = PleTableMode.Ram, CacheMegabytes = 1 });
        long[] a = Enumerable.Range(0, 4000).Select(i => (long)i * 2).ToArray();
        long[] b = [1, 2, 3, 9999];
        _ = pf.Begin(a);                                     // never collected
        var exp = new float[b.Length * rowDim]; var got = new float[exp.Length];
        Qwen4ExpPle.GatherRows(table.Ptr, qt, tableRows, rowDim, b, exp);
        pf.Gather(pf.Begin(a), b, got);                      // request for rows other than the ones gathered
        AssertBitEqual(exp, got, "mismatched request");
        pf.Dispose();                                        // waits for the abandoned worker; must not touch freed memory
    }

    // ── model level: byte-identical logits, chunked prefill, EOS reset ─────

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) LoadSynthetic(PleTableOptions opts, string name, Q4eGeometry? geo = null)
    {
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, name + ".gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(geo ?? new Q4eGeometry(), Q4eQuant.Q8Q51));
        var prev = PleTableOptions.Ambient;
        PleTableOptions.Ambient = opts;
        try
        {
            var (m, g, c) = ModelLoader.LoadFromGguf(path);
            _cleanup.Add(m); _cleanup.Add(g);
            return ((Qwen4ExpTransformerModel)m, c);
        }
        finally { PleTableOptions.Ambient = prev; }
    }

    private static int[] Tokens(int n, int seed, int eos)
    {
        var rng = new Random(seed);
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = i % 7 == 3 ? eos : rng.Next(6, 100);   // EOS (5) resets the hash window every few tokens
        return t;
    }

    private static float[] Run(Qwen4ExpTransformerModel m, int[] ids, int[] chunks)
    {
        var st = m.CreateState();
        var all = new List<float>();
        int pos = 0;
        foreach (int c in chunks)
        {
            var part = ids.AsSpan(pos, c).ToArray();
            using var t = m.Forward(part, Enumerable.Range(pos, c).ToArray(), -1, st, null, lastTokenLogitsOnly: false);
            all.AddRange(new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray());
            pos += c;
        }
        return all.ToArray();
    }

    public static IEnumerable<object[]> ModelArms()
    {
        foreach (var mode in new[] { PleTableMode.Mmap, PleTableMode.Ram })
        foreach (var hint in new[] { PleHintMode.Os, PleHintMode.Touch, PleHintMode.OsThenTouch })
            yield return new object[] { mode, hint };
    }

    [Theory]
    [MemberData(nameof(ModelArms))]
    public void Model_Logits_AreByteIdentical_ToTheNonPrefetchPath_ChunkedAndWithEosReset(PleTableMode mode, PleHintMode hint)
    {
        var (control, cfg) = LoadSynthetic(PleTableOptions.Disabled, "ctl");
        Assert.Null(control.PleTableStats);   // the control arm really is the plain path
        var (treated, _) = LoadSynthetic(new PleTableOptions { Mode = mode, Hint = hint, CacheMegabytes = mode == PleTableMode.Ram ? 1 : 0 }, "trt");

        int eos = cfg.Qwen4Exp!.Ple!.EosTokenId;
        var ids = Tokens(23, 4, eos);
        Assert.Contains(eos, ids);
        // identical forward shapes in both arms (a different chunking changes GEMM partitioning, not the table service)
        foreach (var chunks in new[] { new[] { 23 }, new[] { 8, 8, 7 }, new[] { 1, 1, 1, 20 }, new[] { 5, 1, 17 } })
        {
            var a = Run(control, ids, chunks);
            var b = Run(treated, ids, chunks);
            AssertBitEqual(a, b, $"{mode}/{hint} chunks=[{string.Join(',', chunks)}]");
        }
        var s = treated.PleTableStats;
        Assert.NotNull(s);
        Assert.True(s!.Value.Requests >= 4, "prefetch must actually have been issued (a passing equality alone proves nothing)");
        Assert.True(s.Value.Rows > 0 && s.Value.UniquePages > 0);
    }

    [Fact]
    public void Model_PrefetchIssuedForAWholeBatchBeforeLayerOne_AndConsumedExactlyOnce()
    {
        var (m, cfg) = LoadSynthetic(new PleTableOptions { Hint = PleHintMode.Touch }, "once");
        var ids = Tokens(12, 9, cfg.Qwen4Exp!.Ple!.EosTokenId);
        _ = Run(m, ids, [12]);
        var s = m.PleTableStats!.Value;
        Assert.Equal(1, s.Requests);
        Assert.Equal(1, s.Gathers);
        Assert.Equal(12L * 4, s.Rows);   // 2 n-gram orders x 2 heads per order in the synthetic geometry
    }

    // ── soak (llama.cpp #28933 class of RSS growth) ────────────────────────

    /// <summary>
    /// The #28933 bug class is mapped-file pages piling up in the working set beyond what was touched. <c>PrivateMemorySize64</c>
    /// cannot see mapped pages, so this counts the table's own resident pages (<c>QueryWorkingSetEx</c>) against the pages the
    /// requests actually named: residency must track the touched footprint (bounded read-around amplification), and replaying
    /// already-seen chunks must not add any.
    /// </summary>
    [Fact]
    public void Soak_TwentyPrefillChunks_OnAScaledTable_ResidencyTracksTheTouchedFootprint()
    {
        // 4M-row table: wide enough that the hash does not saturate it in 20 chunks.
        var geo = new Q4eGeometry { TableRows = 1 << 22, Vocab = 100 };   // 4M rows x 32 B = 128 MiB = 32768 pages
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "soak.gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(geo, Q4eQuant.Q8Q51));
        var prev = PleTableOptions.Ambient;
        PleTableOptions.Ambient = new PleTableOptions { Mode = PleTableMode.Mmap, Hint = PleHintMode.OsThenTouch };
        var (mdl, gguf, cfg) = ModelLoader.LoadFromGguf(path);
        PleTableOptions.Ambient = prev;
        _cleanup.Add(mdl); _cleanup.Add(gguf);
        var m = (Qwen4ExpTransformerModel)mdl;
        var ple = cfg.Qwen4Exp!.Ple!;
        var tdesc = gguf.Tensors.Single(t => t.Name == "per_layer_token_embd.weight");
        nint tbl = gguf.TensorDataPointer(tdesc);
        long tableBytes = Dequantize.RowByteSize(ple.RowDim, tdesc.QuantizationType) * tdesc.Shape[1];
        int ps = Environment.SystemPageSize;
        long firstPage = (long)tbl / ps, lastPage = ((long)tbl + tableBytes - 1) / ps;
        var pageAddrs = new nint[lastPage - firstPage + 1];
        for (int i = 0; i < pageAddrs.Length; i++) pageAddrs[i] = (nint)((firstPage + i) * ps);
        long Resident() => ProcessFaults.CountWorkingSetResident(pageAddrs);

        int eos = ple.EosTokenId;
        const int chunk = 48, chunks = 20;
        long r0 = Resident();
        var seen = new List<int[]>();
        var trace = new List<string>();
        for (int i = 0; i < chunks; i++)
        {
            var ids = Tokens(chunk, 100 + i, eos);
            seen.Add(ids);
            using var st = m.CreateState();   // fresh sequence per chunk: the synthetic QSA layer prunes beyond its tiny budget
            using (var t = m.Forward(ids, Enumerable.Range(0, chunk).ToArray(), -1, st, null, lastTokenLogitsOnly: true)) { }
            long touched = m.PleTableStats!.Value.UniquePages;
            trace.Add($"chunk {i + 1}: touched(cum) {touched} resident {Resident() - r0}");
        }
        var stats = m.PleTableStats!.Value;
        long resident = Resident() - r0;
        output.WriteLine(string.Join(Environment.NewLine, trace));
        output.WriteLine($"stats: {stats}");
        output.WriteLine($"table pages {pageAddrs.Length}, touched {stats.UniquePages}, resident growth {resident} (amplification {(double)resident / stats.UniquePages:F2}x)");

        // replay: the same chunks again must not make more of the table resident
        for (int i = 0; i < chunks; i++)
        {
            using var st = m.CreateState();
            using var t = m.Forward(seen[i], Enumerable.Range(0, chunk).ToArray(), -1, st, null, lastTokenLogitsOnly: true);
        }
        long replayGrowth = Resident() - r0 - resident;
        output.WriteLine($"replay growth {replayGrowth} pages");
        if (Resident() >= 0)   // Windows only
        {
            Assert.True(resident <= 2 * stats.UniquePages + 512, $"resident {resident} pages vs {stats.UniquePages} touched");
            Assert.True(replayGrowth <= 64, $"replaying seen chunks made {replayGrowth} more table pages resident");
        }
        Assert.True(stats.Requests == chunks && stats.Gathers == chunks);
    }
}
