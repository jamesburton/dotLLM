using System.Diagnostics;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TurboQuantHistoryProbeCollection
{
    public const string Name = "TurboQuantHistoryProbe";
}

/// <summary>
/// Guards CLAUDE.md rule 2 for the CPU in-place TurboQuant path: the history is never decoded whole (#1892).
/// Runs alone (non-parallel collection): the probes read process-wide counters, so a neighbouring test must not move them.
/// </summary>
[Collection(TurboQuantHistoryProbeCollection.Name)]
public sealed unsafe class TurboQuantInPlaceHistoryProbeTests
{
    private const int Hd = 128, NumKvHeads = 2, NumHeads = 4, Bits = 4;

    private static (TurboQuantInPlaceKvCache Cache, float[] Q) Build(int rows)
    {
        var cache = new TurboQuantInPlaceKvCache(1, NumKvHeads, Hd, rows, Bits, TurboQuantInPlaceAttentionTests.Seed);
        var rng = new Random(5);
        int kvStride = NumKvHeads * Hd;
        TurboQuantInPlaceAttentionTests.Fill(cache, TurboQuantInPlaceAttentionTests.Gauss(rng, rows * kvStride, 1f),
                                             TurboQuantInPlaceAttentionTests.Gauss(rng, rows * kvStride, 1f), 0, kvStride);
        return (cache, TurboQuantInPlaceAttentionTests.Gauss(rng, NumHeads * Hd, 1f));
    }

    private static void Attend(TurboQuantInPlaceKvCache cache, float[] q, float[] o, int rows)
    {
        fixed (float* qp = q)
        fixed (float* op = o)
            Attention.ExecuteTurboQuant(qp, cache, 0, op, 1, rows, NumHeads, NumKvHeads, Hd, rows - 1, null);
    }

    // Primary, deterministic gate: the row scratch the kernel decodes into is one head vector, whatever the history length.
    [Fact]
    public void RowScratch_IsOneHeadVector_AndDoesNotScaleWithHistory()
    {
        var observed = new List<int>();
        foreach (int rows in new[] { 16, 4096 })
        {
            var (cache, q) = Build(rows);
            using (cache)
            {
                Attention.RowScratchFloatsObserved = -1;
                Attend(cache, q, new float[NumHeads * Hd], rows);
                observed.Add(Attention.RowScratchFloatsObserved);
            }
        }
        Assert.Equal(new[] { Hd, Hd }, observed);
    }

    // Backstop for allocations the hook cannot see (a separate decode buffer anywhere, any thread, managed or native).
    // No retry: the collection is non-parallel, so nothing else moves these counters, and the sampler sleeps between reads.
    [Fact]
    public void HistoryIsNeverDecodedWhole()
    {
        const int rows = 16384, reps = 20;
        var (cache, q) = Build(rows);
        using var _ = cache;
        var o = new float[NumHeads * Hd];
        long oneSideDecoded = (long)rows * NumKvHeads * Hd * 4;   // a decoded K (or V) history in F32
        Attend(cache, q, o, rows);                                // warm up the JIT

        // Managed bytes, all threads (the collection is isolated, so nothing else allocates).
        GC.Collect();
        long before = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < reps; i++) Attend(cache, q, o, rows);
        long managedPerCall = (GC.GetTotalAllocatedBytes(precise: true) - before) / reps;
        Assert.True(managedPerCall < oneSideDecoded / 8, $"allocated {managedPerCall} managed bytes per call (decoded side = {oneSideDecoded})");

        // Peak private (committed) memory, sampled with a sleep so the sampler does not load the CPU the workers use.
        GC.Collect();
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long baseline = proc.PrivateMemorySize64, peak = baseline;
        bool stop = false;
        var sampler = new Thread(() =>
        {
            using var p = Process.GetCurrentProcess();
            while (!Volatile.Read(ref stop)) { p.Refresh(); long x = p.PrivateMemorySize64; if (x > peak) peak = x; Thread.Sleep(1); }
        }) { IsBackground = true };
        sampler.Start();
        for (int i = 0; i < reps; i++) Attend(cache, q, o, rows);
        Volatile.Write(ref stop, true);
        sampler.Join();
        Assert.True(peak - baseline < oneSideDecoded / 2, $"peak private memory grew by {peak - baseline} bytes (decoded side = {oneSideDecoded})");

        Assert.Throws<NotSupportedException>(() => { cache.GetKeysRef(0); });
        Assert.Equal(cache.Layout.CodeBytesPerLayer(rows) * 2 + cache.Layout.NormBytesPerLayer(rows) * 2, cache.AllocatedBytes);
    }
}
