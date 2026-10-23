using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #842: per-row recurrent snapshots by replay (one pre-chunk state + keys / decays / deltas per row) instead of a full GDN state per row.
/// </summary>
public sealed unsafe class Qwen4ExpRowSnapshotScratchTests(ITestOutputHelper output) : IDisposable
{
    // ── kernel: replay == the scan's own per-row snapshots, bit for bit ──

    [Theory]
    [InlineData(6, 2, 8, 7)]     // tiled key-head broadcast (vh % nK), 8-wide
    [InlineData(4, 4, 12, 5)]    // 12 is not a multiple of the 8-lane vector: scalar tail
    [InlineData(6, 3, 16, 9)]
    public void Replay_ReproducesEveryScanSnapshot_BitIdentically(int nV, int nK, int dS, int T)
    {
        var rng = new Random(nV * 100 + dS);
        float[] R(int n, float lo, float hi) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = lo + (float)rng.NextDouble() * (hi - lo); return a; }
        var s0 = R(nV * dS * dS, -1, 1);
        var q = R(T * nK * dS, -1, 1); var k = R(T * nK * dS, -1, 1); var v = R(T * nV * dS, -1, 1);
        GatedDeltaNetScan.L2NormalizeHeads(k, dS);
        var g = R(T * nV, 0.6f, 1.0f); var beta = R(T * nV, 0.1f, 0.9f);
        int snapRows = T - 1, stateLen = nV * dS * dS;

        var state = (float[])s0.Clone();
        var snaps = new float[snapRows * stateLen];
        var deltas = new float[snapRows * nV * dS];
        GatedDeltaNetScan.Execute(state, q, k, v, g, beta, new float[T * nV * dS], nV, nK, dS, T, rowSnapshots: snaps, snapshotRows: snapRows, deltaRecord: deltas);

        for (int row = 0; row < snapRows; row++)
        {
            var rebuilt = R(stateLen, 5, 6);                         // garbage: replay must fully overwrite it from the source
            GatedDeltaNetScan.Replay(rebuilt, s0, k, g, deltas, nV, nK, dS, row + 1);
            Assert.Equal(snaps.AsSpan(row * stateLen, stateLen).ToArray(), rebuilt);

            var inPlace = (float[])s0.Clone();                        // same thing in place (source empty)
            GatedDeltaNetScan.Replay(inPlace, default, k, g, deltas, nV, nK, dS, row + 1);
            Assert.Equal(rebuilt, inPlace);
        }
        // Control: ignoring the deltas (replaying decay only) is visibly different, so the comparison above has teeth.
        var decayOnly = (float[])s0.Clone();
        GatedDeltaNetScan.Replay(decayOnly, default, k, g, new float[deltas.Length], nV, nK, dS, 2);
        Assert.NotEqual(snaps.AsSpan(stateLen, stateLen).ToArray(), decayOnly);
    }

    [Fact]
    public void Scan_RecordingDeltas_DoesNotChangeTheScanResult()
    {
        const int nV = 4, nK = 2, dS = 8, T = 6;
        var rng = new Random(3);
        float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)rng.NextDouble(); return a; }
        var s = R(nV * dS * dS); var q = R(T * nK * dS); var k = R(T * nK * dS); var v = R(T * nV * dS); var g = R(T * nV); var b = R(T * nV);
        var s1 = (float[])s.Clone(); var o1 = new float[T * nV * dS];
        GatedDeltaNetScan.Execute(s1, q, k, v, g, b, o1, nV, nK, dS, T);
        var s2 = (float[])s.Clone(); var o2 = new float[T * nV * dS];
        GatedDeltaNetScan.Execute(s2, q, k, v, g, b, o2, nV, nK, dS, T, snapshotRows: T - 1, deltaRecord: new float[(T - 1) * nV * dS]);
        Assert.Equal(s1, s2); Assert.Equal(o1, o2);
    }

    // ── model ──

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-rowscratch-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _d = [];

    public void Dispose()
    {
        foreach (var x in _d) x.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private (Qwen4ExpTransformerModel, ModelConfig) Load()
    {
        Directory.CreateDirectory(_dir);
        var (m, g, c) = ModelLoader.LoadFromGguf(SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "syn.gguf")));
        _d.Add(g); _d.Add(m);
        return ((Qwen4ExpTransformerModel)m, c);
    }

    private static int[] Toks(int n, int seed) { var r = new Random(seed); var t = new int[n]; for (int i = 0; i < n; i++) t[i] = r.Next(4, SyntheticQwen4ExpGguf.VocabSize); return t; }
    private static int[] Pos(int n, int start) => Enumerable.Range(start, n).ToArray();

    private static float[] Fwd(Qwen4ExpTransformerModel m, int[] ids, int start)
    {
        using var t = m.Forward(ids, Pos(ids.Length, start), -1, null);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static float MaxRel(float[] a, float[] b)
    {
        float worst = 0, scale = 1e-12f;
        foreach (float v in a) scale = MathF.Max(scale, MathF.Abs(v));
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst / scale;
    }

    [Fact]
    public void Scratch_IsOneStatePlusSmallPerRowTerms_NotAStatePerRow()
    {
        var (m, cfg) = Load();
        var prefix = Toks(9, 1);
        long BytesFor(int window)
        {
            m.ResetSequenceState();
            Fwd(m, prefix, 0);
            using var _ = m.ForwardWithRecurrentSnapshots(Toks(window, 2), Pos(window, prefix.Length), -1, null, null);
            return m.RecurrentRowSnapshotGdnBytes;
        }
        long b3 = BytesFor(3), b7 = BytesFor(7);
        var gdn = cfg.GdnConfig!.Value;
        int numGdn = cfg.HybridLayout!.LayerKind.Count(k => k == HybridLayerKind.GatedDeltaNet);
        long state = (long)numGdn * gdn.StateElements * 4;                 // one full GDN state
        long conv = (long)numGdn * gdn.ConvStateElements * 4;              // one conv window set (a row's conv snapshot)
        long kgd = (long)numGdn * (gdn.NKHead * gdn.DState + gdn.NVHead + gdn.NVHead * gdn.DState) * 4;   // keys + decays + deltas of one row
        long legacyPerRow = state + conv;
        output.WriteLine($"synthetic: scratch window 3 -> {b3} B, window 7 -> {b7} B; per extra row now {kgd + conv} B vs legacy {legacyPerRow} B; base state {state} B");

        Assert.Equal(state + conv + 2 * (kgd + conv), b3);    // base = the swapped GdnStateCache (matrix + conv window)                          // one pre-chunk state + 2 recorded rows
        Assert.Equal(state + conv + 6 * (kgd + conv), b7);
        Assert.True(kgd + conv < legacyPerRow);                              // a row costs less than a full state snapshot
        Assert.True(b7 > b3);                                                // control: the per-row terms exist and grow
    }

    [Fact]
    public void ConsecutiveWindows_AndAnEntryStateThatIsStillLazy_RestoreCorrectly()
    {
        var (m, _) = Load();
        var prefix = Toks(10, 5); var w1 = Toks(6, 6); var w2 = Toks(5, 7); var detour = Toks(4, 8); var next = Toks(3, 9);
        int row1 = 2, row2 = 3;

        // Reference: the committed prefixes forwarded on their own (different shapes: ULP drift only).
        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        Fwd(m, w1.AsSpan(0, row1 + 1).ToArray(), prefix.Length);
        Fwd(m, w2.AsSpan(0, row2 + 1).ToArray(), prefix.Length + row1 + 1);
        var want = Fwd(m, next, prefix.Length + row1 + 1 + row2 + 1);

        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        var cp = m.CheckpointRecurrentState();
        Fwd(m, detour, prefix.Length);
        m.RestoreRecurrentState(cp);                                   // live state is now lazily reading the checkpoint
        (cp as IDisposable)?.Dispose();                                // (exchanged / materialised on release)
        using (m.ForwardWithRecurrentSnapshots(w1, Pos(w1.Length, prefix.Length), -1, null, null)) { }
        m.RestoreRecurrentStateToRow(row1);
        using (m.ForwardWithRecurrentSnapshots(w2, Pos(w2.Length, prefix.Length + row1 + 1), -1, null, null)) { }
        m.RestoreRecurrentStateToRow(row2);
        var got = Fwd(m, next, prefix.Length + row1 + 1 + row2 + 1);

        Assert.True(MaxRel(want, got) < 2e-5f, $"rel {MaxRel(want, got):E3}");
        // control: not restoring leaves the state far from the reference
        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        using (m.ForwardWithRecurrentSnapshots(w1, Pos(w1.Length, prefix.Length), -1, null, null)) { }
        Assert.True(MaxRel(want, Fwd(m, next, prefix.Length + w1.Length)) > 1e-4f);
    }

    // ── real-size cost (opt-in) ──

    [Fact]
    public void RealSize_ScratchBytesAndRestoreCost()
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_BENCH_842") != "1") return;
        const int layers = 36, nV = 48, nK = 16, dS = 128, dConv = 4;
        int convDim = (2 * nK + nV) * dS;
        long stateFloats = (long)nV * dS * dS, convFloats = (long)(dConv - 1) * convDim;
        double MiB(long floats) => floats * 4 / 1048576.0;
        foreach (int rows in new[] { 1, 3, 4, 7 })
        {
            long legacy = layers * (stateFloats + convFloats) * rows;
            long perRow = layers * ((long)nK * dS + nV + (long)nV * dS + convFloats);
            long now = layers * stateFloats + perRow * rows;
            output.WriteLine($"rows={rows}: scratch legacy {MiB(legacy):F1} MiB -> now {MiB(now):F1} MiB ({(double)legacy / now:F2}x)");
        }

        var rng = new Random(1);
        float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)rng.NextDouble() - 0.5f; return a; }
        int T = 5;
        var s0 = R(nV * dS * dS); var q = R(T * nK * dS); var k = R(T * nK * dS); var v = R(T * nV * dS);
        GatedDeltaNetScan.L2NormalizeHeads(k, dS);
        var g = Enumerable.Repeat(0.97f, T * nV).ToArray(); var beta = Enumerable.Repeat(0.4f, T * nV).ToArray();
        var snaps = new float[(T - 1) * nV * dS * dS]; var deltas = new float[(T - 1) * nV * dS];
        var state = (float[])s0.Clone(); var o = new float[T * nV * dS];

        double Time(Action a, int reps = 9)
        {
            var l = new List<double>();
            for (int i = 0; i < reps; i++) { long t0 = Stopwatch.GetTimestamp(); a(); l.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
            l.Sort(); return l[reps / 2];
        }
        // one layer; x36 for the model
        double scanSnapFull = 0, scanSnapDelta = 0;
        for (int round = 0; round < 3; round++)
        {
            scanSnapFull = Time(() => { s0.CopyTo(state, 0); GatedDeltaNetScan.Execute(state, q, k, v, g, beta, o, nV, nK, dS, T, rowSnapshots: snaps, snapshotRows: T - 1); });
            scanSnapDelta = Time(() => { s0.CopyTo(state, 0); GatedDeltaNetScan.Execute(state, q, k, v, g, beta, o, nV, nK, dS, T, snapshotRows: T - 1, deltaRecord: deltas); });
        }
        output.WriteLine($"per layer, T={T} verify scan: full-state snapshots {scanSnapFull:F2} ms | delta record {scanSnapDelta:F2} ms  (x{layers}: {scanSnapFull * layers:F0} vs {scanSnapDelta * layers:F0} ms)");

        var rebuilt = new float[nV * dS * dS];
        foreach (int row in new[] { 0, 1, 3 })
        {
            double replay = Time(() => GatedDeltaNetScan.Replay(rebuilt, s0, k, g, deltas, nV, nK, dS, row + 1));
            double copy = Time(() => snaps.AsSpan(row * nV * dS * dS, nV * dS * dS).CopyTo(rebuilt));
            output.WriteLine($"restore to row {row}: legacy copy {copy:F2} ms/layer ({copy * layers:F1} ms/model) | replay {replay:F2} ms/layer ({replay * layers:F1} ms/model)");
        }
    }
}
