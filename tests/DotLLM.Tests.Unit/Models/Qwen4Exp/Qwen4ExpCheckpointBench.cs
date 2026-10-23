using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #840 microbench at the released model's state dims (36 GDN layers of 48 x 128 x 128 + conv = ~113 MiB, 12 QSA indexers of
/// 128-wide pooled keys). Opt-in: <c>DOTLLM_BENCH_840=1</c>. Prints ms per checkpoint / restore, old full-copy path vs the new path.
/// </summary>
public sealed unsafe class Qwen4ExpCheckpointBench(ITestOutputHelper output)
{
    private const int GdnLayers = 36, QsaLayers = 12, Layers = GdnLayers + QsaLayers, IdxD = 128, Block = 4;

    private static Qwen4ExpSequenceState Build(int ctx, int seed)
    {
        var gdn = new GatedDeltaNetConfig(FullAttnInterval: 4, NVHead: 48, NKHead: 16, DState: 128, DInner: 6144, DConv: 4);
        var cache = new GdnStateCache(gdn, GdnLayers);
        var rng = new Random(seed);
        for (int l = 0; l < GdnLayers; l++)
        {
            foreach (ref float f in cache.GetGdnState(l)) f = (float)rng.NextDouble();
            foreach (ref float f in cache.GetConvState(l)) f = (float)rng.NextDouble();
        }
        var ple = new Qwen4ExpPleState?[Layers];
        var qsa = new Qwen4ExpQsaState?[Layers];
        for (int i = 0; i < QsaLayers; i++) qsa[i] = new Qwen4ExpQsaState(1024, IdxD, Block);
        var st = new Qwen4ExpSequenceState(cache, ple, qsa);
        var gamma = Enumerable.Repeat(1f, IdxD).ToArray();
        var raw = new float[Math.Min(ctx, 4096) * IdxD];
        foreach (var q in qsa.Where(x => x is not null))
            for (int done = 0; done < ctx; done += 4096)
            {
                for (int i = 0; i < raw.Length; i++) raw[i] = (float)rng.NextDouble();
                int n = Math.Min(4096, ctx - done);
                q!.Indexer.Append(raw, n, gamma, 1e-6f, default, default, 0);
            }
        st.Length = ctx;
        return st;
    }

    private static void RoundWork(Qwen4ExpSequenceState live)
    {
        // What a verify forward does to the state: consume the deferred GDN copy, append a few tokens to every indexer.
        for (int l = 0; l < GdnLayers; l++) live.Gdn.BeginUpdate(l, out _, out _);
        var raw = new float[5 * IdxD]; var gamma = Enumerable.Repeat(1f, IdxD).ToArray();
        foreach (var q in live.Qsa) q?.Indexer.Append(raw, 5, gamma, 1e-6f, default, default, 0);
        live.Length += 5;
    }

    private static double Median(List<double> v) { v.Sort(); return v[v.Count / 2]; }

    [Theory]
    [InlineData(2048)]
    [InlineData(32768)]
    [InlineData(131072)]
    public void CheckpointRestoreCost_OldFullCopy_Vs_Delta(int ctx)
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_BENCH_840") != "1") return;
        using var live = Build(ctx, 1);
        using var shellOld = Build(0, 2);
        using var shellNew = Build(0, 3);
        var scratch = new float[live.Qsa[0]!.Indexer.Pooled.Length + 4096];
        var oldCp = new List<double>(); var oldRs = new List<double>(); var newCp = new List<double>(); var newRs = new List<double>();
        for (int rep = 0; rep < 9; rep++)
        {
            long t = Stopwatch.GetTimestamp();
            shellOld.CopyFrom(live);
            double a = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            t = Stopwatch.GetTimestamp();
            live.CopyFrom(shellOld);
            double b = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            // The pre-#840 CopyFrom also memcpy'd every pooled block on each call; the CopyFrom above is already stamp-delta for that part.
            t = Stopwatch.GetTimestamp();
            foreach (var qs in live.Qsa) if (qs is not null) { qs.Indexer.Pooled.CopyTo(scratch.AsSpan(0, qs.Indexer.Pooled.Length)); }
            double legacyPooled = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            a += legacyPooled; b += legacyPooled;

            t = Stopwatch.GetTimestamp();
            live.CaptureInto(shellNew);
            double c = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            RoundWork(live);                                      // (consumes the deferred copy, as the forward would)
            t = Stopwatch.GetTimestamp();
            live.RestoreFrom(shellNew);
            live.ReleaseSource(shellNew);                         // restore-then-dispose
            double d = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            RoundWork(live);                                      // replay forward
            if (rep > 0) { oldCp.Add(a); oldRs.Add(b); newCp.Add(c); newRs.Add(d); }
        }
        output.WriteLine($"ctx={ctx,7}: checkpoint old {Median(oldCp):F3} ms -> new {Median(newCp):F3} ms | restore old {Median(oldRs):F3} ms -> new {Median(newRs):F3} ms | " +
                         $"state {live.Bytes / 1048576.0:F1} MiB (GDN {live.Gdn.AllocatedBytes / 1048576.0:F1})");
    }

    [Fact]
    public void FusedScan_CostsTheSameAsInPlace_RealDims()
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_BENCH_840") != "1") return;
        const int nV = 48, nK = 16, dS = 128, T = 5;
        var rng = new Random(4);
        float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var src = R(nV * dS * dS); var state = R(nV * dS * dS);
        var q = R(T * nK * dS); var k = R(T * nK * dS); var v = R(T * nV * dS);
        var g = Enumerable.Repeat(0.95f, T * nV).ToArray(); var beta = Enumerable.Repeat(0.5f, T * nV).ToArray();
        var o = new float[T * nV * dS];
        var plain = new List<double>(); var fused = new List<double>(); var copyThenPlain = new List<double>();
        for (int rep = 0; rep < 12; rep++)
        {
            double a = 0, b = 0, c = 0;
            for (int arm = 0; arm < 3; arm++)
            {
                int which = (arm + rep) % 3;                          // rotate the arm order so no arm is always first/last
                src.CopyTo(state);                                    // (untimed) every arm starts from the same, non-denormal state
                long t = Stopwatch.GetTimestamp();
                if (which == 0) GatedDeltaNetScan.Execute(state, q, k, v, g, beta, o, nV, nK, dS, T);
                else if (which == 1) GatedDeltaNetScan.Execute(state, q, k, v, g, beta, o, nV, nK, dS, T, stateSource: src);
                else { src.CopyTo(state); GatedDeltaNetScan.Execute(state, q, k, v, g, beta, o, nV, nK, dS, T); }
                double ms = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                if (which == 0) a = ms; else if (which == 1) b = ms; else c = ms;
            }
            if (rep > 0) { plain.Add(a); fused.Add(b); copyThenPlain.Add(c); }
        }
        output.WriteLine($"GDN layer scan T={T}: in-place {Median(plain):F3} ms | fused-from-source {Median(fused):F3} ms | copy+in-place {Median(copyThenPlain):F3} ms");
    }
}
