using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #841: Q8_0 / Q4_0 engine KV cache for the QSA layers. Fidelity numbers are printed; each bound has a control arm that could fail it
/// (Q4_0 must be measurably worse than Q8_0, and quantisation must be engaged, i.e. not bit-identical to fp32 once rows are read back).
/// </summary>
public sealed unsafe class Qwen4ExpQuantizedKvTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-qkv-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _d = [];

    public void Dispose()
    {
        foreach (var x in _d) x.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) Load()
    {
        Directory.CreateDirectory(_dir);
        // 2 KV heads: a 32-wide KV row, the minimum a quantised cache accepts (the default fixture's 16-wide row cannot be quantised).
        string path = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "syn.gguf"), numKvHeads: 2);
        var (m, g, c) = ModelLoader.LoadFromGguf(path);
        _d.Add(g); _d.Add(m);
        return ((Qwen4ExpTransformerModel)m, c);
    }

    private static int[] Tokens(int n, int seed)
    {
        var r = new Random(seed); var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = r.Next(4, SyntheticQwen4ExpGguf.VocabSize);
        return t;
    }

    private static float[] Logits(Qwen4ExpTransformerModel m, Qwen4ExpSequenceState st, int[] ids, int start, IKvCache kv)
    {
        using var t = m.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, st, kv, lastTokenLogitsOnly: false);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static float[] DefaultLogits(Qwen4ExpTransformerModel m, int[] ids, int start, IKvCache kv)
    {
        using var t = m.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, kv);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    /// <summary>Chunked prefill then token-by-token decode: returns every row's logits.</summary>
    private static float[] Run(Qwen4ExpTransformerModel m, int[] ids, int chunk, IKvCache kv)
    {
        using var st = m.CreateState();
        var all = new List<float>();
        int pos = 0;
        while (pos < ids.Length)
        {
            int n = Math.Min(pos < ids.Length / 2 ? chunk : 1, ids.Length - pos);   // second half is decode
            all.AddRange(Logits(m, st, ids.AsSpan(pos, n).ToArray(), pos, kv));
            pos += n;
        }
        return all.ToArray();
    }

    /// <summary>(mean KL(float||q) over rows after sharpening, top-1 agreement). Logits are scaled up so the tiny model's near-flat
    /// distributions are informative.</summary>
    private static (double Kl, double Top1) Compare(float[] reference, float[] test, int vocab, int fromRow, float sharpen = 40f)
    {
        int rows = reference.Length / vocab, agree = 0;
        double kl = 0;
        for (int r = fromRow; r < rows; r++)
        {
            var p = Softmax(reference.AsSpan(r * vocab, vocab), sharpen);
            var q = Softmax(test.AsSpan(r * vocab, vocab), sharpen);
            for (int i = 0; i < vocab; i++) if (p[i] > 0) kl += p[i] * Math.Log(p[i] / Math.Max(q[i], 1e-30));
            if (ArgMax(reference.AsSpan(r * vocab, vocab)) == ArgMax(test.AsSpan(r * vocab, vocab))) agree++;
        }
        int n = rows - fromRow;
        return (kl / n, (double)agree / n);
    }

    private static double[] Softmax(ReadOnlySpan<float> x, float s)
    {
        double max = double.NegativeInfinity;
        foreach (float v in x) max = Math.Max(max, v * s);
        var e = new double[x.Length]; double sum = 0;
        for (int i = 0; i < x.Length; i++) { e[i] = Math.Exp(x[i] * s - max); sum += e[i]; }
        for (int i = 0; i < e.Length; i++) e[i] /= sum;
        return e;
    }

    private static int ArgMax(ReadOnlySpan<float> x) { int b = 0; for (int i = 1; i < x.Length; i++) if (x[i] > x[b]) b = i; return b; }

    [Theory]
    [InlineData(0)]
    [InlineData(8)]
    public void Synthetic_Q8_0_StaysCloseToFloat_AndQ4_0IsMeasurablyWorse(int window)
    {
        var (m, cfg) = Load();
        var ids = Tokens(62, seed: 3);                       // beyond the 11-token QSA budget: the sparse path is exercised
        var geom = KvGeometry.FromConfig(cfg);

        using var kvF = new SimpleKvCache(geom, 64);
        var reference = Run(m, ids, chunk: 7, kvF);
        using var kv8 = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);
        var q8 = Run(m, ids, 7, kv8);
        using var kv4 = new QuantizedKvCache(geom, 64, KvCacheDType.Q4_0, KvCacheDType.Q4_0, window);
        var q4 = Run(m, ids, 7, kv4);

        int v = cfg.VocabSize, from = 12;
        var (kl8, top8) = Compare(reference, q8, v, from);
        var (kl4, top4) = Compare(reference, q4, v, from);
        output.WriteLine($"window={window}: Q8_0 KL={kl8:E3} top1={top8:P1} | Q4_0 KL={kl4:E3} top1={top4:P1}");

        Assert.NotEqual(reference, q8);                      // quantisation is really engaged (rows are read back dequantised)
        Assert.True(kl8 < 5e-3, $"Q8_0 KL {kl8}");
        Assert.True(top8 >= 0.95, $"Q8_0 top-1 {top8}");
        Assert.True(kl4 > kl8 * 3, $"control: Q4_0 ({kl4}) must be clearly worse than Q8_0 ({kl8})");
    }

    [Fact]
    public void SingleShotPrefill_IsBitIdenticalToFloat_SinceTheChunkIsAttendedAtFullPrecision()
    {
        var (m, cfg) = Load();
        var ids = Tokens(40, seed: 4);
        var geom = KvGeometry.FromConfig(cfg);
        using var stF = m.CreateState(); using var stQ = m.CreateState();
        using var kvF = new SimpleKvCache(geom, 64);
        var f = Logits(m, stF, ids, 0, kvF);
        using var kvQ = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        var q = Logits(m, stQ, ids, 0, kvQ);
        Assert.Equal(f, q);
        // ... and the very next decode step reads the quantised rows, so it must NOT be bit-identical.
        var next = new[] { 5 };
        var f2 = Logits(m, stF, next, ids.Length, kvF);
        var q2 = Logits(m, stQ, next, ids.Length, kvQ);
        Assert.NotEqual(f2, q2);
    }

    [Fact]
    public void WindowedCache_LongChunkIsSplit_AndRowsReadBackFromTheWindowAreExact()
    {
        // A chunk longer than the fp32 window must not quantise ring slots that were never written.
        var (m, cfg) = Load();
        var ids = Tokens(50, seed: 5);
        var geom = KvGeometry.FromConfig(cfg);
        using var stF = m.CreateState(); using var stQ = m.CreateState();
        using var kvF = new SimpleKvCache(geom, 64);
        var f = Logits(m, stF, ids.AsSpan(0, 30).ToArray(), 0, kvF);
        using var kvQ = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 6);   // 30-row chunk > 6-row window
        var q = Logits(m, stQ, ids.AsSpan(0, 30).ToArray(), 0, kvQ);
        Assert.Equal(f, q);
        Assert.Equal(24, kvQ.QuantizedLength);
        // Decode on: the rows quantised by the split update must be the real ones (error stays at Q8_0 level, not garbage).
        var f2 = Logits(m, stF, ids.AsSpan(30, 10).ToArray(), 30, kvF);
        var q2 = Logits(m, stQ, ids.AsSpan(30, 10).ToArray(), 30, kvQ);
        var (kl, top1) = Compare(f2, q2, cfg.VocabSize, 0);
        output.WriteLine($"window=6 long-chunk: KL={kl:E3} top1={top1:P1}");
        Assert.True(kl < 5e-3 && top1 >= 0.9);
    }

    [Fact]
    public void RollbackReplay_OnAQuantisedCache_IsDeterministic()
    {
        var (m, cfg) = Load();
        var ids = Tokens(30, seed: 6); var wrong = Tokens(5, seed: 7); var right = Tokens(5, seed: 8);
        var geom = KvGeometry.FromConfig(cfg);
        float[] Go(bool detour)
        {
            m.ResetSequenceState();
            using var kv = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
            DefaultLogits(m, ids, 0, kv);
            object? cp = null;
            if (detour)
            {
                cp = m.CheckpointRecurrentState();
                DefaultLogits(m, wrong, ids.Length, kv);
                kv.Rollback(ids.Length);
                m.RestoreRecurrentState(cp);
                (cp as IDisposable)?.Dispose();
            }
            return DefaultLogits(m, right, ids.Length, kv);
        }
        Assert.Equal(Go(false), Go(true));
    }

    [Fact]
    public void PrefixSnapshot_FromAQuantisedCache_RestoresIntoAFreshOne()
    {
        var (m, cfg) = Load();
        var prefix = Tokens(20, seed: 9); var next = Tokens(4, seed: 10);
        var geom = KvGeometry.FromConfig(cfg);
        using var kv = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        using var st = m.CreateState();
        using (var t = m.Forward(prefix, Enumerable.Range(0, prefix.Length).ToArray(), -1, st, kv, lastTokenLogitsOnly: false)) { }
        float[] Cont(QuantizedKvCache k, Qwen4ExpSequenceState s)
        {
            using var t = m.Forward(next, Enumerable.Range(prefix.Length, next.Length).ToArray(), -1, s, k, lastTokenLogitsOnly: false);
            return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
        }
        using var snap = m.SnapshotSequencePrefix(kv, st, prefix.Length)!;
        var want = Cont(kv, st);

        using var kv2 = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        using var st2 = m.CreateState();
        m.RestoreSequencePrefix(snap, kv2, st2);
        // Dequantise-then-requantise is idempotent for Q8_0 rows, so the restored cache serves the same rows.
        Assert.Equal(want, Cont(kv2, st2));
    }

    [Fact]
    public void MixedFp32AndQuantisedSides_AreRejectedClearly()
    {
        var (m, cfg) = Load();
        var geom = KvGeometry.FromConfig(cfg);
        using var kv = new QuantizedKvCache(geom, 64, KvCacheDType.Q8_0, KvCacheDType.F32, 4);
        m.ResetSequenceState();
        var ex = Assert.Throws<NotSupportedException>(() => DefaultLogits(m, Tokens(5, 1), 0, kv));
        Assert.Contains("both K and V", ex.Message);
    }

    [Fact]
    public void Accounting_QuantisedEstimate_MatchesTheCacheAllocation_AndIsSmaller()
    {
        var (m, cfg) = Load();
        int ctx = 64;
        var geom = KvGeometry.FromConfig(cfg);
        using var kvQ = new QuantizedKvCache(geom, ctx, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        using var kvQw = new QuantizedKvCache(geom, ctx, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 8);
        var f32 = Qwen4ExpStateBytes.Estimate(cfg, ctx);
        var q8 = Qwen4ExpStateBytes.Estimate(cfg, ctx, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        var q8w = Qwen4ExpStateBytes.Estimate(cfg, ctx, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 8);
        // Since #839 the hybrid-aware geometry gives the cache one slot per QSA layer only, so the allocation equals the estimate.
        Assert.Equal(kvQ.AllocatedBytes, q8.Kv);
        Assert.Equal(kvQw.AllocatedBytes, q8w.Kv);
        Assert.True(q8.Kv * 3 < f32.Kv, $"int8 should be ~3.8x smaller: {q8.Kv} vs {f32.Kv}");
        Assert.Equal(f32.Total - f32.Kv, q8.Total - q8.Kv);   // nothing else changes
    }


    /// <summary>
    /// Decision record for the pooled indexer keys: they stay fp32. Measured here: the top-k block set the indexer picks when the pooled
    /// keys are round-tripped through Q8_0 vs fp32, at the real geometry (128-wide keys, 512-block budget, 3000 blocks = 12K tokens).
    /// Random keys are the harshest case (many near-ties at the cut), so this under-states real-data agreement.
    /// </summary>
    [Fact]
    public void PooledIndexerKeys_Q8_0_RoundTrip_SelectionOverlap_IsReported()
    {
        const int idxD = 128, idxH = 4, nb = 3000, budget = 512, queries = 40;
        var rng = new Random(21);
        var pooled = new float[nb * idxD];
        for (int i = 0; i < pooled.Length; i++) pooled[i] = (float)(rng.NextDouble() * 2 - 1);
        var rt = new float[pooled.Length];
        var qbuf = new byte[KvQuantize.QuantizedRowBytes(idxD, KvCacheDType.Q8_0)];
        fixed (byte* qb = qbuf)
            for (int b = 0; b < nb; b++)
                fixed (float* src = &pooled[b * idxD]) fixed (float* dst = &rt[b * idxD])
                {
                    KvQuantize.F32ToQ8_0(src, qb, idxD);
                    KvQuantize.Q8_0ToF32(qb, dst, idxD);
                }
        double overlap = 0;
        var selA = new int[budget]; var selB = new int[budget]; var scores = new float[nb + 1];
        for (int qn = 0; qn < queries; qn++)
        {
            var iq = new float[idxH * idxD];
            for (int i = 0; i < iq.Length; i++) iq[i] = (float)(rng.NextDouble() * 2 - 1);
            int ta = Qwen4ExpQsa.SelectBlocks(iq, idxH, idxD, pooled, nb, budget, selA, scores);
            int tb = Qwen4ExpQsa.SelectBlocks(iq, idxH, idxD, rt, nb, budget, selB, scores);
            overlap += selA.Take(ta).Intersect(selB.Take(tb)).Count() / (double)ta;
        }
        overlap /= queries;
        output.WriteLine($"pooled indexer keys Q8_0 round trip: mean top-{budget} block-set overlap {overlap:P2} over {queries} random queries " +
                         $"(pooled bytes/token/layer: fp32 {idxD * 4 / 4}, Q8_0 {qbuf.Length / 4}; int8 K+V row/token/layer at the real 512-wide row: {2 * KvQuantize.QuantizedRowBytes(512, KvCacheDType.Q8_0)})");
        Assert.True(overlap > 0.9, $"overlap {overlap}");   // sanity only: the decision (keep fp32) rests on bytes (see PR), not this bound
    }

    // ── beyond the real 2048-token budget, standalone QSA layer ──

    private static Qwen4ExpProjection RandProj(int outDim, int inDim, int seed, float scale)
    {
        var rng = new Random(seed);
        var w = new float[outDim * inDim];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return (input, output, tokens) =>
        {
            fixed (float* wp = w) fixed (float* ip = input) fixed (float* op = output)
                MatMul.GemmF32(wp, ip, op, outDim, inDim, tokens);
        };
    }

    [Fact]
    public void Layer_BeyondRealBudget_Q8_0_TracksFloat_AndSparseSelectionIsUnchanged()
    {
        const int hidden = 64, nH = 2, nKv = 2, d = 16, ropeDim = 8, idxH = 2, idxD = 16, block = 4, budget = 2048, T = 2400, chunk = 256;
        float[] Ones(int n) => Enumerable.Repeat(1f, n).ToArray();
        var cos = new float[T * ropeDim / 2]; var sin = new float[cos.Length];
        RoPE.PrecomputeFrequencyTable(T, ropeDim, 10000f, cos, sin);
        Qwen4ExpQsaLayer Build() => new(hidden, nH, nKv, d, ropeDim, idxH, idxD, block, budget, 1e-6f,
            Ones(d), Ones(d), Ones(idxD), Ones(idxD), cos, sin,
            RandProj(2 * nH * d, hidden, 1, 0.3f), RandProj(nKv * d, hidden, 2, 0.3f), RandProj(nKv * d, hidden, 3, 0.3f),
            RandProj(hidden, nH * d, 4, 0.3f), RandProj(idxH * idxD, hidden, 5, 0.3f), RandProj(idxD, hidden, 6, 0.3f));

        var rng = new Random(11);
        var x = new float[T * hidden];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        // "Needle": a distinctive, large-norm token far back in the context (position 700, well outside the last 2048-token window of the
        // final rows), so the final rows only see it through the sparse selection.
        for (int j = 0; j < hidden; j++) x[700 * hidden + j] *= 6f;

        float[] RunWith(IKvCache kv, out long bytes)
        {
            var layer = Build();
            using var state = layer.CreateState();
            var outp = new float[T * hidden];
            for (int s = 0; s < T; s += chunk)
            {
                int n = Math.Min(chunk, T - s);
                layer.Forward(x.AsSpan(s * hidden, n * hidden), n, state, outp.AsSpan(s * hidden, n * hidden), kv, 0);
            }
            bytes = kv switch { QuantizedKvCache q => q.AllocatedBytes, SimpleKvCache sk => sk.AllocatedBytes, _ => 0 };
            return outp;
        }

        using var kvF = new SimpleKvCache(KvGeometry.Uniform(1, nKv, d), T);
        var f = RunWith(kvF, out long bytesF);
        using var kv8 = new QuantizedKvCache(KvGeometry.Uniform(1, nKv, d), T, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        var q8 = RunWith(kv8, out long bytes8);
        using var kv4 = new QuantizedKvCache(KvGeometry.Uniform(1, nKv, d), T, KvCacheDType.Q4_0, KvCacheDType.Q4_0, 0);
        var q4 = RunWith(kv4, out _);

        // Rows beyond 2051 tokens are the sparse regime; compare relative L2 error of the layer output, and a logit-proxy KL/top-1.
        int from = 2052;
        double Rel(float[] a, float[] b)
        {
            double num = 0, den = 0;
            for (int i = from * hidden; i < a.Length; i++) { double dlt = a[i] - b[i]; num += dlt * dlt; den += (double)a[i] * a[i]; }
            return Math.Sqrt(num / den);
        }
        // Logit proxy: a fixed random 64x32 head over the layer output.
        var head = new float[32 * hidden]; var hr = new Random(99);
        for (int i = 0; i < head.Length; i++) head[i] = (float)(hr.NextDouble() * 2 - 1);
        float[] Proj(float[] o)
        {
            var lg = new float[(T - from) * 32];
            for (int r = from; r < T; r++)
                for (int c = 0; c < 32; c++)
                {
                    float acc = 0;
                    for (int j = 0; j < hidden; j++) acc += o[r * hidden + j] * head[c * hidden + j];
                    lg[(r - from) * 32 + c] = acc;
                }
            return lg;
        }
        var pf = Proj(f);
        var (kl8, top8) = Compare(pf, Proj(q8), 32, 0, sharpen: 3f);
        var (kl4, top4) = Compare(pf, Proj(q4), 32, 0, sharpen: 3f);
        double e8 = Rel(f, q8), e4 = Rel(f, q4);
        output.WriteLine($"T={T} beyond 2051: rel L2 err Q8_0={e8:E3} Q4_0={e4:E3} | proxy-logit KL Q8_0={kl8:E3} (top1 {top8:P1}) Q4_0={kl4:E3} (top1 {top4:P1}) | KV bytes fp32={bytesF} q8={bytes8} ({(double)bytesF / bytes8:F2}x)");

        // The selection (float pooled keys) is identical in all arms, so the quantised arms differ only through K/V rounding.
        Assert.True(e8 < 0.02, $"Q8_0 rel err {e8}");
        Assert.True(top8 >= 0.95, $"Q8_0 proxy top-1 {top8}");
        Assert.True(e4 > e8 * 3, $"control: Q4_0 ({e4}) must be clearly worse than Q8_0 ({e8})");
        // Control for the sparse regime itself: dense attention differs from the sparse float run, so the comparison above really is sparse.
        var denseLayer = Build(); denseLayer.ForceDense = true;
        using var kvD = new SimpleKvCache(KvGeometry.Uniform(1, nKv, d), T);
        using var stD = denseLayer.CreateState();
        var dense = new float[T * hidden];
        for (int s = 0; s < T; s += chunk)
        {
            int n = Math.Min(chunk, T - s);
            denseLayer.Forward(x.AsSpan(s * hidden, n * hidden), n, stD, dense.AsSpan(s * hidden, n * hidden), kvD, 0);
        }
        Assert.True(Rel(f, dense) > 1e-4, "the run never left the exactly-dense regime");
        Assert.True(bytesF / (double)bytes8 > 3.5);
    }
}
