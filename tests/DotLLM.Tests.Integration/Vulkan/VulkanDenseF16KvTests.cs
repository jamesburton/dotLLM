using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1151 / #1549: the dense Vulkan host (Qwen3-4B Q4_K_M here) stores KV as F16 by default and every attention path reads it in place.
/// The F32 cache is the oracle (explicit <c>CreateKvCache(.., VulkanKvStorage.F32)</c>). Accuracy is measured on real wikitext, not random ids.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanDenseF16KvTests
{
    private readonly ITestOutputHelper _out;
    public VulkanDenseF16KvTests(ITestOutputHelper output) => _out = output;

    private sealed class Loaded : IDisposable
    {
        public GgufFile Gguf = null!;
        public DotLLM.Core.Models.ModelConfig Cfg = null!;
        public VulkanTransformerModel Vk = null!;
        public void Dispose() { Vk.Dispose(); Gguf.Dispose(); }
    }

    private static Loaded Load()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));
        var l = new Loaded { Gguf = GgufFile.Open(fx.Path!) };
        l.Cfg = GgufModelConfigExtractor.Extract(l.Gguf.Metadata);
        l.Vk = VulkanTransformerModel.LoadFromGguf(l.Gguf, l.Cfg, VulkanModelLoader.ResolveSpvDir());
        return l;
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var r = new float[t.ElementCount];
        new ReadOnlySpan<float>((void*)t.DataPointer, r.Length).CopyTo(r);
        return r;
    }

    /// <summary>Real text ids: the wikitext-2 LF corpus when present, else a deterministic pseudo-text.</summary>
    private static int[] CorpusIds(GgufFile gguf, int count)
    {
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        if (File.Exists(corpus))
        {
            var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            var ids = tok.Encode(File.ReadAllText(corpus)[..Math.Min(400_000, (int)new FileInfo(corpus).Length)]).ToArray();
            if (ids.Length >= count) return ids[..count];
        }
        return Enumerable.Range(0, count).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
    }

    // KL(p||q) in nats, p = F32-cache distribution, q = F16-cache distribution, for one logit row; also nll of target under each, argmax equality.
    private static (double kl, bool top1, double nllP, double nllQ) Row(float[] a, int aOff, float[] b, int bOff, int V, int target)
    {
        double ma = double.MinValue, mb = double.MinValue; int am = 0, bm = 0;
        for (int i = 0; i < V; i++)
        {
            if (a[aOff + i] > ma) { ma = a[aOff + i]; am = i; }
            if (b[bOff + i] > mb) { mb = b[bOff + i]; bm = i; }
        }
        double za = 0, zb = 0;
        for (int i = 0; i < V; i++) { za += Math.Exp(a[aOff + i] - ma); zb += Math.Exp(b[bOff + i] - mb); }
        double la = Math.Log(za), lb = Math.Log(zb), kl = 0;
        for (int i = 0; i < V; i++)
        {
            double lp = a[aOff + i] - ma - la, lq = b[bOff + i] - mb - lb;
            double p = Math.Exp(lp);
            if (p > 1e-12) kl += p * (lp - lq);
        }
        double nllP = target >= 0 ? -(a[aOff + target] - ma - la) : 0, nllQ = target >= 0 ? -(b[bOff + target] - mb - lb) : 0;
        return (kl, am == bm, nllP, nllQ);
    }

    private static double MaxAbsDiff(float[] a, int ao, float[] b, int bo, int V)
    {
        double m = 0;
        for (int i = 0; i < V; i++) m = Math.Max(m, Math.Abs(a[ao + i] - b[bo + i]));
        return m;
    }

    [SkippableFact]
    public void DenseHost_DefaultsToF16_AndF32IsSelectable()
    {
        using var l = Load();
        using var def = l.Vk.CreateKvCache(1024);
        using var f32 = l.Vk.CreateKvCache(1024, false, VulkanKvStorage.F32);
        Assert.Equal(VulkanKvStorage.F16, l.Vk.DefaultKvStorage);
        Assert.Equal(VulkanKvStorage.F16, def.Storage);
        Assert.Equal(VulkanKvStorage.F32, f32.Storage);
        Assert.Equal(f32.AllocatedBytes / 2, def.AllocatedBytes);   // KV bytes per token halve
        _out.WriteLine($"[1151] KV bytes/token F32 {f32.AllocatedBytes / 1024} KiB-per-1024 vs F16 {def.AllocatedBytes / 1024}");
    }

    /// <summary>
    /// F16 vs F32 KV on real text, 512-row chunked prefill to 8K (so late chunks attend a long F16 history) then 32 teacher-forced decode steps
    /// (split-KV decode). Reports KL(F32 || F16), top-1 agreement and the NLL delta by context band.
    /// </summary>
    [SkippableFact]
    public void Qwen3_4B_F16Kv_VsF32Kv_KlTop1AndNll_To8K()
    {
        using var l = Load();
        var vk = l.Vk;
        const int Chunk = 512, N = 8192, Decode = 32;
        int V = l.Cfg.VocabSize;
        int[] ids = CorpusIds(l.Gguf, N + Decode + 1);
        vk.TrySetAllRowLogitsLimit(Chunk);
        using var cf32 = vk.CreateKvCache(N + Decode + 8, false, VulkanKvStorage.F32);
        using var cf16 = vk.CreateKvCache(N + Decode + 8);
        Assert.Equal(VulkanKvStorage.F16, cf16.Storage);

        // bands: 0 = [0,2K), 1 = [2K,8K), 2 = decode
        var kls = new List<double>[3] { new(), new(), new() };
        var top = new int[3]; var cnt = new int[3]; var nllP = new double[3]; var nllQ = new double[3];
        void Compare(float[] a, float[] b, int rows, int firstPos, int band0Limit)
        {
            for (int r = 0; r < rows; r++)
            {
                int pos = firstPos + r;
                int band = band0Limit < 0 ? 2 : pos < 2048 ? 0 : 1;
                var (kl, t1, p, q) = Row(a, r * V, b, r * V, V, ids[pos + 1]);
                if (kl > 1e-2) _out.WriteLine($"[1151] outlier row pos {pos}: KL {kl:E2}, max|dlogit| {MaxAbsDiff(a, r * V, b, r * V, V):F3}, top-1 {t1}");
                kls[band].Add(kl); cnt[band]++; if (t1) top[band]++; nllP[band] += p; nllQ[band] += q;
            }
        }
        for (int s = 0; s < N; s += Chunk)
        {
            int[] pos = Enumerable.Range(s, Chunk).ToArray();
            float[] a, b;
            using (var ta = vk.Forward(ids.AsSpan(s, Chunk), pos, 0, cf32)) a = Copy(ta);
            using (var tb = vk.Forward(ids.AsSpan(s, Chunk), pos, 0, cf16)) b = Copy(tb);
            Compare(a, b, Chunk, s, 0);
        }
        for (int s = N; s < N + Decode; s++)
        {
            float[] a, b;
            using (var ta = vk.Forward(ids.AsSpan(s, 1), new[] { s }, 0, cf32)) a = Copy(ta);
            using (var tb = vk.Forward(ids.AsSpan(s, 1), new[] { s }, 0, cf16)) b = Copy(tb);
            Compare(a, b, 1, s, -1);
        }
        string[] names = { "prefill [0,2K)", "prefill [2K,8K)", "decode @8K" };
        for (int i = 0; i < 3; i++)
        {
            kls[i].Sort();
            double med = kls[i][kls[i].Count / 2], p99 = kls[i][(int)(kls[i].Count * 0.99)], mean = kls[i].Average();
            _out.WriteLine($"[1151] {names[i],-16} rows {cnt[i],5}  KL median {med:E2} mean {mean:E2} p99 {p99:E2} max {kls[i][^1]:E2}  top-1 {100.0 * top[i] / cnt[i]:F1}%  NLL F32 {nllP[i] / cnt[i]:F4} F16 {nllQ[i] / cnt[i]:F4} (delta {(nllQ[i] - nllP[i]) / cnt[i]:+0.0000;-0.0000})");
        }
        for (int i = 0; i < 3; i++)
        {
            Assert.True(kls[i][kls[i].Count / 2] > 0, "F16 KV is bit-identical to F32: the arms are not different (control insensitive)");
            Assert.True(kls[i][kls[i].Count / 2] < 3e-3, $"{names[i]} median KL {kls[i][kls[i].Count / 2]:E2}");
            Assert.True(100.0 * top[i] / cnt[i] > 93.0, $"{names[i]} top-1 {100.0 * top[i] / cnt[i]:F1}%");
        }
    }

    /// <summary>
    /// The fused batched decode (ForwardBatch) takes F16 caches: two sequences batched equal the same two run one by one on F16 caches, and the
    /// fused path (not the per-sequence fallback) is what ran.
    /// </summary>
    [SkippableFact]
    public void ForwardBatch_WithF16Caches_TakesFusedPath_AndMatchesPerSequence()
    {
        using var l = Load();
        var vk = l.Vk;
        int[] a = CorpusIds(l.Gguf, 300), b = a.Select(x => (x * 7 + 3) % 5000 + 50).ToArray();
        int[] pa = Enumerable.Range(0, 200).ToArray();
        using var ba = vk.CreateKvCache(512); using var bb = vk.CreateKvCache(512);
        using var sa = vk.CreateKvCache(512); using var sb = vk.CreateKvCache(512);
        foreach (var (c, ids) in new[] { (ba, a), (bb, b), (sa, a), (sb, b) })
            using (vk.Forward(ids.AsSpan(0, 200), pa, 0, c)) { }

        long fused0 = vk.ForwardBatchFusedCount;
        double worst = 0; int agree = 0, n = 0;
        for (int t = 200; t < 216; t++)
        {
            var reqs = new[]
            {
                new SequenceForwardRequest { TokenIds = new[] { a[t] }, Positions = new[] { t }, KvCache = ba },
                new SequenceForwardRequest { TokenIds = new[] { b[t] }, Positions = new[] { t }, KvCache = bb },
            };
            var batched = vk.ForwardBatch(reqs, deviceId: -1);
            float[] la = Copy(batched[0]), lb = Copy(batched[1]);
            float[] ra, rb;
            using (var x = vk.Forward(new[] { a[t] }, new[] { t }, 0, sa)) ra = Copy(x);
            using (var x = vk.Forward(new[] { b[t] }, new[] { t }, 0, sb)) rb = Copy(x);
            foreach (var (p, q) in new[] { (ra, la), (rb, lb) })
            {
                var (kl, t1, _, _) = Row(p, 0, q, 0, l.Cfg.VocabSize, -1);
                worst = Math.Max(worst, kl); n++; if (t1) agree++;
            }
        }
        _out.WriteLine($"[1151] ForwardBatch F16: fused calls {vk.ForwardBatchFusedCount - fused0}, worst KL batched-vs-single {worst:E2}, top-1 {agree}/{n}");
        Assert.Equal(16L, vk.ForwardBatchFusedCount - fused0);   // every call took the fused path, none fell back per sequence
        Assert.True(worst < 2e-3, $"worst KL {worst:E2}");
        Assert.True(agree >= n - 1);
    }

    /// <summary>
    /// IPositionedHostKvCache on the model's default (F16) cache: host F32 rows round to nearest-even on write, F16 reads widen exactly,
    /// the round trip is idempotent, and out-of-range values saturate (as the device writers do) instead of becoming infinities.
    /// </summary>
    [SkippableFact]
    public void PositionedRowIo_OnDefaultF16Cache_RoundsRneWidensAndIsIdempotent()
    {
        using var l = Load();
        using var c = l.Vk.CreateKvCache(256);
        Assert.Equal(VulkanKvStorage.F16, c.Storage);
        int stride = c.KvStride, rows = 40, start = 17;
        var rng = new Random(1151);
        var k = new float[rows * stride]; var v = new float[rows * stride];
        for (int i = 0; i < k.Length; i++)
        {
            k[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, rng.Next(-6, 4)));   // spans subnormal-half to 1e3 magnitudes
            v[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        k[0] = 1e6f; k[1] = -1e6f; k[2] = 65504f; k[3] = 65519.9f;   // saturation edge: > max half
        c.WriteRows(3, start, rows, k, v);
        var rk = new float[rows * stride]; var rv = new float[rows * stride];
        c.ReadRows(3, start, rows, rk, rv);
        for (int i = 0; i < k.Length; i++)
        {
            float ek = Math.Clamp(k[i], -65504f, 65504f);
            Assert.True((float)(Half)ek == rk[i], $"K[{i}] {k[i]:R}: expected RNE half {(float)(Half)ek:R}, got {rk[i]:R}");
            Assert.True((float)(Half)v[i] == rv[i], $"V[{i}]");
        }
        // Idempotent: writing back what was read changes nothing, bit for bit.
        c.WriteRows(3, start, rows, rk, rv);
        var rk2 = new float[rows * stride]; var rv2 = new float[rows * stride];
        c.ReadRows(3, start, rows, rk2, rv2);
        Assert.True(rk.AsSpan().SequenceEqual(rk2) && rv.AsSpan().SequenceEqual(rv2));
        // Neighbouring rows untouched, length covers the write.
        Assert.Equal(start + rows, c.CurrentLength);
        // Host-staged transport (handoff / hybrid CPU prefill ingest) works on F16 too: UploadLayer == WriteRows at 0, DownloadLayer widens.
        c.UploadLayer(5, 20, k.AsSpan(0, 20 * stride), v.AsSpan(0, 20 * stride));
        var dk = new float[c.CurrentLength * stride]; var dv = new float[c.CurrentLength * stride];
        c.DownloadLayer(5, dk, dv);
        for (int i = 0; i < 20 * stride; i++) Assert.Equal((float)(Half)Math.Clamp(k[i], -65504f, 65504f), dk[i]);
    }

    /// <summary>
    /// Error accumulation of the device RoPE position shift (RopeShiftKv, #1533) on REAL Qwen3-4B K rows: one shift with delta up to 32K, and an
    /// already-shifted chunk shifted again 1..4 times, F16 cache vs F32 cache. Metric: per (row, head) ||got - ref|| / ||ref||, ref = the source
    /// row rotated by the summed delta in double precision. The F16 cache pays one rounding per store (source write + each shift).
    /// </summary>
    [SkippableFact]
    public void RopeShiftKv_ErrorAccumulation_OnRealQwen3K_F16VsF32()
    {
        using var l = Load();
        var vk = l.Vk;
        int heads = l.Cfg.NumKvHeads, hd = l.Cfg.HeadDim, stride = heads * hd, rows = 512;
        float theta = l.Cfg.RoPEConfig!.Value.Theta;
        int[] ids = CorpusIds(l.Gguf, rows);
        using var src = vk.CreateKvCache(rows + 8, false, VulkanKvStorage.F32);
        using (vk.Forward(ids, Enumerable.Range(0, rows).ToArray(), 0, src)) { }
        using var kernel = RopeShiftKvKernel.Create(src_device(vk), VulkanModelLoader.ResolveSpvDir());

        var worstByCase = new List<string>();
        foreach (int layer in new[] { 0, 18, l.Cfg.NumLayers - 1 })
        {
            var k0 = new float[rows * stride]; var v0 = new float[rows * stride];
            src.ReadRows(layer, 0, rows, k0, v0);
            var cases = new List<int[]>
            {
                new[] { 1 }, new[] { 64 }, new[] { 1024 }, new[] { 8192 }, new[] { 32768 },
                new[] { 4096 }, new[] { 4096, 4096 }, new[] { 4096, 4096, 4096 }, new[] { 4096, 4096, 4096, 4096 },
                new[] { 1000, -700, 2500, -300 },
            };
            foreach (int[] deltas in cases)
            {
                double[] mean = new double[2], worst = new double[2];
                for (int si = 0; si < 2; si++)
                {
                    var storage = si == 0 ? VulkanKvStorage.F32 : VulkanKvStorage.F16;
                    using var c = new VulkanKvCache(src_device(vk), new[] { stride }, rows + 8, storage);
                    c.WriteRows(0, 0, rows, k0, v0);
                    foreach (int d in deltas)
                        kernel.Shift(c, 0, 0, rows, d, heads, hd, hd, theta, DotLLM.Vulkan.Kernels.RopeF32Kernel.Variant.NeoX);
                    var got = new float[rows * stride]; var gv = new float[rows * stride];
                    c.ReadRows(0, 0, rows, got, gv);
                    int total = deltas.Sum();
                    double sum = 0; int n = 0;
                    for (int r = 0; r < rows; r++)
                        for (int h = 0; h < heads; h++)
                        {
                            double dn = 0, rn = 0;
                            int b = r * stride + h * hd, half = hd / 2;
                            for (int i = 0; i < half; i++)
                            {
                                double ang = total * Math.Pow((double)theta, -2.0 * i / hd), cs = Math.Cos(ang), sn = Math.Sin(ang);
                                double x0 = k0[b + i], x1 = k0[b + i + half];
                                double r0 = x0 * cs - x1 * sn, r1 = x0 * sn + x1 * cs;
                                dn += (got[b + i] - r0) * (got[b + i] - r0) + (got[b + i + half] - r1) * (got[b + i + half] - r1);
                                rn += r0 * r0 + r1 * r1;
                            }
                            double rel = Math.Sqrt(dn / rn);
                            sum += rel; n++; worst[si] = Math.Max(worst[si], rel);
                        }
                    mean[si] = sum / n;
                }
                string line = $"layer {layer,2} deltas [{string.Join(",", deltas)}]: F32 mean {mean[0]:E2} worst {worst[0]:E2} | F16 mean {mean[1]:E2} worst {worst[1]:E2}";
                worstByCase.Add(line);
                _out.WriteLine("[1533/1151] " + line);
                // F16: the source write rounds once (<= 2^-11 relative per element), every shift rounds once more: mean error grows like sqrt(1 + shifts).
                Assert.True(mean[1] < 4.5e-4 * Math.Sqrt(1 + deltas.Length) + mean[0], line);
                Assert.True(worst[1] < 3e-3 * Math.Sqrt(1 + deltas.Length) + worst[0], line);
            }
        }
    }

    private static VulkanDevice src_device(VulkanTransformerModel vk) => vk.Device;
}
