using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Vulkan decode-at-depth (#1744): the Granite dense host (TwIL-8B Q4_K_M-balanced) stores KV as F16 by default (it was F32, twice the bytes
/// per token, because <c>Architecture.Granite</c> was not on the dense-F16 list) and decodes with the grouped split-KV pass. The explicit F32
/// cache (<c>CreateKvCache(.., VulkanKvStorage.F32)</c>, per-head split pass) is the oracle and the A/B baseline. Real wikitext, not random ids.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanGraniteF16KvTests
{
    private readonly ITestOutputHelper _out;
    public VulkanGraniteF16KvTests(ITestOutputHelper output) => _out = output;

    private sealed class Loaded : IDisposable
    {
        public GgufFile Gguf = null!;
        public ModelConfig Cfg = null!;
        public VulkanTransformerModel Vk = null!;
        public void Dispose() { Vk.Dispose(); Gguf.Dispose(); }
    }

    private static Loaded Load()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_TWIL_8B_Q4_K_M_GGUF", "webAI-Official", "TwIL-8B",
            "TwIL-8B-Q4_K_M-balanced.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("TwIL-8B Q4_K_M-balanced GGUF"));
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
            var ids = tok.Encode(File.ReadAllText(corpus)).ToArray();
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

    [SkippableFact]
    public void GraniteHost_DefaultsToF16_AndF32IsSelectable()
    {
        using var l = Load();
        using var def = l.Vk.CreateKvCache(1024);
        using var f32 = l.Vk.CreateKvCache(1024, false, VulkanKvStorage.F32);
        Assert.Equal(DotLLM.Core.Configuration.Architecture.Granite, l.Cfg.Architecture);
        Assert.Equal(VulkanKvStorage.F16, l.Vk.DefaultKvStorage);
        Assert.Equal(VulkanKvStorage.F16, def.Storage);
        Assert.Equal(VulkanKvStorage.F32, f32.Storage);
        Assert.Equal(f32.AllocatedBytes / 2, def.AllocatedBytes);
    }

    /// <summary>F16 (grouped split-KV) vs F32 (per-head split-KV) cache on real text: chunked prefill to 8K, then 32 teacher-forced decode steps.</summary>
    [SkippableFact]
    public void TwIL_F16Kv_VsF32Kv_KlTop1AndNll_To8K()
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
        var kls = new List<double>[2] { new(), new() };
        var top = new int[2]; var cnt = new int[2]; var nllP = new double[2]; var nllQ = new double[2];
        void Compare(float[] a, float[] b, int rows, int firstPos, bool decode)
        {
            int band = decode ? 1 : 0;
            for (int r = 0; r < rows; r++)
            {
                var (kl, t1, p, q) = Row(a, r * V, b, r * V, V, ids[firstPos + r + 1]);
                kls[band].Add(kl); cnt[band]++; if (t1) top[band]++; nllP[band] += p; nllQ[band] += q;
            }
        }
        for (int s = 0; s < N; s += Chunk)
        {
            int[] pos = Enumerable.Range(s, Chunk).ToArray();
            float[] a, b;
            using (var ta = vk.Forward(ids.AsSpan(s, Chunk), pos, 0, cf32)) a = Copy(ta);
            using (var tb = vk.Forward(ids.AsSpan(s, Chunk), pos, 0, cf16)) b = Copy(tb);
            Compare(a, b, Chunk, s, false);
        }
        for (int s = N; s < N + Decode; s++)
        {
            float[] a, b;
            using (var ta = vk.Forward(ids.AsSpan(s, 1), new[] { s }, 0, cf32)) a = Copy(ta);
            using (var tb = vk.Forward(ids.AsSpan(s, 1), new[] { s }, 0, cf16)) b = Copy(tb);
            Compare(a, b, 1, s, true);
        }
        string[] names = { "prefill [0,8K)", "decode @8K" };
        for (int i = 0; i < 2; i++)
        {
            kls[i].Sort();
            double med = kls[i][kls[i].Count / 2], p99 = kls[i][(int)(kls[i].Count * 0.99)], mean = kls[i].Average();
            _out.WriteLine($"[1744] {names[i],-16} rows {cnt[i],5}  KL median {med:E2} mean {mean:E2} p99 {p99:E2} max {kls[i][^1]:E2}  top-1 {100.0 * top[i] / cnt[i]:F1}%  NLL F32 {nllP[i] / cnt[i]:F4} F16 {nllQ[i] / cnt[i]:F4} (delta {(nllQ[i] - nllP[i]) / cnt[i]:+0.0000;-0.0000})");
        }
        for (int i = 0; i < 2; i++)
        {
            Assert.True(kls[i][kls[i].Count / 2] > 0, "F16 KV is bit-identical to F32: the arms are not different (control insensitive)");
            Assert.True(kls[i][kls[i].Count / 2] < 3e-3, $"{names[i]} median KL {kls[i][kls[i].Count / 2]:E2}");
            Assert.True(100.0 * top[i] / cnt[i] > 93.0, $"{names[i]} top-1 {100.0 * top[i] / cnt[i]:F1}%");
        }
    }

    /// <summary>
    /// Same-process paired A/B (opt-in, <c>DOTLLM_F16KV_BENCH=1</c>; <c>DOTLLM_F16KV_BENCH_MAX</c> = deepest context, default 32768;
    /// <c>DOTLLM_F16KV_BENCH_ROUNDS</c>, default 2): F32 and F16 caches alternate (A B B A) so clock ramp and load hit both arms alike. One
    /// 512-row-chunked prefill per arm with a 32-step decode at each checkpoint depth (512, 4K, 16K, 32K).
    /// </summary>
    [SkippableFact]
    public void Bench_DecodeAtDepth_F16VsF32_PairedInProcess()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_F16KV_BENCH") == "1", "set DOTLLM_F16KV_BENCH=1");
        using var l = Load();
        var vk = l.Vk;
        int max = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_F16KV_BENCH_MAX"), out int m) ? m : 32768;
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_F16KV_BENCH_ROUNDS"), out int r0) ? r0 : 2;
        const int Steps = 32;
        int[] ids = CorpusIds(l.Gguf, 20000);
        int[] depths = new[] { 512, 4096, 16384, 32768 }.Where(d => d <= max).ToArray();
        var res = new Dictionary<string, List<double>>();
        void Add(string k, double v) { if (!res.TryGetValue(k, out var li)) res[k] = li = new(); li.Add(v); }
        int Tok(int i) => ids[i % ids.Length];
        void Prefill(VulkanKvCache c, int from, int to)
        {
            for (int s = from; s < to; s += 512)
            {
                int n = Math.Min(512, to - s);
                var chunk = Enumerable.Range(s, n).Select(Tok).ToArray();
                using (vk.Forward(chunk, Enumerable.Range(s, n).ToArray(), 0, c)) { }
            }
        }
        double DecodeMs(VulkanKvCache c, int start)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < Steps; i++) using (vk.Forward(new[] { Tok(start + i) }, new[] { start + i }, 0, c)) { }
            return sw.Elapsed.TotalMilliseconds / Steps;
        }
        void Pass(VulkanKvStorage st, bool record)
        {
            using var c = vk.CreateKvCache(max + 512, false, st);
            int pos = 0;
            foreach (int d in depths)
            {
                Prefill(c, pos, d); pos = d;
                double ms = DecodeMs(c, pos); pos += Steps;
                if (record) Add($"{st} decode@{d} tok/s", 1000.0 / ms);
            }
        }
        Pass(VulkanKvStorage.F16, false);   // warm
        var order = new[] { VulkanKvStorage.F32, VulkanKvStorage.F16, VulkanKvStorage.F16, VulkanKvStorage.F32 };
        for (int round = 0; round < rounds; round++)
            foreach (var st in order) Pass(st, true);
        foreach (var kv in res.OrderBy(k => k.Key))
            _out.WriteLine($"[1744 bench] {kv.Key,-26} n={kv.Value.Count} min {kv.Value.Min():F2} median {kv.Value.OrderBy(x => x).ElementAt(kv.Value.Count / 2):F2} max {kv.Value.Max():F2}  all: {string.Join(" ", kv.Value.Select(x => x.ToString("F2")))}");
    }
}
