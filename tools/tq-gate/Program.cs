// tq-gate: gates and benches for in-place TurboQuant KV (#1892 task 10).
//
//   tq-gate --backend cpu|vulkan|cuda --model <gguf> --mode speed|parity|quality --out <file.jsonl> [options]
//
// speed   : in-place vs scratch oracle, interleaved A B A B ..., one discarded warm-up round. --bits --ctx --decode --reps --chunk
// parity  : teacher-forced logits in-place vs scratch oracle (plus the plain-KV baseline for the codec effect), and a
//           free-running greedy identity run. --bits --layers (0 = all) --steps --gen
// quality : KL / PPL / top-1 / greedy identity of the in-place cache against the plain KV (f16 on GPU, f32 SimpleKvCache on CPU).
// Every arm records the cache type it built; GPU arms record the attention-path counters.
// A Vulkan host holds one KV storage override, so on Vulkan give ONE --bits value per process.
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;

var a = ParseArgs(args);
string backend = a["backend"], modelPath = a["model"], mode = a["mode"], outFile = a["out"];
int[] bitsList = a.GetValueOrDefault("bits", "4").Split(',').Select(int.Parse).ToArray();
int ctx = int.Parse(a.GetValueOrDefault("ctx", "7600"));
int decodeN = int.Parse(a.GetValueOrDefault("decode", "64"));
int reps = int.Parse(a.GetValueOrDefault("reps", "3"));
int chunk = int.Parse(a.GetValueOrDefault("chunk", backend == "cuda" ? "1024" : "512"));
int layers = int.Parse(a.GetValueOrDefault("layers", "0"));
int steps = int.Parse(a.GetValueOrDefault("steps", "192"));
int gen = int.Parse(a.GetValueOrDefault("gen", "256"));
int threads = int.Parse(a.GetValueOrDefault("threads", "0"));
string tag = a.GetValueOrDefault("tag", Path.GetFileNameWithoutExtension(modelPath));
string corpus = a.GetValueOrDefault("corpus",
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw"));
const ulong Seed = 0x5DEECE66DUL;   // KvCacheConfig default TurboQuantSeed: what the CLI builds with
if (backend == "vulkan" && bitsList.Length > 1) throw new ArgumentException("one --bits value per process on vulkan");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);

using var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
if (layers > 0) config = config with { NumLayers = layers };
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
int vocab = config.VocabSize;
string text = File.ReadAllText(corpus);
int[] all = tokenizer.Encode(text.Length > 900_000 ? text[..900_000] : text);
Log($"backend={backend} model={tag} layers={config.NumLayers} kvHeads={config.NumKvHeads} hd={config.HeadDim} vocab={vocab} corpusTokens={all.Length}");

IModel model = backend switch
{
    "cpu" => TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(threads)),
    "vulkan" => DotLLM.Vulkan.VulkanTransformerModel.LoadFromGguf(gguf, config, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()),
    "cuda" => DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, 0, null),
    _ => throw new ArgumentException("backend"),
};
using var _m = model;
int dev = backend == "cuda" ? 0 : -1;
string spv = backend == "vulkan" ? DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir() : "";

switch (mode)
{
    case "speed": foreach (int b in bitsList) Speed(b); break;
    case "parity": foreach (int b in bitsList) Parity(b); break;
    case "quality": Quality(); break;
    case "vgpr": foreach (int b in bitsList) Vgpr(b); break;
    default: throw new ArgumentException("mode");
}
return;

// ───────────────────────── caches ─────────────────────────
// Plain() must be called BEFORE InPlaceFactory(): preparing a Vulkan host switches what CreateKvCache builds.
IKvCache Plain(int size) => backend switch
{
    "cpu" => new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, size),
    "vulkan" => ((DotLLM.Vulkan.VulkanTransformerModel)model).CreateKvCache(size),
    _ => ((DotLLM.Cuda.CudaTransformerModel)model).CreateKvCache(size, new KvCacheConfig()),
};

(TurboQuantCodec k, TurboQuantCodec v) Codecs(int bits)
    => (new TurboQuantCodec(config.HeadDim, bits, Seed), new TurboQuantCodec(config.HeadDim, bits, Seed ^ TurboQuantCodec.VSeedMask));

IKvCache Scratch(int size, int bits)
{
    var (k, v) = Codecs(bits);
    return backend switch
    {
        "cpu" => new TurboQuantKvCache(KvGeometry.SlotCount(config), config.NumKvHeads, config.HeadDim, size, bits, Seed, useQjl: false),
        "vulkan" => ((DotLLM.Vulkan.VulkanTransformerModel)model).CreateTurboQuantKvCache(spv, size, bits, k.Centroids, k.RotationSigns, v.RotationSigns, k.InvSqrtD),
        _ => ((DotLLM.Cuda.CudaTransformerModel)model).CreateTurboQuantKvCache(size, bits, k.Centroids, k.RotationSigns, v.RotationSigns, k.InvSqrtD),
    };
}

// The in-place cache is built by the production helper (the one serve/run/chat/bench use).
Func<ModelConfig, int, IKvCache> InPlaceFactory(int bits)
{
    var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, TurboQuantBits: bits);
    string? refusal = TurboQuantKv.PostLoadRefusal(kv, null, KvCommand.Bench, false, config, model, false, false);
    if (refusal is not null) throw new InvalidOperationException("in-place refused: " + refusal);
    return TurboQuantKv.CreateFactory(model, kv);
}

(long attn, long eager, long capture, long replay, long inplaceCaches) Counters()
{
    long attn = 0, eager = 0, cap = 0, rep = 0;
    if (model is DotLLM.Vulkan.VulkanTransformerModel vk) attn = vk.TurboQuantAttentionCount;
    if (model is DotLLM.Cuda.CudaTransformerModel cu)
    {
        eager = Prop(cu, "EagerDecodeCount"); cap = Prop(cu, "GraphCaptureCount"); rep = Prop(cu, "GraphReplayCount");
        var kt = typeof(DotLLM.Cuda.CudaTransformerModel).Assembly.GetType("DotLLM.Cuda.CudaKernels")!;
        foreach (var n in new[] { "AttentionKvTqLaunches", "AttentionFlashPrefillKvTqLaunches" })
            attn += Convert.ToInt64(kt.GetProperty(n, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)?.GetValue(null)
                ?? kt.GetField(n, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)?.GetValue(null) ?? 0L);
    }
    return (attn, eager, cap, rep, TurboQuantKv.InPlaceCachesBuilt);
}
static long Prop(object o, string name)
    => Convert.ToInt64(o.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.GetValue(o));

// ───────────────────────── forward helpers ─────────────────────────
float[] Row(ITensor logits, int row)
{
    var r = new float[vocab];
    unsafe { new ReadOnlySpan<float>((float*)logits.DataPointer + (long)row * vocab, vocab).CopyTo(r); }
    return r;
}

double Prefill(IKvCache c, int[] t, int n)
{
    var sw = Stopwatch.StartNew();
    for (int s = 0; s < n; s += chunk)
    {
        int m = Math.Min(chunk, n - s);
        var pos = new int[m];
        for (int i = 0; i < m; i++) pos[i] = s + i;
        using var l = model.Forward(t.AsSpan(s, m), pos, dev, c, true);
    }
    return sw.Elapsed.TotalSeconds;
}

double Decode(IKvCache c, int[] t, int from, int n)
{
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < n; i++)
    {
        using var l = model.Forward(t.AsSpan(from + i, 1), [from + i], dev, c, true);
    }
    return sw.Elapsed.TotalSeconds;
}

// ───────────────────────── speed ─────────────────────────
static string Vram()
{
    try
    {
        var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used,memory.total --format=csv,noheader,nounits") { RedirectStandardOutput = true, UseShellExecute = false };
        using var p = Process.Start(psi)!;
        string o = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit();
        return o.Replace(" ", "").Replace(",", "/") + "MiB";
    }
    catch { return "n/a"; }
}


void Speed(int bits)
{
    if (all.Length < ctx + decodeN + 2) throw new InvalidOperationException("corpus too short");
    int[] t = all;
    var factory = InPlaceFactory(bits);
    int size = ctx + decodeN + 8;
    var arms = new (string name, Func<IKvCache> make)[]
    {
        ("scratch", () => Scratch(size, bits)),
        ("inplace", () => factory(config, size)),
    };
    var res = new Dictionary<string, List<(double pf, double dc)>> { ["scratch"] = [], ["inplace"] = [] };
    var paths = new Dictionary<string, string>();
    if (a.ContainsKey("reuse-prefill"))
    {
        // Decode-only A/B: ONE timed prefill per arm (reported without a range), then the decode reps roll the cache back to ctx. Same-session, interleaved.
        var caches = new Dictionary<string, IKvCache>(); var pfOnce = new Dictionary<string, double>();
        foreach (var (name, make) in arms) { caches[name] = make(); pfOnce[name] = Prefill(caches[name], t, ctx); Log($"{name} prefill once: {ctx / pfOnce[name]:F1} tok/s ({pfOnce[name]:F0}s) vram={Vram()}"); }
        for (int r = 0; r <= reps; r++)
            foreach (var (name, _) in arms)
            {
                var c0 = Counters();
                double dc = Decode(caches[name], t, ctx, decodeN);
                caches[name].Rollback(ctx);
                var c1 = Counters();
                paths[name] = $"{caches[name].GetType().Name}; reusePrefill; vram={Vram()}; attnDelta={c1.attn - c0.attn} eagerDelta={c1.eager - c0.eager} captureDelta={c1.capture - c0.capture} replayDelta={c1.replay - c0.replay} inPlaceCachesBuilt={c1.inplaceCaches}";
                if (r == 0) { Log($"warmup {name}: decode {decodeN / dc:F2} tok/s [{paths[name]}]"); continue; }
                res[name].Add((pfOnce[name], dc));
                Log($"rep {r} {name}: decode {decodeN / dc:F2} tok/s ({dc:F1}s)");
            }
        foreach (var c in caches.Values) c.Dispose();
        foreach (var name in res.Keys)
        {
            var dcs = res[name].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray();
            Append(new { kind = "speed", backend, model = tag, bits, ctx, decodeN, chunk, reps, arm = name, path = paths[name], decodeOnly = true,
                prefillOnce = ctx / pfOnce[name], decodeMedian = Med(dcs), decodeMin = dcs[0], decodeMax = dcs[^1] });
        }
        Log($"SPEED(decode-only) {backend} tq{bits} ctx={ctx}: decode median scratch {Med(res["scratch"].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray()):F2} vs inplace {Med(res["inplace"].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray()):F2}");
        return;
    }
    // reps+1 rounds; round 0 is the discarded warm-up (one per arm). Order: A B A B ...
    for (int r = 0; r <= reps; r++)
        foreach (var (name, make) in arms)
        {
            var c0 = Counters();
            using var c = make();
            string vramAlloc = Vram();
            double pf = Prefill(c, t, ctx);
            string vramPrefill = Vram();
            double dc = Decode(c, t, ctx, decodeN);
            var c1 = Counters();
            paths[name] = $"{c.GetType().Name}; vramUsedAfterAlloc={vramAlloc} afterPrefill={vramPrefill} afterDecode={Vram()}; attnDelta={c1.attn - c0.attn} eagerDelta={c1.eager - c0.eager} captureDelta={c1.capture - c0.capture} replayDelta={c1.replay - c0.replay} inPlaceCachesBuilt={c1.inplaceCaches}";
            if (r == 0) { Log($"warmup {name}: prefill {ctx / pf:F1} tok/s decode {decodeN / dc:F2} tok/s  [{paths[name]}]"); continue; }
            res[name].Add((pf, dc));
            Log($"rep {r} {name}: prefill {ctx / pf:F1} tok/s ({pf:F1}s)  decode {decodeN / dc:F2} tok/s ({dc:F1}s)");
        }
    foreach (var name in res.Keys)
    {
        var pfs = res[name].Select(x => ctx / x.pf).OrderBy(x => x).ToArray();
        var dcs = res[name].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray();
        Append(new { kind = "speed", backend, model = tag, bits, ctx, decodeN, chunk, reps, arm = name, path = paths[name],
            prefillMedian = Med(pfs), prefillMin = pfs[0], prefillMax = pfs[^1], decodeMedian = Med(dcs), decodeMin = dcs[0], decodeMax = dcs[^1] });
    }
    Log($"SPEED {backend} tq{bits} ctx={ctx}: decode median scratch {Med(res["scratch"].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray()):F2} vs inplace {Med(res["inplace"].Select(x => decodeN / x.dc).OrderBy(x => x).ToArray()):F2}; " +
        $"prefill scratch {Med(res["scratch"].Select(x => ctx / x.pf).OrderBy(x => x).ToArray()):F1} vs inplace {Med(res["inplace"].Select(x => ctx / x.pf).OrderBy(x => x).ToArray()):F1}");
}

// ───────────────────────── parity ─────────────────────────
// Teacher-forced: a 64-token prompt in ONE forward (all rows kept), then one token per forward for `n` tokens. Returns every row.
List<float[]> TeacherForced(IKvCache c, int[] t, int n)
{
    var rows = new List<float[]>();
    const int P = 64;
    using (var l = model.Forward(t.AsSpan(0, P), Enumerable.Range(0, P).ToArray(), dev, c))
    {
        for (int i = 0; i < P - 1; i++) rows.Add([]);   // only the last prompt row is compared (a GPU forward may return only that row)
        rows.Add(Row(l, (int)l.Shape[0] - 1));
    }
    for (int i = P; i < P + n; i++)
    {
        using var l = model.Forward(t.AsSpan(i, 1), [i], dev, c);
        rows.Add(Row(l, 0));
    }
    return rows;
}

// Free-running greedy from a 64-token prompt; records argmax and the top-2 logit gap per step.
(int[] ids, double[] gap) Greedy(IKvCache c, int[] t, int n)
{
    const int P = 64;
    var ids = new int[n]; var gap = new double[n];
    float[] cur;
    using (var l = model.Forward(t.AsSpan(0, P), Enumerable.Range(0, P).ToArray(), dev, c, true)) cur = Row(l, 0);
    for (int i = 0; i < n; i++)
    {
        int b = 0; double b2 = double.NegativeInfinity;
        for (int v = 1; v < cur.Length; v++) if (cur[v] > cur[b]) b = v;
        for (int v = 0; v < cur.Length; v++) if (v != b && cur[v] > b2) b2 = cur[v];
        ids[i] = b; gap[i] = cur[b] - b2;
        if (i == n - 1) break;
        using var l = model.Forward([b], [P + i], dev, c, true);
        cur = Row(l, 0);
    }
    return (ids, gap);
}

void Parity(int bits)
{
    int[] t = all[..(64 + steps + 2)];
    var (k, v) = Codecs(bits);
    if (k.RotationSigns.SequenceEqual(v.RotationSigns)) throw new InvalidOperationException("K and V sign seeds must differ");
    List<float[]> plain; (int[] ids, double[] gap) gPlain;
    using (var c = Plain(64 + steps + 8)) plain = TeacherForced(c, t, steps);
    using (var c = Plain(64 + gen + 8)) gPlain = Greedy(c, all, gen);
    List<float[]> oracle; (int[] ids, double[] gap) gOracle; string oraclePath;
    long a0 = Counters().attn;
    using (var c = Scratch(64 + steps + 8, bits)) { oracle = TeacherForced(c, t, steps); oraclePath = c.GetType().Name; }
    long oracleAttn = Counters().attn - a0;
    using (var c = Scratch(64 + gen + 8, bits)) gOracle = Greedy(c, all, gen);
    var factory = InPlaceFactory(bits);
    var cs0 = Counters();
    List<float[]> inp; (int[] ids, double[] gap) gIn; string inPath;
    using (var c = factory(config, 64 + steps + 8)) { inp = TeacherForced(c, t, steps); inPath = c.GetType().Name; }
    var cs1 = Counters();
    using (var c = factory(config, 64 + gen + 8)) gIn = Greedy(c, all, gen);
    // Compared rows: the last prompt row (prefill) and every decode row.
    var rowsIdx = Enumerable.Range(63, steps + 1).ToArray();
    double minCos = 1, maxMeanAbs = 0, sumEff = 0, sumVs = 0; int agree = 0, worst = -1; var coses = new List<double>();
    foreach (int r in rowsIdx)
    {
        var (vsM, cos, same) = Cmp(inp[r], oracle[r]);
        var (effM, _, _) = Cmp(oracle[r], plain[r]);
        coses.Add(cos); sumVs += vsM; sumEff += effM; if (cos < minCos) { minCos = cos; worst = r; } if (vsM > maxMeanAbs) maxMeanAbs = vsM; if (same) agree++;
    }
    double ratio = sumVs / sumEff;
    int firstDiv = FirstDiff(gIn.ids, gOracle.ids);
    Append(new { kind = "parity", backend, model = tag, layers = config.NumLayers, bits, steps, rows = rowsIdx.Length,
        oraclePath, inPath, oracleAttnDelta = oracleAttn, inPlaceAttnDelta = cs1.attn - cs0.attn, inPlaceCachesBuilt = cs1.inplaceCaches,
        meanAbsInVsOracle = sumVs / rowsIdx.Length, meanAbsCodecEffect = sumEff / rowsIdx.Length, ratio, minCosInVsOracle = minCos, worstRow = worst, p5Cos = coses.OrderBy(x => x).ElementAt(coses.Count / 20),
        maxRowMeanAbs = maxMeanAbs, argmaxAgree = (double)agree / rowsIdx.Length,
        greedyLen = gen, firstDivergence = firstDiv,
        top2GapOracleAtDiv = firstDiv >= 0 ? gOracle.gap[firstDiv] : (double?)null, top2GapInPlaceAtDiv = firstDiv >= 0 ? gIn.gap[firstDiv] : (double?)null,
        greedyInVsPlainFirstDiv = FirstDiff(gIn.ids, gPlain.ids), greedyOracleVsPlainFirstDiv = FirstDiff(gOracle.ids, gPlain.ids) });
    Log($"PARITY {backend} layers={config.NumLayers} tq{bits}: ratio {ratio:F4} (mean|d| {sumVs / rowsIdx.Length:E3} vs codec effect {sumEff / rowsIdx.Length:E3}) minCos {minCos:F6} argmax {(double)agree / rowsIdx.Length:P1}; " +
        $"greedy first divergence {(firstDiv < 0 ? "none" : firstDiv.ToString())} paths {oraclePath}/{inPath} attnDelta oracle {oracleAttn} inplace {cs1.attn - cs0.attn}");
}

static int FirstDiff(int[] x, int[] y) { for (int i = 0; i < Math.Min(x.Length, y.Length); i++) if (x[i] != y[i]) return i; return -1; }

static (double MeanAbs, double Cos, bool Same) Cmp(float[] x, float[] y)
{
    double s = 0, dot = 0, na = 0, nb = 0; int ia = 0, ib = 0;
    for (int i = 0; i < x.Length; i++)
    {
        s += Math.Abs((double)x[i] - y[i]); dot += (double)x[i] * y[i]; na += (double)x[i] * x[i]; nb += (double)y[i] * y[i];
        if (x[i] > x[ia]) ia = i;
        if (y[i] > y[ib]) ib = i;
    }
    return (s / x.Length, dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30), ia == ib);
}

// ───────────────────────── quality ─────────────────────────
void Quality()
{
    int evalRows = int.Parse(a.GetValueOrDefault("rows", "512"));
    int[] t = all[..(ctx + 1)];
    string refName = backend == "cpu" ? "f32 SimpleKvCache" : "f16 KV";
    // Teacher-forced over ctx tokens in chunks; keeps the logits of the last evalRows rows (row i predicts t[i+1]).
    float[][] Run(IKvCache c)
    {
        var keep = new float[evalRows][];
        int qchunk = Math.Max(1, Math.Min(chunk, model.MaxAllRowLogitsLength));   // a GPU forward returns every row only up to this
        for (int s = 0; s < ctx; s += qchunk)
        {
            int m = Math.Min(qchunk, ctx - s);
            var pos = Enumerable.Range(s, m).ToArray();
            using var l = model.Forward(t.AsSpan(s, m), pos, dev, c);
            for (int i = 0; i < m; i++) { int g = s + i; if (g >= ctx - evalRows) keep[g - (ctx - evalRows)] = Row(l, i); }
        }
        return keep;
    }
    float[][] reference; int[] gRef;
    using (var c = Plain(ctx + 8)) reference = Run(c);
    using (var c = Plain(64 + gen + 8)) gRef = Greedy(c, all, gen).ids;
    double pplRef = Ppl(reference, t, ctx - evalRows);
    Log($"reference ({refName}) PPL {pplRef:F4} over {evalRows} rows at ctx {ctx}");
    foreach (int bits in bitsList)
    {
        var factory = InPlaceFactory(bits);
        float[][] cand; int[] gC; string path;
        using (var c = factory(config, ctx + 8)) { cand = Run(c); path = c.GetType().Name; }
        using (var c = factory(config, 64 + gen + 8)) gC = Greedy(c, all, gen).ids;
        double kl = 0; int top1 = 0;
        for (int i = 0; i < evalRows; i++) { kl += Kl(reference[i], cand[i]); if (Cmp(reference[i], cand[i]).Same) top1++; }
        double ppl = Ppl(cand, t, ctx - evalRows);
        int fd = FirstDiff(gC, gRef);
        Append(new { kind = "quality", backend, model = tag, bits, ctx, evalRows, reference = refName, path, kl = kl / evalRows, ppl, pplRef,
            pplDeltaPct = 100 * (ppl / pplRef - 1), top1Agree = (double)top1 / evalRows, greedyLen = gen, greedyFirstDivergenceVsRef = fd });
        Log($"QUALITY {backend} tq{bits}: KL {kl / evalRows:E3} PPL {ppl:F4} vs {pplRef:F4} ({100 * (ppl / pplRef - 1):+0.00;-0.00}%) top1 {(double)top1 / evalRows:P1} greedy first divergence {(fd < 0 ? "none" : fd.ToString())} [{path}]");
    }
}

double Ppl(float[][] rows, int[] t, int firstRow)
{
    double nll = 0;
    for (int i = 0; i < rows.Length; i++) nll -= LogSoftmaxAt(rows[i], t[firstRow + i + 1]);
    return Math.Exp(nll / rows.Length);
}
static double LogSoftmaxAt(float[] r, int idx)
{
    double mx = double.NegativeInfinity; foreach (var x in r) if (x > mx) mx = x;
    double s = 0; foreach (var x in r) s += Math.Exp(x - mx);
    return r[idx] - mx - Math.Log(s);
}
static double Kl(float[] r, float[] c)
{
    double rm = r.Max(), cm = c.Max(), rs = 0, cs = 0;
    for (int i = 0; i < r.Length; i++) { rs += Math.Exp(r[i] - rm); cs += Math.Exp(c[i] - cm); }
    double lr = Math.Log(rs), lc = Math.Log(cs), kl = 0;
    for (int i = 0; i < r.Length; i++) { double lp = r[i] - rm - lr; double p = Math.Exp(lp); if (p > 0) kl += p * (lp - (c[i] - cm - lc)); }
    return kl;
}

// ───────────────────────── vgpr (Vulkan, VK_AMD_shader_info) ─────────────────────────
// Walks the model's kernel objects by reflection, collects every ComputePipeline and asks the driver for its post-compile VGPR / SGPR / LDS /
// scratch use. The TurboQuant twins live under the host's _kvTqSet; the F16 siblings are the same kernel classes under the model's own fields.
void Vgpr(int bits)
{
    var vk = (DotLLM.Vulkan.VulkanTransformerModel)model;
    var factory = InPlaceFactory(bits);
    using (var c = factory(config, 256)) { }   // builds the twin set (and proves it is complete)
    const BindingFlags F = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
    object dev = vk.GetType().GetProperty("Device", F)!.GetValue(vk)!;
    var stat = dev.GetType().GetMethod("GetShaderStatisticsAmd", F)!;
    if (!(bool)dev.GetType().GetProperty("HasShaderInfoAmd", F)!.GetValue(dev)!) throw new InvalidOperationException("no VK_AMD_shader_info");
    Type pipeT = typeof(DotLLM.Vulkan.ComputePipeline);
    var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
    var found = new List<(string path, nint pipe)>();
    void Walk(object? o, string path, int depth)
    {
        if (o is null || depth > 5 || !seen.Add(o)) return;
        var t = o.GetType();
        if (t == pipeT) { found.Add((path, (nint)pipeT.GetProperty("Pipeline")!.GetValue(o)!)); return; }
        if (o is string || t.IsPrimitive || t.IsEnum || t.IsPointer) return;
        if (o is System.Collections.IEnumerable en) { int i = 0; foreach (var x in en) Walk(x, path + "[" + i++ + "]", depth + 1); return; }
        if (t.Assembly != pipeT.Assembly) return;
        if (t.Name.Contains("VulkanDevice") || t.Name.Contains("VulkanModule") || t == typeof(DotLLM.Vulkan.VulkanTransformerModel)) return;
        for (var bt = t; bt != null && bt != typeof(object); bt = bt.BaseType)
            foreach (var f in bt.GetFields(F | BindingFlags.DeclaredOnly))
                if (!f.FieldType.IsPointer) Walk(f.GetValue(o), path + "/" + t.Name + "." + f.Name, depth + 1);
    }
    object tqSet = vk.GetType().GetField("_kvTqSet", F)!.GetValue(vk)!;
    Walk(tqSet, "tq", 0);
    int nTq = found.Count;
    foreach (var f in vk.GetType().GetFields(F))
        if (f.Name != "_kvTqSet" && f.GetValue(vk) is { } v && v.GetType().Assembly == pipeT.Assembly && v.GetType().Name.Contains("Kernel")
            && (f.Name.Contains("ttention") || f.Name.Contains("Flash") || f.Name.Contains("Kv")))
            Walk(v, "ref:" + f.Name, 0);
    Log("VGPR table (columns: VGPR used, SGPR used, LDS used B, scratch B, VGPR available); tq = _kvTqSet twins, ref: = the model's own kernel fields");
    for (int i = 0; i < found.Count; i++)
    {
        object st = stat.Invoke(dev, [found[i].pipe, 32u])!;   // VkShaderStageFlags.Compute = 0x20
        var ru = st.GetType().GetField("resourceUsage", F)!.GetValue(st)!;
        uint G(object o, string n) { var v = o.GetType().GetField(n, F)!.GetValue(o); return v is UIntPtr up ? (uint)up : v is nuint nu ? (uint)nu : Convert.ToUInt32(v); }
        string r = $"{G(ru, "numUsedVgprs")}\t{G(ru, "numUsedSgprs")}\t{G(ru, "ldsUsageSizeInBytes")}\t{G(ru, "scratchMemUsageInBytes")}\t{G(st, "numAvailableVgprs")}";
        Append(new { kind = "vgpr", bits, group = i < nTq ? "tq" : "ref", path = found[i].path, vgpr = G(ru, "numUsedVgprs"), sgpr = G(ru, "numUsedSgprs"),
            lds = G(ru, "ldsUsageSizeInBytes"), scratch = G(ru, "scratchMemUsageInBytes"), vgprAvail = G(st, "numAvailableVgprs") });
        Log($"VGPR {(i < nTq ? "TQ " : "REF")} {found[i].path}\t{r}");
    }
}


// ───────────────────────── plumbing ─────────────────────────
double Med(double[] sorted) => sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
void Append(object row) => File.AppendAllText(outFile, JsonSerializer.Serialize(row) + "\n");
static void Log(string s) => Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {s}");
static Dictionary<string, string> ParseArgs(string[] argv)
{
    var d = new Dictionary<string, string>();
    for (int i = 0; i < argv.Length; i++)
        if (argv[i].StartsWith("--")) d[argv[i][2..]] = i + 1 < argv.Length && !argv[i + 1].StartsWith("--") ? argv[++i] : "true";
    return d;
}
