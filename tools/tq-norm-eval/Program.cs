// tq-norm-eval: measures what storing the TurboQuant per-head-vector norm in a narrower format costs in
// next-token KL, on a real model, CPU only (#1892 task 1).
//
//   tq-norm-eval --model <gguf> --tag <name> --ctx 2048 --bits 4 --formats F32Reference,F16,Bf16,Fp8E4M3,Fp8E5M2,Log8
//                --out <results.jsonl> [--work <dir>] [--corpus <file>] [--rows 64] [--greedy] [--gen 256] [--threads 0]
//
// KL mode (default): teacher-forces ctx tokens of the wikitext corpus through (a) an unquantised
// SimpleKvCache (the full-precision reference; SimpleKvCache has no F16 variant, so its fp32 store is the
// reference, which is the standing exception for baselines) and (b) the scratch-path TurboQuantKvCache with
// the norm round-tripped through TurboQuantNormCodec. KL is taken on the LAST --rows positions, whose
// attention reads the whole ctx-long quantised history. Reference and F32-norm logits are cached in --work
// so a slice that re-runs a format does not redo them. One JSON row is appended to --out per arm.
//
// Greedy mode (--greedy): prefill 256 corpus tokens, greedy-decode --gen tokens with each arm, compare with
// the F32Reference-norm arm of the same bit width.
using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

var a = ParseArgs(args);
string modelPath = a["model"];
string tag = a["tag"];
int ctx = int.Parse(a.GetValueOrDefault("ctx", "2048"));
int[] bitsList = a.GetValueOrDefault("bits", "4").Split(',').Select(int.Parse).ToArray();
// "Null" is a harness-only control arm: F32 norms jittered by up to 2^-12 relative (F16's worst error is
// 2^-11), i.e. a perturbation no larger than F16's. Its dKl / direct KL is the instrument's noise floor.
var formatNames = a.GetValueOrDefault("formats", "F32Reference,F16,Bf16,Fp8E4M3,Fp8E5M2,Log8,Null").Split(',').ToList();
string outFile = a["out"];
string work = a.GetValueOrDefault("work", Path.Combine(Path.GetTempPath(), "tq-norm-eval"));
string corpus = a.GetValueOrDefault("corpus",
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw"));
int rows = int.Parse(a.GetValueOrDefault("rows", "64"));
bool greedy = a.ContainsKey("greedy");
int gen = int.Parse(a.GetValueOrDefault("gen", "256"));
int threads = int.Parse(a.GetValueOrDefault("threads", "0"));
int offset = int.Parse(a.GetValueOrDefault("offset", "0"));   // token offset into the corpus (independent windows)
ulong seed = 0xC0FFEE_1234_5678UL;
Directory.CreateDirectory(work);
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);

using var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
int vocab = config.VocabSize;

int need = (greedy ? 256 : ctx) + 1 + offset;
string text = File.ReadAllText(corpus);
int[] all = tokenizer.Encode(text.Length > 400_000 ? text[..400_000] : text);
if (all.Length < need) throw new InvalidOperationException($"corpus yields {all.Length} tokens, need {need}.");
int[] tokens = all[offset..need];
Console.WriteLine($"model={tag} layers={config.NumLayers} kvHeads={config.NumKvHeads} hd={config.HeadDim} vocab={vocab} ctx={ctx} tokens={tokens.Length}");

using var model = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(threads));

if (greedy) RunGreedy(); else RunKl();
return;

void RunKl()
{
    // Inputs are tokens[0..ctx); row i of the last `rows` predicts tokens[ctx-rows+i+1].
    string refFile = Path.Combine(work, $"ref_{tag}_{ctx}_o{offset}.bin");
    float[][] reference = LoadOrRun(refFile, () =>
    {
        using var c = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, ctx);
        return RunTeacherForced(c);
    });
    foreach (int bits in bitsList)
    {
        string f32File = Path.Combine(work, $"f32norm_{tag}_{ctx}_o{offset}_{bits}.bin");
        float[][] f32 = LoadOrRun(f32File, () => RunTq(bits, TqNormFormat.F32Reference, out _, out _));
        double klRefF32 = MeanKl(reference, f32);
        foreach (var fname in formatNames)
        {
            bool isNull = fname == "Null";
            var fmt = isNull ? TqNormFormat.F32Reference : Enum.Parse<TqNormFormat>(fname);
            var sw = Stopwatch.StartNew();
            long sat = 0, under = 0;
            float[][] cand = fmt == TqNormFormat.F32Reference && !isNull && !a.ContainsKey("rerun-f32") ? f32 : RunTq(bits, fmt, out sat, out under, isNull ? MathF.Pow(2f, -12) : 0f);
            double secs = sw.Elapsed.TotalSeconds;
            double klRef = MeanKl(reference, cand);
            // Per-row paired difference (fmt - F32-norm), both against the reference, with its standard error.
            var diffs = new double[rows];
            for (int i = 0; i < rows; i++) diffs[i] = Kl(reference[i], cand[i]) - Kl(reference[i], f32[i]);
            double dMean = diffs.Average();
            double dSe = Math.Sqrt(diffs.Sum(d => (d - dMean) * (d - dMean)) / (rows - 1) / rows);
            double direct = MeanKl(f32, cand);
            Append(new
            {
                kind = "kl", model = tag, ctx, offset, bits, format = fname, rows,
                klTqVsRef = klRefF32,                 // tq (F32 norms) vs unquantised reference
                klFmtVsRef = klRef,                   // tq (this format) vs unquantised reference
                dKl = dMean, dKlSe = dSe,             // klFmtVsRef - klTqVsRef, paired per row
                dKlOverTq = dMean / klRefF32,
                klFmtVsF32Norm = direct,              // direct KL(F32-norm arm || fmt arm)
                klDirectOverTq = direct / klRefF32,
                top1AgreeRef = Top1(reference, cand),
                top1AgreeF32 = Top1(f32, cand),
                ppl = Ppl(cand), pplRef = Ppl(reference), pplTqF32 = Ppl(f32),
                saturated = sat, underflow = under, seconds = secs,
            });
            Console.WriteLine($"  bits={bits} {fname,-12} klTq={klRefF32:E3} dKl={dMean:E2}+-{dSe:E1} ({dMean / klRefF32:P1}) direct={direct / klRefF32:P1} sat={sat} under={under} {secs:F0}s");
        }
    }
}

void RunGreedy()
{
    int prompt = 256;
    foreach (int bits in bitsList)
    {
        string f32File = Path.Combine(work, $"greedy_f32norm_{tag}_o{offset}_{bits}.json");
        int[] baseline;
        if (File.Exists(f32File)) baseline = JsonSerializer.Deserialize<int[]>(File.ReadAllText(f32File))!;
        else { baseline = Greedy(bits, TqNormFormat.F32Reference, prompt, out _, out _); File.WriteAllText(f32File, JsonSerializer.Serialize(baseline)); }
        foreach (var fname in formatNames.Where(n => n != "Null"))
        {
            var fmt = Enum.Parse<TqNormFormat>(fname);
            var sw = Stopwatch.StartNew();
            long sat = 0, under = 0;
            int[] g = fmt == TqNormFormat.F32Reference ? baseline : Greedy(bits, fmt, prompt, out sat, out under);
            int same = 0, firstDiv = gen;
            for (int i = 0; i < gen; i++) { if (g[i] == baseline[i]) same++; else if (firstDiv == gen) firstDiv = i; }
            Append(new { kind = "greedy", model = tag, offset, bits, format = fmt.ToString(), gen, identical = same, firstDivergence = firstDiv, fullyIdentical = same == gen, saturated = sat, underflow = under, seconds = sw.Elapsed.TotalSeconds });
            Console.WriteLine($"  greedy bits={bits} {fmt,-12} identical={same}/{gen} firstDiv={(firstDiv == gen ? "none" : firstDiv)} {sw.Elapsed.TotalSeconds:F0}s");
        }
    }
}

int[] Greedy(int bits, TqNormFormat fmt, int prompt, out long sat, out long under)
{
    using var cache = new TurboQuantKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, prompt + gen + 1, bits, seed, false, fmt);
    var pt = tokens[..prompt];
    var pp = Enumerable.Range(0, prompt).ToArray();
    int next;
    using (var l = model.Forward(pt, pp, -1, cache, true)) next = ArgmaxRow(l, 0);
    var outTok = new int[gen];
    for (int i = 0; i < gen; i++)
    {
        outTok[i] = next;
        if (i == gen - 1) break;
        using var l = model.Forward(new[] { next }, new[] { prompt + i }, -1, cache);
        next = ArgmaxRow(l, 0);
    }
    sat = cache.NormSaturations; under = cache.NormUnderflows;
    return outTok;
}

unsafe int ArgmaxRow(ITensor logits, int row)
{
    float* p = (float*)logits.DataPointer + (long)row * vocab;
    int best = 0; float bv = p[0];
    for (int i = 1; i < vocab; i++) if (p[i] > bv) { bv = p[i]; best = i; }
    return best;
}

float[][] RunTq(int bits, TqNormFormat fmt, out long sat, out long under, float jitter = 0f)
{
    using var c = new TurboQuantKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, ctx, bits, seed, false, fmt, jitter);
    var r = RunTeacherForced(c);
    sat = c.NormSaturations; under = c.NormUnderflows;
    return r;
}

unsafe float[][] RunTeacherForced(IKvCache cache)
{
    const int chunk = 512;
    int head = ctx - rows;
    for (int s = 0; s < head; s += chunk)
    {
        int n = Math.Min(chunk, head - s);
        var tk = tokens.AsSpan(s, n);
        var ps = new int[n];
        for (int i = 0; i < n; i++) ps[i] = s + i;
        using var _ = model.Forward(tk, ps, -1, cache, true);
    }
    var tail = tokens.AsSpan(head, rows);
    var pos = new int[rows];
    for (int i = 0; i < rows; i++) pos[i] = head + i;
    using var logits = model.Forward(tail, pos, -1, cache);
    var res = new float[rows][];
    for (int i = 0; i < rows; i++)
    {
        res[i] = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)i * vocab, vocab).CopyTo(res[i]);
    }
    return res;
}

float[][] LoadOrRun(string file, Func<float[][]> run)
{
    if (File.Exists(file))
    {
        var bytes = File.ReadAllBytes(file);
        if (bytes.Length == (long)rows * vocab * 4)
        {
            var r = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                r[i] = new float[vocab];
                Buffer.BlockCopy(bytes, i * vocab * 4, r[i], 0, vocab * 4);
            }
            return r;
        }
    }
    var sw = Stopwatch.StartNew();
    var res = run();
    var buf = new byte[(long)rows * vocab * 4];
    for (int i = 0; i < rows; i++) Buffer.BlockCopy(res[i], 0, buf, i * vocab * 4, vocab * 4);
    File.WriteAllBytes(file, buf);
    Console.WriteLine($"  [{Path.GetFileName(file)}] {sw.Elapsed.TotalSeconds:F0}s");
    return res;
}

void Append(object row) => File.AppendAllText(outFile, JsonSerializer.Serialize(row) + "\n");

double MeanKl(float[][] r, float[][] c)
{
    double s = 0;
    for (int i = 0; i < r.Length; i++) s += Kl(r[i], c[i]);
    return s / r.Length;
}

static double Kl(float[] r, float[] c)
{
    double rMax = double.NegativeInfinity, cMax = double.NegativeInfinity;
    for (int i = 0; i < r.Length; i++) { if (r[i] > rMax) rMax = r[i]; if (c[i] > cMax) cMax = c[i]; }
    double rSum = 0, cSum = 0;
    for (int i = 0; i < r.Length; i++) { rSum += Math.Exp(r[i] - rMax); cSum += Math.Exp(c[i] - cMax); }
    double lr = Math.Log(rSum), lc = Math.Log(cSum), kl = 0;
    for (int i = 0; i < r.Length; i++)
    {
        double lpr = (r[i] - rMax) - lr;
        double pr = Math.Exp(lpr);
        if (pr <= 0) continue;
        kl += pr * (lpr - ((c[i] - cMax) - lc));
    }
    return kl;
}

double Top1(float[][] r, float[][] c)
{
    int agree = 0;
    for (int i = 0; i < r.Length; i++) if (Array.IndexOf(r[i], r[i].Max()) == Array.IndexOf(c[i], c[i].Max())) agree++;
    return (double)agree / r.Length;
}

double Ppl(float[][] r)
{
    double nll = 0;
    for (int i = 0; i < r.Length; i++)
    {
        int actual = tokens[ctx - rows + i + 1];
        double mx = r[i].Max(), sum = 0;
        for (int v = 0; v < vocab; v++) sum += Math.Exp(r[i][v] - mx);
        nll += -((r[i][actual] - mx) - Math.Log(sum));
    }
    return Math.Exp(nll / r.Length);
}

static Dictionary<string, string> ParseArgs(string[] argv)
{
    var d = new Dictionary<string, string>();
    for (int i = 0; i < argv.Length; i++)
    {
        if (!argv[i].StartsWith("--", StringComparison.Ordinal)) continue;
        string k = argv[i][2..];
        d[k] = i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal) ? argv[++i] : "true";
    }
    return d;
}
