// Flint KV-format probe (Q8_0 KV): next-token NLL and per-row KL between KV storage formats on real text, one process, one model load.
//
//   FlintKvProbe <flint.gguf> <corpus.txt> <ctxTokens> <evalRows> <arm,arm,...> [needle]
//     arms: f32 | f16 | q8_0 (the first arm is the reference of the KL columns). Load with DOTLLM_VK_KV=q8_0 so the Q8_0 and F16 twins exist.
//     The model prefills tokens [0, ctxTokens - evalRows) (last-row logits only), then scores the last evalRows positions in 64-row chunks with
//     every row's logits, so the scored rows sit at depth ~ctxTokens. NLL is the cross-entropy of the actual next token.
using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;

if (args.Length < 5) { Console.WriteLine("usage: FlintKvProbe <gguf> <corpus.txt> <ctxTokens> <evalRows> <arms>"); return 1; }
string gguf = args[0], corpus = args[1];
int ctx = int.Parse(args[2]), evalRows = int.Parse(args[3]);
string[] arms = args[4].Split(',');

using var file = GgufFile.Open(gguf);
var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
var tok = GgufTokenizerFactory.Load(file.Metadata);
int[] ids = tok.Encode(File.ReadAllText(corpus)).Take(ctx + 1).ToArray();
if (ids.Length < ctx + 1) { Console.WriteLine($"corpus has only {ids.Length} tokens, need {ctx + 1}"); return 1; }
Console.WriteLine($"[kvprobe] corpus {ids.Length} tokens; ctx {ctx}, scored rows {evalRows} at depth {ctx - evalRows}..{ctx}");

var sw = Stopwatch.StartNew();
using var model = VulkanTransformerModel.LoadFromGguf(file, cfg);
Console.WriteLine($"[kvprobe] loaded in {sw.Elapsed.TotalSeconds:F0} s; env DOTLLM_VK_KV={Environment.GetEnvironmentVariable("DOTLLM_VK_KV")}");
int vocab = cfg.VocabSize;

static unsafe float[] CopyRows(ITensor t, int rows, int vocab)
{
    var f = new float[rows * vocab];
    new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
    return f;
}

static void LogSoftmax(float[] logits, int row, int vocab, float[] dst)
{
    var span = logits.AsSpan(row * vocab, vocab);
    float max = float.NegativeInfinity;
    foreach (float x in span) if (x > max) max = x;
    double sum = 0;
    foreach (float x in span) sum += Math.Exp(x - max);
    float lse = max + (float)Math.Log(sum);
    for (int i = 0; i < vocab; i++) dst[i] = span[i] - lse;
}

float[][]? reference = null;   // [row][vocab] log-probs of the first arm
foreach (string arm in arms)
{
    model.KvStorageOverride = arm switch
    {
        "f32" => VulkanKvStorage.F32,
        "f16" => VulkanKvStorage.F16,
        "q8_0" => VulkanKvStorage.Q8_0,
        _ => throw new ArgumentException("arm " + arm),
    };
    sw.Restart();
    using var cache = model.CreateKvCache(ctx + 8);
    int prefillEnd = ctx - evalRows;
    model.TrySetAllRowLogitsLimit(1);
    for (int p = 0; p < prefillEnd; p += 512)
    {
        int n = Math.Min(512, prefillEnd - p);
        using var t = model.Forward(ids.AsSpan(p, n), Enumerable.Range(p, n).ToArray(), 0, cache);
    }
    double prefillSec = sw.Elapsed.TotalSeconds;
    model.TrySetAllRowLogitsLimit(65);
    var lp = new float[evalRows][];
    double nll = 0;
    for (int p = prefillEnd; p < ctx; p += 64)
    {
        int n = Math.Min(64, ctx - p);
        using var t = model.Forward(ids.AsSpan(p, n), Enumerable.Range(p, n).ToArray(), 0, cache);
        float[] logits = CopyRows(t, n, vocab);
        for (int r = 0; r < n; r++)
        {
            var row = new float[vocab];
            LogSoftmax(logits, r, vocab, row);
            lp[p - prefillEnd + r] = row;
            nll -= row[ids[p + r + 1]];
        }
    }
    Console.Write($"[kvprobe] arm {arm,-5} prefill {prefillSec:F0} s, cache {cache.AllocatedBytes / 1048576.0:F0} MiB | mean NLL {nll / evalRows:F4} (ppl {Math.Exp(nll / evalRows):F3})");
    if (reference is null) { reference = lp; Console.WriteLine(" | reference"); continue; }
    var kls = new double[evalRows]; int top1 = 0;
    for (int r = 0; r < evalRows; r++)
    {
        double kl = 0; int a0 = 0, a1 = 0;
        for (int i = 0; i < vocab; i++)
        {
            double pr = Math.Exp(reference[r][i]);
            if (pr > 1e-12) kl += pr * (reference[r][i] - lp[r][i]);
            if (reference[r][i] > reference[r][a0]) a0 = i;
            if (lp[r][i] > lp[r][a1]) a1 = i;
        }
        kls[r] = kl; if (a0 == a1) top1++;
    }
    var sorted = kls.OrderBy(x => x).ToArray();
    Console.WriteLine($" | KL(ref||{arm}) mean {kls.Average():F4} median {sorted[evalRows / 2]:F4} p90 {sorted[(int)(evalRows * 0.9)]:F4} max {sorted[^1]:F3} | top-1 agree {top1}/{evalRows}");
}
return 0;
