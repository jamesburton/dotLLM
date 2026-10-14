using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers.Bpe;

/// <summary>
/// Decode-path quality gate for decode-kernel changes (#991). Perplexity scores through PREFILL and cannot see a decode kernel, so this feeds a
/// fixed token set ONE TOKEN AT A TIME through the DECODE path (KV cache, seqLen == 1) and dumps the full-vocab logits of every step.
/// <list type="bullet">
/// <item><c>decodekl dump gguf cpu|vulkan out.f32</c>: sequences = the teacher-forced texts in <see cref="Texts"/>, steps back to back, no BOS.
/// Variants are chosen by env (<c>DOTLLM_VK_MOE_IQ_V2=0/1</c>) and, for the sensitive-control mutants, by the shader files in the probe's spv folder.</item>
/// <item><c>decodekl cmp ref.f32 test.f32 vocab [label]</c>: KL(ref||test) per step (median, mean, trimmed99, outliers, top-1 agreement).</item>
/// <item><c>decodekl margin gguf vocab a.f32 b.f32 c.f32 [A,B,C]</c>: at every step where the argmaxes are not all equal, top-3 of each with the top-2 gap.</item>
/// </list>
/// </summary>
internal static class DecodeKl
{
    public static readonly string[] Texts =
    {
        "Explain briefly how a transformer language model generates text, step by step.",
        "def fibonacci(n):",
        "Mixture-of-experts models route each token to a small subset of expert networks, which lets the total parameter count grow much faster than the compute per token. Explain in detail how the router is trained and why load balancing matters for throughput.",
        "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Write unit tests for the function above",
        "The French Revolution began in 1789 and transformed the political landscape of Europe. Key events included the storming of the Bastille, the Declaration of the Rights of Man, the Reign of Terror, and the rise of Napoleon Bonaparte, who eventually crowned himself emperor in 1804. Summarize the main causes.",
        // The two greedy continuations the V1 engine produced (the texts V1 and V2 disagreed on).
        "Explain briefly how a transformer language model generates text, step by step. (10 points)\nAnswer:\nA transformer language model generates text through the following steps:\n1. **Tokenization**: The input text is converted into tokens (subword units) using a tokenizer.\n2. **Embedding**: Each token is converted into a numerical vector representation.\n3. **Positional Encoding**: Position",
        "def fibonacci(n): # n-th fibonacci number\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\ndef fibonacci_iterative(n): # n-th fibonacci number\n    if n == 0",
        "The capital of France is Paris, and the capital of Germany is Berlin. Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a by-product. In 2023, the company reported revenue of 4.2 billion dollars, an increase of 12 percent over the previous year.",
    };

    public static int Run(string[] args)
    {
        switch (args[1])
        {
            case "dump": Dump(args[2], args[3], args[4]); return 0;
            case "cmp": Compare(args[2], args[3], int.Parse(args[4]), args.Length > 5 ? args[5] : "test"); return 0;
            case "margin": Margin(args[2], int.Parse(args[3]), args[4..7], args.Length > 7 ? args[7].Split(',') : new[] { "A", "B", "C" }); return 0;
            default: Console.WriteLine("decodekl dump|cmp|margin"); return 1;
        }
    }

    private static int[][] Tokenize(GgufFile gguf)
    {
        var tk = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return Texts.Select(t => tk.Encode(t).ToArray()).ToArray();
    }

    private static unsafe void Dump(string ggufPath, string backend, string outPath)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var seqs = Tokenize(gguf);
        Console.WriteLine($"sequences: {string.Join(",", seqs.Select(s => s.Length))} = {seqs.Sum(s => s.Length)} tokens; backend {backend}");
        IModel model;
        Func<int, IKvCache> kvFactory;
        DotLLM.Vulkan.VulkanDevice? dev = null;
        if (backend == "cpu")
        {
            model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0, 0));
            kvFactory = size => new SimpleKvCache(KvGeometry.FromConfig(config), size);
        }
        else
        {
            dev = DotLLM.Vulkan.VulkanDevice.Create();
            (model, kvFactory) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(dev, gguf, config, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        }
        int vocab = config.VocabSize;
        using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int step = 0;
        foreach (var ids in seqs)
        {
            using var cache = kvFactory(ids.Length + 8);
            for (int i = 0; i < ids.Length; i++)
            {
                using ITensor t = model.Forward(ids.AsSpan(i, 1), new[] { i }, backend == "cpu" ? -1 : 0, cache);
                fs.Write(new ReadOnlySpan<byte>((void*)t.DataPointer, vocab * 4));
                step++;
            }
            Console.WriteLine($"  seq done, {step} steps, {sw.Elapsed.TotalSeconds:F0} s");
        }
        (model as IDisposable)?.Dispose();
        dev?.Dispose();
    }

    private static float[][] ReadAll(string path, int vocab)
    {
        var bytes = File.ReadAllBytes(path);
        int n = bytes.Length / (vocab * 4);
        var all = new float[n][];
        for (int i = 0; i < n; i++) { all[i] = new float[vocab]; Buffer.BlockCopy(bytes, i * vocab * 4, all[i], 0, vocab * 4); }
        return all;
    }

    private static double[] LogSoftmax(float[] l)
    {
        double max = double.MinValue; foreach (var v in l) if (v > max) max = v;
        double z = 0; foreach (var v in l) z += Math.Exp(v - max);
        double lz = Math.Log(z) + max;
        var r = new double[l.Length]; for (int i = 0; i < l.Length; i++) r[i] = l[i] - lz;
        return r;
    }

    private static int Argmax(float[] l) { int b = 0; for (int i = 1; i < l.Length; i++) if (l[i] > l[b]) b = i; return b; }

    private static void Compare(string refPath, string testPath, int vocab, string label)
    {
        var a = ReadAll(refPath, vocab); var b = ReadAll(testPath, vocab);
        int n = Math.Min(a.Length, b.Length);
        var kl = new double[n]; int same = 0;
        for (int s = 0; s < n; s++)
        {
            var la = LogSoftmax(a[s]); var lb = LogSoftmax(b[s]);
            double k = 0; for (int i = 0; i < vocab; i++) { double p = Math.Exp(la[i]); if (p > 1e-12) k += p * (la[i] - lb[i]); }
            kl[s] = Math.Max(0, k);
            if (Argmax(a[s]) == Argmax(b[s])) same++;
        }
        var sorted = kl.OrderBy(x => x).ToArray();
        double median = sorted[n / 2];
        int keep = (int)Math.Ceiling(n * 0.99);
        double trimmed = sorted.Take(keep).Average();
        int outliers = kl.Count(x => x > 0.1);
        Console.WriteLine($"{label,-22} steps={n} meanKL={kl.Average():E3} medianKL={median:E3} trimmed99KL={trimmed:E3} maxKL={sorted[^1]:E3} outliers(>0.1)={outliers} top1={100.0 * same / n:F1}% ({same}/{n})");
    }

    private static void Margin(string ggufPath, int vocab, string[] files, string[] names)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var tk = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var seqs = Tokenize(gguf);
        var all = files.Select(f => ReadAll(f, vocab)).ToArray();
        int step = 0;
        for (int si = 0; si < seqs.Length; si++)
            for (int i = 0; i < seqs[si].Length; i++, step++)
            {
                var am = all.Select(x => Argmax(x[step])).ToArray();
                if (am.Distinct().Count() == 1) continue;
                Console.WriteLine($"seq {si} pos {i} (input token '{Show(tk, seqs[si][i])}', teacher-forced next '{(i + 1 < seqs[si].Length ? Show(tk, seqs[si][i + 1]) : "-")}'):");
                for (int k = 0; k < all.Length; k++)
                {
                    var l = all[k][step];
                    var top = Enumerable.Range(0, vocab).OrderByDescending(j => l[j]).Take(3).ToArray();
                    Console.WriteLine($"   {names[k],-4} top1 '{Show(tk, top[0])}' {l[top[0]]:F3} | top2 '{Show(tk, top[1])}' {l[top[1]]:F3} | top3 '{Show(tk, top[2])}' {l[top[2]]:F3} | gap12 {l[top[0]] - l[top[1]]:F3}");
                }
                foreach (var p in am.Distinct())
                    Console.WriteLine($"   logit of '{Show(tk, p)}': " + string.Join("  ", Enumerable.Range(0, all.Length).Select(k => $"{names[k]} {all[k][step][p]:F3}")));
            }
    }

    private static string Show(object tk, int id)
    {
        try
        {
            var m = tk.GetType().GetMethod("Decode", new[] { typeof(ReadOnlySpan<int>) });
            if (m is not null) return ((string)m.Invoke(tk, new object[] { new int[] { id } })!).Replace("\n", "\\n");
        }
        catch { }
        return id.ToString();
    }
}
