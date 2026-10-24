using System.Reflection;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// Row-count state-leak probe (#1016): <c>leakprobe gguf [steps]</c>, one model load. Each step is a teacher-forced pass over the
/// <see cref="DecodeKl.Texts"/> in chunks of N rows (all-row logits) or a named in-process reset (reflection on the model's private state).
/// Every pass is compared with the FIRST pass (fresh state) and with the previous same-size pass. Default steps: <c>2,4,2,inv,2,realloc,2</c>.
/// </summary>
internal static class LeakProbe
{
    public static unsafe void Run(string[] args)
    {
        string path = args[1];
        var steps = (args.Length > 2 ? args[2] : "2,4,2,inv,2,realloc,2").Split(',');
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        var seqs = DecodeKl.Texts.Select(t => tk.Encode(t).ToArray()).ToList();
        using var dev = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        bool allRows = Environment.GetEnvironmentVariable("DOTLLM_LEAK_NOALLROW") == "1" ? false : model.TrySetAllRowLogitsLimit(4);
        int vocab = cfg.VocabSize;
        Console.WriteLine($"[leak] {cfg.Architecture} as {model.GetType().Name} allRows={allRows} vocab={vocab}");
        var mt = model.GetType();
        object? Get(object o, string f) { for (var t = o.GetType(); t != null; t = t.BaseType) { var fi = t.GetField(f, BindingFlags.NonPublic | BindingFlags.Instance); if (fi != null) return fi.GetValue(o); } throw new Exception("no field " + f); }
        void Set(object o, string f, object v) { for (var t = o.GetType(); t != null; t = t.BaseType) { var fi = t.GetField(f, BindingFlags.NonPublic | BindingFlags.Instance); if (fi != null) { fi.SetValue(o, v); return; } } throw new Exception("no field " + f); }

        float[] Pass(int chunk)
        {
            var list = new List<float[]>();
            foreach (var ids in seqs)
            {
                using var cache = kvFactory(ids.Length + 8);
                for (int i = 0; i < ids.Length; i += chunk)
                {
                    int n = Math.Min(chunk, ids.Length - i);
                    using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                    int rows = (int)t.Shape[0];
                    var a = new float[(allRows ? n : 1) * vocab];
                    new ReadOnlySpan<float>((float*)t.DataPointer + (allRows ? 0 : (long)(rows - 1) * vocab), a.Length).CopyTo(a);
                    list.Add(a);
                }
            }
            return list.SelectMany(x => x).ToArray();
        }
        void Cmp(string tag, float[] a, float[] b)
        {
            int rows = a.Length / vocab, eq = 0, flips = 0; double sumKl = 0, maxKl = 0;
            for (int r = 0; r < rows; r++)
            {
                var ra = new ReadOnlySpan<float>(a, r * vocab, vocab); var rb = new ReadOnlySpan<float>(b, r * vocab, vocab);
                if (ra.SequenceEqual(rb)) { eq++; continue; }
                double ma = double.MinValue, mb = double.MinValue; int ia = 0, ib = 0;
                for (int j = 0; j < vocab; j++) { if (ra[j] > ma) { ma = ra[j]; ia = j; } if (rb[j] > mb) { mb = rb[j]; ib = j; } }
                if (ia != ib) flips++;
                double za = 0, zb = 0; for (int j = 0; j < vocab; j++) { za += Math.Exp(ra[j] - ma); zb += Math.Exp(rb[j] - mb); }
                double lza = Math.Log(za) + ma, lzb = Math.Log(zb) + mb, kl = 0;
                for (int j = 0; j < vocab; j++) { double la = ra[j] - lza; kl += Math.Exp(la) * (la - (rb[j] - lzb)); }
                sumKl += kl; maxKl = Math.Max(maxKl, kl);
            }
            Console.WriteLine($"[leak]   {tag}: rows {rows} bitequal {eq} meanKL {sumKl / rows:E2} maxKL {maxKl:E2} argmaxFlips {flips}");
        }

        var state = Get(model, "_state")!;
        var firstBySize = new Dictionary<int, float[]>(); var lastBySize = new Dictionary<int, float[]>();
        foreach (var s in steps)
        {
            if (int.TryParse(s, out int chunk))
            {
                var r = Pass(chunk);
                Console.WriteLine($"[leak] pass {chunk}");
                if (!firstBySize.TryGetValue(chunk, out var f0)) firstBySize[chunk] = r; else Cmp("vs first same-size", f0, r);
                if (lastBySize.TryGetValue(chunk, out var prev)) Cmp("vs prev same-size", prev, r);
                lastBySize[chunk] = r;
            }
            else
            {
                Console.WriteLine($"[leak] reset {s}");
                if (s.StartsWith("env:")) { var kv = s[4..].Split('=', 2); Environment.SetEnvironmentVariable(kv[0], kv[1].Length == 0 ? null : kv[1].Replace('+', ',')); continue; }
                switch (s)
                {
                    case "inv": mt.GetMethod("InvalidateKernelCaches", BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(model, null); break;
                    case "realloc": Set(state, "_capacitySeqLen", 0); break;   // next Forward re-allocates every state buffer at exactly its row count
                    case "difflogits": Set(model, "_diffusionLogitsCapacityRows", 0); break;
                    default: throw new Exception("unknown reset " + s);
                }
            }
        }
        (model as IDisposable)?.Dispose();
    }
}
