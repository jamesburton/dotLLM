// NemotronProbe: per-position logits of a GGUF model (CPU or Vulkan) for a fixed token-id stream, in the
// layout llama.cpp's `llama-perplexity --save-all-logits` scores (second half of each n_ctx window), so
// `nemprobe_kl.py` can compute per-position KL / top-1 agreement against llama.cpp on IDENTICAL ids (#375).
//
//   NemotronProbe tok    <gguf> <text-file>                                  -> dotLLM token ids (stdout)
//   NemotronProbe logits <gguf> <ids-file> <out-prefix> <ctx> <chunks> [cpu|vulkan]
//       window c = ids[c*ctx .. (c+1)*ctx), fresh sequence state; rows first..ctx-2 (first = ctx/2) are written to
//       <out-prefix>.logits.f32 (chunk-major) and <out-prefix>.meta.txt ("ctx chunks vocab first").
using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Core.Configuration;

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = new System.Text.UTF8Encoding(false);
        if (args.Length < 3) { Console.Error.WriteLine("usage: tok|logits ..."); return 2; }
        using var gguf = GgufFile.Open(args[1]);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (args[0] == "tok")
        {
            var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            Console.WriteLine(string.Join(' ', tok.EncodeRaw(File.ReadAllText(args[2]))));
            return 0;
        }
        if (args[0] == "render")
        {
            // render <gguf> <messages.json> [tools.json]: messages = [{role,content}], tools = [{name,description,parameters}]
            var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            var tmpl = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok, config.Architecture)
                ?? throw new InvalidOperationException("no chat template");
            using var md = System.Text.Json.JsonDocument.Parse(File.ReadAllText(args[2]));
            var msgs = md.RootElement.EnumerateArray().Select(e => new ChatMessage
            {
                Role = e.GetProperty("role").GetString()!,
                Content = e.GetProperty("content").GetString()!,
            }).ToList();
            ToolDefinition[]? tools = null;
            if (args.Length > 3)
            {
                using var td = System.Text.Json.JsonDocument.Parse(File.ReadAllText(args[3]));
                tools = td.RootElement.EnumerateArray().Select(e => new ToolDefinition(
                    e.GetProperty("name").GetString()!, e.GetProperty("description").GetString()!,
                    e.GetProperty("parameters").GetRawText())).ToArray();
            }
            Console.Write(tmpl.Apply(msgs, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools }));
            return 0;
        }
        if (args[0] == "consist")
        {
            // consist <gguf> <ids-file> <M> [cpu|vulkan]: logits of positions M..n-1 from ONE prefill of all ids vs
            // prefill(ids[0..M)) + one-token decode steps. Same maths, so any gap beyond reduction-order noise is a bug.
            int[] cid = File.ReadAllText(args[2]).Split(new[] { ' ', '\n', '\r', ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(int.Parse).ToArray();
            int mm = int.Parse(args[3]);
            string dev = args.Length > 4 ? args[4] : "cpu";
            IModel m3;
            int did;
            if (dev == "vulkan") { (m3, _) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config); did = 0; }
            else { m3 = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0)); did = -1; }
            using (m3)
            {
                int v = config.VocabSize, n = cid.Length;
                m3.TrySetAllRowLogitsLimit(n + 1);
                var all = new float[(long)(n - mm) * v];
                m3.ResetSequenceState();
                using var kvA = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), n + 8);
                using (var lg = m3.Forward(cid, Enumerable.Range(0, n).ToArray(), did, kvA))
                    unsafe { new ReadOnlySpan<float>((float*)lg.DataPointer + (long)mm * v, (n - mm) * v).CopyTo(all); }
                m3.ResetSequenceState();
                using var kvB = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), n + 8);
                using (var lg0 = m3.Forward(cid.AsSpan(0, mm), Enumerable.Range(0, mm).ToArray(), did, kvB)) { }
                double worst = 0, sumKl = 0; int flips = 0;
                for (int i = mm; i < n; i++)
                {
                    using var lg = m3.Forward(cid.AsSpan(i, 1), new[] { i }, did, kvB);
                    var a = new float[v]; var b = new float[v];
                    unsafe
                    {
                        new ReadOnlySpan<float>((float*)lg.DataPointer + (lg.ElementCount / v - 1) * v, v).CopyTo(b);
                    }
                    Array.Copy(all, (long)(i - mm) * v, a, 0, v);
                    double ma = a.Max(), la = ma + Math.Log(a.Sum(x => Math.Exp(x - ma)));
                    double mb = b.Max(), lb = mb + Math.Log(b.Sum(x => Math.Exp(x - mb)));
                    double kl = 0, md = 0;
                    for (int t = 0; t < v; t++)
                    {
                        double pa = a[t] - la, pb = b[t] - lb;
                        kl += Math.Exp(pa) * (pa - pb);
                        md = Math.Max(md, Math.Abs(pa - pb) * Math.Exp(pa) > 1e-4 ? Math.Abs(pa - pb) : 0);
                    }
                    int ia = Array.IndexOf(a, a.Max()), ib = Array.IndexOf(b, b.Max());
                    if (ia != ib) flips++;
                    worst = Math.Max(worst, md); sumKl += kl;
                    Console.WriteLine($"pos {i}: KL(prefill||decode)={kl:E2} maxAbsLogprobDiff(p>1e-4)={md:F4} top1 prefill={ia} decode={ib}");
                }
                Console.WriteLine($"mean KL={sumKl / (n - mm):E3} worst={worst:F4} top1 flips={flips}/{n - mm}");
            }
            return 0;
        }
        if (args[0] == "swa")
        {
            // swa <gguf> <ids-file> <promptLen> <chunk> [q8_0]: compact sliding-window KV (#1169) vs full-history KV on the
            // same stream: prefill ids[0..promptLen) in chunks, then teacher-force the rest one token at a time. Logits of
            // every chunk's last row and every decode step must be identical; prints KV bytes of both caches.
            int[] sid = File.ReadAllText(args[2]).Split(new[] { ' ', '\n', '\r', ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(int.Parse).ToArray();
            int promptLen = int.Parse(args[3]), chunk = int.Parse(args[4]);
            bool q8 = args.Length > 5 && args[5] == "q8_0";
            using var ms = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));
            int v = config.VocabSize, n = sid.Length;
            DotLLM.Core.Attention.IKvCache Make(bool compact)
            {
                var g = DotLLM.Core.Attention.KvGeometry.FromConfig(config, compact);
                return q8
                    ? new DotLLM.Engine.KvCache.QuantizedKvCache(g, n + 8, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0)
                    : new DotLLM.Engine.KvCache.SimpleKvCache(g, n + 8);
            }
            static long Bytes(DotLLM.Core.Attention.IKvCache c) => c switch
            {
                DotLLM.Engine.KvCache.SimpleKvCache s => s.AllocatedBytes,
                DotLLM.Engine.KvCache.QuantizedKvCache q => q.AllocatedBytes,
                _ => -1,
            };
            float[] Run(DotLLM.Core.Attention.IKvCache kv, out double seconds)
            {
                var sw = Stopwatch.StartNew();
                var rows = new List<float>();
                void Take(DotLLM.Core.Tensors.ITensor lg) { unsafe { rows.AddRange(new ReadOnlySpan<float>((float*)lg.DataPointer + (lg.ElementCount / v - 1) * v, v).ToArray()); } }
                for (int p = 0; p < promptLen; p += chunk)
                {
                    int len = Math.Min(chunk, promptLen - p);
                    using var lg = ms.Forward(sid.AsSpan(p, len), Enumerable.Range(p, len).ToArray(), -1, kv, lastTokenLogitsOnly: true);
                    Take(lg);
                }
                for (int i = promptLen; i < n; i++)
                {
                    using var lg = ms.Forward(sid.AsSpan(i, 1), new[] { i }, -1, kv);
                    Take(lg);
                }
                seconds = sw.Elapsed.TotalSeconds;
                return rows.ToArray();
            }
            using var kvFull = Make(false);
            using var kvCompact = Make(true);
            var a = Run(kvFull, out double ta);
            var b = Run(kvCompact, out double tb);
            int steps = a.Length / v, flips = 0, firstDiff = -1;
            double maxAbs = 0;
            for (int s = 0; s < steps; s++)
            {
                var ra = a.AsSpan(s * v, v); var rb = b.AsSpan(s * v, v);
                for (int t = 0; t < v; t++)
                {
                    double d = Math.Abs(ra[t] - rb[t]);
                    if (d > 0 && firstDiff < 0) firstDiff = s;
                    maxAbs = Math.Max(maxAbs, d);
                }
                if (ra.IndexOf(System.Linq.Enumerable.Max(ra.ToArray())) != rb.IndexOf(System.Linq.Enumerable.Max(rb.ToArray()))) flips++;
            }
            var wk = (DotLLM.Core.Attention.IWindowedKvCache)kvCompact;
            int baseL = Enumerable.Range(0, config.NumLayers).Max(wk.RowBaseOf);
            Console.WriteLine($"swa: n={n} prompt={promptLen} chunk={chunk} kv={(q8 ? "q8_0" : "f32")} steps={steps} identical={maxAbs == 0} "
                + $"maxAbsLogitDiff={maxAbs:E3} firstDiffStep={firstDiff} top1Flips={flips}");
            Console.WriteLine($"swa: KV bytes full={Bytes(kvFull) / 1048576.0:F1} MiB compact={Bytes(kvCompact) / 1048576.0:F1} MiB; "
                + $"max row base={baseL}; time full={ta:F0}s compact={tb:F0}s");
            return maxAbs == 0 ? 0 : 1;
        }
        if (args[0] == "forced")
        {
            // forced <gguf> <ids-file> <promptLen> [cpu|vulkan]: teacher-forced top-1 agreement over positions promptLen-1..n-2
            // (does our argmax equal the id the OTHER engine generated?). Mismatches print the logit gap to the forced token.
            int[] fid = File.ReadAllText(args[2]).Split(new[] { ' ', '\n', '\r', ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(int.Parse).ToArray();
            int pl = int.Parse(args[3]);
            string dev = args.Length > 4 ? args[4] : "cpu";
            IModel m4;
            int did;
            if (dev == "vulkan") { (m4, _) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config); did = 0; }
            else { m4 = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0)); did = -1; }
            using (m4)
            {
                int v = config.VocabSize, n = fid.Length;
                m4.TrySetAllRowLogitsLimit(n + 1);
                using var lg = m4.Forward(fid, Enumerable.Range(0, n).ToArray(), did);
                long rows = lg.ElementCount / v;
                if (rows < n) { Console.Error.WriteLine($"only {rows} rows"); return 3; }
                int agree = 0, total = 0;
                var gaps = new List<double>();
                unsafe
                {
                    for (int p = pl - 1; p <= n - 2; p++)
                    {
                        var row = new ReadOnlySpan<float>((float*)lg.DataPointer + (long)p * v, v);
                        int top = 0; for (int t = 1; t < v; t++) if (row[t] > row[top]) top = t;
                        int forcedTok = fid[p + 1];
                        total++;
                        if (top == forcedTok) agree++;
                        else
                        {
                            double gap = row[top] - row[forcedTok];
                            gaps.Add(gap);
                            Console.WriteLine($"  mismatch pos {p + 1}: ours={top} forced={forcedTok} logit gap={gap:F3}");
                        }
                    }
                }
                Console.WriteLine($"teacher-forced top-1: {agree}/{total} ({100.0 * agree / total:F1}%)  mismatch gaps: " +
                    (gaps.Count == 0 ? "none" : string.Join(", ", gaps.Select(g => g.ToString("F2")))));
            }
            return 0;
        }
        if (args[0] == "next")
        {
            // next <gguf> <ids-file> [cpu|vulkan]: top-8 next-token candidates (id, logit, logprob) after the whole id list.
            int[] nid = File.ReadAllText(args[2]).Split(new[] { ' ', '\n', '\r', ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(int.Parse).ToArray();
            string dev = args.Length > 3 ? args[3] : "cpu";
            IModel m2;
            int did;
            if (dev == "vulkan") { (m2, _) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config); did = 0; }
            else { m2 = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0)); did = -1; }
            using (m2)
            {
                int v = config.VocabSize;
                using var lg = m2.Forward(nid, Enumerable.Range(0, nid.Length).ToArray(), did);
                long rows = lg.ElementCount / v;
                var row = new float[v];
                unsafe { new ReadOnlySpan<float>((float*)lg.DataPointer + (rows - 1) * v, v).CopyTo(row); }
                double mx = row.Max(), lse = mx + Math.Log(row.Sum(x => Math.Exp(x - mx)));
                foreach (int i in Enumerable.Range(0, v).OrderByDescending(i => row[i]).Take(8))
                    Console.WriteLine($"{i}\t{row[i]:F4}\t{row[i] - lse:F4}");
            }
            return 0;
        }
        if (args[0] != "logits" || args.Length < 6) { Console.Error.WriteLine("usage: logits <gguf> <ids> <prefix> <ctx> <chunks> [device]"); return 2; }

        // <ids> and <prefix> may be comma-separated lists of equal length: one model load, many corpora.
        string[] idFiles = args[2].Split(',');
        string[] prefixes = args[3].Split(',');
        if (idFiles.Length != prefixes.Length) { Console.Error.WriteLine("ids and prefix lists differ in length"); return 2; }
        int ctx = int.Parse(args[4]);
        int chunks = int.Parse(args[5]);
        string device = args.Length > 6 ? args[6] : "cpu";
        var allIds = idFiles.Select(f => File.ReadAllText(f).Split(new[] { ' ', '\n', '\r', ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(int.Parse).ToArray()).ToArray();
        foreach (var a in allIds)
            if (a.Length < ctx * chunks) { Console.Error.WriteLine($"need {ctx * chunks} ids, have {a.Length}"); return 2; }

        IModel model;
        int deviceId;
        if (device == "vulkan")
        {
            (model, _) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config);
            deviceId = 0;
        }
        else
        {
            model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));
            deviceId = -1;
        }
        using (model)
        {
            int vocab = config.VocabSize;
            int first = ctx / 2;
            model.TrySetAllRowLogitsLimit(ctx + 1);
            var positions = Enumerable.Range(0, ctx).ToArray();
            var rowBuf = new byte[vocab * 4];
            for (int corpus = 0; corpus < allIds.Length; corpus++)
            {
            int[] ids = allIds[corpus];
            string prefix = prefixes[corpus];
            using var fs = File.Create(prefix + ".logits.f32");
            for (int c = 0; c < chunks; c++)
            {
                model.ResetSequenceState();
                var sw = Stopwatch.StartNew();
                var chunk = new ReadOnlySpan<int>(ids, c * ctx, ctx);
                using var logits = model.Forward(chunk, positions, deviceId);
                long rows = logits.ElementCount / vocab;
                if (rows < ctx) { Console.Error.WriteLine($"model returned {rows} rows (< {ctx}); all-row logits unavailable"); return 3; }
                unsafe
                {
                    float* p = (float*)logits.DataPointer;
                    for (int pos = first; pos <= ctx - 2; pos++)
                    {
                        System.Buffer.BlockCopy(new ReadOnlySpan<float>(p + (long)pos * vocab, vocab).ToArray(), 0, rowBuf, 0, rowBuf.Length);
                        fs.Write(rowBuf, 0, rowBuf.Length);
                    }
                }
                Console.Error.WriteLine($"chunk {c}: {sw.Elapsed.TotalSeconds:F1} s ({ctx / sw.Elapsed.TotalSeconds:F1} tok/s)");
            }
            File.WriteAllText(prefix + ".meta.txt", $"{ctx} {chunks} {vocab} {first}\n");
            }
        }
        return 0;
    }
}
