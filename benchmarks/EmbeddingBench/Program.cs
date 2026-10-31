// EmbeddingBench (#1369): CPU embedding survey harness. Emits one JSON object per line on stdout.
//
//   throughput <gguf> [--threads N] [--batches 1,8,32,128] [--seqs 64,256,512] [--cell-seconds 20]
//   latency    <gguf> [--threads N] [--seqs 16,64,256] [--iters 200]
//   profile    <gguf> [--threads N] [--seqs 64,256,512] [--iters 10]     (stage breakdown, BERT-class only)
//   tokenizer  <gguf> [--iters 3]                                          (tokenise docs/s, chars/s)
//   padding    <gguf> [--batches 8,32,128]                                 (padding waste on real RAG chunks)
//   dump       <gguf> <texts.json> <out.json> [--threads N] [--pooling mean|cls|last] [--dims N]
//   llama      <url> <gguf> [--batches ...] [--seqs ...] [--cell-seconds N]  (llama-server /v1/embeddings, token-id inputs)
//
// The engine runs one sequence per ForwardHidden call (no cross-sequence batching), so "batch B" here
// means B documents embedded back to back -- exactly what /v1/embeddings does today.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Tokenizers;
using DotLLM.Engine.Embeddings;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

namespace EmbeddingBench;

internal static class Program
{
    private static readonly JsonSerializerOptions J = new() { WriteIndented = false };
    private static void Emit(object o) => Console.Out.WriteLine(JsonSerializer.Serialize(o, J));

    public static int Main(string[] args)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        if (args.Length < 2) { Console.Error.WriteLine("usage: EmbeddingBench <mode> <gguf|url> ..."); return 1; }
        string mode = args[0];
        var opt = new Dictionary<string, string>();
        var pos = new List<string>();
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i].StartsWith("--")) { opt[args[i][2..]] = i + 1 < args.Length ? args[++i] : ""; }
            else pos.Add(args[i]);
        }
        int[] Ints(string k, string d) => opt.GetValueOrDefault(k, d).Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int threads = int.Parse(opt.GetValueOrDefault("threads", "0"));
        double cellSec = double.Parse(opt.GetValueOrDefault("cell-seconds", "20"), CultureInfo.InvariantCulture);

        if (mode == "http") return Http(pos[0], pos[1], Ints("clients", "1,8,32"), Ints("seqs", "64"), Ints("per-request", "1"), cellSec);
        if (mode == "llama") return Llama(pos[0], pos[1], Ints("batches", "1,8,32,128"), Ints("seqs", "64,256,512"), cellSec);

        if (mode == "tokens")
        {
            // tokens <gguf> <texts.json|fixture.json> <out.json>: token ids per text (no model load); diff against HF/llama.cpp.
            using var tg = GgufFile.Open(pos[0]);
            var tk = GgufTokenizerFactory.Load(tg.Metadata);
            using var doc = JsonDocument.Parse(File.ReadAllText(pos[1]));
            var arr = doc.RootElement.ValueKind == JsonValueKind.Object ? doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("text").GetString()!).ToArray()
                : doc.RootElement.EnumerateArray().Select(c => c.GetString()!).ToArray();
            File.WriteAllText(pos[2], JsonSerializer.Serialize(arr.Select(t => tk.Encode(t)).ToArray()));
            Emit(new { mode = "tokens", tokenizer = tk.GetType().Name, n = arr.Length });
            return 0;
        }

        string path = pos[0];
        BertEncoderModel.UseErfGelu = opt.GetValueOrDefault("erf-gelu", "0") == "1";   // exact-erf GELU (HF) instead of llama.cpp tanh
        BertEncoderModel.UseLegacyGelu = opt.GetValueOrDefault("legacy-gelu", "0") == "1";   // #1403 A/B arm
        BertEncoderModel.UseLegacyAttention = opt.GetValueOrDefault("legacy-attn", "0") == "1";   // #1402 A/B arm
        var (model, gguf, config) = opt.GetValueOrDefault("device", "cpu") == "vulkan"
            ? LoadVulkan(path)
            : ModelLoader.LoadFromGguf(path, new ThreadingConfig(threads, threads));
        using (gguf) using (model)
        {
            if (model is not IEmbeddingModel emb) { Console.Error.WriteLine($"{config.Architecture} has no IEmbeddingModel"); return 2; }
            var tok = GgufTokenizerFactory.Load(gguf.Metadata);
            string label = opt.GetValueOrDefault("label", Path.GetFileNameWithoutExtension(path));
            var pooling = EmbeddingPooler.Resolve(opt.TryGetValue("pooling", out var p) ? Parse(p) : null, emb.DeclaredPoolingType);
            switch (mode)
            {
                case "throughput": Throughput(label, emb, tok, config, pooling, Ints("batches", "1,8,32,128"), Ints("seqs", "64,256,512"), cellSec, threads); break;
                case "vkpack": VulkanPack(label, emb, tok, config, pooling, Ints("seqs", "16,32,64"), Ints("packs", "1,2,4,8,16,32"), int.Parse(opt.GetValueOrDefault("iters", "20")), opt.GetValueOrDefault("proxy", "0") == "1"); break;
                case "batch": BatchThroughput(label, emb, tok, config, Ints("batches", "1,8,32,128"), Ints("seqs", "64,256"), cellSec, threads); break;
                case "latency": Latency(label, emb, tok, config, pooling, Ints("seqs", "16,64,256"), int.Parse(opt.GetValueOrDefault("iters", "200")), threads); break;
                case "profile": Profile(label, emb, tok, config, pooling, Ints("seqs", "64,256,512"), int.Parse(opt.GetValueOrDefault("iters", "10"))); break;
                case "tokenizer": TokenizerBench(label, tok, int.Parse(opt.GetValueOrDefault("iters", "3"))); break;
                case "padding": Padding(label, tok, Ints("batches", "8,32,128")); break;
                case "e2e": E2e(label, emb, tok, config, pooling, pos[1], threads); break;
                case "dump": Suffix = opt.TryGetValue("suffix", out var sf) ? int.Parse(sf) : null; Dump(label, emb, tok, config, pooling, pos[1], pos[2], opt.TryGetValue("dims", out var d) ? int.Parse(d) : null); break;
                default: Console.Error.WriteLine("unknown mode"); return 1;
            }
        }
        return 0;
    }

    private static (IModel, GgufFile, ModelConfig) LoadVulkan(string path)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return (VulkanTransformerModel.LoadFromGguf(gguf, config, VulkanModelLoader.ResolveSpvDir()), gguf, config);
    }

    /// <summary>
    /// #1437: per-document cost of <c>ForwardHiddenForPooling</c> for pack-sized groups of seq-token documents (one call = whole group).
    /// With <paramref name="proxy"/> the group is instead ONE sequence of pack x seq tokens: an upper bound on a packed forward
    /// (full causal attention over the concatenation, no block-diagonal saving).
    /// </summary>
    private static void VulkanPack(string label, IEmbeddingModel emb, ITokenizer rawTok, ModelConfig cfg, PoolingType pooling,
        int[] seqs, int[] packs, int iters, bool proxy)
    {
        var tk = new Tk(rawTok);
        foreach (int seq in seqs)
            foreach (int pack in packs)
            {
                if (proxy && pack * seq > cfg.MaxSequenceLength) continue;
                var docs = proxy ? MakeDocs(tk, 8, pack * seq, 5) : MakeDocs(tk, pack, seq, 5);
                IReadOnlyList<int[]> group = proxy ? [docs[0]] : docs;
                int docsPerCall = proxy ? pack : pack;
                emb.ForwardHiddenForPooling(group, pooling, (_, _) => { }, 0);
                emb.ForwardHiddenForPooling(group, pooling, (_, _) => { }, 0);
                var ms = new List<double>();
                for (int i = 0; i < iters; i++)
                {
                    var t0 = Stopwatch.GetTimestamp();
                    emb.ForwardHiddenForPooling(group, pooling, (_, _) => { }, 0);
                    ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                }
                ms.Sort();
                Emit(new { mode = "vkpack", model = label, proxy, seq = docs[0].Length / (proxy ? pack : 1), pack, call_ms_min = ms[0], call_ms_med = ms[ms.Count / 2], call_ms_max = ms[^1],
                    per_doc_ms_med = ms[ms.Count / 2] / docsPerCall, docs_s_med = docsPerCall * 1000.0 / ms[ms.Count / 2] });
            }
    }

    /// <summary>Token id appended to every dump input (Qwen3-Embedding: the reference appends &lt;|endoftext|&gt; but the GGUF tokenizer does not).</summary>
    private static int? Suffix;

    private static PoolingType? Parse(string s) => s switch { "mean" => PoolingType.Mean, "cls" => PoolingType.Cls, "last" => PoolingType.Last, _ => null };

    // ---------------------------------------------------------------- corpus
    private static string RepoRoot()
    {
        string? d = AppContext.BaseDirectory;
        while (d is not null && !File.Exists(Path.Combine(d, "dotLLM.slnx"))) d = Path.GetDirectoryName(d);
        return d ?? Directory.GetCurrentDirectory();
    }

    /// <summary>RAG-style chunks: paragraphs of docs/*.md (real prose + code, mixed lengths).</summary>
    internal static List<string> Chunks()
    {
        var list = new List<string>();
        foreach (var f in Directory.GetFiles(Path.Combine(RepoRoot(), "docs"), "*.md").OrderBy(x => x, StringComparer.Ordinal))
            foreach (var para in File.ReadAllText(f).Replace("\r\n", "\n").Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
            {
                var t = para.Trim();
                if (t.Length >= 40 && !t.StartsWith('|') && !t.StartsWith("```")) list.Add(t);
            }
        return list;
    }

    /// <summary>Detects the special-token framing the tokenizer adds ([CLS]..[SEP] / none).</summary>
    private sealed class Tk
    {
        private readonly ITokenizer _t; private readonly int[] _pre = [], _suf = [];
        public Tk(ITokenizer t)
        {
            _t = t;
            var a = t.Encode("hello").ToArray(); var b = t.Encode("hello hello").ToArray();
            int bodyLen = b.Length - a.Length;
            int firstBody = Array.IndexOf(a, b[^1]);
            if (firstBody < 0 || bodyLen <= 0) return;
            _pre = a[..firstBody]; _suf = a[(firstBody + bodyLen)..];
        }
        public int FrameLen => _pre.Length + _suf.Length;
        public int[] EncodeBody(string text)
        {
            var ids = _t.Encode(text).ToArray();
            int s = _pre.Length, e = ids.Length - _suf.Length;
            return e > s ? ids[s..e] : ids;
        }
        public int[] Frame(ReadOnlySpan<int> body)
        {
            var r = new int[_pre.Length + body.Length + _suf.Length];
            _pre.CopyTo(r, 0); body.CopyTo(r.AsSpan(_pre.Length)); _suf.CopyTo(r, _pre.Length + body.Length);
            return r;
        }
    }

    private static int[] LongStream(Tk tk, int need)
    {
        var all = new List<int>(); var chunks = Chunks();
        for (int i = 0; all.Count < need; i++) all.AddRange(tk.EncodeBody(chunks[i % chunks.Count]));
        return all.ToArray();
    }

    private static List<int[]> MakeDocs(Tk tk, int count, int seq, int seed = 7)
    {
        int bodyLen = seq - tk.FrameLen;
        var stream = LongStream(tk, bodyLen * 4 + 8192);
        var rng = new Random(seed); var docs = new List<int[]>();
        for (int i = 0; i < count; i++)
            docs.Add(tk.Frame(stream.AsSpan(rng.Next(0, stream.Length - bodyLen), bodyLen)));
        return docs;
    }

    private static unsafe void EmbedOne(IEmbeddingModel emb, int[] ids, int hidden, PoolingType pooling, float[] vec, int[] posBuf)
    {
        for (int i = 0; i < ids.Length; i++) posBuf[i] = i;
        using var h = emb.ForwardHidden(ids, posBuf.AsSpan(0, ids.Length), 0);
        EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, ids.Length * hidden), ids.Length, hidden, pooling, vec);
        EmbeddingPooler.L2Normalize(vec);
    }

    private static double Pct(List<double> sorted, double q) => sorted[Math.Min(sorted.Count - 1, (int)Math.Ceiling(q * sorted.Count) - 1)];

    // ---------------------------------------------------------------- modes
    private static void Throughput(string label, IEmbeddingModel emb, ITokenizer rawTok, ModelConfig cfg, PoolingType pooling,
        int[] batches, int[] seqs, double cellSec, int threads)
    {
        var tk = new Tk(rawTok); int h = cfg.HiddenSize; var vec = new float[h]; var posBuf = new int[cfg.MaxSequenceLength];
        foreach (int seq in seqs)
        {
            if (seq > cfg.MaxSequenceLength) continue;
            foreach (var d in MakeDocs(tk, 2, seq)) EmbedOne(emb, d, h, pooling, vec, posBuf);
            foreach (int b in batches)
            {
                var docs = MakeDocs(tk, b, seq, 11 + b);
                var reps = new List<double>(); int done = 0; var sw = Stopwatch.StartNew();
                for (int rep = 0; rep < 3; rep++)
                {
                    var t0 = Stopwatch.GetTimestamp(); int n = 0;
                    foreach (var d in docs)
                    {
                        EmbedOne(emb, d, h, pooling, vec, posBuf); n++;
                        if (Stopwatch.GetElapsedTime(t0).TotalSeconds > cellSec) break;
                    }
                    reps.Add(n / Stopwatch.GetElapsedTime(t0).TotalSeconds); done = n;
                    if (sw.Elapsed.TotalSeconds > cellSec) break;
                }
                reps.Sort();
                int actualSeq = docs[0].Length;
                Emit(new { mode = "throughput", model = label, threads = threads == 0 ? Environment.ProcessorCount : threads, batch = b, seq = actualSeq, docsPerRep = done,
                    docs_s_min = reps[0], docs_s_med = reps[reps.Count / 2], docs_s_max = reps[^1], tok_s_med = reps[reps.Count / 2] * actualSeq, reps = reps.Count });
            }
        }
    }

    /// <summary>Docs/s through <see cref="IEmbeddingModel.ForwardHiddenBatch"/> (#1405): B documents handed over as one batch.</summary>
    private static void BatchThroughput(string label, IEmbeddingModel emb, ITokenizer rawTok, ModelConfig cfg,
        int[] batches, int[] seqs, double cellSec, int threads)
    {
        var tk = new Tk(rawTok);
        foreach (int seq in seqs)
        {
            if (seq > cfg.MaxSequenceLength) continue;
            emb.ForwardHiddenBatch(MakeDocs(tk, 8, seq), (_, _) => { }, 0);
            foreach (int b in batches)
            {
                var docs = MakeDocs(tk, b, seq, 11 + b);
                var reps = new List<double>(); var sw = Stopwatch.StartNew();
                for (int rep = 0; rep < 5; rep++)
                {
                    var t0 = Stopwatch.GetTimestamp();
                    emb.ForwardHiddenBatch(docs, (_, _) => { }, 0);
                    reps.Add(b / Stopwatch.GetElapsedTime(t0).TotalSeconds);
                    if (sw.Elapsed.TotalSeconds > cellSec && reps.Count >= 3) break;
                }
                reps.Sort();
                Emit(new { mode = "batch", model = label, threads = threads == 0 ? Environment.ProcessorCount : threads, batch = b, seq = docs[0].Length,
                    docs_s_min = reps[0], docs_s_med = reps[reps.Count / 2], docs_s_max = reps[^1], tok_s_med = reps[reps.Count / 2] * docs[0].Length, reps = reps.Count });
            }
        }
    }

    /// <summary>Concurrent HTTP clients against a running dotLLM server's /v1/embeddings, token-id inputs (#1405).</summary>
    private static int Http(string url, string gguf, int[] clients, int[] seqs, int[] perRequest, double cellSec)
    {
        var gf = GgufFile.Open(gguf); var tk = new Tk(GgufTokenizerFactory.Load(gf.Metadata)); var cfg = GgufModelConfigExtractor.Extract(gf.Metadata);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10), BaseAddress = new Uri(url) };
        string label = Path.GetFileNameWithoutExtension(gguf);
        foreach (int seq in seqs)
        foreach (int per in perRequest)
        foreach (int nc in clients)
        {
            var docs = MakeDocs(tk, Math.Max(64, nc * per), seq, 5).ToArray();
            async Task<int> Call(int[][] input)
            {
                string body = JsonSerializer.Serialize(new { input });
                using var r = await http.PostAsync("/v1/embeddings", new StringContent(body, Encoding.UTF8, "application/json"));
                r.EnsureSuccessStatusCode(); _ = (await r.Content.ReadAsStringAsync()).Length; return input.Length;
            }
            Call(docs[..per]).Wait();
            var reps = new List<double>();
            for (int rep = 0; rep < 3; rep++)
            {
                long done = 0; var t0 = Stopwatch.GetTimestamp(); var end = Stopwatch.StartNew();
                var lat = new System.Collections.Concurrent.ConcurrentBag<double>();
                var tasks = Enumerable.Range(0, nc).Select(c => Task.Run(async () =>
                {
                    int i = c * per;
                    while (end.Elapsed.TotalSeconds < cellSec)
                    {
                        var input = Enumerable.Range(0, per).Select(k => docs[(i + k) % docs.Length]).ToArray(); i += nc * per;
                        var s0 = Stopwatch.GetTimestamp(); int n = await Call(input); lat.Add(Stopwatch.GetElapsedTime(s0).TotalMilliseconds);
                        Interlocked.Add(ref done, n);
                    }
                })).ToArray();
                Task.WaitAll(tasks);
                double secs = Stopwatch.GetElapsedTime(t0).TotalSeconds; reps.Add(done / secs);
                var l = lat.OrderBy(x => x).ToArray();
                if (rep == 2) Emit(new { mode = "http", model = label, clients = nc, perRequest = per, seq = docs[0].Length, docs_s_med = reps.OrderBy(x => x).ElementAt(reps.Count / 2), docs_s_min = reps.Min(), docs_s_max = reps.Max(), p50_ms = l[l.Length / 2], p99_ms = l[(int)(l.Length * 0.99)] });
            }
        }
        return 0;
    }

    private static void Latency(string label, IEmbeddingModel emb, ITokenizer rawTok, ModelConfig cfg, PoolingType pooling,
        int[] seqs, int iters, int threads)
    {
        var tk = new Tk(rawTok); int h = cfg.HiddenSize; var vec = new float[h]; var posBuf = new int[cfg.MaxSequenceLength];
        foreach (int seq in seqs)
        {
            var docs = MakeDocs(tk, iters, seq, 3);
            for (int i = 0; i < 5; i++) EmbedOne(emb, docs[i], h, pooling, vec, posBuf);
            var ms = new List<double>();
            foreach (var d in docs) { var t0 = Stopwatch.GetTimestamp(); EmbedOne(emb, d, h, pooling, vec, posBuf); ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
            ms.Sort();
            Emit(new { mode = "latency", model = label, threads = threads == 0 ? Environment.ProcessorCount : threads, seq = docs[0].Length, iters,
                p50_ms = Pct(ms, .5), p90_ms = Pct(ms, .9), p99_ms = Pct(ms, .99), min_ms = ms[0], max_ms = ms[^1] });
        }
    }

    private static void Profile(string label, IEmbeddingModel emb, ITokenizer rawTok, ModelConfig cfg, PoolingType pooling, int[] seqs, int iters)
    {
        if (emb is not BertEncoderModel) { Console.Error.WriteLine("profile: BERT-class encoder only"); return; }
        var tk = new Tk(rawTok); int h = cfg.HiddenSize; var vec = new float[h]; var posBuf = new int[cfg.MaxSequenceLength];
        foreach (int seq in seqs)
        {
            var docs = MakeDocs(tk, iters, seq, 5);
            for (int i = 0; i < 2; i++) EmbedOne(emb, docs[i], h, pooling, vec, posBuf);
            var prof = new EncoderStageProfile(); BertEncoderModel.StageProfile = prof;
            long poolTicks = 0; long total = Stopwatch.GetTimestamp();
            foreach (var d in docs) poolTicks += ProfiledOne(emb, d, h, pooling, vec, posBuf);
            total = Stopwatch.GetTimestamp() - total; BertEncoderModel.StageProfile = null;
            double f = 1000.0 / Stopwatch.Frequency / iters;
            var stages = new Dictionary<string, double>();
            for (int s = 0; s < (int)EncoderStage.Count; s++) stages[((EncoderStage)s).ToString()] = prof.Ticks[s] * f;
            stages["PoolNorm"] = poolTicks * f;
            Emit(new { mode = "profile", model = label, seq = docs[0].Length, iters, total_ms_per_doc = total * f, stages_ms_per_doc = stages });
        }
    }

    private static unsafe long ProfiledOne(IEmbeddingModel emb, int[] d, int h, PoolingType pooling, float[] vec, int[] posBuf)
    {
        for (int i = 0; i < d.Length; i++) posBuf[i] = i;
        using var hid = emb.ForwardHidden(d, posBuf.AsSpan(0, d.Length), 0);
        long t0 = Stopwatch.GetTimestamp();
        EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)hid.DataPointer, d.Length * h), d.Length, h, pooling, vec);
        EmbeddingPooler.L2Normalize(vec);
        return Stopwatch.GetTimestamp() - t0;
    }

    private static void TokenizerBench(string label, ITokenizer tok, int iters)
    {
        var chunks = Chunks(); long chars = chunks.Sum(c => (long)c.Length);
        foreach (var c in chunks.Take(50)) tok.Encode(c);
        long toks = 0; var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) foreach (var c in chunks) toks += tok.Encode(c).Length;
        double s = sw.Elapsed.TotalSeconds;
        Emit(new { mode = "tokenizer", model = label, chunks = chunks.Count, docs_s = chunks.Count * iters / s, chars_s = chars * iters / s, tok_s = toks / s, avg_tokens = (double)toks / (chunks.Count * iters) });
    }

    private static void Padding(string label, ITokenizer tok, int[] batches)
    {
        var lens = Chunks().Select(c => tok.Encode(c).Length).Where(l => l <= 512).ToList();
        var rng = new Random(1);
        foreach (int b in batches)
        {
            double Waste(int[] arr)
            {
                long real = 0, padded = 0;
                for (int i = 0; i + b <= arr.Length; i += b)
                {
                    var g = arr.AsSpan(i, b); int mx = 0; long sum = 0;
                    foreach (var x in g) { mx = Math.Max(mx, x); sum += x; }
                    real += sum; padded += (long)mx * b;
                }
                return 1.0 - (double)real / padded;
            }
            var shuffled = lens.OrderBy(_ => rng.Next()).ToArray();
            var sorted = lens.OrderBy(x => x).ToArray();
            Emit(new { mode = "padding", model = label, batch = b, docs = lens.Count, mean_len = lens.Average(), p50 = sorted[sorted.Length / 2], max_len = sorted[^1],
                waste_arrival_order = Waste(shuffled), waste_length_sorted = Waste(sorted) });
        }
    }

    private static void Dump(string label, IEmbeddingModel emb, ITokenizer tok, ModelConfig cfg, PoolingType pooling, string textsPath, string outPath, int? dims)
    {
        var texts = JsonSerializer.Deserialize<string[]>(File.ReadAllText(textsPath))!;
        int h = cfg.HiddenSize; var posBuf = new int[cfg.MaxSequenceLength]; var res = new List<float[]>(); var ntok = new List<int>();
        foreach (var t in texts)
        {
            var ids = tok.Encode(t).ToArray(); if (Suffix is { } sx) ids = [.. ids, sx]; if (ids.Length > cfg.MaxSequenceLength) { int last = ids[^1]; ids = ids[..cfg.MaxSequenceLength]; ids[^1] = last; }
            var v = DumpOne(emb, ids, h, pooling, posBuf);
            if (dims is { } dd && dd < h) v = v[..dd];
            EmbeddingPooler.L2Normalize(v); res.Add(v); ntok.Add(ids.Length);
        }
        File.WriteAllText(outPath, JsonSerializer.Serialize(new { model = label, pooling = pooling.ToString(), tokens = ntok, vectors = res }));
        Emit(new { mode = "dump", model = label, n = texts.Length, pooling = pooling.ToString(), outPath });
    }

    /// <summary>Text-in/vector-out docs/s over a fixed text set (tokenise + embed + pool + normalise), 3 passes after a warm pass. Same set the ST / llama-server references time.</summary>
    private static void E2e(string label, IEmbeddingModel emb, ITokenizer tok, ModelConfig cfg, PoolingType pooling, string textsPath, int threads)
    {
        var texts = JsonSerializer.Deserialize<string[]>(File.ReadAllText(textsPath))!;
        int h = cfg.HiddenSize; var posBuf = new int[cfg.MaxSequenceLength]; var vec = new float[h];
        void Pass(ref long tokens)
        {
            foreach (var t in texts)
            {
                var ids = tok.Encode(t); if (ids.Length > cfg.MaxSequenceLength) { int last = ids[^1]; ids = ids[..cfg.MaxSequenceLength]; ids[^1] = last; }
                tokens += ids.Length; EmbedOne(emb, ids, h, pooling, vec, posBuf);
            }
        }
        long tk0 = 0; Pass(ref tk0);
        var reps = new List<double>(); long tokens = 0;
        for (int r = 0; r < 3; r++) { tokens = 0; var t0 = Stopwatch.GetTimestamp(); Pass(ref tokens); reps.Add(texts.Length / Stopwatch.GetElapsedTime(t0).TotalSeconds); }
        reps.Sort();
        Emit(new { mode = "e2e", model = label, threads = threads == 0 ? Environment.ProcessorCount : threads, texts = texts.Length, tokens, docs_s_min = reps[0], docs_s_med = reps[1], docs_s_max = reps[2], tok_s_med = reps[1] * tokens / texts.Length });
    }

    private static unsafe float[] DumpOne(IEmbeddingModel emb, int[] ids, int h, PoolingType pooling, int[] posBuf)
    {
        var v = new float[h];
        for (int i = 0; i < ids.Length; i++) posBuf[i] = i;
        using var hid = emb.ForwardHidden(ids, posBuf.AsSpan(0, ids.Length), 0);
        EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)hid.DataPointer, ids.Length * h), ids.Length, h, pooling, v);
        return v;
    }

    // ---------------------------------------------------------------- llama-server
    private static int Llama(string url, string gguf, int[] batches, int[] seqs, double cellSec)
    {
        // Tokenise with dotLLM's tokenizer and send token-id arrays so only the compute is compared.
        var gf = GgufFile.Open(gguf); var tk = new Tk(GgufTokenizerFactory.Load(gf.Metadata)); var cfg = GgufModelConfigExtractor.Extract(gf.Metadata);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10), BaseAddress = new Uri(url) };
        string label = Path.GetFileNameWithoutExtension(gguf);
        foreach (int seq in seqs)
        {
            if (seq > cfg.MaxSequenceLength) continue;
            foreach (int b in batches)
            {
                var docs = MakeDocs(tk, b, seq, 11 + b);
                string body = JsonSerializer.Serialize(new { input = docs, model = "x" });
                void Call()
                {
                    using var r = http.PostAsync("/v1/embeddings", new StringContent(body, Encoding.UTF8, "application/json")).Result;
                    r.EnsureSuccessStatusCode(); _ = r.Content.ReadAsStringAsync().Result.Length;
                }
                Call();
                var reps = new List<double>(); var sw = Stopwatch.StartNew();
                for (int rep = 0; rep < 3; rep++) { var t0 = Stopwatch.GetTimestamp(); Call(); reps.Add(b / Stopwatch.GetElapsedTime(t0).TotalSeconds); if (sw.Elapsed.TotalSeconds > cellSec) break; }
                reps.Sort();
                Emit(new { mode = "llama-server", model = label, batch = b, seq = docs[0].Length, docs_s_min = reps[0], docs_s_med = reps[reps.Count / 2], docs_s_max = reps[^1], tok_s_med = reps[reps.Count / 2] * docs[0].Length });
            }
        }
        return 0;
    }
}
