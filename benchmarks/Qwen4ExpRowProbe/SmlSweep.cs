using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// Regression sweep for the small-row (2..4 rows) dispatch of any generic-path Vulkan model (#996). One process per model, switches flipped at runtime:
/// <c>smlsweep gguf label [reps=7] [ctx=512]</c>. Prints, per model:
/// <list type="bullet">
/// <item>architecture, model class, tensor-type histogram (which levers can possibly fire);</item>
/// <item>harness floor: 1-row run twice (bit-equal?) and a noise-injected 1-row copy (the instrument must see 1e-3 / 1e-2 logit noise);</item>
/// <item>teacher-forced chunk-N (2,3,4) logits vs the 1-row decode, NEW (defaults) and OLD (all switches 0): median / trimmed99 / outliers / top-1,
/// with the dispatch counters per arm (perturbation proof: OLD must zero them);</item>
/// <item>cost(N)/cost(1) wall time at depth ctx, arms rotated per rep (new vs old), and pp (prefill 512) new vs old.</item>
/// </list>
/// </summary>
internal static class SmlSweep
{
    private record struct Stats(double Mean, double Median, double Trim99, double Max, int Outliers, double Top1);

    public static void Run(string[] args)
    {
        string path = args[1], label = args[2];
        int reps = args.Length > 3 ? int.Parse(args[3]) : 7;
        int ctx = args.Length > 4 ? int.Parse(args[4]) : 512;
        string only = args.Length > 5 ? args[5] : "all";   // all | quality | cost
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var hist = gguf.Tensors.GroupBy(t => t.QuantizationType).OrderByDescending(g => g.Count()).Select(g => $"{g.Key}x{g.Count()}");
        Console.WriteLine($"[sweep:{label}] arch={cfg.Architecture} vocab={cfg.VocabSize} layers={cfg.NumLayers} hidden={cfg.HiddenSize} heads={cfg.NumAttentionHeads}/{cfg.NumKvHeads} hd={cfg.HeadDim} types: {string.Join(" ", hist)}");
        if (cfg.Architecture is Architecture.Qwen3MoeHybrid or Architecture.NemotronH or Architecture.NemotronHMoe or Architecture.Qwen3HybridDense or Architecture.Qwen4Exp or Architecture.Mamba3)
        {
            Console.WriteLine($"[sweep:{label}] RESULT off-generic-path: {cfg.Architecture} is built by its own Vulkan model class (VulkanModelLoader), which has no small-row switches");
            return;
        }
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        var seqs = DecodeKl.Texts.Select(t => tk.Encode(t).ToArray()).ToArray();
        string? longPath = Environment.GetEnvironmentVariable("DOTLLM_SWEEP_LONG");
        if (!string.IsNullOrEmpty(longPath))
        {
            // One long natural-text sequence: exercises sliding windows / sink-layer window clipping / split-KV depth, which the short texts (<= ~100 tokens) never reach.
            int cap = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_SWEEP_LONG_TOKENS"), out int lc) ? lc : 1000;
            seqs = seqs.Append(tk.Encode(File.ReadAllText(longPath)).Take(cap).ToArray()).ToArray();
        }
        int total = seqs.Sum(s => s.Length);
        using var dev = VulkanDevice.Create();
        var sw = Stopwatch.StartNew();
        var (imodel, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        Console.WriteLine($"[sweep:{label}] loaded in {sw.Elapsed.TotalSeconds:F0} s as {imodel.GetType().Name}; resident {dev.ResidentLiveBytes() / 1073741824.0:F2} GiB; {total} teacher-forced tokens");
        if (imodel is not VulkanTransformerModel model)
        {
            Console.WriteLine($"[sweep:{label}] RESULT off-generic-path: {imodel.GetType().Name} does not use the small-row levers (no switches)");
            (imodel as IDisposable)?.Dispose();
            return;
        }
        if (!model.TrySetAllRowLogitsLimit(4)) throw new InvalidOperationException("no all-row logits");
        int vocab = cfg.VocabSize;

        (int q6, int f32, int q8, int at, int r2) def = (model.SmallRowQ6KMaxRows, model.SmallRowF32MaxRows, model.SmallRowQ8MaxRows, model.SmallRowAttnMaxRows, model.SmallRowQ6KR2MinRows);
        void New() { model.SmallRowQ6KMaxRows = def.q6; model.SmallRowF32MaxRows = def.f32; model.SmallRowQ8MaxRows = def.q8; model.SmallRowAttnMaxRows = def.at; model.SmallRowQ6KR2MinRows = def.r2; }
        void Old() { model.SmallRowQ6KMaxRows = 0; model.SmallRowF32MaxRows = 0; model.SmallRowQ8MaxRows = 0; model.SmallRowAttnMaxRows = 0; model.SmallRowQ6KR2MinRows = 99; }
        Console.WriteLine($"[sweep:{label}] defaults q6={def.q6} f32={def.f32} q8={def.q8} attn={def.at} r2min={def.r2}");

        if (only is "all" or "quality" or "qa")
        {
            float[][] Collect(int chunk)
            {
                var all = new float[total][];
                int step = 0;
                foreach (var ids in seqs)
                {
                    using var cache = kvFactory(ids.Length + 8);
                    for (int i = 0; i < ids.Length; i += chunk)
                    {
                        int n = Math.Min(chunk, ids.Length - i);
                        using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                        if (t.Shape[0] != n) throw new InvalidOperationException($"expected {n} logits rows, got {t.Shape[0]}");
                        for (int r = 0; r < n; r++) all[step++] = ReadRow(t, r, vocab);
                    }
                }
                return all;
            }

            New();
            var ref1 = Collect(1);
            var ref1b = Collect(1);
            Console.WriteLine($"[sweep:{label}] floor 1-row vs 1-row repeat: {Cmp(ref1, ref1b, vocab)}  bit-equal steps {Enumerable.Range(0, total).Count(s => ref1[s].AsSpan().SequenceEqual(ref1b[s]))}/{total}");
            var rng = new Random(7);
            foreach (double sigma in new[] { 1e-3, 1e-2 })
            {
                var noisy = ref1.Select(r => r.Select(x => x + (float)(sigma * (rng.NextDouble() * 2 - 1))).ToArray()).ToArray();
                Console.WriteLine($"[sweep:{label}] control noise +-{sigma:E0} on 1-row: {Cmp(ref1, noisy, vocab)}");
            }
            // Old-path 1-row must equal new-path 1-row (the switches gate seqLen >= 2 only).
            Old();
            var old1 = Collect(1);
            Console.WriteLine($"[sweep:{label}] 1-row switches=0 vs on: {Cmp(ref1, old1, vocab)}  bit-equal steps {Enumerable.Range(0, total).Count(s => ref1[s].AsSpan().SequenceEqual(old1[s]))}/{total}");
            old1 = null;
            foreach (int chunk in new[] { 2, 3, 4 })
            {
                double[]? klNew = null, klOld = null;
                foreach (bool isNew in new[] { true, false })
                {
                    if (isNew) New(); else Old();
                    long a = model.SmallRowQ6KCount, b = model.SmallRowF32Count, c = model.SmallRowQ8Count, d = model.SmallRowAttnCount;
                    var got = Collect(chunk);
                    string cnt = $"dispatches q6k {model.SmallRowQ6KCount - a} f32 {model.SmallRowF32Count - b} q8 {model.SmallRowQ8Count - c} attn {model.SmallRowAttnCount - d}";
                    Console.WriteLine($"[sweep:{label}] chunk {chunk} {(isNew ? "NEW" : "OLD")}: {Cmp(ref1, got, vocab, out var klx)}  | {cnt}");
                    if (isNew) klNew = klx; else klOld = klx;
                }
                Console.WriteLine($"[sweep:{label}] chunk {chunk} PAIRED new-vs-old: {Paired(klNew!, klOld!)}");
            }
            New();
        }

        if (only == "det")
        {
            // Run-to-run / history determinism of the N-row path: the same config twice in a row, then NEW<->OLD interleaved, byte-compared.
            float[][] C(int chunk)
            {
                var all = new float[total][];
                int step = 0;
                foreach (var ids in seqs)
                {
                    using var cache = kvFactory(ids.Length + 8);
                    for (int i = 0; i < ids.Length; i += chunk)
                    {
                        int n = Math.Min(chunk, ids.Length - i);
                        using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                        for (int r = 0; r < n; r++) all[step++] = ReadRow(t, r, vocab);
                    }
                }
                return all;
            }
            int Same(float[][] x, float[][] y) => Enumerable.Range(0, total).Count(i => x[i].AsSpan().SequenceEqual(y[i]));
            New(); var r1 = C(1);
            Old(); var fa = C(2); var fb = C(2);
            Console.WriteLine($"[sweep:{label}] det after-1-row: first OLD chunk-2 vs second OLD chunk-2 bit-equal {Same(fa, fb)}/{total}");
            foreach (int chunk in new[] { 2, 3 })
            {
                Old(); var o1 = C(chunk); var o2 = C(chunk);
                New(); var n1 = C(chunk); var n2 = C(chunk);
                Old(); var o3 = C(chunk);
                var r1b = C(1);
                Console.WriteLine($"[sweep:{label}] det 1-row before any N-row vs after chunk {chunk} passes: bit-equal {Same(r1, r1b)}/{total}");
                Console.WriteLine($"[sweep:{label}] det chunk {chunk}: OLD,OLD bit-equal {Same(o1, o2)}/{total}; NEW,NEW {Same(n1, n2)}/{total}; OLD after NEW vs first OLD {Same(o1, o3)}/{total}");
            }
            // Row-count invariance: is the N-row output independent of N (per step, bit for bit)? Old and new.
            Old(); var oc2 = C(2); var oc3 = C(3); var oc4 = C(4);
            New(); var nc2 = C(2); var nc3 = C(3); var nc4 = C(4);
            Console.WriteLine($"[sweep:{label}] det row-count invariance: OLD c2==c3 {Same(oc2, oc3)}/{total}, c2==c4 {Same(oc2, oc4)}/{total}; NEW c2==c3 {Same(nc2, nc3)}/{total}, c2==c4 {Same(nc2, nc4)}/{total}");
            New();
        }

        if (only is "attr" or "qa")
        {
            // Lever attribution: single lever on / single lever off, chunk 2..4, KL vs the 1-row decode (same teacher-forced steps).
            float[][] Collect2(int chunk)
            {
                var all = new float[total][];
                int step = 0;
                foreach (var ids in seqs)
                {
                    using var cache = kvFactory(ids.Length + 8);
                    for (int i = 0; i < ids.Length; i += chunk)
                    {
                        int n = Math.Min(chunk, ids.Length - i);
                        using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                        for (int r = 0; r < n; r++) all[step++] = ReadRow(t, r, vocab);
                    }
                }
                return all;
            }
            void Set(int q6, int f32, int q8, int at, int r2) { model.SmallRowQ6KMaxRows = q6; model.SmallRowF32MaxRows = f32; model.SmallRowQ8MaxRows = q8; model.SmallRowAttnMaxRows = at; model.SmallRowQ6KR2MinRows = r2; }
            New(); var refA = Collect2(1);
            var cfgs = new (string name, int q6, int f32, int q8, int at, int r2)[]
            {
                ("old", 0, 0, 0, 0, 99), ("new", def.q6, def.f32, def.q8, def.at, def.r2),
                ("q6k-only", def.q6, 0, 0, 0, def.r2), ("f32-only", 0, def.f32, 0, 0, 99), ("q8-only", 0, 0, def.q8, 0, 99), ("attn-only", 0, 0, 0, def.at, 99),
                ("new-no-q6k", 0, def.f32, def.q8, def.at, 99), ("new-no-f32", def.q6, 0, def.q8, def.at, def.r2), ("new-no-q8", def.q6, def.f32, 0, def.at, def.r2), ("new-no-attn", def.q6, def.f32, def.q8, 0, def.r2),
                ("new-no-r2", def.q6, def.f32, def.q8, def.at, 99),
            };
            double[]? klOldA = null;
            var attrChunks = (Environment.GetEnvironmentVariable("DOTLLM_SWEEP_ATTR_CHUNKS") ?? "2,3,4").Split(',').Select(int.Parse).ToArray();
            foreach (int chunk in attrChunks)
                foreach (var c in cfgs)
                {
                    Set(c.q6, c.f32, c.q8, c.at, c.r2);
                    var got = Collect2(chunk);
                    Console.WriteLine($"[sweep:{label}] attr chunk {chunk} {c.name,-12}: {Cmp(refA, got, vocab, out var klx)}");
                    if (c.name == "old") klOldA = klx;
                    else if (klOldA is not null) Console.WriteLine($"[sweep:{label}] attr chunk {chunk} {c.name,-12}  paired vs old: {Paired(klx, klOldA)}");
                }
            New();
        }

        if (only is "all" or "cost")
        {
            // Pool of tokens for ctx + rows.
            var pool = new List<int>();
            for (int rep = 0; pool.Count < ctx + 4000; rep++) for (int i = 0; i < seqs.Length; i++) pool.AddRange(seqs[(i * 3 + rep) % seqs.Length]);
            using var kv = kvFactory(ctx + 64);
            var ctxIds = pool.Take(ctx).ToArray();
            for (int p = 0; p < ctx; p += 512)
            {
                int m = Math.Min(512, ctx - p);
                using var lg = model.Forward(ctxIds.AsSpan(p, m), Enumerable.Range(p, m).ToArray(), 0, kv);
            }
            double Fwd(int rows, int win)
            {
                kv.Rollback(ctx);
                var ids = pool.Skip(ctx + 7 * win % 2500).Take(rows).ToArray();
                var pos = Enumerable.Range(ctx, rows).ToArray();
                var t0 = Stopwatch.GetTimestamp();
                using var lg = model.Forward(ids, pos, 0, kv);
                return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            }
            var arms = new List<(string name, int rows, bool isNew)> { ("r1", 1, true) };
            for (int rows = 2; rows <= 4; rows++) { arms.Add(($"r{rows} new", rows, true)); arms.Add(($"r{rows} old", rows, false)); }
            var times = arms.ToDictionary(a => a.name, _ => new List<double>());
            foreach (var a in arms) { if (a.isNew) New(); else Old(); Fwd(a.rows, 900); Fwd(a.rows, 901); }
            for (int rep = 0; rep < reps; rep++)
                for (int j = 0; j < arms.Count; j++)
                {
                    var a = arms[(j + rep * 3) % arms.Count];
                    if (a.isNew) New(); else Old();
                    times[a.name].Add(Fwd(a.rows, rep * 13 + j));
                }
            New();
            double med1 = Med(times["r1"]);
            Console.WriteLine($"[sweep:{label}] cost ctx={ctx} reps={reps}: 1-row min/med/max {times["r1"].Min():F1}/{med1:F1}/{times["r1"].Max():F1} ms = {1000 / med1:F1} tok/s");
            foreach (var a in arms.Skip(1))
            {
                var t = times[a.name];
                var paired = t.Select((x, i) => x / times["r1"][i]).OrderBy(x => x).ToList();
                Console.WriteLine($"[sweep:{label}]   {a.name,-7} {t.Min():F1}/{Med(t):F1}/{t.Max():F1} ms  ratio(med) {Med(t) / med1:F2}  paired {paired[0]:F2}..{paired[^1]:F2} (med {paired[paired.Count / 2]:F2})");
            }
            // pp ~512: new vs old, alternating. (Structurally unaffected: 512 rows >> the 4-row gates.)
            var ppIds = pool.Take(512).ToArray();
            var ppPos = Enumerable.Range(0, 512).ToArray();
            double Pp()
            {
                kv.Rollback(0);
                var t0 = Stopwatch.GetTimestamp();
                using var lg = model.Forward(ppIds, ppPos, 0, kv);
                return 512 / Stopwatch.GetElapsedTime(t0).TotalSeconds;
            }
            New(); Pp(); Pp();
            var ppNew = new List<double>(); var ppOld = new List<double>();
            for (int r = 0; r < 4; r++) { New(); ppNew.Add(Pp()); Old(); ppOld.Add(Pp()); }
            New();
            Console.WriteLine($"[sweep:{label}] pp512 tok/s new {ppNew.Min():F0}-{ppNew.Max():F0} (med {Med(ppNew):F0}) old {ppOld.Min():F0}-{ppOld.Max():F0} (med {Med(ppOld):F0})");
        }
        Console.WriteLine($"[sweep:{label}] DONE");
        model.Dispose();
    }

    /// <summary>Paired per-step comparison of two KL series against the same 1-row reference: share of steps where a &lt; b, mean log-ratio with its standard error (negative = a closer to the 1-row decode).</summary>
    private static string Paired(double[] a, double[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        var lr = new double[n]; int better = 0, worse = 0;
        for (int i = 0; i < n; i++) { lr[i] = Math.Log((a[i] + 1e-9) / (b[i] + 1e-9)); if (a[i] < b[i]) better++; else if (a[i] > b[i]) worse++; }
        double m = lr.Average(), sd = Math.Sqrt(lr.Select(x => (x - m) * (x - m)).Sum() / (n - 1)), se = sd / Math.Sqrt(n);
        return $"closer in {better}/{n} steps, farther in {worse}; mean log(KLnew/KLold) = {m:+0.000;-0.000} +- {se:0.000} (z {m / se:+0.0;-0.0}; exp = x{Math.Exp(m):F2})";
    }

    private static double Med(List<double> l) { var s = l.OrderBy(x => x).ToList(); return s[s.Count / 2]; }

    private static unsafe float[] ReadRow(DotLLM.Core.Tensors.ITensor t, int row, int vocab)
    {
        var r = new float[vocab];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * vocab, vocab).CopyTo(r);
        return r;
    }

    private static string Cmp(float[][] a, float[][] b, int vocab) => Cmp(a, b, vocab, out _);

    private static string Cmp(float[][] a, float[][] b, int vocab, out double[] klOut)
    {
        int n = Math.Min(a.Length, b.Length);
        var kl = new double[n]; int[] same = new int[n];
        Parallel.For(0, n, s =>
        {
            var la = LogSoftmax(a[s]); var lb = LogSoftmax(b[s]);
            double k = 0; for (int i = 0; i < vocab; i++) { double p = Math.Exp(la[i]); if (p > 1e-12) k += p * (la[i] - lb[i]); }
            kl[s] = Math.Max(0, k);
            same[s] = Argmax(a[s]) == Argmax(b[s]) ? 1 : 0;
        });
        klOut = kl;
        var sorted = kl.OrderBy(x => x).ToArray();
        int keep = (int)Math.Ceiling(n * 0.99);
        var detail = new System.Text.StringBuilder();
        foreach (int s in Enumerable.Range(0, n).Where(i => kl[i] > 0.05).OrderByDescending(i => kl[i]).Take(3))
            detail.Append($" [step {s} KL {kl[s]:F3} refGap {Gap(a[s]):F2} testGap {Gap(b[s]):F2} flip {Argmax(a[s]) != Argmax(b[s])}]");
        return $"steps={n} meanKL={kl.Average():E2} medianKL={sorted[n / 2]:E2} trim99={sorted.Take(keep).Average():E2} maxKL={sorted[^1]:E2} outliers(>0.1)={kl.Count(x => x > 0.1)} top1={100.0 * same.Sum() / n:F1}%" + (detail.Length > 0 ? " | worst:" + detail : "");
    }

    private static double[] LogSoftmax(float[] l)
    {
        double max = double.MinValue; foreach (var v in l) if (v > max) max = v;
        double z = 0; foreach (var v in l) z += Math.Exp(v - max);
        double lz = Math.Log(z) + max;
        var r = new double[l.Length]; for (int i = 0; i < l.Length; i++) r[i] = l[i] - lz;
        return r;
    }

    private static float Gap(float[] l) { float m1 = float.MinValue, m2 = float.MinValue; foreach (var v in l) { if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v; } return m1 - m2; }

    private static int Argmax(float[] l) { int b = 0; for (int i = 1; i < l.Length; i++) if (l[i] > l[b]) b = i; return b; }
}
