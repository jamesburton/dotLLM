using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;

/// <summary>
/// MTP verify-cost probe for MiMo-V2.6-Flash on the generic Vulkan model (#994). Measures the wall time of one forward over 1..4 rows at a
/// real decode depth (prefill <c>ctx</c> tokens, then roll the KV cache back to <c>ctx</c> before every timed forward, as a speculative round
/// does), with all-row logits enabled so the N-row lm_head is part of the cost. Arms are interleaved round-robin inside every rep.
/// <list type="bullet">
/// <item><c>mimofwd gguf time [reps=7] [ctx=512]</c>: timings (default MMVQ arm vs grouped coopmat arm), counters, then a headroom canary.</item>
/// <item><c>mimofwd gguf attrib [ctx=512]</c>: run with DOTLLM_VULKAN_DECODE_PROFILE=1 DOTLLM_VULKAN_DECODE_PROFILE_GPU=1 DOTLLM_VULKAN_PREFILL_PROFILE=1;
/// 64 one-row forwards (decode profile report) and 2 forwards each of 2/3/4 rows (prefill profile, stage-split submits).</item>
/// </list>
/// Run under the GPU lock. Nothing here is shipped.
/// </summary>
internal static class MimoFwd
{
    public static void Run(string[] args)
    {
        string path = args[1];
        string mode = args[2];
        int reps = (mode == "time" || mode == "sr") && args.Length > 3 ? int.Parse(args[3]) : 7;
        int ctx = args.Length > (mode is "time" or "sr" ? 4 : 3) ? int.Parse(args[mode is "time" or "sr" ? 4 : 3]) : 512;
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tk = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var pool = new List<int>();
        var texts = DecodeKl.Texts;
        for (int rep = 0; pool.Count < ctx + 3000; rep++)
            for (int i = 0; i < texts.Length; i++)
                pool.AddRange(tk.Encode(texts[(i * 3 + rep) % texts.Length]));
        int vocab = cfg.VocabSize;
        using var dev = VulkanDevice.Create();
        var sw = Stopwatch.StartNew();
        var (imodel, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        var model = (VulkanTransformerModel)imodel;
        Console.WriteLine($"[mimofwd] loaded in {sw.Elapsed.TotalSeconds:F0} s, resident live {dev.ResidentLiveBytes() / 1073741824.0:F2} GiB, pool {pool.Count} tokens, vocab {vocab}");
        if (mode != "attrib") model.TrySetAllRowLogitsLimit(4);   // attrib: the all-row head returns before the prefill profiler prints, so profile last-row-only forwards
        using var kv = (VulkanKvCache)kvFactory(ctx + 64);
        var ctxIds = pool.Take(ctx).ToArray();
        for (int p = 0; p < ctx; p += 512)
        {
            int m = Math.Min(512, ctx - p);
            using var lg = model.Forward(ctxIds.AsSpan(p, m), Enumerable.Range(p, m).ToArray(), 0, kv);
        }
        Console.WriteLine($"[mimofwd] prefilled {ctx} tokens, kv length {kv.CurrentLength}");

        double Fwd(int rows, int win, bool checkShape = false)
        {
            kv.Rollback(ctx);
            int start = ctx + 7 * win % 2500;
            var ids = pool.Skip(start).Take(rows).ToArray();
            var pos = Enumerable.Range(ctx, rows).ToArray();
            var t0 = Stopwatch.GetTimestamp();
            using var lg = model.Forward(ids, pos, 0, kv);
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            if (checkShape && lg.Shape[0] != rows) throw new Exception($"logits rows {lg.Shape[0]} != {rows}");
            return ms;
        }

        if (mode == "sr") { SmallRow(model, Fwd, reps, ctx); return; }

        if (mode == "attrib")
        {
            for (int i = 0; i < 64; i++) Fwd(1, i);
            for (int rows = 2; rows <= 4; rows++) { Fwd(rows, 100 + rows); Console.Error.WriteLine($"[mimofwd] ===== rows={rows} (profile above is the following forward) ====="); Fwd(rows, 200 + rows); }
            return;
        }

        // time mode: arms = (rows, mmvqMaxRows). MaxRows 4 = shipped (indexed MMVQ at 2..4 rows); 1 = grouped coopmat from 2 rows.
        var arms = new List<(string name, int rows, int maxRows)>
        {
            ("r1", 1, 4), ("r2 mmvq", 2, 4), ("r3 mmvq", 3, 4), ("r4 mmvq", 4, 4),
            ("r2 grouped", 2, 1), ("r3 grouped", 3, 1), ("r4 grouped", 4, 1),
        };
        var times = arms.ToDictionary(a => a.name, _ => new List<double>());
        var counts = new Dictionary<string, (long mmvq, int grouped, long scalar)>();
        // warm every arm twice
        foreach (var a in arms) { model.GenericMoeMmvqMaxRows = a.maxRows; Fwd(a.rows, 900, checkShape: true); Fwd(a.rows, 901); }
        for (int rep = 0; rep < reps; rep++)
        {
            // rotate the arm order each rep so no arm always follows the same neighbour (clock ramp / cache residue)
            for (int j = 0; j < arms.Count; j++)
            {
                var a = arms[(j + rep) % arms.Count];
                model.GenericMoeMmvqMaxRows = a.maxRows;
                long m0 = model.GenericMoeMmvqLayerCount; int g0 = model.GroupedMoeDispatchCount; long s0 = model.GenericMoeScalarDecodeLayerCount;
                times[a.name].Add(Fwd(a.rows, rep * 13 + j));
                if (rep == 0) counts[a.name] = (model.GenericMoeMmvqLayerCount - m0, model.GroupedMoeDispatchCount - g0, model.GenericMoeScalarDecodeLayerCount - s0);
            }
        }
        model.GenericMoeMmvqMaxRows = 4;
        Console.WriteLine($"[mimofwd] ctx={ctx} reps={reps} (min / median / max ms; per-forward wall incl. all-row logits download)");
        double med1 = Median(times["r1"]), min1 = times["r1"].Min();
        foreach (var a in arms)
        {
            var t = times[a.name]; var c = counts[a.name];
            Console.WriteLine($"  {a.name,-11} {t.Min():F1} / {Median(t):F1} / {t.Max():F1}   ratio vs 1-row: min {t.Min() / min1:F2}, median {Median(t) / med1:F2}   MoE layers mmvq/grouped/scalar {c.mmvq}/{c.grouped}/{c.scalar}");
        }
        Console.WriteLine($"  1-row decode equivalent: {1000.0 / med1:F1} tok/s (median), {1000.0 / min1:F1} (best)");
        foreach (var rows in new[] { 2, 3, 4 })
        {
            var rs = Enumerable.Range(0, reps).Select(i => times[$"r{rows} mmvq"][i] / times["r1"][i]).OrderBy(x => x).ToList();
            Console.WriteLine($"  paired (same rep) ratio r{rows}/r1 mmvq: {rs[0]:F2} .. {rs[^1]:F2}, median {rs[rs.Count / 2]:F2}");
        }

        // Headroom canary: a second device in this process, allocate+touch 1 GiB chunks until failure (#905). Capped to 20 GiB.
        Console.WriteLine($"[mimofwd] resident live (model device) {dev.ResidentLiveBytes() / 1073741824.0:F2} GiB, heap0 fallback {dev.FallbackBytesOnHeap(0) / 1073741824.0:F2} GiB, heap1 {dev.FallbackBytesOnHeap(1) / 1073741824.0:F2} GiB, others {(dev.ReadOtherProcessPressure()?.OtherBytes ?? 0) / 1073741824.0:F2} GiB");
        using var dev2 = VulkanDevice.Create();
        var bufs = new List<VulkanDevice.Buffer>();
        int ok = 0;
        try
        {
            var first = dev2.AllocateDeviceLocal(1L << 30); bufs.Add(first);
            for (int i = 0; i < 20; i++)
            {
                var b = dev2.AllocateDeviceLocal(1L << 30); bufs.Add(b);
                dev2.CopyBufferSynchronous(first, b, 1UL << 30);
                ok++;
            }
        }
        catch (Exception e) { Console.WriteLine($"[mimofwd] canary stopped: {e.GetType().Name}: {e.Message[..Math.Min(100, e.Message.Length)]}"); }
        Console.WriteLine($"[mimofwd] headroom canary: {ok} extra GiB allocated+touched beside the resident trunk (cap 20); model still usable: {Fwd(1, 5000) > 0}");
        foreach (var b in bufs) b.Dispose();
    }

    /// <summary>
    /// <c>mimofwd gguf sr [reps] [ctx]</c>: cost(N)/cost(1) of the 2..4 row forward under cumulative small-row lever configs, switched at runtime in ONE
    /// process (#996). Config = (Q6_K multi, F32 multi, Q8_0 multi, split-KV rows attention). Every rep runs every (config, rows) arm once, rotated.
    /// Also prints dispatch counters so the path is proven, not assumed.
    /// </summary>
    private static void SmallRow(VulkanTransformerModel model, Func<int, int, bool, double> fwd, int reps, int ctx)
    {
        var cfgs = new (string name, int q6, int f32, int q8, int attn)[]
        {
            ("old", 0, 0, 0, 0), ("+q6k", 4, 0, 0, 0), ("+f32", 4, 4, 0, 0), ("+q8", 4, 4, 4, 0), ("+attn", 4, 4, 4, 4),
            ("attn-only", 0, 0, 0, 4), ("f32-only", 0, 4, 0, 0),
        };
        void Apply((string name, int q6, int f32, int q8, int attn) c)
        { model.SmallRowQ6KMaxRows = c.q6; model.SmallRowF32MaxRows = c.f32; model.SmallRowQ8MaxRows = c.q8; model.SmallRowAttnMaxRows = c.attn; }
        var arms = new List<(string name, int rows, int ci)> { ("r1", 1, 4) };
        for (int ci = 0; ci < cfgs.Length; ci++) for (int rows = 2; rows <= 4; rows++) arms.Add(($"r{rows} {cfgs[ci].name}", rows, ci));
        var times = arms.ToDictionary(a => a.name, _ => new List<double>());
        var counts = new Dictionary<string, (long q6, long f32, long q8, long attn)>();
        foreach (var a in arms) { Apply(cfgs[a.ci]); fwd(a.rows, 900, true); fwd(a.rows, 901, false); }
        for (int rep = 0; rep < reps; rep++)
            for (int j = 0; j < arms.Count; j++)
            {
                var a = arms[(j + rep * 5) % arms.Count];
                Apply(cfgs[a.ci]);
                long q60 = model.SmallRowQ6KCount, f0 = model.SmallRowF32Count, q80 = model.SmallRowQ8Count, a0 = model.SmallRowAttnCount;
                times[a.name].Add(fwd(a.rows, rep * 13 + j, false));
                if (rep == 0) counts[a.name] = (model.SmallRowQ6KCount - q60, model.SmallRowF32Count - f0, model.SmallRowQ8Count - q80, model.SmallRowAttnCount - a0);
            }
        Apply(cfgs[4]);
        double med1 = Median(times["r1"]), min1 = times["r1"].Min();
        Console.WriteLine($"[sr] ctx={ctx} reps={reps}; 1-row {min1:F1} / {med1:F1} / {times["r1"].Max():F1} ms (min/median/max) = {1000 / med1:F1} tok/s median");
        foreach (var a in arms.Skip(1))
        {
            var t = times[a.name]; var c = counts[a.name];
            var ratios = t.Select((x, i) => x / times["r1"][i]).OrderBy(x => x).ToList();
            Console.WriteLine($"  {a.name,-16} {t.Min():F1} / {Median(t):F1} / {t.Max():F1} ms   ratio min {t.Min() / min1:F2} median {Median(t) / med1:F2}  paired {ratios[0]:F2}..{ratios[^1]:F2} (med {ratios[ratios.Count / 2]:F2})   dispatches/fwd q6k {c.q6} f32 {c.f32} q8 {c.q8} attn {c.attn}");
        }
    }

    private static double Median(List<double> l) { var s = l.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
}
