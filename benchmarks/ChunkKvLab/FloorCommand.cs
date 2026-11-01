using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.Core.Attention;

namespace ChunkKvLab;

/// <summary>Options of the <c>floor</c> command.</summary>
internal sealed record FloorOptions(
    string ModelPath, string[] Names, string[] CorpusDirs, string[] Hashes, int[] Contexts, string KeptDir, string Mode, int ContextTokens,
    int RefBlock, int[] Blocks, int Threads, string Label, string OutDir);

/// <summary>
/// Phase G Task 2 (Step 0): the numerical floor every later Vulkan parity result is judged against. Loads the model on the CPU and on
/// Vulkan in one process and, for each context (E4 contexts: <see cref="E4Command.SelectPlans"/>), compares full prefills:
/// <list type="bullet">
/// <item><b>(a)</b> <c>a_cpu_vs_vk{ref}</c>: CPU full prefill (reference; CPU kernels are batch-invariant) vs Vulkan full prefill in
/// <c>RefBlock</c>-row blocks;</item>
/// <item><b>(b)</b> <c>b_vk{ref}_vs_vk{n}</c>: Vulkan in <c>RefBlock</c>-row blocks (reference) vs Vulkan in <c>n</c>-row blocks, for each
/// <c>--blocks</c> entry;</item>
/// <item><b>(c)</b> <c>c_vk{ref}_rerun</c>: Vulkan <c>RefBlock</c> vs the same prefill again into a fresh cache (determinism).</item>
/// </list>
/// Per pair: per-layer K and V relative L2 of every context row (‖ΔX_row‖ / ‖X_ref,row‖; <see cref="KvDeviation"/>), pooled over all rows
/// of all contexts (mean, nearest-rank p99, max); and per task the teacher-forced logits of the E4 evaluation block (query prefix, then
/// 32 continuation rows; the same shapes on both sides) compared row by row: KL(ref ‖ other), top-1 agreement, masked-call NLL and
/// exact match. Outputs are written as <c>*.partial</c> and renamed only when every context finished, so a crash never leaves a
/// complete-looking CSV.
/// </summary>
internal static class FloorCommand
{
    public static int Run(FloorOptions o)
    {
        var total = Stopwatch.StartNew();
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        Directory.CreateDirectory(o.OutDir);
        if (o.Names.Length != o.CorpusDirs.Length || o.Names.Length != o.Hashes.Length || o.Names.Length != o.Contexts.Length)
            throw new ArgumentException("--names, --corpus-dirs, --hashes and --contexts must have one entry per corpus.");

        using LabModel vk = LabModel.Load(o.ModelPath, o.Threads, ChunkBackendTags.Vulkan);
        Console.WriteLine($"vulkan device: {vk.DeviceName}");
        using LabModel cpu = LabModel.Load(o.ModelPath, o.Threads, ChunkBackendTags.Cpu);
        int vocab = vk.Config.VocabSize;
        string r = o.RefBlock.ToString(ci);
        string pairA = $"a_cpu_vs_vk{r}", pairC = $"c_vk{r}_rerun";
        string PairB(int n) => $"b_vk{r}_vs_vk{n}";

        string Out(string stem) => Path.Combine(o.OutDir, $"{stem}-{o.Label}.csv");
        var partials = new List<string>();
        StreamWriter Open(string stem, string header)
        {
            string p = Out(stem) + ".partial";
            partials.Add(p);
            var w = new StreamWriter(p) { AutoFlush = true };
            w.WriteLine(header);
            return w;
        }
        using var ctxCsv = Open("contexts", "size,corpus,plan,context_id,context_tokens,pair,layer,kind,n,rel_l2_mean,rel_l2_p99,rel_l2_max");
        using var taskCsv = Open("tasks", "size,corpus,plan,context_id,task_id,task_index,pair,rows,kl_mean,kl_max,top1,ref_mask_nll,mask_nll,ref_mask_em,mask_em");
        using var timeCsv = Open("timing", "size,corpus,plan,context_id,context_tokens,arm,block,prefill_ms,forwards");

        var kv = new FloorAccumulator();
        var kl = new FloorAccumulator();
        var top1 = new Dictionary<string, (int Agree, int Rows)>(StringComparer.Ordinal);
        var nllDiff = new FloorAccumulator();
        var emAgree = new Dictionary<string, (int Agree, int Tasks)>(StringComparer.Ordinal);
        var pairs = new List<string>();
        int layers = vk.Geometry.LayerCount;
        int contexts = 0;

        for (int c = 0; c < o.Names.Length; c++)
        {
            if (o.Contexts[c] <= 0) continue;
            Corpus corpus = TasksRunner.LoadCorpus(o.CorpusDirs[c], "symbol", 512, vk.Tokenizer, o.Hashes[c]);
            var e4 = new E4Options(o.ModelPath, o.CorpusDirs[c], o.Names[c], "symbol", 512, o.Hashes[c], 0, 100,
                new TaskGeneratorOptions { ContextTokens = o.ContextTokens, ContinuationTokens = 32, PrefixTokens = 256, TasksPerContext = 4 },
                o.Mode, Path.Combine(o.KeptDir, $"{o.Names[c]}.validation.csv"), Path.Combine(o.KeptDir, $"{o.Names[c]}.plans.jsonl"),
                o.Contexts[c], [], [], [], [], 16, 0, o.Threads, "", 0, 0, 1);
            IReadOnlyList<CodeContextPlan> plans = E4Command.SelectPlans(e4, corpus);

            for (int p = 0; p < plans.Count; p++)
            {
                var ctxSw = Stopwatch.StartNew();
                CodeContextPlan plan = plans[p];
                CodeTask[] tasks = plan.Tasks.ToArray();
                int[] ids = plan.ContextTokenIds;
                int L = ids.Length;
                int capacity = L + tasks.Max(t => t.PrefixTokenIds.Length + t.ContinuationTokenIds.Length) + 1;
                string contextId = $"{o.Names[c]}/symbol/s0/{o.Mode}/p{p}";
                string Step(string s) => $"[floor {contextId} L {L} +{ctxSw.Elapsed.TotalSeconds:F0} s] {s}";
                string common = string.Join(',', o.Label, o.Names[c], p.ToString(ci), contextId);

                void Time(string arm, int block, double ms) => timeCsv.WriteLine(string.Join(',', common, L.ToString(ci), arm, block.ToString(ci),
                    ms.ToString("F0", ci), ((L + block - 1) / block).ToString(ci)));

                void AddKv(string pair, KvDeviation d)
                {
                    if (!pairs.Contains(pair)) pairs.Add(pair);
                    for (int l = 0; l < layers; l++)
                    {
                        foreach ((string kind, float[] rows) in new[] { ("K", FloorMetrics.KeyRelativeRows(d, l)), ("V", FloorMetrics.ValueRelativeRows(d, l)) })
                        {
                            kv.Add($"{pair}|{l}|{kind}", rows);
                            FloorMetrics.Summary s = FloorMetrics.Summarize(rows);
                            ctxCsv.WriteLine(string.Join(',', common, L.ToString(ci), pair, l.ToString(ci), kind, s.Count.ToString(ci),
                                s.Mean.ToString("G6", ci), s.P99.ToString("G6", ci), s.Max.ToString("G6", ci)));
                        }
                    }
                }

                void AddLogits(string pair, float[][] reference, float[][] other)
                {
                    for (int ti = 0; ti < tasks.Length; ti++)
                    {
                        CodeTask t = tasks[ti];
                        FloorMetrics.LogitsComparison cmp = FloorMetrics.CompareLogits(reference[ti], other[ti], vocab, t.ContinuationTokenIds, t.MaskTokenCount);
                        float[] kls = cmp.Kl.Select(x => (float)x).ToArray();
                        kl.Add(pair, kls);
                        int agree = cmp.Top1Agree.Count(x => x);
                        top1[pair] = top1.TryGetValue(pair, out var a) ? (a.Agree + agree, a.Rows + kls.Length) : (agree, kls.Length);
                        nllDiff.Add(pair, [(float)Math.Abs(cmp.MaskNll - cmp.ReferenceMaskNll)]);
                        int same = cmp.MaskExact == cmp.ReferenceMaskExact ? 1 : 0;
                        emAgree[pair] = emAgree.TryGetValue(pair, out var e) ? (e.Agree + same, e.Tasks + 1) : (same, 1);
                        taskCsv.WriteLine(string.Join(',', common, Quote(t.Id), ti.ToString(ci), pair, kls.Length.ToString(ci),
                            cmp.Kl.Average().ToString("G6", ci), cmp.Kl.Max().ToString("G6", ci), ((double)agree / kls.Length).ToString("G6", ci),
                            cmp.ReferenceMaskNll.ToString("G9", ci), cmp.MaskNll.ToString("G9", ci), cmp.ReferenceMaskExact ? "1" : "0",
                            cmp.MaskExact ? "1" : "0"));
                    }
                }

                // Vulkan reference: full prefill in RefBlock-row blocks, then each task's evaluation block. ---------------------
                Console.WriteLine(Step($"vulkan prefill block {o.RefBlock} (reference)"));
                using IKvCache vref = vk.NewBackendCache(capacity);
                var sw = Stopwatch.StartNew();
                vk.Prefill(vref, ids, 0, o.RefBlock);
                Time($"vk{r}", o.RefBlock, sw.Elapsed.TotalMilliseconds);
                Console.WriteLine(Step("vulkan reference evaluation blocks"));
                float[][] vrefLogits = tasks.Select(t => ForwardTask(vk, vref, L, t)).ToArray();
                // The reference rows are read back once and compared from host memory (each pair would otherwise re-read them).
                Console.WriteLine(Step("read back the vulkan reference"));
                var refK = new float[layers][];
                var refV = new float[layers][];
                for (int l = 0; l < layers; l++)
                {
                    int count = L * vk.Geometry.KvStrideOf(l);
                    refK[l] = new float[count];
                    refV[l] = new float[count];
                    vk.Backend.ReadRows(vref, l, 0, L, refK[l], refV[l]);
                }
                KvDeviation.KvRowsReader readRef = (l, n, k, v) =>
                {
                    int count = n * vk.Geometry.KvStrideOf(l);
                    refK[l].AsSpan(0, count).CopyTo(k);
                    refV[l].AsSpan(0, count).CopyTo(v);
                };
                KvDeviation Against(IChunkBackend backend, IKvCache cache, bool cacheIsReference) => cacheIsReference
                    ? KvDeviation.Compute(layers, L, vk.Geometry.KvStrideOf, (l, n, k, v) => backend.ReadRows(cache, l, 0, n, k, v), readRef)
                    : KvDeviation.Compute(layers, L, vk.Geometry.KvStrideOf, readRef, (l, n, k, v) => backend.ReadRows(cache, l, 0, n, k, v));

                // (c) rerun, then (b) every other block size. ---------------------------------------------------------------
                foreach ((string arm, int block, string pair) in new[] { ($"vk{r}-rerun", o.RefBlock, pairC) }
                             .Concat(o.Blocks.Select(n => ($"vk{n}", n, PairB(n)))))
                {
                    Console.WriteLine(Step($"vulkan prefill block {block} ({arm})"));
                    using IKvCache cache = vk.NewBackendCache(capacity);
                    sw.Restart();
                    vk.Prefill(cache, ids, 0, block);
                    Time(arm, block, sw.Elapsed.TotalMilliseconds);
                    Console.WriteLine(Step($"read back and compare {pair}"));
                    AddKv(pair, Against(vk.Backend, cache, cacheIsReference: false));
                    Console.WriteLine(Step($"evaluation blocks {arm}"));
                    AddLogits(pair, vrefLogits, tasks.Select(t => ForwardTask(vk, cache, L, t)).ToArray());
                }

                // (a) CPU full prefill (reference side of the pair) vs the Vulkan reference. ---------------------------------
                Console.WriteLine(Step($"cpu prefill block {o.RefBlock}"));
                using (IKvCache cc = cpu.NewBackendCache(capacity))
                {
                    sw.Restart();
                    cpu.Prefill(cc, ids, 0, o.RefBlock);
                    Time($"cpu{r}", o.RefBlock, sw.Elapsed.TotalMilliseconds);
                    Console.WriteLine(Step($"read back and compare {pairA}"));
                    AddKv(pairA, Against(cpu.Backend, cc, cacheIsReference: true));
                    Console.WriteLine(Step("cpu evaluation blocks"));
                    float[][] cpuLogits = tasks.Select(t => ForwardTask(cpu, cc, L, t)).ToArray();
                    AddLogits(pairA, cpuLogits, vrefLogits);
                }

                contexts++;
                Console.WriteLine($"[floor {contextId} L {L} tasks {tasks.Length}] done in {ctxSw.Elapsed.TotalSeconds:F0} s; at {total.Elapsed.TotalMinutes:F1} min");
            }
        }

        // Pooled floor over every row of every context. ----------------------------------------------------------------------
        using (var floor = Open("floor", "size,pair,layer,kind,n,rel_l2_mean,rel_l2_p99,rel_l2_max,kl_mean,kl_p99,kl_max,top1,"
                   + "mask_nll_absdiff_mean,mask_nll_absdiff_max,mask_em_agree,contexts"))
        {
            foreach (string pair in pairs)
            {
                for (int l = 0; l < layers; l++)
                    foreach (string kind in new[] { "K", "V" })
                    {
                        FloorMetrics.Summary s = kv.Summary($"{pair}|{l}|{kind}");
                        floor.WriteLine(string.Join(',', o.Label, pair, l.ToString(ci), kind, s.Count.ToString(ci), s.Mean.ToString("G6", ci),
                            s.P99.ToString("G6", ci), s.Max.ToString("G6", ci), "", "", "", "", "", "", "", contexts.ToString(ci)));
                    }
                FloorMetrics.Summary k = kl.Summary(pair), m = nllDiff.Summary(pair);
                floor.WriteLine(string.Join(',', o.Label, pair, "all", "logits", k.Count.ToString(ci), "", "", "", k.Mean.ToString("G6", ci),
                    k.P99.ToString("G6", ci), k.Max.ToString("G6", ci), ((double)top1[pair].Agree / top1[pair].Rows).ToString("G6", ci),
                    m.Mean.ToString("G6", ci), m.Max.ToString("G6", ci), ((double)emAgree[pair].Agree / emAgree[pair].Tasks).ToString("G6", ci),
                    contexts.ToString(ci)));
            }
        }
        ctxCsv.Dispose();
        taskCsv.Dispose();
        timeCsv.Dispose();
        foreach (string p in partials) File.Move(p, p[..^".partial".Length], overwrite: true);
        Console.WriteLine($"floor {o.Label}: {contexts} contexts, {pairs.Count} pairs in {total.Elapsed.TotalMinutes:F1} min -> {o.OutDir}");
        return 0;
    }

    /// <summary>
    /// The E4 evaluation block of <paramref name="task"/> after the context at <c>[0, L)</c> (as <c>E4Command.ForwardTask</c>, on any
    /// backend cache): the prefix minus its last token is prefilled at <c>L</c> in one block, then [last prefix token,
    /// continuation[..^1]] gives one logit row per continuation token. The cache is rolled back to <c>L</c>.
    /// </summary>
    private static float[] ForwardTask(LabModel lab, IKvCache cache, int L, CodeTask task)
    {
        int[] prefix = task.PrefixTokenIds, cont = task.ContinuationTokenIds;
        try
        {
            lab.Prefill(cache, prefix.AsSpan(0, prefix.Length - 1), L, 512);
            int[] ids = [prefix[^1], .. cont.AsSpan(0, cont.Length - 1)];
            return lab.ForwardLogits(cache, ids, L + prefix.Length - 1, allRows: true);
        }
        finally
        {
            cache.Rollback(L);
        }
    }

    private static string Quote(string s) => "\"" + s.Replace("\"", "'", StringComparison.Ordinal) + "\"";
}
