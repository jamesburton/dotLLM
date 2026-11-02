using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.Core.Attention;

namespace ChunkKvLab;

/// <summary>Options of the <c>recompute</c> command.</summary>
internal sealed record RecomputeOptions(
    string ModelPath, string[] Names, string[] CorpusDirs, string[] Hashes, int[] Contexts, string KeptDir, string Mode, int ContextTokens,
    int[] Budgets, string[] Selectors, int Repeats, bool Eval, int CpuContexts, int MaxRows, int[] FullBlocks, int Threads, string Label,
    string OutDir);

/// <summary>
/// Phase G Task 4 (G2): single-pass selective recompute on Vulkan, its speed against the per-run loop and full prefill, and its numerics.
/// Contexts are E4's (<see cref="E4Command.SelectPlans"/>, symbol chunker, seed 0), as in the <c>floor</c> command. Per context, on Vulkan:
/// <list type="bullet">
/// <item>full prefill in each <c>--full-blocks</c> block size, <c>--repeats</c> times (the 512-row prefill is the reference);</item>
/// <item>chunk encode (host entries: forward + readback; resident: forward only) and assembly (device copy and host write, each with the
/// one-submit RoPE shift), <c>--repeats</c> times;</item>
/// <item>per selector and budget: the naive cache is re-assembled (device copy; bit-exact each time) and recomputed single-pass and per-run,
/// <c>--repeats</c> times each.</item>
/// </list>
/// With <c>--eval 1</c> (one process that may widen the all-row logits limit; its timings are not used), also: each task's E4 evaluation block
/// on every arm against the Vulkan full prefill (KL, top-1, masked-call NLL/EM); K/V of single-pass vs per-run; pair (d), the per-query-position
/// attention against the ordinary kernel on the same rows (full prefill in B-row blocks vs the same rows recomputed single-pass in B-row
/// batches over a cache one row longer, B = min(L, 2048)); and on the first <c>--cpu-contexts</c> contexts the CPU per-run recompute of the
/// same random selection against the Vulkan single pass (pair (a)). Outputs are <c>*.partial</c> until every context finished.
/// </summary>
internal static class RecomputeCommand
{
    public static int Run(RecomputeOptions o)
    {
        var total = Stopwatch.StartNew();
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        Directory.CreateDirectory(o.OutDir);
        if (o.Names.Length != o.CorpusDirs.Length || o.Names.Length != o.Hashes.Length || o.Names.Length != o.Contexts.Length)
            throw new ArgumentException("--names, --corpus-dirs, --hashes and --contexts must have one entry per corpus.");
        using var awake = KeepAwake.Hold("ChunkKvLab recompute: long GPU measurement run");
        Console.WriteLine(awake.Description);
        double lastBeatMin = 0;
        using var heartbeat = new Timer(_ =>
        {
            double now = total.Elapsed.TotalMinutes;
            Console.WriteLine($"[heartbeat {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC, +{now:F1} min, {now - lastBeatMin:F1} min since previous]");
            lastBeatMin = now;
        }, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));

        using LabModel vk = LabModel.Load(o.ModelPath, o.Threads, ChunkBackendTags.Vulkan);
        Console.WriteLine($"vulkan device: {vk.DeviceName}; eval {(o.Eval ? "on" : "off (timing run: all-row logits never widened)")}; "
            + $"single-pass row cap {(o.MaxRows > 0 ? o.MaxRows.ToString(ci) : "none")}");
        using LabModel? cpu = o.Eval && o.CpuContexts > 0 ? LabModel.Load(o.ModelPath, o.Threads, ChunkBackendTags.Cpu) : null;
        int vocab = vk.Config.VocabSize, layers = vk.Geometry.LayerCount;
        int maxRows = o.MaxRows > 0 ? o.MaxRows : int.MaxValue;
        var encoder = new ChunkEncoder(vk.Fingerprint);
        var assembler = new ChunkAssembler(vk.Config, vk.Backend);

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
        using var timeCsv = Open("timing", "size,corpus,plan,context_id,context_tokens,chunks,eval_run,arm,selector,budget_pct,rows,runs,rep,ms,forward_calls,qpos_layers");
        using var kvCsv = o.Eval ? Open("kv", "size,corpus,plan,context_id,context_tokens,pair,layer,kind,n,rel_l2_mean,rel_l2_p99,rel_l2_max") : null;
        using var taskCsv = o.Eval
            ? Open("tasks", "size,corpus,plan,context_id,task_id,task_index,pair,selector,budget_pct,rows,kl_mean,kl_max,top1,ref_mask_nll,mask_nll,ref_mask_em,mask_em")
            : null;
        var kvPool = new FloorAccumulator();
        var pairs = new List<string>();
        int contexts = 0;

        // Warm-up (pipelines, scratch buffers): one 512-row prefill of the first context, then discarded.
        bool warmed = false;

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
                int[][] chunkIds = plan.Chunks.Select(ch => ch.TokenIds).ToArray();
                var ctxBase = new ChunkedContext(chunkIds, []);
                int capacity = L + tasks.Max(t => t.PrefixTokenIds.Length + t.ContinuationTokenIds.Length) + 2;
                string contextId = $"{o.Names[c]}/symbol/s0/{o.Mode}/p{p}";
                string common = string.Join(',', o.Label, o.Names[c], p.ToString(ci), contextId);
                string Step(string s) => $"[recompute {contextId} L {L} chunks {plan.Chunks.Count} +{ctxSw.Elapsed.TotalSeconds:F0} s] {s}";
                void Time(string arm, string selector, int pct, int rows, int runs, int rep, double ms, int calls, long qpos) =>
                    timeCsv.WriteLine(string.Join(',', common, L.ToString(ci), plan.Chunks.Count.ToString(ci), o.Eval ? "1" : "0", arm, selector,
                        pct.ToString(ci), rows.ToString(ci), runs.ToString(ci), rep.ToString(ci), ms.ToString("F2", ci), calls.ToString(ci),
                        qpos.ToString(ci)));

                if (!warmed)
                {
                    using IKvCache w = vk.NewBackendCache(capacity);
                    vk.Prefill(w, ids.AsSpan(0, Math.Min(512, L)), 0, 512);
                    warmed = true;
                }

                // Full prefill (reference = the last 512-row prefill). --------------------------------------------------------
                IKvCache? vref = null;
                var sw = new Stopwatch();
                foreach (int block in o.FullBlocks)
                    for (int rep = 0; rep < o.Repeats; rep++)
                    {
                        Console.WriteLine(Step($"full prefill block {block} rep {rep}"));
                        IKvCache cache = vk.NewBackendCache(capacity);
                        sw.Restart();
                        vk.Prefill(cache, ids, 0, block);
                        Time($"full{block}", "", 100, L, (L + block - 1) / block, rep, sw.Elapsed.TotalMilliseconds, (L + block - 1) / block, 0);
                        if (block == 512 && vref is null && rep == o.Repeats - 1) vref = cache;
                        else cache.Dispose();
                    }
                if (vref is null)
                {
                    vref = vk.NewBackendCache(capacity);
                    vk.Prefill(vref, ids, 0, 512);
                }
                using IKvCache reference = vref;
                float[][]? refLogits = null;
                if (o.Eval)
                {
                    Console.WriteLine(Step("reference evaluation blocks"));
                    refLogits = tasks.Select(t => FloorCommand.ForwardTask(vk, reference, L, t)).ToArray();
                }

                // Encode and assembly. ----------------------------------------------------------------------------------------
                Console.WriteLine(Step("encode (host entries: forward + readback)"));
                sw.Restart();
                var host = plan.Chunks.Select(ch => encoder.Encode(vk.Model, vk.Backend, ch.TokenIds)).ToList();
                Time("encode_host", "", 0, L, plan.Chunks.Count, 0, sw.Elapsed.TotalMilliseconds, plan.Chunks.Count, 0);
                Console.WriteLine(Step("encode (resident entries: forward only)"));
                sw.Restart();
                var resident = plan.Chunks.Select(ch => encoder.EncodeResident(vk.Model, vk.Backend, ch.TokenIds)).ToList();
                Time("encode_resident", "", 0, L, plan.Chunks.Count, 0, sw.Elapsed.TotalMilliseconds, plan.Chunks.Count, 0);
                try
                {
                    for (int rep = 0; rep < o.Repeats; rep++)
                    {
                        using IKvCache a = vk.NewBackendCache(capacity);
                        sw.Restart();
                        assembler.Assemble(a, resident, vk.Fingerprint);
                        Time("asm_device", "", 0, L, plan.Chunks.Count, rep, sw.Elapsed.TotalMilliseconds, 0, 0);
                        using IKvCache h = vk.NewBackendCache(capacity);
                        sw.Restart();
                        assembler.Assemble(h, host, vk.Fingerprint);
                        Time("asm_host", "", 0, L, plan.Chunks.Count, rep, sw.Elapsed.TotalMilliseconds, 0, 0);
                    }
                    IKvCache Naive()
                    {
                        IKvCache n = vk.NewBackendCache(capacity);
                        assembler.Assemble(n, resident, vk.Fingerprint);
                        return n;
                    }

                    void AddKv(string pair, IChunkBackend refBackend, IKvCache refCache, IChunkBackend otherBackend, IKvCache otherCache)
                    {
                        if (!pairs.Contains(pair)) pairs.Add(pair);
                        KvDeviation d = KvDeviation.Compute(layers, L, vk.Geometry.KvStrideOf,
                            (l, n, k, v) => refBackend.ReadRows(refCache, l, 0, n, k, v),
                            (l, n, k, v) => otherBackend.ReadRows(otherCache, l, 0, n, k, v));
                        for (int l = 0; l < layers; l++)
                            foreach ((string kind, float[] rows) in new[] { ("K", FloorMetrics.KeyRelativeRows(d, l)), ("V", FloorMetrics.ValueRelativeRows(d, l)) })
                            {
                                kvPool.Add($"{pair}|{l}|{kind}", rows);
                                FloorMetrics.Summary s = FloorMetrics.Summarize(rows);
                                kvCsv!.WriteLine(string.Join(',', common, L.ToString(ci), pair, l.ToString(ci), kind, s.Count.ToString(ci),
                                    s.Mean.ToString("G6", ci), s.P99.ToString("G6", ci), s.Max.ToString("G6", ci)));
                            }
                    }
                    void AddLogits(string pair, string selector, int pct, int rows, float[][] refRows, float[][] other)
                    {
                        for (int ti = 0; ti < tasks.Length; ti++)
                        {
                            CodeTask t = tasks[ti];
                            FloorMetrics.LogitsComparison cmp = FloorMetrics.CompareLogits(refRows[ti], other[ti], vocab, t.ContinuationTokenIds, t.MaskTokenCount);
                            taskCsv!.WriteLine(string.Join(',', common, Quote(t.Id), ti.ToString(ci), pair, selector, pct.ToString(ci), rows.ToString(ci),
                                cmp.Kl.Average().ToString("G6", ci), cmp.Kl.Max().ToString("G6", ci),
                                ((double)cmp.Top1Agree.Count(x => x) / cmp.Kl.Length).ToString("G6", ci), cmp.ReferenceMaskNll.ToString("G9", ci),
                                cmp.MaskNll.ToString("G9", ci), cmp.ReferenceMaskExact ? "1" : "0", cmp.MaskExact ? "1" : "0"));
                        }
                    }
                    float[][] EvalTasks(LabModel lab, IKvCache cache) => tasks.Select(t => FloorCommand.ForwardTask(lab, cache, L, t)).ToArray();

                    if (o.Eval)
                    {
                        Console.WriteLine(Step("naive evaluation blocks"));
                        using IKvCache naive = Naive();
                        AddLogits("naive_vs_vk512", "", 0, 0, refLogits!, EvalTasks(vk, naive));
                    }

                    // Attention-kernel control: the same k contiguous rows (k = the largest budget), recomputed as the tail [L-k, L) (a run
                    // ending at the cache length: the ordinary flash/coopmat attention) and one row earlier [L-k-1, L-1) (inside the cache:
                    // the per-query-position attention). Same forward shape and matmul regime; only the attention kernel differs.
                    {
                        int k = (int)Math.Round(L * o.Budgets.Max() / 100.0, MidpointRounding.AwayFromZero);
                        foreach ((string arm, int first) in new[] { ("tail_ordinary", L - k), ("tail_qpos", L - k - 1) })
                            for (int rep = 0; rep < o.Repeats; rep++)
                            {
                                Console.WriteLine(Step($"control {arm}: {k} rows at {first}.. rep {rep}"));
                                using IKvCache cache = Naive();
                                int[] sel = Enumerable.Range(first, k).ToArray();
                                sw.Restart();
                                SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(vk.Model, vk.Backend, cache, ids, sel, RecomputeMode.SinglePass,
                                    maxRows);
                                Time(arm, "", o.Budgets.Max(), k, 1, rep, sw.Elapsed.TotalMilliseconds, r.ForwardCalls, r.QueryPositionAttentionLayers);
                            }
                    }

                    // Recompute arms. -----------------------------------------------------------------------------------------
                    foreach (string selName in o.Selectors)
                    {
                        IRankedSelector selector = selName switch
                        {
                            "chunk_starts" => new ChunkStartsSelector(),
                            "first_k" => new FirstKSelector(16),
                            "random" => new RandomSelector(),
                            _ => throw new ArgumentException($"--selectors: unknown '{selName}' (chunk_starts, first_k, random)"),
                        };
                        foreach (int pct in o.Budgets)
                        {
                            int budget = (int)Math.Round(L * pct / 100.0, MidpointRounding.AwayFromZero);
                            int[] sel = selector.Select(ctxBase, budget, E4Command.RandomSeed(0, p));
                            int runs = Runs(sel);
                            IKvCache? single = null, perRun = null;
                            try
                            {
                                foreach (RecomputeMode mode in new[] { RecomputeMode.SinglePass, RecomputeMode.PerRun })
                                    for (int rep = 0; rep < o.Repeats; rep++)
                                    {
                                        Console.WriteLine(Step($"{selName} {pct} % ({sel.Length} rows, {runs} runs) {mode} rep {rep}"));
                                        IKvCache cache = Naive();
                                        sw.Restart();
                                        SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(vk.Model, vk.Backend, cache, ids, sel, mode,
                                            mode == RecomputeMode.SinglePass ? maxRows : int.MaxValue);
                                        double ms = sw.Elapsed.TotalMilliseconds;
                                        Time(mode == RecomputeMode.SinglePass ? "single" : "perrun", selName, pct, sel.Length, runs, rep, ms, r.ForwardCalls,
                                            r.QueryPositionAttentionLayers);
                                        if (rep < o.Repeats - 1) cache.Dispose();
                                        else if (mode == RecomputeMode.SinglePass) single = cache;
                                        else perRun = cache;
                                    }
                                if (!o.Eval) continue;
                                Console.WriteLine(Step($"{selName} {pct} %: evaluation blocks and single vs per-run K/V"));
                                float[][] s = EvalTasks(vk, single!), pr = EvalTasks(vk, perRun!);
                                AddLogits("single_vs_vk512", selName, pct, sel.Length, refLogits!, s);
                                AddLogits("perrun_vs_vk512", selName, pct, sel.Length, refLogits!, pr);
                                AddLogits("perrun_vs_single", selName, pct, sel.Length, pr, s);
                                AddKv($"perrun_vs_single_{selName}{pct}", vk.Backend, perRun!, vk.Backend, single!);

                                // Pair (a) on the recomputed state: CPU per-run (reference) vs Vulkan single pass, same selection.
                                if (cpu is not null && contexts < o.CpuContexts && selName == "random" && pct == 10)
                                {
                                    Console.WriteLine(Step("cpu encode, assemble and per-run recompute (pair (a) cross-check)"));
                                    var cpuEncoder = new ChunkEncoder(cpu.Fingerprint);
                                    var cpuEntries = plan.Chunks.Select(ch => cpuEncoder.Encode(cpu.Model, cpu.Backend, ch.TokenIds)).ToList();
                                    using IKvCache cc = cpu.NewBackendCache(capacity);
                                    new ChunkAssembler(cpu.Config, cpu.Backend).Assemble(cc, cpuEntries, cpu.Fingerprint);
                                    sw.Restart();
                                    SelectiveRecomputeResult cr = SelectiveRecomputer.Recompute(cpu.Model, cpu.Backend, cc, ids, sel);
                                    Time("cpu_perrun", selName, pct, sel.Length, runs, 0, sw.Elapsed.TotalMilliseconds, cr.ForwardCalls, 0);
                                    AddKv("a_cpuperrun_vs_vksingle_random10", cpu.Backend, cc, vk.Backend, single!);
                                    AddLogits("a_cpuperrun_vs_vksingle", selName, pct, sel.Length, EvalTasks(cpu, cc), s);
                                }
                            }
                            finally
                            {
                                single?.Dispose();
                                perRun?.Dispose();
                            }
                        }
                    }

                    // Pair (d): the per-query-position attention vs the ordinary kernel on the same rows and forward shapes. ----------
                    if (o.Eval)
                    {
                        int b = Math.Min(Math.Min(L, 2048), maxRows); // --max-rows bounds every submit, here too
                        Console.WriteLine(Step($"pair (d): full prefill in {b}-row blocks"));
                        using IKvCache vb = vk.NewBackendCache(capacity);
                        vk.Prefill(vb, ids, 0, b);
                        using IKvCache q = vk.NewBackendCache(capacity);
                        vk.Prefill(q, ids, 0, b);
                        vk.Prefill(q, ids.AsSpan(0, 1), L, 1); // one row past the context: every recompute batch now sits inside the cache
                        Console.WriteLine(Step($"pair (d): all {L} positions recomputed single-pass in {b}-row batches over {L + 1} rows"));
                        sw.Restart();
                        SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(vk.Model, vk.Backend, q, ids, Enumerable.Range(0, L).ToArray(),
                            RecomputeMode.SinglePass, b);
                        Time("d_qpos_all", "", 100, L, 1, 0, sw.Elapsed.TotalMilliseconds, r.ForwardCalls, r.QueryPositionAttentionLayers);
                        if (r.QueryPositionAttentionLayers != (long)r.ForwardCalls * layers)
                            throw new InvalidOperationException($"pair (d) did not run the per-query-position kernel on every layer ({r.QueryPositionAttentionLayers}).");
                        q.Rollback(L);
                        AddKv($"d_vk{b}_vs_qpos{b}", vk.Backend, vb, vk.Backend, q);
                        AddKv($"d_vk512_vs_qpos{b}", vk.Backend, reference, vk.Backend, q);
                        float[][] qb = EvalTasks(vk, q);
                        AddLogits($"d_vk{b}_vs_qpos{b}", "", 100, L, EvalTasks(vk, vb), qb);
                        AddLogits($"d_vk512_vs_qpos{b}", "", 100, L, refLogits!, qb);
                    }
                }
                finally
                {
                    foreach (ResidentChunkKvEntry e in resident) e.Dispose();
                }

                contexts++;
                Console.WriteLine($"[recompute {contextId} L {L} tasks {tasks.Length}] done in {ctxSw.Elapsed.TotalSeconds:F0} s; at {total.Elapsed.TotalMinutes:F1} min");
            }
        }

        if (o.Eval)
            using (var pooled = Open("kvpooled", "size,pair,layer,kind,n,rel_l2_mean,rel_l2_p99,rel_l2_max,contexts"))
                foreach (string pair in pairs)
                    for (int l = 0; l < layers; l++)
                        foreach (string kind in new[] { "K", "V" })
                        {
                            FloorMetrics.Summary s = kvPool.Summary($"{pair}|{l}|{kind}");
                            pooled.WriteLine(string.Join(',', o.Label, pair, l.ToString(ci), kind, s.Count.ToString(ci), s.Mean.ToString("G6", ci),
                                s.P99.ToString("G6", ci), s.Max.ToString("G6", ci), contexts.ToString(ci)));
                        }
        timeCsv.Dispose();
        kvCsv?.Dispose();
        taskCsv?.Dispose();
        foreach (string p in partials) File.Move(p, p[..^".partial".Length], overwrite: true);
        Console.WriteLine($"recompute {o.Label}: {contexts} contexts in {total.Elapsed.TotalMinutes:F1} min -> {o.OutDir}");
        return 0;
    }

    /// <summary>Number of maximal runs of consecutive positions in an ascending selection (the per-run path's forward count).</summary>
    internal static int Runs(ReadOnlySpan<int> selection)
    {
        int runs = 0;
        for (int i = 0; i < selection.Length; i++)
            if (i == 0 || selection[i] != selection[i - 1] + 1) runs++;
        return runs;
    }

    private static string Quote(string s) => "\"" + s.Replace("\"", "'", StringComparison.Ordinal) + "\"";
}
