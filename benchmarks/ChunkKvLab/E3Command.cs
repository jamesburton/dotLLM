using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.ChunkReuse.Graph;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>Options of the <c>e3</c> command.</summary>
internal sealed record E3Options(
    string ModelPath, string CorpusDir, string CorpusName, string Chunker, int ChunkTokens, string? ExpectHash, int Seed, int Count,
    TaskGeneratorOptions Generator, int MaxContexts, string? KeptPlansJsonl, string? KeptValidationCsv, bool Query, int RawContexts,
    int RandomDraws, double MaxMinutes, int Threads, string OutDir, long ChunkCacheTokens);

/// <summary>
/// E3 (Task 10): per-layer K/V deviation between the full prefill and the assembled chunk cache (isolated chunk encodings,
/// RoPE-shifted by <see cref="ChunkAssembler"/>), by token role from the Roslyn reference graph. Contexts are the Task 8
/// <see cref="CrossFileTaskGenerator"/> plans: the first <c>--contexts</c> plans of the seed, plus (with <c>--kept</c>) every plan
/// holding a task the Task 8 validation kept, flagged in <c>.contexts.csv</c>. The query task is the plan's first kept task, else
/// its first task. With
/// <c>--query 1</c> one task's query prefix (masked call excluded) is forwarded after the context in both arms and its rows are
/// compared too (query rows attend to stale K/V in the assembled arm, so they deviate from layer 1 on: propagated error).
/// Outputs, per run, in <c>--out-dir</c> with stem <c>&lt;corpus&gt;-&lt;chunker&gt;-s&lt;seed&gt;</c>:
/// <c>.contexts.csv</c> (one row per context), <c>.cells.csv</c> (per context × layer × group sums), <c>.capture.csv</c>
/// (per context × layer × metric × k: top-k% share and baselines) and <c>.raw.csv</c> (every row of the first
/// <c>--raw</c> contexts).
/// </summary>
internal static class E3Command
{
    private static readonly double[] Fractions = [0.01, 0.05, 0.10, 0.20, 0.50];
    private static readonly (string Name, int Lo, int Hi)[] OffsetBins =
        [("0", 0, 1), ("1-3", 1, 4), ("4-15", 4, 16), ("16-63", 16, 64), ("64-255", 64, 256), ("256+", 256, int.MaxValue)];
    private static readonly (string Name, int Lo, int Hi)[] CoarseBins = [("0", 0, 1), ("1-15", 1, 16), ("16+", 16, int.MaxValue)];

    private static readonly (string Name, TokenRoleFlags Flag)[] Labels =
    [
        ("cs1", TokenRoleFlags.ChunkStart1), ("cs4", TokenRoleFlags.ChunkStart4), ("cs16", TokenRoleFlags.ChunkStart16),
        ("def", TokenRoleFlags.Definition), ("use", TokenRoleFlags.Use), ("import", TokenRoleFlags.Import),
        ("cross", TokenRoleFlags.CrossChunk), ("ctxcross", TokenRoleFlags.InContextCross), ("qtarget", TokenRoleFlags.QueryTarget),
    ];

    public static int Run(E3Options o)
    {
        var total = Stopwatch.StartNew();
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        Directory.CreateDirectory(o.OutDir);
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        ITokenizer tok = lab.Tokenizer;
        Corpus corpus = TasksRunner.LoadCorpus(o.CorpusDir, o.Chunker, o.ChunkTokens, tok, o.ExpectHash);

        var sw = Stopwatch.StartNew();
        ReferenceGraph graph = new RoslynGraphBuilder(tok).Build(corpus);
        Console.WriteLine($"graph: {graph.Edges.Count} edges in {sw.Elapsed.TotalSeconds:F1} s; mapping-mismatch chunks {graph.MappingMismatchChunks}");
        var indexOf = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < graph.Chunks.Count; i++) indexOf.TryAdd(graph.Chunks[i].Id, i);
        // Every edge end, chunk-local: (token, role flag | CrossChunk when the other end is in another chunk).
        var ends = new List<(int Token, TokenRoleFlags Flags)>[graph.Chunks.Count];
        foreach (ChunkEdge e in graph.Edges)
        {
            TokenRoleFlags cross = e.UseChunk != e.DefinitionChunk ? TokenRoleFlags.CrossChunk : TokenRoleFlags.None;
            (ends[e.UseChunk] ??= []).Add((e.UseToken, TokenRoles.EdgeEndFlag(e.Kind, useEnd: true) | cross));
            (ends[e.DefinitionChunk] ??= []).Add((e.DefinitionToken, TokenRoles.EdgeEndFlag(e.Kind, useEnd: false) | cross));
        }

        sw.Restart();
        IReadOnlyList<CodeContextPlan> plans = new CrossFileTaskGenerator(o.Generator).GeneratePlans(corpus, o.Count, o.Seed);
        // Sample = the first --contexts plans (the generator visits definitions in a seeded order, so this is a seeded random
        // sample; 0 = all). With --kept, every plan holding a kept Task 8 task is added too and flagged.
        Dictionary<int, List<string>>? kept = null;
        if (o.KeptValidationCsv is not null)
        {
            VerifyPlans(plans, o.KeptPlansJsonl ?? throw new ArgumentException("--kept needs --plans-jsonl"));
            kept = ReadKept(o.KeptValidationCsv); // plan -> kept task ids, in file order
        }
        int sample = o.MaxContexts > 0 ? Math.Min(o.MaxContexts, plans.Count) : plans.Count;
        var chosen = new List<(int Plan, CodeTask Task, bool InSample, string Kept)>();
        for (int p = 0; p < plans.Count; p++)
        {
            bool inSample = p < sample;
            List<string>? keptIds = null;
            bool isKept = kept is not null && kept.TryGetValue(p, out keptIds);
            if (!inSample && !isKept) continue;
            CodeTask task = isKept ? plans[p].Tasks.First(t => t.Id == keptIds![0]) : plans[p].Tasks[0];
            chosen.Add((p, task, inSample, kept is null ? "" : isKept ? "1" : "0"));
        }
        Console.WriteLine($"plans: {plans.Count} generated (seed {o.Seed}, {sw.Elapsed.TotalSeconds:F1} s); contexts: {chosen.Count} "
            + $"({chosen.Count(c => c.InSample)} in the first-{sample} sample, {chosen.Count(c => c.Kept == "1")} with a kept task); "
            + $"query rows {(o.Query ? "on" : "off")}");

        string stem = Path.Combine(o.OutDir, $"{o.CorpusName}-{o.Chunker}-s{o.Seed}");
        using var contextsCsv = new StreamWriter(stem + ".contexts.csv") { AutoFlush = true };
        using var cellsCsv = new StreamWriter(stem + ".cells.csv");
        using var captureCsv = new StreamWriter(stem + ".capture.csv");
        using var rawCsv = o.RawContexts > 0 ? new StreamWriter(stem + ".raw.csv") : null;
        contextsCsv.WriteLine("context_id,corpus,chunker,seed,plan,in_sample,kept,context_tokens,chunks,query_tokens,task_id,n_chunkstart,n_definition,n_use,"
            + "n_import,n_other,n_querysite,n_query,n_def_in_cs16,n_cross,n_ctxcross,n_qtarget,query_map_ok,chunk0_max_l2,layer0_max_rel,"
            + "deep_query_mean_rel,seconds");
        cellsCsv.WriteLine("context_id,corpus,chunker,seed,layer,kind,group,n,k_l2,v_l2,kv_l2,k_rel,v_rel,kv_rel,k_ref,v_ref,kv_l2_max");
        captureCsv.WriteLine("context_id,corpus,chunker,seed,layer,metric,k_pct,n,share_top,share_random,share_position,overlap_position,share_layer1set");
        rawCsv?.WriteLine("layer,context_id,seed,chunker,token_index,chunk,offset,token_role,labels,k_l2,v_l2,k_rel,v_rel");

        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config);
        var entryCache = new Dictionary<string, ChunkKvEntry>(StringComparer.Ordinal);
        long cachedTokens = 0;
        int done = 0;
        foreach ((int p, CodeTask task, bool inSample, string keptFlag) in chosen)
        {
            if (o.MaxMinutes > 0 && total.Elapsed.TotalMinutes >= o.MaxMinutes)
            {
                Console.WriteLine($"time cap {o.MaxMinutes} min reached after {done} of {chosen.Count} contexts");
                break;
            }
            var ctxSw = Stopwatch.StartNew();
            CodeContextPlan plan = plans[p];
            string contextId = $"{o.CorpusName}/{o.Chunker}/s{o.Seed}/p{p}";
            int[] ids = plan.ContextTokenIds;
            int L = ids.Length;
            int[] query = o.Query ? task.PrefixTokenIds : [];
            int P = query.Length;
            int[] starts = plan.ChunkStarts;

            // Full prefill (+ query) -------------------------------------------------------------------------------------
            using SimpleKvCache full = lab.NewCache(L + P);
            lab.Prefill(full, ids, 0, 512);
            if (P > 0) lab.Prefill(full, query, L, 512);

            // Assembled chunk cache (+ query over it) ------------------------------------------------------------------------
            var entries = new List<ChunkKvEntry>(plan.Chunks.Count);
            foreach (Chunk c in plan.Chunks)
            {
                if (!entryCache.TryGetValue(c.Id, out ChunkKvEntry? e))
                {
                    if (cachedTokens + c.TokenIds.Length > o.ChunkCacheTokens)
                    {
                        entryCache.Clear();
                        cachedTokens = 0;
                    }
                    e = encoder.Encode(lab.Model, c.TokenIds);
                    entryCache[c.Id] = e;
                    cachedTokens += c.TokenIds.Length;
                }
                entries.Add(e);
            }
            using SimpleKvCache asm = lab.NewCache(L + P);
            assembler.Assemble(asm, entries, lab.Fingerprint);
            if (P > 0) lab.Prefill(asm, query, L, 512);

            KvDeviation d = KvDeviation.Compute(full, asm, L + P);

            // Roles ----------------------------------------------------------------------------------------------------------
            var marks = new List<(int, TokenRoleFlags)>();
            var placed = new Dictionary<int, int>();
            for (int k = 0; k < plan.Chunks.Count; k++)
            {
                int g = indexOf[plan.Chunks[k].Id];
                placed.TryAdd(g, k);
                if (ends[g] is { } list)
                    foreach ((int t, TokenRoleFlags f) in list) marks.Add((starts[k] + t, f));
            }
            var ctx = new ChunkedContext(plan.Chunks.Select(c => c.TokenIds).ToArray(), []);
            foreach (SymbolEdge e in graph.EdgesForContext(ctx, plan.Chunks))
            {
                marks.Add((e.UseTokenIndex, TokenRoleFlags.InContextCross));
                marks.Add((e.DefinitionTokenIndex, TokenRoleFlags.InContextCross));
            }
            bool queryMapOk = true;
            {
                IReadOnlyList<QueryEdge> qe = graph.QueryEdges(task.CallSitePath, task.PrefixStartOffset, task.MaskStartOffset);
                int[]? charStart = P > 0 ? QueryCharStarts(tok, task, out queryMapOk) : null;
                int header = task.Prefix.Length - (task.MaskStartOffset - task.PrefixStartOffset);
                foreach (QueryEdge q in qe)
                {
                    if (!placed.TryGetValue(q.DefinitionChunk, out int k)) continue; // definition not in this context
                    marks.Add((starts[k] + q.DefinitionToken, TokenRoleFlags.QueryTarget));
                    if (charStart is null || !queryMapOk) continue;
                    int c = header + (q.UseOffset - task.PrefixStartOffset);
                    int row = Array.BinarySearch(charStart, c);
                    if (row < 0) row = ~row - 1;
                    marks.Add((L + row, TokenRoles.EdgeEndFlag(q.Kind, useEnd: true)));
                }
            }
            TokenRoleFlags[] flags = TokenRoles.Label(starts, L, P, marks);
            TokenRole[] roles = flags.Select(TokenRoles.Primary).ToArray();
            int[] chunkOf = new int[L], offsetOf = new int[L];
            for (int k = 0; k < starts.Length; k++)
            {
                int end = k + 1 < starts.Length ? starts[k + 1] : L;
                for (int t = starts[k]; t < end; t++) { chunkOf[t] = k; offsetOf[t] = t - starts[k]; }
            }

            // Aggregates -----------------------------------------------------------------------------------------------------
            string meta = string.Join(',', contextId, o.CorpusName, o.Chunker, o.Seed.ToString(ci));
            int stale0 = starts.Length > 1 ? starts[1] : L; // first stale token
            int nStale = L - stale0;
            var groups = new List<(string Kind, string Name, Func<int, bool> Has)>();
            foreach (TokenRole r in Enum.GetValues<TokenRole>())
                groups.Add(("role", r.ToString(), t => roles[t] == r && (t >= L || t >= stale0)));
            foreach ((string name, TokenRoleFlags f) in Labels)
                groups.Add(("label", name, t => t >= stale0 && t < L && (flags[t] & f) != 0));
            groups.Add(("label", "def_in_cs16", t => t >= stale0 && t < L && (flags[t] & TokenRoleFlags.Definition) != 0 && offsetOf[t] < 16));
            groups.Add(("label", "def_after_cs16", t => t >= stale0 && t < L && (flags[t] & TokenRoleFlags.Definition) != 0 && offsetOf[t] >= 16));
            groups.Add(("label", "stale_all", t => t >= stale0 && t < L));
            groups.Add(("label", "chunk0", t => t < stale0));
            groups.Add(("label", "query_all", t => t >= L));
            foreach ((string name, int lo, int hi) in OffsetBins)
                groups.Add(("offset", name, t => t >= stale0 && t < L && offsetOf[t] >= lo && offsetOf[t] < hi));
            foreach (TokenRole r in new[] { TokenRole.DefinitionSite, TokenRole.UseSite, TokenRole.Import, TokenRole.Other })
                foreach ((string name, int lo, int hi) in CoarseBins)
                    groups.Add(("role_offset", $"{r}|{name}", t => t >= stale0 && t < L && roles[t] == r && offsetOf[t] >= lo && offsetOf[t] < hi));
            int[][] members = groups.Select(g => Enumerable.Range(0, L + P).Where(g.Has).ToArray()).ToArray();

            var sb = new StringBuilder();
            for (int l = 0; l < d.Layers; l++)
            {
                for (int gi = 0; gi < groups.Count; gi++)
                {
                    int[] m = members[gi];
                    if (m.Length == 0) continue;
                    double kl = 0, vl = 0, kvl = 0, kr = 0, vr = 0, kvr = 0, kn = 0, vn = 0, mx = 0;
                    foreach (int t in m)
                    {
                        kl += d.KeyL2(l, t); vl += d.ValueL2(l, t); kvl += d.CombinedL2(l, t);
                        kr += d.KeyRelative(l, t); vr += d.ValueRelative(l, t); kvr += d.CombinedRelative(l, t);
                        kn += d.KeyReferenceNorm(l, t); vn += d.ValueReferenceNorm(l, t);
                        mx = Math.Max(mx, d.CombinedL2(l, t));
                    }
                    sb.Append(meta).Append(',').Append(l.ToString(ci)).Append(',').Append(groups[gi].Kind).Append(',').Append(groups[gi].Name)
                        .Append(',').Append(m.Length.ToString(ci));
                    foreach (double x in new[] { kl, vl, kvl, kr, vr, kvr, kn, vn, mx }) sb.Append(',').Append(x.ToString("G6", ci));
                    sb.Append('\n');
                }
            }
            cellsCsv.Write(sb);

            // Capture over the stale context tokens, per layer.
            sb.Clear();
            int[] relStarts = starts.Skip(1).Select(s => s - stale0).ToArray();
            int[] posOrder = nStale > 0 ? TopKCapture.PositionOrder(relStarts, nStale) : [];
            foreach (string metric in new[] { "kv_l2", "kv_rel" })
            {
                float[][] values = new float[d.Layers][];
                for (int l = 0; l < d.Layers; l++)
                {
                    values[l] = new float[nStale];
                    for (int i = 0; i < nStale; i++)
                        values[l][i] = metric == "kv_l2" ? d.CombinedL2(l, stale0 + i) : d.CombinedRelative(l, stale0 + i);
                }
                foreach (double f in Fractions)
                {
                    int count = TopKCapture.Count(nStale, f);
                    int[] layer1Set = d.Layers > 1 ? TopKCapture.TopIndices(values[1], count) : [];
                    int[] posSet = posOrder[..count];
                    for (int l = 0; l < d.Layers; l++)
                    {
                        int[] top = TopKCapture.TopIndices(values[l], count);
                        double share = TopKCapture.ShareOf(values[l], top);
                        double rnd = TopKCapture.RandomShare(values[l], f, o.RandomDraws, HashCode.Combine(p, l, o.Seed));
                        double pos = TopKCapture.ShareOf(values[l], posSet);
                        double overlap = count == 0 ? double.NaN : (double)top.Intersect(posSet).Count() / count;
                        double l1 = d.Layers > 1 ? TopKCapture.ShareOf(values[l], layer1Set) : double.NaN;
                        sb.Append(meta).Append(',').Append(l.ToString(ci)).Append(',').Append(metric).Append(',')
                            .Append((f * 100).ToString("G3", ci)).Append(',').Append(nStale.ToString(ci));
                        foreach (double x in new[] { share, rnd, pos, overlap, l1 }) sb.Append(',').Append(x.ToString("G6", ci));
                        sb.Append('\n');
                    }
                }
            }
            captureCsv.Write(sb);

            if (rawCsv is not null && done < o.RawContexts)
            {
                sb.Clear();
                for (int l = 0; l < d.Layers; l++)
                    for (int t = 0; t < L + P; t++)
                    {
                        sb.Append(l.ToString(ci)).Append(',').Append(contextId).Append(',').Append(o.Seed.ToString(ci)).Append(',').Append(o.Chunker)
                            .Append(',').Append(t.ToString(ci)).Append(',').Append(t < L ? chunkOf[t].ToString(ci) : "q").Append(',')
                            .Append(t < L ? offsetOf[t].ToString(ci) : (t - L).ToString(ci)).Append(',').Append(roles[t]).Append(',')
                            .Append(LabelString(flags[t])).Append(',').Append(d.KeyL2(l, t).ToString("G5", ci)).Append(',')
                            .Append(d.ValueL2(l, t).ToString("G5", ci)).Append(',').Append(d.KeyRelative(l, t).ToString("G5", ci)).Append(',')
                            .Append(d.ValueRelative(l, t).ToString("G5", ci)).Append('\n');
                    }
                rawCsv.Write(sb);
            }

            // Context row + sanity ---------------------------------------------------------------------------------------------
            double chunk0Max = 0, layer0Rel = 0, deepQuery = 0;
            for (int l = 0; l < d.Layers; l++)
                for (int t = 0; t < stale0; t++) chunk0Max = Math.Max(chunk0Max, d.CombinedL2(l, t));
            for (int t = 0; t < L; t++) layer0Rel = Math.Max(layer0Rel, d.CombinedRelative(0, t));
            if (P > 0)
            {
                for (int t = L; t < L + P; t++) deepQuery += d.CombinedRelative(d.Layers - 1, t);
                deepQuery /= P;
            }
            int Count(TokenRole r) => roles.Count(x => x == r);
            int CountFlag(TokenRoleFlags f) => flags.Take(L).Count(x => (x & f) != 0);
            int defIn16 = Enumerable.Range(0, L).Count(t => (flags[t] & TokenRoleFlags.Definition) != 0 && offsetOf[t] < 16);
            double secs = ctxSw.Elapsed.TotalSeconds;
            contextsCsv.WriteLine(string.Join(',', meta, p.ToString(ci), inSample ? "1" : "0", keptFlag, L.ToString(ci), starts.Length.ToString(ci), P.ToString(ci),
                "\"" + task.Id + "\"", Count(TokenRole.ChunkStart), Count(TokenRole.DefinitionSite), Count(TokenRole.UseSite),
                Count(TokenRole.Import), Count(TokenRole.Other), Count(TokenRole.QuerySite), Count(TokenRole.Query), defIn16,
                CountFlag(TokenRoleFlags.CrossChunk), CountFlag(TokenRoleFlags.InContextCross), CountFlag(TokenRoleFlags.QueryTarget),
                queryMapOk ? "1" : "0", chunk0Max.ToString("G6", ci), layer0Rel.ToString("G6", ci), deepQuery.ToString("G6", ci),
                secs.ToString("F1", ci)));
            done++;
            Console.WriteLine($"[{contextId}{(keptFlag == "1" ? " kept" : "")} ctx {L} chunks {starts.Length} query {P}] {secs:F1} s; chunk0 max {chunk0Max:G3}, layer0 max rel {layer0Rel:G3}; "
                + $"roles cs {Count(TokenRole.ChunkStart)} def {Count(TokenRole.DefinitionSite)} use {Count(TokenRole.UseSite)} import {Count(TokenRole.Import)} "
                + $"qsite {Count(TokenRole.QuerySite)}; at {total.Elapsed.TotalMinutes:F1} min");
            if (chunk0Max != 0) Console.WriteLine($"  WARNING: first chunk deviates (max {chunk0Max:G6}); expected exactly 0");
        }
        Console.WriteLine($"e3 {o.CorpusName}/{o.Chunker}/s{o.Seed}: {done} contexts in {total.Elapsed.TotalMinutes:F1} min -> {stem}.*");
        return 0;
    }

    private static string LabelString(TokenRoleFlags f)
    {
        var parts = Labels.Where(x => (f & x.Flag) != 0).Select(x => x.Name).ToList();
        if ((f & TokenRoleFlags.Query) != 0) parts.Add("query");
        return parts.Count == 0 ? "" : string.Join('|', parts);
    }

    /// <summary>Character start of each query token in <see cref="CodeTask.Prefix"/> (cumulative decode); ok = the decode round-trips.</summary>
    private static int[] QueryCharStarts(ITokenizer tok, CodeTask task, out bool ok)
    {
        int[] ids = task.PrefixTokenIds;
        int[] starts = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) starts[i] = i == 0 ? 0 : tok.Decode(ids.AsSpan(0, i).ToArray()).Length;
        ok = tok.Decode(ids) == task.Prefix && starts.Zip(starts.Skip(1)).All(x => x.First <= x.Second);
        return starts;
    }

    private static void VerifyPlans(IReadOnlyList<CodeContextPlan> plans, string jsonl)
    {
        string[] lines = File.ReadAllLines(jsonl);
        if (lines.Length != plans.Count)
            throw new InvalidOperationException($"{jsonl} holds {lines.Length} plans; regenerated {plans.Count}.");
        for (int p = 0; p < lines.Length; p++)
        {
            using JsonDocument doc = JsonDocument.Parse(lines[p]);
            string[] ids = doc.RootElement.GetProperty("chunks").EnumerateArray().Select(x => x.GetString()!).ToArray();
            if (!ids.SequenceEqual(plans[p].Chunks.Select(c => c.Id)))
                throw new InvalidOperationException($"Plan {p}: regenerated chunks differ from {jsonl}.");
        }
        Console.WriteLine($"plans verified against {jsonl}: {lines.Length} plans, identical chunk ids");
    }

    /// <summary>Plan → kept task ids (Keep = 1) from a Task 8 validation CSV.</summary>
    private static Dictionary<int, List<string>> ReadKept(string csv)
    {
        var result = new Dictionary<int, List<string>>();
        foreach (string line in File.ReadLines(csv).Skip(1))
        {
            // Plan,"TaskId",ContextTokens,...,Keep,Reason,PlanSeconds,"Symbol": TaskId is the only quoted field before Keep.
            int q1 = line.IndexOf('"'), q2 = line.IndexOf('"', q1 + 1);
            int plan = int.Parse(line[..line.IndexOf(',')], CultureInfo.InvariantCulture);
            string id = line[(q1 + 1)..q2];
            string[] rest = line[(q2 + 2)..].Split(',');
            // rest: ContextTokens,WithoutTokens,PrefixTokens,MaskTokens,MaskNllWith,MaskNllWithout,ContNllWith,ContNllWithout,
            //       EmWith,EmWithout,GreedyRunWith,GreedyRunWithout,NllGain,Keep,...
            if (rest[13] == "1") (result.TryGetValue(plan, out List<string>? l) ? l : result[plan] = []).Add(id);
        }
        return result;
    }
}
