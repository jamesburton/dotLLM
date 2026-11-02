using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Engine.KvCache;

namespace ChunkKvLab;

/// <summary>
/// Phase S Task 3 check: on the E4 contexts, assembling a naive-concatenation cache from a <see cref="FileChunkStore"/> (cold:
/// misses encoded and stored; warm: a new store instance, every chunk read from disk) against the store-less path (fresh
/// <see cref="ChunkEncoder"/> encodes + <see cref="ChunkAssembler"/>). Reports per context the assembly times and hit/miss
/// counts and whether the three caches are bit-identical; per task the E4 metrics on the fresh and the warm cache and whether
/// their continuation logits are bit-identical.
/// </summary>
internal static class StoreAssemblyCheck
{
    public static int Run(E4Options o, string storeDir)
    {
        var ci = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = ci;
        var total = Stopwatch.StartNew();
        using LabModel lab = LabModel.Load(o.ModelPath, o.Threads);
        int vocab = lab.Config.VocabSize;
        Corpus corpus = TasksRunner.LoadCorpus(o.CorpusDir, o.Chunker, o.ChunkTokens, lab.Tokenizer, o.ExpectHash);
        IReadOnlyList<CodeContextPlan> plans = E4Command.SelectPlans(o, corpus);
        if (Directory.Exists(storeDir) && Directory.EnumerateFileSystemEntries(storeDir).Any())
            throw new InvalidOperationException($"store directory {storeDir} is not empty");
        string tokenizerId = lab.TokenizerId;
        var encoder = new ChunkEncoder(lab.Fingerprint);
        var assembler = new ChunkAssembler(lab.Config);
        var stored = StoredContextAssembler.ForModel(lab.Config, lab.ModelId, lab.Quant, tokenizerId);
        if (stored.ModelFingerprint != lab.Fingerprint) throw new InvalidOperationException("lab and assembler fingerprints differ");
        Console.WriteLine($"store {storeDir}; backend {StoredContextAssembler.Backend}, dtype F32, kv-output version {ChunkReuseSchema.KvOutputVersion}; "
            + $"threads {o.Threads}; tokenizer id {tokenizerId}");

        using var csv = new StreamWriter(o.OutCsv) { AutoFlush = true };
        csv.WriteLine("corpus,chunker,plan,task_id,context_tokens,chunks,fresh_ms,cold_ms,cold_hits,cold_misses,cold_repeats,cold_encoded_tokens,warm_ms,warm_hits,"
            + "warm_misses,warm_repeats,cache_cold_bit_exact,cache_warm_bit_exact,fresh_mask_nll,warm_mask_nll,fresh_mask_em,warm_mask_em,fresh_cont_nll,"
            + "warm_cont_nll,kl_cont_warm_vs_fresh,top1_cont_warm_vs_fresh,logits_bit_exact,max_abs_logit_diff");

        int contexts = 0, cacheMismatches = 0, logitMismatches = 0, tasksDone = 0;
        double freshSum = 0, coldSum = 0, warmSum = 0;
        for (int p = 0; p < plans.Count; p++)
        {
            CodeContextPlan plan = plans[p];
            int L = plan.ContextTokenIds.Length;
            int capacity = L + plan.Tasks.Max(t => t.PrefixTokenIds.Length + t.ContinuationTokenIds.Length) + 1;

            using SimpleKvCache freshCache = lab.NewCache(capacity); // allocated outside the timing, like cold and warm
            var sw = Stopwatch.StartNew();
            var fresh = plan.Chunks.Select(c => encoder.Encode(lab.Model, c.TokenIds)).ToList();
            assembler.Assemble(freshCache, fresh, lab.Fingerprint);
            double freshMs = sw.Elapsed.TotalMilliseconds;
            fresh.Clear(); // free the entries before the next copies

            StoredAssembly cold;
            double coldMs;
            bool coldExact;
            using (SimpleKvCache coldCache = lab.NewCache(capacity))
            {
                sw.Restart();
                cold = stored.Assemble(plan.Chunks, lab.Model, new FileChunkStore(storeDir), coldCache);
                coldMs = sw.Elapsed.TotalMilliseconds;
                coldExact = SameRows(freshCache, coldCache, L);
            }
            cold = cold with { Entries = [] };

            using SimpleKvCache warmCache = lab.NewCache(capacity);
            sw.Restart();
            StoredAssembly warm = stored.Assemble(plan.Chunks, lab.Model, new FileChunkStore(storeDir), warmCache);
            double warmMs = sw.Elapsed.TotalMilliseconds;
            bool warmExact = SameRows(freshCache, warmCache, L);
            if (!coldExact || !warmExact) cacheMismatches++;
            contexts++;
            freshSum += freshMs;
            coldSum += coldMs;
            warmSum += warmMs;
            Console.WriteLine($"plan {p}: L={L}, {plan.Chunks.Count} chunks; fresh {freshMs:F0} ms; cold {coldMs:F0} ms ({cold.Stats}); "
                + $"warm {warmMs:F0} ms ({warm.Stats}); caches bit-exact cold={coldExact} warm={warmExact}");

            foreach (CodeTask task in plan.Tasks)
            {
                float[] freshRows = E4Command.ForwardTask(lab, freshCache, L, task);
                float[] warmRows = E4Command.ForwardTask(lab, warmCache, L, task);
                E4Command.Eval ef = E4Command.Score(freshRows, freshRows, task, vocab, 0);
                E4Command.Eval ew = E4Command.Score(warmRows, freshRows, task, vocab, 0);
                bool same = MemoryMarshal.AsBytes(freshRows.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(warmRows.AsSpan()));
                float maxDiff = 0;
                for (int i = 0; i < freshRows.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(freshRows[i] - warmRows[i]));
                if (!same) logitMismatches++;
                tasksDone++;
                csv.WriteLine(string.Join(',', o.CorpusName, o.Chunker, p.ToString(ci), "\"" + task.Id + "\"", L.ToString(ci),
                    plan.Chunks.Count.ToString(ci), freshMs.ToString("F0", ci), coldMs.ToString("F0", ci), cold.Stats.Hits.ToString(ci),
                    cold.Stats.Misses.ToString(ci), cold.Stats.Repeats.ToString(ci), cold.Stats.EncodedTokens.ToString(ci), warmMs.ToString("F0", ci),
                    warm.Stats.Hits.ToString(ci), warm.Stats.Misses.ToString(ci), warm.Stats.Repeats.ToString(ci), coldExact ? "1" : "0", warmExact ? "1" : "0", ef.MaskNll.ToString("G9", ci),
                    ew.MaskNll.ToString("G9", ci), ef.MaskEm ? "1" : "0", ew.MaskEm ? "1" : "0", ef.ContNll.ToString("G9", ci),
                    ew.ContNll.ToString("G9", ci), ew.KlCont.ToString("G6", ci), ew.Top1Cont.ToString("G6", ci), same ? "1" : "0",
                    maxDiff.ToString("G6", ci)));
            }
        }
        var final = new FileChunkStore(storeDir);
        Console.WriteLine($"done: {contexts} contexts, {tasksDone} tasks in {total.Elapsed.TotalMinutes:F1} min; cache mismatches {cacheMismatches}, "
            + $"logit mismatches {logitMismatches}; mean assembly fresh {freshSum / Math.Max(1, contexts):F0} ms, cold {coldSum / Math.Max(1, contexts):F0} ms, "
            + $"warm {warmSum / Math.Max(1, contexts):F0} ms; store {final.Keys().Count()} entries, {final.TotalBytes} bytes");
        return cacheMismatches == 0 && logitMismatches == 0 ? 0 : 1;
    }

    private static bool SameRows(SimpleKvCache a, SimpleKvCache b, int rows)
    {
        for (int l = 0; l < a.NumLayers; l++)
        {
            int n = rows * a.KvStrideOf(l);
            if (!MemoryMarshal.AsBytes(a.KeysSpan(l)[..n]).SequenceEqual(MemoryMarshal.AsBytes(b.KeysSpan(l)[..n]))) return false;
            if (!MemoryMarshal.AsBytes(a.ValuesSpan(l)[..n]).SequenceEqual(MemoryMarshal.AsBytes(b.ValuesSpan(l)[..n]))) return false;
        }
        return true;
    }
}
