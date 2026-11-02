using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>Settings of the <c>tasks</c> command.</summary>
internal sealed record TasksOptions(
    string ModelPath, string CorpusDir, string Chunker, int ChunkTokens, int Count, int Seed, TaskGeneratorOptions Generator,
    bool Validate, double MaxMinutes, int Threads, string OutPrefix, string? ExpectHash);

/// <summary>
/// Task 8: loads a code corpus, chunks it, generates cross-file completion tasks (plans of one context + several tasks) and,
/// with <c>--validate 1</c>, runs the full-prefill reference with and without the required chunk on CPU (TaskValidator).
/// Writes <c>&lt;out&gt;.plans.jsonl</c> and <c>&lt;out&gt;.validation.csv</c>.
/// </summary>
internal static class TasksRunner
{
    public static IChunker MakeChunker(string name, ITokenizer tokenizer, int tokens) => name switch
    {
        "symbol" => new SymbolChunker(tokenizer, tokens),
        "fixed" => new FixedSizeChunker(tokenizer, tokens),
        "fixed-text" => new TextFirstFixedChunker(tokenizer, tokens),
        _ => throw new ArgumentException($"unknown chunker '{name}' (symbol|fixed|fixed-text)"),
    };

    /// <summary>Loads the corpus and prints its census (files, exclusions, chunks, tokens, hash, tokenizer round trip).</summary>
    public static Corpus LoadCorpus(string dir, string chunker, int tokens, ITokenizer tokenizer, string? expectHash)
    {
        var sw = Stopwatch.StartNew();
        Corpus corpus = Corpus.FromDirectory(dir, MakeChunker(chunker, tokenizer, tokens), tokenizer);
        int roundTrip = corpus.Files.Count(f => tokenizer.Decode(tokenizer.EncodeRaw(f.Text)) == f.Text);
        var byReason = corpus.Excluded.GroupBy(e => e.Reason).Select(g => $"{g.Key} {g.Count()}");
        int[] sizes = corpus.Chunks.Select(c => c.TokenIds.Length).Order().ToArray();
        Console.WriteLine($"corpus {dir}: {corpus.Files.Count} files ({(corpus.Excluded.Count == 0 ? "0" : string.Join(", ", byReason))} excluded), {corpus.Chunker.Name}: "
            + $"{corpus.Chunks.Count} chunks, {corpus.TotalTokens} tokens (chunk tokens median {Median(sizes)}, max {(sizes.Length > 0 ? sizes[^1] : 0)}); "
            + $"Decode(Encode(file)) == file for {roundTrip}/{corpus.Files.Count}; {sw.Elapsed.TotalSeconds:F1} s");
        string hash = corpus.ChunkIdHash;
        Console.WriteLine($"chunk-id sha256 {hash}");
        if (expectHash is not null && !string.Equals(expectHash, hash, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"corpus hash {hash} differs from --expect-hash {expectHash}");
        return corpus;
    }

    private static int Median(int[] sorted) => sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];

    public static int Run(TasksOptions o)
    {
        var total = Stopwatch.StartNew();
        LabModel? lab = o.Validate ? LabModel.Load(o.ModelPath, o.Threads) : null;
        ITokenizer tokenizer;
        DotLLM.Models.Gguf.GgufFile? gguf = null;
        if (lab is not null) tokenizer = lab.Tokenizer;
        else
        {
            gguf = DotLLM.Models.Gguf.GgufFile.Open(o.ModelPath);
            tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        }
        try
        {
            Corpus corpus = LoadCorpus(o.CorpusDir, o.Chunker, o.ChunkTokens, tokenizer, o.ExpectHash);
            var gen = new CrossFileTaskGenerator(o.Generator);
            var sw = Stopwatch.StartNew();
            CallSiteCounts counts = gen.CountCallSites(corpus);
            Console.WriteLine($"call sites ({sw.Elapsed.TotalSeconds:F1} s): {counts}");
            sw.Restart();
            IReadOnlyList<CodeContextPlan> plans = gen.GeneratePlans(corpus, o.Count, o.Seed);
            CodeTask[] tasks = plans.SelectMany(p => p.Tasks).ToArray();
            Console.WriteLine($"generated {tasks.Length} tasks in {plans.Count} plans ({sw.Elapsed.TotalSeconds:F1} s); options {o.Generator}; "
                + $"{tasks.Select(t => t.RequiredChunks[0].Id).Distinct().Count()} distinct definition chunks, "
                + $"{tasks.Select(t => t.CallSitePath).Distinct().Count()} calling files; mean context "
                + $"{plans.Average(p => p.ContextTokenIds.Length):F0} tokens, mean {plans.Average(p => p.Chunks.Count):F1} chunks, "
                + $"mean prefix {tasks.Average(t => t.PrefixTokenIds.Length):F0}, mask {tasks.Average(t => t.MaskTokenCount):F1} tokens");
            int consistent = tasks.Count(t => tokenizer.EncodeRaw(t.Prefix + t.ExpectedContinuation)
                .AsSpan().StartsWith([.. t.PrefixTokenIds, .. t.ContinuationTokenIds.AsSpan(0, t.MaskTokenCount)]));
            Console.WriteLine($"tokenisation at the mask boundary: Encode(prefix+continuation) starts with Encode(prefix)++masked ids for {consistent}/{tasks.Length}");
            WritePlans(o.OutPrefix + ".plans.jsonl", plans);

            if (lab is null) return 0;
            Validate(lab, plans, o, total);
            return 0;
        }
        finally
        {
            lab?.Dispose();
            gguf?.Dispose();
        }
    }

    private static void Validate(LabModel lab, IReadOnlyList<CodeContextPlan> plans, TasksOptions o, Stopwatch total)
    {
        var ci = CultureInfo.InvariantCulture;
        var validator = new TaskValidator(new ModelTaskScorer(lab.Model, cap => lab.NewCache(cap)));
        using var csv = new StreamWriter(o.OutPrefix + ".validation.csv") { AutoFlush = true };
        csv.WriteLine("Plan,TaskId,ContextTokens,WithoutTokens,PrefixTokens,MaskTokens,MaskNllWith,MaskNllWithout,ContNllWith,ContNllWithout,"
            + "EmWith,EmWithout,GreedyRunWith,GreedyRunWithout,NllGain,Keep,Reason,PlanSeconds,Symbol");
        var all = new List<TaskValidation>();
        int done = 0;
        for (int p = 0; p < plans.Count; p++)
        {
            if (o.MaxMinutes > 0 && total.Elapsed.TotalMinutes >= o.MaxMinutes)
            {
                Console.WriteLine($"time cap {o.MaxMinutes} min reached: validated {done} of {plans.Count} plans");
                break;
            }
            var sw = Stopwatch.StartNew();
            IReadOnlyList<TaskValidation> v = validator.Validate(plans[p]);
            double secs = sw.Elapsed.TotalSeconds;
            foreach (TaskValidation x in v)
            {
                csv.WriteLine(string.Join(',', p.ToString(ci), Quote(x.Task.Id), plans[p].ContextTokenIds.Length.ToString(ci),
                    plans[p].ContextWithoutRequired.Length.ToString(ci), x.Task.PrefixTokenIds.Length.ToString(ci), x.Task.MaskTokenCount.ToString(ci),
                    x.With.MaskNll.ToString("G6", ci), x.Without.MaskNll.ToString("G6", ci), x.With.ContinuationNll.ToString("G6", ci),
                    x.Without.ContinuationNll.ToString("G6", ci), x.With.MaskExactMatch ? "1" : "0", x.Without.MaskExactMatch ? "1" : "0",
                    x.With.GreedyMatchedTokens.ToString(ci), x.Without.GreedyMatchedTokens.ToString(ci), x.NllGain.ToString("G6", ci),
                    x.Keep ? "1" : "0", x.Reason, secs.ToString("F1", ci), Quote(x.Task.Symbol)));
            }
            all.AddRange(v);
            done++;
            Console.WriteLine($"[plan {p} ctx {plans[p].ContextTokenIds.Length} tasks {v.Count}] {secs:F1} s; "
                + string.Join("  ", v.Select(x => $"{x.Reason} (EM {(x.With.MaskExactMatch ? 1 : 0)}/{(x.Without.MaskExactMatch ? 1 : 0)}, "
                    + $"NLL {x.With.MaskNll:F2}/{x.Without.MaskNll:F2})")) + $"  at {total.Elapsed.TotalMinutes:F1} min");
        }
        if (all.Count == 0) return;
        Console.WriteLine($"validated {all.Count} tasks in {done} plans: reference EM with chunk {all.Count(x => x.With.MaskExactMatch)}/{all.Count} "
            + $"({100.0 * all.Count(x => x.With.MaskExactMatch) / all.Count:F0} %), without {all.Count(x => x.Without.MaskExactMatch)}/{all.Count}; "
            + $"mean mask NLL with {all.Average(x => x.With.MaskNll):F3} / without {all.Average(x => x.Without.MaskNll):F3} nats; "
            + $"kept {all.Count(x => x.Keep)}/{all.Count} ({100.0 * all.Count(x => x.Keep) / all.Count:F0} %); reasons "
            + string.Join(", ", all.GroupBy(x => x.Reason).OrderBy(g => g.Key).Select(g => $"{g.Key} {g.Count()}"))
            + $"; {total.Elapsed.TotalMinutes:F1} min total");
    }

    private static string Quote(string s) => "\"" + s.Replace("\"", "'", StringComparison.Ordinal) + "\"";

    private static void WritePlans(string path, IReadOnlyList<CodeContextPlan> plans)
    {
        using var w = new StreamWriter(path);
        for (int p = 0; p < plans.Count; p++)
        {
            CodeContextPlan plan = plans[p];
            int L = plan.ContextTokenIds.Length;
            var obj = new
            {
                plan = p,
                seed = plan.Seed,
                contextTokens = L,
                chunks = plan.Chunks.Select(c => c.Id).ToArray(),
                chunkStarts = plan.ChunkStarts,
                required = plan.RequiredIds.Order(StringComparer.Ordinal).ToArray(),
                tasks = plan.Tasks.Select(t => new
                {
                    id = t.Id,
                    symbol = t.Symbol,
                    callSite = t.CallSitePath,
                    line = t.CallSiteLine,
                    maskedText = t.MaskedText,
                    prefix = t.Prefix,
                    expectedContinuation = t.ExpectedContinuation,
                    prefixTokens = t.PrefixTokenIds.Length,
                    continuationTokens = t.ContinuationTokenIds.Length,
                    maskTokenOffset = t.MaskTokenOffset,
                    maskTokenOffsetInContext = L + t.MaskTokenOffset,
                    maskTokens = t.MaskTokenCount,
                    requiredChunks = t.RequiredChunks.Select(c => c.Id).ToArray(),
                    distractorChunks = t.DistractorChunks.Select(c => c.Id).ToArray(),
                }).ToArray(),
            };
            w.WriteLine(JsonSerializer.Serialize(obj));
        }
    }
}
