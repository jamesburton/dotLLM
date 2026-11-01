using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>One task's query: prefix ids, expected continuation ids, and how many leading continuation tokens are masked.</summary>
public sealed record TaskQuery(int[] Prefix, int[] Continuation, int MaskTokens);

/// <summary>Teacher-forced score of one task's continuation after a context.</summary>
/// <param name="MaskNll">Sum of −log p over the masked tokens (nats).</param>
/// <param name="ContinuationNll">Sum of −log p over the whole continuation (nats).</param>
/// <param name="MaskExactMatch">Every masked token is the argmax given the true tokens before it — equivalently, greedy
/// decoding reproduces the masked call exactly.</param>
/// <param name="MaskTokens">Masked token count.</param>
/// <param name="GreedyMatchedTokens">Length of the leading run of continuation tokens that greedy decoding reproduces.</param>
public sealed record TaskScore(double MaskNll, double ContinuationNll, bool MaskExactMatch, int MaskTokens, int GreedyMatchedTokens);

/// <summary>Scores task queries after a context (one context prefill shared by all queries).</summary>
public interface ITaskScorer
{
    /// <summary>Scores each query forwarded right after <paramref name="context"/> (queries do not see each other).</summary>
    IReadOnlyList<TaskScore> Score(ReadOnlySpan<int> context, IReadOnlyList<TaskQuery> queries);
}

/// <summary>A task's validity verdict: the full-prefill reference with and without the required chunk(s).</summary>
/// <param name="Task">The task.</param>
/// <param name="With">Score with the full context.</param>
/// <param name="Without">Score with the required chunk(s) removed.</param>
/// <param name="Keep">Kept for evaluation.</param>
/// <param name="Reason"><c>kept</c>, <c>reference-fails</c>, <c>solved-without</c> or <c>below-margin</c>.</param>
public sealed record TaskValidation(CodeTask Task, TaskScore With, TaskScore Without, bool Keep, string Reason)
{
    /// <summary>Masked-region NLL drop from adding the required chunk (nats; positive = the chunk helps).</summary>
    public double NllGain => Without.MaskNll - With.MaskNll;
}

/// <summary>
/// Validity filter in the spirit of Boxoffice (arXiv 2609.31415): a task is kept only when the full-prefill reference
/// (1) solves it with the required chunk (greedy exact match on the masked call; drops "B0" items the reference itself
/// fails), (2) does not solve it without the chunk (drops "NC" items answerable without it), and (3) the masked-call NLL
/// falls by at least <c>marginNats</c> when the chunk is present (default ln 2: the call is at least twice as likely with
/// the definition in context, so the chunk materially matters even when greedy decoding is brittle).
/// </summary>
public sealed class TaskValidator
{
    private const double Tolerance = 1e-9;
    private readonly ITaskScorer _scorer;
    private readonly double _margin;

    /// <summary>Creates the validator.</summary>
    public TaskValidator(ITaskScorer scorer, double marginNats = 0.69314718055994530942)
    {
        ArgumentNullException.ThrowIfNull(scorer);
        ArgumentOutOfRangeException.ThrowIfNegative(marginNats);
        _scorer = scorer;
        _margin = marginNats;
    }

    /// <summary>Scores every task of <paramref name="plan"/> with and without its required chunk(s) and decides.</summary>
    public IReadOnlyList<TaskValidation> Validate(CodeContextPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        TaskQuery[] queries = plan.Tasks.Select(t => new TaskQuery(t.PrefixTokenIds, t.ContinuationTokenIds, t.MaskTokenCount)).ToArray();
        IReadOnlyList<TaskScore> with = _scorer.Score(plan.ContextTokenIds, queries);
        IReadOnlyList<TaskScore> without = _scorer.Score(plan.ContextWithoutRequired, queries);
        var result = new TaskValidation[queries.Length];
        for (int i = 0; i < result.Length; i++)
        {
            string reason = !with[i].MaskExactMatch ? "reference-fails"
                : without[i].MaskExactMatch ? "solved-without"
                : without[i].MaskNll - with[i].MaskNll < _margin - Tolerance ? "below-margin"
                : "kept";
            result[i] = new TaskValidation(plan.Tasks[i], with[i], without[i], reason == "kept", reason);
        }
        return result;
    }
}

/// <summary>
/// <see cref="ITaskScorer"/> over an <see cref="IModel"/>: the context is prefilled once into a fresh cache; each query is
/// forwarded after it (prefix minus its last token as a prefill, then one forward of [last prefix token, continuation[..^1]]
/// whose logit rows predict the continuation) and the cache is rolled back to the context.
/// </summary>
public sealed unsafe class ModelTaskScorer : ITaskScorer
{
    private readonly IModel _model;
    private readonly Func<int, IKvCache> _newCache;
    private readonly int _block;

    /// <summary>Creates the scorer.</summary>
    /// <param name="model">The model (full prefill reference).</param>
    /// <param name="newCache">Creates a cache with the given capacity in tokens.</param>
    /// <param name="prefillBlock">Prefill block size in tokens (default 512).</param>
    public ModelTaskScorer(IModel model, Func<int, IKvCache> newCache, int prefillBlock = 512)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(newCache);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefillBlock);
        _model = model;
        _newCache = newCache;
        _block = prefillBlock;
    }

    /// <inheritdoc />
    public IReadOnlyList<TaskScore> Score(ReadOnlySpan<int> context, IReadOnlyList<TaskQuery> queries)
    {
        ArgumentNullException.ThrowIfNull(queries);
        int L = context.Length;
        int longest = queries.Count == 0 ? 0 : queries.Max(q => q.Prefix.Length + q.Continuation.Length);
        using IKvCache cache = _newCache(L + longest + 1);
        Prefill(cache, context, 0);

        var scores = new TaskScore[queries.Count];
        for (int q = 0; q < scores.Length; q++)
        {
            TaskQuery query = queries[q];
            if (query.Prefix.Length == 0 || query.Continuation.Length == 0)
                throw new ArgumentException($"Query {q} needs a non-empty prefix and continuation.");
            Prefill(cache, query.Prefix.AsSpan(0, query.Prefix.Length - 1), L);
            int n = query.Continuation.Length;
            int[] ids = new int[n];
            ids[0] = query.Prefix[^1];
            query.Continuation.AsSpan(0, n - 1).CopyTo(ids.AsSpan(1));
            int first = L + query.Prefix.Length - 1;
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = first + i;

            double maskNll = 0, nll = 0;
            bool maskEm = true;
            int greedyRun = 0;
            bool running = true;
            using (ITensor logits = _model.Forward(ids, positions, -1, cache))
            {
                int vocab = logits.Shape[1];
                for (int j = 0; j < n; j++)
                {
                    var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)j * vocab, vocab);
                    int target = query.Continuation[j];
                    double tokenNll = LogSumExp(row) - row[target];
                    bool hit = ArgMax(row) == target;
                    nll += tokenNll;
                    if (j < query.MaskTokens)
                    {
                        maskNll += tokenNll;
                        maskEm &= hit;
                    }
                    if (running && hit) greedyRun++;
                    else running = false;
                }
            }
            cache.Rollback(L);
            scores[q] = new TaskScore(maskNll, nll, maskEm, query.MaskTokens, greedyRun);
        }
        return scores;
    }

    private void Prefill(IKvCache cache, ReadOnlySpan<int> ids, int firstPosition)
    {
        for (int off = 0; off < ids.Length; off += _block)
        {
            int n = Math.Min(_block, ids.Length - off);
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = firstPosition + off + i;
            using ITensor _ = _model.Forward(ids.Slice(off, n), positions, -1, cache, lastTokenLogitsOnly: true);
        }
    }

    private static double LogSumExp(ReadOnlySpan<float> row)
    {
        float max = float.NegativeInfinity;
        foreach (float v in row) if (v > max) max = v;
        double sum = 0;
        foreach (float v in row) sum += Math.Exp(v - max);
        return max + Math.Log(sum);
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }
}
