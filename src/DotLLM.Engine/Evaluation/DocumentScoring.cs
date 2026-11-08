namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Per-position maths for per-document scoring (<c>dotllm perplexity --docs-file</c>, #1706).
/// </summary>
/// <remarks>
/// <b>Precision:</b> every term is formed and summed in <see cref="double"/>. The float32 logits are widened first
/// (<c>(double)x - (double)max</c> is exact), the log-sum-exp is <c>max + log(sum exp(x - max))</c> with the max shift
/// so no <c>exp</c> argument is positive, and per-position values are accumulated sequentially in double. Nothing here
/// depends on a thread count, so equal logits give a bit-equal result. All logs are natural (nats).
/// </remarks>
public static class DocumentScoring
{
    /// <summary>Natural-log log-sum-exp of one logit row, accumulated in double.</summary>
    /// <param name="logits">Unnormalised scores.</param>
    public static double LogSumExp(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty) throw new ArgumentException("Empty logit row.", nameof(logits));
        float max = logits[0];
        for (int j = 1; j < logits.Length; j++)
            if (logits[j] > max) max = logits[j];

        double sum = 0;
        for (int j = 0; j < logits.Length; j++)
            sum += Math.Exp((double)logits[j] - max);
        return max + Math.Log(sum);
    }

    /// <summary><c>log softmax(logits)[target]</c> in nats (double).</summary>
    /// <param name="logits">Unnormalised scores.</param>
    /// <param name="target">Index whose log-probability is returned.</param>
    public static double TargetLogProb(ReadOnlySpan<float> logits, int target)
    {
        if ((uint)target >= (uint)logits.Length) throw new ArgumentOutOfRangeException(nameof(target));
        return logits[target] - LogSumExp(logits);
    }

    /// <summary>
    /// <c>KL(teacher || model) = sum_v p_v (log p_v - log q_v)</c> for one position, where <c>p = exp(teacherLogProbs)</c>
    /// is the reference distribution over the FULL vocabulary and <c>q = softmax(modelLogits)</c>.
    /// Terms with <c>p_v = 0</c> contribute 0. Returns nats (double).
    /// </summary>
    /// <param name="teacherLogProbs">Reference log-probabilities, one per vocab entry.</param>
    /// <param name="modelLogits">Model logits, same length.</param>
    public static double KlFull(ReadOnlySpan<float> teacherLogProbs, ReadOnlySpan<float> modelLogits)
    {
        if (teacherLogProbs.Length != modelLogits.Length)
            throw new ArgumentException("Teacher and model rows must have the same vocabulary size.");
        double lse = LogSumExp(modelLogits);
        double kl = 0;
        for (int v = 0; v < modelLogits.Length; v++)
        {
            double lp = teacherLogProbs[v];
            if (double.IsNegativeInfinity(lp)) continue;
            double lq = modelLogits[v] - lse;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }

    /// <summary>
    /// Lower bound on <c>KL(teacher || model)</c> from a top-K + tail teacher: the exact sum over the K listed ids plus
    /// <c>T log(T / Q)</c> for the lumped tail (<c>T</c> = teacher mass outside the K ids, <c>Q</c> = model mass outside them).
    /// By the log-sum inequality this never exceeds the full-vocabulary KL, and equals it when the teacher's tail is distributed
    /// like the model's. Returns nats (double).
    /// </summary>
    /// <param name="ids">The teacher's top-K token ids (distinct).</param>
    /// <param name="teacherLogProbs">Teacher log-probabilities of <paramref name="ids"/>.</param>
    /// <param name="tailLogProb">Teacher log mass outside <paramref name="ids"/> (<c>-inf</c> for none).</param>
    /// <param name="modelLogits">Model logits over the full vocabulary.</param>
    public static double KlTopKTail(
        ReadOnlySpan<int> ids, ReadOnlySpan<float> teacherLogProbs, double tailLogProb, ReadOnlySpan<float> modelLogits)
    {
        if (ids.Length != teacherLogProbs.Length)
            throw new ArgumentException("ids and teacherLogProbs must have the same length.");
        float max = modelLogits[0];
        for (int j = 1; j < modelLogits.Length; j++)
            if (modelLogits[j] > max) max = modelLogits[j];

        double all = 0;
        for (int j = 0; j < modelLogits.Length; j++) all += Math.Exp((double)modelLogits[j] - max);
        double lse = max + Math.Log(all);

        double kl = 0;
        double topModelShifted = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            if ((uint)ids[i] >= (uint)modelLogits.Length)
                throw new ArgumentOutOfRangeException(nameof(ids),
                    $"Reference token id {ids[i]} is outside the model vocabulary of {modelLogits.Length}.");
            topModelShifted += Math.Exp((double)modelLogits[ids[i]] - max);
            double lp = teacherLogProbs[i];
            if (double.IsNegativeInfinity(lp)) continue;
            kl += Math.Exp(lp) * (lp - (modelLogits[ids[i]] - lse));
        }

        if (!double.IsNegativeInfinity(tailLogProb))
        {
            double t = Math.Exp(tailLogProb);
            double q = Math.Max((all - topModelShifted) / all, 1e-300);
            if (t > 0) kl += t * (tailLogProb - Math.Log(q));
        }
        return kl;
    }
}
