namespace ChunkKvLab;

/// <summary>Quality metrics of a method's logits against the full-prefill reference.</summary>
public static class Metrics
{
    /// <summary>
    /// KL(P_ref ‖ P) in nats, where P_ref = softmax(<paramref name="referenceLogits"/>) and P = softmax(<paramref name="logits"/>);
    /// computed in double with log-sum-exp.
    /// </summary>
    public static double KlDivergence(ReadOnlySpan<float> referenceLogits, ReadOnlySpan<float> logits)
    {
        if (referenceLogits.Length != logits.Length || logits.IsEmpty)
            throw new ArgumentException($"Logit rows differ in length ({referenceLogits.Length} vs {logits.Length}) or are empty.");
        double lseRef = LogSumExp(referenceLogits);
        double lse = LogSumExp(logits);
        double kl = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double logP = referenceLogits[i] - lseRef;
            double p = Math.Exp(logP);
            if (p == 0) continue;
            kl += p * (logP - (logits[i] - lse));
        }
        return Math.Max(0, kl); // rounding can give -1e-17 for identical rows
    }

    /// <summary>Mean per-row KL over row-major <c>[rows, vocab]</c> buffers.</summary>
    public static double MeanKl(ReadOnlySpan<float> referenceRows, ReadOnlySpan<float> rows, int vocab)
    {
        int n = RowCount(referenceRows, rows, vocab);
        double sum = 0;
        for (int r = 0; r < n; r++)
            sum += KlDivergence(referenceRows.Slice(r * vocab, vocab), rows.Slice(r * vocab, vocab));
        return sum / n;
    }

    /// <summary>Index of the largest value; ties go to the lowest index.</summary>
    public static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>Fraction of rows whose argmax equals the reference row's argmax.</summary>
    public static double Top1Agreement(ReadOnlySpan<float> referenceRows, ReadOnlySpan<float> rows, int vocab)
    {
        int n = RowCount(referenceRows, rows, vocab);
        int agree = 0;
        for (int r = 0; r < n; r++)
            if (ArgMax(referenceRows.Slice(r * vocab, vocab)) == ArgMax(rows.Slice(r * vocab, vocab))) agree++;
        return (double)agree / n;
    }

    /// <summary>
    /// Greedy exact match: length of the common leading run of <paramref name="candidate"/> and <paramref name="reference"/>
    /// divided by the reference length (1.0 = all N tokens identical).
    /// </summary>
    public static double ExactMatch(ReadOnlySpan<int> reference, ReadOnlySpan<int> candidate)
    {
        if (reference.IsEmpty) throw new ArgumentException("Empty reference.", nameof(reference));
        int n = Math.Min(reference.Length, candidate.Length);
        int k = 0;
        while (k < n && reference[k] == candidate[k]) k++;
        return (double)k / reference.Length;
    }

    private static int RowCount(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int vocab)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);
        if (a.Length != b.Length || a.Length % vocab != 0 || a.IsEmpty)
            throw new ArgumentException($"Row buffers must be equal, non-empty multiples of vocab {vocab} ({a.Length} vs {b.Length}).");
        return a.Length / vocab;
    }

    private static double LogSumExp(ReadOnlySpan<float> x)
    {
        double max = double.NegativeInfinity;
        foreach (float v in x) if (v > max) max = v;
        double s = 0;
        foreach (float v in x) s += Math.Exp(v - max);
        return max + Math.Log(s);
    }
}
