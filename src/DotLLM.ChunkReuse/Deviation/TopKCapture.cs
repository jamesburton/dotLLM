namespace DotLLM.ChunkReuse.Deviation;

/// <summary>
/// Concentration of a non-negative per-token quantity (E3: one layer's deviation over the stale tokens): the share of the
/// total held by the top-k% tokens, by an arbitrary token set (baselines), and the two baseline orders (random, position in chunk).
/// </summary>
public static class TopKCapture
{
    /// <summary>Tokens in a k% set: <c>round(n · fraction)</c> away from zero, at least 1 for a positive fraction, at most n.</summary>
    public static int Count(int n, double fraction)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        if (fraction < 0 || fraction > 1) throw new ArgumentOutOfRangeException(nameof(fraction));
        if (n == 0 || fraction == 0) return 0;
        int k = (int)Math.Round(n * fraction, MidpointRounding.AwayFromZero);
        return Math.Clamp(k, 1, n);
    }

    /// <summary>Indices of the <paramref name="count"/> largest values, largest first; ties by lower index.</summary>
    public static int[] TopIndices(ReadOnlySpan<float> values, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, values.Length);
        int[] idx = new int[values.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        float[] v = values.ToArray();
        Array.Sort(idx, (a, b) =>
        {
            int d = v[b].CompareTo(v[a]);
            return d != 0 ? d : a.CompareTo(b);
        });
        return idx[..count];
    }

    /// <summary>Share of Σ values held by the top <see cref="Count"/>(n, fraction) values; 0 when the total is 0.</summary>
    public static double Share(ReadOnlySpan<float> values, double fraction)
        => ShareOf(values, TopIndices(values, Count(values.Length, fraction)));

    /// <summary>Share of Σ values held by the tokens in <paramref name="indices"/> (distinct); 0 when the total is 0.</summary>
    public static double ShareOf(ReadOnlySpan<float> values, ReadOnlySpan<int> indices)
    {
        double total = 0, part = 0;
        foreach (float x in values) total += x;
        foreach (int i in indices) part += values[i];
        return total > 0 ? part / total : 0;
    }

    /// <summary>Mean share held by a uniformly random k% set over <paramref name="draws"/> seeded draws.</summary>
    public static double RandomShare(ReadOnlySpan<float> values, double fraction, int draws, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(draws);
        int n = values.Length, k = Count(n, fraction);
        var rng = new Random(seed);
        int[] perm = new int[n];
        double sum = 0;
        for (int d = 0; d < draws; d++)
        {
            for (int i = 0; i < n; i++) perm[i] = i;
            for (int i = 0; i < k; i++) // partial Fisher-Yates
            {
                int j = rng.Next(i, n);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }
            sum += ShareOf(values, perm.AsSpan(0, k));
        }
        return sum / draws;
    }

    /// <summary>
    /// Position-in-chunk order over <c>[0, length)</c>: every chunk's offset 0 (in chunk order), then every offset 1, and so on.
    /// Its top-k% prefix is the set a selector would pick from chunk position alone (equal chunk prefixes).
    /// </summary>
    /// <param name="chunkStarts">Ascending chunk starts relative to the range; the first is 0.</param>
    /// <param name="length">Range length.</param>
    public static int[] PositionOrder(IReadOnlyList<int> chunkStarts, int length)
    {
        ArgumentNullException.ThrowIfNull(chunkStarts);
        var keyed = new (int Offset, int Chunk, int Token)[length];
        for (int c = 0; c < chunkStarts.Count; c++)
        {
            int end = c + 1 < chunkStarts.Count ? chunkStarts[c + 1] : length;
            for (int t = chunkStarts[c]; t < end; t++) keyed[t] = (t - chunkStarts[c], c, t);
        }
        Array.Sort(keyed);
        return keyed.Select(k => k.Token).ToArray();
    }
}
