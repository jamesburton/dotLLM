namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// EPIC / LegoLink-k (arXiv 2410.15332): recompute the first <c>k</c> tokens of every chunk after the first (the attention
/// sinks an isolated encoding creates at each chunk start; chunk 0 is exact). Static: no model, no query, seed ignored.
/// </summary>
/// <remarks>
/// <para>Order: offset 0 of every stale chunk, then offset 1, … up to <c>k</c> (so a budget below <c>k × (chunks − 1)</c>
/// gives every chunk an equal share of its first tokens). The interface asks for an exact budget, and EPIC's native budget
/// (<c>k = 16</c>: 2.3 % of a 2K context of four 512-token chunks) is smaller than the swept 5–20 %; the remainder is spread
/// evenly over the rest of the stale chunks (a bit-reversal order of their remaining tokens), a static uniform coverage. That
/// keeps this selector distinct from <see cref="ChunkStartsSelector"/>, which spends the whole budget on longer chunk
/// prefixes.</para>
/// </remarks>
public sealed class FirstKSelector : RankedSelector
{
    /// <summary>Creates the selector.</summary>
    /// <param name="k">Tokens per chunk start; EPIC's main setting is 16.</param>
    public FirstKSelector(int k = 16)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        K = k;
    }

    /// <summary>Tokens recomputed at the start of each chunk after the first.</summary>
    public int K { get; }

    /// <inheritdoc/>
    public override int[] Rank(ChunkedContext context, int seed)
    {
        ArgumentNullException.ThrowIfNull(context);
        int[] head = ChunkPrefixRoundRobin(context, K).ToArray();
        var taken = new HashSet<int>(head);
        int[] rest = StaleTokens(context).Where(i => !taken.Contains(i)).ToArray();
        return WithFirstChunkLast(context, head.Concat(LowDiscrepancyOrder(rest.Length).Select(i => rest[i])));
    }

    /// <summary>A permutation of <c>[0, n)</c> whose every prefix is spread evenly (bit-reversed counting).</summary>
    internal static IEnumerable<int> LowDiscrepancyOrder(int n)
    {
        if (n == 0) yield break;
        int bits = 0;
        while ((1 << bits) < n) bits++;
        for (int i = 0; i < 1 << bits; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++)
                if ((i & (1 << b)) != 0) r |= 1 << (bits - 1 - b);
            if (r < n) yield return r;
        }
    }
}
