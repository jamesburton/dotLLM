namespace DotLLM.Engine.PromptCache;

/// <summary>
/// How a prefix cache chooses what to keep when a new request competes with cached conversations (issue #1654).
/// </summary>
public enum PrefixCachePolicy
{
    /// <summary>Least-recently-used eviction; any hit may truncate the entry it hits. The behaviour before #1654.</summary>
    Lru,

    /// <summary>
    /// Agent-aware retention. An entry is kept until about its <em>own length</em> of other prompt tokens has been read since
    /// its last use (a 100k-token conversation survives 30 unrelated 3k-token side requests; a 3k entry yields after one), and
    /// while it is unexpired a request that would destroy it (one that shares only part of it and does not belong to the same
    /// conversation) runs without touching it instead of truncating or replacing it. Output is unchanged: the policy decides
    /// only what is reused, never what is computed.
    /// </summary>
    Agent,
}

/// <summary>
/// The shared arithmetic of <see cref="PrefixCachePolicy.Agent"/>: a token clock (prompt tokens the cache has been asked
/// about), an entry's age in tokens since its last use, and the own-length rule. Pure functions so every cache (the live-KV
/// <see cref="PrefixCache"/>, the paged <see cref="PrefixTrie"/>, the recurrent snapshot caches and the scheduler's) applies
/// exactly one rule and the unit tests pin it once.
/// </summary>
public static class PrefixRetention
{
    /// <summary>
    /// Tokens the cache has read since <paramref name="lastUseClock"/>, as a fraction of <paramref name="length"/>: the
    /// entry's "age" in units of its own length. At or above 1 the entry has expired under the own-length rule.
    /// </summary>
    /// <param name="clock">The cache's token clock now (prompt tokens read so far, this request included).</param>
    /// <param name="lastUseClock">The clock when the entry was last stored or hit.</param>
    /// <param name="length">The entry's length in tokens (at least 1 is assumed).</param>
    public static double Age(long clock, long lastUseClock, int length)
        => (double)Math.Max(0, clock - lastUseClock) / Math.Max(1, length);

    /// <summary>True when at least the entry's own length of prompt tokens has been read since its last use.</summary>
    public static bool Expired(long clock, long lastUseClock, int length)
        => clock - lastUseClock >= Math.Max(1, length);

    /// <summary>
    /// Picks the entry to give up. <see cref="PrefixCachePolicy.Lru"/> picks the least recently used.
    /// <see cref="PrefixCachePolicy.Agent"/> picks the entry that is oldest <em>relative to its own length</em> (the largest
    /// <see cref="Age"/>): expired entries go first, and when none has expired the one closest to expiry goes, so a short
    /// stale entry yields before a long conversation that was used a moment ago. Ties fall back to least recently used.
    /// </summary>
    /// <typeparam name="T">The entry type.</typeparam>
    /// <param name="entries">Candidates; must be non-empty.</param>
    /// <param name="lastUseClock">The entry's last-use clock.</param>
    /// <param name="length">The entry's length in tokens.</param>
    /// <param name="clock">The cache's token clock now.</param>
    /// <param name="policy">The policy in force.</param>
    /// <returns>The index of the victim in <paramref name="entries"/>.</returns>
    public static int SelectVictim<T>(IReadOnlyList<T> entries, Func<T, long> lastUseClock, Func<T, int> length,
        long clock, PrefixCachePolicy policy)
    {
        ArgumentNullException.ThrowIfNull(entries);
        if (entries.Count == 0) throw new ArgumentException("No candidates.", nameof(entries));
        int victim = 0;
        for (int i = 1; i < entries.Count; i++)
        {
            long useI = lastUseClock(entries[i]), useV = lastUseClock(entries[victim]);
            if (policy == PrefixCachePolicy.Agent)
            {
                double ageI = Age(clock, useI, length(entries[i])), ageV = Age(clock, useV, length(entries[victim]));
                if (ageI > ageV || ageI == ageV && useI < useV) victim = i;
            }
            else if (useI < useV)
            {
                victim = i;
            }
        }
        return victim;
    }

    /// <summary>
    /// Whether a request that shares <paramref name="shared"/> leading tokens with an entry of <paramref name="length"/>
    /// tokens belongs to the same conversation. Extending the entry always does. With a save point (the start of the entry's
    /// last user message, <see cref="Core.Configuration.InferenceOptions.SavePointPrefix"/>) a request that shares everything
    /// before that message does too: a retry, an edited or compacted last message, or a side request over the same history.
    /// Without one, sharing at least half of the entry stands in for it.
    /// </summary>
    /// <param name="shared">Tokens the request shares with the entry.</param>
    /// <param name="length">The entry's length.</param>
    /// <param name="savePoint">The entry's save point, or -1 when it has none.</param>
    public static bool SameConversation(int shared, int length, int savePoint)
        => shared >= length || (savePoint >= 0 ? shared >= savePoint : 2L * shared >= length);

    /// <summary>
    /// Whether <paramref name="policy"/> protects an entry from a request that would destroy it. A request destroys an entry
    /// when it shares less than the whole entry (a hit would truncate the rows past the share; a miss would replace it). The
    /// entry is protected while it is unexpired and the request is not the same conversation; the request then runs on its
    /// own, leaving the entry for the conversation it belongs to.
    /// </summary>
    /// <param name="policy">The policy in force.</param>
    /// <param name="clock">The cache's token clock now (this request's tokens included).</param>
    /// <param name="lastUseClock">The entry's last-use clock.</param>
    /// <param name="length">The entry's length.</param>
    /// <param name="shared">Tokens the request shares with the entry.</param>
    /// <param name="savePoint">The entry's save point, or -1.</param>
    public static bool Protects(PrefixCachePolicy policy, long clock, long lastUseClock, int length, int shared, int savePoint)
        => policy == PrefixCachePolicy.Agent
           && shared < length
           && !SameConversation(shared, length, savePoint)
           && !Expired(clock, lastUseClock, length);
}
