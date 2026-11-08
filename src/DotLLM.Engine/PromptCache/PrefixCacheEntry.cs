using System.Diagnostics;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// A single cached session: the token prefix and the live KV-cache populated by it.
/// The KV-cache is owned by this entry and disposed on eviction.
/// </summary>
internal sealed class PrefixCacheEntry : IDisposable
{
    /// <summary>Full token sequence (prompt + generated) stored in the KV-cache.</summary>
    public int[] TokenSequence { get; set; }

    /// <summary>Live KV-cache with state for <see cref="TokenSequence"/>.</summary>
    public IKvCache KvCache { get; }

    /// <summary>Timestamp of last access, for LRU eviction.</summary>
    public long LastAccessTicks { get; set; }

    /// <summary>
    /// Snapshot lineage: the first row whose K/V is approximate (came from chunk assembly or
    /// selective recompute, or was computed after such a row). Rows below it are exact.
    /// <c>null</c> = every row is exact (the only case today; nothing in the engine creates
    /// approximate rows yet).
    /// </summary>
    public int? FirstApproxRow { get; set; }

    /// <summary>The cache's token clock when the entry was last stored or hit (<see cref="PrefixRetention"/>, #1654).</summary>
    public long LastUseClock { get; set; }

    /// <summary>The start of the last user message of the request that stored the entry, or -1 (#1654).</summary>
    public int SavePoint { get; set; } = -1;

    public PrefixCacheEntry(int[] tokenSequence, IKvCache kvCache, int? firstApproxRow = null)
    {
        TokenSequence = tokenSequence;
        KvCache = kvCache;
        FirstApproxRow = firstApproxRow;
        LastAccessTicks = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Returns the number of leading tokens of the given prompt this entry may restore. Without
    /// <paramref name="allowApprox"/> the match is capped at <see cref="FirstApproxRow"/>, so an exact
    /// request never restores approximate K/V.
    /// </summary>
    public int GetPrefixMatchLength(ReadOnlySpan<int> promptTokenIds, bool allowApprox = false)
    {
        int match = TokenSequence.AsSpan().CommonPrefixLength(promptTokenIds);
        return !allowApprox && FirstApproxRow is int first ? Math.Min(match, first) : match;
    }

    /// <summary>Number of approximate rows among the first <paramref name="matchedTokens"/> rows.</summary>
    public int ApproxRowsIn(int matchedTokens)
        => FirstApproxRow is int first && matchedTokens > first ? matchedTokens - first : 0;

    public void Dispose() => KvCache.Dispose();
}
