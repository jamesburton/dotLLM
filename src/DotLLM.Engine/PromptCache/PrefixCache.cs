using System.Diagnostics;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// Cache of prefix-matched KV-cache sessions for multi-turn conversations.
/// Stores live KV-cache instances with their associated token sequences.
/// On each generation call, finds the longest matching prefix to skip redundant prefill.
/// </summary>
/// <remarks>
/// A hit hands the entry's live KV to the request, which truncates it to the match and overwrites the rows past it, so a hit
/// on part of an entry destroys the rest. Under <see cref="PrefixCachePolicy.Lru"/> that is always allowed and a full cache
/// evicts its least recently used entry. Under <see cref="PrefixCachePolicy.Agent"/> (#1654) an entry is protected from
/// partial hits by requests outside its conversation until about its own length of other prompt tokens has been read, and
/// eviction picks the entry oldest relative to its own length (<see cref="PrefixRetention"/>).
/// </remarks>
public sealed class PrefixCache : IDisposable
{
    private readonly int _maxEntries;
    private readonly List<PrefixCacheEntry> _entries;
    private readonly PrefixCachePolicy _policy;
    private long _clock, _hits, _misses, _protectedMisses;
    private bool _disposed;

    /// <summary>Number of cached entries.</summary>
    public int EntryCount => _entries.Count;

    /// <summary>The retention policy in force.</summary>
    public PrefixCachePolicy Policy => _policy;

    /// <summary>Lookups that found a reusable entry.</summary>
    public long Hits => _hits;

    /// <summary>Lookups that found no reusable entry.</summary>
    public long Misses => _misses;

    /// <summary>
    /// Lookups that would have hit, destroying an unexpired entry of another conversation, and were made to miss by
    /// <see cref="PrefixCachePolicy.Agent"/>. The protected entry stays for its own conversation's next turn.
    /// </summary>
    public long ProtectedMisses => _protectedMisses;

    /// <summary>Prompt tokens this cache has been asked about: the token clock the own-length rule is measured on.</summary>
    public long TokenClock => _clock;

    /// <summary>
    /// Creates a new prefix cache with the given maximum number of entries.
    /// </summary>
    /// <param name="maxEntries">Maximum cached sessions. When full, an entry is evicted per <paramref name="policy"/>.</param>
    /// <param name="policy">The retention policy (default <see cref="PrefixCachePolicy.Lru"/>, the pre-#1654 behaviour).</param>
    public PrefixCache(int maxEntries = 4, PrefixCachePolicy policy = PrefixCachePolicy.Lru)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxEntries, 1);
        _maxEntries = maxEntries;
        _policy = policy;
        _entries = new List<PrefixCacheEntry>(maxEntries);
    }

    /// <summary>
    /// Finds the entry with the longest common prefix match against the given prompt tokens.
    /// Returns the matched entry and the number of matched tokens, or (null, 0) on miss.
    /// Exact lookup: a match never extends past an entry's first approximate row.
    /// </summary>
    internal (PrefixCacheEntry? Entry, int MatchedTokens) FindMatch(ReadOnlySpan<int> promptTokenIds)
        => FindMatch(promptTokenIds, allowApprox: false, out _);

    /// <summary>
    /// Lineage-aware lookup. <c>MatchedTokens</c> is the number of rows to restore. Without
    /// <paramref name="allowApprox"/> it stops at the entry's first approximate row and
    /// <paramref name="approxRowsReused"/> is 0. With it, approximate rows may be restored and are
    /// reported in <paramref name="approxRowsReused"/>; <c>cached_tokens</c> counts only
    /// <c>MatchedTokens - approxRowsReused</c>.
    /// </summary>
    /// <remarks>
    /// Advances the token clock by the prompt's length. Under <see cref="PrefixCachePolicy.Agent"/> an entry that the prompt
    /// would truncate, that belongs to another conversation and that has not expired is skipped
    /// (<see cref="PrefixRetention.Protects"/>); the next-best entry, or a miss, is returned instead (#1654).
    /// </remarks>
    internal (PrefixCacheEntry? Entry, int MatchedTokens) FindMatch(
        ReadOnlySpan<int> promptTokenIds, bool allowApprox, out int approxRowsReused)
    {
        _clock += promptTokenIds.Length;
        PrefixCacheEntry? bestEntry = null;
        int bestMatch = 0;
        bool protectedSome = false;
        approxRowsReused = 0;

        for (int i = 0; i < _entries.Count; i++)
        {
            var entry = _entries[i];
            int match = entry.GetPrefixMatchLength(promptTokenIds, allowApprox);
            // Longest match wins; on a tie, the entry with more exact rows (as the trie prefers exact children).
            if (match > bestMatch || (match == bestMatch && bestEntry is not null
                && entry.ApproxRowsIn(match) < bestEntry.ApproxRowsIn(bestMatch)))
            {
                if (PrefixRetention.Protects(_policy, _clock, entry.LastUseClock, entry.TokenSequence.Length, match, entry.SavePoint))
                {
                    protectedSome = true;
                    continue;
                }
                bestMatch = match;
                bestEntry = entry;
            }
        }

        if (bestEntry != null)
        {
            bestEntry.LastAccessTicks = Stopwatch.GetTimestamp();
            bestEntry.LastUseClock = _clock;
            approxRowsReused = bestEntry.ApproxRowsIn(bestMatch);
            _hits++;
        }
        else
        {
            _misses++;
            if (protectedSome) _protectedMisses++;
        }

        return (bestEntry, bestMatch);
    }

    /// <summary>
    /// Stores a KV-cache with its associated token sequence.
    /// If the same KV-cache instance already exists, updates the token sequence.
    /// Evicts the LRU entry if at capacity.
    /// </summary>
    /// <param name="tokenSequence">Tokens whose K/V the cache now holds.</param>
    /// <param name="kvCache">The live KV-cache.</param>
    /// <param name="firstApproxRow">Lineage of the stored rows: the first approximate row, or <c>null</c>
    /// when all rows are exact. It replaces any earlier tag on the same cache. A caller that restored
    /// approximate rows (<c>allowApprox</c>) must pass the restored entry's first approximate row (or
    /// the first row it assembled itself, whichever is lower): every row after it attended to
    /// approximate K/V.</param>
    /// <param name="savePoint">The start of the request's last user message within <paramref name="tokenSequence"/>, or -1:
    /// the boundary below which a later request still counts as the same conversation (#1654).</param>
    internal void Store(int[] tokenSequence, IKvCache kvCache, int? firstApproxRow = null, int savePoint = -1)
    {
        if (firstApproxRow is < 0)
            throw new ArgumentOutOfRangeException(nameof(firstApproxRow));

        // Check if this KV-cache is already stored (reuse case)
        for (int i = 0; i < _entries.Count; i++)
        {
            if (ReferenceEquals(_entries[i].KvCache, kvCache))
            {
                // Update token sequence and access time in-place (no allocation)
                _entries[i].TokenSequence = tokenSequence;
                _entries[i].FirstApproxRow = firstApproxRow;
                _entries[i].LastAccessTicks = Stopwatch.GetTimestamp();
                _entries[i].LastUseClock = _clock;
                _entries[i].SavePoint = savePoint;
                return;
            }
        }

        // Evict if at capacity: LRU, or the entry oldest relative to its own length (#1654).
        if (_entries.Count >= _maxEntries)
        {
            int victim = _policy == PrefixCachePolicy.Agent
                ? PrefixRetention.SelectVictim(_entries, e => e.LastUseClock, e => e.TokenSequence.Length, _clock, _policy)
                : PrefixRetention.SelectVictim(_entries, e => e.LastAccessTicks, e => e.TokenSequence.Length, _clock, _policy);
            _entries[victim].Dispose();
            _entries.RemoveAt(victim);
        }

        _entries.Add(new PrefixCacheEntry(tokenSequence, kvCache, firstApproxRow) { LastUseClock = _clock, SavePoint = savePoint });
    }

    /// <summary>
    /// Narrows <paramref name="entry"/> to its first <paramref name="validRows"/> tokens, in place. Call before a request
    /// overwrites the cache's rows from <paramref name="validRows"/> on: a request that then fails or is cancelled leaves the
    /// entry claiming only rows that still hold the K/V of the tokens it names.
    /// </summary>
    internal void Narrow(PrefixCacheEntry entry, int validRows)
    {
        if (validRows >= entry.TokenSequence.Length)
            return;
        entry.TokenSequence = entry.TokenSequence.AsSpan(0, validRows).ToArray();
        if (entry.FirstApproxRow is int first && first >= validRows)
            entry.FirstApproxRow = null;
    }

    /// <summary>
    /// Removes all entries and disposes their KV-caches.
    /// </summary>
    public void Clear()
    {
        for (int i = 0; i < _entries.Count; i++)
            _entries[i].Dispose();
        _entries.Clear();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Clear();
    }
}
