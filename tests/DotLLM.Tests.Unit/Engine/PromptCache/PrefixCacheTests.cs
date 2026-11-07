using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

public sealed class PrefixCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int MaxSeqLen = 64;

    private static SimpleKvCache CreateCache() =>
        new(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

    [Fact]
    public void FindMatch_EmptyCache_ReturnsNull()
    {
        using var cache = new PrefixCache(4);
        var (entry, matched) = cache.FindMatch([1, 2, 3]);

        Assert.Null(entry);
        Assert.Equal(0, matched);
    }

    [Fact]
    public void FindMatch_ExactMatch_ReturnsFullLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        int[] tokens = [1, 2, 3, 4, 5];
        cache.Store(tokens, kvCache);

        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);

        Assert.NotNull(entry);
        Assert.Equal(5, matched);
    }

    [Fact]
    public void FindMatch_PartialMatch_ReturnsMatchedLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3, 4, 5], kvCache);

        // Prompt shares first 3 tokens with stored entry
        var (entry, matched) = cache.FindMatch([1, 2, 3, 10, 11]);

        Assert.NotNull(entry);
        Assert.Equal(3, matched);
    }

    [Fact]
    public void FindMatch_NoMatch_ReturnsZero()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3], kvCache);

        var (entry, matched) = cache.FindMatch([10, 20, 30]);

        Assert.Null(entry); // matchedTokens=0, so entry should be null
        Assert.Equal(0, matched);
    }

    [Fact]
    public void FindMatch_PromptLongerThanStored_MatchesUpToStoredLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3], kvCache);

        // Prompt starts with [1,2,3] then continues
        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5, 6]);

        Assert.NotNull(entry);
        Assert.Equal(3, matched);
    }

    [Fact]
    public void FindMatch_MultpleEntries_ReturnsBestMatch()
    {
        using var cache = new PrefixCache(4);

        var kv1 = CreateCache();
        cache.Store([1, 2], kv1);

        var kv2 = CreateCache();
        cache.Store([1, 2, 3, 4], kv2);

        // Should match the longer prefix
        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);

        Assert.NotNull(entry);
        Assert.Equal(4, matched);
    }

    [Fact]
    public void Store_SameKvCache_UpdatesTokenSequence()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();

        cache.Store([1, 2, 3], kvCache);
        Assert.Equal(1, cache.EntryCount);

        // Store again with same KV-cache but longer sequence
        cache.Store([1, 2, 3, 4, 5], kvCache);
        Assert.Equal(1, cache.EntryCount);

        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);
        Assert.Equal(5, matched);
    }

    [Fact]
    public void Store_EvictsLRU_WhenAtCapacity()
    {
        using var cache = new PrefixCache(2);

        var kv1 = CreateCache();
        cache.Store([10, 20], kv1);

        var kv2 = CreateCache();
        cache.Store([30, 40], kv2);
        Assert.Equal(2, cache.EntryCount);

        // Access kv2 to make kv1 the LRU
        cache.FindMatch([30, 40]);

        // Adding a third should evict kv1 (LRU)
        var kv3 = CreateCache();
        cache.Store([50, 60], kv3);
        Assert.Equal(2, cache.EntryCount);

        // kv1 should be gone
        var (entry1, matched1) = cache.FindMatch([10, 20]);
        Assert.Equal(0, matched1);

        // kv2 should still be there
        var (entry2, matched2) = cache.FindMatch([30, 40]);
        Assert.Equal(2, matched2);
    }

    [Fact]
    public void Clear_RemovesAllEntries()
    {
        using var cache = new PrefixCache(4);
        cache.Store([1, 2], CreateCache());
        cache.Store([3, 4], CreateCache());

        cache.Clear();

        Assert.Equal(0, cache.EntryCount);
        var (entry, matched) = cache.FindMatch([1, 2]);
        Assert.Equal(0, matched);
    }

    [Fact]
    public void Dispose_ClearsAllEntries()
    {
        using var cache = new PrefixCache(4);
        cache.Store([1, 2], CreateCache());
        cache.Store([3, 4], CreateCache());

        cache.Dispose();

        Assert.Equal(0, cache.EntryCount);
    }

    [Fact]
    public void Constructor_ThrowsOnZeroMaxEntries()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => { new PrefixCache(0); });
    }

    // ── agent-aware policy (issue #1654) ──

    private static int[] Seq(int from, int count) => [.. Enumerable.Range(from, count)];

    /// <summary>A 100-token conversation with a 10-token "system prompt" at the front that a side request shares.</summary>
    private static readonly int[] Conversation = Seq(1, 100);
    private static readonly int[] SideRequest = [.. Seq(1, 10), .. Seq(500, 20)];

    [Theory]
    [InlineData(PrefixCachePolicy.Lru, 10)]
    [InlineData(PrefixCachePolicy.Agent, 0)]
    public void Agent_ProtectsAnUnexpiredConversation_FromAShallowHitByAnotherConversation(PrefixCachePolicy policy, int expectedMatch)
    {
        // A hit truncates the entry's live KV to the match and the request overwrites the rest. Under LRU a 30-token side request
        // that shares the 10-token system prompt destroys the 100-token conversation for 10 tokens of saving.
        using var cache = new PrefixCache(4, policy);
        cache.Store(Conversation, CreateCache());

        var (entry, matched) = cache.FindMatch(SideRequest);

        Assert.Equal(expectedMatch, matched);
        Assert.Equal(expectedMatch == 0, entry is null);
        Assert.Equal(policy == PrefixCachePolicy.Agent ? 1 : 0, cache.ProtectedMisses);
        // The conversation's own next turn still hits the whole entry either way.
        Assert.Equal(100, cache.FindMatch([.. Conversation, .. Seq(900, 5)]).MatchedTokens);
    }

    [Fact]
    public void Agent_AllowsTheShallowHit_OnceTheConversationHasExpired()
    {
        // Expiry: the conversation's own length (100) of other prompt tokens read since its last use. Three 30-token side
        // requests (90) are not enough; the fourth (120) is.
        using var cache = new PrefixCache(4, PrefixCachePolicy.Agent);
        cache.Store(Conversation, CreateCache());

        for (int i = 0; i < 3; i++)
            Assert.Equal(0, cache.FindMatch(SideRequest).MatchedTokens);
        Assert.Equal(10, cache.FindMatch(SideRequest).MatchedTokens);
        Assert.Equal(3, cache.ProtectedMisses);
    }

    [Fact]
    public void Agent_TreatsARequestSharingEverythingBeforeTheLastUserMessage_AsTheSameConversation()
    {
        // The entry's save point is the start of its last user message (token 60). A retry or edit of that message shares 60
        // tokens: the same conversation, so it may resume (destructively) from there; a request sharing less may not.
        using var cache = new PrefixCache(4, PrefixCachePolicy.Agent);
        cache.Store(Conversation, CreateCache(), savePoint: 60);

        Assert.Equal(60, cache.FindMatch([.. Seq(1, 60), .. Seq(700, 10)]).MatchedTokens);
        cache.Store(Conversation, CreateCache(), savePoint: 60);   // (restore the entry for the second probe)
        Assert.Equal(0, cache.FindMatch([.. Seq(1, 59), .. Seq(700, 10)]).MatchedTokens);
    }

    [Fact]
    public void Agent_WithoutASavePoint_HalfTheEntryStandsInForIt()
    {
        using var cache = new PrefixCache(4, PrefixCachePolicy.Agent);
        cache.Store(Conversation, CreateCache());

        Assert.Equal(0, cache.FindMatch([.. Seq(1, 49), .. Seq(700, 10)]).MatchedTokens);
        Assert.Equal(50, cache.FindMatch([.. Seq(1, 50), .. Seq(700, 10)]).MatchedTokens);
    }

    [Fact]
    public void Agent_FallsBackToTheNextBestUnprotectedEntry()
    {
        // The side request shares 10 tokens with the protected conversation and 15 with an earlier side request's entry.
        using var cache = new PrefixCache(4, PrefixCachePolicy.Agent);
        cache.Store(Conversation, CreateCache());
        int[] earlierSide = [.. Seq(1, 10), .. Seq(500, 5), .. Seq(800, 10)];
        cache.Store(earlierSide, CreateCache());

        var (entry, matched) = cache.FindMatch(SideRequest);

        Assert.Equal(15, matched);
        Assert.Same(earlierSide, entry!.TokenSequence);
    }

    [Theory]
    [InlineData(PrefixCachePolicy.Lru)]
    [InlineData(PrefixCachePolicy.Agent)]
    public void Eviction_Agent_DropsTheExpiredShortEntry_Lru_DropsTheLongConversation(PrefixCachePolicy policy)
    {
        // Capacity 2. The conversation (100) is hit at clock 100; a 10-token side entry is stored at clock 100; then an unrelated
        // 20-token lookup moves the clock to 120. The side entry is 2x its own length old, the conversation 20%.
        using var cache = new PrefixCache(2, policy);
        cache.Store(Conversation, CreateCache());
        Assert.Equal(100, cache.FindMatch(Conversation).MatchedTokens);
        int[] side = Seq(500, 10);
        cache.Store(side, CreateCache());
        Assert.Equal(0, cache.FindMatch(Seq(900, 20)).MatchedTokens);

        cache.Store(Seq(900, 20), CreateCache());   // third entry: one must go

        Assert.Equal(2, cache.EntryCount);
        if (policy == PrefixCachePolicy.Agent)
        {
            Assert.Equal(100, cache.FindMatch(Conversation).MatchedTokens);
            Assert.Equal(0, cache.FindMatch(side).MatchedTokens);
        }
        else
        {
            Assert.Equal(0, cache.FindMatch(Conversation).MatchedTokens);
            Assert.Equal(10, cache.FindMatch(side).MatchedTokens);
        }
    }

    [Fact]
    public void TokenClock_CountsEveryLookup_HitOrMiss()
    {
        using var cache = new PrefixCache(4, PrefixCachePolicy.Agent);
        cache.Store(Conversation, CreateCache());
        cache.FindMatch(Conversation);
        cache.FindMatch(Seq(900, 7));

        Assert.Equal(107, cache.TokenClock);
        Assert.Equal(1, cache.Hits);
        Assert.Equal(1, cache.Misses);
    }
}
