using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

/// <summary>
/// The own-length rule of the agent-aware prefix policy (issue #1654), pinned once here for every cache that applies it. The
/// discriminating case throughout: a long conversation used a moment ago versus a short side request stored after it. LRU
/// gives up the long conversation; the agent policy gives up the short one.
/// </summary>
public sealed class PrefixRetentionTests
{
    private sealed record E(string Name, long LastUse, int Length);

    private static int Victim(IReadOnlyList<E> entries, long clock, PrefixCachePolicy policy)
        => PrefixRetention.SelectVictim(entries, e => e.LastUse, e => e.Length, clock, policy);

    [Theory]
    [InlineData(100, 0, 100, 1.0)]
    [InlineData(150, 100, 100, 0.5)]
    [InlineData(100, 100, 100, 0.0)]
    [InlineData(50, 100, 100, 0.0)]     // a clock behind the last use (never happens) reads as fresh, not negative
    [InlineData(10, 0, 0, 10.0)]        // a zero-length entry counts as length 1
    public void Age_IsTokensSinceUseOverOwnLength(long clock, long lastUse, int length, double expected)
        => Assert.Equal(expected, PrefixRetention.Age(clock, lastUse, length), 9);

    [Theory]
    [InlineData(199, 100, 100, false)]
    [InlineData(200, 100, 100, true)]   // exactly its own length: expired
    [InlineData(201, 100, 100, true)]
    public void Expired_AtOrPastOwnLength(long clock, long lastUse, int length, bool expected)
        => Assert.Equal(expected, PrefixRetention.Expired(clock, lastUse, length));

    [Fact]
    public void SelectVictim_Agent_GivesUpTheShortStaleEntry_WhereLruGivesUpTheLongConversation()
    {
        // The long conversation (100k) was used at clock 100k; a 3k side request was stored at 103k; now the clock is 109k.
        // LRU: the conversation is least recently used. Agent: the side request is 2x its own length old, the conversation 9%.
        E[] entries = [new("conversation", 100_000, 100_000), new("side", 103_000, 3_000)];

        Assert.Equal(0, Victim(entries, clock: 109_000, PrefixCachePolicy.Lru));
        Assert.Equal(1, Victim(entries, clock: 109_000, PrefixCachePolicy.Agent));
    }

    [Fact]
    public void SelectVictim_Agent_PicksTheLargestAge_NotTheFirstExpired()
    {
        E[] entries = [new("a", 0, 100), new("b", 50, 10), new("c", 90, 1000)];   // ages at 200: 2.0, 15.0, 0.11

        Assert.Equal(1, Victim(entries, clock: 200, PrefixCachePolicy.Agent));
    }

    [Fact]
    public void SelectVictim_Agent_TieBreaksByLeastRecentUse()
    {
        E[] entries = [new("a", 100, 50), new("b", 50, 100)];   // at 200: ages 2.0 and 1.5; at 150: 1.0 and 1.0

        Assert.Equal(0, Victim(entries, clock: 200, PrefixCachePolicy.Agent));
        Assert.Equal(1, Victim(entries, clock: 150, PrefixCachePolicy.Agent));   // equal age: b is older
    }

    [Fact]
    public void SelectVictim_Lru_IgnoresLength()
    {
        E[] entries = [new("a", 10, 1), new("b", 5, 1_000_000)];

        Assert.Equal(1, Victim(entries, clock: 1_000, PrefixCachePolicy.Lru));
    }

    [Fact]
    public void SelectVictim_RejectsNoCandidates()
        => Assert.Throws<ArgumentException>(() => Victim([], 0, PrefixCachePolicy.Agent));

    [Theory]
    [InlineData(100, 100, -1, true)]    // extends the entry
    [InlineData(120, 100, -1, true)]
    [InlineData(70, 100, 60, true)]     // shares everything before the entry's last user message
    [InlineData(60, 100, 60, true)]
    [InlineData(59, 100, 60, false)]    // diverges inside the earlier history
    [InlineData(50, 100, -1, true)]     // no save point: half the entry stands in
    [InlineData(49, 100, -1, false)]
    [InlineData(99, 100, 100, false)]   // the save point sits at the end: only an extension is the same conversation
    public void SameConversation_UsesTheSavePoint_OrHalfTheEntryWithoutOne(int shared, int length, int savePoint, bool expected)
        => Assert.Equal(expected, PrefixRetention.SameConversation(shared, length, savePoint));

    [Fact]
    public void Protects_OnlyUnderAgent_OnlyDestructive_OnlyOtherConversations_OnlyUnexpired()
    {
        // A 100-token entry last used at 100; a 10-token request at clock 110 shares 20 tokens with it.
        Assert.False(PrefixRetention.Protects(PrefixCachePolicy.Lru, 110, 100, 100, shared: 20, savePoint: -1));
        Assert.True(PrefixRetention.Protects(PrefixCachePolicy.Agent, 110, 100, 100, shared: 20, savePoint: -1));
        Assert.False(PrefixRetention.Protects(PrefixCachePolicy.Agent, 110, 100, 100, shared: 100, savePoint: -1));   // extends: not destructive
        Assert.False(PrefixRetention.Protects(PrefixCachePolicy.Agent, 110, 100, 100, shared: 20, savePoint: 20));    // same conversation
        Assert.False(PrefixRetention.Protects(PrefixCachePolicy.Agent, 200, 100, 100, shared: 20, savePoint: -1));    // expired
    }
}
