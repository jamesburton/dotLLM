using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;
using Xunit;
using static DotLLM.Tests.Unit.Engine.PrefixCacheFailedRequestTests;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1780: under the agent prefix policy a request of the same conversation that diverges before the entry's save point (a
/// client inserting a message mid-history) shares more than half of the entry, but a hit would truncate the entry its conversation
/// still needs, so <see cref="PrefixRetention.Protects"/> made it a cold miss (lru reused 9,277 of 11,734 tokens). The miss now seeds
/// its fresh cache with the protected entry's shared rows. Runs the deterministic CPU qwen3 on the <c>--no-paged</c> path
/// (<see cref="SimpleKvCache"/>, no factory); a wrong seeded row changes the greedy continuation.
/// </summary>
public sealed class AgentProtectedSeedTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private const int MaxTokens = 6;
    private readonly TinyQwen _tiny;

    public AgentProtectedSeedTests(TinyQwen tiny) => _tiny = tiny;

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13 + i * i) % (TinyQwen.Vocab - 2);
        return ids;
    }

    private static string Text(int[] ids) => string.Join(",", ids);

    private sealed class Tok(int v) : ITokenizer
    {
        public int VocabSize => v;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(int.Parse).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString();
    }

    // The conversation: 28 tokens of history, then the last user message (its start is the save point).
    private static readonly int[] History = Ids(28, 3);
    private static readonly int[] LastUser = Ids(4, 17);
    private static readonly int[] Turn1 = [.. History, .. LastUser];

    // The same conversation with a message inserted mid-history: shares 22 tokens, more than half of the 38-token entry, but less than
    // its save point (28), so the agent policy protects the entry from this request.
    private const int Shared = 22;
    private static readonly int[] Diverged = [.. History[..Shared], .. Ids(10, 29)];

    private static InferenceOptions Opts(int[]? savePointPrefix = null)
        => new() { Temperature = 0f, MaxTokens = MaxTokens, SavePointPrefix = savePointPrefix is null ? null : Text(savePointPrefix) };

    private (TextGenerator Gen, PrefixCache Prefix) Make(PrefixCachePolicy policy, int seedMin = 8)
    {
        var prefix = new PrefixCache(2, policy);
        var gen = new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab), prefixCache: prefix, prefixCacheHeadroomTokens: 16, prefillChunkSize: 4)
        {
            ProtectedSeedMinTokens = seedMin,
        };
        return (gen, prefix);
    }

    private InferenceResponse Cold(int[] prompt)
        => new TextGenerator(_tiny.Model, new Tok(TinyQwen.Vocab), prefillChunkSize: 4).Generate(Text(prompt), Opts());

    [Fact]
    public void DivergedSameConversationRequest_SeedsFromTheProtectedEntry_AndTheEntryStillServesItsNextTurn()
    {
        var (gen, prefix) = Make(PrefixCachePolicy.Agent);
        using var _ = prefix;
        var first = gen.Generate(Text(Turn1), Opts(History));
        int entryRows = Turn1.Length + MaxTokens - 1;          // the last sampled token was never forwarded

        // The shape the policy protects (the precondition of the issue, pinned so the test cannot drift into a plain hit).
        Assert.True(2 * Shared >= Turn1.Length + MaxTokens && Shared < History.Length);
        Assert.True(PrefixRetention.Protects(PrefixCachePolicy.Agent, prefix.TokenClock + Diverged.Length, prefix.TokenClock,
            Turn1.Length + MaxTokens, Shared, History.Length));

        var diverged = gen.Generate(Text(Diverged), Opts(History[..Shared]));
        Assert.Equal(Shared, diverged.Timings.CachedTokenCount);
        Assert.Equal(Cold(Diverged).GeneratedTokenIds, diverged.GeneratedTokenIds);
        Assert.Equal(1, prefix.ProtectedMisses);

        // The original conversation's next turn still finds its whole entry.
        int[] turn2 = [.. Turn1, .. first.GeneratedTokenIds, .. Ids(6, 41)];
        var next = gen.Generate(Text(turn2), Opts(Turn1));
        Assert.Equal(entryRows, next.Timings.CachedTokenCount);
        Assert.Equal(Cold(turn2).GeneratedTokenIds, next.GeneratedTokenIds);
    }

    [Fact]
    public void ShareBelowTheSeedThreshold_RunsCold_AndKeepsTheEntry()
    {
        var (gen, prefix) = Make(PrefixCachePolicy.Agent, seedMin: Shared + 1);
        using var _ = prefix;
        var first = gen.Generate(Text(Turn1), Opts(History));
        var diverged = gen.Generate(Text(Diverged), Opts(History[..Shared]));
        Assert.Equal(0, diverged.Timings.CachedTokenCount);
        Assert.Equal(Cold(Diverged).GeneratedTokenIds, diverged.GeneratedTokenIds);

        int[] turn2 = [.. Turn1, .. first.GeneratedTokenIds, .. Ids(6, 41)];
        Assert.Equal(Turn1.Length + MaxTokens - 1, gen.Generate(Text(turn2), Opts(Turn1)).Timings.CachedTokenCount);
    }

    /// <summary>Under lru the same request is a plain hit that truncates the entry: the seed path is agent-only.</summary>
    [Fact]
    public void Lru_IsAPlainHit()
    {
        var (gen, prefix) = Make(PrefixCachePolicy.Lru);
        using var _ = prefix;
        gen.Generate(Text(Turn1), Opts(History));
        var diverged = gen.Generate(Text(Diverged), Opts(History[..Shared]));
        Assert.Equal(Shared, diverged.Timings.CachedTokenCount);
        Assert.Equal(Cold(Diverged).GeneratedTokenIds, diverged.GeneratedTokenIds);
        Assert.Equal(1, prefix.EntryCount);                     // adopted, not copied
    }
}
