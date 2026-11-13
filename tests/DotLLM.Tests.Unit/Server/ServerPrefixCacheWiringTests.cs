using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Server;
using DotLLM.Tokenizers;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1591: the CPU server (paged KV + cross-request prefix trie + continuous-batch scheduler) reported
/// <c>cached_tokens = 0</c> for every request, because the scheduler was constructed without the prefix trie. This loads the
/// tiny CPU qwen3 through <see cref="ServerStartup.LoadModel"/> exactly as <c>dotllm serve</c> does and submits two prompts
/// sharing a prefix through the scheduler it built.
/// </summary>
public sealed class ServerPrefixCacheWiringTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public ServerPrefixCacheWiringTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    [Fact]
    public async Task SchedulerBuiltByServerStartup_HitsThePrefixTrie()
    {
        var options = new ServerOptions
        {
            Model = _tiny.GgufPath,
            ModelId = "tiny-1591",
            Device = "cpu",
            UsePaged = true,
            Warmup = WarmupOptions.Disabled,
            PromptCacheEnabled = true,
        };
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, options);
        Assert.NotNull(state.PrefixTrieManager);
        Assert.NotNull(state.Scheduler);
        var scheduler = state.Scheduler!;
        using var cts = new CancellationTokenSource();
        var loop = Task.Run(() => scheduler.RunLoopAsync(cts.Token));
        try
        {
            int[] prefix = Enumerable.Range(0, 40).Select(i => 2 + (i * 13) % 40).ToArray();   // 2.5 blocks of 16
            var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 3 };
            var first = await scheduler.EnqueueAsync(new InferenceRequest { TokenIds = [.. prefix, 5, 6], Options = opts });
            var second = await scheduler.EnqueueAsync(new InferenceRequest { TokenIds = [.. prefix, 7, 8, 9], Options = opts });

            Assert.Equal(0, first.Timings.CachedTokenCount);
            Assert.Equal(32, second.Timings.CachedTokenCount);                  // two whole 16-token blocks
            Assert.True(state.PrefixTrieManager!.Hits >= 1);
        }
        finally
        {
            cts.Cancel();
            try { await loop; } catch (OperationCanceledException) { }
        }
    }

    // ---- #1778: a side request between an agent's turns must not cost the conversation its live-KV prefix ----------------

    /// <summary>Comma-separated token ids, so prompts are exact id sequences on the tiny model (no BPE involved).</summary>
    private sealed class IdTokenizer(int vocab) : ITokenizer
    {
        public int VocabSize => vocab;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(int.Parse).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13) % (PrefixCacheFailedRequestTests.TinyQwen.Vocab - 2);
        return ids;
    }

    private const int TurnTokens = 4, Headroom = 20;   // the live path's next-turn headroom (#1591), scaled to the 64-token model
    private static string Text(int[] ids) => string.Join(",", ids);
    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = TurnTokens };

    /// <summary>
    /// Runs turn 1 of a 30-token conversation, a short unrelated side request sharing only its first 3 tokens (a chat template's
    /// header), then turn 2 of the conversation, on the live-KV <see cref="PrefixCache"/> that <see cref="ServerStartup.LoadModel"/>
    /// builds for a CPU <c>--no-paged</c> server from <paramref name="options"/>.
    /// </summary>
    private (int Turn1Prompt, int Turn2Cached, int SideCached, PrefixCachePolicy Policy) RunConversationWithSideRequest(ServerOptions options)
    {
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, options);
        var cache = state.PrefixCache;
        Assert.NotNull(cache);
        Assert.Null(state.PrefixTrieManager);          // the live-KV path, not the paged trie
        // The server's own generator sizes CPU caches without headroom; the device live path (Vulkan, #1591) adds it so the next,
        // longer turn fits. Same PrefixCache, same lookup; the id tokenizer keeps the prompts exact.
        var generator = new TextGenerator(state.Model!, new IdTokenizer(PrefixCacheFailedRequestTests.TinyQwen.Vocab),
            prefixCache: cache, prefixCacheHeadroomTokens: Headroom);

        int[] turn1 = Ids(30, 3);
        var r1 = generator.Generate(Text(turn1), Opts());
        int[] side = [.. turn1[..3], .. Ids(5, 17)];                               // e.g. a web_fetch summariser
        var rs = generator.Generate(Text(side), Opts());
        int[] turn2 = [.. turn1, .. r1.GeneratedTokenIds, .. Ids(8, 29)];          // the conversation continues
        var r2 = generator.Generate(Text(turn2), Opts());
        return (turn1.Length, r2.Timings.CachedTokenCount, rs.Timings.CachedTokenCount, cache!.Policy);
    }

    private ServerOptions LiveKvOptions(PrefixCachePolicy? policy)
    {
        var options = new ServerOptions
        {
            Model = _tiny.GgufPath,
            ModelId = "tiny-1778",
            Device = "cpu",
            UsePaged = false,                         // --no-paged: the live-KV PrefixCache
            Warmup = WarmupOptions.Disabled,
            PromptCacheEnabled = true,
        };
        return policy is { } p ? options with { PromptCachePolicy = p } : options;
    }

    [Fact]
    public void DefaultPolicy_SideRequestBetweenTurns_KeepsTheConversationsPrefix()
    {
        // #1778: the server's default policy. Under LRU the side request adopted the conversation's entry and truncated it to the
        // shared tokens, so turn 2 re-prefilled almost everything (measured on Strix Halo: 3 of ~6.9k tokens cached).
        var (turn1Prompt, turn2Cached, sideCached, policy) = RunConversationWithSideRequest(LiveKvOptions(policy: null));
        Assert.Equal(PrefixCachePolicy.Agent, policy);
        Assert.Equal(0, sideCached);                                           // the side request ran on its own
        Assert.True(turn2Cached >= turn1Prompt, $"turn 2 reused {turn2Cached} tokens, less than turn 1's {turn1Prompt}-token prompt");
    }

    [Fact]
    public void ExplicitLru_SideRequestBetweenTurns_TruncatesTheConversation()
    {
        // The control arm: --prompt-cache-policy lru stays selectable and reproduces the problem the default fixes, so the
        // scenario above can tell the two policies apart.
        var (_, turn2Cached, sideCached, policy) = RunConversationWithSideRequest(LiveKvOptions(PrefixCachePolicy.Lru));
        Assert.Equal(PrefixCachePolicy.Lru, policy);
        Assert.Equal(3, sideCached);                                           // adopted the conversation's entry
        Assert.True(turn2Cached <= 3, $"LRU turn 2 reused {turn2Cached} tokens; the side request did not truncate the conversation");
    }
}
