using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
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
}
