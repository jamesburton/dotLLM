using DotLLM.Cli.Benchmarking;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// #1892: <c>dotllm bench</c> sent the whole prompt (and each depth-extension chunk) to one <c>Forward</c>, so a long prompt on a model with a hard
/// per-forward limit (Vulkan in-place TurboQuant) met the model's guard. The bench prefill now splits at the limit through
/// <see cref="DotLLM.Engine.PrefillChunkBound"/>.
/// </summary>
public sealed class BenchRunnerPrefillBoundTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public BenchRunnerPrefillBoundTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    /// <summary>The CPU model behind a limit that refuses (as the Vulkan guard does) any forward above it, recording every length.</summary>
    private sealed class LimitedModel(IModel inner, int limit) : IModel
    {
        public List<int> ForwardLengths { get; } = new();
        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public int MaxPrefillForwardTokens => limit;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => Forward(tokenIds, positions, deviceId, null);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            ForwardLengths.Add(tokenIds.Length);
            if (limit > 0 && tokenIds.Length > limit)
                throw new NotSupportedException($"a forward of {tokenIds.Length} rows exceeds the limit of {limit}");
            return inner.Forward(tokenIds, positions, deviceId, kvCache);
        }
        public void Dispose() { }
    }

    private static int[] Tokens(int n) => Enumerable.Range(0, n).Select(i => 1 + i % 40).ToArray();

    private BenchResult Run(IModel model, int promptLen, int depth = 0)
        => BenchRunner.Run(model, size => new SimpleKvCache(KvGeometry.FromConfig(_tiny.Config), size), Tokens(promptLen),
            decodeTokens: 2, reps: 1, depth: depth, loadMs: 0, warmupReps: 0);

    [Fact]
    public void PromptAboveTheLimit_IsSplitAtTheLimit()
    {
        var model = new LimitedModel(_tiny.Model, limit: 5);
        var result = Run(model, promptLen: 23);   // a whole-prompt forward would throw above
        Assert.Equal(23, result.PromptTokens);
        Assert.Equal(new[] { 5, 5, 5, 5, 3 }, model.ForwardLengths.Take(5).ToArray());
        Assert.All(model.ForwardLengths, n => Assert.True(n <= 5));
    }

    [Fact]
    public void DepthExtension_IsBoundedByTheLimitToo()
    {
        var model = new LimitedModel(_tiny.Model, limit: 4);
        Run(model, promptLen: 8, depth: 10);
        Assert.All(model.ForwardLengths, n => Assert.True(n <= 4, $"a forward of {n} rows exceeded the limit"));
    }

    [Fact]
    public void WithoutALimit_ThePromptIsOneForward()
    {
        var model = new LimitedModel(_tiny.Model, limit: 0);
        Run(model, promptLen: 23);
        Assert.Equal(23, model.ForwardLengths[0]);
    }

    [Fact]
    public void EmptyPrompt_IsRefusedWithAClearError_NotANullLogitsCrash()
    {
        var model = new LimitedModel(_tiny.Model, limit: 5);
        var e = Assert.Throws<ArgumentOutOfRangeException>(() => Run(model, promptLen: 0));
        Assert.Contains("promptTokens", e.Message);
        Assert.Empty(model.ForwardLengths);
    }

    [Theory]
    [InlineData(0, 100, 0, 100)]
    [InlineData(0, 100, 7, 7)]
    [InlineData(50, 100, 7, 7)]
    [InlineData(3, 100, 7, 3)]
    [InlineData(512, 20, 0, 512)]
    public void PrefillStep(int configured, int length, int limit, int expected)
        => Assert.Equal(expected, BenchRunner.PrefillStep(new LimitedModel(_tiny.Model, limit), configured, length));
}
