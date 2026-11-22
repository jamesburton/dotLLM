using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tests.Unit.ChunkReuse;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1892: a model's hard per-forward limit (<see cref="IModel.MaxPrefillForwardTokens"/>) bounds every prompt forward the generator makes,
/// whatever chunk size was configured: an explicit size above it is clamped, and "one pass" (0) becomes the limit.
/// </summary>
public sealed class TextGeneratorPrefillBoundTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public TextGeneratorPrefillBoundTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    /// <summary>The CPU model behind a limit and a forward-length recorder.</summary>
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
            return inner.Forward(tokenIds, positions, deviceId, kvCache);
        }
        public void Dispose() { }
    }

    private static string Prompt(int n) => new(Enumerable.Range(0, n).Select(i => (char)(1 + i % 40)).ToArray());

    [Theory]
    [InlineData(0)]       // one pass asked for: becomes the limit
    [InlineData(4)]       // below the limit: stands
    [InlineData(5)]       // at it
    [InlineData(1000)]    // above it: clamped
    public void EveryPromptForward_IsAtMostTheLimit(int configuredChunk)
    {
        var model = new LimitedModel(_tiny.Model, limit: 5);
        var gen = new TextGenerator(model, new CharTokenizer(),
            (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg), size), mtpEnabled: false, prefillChunkSize: configuredChunk);
        Assert.Equal(configuredChunk is > 0 and < 5 ? configuredChunk : 5, gen.EffectivePrefillChunkSize);

        var result = gen.Generate(Prompt(23), new InferenceOptions { Temperature = 0, MaxTokens = 2 });
        Assert.Equal(23, result.PromptTokenCount);
        Assert.All(model.ForwardLengths, n => Assert.True(n <= 5, $"a forward of {n} rows exceeded the limit of 5"));
        Assert.True(model.ForwardLengths.Count >= 23 / 5, "the prompt was not split");
    }

    [Fact]
    public void WithoutALimit_TheConfiguredSizeAndOnePassAreUnchanged()
    {
        var model = new LimitedModel(_tiny.Model, limit: 0);
        var kvf = (ModelConfig cfg, int size) => (IKvCache)new SimpleKvCache(KvGeometry.FromConfig(cfg), size);
        Assert.Equal(0, new TextGenerator(model, new CharTokenizer(), kvf, mtpEnabled: false).EffectivePrefillChunkSize);
        Assert.Equal(1000, new TextGenerator(model, new CharTokenizer(), kvf, mtpEnabled: false, prefillChunkSize: 1000).EffectivePrefillChunkSize);
    }
}
