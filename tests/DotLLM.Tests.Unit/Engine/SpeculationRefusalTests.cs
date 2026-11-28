using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tokenizers;
using Xunit;
using TinyQwen = DotLLM.Tests.Unit.Engine.PrefixCacheFailedRequestTests.TinyQwen;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #2030 review: speculation rolls rejected drafts back by cutting the KV length, and its forwards take no adapter. Where that
/// is not enough, greedy output silently differs from plain decode, so neither draft models nor prompt lookup may engage. Each case
/// pairs the refusal with a control arm that does speculate, so a test cannot pass because nothing ever drafts.
/// </summary>
public sealed class SpeculationRefusalTests : IClassFixture<TinyQwen>
{
    private readonly TinyQwen _tiny;

    public SpeculationRefusalTests(TinyQwen tiny) => _tiny = tiny;

    private const string Prompt = "5,11,23,7,5,11,23,7,5,11,23,7";

    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = 24 };

    /// <summary>The real model, optionally claiming recurrent state that cannot be checkpointed.</summary>
    private sealed class Wrapped(IModel inner, bool recurrent = false, bool checkpoint = false) : IModel
    {
        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => 0;
        public bool RequiresPerSequenceState => recurrent;
        public bool SupportsRecurrentStateCheckpoint => checkpoint;
        public void ResetSequenceState() { }
        public object CheckpointRecurrentState() => new();
        public void RestoreRecurrentState(object? checkpoint) { }
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => inner.Forward(tokenIds, positions, deviceId);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
            => inner.Forward(tokenIds, positions, deviceId, kvCache);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
            => inner.Forward(tokenIds, positions, deviceId, kvCache);
        public void Dispose() { }
    }

    /// <summary>An adapter that adapts nothing: what matters here is that one is attached to the request.</summary>
    private sealed class EmptyAdapter : ILoraAdapter
    {
        public string Name => "empty";
        public int Rank => 1;
        public float Alpha => 1f;
        public IReadOnlyList<string> TargetModules => [];
        public LoraLayerWeights? GetLayerWeights(int layerIndex, string projName, LoraRegion region = LoraRegion.Any) => null;
        public bool IsCompatible(ModelConfig baseConfig) => true;
        public void Dispose() { }
    }

    private int Drafted(IModel target, IModel draft, bool lookup, Func<ModelConfig, int, IKvCache>? kv = null, ILoraAdapter? adapter = null)
    {
        var generator = lookup
            ? new TextGenerator(target, new IdsTokenizer(), kv, promptLookup: PromptLookupOptions.Create(3, 1))
            : new TextGenerator(target, new IdsTokenizer(), kv, draftModel: draft, draftKvCacheFactory: kv, speculativeCandidates: 3);
        var response = generator.Generate(Prompt, Opts(), adapter: adapter);
        return response.Timings.SpeculativeDraftTokens;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RecurrentTargetWithoutCheckpoint_DoesNotSpeculate(bool lookup)
    {
        Assert.True(Drafted(new Wrapped(_tiny.Model), _tiny.Model, lookup) > 0, "control arm drafted nothing");
        Assert.True(Drafted(new Wrapped(_tiny.Model, recurrent: true, checkpoint: true), _tiny.Model, lookup) > 0,
            "a checkpointing recurrent target must keep speculating");
        Assert.Equal(0, Drafted(new Wrapped(_tiny.Model, recurrent: true), _tiny.Model, lookup));
    }

    [Fact]
    public async Task Streaming_RefusesTheSameWay()
    {
        async Task<int> Streamed(IModel target, ILoraAdapter? adapter = null)
        {
            var generator = new TextGenerator(target, new IdsTokenizer(), draftModel: _tiny.Model, speculativeCandidates: 3);
            int drafted = 0;
            await foreach (var t in generator.GenerateStreamingTokensAsync(Prompt, Opts(), default, adapter))
                if (t.Timings is { } timings) drafted = timings.SpeculativeDraftTokens;
            return drafted;
        }

        using var emptyAdapter = new EmptyAdapter();
        Assert.True(await Streamed(new Wrapped(_tiny.Model)) > 0, "control arm drafted nothing");
        Assert.Equal(0, await Streamed(new Wrapped(_tiny.Model, recurrent: true)));
        Assert.Equal(0, await Streamed(new Wrapped(_tiny.Model), emptyAdapter));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WindowedQuantizedKv_DoesNotSpeculate(bool lookup)
    {
        var model = _tiny.Model32;
        Func<ModelConfig, int, IKvCache> Kv(int window) => (cfg, size) => new QuantizedKvCache(
            KvGeometry.FromConfig(cfg, false), size, KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);

        Assert.True(Drafted(model, model, lookup, Kv(0)) > 0, "control arm drafted nothing");
        Assert.Equal(0, Drafted(model, model, lookup, Kv(8)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoraRequest_DoesNotSpeculate(bool lookup)
    {
        var target = new Wrapped(_tiny.Model);
        Assert.True(Drafted(target, _tiny.Model, lookup) > 0, "control arm drafted nothing");
        using var adapter = new EmptyAdapter();
        Assert.Equal(0, Drafted(target, _tiny.Model, lookup, adapter: adapter));
    }

    [Fact]
    public void DraftLoader_LoadsNothingForARecurrentTargetWithoutCheckpoint()
    {
        var warnings = new List<string>();
        var result = DraftModelLoader.LoadWith(new Wrapped(_tiny.Model, recurrent: true), "cpu", _tiny.Config,
            speculativeCandidates: 3, warnings.Add, _ => throw new InvalidOperationException("must not load"));

        Assert.Null(result);
        Assert.Contains("draft model not loaded", Assert.Single(warnings));
    }

    private sealed class IdsTokenizer : ITokenizer
    {
        public int VocabSize => TinyQwen.Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }
}
