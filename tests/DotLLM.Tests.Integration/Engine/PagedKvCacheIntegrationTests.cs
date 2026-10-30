using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests for paged KV-cache with <see cref="TextGenerator"/>.
/// Verifies that <see cref="PagedKvCache"/> produces identical output to <see cref="SimpleKvCache"/>
/// and that prefix caching works correctly with paged allocation.
/// </summary>
[Collection("SmallModel")]
public class PagedKvCacheIntegrationTests
{
    private readonly SmallModelFixture _fixture;

    public PagedKvCacheIntegrationTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer, ModelConfig config) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer, config);
    }

    [Fact]
    public void PagedKvCache_MatchesSimpleKvCache_GreedyOutput()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        // Generate with default SimpleKvCache
        var simpleGenerator = new TextGenerator(model, tokenizer);
        var simpleResponse = simpleGenerator.Generate("The capital of France is", options);

        // Generate with PagedKvCache
        // Bit-identity oracle: F32 reference blocks (default F16 blocks are compared by tolerance, #1268).
        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim, elementType: DType.Float32);
        var pagedGenerator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size));
        var pagedResponse = pagedGenerator.Generate("The capital of France is", options);

        Assert.Equal(simpleResponse.Text, pagedResponse.Text);
        Assert.Equal(simpleResponse.GeneratedTokenIds, pagedResponse.GeneratedTokenIds);
    }

    /// <summary>
    /// F16 (default) and Q8_0 blocks (#1268, #1142 C4), teacher-forced on the F32 simple cache's greedy tokens: the logits
    /// stay as close to the F32 cache as the existing Q8_0 <see cref="QuantizedKvCache"/> does ("qkv" anchor: max |dlogit|
    /// 0.84 on SmolLM-135M; paged F16 0.94, Q8_0 0.90, F32 exactly 0), and attention never touches the legacy staging.
    /// Greedy text can differ at a near-tie (SmolLM-135M: "Paris," vs "Paris." at token 2), so this is not a text compare.
    /// </summary>
    [Theory]
    [InlineData(0)] // F16 blocks
    [InlineData(1)] // Q8_0 blocks
    [InlineData(2)] // F32 blocks: exact
    [InlineData(3)] // QuantizedKvCache Q8_0 anchor
    public unsafe void PagedKvCache_Blocks_TrackSimpleCacheLogits(int variant)
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };
        var simple = new TextGenerator(model, tokenizer).Generate("The capital of France is", options);
        int[] tokens = [.. tokenizer.Encode("The capital of France is"), .. simple.GeneratedTokenIds];

        DType blockType = variant switch { 0 => DType.Float16, 2 => DType.Float32, _ => DType.Q8_0 };
        using var factory = new PagedKvCacheFactory(config.NumLayers, config.NumKvHeads, config.HeadDim, elementType: blockType);
        using var pagedCache = factory.Create(tokens.Length);
        using var anchor = new QuantizedKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, tokens.Length,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        IKvCache candidate = variant == 3 ? anchor : pagedCache;
        using var reference = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, tokens.Length);

        float maxDiff = 0f;
        int topAgree = 0;
        for (int t = 0; t < tokens.Length; t++)
        {
            using var a = model.Forward(tokens.AsSpan(t, 1), [t], -1, reference);
            using var b = model.Forward(tokens.AsSpan(t, 1), [t], -1, candidate);
            var ra = new ReadOnlySpan<float>((void*)a.DataPointer, config.VocabSize);
            var rb = new ReadOnlySpan<float>((void*)b.DataPointer, config.VocabSize);
            for (int i = 0; i < ra.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(ra[i] - rb[i]));
            if (TensorPrimitives.IndexOfMax(ra) == TensorPrimitives.IndexOfMax(rb))
                topAgree++;
        }

        Console.WriteLine($"[variant {variant}] max |dlogit| = {maxDiff}, top-1 agreement {topAgree}/{tokens.Length}");
        Assert.Equal(0, pagedCache.StagingBytes);
        if (variant == 2)
            Assert.Equal(0f, maxDiff);
        Assert.True(maxDiff < 1.5f, $"max |dlogit| {maxDiff}");
        Assert.True(topAgree >= tokens.Length - 2, $"top-1 agreement {topAgree}/{tokens.Length}");
    }

    [Fact]
    public void PagedKvCache_WithPrefixCache_ReusesCacheOnSecondCall()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var prefixCache = new PrefixCache(maxEntries: 2);

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixCache: prefixCache);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        // First call — no cached tokens
        var response1 = generator.Generate("The capital of France is", options);
        Assert.Equal(0, response1.Timings.CachedTokenCount);
        Assert.True(response1.GeneratedTokenCount > 0);

        // Second call with extended prompt (shares prefix with first call's full sequence)
        // The prefix cache stores prompt + generated tokens from call 1.
        // Call 2's prompt = same prompt → shares that prefix.
        var response2 = generator.Generate("The capital of France is", options);
        Assert.True(response2.Timings.CachedTokenCount > 0,
            $"Expected cached tokens > 0 on second call, got {response2.Timings.CachedTokenCount}. " +
            "PrefixCache should reuse the PagedKvCache from the first call.");
    }

    [Fact]
    public void PagedKvCache_WithPrefixCache_DoesNotReuseUndersizedEntry()
    {
        // Regression test: a stored prefix-cache entry whose KV-cache was sized for
        // a small max_tokens must NOT be reused for a later request that needs more
        // generation room. Reusing a too-small cache causes silent early-termination
        // once the decode loop hits pos >= cacheSize.
        //
        // Pre-fix: the ResolveKvCache short-circuit `|| entry.KvCache.MaxLength >= promptLen`
        // accepted any entry big enough to fit just the prompt, with zero room for generation,
        // so the second call would terminate after ~0 decode tokens when cache saturates.
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var prefixCache = new PrefixCache(maxEntries: 2);

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixCache: prefixCache);

        const string prompt = "The capital of France is";
        const int firstMaxTokens = 2;
        const int secondMaxTokens = 32;

        // First call: tiny max_tokens so the stored cache is tightly sized
        // (MaxLength = promptLen + firstMaxTokens).
        var response1 = generator.Generate(prompt, new InferenceOptions { Temperature = 0f, MaxTokens = firstMaxTokens });
        Assert.Equal(firstMaxTokens, response1.GeneratedTokenCount);

        // Second call with the same prompt but much larger max_tokens. The stored cache
        // from call 1 has room for only firstMaxTokens extra positions past the prompt;
        // without the fix, the decode loop breaks once pos >= cacheSize and generation
        // silently truncates at ~firstMaxTokens tokens despite the larger budget.
        var response2 = generator.Generate(prompt, new InferenceOptions { Temperature = 0f, MaxTokens = secondMaxTokens });

        // Require at least 4× the first-call budget — well above the buggy cap of
        // firstMaxTokens + 1 and still safely below secondMaxTokens even if the model
        // hits EOS part-way through.
        int minimumExpected = firstMaxTokens * 4;
        Assert.True(response2.GeneratedTokenCount >= minimumExpected,
            $"Expected response2 to generate at least {minimumExpected} tokens with " +
            $"MaxTokens={secondMaxTokens} after a prior MaxTokens={firstMaxTokens} call " +
            $"(got {response2.GeneratedTokenCount}). A too-small reused prefix-cache entry is truncating output.");
    }

    [Fact]
    public async Task PagedKvCache_StreamingMatchesSynchronous()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size));

        // Synchronous
        var syncResponse = generator.Generate("The capital of France is", options);

        // Streaming
        var sb = new System.Text.StringBuilder();
        await foreach (var token in generator.GenerateStreamingTokensAsync("The capital of France is", options))
            sb.Append(token.Text);

        Assert.Equal(syncResponse.Text, sb.ToString());
    }
}
