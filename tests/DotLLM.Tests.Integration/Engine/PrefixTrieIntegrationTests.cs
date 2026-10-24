using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
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
/// End-to-end tests for the cross-request prefix trie (Step 37) running
/// against a real (small) model. Verifies bit-identical output between the
/// cached and uncached paths plus visible prefill-token savings across
/// concurrent shared-prompt requests.
/// </summary>
[Collection("SmallModel")]
public class PrefixTrieIntegrationTests
{
    private readonly SmallModelFixture _fixture;

    public PrefixTrieIntegrationTests(SmallModelFixture fixture)
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
    public void PrefixTrie_BitIdenticalOutput_VsNoCache()
    {
        // The cached path must produce the SAME tokens as the uncached path.
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        const string prompt = "The capital of France is";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        // Baseline: no caching.
        var baseline = new TextGenerator(model, tokenizer);
        var rBase = baseline.Generate(prompt, options);

        // Cached: prefix trie wired up. First request fills the trie, second hits it.
        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var mgr = new PrefixTrieManager(pagedFactory);
        var cachedGen = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixTrieManager: mgr);

        cachedGen.Generate(prompt, options); // warm
        var rCached = cachedGen.Generate(prompt, options);

        Assert.Equal(rBase.GeneratedTokenIds, rCached.GeneratedTokenIds);
        Assert.Equal(rBase.Text, rCached.Text);
    }

    [Fact]
    public void PrefixTrie_FourSharedPromptRequests_ReusePrefill()
    {
        // Acceptance criterion: with the manager wired into TextGenerator,
        // a shared 32-token prompt issued 4 times should drive only the
        // FIRST request through full prefill. The remaining three should
        // report a CachedTokenCount equal to the prompt length.
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        // Long enough to be at least 1 full block (block size = 16).
        const string prompt = "You are a helpful assistant who answers concisely and accurately. " +
                              "Provide just the answer without commentary. " +
                              "The capital of France is";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 4 };

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var mgr = new PrefixTrieManager(pagedFactory);
        var gen = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixTrieManager: mgr);

        // First request: miss.
        var r1 = gen.Generate(prompt, options);
        Assert.Equal(0, r1.Timings.CachedTokenCount);

        long cachedAfterFirst = mgr.Trie.HitTokens;
        Assert.Equal(0, cachedAfterFirst);

        // Next three: should hit the trie.
        int promptTokens = tokenizer.Encode(prompt).Length;
        int blockSize = mgr.Trie.BlockSize;
        int expectedHitPerRequest = (promptTokens / blockSize) * blockSize;
        Assume.That(expectedHitPerRequest > 0,
            $"Test prompt must span at least one full block (block_size={blockSize}, got {promptTokens} tokens).");

        for (int i = 0; i < 3; i++)
        {
            var r = gen.Generate(prompt, options);
            Assert.Equal(expectedHitPerRequest, r.Timings.CachedTokenCount);
        }

        // The manager's hit-tokens counter records every Lookup hit (including
        // the warm-up's own no-op lookup when its prompt was below block size),
        // so we assert AT LEAST 3 full-prompt hits worth.
        Assert.True(mgr.Trie.HitTokens >= 3L * expectedHitPerRequest,
            $"Expected hit_tokens >= {3L * expectedHitPerRequest}, got {mgr.Trie.HitTokens}");
    }

    [Fact]
    public void PrefixTrie_DisabledConfig_NoReuse()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var mgr = new PrefixTrieManager(pagedFactory, new PrefixCacheConfig { Enabled = false });
        var gen = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixTrieManager: mgr);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 4 };
        gen.Generate("Test prompt to make trie inactive.", options);
        var r2 = gen.Generate("Test prompt to make trie inactive.", options);

        Assert.Equal(0, r2.Timings.CachedTokenCount);
    }
}

internal static class Assume
{
    public static void That(bool condition, string failMessage)
    {
        if (!condition)
            Assert.Fail(failMessage);
    }
}
