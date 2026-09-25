using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// CPU e2e coverage for the issue #141 user toggles:
/// <list type="bullet">
/// <item>Speculative decoding invariant — greedy generation with a draft model (SmolLM drafting
/// for itself) must produce EXACTLY the same tokens as plain greedy decode. Self-drafting also
/// forces full acceptance, so the speculative path demonstrably runs.</item>
/// <item>Prefill chunk size (llama.cpp <c>-ub</c> analog) — chunked prefill must produce exactly
/// the same greedy output as single-pass prefill.</item>
/// </list>
/// </summary>
[Collection("SmallModel")]
public class SpecDecodeBatchTogglesTests
{
    private readonly SmallModelFixture _fixture;

    public SpecDecodeBatchTogglesTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer);
    }

    [Fact]
    public void GreedySpeculative_MatchesPlainGreedy_Exactly()
    {
        var (target, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = target;
        var (draft, draftGguf, _2) = LoadModel(); // SmolLM drafts for itself
        using var ___ = draftGguf;
        using var ____ = draft;

        const string prompt = "The quick brown fox jumps over the lazy dog because";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 24 };

        var plain = new TextGenerator(target, tokenizer).Generate(prompt, options);
        var speculative = new TextGenerator(target, tokenizer,
                draftModel: draft, speculativeCandidates: 4)
            .Generate(prompt, options);

        // Core invariant: greedy speculative output must EXACTLY match plain greedy output.
        Assert.Equal(plain.GeneratedTokenIds, speculative.GeneratedTokenIds);
        Assert.Equal(plain.Text, speculative.Text);

        // Prove the speculative path actually engaged (not a silent fall-through).
        Assert.True(speculative.Timings.SpeculativeDraftTokens > 0,
            "Speculative path did not run (no draft tokens recorded).");
        Assert.True(speculative.Timings.SpeculativeAcceptedTokens > 0,
            "Speculative path accepted no tokens.");
    }

    [Fact]
    public void GreedySpeculative_WithChunkedPrefill_MatchesPlainGreedy()
    {
        var (target, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = target;
        var (draft, draftGguf, _2) = LoadModel();
        using var ___ = draftGguf;
        using var ____ = draft;

        const string prompt = "The capital of France is";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 12 };

        var plain = new TextGenerator(target, tokenizer).Generate(prompt, options);
        var speculative = new TextGenerator(target, tokenizer,
                draftModel: draft, speculativeCandidates: 3, prefillChunkSize: 3)
            .Generate(prompt, options);

        Assert.Equal(plain.GeneratedTokenIds, speculative.GeneratedTokenIds);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void ChunkedPrefill_MatchesSinglePassGreedy_Exactly(int chunkSize)
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        const string prompt = "The quick brown fox jumps over the lazy dog because it";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 16 };

        var single = new TextGenerator(model, tokenizer).Generate(prompt, options);
        var chunked = new TextGenerator(model, tokenizer, prefillChunkSize: chunkSize)
            .Generate(prompt, options);

        Assert.Equal(single.GeneratedTokenIds, chunked.GeneratedTokenIds);
        Assert.Equal(single.Text, chunked.Text);
    }
}
