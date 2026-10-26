using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using static DotLLM.Tests.Unit.Engine.RecurrentPrefixMtpHarness;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>Prompt-lookup (n-gram) drafting (issue #1293): the index, and exactness of the speculative loop it drives.</summary>
public sealed class PromptLookupDrafterTests : IDisposable
{
    private readonly string _scratch = Directory.CreateTempSubdirectory("dotllm-prompt-lookup-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static int[] Draft(int[] context, int k, PromptLookupOptions? options, out int match)
    {
        using var d = new PromptLookupDrafter(options ?? new PromptLookupOptions(), context, [], 256);
        var buf = new int[k];
        int n = d.Draft(buf, out match);
        return buf[..n];
    }

    [Fact]
    public void NoRecurringSuffix_DraftsNothing()
    {
        Assert.Empty(Draft([1, 2, 3, 4, 5, 6, 7], 4, null, out int m));
        Assert.Equal(0, m);
        Assert.Empty(Draft([9], 4, null, out _));
        Assert.Empty(Draft([], 4, null, out _));
    }

    [Fact]
    public void Match_ProposesWhatFollowedIt()
    {
        // Suffix [3, 4] recurs at 2..3; what followed was 5, 6, 7.
        Assert.Equal([5, 6, 7], Draft([1, 3, 4, 5, 6, 7, 8, 3, 4], 3, null, out int m));
        Assert.Equal(2, m);
    }

    [Fact]
    public void SingleTokenRecurrence_IsBelowMinNgram()
        => Assert.Empty(Draft([1, 4, 5, 6, 2, 4], 3, null, out _));

    [Fact]
    public void LongestMatch_BeatsALaterShorterOne()
    {
        // Suffix [7, 3, 4]: matches 3 tokens at the start (→ 10), only [3, 4] later (→ 20).
        int[] ctx = [7, 3, 4, 10, 11, 3, 4, 20, 21, 7, 3, 4];
        Assert.Equal([10, 11], Draft(ctx, 2, null, out int m));
        Assert.Equal(3, m);
    }

    [Fact]
    public void EqualLengthMatches_PreferTheLatest()
    {
        int[] ctx = [3, 4, 10, 1, 3, 4, 20, 0, 3, 4];
        Assert.Equal([20, 0], Draft(ctx, 2, null, out int m));
        Assert.Equal(2, m);
    }

    [Fact]
    public void MatchLength_IsCappedAtMaxNgram()
    {
        int[] ctx = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5];
        Draft(ctx, 1, new PromptLookupOptions(MaxNgram: 3), out int m);
        Assert.Equal(3, m);
        Assert.Equal([9], Draft(ctx, 1, null, out m));
        Assert.Equal(5, m);
    }

    [Fact]
    public void OverlappingMatch_ContinuesThePeriod()
    {
        // Period-3 run: the latest match ends one period back, its continuation runs into the drafts themselves.
        Assert.Equal([1, 2, 3, 1, 2, 3, 1], Draft([1, 2, 3, 1, 2, 3], 7, null, out int m));
        Assert.Equal(3, m);
        // Period 1: a repeated token keeps repeating.
        Assert.Equal([5, 5, 5, 5], Draft([1, 5, 5, 5], 4, null, out _));
    }

    [Fact]
    public void Append_IndexesIncrementally()
    {
        using var d = new PromptLookupDrafter(new PromptLookupOptions(), [1, 2, 3, 9], [], 64);
        var buf = new int[2];
        Assert.Equal(0, d.Draft(buf));
        d.Append(1);
        d.Append(2);
        Assert.Equal(2, d.Draft(buf));
        Assert.Equal([3, 9], buf);
        Assert.Equal(6, d.Count);
    }

    [Fact]
    public void Capacity_StopsIndexing()
    {
        using var d = new PromptLookupDrafter(new PromptLookupOptions(), [1, 2], [], 3);
        d.Append(3);
        d.Append(1);
        Assert.Equal(3, d.Count);
    }

    [Fact]
    public void BucketCollisions_NeverMatchFalsely()
    {
        // Many distinct bigrams in a small table share buckets; none but the true one may match.
        var ctx = new List<int>();
        for (int i = 0; i < 4000; i++) ctx.Add(i);
        ctx.Add(100_000);
        ctx.Add(100_001);
        Assert.Empty(Draft([.. ctx], 3, null, out _));
    }

    [Fact]
    public void ServeFlags_ParseIntoOptions()
    {
        Assert.Null(DotLLM.Server.ServerOptions.Parse(["--model", "m.gguf"]).PromptLookup);
        Assert.Equal(new PromptLookupOptions(5, 4, 2),
            DotLLM.Server.ServerOptions.Parse(["--model", "m.gguf", "--prompt-lookup", "--lookup-draft", "5", "--lookup-ngram", "4"]).PromptLookup);
        Assert.Equal(new PromptLookupOptions(8, 1, 1), PromptLookupOptions.Create(8, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PromptLookupOptions.Create(0, 6));
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]   // GDN + attention trunk: a rejected round restores the recurrent state
    [InlineData(1, true)]
    [InlineData(2, true)]
    public async Task GreedyOutput_IsIdenticalWithAndWithoutLookup(int fullAttnInterval, bool streaming)
    {
        using var gguf = GgufFile.Open(SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"lookup-{fullAttnInterval}.gguf"), withMtp: false, fullAttnInterval: fullAttnInterval, contextLength: 512));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        int attnLayers = model is Qwen3HybridDenseTransformerModel h ? h.AttentionLayerCount : config.NumLayers;
        Func<DotLLM.Core.Models.ModelConfig, int, DotLLM.Core.Attention.IKvCache> kv =
            (cfg, size) => new SimpleKvCache(attnLayers, cfg.NumKvHeads, cfg.HeadDim, size);
        var options = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = 120 };
        // A self-repeating prompt so lookups hit from the first step.
        int[] seed = MtpLongContextHarness.Prompt(20, config.VocabSize);
        string prompt = Csv([.. seed, .. seed, .. seed]);

        var plain = await Run(new TextGenerator(model, new TokenIdsTokenizer(config.VocabSize), kv, mtpEnabled: false), prompt, options, streaming);
        var lookup = await Run(new TextGenerator(model, new TokenIdsTokenizer(config.VocabSize), kv, mtpEnabled: false,
            promptLookup: new PromptLookupOptions(DraftTokens: 5)), prompt, options, streaming);

        Assert.Equal(plain.Tokens, lookup.Tokens);
        Assert.Equal(0, plain.Timings.SpeculativeDraftTokens);
        Assert.True(lookup.Timings.SpeculativeDraftTokens > 0, "no lookup drafted");
        Assert.True(lookup.Timings.SpeculativeAcceptedTokens > 0, "no draft accepted");
        // Some drafts must be rejected too, or the rollback (KV + recurrent state) went unexercised.
        Assert.True(lookup.Timings.SpeculativeAcceptedTokens < lookup.Timings.SpeculativeDraftTokens, "no draft rejected");
    }

    private static async Task<(List<int> Tokens, InferenceTimings Timings)> Run(TextGenerator g, string prompt, InferenceOptions options, bool streaming)
    {
        var tokens = new List<int>();
        if (!streaming)
        {
            var r = g.Generate(prompt, options, onTokenGenerated: tokens.Add);
            return (tokens, r.Timings);
        }
        InferenceTimings? last = null;
        await foreach (var t in g.GenerateStreamingTokensAsync(prompt, options))
        {
            tokens.Add(t.TokenId);
            last = t.Timings ?? last;
        }
        return (tokens, last!.Value);
    }
}
