using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using static DotLLM.Tests.Unit.Engine.RecurrentPrefixMtpHarness;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1088: a turn that hits the recurrent prefix snapshot decodes with MTP. The head's KV for the cached prefix is kept
/// with the snapshot (with the trunk hidden row the next absorb pairs with), so the head drafts on a hit exactly as it would
/// after a cold prefill of the whole prompt. Runs the real CPU hybrid on the synthetic Qwen3.5 fixture with an MTP head.
/// CUDA twin: <c>CudaQwen3HybridDenseMtpPrefixHitTests</c>.
/// </summary>
public sealed class TextGeneratorRecurrentPrefixMtpTests : IDisposable
{
    // Measured: the hit's draft logits equal the cold run's exactly (max |diff| 0 on every turn). The tolerance only allows
    // for a CPU whose GEMM tiles the shorter suffix differently. The synthetic head's logits are small, so a wrong head moves
    // them little: resuming with a stale carry row measured 2e-4 to 7e-4, which a 1e-3 tolerance let through.
    private const float DraftLogitTolerance = 1e-5f;

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public TextGeneratorRecurrentPrefixMtpTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-prefix-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (GgufFile Gguf, ModelConfig Config) Fixture(string name)
    {
        var gguf = GgufFile.Open(SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, name), withMtp: true, contextLength: 512));
        return (gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
    }

    private static Qwen3HybridDenseTransformerModel Load(GgufFile gguf, ModelConfig config)
        => (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

    private static Func<ModelConfig, int, IKvCache> Kv(Qwen3HybridDenseTransformerModel model)
        => (cfg, size) => new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, size);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AgentTurnThatHitsThePrefix_DecodesWithMtp_AndMatchesPlainGreedy(bool streaming)
    {
        var (gguf, config) = Fixture("prefix-mtp-greedy.gguf");
        using var _ = gguf;
        using var cachedModel = Load(gguf, config);
        using var plainModel = Load(gguf, config);
        var cached = Generator(cachedModel, Kv(cachedModel), mtp: true, prefixCache: true);
        var plain = Generator(plainModel, Kv(plainModel), mtp: false, prefixCache: false);

        int[][] turns = AgentTurns(config.VocabSize);
        for (int t = 0; t < turns.Length; t++)
        {
            var want = plain.Generate(Csv(turns[t]), Greedy);
            var (ids, timings) = streaming ? await Stream(cached, Csv(turns[t])) : Run(cached, Csv(turns[t]));

            Assert.Equal(want.GeneratedTokenIds, ids);
            Assert.True(timings.SpeculativeDraftTokens > 0, $"turn {t} decoded without MTP (cached {timings.CachedTokenCount})");
            Assert.Equal(t == 0 ? 0 : turns[t - 1].Length - 1, timings.CachedTokenCount);   // the hit is what this test is about
        }
        cached.ClearRecurrentPrefixCache();
    }

    [Fact]
    public void HitTurn_DraftsWhatAColdMtpRunOfTheSamePromptDrafts()
    {
        // A head resumed with a zeroed or stale carry row, at the wrong length, below a grown KV's length, or over another
        // conversation's rows drafts differently from the cold run (or throws).
        var (gguf, config) = Fixture("prefix-mtp-drafts.gguf");
        using var _ = gguf;
        using var cachedModel = Load(gguf, config);
        using var coldModel = Load(gguf, config);

        AssertHitTurnsDraftLikeColdRuns(cachedModel, Kv(cachedModel), coldModel, Kv(coldModel), DraftLogitTolerance, _out.WriteLine);
    }

    [Fact]
    public void PlainTurns_KeepTheHeadInStep_SoTheNextHitDecodesWithMtp_AndDraftsLikeAColdRun()
    {
        // A turn that decodes plain (here: logprobs requested) still has its prefill absorbed into the head when it takes the
        // snapshot, so the head covers the new boundary and the next hit can decode with MTP. Both the first turn and a hit that
        // moves the snapshot decode plain here; the last turn must still draft exactly what a cold MTP run drafts.
        var (gguf, config) = Fixture("prefix-mtp-plain-turns.gguf");
        using var _ = gguf;
        using var cachedInner = Load(gguf, config);
        using var coldInner = Load(gguf, config);
        var cachedModel = new DraftRecordingModel(cachedInner);
        var coldModel = new DraftRecordingModel(coldInner);
        var cached = Generator(cachedModel, Kv(cachedInner), mtp: true, prefixCache: true);
        var cold = Generator(coldModel, Kv(coldInner), mtp: true, prefixCache: false);
        var logprobs = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens, Logprobs = true };
        int[][] turns = AgentTurns(config.VocabSize);

        Assert.Equal(0, cached.Generate(Csv(turns[0]), logprobs).Timings.SpeculativeDraftTokens);
        var plainHit = cached.Generate(Csv(turns[1]), logprobs);
        Assert.Equal(turns[0].Length - 1, plainHit.Timings.CachedTokenCount);
        Assert.Equal(0, plainHit.Timings.SpeculativeDraftTokens);

        var want = cold.Generate(Csv(turns[2]), Greedy);
        cachedModel.Drafts.Clear();
        var got = cached.Generate(Csv(turns[2]), Greedy);

        Assert.Equal(turns[1].Length - 1, got.Timings.CachedTokenCount);
        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Equal(want.Timings.SpeculativeDraftTokens, got.Timings.SpeculativeDraftTokens);
        Assert.Equal(coldModel.Drafts.Count, cachedModel.Drafts.Count);
        for (int s = 0; s < coldModel.Drafts.Count; s++)
            for (int v = 0; v < config.VocabSize; v++)
                Assert.True(Math.Abs(coldModel.Drafts[s][v] - cachedModel.Drafts[s][v]) <= DraftLogitTolerance, $"draft step {s} differs");
        // One head served all three turns: none was dropped and re-made, none leaked.
        var head = Assert.Single(cachedModel.CreatedStates);
        cached.ClearRecurrentPrefixCache();
        Assert.True(IsDisposed(head));
    }

    [Fact]
    public void HeadThatCannotResume_HitsDecodePlain_AndStayCorrect()
    {
        // A backend whose head state cannot be marked (the Vulkan states today) keeps no head with the snapshot. A hit must then
        // decode plain: a fresh head cannot start absorbing mid-sequence, and trying would fail the request.
        var (gguf, config) = Fixture("prefix-mtp-unmarkable.gguf");
        using var _ = gguf;
        using var cachedInner = Load(gguf, config);
        using var plainModel = Load(gguf, config);
        var cached = Generator(new DraftRecordingModel(cachedInner, hideMarks: true), Kv(cachedInner), mtp: true, prefixCache: true);
        var plain = Generator(plainModel, Kv(plainModel), mtp: false, prefixCache: false);
        int[][] turns = AgentTurns(config.VocabSize);

        for (int t = 0; t < 3; t++)
        {
            var got = cached.Generate(Csv(turns[t]), Greedy);
            Assert.Equal(t == 0 ? 0 : turns[t - 1].Length - 1, got.Timings.CachedTokenCount);
            Assert.Equal(t == 0, got.Timings.SpeculativeDraftTokens > 0);
            Assert.Equal(plain.Generate(Csv(turns[t]), Greedy).GeneratedTokenIds, got.GeneratedTokenIds);
        }
        cached.ClearRecurrentPrefixCache();
    }

    [Fact]
    public void PlainRequestThatTakesNoSnapshot_CreatesNoHead()
    {
        // The head rides along a plain request only to keep a snapshot usable; with no snapshot it would be pure waste.
        var (gguf, config) = Fixture("prefix-mtp-no-snapshot.gguf");
        using var _ = gguf;
        using var inner = Load(gguf, config);
        var model = new DraftRecordingModel(inner);
        var cached = Generator(model, Kv(inner), mtp: true, prefixCache: true);
        var logprobs = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens, Logprobs = true };

        cached.Generate(Csv(AgentTurns(config.VocabSize)[0][..10]), logprobs);   // too short to snapshot (RecurrentPrefixMinTokens)

        Assert.Empty(model.CreatedStates);
        cached.ClearRecurrentPrefixCache();
    }

    [SkippableFact]
    public void AdaptiveGate_ThatDecodesTheFirstTurnPlain_StillGetsMtpOnTheNextHit()
    {
        // The T5500 replay (#1088): the adaptive gate's size prior sends a small model's first request to plain decode. That turn
        // took the snapshot without a head, every later turn hit it, and a hit can only decode with MTP from a kept head, so
        // the gate never got to try MTP for the whole agent session. The plain first turn now keeps the head in step.
        var (gguf, config) = Fixture("prefix-mtp-gate.gguf");
        using var _ = gguf;
        using var model = Load(gguf, config);
        var cached = new TextGenerator(model, new TokenIdsTokenizer(config.VocabSize), Kv(model), mtpAdaptive: true, recurrentPrefixCache: true);
        Skip.If(cached.MtpGate is null, "DOTLLM_MTP_ADAPTIVE=0 disables the adaptive gate");
        int[][] turns = AgentTurns(config.VocabSize);
        // Long enough for the gate to record a sample (MtpAdaptiveGate.MinSampleTokens decode tokens).
        var options = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = 2 * MtpAdaptiveGate.MinSampleTokens };

        Assert.Equal(0, cached.Generate(Csv(turns[0]), options).Timings.SpeculativeDraftTokens);   // the size prior picks plain
        var hit = cached.Generate(Csv(turns[1]), options);

        Assert.Equal(turns[0].Length - 1, hit.Timings.CachedTokenCount);
        Assert.True(hit.Timings.SpeculativeDraftTokens > 0, "the gate never tried MTP on the hit turn");
        cached.ClearRecurrentPrefixCache();
    }

    [SkippableFact]
    public void AdaptiveGate_ThatMeasuresPlainFaster_FeedsNoHeadOnPlainTurns()
    {
        // PR #1091 review (T5500): feeding the head through a plain turn's prefill cost time to first token on every hit turn,
        // for a head that no later turn drafted from once plain decode measured faster. With plain the clear winner and no
        // re-probe due, a plain snapshot turn must not make or feed a head, and the hits stay plain and correct.
        var (gguf, config) = Fixture("prefix-mtp-gate-plain.gguf");
        using var _ = gguf;
        using var inner = Load(gguf, config);
        using var plainModel = Load(gguf, config);
        var model = new DraftRecordingModel(inner);
        var cached = new TextGenerator(model, new TokenIdsTokenizer(config.VocabSize), Kv(inner), mtpAdaptive: true, recurrentPrefixCache: true);
        var plain = Generator(plainModel, Kv(plainModel), mtp: false, prefixCache: false);
        Skip.If(cached.MtpGate is null, "DOTLLM_MTP_ADAPTIVE=0 disables the adaptive gate");
        cached.MtpGate!.RecordSegment(false, 1000, 1.0);      // plain: 0.001 ms/token
        cached.MtpGate!.RecordSegment(true, 1000, 1000.0);    // MTP: 1 ms/token
        Assert.False(cached.MtpGate!.LikelyUsesMtpNext());
        var options = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = 2 * MtpAdaptiveGate.MinSampleTokens };
        int[][] turns = AgentTurns(config.VocabSize);

        for (int t = 0; t < turns.Length; t++)
        {
            var got = cached.Generate(Csv(turns[t]), options);
            Assert.Equal(t == 0 ? 0 : turns[t - 1].Length - 1, got.Timings.CachedTokenCount);
            Assert.Equal(0, got.Timings.SpeculativeDraftTokens);
            Assert.Equal(plain.Generate(Csv(turns[t]), options).GeneratedTokenIds, got.GeneratedTokenIds);
        }

        Assert.Empty(model.CreatedStates);
        cached.ClearRecurrentPrefixCache();
    }

    [Fact]
    public void HeadsThatCannotResume_AreNotFedThroughPlainTurns()
    {
        // A head kept only for the snapshot is wasted work when no hit can resume it (the Vulkan states today). The first state
        // made shows the model's states cannot resume; later plain snapshot turns make none.
        var (gguf, config) = Fixture("prefix-mtp-unmarkable-plain.gguf");
        using var _ = gguf;
        using var inner = Load(gguf, config);
        var model = new DraftRecordingModel(inner, hideMarks: true);
        var cached = Generator(model, Kv(inner), mtp: true, prefixCache: true);
        var logprobs = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens, Logprobs = true };
        int[][] turns = AgentTurns(config.VocabSize);

        for (int conversation = 0; conversation < 3; conversation++)
        {
            Assert.Equal(0, cached.Generate(Csv(turns[0]), logprobs).Timings.CachedTokenCount);   // a cold turn that snapshots
            Assert.Equal(turns[0].Length - 1, cached.Generate(Csv(turns[1]), logprobs).Timings.CachedTokenCount);
            cached.ClearRecurrentPrefixCache();
        }

        var only = Assert.Single(model.CreatedStates);
        Assert.True(IsDisposed(only));   // the real state behind the unmarkable wrapper
    }

    private static (int[] Ids, InferenceTimings Timings) Run(TextGenerator generator, string prompt)
    {
        var r = generator.Generate(prompt, Greedy);
        return (r.GeneratedTokenIds, r.Timings);
    }

    private static async Task<(int[] Ids, InferenceTimings Timings)> Stream(TextGenerator generator, string prompt)
    {
        var ids = new List<int>();
        InferenceTimings? timings = null;
        await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, Greedy))
        {
            if (token.FinishReason is null || token.FinishReason == FinishReason.Length) ids.Add(token.TokenId);
            timings = token.Timings ?? timings;
        }
        return (ids.ToArray(), timings!.Value);
    }
}
