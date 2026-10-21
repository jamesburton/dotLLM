using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1088: a turn that hits the recurrent prefix snapshot decodes with MTP. The head's KV for the cached prefix is kept
/// with the snapshot (with the trunk hidden row the next absorb pairs with), so the head drafts on a hit exactly as it would
/// after a cold prefill of the whole prompt. Runs the real CPU hybrid on the synthetic Qwen3.5 fixture with an MTP head.
/// </summary>
public sealed class TextGeneratorRecurrentPrefixMtpTests : IDisposable
{
    // The hit prefills only the suffix, the cold run the whole prompt: the CPU GEMM tiles by row count, so the trunk rows
    // (and the head's inputs) differ by a few ULPs. A head that pairs the suffix with the wrong trunk row, or attends over
    // missing or foreign rows, is off by orders of magnitude more.
    private const float DraftLogitTolerance = 1e-3f;

    private const int NewTokens = 12;
    private readonly string _scratch;

    public TextGeneratorRecurrentPrefixMtpTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-prefix-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static InferenceOptions Greedy => new() { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens };

    /// <summary>Three agent turns, each prompt extending the previous one.</summary>
    private static int[][] AgentTurns(int vocab)
    {
        int[] all = MtpLongContextHarness.Prompt(112, vocab);
        return [all[..40], all[..76], all[..112]];
    }

    private (GgufFile Gguf, ModelConfig Config) Fixture(string name)
    {
        var gguf = GgufFile.Open(SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, name), withMtp: true, contextLength: 512));
        return (gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
    }

    private static Qwen3HybridDenseTransformerModel Load(GgufFile gguf, ModelConfig config)
        => (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

    private static TextGenerator Generator(IModel model, bool mtp, bool prefixCache)
        => new(model, new TokenIdsTokenizer(model.Config.VocabSize),
            (cfg, size) => new SimpleKvCache(((Qwen3HybridDenseTransformerModel)Unwrap(model)).AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, size),
            mtpEnabled: mtp, recurrentPrefixCache: prefixCache);

    private static IModel Unwrap(IModel model) => model is DraftRecordingModel r ? r.Inner : model;

    private static string Csv(int[] ids) => string.Join(",", ids);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AgentTurnThatHitsThePrefix_DecodesWithMtp_AndMatchesPlainGreedy(bool streaming)
    {
        var (gguf, config) = Fixture("prefix-mtp-greedy.gguf");
        using var _ = gguf;
        using var cachedModel = Load(gguf, config);
        using var plainModel = Load(gguf, config);
        var cached = Generator(cachedModel, mtp: true, prefixCache: true);
        var plain = Generator(plainModel, mtp: false, prefixCache: false);

        int[][] turns = AgentTurns(config.VocabSize);
        for (int t = 0; t < turns.Length; t++)
        {
            var want = plain.Generate(Csv(turns[t]), Greedy);
            var (ids, timings) = streaming ? await Stream(cached, Csv(turns[t])) : Run(cached, Csv(turns[t]));

            Assert.Equal(want.GeneratedTokenIds, ids);
            Assert.True(timings.SpeculativeDraftTokens > 0, $"turn {t} decoded without MTP (cached {timings.CachedTokenCount}, generated [{string.Join(",", ids)}])");
            if (t > 0)
                Assert.Equal(turns[t - 1].Length - 1, timings.CachedTokenCount);   // the hit is what this test is about
        }
        cached.ClearRecurrentPrefixCache();
    }

    [Fact]
    public void HitTurn_DraftsWhatAColdMtpRunOfTheSamePromptDrafts()
    {
        // Verified output equals plain greedy however bad the drafts are, so this checks the head itself: on the hit, every
        // draft step's logits must match a cold MTP run's. A head resumed with a zeroed or stale carry row, at the wrong
        // length, or over another conversation's rows drafts differently.
        var (gguf, config) = Fixture("prefix-mtp-drafts.gguf");
        using var _ = gguf;
        using var cachedInner = Load(gguf, config);
        using var coldInner = Load(gguf, config);
        var cachedModel = new DraftRecordingModel(cachedInner);
        var coldModel = new DraftRecordingModel(coldInner);
        var cached = Generator(cachedModel, mtp: true, prefixCache: true);
        var cold = Generator(coldModel, mtp: true, prefixCache: false);

        int[][] turns = AgentTurns(config.VocabSize);
        for (int t = 0; t < turns.Length; t++)
        {
            coldModel.Drafts.Clear();
            var want = cold.Generate(Csv(turns[t]), Greedy);
            cachedModel.Drafts.Clear();
            var got = cached.Generate(Csv(turns[t]), Greedy);

            Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
            Assert.Equal(want.Timings.SpeculativeDraftTokens, got.Timings.SpeculativeDraftTokens);
            Assert.Equal(want.Timings.SpeculativeAcceptedTokens, got.Timings.SpeculativeAcceptedTokens);
            Assert.Equal(coldModel.Drafts.Count, cachedModel.Drafts.Count);
            Assert.NotEmpty(cachedModel.Drafts);
            for (int s = 0; s < coldModel.Drafts.Count; s++)
            {
                float worst = 0f;
                for (int v = 0; v < config.VocabSize; v++)
                    worst = Math.Max(worst, Math.Abs(coldModel.Drafts[s][v] - cachedModel.Drafts[s][v]));
                Assert.True(worst <= DraftLogitTolerance, $"turn {t} (cached {got.Timings.CachedTokenCount}) draft step {s}: max |diff| {worst}");
            }
        }
        cached.ClearRecurrentPrefixCache();
    }

    [Fact]
    public void HitAfterAPlainTurnThatMovedTheSnapshot_DecodesPlain_AndStaysCorrect()
    {
        // A turn that hits but decodes plain (here: logprobs requested) never absorbs its suffix into the head. If it moves
        // the snapshot forward, the head no longer covers the new boundary, so the next hit must not resume it.
        var (gguf, config) = Fixture("prefix-mtp-plain-move.gguf");
        using var _ = gguf;
        using var cachedModel = Load(gguf, config);
        using var plainModel = Load(gguf, config);
        var cached = Generator(cachedModel, mtp: true, prefixCache: true);
        var plain = Generator(plainModel, mtp: false, prefixCache: false);
        int[][] turns = AgentTurns(config.VocabSize);

        cached.Generate(Csv(turns[0]), Greedy);
        var plainHit = cached.Generate(Csv(turns[1]), new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens, Logprobs = true });
        Assert.Equal(turns[0].Length - 1, plainHit.Timings.CachedTokenCount);
        Assert.Equal(0, plainHit.Timings.SpeculativeDraftTokens);

        var next = cached.Generate(Csv(turns[2]), Greedy);

        Assert.Equal(turns[1].Length - 1, next.Timings.CachedTokenCount);
        Assert.Equal(0, next.Timings.SpeculativeDraftTokens);
        Assert.Equal(plain.Generate(Csv(turns[2]), Greedy).GeneratedTokenIds, next.GeneratedTokenIds);
        cached.ClearRecurrentPrefixCache();
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

    private sealed class TokenIdsTokenizer(int vocabSize) : ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => -1;   // the 12-token fixture would otherwise stop on its own EOS within a few tokens
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>Forwards everything to the real model and records a copy of every MTP draft step's logits.</summary>
    private sealed class DraftRecordingModel(IModel inner) : IModel
    {
        public IModel Inner => inner;
        public List<float[]> Drafts { get; } = [];

        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public void Dispose() { }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => inner.Forward(t, p, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => inner.Forward(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => inner.Forward(t, p, d, kv, last);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => inner.Forward(t, p, d, kv, a);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => inner.Forward(t, p, d, kv, a, m);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => inner.Forward(t, p, d, kv, a, m, last);

        public bool RequiresPerSequenceState => inner.RequiresPerSequenceState;
        public void ResetSequenceState() => inner.ResetSequenceState();
        public bool SupportsRecurrentStateCheckpoint => inner.SupportsRecurrentStateCheckpoint;
        public object? CheckpointRecurrentState() => inner.CheckpointRecurrentState();
        public void RestoreRecurrentState(object? checkpoint) => inner.RestoreRecurrentState(checkpoint);
        public bool SupportsRecurrentRowSnapshots => inner.SupportsRecurrentRowSnapshots;
        public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, IMtpState? m)
            => inner.ForwardWithRecurrentSnapshots(t, p, d, kv, m);
        public void RestoreRecurrentStateToRow(int row) => inner.RestoreRecurrentStateToRow(row);

        public bool SupportsMtp => inner.SupportsMtp;
        public int MaxAllRowLogitsLength => inner.MaxAllRowLogitsLength;
        public bool TrySetAllRowLogitsLimit(int maxSeqLen) => inner.TrySetAllRowLogitsLimit(maxSeqLen);
        public int PreferredPrefillChunkTokens => inner.PreferredPrefillChunkTokens;
        public IMtpState? CreateMtpState() => inner.CreateMtpState();
        public IMtpState? CreateMtpState(int maxSequenceLength) => inner.CreateMtpState(maxSequenceLength);

        public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            ITensor logits = inner.ForwardMtp(state, tokenId, position);
            Drafts.Add(new ReadOnlySpan<float>((void*)logits.DataPointer, Config.VocabSize).ToArray());
            return logits;
        }
    }
}
