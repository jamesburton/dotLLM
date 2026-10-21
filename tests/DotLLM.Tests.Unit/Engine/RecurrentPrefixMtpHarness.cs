using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Backend-agnostic drivers for issue #1088 (MTP on a recurrent-prefix hit): an agent conversation whose every prompt extends
/// the previous one, run through a prefix-cached MTP generator next to a cold one. CPU: <c>TextGeneratorRecurrentPrefixMtpTests</c>;
/// CUDA: <c>CudaQwen3HybridDenseMtpPrefixHitTests</c>.
/// </summary>
internal static class RecurrentPrefixMtpHarness
{
    /// <summary>Tokens generated per turn.</summary>
    public const int NewTokens = 12;

    /// <summary>Greedy options for every turn.</summary>
    public static InferenceOptions Greedy => new() { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = NewTokens };

    /// <summary>
    /// Four agent turns, each prompt extending the previous one. The first request sizes its KV for itself plus 256 positions
    /// of headroom, so the last turn (420 tokens) grows the KV on its hit and the resumed head's ceiling has to follow.
    /// Needs a context of at least 512.
    /// </summary>
    public static int[][] AgentTurns(int vocab)
    {
        int[] all = MtpLongContextHarness.Prompt(420, vocab);
        return [all[..40], all[..76], all[..112], all];
    }

    /// <summary>Comma-separated token ids, the form <see cref="TokenIdsTokenizer"/> reads.</summary>
    public static string Csv(int[] ids) => string.Join(",", ids);

    /// <summary>A generator over <paramref name="model"/> with token-id prompts.</summary>
    public static TextGenerator Generator(IModel model, Func<ModelConfig, int, IKvCache> kvFactory, bool mtp, bool prefixCache)
        => new(model, new TokenIdsTokenizer(model.Config.VocabSize), kvFactory, mtpEnabled: mtp, recurrentPrefixCache: prefixCache);

    /// <summary>
    /// Runs every agent turn cold (MTP, no prefix cache) and through the prefix cache (MTP), and asserts the hit turns hit, decode with
    /// MTP, produce the cold tokens and draft what the cold run drafts: every draft step's logits within <paramref name="tolerance"/>.
    /// Verified output equals plain greedy however bad the drafts are, so the draft logits are what observe the head itself.
    /// </summary>
    /// <param name="cachedInner">Model for the prefix-cached generator.</param>
    /// <param name="cachedKv">KV factory for <paramref name="cachedInner"/>.</param>
    /// <param name="coldInner">A separate instance of the same model for the cold generator.</param>
    /// <param name="coldKv">KV factory for <paramref name="coldInner"/>.</param>
    /// <param name="tolerance">Largest allowed |cold - hit| over each draft step's logits.</param>
    /// <param name="log">Receives one line per turn.</param>
    public static void AssertHitTurnsDraftLikeColdRuns(IModel cachedInner, Func<ModelConfig, int, IKvCache> cachedKv,
        IModel coldInner, Func<ModelConfig, int, IKvCache> coldKv, float tolerance, Action<string>? log = null)
    {
        var cachedModel = new DraftRecordingModel(cachedInner);
        var coldModel = new DraftRecordingModel(coldInner);
        var cached = Generator(cachedModel, cachedKv, mtp: true, prefixCache: true);
        var cold = Generator(coldModel, coldKv, mtp: true, prefixCache: false);
        int vocab = cachedInner.Config.VocabSize;
        try
        {
            int[][] turns = AgentTurns(vocab);
            for (int t = 0; t < turns.Length; t++)
            {
                coldModel.Drafts.Clear();
                var want = cold.Generate(Csv(turns[t]), Greedy);
                cachedModel.Drafts.Clear();
                var got = cached.Generate(Csv(turns[t]), Greedy);

                float worst = 0f;
                int steps = Math.Min(coldModel.Drafts.Count, cachedModel.Drafts.Count);
                for (int s = 0; s < steps; s++)
                    for (int v = 0; v < vocab; v++)
                        worst = Math.Max(worst, Math.Abs(coldModel.Drafts[s][v] - cachedModel.Drafts[s][v]));
                log?.Invoke($"turn {t}: prompt={turns[t].Length} cached={got.Timings.CachedTokenCount} drafted={got.Timings.SpeculativeDraftTokens}/{want.Timings.SpeculativeDraftTokens} accepted={got.Timings.SpeculativeAcceptedTokens}/{want.Timings.SpeculativeAcceptedTokens} draftSteps={cachedModel.Drafts.Count}/{coldModel.Drafts.Count} max|diff|={worst:G3}");

                Assert.Equal(t == 0 ? 0 : turns[t - 1].Length - 1, got.Timings.CachedTokenCount);
                Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
                Assert.NotEmpty(cachedModel.Drafts);
                Assert.Equal(coldModel.Drafts.Count, cachedModel.Drafts.Count);
                Assert.Equal(want.Timings.SpeculativeDraftTokens, got.Timings.SpeculativeDraftTokens);
                Assert.Equal(want.Timings.SpeculativeAcceptedTokens, got.Timings.SpeculativeAcceptedTokens);
                Assert.True(worst <= tolerance, $"turn {t} (cached {got.Timings.CachedTokenCount}): draft logits differ from the cold run by {worst}");
            }
        }
        finally
        {
            cached.ClearRecurrentPrefixCache();
        }

        // Every head state the cached generator made is released once the snapshot is: none leaked across the turns.
        Assert.NotEmpty(cachedModel.CreatedStates);
        Assert.All(cachedModel.CreatedStates, s => Assert.True(IsDisposed(s), "an MTP head state outlived the prefix cache"));
    }

    /// <summary>True when <paramref name="state"/> has been disposed (its <see cref="IMtpState.MarkPrefix"/> throws).</summary>
    public static bool IsDisposed(IMtpState state)
    {
        try { state.MarkPrefix(); return false; }
        catch (ObjectDisposedException) { return true; }
    }

    /// <summary>Prompts are comma-separated token ids; no EOS, so every turn generates <see cref="NewTokens"/> tokens.</summary>
    public sealed class TokenIdsTokenizer(int vocabSize) : ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => -1;   // the 12-token synthetic fixture would otherwise stop on its own EOS within a few tokens
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// A head state that cannot be marked or resumed (the interface defaults), like the Vulkan backends' states: a snapshot taken
    /// with it keeps no head, so a hit on it must decode plain.
    /// </summary>
    public sealed class UnmarkableMtpState(IMtpState inner) : IMtpState
    {
        public IMtpState Inner => inner;
        public int CurrentLength => inner.CurrentLength;
        public void Rollback(int length) => inner.Rollback(length);
        public ReadOnlySpan<float> CapturedHiddenRows => inner.CapturedHiddenRows;
        public int CapturedRowCount => inner.CapturedRowCount;
        public int HiddenSize => inner.HiddenSize;
        public void SeedFromCapturedRow(int rowIndex) => inner.SeedFromCapturedRow(rowIndex);
        public void Dispose() => inner.Dispose();
    }

    /// <summary>
    /// Forwards everything to the real model and records a copy of every MTP draft step's logits. With <c>hideMarks</c> it hands
    /// out <see cref="UnmarkableMtpState"/> wrappers (unwrapped again on the way into the real model).
    /// </summary>
    /// <remarks>
    /// <see cref="IModel.SupportsMtpArgMax"/> is deliberately not forwarded, so the decoder always calls
    /// <see cref="ForwardMtp"/> and every draft step is recorded.
    /// </remarks>
    public sealed class DraftRecordingModel(IModel inner, bool hideMarks = false) : IModel
    {
        public List<float[]> Drafts { get; } = [];

        /// <summary>Every head state this model created, in order.</summary>
        public List<IMtpState> CreatedStates { get; } = [];

        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public void Dispose() { }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => inner.Forward(t, p, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => inner.Forward(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => inner.Forward(t, p, d, kv, last);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => inner.Forward(t, p, d, kv, a);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => inner.Forward(t, p, d, kv, a, Unwrap(m));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => inner.Forward(t, p, d, kv, a, Unwrap(m), last);

        public bool RequiresPerSequenceState => inner.RequiresPerSequenceState;
        public void ResetSequenceState() => inner.ResetSequenceState();
        public bool SupportsRecurrentStateCheckpoint => inner.SupportsRecurrentStateCheckpoint;
        public object? CheckpointRecurrentState() => inner.CheckpointRecurrentState();
        public void RestoreRecurrentState(object? checkpoint) => inner.RestoreRecurrentState(checkpoint);
        public bool SupportsRecurrentRowSnapshots => inner.SupportsRecurrentRowSnapshots;
        public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, IMtpState? m)
            => inner.ForwardWithRecurrentSnapshots(t, p, d, kv, Unwrap(m));
        public void RestoreRecurrentStateToRow(int row) => inner.RestoreRecurrentStateToRow(row);

        public bool SupportsMtp => inner.SupportsMtp;
        public int MaxAllRowLogitsLength => inner.MaxAllRowLogitsLength;
        public bool TrySetAllRowLogitsLimit(int maxSeqLen) => inner.TrySetAllRowLogitsLimit(maxSeqLen);
        public int PreferredPrefillChunkTokens => inner.PreferredPrefillChunkTokens;
        public IMtpState? CreateMtpState() => inner.CreateMtpState();
        public IMtpState? CreateMtpState(int maxSequenceLength)
        {
            var state = inner.CreateMtpState(maxSequenceLength);
            if (state is null) return null;
            CreatedStates.Add(state);
            return hideMarks ? new UnmarkableMtpState(state) : state;
        }

        private static IMtpState? Unwrap(IMtpState? state) => state is UnmarkableMtpState u ? u.Inner : state;

        public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            ITensor logits = inner.ForwardMtp(Unwrap(state)!, tokenId, position);
            Drafts.Add(new ReadOnlySpan<float>((void*)logits.DataPointer, Config.VocabSize).ToArray());
            return logits;
        }
    }
}
