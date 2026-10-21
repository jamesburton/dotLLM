using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The recurrent prefix cache snapshots (KV + recurrent state) at the prefix two consecutive requests share, then serves
/// later requests that start with it by restoring the state, rolling the KV back and prefilling only the suffix.
/// The fake model's next token is a function of its WHOLE history (a running hash), so a missed restore, a stale state or
/// a wrongly positioned suffix shows up as different tokens, and the recorded forward lengths show what was skipped.
/// </summary>
public sealed class TextGeneratorRecurrentPrefixCacheTests
{
    private const int Vocab = 64;
    private const int HiddenSize = 8, NumKvHeads = 1, HeadDim = 4, MaxSeqLen = 512;
    private const int PrefixLen = 20;

    private static string Prompt(params int[] suffix)
        => string.Join(",", Enumerable.Range(2, PrefixLen).Concat(suffix));

    private static InferenceOptions Opts => new() { Temperature = 0f, MaxTokens = 4 };

    private static (TextGenerator Cached, TextGenerator Plain, HistoryModel Model) Build(bool sharedStorage = false, DotLLM.Engine.PromptCache.RecurrentPrefixCacheOptions? snapshots = null)
    {
        var model = new HistoryModel();
        var tok = new CsvTokenizer();
        Func<ModelConfig, int, IKvCache> kv = sharedStorage
            ? (cfg, size) => new SharedStorageKv(new SimpleKvCache(KvGeometry.FromConfig(cfg), size))
            : (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg), size);
        return (new TextGenerator(model, tok, kv, recurrentPrefixCache: true, recurrentPrefixCacheOptions: snapshots),
                new TextGenerator(model, tok, kv), model);
    }

    /// <summary>The prompt grown by <paramref name="count"/> distinct tokens starting at <paramref name="from"/>: an agent step appending output and tool results.</summary>
    private static int[] Run(int from, int count) => [.. Enumerable.Range(from, count)];

    private static InferenceResponse Check(TextGenerator cached, TextGenerator plain, HistoryModel model, string prompt)
    {
        var want = plain.Generate(prompt, Opts);
        model.Calls.Clear();
        var got = cached.Generate(prompt, Opts);
        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Empty(model.KvLengthMismatches);
        return got;
    }

    [Fact]
    public void LongerFollowUp_BeyondTheKvHeadroom_StillReusesTheSnapshot_ByGrowingTheKv()
    {
        // Issue #1025: the snapshot's KV was sized for its own request + 256 tokens, and a later prompt needing more was a
        // full re-prefill, even though it starts with the snapshot. An agent step adds more than 256 tokens, so in an
        // agent loop every request missed. The KV now grows (rows kept) instead.
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));                     // snapshot at PrefixLen, KV for ~281 positions

        var got = Check(cached, plain, model, Prompt([31, .. Run(100, 300)]));

        Assert.Equal(PrefixLen, got.Timings.CachedTokenCount);
    }

    [Fact]
    public void Hit_MovesTheSnapshotForward_SoAnAgentLoopPrefillsOnlyItsNewTurn()
    {
        // Issue #1025: a hit never re-snapshotted, so every later request re-prefilled everything since the FIRST snapshot.
        // Issue #1080: the snapshot then sat at what a prompt shared with the previous one, so each hit reused the prompt from
        // two turns back and re-prefilled the last two turns. A prompt that extends the whole previous one is an agent loop:
        // the snapshot goes at its own end (one token short, to produce logits), and the next turn prefills only its new part.
        // Each step's prompt extends the previous one by 20 tokens.
        var (cached, plain, model) = Build();
        int[] turn = [30, 31];
        Check(cached, plain, model, Prompt(turn));                    // r1: nothing to share yet
        turn = [.. turn, .. Run(100, 20)];
        Check(cached, plain, model, Prompt(turn));                    // r2 extends r1: snapshot at |r2| - 1 = 41
        turn = [.. turn, .. Run(200, 20)];
        Assert.Equal(PrefixLen + 21, Check(cached, plain, model, Prompt(turn)).Timings.CachedTokenCount);   // r3 hits 41, moves it to |r3| - 1 = 61
        turn = [.. turn, .. Run(300, 20)];

        var r4 = Check(cached, plain, model, Prompt(turn));

        Assert.Equal(PrefixLen + 41, r4.Timings.CachedTokenCount);
        Assert.Equal(21, model.Calls.Take(2).Sum());                  // only r4's new turn ran, plus the token held back for logits
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FirstPrompt_SnapshotsAtItsEnd_SoTheSecondAgentTurnHits(bool sharedStorage)
    {
        // Issue #1088 (turn 1 of the #1087 replay): the first prompt had no previous one to share with, so nothing was
        // snapshotted and the second agent turn re-prefilled everything. With no snapshot held, there is nothing to evict:
        // snapshot the prompt's own end, betting that the next prompt extends it.
        var (cached, plain, model) = Build(sharedStorage);
        int[] turn = [30, 31];
        Assert.Equal(0, Check(cached, plain, model, Prompt(turn)).Timings.CachedTokenCount);
        turn = [.. turn, .. Run(100, 20)];

        var r2 = Check(cached, plain, model, Prompt(turn));

        Assert.Equal(PrefixLen + 1, r2.Timings.CachedTokenCount);    // |r1| - 1
        Assert.Equal(21, model.Calls.Take(2).Sum());                 // r2's 20 new tokens plus r1's last, which r1 held back for its logits
    }

    [Fact]
    public void UnrelatedPrompt_OnSharedStorage_StartsANewConversation_ThatItsNextTurnHits()
    {
        // On a model-owned KV a miss drops the snapshot, so an unrelated prompt (a new conversation) evicts nothing by
        // snapshotting its own end, and that conversation's second turn hits.
        var (cached, plain, model) = Build(sharedStorage: true);
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));
        int[] other = [.. Enumerable.Range(40, 25)];
        Assert.Equal(0, Check(cached, plain, model, string.Join(",", other)).Timings.CachedTokenCount);

        var next = Check(cached, plain, model, string.Join(",", [.. other, .. Run(100, 10)]));

        Assert.Equal(other.Length - 1, next.Timings.CachedTokenCount);
    }

    [Fact]
    public void NewConversation_SharingOnlyATemplateHeader_StillSnapshotsItsEnd()
    {
        // A real chat template starts every conversation with the same few tokens (BOS, the system role header). A new
        // conversation that shares only those with the previous prompt, fewer than RecurrentPrefixMinTokens, must still snapshot
        // its own end, or its second turn re-prefills everything (PR #1091 review: "shares nothing" means "nothing usable").
        var (cached, plain, model) = Build(sharedStorage: true);
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));
        int[] other = [2, 3, 4, .. Enumerable.Range(40, 25)];          // Prompt() starts 2, 3, 4, ...
        Assert.Equal(0, Check(cached, plain, model, string.Join(",", other)).Timings.CachedTokenCount);

        var next = Check(cached, plain, model, string.Join(",", [.. other, .. Run(100, 10)]));

        Assert.Equal(other.Length - 1, next.Timings.CachedTokenCount);
    }

    [Fact]
    public void PromptThatDoesNotExtendThePreviousOne_StillSnapshotsTheSharedPrefix()
    {
        // The classifier shape (issue #1025's first case): a fixed system prompt and a different question each time. Snapshotting
        // at a prompt's end would never hit; the shared prefix does.
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30, 31));
        Check(cached, plain, model, Prompt(40, 41));                  // shares PrefixLen with the previous prompt but does not extend it

        var got = Check(cached, plain, model, Prompt(50, 51));

        Assert.Equal(PrefixLen, got.Timings.CachedTokenCount);
        Assert.Equal(2, model.Calls[0]);
    }

    [Fact]
    public void ClassifierOnSharedStorage_KeepsTheSharedPrefixBoundary_AfterTheFirstPromptsEndSnapshotMisses()
    {
        // Issue #1088: the first prompt snapshots its end, and on a model-owned KV the second (a different question) misses
        // and drops it. That leaves no snapshot, but the second prompt does share the system prompt with the first, so it
        // must snapshot there, not at its own end, or the third question misses too.
        var (cached, plain, model) = Build(sharedStorage: true);
        Check(cached, plain, model, Prompt(30, 31));
        Assert.Equal(0, Check(cached, plain, model, Prompt(40, 41)).Timings.CachedTokenCount);

        var got = Check(cached, plain, model, Prompt(50, 51));

        Assert.Equal(PrefixLen, got.Timings.CachedTokenCount);
    }

    [Fact]
    public void SharedKvStorage_DropsTheSnapshot_WhenAnotherRequestRunsInBetween()
    {
        // Issue #1025: a model-owned KV (the CUDA hybrid's length-only handle) is overwritten by any other request's
        // forwards. A later hit on the old snapshot would attend over that request's rows, so the snapshot has to go.
        var (cached, plain, model) = Build(sharedStorage: true);
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));                      // snapshot
        Assert.Equal(PrefixLen, Check(cached, plain, model, Prompt(32)).Timings.CachedTokenCount);

        Check(cached, plain, model, string.Join(",", Enumerable.Range(40, 25)));   // unrelated: drops the snapshot (and takes its own)

        Assert.Equal(0, Check(cached, plain, model, Prompt(33)).Timings.CachedTokenCount);
    }

    [Fact]
    public void FailedRequest_DropsTheSnapshot_SoTheNextRequestDoesNotRetryTheSameFailure()
    {
        // PR #1026 re-review: a hit that grew its handle and then failed (a real OOM after the free-VRAM admission, say) left
        // the snapshot holding a handle whose MaxLength was already raised. The next same-prefix request then skipped the
        // admission and retried the same allocation, failing on every agent step. A failed request now drops the snapshot.
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));                     // snapshot
        Assert.Equal(PrefixLen, Check(cached, plain, model, Prompt(32)).Timings.CachedTokenCount);

        model.FailNextForward = true;
        Assert.Throws<InvalidOperationException>(() => cached.Generate(Prompt(33), Opts));

        var after = Check(cached, plain, model, Prompt(34));
        Assert.Equal(0, after.Timings.CachedTokenCount);
    }

    [Fact]
    public async Task FailedStreamingRequest_DropsTheSnapshot()
    {
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));

        model.FailNextForward = true;
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in cached.GenerateStreamingTokensAsync(Prompt(33), Opts)) { }
        });

        Assert.Equal(0, Check(cached, plain, model, Prompt(34)).Timings.CachedTokenCount);
    }

    [Fact]
    public async Task CompletedStreamingRequest_KeepsTheSnapshot()
    {
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));

        await foreach (var _ in cached.GenerateStreamingTokensAsync(Prompt(33), Opts)) { }

        Assert.Equal(PrefixLen, Check(cached, plain, model, Prompt(34)).Timings.CachedTokenCount);
    }

    [Fact]
    public void ExclusiveKvStorage_KeepsTheSnapshot_AcrossAnUnrelatedRequest()
    {
        var (cached, plain, model) = Build();
        Check(cached, plain, model, Prompt(30));
        Check(cached, plain, model, Prompt(31));
        Check(cached, plain, model, string.Join(",", Enumerable.Range(40, 25)));

        Assert.Equal(PrefixLen, Check(cached, plain, model, Prompt(33)).Timings.CachedTokenCount);
    }

    [Fact]
    public void SharedPrefix_IsSkippedOnLaterRequests_AndTokensMatchTheUncachedRun()
    {
        var (cached, plain, model) = Build();
        int[][] suffixes = [[30, 31], [40, 41, 42], [50], [30, 31], [60, 61]];

        for (int i = 0; i < suffixes.Length; i++)
        {
            string prompt = Prompt(suffixes[i]);
            var want = plain.Generate(prompt, Opts);
            model.Calls.Clear();
            var got = cached.Generate(prompt, Opts);

            Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
            Assert.Empty(model.KvLengthMismatches);

            int prefilled = model.Calls[0];
            if (i >= 2)
            {
                // request 0 has no previous prompt, request 1 takes the snapshot, request 2 onwards hit it
                Assert.Equal(PrefixLen, got.Timings.CachedTokenCount);
                Assert.Equal(suffixes[i].Length, prefilled);          // only the suffix ran through the model
            }
            else
            {
                Assert.Equal(0, got.Timings.CachedTokenCount);
            }
        }
    }

    [Fact]
    public void NonMatchingPrompt_FallsBackToAFullPrefill_AndStaysCorrect()
    {
        var (cached, plain, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);                       // establishes the snapshot

        string other = string.Join(",", Enumerable.Range(40, 25));   // different prefix entirely
        var want = plain.Generate(other, Opts);
        model.Calls.Clear();
        var got = cached.Generate(other, Opts);

        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Equal(0, got.Timings.CachedTokenCount);
        Assert.Equal(25, model.Calls[0]);
    }

    [Fact]
    public void PromptEqualToTheSnapshotPrefix_IsNotReused_BecauseASuffixTokenMustProduceLogits()
    {
        var (cached, plain, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);

        string exactlyThePrefix = Prompt();                          // 20 tokens == snapshot length
        var want = plain.Generate(exactlyThePrefix, Opts);
        var got = cached.Generate(exactlyThePrefix, Opts);

        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Equal(0, got.Timings.CachedTokenCount);
    }

    [Fact]
    public void ClearRecurrentPrefixCache_ForcesAFullPrefillAgain()
    {
        var (cached, _, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);
        Assert.Equal(PrefixLen, cached.Generate(Prompt(33), Opts).Timings.CachedTokenCount);

        cached.ClearRecurrentPrefixCache();
        model.Calls.Clear();
        var after = cached.Generate(Prompt(34), Opts);

        Assert.Equal(0, after.Timings.CachedTokenCount);
        // The whole prompt ran again. It runs as two forwards (split where the new first-prompt snapshot is taken, #1088),
        // then 3 one-token decode steps for the 4 generated tokens.
        Assert.Equal(PrefixLen + 1 + 3, model.Calls.Sum());
    }

    [Fact]
    public void WithoutTheFlag_NothingIsCached()
    {
        var (_, plain, _) = Build();
        for (int i = 0; i < 4; i++)
            Assert.Equal(0, plain.Generate(Prompt(30 + i), Opts).Timings.CachedTokenCount);
    }

    [Fact]
    public void MultiSlot_AlternatingUnrelatedConversationsWarmOnFirstSightAndMatchColdReplay()
    {
        var (cached, plain, model) = Build(sharedStorage: true, snapshots: new() { MaxEntries = 2, HostBudgetBytes = 100000 });
        string a = Prompt(30), b = string.Join(",", Enumerable.Range(35, PrefixLen + 1));
        Assert.Equal(0, Check(cached, plain, model, a).Timings.CachedTokenCount);
        Assert.Equal(0, Check(cached, plain, model, b).Timings.CachedTokenCount);
        for (int i = 0; i < 3; i++)
        {
            var gotA = Check(cached, plain, model, a); Assert.Equal(PrefixLen, gotA.Timings.CachedTokenCount);
            var gotB = Check(cached, plain, model, b); Assert.Equal(PrefixLen, gotB.Timings.CachedTokenCount);
        }
        Assert.Equal(2, cached.RecurrentPrefixStatistics.Entries);
        Assert.Equal(6, cached.RecurrentPrefixStatistics.HostHits);
        cached.ClearRecurrentPrefixCache();
    }
    [Fact]
    public void MultiSlot_AlternatingGrowingConversationsReplaceTheirOwnBoundary()
    {
        var (cached, plain, model) = Build(snapshots: new() { MaxEntries = 2, HostBudgetBytes = 100000 });
        int[] a = Enumerable.Range(2, PrefixLen + 1).ToArray(), b = Enumerable.Range(35, PrefixLen + 1).ToArray();
        Check(cached, plain, model, string.Join(",", a)); Check(cached, plain, model, string.Join(",", b));
        for (int turn = 0; turn < 3; turn++)
        {
            int oldA = a.Length, oldB = b.Length;
            a = [.. a, .. Enumerable.Repeat(20 + turn, 20)];
            b = [.. b, .. Enumerable.Repeat(40 + turn, 20)];
            Assert.Equal(oldA - 1, Check(cached, plain, model, string.Join(",", a)).Timings.CachedTokenCount);
            Assert.Equal(oldB - 1, Check(cached, plain, model, string.Join(",", b)).Timings.CachedTokenCount);
        }
        cached.ClearRecurrentPrefixCache();
    }
    // ── fakes ──

    /// <summary>A KV whose storage another cache's forwards overwrite, like the CUDA hybrid's model-owned handle.</summary>
    private sealed class SharedStorageKv(SimpleKvCache inner) : IKvCache
    {
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public bool HasExclusiveStorage => false;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
        public ITensor GetKeys(int layerIndex) => inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => inner.Rollback(length);
        public void Dispose() => inner.Dispose();
    }

    private sealed class CsvTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => Vocab - 1;
        public int[] Encode(string text)
            => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => Encode(text).Length;
    }

    /// <summary>
    /// A recurrent model: state S = (S * 31 + token + 1) mod 9973 after every token; the next-token logits are one-hot on
    /// S mod (Vocab - 1) (never EOS). Supports checkpoint/restore like the GDN hybrids.
    /// </summary>
    private sealed class HistoryModel : IModel, IRecurrentPrefixSnapshotProvider
    {
        private int _state;

        /// <summary>Token count of every Forward call, in order.</summary>
        public List<int> Calls { get; } = [];

        /// <summary>When set, the next Forward throws (and clears the flag), like an OOM mid-request.</summary>
        public bool FailNextForward { get; set; }

        /// <summary>Forwards whose KV cache length differed from their first position (a stale or un-rolled-back KV).</summary>
        public List<string> KvLengthMismatches { get; } = [];

        public ModelConfig Config => new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = NumKvHeads, NumKvHeads = NumKvHeads,
            HiddenSize = HiddenSize, IntermediateSize = HiddenSize * 4, HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public bool RequiresPerSequenceState => true;
        public void ResetSequenceState() => _state = 0;
        public bool SupportsRecurrentStateCheckpoint => true;
        public object? CheckpointRecurrentState() => _state;
        public void RestoreRecurrentState(object? checkpoint) { if (checkpoint is int s) _state = s; }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Core(t, p, d, null);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Core(t, p, d, kv);

        public string PrefixSnapshotIdentity => "history-model-fixed-config";
        public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage) => storage == RecurrentPrefixSnapshotStorage.Host;
        public long EstimatePrefixSnapshotBytes(IKvCache kvCache, int prefixLength) => 4 + prefixLength * 4L * 8;
        public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kvCache, int prefixLength, RecurrentPrefixSnapshotStorage storage)
        {
            var copy = new SimpleKvCache(1, 1, 4, prefixLength);
            var keys = kvCache.GetKeysRef(0); var values = kvCache.GetValuesRef(0);
            copy.Update(new TensorRef(prefixLength, 4, DType.Float32, -1, keys.DataPointer),
                new TensorRef(prefixLength, 4, DType.Float32, -1, values.DataPointer), Enumerable.Range(0, prefixLength).ToArray(), 0);
            return new PrefixSnapshot(copy, _state, prefixLength, PrefixSnapshotIdentity);
        }
        public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage) => throw new NotSupportedException();
        public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kvCache)
        {
            var snap = (PrefixSnapshot)snapshot;
            kvCache.Rollback(0);
            kvCache.Update(snap.Kv.GetKeysRef(0), snap.Kv.GetValuesRef(0), Enumerable.Range(0, snap.PrefixLength).ToArray(), 0);
            _state = snap.State;
        }
        private sealed class PrefixSnapshot(SimpleKvCache kv, int state, int length, string identity) : IRecurrentPrefixSnapshot
        {
            public SimpleKvCache Kv => kv;
            public int State => state;
            public int PrefixLength => length;
            public long ByteCount => 4 + length * 4L * 8;
            public string Identity => identity;
            public RecurrentPrefixSnapshotStorage Storage => RecurrentPrefixSnapshotStorage.Host;
            public void Dispose() => kv.Dispose();
        }
        private unsafe ITensor Core(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            int seqLen = tokenIds.Length;
            if (FailNextForward)
            {
                FailNextForward = false;
                throw new InvalidOperationException("injected forward failure");
            }
            Calls.Add(seqLen);
            if (kvCache is not null && kvCache.CurrentLength != positions[0])
                KvLengthMismatches.Add($"kv length {kvCache.CurrentLength} but forward starts at position {positions[0]}");

            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)seqLen * Vocab * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int r = 0; r < seqLen; r++)
            {
                _state = (_state * 31 + tokenIds[r] + 1) % 9973;
                var row = new Span<float>(dst + (long)r * Vocab, Vocab);
                row.Fill(-10f);
                row[_state % (Vocab - 1)] = 10f;
            }

            if (kvCache is not null)
            {
                int kvStride = NumKvHeads * HeadDim;
                nuint bytes = (nuint)(seqLen * kvStride * sizeof(float));
                nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                NativeMemory.Clear((void*)kPtr, bytes);
                NativeMemory.Clear((void*)vPtr, bytes);
                kvCache.Update(new TensorRef(seqLen, kvStride, DType.Float32, -1, kPtr),
                               new TensorRef(seqLen, kvStride, DType.Float32, -1, vPtr), positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            return new UnmanagedTensor(new TensorShape(seqLen, Vocab), DType.Float32, deviceId, ptr);
        }
    }
}
