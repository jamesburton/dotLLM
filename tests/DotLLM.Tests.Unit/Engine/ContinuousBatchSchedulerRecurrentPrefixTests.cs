using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The continuous-batch scheduler's recurrent prefix snapshots, with the chat template's generation prompt passed as a hint
/// (issue #1093). The fake model threads a per-sequence state whose next token depends on the WHOLE history, so a wrong
/// restore shows up as different tokens; each request's cached count shows where its snapshot sat.
/// </summary>
public sealed class ContinuousBatchSchedulerRecurrentPrefixTests
{
    private const int Vocab = 64, HiddenSize = 8, HeadDim = 4, MaxSeqLen = 8192, PrefixLen = 20;
    private const string GenerationSuffix = "90,91";
    private static readonly int[] G = [90, 91];
    private static readonly int[] P1 = [.. Enumerable.Range(2, PrefixLen), 30, 31];

    private static int[] Run(int from, int count) => [.. Enumerable.Range(from, count)];

    [Fact]
    public async Task DivergingHistory_WithSuffixHint_HitsTheWholePreviousTurn()
    {
        // Turn N renders as H + G; turn N+1 re-renders turn N's reply as R, which differs from G. The snapshot must sit just
        // before G and advance every turn: turn 2 reuses P1, turn 3 reuses P1 + R1 + U2.
        using var model = new ThreadedHistoryModel();
        var tok = new CsvTokenizer();
        Func<ModelConfig, int, IKvCache> kvf = (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg), size);
        var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 4, GenerationPromptSuffix = GenerationSuffix };
        using var scheduler = new ContinuousBatchScheduler(model, tok, kvf,
            new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = 4 });
        int[] h2 = [.. P1, .. Run(100, 20), .. Run(150, 20)];
        int[] h3 = [.. h2, .. Run(200, 20), .. Run(250, 20)];

        var turn1 = await RunOne(scheduler, [.. P1, .. G], opts);
        var turn2 = await RunOne(scheduler, [.. h2, .. G], opts);
        var turn3 = await RunOne(scheduler, [.. h3, .. G], opts);

        Assert.Equal(0, turn1.Timings.CachedTokenCount);
        Assert.Equal(P1.Length, turn2.Timings.CachedTokenCount);
        Assert.Equal(h2.Length, turn3.Timings.CachedTokenCount);

        // A cold run of each prompt on a fresh scheduler produces the same tokens: the restores were exact.
        await AssertMatchesColdRun([.. h2, .. G], turn2, kvf, opts);
        await AssertMatchesColdRun([.. h3, .. G], turn3, kvf, opts);
    }

    [Fact]
    public async Task FourTurnConversation_WithSuffixHint_KeepsOneEntry_AndDisposesTheSupersededOnes()
    {
        // A hit that moves its snapshot forward replaces the entry it extended. Keeping it would fill the cache with nested
        // copies of one conversation's history (at 128k, several near-full KV copies) and evict other sessions.
        using var model = new ThreadedHistoryModel();
        using var scheduler = NewScheduler(model);
        int[] history = P1;
        int expectedCached = 0;
        for (int turn = 1; turn <= 4; turn++)
        {
            var got = await RunOne(scheduler, [.. history, .. G], HintedOpts);

            Assert.Equal(expectedCached, got.Timings.CachedTokenCount);
            Assert.Equal(1, scheduler.RecurrentPrefixEntryCount);
            Assert.Equal(turn, model.Snapshots.Count);
            Assert.All(model.Snapshots.Take(turn - 1), s => Assert.True(s.IsDisposed));
            Assert.False(model.Snapshots[^1].IsDisposed);
            expectedCached = history.Length;
            history = [.. history, .. Run(100 * turn, 20), .. Run(100 * turn + 50, 20)];
        }
    }

    [Fact]
    public async Task ExactExtension_WithSuffixHint_HitsUpToThePreviousGenerationPrompt()
    {
        // ChatML, or reasoning sent back: each prompt extends the whole previous one, G included. The scheduler has no
        // extension bet (it never had one), so it reuses the previous turn up to its generation prompt.
        using var model = new ThreadedHistoryModel();
        using var scheduler = NewScheduler(model);
        int[] t1 = [.. P1, .. G];
        int[] t2 = [.. t1, .. Run(100, 20), .. Run(150, 20), .. G];
        int[] t3 = [.. t2, .. Run(200, 20), .. Run(250, 20), .. G];

        Assert.Equal(0, (await RunOne(scheduler, t1, HintedOpts)).Timings.CachedTokenCount);
        var turn2 = await RunOne(scheduler, t2, HintedOpts);
        var turn3 = await RunOne(scheduler, t3, HintedOpts);

        Assert.Equal(P1.Length, turn2.Timings.CachedTokenCount);
        Assert.Equal(t2.Length - G.Length, turn3.Timings.CachedTokenCount);
        Assert.Equal(1, scheduler.RecurrentPrefixEntryCount);
        await AssertMatchesColdRun(t3, turn3, Kvf, HintedOpts);
    }

    [Fact]
    public async Task DivergingHistory_WithoutHint_KeepsTodaysBehaviour()
    {
        // No hint: turn 1 shares nothing and is not snapshotted, turn 2 snapshots what it shares with turn 1 (P1), and
        // turn 3 hits that without moving it forward.
        using var model = new ThreadedHistoryModel();
        using var scheduler = NewScheduler(model);
        int[] h2 = [.. P1, .. Run(100, 20), .. Run(150, 20)];
        int[] h3 = [.. h2, .. Run(200, 20), .. Run(250, 20)];

        var turn1 = await RunOne(scheduler, [.. P1, .. G], PlainOpts);
        var turn2 = await RunOne(scheduler, [.. h2, .. G], PlainOpts);
        var turn3 = await RunOne(scheduler, [.. h3, .. G], PlainOpts);

        Assert.Equal(0, turn1.Timings.CachedTokenCount);
        Assert.Equal(0, turn2.Timings.CachedTokenCount);
        Assert.Equal(P1.Length, turn3.Timings.CachedTokenCount);
        Assert.Equal(1, scheduler.RecurrentPrefixEntryCount);
        Assert.Single(model.Snapshots);
    }

    // ── refused growth (issue #1093) ──

    [Fact]
    public async Task UndersizedKv_GrowsExactlyToTheRequest_AndKeepsTheHit()
    {
        // The scheduler sizes each sequence's KV at exactly prompt + max_tokens, so there is no headroom to refuse. A KV that
        // arrives smaller than the matching snapshot must grow to exactly that size before the hit is given up; it used to be
        // skipped, a miss.
        using var model = new ThreadedHistoryModel();
        int cap = int.MaxValue, limit = int.MaxValue;
        Func<ModelConfig, int, IKvCache> kvf = (cfg, size) =>
            new LimitedGrowthKv(new SimpleKvCache(KvGeometry.FromConfig(cfg), Math.Min(size, cap)), limit);
        using var scheduler = new ContinuousBatchScheduler(model, new CsvTokenizer(), kvf,
            new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = 4 });
        // Long enough that the pooled KV of turn 1 (128 positions) does not fit, so turn 2's comes from the factory.
        int[] h2 = [.. P1, .. Run(100, 100), .. Run(300, 100)];
        await RunOne(scheduler, [.. P1, .. G], HintedOpts);           // snapshot just before G: P1
        cap = 10;
        limit = h2.Length + G.Length + HintedOpts.MaxTokens;
        model.Calls.Clear();

        var turn2 = await RunOne(scheduler, [.. h2, .. G], HintedOpts);

        Assert.Equal(P1.Length, turn2.Timings.CachedTokenCount);
        Assert.DoesNotContain(model.Calls, c => c.Start < P1.Length);  // nothing before the snapshot ran again
        await AssertMatchesColdRun([.. h2, .. G], turn2, Kvf, HintedOpts);
    }

    [Fact]
    public async Task RefusedExactGrowth_OnALongPrompt_ThrowsDeviceMemoryAdmission()
    {
        // When even the exact size is refused, a long prompt fails with the context error instead of re-prefilling.
        using var model = new ThreadedHistoryModel();
        int cap = int.MaxValue, limit = int.MaxValue;
        Func<ModelConfig, int, IKvCache> kvf = (cfg, size) =>
            new LimitedGrowthKv(new SimpleKvCache(KvGeometry.FromConfig(cfg), Math.Min(size, cap)), limit);
        using var scheduler = new ContinuousBatchScheduler(model, new CsvTokenizer(), kvf,
            new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = 4 });
        int[] head = Run(2, 4100);
        int[] turn2 = [.. head, .. Run(5000, 200), .. G];             // past turn 1's pooled KV (4224), so from the factory
        await RunOne(scheduler, [.. head, .. G], HintedOpts);         // snapshot at 4100
        cap = 10;
        limit = turn2.Length + HintedOpts.MaxTokens - 1;
        model.Calls.Clear();

        await Assert.ThrowsAsync<DeviceMemoryAdmissionException>(() => RunOne(scheduler, turn2, HintedOpts));

        Assert.Empty(model.Calls);
    }

    private static readonly Func<ModelConfig, int, IKvCache> Kvf = (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg), size);
    private static InferenceOptions PlainOpts => new() { Temperature = 0f, MaxTokens = 4 };
    private static InferenceOptions HintedOpts => PlainOpts with { GenerationPromptSuffix = GenerationSuffix };

    private static ContinuousBatchScheduler NewScheduler(ThreadedHistoryModel model)
        => new(model, new CsvTokenizer(), Kvf, new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = 4 });

    private static async Task AssertMatchesColdRun(int[] prompt, InferenceResponse got, Func<ModelConfig, int, IKvCache> kvf, InferenceOptions opts)
    {
        using var coldModel = new ThreadedHistoryModel();
        using var cold = new ContinuousBatchScheduler(coldModel, new CsvTokenizer(), kvf);
        Assert.Equal((await RunOne(cold, prompt, opts).ConfigureAwait(false)).GeneratedTokenIds, got.GeneratedTokenIds);
    }

    private static async Task<InferenceResponse> RunOne(ContinuousBatchScheduler scheduler, int[] prompt, InferenceOptions opts)
    {
        var handle = scheduler.Submit(new InferenceRequest { TokenIds = prompt, Options = opts });
        for (int i = 0; i < 500 && !scheduler.IsIdle; i++) scheduler.Step();
        Assert.True(scheduler.IsIdle);
        return await handle.Completion.ConfigureAwait(false);
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

    /// <summary>A per-sequence recurrent state: S = (S * 31 + token + 1) mod 9973 after every token.</summary>
    private sealed class HashState : IGdnState
    {
        public int Value { get; set; }
        public int NumGdnLayers => 1;
        public void Reset() => Value = 0;
        public void Dispose() { }
    }

    private sealed class PrefixSnapshot(int state, int length) : IDisposable
    {
        public int State => state;
        public int Length => length;
        public bool IsDisposed { get; private set; }
        public void Dispose() => IsDisposed = true;
    }

    /// <summary>
    /// The scheduler-side counterpart of <c>TextGeneratorRecurrentPrefixCacheTests.HistoryModel</c>: the state is threaded
    /// per sequence through <see cref="ForwardBatch"/>, and logits are one-hot on S mod (Vocab - 1) (never EOS).
    /// </summary>
    private sealed class ThreadedHistoryModel : IModel
    {
        public ModelConfig Config => new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1,
            HiddenSize = HiddenSize, IntermediateSize = HiddenSize * 4, HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public bool RequiresPerSequenceState => true;
        public bool SupportsThreadedSequenceState => true;
        public IRecurrentSequenceState? CreateSequenceState() => new HashState();
        public bool SupportsSequencePrefixSnapshot => true;

        public IDisposable? SnapshotSequencePrefix(IKvCache kvCache, IRecurrentSequenceState? state, int prefixLen)
        {
            Assert.Equal(prefixLen, kvCache.CurrentLength);
            var snapshot = new PrefixSnapshot(((HashState)state!).Value, prefixLen);
            Snapshots.Add(snapshot);
            return snapshot;
        }

        /// <summary>Every forward's first position and token count, in order.</summary>
        public List<(int Start, int Length)> Calls { get; } = [];

        /// <summary>Every snapshot taken, in order.</summary>
        public List<PrefixSnapshot> Snapshots { get; } = [];

        public void RestoreSequencePrefix(IDisposable snapshot, IKvCache kvCache, IRecurrentSequenceState? state)
        {
            var snap = (PrefixSnapshot)snapshot;
            Assert.False(snap.IsDisposed);   // a released entry must never be restored
            kvCache.Rollback(0);
            WriteKv(kvCache, [.. Enumerable.Range(0, snap.Length)]);
            ((HashState)state!).Value = snap.State;
        }

        public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
        {
            var results = new ITensor[requests.Count];
            for (int i = 0; i < requests.Count; i++)
                results[i] = Core(requests[i], deviceId);
            return results;
        }

        private static ITensor Fail() => throw new NotSupportedException("the scheduler must thread the per-sequence state");
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Fail();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Fail();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Fail();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Fail();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Fail();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Fail();

        private unsafe ITensor Core(SequenceForwardRequest request, int deviceId)
        {
            var state = (HashState)request.GdnState!;
            var tokens = request.TokenIds.Span;
            var positions = request.Positions.Span;
            Assert.Equal(positions[0], request.KvCache.CurrentLength);   // a stale or un-rolled-back KV
            Calls.Add((positions[0], tokens.Length));
            int seqLen = tokens.Length;

            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)seqLen * Vocab * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int r = 0; r < seqLen; r++)
            {
                state.Value = (state.Value * 31 + tokens[r] + 1) % 9973;
                var row = new Span<float>(dst + (long)r * Vocab, Vocab);
                row.Fill(-10f);
                row[state.Value % (Vocab - 1)] = 10f;
            }
            WriteKv(request.KvCache, positions);
            return new UnmanagedTensor(new TensorShape(seqLen, Vocab), DType.Float32, deviceId, ptr);
        }

        private static unsafe void WriteKv(IKvCache kvCache, ReadOnlySpan<int> positions)
        {
            if (positions.Length == 0) return;
            nuint bytes = (nuint)(positions.Length * HeadDim * sizeof(float));
            nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
            nint vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
            NativeMemory.Clear((void*)kPtr, bytes);
            NativeMemory.Clear((void*)vPtr, bytes);
            kvCache.Update(new TensorRef(positions.Length, HeadDim, DType.Float32, -1, kPtr),
                           new TensorRef(positions.Length, HeadDim, DType.Float32, -1, vPtr), positions, 0);
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }
}
