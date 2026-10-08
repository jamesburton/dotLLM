using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Engine-citizen contracts of the CPU Qwen4-Exp model (#817): checkpoint/rollback, per-row snapshots, per-sequence state,
/// <c>ForwardBatch</c>, the prefix snapshot and the scheduler / text generator, on the synthetic GGUF
/// (<see cref="SyntheticQwen4ExpGguf"/>: GDN, GDN+PLE, GDN, QSA; pool block 4; PLE conv history 9 rows; the indexer budget of
/// 8 tokens makes QSA prune beyond 11 tokens).
/// </summary>
/// <remarks>
/// "Bit-identical" is only meaningful between runs that issue the same sequence of forward shapes (a different token count
/// changes the GEMM partitioning and so the float summation order), so every exact comparison below keeps the shapes equal and
/// the row-snapshot comparisons (which compare against a differently-shaped fresh run) use a tolerance and a control arm that
/// must fail it.
/// </remarks>
public sealed class Qwen4ExpStateTests : IDisposable
{
    private const int EosToken = SyntheticQwen4ExpGguf.PleEosTokenId;   // 2
    private const int Vocab = SyntheticQwen4ExpGguf.VocabSize;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-state-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _disposables = [];
    private readonly Qwen4ExpTransformerModel _model;
    private readonly ModelConfig _config;

    public Qwen4ExpStateTests()
    {
        Directory.CreateDirectory(_dir);
        string path = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "syn.gguf"));
        var (m, g, c) = ModelLoader.LoadFromGguf(path);
        _disposables.Add(g); _disposables.Add(m);
        _model = (Qwen4ExpTransformerModel)m;
        _config = c;
    }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // ── helpers ──

    private static unsafe float[] Rows(ITensor t)
    {
        using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static int[] Pos(int n, int start = 0) => Enumerable.Range(start, n).ToArray();

    /// <summary>Deterministic token stream over ids 0..15 with EOS (2) sprinkled in: every 5th token.</summary>
    private static int[] Tokens(int n, int seed, bool withEos = true)
    {
        var rng = new Random(seed);
        var t = new int[n];
        for (int i = 0; i < n; i++)
        {
            int v = rng.Next(4, Vocab);
            t[i] = withEos && i % 5 == 3 ? EosToken : v;
        }
        return t;
    }

    private IKvCache NewKv(int len = 64) => new SimpleKvCache(KvGeometry.FromConfig(_config), len);

    private static float MaxRel(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]) / (1f + MathF.Abs(expected[i])));
        return worst;
    }

    private float[] Fwd(int[] ids, int start, IKvCache? kv)
        => Rows(_model.Forward(ids, Pos(ids.Length, start), -1, kv));

    // ── checkpoint / rollback ──

    public static IEnumerable<object[]> RollbackCases()
    {
        foreach (bool engine in new[] { false, true })
        foreach (int prefix in new[] { 5, 8, 12, 13 })            // tail 1, 0 (block boundary), 0, 1; 12/13 are past the 11-token QSA budget
        foreach (int replay in new[] { 3, 11 })                    // < and > the 9-row PLE conv history
        foreach (bool eosAtBoundary in new[] { false, true })      // the EOS window reset straddles the checkpoint
            yield return new object[] { engine, prefix, replay, eosAtBoundary };
    }

    [Theory]
    [MemberData(nameof(RollbackCases))]
    public void CheckpointRestore_ReplayEqualsAFreshRun_BitIdentical(bool engineKv, int prefixLen, int replayLen, bool eosAtBoundary)
    {
        var prefix = Tokens(prefixLen, seed: 1);
        if (eosAtBoundary) prefix[^1] = EosToken;
        var right = Tokens(replayLen, seed: 2);
        right[Math.Min(1, replayLen - 1)] = EosToken;              // an EOS inside the replayed segment
        var wrong = Tokens(replayLen, seed: 3, withEos: false);
        var tail = Tokens(3, seed: 4);

        // Fresh arm.
        _model.ResetSequenceState();
        using var kv1 = engineKv ? NewKv() : null;
        Fwd(prefix, 0, kv1);
        var freshRight = Fwd(right, prefixLen, kv1);
        var freshTail = Fwd(tail, prefixLen + replayLen, kv1);

        // Rollback arm: same forward shapes, with a wrong-token detour that is checkpointed away.
        _model.ResetSequenceState();
        using var kv2 = engineKv ? NewKv() : null;
        Fwd(prefix, 0, kv2);
        var cp = _model.CheckpointRecurrentState();
        Assert.NotNull(cp);
        var wrongLogits = Fwd(wrong, prefixLen, kv2);
        kv2?.Rollback(prefixLen);
        _model.RestoreRecurrentState(cp);
        (cp as IDisposable)?.Dispose();
        var replayRight = Fwd(right, prefixLen, kv2);
        var replayTail = Fwd(tail, prefixLen + replayLen, kv2);

        Assert.NotEqual(freshRight, wrongLogits);                   // the detour is observable, so a no-op restore would be caught
        Assert.Equal(freshRight, replayRight);
        Assert.Equal(freshTail, replayTail);
    }

    [Fact]
    public void CheckpointRestore_IsRepeatable_AndSurvivesStateThatMovedToAnotherHistory()
    {
        // The text generator restores a checkpoint after the live state ran an UNRELATED request: the checkpoint must be a full
        // copy (incl. the pooled indexer keys), not a delta over the live state.
        var promptA = Tokens(14, seed: 11);
        var promptB = Tokens(20, seed: 12, withEos: false);
        var next = Tokens(4, seed: 13);

        _model.ResetSequenceState();
        using var kvA = NewKv();
        Fwd(promptA, 0, kvA);
        var cp = _model.CheckpointRecurrentState();
        var want = Fwd(next, promptA.Length, kvA);

        _model.ResetSequenceState();                                // unrelated request B overwrites the model-owned state
        using var kvB = NewKv();
        Fwd(promptB, 0, kvB);

        kvA.Rollback(promptA.Length);
        _model.RestoreRecurrentState(cp);
        Assert.Equal(want, Fwd(next, promptA.Length, kvA));

        kvA.Rollback(promptA.Length);                               // restoring twice from one checkpoint
        _model.RestoreRecurrentState(cp);
        Assert.Equal(want, Fwd(next, promptA.Length, kvA));
        (cp as IDisposable)?.Dispose();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void Checkpoints_AtEveryChunkBoundary_ReplayTheNextChunkBitIdentically(int chunk)
    {
        var ids = Tokens(22, seed: 21);
        _model.ResetSequenceState();
        using var kv = NewKv();
        for (int i = 0; i < ids.Length; i += chunk)
        {
            int n = Math.Min(chunk, ids.Length - i);
            var cp = _model.CheckpointRecurrentState();
            var first = Fwd(ids.AsSpan(i, n).ToArray(), i, kv);
            kv.Rollback(i);
            _model.RestoreRecurrentState(cp);
            (cp as IDisposable)?.Dispose();
            var again = Fwd(ids.AsSpan(i, n).ToArray(), i, kv);
            Assert.Equal(first, again);
        }
    }

    // ── per-row recurrent snapshots ──

    [Theory]
    [InlineData(false, 5, 4)]
    [InlineData(false, 8, 7)]
    [InlineData(true, 6, 12)]      // verify chunk longer than the PLE conv history, crossing pool blocks
    [InlineData(true, 11, 7)]      // crosses the QSA sparse boundary
    public void RowSnapshots_RestoreToEveryRow_ContinuesLikeAFreshRun(bool engineKv, int prefixLen, int verifyLen)
    {
        var prefix = Tokens(prefixLen, seed: 31);
        var verify = Tokens(verifyLen, seed: 32);
        var next = Tokens(3, seed: 33, withEos: false);
        Assert.True(_model.SupportsRecurrentRowSnapshots);

        for (int row = 0; row < verifyLen - 1; row++)
        {
            // Reference: the committed prefix of the verify chunk is forwarded on its own (a different shape: ULP-level drift).
            _model.ResetSequenceState();
            using var kvR = engineKv ? NewKv() : null;
            Fwd(prefix, 0, kvR);
            Fwd(verify.AsSpan(0, row + 1).ToArray(), prefixLen, kvR);
            var want = Fwd(next, prefixLen + row + 1, kvR);

            _model.ResetSequenceState();
            using var kv = engineKv ? NewKv() : null;
            Fwd(prefix, 0, kv);
            var plain = ((IModel)_model).CheckpointRecurrentState();
            var snapLogits = Rows(_model.ForwardWithRecurrentSnapshots(verify, Pos(verifyLen, prefixLen), -1, kv, null));
            kv?.Rollback(prefixLen + row + 1);
            _model.RestoreRecurrentStateToRow(row);
            var got = Fwd(next, prefixLen + row + 1, kv);

            // The logits of the snapshot forward equal a plain forward of the same shape exactly.
            _model.RestoreRecurrentState(plain);
            (plain as IDisposable)?.Dispose();
            kv?.Rollback(prefixLen);
            Assert.Equal(snapLogits, Fwd(verify, prefixLen, kv));

            float rel = MaxRel(want, got);
            Assert.True(rel < 2e-5f, $"row {row}: continuation after RestoreRecurrentStateToRow drifts by {rel:E3}");
        }
    }

    [Fact]
    public void RowSnapshots_ControlArm_NotRestoringIsFarOutsideTheTolerance()
    {
        // Sensitivity: continuing from the live (fully advanced) state instead of restoring row 3 must miss the 2e-5 tolerance by a
        // wide margin, so the tolerance used above can fail.
        var prefix = Tokens(8, seed: 41);
        var verify = Tokens(7, seed: 42);
        var next = Tokens(3, seed: 43, withEos: false);

        _model.ResetSequenceState();
        Fwd(prefix, 0, null);
        Fwd(verify.AsSpan(0, 4).ToArray(), 8, null);
        var want = Fwd(next, 12, null);

        _model.ResetSequenceState();
        Fwd(prefix, 0, null);
        Rows(_model.ForwardWithRecurrentSnapshots(verify, Pos(7, 8), -1, null, null));
        var liveNotRestored = Fwd(next, 15, null);                    // 8 + 7 tokens consumed, no restore
        Assert.True(MaxRel(want, liveNotRestored) > 1e-4f, $"control rel {MaxRel(want, liveNotRestored):E3}");

        _model.ResetSequenceState();
        Fwd(prefix, 0, null);
        Rows(_model.ForwardWithRecurrentSnapshots(verify, Pos(7, 8), -1, null, null));
        _model.RestoreRecurrentStateToRow(3);
        { float r2 = MaxRel(want, Fwd(next, 12, null)); Assert.True(r2 < 2e-5f, $"restored rel {r2:E3}"); }
    }

    [Fact]
    public void RowSnapshots_LiveRowIsANoOp_AndALaterForwardInvalidatesThem()
    {
        var ids = Tokens(9, seed: 51);
        _model.ResetSequenceState();
        Rows(_model.ForwardWithRecurrentSnapshots(ids, Pos(9), -1, null, null));
        _model.RestoreRecurrentStateToRow(8);                          // the last row: live state, accepted as a no-op
        Rows(_model.ForwardWithRecurrentSnapshots(ids, Pos(9), -1, null, null));   // restarts at position 0
        _model.Forward([5], [9], -1, null).Dispose();                  // any later forward invalidates the snapshots
        Assert.Throws<InvalidOperationException>(() => _model.RestoreRecurrentStateToRow(2));
    }

    // ── per-sequence state, ForwardBatch ──

    private sealed class Seq(int[] tokens, Qwen4ExpTransformerModel model, ModelConfig config) : IDisposable
    {
        public int[] Tokens { get; } = tokens;
        public int Offset;
        public IRecurrentSequenceState State { get; } = model.CreateSequenceState()!;
        public IKvCache Kv { get; } = new SimpleKvCache(KvGeometry.FromConfig(config), 64);
        public void Dispose() { State.Dispose(); Kv.Dispose(); }
    }

    private SequenceForwardRequest Request(Seq s, int n) => new()
    {
        TokenIds = s.Tokens.AsMemory(s.Offset, n),
        Positions = Pos(n, s.Offset).AsMemory(),
        KvCache = s.Kv,
        GdnState = s.State as IGdnState,
    };

    [Fact]
    public void ForwardBatch_InterleavedSequences_EqualRunningThemSeparately()
    {
        int[][] streams = [Tokens(17, seed: 61), Tokens(9, seed: 62, withEos: false), Tokens(23, seed: 63)];
        int[][] schedule =                                   // chunk length per round per sequence (0 = idle that round)
        [
            [5, 3, 8],      // prefill chunks of different sizes
            [4, 0, 6],
            [1, 4, 1],      // decode steps interleaved with prefill
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [5, 0, 7],
        ];

        // Reference: each sequence alone through the same ForwardBatch entry point with the same chunking.
        var want = new List<float[]>[3];
        for (int i = 0; i < 3; i++)
        {
            want[i] = [];
            using var solo = new Seq(streams[i], _model, _config);
            foreach (var round in schedule)
            {
                int n = Math.Min(round[i], streams[i].Length - solo.Offset);
                if (n <= 0) continue;
                var res = ((IModel)_model).ForwardBatch([Request(solo, n)], -1);
                want[i].Add(Rows(res[0]));
                solo.Offset += n;
            }
        }

        // Interleaved: every round is ONE ForwardBatch carrying every sequence that still has tokens.
        var got = new List<float[]>[3] { [], [], [] };
        var seqs = streams.Select(t => new Seq(t, _model, _config)).ToArray();
        try
        {
            foreach (var round in schedule)
            {
                var active = new List<(int Index, int N)>();
                for (int i = 0; i < 3; i++)
                {
                    int n = Math.Min(round[i], streams[i].Length - seqs[i].Offset);
                    if (n > 0) active.Add((i, n));
                }
                if (active.Count == 0) continue;
                var results = ((IModel)_model).ForwardBatch(active.Select(a => Request(seqs[a.Index], a.N)).ToArray(), -1);
                Assert.Equal(active.Count, results.Count);
                for (int k = 0; k < active.Count; k++)
                {
                    Assert.Equal(1, results[k].Shape[0]);   // last-row logits (the scheduler samples Shape[0] - 1)
                    got[active[k].Index].Add(Rows(results[k]));
                    seqs[active[k].Index].Offset += active[k].N;
                }
            }
        }
        finally { foreach (var s in seqs) s.Dispose(); }

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(want[i].Count, got[i].Count);
            for (int k = 0; k < want[i].Count; k++)
                Assert.Equal(want[i][k], got[i][k]);
        }
        // The sequences really are different (an aliasing bug that returned one sequence's logits for all would pass the loop above).
        Assert.NotEqual(got[0][^1], got[2][^1]);
    }

    [Fact]
    public void ForwardBatch_RefusesSharedDefaultStateAndForeignStates()
    {
        using var kv = NewKv();
        var ids = Tokens(3, 71);
        var noState = new SequenceForwardRequest { TokenIds = ids, Positions = Pos(3), KvCache = kv };
        Assert.Throws<ArgumentException>(() => ((IModel)_model).ForwardBatch([noState, noState], -1));
        var foreign = new SequenceForwardRequest
        {
            TokenIds = ids, Positions = Pos(3), KvCache = kv,
            GdnState = new GdnStateCache(_config.GdnConfig!.Value, 1),
        };
        Assert.Throws<ArgumentException>(() => ((IModel)_model).ForwardBatch([foreign], -1));
    }

    [Fact]
    public void ForwardBatch_SingleRequestWithoutState_UsesTheModelOwnedState()
    {
        var ids = Tokens(6, 72);
        _model.ResetSequenceState();
        using var kv = NewKv();
        var viaBatch = Rows(((IModel)_model).ForwardBatch(
            [new SequenceForwardRequest { TokenIds = ids, Positions = Pos(6), KvCache = kv }], -1)[0]);
        _model.ResetSequenceState();
        using var kv2 = NewKv();
        var direct = Rows(_model.Forward(ids, Pos(6), -1, kv2, true));
        Assert.Equal(direct, viaBatch);
    }

    [Fact]
    public void EngineKvCache_AndOwnStore_GiveIdenticalLogits()
    {
        var ids = Tokens(30, seed: 81);                       // well past the 11-token sparse boundary
        _model.ResetSequenceState();
        var own = Fwd(ids, 0, null);
        _model.ResetSequenceState();
        using var kv = NewKv();
        var engine = Fwd(ids, 0, kv);
        Assert.Equal(own, engine);
        Assert.Equal(ids.Length, kv.CurrentLength);
    }

    [Fact]
    public void KvCache_ThatLagsTheState_IsRefused()
    {
        var ids = Tokens(8, 82);
        _model.ResetSequenceState();
        using var kv = NewKv();
        Fwd(ids, 0, kv);
        kv.Rollback(4);                                       // cache rolled back, state not: they must advance together
        Assert.Throws<InvalidOperationException>(() => Fwd(Tokens(2, 83), 8, kv));
    }

    // ── prefix snapshot ──

    [Fact]
    public void PrefixSnapshot_RestoredIntoAFreshSequence_ContinuesBitIdentically()
    {
        Assert.True(_model.SupportsSequencePrefixSnapshot);
        var prefix = Tokens(13, seed: 91);
        var suffix = Tokens(6, seed: 92);

        using var a = new Seq(prefix, _model, _config);
        ((IModel)_model).ForwardBatch([Request(a, 13)], -1)[0].Dispose();
        a.Offset = 13;
        using var snap = _model.SnapshotSequencePrefix(a.Kv, a.State, 13)!;

        // Continue the original (mutates its state: the snapshot must be independent of it).
        var want = Rows(_model.Forward(suffix, Pos(6, 13), -1, (Qwen4ExpSequenceState)a.State, a.Kv, false));

        for (int round = 0; round < 2; round++)
        {
            using var b = new Seq(prefix, _model, _config);
            _model.RestoreSequencePrefix(snap, b.Kv, b.State);
            Assert.Equal(13, b.Kv.CurrentLength);
            Assert.Equal(13, ((Qwen4ExpSequenceState)b.State).Length);
            var got = Rows(_model.Forward(suffix, Pos(6, 13), -1, (Qwen4ExpSequenceState)b.State, b.Kv, false));
            Assert.Equal(want, got);
        }
    }

    // ── accounting ──

    [Fact]
    public void StateAccounting_EstimateMatchesTheAllocatedState()
    {
        var est0 = _model.EstimateSequenceStateBytes(0);
        using var state = _model.CreateState();
        var res = state.ResidentBytes;
        Assert.Equal(res.Gdn, est0.Gdn);
        Assert.Equal(res.Ple, est0.Ple);
        Assert.Equal(res.IndexerTail, est0.IndexerTail);
        Assert.Equal(0, est0.Kv);
        Assert.True(est0.Gdn > 0 && est0.Ple > 0 && est0.IndexerTail > 0);

        // Context-proportional parts: K/V = numQsa * ctx * stride * 2 (K and V) * 4 B; pooled = numQsa * (ctx / R) * d * 4 B.
        var q4 = _config.Qwen4Exp!;
        int numQsa = 1, stride = SyntheticQwen4ExpGguf.NumKvHeads * SyntheticQwen4ExpGguf.HeadDim;
        var est = _model.EstimateSequenceStateBytes(40);
        Assert.Equal((long)numQsa * 40 * stride * 2 * 4, est.Kv);
        Assert.Equal((long)numQsa * (40 / q4.IndexerBlockSize) * q4.IndexerKeyLength * 4, est.IndexerPooled);
        Assert.Equal(est.Total - est.Kv, _model.CheckpointBytes(40));

        // After real work the resident own-store K/V and pooled keys cover the logical estimate.
        var ids = Tokens(40, 95);
        _model.Forward(ids, Pos(40), -1, state).Dispose();
        var after = state.ResidentBytes;
        Assert.True(after.Kv >= est.Kv, $"own K/V {after.Kv} < estimate {est.Kv}");
        Assert.True(after.IndexerPooled >= est.IndexerPooled);
        Assert.Equal(est0.Gdn, after.Gdn);

        // With an engine KV cache the state owns no K/V rows at all.
        using var state2 = _model.CreateState();
        using var kv = NewKv();
        _model.Forward(ids, Pos(40), -1, state2, kv, false).Dispose();
        Assert.Equal(0, state2.ResidentBytes.Kv);
    }

    // ── scheduler + text generator end to end ──

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

    private static string Csv(int[] ids) => string.Join(",", ids);

    [Fact]
    public void TextGenerator_GreedyGeneration_MatchesAManualDecodeLoop_AndTheRecurrentPrefixCache()
    {
        var prompt = Tokens(14, seed: 101, withEos: false);
        var opts = new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f, MaxTokens = 8 };
        var tok = new CsvTokenizer();
        Func<ModelConfig, int, IKvCache> kvf = (c, n) => new SimpleKvCache(KvGeometry.FromConfig(c), n);

        // Manual greedy decode on the model-owned state.
        var manual = new List<int>();
        _model.ResetSequenceState();
        using (var kv = NewKv())
        {
            var logits = Rows(_model.Forward(prompt, Pos(prompt.Length), -1, kv, true));
            for (int step = 0; step < opts.MaxTokens; step++)
            {
                int next = Array.IndexOf(logits, logits.Max());
                if (next == tok.EosTokenId) break;
                manual.Add(next);
                logits = Rows(_model.Forward([next], [prompt.Length + step], -1, kv, true));
            }
        }

        var plain = new TextGenerator(_model, tok, kvf);
        Assert.Equal(manual, plain.Generate(Csv(prompt), opts).GeneratedTokenIds.ToArray());

        // Recurrent prefix cache: a shared prefix is snapshotted via CheckpointRecurrentState and later restored.
        var cached = new TextGenerator(_model, tok, kvf, recurrentPrefixCache: true);
        int[][] suffixes = [[7, 9], [8, 12, 13], [6], [7, 9], [10, 11]];
        var sharedPrefix = Tokens(24, seed: 102, withEos: false);
        for (int i = 0; i < suffixes.Length; i++)
        {
            string p = Csv(sharedPrefix.Concat(suffixes[i]).ToArray());
            var want = plain.Generate(p, opts).GeneratedTokenIds.ToArray();
            var got = cached.Generate(p, opts);
            Assert.Equal(want, got.GeneratedTokenIds.ToArray());
            if (i >= 2) Assert.True(got.Timings.CachedTokenCount > 0, "the shared prefix was never reused");
        }
    }

    [Fact]
    public async Task Scheduler_ConcurrentSequences_MatchTheTextGenerator_AndFreeTheirState()
    {
        var tok = new CsvTokenizer();
        var opts = new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f, MaxTokens = 7 };
        Func<ModelConfig, int, IKvCache> kvf = (c, n) => new SimpleKvCache(KvGeometry.FromConfig(c), n);
        int[][] prompts = [Tokens(12, 111, false), Tokens(5, 112, false), Tokens(19, 113, false)];

        var gen = new TextGenerator(_model, tok, kvf);
        var want = prompts.Select(p => gen.Generate(Csv(p), opts).GeneratedTokenIds.ToArray()).ToArray();

        using var scheduler = new ContinuousBatchScheduler(_model, tok, kvf,
            new ContinuousBatchSchedulerOptions { MaxActiveSequences = 4 });
        var handles = prompts.Select(p => scheduler.Submit(new InferenceRequest { TokenIds = p, Options = opts })).ToArray();
        for (int i = 0; i < 500 && !scheduler.IsIdle; i++) scheduler.Step();
        Assert.True(scheduler.IsIdle);

        for (int i = 0; i < prompts.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(want[i], r.GeneratedTokenIds.ToArray());
        }
    }
}
