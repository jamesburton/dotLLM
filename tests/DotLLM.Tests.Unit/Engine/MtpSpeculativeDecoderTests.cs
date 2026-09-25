using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Tests for <see cref="MtpSpeculativeDecoder"/> (issue #253). Uses a synthetic MTP-capable mock
/// model whose target and MTP-head predictions are each a plain deterministic function of the
/// input token — this isolates the DECODER's draft-verify-accept mechanics (the thing
/// this class is responsible for) from the real MTP head's forward math (covered separately by
/// <c>Qwen3HybridDenseMtpTests</c> against a real, if synthetic, GGUF-loaded model).
/// </summary>
/// <remarks>
/// The central claim under test: <b>MTP self-speculative decoding produces the exact same output
/// token sequence as plain greedy decode of the target model alone</b> — regardless of whether the
/// MTP head's own guesses agree with the target (see <see cref="DraftAndVerify_AllAccepted_MatchesPlainGreedyDecode"/>)
/// or disagree at specific points, forcing rejections (see
/// <see cref="DraftAndVerify_WithDisagreements_StillMatchesPlainGreedyDecode"/>). This is the same
/// correctness property <c>SpeculativeDecoderTests</c> demonstrates for the two-model decoder,
/// carried over to the self-speculative case.
/// </remarks>
public sealed class MtpSpeculativeDecoderTests
{
    private const int VocabSize = 16;
    private const int MaxSeqLen = 256;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;
    private const int HiddenSize = 8;
    private const int MtpNumKvHeads = 1;
    private const int MtpHeadDim = 4;

    [Fact]
    public void Constructor_Throws_WhenNonGreedy()
    {
        var ex = Assert.Throws<NotSupportedException>(() => new MtpSpeculativeDecoder(greedy: false));
        Assert.Contains("greedy", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DraftAndVerify_ModelWithoutMtp_Throws()
    {
        // targetFn == mtpFn is irrelevant here — SupportsMtp is what's being asserted.
        using var model = new MockMtpModel(t => (t + 1) % VocabSize, t => (t + 1) % VocabSize, supportsMtp: false);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };
        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);

        Assert.Throws<ArgumentException>(() =>
        {
            using var mtpState = new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, 32);
            Span<int> outputBuffer = stackalloc int[4];
            decoder.DraftAndVerify(model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position: 1, vocabSize: VocabSize, numCandidates: 3, outputBuffer);
        });
    }

    /// <summary>
    /// When the MTP head's own guesses always agree with the target's argmax, every round should
    /// accept all K draft tokens plus a bonus token, and the resulting sequence must be
    /// byte-identical to running the target's successor function directly (plain greedy decode).
    /// </summary>
    [Fact]
    public void DraftAndVerify_AllAccepted_MatchesPlainGreedyDecode()
    {
        int TargetFn(int t) => (t + 1) % VocabSize;

        using var model = new MockMtpModel(TargetFn, TargetFn, supportsMtp: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        int startToken = 1;
        int totalNewTokens = 12;
        const int k = 3;

        List<int> speculative = RunSpeculative(model, decoder, pipeline, startToken, totalNewTokens, k);
        List<int> plain = RunPlainGreedy(startToken, totalNewTokens, TargetFn);

        Assert.Equal(plain, speculative);
    }

    /// <summary>
    /// Issue #469: a round verifies <c>[lastToken, d1..dK]</c> in ONE trunk forward. The old
    /// decoder also forwarded <c>lastToken</c> alone first, so it paid two forwards per round and
    /// could never beat plain decode.
    /// </summary>
    [Fact]
    public void DraftAndVerify_AllAccepted_CostsOneTrunkForwardPerRound()
    {
        int TargetFn(int t) => (t + 1) % VocabSize;

        using var model = new MockMtpModel(TargetFn, TargetFn, supportsMtp: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        RunSpeculative(model, decoder, pipeline, startToken: 1, totalNewTokens: 12, k: 3, out int rounds);

        Assert.Equal(3, rounds);                 // 12 tokens / (3 drafts + 1 bonus)
        Assert.Equal(rounds, model.TrunkForwardCount);
    }

    /// <summary>
    /// The MTP head disagrees with the target at specific tokens (forcing rejections every other
    /// round), and the target ALSO differs from a naive "always successor" rule at one special
    /// token (7 → 0 instead of 7 → 8) so a plain-decode oracle and a speculative-decode run can be
    /// compared against the SAME non-trivial target function. Despite MTP's wrong guesses, the
    /// final accepted sequence must still exactly match plain greedy decode of the target alone —
    /// MTP never gets to inject a token the target didn't independently agree with.
    /// </summary>
    [Fact]
    public void DraftAndVerify_WithDisagreements_StillMatchesPlainGreedyDecode()
    {
        int TargetFn(int t) => t == 7 ? 0 : (t + 1) % VocabSize;
        // MTP head is deliberately wrong for even tokens (guesses t+2 instead of the target's t+1),
        // right for odd tokens — guarantees a mix of rejections and acceptances across rounds.
        int MtpFn(int t) => (t % 2 == 0) ? (t + 2) % VocabSize : TargetFn(t);

        using var model = new MockMtpModel(TargetFn, MtpFn, supportsMtp: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        int startToken = 2;
        int totalNewTokens = 20;
        const int k = 4;

        List<int> speculative = RunSpeculative(model, decoder, pipeline, startToken, totalNewTokens, k);
        List<int> plain = RunPlainGreedy(startToken, totalNewTokens, TargetFn);

        Assert.Equal(plain, speculative);
    }

    [Fact]
    public void DraftAndVerify_ZeroCandidates_ReturnsEmptyResult()
    {
        int TargetFn(int t) => (t + 1) % VocabSize;
        using var model = new MockMtpModel(TargetFn, TargetFn, supportsMtp: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };

        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, 32);
        Span<int> outputBuffer = stackalloc int[1];

        var result = decoder.DraftAndVerify(model, kvCache, mtpState, pipeline, generatedIds,
            constraint: null, position: 1, vocabSize: VocabSize, numCandidates: 0, outputBuffer);

        Assert.Equal(0, result.AcceptedCount);
    }

    // ── Issue #486: device argmax for an unconstrained greedy draft ─────────

    private static int Successor(int t) => (t + 1) % VocabSize;

    /// <summary>
    /// Every draft row carries a TIE for its maximum between the right token (the trunk's successor)
    /// and a wrong one. The wrong one sits at a HIGHER index — at the wrap the right token is index 0
    /// and the wrong one the last index — except when the right token is itself the last index, where
    /// the wrong one is lower. Lowest-index-wins therefore accepts every draft but that one, on both
    /// paths; a "highest index wins" argmax would reject nearly every draft and change the rounds.
    /// </summary>
    private static void TieRow(int tokenId, Span<float> row)
    {
        row.Fill(-10f);
        int right = Successor(tokenId);
        int wrong = right == VocabSize - 1 ? right - 1
                  : right == 0 ? VocabSize - 1
                  : Math.Min(right + 3, VocabSize - 1);
        row[right] = 10f;
        row[wrong] = 10f;
    }

    [Fact]
    public void DraftArgMax_UnconstrainedGreedy_TakesArgMaxPath_AndMatchesFullLogitsPath()
    {
        var on = RunArgMaxScenario(useDraftArgMax: true, supportsArgMax: true, out int roundsOn, out var modelOn, out var decOn);
        var off = RunArgMaxScenario(useDraftArgMax: false, supportsArgMax: true, out int roundsOff, out var modelOff, out _);
        using (modelOn)
        using (modelOff)
        {
            Assert.Equal(RunPlainGreedy(1, 20, Successor), on);
            Assert.Equal(off, on);
            Assert.Equal(roundsOff, roundsOn);
            Assert.True(modelOn.ArgMaxDraftCalls > 0, "argmax path not taken");
            Assert.Equal(0, modelOn.FullLogitsDraftCalls);
            Assert.Equal(modelOn.ArgMaxDraftCalls, decOn.DraftArgMaxSteps);
            Assert.Equal(0, modelOff.ArgMaxDraftCalls);
            Assert.True(modelOff.FullLogitsDraftCalls > 0);
            // Lowest-index ties accept every draft except the one whose right token is the last
            // index: 20 tokens in far fewer rounds than a one-token-per-round (all-rejected) run.
            Assert.True(roundsOn <= 8, $"ties did not resolve to the lowest index ({roundsOn} rounds)");
        }
    }

    [Fact]
    public void DraftArgMax_ModelWithoutCapability_UsesFullLogits()
    {
        var tokens = RunArgMaxScenario(useDraftArgMax: true, supportsArgMax: false, out _, out var model, out var dec);
        using (model)
        {
            Assert.Equal(RunPlainGreedy(1, 20, Successor), tokens);
            Assert.Equal(0, model.ArgMaxDraftCalls);
            Assert.Equal(0, dec.DraftArgMaxSteps);
        }
    }

    /// <summary>
    /// A constrained draft must mask before the argmax, so it never takes the argmax path. The
    /// constraint forbids the draft head's favourite token; the full-logits path then drafts the
    /// runner-up, the successor. Taking the unmasked argmax would draft the forbidden token instead.
    /// </summary>
    [Fact]
    public void DraftArgMax_WithConstraint_UsesFullLogits_AndHonoursTheMask()
    {
        const int forbidden = 9;
        void Row(int tokenId, Span<float> row)
        {
            row.Fill(-10f);
            row[forbidden] = 10f;
            row[Successor(tokenId)] = 5f;
        }

        using var model = new MockMtpModel(Successor, Successor, supportsMtp: true, draftRow: Row, supportsArgMax: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseDraftArgMax = true };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };
        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, maxSteps: MaxSeqLen);
        Span<int> outputBuffer = stackalloc int[4];

        var result = decoder.DraftAndVerify(model, kvCache, mtpState, pipeline, generatedIds,
            new ForbidTokenConstraint(forbidden, VocabSize), position: 0, vocabSize: VocabSize,
            numCandidates: 3, outputBuffer);

        Assert.Equal(0, model.ArgMaxDraftCalls);
        Assert.Equal(3, model.FullLogitsDraftCalls);
        Assert.Equal(0, decoder.DraftArgMaxSteps);
        // 1 -> 2 -> 3 -> 4 are drafted through the mask and accepted, then the bonus 5.
        Assert.Equal(4, result.AcceptedCount);
        Assert.Equal(new[] { 2, 3, 4, 5 }, outputBuffer.Slice(0, 4).ToArray());
    }

    /// <summary>
    /// Pins the host argmax the fast path must reproduce: <c>TensorPrimitives.IndexOfMax</c> takes the
    /// lowest index on a tie (at index 0, at the last index, and across SIMD lanes), ranks +0 above
    /// -0, and returns the first NaN. The CUDA kernel is tested against the same function
    /// (<c>CudaArgMaxF32Tests</c>).
    /// </summary>
    [Theory]
    [InlineData(new float[] { 3f, 1f, 3f, 2f }, 0)]
    [InlineData(new float[] { 1f, 3f, 2f, 3f }, 1)]
    [InlineData(new float[] { 2f, 1f, 1f, 2f }, 0)]
    [InlineData(new float[] { -0f, 0f }, 1)]
    [InlineData(new float[] { 0f, -0f }, 0)]
    [InlineData(new float[] { 1f, float.NaN, 5f, float.NaN }, 1)]
    [InlineData(new float[] { float.NegativeInfinity, float.NegativeInfinity }, 0)]
    public void HostArgMax_Contract_IsPinned(float[] values, int expected)
    {
        Assert.Equal(expected, System.Numerics.Tensors.TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)values));
    }

    [Fact]
    public void HostArgMax_TieAcrossVectorLanes_PicksLowestIndex()
    {
        var v = new float[1000];
        for (int i = 0; i < v.Length; i++) v[i] = -(i % 7);
        v[517] = 9f; v[3] = 9f; v[999] = 9f;
        Assert.Equal(3, System.Numerics.Tensors.TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)v));
        v[3] = 1f;
        Assert.Equal(517, System.Numerics.Tensors.TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)v));
    }

    private static List<int> RunArgMaxScenario(bool useDraftArgMax, bool supportsArgMax, out int rounds,
                                               out MockMtpModel model, out MtpSpeculativeDecoder decoder)
    {
        model = new MockMtpModel(Successor, Successor, supportsMtp: true, draftRow: TieRow, supportsArgMax: supportsArgMax);
        decoder = new MtpSpeculativeDecoder(greedy: true) { UseDraftArgMax = useDraftArgMax };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        return RunSpeculative(model, decoder, pipeline, startToken: 1, totalNewTokens: 20, k: 3, out rounds);
    }

    /// <summary>Allows every token except one.</summary>
    private sealed class ForbidTokenConstraint(int forbidden, int vocabSize) : DotLLM.Core.Constraints.IDecodingConstraint
    {
        public void Advance(int tokenId) { }

        public DotLLM.Core.Constraints.TokenMask GetAllowedTokens()
        {
            var mask = new DotLLM.Core.Constraints.TokenMask(vocabSize);
            mask.AllowAll();
            mask.Disallow(forbidden);
            return mask;
        }

        public bool IsComplete() => false;
        public DotLLM.Core.Constraints.IDecodingConstraint Clone() => new ForbidTokenConstraint(forbidden, vocabSize);
        public void Reset() { }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static List<int> RunPlainGreedy(int startToken, int totalNewTokens, Func<int, int> targetFn)
    {
        var seq = new List<int> { startToken };
        for (int i = 0; i < totalNewTokens; i++)
            seq.Add(targetFn(seq[^1]));
        return seq;
    }

    private static List<int> RunSpeculative(
        MockMtpModel model, MtpSpeculativeDecoder decoder, SamplerPipeline pipeline,
        int startToken, int totalNewTokens, int k)
        => RunSpeculative(model, decoder, pipeline, startToken, totalNewTokens, k, out _);

    private static List<int> RunSpeculative(
        MockMtpModel model, MtpSpeculativeDecoder decoder, SamplerPipeline pipeline,
        int startToken, int totalNewTokens, int k, out int rounds)
    {
        var generatedIds = new List<int> { startToken };
        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, maxSteps: MaxSeqLen);

        // startToken plays the part of a prefill's sampled token: it sits at position 0 and has not
        // been forwarded through the trunk yet (issue #469's contract), so there is no prefill.
        int position = 0;
        Span<int> outputBuffer = stackalloc int[k + 1];
        int guard = 0;
        rounds = 0;
        while (generatedIds.Count - 1 < totalNewTokens && guard++ < totalNewTokens * 4)
        {
            var result = decoder.DraftAndVerify(
                model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position, vocabSize: VocabSize, numCandidates: k, outputBuffer);

            Assert.True(result.AcceptedCount > 0, "Every round must accept at least the corrected/bonus token.");
            rounds++;

            for (int i = 0; i < result.AcceptedCount && generatedIds.Count - 1 < totalNewTokens; i++)
                generatedIds.Add(outputBuffer[i]);

            position += result.AcceptedCount;

            // The head's KV-cache holds every committed position (issue #469): it is exactly as
            // long as the next round's lastToken position.
            Assert.Equal(position, mtpState.CurrentLength);
        }

        Assert.Empty(model.PairingViolations);
        return generatedIds.Take(totalNewTokens + 1).ToList();
    }

    /// <summary>
    /// Deterministic MTP-capable mock: row <c>t</c>'s target logits argmax is
    /// <c>targetFn(tokenIds[t])</c>; <see cref="ForwardMtp"/>'s argmax is
    /// <c>mtpFn(tokenId)</c>. Mirrors <c>SpeculativeDecoderTests.MockModel</c>'s KV-cache-update
    /// pattern, extended to be per-row input-dependent (needed to prove a genuine multi-round
    /// equivalence property, not just single-round accept/reject).
    /// </summary>
    private sealed class MockMtpModel : IModel
    {
        private readonly Func<int, int> _targetFn;
        private readonly Func<int, int> _mtpFn;
        private readonly bool _supportsMtp;
        private readonly Action<int, Span<float>>? _draftRow;
        private readonly bool _supportsArgMax;

        /// <param name="targetFn">Trunk argmax for an input token.</param>
        /// <param name="mtpFn">Draft argmax for an input token (ignored when <paramref name="draftRow"/> is set).</param>
        /// <param name="supportsMtp">Whether the model reports an MTP head.</param>
        /// <param name="draftRow">Writes a draft step's whole logits row for the input token (issue #486 tie tests).</param>
        /// <param name="supportsArgMax">
        /// Reports <see cref="IModel.SupportsMtpArgMax"/> and implements <see cref="ForwardMtpArgMax"/> with an
        /// independent first-maximum scan — the device kernel's contract — instead of TensorPrimitives.
        /// </param>
        public MockMtpModel(Func<int, int> targetFn, Func<int, int> mtpFn, bool supportsMtp,
                            Action<int, Span<float>>? draftRow = null, bool supportsArgMax = false)
        {
            _targetFn = targetFn;
            _mtpFn = mtpFn;
            _supportsMtp = supportsMtp;
            _draftRow = draftRow;
            _supportsArgMax = supportsArgMax;
        }

        /// <summary>Draft steps served through <see cref="ForwardMtp"/> (full logits).</summary>
        public int FullLogitsDraftCalls { get; private set; }

        /// <summary>Draft steps served through <see cref="ForwardMtpArgMax"/>.</summary>
        public int ArgMaxDraftCalls { get; private set; }

        public bool SupportsMtpArgMax => _supportsArgMax;

        public unsafe int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
        {
            if (!_supportsArgMax)
                throw new NotSupportedException();
            ArgMaxDraftCalls++;
            using ITensor logits = ForwardMtpCore(state, tokenId, position);
            var row = new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize);
            int best = 0;
            for (int i = 1; i < row.Length; i++)
                if (row[i] > row[best]) best = i;   // strict: the lowest index wins a tie
            return best;
        }

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = 1,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HiddenSize,
            IntermediateSize = HiddenSize * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public void Dispose() { }

        public bool SupportsMtp => _supportsMtp;

        public IMtpState? CreateMtpState() => CreateMtpState(MaxSeqLen);

        public IMtpState? CreateMtpState(int maxSequenceLength) =>
            _supportsMtp ? new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, maxSequenceLength) : null;

        /// <summary>Trunk forwards run so far (any overload).</summary>
        public int TrunkForwardCount { get; private set; }

        /// <summary>MTP steps whose pending hidden was neither a chained draft nor h_{position-1}.</summary>
        public List<string> PairingViolations { get; } = [];

        // Captured trunk row for position p is filled with TrunkRowMarker(p); a chained draft step
        // leaves ChainedMarker. The first element of the pending hidden identifies which it was.
        private static float TrunkRowMarker(int position) => 1000f + position;
        private const float ChainedMarker = -1f;

        private void CheckPairing(CpuMtpState state, int position, string where)
        {
            float pending = state.PendingHidden[0];
            bool ok = pending == ChainedMarker
                      || pending == TrunkRowMarker(position - 1)
                      || (position == 0 && pending == 0f);
            if (!ok)
                PairingViolations.Add($"{where} at position {position}: pending={pending}");
        }

        private static void StepHead(CpuMtpState state, int position)
        {
            if (state.CurrentLength > position) state.Rollback(position);
            Assert.Equal(position, state.CurrentLength);
            state.Advance();
        }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
            => Forward(tokenIds, positions, deviceId, kvCache, adapter: null, mtpState: null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache, DotLLM.Core.Lora.ILoraAdapter? adapter)
            => Forward(tokenIds, positions, deviceId, kvCache, adapter, mtpState: null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache, DotLLM.Core.Lora.ILoraAdapter? adapter, IMtpState? mtpState)
        {
            TrunkForwardCount++;
            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            float* dst = (float*)ptr;
            for (int t = 0; t < batchSize; t++)
            {
                var row = new Span<float>(dst + (long)t * VocabSize, VocabSize);
                row.Fill(-10f);
                row[_targetFn(tokenIds[t])] = 10f;
            }

            var shape = new TensorShape(batchSize, VocabSize);

            if (kvCache != null)
            {
                int kvStride = NumKvHeads * HeadDim;
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                NativeMemory.Clear((void*)kPtr, (nuint)(batchSize * kvStride * sizeof(float)));
                NativeMemory.Clear((void*)vPtr, (nuint)(batchSize * kvStride * sizeof(float)));

                var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                kvCache.Update(kRef, vRef, positions, 0);

                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            if (mtpState is CpuMtpState cap)
            {
                // Rows carry their position so the pairing can be checked; the absorb mirrors the
                // real models': token i pairs with the carry (i == 0) or captured row i - 1.
                float[] rows = new float[batchSize * HiddenSize];
                for (int t = 0; t < batchSize; t++)
                    rows.AsSpan(t * HiddenSize, HiddenSize).Fill(TrunkRowMarker(positions[t]));
                cap.SetCapturedRows(rows, batchSize);
                for (int t = 0; t < batchSize; t++)
                {
                    if (t == 0) cap.SetPendingFromCarry();
                    else cap.SetPendingFromCapturedRow(t - 1);
                    CheckPairing(cap, positions[t], "absorb");
                    StepHead(cap, positions[t]);
                }
                cap.SeedFromCapturedRow(batchSize - 1);
            }

            return new UnmanagedTensor(shape, DType.Float32, deviceId, ptr);
        }

        public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            FullLogitsDraftCalls++;
            return ForwardMtpCore(state, tokenId, position);
        }

        private unsafe ITensor ForwardMtpCore(IMtpState state, int tokenId, int position)
        {
            if (!_supportsMtp)
                throw new NotSupportedException();

            long totalFloats = VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            var row = new Span<float>((float*)ptr, VocabSize);
            if (_draftRow is not null)
            {
                _draftRow(tokenId, row);
            }
            else
            {
                row.Fill(-10f);
                row[_mtpFn(tokenId)] = 10f;
            }

            if (state is CpuMtpState cap)
            {
                CheckPairing(cap, position, "draft");
                StepHead(cap, position);
                cap.PendingHiddenMutable.Fill(ChainedMarker);
            }

            return new UnmanagedTensor(new TensorShape(1, VocabSize), DType.Float32, deviceId: -1, ptr);
        }
    }
}
