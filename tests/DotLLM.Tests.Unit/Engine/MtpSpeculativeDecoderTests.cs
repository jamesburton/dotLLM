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
/// input token — this isolates the DECODER's draft-verify-accept-catchup mechanics (the thing
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
    {
        var generatedIds = new List<int> { startToken };
        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, maxSteps: Math.Max(k, 1) + 4);

        // Prefill: seed the target KV-cache with the start token at position 0. `position` must
        // equal lastToken's OWN KV-cache slot (see DraftAndVerify's remarks: "lastToken already
        // occupies position in kvCacheTarget"), matching SpeculativeDecoder's identical
        // convention -- so it starts at 0 here, not 1. This mock model ignores position entirely
        // for logit computation, so this correction has no effect on this test's assertions (both
        // values pass); fixed for realism after a real (position-sensitive) CUDA model test caught
        // an off-by-one derived from copying this exact pattern -- see issue #253's CUDA follow-up.
        using (ITensor _ = model.Forward([startToken], [0], deviceId: -1, kvCache))
        {
        }

        int position = 0;
        Span<int> outputBuffer = stackalloc int[k + 1];
        int guard = 0;
        while (generatedIds.Count - 1 < totalNewTokens && guard++ < totalNewTokens * 4)
        {
            var result = decoder.DraftAndVerify(
                model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position, vocabSize: VocabSize, numCandidates: k, outputBuffer);

            Assert.True(result.AcceptedCount > 0, "Every round must accept at least the corrected/bonus token.");

            for (int i = 0; i < result.AcceptedCount && generatedIds.Count - 1 < totalNewTokens; i++)
                generatedIds.Add(outputBuffer[i]);

            position += result.AcceptedCount;
        }

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

        public MockMtpModel(Func<int, int> targetFn, Func<int, int> mtpFn, bool supportsMtp)
        {
            _targetFn = targetFn;
            _mtpFn = mtpFn;
            _supportsMtp = supportsMtp;
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

        public IMtpState? CreateMtpState() =>
            _supportsMtp ? new CpuMtpState(HiddenSize, MtpNumKvHeads, MtpHeadDim, maxSteps: 32) : null;

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
                // Content is irrelevant to this mock's ForwardMtp (which is purely a function of
                // tokenId), but a real IMtpState consumer must see a well-formed capture.
                float[] fake = new float[batchSize * HiddenSize];
                for (int i = 0; i < fake.Length; i++) fake[i] = 0.01f * i;
                cap.SetCapturedRows(fake, batchSize);
            }

            return new UnmanagedTensor(shape, DType.Float32, deviceId, ptr);
        }

        public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            if (!_supportsMtp)
                throw new NotSupportedException();

            long totalFloats = VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            var row = new Span<float>((float*)ptr, VocabSize);
            row.Fill(-10f);
            row[_mtpFn(tokenId)] = 10f;

            if (state is CpuMtpState cap)
                cap.Advance();

            return new UnmanagedTensor(new TensorShape(1, VocabSize), DType.Float32, deviceId: -1, ptr);
        }
    }
}
