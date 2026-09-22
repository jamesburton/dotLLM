using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #493: <c>TextGenerator</c>'s prompt prefill reads exactly one logits row — row
/// <c>Shape[0] - 1</c> of the final chunk — so it must pass <c>lastTokenLogitsOnly: true</c> and
/// must never assume one row per input token. On a 248,320-token vocabulary at a 4096-token prompt
/// the rows it never reads are ~4.0 GB of device memory.
/// </summary>
/// <remarks>
/// <para>No CPU model honours the hint (it is a GPU VRAM-headroom escape hatch — see
/// <see cref="IModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, bool)"/>), so a
/// real-fixture prefill would return <c>[len, vocab]</c> whatever the engine asks for and could not
/// discriminate the fixed code from the broken code. These tests therefore use a fake that
/// <b>does</b> honour it, and assert both the flag the engine passed and the row count the engine
/// coped with. The real-fixture end-to-end invariance lives in
/// <c>DotLLM.Tests.Integration.Engine.TextGeneratorLastRowPrefillRealFixtureTests</c>.</para>
/// <para>Mutant check: dropping <c>lastTokenLogitsOnly: true</c> in <c>ForwardPrefill</c> (or
/// routing the call back through an overload without it) fails
/// <see cref="Prefill_RequestsLastRowLogits_AndCopesWithOneRow"/> and
/// <see cref="ChunkedPrefill_EveryChunkRequestsLastRow"/>. Indexing row 0 instead of
/// <c>Shape[0]-1</c> fails <see cref="LastRowOnlyModel_ProducesSameTokens_AsAllRowModel"/>.</para>
/// </remarks>
public sealed class TextGeneratorLastRowPrefillTests
{
    private const int VocabSize = 16;
    private const int PromptLen = 10;
    private const int HiddenSize = 8;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;
    private const int MaxSeqLen = 128;

    [Fact]
    public void Prefill_RequestsLastRowLogits_AndCopesWithOneRow()
    {
        using var model = new LastRowHintModel(honourHint: true);
        var generator = new TextGenerator(model, new StubTokenizer());

        var response = generator.Generate("prompt", new InferenceOptions { MaxTokens = 3, Temperature = 0f });

        ForwardRecord prefill = model.Calls[0];
        Assert.Equal(PromptLen, prefill.Length);
        Assert.True(prefill.LastRowHint, "prefill must opt in to lastTokenLogitsOnly (#493)");
        Assert.Equal(1, prefill.ReturnedRows);
        Assert.Equal(3, response.GeneratedTokenIds.Length);
    }

    [Fact]
    public void ChunkedPrefill_EveryChunkRequestsLastRow()
    {
        using var model = new LastRowHintModel(honourHint: true);
        var generator = new TextGenerator(model, new StubTokenizer(), prefillChunkSize: 4);

        generator.Generate("prompt", new InferenceOptions { MaxTokens = 1, Temperature = 0f });

        // 10 prompt tokens in chunks of 4/4/2 — every chunk asks for the last row only, because
        // only the FINAL chunk's last row is ever read and the earlier chunks are disposed unread.
        Assert.Equal(new[] { 4, 4, 2 }, model.Calls.Take(3).Select(c => c.Length).ToArray());
        Assert.All(model.Calls.Take(3), c => Assert.True(c.LastRowHint));
        Assert.All(model.Calls.Take(3), c => Assert.Equal(1, c.ReturnedRows));
    }

    [Fact]
    public void LastRowOnlyModel_ProducesSameTokens_AsAllRowModel()
    {
        // The same deterministic successor function, once behind a model that honours the hint and
        // once behind one that ignores it. Generation must not be able to tell them apart.
        using var honouring = new LastRowHintModel(honourHint: true);
        using var ignoring = new LastRowHintModel(honourHint: false);

        var a = new TextGenerator(honouring, new StubTokenizer())
            .Generate("prompt", new InferenceOptions { MaxTokens = 6, Temperature = 0f });
        var b = new TextGenerator(ignoring, new StubTokenizer())
            .Generate("prompt", new InferenceOptions { MaxTokens = 6, Temperature = 0f });

        Assert.Equal(b.GeneratedTokenIds, a.GeneratedTokenIds);
        Assert.Equal(1, honouring.Calls[0].ReturnedRows);
        Assert.Equal(PromptLen, ignoring.Calls[0].ReturnedRows);   // the fake really does differ
    }

    [Fact]
    public void MtpPrefill_StillAbsorbsEveryPosition_WithLastRowHint()
    {
        // Issue #469's contract: the MTP head absorbs EVERY prefill position, and its pending
        // hidden for token i is the trunk row of position i-1. #493 must not weaken that — the
        // capture is a side effect on the state, independent of how many LOGITS rows come back.
        using var model = new LastRowHintModel(honourHint: true, supportsMtp: true);
        var generator = new TextGenerator(model, new StubTokenizer(), prefillChunkSize: 4);

        generator.Generate("prompt", new InferenceOptions { MaxTokens = 4, Temperature = 0f });

        ForwardRecord[] prefillCalls = model.Calls.Take(3).ToArray();
        Assert.All(prefillCalls, c => Assert.True(c.LastRowHint));
        Assert.All(prefillCalls, c => Assert.True(c.HadMtpState, "prefill must still thread the MTP state"));
        Assert.All(prefillCalls, c => Assert.Equal(1, c.ReturnedRows));
        // The prefill's own absorb covers positions 0..PromptLen-1 in order (the decode
        // loop's verify forwards keep appending beyond that, which is not what's under test).
        Assert.Equal(Enumerable.Range(0, PromptLen), model.AbsorbedPositions.Take(PromptLen));
        Assert.Empty(model.PairingViolations);
    }

    [Fact]
    public void MtpPrefill_LastRowOnly_ProducesSameTokens_AsAllRows()
    {
        using var honouring = new LastRowHintModel(honourHint: true, supportsMtp: true);
        using var ignoring = new LastRowHintModel(honourHint: false, supportsMtp: true);

        var a = new TextGenerator(honouring, new StubTokenizer())
            .Generate("prompt", new InferenceOptions { MaxTokens = 6, Temperature = 0f });
        var b = new TextGenerator(ignoring, new StubTokenizer())
            .Generate("prompt", new InferenceOptions { MaxTokens = 6, Temperature = 0f });

        Assert.Equal(b.GeneratedTokenIds, a.GeneratedTokenIds);
        Assert.Equal(honouring.AbsorbedPositions, ignoring.AbsorbedPositions);
    }

    // ── Fakes ──

    private sealed record ForwardRecord(int Length, int FirstPosition, bool LastRowHint,
                                        bool HadMtpState, int ReturnedRows);

    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => TextGeneratorLastRowPrefillTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => 15;

        public int[] Encode(string text) => Enumerable.Range(2, PromptLen).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => PromptLen;
    }

    /// <summary>
    /// A model whose logits are a deterministic function of the input token, which optionally
    /// HONOURS <c>lastTokenLogitsOnly</c> by returning a <c>[1, vocab]</c> tensor holding the last
    /// position's row (what the CUDA hybrid models do). Records what the engine asked for.
    /// </summary>
    private sealed class LastRowHintModel(bool honourHint, bool supportsMtp = false) : IModel
    {
        private const float ChainedMarker = -1f;

        public List<ForwardRecord> Calls { get; } = [];

        /// <summary>Positions the MTP head absorbed, in order — one entry per trunk token.</summary>
        public List<int> AbsorbedPositions { get; } = [];

        /// <summary>Absorb steps whose pending hidden was not the previous position's trunk row.</summary>
        public List<string> PairingViolations { get; } = [];

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
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public bool SupportsMtp => supportsMtp;
        public IMtpState? CreateMtpState() => CreateMtpState(MaxSeqLen);
        public IMtpState? CreateMtpState(int maxSequenceLength)
            => supportsMtp ? new CpuMtpState(HiddenSize, NumKvHeads, HeadDim, maxSequenceLength) : null;

        private static int TargetFn(int token) => (token + 1) % VocabSize;
        private static float TrunkRowMarker(int position) => 1000f + position;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Core(tokenIds, positions, deviceId, null, null, false);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                               int deviceId, IKvCache? kvCache)
            => Core(tokenIds, positions, deviceId, kvCache, null, false);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                               int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
            => Core(tokenIds, positions, deviceId, kvCache, null, lastTokenLogitsOnly);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                               int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
            => Core(tokenIds, positions, deviceId, kvCache, null, false);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                               int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
            => Core(tokenIds, positions, deviceId, kvCache, mtpState, false);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                               int deviceId, IKvCache? kvCache, ILoraAdapter? adapter,
                               IMtpState? mtpState, bool lastTokenLogitsOnly)
            => Core(tokenIds, positions, deviceId, kvCache, mtpState, lastTokenLogitsOnly);

        private unsafe ITensor Core(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                    int deviceId, IKvCache? kvCache, IMtpState? mtpState,
                                    bool lastTokenLogitsOnly)
        {
            int seqLen = tokenIds.Length;
            int rows = honourHint && lastTokenLogitsOnly ? 1 : seqLen;
            Calls.Add(new ForwardRecord(seqLen, positions[0], lastTokenLogitsOnly,
                                        mtpState is not null, rows));

            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)rows * VocabSize * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int r = 0; r < rows; r++)
            {
                // Row r of the returned tensor corresponds to input token (seqLen - rows + r):
                // with rows == 1 that is the LAST token, which is the whole point of the hint.
                var row = new Span<float>(dst + (long)r * VocabSize, VocabSize);
                row.Fill(-10f);
                row[TargetFn(tokenIds[seqLen - rows + r])] = 10f;
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

            if (mtpState is CpuMtpState cap)
                AbsorbBatch(cap, seqLen, positions);

            return new UnmanagedTensor(new TensorShape(rows, VocabSize), DType.Float32, deviceId, ptr);
        }

        /// <summary>
        /// The trunk capture + head absorb of issue #469, mirrored from the real models: token i
        /// pairs with the carry (i == 0) or with captured row i - 1.
        /// </summary>
        private void AbsorbBatch(CpuMtpState cap, int seqLen, ReadOnlySpan<int> positions)
        {
            float[] captured = new float[seqLen * HiddenSize];
            for (int t = 0; t < seqLen; t++)
                captured.AsSpan(t * HiddenSize, HiddenSize).Fill(TrunkRowMarker(positions[t]));
            cap.SetCapturedRows(captured, seqLen);

            for (int t = 0; t < seqLen; t++)
            {
                if (t == 0) cap.SetPendingFromCarry();
                else cap.SetPendingFromCapturedRow(t - 1);
                CheckPairing(cap, positions[t], "absorb");
                StepHead(cap, positions[t]);
                AbsorbedPositions.Add(positions[t]);
            }
            cap.SeedFromCapturedRow(seqLen - 1);
        }

        private void CheckPairing(CpuMtpState cap, int position, string where)
        {
            float pending = cap.PendingHidden[0];
            bool ok = pending == ChainedMarker
                      || pending == TrunkRowMarker(position - 1)
                      || (position == 0 && pending == 0f);
            if (!ok) PairingViolations.Add($"{where} at position {position}: pending={pending}");
        }

        private static void StepHead(CpuMtpState cap, int position)
        {
            if (cap.CurrentLength > position) cap.Rollback(position);
            cap.Advance();
        }

        public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            if (!supportsMtp) throw new NotSupportedException();

            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(VocabSize * sizeof(float)), 64);
            var row = new Span<float>((float*)ptr, VocabSize);
            row.Fill(-10f);
            row[TargetFn(tokenId)] = 10f;   // the head always agrees with the trunk here

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
