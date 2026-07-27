using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// Tests for <see cref="DisaggregatedScheduler"/> — two model replicas (prefill + decode) sharing one
/// paged KV pool, with the sequence's KV-cache handed off by reference from prefill to decode. Uses a
/// content-driven (ramp) mock so output is replica-independent: the emitted token depends only on the
/// generated-token chain, so a sequence that prefills on replica A and decodes on replica B over the
/// shared cache produces the same tokens as a single scheduler would.
/// </summary>
public sealed class DisaggregatedSchedulerTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int MaxSeqLen = 64;
    private const int EosTokenId = 0;

    // Ramp: a prompt ending in token k generates k+1, k+2, ... up to 9, then EOS.
    private static int Ramp(int lastInputToken) => lastInputToken >= 9 ? EosTokenId : lastInputToken + 1;

    private static int RampExpectedGenerated(int promptLen)
    {
        int n = 0;
        for (int t = promptLen; t < 9; t++) n++;
        return n;
    }

    private static InferenceRequest MakeRequest(int promptLen, int maxTokens, string? apiKey = null)
    {
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;
        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
            ApiKey = apiKey,
        };
    }

    private static void DriveUntilIdle(DisaggregatedScheduler s, int maxIterations = 2000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (s.IsIdle) return;
            s.Step();
        }
        Assert.Fail("Disaggregated scheduler did not reach idle within iteration cap.");
    }

    [Fact]
    public async Task Disaggregated_Output_MatchesSingleSchedulerBaseline()
    {
        int[] promptLens = [2, 3, 5, 7];

        // Single-scheduler baseline.
        int[][] baseline = new int[promptLens.Length][];
        using (var pf = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 64 * BlockSize))
        using (var model = new RampMockModel())
        {
            using var sched = new ContinuousBatchScheduler(model, new MockTokenizer(),
                (_, maxSeq) => pf.Create(maxSeq), options: null, pagedPool: pf.Pool);
            var hs = new ISchedulerRequest[promptLens.Length];
            for (int i = 0; i < promptLens.Length; i++) hs[i] = sched.Submit(MakeRequest(promptLens[i], 32));
            for (int i = 0; i < 2000 && !sched.IsIdle; i++) sched.Step();
            for (int i = 0; i < promptLens.Length; i++) baseline[i] = (await hs[i].Completion).GeneratedTokenIds;
        }

        // Disaggregated: two replicas, shared pool, KV handoff.
        using var fix = new DisFixture();
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], 32));

        DriveUntilIdle(fix.Scheduler);

        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(RampExpectedGenerated(promptLens[i]), r.GeneratedTokenCount);
            Assert.Equal(baseline[i], r.GeneratedTokenIds); // token-identical across the handoff
        }
    }

    [Fact]
    public async Task Disaggregated_DecodeRunsOnDecodeReplica_PrefillReplicaNeverDecodes()
    {
        // Prompts length >= 2 ⇒ prefill forwards are multi-token, decode forwards are single-token.
        // The prefill replica must never issue a single-token (decode) forward; the decode replica must.
        using var fix = new DisFixture();
        var handles = new ISchedulerRequest[3];
        int[] promptLens = [2, 3, 4];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], 32));

        DriveUntilIdle(fix.Scheduler);
        foreach (var h in handles) await h.Completion;

        Assert.True(fix.DecodeModel.SingleTokenForwards > 0, "decode replica never decoded");
        Assert.Equal(0, fix.PrefillModel.SingleTokenForwards); // prefill replica only prefilled
        Assert.True(fix.PrefillModel.MultiTokenForwards > 0, "prefill replica never prefilled");
    }

    [Fact]
    public async Task Disaggregated_PerSeqMaxTokens_Honored()
    {
        // Never emits EOS (prompt last token small + low cap) — each stops at its own max-tokens.
        using var fix = new DisFixture(constantToken: 7);
        int[] maxTokens = [2, 4, 6];
        var handles = new ISchedulerRequest[maxTokens.Length];
        for (int i = 0; i < maxTokens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: maxTokens[i]));

        DriveUntilIdle(fix.Scheduler);

        for (int i = 0; i < maxTokens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Length, r.FinishReason);
            Assert.Equal(maxTokens[i], r.GeneratedTokenCount);
        }
    }

    [Fact]
    public async Task Disaggregated_PerKeyTokenUsage_MergedAcrossReplicas()
    {
        using var fix = new DisFixture();
        var a = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 32, apiKey: "alice"));
        var b = fix.Scheduler.Submit(MakeRequest(promptLen: 5, maxTokens: 32, apiKey: "bob"));
        DriveUntilIdle(fix.Scheduler);
        var ra = await a.Completion; var rb = await b.Completion;

        var usage = fix.Scheduler.GetPerKeyTokenUsage();
        Assert.Equal(ra.GeneratedTokenCount, usage["alice"]);
        Assert.Equal(rb.GeneratedTokenCount, usage["bob"]);
    }

    [Fact]
    public async Task Disaggregated_AsyncRunLoop_EndToEndParity()
    {
        using var fix = new DisFixture();
        using var cts = new CancellationTokenSource();
        var loop = fix.Scheduler.RunLoopAsync(cts.Token);

        int[] promptLens = [2, 3, 5, 7];
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], 32));

        var all = Task.WhenAll(Array.ConvertAll(handles, h => h.Completion));
        var done = await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(10)));
        Assert.Same(all, done); // completed before the timeout

        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(RampExpectedGenerated(promptLens[i]), r.GeneratedTokenCount);
        }

        cts.Cancel();
        try { await loop; } catch (OperationCanceledException) { }
    }

    // ── Helpers ──

    private sealed class DisFixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public RampMockModel PrefillModel { get; }
        public RampMockModel DecodeModel { get; }
        public DisaggregatedScheduler Scheduler { get; }

        public DisFixture(int? constantToken = null)
        {
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize,
                maxTotalTokens: 64 * BlockSize);
            PrefillModel = new RampMockModel(constantToken);
            DecodeModel = new RampMockModel(constantToken);
            Scheduler = new DisaggregatedScheduler(
                PrefillModel, DecodeModel, new MockTokenizer(),
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                options: null, sharedPagedPool: PagedFactory.Pool);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            PagedFactory.Dispose();
            PrefillModel.Dispose();
            DecodeModel.Dispose();
        }
    }

    /// <summary>Content-driven mock: emits ramp(lastInputToken) (or a constant), updates the shared KV
    /// cache, and counts single- vs multi-token forwards so tests can attribute prefill vs decode.</summary>
    private sealed class RampMockModel : IModel
    {
        private readonly int? _constant;
        public int SingleTokenForwards { get; private set; }
        public int MultiTokenForwards { get; private set; }

        public RampMockModel(int? constant = null) => _constant = constant;

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;
            if (batchSize == 1) SingleTokenForwards++; else MultiTokenForwards++;

            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            int lastInput = tokenIds.Length > 0 ? tokenIds[^1] : 0;
            int emit = _constant ?? Ramp(lastInput);
            if ((uint)emit >= VocabSize) emit = 1;

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emit] = 10f;
            }

            if (kvCache is not null)
            {
                int kvStride = NumKvHeads * HeadDim;
                long kvBytes = (long)batchSize * kvStride * sizeof(float);
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                NativeMemory.Clear((void*)kPtr, (nuint)kvBytes);
                NativeMemory.Clear((void*)vPtr, (nuint)kvBytes);
                try
                {
                    for (int layer = 0; layer < NumLayers; layer++)
                    {
                        var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                        var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                        kvCache.Update(kRef, vRef, positions, layer);
                    }
                }
                finally
                {
                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, logitsPtr);
        }

        public void Dispose() { }
    }

    private sealed class MockTokenizer : ITokenizer
    {
        public int VocabSize => DisaggregatedSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => DisaggregatedSchedulerTests.EosTokenId;
        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => 0;
    }
}
