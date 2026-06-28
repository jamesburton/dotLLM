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
/// Tests for the pluggable <see cref="IKvHandoffTransfer"/> seam on <see cref="DisaggregatedScheduler"/>.
/// Proves the <see cref="CopyKvHandoffTransfer"/> path — prefill and decode replicas on <em>separate</em>
/// KV pools, with the KV-cache contents copied across at handoff — produces token-identical output to the
/// zero-copy <see cref="ReferenceKvHandoffTransfer"/> (shared pool) path. This is the in-process stand-in
/// for a cross-process / cross-device KV transfer (which cannot be validated on this single-GPU box).
/// </summary>
public sealed class DisaggregatedKvTransferTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int MaxSeqLen = 64;
    private const int EosTokenId = 0;
    private const int KvStride = NumKvHeads * HeadDim;

    private static int Ramp(int lastInputToken) => lastInputToken >= 9 ? EosTokenId : lastInputToken + 1;

    private static int RampExpectedGenerated(int promptLen)
    {
        int n = 0;
        for (int t = promptLen; t < 9; t++) n++;
        return n;
    }

    private static InferenceRequest MakeRequest(int promptLen, int maxTokens)
    {
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;
        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
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
    public async Task CopyTransfer_SeparatePools_MatchesReferenceTransfer()
    {
        int[] promptLens = [2, 3, 5, 7];

        // Reference transfer: both replicas share one pool, KV handed off by reference (#354 default).
        int[][] reference = await RunDisaggregatedAsync(promptLens, copy: false);

        // Copy transfer: prefill and decode use SEPARATE pools; KV contents are copied across at handoff.
        int[][] copied = await RunDisaggregatedAsync(promptLens, copy: true);

        for (int i = 0; i < promptLens.Length; i++)
        {
            Assert.Equal(RampExpectedGenerated(promptLens[i]), copied[i].Length);
            Assert.Equal(reference[i], copied[i]); // token-identical across the cross-pool copy
        }
    }

    private static async Task<int[][]> RunDisaggregatedAsync(int[] promptLens, bool copy)
    {
        using var prefillPool = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 64 * BlockSize);
        using var decodePool = copy
            ? new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 64 * BlockSize)
            : null;
        using var prefillModel = new RampMockModel();
        using var decodeModel = new RampMockModel();

        var scheduler = copy
            ? new DisaggregatedScheduler(
                prefillModel, decodeModel, new MockTokenizer(),
                (_, maxSeq) => prefillPool.Create(maxSeq),
                options: null, sharedPagedPool: prefillPool.Pool,
                handoffTransfer: CopyKvHandoffTransfer.Instance,
                decodeKvCacheFactory: (_, maxSeq) => decodePool!.Create(maxSeq),
                decodePagedPool: decodePool!.Pool)
            : new DisaggregatedScheduler(
                prefillModel, decodeModel, new MockTokenizer(),
                (_, maxSeq) => prefillPool.Create(maxSeq),
                options: null, sharedPagedPool: prefillPool.Pool);

        try
        {
            var handles = new ISchedulerRequest[promptLens.Length];
            for (int i = 0; i < promptLens.Length; i++)
                handles[i] = scheduler.Submit(MakeRequest(promptLens[i], 32));

            DriveUntilIdle(scheduler);

            var outputs = new int[promptLens.Length][];
            for (int i = 0; i < promptLens.Length; i++)
            {
                var r = await handles[i].Completion;
                Assert.Equal(FinishReason.Stop, r.FinishReason);
                outputs[i] = r.GeneratedTokenIds;
            }
            return outputs;
        }
        finally
        {
            scheduler.Dispose();
        }
    }

    [Fact]
    public void CopyKvHandoffTransfer_TransfersContents_ByteForByte()
    {
        using var srcPool = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 64 * BlockSize);
        using var dstPool = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 64 * BlockSize);

        // Build a source cache with 7 positions of distinct, layer-dependent K/V values.
        const int length = 7;
        var source = srcPool.Create(MaxSeqLen);
        FillCache(source, length, seed: 100);

        var config = new RampMockModel().Config;
        IKvCache dest = CopyKvHandoffTransfer.Instance.Transfer(
            source, config, (_, maxSeq) => dstPool.Create(maxSeq));

        Assert.NotSame(source, dest);
        Assert.Equal(length, dest.CurrentLength);

        // The destination's per-layer K/V must equal what the source held.
        var expected = srcPool.Create(MaxSeqLen);
        FillCache(expected, length, seed: 100);
        try
        {
            for (int layer = 0; layer < NumLayers; layer++)
            {
                AssertRefEqual(expected.GetKeysRef(layer), dest.GetKeysRef(layer), length);
                AssertRefEqual(expected.GetValuesRef(layer), dest.GetValuesRef(layer), length);
            }
        }
        finally
        {
            expected.Dispose();
            dest.Dispose();
        }
    }

    private static unsafe void FillCache(IKvCache cache, int length, int seed)
    {
        long bytes = (long)length * KvStride * sizeof(float);
        nint k = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        nint v = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        var positions = new int[length];
        try
        {
            for (int layer = 0; layer < NumLayers; layer++)
            {
                float* kp = (float*)k;
                float* vp = (float*)v;
                for (int p = 0; p < length; p++)
                {
                    positions[p] = p;
                    for (int d = 0; d < KvStride; d++)
                    {
                        kp[p * KvStride + d] = seed + layer * 1000 + p * 10 + d;
                        vp[p * KvStride + d] = -(seed + layer * 1000 + p * 10 + d);
                    }
                }
                var kRef = new TensorRef(length, KvStride, DType.Float32, -1, k);
                var vRef = new TensorRef(length, KvStride, DType.Float32, -1, v);
                cache.Update(kRef, vRef, positions, layer);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)k);
            NativeMemory.AlignedFree((void*)v);
        }
    }

    private static unsafe void AssertRefEqual(TensorRef expected, TensorRef actual, int length)
    {
        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        for (int i = 0; i < length * KvStride; i++)
            Assert.Equal(e[i], a[i]);
    }

    // ── Helpers (mirrors DisaggregatedSchedulerTests' content-driven ramp mock) ──

    private sealed class RampMockModel : IModel
    {
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
            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            int lastInput = tokenIds.Length > 0 ? tokenIds[^1] : 0;
            int emit = Ramp(lastInput);
            if ((uint)emit >= VocabSize) emit = 1;

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                float* row = dst + (long)b * VocabSize;
                for (int vv = 0; vv < VocabSize; vv++) row[vv] = -10f;
                row[emit] = 10f;
            }

            if (kvCache is not null)
            {
                long kvBytes = (long)batchSize * KvStride * sizeof(float);
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                // Content depends on the last generated token so the copy must be faithful for parity.
                NativeMemory.Fill((void*)kPtr, (nuint)kvBytes, (byte)(emit & 0xFF));
                NativeMemory.Fill((void*)vPtr, (nuint)kvBytes, (byte)((emit + 1) & 0xFF));
                try
                {
                    for (int layer = 0; layer < NumLayers; layer++)
                    {
                        var kRef = new TensorRef(batchSize, KvStride, DType.Float32, -1, kPtr);
                        var vRef = new TensorRef(batchSize, KvStride, DType.Float32, -1, vPtr);
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
        public int VocabSize => DisaggregatedKvTransferTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => DisaggregatedKvTransferTests.EosTokenId;
        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString();
        public int CountTokens(string text) => 0;
    }
}
