using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Allocator-logic and gather-correctness tests for <see cref="CudaPagedKvCache"/> (issue #252).
/// Mirrors <c>PagedKvCacheTests</c> (the CPU reference) for the parts that translate 1:1
/// (construction validation, current-length bookkeeping, rollback) and adds device-specific
/// coverage for the block-scattered write path (<see cref="CudaPagedKvCache.UpdateDevice"/>) and
/// the gather-into-scratch path (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/>) that the
/// CPU cache does not need (CPU gather is host <c>Buffer.MemoryCopy</c>; this is D2D).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPagedKvCacheTests : IDisposable
{
    private const int NumLayers = 1;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int KvStride = NumKvHeads * HeadDim; // 8
    private const int BlockSize = 4;
    private const int TotalBlocks = 32;
    private const int MaxSeqLen = 32;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;

    public CudaPagedKvCacheTests()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
    }

    public void Dispose()
    {
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    private CudaKvBlockPool CreatePool() =>
        new(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks, _ctx);

    [SkippableFact]
    public void Constructor_InitializesCorrectly()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(MaxSeqLen, cache.MaxLength);
        Assert.Equal(1, ((DotLLM.Core.Attention.IPerLayerKvCache)cache).LayerCount);
        Assert.Equal(KvStride, cache.KvStrideOf(0));
        Assert.Equal(2L * MaxSeqLen * KvStride * sizeof(ushort), cache.AllocatedBytes);
    }

    [SkippableFact]
    public void Constructor_RejectsInvalidMaxSeqLen()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaPagedKvCache(pool, 0));
    }

    [SkippableFact]
    public unsafe void UpdateDevice_ThenGather_RoundTripsExactBlockBoundary()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        // Exactly BlockSize tokens — no tail partial block.
        WriteAndGatherAndVerify(cache, seqLen: BlockSize, startPos: 0);
    }

    [SkippableFact]
    public unsafe void UpdateDevice_ThenGather_RoundTripsPartialTailBlock()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        // BlockSize + 1 tokens spans a full block plus a 1-token tail block.
        WriteAndGatherAndVerify(cache, seqLen: BlockSize + 1, startPos: 0);
    }

    [SkippableFact]
    public unsafe void UpdateDevice_ThenGather_RoundTripsSingleBlockSequence()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        WriteAndGatherAndVerify(cache, seqLen: 1, startPos: 0);
    }

    [SkippableFact]
    public unsafe void UpdateDevice_IncrementalDecodeSteps_MatchOneShotPrefill()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cacheDecode = new CudaPagedKvCache(pool, MaxSeqLen);

        const int totalTokens = BlockSize * 2 + 2; // crosses two block boundaries

        // Simulate prefill(1) + one-token-at-a-time decode, like the real Forward() dispatch.
        for (int pos = 0; pos < totalTokens; pos++)
        {
            nint kRow = AllocAndFillDeviceFp16(KvStride, i => RowValue(pos, i, keyOffset: 0));
            nint vRow = AllocAndFillDeviceFp16(KvStride, i => RowValue(pos, i, keyOffset: 1));
            try
            {
                cacheDecode.UpdateDevice(kRow, vRow, [pos], 1, layerIndex: 0, _stream!.Handle);
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(kRow);
                CudaDriverApi.cuMemFree_v2(vRow);
            }
        }
        _stream!.Synchronize();

        Assert.Equal(totalTokens, cacheDecode.CurrentLength);

        var (kPtr, vPtr) = cacheDecode.PrepareAttentionScratch(0, _stream.Handle);
        _stream.Synchronize();

        AssertGatheredRows(kPtr, totalTokens, keyOffset: 0);
        AssertGatheredRows(vPtr, totalTokens, keyOffset: 1);
    }

    [SkippableFact]
    public void Rollback_TruncatesCurrentLength()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        nint kRow = AllocAndFillDeviceFp16(KvStride, i => 1.0f);
        nint vRow = AllocAndFillDeviceFp16(KvStride, i => 2.0f);
        try
        {
            for (int pos = 0; pos < 6; pos++)
                cache.UpdateDevice(kRow, vRow, [pos], 1, layerIndex: 0, _stream!.Handle);
            _stream!.Synchronize();
            Assert.Equal(6, cache.CurrentLength);

            cache.Rollback(3);
            Assert.Equal(3, cache.CurrentLength);

            Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(10));
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kRow);
            CudaDriverApi.cuMemFree_v2(vRow);
        }
    }

    [SkippableFact]
    public void PrefixSharing_ForkThenCopyOnWrite_PreservesSourceBlocks()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var source = new CudaPagedKvCache(pool, MaxSeqLen);
        using var target = new CudaPagedKvCache(pool, MaxSeqLen);

        nint kRow = AllocAndFillDeviceFp16(KvStride, i => 5.0f);
        nint vRow = AllocAndFillDeviceFp16(KvStride, i => 6.0f);
        try
        {
            for (int pos = 0; pos < BlockSize; pos++)
                source.UpdateDevice(kRow, vRow, [pos], 1, layerIndex: 0, _stream!.Handle);
            _stream!.Synchronize();

            var blockIds = new List<int>();
            source.SnapshotFullBlocks(BlockSize, blockIds);
            Assert.Single(blockIds);

            // Simulate the prefix-trie handoff: AddRef the shared block, seed the target.
            pool.AddRef(blockIds[0]);
            target.SeedSharedPrefix(blockIds, BlockSize);

            Assert.Equal(BlockSize, target.CurrentLength);
            Assert.Equal(2, pool.RefCount(blockIds[0]));

            // Disposing the target releases its ref; the source's block (and data) survive.
            target.Dispose();
            Assert.Equal(1, pool.RefCount(blockIds[0]));

            var (kPtr, _) = source.PrepareAttentionScratch(0, _stream.Handle);
            _stream.Synchronize();
            AssertGatheredRows(kPtr, BlockSize, keyOffset: 0, expected: _ => 5.0f);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kRow);
            CudaDriverApi.cuMemFree_v2(vRow);
        }
    }

    [SkippableFact]
    public void HostTensorSurface_Throws()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);
        DotLLM.Core.Attention.IKvCache ikv = cache;

        Assert.Throws<NotSupportedException>(() => ikv.Update(default(ITensor)!, default(ITensor)!, [], 0));
        Assert.Throws<NotSupportedException>(() => ikv.Update(default(TensorRef), default(TensorRef), [], 0));
        Assert.Throws<NotSupportedException>(() => ikv.GetKeys(0));
        Assert.Throws<NotSupportedException>(() => ikv.GetValues(0));
        Assert.Throws<NotSupportedException>(() => ikv.GetKeysRef(0));
        Assert.Throws<NotSupportedException>(() => ikv.GetValuesRef(0));
    }

    // ── helpers ──

    // Deliberately built from powers-of-two fractions (0.25 / 0.0625) at small magnitude so the
    // FP16 round trip (write via Half, read back via Half) is exact — no rounding-tolerance
    // guesswork needed to discriminate a real gather bug from ordinary FP16 quantization noise.
    private static float RowValue(int pos, int elemIndex, int keyOffset) => keyOffset + pos * 0.25f + elemIndex * 0.0625f;

    private unsafe void WriteAndGatherAndVerify(CudaPagedKvCache cache, int seqLen, int startPos)
    {
        var positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = startPos + i;

        nint kSeq = AllocAndFillDeviceFp16(seqLen * KvStride, i => RowValue(positions[i / KvStride], i % KvStride, keyOffset: 0));
        nint vSeq = AllocAndFillDeviceFp16(seqLen * KvStride, i => RowValue(positions[i / KvStride], i % KvStride, keyOffset: 1));
        try
        {
            cache.UpdateDevice(kSeq, vSeq, positions, seqLen, layerIndex: 0, _stream!.Handle);
            _stream.Synchronize();

            Assert.Equal(startPos + seqLen, cache.CurrentLength);

            var (kPtr, vPtr) = cache.PrepareAttentionScratch(0, _stream.Handle);
            _stream.Synchronize();

            AssertGatheredRows(kPtr, startPos + seqLen, keyOffset: 0);
            AssertGatheredRows(vPtr, startPos + seqLen, keyOffset: 1);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kSeq);
            CudaDriverApi.cuMemFree_v2(vSeq);
        }
    }

    private static unsafe nint AllocAndFillDeviceFp16(int count, Func<int, float> pattern)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)pattern(i));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return devPtr;
    }

    private static unsafe void AssertGatheredRows(nint devicePtr, int rowCount, int keyOffset, Func<int, float>? expected = null)
    {
        int count = rowCount * KvStride;
        var host = new ushort[count];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(ushort))).ThrowOnError();

        for (int pos = 0; pos < rowCount; pos++)
        {
            for (int e = 0; e < KvStride; e++)
            {
                float actual = (float)BitConverter.UInt16BitsToHalf(host[pos * KvStride + e]);
                float want = expected is not null ? expected(e) : RowValue(pos, e, keyOffset);
                // want is built from small powers-of-two fractions (see RowValue), so the FP16
                // round trip is exact — an exact compare discriminates a real gather/index bug
                // from ordinary FP16 quantization noise (which this test deliberately avoids).
                Assert.Equal(want, actual);
            }
        }
    }
}
