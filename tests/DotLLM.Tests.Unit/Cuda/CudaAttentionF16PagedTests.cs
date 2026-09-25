using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness coverage for issue #200's direct block-table-read decode attention kernel
/// (<c>attention_f16_paged</c> / <see cref="CudaKernels.LaunchAttentionPaged"/>), the opt-in
/// (<c>DOTLLM_ATTN_PAGED_NATIVE=1</c>) alternative to <see cref="CudaPagedKvCache"/>'s default
/// gather-into-scratch dispatch (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/> +
/// <see cref="CudaKernels.LaunchAttention"/>).
/// </summary>
/// <remarks>
/// <b>Why these assert bit-exact equality, not a tolerance</b>: <c>attention_f16_paged</c> is a
/// deliberate full copy of <c>attention_f16_body</c> (see <c>attention.cu</c>'s header comment)
/// with ONLY the K/V row address resolution changed — every tile's traversal order, every
/// accumulation, every reduction is byte-identical to the flat-buffer kernel. Given the SAME
/// underlying KV content (written into the pool, then either gathered into a contiguous scratch
/// buffer or read directly through the block-pointer array), both code paths must compute the
/// exact same FP16 output — there is no reassociation here (unlike the split-KV kernels' partial
/// combine), so a tolerance-based comparison would mask a real indexing bug instead of catching
/// it. This mirrors the issue's own risk framing: "the math is unchanged... this is purely a
/// memory-access-pattern change."
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaAttentionF16PagedTests : IDisposable
{
    private const int NumLayers = 1;
    private const int BlockSize = 4;
    private const int TotalBlocks = 80;
    private const int MaxSeqLen = 320;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;

    public CudaAttentionF16PagedTests()
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

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    private CudaKvBlockPool CreatePool(int numKvHeads, int headDim) =>
        new(NumLayers, numKvHeads, headDim, BlockSize, TotalBlocks, _ctx);

    /// <summary>
    /// Writes a deterministic, exactly-FP16-representable KV history (small powers-of-two
    /// fractions, same trick <c>CudaPagedKvCacheTests</c> uses) into <paramref name="cache"/> one
    /// token at a time (matching real incremental decode), spanning block boundaries as dictated
    /// by <paramref name="seqLen"/>.
    /// </summary>
    private unsafe void FillCacheIncrementally(CudaPagedKvCache cache, int seqLen, int kvStride)
    {
        for (int pos = 0; pos < seqLen; pos++)
        {
            nint kRow = AllocAndFillDeviceFp16(kvStride, e => RowValue(pos, e, keyOffset: 0));
            nint vRow = AllocAndFillDeviceFp16(kvStride, e => RowValue(pos, e, keyOffset: 1));
            try
            {
                cache.UpdateDevice(kRow, vRow, [pos], 1, layerIndex: 0, _stream!.Handle);
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(kRow);
                CudaDriverApi.cuMemFree_v2(vRow);
            }
        }
        _stream!.Synchronize();
    }

    private static float RowValue(int pos, int elemIndex, int keyOffset) =>
        keyOffset + pos * 0.25f + elemIndex * 0.0625f;

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

    private static unsafe nint AllocAndFillDeviceFp16Random(Random rng, int count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return devPtr;
    }

    private static unsafe ushort[] ReadDeviceFp16(nint devicePtr, int count)
    {
        var host = new ushort[count];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return host;
    }

    /// <summary>
    /// Core comparison: build a scattered-block KV history of <paramref name="seqKv"/> tokens,
    /// then run the SAME decode query through (a) the existing gather-based dispatch
    /// (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/> + <see cref="CudaKernels.LaunchAttention"/>)
    /// and (b) the new direct block-table-read kernel
    /// (<see cref="CudaPagedKvCache.PrepareNativeBlockPtrs"/> + <see cref="CudaKernels.LaunchAttentionPaged"/>),
    /// asserting bit-exact FP16 output equality.
    /// </summary>
    private unsafe void AssertPagedNativeMatchesGatherPath(
        int numHeads, int numKvHeads, int headDim, int seqKv, int slidingWindow = 0)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF16Paged, "attention_f16_paged not present in PTX (stale build)");

        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        using var pool = CreatePool(numKvHeads, headDim);
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);
        FillCacheIncrementally(cache, seqKv, kvStride);
        Assert.Equal(seqKv, cache.CurrentLength);

        var rng = new Random(0xC0DE200 ^ numHeads ^ (numKvHeads << 4) ^ (headDim << 8) ^ (seqKv << 16));
        nint dQ = AllocAndFillDeviceFp16Random(rng, qStride);
        nint dOutGather = 0, dOutPaged = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dOutGather, (nuint)(qStride * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutPaged, (nuint)(qStride * sizeof(ushort))).ThrowOnError();

            nint s = _stream!.Handle;
            int positionOffset = seqKv - 1; // causal: query is the most-recently-cached position

            // (a) existing gather-based dispatch
            var (kGather, vGather) = cache.PrepareAttentionScratch(0, s);
            _stream.Synchronize();
            kernels.LaunchAttention(dQ, kGather, vGather, dOutGather,
                seqQ: 1, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow, s);
            _stream.Synchronize();

            // (b) new direct block-table-read dispatch
            var (kBlockPtrs, vBlockPtrs, blockCount) = cache.PrepareNativeBlockPtrs(0, s);
            Assert.Equal((seqKv + BlockSize - 1) / BlockSize, blockCount);
            kernels.LaunchAttentionPaged(dQ, kBlockPtrs, vBlockPtrs, dOutPaged,
                seqQ: 1, seqKv, BlockSize, numHeads, numKvHeads, headDim, positionOffset, slidingWindow, s);
            _stream.Synchronize();

            var gather = ReadDeviceFp16(dOutGather, qStride);
            var paged = ReadDeviceFp16(dOutPaged, qStride);

            for (int i = 0; i < qStride; i++)
            {
                float p = (float)BitConverter.UInt16BitsToHalf(paged[i]);
                Assert.False(float.IsNaN(p) || float.IsInfinity(p), $"NaN/Inf in paged-native output at index {i}");
                Assert.Equal(gather[i], paged[i]); // bit-exact — see class remarks
            }
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dOutGather != 0) CudaDriverApi.cuMemFree_v2(dOutGather);
            if (dOutPaged != 0) CudaDriverApi.cuMemFree_v2(dOutPaged);
        }
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_AtExactBlockBoundary()
    {
        // seqKv is an exact multiple of BlockSize (4) -- no tail partial block.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 3);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithPartialTailBlock()
    {
        // seqKv is NOT a multiple of BlockSize -- exercises the tail-block offset arithmetic.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 3 + 1);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_SingleBlockSequence()
    {
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: 1);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithGqaGroupBroadcast()
    {
        // numHeads > numKvHeads (group_size = 4): exercises the hkv = hq / group_size broadcast
        // together with block-table indirection, not just a 1:1 head mapping.
        AssertPagedNativeMatchesGatherPath(numHeads: 8, numKvHeads: 2, headDim: 16, seqKv: BlockSize * 5 + 2);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithSlidingWindow()
    {
        // Sliding window smaller than seqKv, spanning a block boundary -- exercises the
        // pos_q - tkv >= sliding_window masking branch alongside block-table indirection.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 5, slidingWindow: 6);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_AtRealBonsaiShape()
    {
        // Real Bonsai-27B decode shape (numHeads=24, numKvHeads=4, headDim=256), depth spanning
        // several blocks with a partial tail -- the shape this issue's motivating investigation
        // (docs/CUDA.md's Future Work entry) actually profiled.
        AssertPagedNativeMatchesGatherPath(numHeads: 24, numKvHeads: 4, headDim: 256, seqKv: 258);
    }

    [SkippableFact]
    public unsafe void PrepareNativeBlockPtrs_ResolvesSamePointersAsPoolAccessors()
    {
        // Allocator-logic check (not full attention numerics): the block-pointer array
        // PrepareNativeBlockPtrs uploads must resolve to EXACTLY the same device addresses
        // CudaKvBlockPool.GetKeyPtr/GetValuePtr would hand back directly for each logical block
        // in this sequence's table -- i.e. the array is a faithful, order-preserving copy, not
        // an off-by-one or stale snapshot.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int numKvHeads = 2, headDim = 16;
        int kvStride = numKvHeads * headDim;

        using var pool = CreatePool(numKvHeads, headDim);
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        int seqKv = BlockSize * 3 + 2; // 4 logical blocks, partial tail
        FillCacheIncrementally(cache, seqKv, kvStride);

        var (kBlockPtrsDevice, vBlockPtrsDevice, blockCount) = cache.PrepareNativeBlockPtrs(0, _stream!.Handle);
        _stream.Synchronize();
        Assert.Equal((seqKv + BlockSize - 1) / BlockSize, blockCount);

        var kHost = ReadDevicePtrArray(kBlockPtrsDevice, blockCount);
        var vHost = ReadDevicePtrArray(vBlockPtrsDevice, blockCount);

        for (int b = 0; b < blockCount; b++)
        {
            var (blockId, offset) = cache.BlockTable.Resolve(b * BlockSize);
            Assert.Equal(0, offset); // block-aligned position by construction
            Assert.Equal(pool.GetKeyPtr(blockId, 0), (nint)kHost[b]);
            Assert.Equal(pool.GetValuePtr(blockId, 0), (nint)vHost[b]);
        }
    }

    private static unsafe nint[] ReadDevicePtrArray(nint devicePtr, int count)
    {
        var host = new nint[count];
        fixed (nint* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(nint))).ThrowOnError();
        return host;
    }
}
