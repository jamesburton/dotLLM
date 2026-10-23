using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Allocator-logic tests for <see cref="CudaKvBlockPool"/> (issue #252). Mirrors
/// <c>KvBlockPoolTests</c> (the CPU <see cref="DotLLM.Engine.KvCache.KvBlockPool"/> reference) —
/// same allocate/free/refcount/CopyBlock semantics — adapted for a real CUDA context and
/// device-resident FP16 storage (round-tripped through host <see cref="Half"/> for verification).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaKvBlockPoolTests : IDisposable
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int BlockSize = 4;
    private const int TotalBlocks = 8;
    private const int KvStride = NumKvHeads * HeadDim; // 32

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;

    public CudaKvBlockPoolTests()
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
    public void Constructor_AllBlocksFree()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        Assert.Equal(BlockSize, pool.BlockSize);
        Assert.Equal(TotalBlocks, pool.TotalBlocks);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
        Assert.Equal(NumLayers, pool.NumLayers);
        Assert.Equal(2L * NumLayers * TotalBlocks * BlockSize * KvStride * sizeof(ushort), pool.AllocatedBytes);
    }

    [SkippableFact]
    public void Constructor_RejectsInvalidDimensions()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaKvBlockPool(NumLayers, NumKvHeads, HeadDim, 0, TotalBlocks, _ctx));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaKvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, 0, _ctx));
    }

    [SkippableFact]
    public void Allocate_ReturnsBlockWithRefCount1()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        int blockId = pool.Allocate();

        Assert.Equal(1, pool.RefCount(blockId));
        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);
    }

    [SkippableFact]
    public void Allocate_ReturnsDistinctBlocks()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        var ids = new HashSet<int>();
        for (int i = 0; i < TotalBlocks; i++)
            ids.Add(pool.Allocate());

        Assert.Equal(TotalBlocks, ids.Count);
        Assert.Equal(0, pool.FreeBlocks);
    }

    [SkippableFact]
    public void Allocate_ThrowsWhenExhausted()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        for (int i = 0; i < TotalBlocks; i++)
            pool.Allocate();

        Assert.Throws<InvalidOperationException>(() => pool.Allocate());
    }

    [SkippableFact]
    public void Release_ReturnsBlockToPool()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        int blockId = pool.Allocate();
        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);

        pool.Release(blockId);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
        Assert.Equal(0, pool.RefCount(blockId));
    }

    [SkippableFact]
    public void Release_ThrowsOnFreeBlock()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        int blockId = pool.Allocate();
        pool.Release(blockId);

        Assert.Throws<InvalidOperationException>(() => pool.Release(blockId));
    }

    [SkippableFact]
    public void AddRef_ThrowsOnFreeBlock()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        int blockId = pool.Allocate();
        pool.Release(blockId);

        Assert.Throws<InvalidOperationException>(() => pool.AddRef(blockId));
    }

    [SkippableFact]
    public void Release_DoesNotFreeUntilRefCountZero()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        int blockId = pool.Allocate();
        pool.AddRef(blockId);    // refCount = 2
        pool.Release(blockId);   // refCount = 1

        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);
        Assert.Equal(1, pool.RefCount(blockId));

        pool.Release(blockId);   // refCount = 0, returned to pool
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
    }

    [SkippableFact]
    public unsafe void GetKeyPtr_RoundTripsFp16DataAcrossLayers()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        int blockId = pool.Allocate();
        int blockFloats = BlockSize * KvStride;

        // Layer 0 pattern
        WriteRow(pool.GetKeyPtr(blockId, 0), blockFloats, i => i + 1.0f);
        // Layer 1 pattern (independent storage — must not alias layer 0)
        WriteRow(pool.GetKeyPtr(blockId, 1), blockFloats, i => -(i + 1.0f));

        AssertRow(pool.GetKeyPtr(blockId, 0), blockFloats, i => i + 1.0f);
        AssertRow(pool.GetKeyPtr(blockId, 1), blockFloats, i => -(i + 1.0f));
    }

    [SkippableFact]
    public unsafe void CopyBlock_CreatesIndependentDeviceCopy()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();

        int srcId = pool.Allocate();
        int blockFloats = BlockSize * KvStride;
        for (int layer = 0; layer < NumLayers; layer++)
        {
            WriteRow(pool.GetKeyPtr(srcId, layer), blockFloats, i => layer * 100 + i);
            WriteRow(pool.GetValuePtr(srcId, layer), blockFloats, i => layer * 1000 + i);
        }

        int copyId = pool.CopyBlock(srcId, _stream!.Handle);
        _stream.Synchronize();

        Assert.NotEqual(srcId, copyId);
        for (int layer = 0; layer < NumLayers; layer++)
        {
            AssertRow(pool.GetKeyPtr(copyId, layer), blockFloats, i => layer * 100 + i);
            AssertRow(pool.GetValuePtr(copyId, layer), blockFloats, i => layer * 1000 + i);
        }

        // Mutate the copy — source must be unaffected (independent device allocation).
        WriteRow(pool.GetKeyPtr(copyId, 0), blockFloats, _ => -1.0f);
        AssertRow(pool.GetKeyPtr(srcId, 0), blockFloats, i => 0 * 100 + i);
    }

    [SkippableFact]
    public void Dispose_IsSafeToCallMultipleTimes()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        pool.Dispose(); // explicit dispose; the `using` scope disposes again on exit — should not throw
    }

    // ── FP16 device round-trip helpers (H2D write / D2H read via host Half) ──

    private static unsafe void WriteRow(nint devicePtr, int count, Func<int, float> pattern)
    {
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)pattern(i));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devicePtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
    }

    private static unsafe void AssertRow(nint devicePtr, int count, Func<int, float> expected)
    {
        var host = new ushort[count];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        for (int i = 0; i < count; i++)
            Assert.Equal(expected(i), (float)BitConverter.UInt16BitsToHalf(host[i]), 3);
    }
}
