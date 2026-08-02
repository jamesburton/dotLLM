using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Allocator-logic tests for <see cref="CudaKvBlockTable"/> (issue #252). Mirrors
/// <c>KvBlockTableTests</c> (the CPU <see cref="DotLLM.Engine.KvCache.KvBlockTable"/> reference) —
/// same EnsureCapacity/Resolve/Fork/EnsureWritable(CoW)/Free semantics — driven against a real
/// GPU-resident <see cref="CudaKvBlockPool"/> instead of the CPU pool.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaKvBlockTableTests : IDisposable
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int BlockSize = 4;
    private const int TotalBlocks = 16;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;

    public CudaKvBlockTableTests()
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
    public void InitialState_EmptyTable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        Assert.Equal(0, table.CurrentLength);
        Assert.Equal(0, table.BlockCount);
    }

    [SkippableFact]
    public void Advance_AllocatesBlocksAsNeeded()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.Advance(3);  // 3 tokens, fits in 1 block (blockSize=4)
        Assert.Equal(3, table.CurrentLength);
        Assert.Equal(1, table.BlockCount);

        table.Advance(5);  // crosses into 2nd block
        Assert.Equal(5, table.CurrentLength);
        Assert.Equal(2, table.BlockCount);
    }

    [SkippableFact]
    public void Resolve_ReturnsCorrectBlockAndOffset()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.Advance(10);  // 3 blocks needed (4+4+2)

        var (block0, offset0) = table.Resolve(0);
        Assert.Equal(0, offset0);

        var (block0b, offset3) = table.Resolve(3);
        Assert.Equal(block0, block0b);
        Assert.Equal(3, offset3);

        var (block1, offset4) = table.Resolve(4);
        Assert.NotEqual(block0, block1);
        Assert.Equal(0, offset4);

        var (_, offset9) = table.Resolve(9);
        Assert.Equal(1, offset9);
    }

    [SkippableFact]
    public void EnsureCapacity_DoesNotAllocateExtraBlocks()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.EnsureCapacity(4);  // exactly 1 block
        Assert.Equal(1, table.BlockCount);

        table.EnsureCapacity(4);  // same capacity, no new blocks
        Assert.Equal(1, table.BlockCount);

        table.EnsureCapacity(5);  // needs 2 blocks now
        Assert.Equal(2, table.BlockCount);
    }

    [SkippableFact]
    public void Fork_SharesBlocksViaRefCount()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var source = new CudaKvBlockTable(pool);
        var target = new CudaKvBlockTable(pool);

        source.Advance(6);  // 2 blocks
        Assert.Equal(TotalBlocks - 2, pool.FreeBlocks);

        source.Fork(target);

        Assert.Equal(6, target.CurrentLength);
        Assert.Equal(2, target.BlockCount);
        Assert.Equal(TotalBlocks - 2, pool.FreeBlocks); // shared, no new allocation

        var (srcBlock0, _) = source.Resolve(0);
        var (tgtBlock0, _) = target.Resolve(0);
        Assert.Equal(srcBlock0, tgtBlock0);
        Assert.Equal(2, pool.RefCount(srcBlock0));
    }

    [SkippableFact]
    public void EnsureWritable_CopiesSharedBlockViaDeviceToDeviceCopy()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var source = new CudaKvBlockTable(pool);
        var target = new CudaKvBlockTable(pool);

        source.Advance(4);  // 1 block

        var (srcBlockBefore, _) = source.Resolve(0);
        Assert.Equal(1, pool.RefCount(srcBlockBefore));

        source.Fork(target);
        Assert.Equal(2, pool.RefCount(srcBlockBefore));

        // CoW: target's write path must duplicate the shared block onto its own stream.
        target.EnsureWritable(0, _stream!.Handle);
        _stream.Synchronize();

        var (tgtBlockAfter, _) = target.Resolve(0);
        Assert.NotEqual(srcBlockBefore, tgtBlockAfter);   // new physical block
        Assert.Equal(1, pool.RefCount(srcBlockBefore));   // source refcount back to 1
        Assert.Equal(1, pool.RefCount(tgtBlockAfter));    // new block owns its own ref
    }

    [SkippableFact]
    public void EnsureWritable_NoOpWhenNotShared()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.Advance(4);
        var (blockBefore, _) = table.Resolve(0);

        table.EnsureWritable(0, _stream!.Handle); // refcount=1 already — must be a no-op, no new block
        _stream.Synchronize();

        var (blockAfter, _) = table.Resolve(0);
        Assert.Equal(blockBefore, blockAfter);
    }

    [SkippableFact]
    public void Free_ReleasesAllBlocksToPool()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.Advance(10);  // 3 blocks
        Assert.Equal(TotalBlocks - 3, pool.FreeBlocks);

        table.Free();

        Assert.Equal(0, table.CurrentLength);
        Assert.Equal(0, table.BlockCount);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
    }

    [SkippableFact]
    public void SetCurrentLength_TruncatesVisibleLengthWithoutFreeingBlocks()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var pool = CreatePool();
        var table = new CudaKvBlockTable(pool);

        table.Advance(10);
        Assert.Equal(10, table.CurrentLength);

        table.SetCurrentLength(5);
        Assert.Equal(5, table.CurrentLength);
        Assert.Equal(3, table.BlockCount);  // blocks not freed — just length truncated
    }
}
