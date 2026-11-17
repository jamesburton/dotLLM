using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only coverage for the host-side bookkeeping behind CUDA paged decode (issue #1839): the
/// block table's <see cref="CudaKvBlockTable.Version"/>, the re-upload decision
/// (<see cref="CudaPagedKvCache.NeedsBlockPtrUpload"/>), the combined pointer-array layout
/// (<see cref="CudaPagedKvCache.PtrSlot"/>) and the dispatch gate
/// (<see cref="CudaTransformerModel.UsePagedNativeAttention"/>). No CUDA device: the table runs
/// over <see cref="FakeAllocator"/>.
/// </summary>
/// <remarks>
/// The device block-pointer arrays are uploaded only when the table version changes. A missed
/// bump leaves the attention kernel reading a stale block pointer, which is silently wrong
/// output, not a crash. So every mutation of a block ID must bump it, and appends inside an
/// allocated block must not (otherwise the per-step upload cost #1839 removed comes back).
/// </remarks>
public sealed class CudaPagedDecodeBookkeepingTests
{
    private const int BlockSize = 16;

    /// <summary>Host-only block allocator with ref counts; IDs are handed out in order.</summary>
    private sealed class FakeAllocator : ICudaKvBlockAllocator
    {
        private readonly Dictionary<int, int> _refs = [];
        private int _next;

        public int BlockSize => CudaPagedDecodeBookkeepingTests.BlockSize;

        public int Allocate()
        {
            int id = _next++;
            _refs[id] = 1;
            return id;
        }

        public void AddRef(int blockId) => _refs[blockId]++;

        public int RefCount(int blockId) => _refs.TryGetValue(blockId, out int r) ? r : 0;

        public void Release(int blockId)
        {
            if (--_refs[blockId] == 0) _refs.Remove(blockId);
        }

        public int CopyBlock(int sourceBlockId, nint stream) => Allocate();
    }

    /// <summary>
    /// Mirrors CudaPagedKvCache's per-layer decode loop: write the row (allocate + CoW + advance,
    /// as FusedRopeAndUpdateDevice does), then each layer's PrepareNativeBlockPtrs asks whether
    /// the arrays must be re-uploaded. Returns the number of uploads.
    /// </summary>
    private static int SimulateDecode(CudaKvBlockTable table, int fromPos, int toPos, int numLayers,
                                      ref int uploadedVersion, ref int uploadedBlocks)
    {
        int uploads = 0;
        for (int pos = fromPos; pos < toPos; pos++)
        {
            table.EnsureCapacity(pos + 1);
            table.EnsureWritable(pos, 0);
            if (pos + 1 > table.CurrentLength) table.Advance(pos + 1);

            for (int layer = 0; layer < numLayers; layer++)
            {
                if (CudaPagedKvCache.NeedsBlockPtrUpload(table.Version, uploadedVersion, table.BlockCount, uploadedBlocks))
                {
                    uploads++;
                    uploadedVersion = table.Version;
                    uploadedBlocks = table.BlockCount;
                }
            }
        }
        return uploads;
    }

    [Fact]
    public void Decode_UploadsOncePerNewBlock_NotPerLayerOrPerStep()
    {
        var table = new CudaKvBlockTable(new FakeAllocator());
        int uv = -1, ub = 0;

        // 686 tokens (the #1839 measurement point) over 40 layers.
        int uploads = SimulateDecode(table, 0, 686, numLayers: 40, ref uv, ref ub);

        int blocks = (686 + BlockSize - 1) / BlockSize; // 43
        Assert.Equal(blocks, table.BlockCount);
        Assert.Equal(blocks, uploads); // was 686 * 40 = 27,440 per-layer synchronous uploads before #1839
    }

    [Fact]
    public void Version_BumpsOnAllocation_NotOnAppendInsideAllocatedBlock()
    {
        var table = new CudaKvBlockTable(new FakeAllocator());
        table.Advance(1);
        int v1 = table.Version;

        table.Advance(BlockSize); // fills the first block: no new block
        Assert.Equal(v1, table.Version);

        table.Advance(BlockSize + 1); // spills into block 2
        Assert.NotEqual(v1, table.Version);
    }

    [Fact]
    public void Version_BumpsOnCopyOnWrite_AndTheTableChangesBlock()
    {
        var alloc = new FakeAllocator();
        var source = new CudaKvBlockTable(alloc);
        source.Advance(BlockSize * 2);

        var ids = new List<int>();
        source.SnapshotFullBlocks(BlockSize * 2, ids);
        foreach (int id in ids) alloc.AddRef(id);

        var target = new CudaKvBlockTable(alloc);
        target.SeedSharedBlocks(ids, BlockSize * 2);
        target.SetCurrentLength(BlockSize + 3); // rollback into the shared second block
        int uv = target.Version, ub = target.BlockCount;
        int before = target.BlockIdAt(1);

        // The next decode write lands in a shared block -> CoW -> new physical block -> must re-upload.
        int uploads = SimulateDecode(target, BlockSize + 3, BlockSize + 4, numLayers: 4, ref uv, ref ub);

        Assert.NotEqual(before, target.BlockIdAt(1));
        Assert.Equal(1, uploads);
    }

    [Fact]
    public void Version_UnchangedByWriteToUnsharedBlock()
    {
        var table = new CudaKvBlockTable(new FakeAllocator());
        table.Advance(5);
        int v = table.Version;
        table.EnsureWritable(3, 0);
        Assert.Equal(v, table.Version);
    }

    [Fact]
    public void Version_UnchangedByRollback_SoReDecodingTheSameSlotsDoesNotUpload()
    {
        var table = new CudaKvBlockTable(new FakeAllocator());
        int uv = -1, ub = 0;
        SimulateDecode(table, 0, 40, numLayers: 2, ref uv, ref ub);

        table.SetCurrentLength(20); // speculative rollback inside allocated blocks: IDs unchanged
        int uploads = SimulateDecode(table, 20, 40, numLayers: 2, ref uv, ref ub);
        Assert.Equal(0, uploads);
    }

    [Theory]
    // #1843: blocks a cache still needs to hold maxLength positions, given what it holds; -1 past the whole pool.
    [InlineData(32, 0, 10, 2)]
    [InlineData(33, 2, 10, 1)]
    [InlineData(16, 3, 10, 0)]     // holds more than it needs (rolled back): nothing to allocate
    [InlineData(160, 0, 10, 10)]
    [InlineData(161, 0, 10, -1)]
    [InlineData(161, 10, 10, -1)]
    public void BlocksStillNeeded_CountsOnlyMissingBlocks_AndRefusesPastThePool(int maxLength, int held, int total, int expected)
        => Assert.Equal(expected, CudaPagedKvCache.BlocksStillNeeded(maxLength, held, BlockSize, total));

    [Fact]
    public void Version_BumpsOnSeed()
    {
        var alloc = new FakeAllocator();
        int id = alloc.Allocate();
        var table = new CudaKvBlockTable(alloc);
        int v = table.Version;
        table.SeedSharedBlocks([id], BlockSize);
        Assert.NotEqual(v, table.Version);
    }

    [Fact]
    public void Version_BumpsOnFreeThenReallocate_EvenWhenBlockCountMatches()
    {
        // Free + reallocate to the same block count must still re-upload: the IDs differ.
        var alloc = new FakeAllocator();
        var table = new CudaKvBlockTable(alloc);
        int uv = -1, ub = 0;
        SimulateDecode(table, 0, BlockSize, numLayers: 1, ref uv, ref ub);
        int firstId = table.BlockIdAt(0);

        var other = new CudaKvBlockTable(alloc); // occupies the freed ID later
        table.Free();
        other.Advance(1);
        int uploads = SimulateDecode(table, 0, BlockSize, numLayers: 1, ref uv, ref ub);

        Assert.NotEqual(firstId, table.BlockIdAt(0));
        Assert.Equal(1, uploads);
    }

    [Fact]
    public void Version_BumpsOnForkTarget()
    {
        var alloc = new FakeAllocator();
        var source = new CudaKvBlockTable(alloc);
        source.Advance(BlockSize);
        var target = new CudaKvBlockTable(alloc);
        int v = target.Version;
        source.Fork(target);
        Assert.NotEqual(v, target.Version);
    }

    [Theory]
    [InlineData(3, 3, 5, 5, false)]
    [InlineData(4, 3, 5, 5, true)]  // table changed
    [InlineData(3, 3, 6, 5, true)]  // more blocks than uploaded
    [InlineData(3, 3, 4, 5, false)] // fewer blocks (rollback): uploaded entries still valid
    public void NeedsBlockPtrUpload_TruthTable(int version, int uploadedVersion, int blocks, int uploadedBlocks, bool expected)
        => Assert.Equal(expected, CudaPagedKvCache.NeedsBlockPtrUpload(version, uploadedVersion, blocks, uploadedBlocks));

    [Fact]
    public void PtrSlot_LayoutIsKvMajorThenLayerThenBlock_AndCoversTheArrayExactly()
    {
        const int numLayers = 5, maxBlocks = 7;
        var seen = new HashSet<int>();
        for (int kv = 0; kv < 2; kv++)
        for (int layer = 0; layer < numLayers; layer++)
        for (int b = 0; b < maxBlocks; b++)
        {
            int slot = CudaPagedKvCache.PtrSlot(kv, layer, b, numLayers, maxBlocks);
            Assert.InRange(slot, 0, 2 * numLayers * maxBlocks - 1);
            Assert.True(seen.Add(slot), $"slot {slot} reused");
        }

        // Each layer's K (or V) row is contiguous: the kernel indexes block_ptrs[logical_block].
        Assert.Equal(CudaPagedKvCache.PtrSlot(0, 2, 0, numLayers, maxBlocks) + 3,
                     CudaPagedKvCache.PtrSlot(0, 2, 3, numLayers, maxBlocks));
        // Layer stride is maxBlocks, V region starts after all K rows.
        Assert.Equal(2 * maxBlocks, CudaPagedKvCache.PtrSlot(0, 2, 0, numLayers, maxBlocks));
        Assert.Equal(numLayers * maxBlocks, CudaPagedKvCache.PtrSlot(1, 0, 0, numLayers, maxBlocks));
    }

    [Theory]
    [InlineData(1, 0, true, true, true)]
    [InlineData(2, 0, true, true, false)]   // prefill keeps the gather (G3 / flash need a contiguous buffer)
    [InlineData(1, 1, true, true, false)]   // sinks: paged kernel has no sink epilogue
    [InlineData(1, 0, false, true, false)]  // DOTLLM_ATTN_PAGED_NATIVE=0
    [InlineData(1, 0, true, false, false)]  // stale PTX
    public void UsePagedNativeAttention_Gate(int seqLen, long sinks, bool enabled, bool hasKernel, bool expected)
        => Assert.Equal(expected, CudaTransformerModel.UsePagedNativeAttention(seqLen, (nint)sinks, enabled, hasKernel));
}
