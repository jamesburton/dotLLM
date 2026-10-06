using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The chunking arithmetic behind the #510 staging-upload fix, as a pure CPU test —
/// no Vulkan device, so the submit-count property can be asserted without the GPU.
/// </summary>
/// <remarks>
/// The discriminating property is the <b>region count</b>. Before #510 a packed tensor
/// whose byte size was not a 4-byte multiple (every Q8_0 at 34 B/block and Q6_K at
/// 210 B/block with an odd block count) paid a SECOND <c>UploadBytes</c> call for its
/// 0-3 pad bytes: another command buffer, another <c>vkQueueSubmit</c> and another full
/// <c>vkWaitForFences</c> host stall. The plan must therefore emit exactly
/// <c>ceil(bytes / limit)</c> chunks whether or not there is a pad — emitting the pad as
/// its own chunk (the pre-#510 shape) is the mutant these tests kill.
/// </remarks>
public class VulkanStagingChunkPlanTests
{
    private static List<(long SrcOffset, long Length, long ZeroTail)> Drain(
        long bytes, long zeroTail, long capacity)
    {
        var plan = new VulkanStagingChunkPlan(bytes, zeroTail, capacity);
        var chunks = new List<(long, long, long)>();
        while (plan.MoveNext())
            chunks.Add((plan.SrcOffset, plan.Length, plan.ZeroTail));
        return chunks;
    }

    /// <summary>
    /// A Q8_0 tensor of odd block count: 34 B × 2049 blocks = 69,666 B ≡ 2 (mod 4).
    /// One chunk, pad folded in, and the copied region covers data + pad.
    /// </summary>
    [Fact]
    public void PadRidesInTheLastChunk_NotItsOwnSubmit()
    {
        const long bytes = 34L * 2049;  // 69,666 — ≡ 2 (mod 4)
        const long pad = 2;
        var chunks = Drain(bytes, pad, capacity: 1 << 20);

        Assert.Single(chunks);
        Assert.Equal(0L, chunks[0].SrcOffset);
        Assert.Equal(bytes, chunks[0].Length);
        Assert.Equal(pad, chunks[0].ZeroTail);
    }

    /// <summary>
    /// The same tensor with and without a pad must cost the SAME number of submits.
    /// This is the assertion that fails if the pad is ever re-split into its own region.
    /// (The one case where it legitimately costs one more is a tensor that exactly fills
    /// a whole number of slots — covered by
    /// <see cref="PadIsReservedFromTheSlot_NotAddedToIt"/>; a real packed tensor needing
    /// a pad is ≡ 2 (mod 4) and so cannot be a multiple of a 4 KiB-aligned slot.)
    /// </summary>
    [Theory]
    [InlineData(34L * 2049, 2L, 4096L)]
    [InlineData(210L * 777, 2L, 4096L)]
    [InlineData((1L << 26) - 1000, 3L, 1L << 25)]
    [InlineData(1000L, 3L, 4096L)]
    public void PaddedUpload_CostsTheSameSubmitsAsUnpadded(long bytes, long pad, long capacity)
    {
        int withPad = Drain(bytes, pad, capacity).Count;
        int withoutPad = Drain(bytes, 0, capacity).Count;

        Assert.Equal(withoutPad, withPad);
        Assert.Equal((int)new VulkanStagingChunkPlan(bytes, pad, capacity).ChunkCount, withPad);
    }

    /// <summary>
    /// Chunks must tile the source exactly, in order, with nothing exceeding the slot —
    /// including the pad, which shares the slot with the final data chunk.
    /// </summary>
    [Theory]
    [InlineData(0L, 0L)]
    [InlineData(1L, 3L)]
    [InlineData(4095L, 1L)]
    [InlineData(4096L, 2L)]          // exact multiple of the limit-less-pad boundary
    [InlineData(4094L, 2L)]          // data exactly fills capacity once the pad is reserved
    [InlineData(100_000L, 2L)]
    [InlineData(100_000L, 0L)]
    public void ChunksTileTheSourceAndNeverExceedTheSlot(long bytes, long pad)
    {
        const long capacity = 4096;
        var chunks = Drain(bytes, pad, capacity);

        long expectedNext = 0;
        long totalCopied = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            Assert.Equal(expectedNext, chunks[i].SrcOffset);
            Assert.True(chunks[i].Length + chunks[i].ZeroTail <= capacity,
                $"chunk {i} writes {chunks[i].Length + chunks[i].ZeroTail} B into a {capacity} B slot");
            Assert.Equal(i == chunks.Count - 1 ? pad : 0, chunks[i].ZeroTail);
            expectedNext += chunks[i].Length;
            totalCopied += chunks[i].Length + chunks[i].ZeroTail;
        }

        Assert.Equal(bytes, expectedNext);
        Assert.Equal(bytes + pad, totalCopied);
    }

    /// <summary>Nothing to upload and nothing to pad is zero submits, not one empty one.</summary>
    [Fact]
    public void EmptyUpload_EmitsNothing()
    {
        Assert.Empty(Drain(0, 0, 4096));
        Assert.Equal(0L, new VulkanStagingChunkPlan(0, 0, 4096).ChunkCount);
    }

    /// <summary>A pad with no data still has to be written — one chunk, all tail.</summary>
    [Fact]
    public void PadOnly_EmitsOneChunk()
    {
        var chunks = Drain(0, 3, 4096);
        Assert.Single(chunks);
        Assert.Equal(0L, chunks[0].Length);
        Assert.Equal(3L, chunks[0].ZeroTail);
    }

    /// <summary>
    /// The pad must be reserved out of the slot, not added on top of it: a tensor that
    /// is an exact multiple of the capacity needs one MORE chunk when padded, because
    /// the final full chunk has no room for the tail.
    /// </summary>
    [Fact]
    public void PadIsReservedFromTheSlot_NotAddedToIt()
    {
        const long capacity = 4096;
        var plan = new VulkanStagingChunkPlan(capacity, 2, capacity);
        Assert.Equal(capacity - 2, plan.Limit);

        var chunks = Drain(capacity, 2, capacity);
        Assert.Equal(2, chunks.Count);
        Assert.All(chunks, c => Assert.True(c.Length + c.ZeroTail <= capacity));
    }
}
