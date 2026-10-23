using DotLLM.Cuda.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #478, host-only: the CUDA hybrid models' KV length bookkeeping after a speculative
/// rollback. No GPU needed — <see cref="HybridKvLengthBookkeeping"/> is the pure rule both models
/// call once per forward; <see cref="CudaQwen3HybridDenseKvRollbackTest"/> covers the wiring on
/// real hardware.
/// </summary>
public sealed class HybridKvLengthBookkeepingTests
{
    /// <summary>
    /// The MTP round shape: prefill [0,3), verify [3,6) (cursor 6), a partial rejection rolls the
    /// handle back to 4, and the next forward appends position 4. Both lengths must shrink to 4 so
    /// that append is incremental; the pre-#478 grow-only rule left them at 6, which forced the
    /// full-range reconversion.
    /// </summary>
    [Fact]
    public void RollbackThenAppend_ShrinksBothLengths_AndStaysIncremental()
    {
        int f16 = 0;
        int[] valid = [0, 0];

        // Prefill + verify, each an incremental append on every slot.
        foreach (int[] batch in new[] { new[] { 0, 1, 2 }, new[] { 3, 4, 5 } })
        {
            f16 = HybridKvLengthBookkeeping.SyncToCommitted(f16, valid, committedLength: batch[0]);
            f16 = HybridKvLengthBookkeeping.LengthAfterWrite(f16, batch);
            for (int s = 0; s < valid.Length; s++)
            {
                Assert.True(HybridKvLengthBookkeeping.IsIncrementalAppend(batch, valid[s], forceFull: false));
                valid[s] = batch[^1] + 1;
            }
        }
        Assert.Equal(6, f16);

        // Rollback to 4 (the handle's length before the next forward), then append [4].
        int[] next = [4];
        f16 = HybridKvLengthBookkeeping.SyncToCommitted(f16, valid, committedLength: 4);
        Assert.Equal(4, f16);
        Assert.All(valid, v => Assert.Equal(4, v));
        foreach (int v in valid)
            Assert.True(HybridKvLengthBookkeeping.IsIncrementalAppend(next, v, forceFull: false));

        f16 = HybridKvLengthBookkeeping.LengthAfterWrite(f16, next);
        Assert.Equal(5, f16);
    }

    /// <summary>A committed length at or above the live lengths (plain decode) changes nothing.</summary>
    [Fact]
    public void SyncToCommitted_WithoutRollback_IsANoOp()
    {
        int[] valid = [7, 5];
        Assert.Equal(7, HybridKvLengthBookkeeping.SyncToCommitted(7, valid, committedLength: 7));
        Assert.Equal(7, HybridKvLengthBookkeeping.SyncToCommitted(7, valid, committedLength: 9));
        Assert.Equal([7, 5], valid);
    }

    /// <summary>Only the slots above the committed length shrink; a lower one keeps its value.</summary>
    [Fact]
    public void SyncToCommitted_ClampsEachSlotIndependently()
    {
        int[] valid = [8, 3, 6];
        Assert.Equal(5, HybridKvLengthBookkeeping.SyncToCommitted(8, valid, committedLength: 5));
        Assert.Equal([5, 3, 5], valid);
    }

    [Fact]
    public void LengthAfterWrite_NeverShrinks()
    {
        Assert.Equal(9, HybridKvLengthBookkeeping.LengthAfterWrite(9, [2, 3]));
        Assert.Equal(4, HybridKvLengthBookkeeping.LengthAfterWrite(0, [3, 1]));
    }

    [Theory]
    [InlineData(new[] { 4 }, 4, false, true)]
    [InlineData(new[] { 4, 5, 6 }, 4, false, true)]
    [InlineData(new[] { 4, 5, 6 }, 4, true, false)]   // test-only force-full escape hatch
    [InlineData(new[] { 5 }, 4, false, false)]        // gap ahead of the valid rows
    [InlineData(new[] { 3 }, 4, false, false)]        // below the valid length (the pre-#478 post-rollback case)
    [InlineData(new[] { 4, 6 }, 4, false, false)]     // non-contiguous
    [InlineData(new int[0], 0, false, false)]
    public void IsIncrementalAppend_Cases(int[] positions, int prevValid, bool force, bool expected)
        => Assert.Equal(expected, HybridKvLengthBookkeeping.IsIncrementalAppend(positions, prevValid, force));
}
