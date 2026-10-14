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
    /// handle back to 4, and the next forward appends position 4. The F16 length must shrink to 4
    /// so the next forward converts only the live range.
    /// </summary>
    [Fact]
    public void RollbackThenAppend_ShrinksCursor()
    {
        int f16 = 0;

        // Prefill + verify.
        foreach (int[] batch in new[] { new[] { 0, 1, 2 }, new[] { 3, 4, 5 } })
        {
            f16 = HybridKvLengthBookkeeping.SyncToCommitted(f16, committedLength: batch[0]);
            f16 = HybridKvLengthBookkeeping.LengthAfterWrite(f16, batch);
        }
        Assert.Equal(6, f16);

        // Rollback to 4 (the handle's length before the next forward), then append [4].
        f16 = HybridKvLengthBookkeeping.SyncToCommitted(f16, committedLength: 4);
        Assert.Equal(4, f16);

        f16 = HybridKvLengthBookkeeping.LengthAfterWrite(f16, [4]);
        Assert.Equal(5, f16);
    }

    /// <summary>A committed length at or above the live length (plain decode) changes nothing.</summary>
    [Fact]
    public void SyncToCommitted_WithoutRollback_IsANoOp()
    {
        Assert.Equal(7, HybridKvLengthBookkeeping.SyncToCommitted(7, committedLength: 7));
        Assert.Equal(7, HybridKvLengthBookkeeping.SyncToCommitted(7, committedLength: 9));
    }

    [Fact]
    public void SyncToCommitted_ClampsBelowF16Length()
    {
        Assert.Equal(5, HybridKvLengthBookkeeping.SyncToCommitted(8, committedLength: 5));
    }

    [Fact]
    public void LengthAfterWrite_NeverShrinks()
    {
        Assert.Equal(9, HybridKvLengthBookkeeping.LengthAfterWrite(9, [2, 3]));
        Assert.Equal(4, HybridKvLengthBookkeeping.LengthAfterWrite(0, [3, 1]));
    }
}
