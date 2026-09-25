using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Lifecycle and sizing tests for <see cref="Mamba3ForwardScratch"/>. Focused
/// on the invariants the production call path relies on: power-of-two growth,
/// idempotent Dispose, and that a second EnsureCapacity at the same-or-smaller
/// seqLen does not reallocate.
/// </summary>
public class Mamba3ForwardScratchTests
{
    // Shape close to the ib-ssm 370M per-layer block — d_inner = H · P = 32 · 64.
    private const int DInner = 2048;
    private const int NHead = 32;
    private const int DState = 128;
    private const int NumBcHeads = 1;
    private const int NumRopeAngles = 32;

    [Fact]
    public void FromDimensions_LazyInit_ZeroCapacityUntilEnsure()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);

        Assert.Equal(0, scratch.Capacity);
        Assert.Equal(0, scratch.AllocatedBytes);
    }

    [Fact]
    public void EnsureCapacity_RoundsUpToPowerOfTwo()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);

        scratch.EnsureCapacity(5);

        Assert.Equal(8, scratch.Capacity);
        Assert.True(scratch.AllocatedBytes > 0);
    }

    [Fact]
    public void EnsureCapacity_AtOrBelowExisting_DoesNotShrinkOrReallocate()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        scratch.EnsureCapacity(17); // rounds to 32

        int startCap = scratch.Capacity;
        long startBytes = scratch.AllocatedBytes;
        Assert.Equal(32, startCap);

        scratch.EnsureCapacity(32); // same cap
        Assert.Equal(startCap, scratch.Capacity);
        Assert.Equal(startBytes, scratch.AllocatedBytes);

        scratch.EnsureCapacity(10); // smaller — still fits, no-op.
        Assert.Equal(startCap, scratch.Capacity);
        Assert.Equal(startBytes, scratch.AllocatedBytes);
    }

    [Fact]
    public void EnsureCapacity_BeyondExisting_GrowsToNextPowerOfTwo()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        scratch.EnsureCapacity(8);
        Assert.Equal(8, scratch.Capacity);

        scratch.EnsureCapacity(9); // rounds to 16
        Assert.Equal(16, scratch.Capacity);

        scratch.EnsureCapacity(17); // rounds to 32
        Assert.Equal(32, scratch.Capacity);
    }

    [Fact]
    public void Accessors_ReturnSpansOfExpectedLength()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        int T = 7; // will round to 8

        scratch.EnsureCapacity(T);

        Assert.Equal(T * DInner, scratch.X(T).Length);
        Assert.Equal(T * DInner, scratch.Z(T).Length);
        Assert.Equal(T * DInner, scratch.YScan(T).Length);
        Assert.Equal(T * NHead, scratch.Dt(T).Length);
        Assert.Equal(T * NHead, scratch.Adt(T).Length);
        Assert.Equal(T * NHead, scratch.Trap(T).Length);
        Assert.Equal(T * NHead, scratch.Gamma(T).Length);
        Assert.Equal(T * NHead, scratch.Scale(T).Length);
        Assert.Equal(T * NHead, scratch.QkPreDot(T).Length);
        Assert.Equal(T * NumRopeAngles, scratch.AnglesRaw(T).Length);
        // B/C SISO width: 1·H·N.
        Assert.Equal(T * NHead * DState, scratch.B(T).Length);
        Assert.Equal(T * NHead * DState, scratch.C(T).Length);
        // Proj: full in_proj row — 2·d_inner + 2·N·G·R + 3·H + S.
        Assert.Equal(T * scratch.InProjWidth, scratch.Proj(T).Length);
    }

    [Fact]
    public void FromDimensions_Mimo_SizesBcForRank()
    {
        const int MimoRank = 4;
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: MimoRank);
        scratch.EnsureCapacity(3);

        // MIMO B/C: T · R · H · N.
        Assert.Equal(3 * MimoRank * NHead * DState, scratch.B(3).Length);
        Assert.Equal(3 * MimoRank * NHead * DState, scratch.C(3).Length);
    }

    [Fact]
    public void Dispose_ReleasesBuffers_IsIdempotent()
    {
        var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        scratch.EnsureCapacity(4);
        Assert.True(scratch.AllocatedBytes > 0);

        scratch.Dispose();
        Assert.Equal(0, scratch.Capacity);

        // Second Dispose is a no-op — does not throw.
        scratch.Dispose();
    }

    [Fact]
    public void Accessor_AfterDispose_Throws()
    {
        var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        scratch.EnsureCapacity(4);
        scratch.Dispose();

        Assert.Throws<ObjectDisposedException>(() => scratch.X(1));
        Assert.Throws<ObjectDisposedException>(() => scratch.EnsureCapacity(8));
    }

    [Fact]
    public void Accessor_BeyondCapacity_Throws()
    {
        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);
        scratch.EnsureCapacity(4); // -> 4

        // Asking for more than capacity without EnsureCapacity first is a bug.
        Assert.Throws<InvalidOperationException>(() => scratch.X(5));
    }
}
