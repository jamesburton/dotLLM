using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1088: <see cref="IMtpState.MarkPrefix"/> / <see cref="IMtpState.TryResumeAt"/> on the CPU head state. The engine
/// resumes a state only with its own mark, so the refusals below are what keep a foreign or stale mark from resuming.
/// </summary>
public sealed class CpuMtpStatePrefixMarkTests
{
    private const int Hidden = 4, KvHeads = 1, HeadDim = 2;

    /// <summary>A state that has absorbed <paramref name="length"/> positions, with captured row <c>i</c> filled with <c>i + 1</c>.</summary>
    private static CpuMtpState Absorbed(int length, int maxSteps = 64)
    {
        var st = new CpuMtpState(Hidden, KvHeads, HeadDim, maxSteps);
        var rows = new float[length * Hidden];
        for (int i = 0; i < length; i++)
            rows.AsSpan(i * Hidden, Hidden).Fill(i + 1);
        st.SetCapturedRows(rows, length);
        st.BeginAbsorb(0, length);
        st.EndAbsorb(length);
        st.SeedFromCapturedRow(length - 1);   // what an absorb leaves: the carry is the last absorbed row
        return st;
    }

    private static float[] PairingRow(CpuMtpState st)
    {
        var dest = new float[Hidden];
        st.CopyAbsorbPairingRows(1, dest);
        return dest;
    }

    [Fact]
    public void Resume_RollsBackToTheMark_AndRestoresTheCarryTheNextAbsorbPairsWith()
    {
        using var st = Absorbed(5);
        object mark = st.MarkPrefix()!;
        // A later request absorbs and decodes past the mark, moving the carry.
        st.SetCapturedRows(new float[3 * Hidden].Select((_, i) => 100f + i).ToArray(), 3);
        st.BeginAbsorb(5, 3);
        st.EndAbsorb(8);
        st.SeedFromCapturedRow(2);

        Assert.True(st.TryResumeAt(mark, maxSequenceLength: 64));

        Assert.Equal(5, st.CurrentLength);
        Assert.Equal([5f, 5f, 5f, 5f], PairingRow(st));
        Assert.Equal([5f, 5f, 5f, 5f], st.PendingHidden.ToArray());
    }

    [Fact]
    public void Resume_RaisesTheCeiling_ToAGrownTrunkKv()
    {
        using var st = Absorbed(5, maxSteps: 16);
        object mark = st.MarkPrefix()!;

        Assert.True(st.TryResumeAt(mark, maxSequenceLength: 40));

        Assert.Equal(40, st.MaxSteps);
        st.BeginAbsorb(5, 30);   // past the old ceiling of 16
    }

    [Fact]
    public void Resume_NeverLowersTheCeiling()
    {
        using var st = Absorbed(5, maxSteps: 64);
        Assert.True(st.TryResumeAt(st.MarkPrefix()!, maxSequenceLength: 10));
        Assert.Equal(64, st.MaxSteps);
    }

    [Fact]
    public void AnotherStatesMark_IsRefused_AndChangesNothing()
    {
        using var st = Absorbed(5);
        using var other = Absorbed(3);
        object foreign = other.MarkPrefix()!;

        Assert.False(st.TryResumeAt(foreign, maxSequenceLength: 64));
        Assert.Equal(5, st.CurrentLength);
        Assert.Equal([5f, 5f, 5f, 5f], PairingRow(st));
    }

    [Fact]
    public void MarkBeyondTheCommittedRows_IsRefused()
    {
        // Rows the head rolled back past are no longer the prefix's.
        using var st = Absorbed(5);
        object mark = st.MarkPrefix()!;
        st.Rollback(3);

        Assert.False(st.TryResumeAt(mark, maxSequenceLength: 64));
        Assert.Equal(3, st.CurrentLength);
    }

    [Fact]
    public void CeilingBelowTheMark_IsRefused()
    {
        using var st = Absorbed(5);
        Assert.False(st.TryResumeAt(st.MarkPrefix()!, maxSequenceLength: 4));
    }

    [Fact]
    public void MarkIsACopy_SoLaterAbsorbsDoNotMoveIt()
    {
        using var st = Absorbed(5);
        var mark = (MtpPrefixMark)st.MarkPrefix()!;
        st.SetCapturedRows(Enumerable.Repeat(9f, Hidden).ToArray(), 1);
        st.SeedFromCapturedRow(0);

        Assert.Equal(5, mark.Length);
        Assert.Equal([5f, 5f, 5f, 5f], mark.CarryHidden);
    }
}
