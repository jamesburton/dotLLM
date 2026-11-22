using DotLLM.Core.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

/// <summary>
/// Core tap API (#1920) against a fake host: filtering by point and layer, row selection, dispose, validation, abort and
/// counters. No model involved, so these also pin the contract Vulkan / CUDA hosts must honour.
/// </summary>
[Collection("HiddenStateTaps")]
public sealed class HiddenStateTapRegistryTests
{
    private const int Layers = 4, Hidden = 8;

    private static float[] Rows(int rows)
    {
        var a = new float[rows * Hidden];
        for (int i = 0; i < a.Length; i++) a[i] = i + 1;
        return a;
    }

    private static readonly int[] Positions = [10, 11, 12];

    private static bool Fire(HiddenStateTapRegistry reg, HookPoint p, int layer, int rows = 3)
    {
        var snap = reg.Current;
        if (snap is null || !snap.Wants(p, layer)) return true;
        return snap.Dispatch(p, layer, Rows(rows), rows, 0, Hidden, Positions, 7, 3, out _);
    }

    [Fact]
    public void NoTap_CurrentIsNull_AndWantsNothing()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        Assert.Null(reg.Current);
    }

    [Fact]
    public void Filters_ByPointAndLayer()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        var seen = new List<(HookPoint, int)>();
        using var h = reg.Register(new HiddenStateTapSubscription([1, 3], [HookPoint.PostLayer, HookPoint.PreLmHead]),
            s => seen.Add((s.Point, s.LayerIndex)), null);
        for (int l = 0; l < Layers; l++) Fire(reg, HookPoint.PostLayer, l);
        Fire(reg, HookPoint.PostEmbedding, -1);
        Fire(reg, HookPoint.PreFinalNorm, -1);
        Fire(reg, HookPoint.PreLmHead, -1);
        Assert.Equal([(HookPoint.PostLayer, 1), (HookPoint.PostLayer, 3), (HookPoint.PreLmHead, -1)], seen);
    }

    [Fact]
    public void NullLayers_MeansEveryLayer()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        int n = 0;
        using var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]), _ => n++, null);
        for (int l = 0; l < Layers; l++) Fire(reg, HookPoint.PostLayer, l);
        Assert.Equal(Layers, n);
    }

    [Fact]
    public void Rows_Last_DeliversOneSample_All_DeliversEveryRowWithPositions()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        var last = new List<HiddenStateSample>();
        var all = new List<(int Row, int Pos, float First)>();
        using var a = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]),
            s => last.Add(s with { Data = s.Data.ToArray() }), null);
        using var b = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer], TapRows.All),
            s => all.Add((s.RowIndex, s.TokenPosition, s.Data.Span[0])), null);
        Fire(reg, HookPoint.PostLayer, 2);

        var one = Assert.Single(last);
        Assert.Equal(2, one.RowIndex);
        Assert.Equal(12, one.TokenPosition);
        Assert.Equal(7, one.SequenceId);
        Assert.Equal(3, one.Step);
        Assert.Equal(Hidden, one.Data.Length);
        Assert.Equal(2 * Hidden + 1, one.Data.Span[0]);
        Assert.Equal([(0, 10, 1f), (1, 11, (float)Hidden + 1), (2, 12, 2f * Hidden + 1)], all);
    }

    [Fact]
    public void RowOffset_ShiftsRowIndexAndPosition()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        HiddenStateSample got = default;
        using var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PreLmHead]), s => got = s, null);
        // One row passed, which is row 2 of a 3-row forward.
        Assert.True(reg.Current!.Dispatch(HookPoint.PreLmHead, -1, Rows(1), 1, 2, Hidden, Positions, 0, 0, out _));
        Assert.Equal(2, got.RowIndex);
        Assert.Equal(12, got.TokenPosition);
    }

    [Fact]
    public void Dispose_Unregisters_AndIsIdempotent()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        int n = 0;
        var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => n++, null);
        Fire(reg, HookPoint.PostEmbedding, -1);
        Assert.Equal(1, n);
        h.Dispose();
        h.Dispose();
        Assert.Null(reg.Current);
        Fire(reg, HookPoint.PostEmbedding, -1);
        Assert.Equal(1, n);
    }

    [Fact]
    public void Dispose_OfOneTap_KeepsTheOther()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        int a = 0, b = 0;
        var ha = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => a++, null);
        using var hb = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => b++, null);
        ha.Dispose();
        Fire(reg, HookPoint.PostEmbedding, -1);
        Assert.Equal((0, 1), (a, b));
    }

    [Theory]
    [InlineData(HookPoint.PreAttention)]
    [InlineData(HookPoint.PostAttention)]
    [InlineData(HookPoint.PreFfn)]
    [InlineData(HookPoint.PostFfn)]
    [InlineData(HookPoint.PostLmHead)]
    public void UnsupportedPoints_ThrowNotSupported_AtRegistration(HookPoint point)
    {
        var reg = new HiddenStateTapRegistry(Layers);
        Assert.Throws<NotSupportedException>(() =>
            reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer, point]), _ => { }, null));
        Assert.Null(reg.Current);
    }

    [Fact]
    public void LayerIndices_AreValidated()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        Assert.Throws<ArgumentException>(() =>
            reg.Register(new HiddenStateTapSubscription([Layers], [HookPoint.PostLayer]), _ => { }, null));
        Assert.Throws<ArgumentException>(() =>
            reg.Register(new HiddenStateTapSubscription([-1], [HookPoint.PostLayer]), _ => { }, null));
    }

    [Fact]
    public void EmptyPoints_Throw()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        Assert.Throws<ArgumentException>(() => reg.Register(new HiddenStateTapSubscription(null, []), _ => { }, null));
    }

    [Fact]
    public void Abortable_AbortForward_StopsDispatch_AndReportsSite()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        int after = 0;
        using var a = reg.Register(new HiddenStateTapSubscription([2], [HookPoint.PostLayer]), null, _ => HookResult.AbortForward);
        using var b = reg.Register(new HiddenStateTapSubscription([2], [HookPoint.PostLayer]), _ => after++, null);
        Assert.True(reg.Current!.HasAbortable);
        Assert.False(reg.Current.Dispatch(HookPoint.PostLayer, 2, Rows(3), 3, 0, Hidden, Positions, 0, 0, out var abort));
        Assert.Equal(new ForwardAbortInfo(2, HookPoint.PostLayer), abort);
        Assert.Equal(0, after);
    }

    [Fact]
    public void Abortable_Continue_Proceeds()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        using var a = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]), null, _ => HookResult.Continue);
        Assert.True(Fire(reg, HookPoint.PostLayer, 0));
    }

    [Fact]
    public void Abortable_Replace_IsRejected()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        using var a = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]), null,
            _ => HookResult.Replace(new float[Hidden]));
        Assert.Throws<InvalidOperationException>(() => Fire(reg, HookPoint.PostLayer, 0));
    }

    [Fact]
    public void ReadOnlyTaps_DoNotMarkSnapshotAbortable()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        using var a = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]), _ => { }, null);
        Assert.False(reg.Current!.HasAbortable);
    }

    [Fact]
    public void BufferProvider_BacksTheSampleData()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        var mine = new float[Hidden + 3];
        int asked = -1;
        using var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding], TapRows.Last,
            n => { asked = n; return mine; }), s => Assert.Equal(mine.AsMemory(0, Hidden).ToArray(), s.Data.ToArray()), null);
        Fire(reg, HookPoint.PostEmbedding, -1);
        Assert.Equal(Hidden, asked);
        Assert.Equal(2f * Hidden + 1, mine[0]);   // last of 3 rows
    }

    [Fact]
    public void BufferProvider_TooSmall_Throws()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        using var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding], TapRows.Last,
            _ => new float[1]), _ => { }, null);
        Assert.Throws<InvalidOperationException>(() => Fire(reg, HookPoint.PostEmbedding, -1));
    }

    [Fact]
    public void Samples_AreCopies_TheSourceIsNeverWritten()
    {
        var reg = new HiddenStateTapRegistry(Layers);
        var src = Rows(3);
        var before = (float[])src.Clone();
        using var h = reg.Register(new HiddenStateTapSubscription(null, [HookPoint.PostLayer], TapRows.All), s =>
        {
            // The sample owns its buffer: scribbling on it must not reach the residual.
            MemoryMarshalWrite(s.Data);
        }, null);
        reg.Current!.Dispatch(HookPoint.PostLayer, 0, src, 3, 0, Hidden, Positions, 0, 0, out _);
        Assert.Equal(before, src);
    }

    private static void MemoryMarshalWrite(ReadOnlyMemory<float> data)
        => System.Runtime.InteropServices.MemoryMarshal.AsMemory(data).Span.Fill(-1f);

    [Fact]
    public void PreFinalNorm_IsAppendedAtTheEndOfTheEnum()
    {
        Assert.Equal(8, (int)HookPoint.PreFinalNorm);
        Assert.Equal(7, (int)HookPoint.PostLmHead);
        Assert.Equal(6, (int)HookPoint.PreLmHead);
    }

    [Fact]
    public void PathCounters_CountAndReset()
    {
        HiddenStateTapRegistry.PathCounters.Reset();
        HiddenStateTapRegistry.PathCounters.AddTapForward();
        HiddenStateTapRegistry.PathCounters.AddPackedBatchFallback();
        HiddenStateTapRegistry.PathCounters.AddAbortedForward();
        HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
        HiddenStateTapRegistry.PathCounters.AddCudaGraphBypassed();
        HiddenStateTapRegistry.PathCounters.AddVulkanFusedDecodeBypassed();
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.TapForwards);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.PackedBatchFallbacks);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.AbortedForwards);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.AbortCheckpoints);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.CudaGraphBypassed);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.VulkanFusedDecodeBypassed);
        HiddenStateTapRegistry.PathCounters.Reset();
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.TapForwards);
    }
}

[CollectionDefinition("HiddenStateTaps", DisableParallelization = true)]
public sealed class HiddenStateTapsCollection;
