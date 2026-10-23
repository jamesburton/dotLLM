using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

public sealed unsafe class Lfm2SnapshotsTests
{
    [Theory]
    [InlineData(0)] // Quantized F32 recent window is not the full lossless cache.
    [InlineData(1)] // TurboQuant exposes dequantized F32 scratch, not raw storage.
    [InlineData(2)] // Paged storage has no complete per-layer geometry contract.
    public void PrefixSnapshot_UnsupportedStorageIsNotCaptured(int kind)
    {
        using var fixture = new Lfm2TestFixture(64);
        using var state = fixture.Model.CreateSequenceState();
        using var pool = new KvBlockPool(1, 1, 32, totalBlocks: 4);
        using var cache = CreateUnsupportedCache(kind, pool);
        Populate(cache);
        using var snapshot = fixture.Model.SnapshotSequencePrefix(cache, state, 3);
        Assert.Null(snapshot);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PrefixSnapshot_UnsupportedRestoreLeavesDestinationUnchanged(int kind)
    {
        using var fixture = new Lfm2TestFixture(64);
        using var state = fixture.Model.CreateSequenceState();
        using var source = new SimpleKvCache(1, 1, 32, 16);
        fixture.Model.Forward([1, 2, 3], [0, 1, 2], -1, source, (ISsmState)state!).Dispose();
        using var snapshot = fixture.Model.SnapshotSequencePrefix(source, state, 3);
        Assert.NotNull(snapshot);
        using var pool = new KvBlockPool(1, 1, 32, totalBlocks: 4);
        using var target = CreateUnsupportedCache(kind, pool);
        Populate(target);
        float[] before = ReadKeys(target);
        Assert.Throws<ArgumentException>(() => fixture.Model.RestoreSequencePrefix(snapshot, target, state));
        Assert.Equal(3, target.CurrentLength);
        Assert.Equal(before, ReadKeys(target));
    }

    [Fact]
    public void PrefixSnapshot_RejectsDisposedCachesBeforeCopyOrRollback()
    {
        using var fixture = new Lfm2TestFixture();
        using var state = fixture.Model.CreateSequenceState();
        using var source = new SimpleKvCache(1, 1, 4, 16);
        fixture.Model.Forward([1, 2, 3], [0, 1, 2], -1, source, (ISsmState)state!).Dispose();
        using var snapshot = fixture.Model.SnapshotSequencePrefix(source, state, 3);
        Assert.NotNull(snapshot);
        source.Dispose();
        Assert.Throws<ArgumentException>(() => fixture.Model.SnapshotSequencePrefix(source, state, 3));
        Assert.Throws<ArgumentException>(() => fixture.Model.RestoreSequencePrefix(snapshot, source, state));
        Assert.Equal(3, source.CurrentLength);
    }

    private static IKvCache CreateUnsupportedCache(int kind, KvBlockPool pool) => kind switch
    {
        0 => new QuantizedKvCache(1, 1, 32, 16, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 1),
        1 => new TurboQuantKvCache(1, 1, 32, 16, 3, 1028),
        _ => new PagedKvCache(pool, 1, 32, 16)
    };

    private static void Populate(IKvCache cache)
    {
        float[] rows = Enumerable.Range(0, 96).Select(i => (i + 1) * .01f).ToArray();
        fixed (float* p = rows)
            cache.Update(new TensorRef(3, 32, DType.Float32, -1, (nint)p),
                new TensorRef(3, 32, DType.Float32, -1, (nint)p), [0, 1, 2], 0);
    }

    private static float[] ReadKeys(IKvCache cache)
    {
        var rows = cache.GetKeysRef(0);
        return new ReadOnlySpan<float>((void*)rows.DataPointer, (int)rows.ElementCount).ToArray();
    }

    [Fact]
    public void Checkpoint_RestoresContinuationAfterRejectedHistory()
    {
        using var fixture = new Lfm2TestFixture();
        IModel model = fixture.Model;
        using var kv = new SimpleKvCache(1, 1, 4, 16);
        model.Forward([1, 2, 3], [0, 1, 2], -1, kv).Dispose();
        object? checkpoint = model.CheckpointRecurrentState();
        Assert.NotNull(checkpoint);
        using var owned = Assert.IsAssignableFrom<IDisposable>(checkpoint);
        float[] expected = Forward(model, kv, 4, 3);
        model.Forward([6, 5], [4, 5], -1, kv).Dispose();
        model.RestoreRecurrentState(checkpoint);
        kv.Rollback(3);
        Assert.Equal(expected, Forward(model, kv, 4, 3));
        model.RestoreRecurrentState(checkpoint);
        kv.Rollback(3);
        Assert.Equal(expected, Forward(model, kv, 4, 3));
    }

    [Fact]
    public void PrefixSnapshot_IsIndependentAndRestoresIntoDifferentSequence()
    {
        using var fixture = new Lfm2TestFixture();
        IModel model = fixture.Model;
        using var source = model.CreateSequenceState();
        using var target = model.CreateSequenceState();
        using var sourceKv = new SimpleKvCache(1, 1, 4, 16);
        using var targetKv = new SimpleKvCache(1, 1, 4, 16);
        fixture.Model.Forward([1, 2, 3], [0, 1, 2], -1, sourceKv, (ISsmState)source!).Dispose();
        using var snapshot = model.SnapshotSequencePrefix(sourceKv, source, 3);
        Assert.NotNull(snapshot);
        float[] expected = Forward(model, sourceKv, 4, 3, source);
        fixture.Model.Forward([6, 5], [0, 1], -1, targetKv, (ISsmState)target!).Dispose();
        source!.Reset(); sourceKv.Rollback(0);
        model.RestoreSequencePrefix(snapshot, targetKv, target);
        Assert.Equal(3, targetKv.CurrentLength);
        Assert.Equal(expected, Forward(model, targetKv, 4, 3, target));
        model.RestoreSequencePrefix(snapshot, sourceKv, source);
        Assert.Equal(expected, Forward(model, sourceKv, 4, 3, source));
    }

    [Fact]
    public void StateCopy_RejectsSameElementCountWithDifferentGeometry()
    {
        using var a = new Lfm2ConvStateCache(2, 2, 4);
        using var b = new Lfm2ConvStateCache(2, 4, 2);
        Assert.Throws<ArgumentException>(() => a.CopyFrom(b));
    }

    [Fact]
    public void StateCopy_RejectsDisposedSourceAndDestination()
    {
        using var a = new Lfm2ConvStateCache(2, 2, 4);
        var b = new Lfm2ConvStateCache(2, 2, 4);
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => b.CopyFrom(a));
        Assert.Throws<ObjectDisposedException>(() => a.CopyFrom(b));
    }

    [Fact]
    public void Snapshots_RejectOtherModelAndDisposedSnapshots()
    {
        using var fixture = new Lfm2TestFixture();
        using var other = new Lfm2TestFixture();
        IModel model = fixture.Model;
        object? checkpoint = model.CheckpointRecurrentState();
        Assert.NotNull(checkpoint);
        using var owned = Assert.IsAssignableFrom<IDisposable>(checkpoint);
        Assert.Throws<ArgumentException>(() => ((IModel)other.Model).RestoreRecurrentState(checkpoint));
        owned.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.RestoreRecurrentState(checkpoint));
        using var state = model.CreateSequenceState();
        using var kv = new SimpleKvCache(1, 1, 4, 16);
        fixture.Model.Forward([1], [0], -1, kv, (ISsmState)state!).Dispose();
        using var prefix = model.SnapshotSequencePrefix(kv, state, 1);
        Assert.NotNull(prefix);
        Assert.Throws<ArgumentException>(() => ((IModel)other.Model).RestoreSequencePrefix(prefix, kv, state));
        prefix.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.RestoreSequencePrefix(prefix, kv, state));
    }
    [Fact]
    public void PrefixSnapshot_RejectsInvalidLengthAndGeometryBeforeChangingDestination()
    {
        using var fixture = new Lfm2TestFixture();
        IModel model = fixture.Model;
        using var state = model.CreateSequenceState();
        using var kv = new SimpleKvCache(1, 1, 4, 16);
        fixture.Model.Forward([1, 2, 3], [0, 1, 2], -1, kv, (ISsmState)state!).Dispose();
        Assert.Throws<ArgumentOutOfRangeException>(() => model.SnapshotSequencePrefix(kv, state, 4));
        using var wrongState = new Lfm2ConvStateCache(2, 4, 4); // same total elements; different slots/channels
        Assert.Throws<ArgumentException>(() => model.SnapshotSequencePrefix(kv, wrongState, 3));
        using var snapshot = model.SnapshotSequencePrefix(kv, state, 3);
        Assert.NotNull(snapshot);
        using var target = new SimpleKvCache(1, 1, 8, 16);
        float[] sentinel = [10, 11, 12, 13, 14, 15, 16, 17];
        fixed (float* p = sentinel)
            target.Update(new TensorRef(1, 8, DType.Float32, -1, (nint)p), new TensorRef(1, 8, DType.Float32, -1, (nint)p), [0], 0);
        Assert.Throws<ArgumentException>(() => model.RestoreSequencePrefix(snapshot, target, state));
        Assert.Equal(1, target.CurrentLength);
        Assert.Equal(sentinel, new ReadOnlySpan<float>((void*)target.GetKeysRef(0).DataPointer, 8).ToArray());
    }
    private static float[] Forward(IModel model, SimpleKvCache kv, int token, int pos, IRecurrentSequenceState? state = null)
    {
        using var logits = state is null ? model.Forward([token], [pos], -1, kv)
            : ((Lfm2MoeTransformerModel)model).Forward([token], [pos], -1, kv, (ISsmState)state);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
    }

}
