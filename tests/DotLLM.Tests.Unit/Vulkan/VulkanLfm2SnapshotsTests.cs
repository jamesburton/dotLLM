using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Lfm2Moe;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanLfm2SnapshotsTests
{
    [SkippableFact]
    public void ForwardBatch_InterleavedSequencesMatchIsolatedContinuations()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = new Lfm2TestFixture();
        using var gguf = GgufFile.Open(fixture.Path);
        using var device = VulkanDevice.Create();
        using var model = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);
        IModel api = model;
        using var a = api.CreateSequenceState();
        using var b = api.CreateSequenceState();
        using var ka = model.CreateKvCache(16);
        using var kb = model.CreateKvCache(16);
        model.Forward([1, 2, 3], [0, 1, 2], 0, ka, (ISsmState)a!).Dispose();
        float[] expectedA = Forward(model, ka, 4, 3, a);
        model.Forward([6, 5], [0, 1], 0, kb, (ISsmState)b!).Dispose();
        float[] expectedB = Forward(model, kb, 7, 2, b);
        a!.Reset(); b!.Reset(); ka.Rollback(0); kb.Rollback(0);
        var prefill = api.ForwardBatch([
            new() { TokenIds = new int[] { 1, 2, 3 }, Positions = new int[] { 0, 1, 2 }, KvCache = ka, SsmState = (ISsmState)a },
            new() { TokenIds = new int[] { 6, 5 }, Positions = new int[] { 0, 1 }, KvCache = kb, SsmState = (ISsmState)b }
        ], 0);
        foreach (var row in prefill) row.Dispose();
        // Reverse order at decode to expose any accidental model-owned state reuse.
        var decode = api.ForwardBatch([
            new() { TokenIds = new int[] { 7 }, Positions = new int[] { 2 }, KvCache = kb, SsmState = (ISsmState)b },
            new() { TokenIds = new int[] { 4 }, Positions = new int[] { 3 }, KvCache = ka, SsmState = (ISsmState)a }
        ], 0);
        try
        {
            Assert.Equal(expectedB, Read(decode[0]));
            Assert.Equal(expectedA, Read(decode[1]));
            Assert.True(api.SupportsThreadedSequenceState);
        }
        finally { foreach (var row in decode) row.Dispose(); }
    }

    [SkippableFact]
    public void Snapshots_RejectForeignOwnersGeometryAndDisposedStateBeforeMutation()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = new Lfm2TestFixture();
        using var gguf = GgufFile.Open(fixture.Path);
        using var device = VulkanDevice.Create();
        using var model = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);
        using var other = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);
        IModel api = model;
        using var checkpoint = Assert.IsAssignableFrom<IDisposable>(api.CheckpointRecurrentState());
        Assert.Throws<ArgumentException>(() => ((IModel)other).RestoreRecurrentState(checkpoint));
        checkpoint.Dispose();
        Assert.Throws<ObjectDisposedException>(() => api.RestoreRecurrentState(checkpoint));
        using var source = api.CreateSequenceState();
        using var kv = model.CreateKvCache(16);
        model.Forward([1, 2, 3], [0, 1, 2], 0, kv, (ISsmState)source!).Dispose();
        using var wrong = new VulkanLfm2ConvStateCache(device, 2, 4, 4);
        Assert.Throws<ArgumentException>(() => api.SnapshotSequencePrefix(kv, wrong, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.SnapshotSequencePrefix(kv, source, 4));
        using var snapshot = Assert.IsAssignableFrom<IDisposable>(api.SnapshotSequencePrefix(kv, source, 3));
        Assert.Throws<ArgumentException>(() => ((IModel)other).RestoreSequencePrefix(snapshot, kv, source));
        using var targetState = api.CreateSequenceState();
        using var targetKv = model.CreateKvCache(16);
        model.Forward([6, 5], [0, 1], 0, targetKv, (ISsmState)targetState!).Dispose();
        Assert.Throws<ArgumentException>(() => api.RestoreSequencePrefix(snapshot, targetKv, wrong));
        Assert.Equal(2, targetKv.CurrentLength);
        using var wrongKv = new VulkanNemotronHKvCache(device, [-1, 0, -1], 1, 1, 8, 16);
        Assert.Throws<ArgumentException>(() => api.RestoreSequencePrefix(snapshot, wrongKv, targetState));
        Assert.Equal(0, wrongKv.CurrentLength);
        using var secondDevice = VulkanDevice.Create();
        using var foreignState = new VulkanLfm2ConvStateCache(secondDevice, 2, 2, 8);
        Assert.Throws<ArgumentException>(() => api.RestoreSequencePrefix(snapshot, targetKv, foreignState));
        targetState!.Dispose();
        Assert.Throws<ObjectDisposedException>(() => api.RestoreSequencePrefix(snapshot, targetKv, targetState));
        Assert.Equal(2, targetKv.CurrentLength);
        snapshot.Dispose();
        Assert.Throws<ObjectDisposedException>(() => api.RestoreSequencePrefix(snapshot, kv, source));
    }

    [SkippableFact]
    public void CheckpointAndPrefix_RestoreTheSameContinuation()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = new Lfm2TestFixture();
        using var gguf = GgufFile.Open(fixture.Path);
        using var device = VulkanDevice.Create();
        using var model = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);
        IModel api = model;
        using var initial = Assert.IsAssignableFrom<IDisposable>(api.CheckpointRecurrentState());
        using var kv = model.CreateKvCache(16);
        model.Forward([1, 2, 3], [0, 1, 2], 0, kv).Dispose();
        var checkpoint = api.CheckpointRecurrentState();
        Assert.NotNull(checkpoint);
        using var owned = Assert.IsAssignableFrom<IDisposable>(checkpoint);
        float[] expected = Forward(model, kv, 4, 3);
        model.Forward([6, 5], [4, 5], 0, kv).Dispose();
        api.RestoreRecurrentState(checkpoint); kv.Rollback(3);
        Assert.Equal(expected, Forward(model, kv, 4, 3));

        using var source = api.CreateSequenceState();
        using var target = api.CreateSequenceState();
        using var sourceKv = model.CreateKvCache(16);
        using var targetKv = model.CreateKvCache(16);
        model.Forward([1, 2, 3], [0, 1, 2], 0, sourceKv, (ISsmState)source!).Dispose();
        using var snapshot = api.SnapshotSequencePrefix(sourceKv, source, 3);
        Assert.NotNull(snapshot);
        expected = Forward(model, sourceKv, 4, 3, source);
        model.Forward([6, 5], [0, 1], 0, targetKv, (ISsmState)target!).Dispose();
        source!.Reset(); sourceKv.Rollback(0);
        api.RestoreSequencePrefix(snapshot, targetKv, target);
        Assert.Equal(3, targetKv.CurrentLength);
        Assert.Equal(expected, Forward(model, targetKv, 4, 3, target));
        api.RestoreSequencePrefix(snapshot, sourceKv, source);
        Assert.Equal(expected, Forward(model, sourceKv, 4, 3, source));
    }

    private static float[] Forward(VulkanLfm2MoeTransformerModel model, VulkanNemotronHKvCache kv, int token, int pos, IRecurrentSequenceState? state = null)
    {
        using var logits = state is null ? model.Forward([token], [pos], 0, kv)
            : model.Forward([token], [pos], 0, kv, (ISsmState)state);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
    }

    private static float[] Read(ITensor logits)
        => new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
}
