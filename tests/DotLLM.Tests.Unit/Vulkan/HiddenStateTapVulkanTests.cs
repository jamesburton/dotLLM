using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using DotLLM.Tests.Unit.Diagnostics;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>Vulkan model of one synthetic family for the #1920 tap suite.</summary>
internal sealed class VulkanHiddenStateTapFixture : HiddenStateTapFixture
{
    private readonly string _dir;
    private readonly GgufFile _gguf;
    private readonly ModelConfig _config;
    private readonly string _family;
    private readonly Func<int, IKvCache> _kvFactory;
    private readonly VulkanDevice _device;

    public VulkanHiddenStateTapFixture(string family, bool withMtp = false)
    {
        _family = family;
        _dir = Path.Combine(Path.GetTempPath(), $"dotllm-taps-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "m.gguf");
        switch (family)
        {
            case HiddenStateTapBackendSuite.Llama:
                File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
                Prompt = [3, 9, 14, 5, 21, 8, 30, 2];
                break;
            case HiddenStateTapBackendSuite.HybridDense:
                SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: withMtp, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            case HiddenStateTapBackendSuite.MoeHybrid:
                SyntheticQwen35MoeGguf.Write(path, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family));
        }
        _gguf = GgufFile.Open(path);
        _config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        (Model, _kvFactory) = VulkanModelLoader.CreateOnDeviceFromGguf(_gguf, _config, physicalIndex: null);
        _device = Model switch
        {
            VulkanTransformerModel m => m.Device,
            VulkanQwen3HybridDenseTransformerModel m => m.Device,
            VulkanQwen3MoeHybridTransformerModel m => m.Device,
            _ => throw new InvalidOperationException(Model.GetType().Name),
        };
    }

    public override IModel Model { get; }
    public override int NumLayers => _config.NumLayers;
    public override int Hidden => _config.HiddenSize;
    public override int Vocab => _config.VocabSize;
    public override int[] Prompt { get; }
    public override bool ExpectsPackedBatchFallback => _family != HiddenStateTapBackendSuite.MoeHybrid;

    public override IKvCache NewKv(int maxLen = 96) => _kvFactory(maxLen);

    /// <summary>Merge base d1275eef, 16 untapped decode forwards: managed bytes on this thread.</summary>
    public override long? BaselineAllocBytesPer16Decodes => 1872;

    public override long[]? DeviceCounters() =>
    [
        DotLLM.Vulkan.Interop.ProfileCounters.Dispatches, DotLLM.Vulkan.Interop.ProfileCounters.Copies,
        DotLLM.Vulkan.Interop.ProfileCounters.Barriers, DotLLM.Vulkan.Interop.ProfileCounters.DescriptorWrites,
        DotLLM.Vulkan.Interop.ProfileCounters.DescriptorAllocs, DotLLM.Vulkan.Interop.ProfileCounters.Submits,
    ];

    /// <summary>
    /// Merge base d1275eef (dispatches, copies, barriers, descriptor writes, descriptor allocs, queue submits), per 16 untapped decodes.
    /// Queue submits equal fence waits (every forward submit is awaited by a fence or timeline).
    /// </summary>
    public override long[]? BaselineDeviceCountersPer16Decodes => _family switch
    {
        HiddenStateTapBackendSuite.Llama => [740, 32, 824, 4, 4, 16],
        HiddenStateTapBackendSuite.HybridDense => [716, 208, 824, 2, 2, 16],
        _ => [1036, 176, 1128, 2, 2, 16],
    };

    public override IKvCache? NewWindowedKv(int maxLen)
        => _family != HiddenStateTapBackendSuite.Llama ? null
            : new VulkanKvCache(_device, KvGeometry.FromConfig(_config with { SlidingWindowSize = 8, SlidingWindowPattern = 2 }, compactSlidingWindows: true), maxLen);

    private readonly List<IModel> _extra = [];

    public override float[] OutputNormWeight => ReadF32Row(_gguf, "output_norm.weight", 0, _config.HiddenSize);
    public override float NormEps => _config.NormEpsilon;
    public override float[] EmbeddingRow(int token) => ReadF32Row(_gguf, "token_embd.weight", token, _config.HiddenSize);

    public override DotLLM.Core.Diagnostics.IHiddenStateTapHost? NewSlidingWindowConfigHost()
    {
        if (_family != HiddenStateTapBackendSuite.Llama) return null;
        var (m, _) = VulkanModelLoader.CreateOnDeviceFromGguf(_gguf, _config with { SlidingWindowSize = 8, SlidingWindowPattern = 2 }, physicalIndex: null);
        _extra.Add(m);
        return (DotLLM.Core.Diagnostics.IHiddenStateTapHost)m;
    }

    public override float[] KvBytes(IKvCache kv)
    {
        var all = new List<float>();
        switch (kv)
        {
            case VulkanKvCache c:
                for (int l = 0; l < _config.NumLayers; l++)
                {
                    int n = c.StagedLayerElementCount(l);
                    if (n == 0) continue;
                    var k = new float[n]; var v = new float[n];
                    c.ReadRows(l, 0, c.CurrentLength, k, v);
                    all.AddRange(k); all.AddRange(v);
                }
                break;
            case VulkanNemotronHKvCache c:
                long rowBytes = c.BytesPerPosition / (2L * c.AttentionLayerCount);
                int floats = (int)((c.CurrentLength * rowBytes + 3) / 4);
                for (int l = 0; l < _config.NumLayers; l++)
                {
                    VulkanDevice.Buffer kb, vb;
                    try { kb = c.GetKeysBuffer(l); vb = c.GetValuesBuffer(l); }
                    catch (Exception) { continue; }
                    if (floats == 0) continue;
                    var k = new float[floats]; var v = new float[floats];
                    _device.Download(kb, k); _device.Download(vb, v);
                    all.AddRange(k); all.AddRange(v);
                }
                break;
            default:
                throw new NotSupportedException(kv.GetType().Name);
        }
        return [.. all];
    }

    public override void Dispose()
    {
        foreach (var m in _extra) m.Dispose();
        Model.Dispose();
        _gguf.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }
}

/// <summary>The #1920 tap suite on the Vulkan backend (all three families).</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class HiddenStateTapVulkanTests : HiddenStateTapBackendSuite
{
    protected override HiddenStateTapFixture Open(string family)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        return new VulkanHiddenStateTapFixture(family);
    }

    protected override HiddenStateTapFixture? OpenHybridDenseWithMtp()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        return new VulkanHiddenStateTapFixture(HybridDense, withMtp: true);
    }

    private static readonly DotLLM.Core.Diagnostics.HookPoint[] All4 =
    [
        DotLLM.Core.Diagnostics.HookPoint.PostEmbedding, DotLLM.Core.Diagnostics.HookPoint.PostLayer,
        DotLLM.Core.Diagnostics.HookPoint.PreFinalNorm, DotLLM.Core.Diagnostics.HookPoint.PreLmHead,
    ];

    private static List<(DotLLM.Core.Diagnostics.HookPoint P, int L, int R, float[] D)> Collect(HiddenStateTapFixture fx, bool immediate)
    {
        VulkanHiddenStateTaps.ForceImmediateForTest = immediate;
        try
        {
            var seen = new List<(DotLLM.Core.Diagnostics.HookPoint, int, int, float[])>();
            using var h = fx.Host.RegisterTap(
                new DotLLM.Core.Diagnostics.HiddenStateTapSubscription(null, All4, DotLLM.Core.Diagnostics.TapRows.All),
                s => seen.Add((s.Point, s.LayerIndex, s.RowIndex, s.Data.ToArray())));
            using var kv = fx.NewKv();
            using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
            using (fx.Model.Forward([fx.Prompt[0]], [fx.Prompt.Length], -1, kv)) { }
            return seen;
        }
        finally { VulkanHiddenStateTaps.ForceImmediateForTest = false; }
    }

    /// <summary>The deferred plan and the site-by-site (immediate) plan deliver identical samples in the same order.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void DeferredAndImmediatePlans_DeliverIdenticalSamples(string family)
    {
        using var a = Open(family);
        using var b = Open(family);
        var deferred = Collect(a, immediate: false);
        var immediate = Collect(b, immediate: true);
        Assert.NotEmpty(deferred);
        Assert.Equal(deferred.Count, immediate.Count);
        for (int i = 0; i < deferred.Count; i++)
        {
            Assert.Equal((deferred[i].P, deferred[i].L, deferred[i].R), (immediate[i].P, immediate[i].L, immediate[i].R));
            Assert.True(deferred[i].D.AsSpan().SequenceEqual(immediate[i].D), $"sample {i} differs");
        }
    }

    /// <summary>
    /// With a tap attached the MoE-hybrid single-token decode must leave the fused-decode path (it has no layer boundaries) and say so in the
    /// counter; the untapped control proves the fused path is the default for this fixture, so the bypass is observable.
    /// </summary>
    [SkippableFact]
    public void MoeHybrid_FusedDecode_IsBypassed_WhenATapIsAttached()
    {
        using var fx = (VulkanHiddenStateTapFixture)Open(MoeHybrid);
        var model = (VulkanQwen3MoeHybridTransformerModel)fx.Model;
        using var kv = fx.NewKv();
        int[] pp = Enumerable.Range(0, fx.Prompt.Length).ToArray();
        using (model.Forward(fx.Prompt, pp, -1, kv)) { }
        long fusedBefore = model.FusedDecodeForwards;
        using (model.Forward([fx.Prompt[0]], [fx.Prompt.Length], -1, kv)) { }
        Skip.If(model.FusedDecodeForwards == fusedBefore, "the synthetic MoE fixture does not take the fused decode path on this device");

        int rows = 0;
        using var h = fx.Host.RegisterTap(new DotLLM.Core.Diagnostics.HiddenStateTapSubscription(null, All4), _ => rows++);
        DotLLM.Core.Diagnostics.HiddenStateTapRegistry.PathCounters.Reset();
        long fused = model.FusedDecodeForwards;
        using (model.Forward([fx.Prompt[1]], [fx.Prompt.Length + 1], -1, kv)) { }
        Assert.Equal(fused, model.FusedDecodeForwards);
        Assert.Equal(1, DotLLM.Core.Diagnostics.HiddenStateTapRegistry.PathCounters.VulkanFusedDecodeBypassed);
        Assert.Equal(3 + fx.NumLayers, rows);
    }
}

