using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1888: <see cref="ServerStartup.LoadModel"/> (exactly what <c>dotllm serve</c> runs) either builds a real TurboQuant cache or
/// refuses. Before this, <c>--cache-type-k tq4</c> fell through every branch (<c>IsQuantized</c> excludes TurboQuant) and the paged
/// default served it with F16 blocks.
/// </summary>
public sealed class ServerTurboQuantWiringTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public ServerTurboQuantWiringTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    private ServerOptions Options(string? k, string? v, bool paged, string device = "cpu") => new()
    {
        Model = _tiny.GgufPath,
        ModelId = "tiny-1888",
        Device = device,
        UsePaged = paged,
        Warmup = WarmupOptions.Disabled,
        CacheTypeK = k,
        CacheTypeV = v,
    };

    [Fact]
    public void Tq4_NoPaged_BuildsTheInPlaceCache()
    {
        // #1892: the MSE stage is read in place (codes + norms), no longer the dequant-scratch TurboQuantKvCache.
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, Options("tq4", "tq4", paged: false));
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        var tq = Assert.IsType<TurboQuantInPlaceKvCache>(cache);
        Assert.Equal(KvCacheDType.TurboQuant, tq.KeyDType);
        Assert.Equal(0, tq.WindowCapacity);
    }

    [Theory]
    [InlineData("tq2")]
    [InlineData("tq3")]
    [InlineData("tq8")]
    public void TqBits_ReachTheInPlaceCache(string flag)
    {
        // run / chat / bench used to drop the bit width at parse time: the cache must carry the width that was asked for.
        int bits = int.Parse(flag[2..]);
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, Options(flag, flag, paged: false));
        Assert.Equal(bits, state.KvCacheConfig.TurboQuantBits);
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        var layout = new DotLLM.Core.Attention.TurboQuantLayout(_tiny.Config.HeadDim, _tiny.Config.NumKvHeads, bits, DotLLM.Core.Attention.TurboQuantLayout.DefaultNorm);
        long expected = (long)DotLLM.Core.Attention.KvGeometry.SlotCount(_tiny.Config) * 2 * (layout.CodeBytesPerLayer(32) + layout.NormBytesPerLayer(32));
        Assert.Equal(expected, Assert.IsType<TurboQuantInPlaceKvCache>(cache).AllocatedBytes);
    }

    [Fact]
    public void Tq4q_NoPaged_StaysOnTheScratchQjlCache_UntilStep5()
    {
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, Options("tq4q", "tq4q", paged: false));
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        Assert.True(Assert.IsType<TurboQuantKvCache>(cache).UseQjl);
    }

    [Fact]
    public void Tq4_Paged_Refuses()
    {
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(_tiny.GgufPath, Options("tq4", "tq4", paged: true)));
        Assert.Contains("--no-paged", e.Message);
    }

    [Fact]
    public void Tq4_VOnly_Refuses()
    {
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(_tiny.GgufPath, Options(null, "tq4", paged: false)));
        Assert.Contains("step 6", e.Message);
    }

    [Fact]
    public void Tq4_KOnly_Refuses_WithBothFlagsHint()
    {
        // K-only used to mean "both"; K and V are now judged independently, so say how to ask for both.
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(_tiny.GgufPath, Options("tq4", null, paged: false)));
        Assert.Contains("--cache-type-v", e.Message);
    }

    [Fact]
    public void TqBitsDiffer_Refuses()
    {
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(_tiny.GgufPath, Options("tq3", "tq4", paged: false)));
        Assert.Contains("step 6", e.Message);
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void Tq4_GpuDevice_IsNoLongerRefusedBeforeLoad(string device)
        => Assert.Null(ServerStartup.PreLoadTurboQuantRefusal(Options("tq4", "tq4", paged: false, device), _tiny.Config));

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void Tq4q_GpuDevice_RefusesBeforeLoad_CitingStep5(string device)
    {
        string? reason = ServerStartup.PreLoadTurboQuantRefusal(Options("tq4q", "tq4q", paged: false, device), _tiny.Config);
        Assert.NotNull(reason);
        Assert.Contains("#1894", reason);
        Assert.Contains(device, reason);
    }

    [Fact]
    public void CacheWindow_WithTq_Refuses()
    {
        // ServerOptions has no --cache-window; run and chat do, and refuse before anything loads (see CliTurboQuantRefusalTests). Here the
        // shared parse carries the window into the same resolver the server uses.
        var kv = TurboQuantKv.Parse("tq4", "tq4", cacheWindow: 64, out _);
        Assert.Equal(64, kv.MixedPrecisionWindowSize);
        string? reason = TurboQuantKv.PreLoadRefusal(kv, null, KvCommand.Serve, KvBackendKind.Cpu, paged: false, _tiny.Config, hasDraftModel: false);
        Assert.Contains("--cache-window 64", reason);
        // ...and after the load, so the CPU window guard in ExecuteQuantizedCore is never what a user meets.
        Assert.Contains("--cache-window 64", TurboQuantKv.PostLoadRefusal(kv, null, KvCommand.Serve, paged: false, _tiny.Config, _tiny.Model, false, false));
    }

    [Fact]
    public void Tq4_Paged_RefusesBeforeLoad()
    {
        // The paged default is the commonest accident; refuse it before the weights load, not after.
        Assert.Contains("--no-paged", ServerStartup.PreLoadTurboQuantRefusal(Options("tq4", "tq4", paged: true), _tiny.Config));
    }

    [Fact]
    public void Tq4_DeviceAuto_IsDecidedAfterLoad()
    {
        // The backend is unknown before an auto device loads; the post-load pass decides (here it lands on the CPU, which is real).
        Assert.Null(ServerStartup.PreLoadTurboQuantRefusal(Options("tq4", "tq4", paged: false, "auto"), _tiny.Config));
    }
}

/// <summary>#1888: the post-load rules (host model type, an attaching MTP head) that the pre-load pass cannot see.</summary>
public sealed class ServerTurboQuantPostLoadTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public ServerTurboQuantPostLoadTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    private static ServerOptions Tq4(bool mtp = true) => new()
    {
        Model = "x", ModelId = "x", Device = "cpu", UsePaged = false, Warmup = WarmupOptions.Disabled,
        CacheTypeK = "tq4", CacheTypeV = "tq4", MtpEnabled = mtp,
    };

    /// <summary>A CPU model that is not the plain <c>TransformerModel</c>, optionally with an MTP head.</summary>
    private sealed class OtherCpuModel(ModelConfig config, bool mtp) : IModel
    {
        public ModelConfig Config { get; } = config;
        public long ComputeMemoryBytes => 0;
        public bool SupportsMtp => mtp;
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, DotLLM.Core.Attention.IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    [Fact]
    public void PlainTransformer_IsReal()
        => Assert.Null(ServerStartup.PostLoadTurboQuantRefusal(Tq4(), _tiny.Config, _tiny.Model));

    [Fact]
    public void OtherCpuHost_Refuses_CitingTheHostGap()
    {
        string? reason = ServerStartup.PostLoadTurboQuantRefusal(Tq4(), _tiny.Config, new OtherCpuModel(_tiny.Config, mtp: false));
        Assert.Contains("#1900", reason);
    }

    [Fact]
    public void AttachingMtpHead_Refuses_WithNoMtpHint()
    {
        string? reason = ServerStartup.PostLoadTurboQuantRefusal(Tq4(), _tiny.Config, new OtherCpuModel(_tiny.Config, mtp: true));
        Assert.Contains("--no-mtp", reason);
    }

    [Fact]
    public void MtpDisabled_DoesNotRefuseForMtp()
    {
        string? reason = ServerStartup.PostLoadTurboQuantRefusal(Tq4(mtp: false), _tiny.Config, new OtherCpuModel(_tiny.Config, mtp: true));
        Assert.DoesNotContain("--no-mtp", reason);
    }
}

/// <summary>
/// #1888 Review Focus 1, inverted by #1892: with in-place TurboQuant on the GPUs, a tq request that <c>--device auto</c> lands on a GPU must
/// build that GPU's TurboQuant cache - not refuse (step 1), not serve f16, and not quietly move to the CPU. Whatever backend auto lands on, the
/// cache is that backend's in-place type: <c>VulkanKvCache</c> in TurboQuant storage on Vulkan, <c>CudaTurboQuantInPlaceKvCache</c> on CUDA,
/// <c>TurboQuantInPlaceKvCache</c> on the CPU. Needs a GPU, so it runs under the GPU lock (Strix: Vulkan; framework: CUDA).
/// </summary>
[Trait("Category", "GPU")]
public sealed class ServerTurboQuantAutoDeviceTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public ServerTurboQuantAutoDeviceTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    [Fact]
    public void Tq4_DeviceAuto_BuildsTheInPlaceCacheOfWhicheverBackendItLandsOn()
    {
        var options = new ServerOptions
        {
            Model = _tiny.GgufPath, ModelId = "tiny-1892-auto", Device = "auto", UsePaged = false, Warmup = WarmupOptions.Disabled,
            CacheTypeK = "tq4", CacheTypeV = "tq4",
        };
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, options);   // no refusal: the GPU runs it
        var backend = TurboQuantKvSupport.BackendOf(state.Model!);
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        switch (backend)
        {
            case KvBackendKind.Vulkan:
                var vk = Assert.IsType<DotLLM.Vulkan.VulkanKvCache>(cache);
                Assert.Equal(DotLLM.Vulkan.VulkanKvStorage.Tq, vk.Storage);
                break;
            case KvBackendKind.Cuda:
                Assert.IsType<DotLLM.Cuda.CudaTurboQuantInPlaceKvCache>(cache);
                break;
            default:
                Assert.Equal(KvBackendKind.Cpu, backend);
                Assert.IsType<TurboQuantInPlaceKvCache>(cache);
                break;
        }
        Console.WriteLine($"auto landed on {backend}: {cache.GetType().Name}");
    }
}
