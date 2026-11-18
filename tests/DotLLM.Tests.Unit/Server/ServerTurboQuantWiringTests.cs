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
    public void Tq4_NoPaged_BuildsTurboQuantKvCache()
    {
        using var state = ServerStartup.LoadModel(_tiny.GgufPath, Options("tq4", "tq4", paged: false));
        using var cache = state.KvCacheFactory!(_tiny.Config, 32);
        var tq = Assert.IsType<TurboQuantKvCache>(cache);
        Assert.Equal(4, tq.Bits);
        Assert.False(tq.UseQjl);
    }

    [Fact]
    public void Tq4q_NoPaged_BuildsQjlCache()
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
    public void Tq4_GpuDevice_RefusesBeforeLoad(string device)
    {
        string? reason = ServerStartup.PreLoadTurboQuantRefusal(Options("tq4", "tq4", paged: false, device), _tiny.Config);
        Assert.NotNull(reason);
        Assert.Contains(device, reason);
    }

    [Fact]
    public void Tq4_DeviceAuto_IsDecidedAfterLoad()
    {
        // The backend is unknown before an auto device loads; the post-load pass decides (here it lands on the CPU, which is real).
        Assert.Null(ServerStartup.PreLoadTurboQuantRefusal(Options("tq4", "tq4", paged: false, "auto"), _tiny.Config));
    }
}
