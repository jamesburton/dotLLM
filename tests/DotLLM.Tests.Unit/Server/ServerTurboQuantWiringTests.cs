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
    public void OtherCpuHost_Refuses()
    {
        string? reason = ServerStartup.PostLoadTurboQuantRefusal(Tq4(), _tiny.Config, new OtherCpuModel(_tiny.Config, mtp: false));
        Assert.Contains("CPU model", reason);
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
/// #1888 Review Focus 1, <b>step 1 only</b>: GPUs have no in-place TurboQuant kernels until #1887 step 2, so while that holds a tq request
/// that <c>--device auto</c> lands on a GPU must be refused - not served f16 as before, and not quietly moved to the CPU.
/// <b>Step 2 inverts this test</b>: auto on a GPU must then build the GPU TurboQuant cache. Auto loads each candidate as an explicit device, so the pre-load pass refuses the GPU candidate before its weights load and the refusal
/// reaches the user (measured on gfx1151: "trying vulkan:0", then the refusal, no CPU attempt). The post-load backend rule is a backstop
/// this test cannot isolate. Needs a GPU, so it runs under the GPU lock.
/// </summary>
[Trait("Category", "GPU")]
public sealed class ServerTurboQuantAutoDeviceTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public ServerTurboQuantAutoDeviceTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    [Fact]
    public void Tq4_DeviceAuto_RunsOnlyWhereTurboQuantIsReal()
    {
        var options = new ServerOptions
        {
            Model = _tiny.GgufPath, ModelId = "tiny-1888-auto", Device = "auto", UsePaged = false, Warmup = WarmupOptions.Disabled,
            CacheTypeK = "tq4", CacheTypeV = "tq4",
        };
        ServerState state;
        try { state = ServerStartup.LoadModel(_tiny.GgufPath, options); }
        catch (InvalidServerOptionException e)
        {
            // Landed on a GPU: refused, naming the backend.
            Assert.Matches("(?i)vulkan|cuda|hip", e.Message);
            return;
        }
        using (state)
        {
            // Landed on the CPU: the real path, on the CPU model.
            Assert.Equal(KvBackendKind.Cpu, TurboQuantKvSupport.BackendOf(state.Model!));
            using var cache = state.KvCacheFactory!(_tiny.Config, 32);
            Assert.IsType<TurboQuantKvCache>(cache);
        }
    }
}
