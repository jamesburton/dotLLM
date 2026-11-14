using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Hip;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tests.Unit.Hip;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

[Collection("HipGpu")]
public sealed class HipDeviceResolutionTests
{
    private const long GiB = 1L << 30;

    private static BackendInfoDto Hip(Action<int>? validate = null) => DeviceEndpoint.DescribeHip(
        () => true, () => 2,
        i => new DeviceInfoDto { Index = i, Name = $"AMD {i}", TotalMemoryBytes = 16 * GiB },
        validate ?? (_ => { }));

    private static DeviceListResponse Devices(BackendInfoDto hip) => new() { Backends = [hip] };

    [Fact]
    public void RuntimeAbsence_DoesNotEnumerateOrOfferHip()
    {
        var backend = DeviceEndpoint.DescribeHip(() => false,
            () => throw new InvalidOperationException("must not enumerate"),
            _ => throw new InvalidOperationException("must not describe"),
            _ => throw new InvalidOperationException("must not initialize"));
        Assert.False(backend.Available);
        Assert.False(backend.Servable);
        Assert.Empty(backend.Devices);
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("hip"), GiB, Devices(backend));
        Assert.Empty(plan.Candidates);
        Assert.Contains("unavailable", plan.Error);
    }

    [Theory]
    [InlineData("dense_f16.co is missing")]
    [InlineData("hipErrorNoBinaryForGpu: gfx1036")]
    [InlineData("DirectStorage DLLs were not found")]
    [InlineData("Windows x64 is required")]
    public void RuntimePresence_IsNotInferenceReadiness(string reason)
    {
        var backend = Hip(i => { if (i == 1) throw new NotSupportedException(reason); });
        Assert.True(backend.Available);
        Assert.True(backend.Servable);
        Assert.Equal(2, backend.DeviceCount);
        Assert.Equal("hip:0", backend.Devices[0].DeviceString);
        Assert.True(backend.Devices[0].Servable);
        Assert.False(backend.Devices[1].Servable);
        Assert.Null(backend.Devices[1].DeviceString);
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("hip:1"), GiB, Devices(backend));
        Assert.Empty(plan.Candidates);
        Assert.Contains(reason, plan.Error);
    }

    [Fact]
    public void ExplicitOrdinal_IsPreserved_AndUnknownOrdinalFails()
    {
        var devices = Devices(Hip());
        Assert.Equal(["hip:0"], DeviceSelector.Plan(DeviceSpec.Parse("hip"), GiB, devices).Candidates);
        Assert.Equal(["hip:1"], DeviceSelector.Plan(DeviceSpec.Parse("HIP:1"), GiB, devices).Candidates);
        var missing = DeviceSelector.Plan(DeviceSpec.Parse("hip:2"), GiB, devices);
        Assert.Empty(missing.Candidates);
        Assert.Contains("device 2 does not exist", missing.Error);
    }

    [Fact]
    public void Auto_DoesNotOptIntoExperimentalHip()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("auto"), GiB, Devices(Hip()));
        Assert.Equal(["cpu"], plan.Candidates);
        Assert.Contains("select explicitly", plan.CpuWarning);
    }

    [Fact]
    public void WeightAdmission_RejectsModelsLargerThanDevice_EvenWithPartialOffloadFlag()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("hip"), 17 * GiB, Devices(Hip()), fullOffload: false);
        Assert.Empty(plan.Candidates);
        Assert.Contains("17.0 GiB", plan.Error);
        Assert.Contains("no partial offload", plan.Error);
    }

    [Fact]
    public void HipLoad_KeepsDeviceAndFactory_WithoutCpuFallback()
    {
        var tried = new List<string>();
        Func<int, IKvCache> cacheFactory = _ => null!;
        var result = DeviceModelLoader.LoadWith(null!, null!, "hip:1", null, default, GiB, "test.gguf",
            _ => { }, _ => { }, Devices(Hip()), (device, _) =>
            {
                tried.Add(device);
                return (null!, cacheFactory);
            });
        Assert.Equal(["hip:1"], tried);
        Assert.Equal("hip:1", result.ResolvedDevice);
        Assert.Same(cacheFactory, result.KvCacheFactory);
        Assert.Null(result.Warning);
    }

    [Theory]
    [InlineData("out of HIP device memory")]
    [InlineData("DirectStorage read failed")]
    [InlineData("dense_f16.co was removed")]
    public void ExplicitLoadFailure_NeverAttemptsCpu(string reason)
    {
        var tried = new List<string>();
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(
            null!, null!, "hip:1", null, default, GiB, "test.gguf", _ => { }, _ => { }, Devices(Hip()),
            (device, _) => { tried.Add(device); throw new InvalidOperationException(reason); }));
        Assert.Equal(["hip:1"], tried);
        Assert.Contains(reason, ex.Message);
        Assert.Contains("--device cpu", ex.Message);
    }

    [Theory]
    [InlineData(null, null, 0, true)]
    [InlineData("f16", "F16", 0, true)]
    [InlineData("fp16", null, 0, true)]
    [InlineData("f32", null, 0, false)]
    [InlineData("q8_0", "f16", 0, true)]
    [InlineData("f16", "q8_0", 0, true)]
    [InlineData("q8_0", "Q8_0", 0, true)]
    [InlineData("f16", "q4_0", 0, false)]
    [InlineData("tq4", null, 0, false)]
    [InlineData(null, null, 16, false)]
    public void CacheOptions_AllowF16OrQ8_NoMixedPrecisionWindow(string? key, string? value, int window, bool valid)
    {
        using var environment = new HipQ8Environment("1");
        Assert.Equal(valid, DeviceModelLoader.HipKvOptionError("hip:1", key, value, window) is null);
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("false", false)]
    [InlineData(" 1 ", false)]
    [InlineData("1", true)]
    public void Q8Kv_RequiresExplicitAcknowledgementForEitherPlane(string? optIn, bool allowed)
    {
        using var environment = new HipQ8Environment(optIn);
        Assert.Null(DeviceModelLoader.HipKvOptionError("hip", null, null));
        Assert.Null(DeviceModelLoader.HipKvOptionError("hip", "f16", "fp16"));
        Assert.Null(DeviceModelLoader.HipKvOptionError("cpu", "q8_0", "q8_0"));
        Assert.Null(HipPagedKvCache.ExperimentalQ8Error(KvCacheDType.F32, KvCacheDType.F32));
        foreach (var (key, value) in new[] { ("q8_0", "f16"), ("f16", "q8_0"), (" Q8_0 ", "q8_0") })
        {
            string? error = DeviceModelLoader.HipKvOptionError("hip:0", key, value);
            Assert.Equal(allowed, error is null);
            if (!allowed)
            {
                Assert.Contains("precision loss", error, StringComparison.Ordinal);
                Assert.Contains("DOTLLM_HIP_ALLOW_Q8_KV=1", error, StringComparison.Ordinal);
            }
        }
    }

    [Theory]
    [InlineData("q8_0", null)]
    [InlineData(null, "q8_0")]
    public void Server_RejectsUnacknowledgedQ8BeforeOpeningTheModel(string? key, string? value)
    {
        using var environment = new HipQ8Environment(null);
        var error = Assert.Throws<InvalidServerOptionException>(() =>
        {
            using var rejected = ServerStartup.LoadModel("does-not-exist.gguf",
                new ServerOptions { Model = "test", Device = "hip:0", CacheTypeK = key, CacheTypeV = value });
        });
        Assert.Contains("DOTLLM_HIP_ALLOW_Q8_KV=1", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void PartialOffload_FailsBeforeAnyNativeLoad(int layers)
    {
        var config = new ModelConfig { Architecture = Architecture.Llama, VocabSize = 32, HiddenSize = 32,
            IntermediateSize = 64, NumLayers = 2, NumAttentionHeads = 2, NumKvHeads = 1, HeadDim = 16, MaxSequenceLength = 32 };
        var ex = Assert.Throws<NotSupportedException>(() => DeviceModelLoader.LoadExact(
            null!, config, "hip:0", layers, default, _ => { }, _ => { }));
        Assert.Contains("partial offload", ex.Message);
    }

    [Fact]
    public void Server_RejectsUnsupportedHipCacheBeforeOpeningTheModel()
    {
        var ex = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel("does-not-exist.gguf",
            new ServerOptions { Model = "test", Device = "hip:0", CacheTypeV = "q4_0" }));
        Assert.Contains("only F16", ex.Message);
    }
}
