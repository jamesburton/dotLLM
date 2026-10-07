using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #790: <c>--device</c> must never silently mean CPU. Pins the one parser, the pure resolution (<see cref="DeviceSelector.Plan"/>) and
/// the load orchestration (<see cref="DeviceModelLoader"/>) with an injected loader, so a requested GPU that resolves to CPU fails here
/// without needing any hardware.
/// </summary>
public sealed class DeviceResolutionTests
{
    private const long GiB = 1L << 30;

    private static BackendInfoDto Cpu() => new()
    {
        Name = "cpu", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }],
    };

    private static BackendInfoDto Cuda(long mem = 12 * GiB) => new()
    {
        Name = "cuda", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "gpu0", DeviceString = "gpu:0", TotalMemoryBytes = mem }],
    };

    private static BackendInfoDto NoCuda() => new()
    {
        Name = "cuda", Available = false, DeviceCount = 0, Servable = false, Note = "No CUDA driver / no CUDA-capable GPU detected.",
    };

    private static BackendInfoDto Vulkan() => new()
    {
        Name = "vulkan", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "Vulkan GPU", DeviceString = "vulkan" }],
    };

    private static BackendInfoDto NoVulkan() => new()
    {
        Name = "vulkan", Available = false, DeviceCount = 0, Servable = false, Note = "No Vulkan loader / no Vulkan-capable device detected.",
    };

    private static DeviceListResponse Devices(params BackendInfoDto[] b) => new() { Backends = b };

    private static (IModel Model, Func<int, IKvCache>? Kv) FakeModel() => (null!, null);

    // ---- parser ------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("auto", DeviceKind.Auto, 0, "auto")]
    [InlineData("AUTO", DeviceKind.Auto, 0, "auto")]
    [InlineData("", DeviceKind.Auto, 0, "auto")]
    [InlineData("cpu", DeviceKind.Cpu, 0, "cpu")]
    [InlineData("vulkan", DeviceKind.Vulkan, 0, "vulkan")]
    [InlineData("Vulkan:0", DeviceKind.Vulkan, 0, "vulkan")]
    [InlineData("gpu", DeviceKind.Cuda, 0, "gpu:0")]
    [InlineData("GPU:1", DeviceKind.Cuda, 1, "gpu:1")]
    [InlineData("cuda", DeviceKind.Cuda, 0, "gpu:0")]
    [InlineData("cuda:2", DeviceKind.Cuda, 2, "gpu:2")]
    public void Parser_AcceptsEveryDocumentedForm_AndNormalises(string input, DeviceKind kind, int ordinal, string canonical)
    {
        Assert.True(DeviceSpec.TryParse(input, out var spec, out var error), error);
        Assert.Equal(new DeviceSpec(kind, ordinal), spec);
        Assert.Equal(canonical, spec.Canonical);
    }

    [Theory]
    [InlineData("foo")]
    [InlineData("vulkn")]
    [InlineData("cuda:x")]
    [InlineData("cuda:-1")]
    [InlineData("gpu:")]
    [InlineData("vulkan:1")]
    [InlineData("cpu:0")]
    [InlineData("auto:1")]
    [InlineData("hip")]
    public void Parser_RejectsAnythingElse_ListingTheAcceptedValues(string input)
    {
        Assert.False(DeviceSpec.TryParse(input, out _, out var error));
        Assert.Contains(DeviceSpec.Accepted, error);
        Assert.Throws<ArgumentException>(() => DeviceSpec.Parse(input));
    }

    [Theory]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("cuda:0")]
    [InlineData("GPU:0")]
    public void EveryCudaSpelling_ResolvesToAllLayersOnTheGpu_NotCpu(string device) =>
        // Before #790 only a "gpu"-prefixed string counted as CUDA: "cuda" got 0 layers = a silent CPU load.
        Assert.Equal(42, ServerStartup.ResolveGpuLayers(null, device, 42));

    // ---- pure resolution ---------------------------------------------------------------------------------------

    [Theory]
    [InlineData("vulkan", "vulkan")]
    [InlineData("gpu", "gpu:0")]
    [InlineData("cuda", "gpu:0")]
    [InlineData("gpu:0", "gpu:0")]
    public void ExplicitGpu_ThatTheMachineHas_ResolvesToThatGpu(string requested, string expected)
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse(requested), GiB, Devices(Cpu(), Cuda(), Vulkan()));
        Assert.Null(plan.Error);
        Assert.Equal([expected], plan.Candidates);
    }

    [Theory]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("gpu:1")]      // exists as a string, not on this machine
    public void ExplicitCuda_WithoutThatDevice_IsAnError_NeverCpu(string requested)
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse(requested), GiB, Devices(Cpu(), requested == "gpu:1" ? Cuda() : NoCuda(), Vulkan()));
        Assert.NotNull(plan.Error);
        Assert.Empty(plan.Candidates);              // no candidate list at all - in particular no "cpu"
        Assert.Contains("--device vulkan", plan.Error);   // points at the GPU path that does work
        if (requested != "gpu:1") Assert.Contains("no NVIDIA/CUDA device is available", plan.Error);   // "no CUDA here", not "CUDA load failed"
    }

    [Fact]
    public void ExplicitVulkan_WhenNotServable_IsAnError_NeverCpu()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("vulkan"), GiB, Devices(Cpu(), NoCuda(), NoVulkan()));
        Assert.NotNull(plan.Error);
        Assert.Empty(plan.Candidates);
        Assert.Contains("No Vulkan", plan.Error);
    }

    [Fact]
    public void Auto_WithNoServableGpu_ResolvesToCpuAndAlwaysCarriesAWarning()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("auto"), GiB, Devices(Cpu(), NoCuda(), NoVulkan()));
        Assert.Equal(["cpu"], plan.Candidates);
        Assert.NotNull(plan.CpuWarning);
        Assert.Contains("No CUDA driver", plan.CpuWarning);   // says why
        Assert.Contains("--device cpu", plan.CpuWarning);     // and how to opt in / silence
    }

    [Fact]
    public void Auto_WithAGpu_PrefersItAndDoesNotWarn()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("auto"), GiB, Devices(Cpu(), NoCuda(), Vulkan()));
        Assert.Equal(["vulkan", "cpu"], plan.Candidates);
        Assert.Null(plan.CpuWarning);
    }

    [Fact]
    public void ExplicitCpu_IsCpu_WithoutWarning()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("cpu"), GiB, Devices(Cpu(), Cuda(), Vulkan()));
        Assert.Equal(["cpu"], plan.Candidates);
        Assert.Null(plan.CpuWarning);
    }

    [Fact]
    public void ErrorMessage_SaysWhatWhyAndHowToOptIntoTheCpu_AndLinksTheIssue()
    {
        string msg = DeviceSelector.ExplicitFailureMessage("vulkan", "m.gguf", 3 * GiB, "out of device memory");
        Assert.Contains("--device vulkan was requested for 'm.gguf'", msg);   // what
        Assert.Contains("out of device memory", msg);                         // why
        Assert.Contains("3.0 GiB", msg);                                      // size
        Assert.Contains("--device cpu", msg);                                 // opt-in
        Assert.Contains(DeviceSelector.PolicyIssueUrl, msg);                  // link
    }

    // ---- orchestration: a requested GPU must never resolve to CPU ---------------------------------------------------

    [Theory]
    [InlineData("vulkan")]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("gpu:0")]
    public void RequestedGpu_ThatLoads_IsReportedAsThatGpu_NeverCpu(string requested)
    {
        var tried = new List<string>();
        var result = DeviceModelLoader.LoadWith(null!, null!, requested, null, default, GiB, "m", _ => { }, _ => { },
            Devices(Cpu(), Cuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); });
        Assert.NotEqual("cpu", result.ResolvedDevice);
        Assert.DoesNotContain("cpu", tried);
        Assert.Null(result.Warning);
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("gpu")]
    public void RequestedGpu_ThatFailsToLoad_Throws_AndNeverTriesTheCpu(string requested)
    {
        var tried = new List<string>();
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, requested, null, default, GiB, "m",
            _ => { }, _ => { }, Devices(Cpu(), Cuda(), Vulkan()),
            (d, _) => { tried.Add(d); throw new InvalidOperationException("VK_ERROR_OUT_OF_DEVICE_MEMORY"); }));
        Assert.DoesNotContain("cpu", tried);
        Assert.Contains("VK_ERROR_OUT_OF_DEVICE_MEMORY", ex.Message);
        Assert.Contains("--device cpu", ex.Message);
    }

    [Fact]
    public void RequestedCuda_OnAMachineWithoutCuda_NeverReachesAnyLoader()
    {
        var tried = new List<string>();
        Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, "cuda", null, default, GiB, "m",
            _ => { }, _ => { }, Devices(Cpu(), NoCuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); }));
        Assert.Empty(tried);
    }

    [Fact]
    public void Auto_FallsThroughFailedGpus_ToCpu_WithAProminentWarningNamingTheReasons()
    {
        var tried = new List<string>();
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), Cuda(), Vulkan()),
            (d, _) => { tried.Add(d); return d == "cpu" ? FakeModel() : throw new InvalidOperationException($"{d} boom"); });
        Assert.Equal(["gpu:0", "vulkan", "cpu"], tried);
        Assert.Equal("cpu", result.ResolvedDevice);
        Assert.NotNull(result.Warning);
        Assert.Contains("gpu:0 boom", result.Warning);
        Assert.Contains("vulkan boom", result.Warning);
    }

    [Fact]
    public void Auto_WithNoGpuAtAll_StillWarnsWhenItEndsOnCpu()
    {
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), NoCuda(), NoVulkan()), (d, _) => FakeModel());
        Assert.Equal("cpu", result.ResolvedDevice);
        Assert.NotNull(result.Warning);
    }

    [Fact]
    public void Auto_ThatLoadsOnAGpu_DoesNotWarn()
    {
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), NoCuda(), Vulkan()), (d, _) => FakeModel());
        Assert.Equal("vulkan", result.ResolvedDevice);
        Assert.Null(result.Warning);
        Assert.True(result.IsVulkan);
    }

    [Fact]
    public void ExplicitGpuLayersZero_UnderAuto_IsTheUsersChoiceOfCpu_NoWarning()
    {
        var tried = new List<string>();
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", 0, default, GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), Cuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); });
        Assert.Equal(["cpu"], tried);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void UnrecognisedDevice_ThrowsInsteadOfLoadingOnCpu()
    {
        var tried = new List<string>();
        Assert.Throws<ArgumentException>(() => DeviceModelLoader.LoadWith(null!, null!, "vulkn", null, default, GiB, "m", _ => { }, _ => { },
            Devices(Cpu(), Cuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); }));
        Assert.Empty(tried);
    }

    [Fact]
    public void EveryStringCandidatesCanEmit_RoundTripsToANonCpuPlan_ForGpus()
    {
        // The candidate strings feed LoadExact, which feeds ResolveGpuLayers: a GPU candidate that parsed to 0 layers would load the CPU.
        foreach (string c in DeviceSelector.Candidates(GiB, Devices(Cpu(), Cuda(), Vulkan())))
        {
            Assert.True(DeviceSpec.TryParse(c, out var spec, out _), c);
            Assert.NotEqual(DeviceKind.Auto, spec.Kind);
            if (c != "cpu")
            {
                Assert.True(spec.IsGpu, c);
                if (spec.Kind == DeviceKind.Cuda)
                    Assert.Equal(10, ServerStartup.ResolveGpuLayers(null, c, 10));
            }
        }
    }
}
