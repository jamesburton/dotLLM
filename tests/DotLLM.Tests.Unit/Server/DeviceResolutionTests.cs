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
    [InlineData("Vulkan:0", 0, "vulkan:0")]
    [InlineData("vulkan:1", 1, "vulkan:1")]
    [InlineData("vulkan:12", 12, "vulkan:12")]
    public void Parser_AcceptsVulkanIndex_AndKeepsItInTheCanonicalString(string input, int index, string canonical)
    {
        // #1704. Plain "vulkan" is the process-default device, not index 0, so vulkan:0 must stay distinguishable from it.
        Assert.True(DeviceSpec.TryParse(input, out var spec, out var error), error);
        Assert.Equal(DeviceKind.Vulkan, spec.Kind);
        Assert.Equal(index, spec.VulkanIndex);
        Assert.Equal(canonical, spec.Canonical);
        Assert.True(spec.IsGpu);
    }

    [Fact]
    public void PlainVulkan_HasNoIndex_SoItMeansTheDefaultDevice()
    {
        var spec = DeviceSpec.Parse("vulkan");
        Assert.Null(spec.VulkanIndex);
        Assert.Equal("vulkan", spec.Canonical);
        Assert.Null(new DeviceSpec(DeviceKind.Cuda, 1).VulkanIndex);   // a CUDA ordinal is never a Vulkan index
    }

    [Theory]
    [InlineData("foo")]
    [InlineData("vulkn")]
    [InlineData("cuda:x")]
    [InlineData("cuda:-1")]
    [InlineData("gpu:")]
    [InlineData("vulkan:")]
    [InlineData("vulkan:-1")]
    [InlineData("vulkan:x")]
    [InlineData("vulkan:1.5")]
    [InlineData("vulkan:+1")]
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

    // ---- Vulkan device index (#1704) ---------------------------------------------------------------------------

    private static BackendInfoDto TwoVulkan() => new()
    {
        Name = "vulkan", Available = true, DeviceCount = 2, Servable = true,
        Devices =
        [
            new DeviceInfoDto { Index = 0, Name = "NVIDIA eGPU", DeviceString = "vulkan:0", TotalMemoryBytes = 12 * GiB },
            new DeviceInfoDto { Index = 1, Name = "Radeon iGPU", DeviceString = "vulkan:1", TotalMemoryBytes = 64 * GiB, IsDefault = true },
        ],
    };

    [Fact]
    public void ExplicitVulkanIndex_IsPlannedOnThatExactDevice()
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse("vulkan:1"), GiB, Devices(Cpu(), NoCuda(), TwoVulkan()));
        Assert.Null(plan.Error);
        Assert.Equal(["vulkan:1"], plan.Candidates);
        plan = DeviceSelector.Plan(DeviceSpec.Parse("vulkan:0"), GiB, Devices(Cpu(), NoCuda(), TwoVulkan()));
        Assert.Equal(["vulkan:0"], plan.Candidates);   // not collapsed to plain "vulkan" (= the default, which is device 1 here)
    }

    [Fact]
    public void PlainVulkan_StaysPlain_SoItKeepsTheDefaultDeviceBehaviour() =>
        Assert.Equal(["vulkan"], DeviceSelector.Plan(DeviceSpec.Parse("vulkan"), GiB, Devices(Cpu(), NoCuda(), TwoVulkan())).Candidates);

    [Theory]
    [InlineData("vulkan:2")]
    [InlineData("vulkan:9")]
    public void OutOfRangeVulkanIndex_IsAClearError_NeverAFallback(string requested)
    {
        var plan = DeviceSelector.Plan(DeviceSpec.Parse(requested), GiB, Devices(Cpu(), NoCuda(), TwoVulkan()));
        Assert.Empty(plan.Candidates);
        Assert.Contains("does not exist", plan.Error);
        Assert.Contains("2 Vulkan device(s)", plan.Error);
        Assert.Contains("vulkan:0..vulkan:1", plan.Error);
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, requested, null, default, GiB, "m",
            _ => { }, _ => { }, Devices(Cpu(), NoCuda(), TwoVulkan()), (d, _) => throw new InvalidOperationException("must not load")));
        Assert.Contains("--device cpu", ex.Message);
    }

    [Fact]
    public void LoadOnVulkanIndex_RoutesThatIndexToTheLoader_AndReportsIt()
    {
        // The discriminating routing test: a loader that dropped the index would be handed "vulkan" (or vulkan:0) here.
        var tried = new List<string>();
        var result = DeviceModelLoader.LoadWith(null!, null!, "vulkan:1", null, default, GiB, "m", _ => { }, _ => { },
            Devices(Cpu(), NoCuda(), TwoVulkan()), (d, _) => { tried.Add(d); return FakeModel(); });
        Assert.Equal(["vulkan:1"], tried);
        Assert.Equal("vulkan:1", result.ResolvedDevice);
        Assert.True(result.IsVulkan);
    }

    [Fact]
    public void AutoPrefersTheDefaultVulkanDevice_NotJustTheFirst()
    {
        Assert.Equal("vulkan:1", DeviceSelector.Candidates(GiB, Devices(Cpu(), NoCuda(), TwoVulkan()))[0]);
        Assert.Equal("gpu:0", DeviceSelector.Candidates(GiB, Devices(Cpu(), Cuda(), TwoVulkan()))[0]);   // CUDA still first
    }

    [Fact]
    public void DescribeVulkan_ListsEveryPhysicalDevice_WithIndexNameMemoryAndDefault()
    {
        var info = DotLLM.Server.Endpoints.DeviceEndpoint.DescribeVulkan(
        [
            new DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo(0, "NVIDIA RTX", 0x10DE, 2, 12L * GiB, false),
            new DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo(1, "AMD Radeon 8060S", 0x1002, 1, 64L * GiB, true),
            new DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo(2, "llvmpipe", 0x10005, 4, 0, false),
        ], shaders: true);
        Assert.True(info.Servable);
        Assert.Equal(3, info.DeviceCount);
        Assert.Equal(["vulkan:0", "vulkan:1", "vulkan:2"], info.Devices.Select(d => d.DeviceString));
        Assert.Equal(["NVIDIA RTX", "AMD Radeon 8060S", "llvmpipe"], info.Devices.Select(d => d.Name));
        Assert.Equal([12L * GiB, 64L * GiB, (long?)null], info.Devices.Select(d => d.TotalMemoryBytes));   // 0 is "unknown", not a figure
        Assert.Equal([null, true, null], info.Devices.Select(d => d.IsDefault));
        // Without the shader blobs the devices exist but are not offered as load targets.
        var noShaders = DotLLM.Server.Endpoints.DeviceEndpoint.DescribeVulkan([new DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo(0, "x", 0, 1, GiB, true)], shaders: false);
        Assert.False(noShaders.Servable);
        Assert.Empty(noShaders.Devices);
    }
}
