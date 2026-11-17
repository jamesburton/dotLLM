using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1851: <c>--gpu-layers auto</c> and the automatic split under <c>--device auto</c>. The user decisions pinned here:
/// the split happens only under <c>auto</c>; an explicit <c>--device gpu</c> that does not fit still refuses, now with a
/// <c>--gpu-layers auto</c> hint; the dGPU tier is tried before Vulkan and the CPU.
/// </summary>
public sealed class AutoGpuLayersResolutionTests
{
    private const long GiB = 1L << 30;

    private static BackendInfoDto Cpu() => new()
    {
        Name = "cpu", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }],
    };

    // 4 GiB free of 12: an 8 GiB model does not fit whole (FitsCuda needs weights + 15 % of total free).
    private static BackendInfoDto SmallFreeCuda() => new()
    {
        Name = "cuda", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "gpu0", DeviceString = "gpu:0", TotalMemoryBytes = 12 * GiB, FreeMemoryBytes = 4 * GiB }],
    };

    private static BackendInfoDto Vulkan() => new()
    {
        Name = "vulkan", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "Vulkan GPU", DeviceString = "vulkan" }],
    };

    private static DeviceListResponse Devices(params BackendInfoDto[] b) => new() { Backends = b };

    private static (IModel Model, Func<int, IKvCache>? Kv) FakeModel() => (null!, null);

    [Theory]
    [InlineData("auto", GpuOffloadPlanner.AutoGpuLayers)]
    [InlineData("AUTO", GpuOffloadPlanner.AutoGpuLayers)]
    [InlineData("all", -1)]
    [InlineData("-1", -1)]
    [InlineData("0", 0)]
    [InlineData("28", 28)]
    public void Parser_AcceptsAutoAllAndCounts(string value, int expected)
    {
        Assert.True(GpuOffloadPlanner.TryParseGpuLayers(value, out int? layers, out string? error));
        Assert.Null(error);
        Assert.Equal(expected, layers);
    }

    [Theory]
    [InlineData("-2")]
    [InlineData("-5")]
    [InlineData("half")]
    public void Parser_RejectsEverythingElse(string value)
    {
        Assert.False(GpuOffloadPlanner.TryParseGpuLayers(value, out _, out string? error));
        Assert.Contains("auto", error);
    }

    [Fact]
    public void ServeArgs_AcceptAuto()
    {
        var options = ServerOptions.Parse(["--model", "m.gguf", "--gpu-layers", "auto"]);
        Assert.Equal(GpuOffloadPlanner.AutoGpuLayers, options.GpuLayers);
    }

    [Fact]
    public void ResolveGpuLayers_RefusesAuto_SoItIsNeverReadAsAllLayers()
        => Assert.Throws<ArgumentException>(() => ServerStartup.ResolveGpuLayers(GpuOffloadPlanner.AutoGpuLayers, "gpu:0", 40));

    [Fact]
    public void Auto_ModelThatDoesNotFitCuda_TriesTheCudaSplitFirst_BeforeVulkan()
    {
        var tried = new List<(string Device, int? Layers)>();
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, 8 * GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, l) => { tried.Add((d, l)); return FakeModel(); },
            autoSplitBlocker: _ => null);
        Assert.Equal([("gpu:0", GpuOffloadPlanner.AutoGpuLayers)], tried);
        Assert.Equal("gpu:0", result.ResolvedDevice);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void Auto_SplitThatFails_FallsThroughToVulkanThenCpu_WithLayersUnset()
    {
        var tried = new List<(string Device, int? Layers)>();
        var result = DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, 8 * GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()),
            (d, l) => { tried.Add((d, l)); return d == "cpu" ? FakeModel() : throw new DeviceMemoryAdmissionException($"{d} does not fit"); },
            autoSplitBlocker: _ => null);
        Assert.Equal([("gpu:0", GpuOffloadPlanner.AutoGpuLayers), ("vulkan", null), ("cpu", null)], tried);
        Assert.Contains("gpu:0 does not fit", result.Warning);
    }

    [Fact]
    public void Auto_ModelTheSplitCannotRun_SkipsCudaAsBefore()
    {
        var tried = new List<string>();
        DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, 8 * GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); },
            autoSplitBlocker: _ => "Granite runs only on the FP32 dense forward");
        Assert.Equal(["vulkan"], tried);
    }

    [Fact]
    public void Auto_ModelThatFitsCuda_LoadsWholeAsBefore_NotAsASplit()
    {
        var tried = new List<(string Device, int? Layers)>();
        DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, 1 * GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, l) => { tried.Add((d, l)); return FakeModel(); },
            autoSplitBlocker: _ => null);
        Assert.Equal([("gpu:0", (int?)null)], tried);
    }

    [Fact]
    public void ExplicitGpu_ThatDoesNotFit_StillRefuses_WithTheAutoHint_AndNeverSplitsBehindTheUsersBack()
    {
        var tried = new List<string>();
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, "gpu", null, default, 8 * GiB,
            "m.gguf", _ => { }, _ => { }, Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, _) => { tried.Add(d); return FakeModel(); },
            autoSplitBlocker: _ => null));
        Assert.Empty(tried);
        Assert.Contains("--gpu-layers auto", ex.Message);
    }

    [Fact]
    public void ExplicitGpu_WithGpuLayersAuto_IsNotRefusedOnTheWholeModel_AndReachesTheLoaderAsAuto()
    {
        var tried = new List<(string Device, int? Layers)>();
        DeviceModelLoader.LoadWith(null!, null!, "gpu", GpuOffloadPlanner.AutoGpuLayers, default, 8 * GiB, "m.gguf", _ => { }, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, l) => { tried.Add((d, l)); return FakeModel(); });
        Assert.Equal([("gpu:0", GpuOffloadPlanner.AutoGpuLayers)], tried);
    }

    [Fact]
    public void ExplicitGpu_ThatDoesNotFit_AndCannotSplit_GetsNoAutoHint()
    {
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, "gpu", null, default, 8 * GiB,
            "m.gguf", _ => { }, _ => { }, Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, _) => FakeModel(),
            autoSplitBlocker: _ => "Granite runs only on the FP32 dense forward"));
        Assert.DoesNotContain("--gpu-layers auto", ex.Message);
    }

    [Fact]
    public void Auto_SkippingCudaForABlockedModel_SaysWhy()
    {
        var logs = new List<string>();
        DeviceModelLoader.LoadWith(null!, null!, "auto", null, default, 8 * GiB, "m.gguf", logs.Add, _ => { },
            Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, _) => FakeModel(),
            autoSplitBlocker: _ => "Granite runs only on the FP32 dense forward");
        Assert.Contains(logs, l => l.Contains("cannot be split") && l.Contains("Granite"));
    }

    [Fact]
    public void ExplicitGpu_WithAllLayers_IsStillJudgedOnTheWholeModel()
    {
        Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, null!, "gpu", -1, default, 8 * GiB, "m.gguf",
            _ => { }, _ => { }, Devices(Cpu(), SmallFreeCuda(), Vulkan()), (d, _) => FakeModel()));
    }
}
