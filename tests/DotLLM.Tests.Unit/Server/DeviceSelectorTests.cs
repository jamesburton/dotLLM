using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>Issue #722: <c>--device auto</c> candidate order - CUDA if present and the model fits, then Vulkan, then CPU.</summary>
public sealed class DeviceSelectorTests
{
    private const long GiB = 1L << 30;

    private static BackendInfoDto Cuda(params long[] memory) => new()
    {
        Name = "cuda", Available = true, DeviceCount = memory.Length, Servable = true,
        Devices = memory.Select((m, i) => new DeviceInfoDto { Index = i, Name = $"gpu{i}", DeviceString = $"gpu:{i}", TotalMemoryBytes = m }).ToArray(),
    };

    private static BackendInfoDto Vulkan(bool servable = true) => new()
    {
        Name = "vulkan", Available = true, DeviceCount = 1, Servable = servable,
        Devices = servable ? [new DeviceInfoDto { Index = 0, Name = "Vulkan GPU", DeviceString = "vulkan" }] : [],
    };

    private static BackendInfoDto Cpu() => new()
    {
        Name = "cpu", Available = true, DeviceCount = 1, Servable = true,
        Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }],
    };

    private static DeviceListResponse Devices(params BackendInfoDto[] b) => new() { Backends = b };

    [Fact]
    public void CudaThatFits_IsFirst_ThenVulkan_ThenCpu() =>
        Assert.Equal(["gpu:0", "vulkan", "cpu"], DeviceSelector.Candidates(4 * GiB, Devices(Cpu(), Cuda(12 * GiB), Vulkan())));

    [Fact]
    public void CudaThatDoesNotFit_IsSkipped_SoALargeModelGoesToUnifiedMemoryVulkan() =>
        Assert.Equal(["vulkan", "cpu"], DeviceSelector.Candidates(20 * GiB, Devices(Cpu(), Cuda(12 * GiB), Vulkan())));

    [Fact]
    public void TheCudaDeviceWithMostMemoryThatFitsIsChosen() =>
        Assert.Equal("gpu:1", DeviceSelector.Candidates(6 * GiB, Devices(Cpu(), Cuda(8 * GiB, 24 * GiB))).First());

    [Fact]
    public void VulkanOnly_AndAnUnservableVulkanIsNeverOffered()
    {
        Assert.Equal(["vulkan", "cpu"], DeviceSelector.Candidates(GiB, Devices(Cpu(), Vulkan())));
        Assert.Equal(["cpu"], DeviceSelector.Candidates(GiB, Devices(Cpu(), Vulkan(servable: false))));
    }

    [Fact]
    public void NothingElse_IsCpu() => Assert.Equal(["cpu"], DeviceSelector.Candidates(GiB, Devices()));

    [Theory]
    [InlineData("auto", true)]
    [InlineData("AUTO", true)]
    [InlineData(" auto ", true)]
    [InlineData("cpu", false)]
    [InlineData("vulkan", false)]
    [InlineData(null, false)]
    public void IsAuto_IsExactlyTheAutoPseudoDevice(string? device, bool expected) => Assert.Equal(expected, DeviceSelector.IsAuto(device));

    [Fact]
    public void FallbackWarning_NamesModelReasonAndPerfConsequence()
    {
        string w = DeviceSelector.FallbackWarning("nemotron-h-nano", ["gpu:0: out of memory", "vulkan: device lost"]);

        Assert.Contains("nemotron-h-nano", w, StringComparison.Ordinal);
        Assert.Contains("out of memory", w, StringComparison.Ordinal);
        Assert.Contains("vulkan: device lost", w, StringComparison.Ordinal);
        Assert.Contains("slower", w, StringComparison.Ordinal);
        Assert.Contains("CPU", w, StringComparison.Ordinal);
    }

    [Fact]
    public void PropsResponse_SurfacesDeviceFallbackWarning_OnlyWhenSet()
    {
        var set = new DotLLM.Server.Models.PropsResponse { SamplingDefaults = new() , DeviceFallbackWarning = "slow" };
        var unset = new DotLLM.Server.Models.PropsResponse { SamplingDefaults = new() };

        string a = System.Text.Json.JsonSerializer.Serialize(set, DotLLM.Server.ServerJsonContext.Default.PropsResponse);
        string b = System.Text.Json.JsonSerializer.Serialize(unset, DotLLM.Server.ServerJsonContext.Default.PropsResponse);
        Assert.Contains("\"device_fallback_warning\":\"slow\"", a, StringComparison.Ordinal);
        Assert.DoesNotContain("device_fallback_warning", b, StringComparison.Ordinal);
    }
}
