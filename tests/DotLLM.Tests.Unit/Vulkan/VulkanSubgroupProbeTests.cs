using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Reports the <see cref="VulkanDevice.SubgroupSize"/> and
/// <see cref="VulkanDevice.HasSubgroupArithmetic"/> probe results to test
/// output — used both as a sanity check that the Vulkan 1.1 property query
/// does not throw and as a human-readable record of the host GPU's subgroup
/// capabilities. Always passes; the values are informational.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSubgroupProbeTests
{
    private readonly ITestOutputHelper _output;

    public VulkanSubgroupProbeTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Probe_ReportsSubgroupCapability()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device           : {device.DeviceName}");
        _output.WriteLine($"VendorId         : 0x{device.VendorId:X4}");
        _output.WriteLine($"SubgroupSize     : {device.SubgroupSize}");
        _output.WriteLine($"HasSubgroupArith : {device.HasSubgroupArithmetic}");

        // A Vulkan 1.1+ driver must advertise SOME non-zero subgroup width.
        // If SubgroupSize is 0 we're on a Vulkan 1.0-only driver — which is
        // a valid configuration; the shared-memory path must cover it.
        if (device.HasSubgroupArithmetic)
            Assert.True(device.SubgroupSize > 0, "HasSubgroupArithmetic=true implies SubgroupSize>0");
    }
}
