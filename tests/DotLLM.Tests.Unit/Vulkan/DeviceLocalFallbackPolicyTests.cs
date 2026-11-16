using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1849: the device-local -> host-visible fallback policy as a pure function of (device type, env value). Discrete
/// GPUs refuse by default; integrated/UMA keep the fallback; <c>DOTLLM_VULKAN_STRICT_DEVICE_LOCAL=1</c> forces strict
/// everywhere and <c>=0</c> forces the fallback everywhere.
/// </summary>
public sealed class DeviceLocalFallbackPolicyTests
{
    [Theory]
    // default (unset / empty / unrecognised): only discrete is strict
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, null, false)]
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, "", false)]
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, "yes", false)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, null, true)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, "", true)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, "yes", true)]
    [InlineData(VkPhysicalDeviceType.Cpu, null, true)]
    [InlineData(VkPhysicalDeviceType.VirtualGpu, null, true)]
    [InlineData(VkPhysicalDeviceType.Other, null, true)]
    // =1 forces strict everywhere
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, "1", false)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, "1", false)]
    [InlineData(VkPhysicalDeviceType.Cpu, "1", false)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, " 1 ", false)]
    // =0 forces the fallback everywhere, discrete included
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, "0", true)]
    [InlineData(VkPhysicalDeviceType.DiscreteGpu, " 0", true)]
    [InlineData(VkPhysicalDeviceType.IntegratedGpu, "0", true)]
    public void Policy_MatchesDeviceTypeAndOverride(int deviceType, string? env, bool expectedAllowsFallback) =>
        Assert.Equal(expectedAllowsFallback, VulkanDevice.AllowsDeviceLocalFallback(deviceType, env));

    [Fact]
    public void EnvVarName_IsTheDocumentedOne() =>
        Assert.Equal("DOTLLM_VULKAN_STRICT_DEVICE_LOCAL", VulkanDevice.StrictDeviceLocalEnvVar);
}
