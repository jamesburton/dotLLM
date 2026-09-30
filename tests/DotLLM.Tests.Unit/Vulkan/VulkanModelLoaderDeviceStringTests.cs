using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <c>--device vulkan</c> used to fall through to the CPU path silently in <c>run</c>/<c>serve</c> (the string does
/// not start with "gpu"). The device-string predicate that now routes them is pinned here.
/// </summary>
public sealed class VulkanModelLoaderDeviceStringTests
{
    [Theory]
    [InlineData("vulkan", true)]
    [InlineData("Vulkan", true)]
    [InlineData("VULKAN:0", true)]
    [InlineData("cpu", false)]
    [InlineData("gpu", false)]
    [InlineData("gpu:1", false)]
    [InlineData("cuda", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void IsVulkanDeviceString_RecognisesOnlyVulkan(string? device, bool expected) =>
        Assert.Equal(expected, VulkanModelLoader.IsVulkanDeviceString(device));
}
