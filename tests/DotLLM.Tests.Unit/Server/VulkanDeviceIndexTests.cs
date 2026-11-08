using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1704: <see cref="VulkanModelLoader.GetDevice"/> binds the physical device the index names. Needs a Vulkan loader (skips otherwise);
/// creates logical devices only, no model, no shaders.
/// </summary>
public sealed class VulkanDeviceIndexTests
{
    [SkippableFact]
    public void GetDevice_BindsTheIndexedPhysicalDevice_AndCachesIt()
    {
        var infos = VulkanDevice.EnumeratePhysicalDevices();
        Skip.If(infos.Length == 0, "No Vulkan device.");
        foreach (var info in infos)
        {
            var d = VulkanModelLoader.GetDevice(info.Index);
            Assert.Equal(info.Index, d.PhysicalDeviceIndex);
            Assert.Equal(info.Name, d.DeviceName);
            Assert.Same(d, VulkanModelLoader.GetDevice(info.Index));   // one device per index
        }
    }

    [SkippableFact]
    public void DefaultFlag_MatchesTheSharedDevice_AndPlainVulkanSharesItsInstance()
    {
        var infos = VulkanDevice.EnumeratePhysicalDevices();
        Skip.If(infos.Length == 0, "No Vulkan device.");
        Assert.Single(infos, i => i.IsDefault);
        int def = infos.Single(i => i.IsDefault).Index;
        Assert.Equal(def, VulkanModelLoader.SharedDevice.PhysicalDeviceIndex);
        Assert.Same(VulkanModelLoader.SharedDevice, VulkanModelLoader.GetDevice(def));   // vulkan == vulkan:<default>, no second device
    }

    [SkippableFact]
    public void GetDevice_OutOfRange_ThrowsAClearError()
    {
        var infos = VulkanDevice.EnumeratePhysicalDevices();
        Skip.If(infos.Length == 0, "No Vulkan device.");
        var ex = Assert.Throws<VulkanException>(() => VulkanModelLoader.GetDevice(infos.Length));
        Assert.Contains("out of range", ex.Message);
        Assert.Contains($"0..{infos.Length - 1}", ex.Message);
    }
}
