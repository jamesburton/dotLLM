using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #374: the device enables the 16-bit / 8-bit storage and float16 / int8 features its shaders declare. Both validation
/// targets, the Strix Halo (AMD) and the Meteor Lake Arc (Intel), support all of them, so on those vendors every bit must come up
/// set: <see cref="VulkanDevice.NumericFeatures"/> reports what was enabled at device creation.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNumericFeaturesTests
{
    private const uint AmdVendor = 0x1002;
    private const uint IntelVendor = 0x8086;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the test with the xUnit output sink.</summary>
    /// <param name="output">Sink for the feature report.</param>
    public VulkanNumericFeaturesTests(ITestOutputHelper output) => _output = output;

    /// <summary>On the AMD and Intel targets the storage and arithmetic bits the shaders need are enabled.</summary>
    [SkippableFact]
    public void TargetDevices_EnableTheNarrowTypeFeatures()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        var f = device.NumericFeatures;
        _output.WriteLine($"{device.DeviceName} (0x{device.VendorId:x4}): {f}");
        Skip.IfNot(device.VendorId is AmdVendor or IntelVendor, "only the AMD and Intel targets are known to support every bit");

        Assert.True(f.StorageBuffer16BitAccess, "storageBuffer16BitAccess");
        Assert.True(f.ShaderFloat16, "shaderFloat16");
        Assert.True(f.ShaderInt8, "shaderInt8");
        Assert.True(f.StorageBuffer8BitAccess, "storageBuffer8BitAccess");
    }
}
