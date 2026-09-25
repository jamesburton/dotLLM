using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Guards the 32-bit ceiling that keeps every GLSL byte-offset computation in
/// <c>native/vulkan/shaders/*.comp</c> sound.
/// </summary>
/// <remarks>
/// The shaders index storage buffers with 32-bit <c>uint</c> byte offsets (GLSL has no
/// 64-bit integer arithmetic unless <c>GL_EXT_shader_explicit_arithmetic_types_int64</c>
/// is enabled, which none of our compute kernels do). Every kernel binds its buffers
/// with <c>VK_WHOLE_SIZE</c> at offset 0, so the largest offset a shader can compute for
/// in-range data is bounded by the buffer size — which Vulkan bounds by
/// <c>maxStorageBufferRange</c>. As long as we never allocate past that limit, the
/// <c>uint</c> arithmetic cannot wrap.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanStorageBufferRangeTests
{
    /// <summary>
    /// Reads <c>maxStorageBufferRange</c> off the device and checks it is plausible.
    /// This is also the regression test for the hand-computed byte offset into the
    /// opaque <c>VkPhysicalDeviceLimits</c> tail: the neighbouring fields
    /// (<c>maxUniformBufferRange</c> ≥ 16 KiB, <c>maxPushConstantsSize</c> ≥ 128 B,
    /// <c>maxTexelBufferElements</c> ≥ 64 Ki) are all orders of magnitude below the
    /// 2²⁷ Vulkan floor asserted here, so a wrong offset fails.
    /// </summary>
    [SkippableFact]
    public void MaxStorageBufferRange_IsAtLeastTheVulkanSpecFloor()
    {
        Skip.If(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN"), "1", StringComparison.Ordinal), "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();

        ulong range = device.MaxStorageBufferRange;

        // Vulkan's required minimum is 2^27 (128 MiB); the field is a uint32_t so it
        // can never exceed uint.MaxValue.
        // (Verified against `vulkaninfo` on the gfx1151 target: reads exactly
        // 4294967295. Asserted as a range so the test stays portable.)
        Assert.InRange(range, 1UL << 27, uint.MaxValue);
    }

    [Fact]
    public void ThrowIfExceedsStorageBufferRange_AllowsSizesAtOrBelowTheLimit()
    {
        VulkanDevice.ThrowIfExceedsStorageBufferRange(1, uint.MaxValue);
        VulkanDevice.ThrowIfExceedsStorageBufferRange(uint.MaxValue, uint.MaxValue);
        // 2 GiB + 1: past int.MaxValue but still bindable when the device says so.
        VulkanDevice.ThrowIfExceedsStorageBufferRange(2_147_483_649L, uint.MaxValue);
    }

    [Fact]
    public void ThrowIfExceedsStorageBufferRange_RejectsSizesPastTheLimit()
    {
        // 4 GiB — one byte past the largest uint byte offset a shader can represent.
        Assert.Throws<InvalidOperationException>(
            () => VulkanDevice.ThrowIfExceedsStorageBufferRange(4L * 1024 * 1024 * 1024, uint.MaxValue));

        // A device reporting only the spec floor rejects anything larger.
        Assert.Throws<InvalidOperationException>(
            () => VulkanDevice.ThrowIfExceedsStorageBufferRange((1L << 27) + 1, 1UL << 27));
    }

    [Fact]
    public void ThrowIfExceedsStorageBufferRange_IsInertWhenTheLimitIsUnreadable()
    {
        // A zero limit means we could not read the property — never block the caller on it.
        VulkanDevice.ThrowIfExceedsStorageBufferRange(long.MaxValue, 0);
    }
}
