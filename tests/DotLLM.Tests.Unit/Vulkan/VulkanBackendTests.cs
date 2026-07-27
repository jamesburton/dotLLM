using DotLLM.Core.Tensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for <see cref="VulkanBackend"/>: device availability, tensor allocation,
/// and host↔device round-trip correctness.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanBackendTests
{
    // ─────────────────────────────────────────────────────────────────
    // Construction and device identity
    // ─────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void DeviceCount_IsOne_WhenVulkanAvailable()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        using var backend = new VulkanBackend();

        Assert.Equal(1, backend.DeviceCount);
    }

    [SkippableFact]
    public void AllocateOnDevice_ReturnsVulkanTensor_WithCorrectMetadata()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        using var backend = new VulkanBackend();
        var shape = new TensorShape(4, 8);

        using var tensor = backend.AllocateOnDevice(deviceId: 0, shape, DType.Float32);

        Assert.IsType<VulkanTensor>(tensor);
        Assert.Equal(shape, tensor.Shape);
        Assert.Equal(DType.Float32, tensor.DType);
        Assert.Equal(0, tensor.DeviceId);
        Assert.Equal(32L, tensor.ElementCount);   // 4 × 8
        Assert.Equal(128L, tensor.ByteCount);      // 32 × 4 bytes
    }

    [SkippableFact]
    public void AllocateOnDevice_DeviceIdNot0_Throws()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        using var backend = new VulkanBackend();

        // IDISP005 false positive: AllocateOnDevice already declares the disposable
        // ITensor return type; this call is expected to throw before any tensor
        // is constructed, so there is nothing to dispose here.
        Assert.Throws<ArgumentException>(
            () => backend.AllocateOnDevice(deviceId: 1, new TensorShape(4), DType.Float32));
    }

    // ─────────────────────────────────────────────────────────────────
    // Round-trip: host → Vulkan → host, bit-exact
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Verifies that <see cref="VulkanBackend.CopyBetweenDevices"/> correctly
    /// transfers FP32 data host→device then device→host with bit-exact results.
    /// This is a pure byte-copy path — any mismatch indicates a real bug.
    /// </summary>
    [SkippableFact]
    public void CopyBetweenDevices_RoundTrip_BitExact()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        const int elementCount = 256;
        var shape = new TensorShape(elementCount);
        var dtype = DType.Float32;

        // Build deterministic source data.
        float[] original = new float[elementCount];
        for (int i = 0; i < elementCount; i++)
            original[i] = (i % 37) * 0.01f - 0.5f;

        using var backend = new VulkanBackend();

        // Allocate host (CPU) source tensor.
        using var hostSrc = UnmanagedTensor.Allocate(shape, dtype, deviceId: -1);
        WriteFloats(hostSrc.DataPointer, original);

        // Allocate device-local Vulkan tensor.
        using var gpuTensor = backend.AllocateOnDevice(deviceId: 0, shape, dtype);

        // Allocate host destination tensor (for the download).
        using var hostDst = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        // Upload: host → Vulkan.
        backend.CopyBetweenDevices(hostSrc, gpuTensor);

        // Download: Vulkan → host.
        backend.CopyBetweenDevices(gpuTensor, hostDst);

        // Verify bit-exact equality — this is pure byte-copy, no conversion.
        float[] result = ReadFloats(hostDst.DataPointer, elementCount);
        for (int i = 0; i < elementCount; i++)
        {
            Assert.Equal(original[i], result[i]);
        }
    }

    /// <summary>
    /// Verifies that the Intel Arc iGPU (vendor 0x8086) is selected when
    /// <c>DOTLLM_VULKAN_DEVICE_VENDOR=0x8086</c> is set in the environment.
    /// This guards against tests silently running on the wrong GPU when
    /// multiple Vulkan devices are present.
    /// </summary>
    [SkippableFact]
    public void VulkanBackend_SelectsIntelArc_WhenVendorEnvSet()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");
        Skip.If(
            !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR"), "0x8086", StringComparison.Ordinal),
            "DOTLLM_VULKAN_DEVICE_VENDOR not set to 0x8086; skipping Intel Arc vendor check.");

        // The env var is read by VulkanDevice.Create() -> ResolveForcedDevice().
        // We need to probe the device directly to check the vendor ID.
        using var device = VulkanDevice.Create();

        Assert.Equal(0x8086u, device.VendorId);
    }

    // ─────────────────────────────────────────────────────────────────
    // Error cases
    // ─────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void CopyBetweenDevices_SizeMismatch_Throws()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        using var backend = new VulkanBackend();
        using var hostSrc = UnmanagedTensor.Allocate(new TensorShape(4), DType.Float32, deviceId: -1);
        using var gpuDst = backend.AllocateOnDevice(deviceId: 0, new TensorShape(8), DType.Float32);

        Assert.Throws<ArgumentException>(
            () => backend.CopyBetweenDevices(hostSrc, gpuDst));
    }

    [SkippableFact]
    public void CopyBetweenDevices_HostToHost_Throws()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");

        using var backend = new VulkanBackend();
        using var hostA = UnmanagedTensor.Allocate(new TensorShape(4), DType.Float32, deviceId: -1);
        using var hostB = UnmanagedTensor.Allocate(new TensorShape(4), DType.Float32, deviceId: -1);

        Assert.Throws<NotSupportedException>(
            () => backend.CopyBetweenDevices(hostA, hostB));
    }

    // ─────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────

    private static unsafe void WriteFloats(nint ptr, float[] values)
    {
        var span = new Span<float>((void*)ptr, values.Length);
        values.CopyTo(span);
    }

    private static unsafe float[] ReadFloats(nint ptr, int count)
    {
        var result = new float[count];
        new ReadOnlySpan<float>((void*)ptr, count).CopyTo(result);
        return result;
    }
}
