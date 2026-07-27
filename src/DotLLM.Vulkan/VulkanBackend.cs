using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// GPU backend using Vulkan compute via raw P/Invoke. Supports single-device
/// operations against the first available (or
/// <c>DOTLLM_VULKAN_DEVICE_VENDOR</c>-targeted) Vulkan device.
/// Multi-device operations (<see cref="AllReduce"/>, <see cref="Send"/>,
/// <see cref="Receive"/>) throw <see cref="NotSupportedException"/> — they are
/// deferred to the async-pipelining milestone (M3).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="AllocateOnDevice"/> produces a <see cref="VulkanTensor"/> backed
/// by a device-local <c>VkBuffer</c>. The returned tensor's
/// <see cref="VulkanTensor.DataPointer"/> is the opaque <c>VkBuffer</c> handle;
/// it is NOT a host-accessible address.
/// </para>
/// <para>
/// <see cref="CopyBetweenDevices"/> routes:
/// <list type="bullet">
///   <item>Host → Vulkan: alloc HOST_VISIBLE staging buffer, map + copy FP32 bytes,
///     synchronous <c>vkCmdCopyBuffer</c> staging → device-local, dispose staging.
///     This is the same path as <c>VulkanKvCache.IngestFromHost</c>.</item>
///   <item>Vulkan → Host: synchronous <c>vkCmdCopyBuffer</c> device-local →
///     HOST_VISIBLE staging, map + copy FP32 bytes out, dispose staging.</item>
///   <item>Vulkan ↔ Vulkan (device 0 → device 0): not yet supported;
///     throws <see cref="NotSupportedException"/>.</item>
/// </list>
/// All operations are synchronous (fence wait). Async pipelining via Vulkan
/// timeline semaphores is M3 scope.
/// </para>
/// </remarks>
public sealed class VulkanBackend : IBackend
{
    private readonly VulkanDevice _device;

    /// <inheritdoc/>
    /// <remarks>Always 1 — this backend binds to a single Vulkan physical device.</remarks>
    public int DeviceCount => 1;

    /// <summary>
    /// Creates a Vulkan backend, initialising the Vulkan loader and selecting
    /// the first available (or vendor-targeted) physical device.
    /// </summary>
    /// <exception cref="VulkanException">Thrown when no Vulkan device is available.</exception>
    public VulkanBackend()
    {
        _device = VulkanDevice.Create();
    }

    /// <summary>
    /// Creates a Vulkan backend that borrows an existing <see cref="VulkanDevice"/>.
    /// The device is NOT disposed when this backend is disposed — the caller
    /// retains ownership. Useful when the device is already held by a
    /// <see cref="VulkanTransformerModel"/> that must outlive the backend reference.
    /// </summary>
    /// <param name="device">An already-created Vulkan device to borrow.</param>
    public VulkanBackend(VulkanDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _ownsDevice = false;
    }

    private readonly bool _ownsDevice = true;

    /// <inheritdoc/>
    /// <remarks>
    /// Allocates a device-local Vulkan buffer of the required byte size and
    /// wraps it in a <see cref="VulkanTensor"/>. <paramref name="deviceId"/>
    /// must be 0 (this backend has exactly one device).
    /// </remarks>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        if (deviceId != 0)
            throw new ArgumentException(
                $"VulkanBackend has DeviceCount=1 — only deviceId=0 is valid, got {deviceId}.",
                nameof(deviceId));

        return VulkanTensor.Allocate(_device, shape, dtype, deviceId: 0);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exactly one of <paramref name="source"/> and <paramref name="destination"/>
    /// must be a <see cref="VulkanTensor"/> (deviceId=0); the other must be a
    /// host tensor (deviceId=-1) with a valid <see cref="ITensor.DataPointer"/>.
    /// </para>
    /// <para>
    /// Host → Vulkan: allocates a HOST_VISIBLE staging buffer, maps it and
    /// copies from <c>source.DataPointer</c>, then issues a synchronous
    /// <c>vkCmdCopyBuffer</c> staging → device-local.
    /// </para>
    /// <para>
    /// Vulkan → Host: issues a synchronous <c>vkCmdCopyBuffer</c> device-local →
    /// HOST_VISIBLE staging, maps the staging buffer and copies to
    /// <c>destination.DataPointer</c>.
    /// </para>
    /// </remarks>
    public unsafe void CopyBetweenDevices(ITensor source, ITensor destination)
    {
        if (source.ByteCount != destination.ByteCount)
            throw new ArgumentException(
                $"Source ({source.ByteCount} bytes) and destination ({destination.ByteCount} bytes) sizes differ.");

        long bytes = source.ByteCount;

        if (source.DeviceId == -1 && destination.DeviceId == 0)
        {
            // Host → Vulkan: staging copy, same pattern as VulkanKvCache.IngestFromHost
            var dst = CastToVulkanTensor(destination, nameof(destination));
            using var staging = _device.Allocate(bytes);
            MapAndCopyIn(staging, source.DataPointer, bytes);
            _device.CopyBufferRangeSynchronous(staging, dst.DeviceBuffer,
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        }
        else if (source.DeviceId == 0 && destination.DeviceId == -1)
        {
            // Vulkan → Host: staging copy then map-read
            var src = CastToVulkanTensor(source, nameof(source));
            using var staging = _device.Allocate(bytes);
            _device.CopyBufferRangeSynchronous(src.DeviceBuffer, staging,
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
            MapAndCopyOut(staging, destination.DataPointer, bytes);
        }
        else
        {
            throw new NotSupportedException(
                $"VulkanBackend.CopyBetweenDevices does not support {source.DeviceId}→{destination.DeviceId}. "
                + "Supported: host(-1)→Vulkan(0) and Vulkan(0)→host(-1).");
        }
    }

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors) =>
        throw new NotSupportedException(
            "VulkanBackend does not support AllReduce (requires multi-device + timeline semaphores, M3 scope).");

    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice) =>
        throw new NotSupportedException(
            "VulkanBackend does not support Send (requires multi-device + timeline semaphores, M3 scope).");

    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype) =>
        throw new NotSupportedException(
            "VulkanBackend does not support Receive (requires multi-device + timeline semaphores, M3 scope).");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsDevice)
            _device.Dispose();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static VulkanTensor CastToVulkanTensor(ITensor tensor, string paramName)
    {
        if (tensor is not VulkanTensor vt)
            throw new ArgumentException(
                $"Expected a VulkanTensor for deviceId=0 but got {tensor.GetType().Name}.",
                paramName);
        return vt;
    }

    /// <summary>
    /// Maps <paramref name="staging"/> (HOST_VISIBLE) and copies
    /// <paramref name="bytes"/> bytes FROM <paramref name="hostPtr"/> into it.
    /// </summary>
    private unsafe void MapAndCopyIn(VulkanDevice.Buffer staging, nint hostPtr, long bytes)
    {
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)bytes, "vkMapMemory VulkanBackend H2D staging");
        try
        {
            System.Buffer.MemoryCopy((void*)hostPtr, (void*)mapped, bytes, bytes);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
    }

    /// <summary>
    /// Maps <paramref name="staging"/> (HOST_VISIBLE) and copies
    /// <paramref name="bytes"/> bytes TO <paramref name="hostPtr"/> from it.
    /// </summary>
    private unsafe void MapAndCopyOut(VulkanDevice.Buffer staging, nint hostPtr, long bytes)
    {
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)bytes, "vkMapMemory VulkanBackend D2H staging");
        try
        {
            System.Buffer.MemoryCopy((void*)mapped, (void*)hostPtr, bytes, bytes);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
    }
}
