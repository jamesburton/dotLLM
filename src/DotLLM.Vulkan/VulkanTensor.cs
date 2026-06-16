using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan;

/// <summary>
/// Tensor implementation backed by a device-local Vulkan buffer.
/// The <see cref="DataPointer"/> holds the <c>VkBuffer</c> handle — it is
/// an opaque GPU resource pointer and must NOT be dereferenced from C#.
/// Use <see cref="VulkanBackend.CopyBetweenDevices"/> (which down-casts to this
/// type to access <see cref="DeviceBuffer"/> directly) rather than reading or
/// writing via <see cref="DataPointer"/>.
/// </summary>
/// <remarks>
/// Mirrors <c>CudaTensor</c>'s "opaque device pointer" convention: the Vulkan
/// runtime owns the allocation; .NET code never touches the bytes directly.
/// Disposal destroys the backing <c>VkBuffer</c> + <c>VkDeviceMemory</c>.
/// </remarks>
public sealed class VulkanTensor : ITensor
{
    private readonly VulkanDevice.Buffer _buffer;
    private bool _disposed;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <summary>
    /// The underlying Vulkan buffer. Use this (not <see cref="DataPointer"/>)
    /// when issuing <c>vkCmd*</c> operations or staging copies.
    /// </summary>
    public VulkanDevice.Buffer DeviceBuffer => _buffer;

    /// <summary>
    /// Returns the <c>VkBuffer</c> handle as an opaque <see cref="nint"/>.
    /// This is NOT a host-accessible memory address — do not dereference it.
    /// </summary>
    public nint DataPointer => _buffer.Handle;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, DataPointer);

    /// <inheritdoc/>
    public long ElementCount { get; }

    /// <inheritdoc/>
    public long ByteCount { get; }

    private VulkanTensor(VulkanDevice.Buffer buffer, TensorShape shape, DType dtype, int deviceId)
    {
        _buffer = buffer;
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        ElementCount = shape.ElementCount;
        ByteCount = dtype.ComputeByteCount(ElementCount);
    }

    /// <summary>
    /// Allocates a device-local Vulkan buffer of the correct size and wraps it
    /// in a <see cref="VulkanTensor"/>. The buffer contents are uninitialised.
    /// </summary>
    /// <param name="device">The Vulkan device that owns the allocation.</param>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">Logical device ID (typically 0 for the single Vulkan device).</param>
    /// <returns>A newly allocated tensor. Caller owns disposal.</returns>
    public static VulkanTensor Allocate(VulkanDevice device, TensorShape shape, DType dtype, int deviceId = 0)
    {
        ArgumentNullException.ThrowIfNull(device);
        long bytes = dtype.ComputeByteCount(shape.ElementCount);
        var buffer = device.AllocateDeviceLocal(bytes);
        return new VulkanTensor(buffer, shape, dtype, deviceId);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _buffer.Dispose();
    }
}
