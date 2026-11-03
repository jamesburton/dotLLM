using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Writes F32 K / V rows into an F16 <see cref="VulkanNemotronHKvCache"/> (the device-side convert the cache records).</summary>
internal interface IVulkanKvF16Writer
{
    /// <summary>
    /// Records the conversion of <paramref name="count"/> F32 elements from the start of <paramref name="src"/> into F16 at element
    /// offset <paramref name="dstElemOffset"/> of <paramref name="dst"/>. Both must be even. Values saturate at the F16 range.
    /// </summary>
    /// <param name="cmd">The command buffer.</param>
    /// <param name="src">F32 source rows.</param>
    /// <param name="dst">The F16 (uint-backed) destination.</param>
    /// <param name="dstElemOffset">Destination element offset (even).</param>
    /// <param name="count">Elements to convert (even).</param>
    void RecordF32ToF16(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long dstElemOffset, long count);

    /// <summary>
    /// Records the conversion of <paramref name="rows"/> F32 rows (<paramref name="kvStride"/> elements each, starting at source row <paramref name="srcRow0"/> of
    /// <paramref name="src"/>) into F16 rows <c>positions[srcRow0 + t] - rowBase</c> of <paramref name="dst"/>: the non-contiguous write of selective recompute (#1151).
    /// </summary>
    void RecordScatterRows(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer positions, int kvStride, int rows, int rowBase, int srcRow0 = 0)
        => throw new NotSupportedException("This F16 KV writer cannot scatter rows to non-contiguous positions.");
}

/// <summary>
/// The F32-to-F16 KV row writer for hosts other than Qwen4-Exp (issue #1102): one <c>convert_f32_to_f16_range.spv</c> pipeline, the
/// same shader <see cref="Qwen4ExpQsaKernels"/> uses. It rounds to nearest-even in the integer domain, so the stored halves are
/// driver independent and equal <see cref="Half"/>'s conversion of the F32 row (the parity tests' oracle).
/// </summary>
internal sealed class VulkanKvF16WriteKernel : IVulkanKvF16Writer, IDisposable
{
    private const int PushConstantBytes = 2 * sizeof(uint);
    private const int WorkgroupSize = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private VulkanModule? _rowsModule;
    private ComputePipeline? _rowsPipeline;
    private nint _rowsPool;
    private DescriptorSetCache? _rowsCache;
    private bool _disposed;

    private VulkanKvF16WriteKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _pool = pool;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>Dispatches recorded so far (test hook).</summary>
    public long Dispatches { get; private set; }

    /// <summary>Loads <c>convert_f32_to_f16_range.spv</c>.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V is missing.</exception>
    public static VulkanKvF16WriteKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "convert_f32_to_f16_range.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        var kernel = new VulkanKvF16WriteKernel(device, module, pipeline, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2));
        kernel.TryLoadRows(spvDir);
        return kernel;
    }

    /// <summary>True when <c>convert_f32_to_f16_rows.spv</c> loaded, so non-contiguous positions can be written (#1151).</summary>
    public bool SupportsScatter => _rowsPipeline is not null;

    private void TryLoadRows(string spvDir)
    {
        string path = Path.Combine(spvDir, "convert_f32_to_f16_rows.spv");
        if (!File.Exists(path)) return;
        _rowsModule = VulkanModule.LoadFromFile(_device, path);
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
        bindings[0] = new VkDescriptorBinding(0);
        bindings[1] = new VkDescriptorBinding(1);
        bindings[2] = new VkDescriptorBinding(2);
        _rowsPipeline = _rowsModule.CreateComputePipeline("main", bindings, 4 * sizeof(uint));
        _rowsPool = KernelSupport.CreateDescriptorPool(_device, buffersPerSet: 3);
        _rowsCache = new DescriptorSetCache(_device, _rowsPool, _rowsPipeline, buffersPerSet: 3);
    }

    /// <inheritdoc/>
    public unsafe void RecordScatterRows(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer positions, int kvStride, int rows, int rowBase, int srcRow0 = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_rowsPipeline is null) throw new NotSupportedException("convert_f32_to_f16_rows.spv is missing; rebuild the Vulkan shaders.");
        if (rows <= 0) return;
        if ((kvStride & 1) != 0) throw new ArgumentException("F16 KV row width must be even.", nameof(kvStride));
        long words = (long)rows * (kvStride / 2);
        if (words > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(rows));
        if (src.Size < ((long)srcRow0 + rows) * kvStride * sizeof(float)) throw new ArgumentException("Source buffer too small.", nameof(src));
        if (positions.Size < ((long)srcRow0 + rows) * sizeof(int)) throw new ArgumentException("Positions buffer too small.", nameof(positions));

        Span<nint> buffers = stackalloc nint[3] { src.Handle, dst.Handle, positions.Handle };
        nint set = _rowsCache!.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _rowsPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _rowsPipeline.Layout, 0, 1, set, 0, 0);
        uint* pc = stackalloc uint[4];
        pc[0] = (uint)(kvStride / 2);
        pc[1] = (uint)rows;
        pc[2] = (uint)rowBase;
        pc[3] = (uint)srcRow0;
        VulkanApi.vkCmdPushConstants(cmd, _rowsPipeline.Layout, VkShaderStageFlags.Compute, 0, 4 * sizeof(uint), (nint)pc);
        VulkanApi.vkCmdDispatch(cmd, (uint)((words + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
        Dispatches++;
    }

    /// <inheritdoc/>
    public unsafe void RecordF32ToF16(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long dstElemOffset, long count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (count <= 0) return;
        if ((dstElemOffset & 1) != 0 || (count & 1) != 0) throw new ArgumentException("F16 convert offsets and counts must be even.");
        long words = count / 2;
        if (dstElemOffset / 2 + words > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(count));
        if (src.Size < count * sizeof(float)) throw new ArgumentException("Source buffer too small.", nameof(src));
        if (dst.Size < (dstElemOffset + count) * sizeof(ushort)) throw new ArgumentException("Destination buffer too small.", nameof(dst));

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        uint* pc = stackalloc uint[2];
        pc[0] = (uint)(dstElemOffset / 2);
        pc[1] = (uint)words;
        VulkanApi.vkCmdPushConstants(cmd, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pc);
        VulkanApi.vkCmdDispatch(cmd, (uint)((words + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
        Dispatches++;
    }

    /// <summary>Drops the cached descriptor sets (a KV buffer was freed; its handle may be recycled).</summary>
    public void InvalidateDescriptorCache() { _cache.Reset(); _rowsCache?.Reset(); }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        if (_rowsPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _rowsPool, 0);
        _rowsPipeline?.Dispose();
        _rowsModule?.Dispose();
        _pipeline.Dispose();
        _module.Dispose();
    }
}
