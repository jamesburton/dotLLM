using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #1109 precision-study experiment (<see cref="VulkanPrecisionExperiment.KvRoundF16"/>): rounds a range of an F32 KV store
/// to F16-representable values in place (<c>round_f16_inplace_f32.spv</c>, the same round-to-nearest-even and clamp as the F16 KV
/// writer). The attention kernels then compute on exactly the values an F16 store would hold, so the KL against the unrounded run
/// is the accuracy cost of narrowing that store, measured without an F16 kernel for it.
/// </summary>
/// <remarks>
/// One kernel per device, created on first use. Descriptor sets are cached by buffer handle; the harness that toggles the experiment
/// calls <see cref="InvalidateAll"/> whenever it frees KV caches, so a recycled handle never meets a stale set.
/// </remarks>
internal sealed class VulkanKvRoundF16Kernel : IDisposable
{
    private const int PushConstantBytes = 2 * sizeof(uint);
    private const int WorkgroupSize = 256;
    private const long MaxElementsPerDispatch = 65_535L * WorkgroupSize;

    private static readonly Dictionary<VulkanDevice, VulkanKvRoundF16Kernel> PerDevice = [];

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private bool _disposed;

    private VulkanKvRoundF16Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _pool = pool;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 1);
    }

    /// <summary>Dispatches recorded by every instance (test and harness hook).</summary>
    public static long TotalDispatches { get; private set; }

    /// <summary>
    /// When <see cref="VulkanPrecisionExperiment.KvRoundF16"/> is set, records the rounding of <paramref name="count"/> elements at
    /// <paramref name="elemOffset"/> of <paramref name="buffer"/>, bracketed by blanket barriers (the writer before it may be a
    /// transfer or a compute dispatch, the reader after it a compute dispatch). A no-op otherwise.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="cmd">The command buffer.</param>
    /// <param name="buffer">An F32 KV buffer.</param>
    /// <param name="elemOffset">First element to round.</param>
    /// <param name="count">Elements to round.</param>
    public static void RecordIfEnabled(VulkanDevice device, nint cmd, VulkanDevice.Buffer buffer, long elemOffset, long count)
    {
        if (!VulkanPrecisionExperiment.KvRoundF16 || count <= 0) return;
        if (!PerDevice.TryGetValue(device, out var kernel))
        {
            kernel = Create(device, VulkanModelLoader.ResolveSpvDir());
            PerDevice[device] = kernel;
        }

        KernelSupport.TransferToComputeBarrier(cmd);
        KernelSupport.ComputeToComputeBarrier(cmd);
        kernel.Record(cmd, buffer, elemOffset, count);
        KernelSupport.ComputeToComputeBarrier(cmd);
        KernelSupport.ComputeToTransferBarrier(cmd);
    }

    /// <summary>Drops every instance's cached descriptor sets (call after freeing KV buffers).</summary>
    public static void InvalidateAll()
    {
        foreach (var k in PerDevice.Values) k._cache.Reset();
    }

    private static VulkanKvRoundF16Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "round_f16_inplace_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[1];
            bindings[0] = new VkDescriptorBinding(0);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        return new VulkanKvRoundF16Kernel(device, module, pipeline, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1));
    }

    private unsafe void Record(nint cmd, VulkanDevice.Buffer buffer, long elemOffset, long count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (buffer.Size < (elemOffset + count) * sizeof(float)) throw new ArgumentException("Buffer too small.", nameof(buffer));
        if (elemOffset + count > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(count));
        Span<nint> buffers = stackalloc nint[1] { buffer.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        uint* pc = stackalloc uint[2];
        for (long done = 0; done < count; done += MaxElementsPerDispatch)
        {
            long n = Math.Min(MaxElementsPerDispatch, count - done);
            pc[0] = (uint)(elemOffset + done);
            pc[1] = (uint)n;
            VulkanApi.vkCmdPushConstants(cmd, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pc);
            VulkanApi.vkCmdDispatch(cmd, (uint)((n + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
            TotalDispatches++;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
