using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Computes routed MoE expert counts and exclusive offsets on-GPU.
/// </summary>
public sealed class MoeExpertOffsetsKernel : IDisposable
{
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeExpertOffsetsKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_expert_offsets.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeExpertOffsetsKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_expert_offsets.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeExpertOffsetsKernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer indices, VulkanDevice.Buffer counts,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer groupCounters,
        int rows, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, indices, counts, offsets, groupCounters, rows, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the offsets dispatch. <paramref name="groupCounters"/> is zeroed by this
    /// kernel and can be passed directly to <see cref="MoeExpandGroupByExpertF32Kernel"/>.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer counts,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer groupCounters,
        int rows, int numExperts)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));

        if (indices.Size < (long)rows * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (counts.Size < (long)numExperts * sizeof(uint)) throw new ArgumentException("counts buffer too small.", nameof(counts));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (groupCounters.Size < (long)numExperts * sizeof(uint)) throw new ArgumentException("groupCounters buffer too small.", nameof(groupCounters));

        Span<nint> buffers = stackalloc nint[4]
        {
            indices.Handle, counts.Handle, offsets.Handle, groupCounters.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)rows, (uint)numExperts };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, 1, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
