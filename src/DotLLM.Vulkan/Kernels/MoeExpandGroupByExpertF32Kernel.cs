using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Packs routed MoE rows into expert-contiguous ranges using caller-provided
/// expert offsets and a zeroed per-expert counter buffer.
/// </summary>
public sealed class MoeExpandGroupByExpertF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeExpandGroupByExpertF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 6);
    }

    /// <summary>Loads <c>moe_expand_group_by_expert_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeExpandGroupByExpertF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_expand_group_by_expert_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[6];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 6);
        return new MoeExpandGroupByExpertF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer counters, VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation,
        int rows, int hidden, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, indices, offsets, counters, packed, permutation,
            rows, hidden, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the grouping dispatch into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer counters, VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation,
        int rows, int hidden, int numExperts)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));

        long matrixBytes = (long)rows * hidden * sizeof(float);
        if (x.Size < matrixBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (packed.Size < matrixBytes) throw new ArgumentException("packed buffer too small.", nameof(packed));
        if (indices.Size < (long)rows * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (permutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("permutation buffer too small.", nameof(permutation));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (counters.Size < (long)numExperts * sizeof(uint)) throw new ArgumentException("counters buffer too small.", nameof(counters));

        Span<nint> buffers = stackalloc nint[6]
        {
            x.Handle, indices.Handle, offsets.Handle, counters.Handle, packed.Handle, permutation.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)rows, (uint)hidden, (uint)numExperts };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rows, 1, 1);
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
