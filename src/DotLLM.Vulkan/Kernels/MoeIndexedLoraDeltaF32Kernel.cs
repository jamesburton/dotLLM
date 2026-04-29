using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Applies a LoRA delta to routed MoE expanded rows for one expert only.
/// Rows whose <c>indices[row]</c> do not match the requested expert are
/// left unchanged.
/// </summary>
public sealed class MoeIndexedLoraDeltaF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // outputDim, inputDim, rank, rows, expert (all u32)
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedLoraDeltaF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
    }

    /// <summary>Loads <c>moe_indexed_lora_delta_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedLoraDeltaF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_lora_delta_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            bindings[4] = new VkDescriptorBinding(4);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MoeIndexedLoraDeltaF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer bWeight, VulkanDevice.Buffer aWeight, VulkanDevice.Buffer y,
        int rows, int inputDim, int outputDim, int rank, int expert)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, indices, bWeight, aWeight, y,
            rows, inputDim, outputDim, rank, expert);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the masked LoRA delta dispatch.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer bWeight, VulkanDevice.Buffer aWeight, VulkanDevice.Buffer y,
        int rows, int inputDim, int outputDim, int rank, int expert)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));
        if (expert < 0) throw new ArgumentOutOfRangeException(nameof(expert));

        long xBytes = (long)rows * inputDim * sizeof(float);
        long idxBytes = (long)rows * sizeof(int);
        long bBytes = (long)rank * inputDim * sizeof(float);
        long aBytes = (long)outputDim * rank * sizeof(float);
        long yBytes = (long)rows * outputDim * sizeof(float);
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (bWeight.Size < bBytes) throw new ArgumentException("bWeight buffer too small.", nameof(bWeight));
        if (aWeight.Size < aBytes) throw new ArgumentException("aWeight buffer too small.", nameof(aWeight));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
        {
            x.Handle, indices.Handle, bWeight.Handle, aWeight.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)outputDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)inputDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)rank);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)rows);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)expert);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((outputDim + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((rows + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
