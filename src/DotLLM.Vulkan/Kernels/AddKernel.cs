using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Proof-of-pipeline compute kernel that performs <c>c[i] = a[i] + b[i]</c>
/// over three FP32 storage buffers.
/// </summary>
/// <remarks>
/// This kernel exists to demonstrate and exercise the full Vulkan compute
/// path — SPIR-V load, descriptor set, push constant, command buffer record,
/// queue submit, wait. Real LLM kernels (rmsnorm, rope, attention, swiglu,
/// embedding, dequant) follow the same scaffolding.
/// </remarks>
public sealed class AddKernel : IDisposable
{
    private const int WorkgroupSize = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private AddKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>add.spv</c> from the given directory and creates the pipeline.</summary>
    public static AddKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "add.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: sizeof(uint)); // just `n`
        }
        catch
        {
            module.Dispose();
            throw;
        }

        // Single, small descriptor pool with one set for this kernel. A real
        // implementation pools descriptors across launches; for the proof-of-
        // pipeline scaffold, one set per kernel instance is fine.
        nint pool = CreateDescriptorPool(device);
        return new AddKernel(device, module, pipeline, pool);
    }

    private static unsafe nint CreateDescriptorPool(VulkanDevice device)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = 3,
        };
        VkDescriptorPoolCreateInfo ci = default;
        ci.sType = VkStructureType.DescriptorPoolCreateInfo;
        ci.maxSets = 1;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = (nint)(&poolSize);
        VulkanApi.vkCreateDescriptorPool(device.Handle, ci, 0, out nint pool)
            .ThrowOnError("vkCreateDescriptorPool");
        return pool;
    }

    /// <summary>
    /// Dispatches the add kernel: <c>c[i] = a[i] + b[i]</c> for <paramref name="n"/>
    /// FP32 elements. All three buffers must be at least <c>n * sizeof(float)</c> bytes.
    /// Synchronous — the call returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public unsafe void Launch(VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        // 1. Allocate one descriptor set.
        nint setLayout = _pipeline.DescriptorSetLayout;
        var dsai = new VkDescriptorSetAllocateInfo
        {
            sType = VkStructureType.DescriptorSetAllocateInfo,
            descriptorPool = _descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = (nint)(&setLayout),
        };
        VulkanApi.vkAllocateDescriptorSets(_device.Handle, dsai, out nint descriptorSet)
            .ThrowOnError("vkAllocateDescriptorSets");

        // 2. Write buffer bindings into the descriptor set.
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[3];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = a.Handle, offset = 0, range = ulong.MaxValue }; // VK_WHOLE_SIZE
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = b.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = c.Handle, offset = 0, range = ulong.MaxValue };

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[3];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < 3; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VkStructureType.WriteDescriptorSet,
                    dstSet = descriptorSet,
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    pBufferInfo = (nint)(bufPtr + i),
                };
            }

            fixed (VkWriteDescriptorSet* writesPtr = writes)
            {
                VulkanApi.vkUpdateDescriptorSets(_device.Handle, 3, (nint)writesPtr, 0, 0);
            }
        }

        // 3. Allocate and record a one-shot command buffer.
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _device.CommandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device.Handle, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers");

        try
        {
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(cmdBuf, begin)
                .ThrowOnError("vkBeginCommandBuffer");

            VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(
                cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
                0, 1, descriptorSet, 0, 0);

            uint pushN = (uint)n;
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, sizeof(uint), (nint)(&pushN));

            uint groups = (uint)((n + WorkgroupSize - 1) / WorkgroupSize);
            VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);

            VulkanApi.vkEndCommandBuffer(cmdBuf)
                .ThrowOnError("vkEndCommandBuffer");

            // 4. Submit and wait. Fence-based pipelining comes later.
            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBuf),
            };
            VulkanApi.vkQueueSubmit(_device.Queue, 1, submit, 0)
                .ThrowOnError("vkQueueSubmit");
            VulkanApi.vkQueueWaitIdle(_device.Queue)
                .ThrowOnError("vkQueueWaitIdle");
        }
        finally
        {
            VulkanApi.vkFreeCommandBuffers(_device.Handle, _device.CommandPool, 1, cmdBuf);
        }
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
