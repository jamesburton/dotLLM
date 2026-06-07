using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE weighted scatter — final combine step of a MoE forward pass.
/// Collapses the per-(token, slot) expert outputs into per-token outputs
/// by summing each token's <c>topK</c> rows scaled by its routing weights:
/// <code>
///     out[t, h] = sum_{slot=0..topK-1} weights[t * topK + slot] * x[t * topK + slot, h]
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the per-(token, slot) accumulation tail of
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.Execute</c> — same per-token slot
/// order so the result is bit-stable wrt the CPU reference modulo F32
/// rounding upstream in the matmul / SwiGLU phases.
/// </para>
/// <para>
/// Adapted to the inline-pool pattern: each <see cref="Launch"/> allocates a
/// fresh descriptor set, records a one-time-submit command buffer, waits on
/// the queue, then resets the descriptor pool. Matches the existing
/// <c>MoeTopKSoftmaxF32Kernel</c> Launch convention.
/// </para>
/// </remarks>
public sealed class MoeWeightedScatterF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, topK, hiddenSize (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MoeWeightedScatterF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>moe_weighted_scatter_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeWeightedScatterF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_weighted_scatter_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = CreateDescriptorPool(device);
        return new MoeWeightedScatterF32Kernel(device, module, pipeline, pool);
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
    /// Synchronous dispatch of the weighted scatter. Returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="x">Per-(token, slot) expert outputs [<paramref name="seqLen"/> × <paramref name="topK"/> × <paramref name="hiddenSize"/>] F32 row-major.</param>
    /// <param name="weights">Routing weights [<paramref name="seqLen"/> × <paramref name="topK"/>] F32 row-major.</param>
    /// <param name="output">Combined per-token output [<paramref name="seqLen"/> × <paramref name="hiddenSize"/>]; fully overwritten.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="topK">Per-token expert count (1 ≤ topK).</param>
    /// <param name="hiddenSize">Per-token feature dim.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer weights, VulkanDevice.Buffer output,
        int seqLen, int topK, int hiddenSize)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));

        long xBytes = (long)seqLen * topK * hiddenSize * sizeof(float);
        long wBytes = (long)seqLen * topK * sizeof(float);
        long outBytes = (long)seqLen * hiddenSize * sizeof(float);
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (weights.Size < wBytes) throw new ArgumentException("weights buffer too small.", nameof(weights));
        if (output.Size < outBytes) throw new ArgumentException("output buffer too small.", nameof(output));

        // 1. Allocate descriptor set.
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

        // 2. Bind buffers.
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[3];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = x.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = weights.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = output.Handle, offset = 0, range = ulong.MaxValue };

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

        // 3. Record and submit.
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
            VulkanApi.vkBeginCommandBuffer(cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer");

            VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(
                cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
                0, 1, descriptorSet, 0, 0);

            Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)topK);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)hiddenSize);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            uint groupX = (uint)((hiddenSize + WorkgroupX - 1) / WorkgroupX);
            uint groupY = (uint)((seqLen + WorkgroupY - 1) / WorkgroupY);
            VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);

            VulkanApi.vkEndCommandBuffer(cmdBuf).ThrowOnError("vkEndCommandBuffer");

            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBuf),
            };
            VulkanApi.vkQueueSubmit(_device.Queue, 1, submit, 0).ThrowOnError("vkQueueSubmit");
            VulkanApi.vkQueueWaitIdle(_device.Queue).ThrowOnError("vkQueueWaitIdle");
        }
        finally
        {
            VulkanApi.vkFreeCommandBuffers(_device.Handle, _device.CommandPool, 1, cmdBuf);
            // Pool-reset after the queue wait frees the descriptor set for the next Launch();
            // without this the single-set pool exhausts on the second invocation.
            VulkanApi.vkResetDescriptorPool(_device.Handle, _descriptorPool, 0);
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
