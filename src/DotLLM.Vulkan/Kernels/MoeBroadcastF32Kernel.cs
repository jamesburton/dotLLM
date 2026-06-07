using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE broadcast — expands per-token <c>NormOutput</c> into the per-(token,
/// slot) expanded input that feeds the indexed expert matmul:
/// <code>
///     out[n, h] = in[n / topK, h]
/// </code>
/// where the output is <c>[seqLen * topK, hiddenSize]</c> and the input is
/// <c>[seqLen, hiddenSize]</c>. Replaces the
/// <c>seqLen × topK</c> loop of <c>vkCmdCopyBuffer</c> regions (or the host
/// map/memcpy fallback in the scaffold path) in the MoE forward path with a
/// single compute dispatch — one transfer↔compute stage transition pair
/// drops out and the per-region command-recording overhead disappears at
/// prefill.
/// </summary>
/// <remarks>
/// <para>
/// Math is bit-exact wrt the copy loop it replaces — no FP arithmetic, just
/// a strided F32 load/store. Tolerance for the parity tests is therefore
/// zero (bit equality).
/// </para>
/// <para>
/// Inline-pool one-shot Launch convention (matches
/// <c>MoeWeightedScatterF32Kernel</c> / <c>MoeTopKSoftmaxF32Kernel</c>):
/// each <see cref="Launch"/> allocates a fresh descriptor set, records a
/// one-time-submit command buffer, waits on the queue, then resets the
/// descriptor pool.
/// </para>
/// </remarks>
public sealed class MoeBroadcastF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, topK, hidden (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MoeBroadcastF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>moe_broadcast_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeBroadcastF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_broadcast_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
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
        return new MoeBroadcastF32Kernel(device, module, pipeline, pool);
    }

    private static unsafe nint CreateDescriptorPool(VulkanDevice device)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = 2,
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
    /// Synchronous dispatch of the MoE broadcast expansion. Returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="input">Per-token input [<paramref name="seqLen"/> × <paramref name="hidden"/>] F32 row-major.</param>
    /// <param name="output">Per-(token, slot) expanded output [<paramref name="seqLen"/> × <paramref name="topK"/> × <paramref name="hidden"/>] F32 row-major; fully overwritten.</param>
    /// <param name="seqLen">Number of source tokens.</param>
    /// <param name="topK">Per-token replication factor (1 ≤ topK).</param>
    /// <param name="hidden">Per-token feature dim.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int seqLen, int topK, int hidden)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));

        long inBytes = (long)seqLen * hidden * sizeof(float);
        long outBytes = (long)seqLen * topK * hidden * sizeof(float);
        if (input.Size < inBytes) throw new ArgumentException("input buffer too small.", nameof(input));
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
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[2];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = input.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = output.Handle, offset = 0, range = ulong.MaxValue };

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[2];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < 2; i++)
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
                VulkanApi.vkUpdateDescriptorSets(_device.Handle, 2, (nint)writesPtr, 0, 0);
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
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)hidden);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            long expandedRows = (long)seqLen * topK;
            uint groupX = (uint)((hidden + WorkgroupX - 1) / WorkgroupX);
            uint groupY = (uint)((expandedRows + WorkgroupY - 1) / WorkgroupY);
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
