using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MLA kv_b split kernel. After <c>kv_b_proj</c> produces a fused
/// <c>[seqLen, numHeads * (qkNope + vHead)]</c> tensor with each head's
/// block laid out as <c>[qkNope | vHead]</c>, this kernel splits it
/// into two densely-packed buffers — per-head <c>K_nope</c> and per-head
/// <c>V</c> — that the downstream RoPE / cache / attention kernels expect.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the per-head copy loop in
/// <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>. One workgroup per
/// token; threads stride over the per-token element count and dispatch
/// each element to <c>K_nope</c> or <c>V</c> based on its position within
/// the head's block.
/// </para>
/// <para>
/// FORWARD-REFERENCE ADAPTATION: the source feature branch used a
/// <c>DescriptorSetCache</c> + persistent <c>SubmitContext</c>; the
/// upstream Vulkan scaffold (PR #205) has neither. The kernel is wired to
/// the inline-pool / inline-submit pattern instead — same precedent as
/// <c>BiasAddF32Kernel</c>.
/// </para>
/// </remarks>
public sealed class MlaKvSplitF32Kernel : IDisposable
{
    // seqLen, numHeads, qkNopeHeadDim, vHeadDim (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MlaKvSplitF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>mla_kv_split_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MlaKvSplitF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "mla_kv_split_f32.spv");
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
        return new MlaKvSplitF32Kernel(device, module, pipeline, pool);
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
    /// Synchronous launch — splits the fused <paramref name="kvBExpanded"/>
    /// tensor into per-head <paramref name="kNope"/> and <paramref name="v"/>.
    /// </summary>
    /// <param name="kvBExpanded">Fused [seqLen, numHeads * (qkNope + vHead)] (FP32).</param>
    /// <param name="kNope">Output [seqLen, numHeads * qkNopeHeadDim] (FP32).</param>
    /// <param name="v">Output [seqLen, numHeads * vHeadDim] (FP32).</param>
    /// <param name="seqLen">Number of tokens (rows) in the inputs.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="qkNopeHeadDim">Non-rope Q·K sub-dim per head — width of the K_nope slot.</param>
    /// <param name="vHeadDim">V head dim — width of the V slot.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer kvBExpanded, VulkanDevice.Buffer kNope, VulkanDevice.Buffer v,
        int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));

        long expBytes = (long)seqLen * numHeads * (qkNopeHeadDim + vHeadDim) * sizeof(float);
        long kNopeBytes = (long)seqLen * numHeads * qkNopeHeadDim * sizeof(float);
        long vBytes = (long)seqLen * numHeads * vHeadDim * sizeof(float);
        if (kvBExpanded.Size < expBytes) throw new ArgumentException("kvBExpanded buffer too small.", nameof(kvBExpanded));
        if (kNope.Size < kNopeBytes) throw new ArgumentException("kNope buffer too small.", nameof(kNope));
        if (v.Size < vBytes) throw new ArgumentException("V buffer too small.", nameof(v));

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
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = kvBExpanded.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = kNope.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = v.Handle, offset = 0, range = ulong.MaxValue };

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
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)numHeads);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)qkNopeHeadDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)vHeadDim);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            // One workgroup per token.
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)seqLen, 1, 1);

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
