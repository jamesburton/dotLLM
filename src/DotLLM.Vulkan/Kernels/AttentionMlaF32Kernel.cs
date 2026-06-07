using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// FP32 Multi-head Latent Attention (DeepSeek-V2/V3) — post-projection
/// attention loop. Mirrors <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>'s
/// per-head SDPA: causal mask, online softmax, weighted V sum.
/// </summary>
/// <remarks>
/// <para>
/// MLA differs from regular MHA/GQA in three ways the kernel handles:
/// <list type="bullet">
///   <item>Q has two contiguous sub-dims per head: <c>q_nope</c> (no
///     positional encoding) and <c>q_pe</c> (RoPE-rotated). Total Q head
///     dim is <c>qk_nope_head_dim + qk_rope_head_dim</c>.</item>
///   <item>K_pe is MQA-style shared across all heads — one rope-K per
///     token instead of per-head. Stored in its own buffer.</item>
///   <item>V uses its own head dim (<c>v_head_dim</c>) which may differ
///     from <c>qk_head_dim</c>. Output is per-head <c>v_head_dim</c>.</item>
/// </list>
/// </para>
/// <para>
/// First-pass implementation uses the shared-memory online-softmax variant
/// (no subgroup / coopmat tiling). Adding those variants is a follow-up
/// once the integration path is wired and end-to-end DeepSeek-V2-Lite
/// argmax parity is locked.
/// </para>
/// <para>
/// FORWARD-REFERENCE ADAPTATION: the source feature branch used a
/// <c>DescriptorSetCache</c> + a persistent <c>SubmitContext</c>; the
/// upstream Vulkan scaffold (PR #205) has neither. The kernel is wired to
/// the inline-pool / inline-submit pattern instead — same precedent as
/// <c>BiasAddF32Kernel</c>. The Vulkan perf chain restores both
/// <c>DescriptorSetCache</c> and <c>SubmitContext</c>; at that point a
/// caller-driven <c>Record</c> path can be re-introduced.
/// </para>
/// </remarks>
public sealed class AttentionMlaF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on (qk_nope + qk_rope) per head — must mirror <c>MAX_QK_HEAD_DIM</c> in the shader.</summary>
    public const int MaxQkHeadDim = 256;

    /// <summary>Compile-time upper bound on v_head_dim — must mirror <c>MAX_V_HEAD_DIM</c> in the shader.</summary>
    public const int MaxVHeadDim = 256;

    // seqQ, seqKv, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim, positionOffset (u32) + scale (f32)
    private const int PushConstantBytes = 7 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private AttentionMlaF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>attention_mla_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static AttentionMlaF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "attention_mla_f32.spv");
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

        nint pool = CreateDescriptorPool(device);
        return new AttentionMlaF32Kernel(device, module, pipeline, pool);
    }

    private static unsafe nint CreateDescriptorPool(VulkanDevice device)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = 5,
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
    /// Synchronous launch — allocates a descriptor set, records the dispatch
    /// into a one-time command buffer, submits, and waits for completion.
    /// Resets the descriptor pool at the end so the single-set pool is
    /// re-usable on the next call.
    /// </summary>
    public unsafe void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer kNope, VulkanDevice.Buffer v,
        VulkanDevice.Buffer kPe, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float scale)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (qkRopeHeadDim < 0) throw new ArgumentOutOfRangeException(nameof(qkRopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        if (qkHeadDim > MaxQkHeadDim)
            throw new ArgumentException(
                $"qkNopeHeadDim + qkRopeHeadDim must be ≤ {MaxQkHeadDim}, got {qkHeadDim}.", nameof(qkNopeHeadDim));
        if (vHeadDim > MaxVHeadDim)
            throw new ArgumentException($"vHeadDim must be ≤ {MaxVHeadDim}, got {vHeadDim}.", nameof(vHeadDim));

        long qBytes = (long)seqQ * numHeads * qkHeadDim * sizeof(float);
        long kNopeBytes = (long)seqKv * numHeads * qkNopeHeadDim * sizeof(float);
        long vBytes = (long)seqKv * numHeads * vHeadDim * sizeof(float);
        long kPeBytes = (long)seqKv * qkRopeHeadDim * sizeof(float);
        long outBytes = (long)seqQ * numHeads * vHeadDim * sizeof(float);
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (kNope.Size < kNopeBytes) throw new ArgumentException("K_nope buffer too small.", nameof(kNope));
        if (v.Size < vBytes) throw new ArgumentException("V buffer too small.", nameof(v));
        if (kPe.Size < kPeBytes) throw new ArgumentException("K_pe buffer too small.", nameof(kPe));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

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
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[5];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = q.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = kNope.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = v.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[3] = new VkDescriptorBufferInfo { buffer = kPe.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[4] = new VkDescriptorBufferInfo { buffer = output.Handle, offset = 0, range = ulong.MaxValue };

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[5];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < 5; i++)
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
                VulkanApi.vkUpdateDescriptorSets(_device.Handle, 5, (nint)writesPtr, 0, 0);
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
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqQ);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)seqKv);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)numHeads);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)qkNopeHeadDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)qkRopeHeadDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[20..], (uint)vHeadDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[24..], (uint)positionOffset);
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[28..], scale);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            // One workgroup per (tq, hq) pair.
            uint groupCount = (uint)seqQ * (uint)numHeads;
            VulkanApi.vkCmdDispatch(cmdBuf, groupCount, 1, 1);

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
