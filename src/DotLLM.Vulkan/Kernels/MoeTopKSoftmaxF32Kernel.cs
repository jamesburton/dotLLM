using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE router top-k softmax kernel — for each token, full softmax over
/// router logits then picks the top-k entries (stable on ties, lower
/// index wins). Optional renormalisation of the top-k weights to sum
/// to 1.0 (Mixtral / Qwen3-MoE convention; Qwen1.5-MoE uses raw probs).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the routing path in
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.SelectTopK</c>. The top-k selection
/// is sequential (single-threaded inside each workgroup) to preserve the
/// lower-index-wins-on-tie property — a parallel reduction would not.
/// For typical Mixtral / Qwen-MoE / Phi-3.5-MoE shapes (numExperts up
/// to 64, k up to 8) the cost is trivial.
/// </para>
/// <para>
/// Build block of the Vulkan MoE forward path (issue #4): the matmul
/// kernels (gate projection, expert MLPs) are reused; this is the only
/// genuinely new compute kernel. Forward-pass orchestration is a
/// follow-up.
/// </para>
/// <para>
/// Adapted to the inline-pool pattern: each <see cref="Launch"/> allocates a
/// fresh descriptor set, records a one-time-submit command buffer, waits on
/// the queue, then resets the descriptor pool. The forthcoming
/// <c>DescriptorSetCache</c>-based variant will be wired in once that
/// infrastructure lands on <c>main</c>.
/// </para>
/// </remarks>
public sealed class MoeTopKSoftmaxF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on numExperts (mirrors <c>MAX_EXPERTS</c> in the shader).</summary>
    public const int MaxExperts = 256;

    /// <summary>Compile-time upper bound on top-k (mirrors <c>MAX_K</c> in the shader).</summary>
    public const int MaxK = 16;

    // seqLen, numExperts, k, normTopKProb (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MoeTopKSoftmaxF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>moe_topk_softmax_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeTopKSoftmaxF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_topk_softmax_f32.spv");
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
        return new MoeTopKSoftmaxF32Kernel(device, module, pipeline, pool);
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
    /// Synchronous dispatch of the top-k softmax routing. Returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="logits">F32 router logits, layout <c>[seqLen, numExperts]</c> row-major.</param>
    /// <param name="indices">int32 top-k indices output, layout <c>[seqLen, k]</c> row-major.</param>
    /// <param name="weights">F32 top-k weights output, layout <c>[seqLen, k]</c> row-major.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="numExperts">Total experts per layer (must be in [1, <see cref="MaxExperts"/>]).</param>
    /// <param name="k">Top-k count (1 ≤ k ≤ <see cref="MaxK"/> ≤ numExperts).</param>
    /// <param name="normTopKProb">When <c>true</c>, divides the picked weights by their sum.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer logits, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights,
        int seqLen, int numExperts, int k, bool normTopKProb)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numExperts <= 0 || numExperts > MaxExperts)
            throw new ArgumentOutOfRangeException(nameof(numExperts),
                $"numExperts must be in [1, {MaxExperts}], got {numExperts}.");
        if (k <= 0 || k > MaxK || k > numExperts)
            throw new ArgumentOutOfRangeException(nameof(k),
                $"k must be in [1, min({MaxK}, numExperts)], got {k} (numExperts={numExperts}).");

        long logitBytes = (long)seqLen * numExperts * sizeof(float);
        long idxBytes = (long)seqLen * k * sizeof(int);
        long wtBytes = (long)seqLen * k * sizeof(float);
        if (logits.Size < logitBytes) throw new ArgumentException("logits buffer too small.", nameof(logits));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (weights.Size < wtBytes) throw new ArgumentException("weights buffer too small.", nameof(weights));

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
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = logits.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = indices.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = weights.Handle, offset = 0, range = ulong.MaxValue };

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
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)numExperts);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)k);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], normTopKProb ? 1u : 0u);
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
