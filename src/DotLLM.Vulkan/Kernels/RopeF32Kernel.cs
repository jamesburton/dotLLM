using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// RoPE (Rotary Position Embedding) kernel with FP32 Q/K data. Rotates Q and
/// K tensors in place by their token positions; frequencies are reconstructed
/// on the GPU from <c>theta</c> — no pre-computed cos/sin tables crossing the
/// P/Invoke boundary.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CUDA kernel <c>rope_f32</c> in
/// <c>native/kernels/rope_f32.cu</c>. One shader invocation per rotation pair;
/// Q and K are rotated in the same dispatch because their index ranges are
/// independent (GQA reduces the K range relative to Q).
/// </para>
/// <para>
/// Element-pairing variants:
/// <list type="bullet">
///   <item><b>Norm</b> (<c>ropeType = 0</c>): pair <c>(2i, 2i+1)</c> within a head — used by Llama-family, SmolLM, Phi-3.</item>
///   <item><b>NeoX</b> (<c>ropeType = 1</c>): pair <c>(i, i + halfRope)</c> — GPT-NeoX / HuggingFace <c>rotate_half</c>.</item>
/// </list>
/// </para>
/// </remarks>
public sealed class RopeF32Kernel : IDisposable
{
    /// <summary>RoPE element-pairing variant. Must match the model's RoPE convention.</summary>
    public enum Variant
    {
        /// <summary>Interleaved pairs <c>(2i, 2i+1)</c>. Llama-family, SmolLM, Phi-3.</summary>
        Norm = 0,
        /// <summary>Rotate-half pairs <c>(i, i + halfRope)</c>. GPT-NeoX / HuggingFace.</summary>
        NeoX = 1,
    }

    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 6 * sizeof(uint) + sizeof(float); // seqLen, numHeads, numKvHeads, headDim, ropeDim, ropeType, theta

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private RopeF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>rope_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static RopeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rope_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = CreateDescriptorPool(device);
        return new RopeF32Kernel(device, module, pipeline, pool);
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
    /// Applies RoPE to Q and K in place. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="q">Query buffer (FP32), layout <c>[seqLen, numHeads * headDim]</c>.</param>
    /// <param name="k">Key buffer (FP32), layout <c>[seqLen, numKvHeads * headDim]</c>.</param>
    /// <param name="positions">Position indices buffer (int32), length <paramref name="seqLen"/>.</param>
    /// <param name="seqLen">Number of query/key positions.</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="ropeDim">Number of dims to rotate per head (even, &lt;= headDim).</param>
    /// <param name="theta">RoPE base (typical 10000 for Llama-2, 500000 for Llama-3).</param>
    /// <param name="variant">Pair-layout variant.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer positions,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta,
        Variant variant = Variant.Norm)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (ropeDim <= 0 || (ropeDim & 1) != 0) throw new ArgumentException($"ropeDim must be a positive even integer, got {ropeDim}", nameof(ropeDim));
        if (ropeDim > headDim) throw new ArgumentException($"ropeDim ({ropeDim}) must be <= headDim ({headDim})", nameof(ropeDim));

        long qBytes = (long)seqLen * numHeads * headDim * sizeof(float);
        long kBytes = (long)seqLen * numKvHeads * headDim * sizeof(float);
        long posBytes = (long)seqLen * sizeof(int);
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (k.Size < kBytes) throw new ArgumentException("K buffer too small.", nameof(k));
        if (positions.Size < posBytes) throw new ArgumentException("Positions buffer too small.", nameof(positions));

        int halfRope = ropeDim / 2;
        long totalQ = (long)seqLen * numHeads * halfRope;
        long totalK = (long)seqLen * numKvHeads * halfRope;
        long maxPairs = Math.Max(totalQ, totalK);

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
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = q.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = k.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = positions.Handle, offset = 0, range = ulong.MaxValue };

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

            // Push constants: 6 uint + 1 float = 28 bytes.
            Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[0..],  (uint)seqLen);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],  (uint)numHeads);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],  (uint)numKvHeads);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)headDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)ropeDim);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[20..], (uint)variant);
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[24..], theta);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            uint groups = (uint)((maxPairs + WorkgroupSize - 1) / WorkgroupSize);
            VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);

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
