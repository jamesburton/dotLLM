using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Full FP32 RMS Normalization: <c>output = (input / rms(input)) * weight</c>
/// with <c>rms = sqrt(mean(x^2) + eps)</c>. Processes a batch of rows in a
/// single launch — one workgroup per row.
/// </summary>
/// <remarks>
/// Mirrors the CUDA kernel <c>rmsnorm_f32</c> in
/// <c>native/kernels/rmsnorm_f32.cu</c> and matches the algorithm used by the
/// CPU path (sum-of-squares, divide by length, add epsilon under the sqrt).
/// </remarks>
public sealed class RmsNormF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint) + sizeof(float); // n, eps

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private RmsNormF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>rmsnorm_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static RmsNormF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rmsnorm_f32.spv");
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
        return new RmsNormF32Kernel(device, module, pipeline, pool);
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
    /// Dispatches RMS norm over <paramref name="rowCount"/> rows of length
    /// <paramref name="n"/>. Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="input">FP32 input buffer, <c>[rowCount, n]</c> row-major.</param>
    /// <param name="weight">FP32 per-feature scale, <c>[n]</c>.</param>
    /// <param name="output">FP32 output buffer, <c>[rowCount, n]</c> row-major.</param>
    /// <param name="rowCount">Number of rows to normalize.</param>
    /// <param name="n">Row length (number of features).</param>
    /// <param name="eps">Epsilon under the square root. Typical: 1e-5 or 1e-6.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer output,
        int rowCount, int n, float eps)
    {
        if (rowCount <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        long rowBytes = (long)n * sizeof(float);
        if (input.Size < rowBytes * rowCount) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (weight.Size < rowBytes) throw new ArgumentException("Weight buffer too small.", nameof(weight));
        if (output.Size < rowBytes * rowCount) throw new ArgumentException("Output buffer too small.", nameof(output));

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
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = input.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = weight.Handle, offset = 0, range = ulong.MaxValue };
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

            // Push constants: uint n, float eps (8 bytes total).
            Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)n);
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[4..], eps);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            // One workgroup per row.
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)rowCount, 1, 1);

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
