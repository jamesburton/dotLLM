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
/// Mirrors the per-head copy loop in
/// <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>. One workgroup per
/// token; threads stride over the per-token element count and dispatch
/// each element to <c>K_nope</c> or <c>V</c> based on its position within
/// the head's block.
/// </remarks>
public sealed class MlaKvSplitF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    // seqLen, numHeads, qkNopeHeadDim, vHeadDim (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MlaKvSplitF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MlaKvSplitF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer kvBExpanded, VulkanDevice.Buffer kNope, VulkanDevice.Buffer v,
        int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, kvBExpanded, kNope, v, seqLen, numHeads, qkNopeHeadDim, vHeadDim);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the split dispatch. <paramref name="kvBExpanded"/> layout is
    /// <c>[seqLen, numHeads * (qkNopeHeadDim + vHeadDim)]</c>; outputs are
    /// <c>kNope[seqLen, numHeads * qkNopeHeadDim]</c> and
    /// <c>v[seqLen, numHeads * vHeadDim]</c>.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
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

        Span<nint> buffers = stackalloc nint[3] { kvBExpanded.Handle, kNope.Handle, v.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

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
