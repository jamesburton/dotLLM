using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place per-head L2 normalisation used by Qwen3MoeHybrid GDN token mixing
/// (step 4 of <c>ForwardGdnBody</c>): the Q and K tensors must be unit-norm per
/// head before the GDN delta-rule scan. Mirrors
/// <see cref="DotLLM.Cpu.Kernels.GatedDeltaNetScan.L2NormalizeHeads"/> bit-for-bit.
/// </summary>
/// <remarks>
/// <para>
/// For each head <c>h</c> in <c>[0, totalHeads)</c>, the kernel computes
/// <c>invN = 1 / (sqrt(sumSq) + eps)</c> (epsilon OUTSIDE the sqrt, unlike
/// <see cref="RmsNormF32Kernel"/>) and scales the head's <c>dState</c> elements
/// in place. <c>totalHeads</c> covers all <c>seqLen * numHeads</c>
/// slices when the buffer holds the full <c>[seqLen, numHeads, dState]</c>
/// tensor — the kernel only sees a flat <c>[totalHeads, dState]</c> layout.
/// </para>
/// <para>
/// The matching shader <c>gdn_l2_normalize_heads_f32.comp</c> uses one
/// workgroup per head with <c>local_size_x = 128</c>; <c>dState</c>
/// is read from a push constant so smaller widths still work (the shader
/// strides over <c>[0, dState)</c>).
/// </para>
/// </remarks>
public sealed class GdnL2NormalizeHeadsF32Kernel : IDisposable
{
    private const int WorkgroupX = 128;
    // totalHeads (u32), dState (u32), eps (f32)
    private const int PushConstantBytes = 2 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnL2NormalizeHeadsF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 1);
    }

    /// <summary>Loads <c>gdn_l2_normalize_heads_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnL2NormalizeHeadsF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_l2_normalize_heads_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[1];
            bindings[0] = new VkDescriptorBinding(0);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1);
        return new GdnL2NormalizeHeadsF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer x, int totalHeads, int dState, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, totalHeads, dState, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the in-place per-head L2 normalisation dispatch.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="x">F32 buffer of shape <c>[totalHeads, dState]</c> row-major. Normalized in place.</param>
    /// <param name="totalHeads">Number of head slices (typically <c>seqLen * numHeads</c>).</param>
    /// <param name="dState">Per-head state width.</param>
    /// <param name="eps">Stabilising constant added OUTSIDE the sqrt (matches CPU reference).</param>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer x, int totalHeads, int dState, float eps)
    {
        if (totalHeads <= 0) throw new ArgumentOutOfRangeException(nameof(totalHeads));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));

        long bytes = (long)totalHeads * dState * sizeof(float);
        if (x.Size < bytes) throw new ArgumentException("x buffer too small.", nameof(x));

        Span<nint> buffers = stackalloc nint[1] { x.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)totalHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[8..], eps);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per head — shader is dispatched as (totalHeads, 1, 1).
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)totalHeads, 1, 1);
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
