using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place per-head sigmoid-gated multiply: <c>attnOut[t, h, d] *= 1 / (1 + exp(-gate[t, h]))</c>. LFM3 attention
/// (llama.cpp <c>lfm3.cpp</c>) has one gate value per (token, head), broadcast over <c>head_dim</c>; the per-element
/// <see cref="SigmoidGateMulF32Kernel"/> does not fit.
/// </summary>
/// <remarks>
/// Pointwise, no reduction. One thread per <c>attnOut</c> element.
/// </remarks>
public sealed class SigmoidGateMulHeadsF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 2 * sizeof(uint); // n_total, head_dim

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SigmoidGateMulHeadsF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>Loads <c>sigmoid_gate_mul_heads_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SigmoidGateMulHeadsF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "sigmoid_gate_mul_heads_f32.spv");
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new SigmoidGateMulHeadsF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer attnOut, VulkanDevice.Buffer gate, int nTotal, int headDim)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, attnOut, gate, nTotal, headDim);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the in-place sigmoid-gated multiply dispatch.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="attnOut">F32 buffer of length <paramref name="nTotal"/>, mutated in place.</param>
    /// <param name="gate">F32 buffer of length <c>nTotal / headDim</c>, read-only.</param>
    /// <param name="nTotal">Total element count of <paramref name="attnOut"/> (<c>seqLen * numHeads * headDim</c>).</param>
    /// <param name="headDim">Elements sharing one gate value; <paramref name="gate"/> holds <c>nTotal / headDim</c> values.</param>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer attnOut, VulkanDevice.Buffer gate, int nTotal, int headDim)
    {
        if (nTotal <= 0) throw new ArgumentOutOfRangeException(nameof(nTotal));
        if (headDim <= 0 || nTotal % headDim != 0) throw new ArgumentException($"headDim {headDim} must divide nTotal {nTotal}.", nameof(headDim));

        long bytes = (long)nTotal * sizeof(float);
        if (attnOut.Size < bytes) throw new ArgumentException("attnOut buffer too small.", nameof(attnOut));
        if (gate.Size < bytes / headDim) throw new ArgumentException("gate buffer too small.", nameof(gate));

        Span<nint> buffers = stackalloc nint[2] { attnOut.Handle, gate.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> push = stackalloc uint[2] { (uint)nTotal, (uint)headDim };
        fixed (uint* pPush = push)
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, 2 * sizeof(uint), (nint)pPush);

        uint groups = (uint)((nTotal + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
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
