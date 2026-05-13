using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place per-element sigmoid-gated multiply:
/// <c>attnOut[i] *= 1 / (1 + exp(-gate[i]))</c>. Used by the Qwen3MoeHybrid
/// full-attention layer (step 8 of <c>ForwardAttnBody</c>) to fold the
/// per-element gate into the attention output before the O-projection.
/// </summary>
/// <remarks>
/// Pointwise, no reduction. One thread per element across both buffers.
/// Mirrors the CPU expression
/// <c>aRow[i] *= 1f / (1f + MathF.Exp(-gRow[i]))</c>.
/// </remarks>
public sealed class SigmoidGateMulF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint); // n_total

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SigmoidGateMulF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 2);
    }

    /// <summary>Loads <c>sigmoid_gate_mul_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SigmoidGateMulF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "sigmoid_gate_mul_f32.spv");
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
        return new SigmoidGateMulF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer attnOut, VulkanDevice.Buffer gate, int nTotal)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, attnOut, gate, nTotal);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the in-place sigmoid-gated multiply dispatch.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="attnOut">F32 buffer of length <paramref name="nTotal"/>, mutated in place.</param>
    /// <param name="gate">F32 buffer of length <paramref name="nTotal"/>, read-only.</param>
    /// <param name="nTotal">Total element count (typically <c>seqLen * numHeads * headDim</c>).</param>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer attnOut, VulkanDevice.Buffer gate, int nTotal)
    {
        if (nTotal <= 0) throw new ArgumentOutOfRangeException(nameof(nTotal));

        long bytes = (long)nTotal * sizeof(float);
        if (attnOut.Size < bytes) throw new ArgumentException("attnOut buffer too small.", nameof(attnOut));
        if (gate.Size < bytes) throw new ArgumentException("gate buffer too small.", nameof(gate));

        Span<nint> buffers = stackalloc nint[2] { attnOut.Handle, gate.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        uint pushN = (uint)nTotal;
        VulkanApi.vkCmdPushConstants(
            cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
            0, sizeof(uint), (nint)(&pushN));

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
