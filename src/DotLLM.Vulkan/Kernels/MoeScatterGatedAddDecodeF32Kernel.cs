using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE single-token combine (#885): <c>out[h] = sum_slot weights[slot] * x[slot, h] + sigmoid(gateLogit) * b[h]</c> in one dispatch - the weighted scatter and the
/// shared-expert sigmoid-gated add of a decode row, which were two badly shaped dispatches (16 live threads per 256-thread group; one 64-thread workgroup over the row).
/// </summary>
public sealed class MoeScatterGatedAddDecodeF32Kernel : IDisposable
{
    private const int Workgroup = 256;
    // topK, hiddenSize (u32)
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeScatterGatedAddDecodeF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>Loads <c>moe_scatter_gated_add_decode_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeScatterGatedAddDecodeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_scatter_gated_add_decode_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < 5; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MoeScatterGatedAddDecodeF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch - wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer x, VulkanDevice.Buffer weights, VulkanDevice.Buffer shared, VulkanDevice.Buffer gateLogit,
        VulkanDevice.Buffer output, int topK, int hiddenSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, weights, shared, gateLogit, output, topK, hiddenSize);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the fused combine for ONE token into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="x">Per-slot expert outputs <c>[topK * hiddenSize]</c>.</param>
    /// <param name="weights">Routing weights <c>[topK]</c>.</param>
    /// <param name="shared">Shared-expert output <c>[hiddenSize]</c>.</param>
    /// <param name="gateLogit">Raw (pre-sigmoid) shared-expert gate logit <c>[1]</c>.</param>
    /// <param name="output">Combined output <c>[hiddenSize]</c>; fully overwritten.</param>
    /// <param name="topK">Experts per token.</param>
    /// <param name="hiddenSize">Feature dim.</param>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer x, VulkanDevice.Buffer weights, VulkanDevice.Buffer shared,
        VulkanDevice.Buffer gateLogit, VulkanDevice.Buffer output, int topK, int hiddenSize)
    {
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (x.Size < (long)topK * hiddenSize * sizeof(float)) throw new ArgumentException("x buffer too small.", nameof(x));
        if (weights.Size < (long)topK * sizeof(float)) throw new ArgumentException("weights buffer too small.", nameof(weights));
        if (shared.Size < (long)hiddenSize * sizeof(float)) throw new ArgumentException("shared buffer too small.", nameof(shared));
        if (gateLogit.Size < sizeof(float)) throw new ArgumentException("gate logit buffer too small.", nameof(gateLogit));
        if (output.Size < (long)hiddenSize * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { x.Handle, weights.Handle, shared.Handle, gateLogit.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)topK, (uint)hiddenSize };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((hiddenSize + Workgroup - 1) / Workgroup), 1, 1);
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
