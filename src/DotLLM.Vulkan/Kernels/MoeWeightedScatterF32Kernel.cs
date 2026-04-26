using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE weighted scatter — final combine step of a MoE forward pass.
/// Collapses the per-(token, slot) expert outputs into per-token outputs
/// by summing each token's <c>topK</c> rows scaled by its routing weights:
/// <code>
///     out[t, h] = sum_{slot=0..topK-1} weights[t * topK + slot] * x[t * topK + slot, h]
/// </code>
/// </summary>
/// <remarks>
/// Mirrors the per-(token, slot) accumulation tail of
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.Execute</c> — same per-token slot
/// order so the result is bit-stable wrt the CPU reference modulo F32
/// rounding upstream in the matmul / SwiGLU phases.
/// </remarks>
public sealed class MoeWeightedScatterF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, topK, hiddenSize (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeWeightedScatterF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>moe_weighted_scatter_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeWeightedScatterF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_weighted_scatter_f32.spv");
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
        return new MoeWeightedScatterF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer weights, VulkanDevice.Buffer output,
        int seqLen, int topK, int hiddenSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, weights, output, seqLen, topK, hiddenSize);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the weighted-scatter dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="x">Per-(token, slot) expert outputs [<paramref name="seqLen"/> × <paramref name="topK"/> × <paramref name="hiddenSize"/>] F32 row-major.</param>
    /// <param name="weights">Routing weights [<paramref name="seqLen"/> × <paramref name="topK"/>] F32 row-major.</param>
    /// <param name="output">Combined per-token output [<paramref name="seqLen"/> × <paramref name="hiddenSize"/>]; fully overwritten.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="topK">Per-token expert count (1 ≤ topK).</param>
    /// <param name="hiddenSize">Per-token feature dim.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer weights, VulkanDevice.Buffer output,
        int seqLen, int topK, int hiddenSize)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));

        long xBytes = (long)seqLen * topK * hiddenSize * sizeof(float);
        long wBytes = (long)seqLen * topK * sizeof(float);
        long outBytes = (long)seqLen * hiddenSize * sizeof(float);
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (weights.Size < wBytes) throw new ArgumentException("weights buffer too small.", nameof(weights));
        if (output.Size < outBytes) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[3] { x.Handle, weights.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)topK);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)hiddenSize);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((hiddenSize + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((seqLen + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
