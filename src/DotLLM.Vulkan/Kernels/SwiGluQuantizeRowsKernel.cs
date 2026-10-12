using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused gated activation + row-wise Q8_1 quantization for the generic-MoE decode arm (#912): <c>xq/xds = Q8_1(act(gate, up))</c> in one dispatch,
/// where <c>act</c> is SwiGLU or the gpt-oss clamped SwiGLU, replacing the activation → barrier → <see cref="QuantizeQ8_1RowsKernel"/> pair in
/// front of the routed-expert down projection. Rows are contiguous runs of 32-blocks (row length a multiple of 32) so the whole
/// <c>[rows * K]</c> activation is processed as one flat run, one thread per block. Bit-identical to the standalone pair (the F32 activation is
/// simply not materialised; LoRA, which reads it, never takes the MMVQ arm).
/// </summary>
public sealed class SwiGluQuantizeRowsKernel : IDisposable
{
    /// <summary>Elements per Q8_1 block.</summary>
    public const int GroupSize = QuantFormat.LegacyGroupSize;

    private const int BuffersPerSet = 4;
    private const int PushConstantBytes = 4 * sizeof(uint); // blocks, mode, alpha, limit
    private const int ThreadsPerGroup = 64;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SwiGluQuantizeRowsKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>Loads <c>swiglu_quantize_q8_1_rows.spv</c>; <c>null</c> when the SPIR-V is missing.</summary>
    public static SwiGluQuantizeRowsKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "swiglu_quantize_q8_1_rows.spv");
        if (!File.Exists(path))
            return null;
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BuffersPerSet];
            for (int i = 0; i < BuffersPerSet; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new SwiGluQuantizeRowsKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records <c>Q8_1(act(gate, up))</c> over <paramref name="n"/> elements (<paramref name="n"/> a multiple of 32; the quantization rows are the
    /// consecutive 32-blocks, so any row length that divides into 32-blocks gives the same bytes as the per-row quantizer).
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="gate">F32 gate, <c>[n]</c>.</param>
    /// <param name="up">F32 up, <c>[n]</c>.</param>
    /// <param name="xq">Packed int8 out, <c>n/4</c> uints.</param>
    /// <param name="xds">(scale, sum) out, <c>n/32</c> vec2.</param>
    /// <param name="n">Element count.</param>
    /// <param name="oai"><c>true</c> = gpt-oss clamped SwiGLU (alpha 1.702, limit 7), <c>false</c> = plain SwiGLU.</param>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer gate, VulkanDevice.Buffer up, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int n, bool oai)
    {
        if (n <= 0 || (n % GroupSize) != 0) throw new ArgumentException($"n must be a positive multiple of {GroupSize}, got {n}", nameof(n));
        long bytes = (long)n * sizeof(float);
        if (gate.Size < bytes) throw new ArgumentException("Gate buffer too small.", nameof(gate));
        if (up.Size < bytes) throw new ArgumentException("Up buffer too small.", nameof(up));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(1, n)) throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(1, n)) throw new ArgumentException("Scale output buffer too small.", nameof(xds));

        Span<nint> buffers = stackalloc nint[BuffersPerSet] { gate.Handle, up.Handle, xq.Handle, xds.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        int blocks = n / GroupSize;
        Span<uint> pc = stackalloc uint[4]
        {
            (uint)blocks, oai ? 1u : 0u,
            BitConverter.SingleToUInt32Bits(VulkanGptOssKernels.SwiGluOaiAlpha), BitConverter.SingleToUInt32Bits(VulkanGptOssKernels.SwiGluOaiLimit),
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((blocks + ThreadsPerGroup - 1) / ThreadsPerGroup), 1, 1);
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
