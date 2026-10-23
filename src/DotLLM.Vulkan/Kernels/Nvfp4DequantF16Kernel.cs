using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// NVFP4 (GGUF type 40) to FP16 dequantization (#1060): the NVFP4 member of <see cref="IqF16PrefillMatmul"/>'s dequantise-then-coopmat
/// prefill. The decode is exact in F16 (E2M1 x UE4M3 needs at most 6 mantissa bits and stays in F16's normal range). The per-tensor
/// <c>X.scale</c> is not applied: the caller multiplies the GEMM output by it.
/// </summary>
/// <remarks>Layout: 36-byte blocks of 64 elements (<c>nvfp4_decode.glsl</c>, llama.cpp <c>block_nvfp4</c>).</remarks>
public sealed class Nvfp4DequantF16Kernel : IDisposable
{
    private const int PushConstantBytes = sizeof(uint);
    private const int ElementsPerWorkgroup = 256;
    private const string SpvName = "nvfp4_dequant_f16.spv";

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Nvfp4DequantF16Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>True when the SPIR-V for this kernel is present in <paramref name="spvDir"/>.</summary>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <returns>Whether <see cref="Create"/> can succeed.</returns>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, SpvName));

    /// <summary>Loads <c>nvfp4_dequant_f16.spv</c> and creates the pipeline.</summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V is missing.</exception>
    public static Nvfp4DequantF16Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, SpvName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new Nvfp4DequantF16Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the dequant synchronously (tests).</summary>
    /// <param name="src">Packed NVFP4 blocks.</param>
    /// <param name="dst">F16 output, one element per weight.</param>
    /// <param name="totalBlocks">Number of 64-element blocks.</param>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalBlocks)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, totalBlocks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="src">Packed NVFP4 blocks.</param>
    /// <param name="dst">F16 output, one element per weight.</param>
    /// <param name="totalBlocks">Number of 64-element blocks.</param>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalBlocks)
    {
        if (totalBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalBlocks));
        long srcMin = (long)totalBlocks * QuantFormat.Nvfp4BlockBytes;
        long elements = (long)totalBlocks * QuantFormat.Nvfp4GroupSize;
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < elements * sizeof(ushort))
            throw new ArgumentException($"Destination buffer too small: need >= {elements * sizeof(ushort)} bytes.", nameof(dst));

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);
        uint pc = (uint)totalBlocks;
        VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)(&pc));
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((elements + ElementsPerWorkgroup - 1) / ElementsPerWorkgroup), 1, 1);
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
