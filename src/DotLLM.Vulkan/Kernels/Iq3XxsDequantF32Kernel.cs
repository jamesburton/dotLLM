using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ3_XXS → FP32 dequantization. Reads a tightly-packed IQ3_XXS blob and
/// produces a contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ3_XXS</c> and
/// llama.cpp's <c>block_iq3_xxs</c>: 98 bytes per 256-element super-block,
/// 256-entry codebook (4 uint8 grid points per entry) + 7-bit sign indices
/// resolved via the shared <c>ksigns_iq2xs</c> table.
/// </remarks>
public sealed class Iq3XxsDequantF32Kernel : IDisposable
{
    /// <summary>IQ3_XXS super-block: 2 + 64 + 32 = 98 bytes.</summary>
    public const int IQ3_XXSBlockBytes = 98;

    /// <summary>Elements per IQ3_XXS super-block.</summary>
    public const int IQ3_XXSGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq3Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private Iq3XxsDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads <c>iq3_xxs_dequant_f32.spv</c> from the given directory, allocates the
    /// shared codebook SSBOs, and creates the pipeline.</summary>
    public static Iq3XxsDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq3Codebooks.Create(device);
        try
        {
            return CreateInternal(device, spvDir, codebooks, ownsCodebooks: true);
        }
        catch
        {
            codebooks.Dispose();
            throw;
        }
    }

    internal static Iq3XxsDequantF32Kernel CreateWithCodebooks(VulkanDevice device, string spvDir, Iq3Codebooks codebooks)
        => CreateInternal(device, spvDir, codebooks, ownsCodebooks: false);

    private static Iq3XxsDequantF32Kernel CreateInternal(VulkanDevice device, string spvDir, Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, "iq3_xxs_dequant_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < 4; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new Iq3XxsDequantF32Kernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the dequant synchronously.</summary>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalSuperblocks)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, totalSuperblocks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalSuperblocks)
    {
        if (totalSuperblocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalSuperblocks));

        long srcMin = (long)totalSuperblocks * IQ3_XXSBlockBytes;
        long dstMin = (long)totalSuperblocks * IQ3_XXSGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[4]
        {
            src.Handle,
            dst.Handle,
            _codebooks.Iq3XxsGrid.Handle,
            _codebooks.Ksigns.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)totalSuperblocks, (uint)srcUints };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)totalSuperblocks, 1, 1);
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
        if (_ownsCodebooks) _codebooks.Dispose();
    }
}
