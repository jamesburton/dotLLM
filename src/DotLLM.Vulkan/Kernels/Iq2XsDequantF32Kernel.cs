using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ2_XS → FP32 dequantization. Reads a tightly-packed IQ2_XS blob and
/// produces a contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ2_XS</c> and
/// llama.cpp's <c>block_iq2_xs</c>: 74 bytes per 256-element super-block,
/// 512-entry codebook + 7-bit sign indices in the upper bits of each qs uint16.
/// </remarks>
public sealed class Iq2XsDequantF32Kernel : IDisposable
{
    /// <summary>IQ2_XS super-block: 2 + 64 + 8 = 74 bytes.</summary>
    public const int IQ2_XSBlockBytes = 74;

    /// <summary>Elements per IQ2_XS super-block.</summary>
    public const int IQ2_XSGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq2Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private Iq2XsDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads the .spv from the given directory, allocates the shared codebook SSBOs, and creates the pipeline.</summary>
    public static Iq2XsDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq2Codebooks.Create(device);
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

    internal static Iq2XsDequantF32Kernel CreateWithCodebooks(VulkanDevice device, string spvDir, Iq2Codebooks codebooks)
        => CreateInternal(device, spvDir, codebooks, ownsCodebooks: false);

    private static Iq2XsDequantF32Kernel CreateInternal(VulkanDevice device, string spvDir, Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, "iq2_xs_dequant_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
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
        return new Iq2XsDequantF32Kernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
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

        long srcMin = (long)totalSuperblocks * IQ2_XSBlockBytes;
        long dstMin = (long)totalSuperblocks * IQ2_XSGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[4]
        {
            src.Handle,
            dst.Handle,
            _codebooks.Iq2XsGrid.Handle,
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
