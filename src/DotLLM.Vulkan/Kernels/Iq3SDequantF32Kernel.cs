using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ3_S → FP32 dequantization. Reads a tightly-packed IQ3_S blob and
/// produces a contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ3_S</c> and
/// llama.cpp's <c>block_iq3_s</c>: 110 bytes per 256-element super-block,
/// 512-entry codebook (4 uint8 grid points per entry), explicit 8-bit sign
/// mask per pair (no ksigns indirection).
/// </remarks>
public sealed class Iq3SDequantF32Kernel : IDisposable
{
    /// <summary>IQ3_S super-block: 2 + 64 + 8 + 32 + 4 = 110 bytes.</summary>
    public const int IQ3_SBlockBytes = 110;

    /// <summary>Elements per IQ3_S super-block.</summary>
    public const int IQ3_SGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq3Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private Iq3SDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads <c>iq3_s_dequant_f32.spv</c> from the given directory, allocates the
    /// shared codebook SSBOs, and creates the pipeline.</summary>
    public static Iq3SDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq3Codebooks.Create(device);
        try { return CreateInternal(device, spvDir, codebooks, ownsCodebooks: true); }
        catch { codebooks.Dispose(); throw; }
    }

    internal static Iq3SDequantF32Kernel CreateWithCodebooks(VulkanDevice device, string spvDir, Iq3Codebooks codebooks)
        => CreateInternal(device, spvDir, codebooks, ownsCodebooks: false);

    private static Iq3SDequantF32Kernel CreateInternal(VulkanDevice device, string spvDir, Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, "iq3_s_dequant_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            for (int i = 0; i < 3; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new Iq3SDequantF32Kernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
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

        long srcMin = (long)totalSuperblocks * IQ3_SBlockBytes;
        long dstMin = (long)totalSuperblocks * IQ3_SGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[3]
        {
            src.Handle, dst.Handle, _codebooks.Iq3SGrid.Handle,
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
