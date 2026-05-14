using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ3_XXS decode-path GEMV: <c>y[M] = W_iq3xxs[M,K] @ x[K]</c>.
/// </summary>
public sealed class MatMulIq3XxsGemvF32Kernel : IDisposable
{
    /// <summary>IQ3_XXS super-block size in bytes.</summary>
    public const int IQ3_XXSBlockBytes = 98;
    /// <summary>Elements per IQ3_XXS super-block.</summary>
    public const int IQ3_XXSGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq3Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private MatMulIq3XxsGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads <c>matmul_iq3_xxs_f32_gemv.spv</c>, allocates the shared codebook
    /// SSBOs, and creates the pipeline.</summary>
    public static MatMulIq3XxsGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq3Codebooks.Create(device);
        try { return CreateInternal(device, spvDir, codebooks, ownsCodebooks: true); }
        catch { codebooks.Dispose(); throw; }
    }

    internal static MatMulIq3XxsGemvF32Kernel CreateWithCodebooks(VulkanDevice device, string spvDir, Iq3Codebooks codebooks)
        => CreateInternal(device, spvDir, codebooks, ownsCodebooks: false);

    private static MatMulIq3XxsGemvF32Kernel CreateInternal(VulkanDevice device, string spvDir, Iq3Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, "matmul_iq3_xxs_f32_gemv.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < 5; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MatMulIq3XxsGemvF32Kernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weights, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % IQ3_XXSGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {IQ3_XXSGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / IQ3_XXSGroupSize;
        long rowBytes = (long)blocksPerRow * IQ3_XXSBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weights.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weights.Size}.", nameof(weights));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
        {
            weights.Handle, x.Handle, y.Handle, _codebooks.Iq3XxsGrid.Handle, _codebooks.Ksigns.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, (uint)rowUints };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
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
