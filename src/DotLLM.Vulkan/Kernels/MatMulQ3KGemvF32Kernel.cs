using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q3_K decode-path GEMV: <c>y[M] = W_q3k[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ3_KScalar</c>
/// and llama.cpp's <c>block_q3_K</c>: each 256 contiguous columns of a row form
/// one Q3_K super-block of 110 bytes — 32 bytes hmask (1 bit per element),
/// 64 bytes of 2-bit qs, 12 bytes packed 6-bit signed-after-bias scales,
/// fp16 d. Activation <c>x</c> and output <c>y</c> are FP32. One workgroup per
/// output row, 128 threads, shared-memory tree reduce. No coopmat path —
/// follow-up ticket sibling of the Q4_K / Q5_K / Q6_K coopmat work.
/// </remarks>
public sealed class MatMulQ3KGemvF32Kernel : IDisposable
{
    /// <summary>Q3_K super-block: 32 + 64 + 12 + 2 = 110 bytes.</summary>
    public const int Q3_KBlockBytes = 110;

    /// <summary>Elements per Q3_K super-block.</summary>
    public const int Q3_KGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ3KGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q3_k_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ3KGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q3_k_f32_gemv.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
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
        return new MatMulQ3KGemvF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ3K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ3K, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q3_K GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ3K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q3_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q3_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q3_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q3_KBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ3K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ3K.Size}.",
                nameof(weightsQ3K));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ3K.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
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
    }
}
