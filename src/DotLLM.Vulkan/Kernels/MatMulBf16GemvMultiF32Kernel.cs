using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// BF16 multi-column GEMV (issue #706): <c>y[N, M] = x[N, K] @ W_bf16[M, K]^T</c> for <c>N</c> in 1..<see cref="MaxColumns"/>.
/// </summary>
/// <remarks>
/// Reads each weight row once for all N activation vectors, so a 2..8-token MTP verify / small batch costs about one GEMV instead of the
/// 16x16-tile prefill GEMM (<see cref="MatMulBf16GemmF32Kernel"/>), whose tile wastes most lanes on thin projections such as Bonsai's
/// 48-row <c>ssm_alpha</c> / <c>ssm_beta</c>. Same weight layout as <see cref="MatMulBf16GemvF32Kernel"/>; <c>k</c> must be even.
/// </remarks>
public sealed class MatMulBf16GemvMultiF32Kernel : IDisposable
{
    /// <summary>Bytes per BF16 element on device.</summary>
    public const int Bf16ElementBytes = 2;

    /// <summary>Largest supported column count.</summary>
    public const int MaxColumns = 8;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, pairsPerRow, rowUints, N

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulBf16GemvMultiF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_bf16_gemv_multi_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulBf16GemvMultiF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_bf16_gemv_multi_f32.spv");
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
        return new MatMulBf16GemvMultiF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the BF16 multi-column GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsBf16, x, y, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the BF16 multi-column GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n < 1 || n > MaxColumns) throw new ArgumentOutOfRangeException(nameof(n), $"n must be 1..{MaxColumns}, got {n}");
        if ((k & 1) != 0)
            throw new ArgumentException($"k must be a multiple of 2, got {k}", nameof(k));

        int pairsPerRow = k / 2;
        long rowBytes = (long)k * Bf16ElementBytes;
        int rowUints = pairsPerRow;

        long weightsMin = (long)m * rowBytes;
        if (weightsBf16.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsBf16.Size}.",
                nameof(weightsBf16));
        if (x.Size < (long)n * k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsBf16.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m,
            (uint)k,
            (uint)pairsPerRow,
            (uint)rowUints,
            (uint)n,
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
