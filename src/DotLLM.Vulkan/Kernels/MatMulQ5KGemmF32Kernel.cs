using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q5_K prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_q5k[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Companion to <see cref="MatMulQ5KGemvF32Kernel"/> (decode-path GEMV) and
/// structural sibling of <see cref="MatMulQ4KGemmF32Kernel"/>: same byte
/// layout shape (per-row super-blocks of 176 bytes), same tiling (16x16
/// output tile, K-chunk = 32 elements = one Q5_K sub-block), same scale/min
/// unpack (Q4_K and Q5_K share <c>UnpackQ4Q5Scales</c>). Q5_K's only
/// structural delta versus Q4_K is one extra high-bit table (qh) per
/// super-block contributing the 5th bit of every element.
/// </para>
/// <para>
/// Tiling: 16x16 output tile per workgroup (one thread per output cell). The
/// K-axis is iterated in chunks of 32 elements (one Q5_K <i>sub-block</i>); 8
/// such chunks per super-block. Per K-chunk we cooperatively dequantise a
/// 16x32 weight tile into shared memory once and reuse it across all 16 token
/// rows of B. No subgroup / cooperative-matrix intrinsics in this Phase 1
/// kernel — broadest driver portability and correctness first; perf
/// follow-ups (coopmat, F16 acc) are tracked separately.
/// </para>
/// </remarks>
public sealed class MatMulQ5KGemmF32Kernel : IDisposable
{
    /// <summary>Q5_K super-block: 2 + 2 + 12 + 32 + 128 = 176 bytes.</summary>
    public const int Q5_KBlockBytes = 176;

    /// <summary>Elements per Q5_K super-block.</summary>
    public const int Q5_KGroupSize = 256;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ5KGemmF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q5_k_gemm_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ5KGemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q5_k_gemm_f32.spv");
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
        return new MatMulQ5KGemmF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the batched GEMM: <c>C[N, M] = B[N, K] @ W_q5k[M, K]^T</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ5K, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ5K, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q5_K GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ5K">
    /// Raw Q5_K blob of <c>M * (K/256) * 176</c> bytes, rows contiguous.
    /// </param>
    /// <param name="inputB">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 256).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ5K, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q5_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q5_KBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ5K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ5K.Size}.",
                nameof(weightsQ5K));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsQ5K.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m,
            (uint)k,
            (uint)n,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((m + TileM - 1) / TileM);
        uint groupsY = (uint)((n + TileN - 1) / TileN);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
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
