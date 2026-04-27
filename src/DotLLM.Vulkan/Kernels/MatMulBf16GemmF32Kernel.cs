using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// BF16 native prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_bf16[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Companion to <see cref="MatMulBf16GemvF32Kernel"/> (decode-path GEMV).
/// Same byte layout: weights row-major <c>[M, K]</c> in IEEE-754-truncated
/// brain float (2 bytes/element). B and C are FP32 row-major.
/// </para>
/// <para>
/// Tiling identical to <see cref="MatMulF16GemmF32Kernel"/> — 16x16 output
/// tile, 256 threads/workgroup, K-chunk 32. The only inner-loop difference
/// is the BF16 → F32 unpack (shift-left-16 + uintBitsToFloat) at sharedW
/// staging. No coopmat path for BF16 today: KHR_cooperative_matrix exposes
/// F16 / Sint8 operands on this hardware, not BF16, so the prefill GEMM
/// stays on the scalar tiled path.
/// </para>
/// <para>
/// Alignment requirement: <c>k</c> must be a multiple of 32 (matches the
/// K-chunk size).
/// </para>
/// </remarks>
public sealed class MatMulBf16GemmF32Kernel : IDisposable
{
    /// <summary>Bytes per BF16 element on device.</summary>
    public const int Bf16ElementBytes = 2;

    /// <summary>K must be a multiple of this value.</summary>
    public const int KChunk = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, pairsPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulBf16GemmF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_bf16_gemm_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulBf16GemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_bf16_gemm_f32.spv");
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
        return new MatMulBf16GemmF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the BF16 GEMM synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsBf16, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the BF16 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % KChunk) != 0)
            throw new ArgumentException($"k must be a multiple of {KChunk}, got {k}", nameof(k));

        int pairsPerRow = k / 2;
        long rowBytes = (long)k * Bf16ElementBytes;
        int rowUints = pairsPerRow;

        long weightsMin = (long)m * rowBytes;
        if (weightsBf16.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsBf16.Size}.",
                nameof(weightsBf16));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsBf16.Handle, inputB.Handle, outputC.Handle };
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
            (uint)pairsPerRow,
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
