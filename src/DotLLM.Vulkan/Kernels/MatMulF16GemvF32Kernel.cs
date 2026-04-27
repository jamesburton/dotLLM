using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F16 native decode-path GEMV: <c>y[M] = W_f16[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weights are stored on device in their source IEEE-754 binary16 layout —
/// 2 bytes per element, row-major <c>[M, K]</c>. Activations / output are
/// FP32; accumulation is FP32. Companion to the F16 prefill GEMM
/// (<see cref="MatMulF16GemmF32Kernel"/> /
/// <see cref="MatMulF16GemmCoopmatKernel"/>); decode-path sibling of
/// <see cref="MatMulQ8_0Kernel"/> and <see cref="MatMulQ4KGemvF32Kernel"/>.
/// </para>
/// <para>
/// Why this kernel exists: BF16 / F16 SafeTensors weights used to be expanded
/// to F32 at upload, doubling the matmul-weight VRAM cost. With F16 native
/// upload + this kernel, F16 weights stay 2 bytes per element on device,
/// halving VRAM and per-forward weight bandwidth on the SafeTensors path that
/// dominates the BF16 / F16 HuggingFace ecosystem.
/// </para>
/// <para>
/// Dispatch: one workgroup per output row, 128 threads per workgroup,
/// shared-memory tree reduction. Mirrors the
/// <see cref="MatMulQ8_0Kernel"/> / <see cref="MatMulQ4KGemvF32Kernel"/>
/// dispatch geometry.
/// </para>
/// <para>
/// Alignment requirement: <c>k</c> must be a multiple of 2 (each storage
/// uint holds two F16 elements via <c>unpackHalf2x16</c>). Real-model K is
/// always at least <c>head_dim ≥ 32</c>, so this is never a binding constraint
/// in practice.
/// </para>
/// </remarks>
public sealed class MatMulF16GemvF32Kernel : IDisposable
{
    /// <summary>Bytes per F16 element on device.</summary>
    public const int F16ElementBytes = 2;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, pairsPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulF16GemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_f16_gemv_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulF16GemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_f16_gemv_f32.spv");
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
        return new MatMulF16GemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W_f16[M,K] @ x[K]</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsF16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsF16, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the F16 GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsF16">Raw F16 blob of <c>M * K * 2</c> bytes, rows contiguous.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 2).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsF16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k & 1) != 0)
            throw new ArgumentException($"k must be a multiple of 2, got {k}", nameof(k));

        int pairsPerRow = k / 2;
        long rowBytes = (long)k * F16ElementBytes; // = pairsPerRow * 4
        int rowUints = pairsPerRow;                 // 2 F16 / uint

        long weightsMin = (long)m * rowBytes;
        if (weightsF16.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsF16.Size}.",
                nameof(weightsF16));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsF16.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)pairsPerRow,
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
