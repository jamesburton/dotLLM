using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_K decode-path GEMV: <c>y[M] = W_q4k[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ4_KScalar</c>
/// and llama.cpp's <c>block_q4_K</c>: each 256 contiguous columns of a row form one
/// Q4_K super-block of 144 bytes — 2 bytes fp16 <c>d</c>, 2 bytes fp16 <c>dmin</c>,
/// 12 bytes packed (8x6-bit) scales/mins (see <c>UnpackQ4Q5Scales</c>), 128 bytes
/// of 4-bit quants (two per byte).
/// </para>
/// <para>
/// The activation vector <c>x</c> is FP32 (not pre-quantized) — N=1 decode-path,
/// matching the convention of <see cref="MatMulQ8_0Kernel"/>. Output <c>y</c> is FP32.
/// </para>
/// <para>
/// Dispatch: one workgroup per output row, 128 threads per workgroup,
/// shared-memory reduction. No subgroup / cooperative-matrix intrinsics —
/// broadest driver portability and bit-equal correctness first; perf
/// follow-ups (subgroup tile, F16 acc) are tracked separately.
/// </para>
/// <para>
/// <b>Bit-packing note.</b> The 12-byte scales region holds 8x6-bit scales and
/// 8x6-bit mins. Sub-blocks 0..3 take the low 6 bits of bytes [0..3]
/// (scales) / [4..7] (mins). Sub-blocks 4..7 take the low 4 bits of bytes
/// [8..11] ored with the top 2 bits of bytes [0..3] (scales) / [4..7] (mins)
/// shifted to position [4..5]. This is exactly llama.cpp's
/// <c>get_scale_min_k4()</c> and is implemented in the shader's
/// <c>unpackScaleMin</c> helper. The layout is also implemented in
/// <c>DequantizeKQuants.UnpackQ4Q5Scales</c> which serves as the byte-exact
/// numerical oracle.
/// </para>
/// </remarks>
public sealed class MatMulQ4KGemvF32Kernel : IDisposable
{
    /// <summary>Q4_K super-block: 2 + 2 + 12 + 128 = 144 bytes.</summary>
    public const int Q4_KBlockBytes = 144;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4_KGroupSize = 256;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ4KGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q4_k_gemv_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ4KGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q4_k_gemv_f32.spv");
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
        return new MatMulQ4KGemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W_q4k[M,K] @ x[K]</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ4K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ4K, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q4_K GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ4K">
    /// Raw Q4_K blob of <c>M * (K/256) * 144</c> bytes, rows contiguous.
    /// </param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 256).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q4_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q4_KBlockBytes;
        // 144 is divisible by 4, so rowBytes is always 4-byte aligned. rowUints kept
        // for parity with the Q8_0 kernel signature.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ4K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ4K.Size}.",
                nameof(weightsQ4K));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ4K.Handle, x.Handle, y.Handle };
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
