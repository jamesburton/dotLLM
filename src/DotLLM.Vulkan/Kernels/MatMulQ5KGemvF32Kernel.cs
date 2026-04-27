using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q5_K decode-path GEMV: <c>y[M] = W_q5k[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Sibling of <see cref="MatMulQ4KGemvF32Kernel"/> — same dispatch model
/// (one workgroup per output row, 128 threads, shared-memory tree reduce),
/// same scale/min unpack (Q4_K and Q5_K share <c>UnpackQ4Q5Scales</c>), same
/// nibble selection. Q5_K's only structural delta is one extra high-bit
/// table (qh) per super-block contributing the 5th bit of every element.
/// </para>
/// <para>
/// Weight layout mirrors the CPU oracle
/// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ5_KScalar</c> and
/// llama.cpp's <c>block_q5_K</c>: each 256 contiguous columns of a row form
/// one Q5_K super-block of 176 bytes — 2 bytes fp16 <c>d</c>, 2 bytes fp16
/// <c>dmin</c>, 12 bytes packed (8x6-bit) scales/mins, 32 bytes <c>qh</c>
/// (one bit per element across 8 sub-blocks), 128 bytes of 4-bit low
/// nibbles (two per byte).
/// </para>
/// <para>
/// The activation vector <c>x</c> is FP32 (not pre-quantized) — N=1
/// decode-path, matching <see cref="MatMulQ8_0Kernel"/> /
/// <see cref="MatMulQ4KGemvF32Kernel"/>. Output <c>y</c> is FP32.
/// </para>
/// <para>
/// <b>qh bit-indexing.</b> The <c>qh</c> array is 32 bytes — one byte per
/// position-within-sub-block (i in 0..31). Bit <c>j</c> of <c>qh[i]</c>
/// (j in 0..7) is the 5th bit of element <c>j*32 + i</c>. This is NOT a
/// flat bitfield; the indexing matches llama.cpp's
/// <c>dequantize_row_q5_K</c> and the CPU oracle's <c>(qh[i] &gt;&gt; j) &amp; 1</c>
/// expression.
/// </para>
/// </remarks>
public sealed class MatMulQ5KGemvF32Kernel : IDisposable
{
    /// <summary>Q5_K super-block: 2 + 2 + 12 + 32 + 128 = 176 bytes.</summary>
    public const int Q5_KBlockBytes = 176;

    /// <summary>Elements per Q5_K super-block.</summary>
    public const int Q5_KGroupSize = 256;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ5KGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q5_k_gemv_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ5KGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q5_k_gemv_f32.spv");
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
        return new MatMulQ5KGemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W_q5k[M,K] @ x[K]</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ5K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ5K, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q5_K GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ5K">
    /// Raw Q5_K blob of <c>M * (K/256) * 176</c> bytes, rows contiguous.
    /// </param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 256).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ5K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q5_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q5_KBlockBytes;
        // 176 is divisible by 4, so rowBytes is always 4-byte aligned. rowUints kept
        // for parity with the Q4_K / Q8_0 kernel signatures.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ5K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ5K.Size}.",
                nameof(weightsQ5K));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ5K.Handle, x.Handle, y.Handle };
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
