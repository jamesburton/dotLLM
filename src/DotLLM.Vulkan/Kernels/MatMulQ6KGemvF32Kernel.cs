using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q6_K decode-path GEMV: <c>y[M] = W_q6k[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Sibling of <see cref="MatMulQ4KGemvF32Kernel"/> /
/// <see cref="MatMulQ5KGemvF32Kernel"/> — same dispatch model (one workgroup
/// per output row, 128 threads, shared-memory tree reduce). Q6_K is
/// structurally simpler than Q4_K / Q5_K on the metadata side: scale-only
/// reconstruction (no <c>dmin</c>), 16 signed <c>int8</c> scales (no 6-bit
/// packed scale table), and signed quants in <c>-32..31</c>. The byte
/// extraction, however, is more involved because a Q6_K super-block is laid
/// out as two 128-element halves, each with 4 groups of 32 elements that
/// share a 32-byte <c>qh</c> slab.
/// </para>
/// <para>
/// Weight layout mirrors the CPU oracle
/// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ6_KScalar</c> and
/// llama.cpp's <c>block_q6_K</c>: each 256 contiguous columns of a row form
/// one Q6_K super-block of 210 bytes — 128 bytes <c>ql</c> (low 4 bits, two
/// per byte), 64 bytes <c>qh</c> (high 2 bits per element, four per byte),
/// 16 bytes <c>scales</c> (<c>int8</c>, signed, one per 16-element
/// sub-block), 2 bytes fp16 <c>d</c>.
/// </para>
/// <para>
/// The activation vector <c>x</c> is FP32 (not pre-quantized) — N=1
/// decode-path, matching <see cref="MatMulQ8_0Kernel"/> /
/// <see cref="MatMulQ4KGemvF32Kernel"/> /
/// <see cref="MatMulQ5KGemvF32Kernel"/>. Output <c>y</c> is FP32.
/// </para>
/// <para>
/// <b>Bit-extraction.</b> Given output element index <c>i</c> in <c>[0,256)</c>
/// within a super-block, with <c>half = i / 128</c>, <c>local = i % 128</c>,
/// <c>group = local / 32</c>, <c>l = local % 32</c>, <c>isc = l / 16</c>:
/// the element value is
/// <c>d * scales[half*8 + isc + 2*group] *
///   ((nibble | (high2 &lt;&lt; 4)) - 32)</c>
/// where the low 4 bits come from <c>ql[half*64 + (group&amp;1)*32 + l]</c>
/// (low nibble for groups 0,1; high nibble for groups 2,3) and the high 2 bits
/// from <c>(qh[half*32 + l] &gt;&gt; (group*2)) &amp; 3</c>. This is a
/// byte-exact replica of <c>DequantizeQ6_KScalar</c>.
/// </para>
/// <para>
/// Per-row stride is <c>blocksPerRow * 210</c> bytes — NOT 4-byte aligned in
/// general (210 % 4 == 2), unlike Q4_K / Q5_K. The shader's <c>readByte</c>
/// and <c>readHalf</c> helpers handle straddle.
/// </para>
/// </remarks>
public sealed class MatMulQ6KGemvF32Kernel : IDisposable
{
    /// <summary>Q6_K super-block: 128(ql) + 64(qh) + 16(scales) + 2(d) = 210 bytes.</summary>
    public const int Q6_KBlockBytes = 210;

    /// <summary>Elements per Q6_K super-block.</summary>
    public const int Q6_KGroupSize = 256;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ6KGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q6_k_gemv_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ6KGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q6_k_gemv_f32.spv");
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
        return new MatMulQ6KGemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W_q6k[M,K] @ x[K]</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ6K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ6K, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q6_K GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ6K">
    /// Raw Q6_K blob of <c>M * (K/256) * 210</c> bytes, rows contiguous.
    /// </param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 256).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ6K, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q6_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q6_KBlockBytes;
        // 210 is NOT divisible by 4 (210 % 4 == 2); rowBytes will not be
        // 4-byte aligned for odd blocksPerRow. The shader's per-byte / per-fp16
        // readers handle this; rowUints is kept for parity with the Q4_K /
        // Q5_K / Q8_0 kernel signatures.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ6K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ6K.Size}.",
                nameof(weightsQ6K));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ6K.Handle, x.Handle, y.Handle };
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
