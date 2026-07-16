using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed Q5_1 expert bank, with a per-expert
/// output scale folded in:
/// <c>y[n, m] = downScale[indices[n]] * (dequant_q5_1(bank[indices[n], m, :]) dot x[n, :])</c>.
/// </summary>
/// <remarks>
/// <para>
/// Per-row Q5_1 dequantisation in the inner loop — the bank stays in its
/// 24-byte-per-block (32-element) GGUF layout on device. Sibling of
/// <see cref="MoeIndexedMatmulQ6_KF32Kernel"/> with one extra binding (the
/// per-expert <c>downScale</c> array) and one extra step (the post-scale).
/// The dequant matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ5_1Scalar</c>
/// byte-for-byte; the indexed expert lookup comes from
/// <see cref="MoeIndexedMatmulF32Kernel"/>.
/// </para>
/// <para>
/// The per-expert scale is Gemma-4's <c>ffn_down_exps.scale[e]</c>. Applying it
/// to the accumulator is algebraically identical to scaling every dequantised
/// weight and matches the CPU fold (<c>down_output * DownExpertScale[e]</c>)
/// applied before the routing weight, so the downstream weighted-scatter stays
/// unchanged. Keeps the Gemma-4 down experts quantized on device for the real 26B.
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ5_1F32Kernel : IDisposable
{
    /// <summary>Q5_1 block: 2(d) + 2(m) + 4(qh) + 16(qs) = 24 bytes.</summary>
    public const int Q5_1BlockBytes = 24;

    /// <summary>Elements per Q5_1 block.</summary>
    public const int Q5_1GroupSize = 32;

    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // M, K, N, numExperts, blocksPerRow
    private const int PushConstantBytes = 5 * sizeof(uint);
    private const int BuffersPerSet = 5;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulQ5_1F32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>Loads <c>moe_indexed_matmul_q5_1_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedMatmulQ5_1F32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_q5_1_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BuffersPerSet];
            for (int i = 0; i < BuffersPerSet; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new MoeIndexedMatmulQ5_1F32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer bankQ5_1, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        VulkanDevice.Buffer downScale, int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankQ5_1, x, indices, y, downScale, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed Q5_1 expert-bank matmul dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bankQ5_1">
    /// Raw Q5_1 bank of <c>numExperts * M * (K/32) * 24</c> bytes, expert
    /// matrices contiguous, rows contiguous within each expert.
    /// </param>
    /// <param name="x">F32 input rows [<paramref name="n"/> * K] row-major.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="downScale">F32 per-expert output scale [<paramref name="numExperts"/>].</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 32).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — bounds-checks the index lookup and indexes <paramref name="downScale"/>.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bankQ5_1, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        VulkanDevice.Buffer downScale,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Q5_1GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5_1GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5_1GroupSize;
        long rowBytes = (long)blocksPerRow * Q5_1BlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        long scaleBytes = (long)numExperts * sizeof(float);
        if (bankQ5_1.Size < bankBytes) throw new ArgumentException("bankQ5_1 buffer too small.", nameof(bankQ5_1));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));
        if (downScale.Size < scaleBytes) throw new ArgumentException("downScale buffer too small.", nameof(downScale));

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            bankQ5_1.Handle, x.Handle, indices.Handle, y.Handle, downScale.Handle,
        };
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
            (uint)numExperts,
            (uint)blocksPerRow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((m + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((n + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
