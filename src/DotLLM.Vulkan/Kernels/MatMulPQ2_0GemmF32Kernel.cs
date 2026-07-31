using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// PQ2_0 (PrismML Bonsai ternary) prefill-path batched GEMM:
/// <c>C[N, M] = B[N, K] @ W_pq2_0[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Companion to <see cref="MatMulPQ2_0GemvF32Kernel"/> and the PQ2_0 analog of
/// <see cref="MatMulI2SGemmF32Kernel"/>. The shader is a direct port of the shipped
/// register-blocked I2_S GEMM (<c>matmul_i2_s_f32_gemm_rb.comp</c>): 32x32 output tile per
/// workgroup, 16x16 threads each owning a 2x2 micro-tile, K-chunk = 128 elements (one PQ2_0
/// group), 32 KB of shared memory staging a 32x128 token tile and a 32x128 dequantised weight
/// tile per chunk.
/// </para>
/// <para>
/// The cooperative-matrix line was deliberately not taken: issue #229 (commit <c>a1161cbc</c>)
/// confirmed the multi-fragment coopmat warptile hypothesis and the variant <i>still</i> lost to
/// the F32 register-blocked kernel on gfx1151, and commit <c>0ea1bd7f</c> found plain I2_S
/// coopmat loses to register-blocked on an RTX 3060. Two independent negatives; PQ2_0 starts
/// from the design that won.
/// </para>
/// <para>
/// Weight layout matches the CPU oracle <c>DotLLM.Cpu.Kernels.MatMul.UnpackPQ2_0Row</c>: each
/// 128 contiguous columns of a row form a 34-byte group — 2 bytes little-endian fp16 group scale
/// then 32 packed code bytes (byte <c>gp</c> holds group-relative positions
/// <c>{gp, gp+32, gp+64, gp+96}</c> at bit offsets <c>{6,4,2,0}</c>, value = code − 1). Row
/// stride is <c>(K/128)·34</c> bytes and there is <b>no</b> per-tensor tail scale (contrast I2_S,
/// whose single float32 scale sits at byte offset <c>M·(K/4)</c> and is applied once to the
/// finished accumulator). Because each group owns its scale, the scale is folded into the shared
/// weight tile at staging time — mirroring the CPU reference, which writes
/// <c>(code − 1) · scale</c> before the dot — so the inner MAC loop stays identical to the I2_S
/// kernel's.
/// </para>
/// </remarks>
public sealed class MatMulPQ2_0GemmF32Kernel : IDisposable
{
    /// <summary>PQ2_0 group: 2 bytes fp16 scale + 32 bytes packed ternary codes.</summary>
    public const int PQ2_0GroupBytes = MatMulPQ2_0GemvF32Kernel.PQ2_0GroupBytes;

    /// <summary>Elements per PQ2_0 group.</summary>
    public const int PQ2_0GroupSize = MatMulPQ2_0GemvF32Kernel.PQ2_0GroupSize;

    /// <summary>Weight rows of <c>C</c> produced per workgroup.</summary>
    private const int TileM = 32;

    /// <summary>Token rows of <c>C</c> produced per workgroup.</summary>
    private const int TileN = 32;

    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulPQ2_0GemmF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_pq2_0_f32_gemm_rb.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulPQ2_0GemmF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, "matmul_pq2_0_f32_gemm_rb.spv");

    /// <summary>
    /// Loads the named SPIR-V from <paramref name="spvDir"/> and creates the pipeline. The
    /// <paramref name="spvFileName"/> overload exists so a benchmark can A/B successive kernel
    /// variants side by side in one process — the interleaved-paired methodology this box
    /// requires cannot span processes.
    /// </summary>
    public static MatMulPQ2_0GemmF32Kernel Create(VulkanDevice device, string spvDir, string spvFileName)
    {
        string path = Path.Combine(spvDir, spvFileName);
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
        return new MatMulPQ2_0GemmF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMM synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsPQ2_0, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the PQ2_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % PQ2_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / PQ2_0GroupSize;
        long rowBytes = (long)blocksPerRow * PQ2_0GroupBytes;
        // 34 is not divisible by 4 — every group past the first straddles uint
        // boundaries. rowUints rounds up to cover safe straddle reads.
        int rowUints = (int)((rowBytes + 3) / 4);

        // Packed rows only — PQ2_0 has no per-tensor tail scale (contrast I2_S).
        long weightsMin = (long)m * rowBytes;
        if (weightsPQ2_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·(k/128)·34), got {weightsPQ2_0.Size}.",
                nameof(weightsPQ2_0));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsPQ2_0.Handle, inputB.Handle, outputC.Handle };
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
