using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Weight storage formats the F16-activation encoder GEMM reads in place (issues #1473 / #1474).</summary>
public enum EncoderWeightFormat
{
    /// <summary>IEEE binary16, 2 bytes per element.</summary>
    F16,
    /// <summary>Q8_0 blocks (fp16 scale + 32 int8 codes, 34 bytes per 32 elements), dequantised in the LDS tile load.</summary>
    Q8_0,
    /// <summary>IEEE binary32, narrowed to F16 in the LDS tile load.</summary>
    F32,
}

/// <summary>
/// Blocked 128x128 coopmat GEMM with F16 activations in and out and in-place weights (issue #1473):
/// <c>C[N, M] = B[N, K] @ W[M, K]^T</c>, <c>B</c> and <c>C</c> row-major F16, <c>W</c> F16, Q8_0 or F32 as stored in the GGUF. The weights are never
/// expanded into a wider copy: dequantisation / narrowing happens inside the tile load, and the F32 accumulator is rounded to F16 once at the store.
/// </summary>
/// <remarks>
/// Needs a wave64 device (the shared blocked template's correctness gate, <see cref="F16GemmCoopmatVariant.RequiresNativeSubgroupSize"/>). The F16
/// format additionally has the any-wave fallback <see cref="MatMulF16GemmCoopmatKernel"/>; Q8_0 / F32 weights refuse on other devices.
/// </remarks>
public sealed class MatMulEncoderGemmKernel : IDisposable
{
    private const int PushConstantBytes = 5 * sizeof(uint);   // M, K, N, blocksPerRow, rowUints
    private const int Tile = 128;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private bool _disposed;

    /// <summary>Weight format this instance reads.</summary>
    public EncoderWeightFormat Format { get; }

    private MatMulEncoderGemmKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, EncoderWeightFormat format)
    {
        _device = device; _module = module; _pipeline = pipeline; _pool = pool; Format = format;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    private static string SpvName(EncoderWeightFormat f) => f switch
    {
        EncoderWeightFormat.Q8_0 => "matmul_q8_0_gemm_coopmat_128x128x4_hh.spv",
        EncoderWeightFormat.F32 => "matmul_f32w_gemm_coopmat_128x128x4_hh.spv",
        _ => "matmul_f16_gemm_coopmat_128x128x4_hh.spv",
    };

    /// <summary>Whether the device can run the in-place kernel for <paramref name="format"/> (cooperative matrices, wave64, SPIR-V present).</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, EncoderWeightFormat format)
        => device.HasCooperativeMatrix && device.SubgroupSize == 64 && File.Exists(Path.Combine(spvDir, SpvName(format)));

    /// <summary>Creates the kernel; check <see cref="IsSupportedOn"/> first.</summary>
    public static MatMulEncoderGemmKernel Create(VulkanDevice device, string spvDir, EncoderWeightFormat format)
    {
        if (!IsSupportedOn(device, spvDir, format))
            throw new NotSupportedException($"The {format} encoder GEMM needs VK_KHR_cooperative_matrix, a wave64 device and {SpvName(format)}.");
        var module = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, SpvName(format)));
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            for (int i = 0; i < 3; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch { module.Dispose(); throw; }
        return new MatMulEncoderGemmKernel(device, module, pipeline, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3), format);
    }

    /// <summary>Stored bytes of a <c>[m, k]</c> weight in <paramref name="format"/> (rows contiguous).</summary>
    public static long WeightBytes(EncoderWeightFormat format, int m, int k) => format switch
    {
        EncoderWeightFormat.Q8_0 => (long)m * (k / 32) * 34,
        EncoderWeightFormat.F32 => (long)m * k * 4,
        _ => (long)m * k * 2,
    };

    /// <summary>Whether a K of this value is legal for <paramref name="format"/> (Q8_0 rows must be 4-byte aligned: an even block count).</summary>
    public static bool SupportsK(EncoderWeightFormat format, int k)
        => k % 32 == 0 && (format != EncoderWeightFormat.Q8_0 || (k / 32) % 2 == 0);

    /// <summary>Drops every cached descriptor set.</summary>
    internal void InvalidateDescriptorCache() => _cache.Reset();

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC, int m, int k, int n)
    {
        if (m <= 0 || k <= 0 || n <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (!SupportsK(Format, k)) throw new ArgumentException($"k={k} is not legal for {Format}.", nameof(k));
        if (weights.Size < WeightBytes(Format, m, k)) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (inputB.Size < (long)n * k * 2) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * 2) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        uint blocksPerRow = Format switch { EncoderWeightFormat.Q8_0 => (uint)(k / 32), EncoderWeightFormat.F32 => (uint)k, _ => (uint)(k / 2) };
        uint rowUints = Format switch
        {
            EncoderWeightFormat.Q8_0 => (uint)((WeightBytes(Format, 1, k) + 3) / 4),
            EncoderWeightFormat.F32 => (uint)k,
            _ => (uint)(k / 2),
        };
        Span<nint> buffers = stackalloc nint[3] { weights.Handle, inputB.Handle, outputC.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, blocksPerRow, rowUints };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Tile - 1) / Tile), (uint)((n + Tile - 1) / Tile), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
