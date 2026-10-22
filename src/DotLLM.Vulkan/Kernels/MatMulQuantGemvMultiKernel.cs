using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Block formats served by <see cref="MatMulQuantGemvMultiKernel"/>.</summary>
public enum QuantGemvMultiFormat
{
    /// <summary>Q4_K, 144-byte super-blocks.</summary>
    Q4_K,
    /// <summary>Q5_K, 176-byte super-blocks.</summary>
    Q5_K,
    /// <summary>Q6_K, 210-byte super-blocks.</summary>
    Q6_K,
    /// <summary>Q6_K, 210-byte super-blocks, two output rows per workgroup sharing the activation loads (#996; faster than <see cref="Q6_K"/> at 3..4 columns).</summary>
    Q6_K_R2,
    /// <summary>IQ4_XS, 136-byte super-blocks.</summary>
    IQ4_XS,
    /// <summary>IQ3_S, 110-byte super-blocks (needs the shared grid codebook).</summary>
    IQ3_S,
}

/// <summary>
/// Multi-column (2..<see cref="MaxColumns"/> activation rows) F32-in GEMV for the K-quant and IQ dense-projection formats (#902):
/// <c>Y[N, M] = X[N, K] @ dequant(W[M, K])^T</c>. Each shader is its single-row decode GEMV with NC accumulators: every weight is decoded ONCE and
/// FMA'd against all N rows, so a 2..8-token MTP verify batch costs about one GEMV per projection instead of the coopmat / tiled GEMM
/// (a fixed ~100 ms penalty per forward on the ISTA GSQ-RCO qwen4exp files, whose dense projections are IQ4_XS / IQ3_S / Q4_K / Q5_K / Q6_K).
/// </summary>
/// <remarks>One workgroup per output row; <c>NC</c> is a specialization constant (one pipeline per column count). K must be a multiple of 256.</remarks>
public sealed class MatMulQuantGemvMultiKernel : IDisposable
{
    /// <summary>Largest supported column count.</summary>
    public const int MaxColumns = 8;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints
    private const int GroupSize = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _buffersPerSet;
    private readonly int _blockBytes;
    private bool _disposed;

    /// <summary>The format this instance serves.</summary>
    public QuantGemvMultiFormat Format { get; }

    private MatMulQuantGemvMultiKernel(QuantGemvMultiFormat format, VulkanDevice device, VulkanModule module, nint pool, int buffersPerSet, int blockBytes)
    {
        Format = format;
        _device = device; _module = module; _descriptorPool = pool; _buffersPerSet = buffersPerSet; _blockBytes = blockBytes;
        var bindings = new VkDescriptorBinding[buffersPerSet];
        for (int i = 0; i < buffersPerSet; i++) bindings[i] = new VkDescriptorBinding((uint)i);
        for (int n = 2; n <= MaxColumns; n++)
            _pipelines[n] = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes, specConstants: [(uint)n]);
        _descriptorCache = new DescriptorSetCache(device, pool, _pipelines[2]!, buffersPerSet: buffersPerSet);
    }

    private static (string Spv, int BlockBytes, int Buffers, bool NeedsSubgroup) Describe(QuantGemvMultiFormat f) => f switch
    {
        QuantGemvMultiFormat.Q4_K => ("matmul_q4_k_gemv_f32_multi.spv", 144, 3, true),
        QuantGemvMultiFormat.Q5_K => ("matmul_q5_k_gemv_f32_multi.spv", 176, 3, true),
        QuantGemvMultiFormat.Q6_K => ("matmul_q6_k_gemv_f32_multi.spv", 210, 3, true),
        QuantGemvMultiFormat.Q6_K_R2 => ("matmul_q6_k_gemv_f32_multi_r2.spv", 210, 3, true),
        QuantGemvMultiFormat.IQ4_XS => ("matmul_iq4_xs_f32_gemv_multi.spv", 136, 3, false),
        QuantGemvMultiFormat.IQ3_S => ("matmul_iq3_s_f32_gemv_multi.spv", 110, 4, false),
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    /// <summary>Loads the format's multi-column shader; null when it is missing or the device lacks the subgroup arithmetic the K-quant variants reduce with.</summary>
    public static MatMulQuantGemvMultiKernel? TryCreate(VulkanDevice device, string spvDir, QuantGemvMultiFormat format)
    {
        var (spv, blockBytes, buffers, needsSubgroup) = Describe(format);
        if (needsSubgroup && !device.HasSubgroupArithmetic) return null;
        string path = Path.Combine(spvDir, spv);
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
            return new MatMulQuantGemvMultiKernel(format, device, module, pool, buffers, blockBytes);
        }
        catch { module.Dispose(); throw; }
    }

    /// <summary>True when this kernel serves a matmul with <paramref name="n"/> rows and contraction <paramref name="k"/>.</summary>
    public static bool Accepts(int n, int k) => n >= 2 && n <= MaxColumns && k > 0 && (k % GroupSize) == 0;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the multi-column GEMV (input <c>[n, k]</c> F32, output <c>[n, m]</c>); <paramref name="codebook"/> is required for IQ3_S.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n,
                              VulkanDevice.Buffer? codebook = null)
    {
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        int blocksPerRow = k / GroupSize;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (weights.Size < (long)m * rowBytes) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (input.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (output.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(output));
        if (_buffersPerSet == 4 && codebook is null) throw new ArgumentNullException(nameof(codebook), $"{Format} needs its grid codebook.");

        Span<nint> buffers = _buffersPerSet == 4
            ? stackalloc nint[4] { weights.Handle, input.Handle, output.Handle, codebook!.Handle }
            : stackalloc nint[3] { weights.Handle, input.Handle, output.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        var pipe = _pipelines[n]!;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, (uint)((rowBytes + 3) / 4) };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipe.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, Format == QuantGemvMultiFormat.Q6_K_R2 ? (uint)((m + 1) / 2) : (uint)m, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        foreach (var p in _pipelines) p?.Dispose();
        _module.Dispose();
    }
}
