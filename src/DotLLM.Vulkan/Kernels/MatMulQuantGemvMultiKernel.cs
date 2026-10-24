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
    /// <summary>
    /// Q6_K, 210-byte super-blocks, NR rows per workgroup, 1..8 columns, deterministic (fixed reduction tree; #1220): the dense qwen35 model
    /// decodes and verifies on it so the two are bit-identical per column.
    /// </summary>
    Q6_K_Rows,
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
    private readonly int _rowsPerWorkgroup;
    private readonly int[] _rowsForColumns = new int[MaxColumns + 1];   // NR per column count (#1220)
    private bool _disposed;

    /// <summary>The format this instance serves.</summary>
    public QuantGemvMultiFormat Format { get; }

    /// <summary>Output rows per workgroup (<c>NR</c>, #1220); 1 except for the formats whose shader takes the constant.</summary>
    public int RowsPerWorkgroup => _rowsPerWorkgroup;

    /// <summary>Output rows per workgroup used for an <paramref name="n"/>-column dispatch.</summary>
    /// <param name="n">Column count, 1..<see cref="MaxColumns"/>.</param>
    /// <returns>NR for that pipeline.</returns>
    public int RowsPerWorkgroupFor(int n) => _rowsForColumns[n];

    /// <summary>True when the shader also serves a single activation row (n = 1), i.e. can stand in for the decode GEMV.</summary>
    public bool SupportsSingleColumn => TakesRows(Format);

    private static bool TakesRows(QuantGemvMultiFormat f) => f is QuantGemvMultiFormat.Q4_K or QuantGemvMultiFormat.Q6_K_Rows;

    private MatMulQuantGemvMultiKernel(QuantGemvMultiFormat format, VulkanDevice device, VulkanModule module, nint pool, int buffersPerSet, int blockBytes,
                                       uint requiredSubgroupSize, int rowsPerWorkgroup, int multiColumnRowsPerWorkgroup)
    {
        Format = format;
        _device = device; _module = module; _descriptorPool = pool; _buffersPerSet = buffersPerSet; _blockBytes = blockBytes;
        _rowsPerWorkgroup = rowsPerWorkgroup;
        for (int n = 1; n <= MaxColumns; n++)
            _rowsForColumns[n] = n == 1 ? rowsPerWorkgroup : multiColumnRowsPerWorkgroup;
        var bindings = new VkDescriptorBinding[buffersPerSet];
        for (int i = 0; i < buffersPerSet; i++) bindings[i] = new VkDescriptorBinding((uint)i);
        for (int n = SupportsSingleColumn ? 1 : 2; n <= MaxColumns; n++)
            _pipelines[n] = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes, requiredSubgroupSize: requiredSubgroupSize,
                specConstants: [(uint)n, (uint)_rowsForColumns[n]]);
        _descriptorCache = new DescriptorSetCache(device, pool, _pipelines[2]!, buffersPerSet: buffersPerSet);
    }

    /// <summary>
    /// Shader facts per format. <c>NeedsWave32</c>: the shader is ONE 32-lane workgroup whose whole reduction is a single <c>subgroupAdd</c> followed by a
    /// <c>subgroupElect</c> store, so it is only correct when the workgroup is exactly one subgroup. A driver free to pick a narrower subgroup (Intel Arc
    /// picks 8/16) splits the row over several subgroups that each store a partial sum: the IQ4_XS kernel returned about a quarter of the dot product
    /// there and every 2..8-row qwen4exp forward (MTP verify) produced garbage (#1104). Such a pipeline is pinned to wave32, or not created.
    /// </summary>
    private static (string Spv, int BlockBytes, int Buffers, bool NeedsSubgroup, bool NeedsWave32) Describe(QuantGemvMultiFormat f) => f switch
    {
        QuantGemvMultiFormat.Q4_K => ("matmul_q4_k_gemv_f32_multi.spv", 144, 3, false, false),
        QuantGemvMultiFormat.Q5_K => ("matmul_q5_k_gemv_f32_multi.spv", 176, 3, true, false),
        QuantGemvMultiFormat.Q6_K => ("matmul_q6_k_gemv_f32_multi.spv", 210, 3, true, false),
        QuantGemvMultiFormat.Q6_K_R2 => ("matmul_q6_k_gemv_f32_multi_r2.spv", 210, 3, true, false),
        QuantGemvMultiFormat.Q6_K_Rows => ("matmul_q6_k_gemv_f32_rows.spv", 210, 3, false, false),
        QuantGemvMultiFormat.IQ4_XS => ("matmul_iq4_xs_f32_gemv_multi.spv", 136, 3, true, true),
        QuantGemvMultiFormat.IQ3_S => ("matmul_iq3_s_f32_gemv_multi.spv", 110, 4, false, false),
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    /// <summary>True when <paramref name="format"/>'s shader reduces with a single subgroup per 32-lane workgroup and must run pinned to wave32.</summary>
    /// <param name="format">The block format.</param>
    /// <returns>Whether the pipeline needs a required subgroup size of 32.</returns>
    internal static bool RequiresWave32(QuantGemvMultiFormat format) => Describe(format).NeedsWave32;

    /// <summary>
    /// The required subgroup size to create <paramref name="format"/>'s pipelines with: 0 (driver default) for shaders that reduce across subgroups,
    /// 32 for single-subgroup shaders when the device can pin it, or null when the shader cannot run correctly on this device (no pin available, or the
    /// <c>DOTLLM_VULKAN_DISABLE_WAVE32</c> opt-out is set), in which case the caller falls back to the GEMM path.
    /// </summary>
    /// <param name="format">The block format.</param>
    /// <param name="wave32Pin">What <see cref="Wave32SubgroupControl.RequiredSubgroupSizeFor"/> returned for the device (32 or 0).</param>
    /// <returns>The pipeline's required subgroup size, or null to refuse the kernel.</returns>
    internal static uint? PipelineSubgroupSize(QuantGemvMultiFormat format, uint wave32Pin)
        => !RequiresWave32(format) ? 0u : wave32Pin == Wave32SubgroupControl.Wave32 ? wave32Pin : null;

    /// <summary>
    /// Loads the format's multi-column shader; null when it is missing, the device lacks the subgroup arithmetic the shader reduces with, or the shader
    /// needs a wave32 pin the device cannot provide.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V.</param>
    /// <param name="format">The block format.</param>
    /// <param name="rowsPerWorkgroup">Output rows per workgroup (1..8); only <see cref="QuantGemvMultiFormat.Q4_K"/> and <see cref="QuantGemvMultiFormat.Q6_K_Rows"/> take more than 1 (#1220). Used by the one-column (decode) pipeline.</param>
    /// <param name="multiColumnRowsPerWorkgroup">NR of the 2..8-column pipelines; 0 = <paramref name="rowsPerWorkgroup"/>. A row's bits do not depend on NR.</param>
    /// <returns>The kernel, or null when it cannot run correctly here.</returns>
    public static MatMulQuantGemvMultiKernel? TryCreate(VulkanDevice device, string spvDir, QuantGemvMultiFormat format, int rowsPerWorkgroup = 1,
                                                          int multiColumnRowsPerWorkgroup = 0)
    {
        if (multiColumnRowsPerWorkgroup == 0) multiColumnRowsPerWorkgroup = rowsPerWorkgroup;
        if (multiColumnRowsPerWorkgroup < 1 || multiColumnRowsPerWorkgroup > 8 || (multiColumnRowsPerWorkgroup > 1 && !TakesRows(format)))
            throw new ArgumentOutOfRangeException(nameof(multiColumnRowsPerWorkgroup));
        if (rowsPerWorkgroup < 1 || rowsPerWorkgroup > 8 || (rowsPerWorkgroup > 1 && !TakesRows(format)))
            throw new ArgumentOutOfRangeException(nameof(rowsPerWorkgroup), $"{format} does not take {rowsPerWorkgroup} rows per workgroup.");
        var (spv, blockBytes, buffers, needsSubgroup, _) = Describe(format);
        if (needsSubgroup && !device.HasSubgroupArithmetic) return null;
        if (PipelineSubgroupSize(format, Wave32SubgroupControl.RequiredSubgroupSizeFor(device)) is not { } subgroupSize) return null;
        string path = Path.Combine(spvDir, spv);
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
            return new MatMulQuantGemvMultiKernel(format, device, module, pool, buffers, blockBytes, subgroupSize, rowsPerWorkgroup, multiColumnRowsPerWorkgroup);
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
        if (!Accepts(n, k) && !(n == 1 && SupportsSingleColumn && k > 0 && (k % GroupSize) == 0))
            throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
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
        int rowsPerGroup = Format == QuantGemvMultiFormat.Q6_K_R2 ? 2 : _rowsForColumns[n];
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + rowsPerGroup - 1) / rowsPerGroup), 1, 1);
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
