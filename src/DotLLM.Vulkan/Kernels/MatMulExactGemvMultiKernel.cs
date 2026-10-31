using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Block formats served by <see cref="MatMulExactGemvMultiKernel"/>.</summary>
public enum ExactGemvFormat
{
    /// <summary>Q8_0 (<c>matmul_q8_0_gemv_multi</c>, #1220).</summary>
    Q8_0,

    /// <summary>NVFP4 with a per-tensor <c>X.scale</c> (<c>matmul_nvfp4_gemv_multi</c>, #1244).</summary>
    Nvfp4,

    /// <summary>IQ2_XXS (<c>matmul_iq2_xxs_gemv_multi</c>, needs the grid and ksigns tables; #1244).</summary>
    Iq2Xxs,

    /// <summary>Q5_0 (<c>matmul_q5_0_gemv_multi</c>, #1420).</summary>
    Q5_0,

    /// <summary>Q4_0 (<c>matmul_q4_0_gemv_multi</c>, #1420).</summary>
    Q4_0,
}

/// <summary>
/// Deterministic FP32-activation GEMV over 1..<see cref="MaxColumns"/> activation columns (#1244): <c>Y[n, M] = X[n, K] @ dequant(W[M, K])ᵀ</c>.
/// Every lane decodes its weights once per row and FMAs them against all <c>n</c> columns, and a column's arithmetic (explicit fma chain under
/// <c>precise</c>, fixed reduction tree) does not depend on <c>n</c> or the rows per workgroup. So a K+1-row speculative verify is
/// bit-identical per row to <c>n = 1</c> decode dispatches through this kernel, at about one GEMV's cost (the <c>cols</c> GEMVs of #1237 paid
/// one dequant per column).
/// </summary>
/// <remarks>One pipeline per column count (<c>NC</c> and <c>NR</c> are specialization constants), created on first use.</remarks>
public sealed class MatMulExactGemvMultiKernel : IDisposable
{
    /// <summary>Largest supported column count (a DFlash block of 16).</summary>
    public const int MaxColumns = GemvColumns.MaxColumns;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints | X.scale bits

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly VkDescriptorBinding[] _bindings;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _groupSize;
    private readonly int _blockBytes;
    private readonly int _decodeRows;
    private readonly int _multiRows;
    private bool _disposed;

    /// <summary>Dispatches recorded by every instance, read by tests to prove which GEMV ran.</summary>
    internal static long Dispatches;

    private MatMulExactGemvMultiKernel(ExactGemvFormat format, VulkanDevice device, VulkanModule module, int buffers, int groupSize, int blockBytes,
                                       int decodeRows, int multiRows)
    {
        Format = format;
        _device = device;
        _module = module;
        _groupSize = groupSize;
        _blockBytes = blockBytes;
        _decodeRows = decodeRows;
        _multiRows = multiRows;
        _bindings = new VkDescriptorBinding[buffers];
        for (int i = 0; i < buffers; i++) _bindings[i] = new VkDescriptorBinding((uint)i);
        _descriptorPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
        _descriptorCache = new DescriptorSetCache(device, _descriptorPool, Pipeline(1), buffersPerSet: buffers);
    }

    /// <summary>The format this instance serves.</summary>
    public ExactGemvFormat Format { get; }

    /// <summary>
    /// Optional per-column-count NR override (index = n, 0 = use the multi-column default). #1459: the best NR is not monotone in n (Nemotron-H Nano,
    /// Q5_0/Q8_0: 5 and 7 columns want NR 2, 6 and 8 want 4). Must be set before the first <see cref="Record"/> (pipelines are created lazily).
    /// A row's bits do not depend on NR.
    /// </summary>
    public int[]? RowsByColumns { get; set; }

    /// <summary>Loads the format's shader; null when it is missing.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V.</param>
    /// <param name="format">The block format.</param>
    /// <param name="decodeRows">Output rows per workgroup of the one-column (decode) pipeline, 1..8.</param>
    /// <param name="multiRows">Output rows per workgroup of the 2..16-column pipelines, 1..8.</param>
    /// <returns>The kernel, or null when the shader is missing.</returns>
    public static MatMulExactGemvMultiKernel? TryCreate(VulkanDevice device, string spvDir, ExactGemvFormat format, int decodeRows, int multiRows)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(decodeRows, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(decodeRows, 8);
        ArgumentOutOfRangeException.ThrowIfLessThan(multiRows, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(multiRows, 8);
        var (spv, buffers, groupSize, blockBytes) = format switch
        {
            ExactGemvFormat.Q8_0 => ("matmul_q8_0_gemv_multi.spv", 3, 32, 34),
            ExactGemvFormat.Nvfp4 => ("matmul_nvfp4_gemv_multi.spv", 3, QuantFormat.Nvfp4GroupSize, QuantFormat.Nvfp4BlockBytes),
            ExactGemvFormat.Iq2Xxs => ("matmul_iq2_xxs_gemv_multi.spv", 5, QuantFormat.KQuantGroupSize, QuantFormat.IQ2_XXSBlockBytes),
            ExactGemvFormat.Q5_0 => ("matmul_q5_0_gemv_multi.spv", 3, 32, 22),
            ExactGemvFormat.Q4_0 => ("matmul_q4_0_gemv_multi.spv", 3, 32, 18),
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
        string path = Path.Combine(spvDir, spv);
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            return new MatMulExactGemvMultiKernel(format, device, module, buffers, groupSize, blockBytes, decodeRows, multiRows);
        }
        catch
        {
            module.Dispose();
            throw;
        }
    }

    /// <summary>First column count at which IQ2_XXS uses at least 4 rows per workgroup (#1366).</summary>
    internal const int Iq2WideColumns = 14;

    /// <summary>Output rows per workgroup of an <paramref name="n"/>-column dispatch (a row's bits do not depend on it).</summary>
    /// <param name="n">Column count.</param>
    /// <returns>NR for that pipeline.</returns>
    public int RowsPerWorkgroupFor(int n)
    {
        int nr = n == 1 ? _decodeRows : _multiRows;
        if (n >= 2 && RowsByColumns is { } table && n < table.Length && table[n] > 0) nr = table[n];   // #1459 per-n tuning
        // #1366: IQ2_XXS at 14..16 columns with NR = 2 falls off a cliff (moe_dense of a Bolt verify, ms over 88 layers: 19 at 13 columns, 46-58 at
        // 14..16, slower than the GEMM path at 32 rows); NR = 4 gives 27 / 32. A row's bits do not depend on NR.
        if (Format == ExactGemvFormat.Iq2Xxs && n >= Iq2WideColumns) nr = Math.Max(nr, 4);
        while (nr > 1 && nr * n > 64) nr >>= 1;   // the shader's output pass needs NR * NC <= 64 lanes
        return nr;
    }

    /// <summary>True when the kernel serves <paramref name="n"/> columns of contraction <paramref name="k"/>.</summary>
    /// <param name="n">Column count.</param>
    /// <param name="k">Contraction length.</param>
    /// <returns>Whether <see cref="Record"/> accepts the shape.</returns>
    public bool Accepts(int n, int k) => n >= 1 && n <= MaxColumns && k > 0 && k % _groupSize == 0;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    private ComputePipeline Pipeline(int n)
        => _pipelines[n] ??= _module.CreateComputePipeline(
            entryPoint: "main", bindings: _bindings, pushConstantBytes: PushConstantBytes,
            specConstants: [(uint)n, (uint)RowsPerWorkgroupFor(n)]);

    /// <summary>
    /// #1459: when true, an column count of 5 or 7 runs the next even column count's pipeline if the input and output buffers have room for
    /// the extra row (its input row is never read by a result that is kept; its output row lands past the n rows the caller owns). The first n columns are
    /// bit-identical (a column's bits do not depend on NC). Measured on Nemotron-H Nano: 5 and 7 columns cost 1.5x the 6 and 8 column dispatches.
    /// </summary>
    public bool PadOddColumns { get; set; }

    private int RunColumns(int n, int k, int m, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int maxN)
        => PadOddColumns && (n & 1) == 1 && (n == 5 || n == 7) && n < maxN
           && input.Size >= (long)(n + 1) * k * sizeof(float) && output.Size >= (long)(n + 1) * m * sizeof(float) ? n + 1 : n;

    /// <summary>Records the GEMV over <paramref name="n"/> activation columns (input <c>[n, k]</c>, output <c>[n, m]</c>).</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="weights">Packed weights, <paramref name="m"/> rows.</param>
    /// <param name="input">FP32 input rows.</param>
    /// <param name="output">FP32 output rows.</param>
    /// <param name="m">Output rows of the weight.</param>
    /// <param name="k">Contraction length.</param>
    /// <param name="n">Activation columns, 1..<see cref="MaxColumns"/>.</param>
    /// <param name="scale">Per-tensor output scale (NVFP4 <c>X.scale</c>; must be 1 for the other formats).</param>
    /// <param name="grid">IQ2_XXS grid table (required for <see cref="ExactGemvFormat.Iq2Xxs"/>).</param>
    /// <param name="ksigns">IQ2 ksigns table (required for <see cref="ExactGemvFormat.Iq2Xxs"/>).</param>
    /// <exception cref="ArgumentException">A dimension, buffer size or table is invalid.</exception>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n,
                              float scale = 1f, VulkanDevice.Buffer? grid = null, VulkanDevice.Buffer? ksigns = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        if (Format != ExactGemvFormat.Nvfp4 && scale != 1f) throw new ArgumentException($"{Format} takes no output scale.", nameof(scale));
        int blocksPerRow = k / _groupSize;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (weights.Size < m * rowBytes) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (input.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (output.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[_bindings.Length];
        buffers[0] = weights.Handle;
        buffers[1] = input.Handle;
        buffers[2] = output.Handle;
        if (Format == ExactGemvFormat.Iq2Xxs)
        {
            buffers[3] = (grid ?? throw new ArgumentNullException(nameof(grid))).Handle;
            buffers[4] = (ksigns ?? throw new ArgumentNullException(nameof(ksigns))).Handle;
        }

        nint set = _descriptorCache.GetOrCreate(buffers);
        int nRun = RunColumns(n, k, m, input, output, MaxColumns);
        var pipe = Pipeline(nRun);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        uint extra = Format == ExactGemvFormat.Nvfp4 ? BitConverter.SingleToUInt32Bits(scale) : (uint)((rowBytes + 3) / 4);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, extra };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipe.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        int nr = RowsPerWorkgroupFor(nRun);
        GemvColumns.Dispatch(cmdBuf, (m + nr - 1) / nr);
        Interlocked.Increment(ref Dispatches);
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
