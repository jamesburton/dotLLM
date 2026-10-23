using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// I2_S (BitNet b1.58 ternary) decode-path GEMV: <c>y[M] = scale · W_i2s[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.MatMul.GemvI2_S</c> and
/// bitnet.cpp's I2_S: each 128 contiguous columns of a row form one 32-byte block —
/// byte <c>gp</c> (0..31) packs 4 ternary codes (2 bits each) for elements
/// <c>{gp, gp+32, gp+64, gp+96}</c> at bit offsets <c>{6,4,2,0}</c>, value = code − 1.
/// Row stride is <c>K/4</c> bytes; a single per-tensor float32 scale for the whole
/// matrix sits at byte offset <c>M·(K/4)</c> (the buffer tail) and is read in-shader.
/// Activation <c>x</c> and output <c>y</c> are FP32. One workgroup per output row,
/// 128 threads, shared-memory tree reduce. Correctness-first baseline (no coopmat /
/// W2A8 path yet) — the Meteor-Lake/Arc perf tuning and the AMD/Strix refinements
/// build on this.
/// </remarks>
public sealed class MatMulI2SGemvF32Kernel : IDisposable
{
    /// <summary>I2_S block: 128 ternary codes packed into 32 bytes.</summary>
    public const int I2SBlockBytes = QuantFormat.I2_SBlockBytes;

    /// <summary>Elements per I2_S block.</summary>
    public const int I2SGroupSize = QuantFormat.TernaryGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _rowsPerWorkgroup;
    private bool _disposed;

    private MatMulI2SGemvF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int rowsPerWorkgroup)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _rowsPerWorkgroup = rowsPerWorkgroup;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_i2_s_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    /// <summary>
    /// SPIR-V used by the production decode path: the 8-row multi-row GEMV.
    /// </summary>
    /// <remarks>
    /// The original kernel gave each workgroup a single output row, which at BitNet dims is only
    /// ~5 loop iterations per thread followed by a 7-step tree reduce — the reduction costs more
    /// than the dot product, and M=2560 launches 2560 such workgroups. Giving each workgroup 8 rows
    /// amortizes that overhead and lets each x[] value be loaded once and reused across all 8 rows,
    /// cutting activation traffic 8x. Measured 1.30-2.02x on Meteor-Lake Arc and BIT-IDENTICAL to
    /// the single-row kernel (same k, same per-thread stride, same tree reduce — only the
    /// row-to-workgroup mapping changes).
    /// </remarks>
    public const string ProductionSpv = "matmul_i2_s_f32_gemv_mr8.spv";

    /// <summary>The single-row kernel, retained as the benchmark comparand and bit-exactness reference.</summary>
    public const string SingleRowSpv = "matmul_i2_s_f32_gemv.spv";

    /// <summary>Loads the production decode GEMV and creates the pipeline.</summary>
    /// <remarks>
    /// Falls back to the single-row kernel when the multi-row SPIR-V is absent, so a build whose
    /// shader directory predates this variant keeps working. Both produce identical results.
    /// </remarks>
    public static MatMulI2SGemvF32Kernel Create(VulkanDevice device, string spvDir)
        => File.Exists(Path.Combine(spvDir, ProductionSpv))
            ? Create(device, spvDir, ProductionSpv)
            : Create(device, spvDir, SingleRowSpv);

    /// <summary>
    /// Output rows a given SPIR-V produces per workgroup. The production kernel maps one row to
    /// one workgroup; multi-row variants amortize launch and reduction cost over several rows and
    /// therefore need a proportionally smaller dispatch.
    /// </summary>
    /// <param name="spvFileName">SPIR-V file name.</param>
    /// <returns>Rows produced per workgroup by that shader.</returns>
    private static int RowsPerWorkgroupFor(string spvFileName)
        => spvFileName.Contains("_mr8", StringComparison.Ordinal) ? 8
         : spvFileName.Contains("_mr4", StringComparison.Ordinal) ? 4
         : 1;

    /// <summary>
    /// Loads the named SPIR-V from <paramref name="spvDir"/> and creates the pipeline. The
    /// <paramref name="spvFileName"/> overload exists so a benchmark can A/B successive kernel
    /// variants (e.g. an alternate-mapping <c>.spv</c>) side by side in one process.
    /// </summary>
    public static MatMulI2SGemvF32Kernel Create(VulkanDevice device, string spvDir, string spvFileName)
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
        return new MatMulI2SGemvF32Kernel(device, module, pipeline, pool, RowsPerWorkgroupFor(spvFileName));
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsI2S, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsI2S, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the I2_S GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsI2S, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % I2SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {I2SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / I2SGroupSize;
        long rowBytes = (long)k / 4;                         // K/4 packed bytes per row
        int rowUints = (int)((rowBytes + 3) / 4);

        // Packed rows (m·K/4) plus the per-tensor float32 scale at the tail.
        long weightsMin = (long)m * rowBytes + sizeof(float);
        if (weightsI2S.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·K/4 + scale), got {weightsI2S.Size}.",
                nameof(weightsI2S));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsI2S.Handle, x.Handle, y.Handle };
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

        uint groupsX = (uint)((m + _rowsPerWorkgroup - 1) / _rowsPerWorkgroup);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, 1, 1);
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
