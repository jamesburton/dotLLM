using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Identifies an I2_S GEMM shader variant: which SPIR-V to load and the output-tile
/// dimensions that shader was compiled for (they determine the dispatch group count).
/// </summary>
/// <param name="SpvFileName">File name of the SPIR-V module within the <c>spv</c> directory.</param>
/// <param name="TileM">Weight rows of <c>C</c> produced per workgroup.</param>
/// <param name="TileN">Token rows of <c>C</c> produced per workgroup.</param>
/// <param name="RequiresCooperativeMatrix">
/// <c>true</c> when the variant's SPIR-V uses <c>VK_KHR_cooperative_matrix</c> and therefore
/// cannot be created on a device that does not advertise it.
/// </param>
/// <remarks>
/// Variants exist so a benchmark can A/B them side by side in one process — the
/// interleaved-paired methodology the Arc requires cannot span processes. Every variant
/// computes the same result; the thread-to-output mapping differs, and the coopmat variant
/// additionally carries F16 operand rounding (see <see cref="Coopmat"/>).
/// </remarks>
public readonly record struct I2SGemmVariant(
    string SpvFileName, int TileM, int TileN, bool RequiresCooperativeMatrix = false)
{
    /// <summary>
    /// Baseline: 16x16 tile, one thread per output cell (0.5 MAC per shared load).
    /// Retained as the benchmark comparand; <see cref="RegisterBlocked"/> supersedes it in production.
    /// </summary>
    public static I2SGemmVariant Scalar => new("matmul_i2_s_f32_gemm.spv", 16, 16);

    /// <summary>
    /// Register-blocked: 32x32 tile, one 2x2 micro-tile per thread (1.0 MAC per shared load).
    /// Measured 1.63-1.67x faster than <see cref="Scalar"/> across the BitNet 2B4T projection
    /// shapes on Meteor-Lake Arc (Xe-LPG) — the kernel is shared-memory-bound, so doubling
    /// arithmetic intensity converts almost directly to wall-clock.
    /// </summary>
    public static I2SGemmVariant RegisterBlocked => new("matmul_i2_s_f32_gemm_rb.spv", 32, 32);

    /// <summary>
    /// Cooperative-matrix: 16x16 tile, one subgroup per workgroup, F16 operands into an F32
    /// accumulator. Requires <c>VK_KHR_cooperative_matrix</c>.
    /// </summary>
    /// <remarks>
    /// Numerically looser than the F32-throughout scalar and register-blocked variants, but
    /// tighter than the Q8_0 coopmat kernel: I2_S has a single per-tensor scale, so the raw
    /// ternary values {-1, 0, +1} are staged exactly in F16 and the scale is applied once to
    /// the F32 accumulator. The A operand therefore contributes no rounding error — unlike
    /// Q8_0, which must fold its per-block scale into the F16 A operand. All F16 error comes
    /// from the activation cast alone.
    /// </remarks>
    public static I2SGemmVariant Coopmat =>
        new("matmul_i2_s_f32_gemm_coopmat.spv", 16, 16, RequiresCooperativeMatrix: true);

    /// <summary>
    /// Cooperative-matrix probe with a 32-thread workgroup and workgroup-size-agnostic staging.
    /// </summary>
    /// <remarks>
    /// Diagnostic for why <see cref="Coopmat"/> underperforms. That kernel declares a 64-thread
    /// workgroup (copied from the Q8_0 coopmat kernel, sized for AMD's 64-wide wave), so on a
    /// 32-wide device it contains TWO subgroups — and because the coopmat ops are at
    /// <c>gl_ScopeSubgroup</c>, both subgroups redundantly compute the SAME 16x16 tile and both
    /// store it. Correct, but roughly double the necessary compute. This variant uses one
    /// 32-thread workgroup so a wave32 device gets exactly one subgroup.
    /// Not portable as-is: on a 64-wide device a 32-thread workgroup is half a wave. The real fix
    /// is a specialization constant for the workgroup size, set per device from
    /// <see cref="VulkanDevice.SubgroupSize"/>.
    /// </remarks>
    public static I2SGemmVariant Coopmat32 =>
        new("matmul_i2_s_f32_gemm_coopmat32.spv", 16, 16, RequiresCooperativeMatrix: true);

    /// <summary>The variant used by the production forward path.</summary>
    public static I2SGemmVariant Production => RegisterBlocked;
}

/// <summary>
/// I2_S (BitNet b1.58 ternary) prefill-path batched GEMM:
/// <c>C[N, M] = B[N, K] @ W_i2s[M, K]^T</c>.
/// </summary>
/// <remarks>
/// Companion to <see cref="MatMulI2SGemvF32Kernel"/>. Same byte layout — each
/// 128-element block of a weight row is 32 packed bytes; row stride is
/// <c>K/4</c> bytes; a single per-tensor float32 scale sits at byte offset
/// <c>M·(K/4)</c> (the buffer tail) and is read in-shader. Tiling is 16x16
/// cells per workgroup, K-chunk = 128 elements (one I2_S block); each K-chunk
/// stages a 16x128 B tile and dequantises a 16x128 weight tile into shared
/// memory once and reuses both across the 16x16 cells.
/// </remarks>
public sealed class MatMulI2SGemmF32Kernel : IDisposable
{
    /// <summary>I2_S block: 128 ternary codes packed into 32 bytes.</summary>
    public const int I2SBlockBytes = 32;

    /// <summary>Elements per I2_S block.</summary>
    public const int I2SGroupSize = 128;

    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _tileM;
    private readonly int _tileN;
    private bool _disposed;

    private MatMulI2SGemmF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int tileM, int tileN)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _tileM = tileM;
        _tileN = tileN;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads the production I2_S GEMM variant and creates the pipeline.</summary>
    public static MatMulI2SGemmF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, I2SGemmVariant.Production);

    /// <summary>Loads the SPIR-V for <paramref name="variant"/> and creates the pipeline.</summary>
    /// <param name="device">Device to create the pipeline on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V modules.</param>
    /// <param name="variant">Which GEMM shader variant to load.</param>
    /// <returns>A kernel bound to the requested variant.</returns>
    /// <exception cref="FileNotFoundException">The variant's SPIR-V is missing from <paramref name="spvDir"/>.</exception>
    /// <exception cref="InvalidOperationException">
    /// The variant needs <c>VK_KHR_cooperative_matrix</c> and the device does not advertise it.
    /// Callers should check <see cref="VulkanDevice.HasCooperativeMatrix"/> first and fall back
    /// to <see cref="I2SGemmVariant.RegisterBlocked"/>.
    /// </exception>
    public static MatMulI2SGemmF32Kernel Create(VulkanDevice device, string spvDir, I2SGemmVariant variant)
    {
        if (variant.RequiresCooperativeMatrix && !device.HasCooperativeMatrix)
            throw new InvalidOperationException(
                $"I2_S GEMM variant '{variant.SpvFileName}' requires VK_KHR_cooperative_matrix support. " +
                "Check VulkanDevice.HasCooperativeMatrix before calling Create() and fall back to " +
                "I2SGemmVariant.RegisterBlocked when it is false.");

        string path = Path.Combine(spvDir, variant.SpvFileName);
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
        return new MatMulI2SGemmF32Kernel(device, module, pipeline, pool, variant.TileM, variant.TileN);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMM synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsI2S, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsI2S, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the I2_S GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsI2S, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
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
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsI2S.Handle, inputB.Handle, outputC.Handle };
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

        uint groupsX = (uint)((m + _tileM - 1) / _tileM);
        uint groupsY = (uint)((n + _tileN - 1) / _tileN);
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
