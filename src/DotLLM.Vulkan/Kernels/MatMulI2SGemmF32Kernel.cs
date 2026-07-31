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

    /// <summary>
    /// Cooperative-matrix warptile: a 2x2 grid of 16x16 fragments giving a 32x32 output tile
    /// per workgroup, one subgroup (32 threads). Requires <c>VK_KHR_cooperative_matrix</c>.
    /// </summary>
    /// <remarks>
    /// Tests the leading explanation for why <see cref="Coopmat"/> loses to
    /// <see cref="RegisterBlocked"/>: that kernel emits a single 16x16 fragment per workgroup, so
    /// it launches 4x the workgroups for the same output and gets half the data reuse. Here each
    /// K-slice loads 2 A and 2 B fragments and issues 4 <c>coopMatMulAdd</c>, so every loaded
    /// fragment feeds two multiplies — llama.cpp <c>mul_mm</c>'s warptile idea — and the output
    /// tile matches <see cref="RegisterBlocked"/> exactly.
    /// Pinned to one subgroup per workgroup so that <see cref="Coopmat32"/> to this variant is a
    /// controlled A/B in which only the tile size changes.
    /// </remarks>
    public static I2SGemmVariant CoopmatWarptile =>
        new("matmul_i2_s_f32_gemm_coopmat_wt.spv", 32, 32, RequiresCooperativeMatrix: true);

    /// <summary>
    /// Register-blocked with a wide weight unpack: one aligned 32-bit load per thread
    /// (4 packed bytes = 16 ternary codes) instead of four redundant loads of the same word.
    /// </summary>
    /// <remarks>
    /// Same 32x32 tile and 2x2 micro-tile as <see cref="RegisterBlocked"/>; only the unpack
    /// differs. Motivated by the coopmat findings (issue #229): three cooperative-matrix attempts
    /// all lost to the F32 kernel because this GEMM is bound by I2_S unpacking and shared staging
    /// rather than by the multiply, so the unpack is where the remaining time actually is.
    /// The four byte offsets the baseline fetched separately are provably one aligned word —
    /// <c>rowBytes = K/4</c> is a multiple of 32 (K is a multiple of 128), and both
    /// <c>blk*32</c> and the 4-aligned in-block offset preserve alignment.
    /// </remarks>
    public static I2SGemmVariant RegisterBlockedWide => new("matmul_i2_s_f32_gemm_rb_w4.spv", 32, 32);

    /// <summary>
    /// Register-blocked with one padding word on the <c>sharedW</c> row stride (129 instead of 128)
    /// to eliminate shared-memory bank conflicts in the inner loop.
    /// </summary>
    /// <remarks>
    /// Shared memory has 32 four-byte banks, so bank = word index mod 32. A 128-word row stride is
    /// 0 mod 32, so all 16 distinct <c>lx</c> values read the SAME bank for a given <c>j</c> — a
    /// 16-way conflict on every one of the 128 inner iterations. A 129-word stride gives
    /// bank = (lx + j) mod 32 (since 129 = 1 mod 32), spreading the 16 lanes across 16 banks.
    /// <c>sharedB</c> stays unpadded: threads in a warp share <c>ly</c>, so those reads broadcast
    /// one address and are already conflict-free.
    /// Bit-exact with <see cref="RegisterBlocked"/> — only the shared layout changes.
    /// </remarks>
    public static I2SGemmVariant RegisterBlockedPadded => new("matmul_i2_s_f32_gemm_rb_pad.spv", 32, 32);

    /// <summary>
    /// Occupancy probe: identical tiling to <see cref="RegisterBlocked"/> but both shared tiles are
    /// stored as F16, halving shared memory from 32 KB to 16 KB per workgroup.
    /// </summary>
    /// <remarks>
    /// Discriminates the two readings of VTune's "XVE Array Stalled/Idle: 88.8%" on the Arc, which
    /// an unprivileged collection cannot split. If the kernel is <i>idle</i> (occupancy-limited),
    /// halving shared memory raises resident workgroups per Xe-core and throughput should rise; if
    /// it is <i>stalled</i> on latency with threads already resident, this measures flat. Opposite
    /// fixes, so the experiment is worth running either way.
    /// NOT bit-exact and not a shipping candidate: the ternary weights are lossless in F16, but the
    /// activations in <c>sharedB</c> do round. A shipping version would keep activations at F32 and
    /// shrink only the weight tile (32 KB to 24 KB), or pack the weights as int8 (to 20 KB).
    /// </remarks>
    public static I2SGemmVariant RegisterBlockedF16Shared => new("matmul_i2_s_f32_gemm_rb_f16s.spv", 32, 32);

    /// <summary>
    /// 4x4 register-blocked (ILP) variant: same 32x32 output tile and same 32 KB of shared memory
    /// as <see cref="RegisterBlocked"/>, but 64 threads instead of 256, each owning a 4x4 micro-tile.
    /// </summary>
    /// <remarks>
    /// Chosen from profile data rather than guesswork. Elevated VTune gpu-hotspots on the Arc reports
    /// the production 2x2 kernel at 86.9% XVE stalled/idle with occupancy at 73.3% of peak — occupancy
    /// is high, so those are threads STALLED on memory, not idle. Footprint fixes therefore cannot
    /// help; the lever is instruction-level parallelism. Each thread here issues 8 shared loads to feed
    /// 16 FMAs across 16 INDEPENDENT accumulator chains, so many loads stay outstanding instead of
    /// stalling one at a time. Shared memory is held constant deliberately, isolating ILP as the only
    /// changed variable. The trade is fewer threads per workgroup; at 73.3% of peak there is occupancy
    /// headroom to spend, and whether ILP buys more than it costs is exactly what the benchmark decides.
    /// Accumulation order per output cell is unchanged, so results are bit-identical to
    /// <see cref="RegisterBlocked"/>.
    /// </remarks>
    public static I2SGemmVariant RegisterBlocked4x4 => new("matmul_i2_s_f32_gemm_rb4.spv", 32, 32);

    /// <summary>
    /// Register-blocked with the WEIGHT tile stored as F16 (activations stay F32). Bit-exact with
    /// <see cref="RegisterBlocked"/>; shared memory 32 KB -> 24 KB and weight-side SLM traffic halved.
    /// </summary>
    /// <remarks>
    /// Bit-exact because the staged values are raw ternary {-1, 0, +1}, each exactly representable in
    /// F16, so <c>float(float16_t(v)) == v</c> and neither the products nor the accumulation order
    /// change. Activations are never rounded.
    /// Also discriminates why <see cref="RegisterBlockedF16Shared"/> won: that halves both footprint
    /// (to 16 KB) and SLM bytes-per-access. This halves only weight traffic and takes footprint to
    /// 24 KB, so recovering most of the win implicates SLM TRAFFIC, while a much smaller gain would
    /// implicate footprint/occupancy instead.
    /// </remarks>
    public static I2SGemmVariant RegisterBlockedWeightF16 => new("matmul_i2_s_f32_gemm_rb_wf16.spv", 32, 32);

    /// <summary>The variant used by the production forward path.</summary>
    /// <remarks>
    /// <see cref="RegisterBlockedWeightF16"/>: bit-identical to <see cref="RegisterBlocked"/> and
    /// measured 1.22-1.50x faster on Meteor-Lake Arc. Halving weight-side SLM traffic recovers the
    /// full win of the all-F16 probe without rounding activations, which identified SLM read traffic
    /// (not occupancy, not bank conflicts, not global loads) as the real bottleneck.
    /// </remarks>
    public static I2SGemmVariant Production => RegisterBlockedWeightF16;
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
