using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Identifies a PQ2_0 GEMM shader variant: which SPIR-V to load and the output-tile
/// dimensions that shader was compiled for (they determine the dispatch group count).
/// </summary>
/// <param name="SpvFileName">File name of the SPIR-V module within the <c>spv</c> directory.</param>
/// <param name="TileM">Weight rows of <c>C</c> produced per workgroup.</param>
/// <param name="TileN">Token rows of <c>C</c> produced per workgroup.</param>
/// <param name="RequiresCooperativeMatrix">
/// <c>true</c> when the variant's SPIR-V uses <c>VK_KHR_cooperative_matrix</c> and therefore
/// cannot be created on a device that does not advertise it.
/// </param>
/// <param name="RequiresSubgroupSize">
/// When non-zero, the variant declares a workgroup size that only maps to exactly one subgroup
/// on a device with this <see cref="VulkanDevice.SubgroupSize"/>. Selecting it elsewhere is
/// correct but wasteful (or, on a wider device, half a wave).
/// </param>
/// <remarks>
/// Variants exist so a benchmark can A/B them side by side in one process — the interleaved
/// order-reversed methodology this box requires cannot span processes. Every variant computes the
/// same result; only the thread-to-output mapping and the accumulation path differ.
/// </remarks>
public readonly record struct PQ2_0GemmVariant(
    string SpvFileName, int TileM, int TileN,
    bool RequiresCooperativeMatrix = false, int RequiresSubgroupSize = 0)
{
    /// <summary>
    /// Register-blocked F32: 32x32 tile, 16x16 threads each owning a 2x2 micro-tile
    /// (1.0 MAC per shared load). Shipped by issue #233 and the correctness oracle for
    /// every later variant.
    /// </summary>
    public static PQ2_0GemmVariant RegisterBlocked => new("matmul_pq2_0_f32_gemm_rb.spv", 32, 32);

    /// <summary>
    /// Cooperative-matrix: 16x16 tile, F16 operands into an F32 accumulator, 64-thread
    /// workgroup. Requires <c>VK_KHR_cooperative_matrix</c>.
    /// </summary>
    /// <remarks>
    /// Unlike the Q8_0 coopmat kernel, the F16 A operand here carries <b>no</b> rounding error:
    /// PQ2_0 stores its group scale as an IEEE binary16 value and the ternary code is exactly
    /// {-1, 0, +1}, so the staged product is {-scale, ±0, +scale} — a sign flip or a zero, both
    /// exact in F16 and bit-identical to what the file holds. All F16 error comes from the
    /// activation cast alone.
    /// </remarks>
    public static PQ2_0GemmVariant Coopmat =>
        new("matmul_pq2_0_f32_gemm_coopmat.spv", 16, 16, RequiresCooperativeMatrix: true);

    /// <summary>
    /// Cooperative-matrix with a 32-thread workgroup — one subgroup per workgroup on a wave32
    /// device such as gfx1151 (RDNA3.5).
    /// </summary>
    /// <remarks>
    /// The coopmat ops are at <c>gl_ScopeSubgroup</c>, so <see cref="Coopmat"/>'s 64-thread
    /// workgroup contains two subgroups on a wave32 part and both redundantly compute and store
    /// the same 16x16 tile. Not portable to a 64-wide device, where 32 threads is half a wave —
    /// hence <c>RequiresSubgroupSize = 32</c>.
    /// </remarks>
    public static PQ2_0GemmVariant Coopmat32 =>
        new("matmul_pq2_0_f32_gemm_coopmat32.spv", 16, 16,
            RequiresCooperativeMatrix: true, RequiresSubgroupSize: 32);

    /// <summary>
    /// Picks the fastest variant this device can actually run.
    /// </summary>
    /// <remarks>
    /// Issue #236 measured the coopmat path on gfx1151 against #233's register-blocked kernel;
    /// see <c>MatMulPQ2_0GemmF32Kernel</c>'s remarks for the verdict that drives this choice.
    /// Devices without <c>VK_KHR_cooperative_matrix</c> always get
    /// <see cref="RegisterBlocked"/>, which is F32 throughout and needs no extension.
    /// </remarks>
    /// <param name="device">Device the pipeline will be created on.</param>
    /// <returns>The variant to load.</returns>
    public static PQ2_0GemmVariant SelectFor(VulkanDevice device)
    {
        if (Coopmat32.IsSupportedOn(device)) return Coopmat32;
        if (Coopmat.IsSupportedOn(device)) return Coopmat;
        return RegisterBlocked;
    }

    /// <summary>
    /// Whether <paramref name="device"/> can create a pipeline for this variant: the coopmat
    /// extension when the variant needs it, and a pinnable compute subgroup size when the
    /// variant declares one.
    /// </summary>
    /// <param name="device">Device to test.</param>
    /// <returns><c>true</c> when <see cref="MatMulPQ2_0GemmF32Kernel.Create(VulkanDevice, string, PQ2_0GemmVariant)"/> can succeed.</returns>
    public bool IsSupportedOn(VulkanDevice device)
    {
        if (RequiresCooperativeMatrix && !device.HasCooperativeMatrix) return false;
        if (RequiresSubgroupSize != 0
            && !device.SupportsRequiredSubgroupSize((uint)RequiresSubgroupSize, VkShaderStageFlags.Compute))
            return false;
        return true;
    }

    /// <summary>
    /// Every variant <paramref name="device"/> can run, cheapest-first. Benchmarks and the
    /// correctness gates enumerate this so a variant cannot rot unmeasured.
    /// </summary>
    /// <param name="device">Device to enumerate for.</param>
    /// <returns>The runnable variants, register-blocked first.</returns>
    public static IEnumerable<PQ2_0GemmVariant> AvailableOn(VulkanDevice device)
    {
        yield return RegisterBlocked;
        if (Coopmat.IsSupportedOn(device)) yield return Coopmat;
        if (Coopmat32.IsSupportedOn(device)) yield return Coopmat32;
    }
}

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
/// <b>Issue #236 reversed #233's coopmat decision on gfx1151.</b> #233 skipped the
/// cooperative-matrix line citing #229 (commit <c>a1161cbc</c>, warptile loses to plain coopmat)
/// and <c>0ea1bd7f</c> (plain I2_S coopmat loses to register-blocked on an RTX 3060). Measured
/// directly on the Radeon 8060S, neither transfers: same-session order-reversed A/B puts
/// <see cref="PQ2_0GemmVariant.Coopmat32"/> 1.81-2.15x ahead of
/// <see cref="PQ2_0GemmVariant.RegisterBlocked"/> — 1.81x on <c>lm_head</c>, the one shape too
/// large to sit in the 32 MB MALL and therefore the honest row. Against the pre-#233 looped-GEMV
/// fallback the same shape moves from 1.03x (register-blocked, i.e. parity) to 2.14x.
/// </para>
/// <para>
/// Most of that comes from the wave width, not the tile: the 64-thread coopmat variant is only
/// 1.33-1.62x over register-blocked, and pinning a 32-thread workgroup to wave32 via
/// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c> adds a further 1.29-1.79x. RDNA3.5's
/// WMMA is a wave32 instruction and the driver defaults compute to wave64 here
/// (<see cref="VulkanDevice.SubgroupSize"/> reports 64). Note that a 32-thread workgroup WITHOUT
/// the pin is a pessimisation — half a wave — which is what <c>I2SGemmVariant.Coopmat32</c>
/// measures at 0.66-0.81x.
/// </para>
/// <para>
/// The coopmat variants are <i>not</i> bit-exact against the CPU oracle and cannot be made so on
/// this device: gfx1151's <c>coopMatMulAdd</c> returns a result one F32 ULP toward −∞ on roughly a
/// third of cells even when the exact answer is representable. See
/// <c>VulkanMatMulPQ2_0GemmF32KernelTests.RealBonsaiTensor_DequantIsBitExactVsCpuOracle</c> for the
/// evidence. <see cref="PQ2_0GemmVariant.RegisterBlocked"/> stays the F32 reference.
/// </para>
/// <para>
/// Weight layout matches the CPU oracle <c>DotLLM.Cpu.Kernels.MatMul.UnpackPQ2_0Row</c>: each
/// 128 contiguous columns of a row form a 34-byte group — 2 bytes little-endian fp16 group scale
/// then 32 packed code bytes (byte <c>b</c> holds 4 CONSECUTIVE group-relative positions
/// <c>{4b, 4b+1, 4b+2, 4b+3}</c> at ASCENDING bit offsets <c>{0,2,4,6}</c>, value = code − 1 —
/// PrismML's real format, see #271). Row
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

    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _tileM;
    private readonly int _tileN;
    private bool _disposed;

    private MatMulPQ2_0GemmF32Kernel(
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

    /// <summary>
    /// Creates the pipeline for the fastest variant <paramref name="device"/> can run
    /// (<see cref="PQ2_0GemmVariant.SelectFor"/>).
    /// </summary>
    public static MatMulPQ2_0GemmF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, PQ2_0GemmVariant.SelectFor(device));

    /// <summary>
    /// Loads the SPIR-V for <paramref name="variant"/> and creates the pipeline. The explicit
    /// overload exists so a benchmark can A/B successive kernel variants side by side in one
    /// process — the interleaved order-reversed methodology this box requires cannot span
    /// processes.
    /// </summary>
    /// <param name="device">Device to create the pipeline on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V modules.</param>
    /// <param name="variant">Which GEMM shader variant to load.</param>
    /// <returns>A kernel bound to the requested variant.</returns>
    /// <exception cref="FileNotFoundException">The variant's SPIR-V is missing from <paramref name="spvDir"/>.</exception>
    /// <exception cref="InvalidOperationException">
    /// The variant needs <c>VK_KHR_cooperative_matrix</c> and the device does not advertise it.
    /// Callers should use <see cref="PQ2_0GemmVariant.SelectFor"/> rather than hard-coding.
    /// </exception>
    public static MatMulPQ2_0GemmF32Kernel Create(VulkanDevice device, string spvDir, PQ2_0GemmVariant variant)
    {
        if (variant.RequiresCooperativeMatrix && !device.HasCooperativeMatrix)
            throw new InvalidOperationException(
                $"PQ2_0 GEMM variant '{variant.SpvFileName}' requires VK_KHR_cooperative_matrix support. " +
                "Use PQ2_0GemmVariant.SelectFor(device), which falls back to RegisterBlocked.");

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
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: (uint)variant.RequiresSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulPQ2_0GemmF32Kernel(device, module, pipeline, pool, variant.TileM, variant.TileN);
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
