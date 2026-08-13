using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

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
/// <param name="RequiredSubgroupSize">
/// Wave width this variant's workgroup size REQUIRES, pinned via
/// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c>, or <c>0</c> when the variant is
/// wave-width agnostic and the driver's own choice is fine. A variant with a non-zero value
/// cannot be created on a device that does not support pinning that size — the pin and the
/// declared workgroup size are a PAIR (issue #239): a 32-thread workgroup running as half a
/// 64-wide wave wastes half the machine, which is precisely why the unpinned
/// <see cref="Coopmat32"/> measured 0.66-0.81x.
/// </param>
/// <remarks>
/// Variants exist so a benchmark can A/B them side by side in one process — the
/// interleaved-paired methodology the Arc requires cannot span processes. Every variant
/// computes the same result; the thread-to-output mapping differs, and the coopmat variant
/// additionally carries F16 operand rounding (see <see cref="Coopmat"/>).
/// </remarks>
public readonly record struct I2SGemmVariant(
    string SpvFileName, int TileM, int TileN, bool RequiresCooperativeMatrix = false,
    uint RequiredSubgroupSize = 0)
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
    /// <see cref="Coopmat32"/> with its 32-thread workgroup PINNED to wave32 via
    /// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c>.
    /// </summary>
    /// <remarks>
    /// The pin is not an optimisation bolted onto <see cref="Coopmat32"/>; it is the missing
    /// half of it. RDNA3.5 (gfx1151) defaults compute dispatch to wave64, so an unpinned
    /// 32-thread workgroup is HALF A WAVE and half the lanes idle — which is exactly the
    /// 0.66-0.81x that <see cref="Coopmat32"/> measures there. Issue #236 / PR #238 measured
    /// the pin as worth a further 1.29-1.79x on top of the coopmat tile for the PQ2_0 GEMM.
    /// </remarks>
    public static I2SGemmVariant Coopmat32Wave32 =>
        new("matmul_i2_s_f32_gemm_coopmat32.spv", 16, 16,
            RequiresCooperativeMatrix: true, RequiredSubgroupSize: 32);

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

    /// <summary><see cref="CoopmatWarptile"/> with its 32-thread workgroup pinned to wave32.</summary>
    /// <remarks>Same pairing argument as <see cref="Coopmat32Wave32"/>; larger output tile.</remarks>
    public static I2SGemmVariant CoopmatWarptileWave32 =>
        new("matmul_i2_s_f32_gemm_coopmat_wt.spv", 32, 32,
            RequiresCooperativeMatrix: true, RequiredSubgroupSize: 32);

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

    /// <summary>
    /// Register-blocked with the WEIGHT tile stored as int8 (activations stay F32). Bit-exact with
    /// <see cref="RegisterBlocked"/>; shared memory 32 KB -> 20 KB and weight-side SLM traffic quartered.
    /// </summary>
    /// <remarks>
    /// Follow-on from the SLM-traffic finding: F32 moves 4 bytes per weight access, F16 moves 2 and
    /// won 1.22-1.50x, so int8 at 1 byte tests whether the gain keeps scaling with bytes moved.
    /// Bit-exact for the same reason as F16 — ternary {-1, 0, +1} is exactly representable in int8,
    /// so <c>float(int8_t(v)) == v</c>, leaving products and accumulation order unchanged.
    /// Requires <c>GL_EXT_shader_8bit_storage</c>; see <see cref="ProductionPreference"/> for how a
    /// device lacking it degrades.
    /// </remarks>
    public static I2SGemmVariant RegisterBlockedWeightInt8 => new("matmul_i2_s_f32_gemm_rb_wi8.spv", 32, 32);

    /// <summary>
    /// The variant used by the production forward path when no device-specific selection is
    /// made. Bit-exact F32 throughout; the correctness oracle for every coopmat and reduced-
    /// precision-storage challenger.
    /// </summary>
    public static I2SGemmVariant Production => RegisterBlocked;

    /// <summary>
    /// Picks the I2_S GEMM variant for <paramref name="device"/>, preferring a coopmat variant
    /// only where the device can supply BOTH cooperative matrix and a pinned wave32 compute
    /// subgroup size, and falling back to <see cref="RegisterBlocked"/> otherwise (the caller,
    /// <see cref="MatMulI2SGemmF32Kernel.Create(VulkanDevice, string)"/>, then tries
    /// <see cref="ProductionPreference"/> instead of stopping at that fallback).
    /// </summary>
    /// <param name="device">Device the pipeline will be created on.</param>
    /// <remarks>
    /// <para>
    /// Measured on gfx1151 (Radeon 8060S), 64 tokens, interleaved order-alternated paired A/B,
    /// median of 9 passes, against the production register-blocked bar:
    /// <c>o_proj 1.74x</c>, <c>gate/up 1.67x</c>, <c>down 1.64x</c> for
    /// <see cref="Coopmat32Wave32"/> — consistent across all three projection shapes, and the
    /// best of the four coopmat variants (plain 64-thread <see cref="Coopmat"/> is faster on
    /// <c>o_proj</c> at 2.35x but collapses to 1.17x on <c>down</c>;
    /// <see cref="CoopmatWarptileWave32"/> is a flat 1.42-1.52x).
    /// </para>
    /// <para>
    /// DEFAULTED ON FOR AMD ONLY, because a coopmat win does not generalise: the I2_S coopmat
    /// GEMM was validated on an RTX 3060 and LOST to register-blocked there, and #233/#236
    /// already produced one discrete-vs-UMA inversion. Other vendors keep the bit-exact
    /// register-blocked kernel unless <c>DOTLLM_VULKAN_I2S_COOPMAT=1</c> forces evaluation;
    /// <c>DOTLLM_VULKAN_DISABLE_I2S_COOPMAT=1</c> forces register-blocked everywhere.
    /// </para>
    /// <para>
    /// The wave32 pin itself measured NEUTRAL here (1.00x on both coopmat32 and warptile — see
    /// issue #241): on this driver a 32-thread workgroup already compiles to wave32, so the
    /// gain is the one-subgroup workgroup and tile shape, not the pin. The pin is retained
    /// because it is the correct portable contract for a kernel whose workgroup IS its
    /// subgroup — without it, a driver that chose wave64 would run the workgroup as half a
    /// wave, and a 16-wide device would run two subgroups redundantly over the same tile.
    /// </para>
    /// <para>
    /// Coopmat carries F16 operand rounding, so it is held to 1 ULP rather than bit-exactness
    /// (a gfx1151 <c>coopMatMulAdd</c> device property established in PR #238), whereas
    /// <see cref="RegisterBlocked"/> stays asserted bit-exact.
    /// </para>
    /// </remarks>
    public static I2SGemmVariant SelectFor(VulkanDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (IsCoopmatDisabled()) return Production;
        if (!device.HasCooperativeMatrix) return Production;
        // The 32-thread workgroup and the pin are a pair — refuse the variant outright where
        // the wave width cannot be pinned, rather than shipping a half-wave dispatch.
        if (!device.SupportsRequiredSubgroupSize(32, VkShaderStageFlags.Compute)) return Production;
        if (device.VendorId != AmdVendorId && !IsCoopmatForced()) return Production;
        return Coopmat32Wave32;
    }

    /// <summary>PCI vendor ID for AMD — the only vendor this coopmat path is measured on.</summary>
    private const uint AmdVendorId = 0x1002;

    /// <summary>
    /// Env opt-out for the coopmat + wave32 I2_S GEMM
    /// (<c>DOTLLM_VULKAN_DISABLE_I2S_COOPMAT=1</c>) — forces the bit-exact register-blocked
    /// kernel on every device. Mirrors the <c>DOTLLM_VULKAN_DISABLE_WAVE32</c> convention.
    /// </summary>
    internal const string DisableCoopmatEnvVar = "DOTLLM_VULKAN_DISABLE_I2S_COOPMAT";

    /// <summary>
    /// Env opt-in (<c>DOTLLM_VULKAN_I2S_COOPMAT=1</c>) extending the coopmat selection to
    /// non-AMD devices, for evaluating it on hardware it has not been measured on.
    /// </summary>
    internal const string ForceCoopmatEnvVar = "DOTLLM_VULKAN_I2S_COOPMAT";

    private static bool IsCoopmatDisabled() =>
        string.Equals(Environment.GetEnvironmentVariable(DisableCoopmatEnvVar), "1", StringComparison.Ordinal);

    private static bool IsCoopmatForced() =>
        string.Equals(Environment.GetEnvironmentVariable(ForceCoopmatEnvVar), "1", StringComparison.Ordinal);

    /// <summary>
    /// Production variants in preference order for devices where <see cref="SelectFor"/> does not
    /// pick a coopmat variant, each falling back to the next when the device cannot create it. The
    /// final entry requires no optional features and is always creatable.
    /// </summary>
    /// <remarks>
    /// The faster variants narrow the weight tile to shrink SLM traffic, which needs small-type
    /// storage: F16 needs 16-bit storage, int8 needs 8-bit storage (int8 measured no further gain
    /// over F16 on Xe-LPG — see <see cref="RegisterBlockedWeightInt8"/> — so it is not in this
    /// chain). Those are optional Vulkan features, so rather than probe each one this chain simply
    /// attempts creation and degrades. Every entry is bit-identical to <see cref="RegisterBlocked"/>,
    /// so which one a device picks changes speed only — never results.
    /// </remarks>
    public static IReadOnlyList<I2SGemmVariant> ProductionPreference { get; } =
    [
        RegisterBlockedWeightF16,
        RegisterBlocked,
    ];
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
    public const int I2SBlockBytes = QuantFormat.I2_SBlockBytes;

    /// <summary>Elements per I2_S block.</summary>
    public const int I2SGroupSize = QuantFormat.TernaryGroupSize;

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

    /// <summary>
    /// Creates the fastest production I2_S GEMM variant for <paramref name="device"/>: the
    /// coopmat + wave32 variant where <see cref="I2SGemmVariant.SelectFor"/> picks it (AMD only,
    /// issue #239), else the best creatable entry in
    /// <see cref="I2SGemmVariant.ProductionPreference"/> (F16 weight-tile where the device
    /// supports 16-bit storage, else the bit-exact register-blocked F32 kernel that needs no
    /// optional feature).
    /// </summary>
    /// <param name="device">Device to create the pipeline on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V modules.</param>
    /// <returns>A kernel bound to the best creatable variant.</returns>
    /// <remarks>
    /// The two tracks are independent and vendor-disjoint in practice (coopmat32wave32 is
    /// measured only on AMD gfx1151; the F16 weight-tile win is measured on Intel Xe-LPG), so
    /// <see cref="I2SGemmVariant.SelectFor"/> is tried first and, if it declines (returns the
    /// plain <see cref="I2SGemmVariant.RegisterBlocked"/> fallback), <see cref="I2SGemmVariant.ProductionPreference"/>
    /// takes over rather than stopping at that fallback immediately. Every candidate across both
    /// tracks is bit-identical (coopmat is held to 1 ULP instead — see <see cref="I2SGemmVariant.SelectFor"/>),
    /// so which one a device ends up on changes speed only — never results.
    /// </remarks>
    public static MatMulI2SGemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        I2SGemmVariant coopmatPick = I2SGemmVariant.SelectFor(device);
        if (coopmatPick != I2SGemmVariant.RegisterBlocked)
        {
            // SelectFor already verified every precondition for this exact variant (coopmat
            // support, the wave32 pin, vendor/opt-in), so creation is expected to succeed.
            return Create(device, spvDir, coopmatPick);
        }

        var candidates = I2SGemmVariant.ProductionPreference;
        for (int i = 0; i < candidates.Count; i++)
        {
            // The final candidate requires no optional feature — let its failure surface.
            if (i == candidates.Count - 1)
                return Create(device, spvDir, candidates[i]);

            try
            {
                return Create(device, spvDir, candidates[i]);
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidOperationException or VulkanException)
            {
                // Device lacks the storage feature, or the SPIR-V predates this build. Try the next.
            }
        }

        throw new InvalidOperationException("No I2_S GEMM variant could be created.");
    }

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

        // A variant that declares a required wave width cannot be created without it: its
        // workgroup size was chosen to be exactly one subgroup, so running it at the driver's
        // default width would silently halve (or double-count) the work. Refuse rather than
        // silently drop the pin — SelectFor is responsible for not asking on such a device.
        if (variant.RequiredSubgroupSize != 0
            && !device.SupportsRequiredSubgroupSize(variant.RequiredSubgroupSize, VkShaderStageFlags.Compute))
        {
            throw new InvalidOperationException(
                $"I2_S GEMM variant '{variant.SpvFileName}' requires a pinned compute subgroup size of " +
                $"{variant.RequiredSubgroupSize}, which this device does not support. Check " +
                "VulkanDevice.SupportsRequiredSubgroupSize before calling Create() and fall back to " +
                "I2SGemmVariant.RegisterBlocked (I2SGemmVariant.SelectFor does this).");
        }

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
                requiredSubgroupSize: variant.RequiredSubgroupSize);
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
