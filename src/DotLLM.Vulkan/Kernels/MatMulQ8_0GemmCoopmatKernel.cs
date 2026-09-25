using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Identifies a Q8_0 coopmat GEMM shader variant: which SPIR-V to load and the
/// pipeline subgroup-size pin (if any) it requires.
/// </summary>
/// <param name="SpvFileName">File name of the SPIR-V module within the <c>spv</c> directory.</param>
/// <param name="RequiredSubgroupSize">
/// When non-zero, pins the pipeline to this wave width via
/// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c> (see
/// <see cref="VulkanDevice.SupportsRequiredSubgroupSize"/>). The variant's shader
/// declares a workgroup size that maps to exactly one subgroup at this width.
/// </param>
/// <remarks>
/// Issue #240 audit: #236/PR#238 measured that pinning a coopmat pipeline's
/// workgroup to one subgroup at its true wave width (32 on gfx1151, where the
/// driver otherwise defaults compute to wave64) is worth a further 1.29-1.79x on
/// top of the coopmat tile itself (PQ2_0 GEMM). This mirrors that fix
/// (<c>PQ2_0GemmVariant</c>) for the Q8_0 GEMM coopmat kernel.
/// </remarks>
/// <param name="TileM">Weight rows of <c>C</c> produced per workgroup (sets the dispatch grid).</param>
/// <param name="TileN">Token rows of <c>C</c> produced per workgroup (sets the dispatch grid).</param>
/// <param name="RequiresNativeSubgroupSize">
/// When non-zero, the variant's fixed workgroup size only maps to the subgroup grid its shader
/// assumes on a device whose <see cref="VulkanDevice.SubgroupSize"/> is exactly this. See
/// <see cref="Blocked128x128x4"/>.
/// </param>
public readonly record struct Q8_0GemmCoopmatVariant(
    string SpvFileName, int RequiredSubgroupSize, int TileM = 16, int TileN = 16,
    int RequiresNativeSubgroupSize = 0)
{
    /// <summary>
    /// Environment variable that restores the pre-issue-#443 preference in
    /// <see cref="SelectFor"/>.
    /// </summary>
    public const string LegacyEnvVar = "DOTLLM_VK_Q8_0_GEMM_LEGACY";

    /// <summary>Baseline 64-thread coopmat kernel — one subgroup on a wave64 device, two on wave32.</summary>
    public static Q8_0GemmCoopmatVariant Coopmat64 => new("matmul_q8_0_gemm_coopmat.spv", 0);

    /// <summary>
    /// Issue #443: the proven 128x128, BK=32, four-wave64-subgroup blocked tile, instantiated
    /// from the shared <c>gemm_coopmat_blocked_*.glsl</c> template.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BK=32 is not a change for Q8_0 — <see cref="Coopmat64"/> already stepped K one 32-element
    /// block at a time — so the <b>only</b> thing this variant changes is the output tile, and
    /// with it the <c>M*K*N / BN</c> weight-staging term. The per-output reduction order is
    /// identical, so it is held to the existing coopmat parity tolerance.
    /// </para>
    /// <para>
    /// <b>This is not the Q8_0 prefill path on a device with dp4a.</b>
    /// <c>VulkanTransformerModel</c> prefers <c>MatMulQ8_0MmqKernel</c> (issue #50) and reaches
    /// the coopmat GEMM only as a fallback, so beating <see cref="Coopmat64"/> is necessary but
    /// not sufficient for an end-to-end win; the honest competitor is MMQ.
    /// </para>
    /// <para>
    /// <b>wave64 ONLY, and that is a correctness gate.</b> The shader declares
    /// <c>local_size_x = 256</c> and lays four subgroups out as a 2x2 grid over the tile via
    /// <c>gl_SubgroupID</c>. On a 32-wide device those 256 threads form EIGHT subgroups, and ids
    /// 4-7 index past the grid — reading <c>sharedB</c> out of bounds and storing into the next
    /// tile's rows while <c>tileAllIn</c> still says the fast path is safe. Hence
    /// <c>RequiresNativeSubgroupSize = 64</c>; #298 ran these Q8_0 coopmat kernels on an RTX 3060,
    /// so this is a reachable device, not a hypothetical one.
    /// </para>
    /// </remarks>
    public static Q8_0GemmCoopmatVariant Blocked128x128x4 =>
        new("matmul_q8_0_gemm_coopmat_128x128x4.spv", 0, TileM: 128, TileN: 128,
            RequiresNativeSubgroupSize: 64);

    /// <summary>
    /// 32-thread workgroup pinned to wave32 via <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c>.
    /// </summary>
    /// <remarks>
    /// Not portable to a 64-wide device unpinned — a 32-thread workgroup would be
    /// half a wave there, which issue #236 measured as a pessimisation (0.66-0.81x)
    /// for the equivalent PQ2_0 case. Hence the <see cref="VulkanDevice.SupportsRequiredSubgroupSize"/>
    /// gate in <see cref="IsSupportedOn"/>.
    /// </remarks>
    public static Q8_0GemmCoopmatVariant Coopmat32 => new("matmul_q8_0_gemm_coopmat32.spv", 32);

    /// <summary>
    /// Whether <paramref name="device"/> can create a pipeline for this variant right now: the
    /// pinnable subgroup size the variant declares (when it declares one), AND the compiled
    /// SPIR-V actually present in <paramref name="spvDir"/>.
    /// </summary>
    /// <remarks>
    /// The file-existence check matters because <see cref="Coopmat32"/>'s shader source may be
    /// checked in ahead of its compiled <c>.spv</c> (e.g. authored on a machine without the Vulkan
    /// SDK/<c>glslc</c> — see <c>native/vulkan/build.ps1</c>/<c>build.sh</c>). Gating on the file
    /// keeps <see cref="SelectFor"/> safe to call unconditionally: it silently falls back to
    /// <see cref="Coopmat64"/> until the shader is actually compiled and its <c>.spv</c> lands in
    /// <paramref name="spvDir"/>, rather than throwing <see cref="FileNotFoundException"/> out of
    /// what every existing caller treats as an unconditional default-selection call.
    /// </remarks>
    public bool IsSupportedOn(VulkanDevice device, string spvDir)
    {
        if (RequiredSubgroupSize != 0
            && !device.SupportsRequiredSubgroupSize((uint)RequiredSubgroupSize, VkShaderStageFlags.Compute))
            return false;
        // Issue #443: a fixed workgroup size only yields the subgroup grid the shader assumes at
        // one native wave width. Getting this wrong is silent corruption, not a pipeline failure.
        if (RequiresNativeSubgroupSize != 0 && device.SubgroupSize != (uint)RequiresNativeSubgroupSize)
            return false;
        return File.Exists(Path.Combine(spvDir, SpvFileName));
    }

    /// <summary>
    /// Picks the default variant for this device. Defaults to <see cref="Coopmat64"/>.
    /// </summary>
    /// <remarks>
    /// Issue #298: this used to unconditionally prefer <see cref="Coopmat32"/> whenever
    /// <see cref="IsSupportedOn"/> allowed it, on the assumption that pinning the true wave
    /// width is always a win (per #236/PQ2_0's 1.29-1.79x). Real same-session A/B measurement
    /// (<c>VulkanCoopmat32SubgroupPinBench</c>) on BOTH measured hardware families falsified
    /// that assumption for this kernel: RTX 3060 (576/1536/4096 shapes) 1.31x/1.43x/1.00x, but
    /// gfx1151 (Strix Halo, dotLLM's primary non-CUDA target) 1.17x/0.74x/0.65x — a real 27-35%
    /// REGRESSION at medium/large shapes, the ones closest to real inference workload. No
    /// shape/vendor pattern was consistent enough to justify a default-ON heuristic, so this now
    /// defaults to the always-safe baseline. <see cref="Coopmat32"/> remains available via the
    /// explicit-selection overload for A/B benchmarking; do not flip this default back without
    /// new measured evidence superseding the above.
    /// </remarks>
    public static Q8_0GemmCoopmatVariant SelectFor(VulkanDevice device, string spvDir)
    {
        // DOTLLM_VK_Q8_0_GEMM_LEGACY=1 restores the pre-#443 preference so the blocked tile can
        // be A/B'd against what shipped, in alternating processes, on a real model.
        if (Environment.GetEnvironmentVariable(LegacyEnvVar) != "1"
            && Blocked128x128x4.IsSupportedOn(device, spvDir))
            return Blocked128x128x4;

        return Coopmat64;
    }

    /// <summary>
    /// Every variant <paramref name="device"/> can run. Benchmarks and the parity gates
    /// enumerate this so a variant cannot rot unmeasured.
    /// </summary>
    /// <param name="device">Device to enumerate for.</param>
    /// <param name="spvDir">Directory the compiled SPIR-V is loaded from.</param>
    /// <returns>The runnable variants, legacy first.</returns>
    public static IEnumerable<Q8_0GemmCoopmatVariant> AvailableOn(VulkanDevice device, string spvDir)
    {
        if (Coopmat64.IsSupportedOn(device, spvDir)) yield return Coopmat64;
        if (Coopmat32.IsSupportedOn(device, spvDir)) yield return Coopmat32;
        if (Blocked128x128x4.IsSupportedOn(device, spvDir)) yield return Blocked128x128x4;
    }
}

/// <summary>
/// Q8_0 batched GEMM via <c>VK_KHR_cooperative_matrix</c>:
/// <c>C[N, M] = B[N, K] @ W_q8[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Semantic and binding parity with <see cref="MatMulQ8_0GemmKernel"/> — same
/// descriptor layout (3 storage buffers: Q8_0 weights, F32 input, F32 output),
/// same push constants (M, K, N, blocksPerRow, rowUints), same weight byte
/// format. The <i>numerical</i> result matches the scalar kernel to
/// abs 1e-4 / rel 1e-3 (F32 accumulator, F16 operand staging through
/// shared memory).
/// </para>
/// <para>
/// Availability: this kernel requires the physical device to advertise
/// <c>VK_KHR_cooperative_matrix</c> with a 16×16×16 F16×F16→F32 subgroup tile.
/// Callers must check <see cref="VulkanDevice.HasCooperativeMatrix"/> before
/// calling <see cref="Create(VulkanDevice, string)"/> — otherwise an exception is thrown. The
/// orchestrator wires runtime dispatch selection (coopmat vs scalar) in a
/// separate integration commit; this class only loads the SPIR-V and
/// dispatches.
/// </para>
/// <para>
/// Dispatch: 2-D grid, workgroup <c>(64, 1, 1)</c> — one subgroup per
/// output tile. Tile shape: 16 rows × 16 cols of C, K stepped 32 at a time
/// (one Q8_0 block per outer iteration, two <c>coopMatMulAdd</c> calls per
/// block with <c>TK=16</c>).
/// </para>
/// </remarks>
public sealed class MatMulQ8_0GemmCoopmatKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    // Dispatch grid comes from the VARIANT, not a constant: issue #443's blocked instantiation
    // produces a 128x128 tile where the legacy kernel produces 16x16, and a stale 16 here would
    // launch 64x too many workgroups, each recomputing and overwriting the same output.
    private readonly int _tileM;
    private readonly int _tileN;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0GemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        int tileM, int tileN)
    {
        _tileM = tileM;
        _tileN = tileN;
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads the fastest coopmat variant <paramref name="device"/> can run right now
    /// (<see cref="Q8_0GemmCoopmatVariant.SelectFor"/>) and creates the pipeline. Requires
    /// <see cref="VulkanDevice.HasCooperativeMatrix"/> to be <c>true</c> —
    /// throws <see cref="InvalidOperationException"/> otherwise so the caller
    /// can fall back to <see cref="MatMulQ8_0GemmKernel"/>.
    /// </summary>
    public static MatMulQ8_0GemmCoopmatKernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, Q8_0GemmCoopmatVariant.SelectFor(device, spvDir));

    /// <summary>
    /// Loads the SPIR-V for <paramref name="variant"/> and creates the pipeline. The explicit
    /// overload exists so a benchmark can A/B <see cref="Q8_0GemmCoopmatVariant.Coopmat64"/> vs
    /// <see cref="Q8_0GemmCoopmatVariant.Coopmat32"/> side by side in one process.
    /// </summary>
    /// <exception cref="FileNotFoundException">The variant's SPIR-V is missing from <paramref name="spvDir"/>.</exception>
    public static MatMulQ8_0GemmCoopmatKernel Create(VulkanDevice device, string spvDir, Q8_0GemmCoopmatVariant variant)
    {
        if (!device.HasCooperativeMatrix)
            throw new InvalidOperationException(
                "MatMulQ8_0GemmCoopmatKernel requires VK_KHR_cooperative_matrix support. " +
                "Check VulkanDevice.HasCooperativeMatrix before calling Create() and fall " +
                "back to MatMulQ8_0GemmKernel when it is false.");

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
                requiredSubgroupSize: (uint)variant.RequiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulQ8_0GemmCoopmatKernel(device, module, pipeline, pool, variant.TileM, variant.TileN);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the coopmat GEMM synchronously (wraps <see cref="Record"/>
    /// with a one-shot submit + fence wait). Use <see cref="Record"/> directly
    /// on the forward-pass command buffer in production.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the coopmat Q8_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K / 32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="inputB">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 32).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsQ8.Handle, inputB.Handle, outputC.Handle };
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
