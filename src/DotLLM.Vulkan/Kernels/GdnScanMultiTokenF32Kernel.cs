using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Multi-token Gated DeltaNet (GDN) scan for Qwen3MoeHybrid — binds the full
/// whole-sequence q/k/v/g/beta/output buffers and walks the <c>seqLen</c>
/// dimension INSIDE the shader, mutating the per-sequence state matrix in
/// place between tokens.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the host-driven <c>for t in 0..seqLen { CopyTokenRow; GdnScanStep }</c>
/// loop in <see cref="VulkanQwen3MoeHybridTransformerModel.RecordGdnLayer"/>:
/// the per-token shader required six D2D copies per token before each
/// dispatch, growing kernel launches as O(seqLen). The multi-token kernel
/// collapses the entire scan into a single dispatch per layer.
/// </para>
/// <para>
/// Bit-parity vs the per-token <see cref="GdnScanStepF32Kernel"/> and the CPU
/// reference is preserved: the t-loop runs in order with a barrier at the end
/// of each token, and each phase's reduction order is identical.
/// </para>
/// </remarks>
public sealed class GdnScanMultiTokenF32Kernel : IDisposable
{
    // n_v_head, n_k_head, d_state, v_heads_per_k_head, seq_len (all u32)
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnScanMultiTokenF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, Variant variant)
    {
        _variant = variant;
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 7);
    }

    /// <summary>
    /// #445 factorial arms for the scan. Two independent hypotheses about why this kernel
    /// costs 26 % of a pp512 pass, crossed so they can be told apart in four measurements
    /// rather than guessed at in a ladder:
    /// <list type="bullet">
    /// <item><description><b>A — phase fusion.</b> Each state element is touched six times
    /// per token (decay r+w, retrieve r, rank-1 r+w, read r); fusing decay into retrieve and
    /// rank-1 into read-out makes it four.</description></item>
    /// <item><description><b>B — LDS residency.</b> The state matrix lives in a global SSBO.
    /// Splitting each head's columns across four workgroups of 32 lanes puts a
    /// <c>d_state x 32</c> slab (16 KiB) in LDS, read from and written to global exactly once
    /// per dispatch.</description></item>
    /// </list>
    /// Every arm is bit-exact against <c>DotLLM.Cpu.Kernels.GatedDeltaNetScan</c>; the arms
    /// change where values live and how often they are re-read, never what they are.
    /// Select with <c>DOTLLM_VK_GDN_SCAN_VARIANT</c> = <c>base</c> | <c>fused</c> |
    /// <c>lds</c> | <c>ldsfused</c>. Default <see cref="Baseline"/> — nothing ships changed
    /// until the A/B says which cell wins.
    /// </summary>
    public enum Variant
    {
        /// <summary>Arm 00 — shipping: global state, six accesses per token.</summary>
        Baseline,

        /// <summary>Arm 10 — factor A only: global state, four accesses per token.</summary>
        Fused,

        /// <summary>Arm 01 — factor B only: LDS-resident state, six accesses per token.</summary>
        Lds,

        /// <summary>Arm 11 — both factors.</summary>
        LdsFused,

        /// <summary>
        /// Arm 01 at COLS=64 — the WAVE-COUNT-MATCHED control for factor B. The device reports
        /// <c>subgroupSize</c> 64, so the COLS=32 arms run 192 half-empty wave64s where the
        /// shipping kernel runs 96 full ones: they change the memory level AND double the
        /// wavefront count. At COLS=64 the wavefront count is identical to the shipping kernel
        /// (48 heads x 128 lanes / 64 = 96 either way) and only the memory level differs.
        /// </summary>
        Lds64,

        /// <summary>Arm 11 at COLS=64 — wave-count-matched, both factors.</summary>
        Lds64Fused,
    }

    /// <summary>Lanes per workgroup, and therefore state columns per workgroup, in the LDS arms.</summary>
    private const int LdsColsPerGroup = 32;

    /// <summary>Columns per workgroup in the wave64-matched LDS arms — one full wave64 per group.</summary>
    private const int LdsColsPerGroupWave64 = 64;

    private readonly Variant _variant;

    /// <summary>Which factorial arm this pipeline was built from (#445).</summary>
    public Variant ActiveVariant => _variant;

    /// <summary>
    /// Raw <c>VkPipeline</c> handle — diagnostics only, so
    /// <c>VulkanDevice.GetShaderStatisticsAmd</c> can report this kernel's post-compile
    /// VGPR/SGPR/LDS/scratch allocation. No codepath's correctness or performance depends on it.
    /// </summary>
    internal nint PipelineHandle => _pipeline.Pipeline;

    private static Variant VariantFromEnv() =>
        Environment.GetEnvironmentVariable("DOTLLM_VK_GDN_SCAN_VARIANT") switch
        {
            "fused" => Variant.Fused,
            "lds" => Variant.Lds,
            "ldsfused" => Variant.LdsFused,
            "lds64" => Variant.Lds64,
            "lds64fused" => Variant.Lds64Fused,
            _ => Variant.Baseline,
        };

    private static string SpvFor(Variant v) => v switch
    {
        Variant.Fused => "gdn_scan_multi_token_fused_f32.spv",
        Variant.Lds => "gdn_scan_multi_token_lds_f32.spv",
        Variant.LdsFused => "gdn_scan_multi_token_lds_fused_f32.spv",
        Variant.Lds64 => "gdn_scan_multi_token_lds64_f32.spv",
        Variant.Lds64Fused => "gdn_scan_multi_token_lds64_fused_f32.spv",
        _ => "gdn_scan_multi_token_f32.spv",
    };

    /// <summary>Loads the selected variant's SPIR-V from <paramref name="spvDir"/>.</summary>
    public static GdnScanMultiTokenF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, VariantFromEnv());

    /// <summary>Loads a specific #445 factorial arm — used by the A/B harness and the parity tests.</summary>
    /// <param name="device">Device to create the pipeline on.</param>
    /// <param name="spvDir">Directory holding the compiled <c>.spv</c> blobs.</param>
    /// <param name="variant">Which arm to build.</param>
    public static GdnScanMultiTokenF32Kernel Create(VulkanDevice device, string spvDir, Variant variant)
    {
        string path = Path.Combine(spvDir, SpvFor(variant));
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[7];
            for (int i = 0; i < 7; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 7);
        return new GdnScanMultiTokenF32Kernel(device, module, pipeline, pool, variant);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, q, k, v, g, beta, output, seqLen, nVHead, nKHead, dState);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the multi-token GDN scan dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are full-sequence views (<c>q/k</c> shape <c>[seqLen, nKHead, dState]</c>,
    /// <c>v/output</c> shape <c>[seqLen, nVHead, dState]</c>, <c>g/beta</c> shape
    /// <c>[seqLen, nVHead]</c>); the per-token slicing happens inside the shader.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));
        if (nKHead <= 0) throw new ArgumentOutOfRangeException(nameof(nKHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nVHead % nKHead != 0)
            throw new ArgumentException(
                $"nVHead ({nVHead}) must be a multiple of nKHead ({nKHead}).");

        int vHeadsPerKHead = nVHead / nKHead;

        Span<nint> buffers = stackalloc nint[7]
        {
            state.Handle, q.Handle, k.Handle, v.Handle,
            g.Handle, beta.Handle, output.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)nVHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)nKHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)vHeadsPerKHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)seqLen);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Arms 00/10 keep one workgroup of d_state lanes per value head. Arms 01/11 split
        // each head's columns across ceil(d_state / 32) single-wave workgroups so the slab
        // fits LDS; total threads are identical either way, so this adds no wavefronts.
        int cols = _variant switch
        {
            Variant.Lds or Variant.LdsFused => LdsColsPerGroup,
            Variant.Lds64 or Variant.Lds64Fused => LdsColsPerGroupWave64,
            _ => 0,
        };
        uint groupsY = cols == 0 ? 1u : (uint)((dState + cols - 1) / cols);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nVHead, groupsY, 1);
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
