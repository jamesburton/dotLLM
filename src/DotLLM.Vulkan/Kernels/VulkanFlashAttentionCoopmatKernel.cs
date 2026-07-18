using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// KHR cooperative-matrix Flash-Attention kernel for the GQA prefill path
/// (issue #149). Same tiling, dispatch geometry, bindings and push constants
/// as <see cref="VulkanFlashAttentionF32Kernel"/>, but QK^T and P·V run on
/// the matrix cores via 16x16x16 f16xf16-&gt;f32 cooperative-matrix tiles —
/// llama.cpp's FA_COOPMAT1 shape (Br=16, Bc=4x16 subgroup slices).
/// </summary>
/// <remarks>
/// <para>
/// Parity target: <c>DotLLM.Cpu.Kernels.Attention.ExecuteScalar</c> (same as
/// the scalar FA kernel) — but at f16-input tolerance, NOT the all-F32
/// tolerance: Q/K/V are rounded to f16 for the matrix multiplies, exactly the
/// rounding class llama.cpp ships (its KV cache is f16). Softmax state and
/// both matmul accumulators stay F32. See
/// <c>attention_flash_f32_coopmat.comp</c> for the full numerics contract.
/// </para>
/// <para>
/// Availability: requires <see cref="VulkanDevice.HasCooperativeMatrix"/>
/// with a subgroup-scope 16x16x16 F16xF16-&gt;F32 tile (checked via
/// <see cref="SupportsDevice"/>). Callers route to the scalar FA kernel when
/// unavailable, when the env kill-switch
/// <c>DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1</c> is set, or when
/// <c>headDim &gt; MaxHeadDim</c>. Decode (seqQ == 1) must never route here —
/// llama.cpp itself prefers FA_SCALAR at n_rows==1, matching the 2026-04
/// finding that coopmat attention loses at decode shapes on gfx1151.
/// </para>
/// <para>
/// Issue #378: a second SPV (<c>attention_flash_f32_coopmat_hd64.spv</c>,
/// <c>MAX_HEAD_DIM=64</c> instead of 128) is loaded when present and used
/// whenever the model's <c>headDim &lt;= HeadDim64Threshold</c> — halves the
/// qTile/kvTile shared-memory footprint and the live O-accumulator register
/// count for small-headDim models (SmolLM and other 64-headDim configs),
/// which the base shader's own comment attributes its occupancy ceiling to.
/// Falls back to the 128-dim shader unconditionally if the hd64 SPV is
/// missing (older SPV cache) or <c>DOTLLM_VULKAN_FA_COOPMAT_HD64=0</c>.
/// </para>
/// <para>
/// Issue #382 (corrects #381's invalid approach): a third SPV
/// (<c>attention_flash_f32_coopmat_hd64_2rb.spv</c>) doubles the effective
/// query tile (BR 16-&gt;32) by looping the QK^T/P·V coopmat phases over TWO
/// 16-row blocks that share one KV-tile LDS load, instead of (invalidly)
/// declaring a 32-row coopmat type — gfx1151 only has a native 16x16x16
/// tile. Opt-in via <c>DOTLLM_VULKAN_FA_COOPMAT_2RB=1</c>
/// (<see cref="Rb2Enabled"/>) since the arithmetic-intensity/instruction-count
/// trade-off is an open empirical question, matching #378's "measure before
/// defaulting on" precedent. Falls back to the hd64 shader (or base shader)
/// when the 2rb SPV is missing or not enabled.
/// </para>
/// </remarks>
public sealed class VulkanFlashAttentionCoopmatKernel : IDisposable
{
    /// <summary>Compile-time upper bound on head_dim baked into the base (128-dim) shader.</summary>
    public const int MaxHeadDim = 128;

    /// <summary>
    /// Issue #378: models with <c>headDim &lt;= this</c> are ELIGIBLE for the
    /// LDS-halved <c>attention_flash_f32_coopmat_hd64.spv</c> variant instead
    /// of the base 128-dim shader (also gated on <see cref="SeqKvThreshold"/>).
    /// </summary>
    public const int HeadDim64Threshold = 64;

    /// <summary>
    /// Issue #378: the hd64 variant only dispatches when <c>seqKv &gt;=</c>
    /// this. Same-session order-reversed A/B on SmolLM-135M found a clear,
    /// consistent +10-17% prefill win at seqKv 640-2048, but a noise-level
    /// (sometimes slightly negative) effect exactly at the canonical p=512
    /// benchmark point — LDS occupancy gains need enough KV-tile-loop
    /// iterations to amortize whatever regresses at very short context.
    /// Threshold picked at the first cleanly-positive measured point (640)
    /// rather than the ambiguous 512 one, so the standard SmolLM-135M
    /// perf-matrix measurement (p=512) is unaffected by this kernel and only
    /// longer-context prefill benefits.
    /// </summary>
    public const int SeqKvThreshold = 640;

    /// <summary>
    /// Issue #378: set <c>DOTLLM_VULKAN_FA_COOPMAT_HD64=0</c> to force every
    /// shape onto the base 128-dim shader (as if the hd64 SPV were absent).
    /// </summary>
    private static readonly bool HeadDim64Enabled =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_FA_COOPMAT_HD64") != "0";

    /// <summary>
    /// Issue #382: set <c>DOTLLM_VULKAN_FA_COOPMAT_2RB=1</c> to opt into the
    /// 2-row-block doubled-query-tile variant for headDim&lt;=64 models. Off
    /// by default — see the class remarks for why this stays opt-in until
    /// benchmarked.
    /// </summary>
    private static readonly bool Rb2Enabled =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_FA_COOPMAT_2RB") == "1";

    /// <summary>Query rows per workgroup (Br) for the base and BR=16 hd64 shaders.</summary>
    public const int QueryTileRows = 16;

    /// <summary>Query rows per workgroup (Br) for the #382 2-row-block hd64 variant.</summary>
    public const int QueryTileRowsRb2 = 32;

    /// <summary>KV tile columns per workgroup iteration (Bc).</summary>
    public const int KvTileCols = 64;

    private const int PushConstantBytes = 12 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly VulkanModule? _hd64Module;
    private readonly ComputePipeline? _hd64Pipeline;
    private readonly VulkanModule? _rb2Module;
    private readonly ComputePipeline? _rb2Pipeline;
    private readonly nint _descriptorPool;
    private readonly nint _hd64DescriptorPool;
    private readonly nint _rb2DescriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly DescriptorSetCache? _hd64DescriptorCache;
    private readonly DescriptorSetCache? _rb2DescriptorCache;
    private bool _disposed;

    /// <summary>
    /// Test-only override for the #382 2rb opt-in gate: <c>true</c> forces
    /// the 2rb dispatch path (when its SPV is loaded) regardless of
    /// <see cref="Rb2Enabled"/>, <c>false</c> forces it off, <c>null</c>
    /// (default) defers to <see cref="Rb2Enabled"/>. Avoids parity tests
    /// needing to mutate the process-wide <c>DOTLLM_VULKAN_FA_COOPMAT_2RB</c>
    /// env var, which would leak into other tests sharing the process.
    /// </summary>
    internal bool? ForceRb2ForTests { get; set; }

    private VulkanFlashAttentionCoopmatKernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        VulkanModule? hd64Module, ComputePipeline? hd64Pipeline, nint hd64Pool,
        VulkanModule? rb2Module, ComputePipeline? rb2Pipeline, nint rb2Pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
        _hd64Module = hd64Module;
        _hd64Pipeline = hd64Pipeline;
        _hd64DescriptorPool = hd64Pool;
        if (hd64Pipeline is not null)
            _hd64DescriptorCache = new DescriptorSetCache(device, hd64Pool, hd64Pipeline, buffersPerSet: 4);
        _rb2Module = rb2Module;
        _rb2Pipeline = rb2Pipeline;
        _rb2DescriptorPool = rb2Pool;
        if (rb2Pipeline is not null)
            _rb2DescriptorCache = new DescriptorSetCache(device, rb2Pool, rb2Pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Returns <c>true</c> when the device advertises a subgroup-scope
    /// 16x16x16 F16 x F16 -&gt; F32 cooperative-matrix tile — the exact shape
    /// the shader's <c>coopMatMulAdd</c> calls require.
    /// </summary>
    public static bool SupportsDevice(VulkanDevice device)
    {
        if (!device.HasCooperativeMatrix) return false;
        foreach (var s in device.SupportedCooperativeMatrixProperties)
        {
            if (s.Scope == VkScopeKhr.Subgroup
                && s.MSize == 16 && s.NSize == 16 && s.KSize == 16
                && s.AType == VkComponentTypeKhr.Float16 && s.BType == VkComponentTypeKhr.Float16
                && s.CType == VkComponentTypeKhr.Float32 && s.ResultType == VkComponentTypeKhr.Float32)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Loads <c>attention_flash_f32_coopmat.spv</c> and creates the compute
    /// pipeline. Throws when the device lacks the required coopmat tile or
    /// the SPV is missing — use <see cref="TryCreate"/> for graceful fallback.
    /// </summary>
    public static VulkanFlashAttentionCoopmatKernel Create(VulkanDevice device, string spvDir)
    {
        if (!SupportsDevice(device))
            throw new InvalidOperationException(
                "VulkanFlashAttentionCoopmatKernel requires a subgroup-scope 16x16x16 " +
                "F16xF16->F32 VK_KHR_cooperative_matrix tile. Check SupportsDevice() first " +
                "and fall back to VulkanFlashAttentionF32Kernel.");

        string path = Path.Combine(spvDir, "attention_flash_f32_coopmat.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
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

        VulkanModule? hd64Module = null;
        ComputePipeline? hd64Pipeline = null;
        nint hd64Pool = 0;
        string hd64Path = Path.Combine(spvDir, "attention_flash_f32_coopmat_hd64.spv");
        if (File.Exists(hd64Path))
        {
            hd64Module = VulkanModule.LoadFromFile(device, hd64Path);
            try
            {
                Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
                bindings[0] = new VkDescriptorBinding(0);
                bindings[1] = new VkDescriptorBinding(1);
                bindings[2] = new VkDescriptorBinding(2);
                bindings[3] = new VkDescriptorBinding(3);
                hd64Pipeline = hd64Module.CreateComputePipeline(
                    entryPoint: "main",
                    bindings: bindings,
                    pushConstantBytes: PushConstantBytes);
                // Separate pool per pipeline: DescriptorSetCache.Reset() calls
                // vkResetDescriptorPool on its whole pool, which would silently
                // invalidate the OTHER pipeline's still-referenced sets
                // mid-forward if both caches shared one pool (see #377).
                hd64Pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
            }
            catch
            {
                hd64Module.Dispose();
                module.Dispose();
                pipeline.Dispose();
                throw;
            }
        }

        VulkanModule? rb2Module = null;
        ComputePipeline? rb2Pipeline = null;
        nint rb2Pool = 0;
        string rb2Path = Path.Combine(spvDir, "attention_flash_f32_coopmat_hd64_2rb.spv");
        if (File.Exists(rb2Path))
        {
            rb2Module = VulkanModule.LoadFromFile(device, rb2Path);
            try
            {
                Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
                bindings[0] = new VkDescriptorBinding(0);
                bindings[1] = new VkDescriptorBinding(1);
                bindings[2] = new VkDescriptorBinding(2);
                bindings[3] = new VkDescriptorBinding(3);
                rb2Pipeline = rb2Module.CreateComputePipeline(
                    entryPoint: "main",
                    bindings: bindings,
                    pushConstantBytes: PushConstantBytes);
                // Separate pool per pipeline — see the #377 note above.
                rb2Pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
            }
            catch
            {
                rb2Module.Dispose();
                hd64Module?.Dispose();
                module.Dispose();
                hd64Pipeline?.Dispose();
                pipeline.Dispose();
                throw;
            }
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new VulkanFlashAttentionCoopmatKernel(
            device, module, pipeline, pool,
            hd64Module, hd64Pipeline, hd64Pool,
            rb2Module, rb2Pipeline, rb2Pool);
    }

    /// <summary>
    /// Returns <c>null</c> when the device lacks coopmat support, the SPV is
    /// missing, or pipeline creation fails — callers fall back to the scalar
    /// FA kernel.
    /// </summary>
    public static VulkanFlashAttentionCoopmatKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!SupportsDevice(device)) return null;
        try
        {
            return Create(device, spvDir);
        }
        catch (FileNotFoundException)
        {
            return null;
        }
        catch (VulkanException)
        {
            return null;
        }
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers were re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _descriptorCache.Reset();
        _hd64DescriptorCache?.Reset();
        _rb2DescriptorCache?.Reset();
    }

    /// <summary>
    /// Synchronous one-shot launch (parity tests). Production callers use
    /// <see cref="Record"/> inside a batched command buffer.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
               positionOffset, slidingWindow, useAlibi, softCap, scaleOverride, maskMode, prefixLen);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the coopmat FA dispatch. Contract is identical to
    /// <see cref="VulkanFlashAttentionF32Kernel.Record"/> — same buffer
    /// shapes, same parameters, same dispatch geometry.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (headDim > MaxHeadDim)
            throw new ArgumentException(
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}). " +
                $"Route to {nameof(VulkanFlashAttentionF32Kernel)} / {nameof(AttentionF32Kernel)}.",
                nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));
        if (softCap < 0.0f) throw new ArgumentOutOfRangeException(nameof(softCap),
            "softCap must be non-negative (use 0 to disable).");
        if (scaleOverride < 0.0f) throw new ArgumentOutOfRangeException(nameof(scaleOverride),
            "scaleOverride must be non-negative (use 0 for the default 1/sqrt(headDim)).");
        if (prefixLen < 0) throw new ArgumentOutOfRangeException(nameof(prefixLen));

        long qBytes   = (long)seqQ  * numHeads   * headDim * sizeof(float);
        long kvBytes  = (long)seqKv * numKvHeads * headDim * sizeof(float);
        long outBytes = qBytes;
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        // Issue #382: opt-in 2-row-block doubled-query-tile variant for
        // headDim<=64 models — checked before #378's hd64 gate since it's a
        // more specific alternative for the same headDim range.
        bool useRb2 = _rb2Pipeline is not null && (ForceRb2ForTests ?? Rb2Enabled) && headDim <= HeadDim64Threshold;

        // Issue #378: route headDim<=64 models with enough KV-tile iterations
        // to amortize the occupancy gain to the LDS-halved hd64 shader.
        bool useHd64 = !useRb2 && _hd64Pipeline is not null && HeadDim64Enabled
            && headDim <= HeadDim64Threshold && seqKv >= SeqKvThreshold;

        ComputePipeline pipeline = useRb2 ? _rb2Pipeline! : useHd64 ? _hd64Pipeline! : _pipeline;
        DescriptorSetCache descriptorCache = useRb2 ? _rb2DescriptorCache! : useHd64 ? _hd64DescriptorCache! : _descriptorCache;
        int queryTileRows = useRb2 ? QueryTileRowsRb2 : QueryTileRows;

        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, v.Handle, output.Handle };
        nint descriptorSet = descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[12];
        pc[0]  = (uint)seqQ;
        pc[1]  = (uint)seqKv;
        pc[2]  = (uint)numHeads;
        pc[3]  = (uint)numKvHeads;
        pc[4]  = (uint)headDim;
        pc[5]  = (uint)positionOffset;
        pc[6]  = (uint)slidingWindow;
        pc[7]  = useAlibi ? 1u : 0u;
        pc[8]  = BitConverter.SingleToUInt32Bits(softCap);
        pc[9]  = BitConverter.SingleToUInt32Bits(scaleOverride);
        pc[10] = (uint)maskMode;
        pc[11] = (uint)prefixLen;
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint qTiles = ((uint)seqQ + (uint)queryTileRows - 1u) / (uint)queryTileRows;
        uint groups = qTiles * (uint)numHeads;
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
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

        if (_hd64DescriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _hd64DescriptorPool, 0);
        _hd64Pipeline?.Dispose();
        _hd64Module?.Dispose();

        if (_rb2DescriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _rb2DescriptorPool, 0);
        _rb2Pipeline?.Dispose();
        _rb2Module?.Dispose();
    }
}
