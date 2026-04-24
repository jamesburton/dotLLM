using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// FP32 scaled-dot-product attention with causal masking, GQA head broadcast,
/// and flash-attention-style online softmax. Three pipeline variants —
/// cooperative-matrix (KHR), subgroup-arithmetic, and shared-memory — are
/// loaded at construction time and the runtime picks the best available.
/// </summary>
/// <remarks>
/// <para>
/// Parity target: the scalar CPU reference (<c>DotLLM.Cpu.Kernels.Attention.ExecuteScalar</c>).
/// All three variants do a running max / sum_exp update per KV tile, rescale
/// the output accumulator by <c>exp(oldMax - newMax)</c>, and finally divide
/// by the running sum. The coopmat variant uses a 16×16×16 f16/f16/f32 tile
/// shape from <c>VK_KHR_cooperative_matrix</c> for the per-tile Q·K^T and
/// P·V multiplies; the subgroup variant replaces tree reductions with
/// <c>subgroupAdd</c> / <c>subgroupMax</c>; the shared-mem variant is the
/// broadest-portable baseline.
/// </para>
/// <para>
/// Tile size <c>TILE_KV = 256</c> matches CUDA (shared/subgroup). The coopmat
/// variant dispatches query-rows in blocks of 16 and processes KV tiles of
/// 16 columns per iteration. <c>MAX_HEAD_DIM = 256</c> in every shader
/// bounds the shared-memory footprint — well above any current
/// Llama/Mistral/Phi/DeepSeek/SmolLM head dim (64 or 128).
/// </para>
/// <para>
/// <b>Dispatch priority:</b>
/// <list type="number">
///   <item>Coopmat only when <see cref="VulkanDevice.HasCooperativeMatrix"/> AND <c>DOTLLM_VULKAN_USE_COOPMAT_ATTENTION=1</c> (opt-in, off by default — see note below).</item>
///   <item>Subgroup when <see cref="VulkanDevice.HasSubgroupArithmetic"/> and <c>DOTLLM_VULKAN_FORCE_SHARED_REDUCE</c> is not <c>1</c>.</item>
///   <item>Shared-mem — always present as the deepest fallback.</item>
/// </list>
/// Coopmat is off-by-default because on AMD RDNA3.5 iGPU (the reference
/// hardware) it measured 1.27–1.56× SLOWER than the shared-mem reduce at
/// decode/prefill shapes — wave=64 underfills the 16-lane WMMA tile and
/// the f32→f16 conversion overhead is not amortized. NVIDIA tensor cores
/// (subgroup=32) and AMD discrete (gfx110x) are expected to flip the
/// balance — opt in there via the env var.
/// </para>
/// </remarks>
public sealed class AttentionF32Kernel : IDisposable
{
    /// <summary>Fixed compile-time upper bound on head_dim in the shader.</summary>
    public const int MaxHeadDim = 256;

    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 7 * sizeof(uint); // seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow

    /// <summary>
    /// Env-var opt-in: use the coopmat attention pipeline. When set to
    /// <c>1</c>, the coopmat dispatch is preferred over subgroup/shared.
    /// Default is off because on AMD RDNA3.5 iGPU (the reference hardware
    /// measured at landing time) coopmat attention is 1.27–1.56× SLOWER than
    /// the shared-mem reduce — wave=64 underfills the 16-lane WMMA tile and
    /// the f32→f16 conversion overhead is not amortized at decode shapes.
    /// NVIDIA tensor cores (subgroup=32) and AMD discrete (gfx110x) are
    /// expected to flip this; set this env var on those targets to opt in.
    /// See <c>.perf-runs/vulkan-coopmat-attention-20260424/README.md</c>
    /// for the per-shape measurements and analysis.
    /// </summary>
    internal const string UseCoopmatEnvVar = "DOTLLM_VULKAN_USE_COOPMAT_ATTENTION";

    private readonly VulkanDevice _device;
    private readonly VulkanModule _sharedModule;
    private readonly ComputePipeline _sharedPipeline;
    private readonly VulkanModule? _subgroupModule;
    private readonly ComputePipeline? _subgroupPipeline;
    private readonly VulkanModule? _coopmatModule;
    private readonly ComputePipeline? _coopmatPipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly DispatchMode _mode;
    private bool _disposed;

    /// <summary>The three pipeline families this kernel may dispatch.</summary>
    public enum DispatchMode
    {
        /// <summary><c>attention_f32.spv</c> — shared-memory tree reduces.</summary>
        SharedMem,
        /// <summary><c>attention_f32_sg.spv</c> — subgroup-arithmetic reduces.</summary>
        Subgroup,
        /// <summary><c>attention_f32_coopmat.spv</c> — <c>VK_KHR_cooperative_matrix</c> tiled multiplies.</summary>
        Coopmat,
    }

    /// <summary>The dispatch mode chosen at construction; fixed for the lifetime of the kernel.</summary>
    public DispatchMode Mode => _mode;

    /// <summary>
    /// True when this kernel dispatches the <c>attention_f32_sg.spv</c>
    /// subgroup-arithmetic variant. Retained for backward-compatible
    /// telemetry; prefer <see cref="Mode"/> for new call sites.
    /// </summary>
    public bool UsesSubgroupReduce => _mode == DispatchMode.Subgroup;

    /// <summary>True when this kernel dispatches the cooperative-matrix SPV.</summary>
    public bool UsesCooperativeMatrix => _mode == DispatchMode.Coopmat;

    private AttentionF32Kernel(
        VulkanDevice device,
        VulkanModule sharedModule, ComputePipeline sharedPipeline,
        VulkanModule? subgroupModule, ComputePipeline? subgroupPipeline,
        VulkanModule? coopmatModule, ComputePipeline? coopmatPipeline,
        nint pool, DispatchMode mode)
    {
        _device = device;
        _sharedModule = sharedModule;
        _sharedPipeline = sharedPipeline;
        _subgroupModule = subgroupModule;
        _subgroupPipeline = subgroupPipeline;
        _coopmatModule = coopmatModule;
        _coopmatPipeline = coopmatPipeline;
        _descriptorPool = pool;
        _mode = mode;
        ComputePipeline active = mode switch
        {
            DispatchMode.Coopmat  => coopmatPipeline!,
            DispatchMode.Subgroup => subgroupPipeline!,
            _                     => sharedPipeline,
        };
        _descriptorCache = new DescriptorSetCache(device, pool, active.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>attention_f32.spv</c> (always), <c>attention_f32_sg.spv</c>
    /// (when the device advertises subgroup arithmetic), and
    /// <c>attention_f32_coopmat.spv</c> (when the device advertises
    /// <c>VK_KHR_cooperative_matrix</c> with 16×16×16 f16/f32 subgroup tiles)
    /// from the given directory and creates the pipelines.
    /// </summary>
    /// <remarks>
    /// The chosen dispatch is fixed at construction. See <see cref="Mode"/>.
    /// Both optional SPVs silently degrade when missing from disk — older
    /// builds keep working without the newer shaders.
    /// </remarks>
    public static AttentionF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string sharedPath = Path.Combine(spvDir, "attention_f32.spv");
        if (!File.Exists(sharedPath))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {sharedPath}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule sharedModule = VulkanModule.LoadFromFile(device, sharedPath);
        ComputePipeline sharedPipeline;
        try
        {
            sharedPipeline = CreatePipeline(sharedModule);
        }
        catch
        {
            sharedModule.Dispose();
            throw;
        }

        // ── Subgroup variant (optional) ─────────────────────────────────
        VulkanModule? subgroupModule = null;
        ComputePipeline? subgroupPipeline = null;
        bool subgroupAvailable = device.HasSubgroupArithmetic && !RmsNormF32Kernel.IsForceSharedReduce();
        if (subgroupAvailable)
        {
            string subgroupPath = Path.Combine(spvDir, "attention_f32_sg.spv");
            if (File.Exists(subgroupPath))
            {
                try
                {
                    subgroupModule = VulkanModule.LoadFromFile(device, subgroupPath);
                    subgroupPipeline = CreatePipeline(subgroupModule);
                }
                catch
                {
                    subgroupModule?.Dispose();
                    subgroupModule = null;
                    subgroupPipeline = null;
                }
            }
        }

        // ── Cooperative-matrix variant (optional) ───────────────────────
        //
        // Gated on HasCooperativeMatrix AND opt-in. Default is off (see class
        // XML doc for rationale). DOTLLM_VULKAN_FORCE_SHARED_REDUCE=1 forces
        // the shared-mem reference regardless.
        VulkanModule? coopmatModule = null;
        ComputePipeline? coopmatPipeline = null;
        // Coopmat is off by default on the reference hardware (AMD RDNA3.5
        // iGPU) because it measured 1.27–1.56× SLOWER than the shared-mem
        // reduce at decode/prefill shapes there. Opt-in via
        // DOTLLM_VULKAN_USE_COOPMAT_ATTENTION=1 on hardware where it wins
        // (NVIDIA tensor cores, AMD discrete). DOTLLM_VULKAN_FORCE_SHARED_REDUCE
        // remains the deepest fallback that disables both subgroup and coopmat.
        bool coopmatAvailable = device.HasCooperativeMatrix
            && IsUseCoopmat()
            && !RmsNormF32Kernel.IsForceSharedReduce();
        if (coopmatAvailable)
        {
            string coopmatPath = Path.Combine(spvDir, "attention_f32_coopmat.spv");
            if (File.Exists(coopmatPath))
            {
                try
                {
                    coopmatModule = VulkanModule.LoadFromFile(device, coopmatPath);
                    coopmatPipeline = CreatePipeline(coopmatModule);
                }
                catch
                {
                    coopmatModule?.Dispose();
                    coopmatModule = null;
                    coopmatPipeline = null;
                }
            }
        }

        // Dispatch-priority gate: coopmat > subgroup > shared.
        DispatchMode mode = DispatchMode.SharedMem;
        if (coopmatPipeline != null)
        {
            mode = DispatchMode.Coopmat;
        }
        else if (subgroupPipeline != null)
        {
            mode = DispatchMode.Subgroup;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new AttentionF32Kernel(
            device, sharedModule, sharedPipeline,
            subgroupModule, subgroupPipeline,
            coopmatModule, coopmatPipeline,
            pool, mode);
    }

    internal static bool IsUseCoopmat() =>
        Environment.GetEnvironmentVariable(UseCoopmatEnvVar) == "1";

    private static ComputePipeline CreatePipeline(VulkanModule module)
    {
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
        bindings[0] = new VkDescriptorBinding(0);
        bindings[1] = new VkDescriptorBinding(1);
        bindings[2] = new VkDescriptorBinding(2);
        bindings[3] = new VkDescriptorBinding(3);
        return module.CreateComputePipeline(
            entryPoint: "main",
            bindings: bindings,
            pushConstantBytes: PushConstantBytes);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches attention: <c>output = softmax((Q K^T)/sqrt(headDim) + mask) V</c>
    /// for every (query token, query head) pair. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="q">FP32 Q tensor, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">FP32 K tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">FP32 V tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">FP32 output, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query length.</param>
    /// <param name="seqKv">Key/value length (total context).</param>
    /// <param name="numHeads">Query-head count.</param>
    /// <param name="numKvHeads">KV-head count (must divide <paramref name="numHeads"/>).</param>
    /// <param name="headDim">Per-head dimension; must be &lt;= <see cref="MaxHeadDim"/>.</param>
    /// <param name="positionOffset">Offset added to q positions for causal masking (decode: cached-tokens count).</param>
    /// <param name="slidingWindow">Sliding-window size in tokens; <c>0</c> disables.</param>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow);
        ctx.SubmitAndWait();
    }

    /// <summary>Records attention into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0)
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
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}). Rebuild attention_f32.comp with a larger bound.",
                nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));

        long qBytes   = (long)seqQ  * numHeads   * headDim * sizeof(float);
        long kvBytes  = (long)seqKv * numKvHeads * headDim * sizeof(float);
        long outBytes = qBytes;
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        ComputePipeline pipeline = _mode switch
        {
            DispatchMode.Coopmat  => _coopmatPipeline!,
            DispatchMode.Subgroup => _subgroupPipeline!,
            _                     => _sharedPipeline,
        };

        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, v.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7]
        {
            (uint)seqQ,
            (uint)seqKv,
            (uint)numHeads,
            (uint)numKvHeads,
            (uint)headDim,
            (uint)positionOffset,
            (uint)slidingWindow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Workgroup count depends on dispatch mode:
        //   coopmat: one WG per (query-tile-of-16-rows, query-head)
        //   shared / subgroup: one WG per (query-token, query-head)
        uint groups;
        if (_mode == DispatchMode.Coopmat)
        {
            const int Br = 16;
            uint qTiles = ((uint)seqQ + (uint)Br - 1u) / (uint)Br;
            groups = qTiles * (uint)numHeads;
        }
        else
        {
            groups = (uint)seqQ * (uint)numHeads;
        }
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _coopmatPipeline?.Dispose();
        _coopmatModule?.Dispose();
        _subgroupPipeline?.Dispose();
        _subgroupModule?.Dispose();
        _sharedPipeline.Dispose();
        _sharedModule.Dispose();
    }
}
