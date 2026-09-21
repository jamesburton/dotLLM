using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Which wide-head (<c>headDim &gt; 128</c>) SPIR-V variant the scalar flash-attention
/// kernel loads alongside the 128-dim base shader (issue #441).
/// </summary>
/// <remarks>
/// The base <c>attention_flash_f32.comp</c> bakes <c>MAX_HEAD_DIM = 128</c> into its
/// <c>qTile</c>/<c>outAccum</c> shared-memory declarations, so a model with a wider head
/// cannot dispatch it at all. Bonsai 2 (qwen35) declares
/// <c>attention.key_length = value_length = 256</c> and therefore fell back to the
/// per-token <see cref="AttentionF32Kernel"/> on every prefill pass, silently.
/// <para>
/// The two 256-dim variants trade KV amortisation against occupancy:
/// <c>Hd256Br16</c> keeps the base shader's 16 query rows per workgroup (36.2 KB LDS,
/// one workgroup resident per RDNA3.5 CU); <c>Hd256Br8</c> halves the tile to 8 rows
/// (18.1 KB LDS, three workgroups resident). Selectable at runtime through
/// <c>DOTLLM_VK_FLASH_HD256</c> so the two can be A/B'd inside one process.
/// </para>
/// </remarks>
public enum FlashAttentionWideVariant
{
    /// <summary>Load no wide variant; <c>headDim &gt; 128</c> falls back to the per-token kernel.</summary>
    None = 0,

    /// <summary><c>attention_flash_f32_hd256_br16.spv</c> - 256-dim heads, 16 query rows per workgroup.</summary>
    Hd256Br16 = 1,

    /// <summary><c>attention_flash_f32_hd256_br8.spv</c> - 256-dim heads, 8 query rows per workgroup.</summary>
    Hd256Br8 = 2,

    /// <summary><c>attention_flash_f32_hd256_br4.spv</c> - 256-dim heads, 4 query rows per workgroup.</summary>
    Hd256Br4 = 3,
}

/// <summary>
/// Flash-Attention-v2 style FP32 attention kernel for the GQA prefill path.
/// Each workgroup processes one (query-head, query-tile of <see cref="QueryTileRows"/>
/// rows) pair, amortising K/V reads across the resident Q-tile.
/// </summary>
/// <remarks>
/// <para>
/// Parity target: <c>DotLLM.Cpu.Kernels.Attention.ExecuteTiled</c> (the CPU
/// online-softmax tiled reference). Same scale, same masking semantics, same
/// softmax shape. Numerical drift comes from reordered reductions; tolerance
/// is in line with the per-token <see cref="AttentionF32Kernel"/> (abs 1e-4 /
/// rel 1e-3 on Llama-shape configs).
/// </para>
/// <para>
/// The dispatch geometry expects <c>seqQ &gt; 1</c>: the FA path reduces K/V
/// traffic by <c>BR</c> when many Q-rows share a KV tile. For <c>seqQ == 1</c>
/// (decode) the legacy <see cref="AttentionF32Kernel"/> already does one KV
/// read per workgroup — FA would only add overhead. Callers MUST route decode
/// to the legacy kernel.
/// </para>
/// <para>
/// Soft-cap support: when the model carries a non-zero attention soft-cap
/// (Gemma 2, Qwen3 thinking variants), raw scores are passed through
/// <c>softCap * tanh(score / softCap)</c> before softmax. Pass <c>0.0f</c> to
/// disable.
/// </para>
/// </remarks>
public sealed class VulkanFlashAttentionF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on head_dim baked into the base shader.</summary>
    /// <remarks>
    /// This is the bound of <c>attention_flash_f32.spv</c> only. Since issue #441 the kernel
    /// can additionally carry a 256-dim variant; ask the INSTANCE
    /// (<see cref="SupportedMaxHeadDim"/>) what it can actually dispatch, and use
    /// <see cref="MaxSupportedHeadDim"/> for a pre-construction gate.
    /// </remarks>
    public const int MaxHeadDim = 128;

    /// <summary>Compile-time upper bound on head_dim baked into the wide (issue #441) variants.</summary>
    public const int WideMaxHeadDim = 256;

    /// <summary>
    /// Largest head_dim any variant of this kernel can dispatch. Load-time gates that run
    /// before a kernel instance exists should compare against this, then consult the created
    /// instance's <see cref="SupportedMaxHeadDim"/> (the wide SPV may be absent from an
    /// older build).
    /// </summary>
    public static int MaxSupportedHeadDim => WideMaxHeadDim;

    /// <summary>Environment override for the wide-head variant: <c>br16</c> (default), <c>br8</c>, or <c>off</c>.</summary>
    public const string WideVariantEnvVar = "DOTLLM_VK_FLASH_HD256";

    /// <summary>
    /// Number of query rows processed by a single workgroup. The KV stream
    /// is read once per workgroup, so this is the amortisation factor for
    /// KV memory traffic on the prefill path.
    /// </summary>
    public const int QueryTileRows = 16;

    /// <summary>KV tile (columns) per workgroup iteration.</summary>
    public const int KvTileCols = 64;

    private const int WorkgroupSize = KvTileCols;

    // 8 uints + 2 floats (softCap, scaleOverride) + 2 uints (maskMode, prefixLen) = 12 * 4 = 48 bytes.
    private const int PushConstantBytes = 12 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly VulkanModule? _wideModule;
    private readonly ComputePipeline? _widePipeline;
    private readonly nint _widePool;
    private readonly DescriptorSetCache? _wideDescriptorCache;
    private readonly int _wideQueryTileRows;
    private bool _disposed;

    private VulkanFlashAttentionF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        VulkanModule? wideModule, ComputePipeline? widePipeline, nint widePool,
        FlashAttentionWideVariant wideVariant)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
        _wideModule = wideModule;
        _widePipeline = widePipeline;
        _widePool = widePool;
        WideVariant = widePipeline is null ? FlashAttentionWideVariant.None : wideVariant;
        _wideQueryTileRows = WideVariant switch
        {
            FlashAttentionWideVariant.Hd256Br8 => 8,
            FlashAttentionWideVariant.Hd256Br4 => 4,
            _ => QueryTileRows,
        };
        if (widePipeline is not null)
            _wideDescriptorCache = new DescriptorSetCache(device, widePool, widePipeline, buffersPerSet: 4);
    }

    /// <summary>The wide-head variant this instance actually loaded, or <see cref="FlashAttentionWideVariant.None"/>.</summary>
    public FlashAttentionWideVariant WideVariant { get; }

    /// <summary>
    /// Largest head_dim THIS instance can dispatch: <see cref="WideMaxHeadDim"/> when a wide
    /// variant loaded, otherwise <see cref="MaxHeadDim"/>. Dispatch gates must use this, not
    /// the <see cref="MaxHeadDim"/> constant.
    /// </summary>
    public int SupportedMaxHeadDim =>
        WideVariant == FlashAttentionWideVariant.None ? MaxHeadDim : WideMaxHeadDim;

    /// <summary>Query rows per workgroup for a given head_dim (the wide br8 variant uses a smaller tile).</summary>
    /// <param name="headDim">Per-head dimension of the dispatch.</param>
    /// <returns>The <c>BR</c> the selected shader was compiled with.</returns>
    public int QueryTileRowsFor(int headDim) => headDim > MaxHeadDim ? _wideQueryTileRows : QueryTileRows;

    /// <summary>Resolves the wide variant requested by <see cref="WideVariantEnvVar"/>.</summary>
    /// <returns>The requested variant; <see cref="FlashAttentionWideVariant.Hd256Br16"/> when unset.</returns>
    public static FlashAttentionWideVariant WideVariantFromEnvironment()
        => (Environment.GetEnvironmentVariable(WideVariantEnvVar) ?? string.Empty).Trim().ToLowerInvariant() switch
        {
            // br4 is the DEFAULT on measurement, not on symmetry. Same-process, order-reversed,
            // interleaved A/B at Bonsai 2's real attention shape (24/4 heads, headDim 256),
            // min-ms over 5 rounds on gfx1151:
            //   seq  512: br16 11.71 | br8 5.02 | br4 3.32
            //   seq 2048: br16 174.1 | br8 90.8 | br4 65.8
            // (The same bench's per-token-kernel arm is NOT quoted: it measures 84.11 ms for a
            // dispatch the model's own GPU timestamps put at ~38 ms, unexplained. The size of the
            // win over the fallback comes from the end-to-end profile instead - attn_core
            // 604-811 ms -> 42-51 ms on a pp512 Bonsai 2 pass.)
            // Per-arm ranges are disjoint at both lengths. The ordering is the opposite of what
            // KV-traffic amortisation alone predicts - a smaller tile reads each KV row MORE
            // times - so amortisation is not the binding constraint at this head width: LDS
            // residency is. qTile + outAccum both scale with BR * MAX_HEAD_DIM, so at 256 dims
            // BR=16 costs 36.2 KB and pins one workgroup (4 wave64) to a CU, BR=8 costs 18.1 KB
            // and BR=4 costs 9.2 KB, which is ~6 workgroups / 24 waves of latency hiding.
            // BR=4 is the floor for this geometry, not a measured optimum: ROWS_PER_SLICE is
            // BR / (WG_SIZE / BC) = BR / 4, so BR=2 would give slices zero rows to own. Going
            // below it needs a narrower workgroup, which is a separate change.
            "" or "br4" or "1" or "on" => FlashAttentionWideVariant.Hd256Br4,
            "br8"                      => FlashAttentionWideVariant.Hd256Br8,
            "br16"                     => FlashAttentionWideVariant.Hd256Br16,
            "off" or "0" or "none"     => FlashAttentionWideVariant.None,
            var other => throw new ArgumentException(
                $"Unrecognised {WideVariantEnvVar} value '{other}'; expected br4, br8, br16 or off."),
        };

    /// <summary>SPIR-V file name for a wide variant.</summary>
    private static string WideSpvName(FlashAttentionWideVariant variant) => variant switch
    {
        FlashAttentionWideVariant.Hd256Br16 => "attention_flash_f32_hd256_br16.spv",
        FlashAttentionWideVariant.Hd256Br8  => "attention_flash_f32_hd256_br8.spv",
        FlashAttentionWideVariant.Hd256Br4  => "attention_flash_f32_hd256_br4.spv",
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    /// <summary>
    /// Loads <c>attention_flash_f32.spv</c> from the given directory and
    /// creates the compute pipeline. Throws if the SPV is missing — callers
    /// that want a graceful fallback should wrap in try/catch and route to
    /// <see cref="AttentionF32Kernel"/>.
    /// </summary>
    public static VulkanFlashAttentionF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, WideVariantFromEnvironment());

    /// <summary>
    /// Loads <c>attention_flash_f32.spv</c> plus, when <paramref name="wideVariant"/> asks for
    /// one and its SPV is present, a 256-dim wide-head variant (issue #441). A missing wide SPV
    /// is NOT an error - the kernel then reports <see cref="SupportedMaxHeadDim"/> = 128 and the
    /// caller's dispatch gate routes wide heads to the per-token kernel as before.
    /// </summary>
    /// <param name="device">Device to create the pipelines on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <param name="wideVariant">Wide-head variant to attempt to load.</param>
    /// <returns>The created kernel.</returns>
    public static VulkanFlashAttentionF32Kernel Create(
        VulkanDevice device, string spvDir, FlashAttentionWideVariant wideVariant)
    {
        string path = Path.Combine(spvDir, "attention_flash_f32.spv");
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);

        VulkanModule? wideModule = null;
        ComputePipeline? widePipeline = null;
        nint widePool = 0;
        if (wideVariant != FlashAttentionWideVariant.None)
        {
            string widePath = Path.Combine(spvDir, WideSpvName(wideVariant));
            if (File.Exists(widePath))
            {
                // The LOAD is inside the try as well as the pipeline build: a corrupt or
                // driver-rejected wide SPV must degrade this kernel to base-only, not throw out
                // of Create() where TryCreate would swallow it, lose flash attention entirely,
                // leak the base module/pipeline/pool, and then report "attention_flash_f32.spv is
                // missing" - which would be false. (attention_flash_f32_coopmat_hd64's loader
                // still has the un-guarded shape; noted, not copied.)
                try
                {
                    wideModule = VulkanModule.LoadFromFile(device, widePath);
                    Span<VkDescriptorBinding> wideBindings = stackalloc VkDescriptorBinding[4];
                    wideBindings[0] = new VkDescriptorBinding(0);
                    wideBindings[1] = new VkDescriptorBinding(1);
                    wideBindings[2] = new VkDescriptorBinding(2);
                    wideBindings[3] = new VkDescriptorBinding(3);
                    widePipeline = wideModule.CreateComputePipeline(
                        entryPoint: "main",
                        bindings: wideBindings,
                        pushConstantBytes: PushConstantBytes);
                    widePool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
                }
                catch
                {
                    widePipeline?.Dispose();
                    wideModule?.Dispose();
                    wideModule = null;
                    widePipeline = null;
                    widePool = 0;
                }
            }
        }

        return new VulkanFlashAttentionF32Kernel(
            device, module, pipeline, pool, wideModule, widePipeline, widePool, wideVariant);
    }

    /// <summary>
    /// <c>TryCreate</c> companion — returns <c>null</c> when the SPV is
    /// missing or pipeline creation fails, instead of throwing. Used by the
    /// model loader so older builds without the FA SPV silently fall back to
    /// the per-token shader.
    /// </summary>
    public static VulkanFlashAttentionF32Kernel? TryCreate(VulkanDevice device, string spvDir)
        => TryCreate(device, spvDir, WideVariantFromEnvironment());

    /// <summary>
    /// <see cref="Create(VulkanDevice, string, FlashAttentionWideVariant)"/> companion that
    /// returns <c>null</c> instead of throwing when the base SPV is missing.
    /// </summary>
    /// <param name="device">Device to create the pipelines on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <param name="wideVariant">Wide-head variant to attempt to load.</param>
    /// <returns>The created kernel, or <c>null</c>.</returns>
    public static VulkanFlashAttentionF32Kernel? TryCreate(
        VulkanDevice device, string spvDir, FlashAttentionWideVariant wideVariant)
    {
        try
        {
            return Create(device, spvDir, wideVariant);
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
        _wideDescriptorCache?.Reset();
    }

    /// <summary>
    /// Synchronous one-shot launch. Mirrors <see cref="AttentionF32Kernel.Launch"/>
    /// for parity tests; production callers should use <see cref="Record"/>
    /// inside a batched command buffer.
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
    /// Records the FA dispatch into <paramref name="cmdBuf"/> without
    /// submitting. The contract mirrors <see cref="AttentionF32Kernel.Record"/>
    /// — same buffer shapes, same parameters — with one extra
    /// <paramref name="softCap"/> argument that defaults to disabled.
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
        if (headDim > SupportedMaxHeadDim)
            throw new ArgumentException(
                $"headDim ({headDim}) exceeds this kernel's MAX_HEAD_DIM ({SupportedMaxHeadDim}; " +
                $"wide variant = {WideVariant}). Either build the matching " +
                $"attention_flash_f32_hd*.comp variant or route to {nameof(AttentionF32Kernel)}. " +
                $"Callers should gate on {nameof(SupportedMaxHeadDim)}, not the {nameof(MaxHeadDim)} constant.",
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

        // Issue #441: heads wider than the base shader's MAX_HEAD_DIM dispatch the wide SPV,
        // which has its own pipeline, descriptor pool/cache and (for br8) its own BR.
        bool wide = headDim > MaxHeadDim;
        ComputePipeline pipeline = wide ? _widePipeline! : _pipeline;
        DescriptorSetCache cache = wide ? _wideDescriptorCache! : _descriptorCache;
        int queryTileRows = wide ? _wideQueryTileRows : QueryTileRows;

        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, v.Handle, output.Handle };
        nint descriptorSet = cache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // Push-constant layout (matches the shader's PushConstants block):
        //   [0] seqQ, [1] seqKv, [2] numHeads, [3] numKvHeads,
        //   [4] headDim, [5] positionOffset, [6] slidingWindow, [7] useAlibi,
        //   [8] softCap (float, reinterpreted), [9] scaleOverride (float, reinterpreted),
        //   [10] maskMode (0=Causal,1=Bidirectional,2=Hybrid), [11] prefixLen (Hybrid only).
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

        if (_widePool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _widePool, 0);
        _widePipeline?.Dispose();
        _wideModule?.Dispose();

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
