using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Split-KV (Flash-Decoding) FP32 attention for the <b>decode</b> path
/// (<c>seqQ == 1</c>). Splits the KV range across many workgroups to raise
/// occupancy above the per-token kernel's ~<c>numHeads</c> floor, and coalesces
/// the Q·K read inside each split — together the long-context decode lever that
/// the coalesced read alone could not deliver (it serialised KV positions at low
/// occupancy and regressed).
/// </summary>
/// <remarks>
/// <para>
/// Two-pass: <c>attention_f32_splitkv.comp</c> computes a per-(head, split)
/// partial online-softmax state (running max <c>m</c>, running sum <c>l</c>,
/// un-normalised accumulator <c>acc[head_dim]</c>) into kernel-owned scratch;
/// <c>attention_f32_splitkv_merge.comp</c> combines the <c>S</c> partials per
/// head into the final normalised row. A compute→compute barrier separates the
/// passes, recorded inside a single <see cref="Record"/> call.
/// </para>
/// <para>
/// Parity target: the scalar CPU reference
/// <c>DotLLM.Cpu.Kernels.Attention.ExecuteScalar</c> — same masking (causal /
/// bidirectional / hybrid), GQA broadcast, sliding window, ALiBi, soft-cap and
/// scale-override semantics as <see cref="AttentionF32Kernel"/>. Numerical drift
/// is reduction-order only (abs 1e-4 / rel 1e-3).
/// </para>
/// <para>
/// <b>Decode-only.</b> The shaders assume <c>seqQ == 1</c> (the single query sits
/// at absolute position <c>positionOffset</c>). Callers must route prefill
/// (<c>seqQ &gt; 1</c>) to the flash / per-token kernels.
/// </para>
/// <para>
/// <b>Fused single-dispatch split+merge — assessed (#370) and deliberately NOT adopted.</b>
/// The ceiling is too small for the hazard. At the tuned 256/256 heuristic (Llama-3.2-3B,
/// 24 q-heads / 8 kv-heads / head_dim 128, ctx 4096 ⇒ S=10) the merge pass moves ≈137 KB of
/// partials per layer against the split pass's ≈33.5 MB unavoidable K/V read (~0.4% of attention
/// bytes), and the extra dispatch + barrier per attention layer costs ≲0.6% of a decode step —
/// consistent with the #347 sweep, where merge overhead only became visible in the mis-tuned
/// many-small-splits regime the heuristic avoids. The only true single-dispatch shape (an
/// atomic-counter "last workgroup through merges" à la decoupled-lookback) requires
/// forward-progress guarantees Vulkan does not portably provide — a workgroup spin-waiting on
/// peers the driver has not scheduled can deadlock, and this codebase already documents gfx1151
/// driver fragility (IQ2_XXS MMQ). Prior art agrees: CUDA Flash-Decoding and llama.cpp's Vulkan
/// flash-attention both ship a separate reduce dispatch. Revisit only if a portable device-scope
/// forward-progress primitive lands in Vulkan.
/// </para>
/// </remarks>
public sealed class VulkanSplitKvAttentionKernel : IDisposable
{
    /// <summary>Compile-time head_dim bound baked into the shaders (matches <see cref="AttentionF32Kernel.MaxHeadDim"/>).</summary>
    public const int MaxHeadDim = 512;

    // Split heuristic. A split is only taken when it produces >= 2 splits; at 1
    // split the caller falls back to the single-pass kernel (so short context is
    // bit-identical to the legacy path — the regression guard).
    //
    // S = clamp(TargetWorkgroups / numHeads, 1, ceil(seqKv / MinKvPerSplit)).
    //   - TargetWorkgroups: occupancy target (split-pass workgroups = numHeads*S).
    //   - MinKvPerSplit: floor on KV rows per split so each split has enough work
    //     to amortise its launch + its share of the merge.
    //
    // TargetWorkgroups defaults to 256 — SWEPT on Llama-3.2-3B / gfx1151 (#347,
    // decode_min_ms, ctx 512-4096): more total workgroups regress (~+12% at
    // ctx4096: smaller splits, merge overhead dominates); fewer regress too
    // (~+6-28%: lost occupancy). MinKvPerSplit defaults to 16 (issue #143 —
    // re-swept including a small-head model): it exists only to stop degenerate
    // splits; the occupancy target above is what should normally bound S. The
    // original 256 floor capped S=3 on SmolLM-135M (9 heads, ctx 512-640) —
    // 27 split workgroups on a 40-CU part — making decode attention 4x slower
    // than S=28 (2.21 → 0.53 ms/token GPU time). Lowering the floor to 16 also
    // measured FASTER on the original sweep models at ctx 512 (Llama-3.2-3B
    // IQ4_XS 68.5 → 74.2 tok/s; Llama-3.1-8B Q4_K_M 27.5 → 28.4 tok/s), and at
    // long ctx the occupancy bound binds first so S is unchanged there.
    // Overridable via DOTLLM_VULKAN_SPLIT_TARGET_WG / DOTLLM_VULKAN_SPLIT_MIN_KV
    // so other archs/deployments can be re-tuned without recompiling. Read once
    // at type initialization — zero decode-hot-path cost. An invalid/
    // non-positive value falls back to the default.
    internal const string TargetWorkgroupsEnvVar = "DOTLLM_VULKAN_SPLIT_TARGET_WG";
    internal const string MinKvPerSplitEnvVar = "DOTLLM_VULKAN_SPLIT_MIN_KV";
    internal static readonly int TargetWorkgroups = EnvIntOrDefault(TargetWorkgroupsEnvVar, 256);
    internal static readonly int MinKvPerSplit = EnvIntOrDefault(MinKvPerSplitEnvVar, 16);

    private static int EnvIntOrDefault(string envVar, int fallback)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        return int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    // 14 uints (the shared 12-uint attention block + numSplits + kvStart) = 56 bytes.
    private const int SplitPushConstantBytes = 14 * sizeof(uint);
    // numHeads, headDim, numSplits, hasSinks = 16 bytes.
    private const int MergePushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _splitModule;
    private readonly ComputePipeline _splitPipeline;
    private readonly VulkanModule _mergeModule;
    private readonly ComputePipeline _mergePipeline;
    private readonly nint _splitPool;
    private readonly nint _mergePool;
    private readonly DescriptorSetCache _splitCache;   // (Q, K, V, partOut, partMS)
    private readonly DescriptorSetCache _mergeCache;   // (partOut, partMS, output)

    // Optional GQA-grouped split pass (issue #1102): one workgroup per (KV head, split) reads each K / V element once for all of
    // that KV head's query heads. Same bindings, push constants and partial layout as the per-query-head pass.
    private readonly VulkanModule? _gqaModule;
    private readonly ComputePipeline? _gqaPipeline;
    private readonly nint _gqaPool;
    private readonly DescriptorSetCache? _gqaCache;

    /// <summary>Largest query-head group (numHeads / numKvHeads) the GQA-grouped pass serves.</summary>
    public const int GqaMaxGroup = 8;

    /// <summary>Largest head_dim the GQA-grouped pass serves (one thread per column of a 256-thread workgroup).</summary>
    public const int GqaMaxHeadDim = 256;

    /// <summary>
    /// Override for the GQA-grouped pass: <c>DOTLLM_VULKAN_SPLITKV_GQA=1</c> takes it on every host, <c>0</c> on none; unset leaves
    /// it to the host (<see cref="EnableGqaByDefault"/>).
    /// </summary>
    internal const string GqaEnvVar = "DOTLLM_VULKAN_SPLITKV_GQA";

    private static readonly bool? GqaFromEnvironment = Environment.GetEnvironmentVariable(GqaEnvVar) switch
    {
        "1" => true,
        "0" => false,
        _ => null,
    };

    /// <summary>
    /// Whether <see cref="Record"/> may take the GQA-grouped pass. Off unless the host opts in (<see cref="EnableGqaByDefault"/>) or
    /// <c>DOTLLM_VULKAN_SPLITKV_GQA=1</c>; tests set it directly to pin either pass.
    /// </summary>
    internal bool PreferGqa { get; set; } = GqaFromEnvironment ?? false;

    /// <summary>A host that measured the grouped pass opts in here; <c>DOTLLM_VULKAN_SPLITKV_GQA</c> still wins when set.</summary>
    internal void EnableGqaByDefault() => PreferGqa = GqaFromEnvironment ?? true;

    /// <summary>Dispatches that took the GQA-grouped pass (test hook).</summary>
    internal long GqaDispatches { get; private set; }

    // Kernel-owned scratch for the partials. Shared across layers within one
    // forward (each layer overwrites them; the leading barrier in Record orders
    // a layer's split write after the previous layer's merge read).
    private VulkanDevice.Buffer? _partOut;   // [numHeads * S * headDim] floats
    private VulkanDevice.Buffer? _partMS;    // [numHeads * S * 2] floats
    private long _partOutFloats;
    private long _partMSFloats;

    private bool _disposed;

    private VulkanSplitKvAttentionKernel(
        VulkanDevice device,
        VulkanModule splitModule, ComputePipeline splitPipeline, nint splitPool,
        VulkanModule mergeModule, ComputePipeline mergePipeline, nint mergePool, VulkanKvStorage kvStorage,
        VulkanModule? gqaModule, ComputePipeline? gqaPipeline, nint gqaPool)
    {
        _device = device;
        KvStorage = kvStorage;
        _gqaModule = gqaModule;
        _gqaPipeline = gqaPipeline;
        _gqaPool = gqaPool;
        if (gqaPipeline is not null)
            _gqaCache = new DescriptorSetCache(device, gqaPool, gqaPipeline, buffersPerSet: 5);
        _splitModule = splitModule;
        _splitPipeline = splitPipeline;
        _splitPool = splitPool;
        _mergeModule = mergeModule;
        _mergePipeline = mergePipeline;
        _mergePool = mergePool;
        _splitCache = new DescriptorSetCache(device, splitPool, splitPipeline, buffersPerSet: 5);
        _mergeCache = new DescriptorSetCache(device, mergePool, mergePipeline, buffersPerSet: 4);
    }

    /// <summary>Element format of the K / V buffers this instance reads (issue #1102); fixed at creation.</summary>
    public VulkanKvStorage KvStorage { get; }

    /// <summary>
    /// Loads <c>attention_f32_splitkv.spv</c> + <c>attention_f32_splitkv_merge.spv</c>
    /// and creates both pipelines. Throws if either SPV is missing — callers
    /// wanting graceful fallback should use <see cref="TryCreate(VulkanDevice, string)"/>.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <returns>An instance reading F32 K / V.</returns>
    public static VulkanSplitKvAttentionKernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, VulkanKvStorage.F32);

    /// <summary>
    /// Loads the split pass compiled for <paramref name="kvStorage"/> (<c>attention_f32_splitkv[_kvf16].spv</c>, issue #1102) and the
    /// shared merge pass. The F16 split pass reads the cache's packed halves in place: no F32 copy of the history exists. Throws if
    /// either SPV is missing.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="kvStorage">Element format of the K / V buffers <see cref="Record"/> will be given.</param>
    /// <returns>The kernel.</returns>
    public static VulkanSplitKvAttentionKernel Create(VulkanDevice device, string spvDir, VulkanKvStorage kvStorage)
    {
        string splitPath = Path.Combine(spvDir, kvStorage.ShaderName("attention_f32_splitkv") + ".spv");
        string mergePath = Path.Combine(spvDir, "attention_f32_splitkv_merge.spv");
        if (!File.Exists(splitPath))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {splitPath}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
        if (!File.Exists(mergePath))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {mergePath}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");

        VulkanModule splitModule = VulkanModule.LoadFromFile(device, splitPath);
        ComputePipeline? splitPipeline = null;
        VulkanModule? mergeModule = null;
        ComputePipeline? mergePipeline = null;
        try
        {
            splitPipeline = CreatePipeline(splitModule, bindingCount: 5, SplitPushConstantBytes);
            mergeModule = VulkanModule.LoadFromFile(device, mergePath);
            mergePipeline = CreatePipeline(mergeModule, bindingCount: 4, MergePushConstantBytes);
        }
        catch
        {
            mergePipeline?.Dispose();
            mergeModule?.Dispose();
            splitPipeline?.Dispose();
            splitModule.Dispose();
            throw;
        }

        nint splitPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        nint mergePool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);

        // The GQA-grouped pass is optional: a missing or rejected SPV leaves the per-query-head pass in charge.
        VulkanModule? gqaModule = null;
        ComputePipeline? gqaPipeline = null;
        nint gqaPool = 0;
        string gqaPath = Path.Combine(spvDir, kvStorage.ShaderName("attention_f32_splitkv_gqa") + ".spv");
        if (File.Exists(gqaPath))
        {
            try
            {
                gqaModule = VulkanModule.LoadFromFile(device, gqaPath);
                gqaPipeline = CreatePipeline(gqaModule, bindingCount: 5, SplitPushConstantBytes);
                gqaPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
            }
            catch (VulkanException)
            {
                gqaPipeline?.Dispose();
                gqaModule?.Dispose();
                gqaModule = null;
                gqaPipeline = null;
            }
        }

        return new VulkanSplitKvAttentionKernel(
            device, splitModule, splitPipeline, splitPool, mergeModule!, mergePipeline, mergePool, kvStorage,
            gqaModule, gqaPipeline, gqaPool);
    }

    /// <summary>
    /// <c>TryCreate</c> companion — returns <c>null</c> when an SPV is missing or
    /// pipeline creation fails, so older builds without the split-KV SPVs fall
    /// back to the per-token / flash kernels.
    /// </summary>
    public static VulkanSplitKvAttentionKernel? TryCreate(VulkanDevice device, string spvDir)
        => TryCreate(device, spvDir, VulkanKvStorage.F32);

    /// <summary><see cref="Create(VulkanDevice, string, VulkanKvStorage)"/>, or <see langword="null"/> when an SPV is missing or a pipeline fails.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="kvStorage">Element format of the K / V buffers.</param>
    /// <returns>The kernel, or <see langword="null"/>.</returns>
    public static VulkanSplitKvAttentionKernel? TryCreate(VulkanDevice device, string spvDir, VulkanKvStorage kvStorage)
    {
        try
        {
            return Create(device, spvDir, kvStorage);
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

    private static ComputePipeline CreatePipeline(VulkanModule module, int bindingCount, int pushConstantBytes)
    {
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[bindingCount];
        for (int i = 0; i < bindingCount; i++)
            bindings[i] = new VkDescriptorBinding((uint)i);
        return module.CreateComputePipeline(
            entryPoint: "main",
            bindings: bindings,
            pushConstantBytes: (uint)pushConstantBytes);
    }

    /// <summary>
    /// Number of KV splits for the given shape, or <c>1</c> when splitting is not
    /// worthwhile (short context / many heads). The caller uses <c>&gt;= 2</c> as
    /// the gate to route decode through this kernel rather than the per-token one.
    /// </summary>
    public static int ComputeSplits(int seqKv, int numHeads)
    {
        if (seqKv <= 0 || numHeads <= 0) return 1;
        int byKv = (seqKv + MinKvPerSplit - 1) / MinKvPerSplit;     // ceil(seqKv / MinKvPerSplit)
        int byOccupancy = Math.Max(1, TargetWorkgroups / numHeads);
        int s = Math.Min(byOccupancy, byKv);
        return Math.Max(1, s);
    }

    /// <summary>True when <see cref="ComputeSplits"/> yields &gt;= 2 splits for this shape.</summary>
    public static bool WouldSplit(int seqKv, int numHeads) => ComputeSplits(seqKv, numHeads) >= 2;

    /// <summary>
    /// The <b>seqKv-independent</b> upper bound on <see cref="ComputeSplits"/> for a given head
    /// count — the scratch sizing bound (#465).
    /// </summary>
    /// <remarks>
    /// <see cref="ComputeSplits"/> is <c>min(byOccupancy, byKv)</c>, and only <c>byKv</c> depends
    /// on the cache length. So <c>byOccupancy</c> alone bounds every split count this kernel can
    /// request for <paramref name="numHeads"/>, at any context length. Sizing the partial scratch
    /// on it means a varying <c>seqKv</c> never grows the buffer — which is what makes
    /// <see cref="EnsureScratch"/> safe to call while a command buffer is open.
    /// </remarks>
    /// <summary>True when <see cref="Record"/> takes the GQA-grouped pass for this shape.</summary>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <returns>Whether the grouped pass serves the shape.</returns>
    public bool UsesGqa(int numHeads, int numKvHeads, int headDim)
    {
        if (_gqaPipeline is null || !PreferGqa || numKvHeads <= 0 || numHeads % numKvHeads != 0) return false;
        int group = numHeads / numKvHeads;
        return group >= 2 && group <= GqaMaxGroup && headDim <= GqaMaxHeadDim;
    }

    /// <summary>
    /// Splits per KV head for the GQA-grouped pass: the same occupancy target and per-split floor as
    /// <see cref="ComputeSplits"/>, counted in KV-head workgroups.
    /// </summary>
    /// <param name="seqKv">KV positions the splits cover.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <returns>The split count (at least 1).</returns>
    public static int ComputeGqaSplits(int seqKv, int numKvHeads) => ComputeSplits(seqKv, numKvHeads);

    internal static int MaxSplitsFor(int numHeads)
        => numHeads <= 0 ? 1 : Math.Max(1, TargetWorkgroups / numHeads);

    /// <summary>
    /// Count of partial-scratch (re)allocations. Test-only observability for #465: the invariant
    /// is that this reaches 1 for a given head count and never increments again, however
    /// <c>seqKv</c> varies.
    /// </summary>
    internal int ScratchAllocationCount { get; private set; }

    /// <summary>Drops every cached descriptor set; call when the partial scratch was re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _splitCache.Reset();
        _mergeCache.Reset();
        _gqaCache?.Reset();
    }

    /// <summary>
    /// Synchronous one-shot launch (parity tests). Production callers use
    /// <see cref="Record"/> inside the batched forward command buffer.
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
    /// Records the two-pass split-KV decode attention into <paramref name="cmdBuf"/>.
    /// Contract mirrors <see cref="AttentionF32Kernel.Record"/> exactly (same
    /// buffer shapes / parameters); <paramref name="seqQ"/> must be 1.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
        Action<nint>? interPassStamp = null,
        VulkanDevice.Buffer? sinks = null, int kvStart = 0)
    {
        if (seqQ != 1)
            throw new ArgumentException("VulkanSplitKvAttentionKernel is decode-only (seqQ must be 1).", nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (headDim > MaxHeadDim)
            throw new ArgumentException(
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}).", nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));
        if (softCap < 0.0f) throw new ArgumentOutOfRangeException(nameof(softCap),
            "softCap must be non-negative (use 0 to disable).");
        if (scaleOverride < 0.0f) throw new ArgumentOutOfRangeException(nameof(scaleOverride),
            "scaleOverride must be non-negative (use 0 for the default 1/sqrt(headDim)).");
        if (prefixLen < 0) throw new ArgumentOutOfRangeException(nameof(prefixLen));

        long qBytes   = (long)numHeads   * headDim * sizeof(float);   // seqQ == 1
        long kvBytes  = (long)seqKv * numKvHeads * headDim * KvStorage.ElementBytes();
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < qBytes)   throw new ArgumentException("Output buffer too small.", nameof(output));

        if (kvStart < 0 || kvStart >= seqKv) throw new ArgumentOutOfRangeException(nameof(kvStart));
        if (sinks is not null && sinks.Size < (long)numHeads * sizeof(float)) throw new ArgumentException("Sinks buffer too small.", nameof(sinks));
        bool gqa = UsesGqa(numHeads, numKvHeads, headDim);
        int numSplits = gqa ? ComputeGqaSplits(seqKv - kvStart, numKvHeads) : ComputeSplits(seqKv - kvStart, numHeads);

        EnsureScratch(numHeads);

        // Order this layer's split write after any prior compute reads of the
        // shared partial scratch (the previous layer's merge pass) and after any
        // prior writes — self-contained WAR/RAW protection independent of the
        // model's inter-kernel barriers. When the hazard tracker (issue #144)
        // is armed, the split dispatch's own guard (via GetOrCreate below)
        // detects the same WAR/RAW on _partOut/_partMS and emits the minimal
        // batched barrier — the blanket one here would just double up.
        if (_device.ActiveHazards is null)
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Pass 1: split ────────────────────────────────────────────────
        Span<nint> splitBuffers = stackalloc nint[5]
            { q.Handle, k.Handle, v.Handle, _partOut!.Handle, _partMS!.Handle };
        ComputePipeline splitPipeline = gqa ? _gqaPipeline! : _splitPipeline;
        nint splitSet = (gqa ? _gqaCache! : _splitCache).GetOrCreate(splitBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, splitPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, splitPipeline.Layout, 0, 1, splitSet, 0, 0);

        Span<uint> spc = stackalloc uint[14];
        spc[0]  = 1u;                  // seqQ
        spc[1]  = (uint)seqKv;
        spc[2]  = (uint)numHeads;
        spc[3]  = (uint)numKvHeads;
        spc[4]  = (uint)headDim;
        spc[5]  = (uint)positionOffset;
        spc[6]  = (uint)slidingWindow;
        spc[7]  = useAlibi ? 1u : 0u;
        spc[8]  = BitConverter.SingleToUInt32Bits(softCap);
        spc[9]  = BitConverter.SingleToUInt32Bits(scaleOverride);
        spc[10] = (uint)maskMode;
        spc[11] = (uint)prefixLen;
        spc[12] = (uint)numSplits;
        spc[13] = (uint)kvStart;
        fixed (uint* pcPtr = spc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, splitPipeline.Layout, VkShaderStageFlags.Compute, 0, SplitPushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((gqa ? numKvHeads : numHeads) * numSplits), 1, 1);
        if (gqa) GqaDispatches++;

        // Split writes partOut/partMS → merge reads them. Under the hazard
        // tracker the merge's guard emits this RAW barrier itself.
        if (_device.ActiveHazards is null)
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Optional profiler hook (issue #145): a timestamp written here — after
        // the split-pass barrier, before the merge dispatch — separates the
        // split-pass GPU time from the merge-pass GPU time in the decode
        // profiler's barrier-serialised timeline. Null (production) costs nothing.
        interPassStamp?.Invoke(cmdBuf);

        // ── Pass 2: merge ────────────────────────────────────────────────
        Span<nint> mergeBuffers = stackalloc nint[4] { _partOut!.Handle, _partMS!.Handle, output.Handle, (sinks ?? output).Handle };
        nint mergeSet = _mergeCache.GetOrCreate(mergeBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Layout, 0, 1, mergeSet, 0, 0);

        Span<uint> mpc = stackalloc uint[4];
        mpc[0] = (uint)numHeads;
        mpc[1] = (uint)headDim;
        mpc[2] = (uint)numSplits;
        mpc[3] = sinks is null ? 0u : 1u;
        fixed (uint* pcPtr = mpc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _mergePipeline.Layout, VkShaderStageFlags.Compute, 0, MergePushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)numHeads, 1, 1);
    }

    private void EnsureScratch(int numHeads)
    {
        // Size on BOUNDS, never on this call's values — the buffer must not grow while a
        // command buffer is open, because InvalidateDescriptorCache() below resets the
        // descriptor pool and would free sets already recorded for earlier layers.
        //
        //   headDim  -> MaxHeadDim       (headDim varies per layer, e.g. Gemma global vs sliding)
        //   numSplits-> MaxSplitsFor(n)  (#465: numSplits varies per SEQUENCE under ForwardBatch)
        //
        // The previous version sized on the call's numSplits and justified it with "within one
        // forward seqKv (hence numSplits) is constant". That holds for single-sequence Forward
        // and is FALSE for ForwardBatch, which reads seqKv from each sequence's own cache inside
        // the per-sequence loop (VulkanTransformerModel.cs) — so a batch with ascending KV
        // lengths grew the scratch mid-recording. MaxSplitsFor() is seqKv-independent, so the
        // invariant now holds by construction rather than by assumption.
        //
        // The shaders index with the actual headDim/numSplits, staying within the buffer.
        //   GQA pass (#1102) -> up to TargetWorkgroups splits per query head (numKvHeads >= 1) at head_dim <= GqaMaxHeadDim,
        //                       a bound independent of the layer's KV-head count.
        int maxSplits = MaxSplitsFor(numHeads);
        long needOut = (long)numHeads * maxSplits * MaxHeadDim;
        long needMS  = (long)numHeads * maxSplits * 2;
        if (_gqaPipeline is not null)
        {
            needOut = Math.Max(needOut, (long)numHeads * TargetWorkgroups * GqaMaxHeadDim);
            needMS = Math.Max(needMS, (long)numHeads * TargetWorkgroups * 2);
        }
        if (_partOut is not null && needOut <= _partOutFloats && _partMS is not null && needMS <= _partMSFloats)
            return;

        _partOut?.Dispose();
        _partMS?.Dispose();
        _partOut = _device.AllocateDeviceLocal(needOut * sizeof(float));
        _partMS  = _device.AllocateDeviceLocal(needMS  * sizeof(float));
        _partOutFloats = needOut;
        _partMSFloats  = needMS;
        ScratchAllocationCount++;
        // Cached descriptor sets reference the old scratch handles — drop them.
        InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_splitPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _splitPool, 0);
        if (_mergePool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _mergePool, 0);
        if (_gqaPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _gqaPool, 0);
        _gqaPipeline?.Dispose();
        _gqaModule?.Dispose();
        _partOut?.Dispose();
        _partMS?.Dispose();
        _mergePipeline.Dispose();
        _mergeModule.Dispose();
        _splitPipeline.Dispose();
        _splitModule.Dispose();
    }
}
