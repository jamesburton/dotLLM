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
    private static readonly int TargetWorkgroups = EnvIntOrDefault(TargetWorkgroupsEnvVar, 256);
    private static readonly int MinKvPerSplit = EnvIntOrDefault(MinKvPerSplitEnvVar, 16);

    private static int EnvIntOrDefault(string envVar, int fallback)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        return int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    // 13 uints (the shared 12-uint attention block + numSplits) = 52 bytes.
    private const int SplitPushConstantBytes = 13 * sizeof(uint);
    // numHeads, headDim, numSplits = 12 bytes.
    private const int MergePushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _splitModule;
    private readonly ComputePipeline _splitPipeline;
    private readonly VulkanModule _mergeModule;
    private readonly ComputePipeline _mergePipeline;
    private readonly nint _splitPool;
    private readonly nint _mergePool;
    private readonly DescriptorSetCache _splitCache;   // (Q, K, V, partOut, partMS)
    private readonly DescriptorSetCache _mergeCache;   // (partOut, partMS, output)

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
        VulkanModule mergeModule, ComputePipeline mergePipeline, nint mergePool)
    {
        _device = device;
        _splitModule = splitModule;
        _splitPipeline = splitPipeline;
        _splitPool = splitPool;
        _mergeModule = mergeModule;
        _mergePipeline = mergePipeline;
        _mergePool = mergePool;
        _splitCache = new DescriptorSetCache(device, splitPool, splitPipeline, buffersPerSet: 5);
        _mergeCache = new DescriptorSetCache(device, mergePool, mergePipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>attention_f32_splitkv.spv</c> + <c>attention_f32_splitkv_merge.spv</c>
    /// and creates both pipelines. Throws if either SPV is missing — callers
    /// wanting graceful fallback should use <see cref="TryCreate"/>.
    /// </summary>
    public static VulkanSplitKvAttentionKernel Create(VulkanDevice device, string spvDir)
    {
        string splitPath = Path.Combine(spvDir, "attention_f32_splitkv.spv");
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
            mergePipeline = CreatePipeline(mergeModule, bindingCount: 3, MergePushConstantBytes);
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
        nint mergePool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new VulkanSplitKvAttentionKernel(
            device, splitModule, splitPipeline, splitPool, mergeModule!, mergePipeline, mergePool);
    }

    /// <summary>
    /// <c>TryCreate</c> companion — returns <c>null</c> when an SPV is missing or
    /// pipeline creation fails, so older builds without the split-KV SPVs fall
    /// back to the per-token / flash kernels.
    /// </summary>
    public static VulkanSplitKvAttentionKernel? TryCreate(VulkanDevice device, string spvDir)
    {
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

    /// <summary>Drops every cached descriptor set; call when the partial scratch was re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _splitCache.Reset();
        _mergeCache.Reset();
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
        Action<nint>? interPassStamp = null)
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
        long kvBytes  = (long)seqKv * numKvHeads * headDim * sizeof(float);
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < qBytes)   throw new ArgumentException("Output buffer too small.", nameof(output));

        int numSplits = ComputeSplits(seqKv, numHeads);

        EnsureScratch(numHeads, numSplits);

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
        nint splitSet = _splitCache.GetOrCreate(splitBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _splitPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _splitPipeline.Layout, 0, 1, splitSet, 0, 0);

        Span<uint> spc = stackalloc uint[13];
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
        fixed (uint* pcPtr = spc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _splitPipeline.Layout, VkShaderStageFlags.Compute, 0, SplitPushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(numHeads * numSplits), 1, 1);

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
        Span<nint> mergeBuffers = stackalloc nint[3] { _partOut!.Handle, _partMS!.Handle, output.Handle };
        nint mergeSet = _mergeCache.GetOrCreate(mergeBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Layout, 0, 1, mergeSet, 0, 0);

        Span<uint> mpc = stackalloc uint[3];
        mpc[0] = (uint)numHeads;
        mpc[1] = (uint)headDim;
        mpc[2] = (uint)numSplits;
        fixed (uint* pcPtr = mpc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _mergePipeline.Layout, VkShaderStageFlags.Compute, 0, MergePushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)numHeads, 1, 1);
    }

    private void EnsureScratch(int numHeads, int numSplits)
    {
        // Size partOut by MaxHeadDim, NOT the call's headDim. Within one forward
        // seqKv (hence numSplits) is constant and numHeads is constant for every
        // known architecture, but headDim can vary per layer (e.g. Gemma global vs
        // sliding). Sizing on the compile-time bound means per-layer headDim
        // variation never grows the buffer, so EnsureScratch never reallocates
        // (and resets the descriptor pool) MID command-buffer — which would free
        // descriptor sets already recorded for earlier layers of this forward.
        // The shaders index with the actual headDim, staying within the buffer.
        long needOut = (long)numHeads * numSplits * MaxHeadDim;
        long needMS  = (long)numHeads * numSplits * 2;
        if (_partOut is not null && needOut <= _partOutFloats && _partMS is not null && needMS <= _partMSFloats)
            return;

        _partOut?.Dispose();
        _partMS?.Dispose();
        _partOut = _device.AllocateDeviceLocal(needOut * sizeof(float));
        _partMS  = _device.AllocateDeviceLocal(needMS  * sizeof(float));
        _partOutFloats = needOut;
        _partMSFloats  = needMS;
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
        _partOut?.Dispose();
        _partMS?.Dispose();
        _mergePipeline.Dispose();
        _mergeModule.Dispose();
        _splitPipeline.Dispose();
        _splitModule.Dispose();
    }
}
