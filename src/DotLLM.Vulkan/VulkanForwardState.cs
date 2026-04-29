namespace DotLLM.Vulkan;

/// <summary>
/// Owns all per-forward-pass scratch buffers on the Vulkan device.
/// Sized for the maximum <c>seqLen</c> the caller has used so far; grows monotonically.
/// Mirrors <c>DotLLM.Cuda.CudaForwardState</c> but for FP32 storage on a Vulkan device.
/// </summary>
/// <remarks>
/// All buffers are host-visible host-coherent. The Vulkan scaffold does not
/// have a staging path yet; <see cref="VulkanDevice.Allocate"/> returns memory
/// that is mappable from both host and GPU, which is slower than device-local
/// memory for real kernels but keeps weight upload / result download trivial.
/// Optimising this is explicitly out of scope for the end-to-end wave.
/// </remarks>
internal sealed class VulkanForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;
    // MLA dims — zero unless the model carries an MLA layer.
    private readonly int _mlaQLoraRank;
    private readonly int _mlaKvLoraRank;
    private readonly int _mlaQkNopeHeadDim;
    private readonly int _mlaQkRopeHeadDim;
    private readonly int _mlaVHeadDim;
    private readonly int _mlaNumHeads;
    // MoE dims — zero unless the model carries a MoE layer.
    private readonly int _moeNumExperts;
    private readonly int _moeTopK;
    private readonly int _moeIntermediateSize;
    // MoE shared-expert dims — zero unless any MoE layer carries shared experts.
    private readonly int _moeSharedIntermediateSize;
    private readonly int _moeNumSharedExperts;
    private int _capacitySeqLen;

    // ── Transformer layer scratch (all FP32) ──────────────────────────
    //
    // Hidden / AddScratch form a 2-slot pair. The "current hidden" sits in
    // one slot; the "scratch that the residual-add writes into" sits in the
    // other. After each residual add we swap the slot indices — the buffer
    // that was just written becomes the new HiddenState, and the previous
    // HiddenState becomes the new AddScratch. This replaces two vkCmdCopyBuffer
    // dispatches per residual cycle (HiddenState→Residual snapshot copy and
    // AddScratch→HiddenState writeback copy) with pure pointer relabelling —
    // Residual is simply an alias for the current HiddenState, because the
    // snapshot read and the norm read come from the same buffer.
    private readonly VulkanDevice.Buffer[] _hiddenSlots = new VulkanDevice.Buffer[2];
    private int _hiddenIdx; // index into _hiddenSlots of the current HiddenState

    /// <summary>Currently active hidden-state buffer (rotates on each residual add).</summary>
    public VulkanDevice.Buffer HiddenState => _hiddenSlots[_hiddenIdx];

    /// <summary>
    /// Residual snapshot for the current pre-norm → post-add cycle. By design
    /// it aliases <see cref="HiddenState"/>: the snapshot is always the
    /// hidden state as-of the start of the cycle, and rotating the hidden
    /// slot only happens after the residual add finishes reading from here.
    /// </summary>
    public VulkanDevice.Buffer Residual => _hiddenSlots[_hiddenIdx];

    /// <summary>Alternate hidden slot — residual-add writes here, then <see cref="RotateHiddenSlot"/> promotes it.</summary>
    public VulkanDevice.Buffer AddScratch => _hiddenSlots[_hiddenIdx ^ 1];

    /// <summary>
    /// Flips <see cref="HiddenState"/> and <see cref="AddScratch"/>. Call after
    /// the residual add has been dispatched and the post-add barrier is in
    /// place — the old <see cref="AddScratch"/> now carries the new hidden
    /// state and the old <see cref="HiddenState"/> becomes free scratch.
    /// </summary>
    public void RotateHiddenSlot() => _hiddenIdx ^= 1;

    /// <summary>
    /// Resets the hidden-slot rotation so the next forward starts with the
    /// canonical slot 0 as <see cref="HiddenState"/>. Called at the start of
    /// every forward — without this, the forward pass would alternate the
    /// physical buffer it writes the initial embedding into, defeating any
    /// per-buffer descriptor-set caching that depends on the buffer handle
    /// staying stable across forwards.
    /// </summary>
    public void ResetHiddenSlot() => _hiddenIdx = 0;

    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
    public VulkanDevice.Buffer Q { get; private set; } = null!;
    public VulkanDevice.Buffer K { get; private set; } = null!;
    public VulkanDevice.Buffer V { get; private set; } = null!;
    public VulkanDevice.Buffer AttnOutput { get; private set; } = null!;
    public VulkanDevice.Buffer FfnGate { get; private set; } = null!;
    public VulkanDevice.Buffer FfnUp { get; private set; } = null!;
    public VulkanDevice.Buffer SiluOutput { get; private set; } = null!;

    // ── MoE scratch (Mixtral / Qwen-MoE) ─────────────────────────────
    // Allocated only when the model carries a MoE layer (moeNumExperts > 0
    // at construction). Sizes are seqLen-dependent so they grow with
    // EnsureCapacity. Names mirror the steps in MoeSwiGluMlp.Execute.
    public VulkanDevice.Buffer? MoeRouterLogits { get; private set; }   // [seqLen, numExperts]
    public VulkanDevice.Buffer? MoeTopkIndices { get; private set; }    // [seqLen, topK]   int32
    public VulkanDevice.Buffer? MoeTopkWeights { get; private set; }    // [seqLen, topK]   F32
    public VulkanDevice.Buffer? MoeExpandedInput { get; private set; }  // [seqLen * topK, hidden] (broadcast of NormOutput)
    public VulkanDevice.Buffer? MoeGateInter { get; private set; }      // [seqLen * topK, intermediate]
    public VulkanDevice.Buffer? MoeUpInter { get; private set; }        // [seqLen * topK, intermediate]
    public VulkanDevice.Buffer? MoeSiluInter { get; private set; }      // [seqLen * topK, intermediate]
    public VulkanDevice.Buffer? MoeDownRows { get; private set; }       // [seqLen * topK, hidden]
    public VulkanDevice.Buffer? MoeExpertCounts { get; private set; }   // [numExperts] uint32
    public VulkanDevice.Buffer? MoeExpertOffsets { get; private set; }  // [numExperts + 1] uint32
    public VulkanDevice.Buffer? MoeExpertCounters { get; private set; } // [numExperts] uint32
    public VulkanDevice.Buffer? MoePermutation { get; private set; }    // [seqLen * topK] uint32
    public VulkanDevice.Buffer? MoeGroupedHidden { get; private set; }  // [seqLen * topK, hidden]
    public VulkanDevice.Buffer? MoeGroupedGateInter { get; private set; } // [seqLen * topK, intermediate]
    public VulkanDevice.Buffer? MoeGroupedUpInter { get; private set; } // [seqLen * topK, intermediate]

    // ── MoE shared-expert scratch (DeepSeek-V2/V3) ────────────────────
    // Allocated only when the model carries an MoE layer with shared
    // experts (moeNumSharedExperts > 0 at construction). Each shared
    // expert is a dense SwiGLU MLP run over the full [seqLen, hidden]
    // input; the running sum is accumulated via a SumA / SumB ping-pong
    // pair so we never have to alias a buffer in the add kernel.
    public VulkanDevice.Buffer? MoeSharedInput { get; private set; }    // [seqLen, hidden] — post-rmsnorm hidden state, fed to every shared expert
    public VulkanDevice.Buffer? MoeSharedGate { get; private set; }     // [seqLen, sharedIntermediate]
    public VulkanDevice.Buffer? MoeSharedUp { get; private set; }       // [seqLen, sharedIntermediate]
    public VulkanDevice.Buffer? MoeSharedSilu { get; private set; }     // [seqLen, sharedIntermediate]
    public VulkanDevice.Buffer? MoeSharedDown { get; private set; }     // [seqLen, hidden] — per-expert down output
    public VulkanDevice.Buffer? MoeSharedSumA { get; private set; }     // [seqLen, hidden] — running shared sum, ping side A
    public VulkanDevice.Buffer? MoeSharedSumB { get; private set; }     // [seqLen, hidden] — running shared sum, ping side B
    public VulkanDevice.Buffer? MoeSharedGateLogits { get; private set; } // [seqLen] — pre-sigmoid gate logits (Qwen1.5-MoE)

    // ── MLA scratch (DeepSeek-V2/V3) ──────────────────────────────────
    // Allocated only when the model carries an MLA layer (mla* dims > 0
    // at construction). Names match MlaAttention.Execute one-for-one.
    public VulkanDevice.Buffer? MlaQLatent { get; private set; }       // [seqLen, qLoraRank]
    public VulkanDevice.Buffer? MlaQLatentNorm { get; private set; }   // [seqLen, qLoraRank]
    public VulkanDevice.Buffer? MlaQ { get; private set; }             // [seqLen, numHeads * (qkNope + qkRope)]
    public VulkanDevice.Buffer? MlaKvLatent { get; private set; }      // [seqLen, kvLoraRank]
    public VulkanDevice.Buffer? MlaKvLatentNorm { get; private set; }  // [seqLen, kvLoraRank]
    public VulkanDevice.Buffer? MlaKvBExpanded { get; private set; }   // [seqLen, numHeads * (qkNope + vHead)]
    public VulkanDevice.Buffer? MlaKNope { get; private set; }         // [seqLen, numHeads * qkNope]
    public VulkanDevice.Buffer? MlaV { get; private set; }             // [seqLen, numHeads * vHead]
    public VulkanDevice.Buffer? MlaKPe { get; private set; }           // [seqLen, qkRope]
    public VulkanDevice.Buffer? MlaAttnOutput { get; private set; }    // [seqLen, numHeads * vHead]

    // ── Logits (last token only) ──────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; private set; }

    // ── Host → device transfer scratch (tokens + positions) ──────────
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    // ── LoRA delta scratch (Phase 4b) ─────────────────────────────────
    // Allocated lazily on first LoRA-aware forward via EnsureLoraScratch
    // (otherwise null — forward pass with no adapter pays no extra VRAM).
    // Sized for [seqLen, max(rank)] / [seqLen, max(outputDim)] so we can
    // dispatch the two-stage LoRA delta as:
    //   LoraTmp[seqLen, rank] = matmul_f32(B_scaled, x)
    //   LoraDelta[seqLen, outputDim] = matmul_f32(A, LoraTmp)
    //   LoraDeltaSum[seqLen, outputDim] = AddKernel(y, LoraDelta)
    //   vkCmdCopyBuffer(LoraDeltaSum -> y)
    // The third buffer is needed because AddKernel writes to a separate
    // output (read-only A, write-only C); we copy the sum back into y.
    private int _loraCapacityRank;
    private int _loraCapacityOutputDim;
    public VulkanDevice.Buffer? LoraTmp { get; private set; }       // [seqLen, rank]
    public VulkanDevice.Buffer? LoraDelta { get; private set; }     // [seqLen, outputDim]
    public VulkanDevice.Buffer? LoraDeltaSum { get; private set; }  // [seqLen, outputDim]

    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    public VulkanForwardState(
        VulkanDevice device,
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int initialSeqLen,
        int mlaNumHeads = 0, int mlaQkNopeHeadDim = 0, int mlaQkRopeHeadDim = 0,
        int mlaVHeadDim = 0, int mlaQLoraRank = 0, int mlaKvLoraRank = 0,
        int moeNumExperts = 0, int moeTopK = 0, int moeIntermediateSize = 0,
        int moeSharedIntermediateSize = 0, int moeNumSharedExperts = 0)
    {
        _device = device;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _mlaNumHeads = mlaNumHeads;
        _mlaQkNopeHeadDim = mlaQkNopeHeadDim;
        _mlaQkRopeHeadDim = mlaQkRopeHeadDim;
        _mlaVHeadDim = mlaVHeadDim;
        _mlaQLoraRank = mlaQLoraRank;
        _mlaKvLoraRank = mlaKvLoraRank;
        _moeNumExperts = moeNumExperts;
        _moeTopK = moeTopK;
        _moeIntermediateSize = moeIntermediateSize;
        _moeSharedIntermediateSize = moeSharedIntermediateSize;
        _moeNumSharedExperts = moeNumSharedExperts;

        // LM-head logits are always one token (last). Positions buffer sized for some reasonable
        // default; grows with EnsureCapacity.
        Logits = device.Allocate((long)vocabSize * sizeof(float));
        PositionsBuffer = device.Allocate(Math.Max(1, initialSeqLen) * sizeof(int));

        AllocateForCapacity(Math.Max(1, initialSeqLen));
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough to host <paramref name="seqLen"/> tokens.
    /// Grows monotonically; never shrinks.
    /// </summary>
    /// <returns><c>true</c> when scratch was re-allocated (so every cached VkBuffer handle
    /// is now stale); <c>false</c> when existing capacity was already sufficient.</returns>
    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return false;

        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
        return true;
    }

    /// <summary>
    /// Ensures the LoRA scratch buffers are sized for at least
    /// <paramref name="rank"/> × <paramref name="outputDim"/> at the current
    /// <see cref="EnsureCapacity"/>-honoured seqLen capacity. Allocated
    /// lazily (so non-LoRA forwards never pay this VRAM cost) and grows
    /// monotonically — multiple adapters with different ranks share one
    /// scratch sized to the largest seen so far.
    /// </summary>
    /// <returns>
    /// <c>true</c> when scratch was re-allocated (so cached descriptor sets
    /// pointing at the old <see cref="LoraTmp"/> / <see cref="LoraDelta"/> /
    /// <see cref="LoraDeltaSum"/> handles are now stale); <c>false</c> when
    /// existing capacity was sufficient.
    /// </returns>
    public bool EnsureLoraScratch(int rank, int outputDim)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        bool needRealloc =
            LoraTmp is null || LoraDelta is null || LoraDeltaSum is null
            || rank > _loraCapacityRank
            || outputDim > _loraCapacityOutputDim;
        if (!needRealloc) return false;

        // Grow to the max ever requested (monotonic — small adapters
        // benefit from a previous larger allocation; large adapters force
        // a one-shot resize).
        int newRank = Math.Max(_loraCapacityRank, rank);
        int newOutputDim = Math.Max(_loraCapacityOutputDim, outputDim);

        LoraTmp?.Dispose();
        LoraDelta?.Dispose();
        LoraDeltaSum?.Dispose();

        long tmpBytes = (long)_capacitySeqLen * newRank * sizeof(float);
        long deltaBytes = (long)_capacitySeqLen * newOutputDim * sizeof(float);

        // Device-local: written by matmul_f32 / add kernels, never host-mapped.
        LoraTmp = _device.AllocateDeviceLocal(tmpBytes);
        LoraDelta = _device.AllocateDeviceLocal(deltaBytes);
        LoraDeltaSum = _device.AllocateDeviceLocal(deltaBytes);
        _loraCapacityRank = newRank;
        _loraCapacityOutputDim = newOutputDim;
        return true;
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long qBytes = (long)seqLen * _numHeads * _headDim * sizeof(float);
        long kvBytes = (long)seqLen * _numKvHeads * _headDim * sizeof(float);
        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        // Buffers never read or written from the host go device-local — the
        // driver can pick its tiled / swizzled native layout instead of the
        // host-coherent linear layout that Allocate() returns. On UMA (Strix
        // Halo iGPU) this lets the compute path use the GPU's preferred
        // memory access pattern; on dGPU it keeps the bytes off the
        // PCIe-host-coherent path.
        //
        // Bias-add receiver buffers (Q, K, V, NormOutput, FfnGate, FfnUp)
        // stay host-visible because AddBiasRows host-maps them when a bias
        // tensor is present. SmolLM-135M has no biases so the host-map
        // never fires, but Phi-3 / Qwen3 / DeepSeek-V2 do — moving them
        // requires a bias_add_f32 compute kernel (issue #7).
        _hiddenSlots[0] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenSlots[1] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenIdx = 0;
        NormOutput = _device.Allocate(hiddenBytes);

        Q = _device.Allocate(qBytes);
        K = _device.Allocate(kvBytes);
        V = _device.Allocate(kvBytes);
        AttnOutput = _device.AllocateDeviceLocal(qBytes);

        FfnGate = _device.Allocate(ffnBytes);
        FfnUp = _device.Allocate(ffnBytes);
        SiluOutput = _device.AllocateDeviceLocal(ffnBytes);

        long mlaBytes = AllocateMlaScratch(seqLen);
        long moeBytes = AllocateMoeScratch(seqLen);

        // Resize positions buffer — host writes positions per forward.
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;

        // hiddenBytes × 3: two hidden slots (HiddenState/AddScratch rotate) + NormOutput.
        AllocatedBytes = hiddenBytes * 3 + qBytes * 2 + kvBytes * 2 + ffnBytes * 3
                       + mlaBytes + moeBytes
                       + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int);
    }

    private long AllocateMoeScratch(int seqLen)
    {
        if (_moeNumExperts == 0) return 0;

        long routerBytes = (long)seqLen * _moeNumExperts * sizeof(float);
        long topkIdxBytes = (long)seqLen * _moeTopK * sizeof(int);
        long topkWtBytes = (long)seqLen * _moeTopK * sizeof(float);
        long expandedBytes = (long)seqLen * _moeTopK * _hiddenSize * sizeof(float);
        long interBytes = (long)seqLen * _moeTopK * _moeIntermediateSize * sizeof(float);
        long downBytes = expandedBytes;
        long expertCountsBytes = (long)_moeNumExperts * sizeof(uint);
        long expertOffsetsBytes = (long)(_moeNumExperts + 1) * sizeof(uint);
        long routedRowsBytes = (long)seqLen * _moeTopK * sizeof(uint);

        MoeRouterLogits = _device.AllocateDeviceLocal(routerBytes);
        MoeTopkIndices = _device.AllocateDeviceLocal(topkIdxBytes);
        MoeTopkWeights = _device.AllocateDeviceLocal(topkWtBytes);
        MoeExpandedInput = _device.AllocateDeviceLocal(expandedBytes);
        MoeGateInter = _device.AllocateDeviceLocal(interBytes);
        MoeUpInter = _device.AllocateDeviceLocal(interBytes);
        MoeSiluInter = _device.AllocateDeviceLocal(interBytes);
        MoeDownRows = _device.AllocateDeviceLocal(downBytes);
        MoeExpertCounts = _device.AllocateDeviceLocal(expertCountsBytes);
        MoeExpertOffsets = _device.AllocateDeviceLocal(expertOffsetsBytes);
        MoeExpertCounters = _device.AllocateDeviceLocal(expertCountsBytes);
        MoePermutation = _device.AllocateDeviceLocal(routedRowsBytes);
        MoeGroupedHidden = _device.AllocateDeviceLocal(expandedBytes);
        MoeGroupedGateInter = _device.AllocateDeviceLocal(interBytes);
        MoeGroupedUpInter = _device.AllocateDeviceLocal(interBytes);

        long total = routerBytes + topkIdxBytes + topkWtBytes + expandedBytes
                   + interBytes * 3 + downBytes
                   + expertCountsBytes * 2 + expertOffsetsBytes + routedRowsBytes
                   + expandedBytes + interBytes * 2;

        if (_moeNumSharedExperts > 0)
        {
            long sharedInterBytes = (long)seqLen * _moeSharedIntermediateSize * sizeof(float);
            long sharedHiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);

            MoeSharedInput = _device.AllocateDeviceLocal(sharedHiddenBytes);
            MoeSharedGate = _device.AllocateDeviceLocal(sharedInterBytes);
            MoeSharedUp = _device.AllocateDeviceLocal(sharedInterBytes);
            MoeSharedSilu = _device.AllocateDeviceLocal(sharedInterBytes);
            MoeSharedDown = _device.AllocateDeviceLocal(sharedHiddenBytes);
            MoeSharedSumA = _device.AllocateDeviceLocal(sharedHiddenBytes);
            MoeSharedSumB = _device.AllocateDeviceLocal(sharedHiddenBytes);

            // Per-token gate logits — only used by the Qwen1.5-MoE sigmoid
            // gate path. Allocated unconditionally when shared experts exist
            // so the same state object covers both DeepSeek (unused) and
            // Qwen1.5-MoE (used) layers; the size is tiny (seqLen × 4B).
            long gateLogitBytes = (long)seqLen * sizeof(float);
            MoeSharedGateLogits = _device.AllocateDeviceLocal(gateLogitBytes);

            total += sharedInterBytes * 3 + sharedHiddenBytes * 4 + gateLogitBytes;
        }

        return total;
    }

    private long AllocateMlaScratch(int seqLen)
    {
        if (_mlaNumHeads == 0) return 0;

        int qkHeadDim = _mlaQkNopeHeadDim + _mlaQkRopeHeadDim;
        long qLatentBytes = _mlaQLoraRank > 0 ? (long)seqLen * _mlaQLoraRank * sizeof(float) : 0;
        long qBytes = (long)seqLen * _mlaNumHeads * qkHeadDim * sizeof(float);
        long kvLatentBytes = (long)seqLen * _mlaKvLoraRank * sizeof(float);
        long kvBExpandedBytes = (long)seqLen * _mlaNumHeads * (_mlaQkNopeHeadDim + _mlaVHeadDim) * sizeof(float);
        long kNopeBytes = (long)seqLen * _mlaNumHeads * _mlaQkNopeHeadDim * sizeof(float);
        long vBytes = (long)seqLen * _mlaNumHeads * _mlaVHeadDim * sizeof(float);
        long kPeBytes = (long)seqLen * _mlaQkRopeHeadDim * sizeof(float);
        long attnOutBytes = (long)seqLen * _mlaNumHeads * _mlaVHeadDim * sizeof(float);

        // Latent / latent-norm only when the Q path is LoRA-factored.
        if (_mlaQLoraRank > 0)
        {
            MlaQLatent = _device.AllocateDeviceLocal(qLatentBytes);
            MlaQLatentNorm = _device.AllocateDeviceLocal(qLatentBytes);
        }
        MlaQ = _device.AllocateDeviceLocal(qBytes);
        MlaKvLatent = _device.AllocateDeviceLocal(kvLatentBytes);
        MlaKvLatentNorm = _device.AllocateDeviceLocal(kvLatentBytes);
        MlaKvBExpanded = _device.AllocateDeviceLocal(kvBExpandedBytes);
        MlaKNope = _device.AllocateDeviceLocal(kNopeBytes);
        MlaV = _device.AllocateDeviceLocal(vBytes);
        MlaKPe = _device.AllocateDeviceLocal(kPeBytes);
        MlaAttnOutput = _device.AllocateDeviceLocal(attnOutBytes);

        return qLatentBytes * 2 + qBytes + kvLatentBytes * 2 + kvBExpandedBytes
             + kNopeBytes + vBytes + kPeBytes + attnOutBytes;
    }

    private void ReleaseLayerScratch()
    {
        _hiddenSlots[0]?.Dispose();
        _hiddenSlots[1]?.Dispose();
        _hiddenSlots[0] = null!;
        _hiddenSlots[1] = null!;
        NormOutput?.Dispose();
        Q?.Dispose();
        K?.Dispose();
        V?.Dispose();
        AttnOutput?.Dispose();
        FfnGate?.Dispose();
        FfnUp?.Dispose();
        SiluOutput?.Dispose();

        MlaQLatent?.Dispose(); MlaQLatent = null;
        MlaQLatentNorm?.Dispose(); MlaQLatentNorm = null;
        MlaQ?.Dispose(); MlaQ = null;
        MlaKvLatent?.Dispose(); MlaKvLatent = null;
        MlaKvLatentNorm?.Dispose(); MlaKvLatentNorm = null;
        MlaKvBExpanded?.Dispose(); MlaKvBExpanded = null;
        MlaKNope?.Dispose(); MlaKNope = null;
        MlaV?.Dispose(); MlaV = null;
        MlaKPe?.Dispose(); MlaKPe = null;
        MlaAttnOutput?.Dispose(); MlaAttnOutput = null;

        MoeRouterLogits?.Dispose(); MoeRouterLogits = null;
        MoeTopkIndices?.Dispose(); MoeTopkIndices = null;
        MoeTopkWeights?.Dispose(); MoeTopkWeights = null;
        MoeExpandedInput?.Dispose(); MoeExpandedInput = null;
        MoeGateInter?.Dispose(); MoeGateInter = null;
        MoeUpInter?.Dispose(); MoeUpInter = null;
        MoeSiluInter?.Dispose(); MoeSiluInter = null;
        MoeDownRows?.Dispose(); MoeDownRows = null;
        MoeExpertCounts?.Dispose(); MoeExpertCounts = null;
        MoeExpertOffsets?.Dispose(); MoeExpertOffsets = null;
        MoeExpertCounters?.Dispose(); MoeExpertCounters = null;
        MoePermutation?.Dispose(); MoePermutation = null;
        MoeGroupedHidden?.Dispose(); MoeGroupedHidden = null;
        MoeGroupedGateInter?.Dispose(); MoeGroupedGateInter = null;
        MoeGroupedUpInter?.Dispose(); MoeGroupedUpInter = null;

        MoeSharedInput?.Dispose(); MoeSharedInput = null;
        MoeSharedGate?.Dispose(); MoeSharedGate = null;
        MoeSharedUp?.Dispose(); MoeSharedUp = null;
        MoeSharedSilu?.Dispose(); MoeSharedSilu = null;
        MoeSharedDown?.Dispose(); MoeSharedDown = null;
        MoeSharedSumA?.Dispose(); MoeSharedSumA = null;
        MoeSharedSumB?.Dispose(); MoeSharedSumB = null;
        MoeSharedGateLogits?.Dispose(); MoeSharedGateLogits = null;

        // LoRA scratch (Phase 4b) — sized in seqLen × rank / outputDim, so
        // it grows alongside the main scratch on EnsureCapacity. Reset the
        // capacity counters so the next EnsureLoraScratch call re-allocates
        // at the new seqLen.
        LoraTmp?.Dispose(); LoraTmp = null;
        LoraDelta?.Dispose(); LoraDelta = null;
        LoraDeltaSum?.Dispose(); LoraDeltaSum = null;
        _loraCapacityRank = 0;
        _loraCapacityOutputDim = 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        ReleaseLayerScratch();
        Logits?.Dispose();
        PositionsBuffer?.Dispose();
    }
}
