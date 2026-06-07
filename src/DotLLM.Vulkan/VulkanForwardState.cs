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

    // MoE dims — zero unless the model carries a MoE layer.
    private readonly int _moeNumExperts;
    private readonly int _moeTopK;
    private readonly int _moeIntermediateSize;
    // MoE shared-expert dims — zero unless any MoE layer carries shared experts.
    private readonly int _moeSharedIntermediateSize;
    private readonly int _moeNumSharedExperts;
    private int _capacitySeqLen;

    // ── Transformer layer scratch (all FP32) ──────────────────────────
    public VulkanDevice.Buffer HiddenState { get; private set; } = null!;
    public VulkanDevice.Buffer Residual { get; private set; } = null!;
    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
    public VulkanDevice.Buffer AddScratch { get; private set; } = null!;
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


    // ── Logits (last token only) ──────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; private set; }

    // ── Host → device transfer scratch (tokens + positions) ──────────
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    public VulkanForwardState(
        VulkanDevice device,
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int initialSeqLen,
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
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return;

        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long qBytes = (long)seqLen * _numHeads * _headDim * sizeof(float);
        long kvBytes = (long)seqLen * _numKvHeads * _headDim * sizeof(float);
        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        HiddenState = _device.Allocate(hiddenBytes);
        Residual = _device.Allocate(hiddenBytes);
        NormOutput = _device.Allocate(hiddenBytes);
        AddScratch = _device.Allocate(hiddenBytes);

        Q = _device.Allocate(qBytes);
        K = _device.Allocate(kvBytes);
        V = _device.Allocate(kvBytes);
        AttnOutput = _device.Allocate(qBytes);

        FfnGate = _device.Allocate(ffnBytes);
        FfnUp = _device.Allocate(ffnBytes);
        SiluOutput = _device.Allocate(ffnBytes);

        long moeBytes = AllocateMoeScratch(seqLen);

        // Resize positions buffer.
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;

        AllocatedBytes = hiddenBytes * 4 + qBytes * 2 + kvBytes * 2 + ffnBytes * 3
                       + moeBytes
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

        MoeRouterLogits = _device.Allocate(routerBytes);
        MoeTopkIndices = _device.Allocate(topkIdxBytes);
        MoeTopkWeights = _device.Allocate(topkWtBytes);
        MoeExpandedInput = _device.Allocate(expandedBytes);
        MoeGateInter = _device.Allocate(interBytes);
        MoeUpInter = _device.Allocate(interBytes);
        MoeSiluInter = _device.Allocate(interBytes);
        MoeDownRows = _device.Allocate(downBytes);

        long total = routerBytes + topkIdxBytes + topkWtBytes + expandedBytes
                   + interBytes * 3 + downBytes;

        if (_moeNumSharedExperts > 0)
        {
            long sharedInterBytes = (long)seqLen * _moeSharedIntermediateSize * sizeof(float);
            long sharedHiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);

            MoeSharedInput = _device.Allocate(sharedHiddenBytes);
            MoeSharedGate = _device.Allocate(sharedInterBytes);
            MoeSharedUp = _device.Allocate(sharedInterBytes);
            MoeSharedSilu = _device.Allocate(sharedInterBytes);
            MoeSharedDown = _device.Allocate(sharedHiddenBytes);
            MoeSharedSumA = _device.Allocate(sharedHiddenBytes);
            MoeSharedSumB = _device.Allocate(sharedHiddenBytes);

            total += sharedInterBytes * 3 + sharedHiddenBytes * 4;
        }

        return total;
    }

    private void ReleaseLayerScratch()
    {
        HiddenState?.Dispose();
        Residual?.Dispose();
        NormOutput?.Dispose();
        AddScratch?.Dispose();
        Q?.Dispose();
        K?.Dispose();
        V?.Dispose();
        AttnOutput?.Dispose();
        FfnGate?.Dispose();
        FfnUp?.Dispose();
        SiluOutput?.Dispose();

        MoeRouterLogits?.Dispose(); MoeRouterLogits = null;
        MoeTopkIndices?.Dispose(); MoeTopkIndices = null;
        MoeTopkWeights?.Dispose(); MoeTopkWeights = null;
        MoeExpandedInput?.Dispose(); MoeExpandedInput = null;
        MoeGateInter?.Dispose(); MoeGateInter = null;
        MoeUpInter?.Dispose(); MoeUpInter = null;
        MoeSiluInter?.Dispose(); MoeSiluInter = null;
        MoeDownRows?.Dispose(); MoeDownRows = null;

        MoeSharedInput?.Dispose(); MoeSharedInput = null;
        MoeSharedGate?.Dispose(); MoeSharedGate = null;
        MoeSharedUp?.Dispose(); MoeSharedUp = null;
        MoeSharedSilu?.Dispose(); MoeSharedSilu = null;
        MoeSharedDown?.Dispose(); MoeSharedDown = null;
        MoeSharedSumA?.Dispose(); MoeSharedSumA = null;
        MoeSharedSumB?.Dispose(); MoeSharedSumB = null;
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
