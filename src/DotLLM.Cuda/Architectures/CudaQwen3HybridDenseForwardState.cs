using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Pre-allocated F32 device scratch buffers for the Qwen3HybridDense (<c>qwen35</c>, e.g.
/// PrismML's Bonsai-27B) CUDA forward pass. Mirror of
/// <see cref="CudaQwen3MoeHybridForwardState"/> with the MoE routing scratch replaced by
/// dense SwiGLU FFN scratch (<see cref="FfnGate"/>/<see cref="FfnUp"/>/<see cref="SiluOutput"/>,
/// sized to <c>intermediateSize</c>) — matches
/// <c>Qwen3HybridDenseForwardState</c>'s (CPU) buffer naming/shape convention.
/// </summary>
internal sealed unsafe class CudaQwen3HybridDenseForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _qElems;        // numAttentionHeads * headDim
    private readonly int _kvElems;       // numKvHeads * headDim
    private readonly int _convDim;       // (2 * NKHead + NVHead) * DState
    private readonly int _dConv;
    private readonly int _gdnVDim;       // NVHead * DState
    private readonly int _gdnKDim;       // NKHead * DState
    private readonly int _gdnHeads;      // NVHead
    private readonly int _intermediateSize;

    private int _currentSeqLen;
    private int _logitsCurrentRows;
    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    // ── Shared across all sub-layers ──────────────────────────────────────────
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;

    // Sized independently of the other per-token buffers above (issue #185) -- unlike
    // HiddenState/NormOutput/etc., which every position's value feeds into the NEXT layer's
    // computation for every OTHER position (via attention/GDN mixing) and so must stay sized to
    // the full sequence length, Logits is a purely trailing projection nothing else reads back
    // from within the same Forward call. The IKvCache-decode overload's documented contract
    // (see IModel.Forward's XML doc) returns only the LAST token's logits, and every real caller
    // (BenchRunner, TextGenerator) only ever reads the last row regardless of what shape comes
    // back -- so sizing this cap*vocabSize*sizeof(float) (e.g. ~970 MiB at cap=1024 for a
    // 248,320-vocab model) was pure waste for every kvCache-enabled call, and was the single
    // largest contributor to the VRAM-ceiling hang `dotllm bench --depth` hit beyond ~650-768 on
    // a 12GB card (confirmed via cuMemGetInfo instrumentation: EnsureCapacity's allocation for
    // cap=1024 landed on EXACTLY 0 MiB free). See EnsureLogitsCapacity.
    public nint Logits;

    // ── GDN sub-layer ─────────────────────────────────────────────────────────
    public nint GdnConvInput;   // [(DConv-1 + seqLen) * convDim]
    public nint GdnQkvBuf;      // [seqLen * convDim] — also conv1d output (in-place SiLU)
    public nint GdnZBuf;        // [seqLen * gdnVDim] — attn_gate projection
    public nint GdnAlphaBuf;    // [seqLen * NVHead] — alpha proj → g after softplus+exp
    public nint GdnBetaBuf;     // [seqLen * NVHead] — beta proj → sigmoid
    public nint GdnQBuf;        // [seqLen * gdnKDim]
    public nint GdnKBuf;        // [seqLen * gdnKDim]
    public nint GdnVBuf;        // [seqLen * gdnVDim]
    public nint GdnOut;         // [seqLen * gdnVDim] — scan output / post-gate

    // ── Opt-in cooperative-split GDN scan scratch (issue #180, DOTLLM_GDN_SCAN_APPROX_SPLIT4) ──
    // Fixed-size (independent of seqLen — one decode step's worth of scratch, reused across every
    // token/layer/step since it's transient, not accumulated state). Only allocated/used when
    // CudaKernels.EnableGdnScanApproxSplit4 is set; harmless small allocation otherwise (NVHead*4*
    // DState floats, e.g. 48*4*128*4 bytes = 98KB for Bonsai-27B — negligible vs the multi-GB model).
    public nint GdnScanPartialTmp;  // [NVHead, 4, DState]
    public nint GdnScanPartialOut;  // [NVHead, 4, DState]

    // ── Full GQA attention sub-layer ──────────────────────────────────────────
    public nint QGateScratch;   // [seqLen * 2 * qElems]
    public nint QScratch;       // [seqLen * qElems]
    public nint GateScratch;    // [seqLen * qElems]
    public nint KScratch;       // [seqLen * kvElems]
    public nint VScratch;       // [seqLen * kvElems]
    public nint AttnOutput;     // [seqLen * qElems]

    // ── Dense SwiGLU FFN scratch (replaces MoE routing scratch) ───────────────
    public nint FfnGate;        // [seqLen * intermediateSize] — ffn_gate.weight @ normOutput
    public nint FfnUp;          // [seqLen * intermediateSize] — ffn_up.weight @ normOutput
    public nint SiluOutput;     // [seqLen * intermediateSize] — silu(FfnGate) * FfnUp

    // ── Token-id / position H2D staging ───────────────────────────────────────
    public nint TokenIdsDevice;
    public nint PositionsDevice;

    public CudaQwen3HybridDenseForwardState(
        int hiddenSize,
        int vocabSize,
        int qElems,
        int kvElems,
        int convDim,
        int dConv,
        int nVHead,
        int nKHead,
        int dState,
        int intermediateSize)
    {
        _hiddenSize = hiddenSize;
        _vocabSize = vocabSize;
        _qElems = qElems;
        _kvElems = kvElems;
        _convDim = convDim;
        _dConv = dConv;
        _gdnVDim = nVHead * dState;
        _gdnKDim = nKHead * dState;
        _gdnHeads = nVHead;
        _intermediateSize = intermediateSize;

        _currentSeqLen = 0;
        EnsureCapacity(1);
        EnsureLogitsCapacity(1);

        // Fixed-size, seqLen-independent — allocated once, not part of FreeSequenceBuffers/EnsureCapacity.
        GdnScanPartialTmp = AllocDevice((long)_gdnHeads * 4 * dState * sizeof(float));
        GdnScanPartialOut = AllocDevice((long)_gdnHeads * 4 * dState * sizeof(float));
    }

    /// <summary>
    /// Rounding granularity (in tokens) <see cref="EnsureCapacity"/> switches to once its
    /// requested length exceeds this size -- see the method's remarks for the full issue #188
    /// rationale. 256 is a round number comfortably larger than typical single-token
    /// decode churn, so ordinary serving still amortizes reallocation across small prompt-length
    /// variation, while bounding the worst-case waste at any depth to at most
    /// <c>CapacityGranularity - 1</c> tokens' worth of buffers instead of the up-to-2x waste
    /// power-of-two rounding produces right after crossing a pow2 boundary.
    /// </summary>
    internal const int CapacityGranularity = 256;

    /// <summary>
    /// Rounds <paramref name="seqLen"/> up to the next allocation-worthy capacity for
    /// <see cref="EnsureCapacity"/>. See <see cref="CapacityGranularity"/> and
    /// <see cref="EnsureCapacity"/>'s remarks for the issue #188 rationale behind the two-regime
    /// split.
    /// </summary>
    internal static int RoundUpCapacity(int seqLen)
    {
        if (seqLen <= CapacityGranularity)
            return (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);

        return (seqLen + CapacityGranularity - 1) / CapacityGranularity * CapacityGranularity;
    }

    /// <summary>
    /// Grows all per-token buffers to cover at least <paramref name="seqLen"/> tokens.
    /// No-op when capacity already suffices. Does NOT size <see cref="Logits"/> -- see
    /// <see cref="EnsureLogitsCapacity"/>.
    /// </summary>
    /// <remarks>
    /// Issue #188: below <see cref="CapacityGranularity"/> tokens, capacity still rounds up to
    /// the next power of two (unchanged from the original scheme) -- at this scale the absolute
    /// byte cost of over-allocating is small, and pow2 rounding gives cheap amortization for the
    /// common case of many small, differently-sized calls (e.g. varying prompt lengths in normal
    /// serving) without reallocating on every single-token-different request. Above the
    /// granularity threshold, capacity instead rounds up to the next multiple of
    /// <see cref="CapacityGranularity"/> tokens -- this bounds the worst-case waste to at most
    /// <c>CapacityGranularity - 1</c> tokens' worth of buffers REGARDLESS of how large seqLen
    /// gets, instead of the up-to-2x waste power-of-two rounding produces right after crossing a
    /// pow2 boundary (e.g. seqLen=1025 previously rounded to cap=2048, wasting ~1023 tokens'
    /// worth of every per-token buffer this class owns). That 2x-at-scale behavior is exactly
    /// what let `dotllm bench --depth` land on EXACTLY 0 MiB free VRAM at depth 1536 (rounds to
    /// cap=2048 under pure pow2) on a 12 GB card -- see issue #188, a recurrence of #185's
    /// original Logits-buffer finding but for every OTHER per-token buffer in this class, which
    /// #185 deliberately left alone (unlike Logits, they're live in ordinary variable-length
    /// prefill too, so unconditionally exact-sizing them like Logits would defeat the
    /// reallocation-amortization these buffers exist for in real serving). A fixed-granularity
    /// step still reallocates only when a call's seqLen crosses into a new granularity bucket
    /// (same amortization property as pow2, just with smaller, constant-size buckets instead of
    /// buckets that double forever), while capping the worst case that made #188 possible.
    /// </remarks>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return;

        int cap = RoundUpCapacity(seqLen);
        FreeSequenceBuffers();

        HiddenState = AllocDevice((long)cap * _hiddenSize * sizeof(float));
        Residual = AllocDevice((long)cap * _hiddenSize * sizeof(float));
        NormOutput = AllocDevice((long)cap * _hiddenSize * sizeof(float));

        GdnConvInput = AllocDevice((long)(_dConv - 1 + cap) * _convDim * sizeof(float));
        GdnQkvBuf = AllocDevice((long)cap * _convDim * sizeof(float));
        GdnZBuf = AllocDevice((long)cap * _gdnVDim * sizeof(float));
        GdnAlphaBuf = AllocDevice((long)cap * _gdnHeads * sizeof(float));
        GdnBetaBuf = AllocDevice((long)cap * _gdnHeads * sizeof(float));
        GdnQBuf = AllocDevice((long)cap * _gdnKDim * sizeof(float));
        GdnKBuf = AllocDevice((long)cap * _gdnKDim * sizeof(float));
        GdnVBuf = AllocDevice((long)cap * _gdnVDim * sizeof(float));
        GdnOut = AllocDevice((long)cap * _gdnVDim * sizeof(float));

        QGateScratch = AllocDevice((long)cap * 2 * _qElems * sizeof(float));
        QScratch = AllocDevice((long)cap * _qElems * sizeof(float));
        GateScratch = AllocDevice((long)cap * _qElems * sizeof(float));
        KScratch = AllocDevice((long)cap * _kvElems * sizeof(float));
        VScratch = AllocDevice((long)cap * _kvElems * sizeof(float));
        AttnOutput = AllocDevice((long)cap * _qElems * sizeof(float));

        FfnGate = AllocDevice((long)cap * _intermediateSize * sizeof(float));
        FfnUp = AllocDevice((long)cap * _intermediateSize * sizeof(float));
        SiluOutput = AllocDevice((long)cap * _intermediateSize * sizeof(float));

        TokenIdsDevice = AllocDevice((long)cap * sizeof(int));
        PositionsDevice = AllocDevice((long)cap * sizeof(int));

        _currentSeqLen = cap;
    }

    /// <summary>
    /// Grows <see cref="Logits"/> to cover at least <paramref name="rows"/> rows (issue #185).
    /// Sized independently of <see cref="EnsureCapacity"/>'s <c>cap</c> -- callers pass 1 for the
    /// common kvCache-enabled decode/depth-extension path (only the last token's logits are ever
    /// consumed) and the real row count for the uncached multi-token path. No power-of-two
    /// rounding: the two regimes are wildly different scales and this buffer is never in a hot
    /// per-token growth loop the way the other per-token buffers are.
    /// </summary>
    public void EnsureLogitsCapacity(int rows)
    {
        if (rows <= _logitsCurrentRows) return;
        FreeIfNonZero(ref Logits);
        Logits = AllocDevice((long)rows * _vocabSize * sizeof(float));
        _logitsCurrentRows = rows;
    }

    private nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            CudaDriverApi.cuMemFree_v2(ptr);
            ptr = 0;
        }
    }

    private void FreeSequenceBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref GdnConvInput);
        FreeIfNonZero(ref GdnQkvBuf);
        FreeIfNonZero(ref GdnZBuf);
        FreeIfNonZero(ref GdnAlphaBuf);
        FreeIfNonZero(ref GdnBetaBuf);
        FreeIfNonZero(ref GdnQBuf);
        FreeIfNonZero(ref GdnKBuf);
        FreeIfNonZero(ref GdnVBuf);
        FreeIfNonZero(ref GdnOut);
        FreeIfNonZero(ref QGateScratch);
        FreeIfNonZero(ref QScratch);
        FreeIfNonZero(ref GateScratch);
        FreeIfNonZero(ref KScratch);
        FreeIfNonZero(ref VScratch);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref FfnGate);
        FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput);
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        FreeSequenceBuffers();
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref GdnScanPartialTmp);
        FreeIfNonZero(ref GdnScanPartialOut);
        _currentSeqLen = 0;
        _logitsCurrentRows = 0;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CudaQwen3HybridDenseForwardState()
    {
        if (_disposed) return;
        FreeSequenceBuffers();
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref GdnScanPartialTmp);
        FreeIfNonZero(ref GdnScanPartialOut);
    }
}
