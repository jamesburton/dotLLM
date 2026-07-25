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
    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    // ── Shared across all sub-layers ──────────────────────────────────────────
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
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

        // Fixed-size, seqLen-independent — allocated once, not part of FreeSequenceBuffers/EnsureCapacity.
        GdnScanPartialTmp = AllocDevice((long)_gdnHeads * 4 * dState * sizeof(float));
        GdnScanPartialOut = AllocDevice((long)_gdnHeads * 4 * dState * sizeof(float));
    }

    /// <summary>
    /// Grows all per-token buffers to cover at least <paramref name="seqLen"/> tokens,
    /// reallocating in power-of-two increments. No-op when capacity already suffices.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        HiddenState = AllocDevice((long)cap * _hiddenSize * sizeof(float));
        Residual = AllocDevice((long)cap * _hiddenSize * sizeof(float));
        NormOutput = AllocDevice((long)cap * _hiddenSize * sizeof(float));
        Logits = AllocDevice((long)cap * _vocabSize * sizeof(float));

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
        FreeIfNonZero(ref Logits);
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
        FreeIfNonZero(ref GdnScanPartialTmp);
        FreeIfNonZero(ref GdnScanPartialOut);
        _currentSeqLen = 0;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CudaQwen3HybridDenseForwardState()
    {
        if (_disposed) return;
        FreeSequenceBuffers();
        FreeIfNonZero(ref GdnScanPartialTmp);
        FreeIfNonZero(ref GdnScanPartialOut);
    }
}
