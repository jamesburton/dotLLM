using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Pre-allocated F32 device scratch buffers for the Qwen3MoeHybrid CUDA forward pass.
/// Mirror of <c>Qwen3MoeHybridForwardState</c> (CPU) but allocating GPU memory via
/// <c>cuMemAlloc_v2</c>. Sized for the widest of the GDN and full-attention sub-layer
/// paths and grown in power-of-two steps by <see cref="EnsureCapacity"/> so the hot
/// path stays allocation-free.
/// </summary>
/// <remarks>
/// <para>
/// F32 throughout for clean parity with the verified CPU reference. An FP16 fast path
/// is left to follow-up optimisation work — initial correctness must hold first.
/// </para>
/// <para>
/// Buffer ownership: this state owns the activation scratch. Weights are owned by
/// the model; KV cache and GDN state cache are separate objects.
/// </para>
/// </remarks>
internal sealed unsafe class CudaQwen3MoeHybridForwardState : IDisposable
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
    private readonly int _moeNumExperts;
    private readonly int _moeIntermediate;
    private readonly long _moeW1ElemsPerExpert;
    private readonly long _moeW2ElemsPerExpert;

    private int _currentSeqLen;
    private int _logitsCurrentRows;
    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    // ── Shared across all sub-layers ──────────────────────────────────────────
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;

    // Sized independently of the other per-token buffers above (issue #185) -- see
    // CudaQwen3HybridDenseForwardState's identically-named field doc for the full rationale.
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

    // ── Full GQA attention sub-layer ──────────────────────────────────────────
    public nint QGateScratch;   // [seqLen * 2 * qElems]
    public nint QScratch;       // [seqLen * qElems]
    public nint GateScratch;    // [seqLen * qElems]
    public nint KScratch;       // [seqLen * kvElems]
    public nint VScratch;       // [seqLen * kvElems]
    public nint AttnOutput;     // [seqLen * qElems]

    // ── MoE routed-expert on-demand dequant scratch (per-layer reuse) ─────────
    //
    // These three scratches are allocated ONLY when the ctor flag
    // allocFullExpertDequantScratch is true (the abandoned "all-experts-to-F32-once" path).
    // The production GPU dispatcher uses CudaMoeFfn's per-expert dequant-then-GEMV (or the
    // Phase-B grouped quantized GEMV) and leaves all three pointers at zero. Kept as
    // public fields for backwards compatibility with the original (now-unused) path.
    /// <summary>Per-call routed-expert gate-proj scratch: [numExperts × intermediate × hidden] F32. Zero on the dispatcher path.</summary>
    public nint MoeW1Scratch;
    /// <summary>Per-call routed-expert down-proj scratch: [numExperts × hidden × intermediate] F32. Zero on the dispatcher path.</summary>
    public nint MoeW2Scratch;
    /// <summary>Per-call routed-expert up-proj scratch: [numExperts × intermediate × hidden] F32. Zero on the dispatcher path.</summary>
    public nint MoeW3Scratch;

    /// <summary>Per-expert device pointer arrays — fixed for the model lifetime. Null on the dispatcher path.</summary>
    public readonly nint[]? MoeW1Ptrs;
    public readonly nint[]? MoeW2Ptrs;
    public readonly nint[]? MoeW3Ptrs;

    /// <summary>Element count per expert slice in MoeW1/W3 (intermediate × hidden).</summary>
    public long MoeW1ElemsPerExpert => _moeW1ElemsPerExpert;
    /// <summary>Element count per expert slice in MoeW2 (hidden × intermediate).</summary>
    public long MoeW2ElemsPerExpert => _moeW2ElemsPerExpert;

    // ── Token-id / position H2D staging ───────────────────────────────────────
    public nint TokenIdsDevice;
    public nint PositionsDevice;

    /// <summary>
    /// Initialises forward-state buffers for a Qwen3MoeHybrid model.
    /// </summary>
    /// <param name="hiddenSize">Model hidden / residual dimension.</param>
    /// <param name="vocabSize">Output vocabulary size.</param>
    /// <param name="qElems">numAttentionHeads * headDim.</param>
    /// <param name="kvElems">numKvHeads * headDim.</param>
    /// <param name="convDim">(2 * NKHead + NVHead) * DState for the GDN conv1d input width.</param>
    /// <param name="dConv">GDN conv kernel width.</param>
    /// <param name="nVHead">Number of GDN value heads.</param>
    /// <param name="nKHead">Number of GDN key heads.</param>
    /// <param name="dState">GDN per-head state dim.</param>
    /// <param name="moeNumExperts">Routed MoE expert count (e.g. 256 for qwen35moe).</param>
    /// <param name="moeIntermediate">MoE per-expert intermediate width.</param>
    /// <param name="allocFullExpertDequantScratch">
    /// When <see langword="true"/>, pre-allocate F32 dequant scratch covering ALL routed experts
    /// (<c>3 × numExperts × intermediate × hidden × sizeof(float)</c>; ~3.2 GiB for qwen35moe).
    /// When <see langword="false"/> (default; the GPU dispatcher path), this scratch is left
    /// unallocated and the MoE forward goes through on-demand per-expert dequant via
    /// <see cref="CudaMoeFfn"/> grouped GEMV. Keep <see langword="false"/> on any consumer-class
    /// GPU — the full-expert F32 scratch alone exceeds 12 GiB cards.
    /// </param>
    public CudaQwen3MoeHybridForwardState(
        int hiddenSize,
        int vocabSize,
        int qElems,
        int kvElems,
        int convDim,
        int dConv,
        int nVHead,
        int nKHead,
        int dState,
        int moeNumExperts,
        int moeIntermediate,
        bool allocFullExpertDequantScratch = false)
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
        _moeNumExperts = moeNumExperts;
        _moeIntermediate = moeIntermediate;

        if (allocFullExpertDequantScratch && moeNumExperts > 0 && moeIntermediate > 0)
        {
            _moeW1ElemsPerExpert = (long)moeIntermediate * hiddenSize;
            _moeW2ElemsPerExpert = (long)hiddenSize * moeIntermediate;
            long totalW1Elems = (long)moeNumExperts * _moeW1ElemsPerExpert;
            long totalW2Elems = (long)moeNumExperts * _moeW2ElemsPerExpert;
            MoeW1Scratch = AllocDevice(totalW1Elems * sizeof(float));
            MoeW2Scratch = AllocDevice(totalW2Elems * sizeof(float));
            MoeW3Scratch = AllocDevice(totalW1Elems * sizeof(float));

            MoeW1Ptrs = new nint[moeNumExperts];
            MoeW2Ptrs = new nint[moeNumExperts];
            MoeW3Ptrs = new nint[moeNumExperts];
            for (int e = 0; e < moeNumExperts; e++)
            {
                MoeW1Ptrs[e] = MoeW1Scratch + (nint)(e * _moeW1ElemsPerExpert * sizeof(float));
                MoeW2Ptrs[e] = MoeW2Scratch + (nint)(e * _moeW2ElemsPerExpert * sizeof(float));
                MoeW3Ptrs[e] = MoeW3Scratch + (nint)(e * _moeW1ElemsPerExpert * sizeof(float));
            }
        }

        _currentSeqLen = 0;
        EnsureCapacity(1);
        EnsureLogitsCapacity(1);
    }

    /// <summary>
    /// Rounding granularity (in tokens) <see cref="EnsureCapacity"/> switches to once its
    /// requested length exceeds this size -- mirrors
    /// <see cref="CudaQwen3HybridDenseForwardState.CapacityGranularity"/>; see that type's
    /// identically-named member and <see cref="EnsureCapacity"/>'s remarks for the full
    /// issue #188 rationale.
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
    /// pow2 boundary. See <see cref="CudaQwen3HybridDenseForwardState.EnsureCapacity"/> for the
    /// full issue #188 finding this mirrors.
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

        TokenIdsDevice = AllocDevice((long)cap * sizeof(int));
        PositionsDevice = AllocDevice((long)cap * sizeof(int));

        _currentSeqLen = cap;
    }

    /// <summary>
    /// Grows <see cref="Logits"/> to cover at least <paramref name="rows"/> rows (issue #185) --
    /// see <see cref="CudaQwen3HybridDenseForwardState.EnsureLogitsCapacity"/> for the rationale.
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
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
    }

    private void FreeMoeScratch()
    {
        FreeIfNonZero(ref MoeW1Scratch);
        FreeIfNonZero(ref MoeW2Scratch);
        FreeIfNonZero(ref MoeW3Scratch);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        FreeSequenceBuffers();
        FreeIfNonZero(ref Logits);
        FreeMoeScratch();
        _currentSeqLen = 0;
        _logitsCurrentRows = 0;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CudaQwen3MoeHybridForwardState()
    {
        if (_disposed) return;
        FreeSequenceBuffers();
        FreeIfNonZero(ref Logits);
        FreeMoeScratch();
    }
}
