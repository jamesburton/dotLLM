using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated, 64-byte-aligned scratch buffers for the Qwen3MoeHybrid forward pass.
/// Sized for the widest of the GDN and full-attention sub-layer paths and grown in
/// power-of-two steps by <see cref="EnsureCapacity"/> to keep the hot path allocation-free.
/// </summary>
/// <remarks>
/// <para>
/// Each layer in Qwen3MoeHybrid has two sub-layers sharing a single residual stream:
/// a token-mixing path (GDN or full GQA attention) and a sparse MoE FFN. The buffers below
/// support both paths — GDN-specific buffers are unused on full-attention layers and
/// vice versa.
/// </para>
/// </remarks>
internal sealed unsafe class Qwen3MoeHybridForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _qElems;         // numQHeads * headDim
    private readonly int _kvElems;        // numKvHeads * headDim
    private readonly int _convDim;        // (2*NKHead + NVHead) * DState
    private readonly int _dConv;
    private readonly int _gdnVDim;        // NVHead * DState
    private readonly int _gdnKDim;        // NKHead * DState
    private readonly int _gdnHeads;       // NVHead
    private readonly int _inputScratchRowBytes;

    // MoE routing scratch sizing (fixed for the model lifetime).
    private readonly int _moeNumExperts;
    private readonly int _moeNumExpertsPerTok;

    private int _currentSeqLen;

    // ── Shared across all sub-layers ──────────────────────────────────────────
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint Logits;
    public nint InputQ8Scratch;           // byte-sized scratch for input pre-quantisation

    // ── GDN sub-layer ─────────────────────────────────────────────────────────

    /// <summary>
    /// Pre-conv rolling buffer: <c>[(DConv−1 + seqLen), convDim]</c>.
    /// The first <c>DConv−1</c> rows hold the per-layer conv state (copied in before each
    /// forward call); the trailing <c>seqLen</c> rows hold the current <c>attn_qkv</c>
    /// projection output. Passed directly to <c>Conv1dCausal.Execute</c> as input.
    /// </summary>
    public nint GdnConvInput;

    /// <summary>
    /// Conv output buffer: <c>[seqLen, convDim]</c>.
    /// Filled by <c>Conv1dCausal.Execute</c>, then SiLU-activated in place.
    /// The de-interleaved Q/K/V slices are subsequently copied from here into
    /// <see cref="GdnQBuf"/>, <see cref="GdnKBuf"/>, and <see cref="GdnVBuf"/>.
    /// </summary>
    public nint GdnQkvBuf;

    /// <summary>Gate projection from <c>attn_gate.weight</c>: <c>[seqLen, NVHead*DState]</c>.</summary>
    public nint GdnZBuf;

    /// <summary>
    /// Alpha projection buffer: <c>[seqLen, NVHead]</c>.
    /// Computed as <c>ssm_alpha.weight @ input</c>, then in-place transformed to
    /// per-token decay scalars: <c>g = exp(softplus(alpha + DtBias) * A)</c>.
    /// </summary>
    public nint GdnAlphaBuf;

    /// <summary>
    /// Beta projection buffer: <c>[seqLen, NVHead]</c>.
    /// Computed as <c>ssm_beta.weight @ input</c>, then sigmoid-activated in place.
    /// </summary>
    public nint GdnBetaBuf;

    /// <summary>Q after de-interleave and L2-normalisation: <c>[seqLen, NKHead*DState]</c>.</summary>
    public nint GdnQBuf;

    /// <summary>K after de-interleave and L2-normalisation: <c>[seqLen, NKHead*DState]</c>.</summary>
    public nint GdnKBuf;

    /// <summary>V after de-interleave: <c>[seqLen, NVHead*DState]</c>.</summary>
    public nint GdnVBuf;

    /// <summary>
    /// GDN scan output: <c>[seqLen, NVHead*DState]</c>.
    /// Overwritten in place by the per-head RMSNorm + silu(Z) gate step, then passed
    /// as input to the <c>ssm_out</c> projection.
    /// </summary>
    public nint GdnOut;

    // ── Full GQA attention sub-layer ──────────────────────────────────────────

    /// <summary>
    /// Fused Q+Gate projection output: <c>[seqLen, 2 * nQ * headDim]</c>.
    /// Output of <c>attn_q.weight @ input</c> before per-head de-interleave into
    /// <see cref="QScratch"/> (Q) and <see cref="GateScratch"/> (Gate).
    /// </summary>
    public nint QGateScratch;

    /// <summary>Q slice after de-interleave: <c>[seqLen, nQ * headDim]</c>.</summary>
    public nint QScratch;

    /// <summary>
    /// Gate slice after de-interleave: <c>[seqLen, nQ * headDim]</c>. Will be sigmoid'd in place
    /// and elementwise-multiplied with <see cref="AttnOutput"/> before the O-projection.
    /// </summary>
    public nint GateScratch;

    public nint KScratch;
    public nint VScratch;
    public nint AttnOutput;

    // ── MoE routing scratch ──────────────────────────────────────────────────
    // Small managed buffers (few KB) that survive across forward calls — no MoE-weight
    // dequant scratch is allocated here any more: the routed path now consumes the raw
    // GGUF quant view directly via MoeSwiGluMlp.ExecuteRoutedFromAssignments.

    /// <summary>Per-(token,slot) expert id [seqLen * numExpertsPerTok].</summary>
    public int[] MoeAssignExpert = Array.Empty<int>();

    /// <summary>Per-(token,slot) gate probability [seqLen * numExpertsPerTok].</summary>
    public float[] MoeAssignWeight = Array.Empty<float>();

    /// <summary>Per-expert exclusive-scan offsets [numExperts + 1] into <see cref="MoeBucketTokens"/>.</summary>
    public int[] MoeBucketCursors;

    /// <summary>Per-bucket-entry token index [seqLen * numExpertsPerTok].</summary>
    public int[] MoeBucketTokens = Array.Empty<int>();

    /// <summary>Per-bucket-entry top-k slot index [seqLen * numExpertsPerTok].</summary>
    public int[] MoeBucketSlots = Array.Empty<int>();

    /// <summary>Ordered list of distinct expert ids actually used [≤ seqLen * numExpertsPerTok].</summary>
    public int[] MoeUniqueExperts;

    // ── Computed properties ────────────────────────────────────────────────────

    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;
            long floats = 0;
            floats += s * _hiddenSize * 3;                        // HiddenState, Residual, NormOutput
            floats += s * _vocabSize;                              // Logits
            floats += (_dConv - 1 + s) * _convDim;                // GdnConvInput
            floats += s * _convDim;                                // GdnQkvBuf
            floats += s * _gdnVDim;                                // GdnZBuf
            floats += s * _gdnHeads * 2;                           // GdnAlphaBuf, GdnBetaBuf
            floats += s * _gdnKDim * 2;                            // GdnQBuf, GdnKBuf
            floats += s * _gdnVDim * 2;                            // GdnVBuf, GdnOut
            floats += s * _qElems * 4;                             // QGateScratch (2x), QScratch, GateScratch, AttnOutput
            floats += s * _kvElems * 2;                            // KScratch, VScratch
            long bytes = floats * sizeof(float);
            bytes += s * _inputScratchRowBytes;                    // InputQ8Scratch (byte-sized)
            return bytes;
        }
    }

    public Qwen3MoeHybridForwardState(
        int hiddenSize,
        int vocabSize,
        int qElems,
        int kvElems,
        int convDim,
        int dConv,
        int nVHead,
        int nKHead,
        int dState,
        int moeNumExperts = 0,
        int moeNumExpertsPerTok = 0)
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

        // InputQ8Scratch row size: largest quantised-input row across all GEMM kinds.
        // Base dim is the max of hidden, qElems (for attn output proj), gdnVDim (ssm_out input).
        int scratchBase = Math.Max(Math.Max(hiddenSize, qElems), nVHead * dState);
        int q8_0RowBytes = (scratchBase / 32) * 34;
        int q8_1RowBytes = (scratchBase / 32) * 36;
        int q8_kRowBytes = (scratchBase / 256) * 292;
        _inputScratchRowBytes = Math.Max(Math.Max(q8_0RowBytes, q8_1RowBytes), q8_kRowBytes);

        // MoE routing scratch: small persistent arrays sized from the routing fan-out.
        // The bucket cursors / unique-experts arrays scale with numExperts (small constants
        // for production MoE models — 64-256). The per-(token,slot) arrays scale with seqLen,
        // grown lazily in EnsureCapacity.
        _moeNumExperts = moeNumExperts;
        _moeNumExpertsPerTok = moeNumExpertsPerTok;
        MoeBucketCursors = moeNumExperts > 0 ? new int[moeNumExperts + 1] : Array.Empty<int>();
        MoeUniqueExperts = moeNumExperts > 0 ? new int[moeNumExperts] : Array.Empty<int>();

        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    /// <summary>
    /// Grows all buffers to cover at least <paramref name="seqLen"/> tokens, reallocating
    /// in power-of-two increments. No-op when current capacity already suffices.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        HiddenState = AllocFloats((long)cap * _hiddenSize);
        Residual = AllocFloats((long)cap * _hiddenSize);
        NormOutput = AllocFloats((long)cap * _hiddenSize);
        Logits = AllocFloats((long)cap * _vocabSize);
        InputQ8Scratch = AllocBytes((long)cap * _inputScratchRowBytes);

        GdnConvInput = AllocFloats((long)(_dConv - 1 + cap) * _convDim);
        GdnQkvBuf = AllocFloats((long)cap * _convDim);
        GdnZBuf = AllocFloats((long)cap * _gdnVDim);
        GdnAlphaBuf = AllocFloats((long)cap * _gdnHeads);
        GdnBetaBuf = AllocFloats((long)cap * _gdnHeads);
        GdnQBuf = AllocFloats((long)cap * _gdnKDim);
        GdnKBuf = AllocFloats((long)cap * _gdnKDim);
        GdnVBuf = AllocFloats((long)cap * _gdnVDim);
        GdnOut = AllocFloats((long)cap * _gdnVDim);

        QGateScratch = AllocFloats((long)cap * 2 * _qElems);
        QScratch = AllocFloats((long)cap * _qElems);
        GateScratch = AllocFloats((long)cap * _qElems);
        KScratch = AllocFloats((long)cap * _kvElems);
        VScratch = AllocFloats((long)cap * _kvElems);
        AttnOutput = AllocFloats((long)cap * _qElems);

        // MoE per-(token,slot) routing arrays scale with seqLen × k. Reallocated as managed
        // arrays — the public MoeSwiGluMlp.Route + ExecuteRoutedFromAssignments API consumes
        // Span<T>/ReadOnlySpan<T>, so plain arrays work without extra pinning.
        if (_moeNumExperts > 0 && _moeNumExpertsPerTok > 0)
        {
            int totalAssignments = cap * _moeNumExpertsPerTok;
            MoeAssignExpert = new int[totalAssignments];
            MoeAssignWeight = new float[totalAssignments];
            MoeBucketTokens = new int[totalAssignments];
            MoeBucketSlots = new int[totalAssignments];
        }

        _currentSeqLen = cap;
    }

    private static nint AllocFloats(long count)
        => (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);

    private static nint AllocBytes(long count)
        => (nint)NativeMemory.AlignedAlloc((nuint)count, 64);

    private void FreeBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref InputQ8Scratch);
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
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        FreeBuffers();
        _currentSeqLen = 0;
    }
}
