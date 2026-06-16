using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated scratch buffers for the transformer forward pass. Reused across calls to
/// achieve zero per-call allocation on the hot path. Call <see cref="EnsureCapacity"/>
/// before each forward pass to resize if the sequence length has grown.
/// </summary>
internal sealed unsafe class TransformerForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;
    // Per-token element counts that drive the Q/AttnOutput and K/V scratch
    // allocation. For a uniform head dim these are numHeads*headDim and
    // numKvHeads*headDim respectively. For Gemma 4 with a distinct
    // global_head_dim the full-attention layers project a wider per-head
    // slice (and a different KV-head count), so these are sized to the LARGER
    // per-layer block so a single allocation covers both layer types.
    private readonly int _qBlockElems;   // max over layer types of numHeads   * layerHeadDim
    private readonly int _kvBlockElems;  // max over layer types of kvHeads(L) * layerHeadDim

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated for inference scratch buffers.</summary>
    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;

            long bytes = 0;
            bytes += s * _hiddenSize * 3;                    // HiddenState + Residual + NormOutput
            bytes += s * _qBlockElems * 2;                   // Q + AttnOutput
            bytes += s * _kvBlockElems * 2;                  // K + V
            bytes += s * _intermediateSize * 3;              // FfnGate + FfnUp + SiluOutput
            bytes += s * _vocabSize;                          // Logits (all positions for speculative verify)
            bytes *= sizeof(float);
            // InputQ8Scratch: seqLen × max(Q8_0, Q8_1, Q8_K) row bytes
            int maxInputDim = Math.Max(_hiddenSize, _intermediateSize);
            int q8_0Bytes = (maxInputDim / 32) * 34;
            int q8_1Bytes = (maxInputDim / 32) * 36;
            int q8_kBytes = (maxInputDim / 256) * 292;
            bytes += s * Math.Max(Math.Max(q8_0Bytes, q8_1Bytes), q8_kBytes);
            // RoPE tables (managed, but still part of compute memory)
            bytes += (CosTable.Length + SinTable.Length) * sizeof(float);
            if (GlobalCosTable is not null && GlobalSinTable is not null)
                bytes += (GlobalCosTable.Length + GlobalSinTable.Length) * sizeof(float);
            return bytes;
        }
    }

    // All pointers are 64-byte-aligned via NativeMemory.AlignedAlloc.
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint Q;
    public nint K;
    public nint V;
    public nint AttnOutput;
    public nint FfnGate;
    public nint FfnUp;
    public nint SiluOutput;
    public nint Logits;

    /// <summary>
    /// Scratch buffer for pre-quantized input rows [seqLen × rowBytes].
    /// Sized for the largest of Q8_0 (34B/32el), Q8_1 (36B/32el), and Q8_K (292B/256el).
    /// Used to quantize the input once and reuse across Q/K/V and Gate/Up projections.
    /// </summary>
    public nint InputQ8Scratch;

    /// <summary>Pre-computed RoPE cosine table [maxSeqLen * halfDim].</summary>
    public float[] CosTable { get; }

    /// <summary>Pre-computed RoPE sine table [maxSeqLen * halfDim].</summary>
    public float[] SinTable { get; }

    /// <summary>
    /// Number of rotated dimensions for the primary (sliding-window) RoPE table.
    /// Equals the <c>ropeDim</c> the primary table was built with.
    /// </summary>
    public int RopeDim { get; }

    /// <summary>
    /// Optional secondary RoPE cosine table for the FULL-attention layers
    /// (Gemma 4 per-attention-type RoPE — different base theta + partial-rotary
    /// factor). Null when the model uses a single RoPE configuration for every
    /// layer.
    /// </summary>
    public float[]? GlobalCosTable { get; }

    /// <summary>Optional secondary RoPE sine table for the full-attention layers. Null when unused.</summary>
    public float[]? GlobalSinTable { get; }

    /// <summary>
    /// Number of rotated dimensions for the secondary (full-attention) RoPE
    /// table. May be smaller than <see cref="RopeDim"/> when a partial-rotary
    /// factor is in effect (only the leading <c>GlobalRopeDim</c> dims rotate).
    /// Zero when <see cref="GlobalCosTable"/> is null.
    /// </summary>
    public int GlobalRopeDim { get; }

    public TransformerForwardState(
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int maxSeqLen, int ropeDim,
        float ropeTheta,
        int globalRopeDim = 0, float globalRopeTheta = 0f,
        int qBlockElems = 0, int kvBlockElems = 0)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        // Default the per-token Q/KV block sizes to the uniform-head-dim layout
        // when the caller does not pass explicit (per-layer max) sizes — keeps
        // every existing caller byte-identical.
        _qBlockElems = qBlockElems > 0 ? qBlockElems : numHeads * headDim;
        _kvBlockElems = kvBlockElems > 0 ? kvBlockElems : numKvHeads * headDim;

        // Pre-compute RoPE frequency tables
        int halfDim = ropeDim / 2;
        RopeDim = ropeDim;
        CosTable = new float[maxSeqLen * halfDim];
        SinTable = new float[maxSeqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(maxSeqLen, ropeDim, ropeTheta, CosTable, SinTable);

        // Optional secondary table for the full-attention layers (Gemma 4):
        // different base theta and (via globalRopeDim < ropeDim) partial rotary.
        if (globalRopeDim > 0)
        {
            GlobalRopeDim = globalRopeDim;
            int globalHalfDim = globalRopeDim / 2;
            GlobalCosTable = new float[maxSeqLen * globalHalfDim];
            GlobalSinTable = new float[maxSeqLen * globalHalfDim];
            DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(
                maxSeqLen, globalRopeDim, globalRopeTheta, GlobalCosTable, GlobalSinTable);
        }

        // Initial allocation for 1 token (decode mode)
        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough for <paramref name="seqLen"/> tokens.
    /// Uses power-of-2 growth to amortize reallocation cost.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        HiddenState = AllocFloats(newCapacity * _hiddenSize);
        Residual = AllocFloats(newCapacity * _hiddenSize);
        NormOutput = AllocFloats(newCapacity * _hiddenSize);
        Q = AllocFloats((long)newCapacity * _qBlockElems);
        K = AllocFloats((long)newCapacity * _kvBlockElems);
        V = AllocFloats((long)newCapacity * _kvBlockElems);
        AttnOutput = AllocFloats((long)newCapacity * _qBlockElems);
        FfnGate = AllocFloats(newCapacity * _intermediateSize);
        FfnUp = AllocFloats(newCapacity * _intermediateSize);
        SiluOutput = AllocFloats(newCapacity * _intermediateSize);
        Logits = AllocFloats((long)newCapacity * _vocabSize); // All positions' logits for speculative verify

        // InputQ8Scratch: seqLen × max(q8_0RowBytes, q8_1RowBytes, q8_kRowBytes) for pre-quantized GEMM input reuse.
        // Q8_0: 34 bytes per 32-element block. Q8_1: 36 bytes per 32-element block. Q8_K: 292 bytes per 256-element block.
        int maxInputDim = Math.Max(_hiddenSize, _intermediateSize);
        int q8_0RowBytes = (maxInputDim / 32) * 34;
        int q8_1RowBytes = (maxInputDim / 32) * 36;
        int q8_kRowBytes = (maxInputDim / 256) * 292;
        int scratchRowBytes = Math.Max(Math.Max(q8_0RowBytes, q8_1RowBytes), q8_kRowBytes);
        InputQ8Scratch = AllocBytes(newCapacity * scratchRowBytes);

        _currentSeqLen = newCapacity;
    }

    private static nint AllocFloats(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
    }

    private static nint AllocBytes(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)count, 64);
    }

    private void FreeBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Q);
        FreeIfNonZero(ref K);
        FreeIfNonZero(ref V);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref FfnGate);
        FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput);
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref InputQ8Scratch);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }

    public void Dispose()
    {
        FreeBuffers();
        _currentSeqLen = 0;
    }
}
