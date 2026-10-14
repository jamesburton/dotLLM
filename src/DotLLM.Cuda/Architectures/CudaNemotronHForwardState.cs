using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side, power-of-two-growth scratch buffers for the NemotronH hybrid forward pass on
/// CUDA. Mirrors <see cref="DotLLM.Models.Architectures.NemotronHForwardState"/> (CPU) —
/// same buffer inventory and growth strategy, ported to <c>cuMemAlloc_v2</c> device pointers.
/// Adds <see cref="TokenIdsDevice"/>/<see cref="PositionsDevice"/> (int32 device arrays the
/// embedding-lookup and RoPE CUDA kernels read from device memory) which the CPU host doesn't
/// need (it reads <c>ReadOnlySpan&lt;int&gt;</c> directly).
/// </summary>
internal sealed class CudaNemotronHForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _maxIntermediateSize;
    private readonly int _vocabSize;
    private readonly int _qElems;
    private readonly int _kvElems;

    private readonly int _inputProjectionDim;
    private readonly int _convDim;
    private readonly int _dConv;
    private readonly int _dInner;
    private readonly int _nHead;
    private readonly int _bcDim; // n_group * d_state

    private int _currentSeqLen;
    private readonly int _maxSeqLen; // cap for TokenIdsDevice/PositionsDevice (int32, not resized per-call)

    /// <summary>Fixed allocated capacity (in elements) of <see cref="TokenIdsDevice"/> and
    /// <see cref="PositionsDevice"/> — unlike every other buffer here, these are sized once at
    /// construction and never grown by <see cref="EnsureCapacity"/>. Callers writing into either
    /// buffer must bound their write length against this value themselves.</summary>
    public int MaxSeqLen => _maxSeqLen;

    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint FfnIntermediate;
    public nint Logits;

    public nint QScratch;
    public nint KScratch;
    public nint VScratch;
    public nint AttnOutput;

    public nint Zxbcdt;
    public nint ConvInput;
    public nint XBC;
    public nint DtBuffer;
    public nint SsmX;
    public nint SsmB;
    public nint SsmC;
    public nint SsmY;
    /// <summary>Extraction scratch for the SwiGLU gate `z` slice of Zxbcdt (CPU/Vulkan step
    /// 8/10) — needed because Zxbcdt's per-token row stride is <c>inputProjectionDim</c>, not
    /// <c>dInner</c>, so `z` cannot be passed directly to a single fused
    /// <c>LaunchSwiGLUF32</c> call over all seqLen tokens at once without first being copied
    /// into a contiguous dInner-strided buffer. See Task 9.</summary>
    public nint SsmZ;

    /// <summary>Device int32 array of the current call's token ids, length &gt;= seqLen. Plain
    /// field (not a property) like every other buffer here, so it can be passed by
    /// <c>ref</c> to <see cref="FreeIfNonZero"/> in <see cref="Dispose"/>.</summary>
    public nint TokenIdsDevice;

    /// <summary>Device int32 array of the current call's positions, length &gt;= seqLen.</summary>
    public nint PositionsDevice;

    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;
            long floats = 0;
            floats += s * _hiddenSize * 3;             // HiddenState, Residual, NormOutput
            floats += s * _maxIntermediateSize;         // FfnIntermediate
            floats += s * _vocabSize;                   // Logits
            floats += s * _qElems;                      // QScratch
            floats += s * _kvElems * 2;                 // KScratch, VScratch
            floats += s * _qElems;                       // AttnOutput
            floats += s * _inputProjectionDim;           // Zxbcdt
            floats += (_dConv - 1 + s) * _convDim;        // ConvInput
            floats += s * _convDim;                       // XBC
            floats += s * _nHead;                         // DtBuffer
            floats += s * _dInner;                        // SsmX
            floats += s * _bcDim * 2;                     // SsmB, SsmC
            floats += s * _dInner;                        // SsmY
            floats += s * _dInner;                        // SsmZ
            long bytes = floats * sizeof(float);
            bytes += (long)_maxSeqLen * sizeof(int) * 2;  // TokenIdsDevice, PositionsDevice
            return bytes;
        }
    }

    public CudaNemotronHForwardState(
        int hiddenSize, int maxIntermediateSize, int vocabSize, int qElems, int kvElems,
        int inputProjectionDim, int convDim, int dConv, int dInner, int nHead, int nGroup,
        int dState, int maxSeqLen)
    {
        _hiddenSize = hiddenSize;
        _maxIntermediateSize = maxIntermediateSize;
        _vocabSize = vocabSize;
        _qElems = qElems;
        _kvElems = kvElems;
        _inputProjectionDim = inputProjectionDim;
        _convDim = convDim;
        _dConv = dConv;
        _dInner = dInner;
        _nHead = nHead;
        _bcDim = nGroup * dState;
        _maxSeqLen = maxSeqLen;

        TokenIdsDevice = AllocInts(maxSeqLen);
        PositionsDevice = AllocInts(maxSeqLen);

        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    /// <summary>Grows every seqLen-dependent buffer to at least <paramref name="seqLen"/> rows
    /// (rounded up to the next power of two), freeing and reallocating if it grows. Returns
    /// true iff a reallocation happened (callers that cache descriptor/graph state keyed on
    /// buffer identity must invalidate on true).</summary>
    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return false;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSeqBuffers();

        HiddenState = AllocFloats((long)cap * _hiddenSize);
        Residual = AllocFloats((long)cap * _hiddenSize);
        NormOutput = AllocFloats((long)cap * _hiddenSize);
        FfnIntermediate = AllocFloats((long)cap * _maxIntermediateSize);
        Logits = AllocFloats((long)cap * _vocabSize);

        QScratch = AllocFloats((long)cap * _qElems);
        KScratch = AllocFloats((long)cap * _kvElems);
        VScratch = AllocFloats((long)cap * _kvElems);
        AttnOutput = AllocFloats((long)cap * _qElems);

        Zxbcdt = AllocFloats((long)cap * _inputProjectionDim);
        ConvInput = AllocFloats((long)(_dConv - 1 + cap) * _convDim);
        XBC = AllocFloats((long)cap * _convDim);
        DtBuffer = AllocFloats((long)cap * _nHead);
        SsmX = AllocFloats((long)cap * _dInner);
        SsmB = AllocFloats((long)cap * _bcDim);
        SsmC = AllocFloats((long)cap * _bcDim);
        SsmY = AllocFloats((long)cap * _dInner);
        SsmZ = AllocFloats((long)cap * _dInner);

        _currentSeqLen = cap;
        return true;
    }

    private static nint AllocFloats(long count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(count * sizeof(float))).ThrowOnError();
        return ptr;
    }

    private static nint AllocInts(long count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(count * sizeof(int))).ThrowOnError();
        return ptr;
    }

    private void FreeSeqBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref FfnIntermediate);
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref QScratch);
        FreeIfNonZero(ref KScratch);
        FreeIfNonZero(ref VScratch);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref Zxbcdt);
        FreeIfNonZero(ref ConvInput);
        FreeIfNonZero(ref XBC);
        FreeIfNonZero(ref DtBuffer);
        FreeIfNonZero(ref SsmX);
        FreeIfNonZero(ref SsmB);
        FreeIfNonZero(ref SsmC);
        FreeIfNonZero(ref SsmY);
        FreeIfNonZero(ref SsmZ);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    public void Dispose()
    {
        FreeSeqBuffers();
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
        _currentSeqLen = 0;
    }
}
