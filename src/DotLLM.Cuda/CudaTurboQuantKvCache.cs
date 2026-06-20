using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// CUDA-side TurboQuant (MSE-stage) KV cache — the GPU-resident CUDA analogue of the Vulkan
/// <c>VulkanTurboQuantKvCache</c> and the CPU <c>TurboQuantKvCache</c>. Stores per-layer compressed
/// codes + an fp32 norm per cached head-vector in device memory and, on attention prep, dequantizes
/// the live range into a shared <c>half</c> scratch the existing CUDA attention kernel reads.
/// </summary>
/// <remarks>
/// <para>GPU-resident encode + decode via <c>turboquant_{encode,dequant}_f16</c>: the CUDA forward's
/// K/V activations and attention scratch are FP16, so these kernels read/write <c>half</c> while the
/// codec math runs in fp32. Codec constants (centroids/signs/invSqrtD) are supplied by the caller —
/// the Cuda project does not depend on the Engine codec.</para>
/// <para>Scope: uniform geometry, pure-quantized, headDim a power of two ≤ 256, contiguous positions
/// (prefill / single-token decode). Eager decode only (no CUDA-graph capture — graph capture is gated
/// to <c>CudaKvCache</c>/<c>CudaQuantizedKvCache</c>). The shared dequant scratch is reused per layer,
/// so the permanent footprint is codes+norms, not FP16.</para>
/// </remarks>
public sealed class CudaTurboQuantKvCache : IKvCache, IPerLayerKvCache
{
    private readonly nint[] _keyCodes;     // [numLayers] uint device buffers
    private readonly nint[] _valueCodes;
    private readonly nint[] _keyNorms;     // [numLayers] fp32 device buffers
    private readonly nint[] _valueNorms;
    private readonly nint _centroids;      // fp32 device
    private readonly nint _signsK;
    private readonly nint _signsV;
    private readonly nint _kScratch;       // half device, shared across layers
    private readonly nint _vScratch;

    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _stride;
    private readonly int _maxSeqLen;
    private readonly int _mseBits;
    private readonly int _codeUintsPerVec;
    private readonly float _invSqrtD;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;
    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;
    /// <summary>Number of layers.</summary>
    public int LayerCount => _numLayers;
    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _stride;

    /// <summary>
    /// Creates a uniform-geometry CUDA TurboQuant KV cache on the current context. The caller supplies
    /// the codec constants: <paramref name="centroids"/> (length 2^mseBits, scaled by 1/√d), per-K/V
    /// RHT signs (length headDim, ±1), and <paramref name="invSqrtD"/>.
    /// </summary>
    public unsafe CudaTurboQuantKvCache(
        int numLayers, int numKvHeads, int headDim, int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > 256)
            throw new ArgumentException($"headDim must be a power of two in [1,256]; got {headDim}.", nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (mseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(mseBits));
        if (centroids.Length != (1 << mseBits)) throw new ArgumentException("centroids must have 2^mseBits entries.", nameof(centroids));
        if (signsK.Length != headDim || signsV.Length != headDim) throw new ArgumentException("signs must have headDim entries.");

        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _mseBits = mseBits;
        _invSqrtD = invSqrtD;

        int codeBytes = (headDim * mseBits + 7) / 8;
        _codeUintsPerVec = (codeBytes + 3) / 4;
        long codeUintsPerLayer = (long)maxSeqLen * numKvHeads * _codeUintsPerVec + 1; // +1 dequant guard
        nuint codeBytesPerLayer = (nuint)(codeUintsPerLayer * sizeof(uint));
        nuint normBytesPerLayer = (nuint)((long)maxSeqLen * numKvHeads * sizeof(float));
        nuint scratchBytes = (nuint)((long)maxSeqLen * _stride * sizeof(ushort)); // half

        _keyCodes = new nint[numLayers];
        _valueCodes = new nint[numLayers];
        _keyNorms = new nint[numLayers];
        _valueNorms = new nint[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _keyCodes[i], codeBytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valueCodes[i], codeBytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _keyNorms[i], normBytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valueNorms[i], normBytesPerLayer).ThrowOnError();
            // Zero codes so the dequant guard-uint read is well-defined.
            CudaDriverApi.cuMemsetD8_v2(_keyCodes[i], 0, codeBytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(_valueCodes[i], 0, codeBytesPerLayer).ThrowOnError();
        }
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, scratchBytes).ThrowOnError();

        _centroids = UploadFloats(centroids);
        _signsK = UploadFloats(signsK);
        _signsV = UploadFloats(signsV);
    }

    private static unsafe nint UploadFloats(ReadOnlySpan<float> src)
    {
        nuint bytes = (nuint)(src.Length * sizeof(float));
        CudaDriverApi.cuMemAlloc_v2(out nint dev, bytes).ThrowOnError();
        fixed (float* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, bytes).ThrowOnError();
        return dev;
    }

    /// <summary>Encodes fresh device K/V (FP16) for one layer into the layer's code/norm store. Positions
    /// must be contiguous ascending. Advances CurrentLength.</summary>
    public void UpdateDevice(nint keysDevice, nint valuesDevice, ReadOnlySpan<int> positions, int seqLen,
                             int layerIndex, nint stream, CudaKernels kernels)
    {
        int startPos = ValidateContiguous(positions, seqLen);
        int levelCount = 1 << _mseBits;
        kernels.LaunchTurboQuantEncodeF16(keysDevice, _centroids, _signsK, _keyCodes[layerIndex], _keyNorms[layerIndex],
            seqLen, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, levelCount, startPos, _invSqrtD, stream);
        kernels.LaunchTurboQuantEncodeF16(valuesDevice, _centroids, _signsV, _valueCodes[layerIndex], _valueNorms[layerIndex],
            seqLen, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, levelCount, startPos, _invSqrtD, stream);
        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <summary>Dequantizes <c>[0, CurrentLength)</c> for the layer into the shared FP16 scratch and
    /// returns the (keys, values) device pointers the attention kernel reads.</summary>
    public (nint kPtr, nint vPtr) PrepareAttentionScratch(int layerIndex, nint stream, CudaKernels kernels)
    {
        int numVectors = _currentLength * _numKvHeads;
        kernels.LaunchTurboQuantDequantF16(_keyCodes[layerIndex], _keyNorms[layerIndex], _centroids, _signsK, _kScratch,
            numVectors, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, _invSqrtD, stream);
        kernels.LaunchTurboQuantDequantF16(_valueCodes[layerIndex], _valueNorms[layerIndex], _centroids, _signsV, _vScratch,
            numVectors, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, _invSqrtD, stream);
        return (_kScratch, _vScratch);
    }

    private int ValidateContiguous(ReadOnlySpan<int> positions, int seqLen)
    {
        if (positions.Length != seqLen) throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        int start = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] != start + i)
                throw new NotSupportedException("CudaTurboQuantKvCache requires contiguous ascending positions.");
            if ((uint)positions[i] >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions));
        }
        return start;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength) throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length (new sequence).</summary>
    public void Reset() => _currentLength = 0;

    // Host-side IKvCache members are unused on the CUDA device path (mirrors CudaKvCache).
    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("CudaTurboQuantKvCache is updated via UpdateDevice from the CUDA forward pass.");
    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("CudaTurboQuantKvCache is updated via UpdateDevice from the CUDA forward pass.");
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException("CUDA cache exposes device pointers via PrepareAttentionScratch.");
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException("CUDA cache exposes device pointers via PrepareAttentionScratch.");
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("CUDA cache does not materialise cached keys as CPU tensors.");
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => throw new NotSupportedException("CUDA cache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            Free(_keyCodes[i]); Free(_valueCodes[i]); Free(_keyNorms[i]); Free(_valueNorms[i]);
        }
        Free(_kScratch); Free(_vScratch); Free(_centroids); Free(_signsK); Free(_signsV);
    }

    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
}
