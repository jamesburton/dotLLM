using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache.Codecs;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// KV-cache backed by the data-oblivious <see cref="TurboQuantCodec"/> (arXiv:2504.19874,
/// MSE stage). Each cached K/V <b>head vector</b> (dimension <c>headDim</c>) is stored as
/// per-coordinate Lloyd–Max codes plus one norm (F16 by default); attention dequantizes the whole live
/// range into a shared fp32 scratch on demand, so the model's existing full-precision
/// attention path runs unchanged.
/// </summary>
/// <remarks>
/// <para><b>Why a sibling of <see cref="QuantizedKvCache"/>, not a codec inside it.</b>
/// TurboQuant's storage is <i>per-row</i> (<c>ceil(headDim*bits/8)</c> code bytes + a norm,
/// per head) — not the per-32-element block layout the Q8_0/Q4_0 cache assumes. Keeping it
/// separate avoids destabilising the shipping Q8_0/Q4_0 path.</para>
///
/// <para><b>Per-head, not per-row.</b> The rotation only Gaussianises a coherent vector, and
/// attention treats each KV head independently (GQA), so each of the <c>numKvHeads</c> head
/// vectors in a row is encoded separately with the same codec.</para>
///
/// <para><b>Attention integration.</b> This implements <see cref="IKvCache"/> but NOT
/// <see cref="IQuantizedKvCache"/>, so the forward pass takes the plain-fp32 branch:
/// <see cref="GetKeysRef"/>/<see cref="GetValuesRef"/> dequantize <c>[0, CurrentLength)</c>
/// into a shared scratch and return a contiguous fp32 view. (A fused dequant-in-kernel path
/// — and the optional QJL residual + mixed bit-widths — are later slices.)</para>
///
/// <para><b>Scope (slice 2).</b> Uniform geometry only (single <c>headDim</c>/<c>numKvHeads</c>),
/// pure-quantized (every token encoded immediately, no fp32 recency window). These cover the
/// dense / GQA models that would actually pick TurboQuant KV; per-layer (Gemma-4) and a recency
/// window are additive follow-ups.</para>
/// </remarks>
public sealed unsafe class TurboQuantKvCache : IKvCache, IPerLayerKvCache
{
    private readonly nint[] _keyCodes;    // [numLayers] -> [maxSeqLen * numKvHeads * codeBytes]
    private readonly nint[] _valueCodes;  // [numLayers]
    private readonly nint[] _keyNorms;    // [numLayers] -> [maxSeqLen * numKvHeads * normBytes] in _normFormat
    private readonly nint[] _valueNorms;  // [numLayers]
    private readonly nint _kScratch;      // [maxSeqLen * stride] fp32, shared across layers
    private readonly nint _vScratch;

    private readonly TurboQuantCodec _codecK;
    private readonly TurboQuantCodec _codecV;
    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _stride;         // numKvHeads * headDim
    private readonly int _codeBytes;      // per head vector
    private readonly int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;
    private readonly TqNormFormat _normFormat;
    private readonly int _normBytes;      // bytes per stored norm
    private long _normSaturations;
    private long _normUnderflows;
    private readonly float _normJitter;
    private ulong _jitterState = 0x9E3779B97F4A7C15UL;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of layers in this cache.</summary>
    public int NumLayers => _numLayers;

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _stride;

    /// <summary>Total per-coordinate bit budget (the codec's bit-width).</summary>
    public int Bits => _codecK.Bits;

    /// <summary>Whether the QJL unbiased-score residual stage is active.</summary>
    public bool UseQjl => _codecK.UseQjl;

    /// <summary>Format the per-head-vector norms are stored in (default <see cref="TurboQuantLayout.DefaultNorm"/>).</summary>
    public TqNormFormat NormFormat => _normFormat;

    /// <summary>Norms that exceeded the format's range and were clamped on store.</summary>
    public long NormSaturations => _normSaturations;

    /// <summary>Non-zero norms that decoded to zero under the storage format.</summary>
    public long NormUnderflows => _normUnderflows;

    /// <summary>Total bytes for the permanent compressed store (codes + norms; excludes scratch).</summary>
    public long AllocatedBytes =>
        (long)_numLayers * 2 * _maxSeqLen * _numKvHeads * ((long)_codeBytes + _normBytes);

    /// <summary>
    /// Creates a TurboQuant KV-cache for a uniform-geometry model.
    /// </summary>
    /// <param name="numLayers">Transformer layers.</param>
    /// <param name="numKvHeads">KV heads per layer.</param>
    /// <param name="headDim">Per-head dimension (must be a power of two — RHT rotation).</param>
    /// <param name="maxSeqLen">Max cached positions.</param>
    /// <param name="bits">Bits per coordinate (1–8; 4 ≈ quality-neutral, lower = more compression).
    /// With <paramref name="useQjl"/> this is the total budget (MSE runs at <c>bits-1</c>; requires ≥2).</param>
    /// <param name="seed">Rotation seed; persist it so rollback / prefix reuse stay valid. Keys and
    /// values use independent rotations derived from this seed.</param>
    /// <param name="useQjl">Enable the QJL 1-bit residual stage for unbiased attention scores
    /// (debiases the MSE contraction; raises ℓ2 error but removes score bias — see
    /// <see cref="TurboQuantCodec"/>).</param>
    /// <param name="normFormat">Storage format of the per-head-vector norm (default <see cref="TurboQuantLayout.DefaultNorm"/>, F16). Norms are really stored in this format and decoded through <see cref="TurboQuantNormCodec"/>, the same codec the GPU backends match. <see cref="TqNormFormat.F32Reference"/> is for oracles and A/B only.</param>
    /// <param name="normJitter">Measurement-only null control (#1892): multiplies every stored norm by <c>1 + u*normJitter</c>, u pseudo-random in [-1,1]. A perturbation of known tiny size, to measure the instrument's noise floor. Default 0 = off.</param>
    public TurboQuantKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                             int bits, ulong seed, bool useQjl = false,
                             TqNormFormat? normFormat = null,
                             float normJitter = 0f)
    {
        _normFormat = normFormat ?? TurboQuantLayout.DefaultNorm;
        _normBytes = TurboQuantNormCodec.BytesPerNorm(_normFormat);
        _normJitter = normJitter;
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;

        // Independent K/V rotations (paper-consistent): derive a distinct V seed.
        _codecK = new TurboQuantCodec(headDim, bits, seed, useQjl);
        _codecV = new TurboQuantCodec(headDim, bits, seed ^ 0xD1B54A32D192ED03UL, useQjl);
        _codeBytes = _codecK.CodeBytesPerVector;

        _keyCodes = new nint[numLayers];
        _valueCodes = new nint[numLayers];
        _keyNorms = new nint[numLayers];
        _valueNorms = new nint[numLayers];

        nuint codeBytesPerLayer = (nuint)((long)maxSeqLen * numKvHeads * _codeBytes);
        nuint normBytesPerLayer = (nuint)((long)maxSeqLen * numKvHeads * _normBytes);
        for (int i = 0; i < numLayers; i++)
        {
            _keyCodes[i] = (nint)NativeMemory.AlignedAlloc(codeBytesPerLayer, 64);
            _valueCodes[i] = (nint)NativeMemory.AlignedAlloc(codeBytesPerLayer, 64);
            _keyNorms[i] = (nint)NativeMemory.AlignedAlloc(normBytesPerLayer, 64);
            _valueNorms[i] = (nint)NativeMemory.AlignedAlloc(normBytesPerLayer, 64);
        }

        nuint scratchBytes = (nuint)((long)maxSeqLen * _stride * sizeof(float));
        _kScratch = (nint)NativeMemory.AlignedAlloc(scratchBytes, 64);
        _vScratch = (nint)NativeMemory.AlignedAlloc(scratchBytes, 64);
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~TurboQuantKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;

        byte* kCodes = (byte*)_keyCodes[layerIndex];
        byte* vCodes = (byte*)_valueCodes[layerIndex];
        byte* kNorms = (byte*)_keyNorms[layerIndex];
        byte* vNorms = (byte*)_valueNorms[layerIndex];

        int maxPos = -1;
        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;

            for (int h = 0; h < _numKvHeads; h++)
            {
                int srcOff = i * _stride + h * _headDim;
                int headIdx = pos * _numKvHeads + h;

                var kVec = new ReadOnlySpan<float>(kSrc + srcOff, _headDim);
                var vVec = new ReadOnlySpan<float>(vSrc + srcOff, _headDim);
                var kDst = new Span<byte>(kCodes + (long)headIdx * _codeBytes, _codeBytes);
                var vDst = new Span<byte>(vCodes + (long)headIdx * _codeBytes, _codeBytes);

                StoreNorm(kNorms, headIdx, _codecK.Encode(kVec, kDst));
                StoreNorm(vNorms, headIdx, _codecV.Encode(vVec, vDst));
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    // Stores one norm in the cache's format via the shared codec. normJitter is a harness-only null control
    // (#1892) that perturbs the norm before it is stored; it is 0 (off) everywhere else.
    private void StoreNorm(byte* norms, int headIdx, float norm)
    {
        if (_normJitter != 0f)
        {
            _jitterState += 0x9E3779B97F4A7C15UL;
            ulong z = _jitterState;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            float u = ((z >> 40) * (1.0f / 8388608.0f)) - 1.0f; // [-1, 1)
            norm *= 1f + u * _normJitter;
        }
        var dst = new Span<byte>(norms + (long)headIdx * _normBytes, _normBytes);
        TurboQuantNormCodec.Encode(_normFormat, new ReadOnlySpan<float>(in norm), dst, out int sat);
        _normSaturations += sat;
        if (norm > 0f && TurboQuantNormCodec.Decode(_normFormat, dst, 0) == 0f) _normUnderflows++;
    }

    /// <summary>
    /// The raw stored norm bytes of a layer for positions <c>[0, CurrentLength)</c> (all KV heads), in
    /// <see cref="NormFormat"/>. Exposed so tests can compare against the shared codec and other backends.
    /// </summary>
    public ReadOnlySpan<byte> StoredNormBytes(int layerIndex, bool values)
        => new((void*)(values ? _valueNorms[layerIndex] : _keyNorms[layerIndex]),
               checked(_currentLength * _numKvHeads * _normBytes));

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _stride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _stride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        Dequantize(_keyCodes[layerIndex], _keyNorms[layerIndex], _kScratch, _codecK);
        return new TensorRef(_currentLength, _stride, DType.Float32, -1, _kScratch);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        Dequantize(_valueCodes[layerIndex], _valueNorms[layerIndex], _vScratch, _codecV);
        return new TensorRef(_currentLength, _stride, DType.Float32, -1, _vScratch);
    }

    // Dequantizes [0, CurrentLength) head vectors from the compressed store into the shared
    // fp32 scratch (contiguous [CurrentLength, stride]).
    private void Dequantize(nint codesPtr, nint normsPtr, nint scratch, TurboQuantCodec codec)
    {
        byte* codes = (byte*)codesPtr;
        var norms = new ReadOnlySpan<byte>((void*)normsPtr, _maxSeqLen * _numKvHeads * _normBytes);
        float* dst = (float*)scratch;

        for (int pos = 0; pos < _currentLength; pos++)
        {
            for (int h = 0; h < _numKvHeads; h++)
            {
                int headIdx = pos * _numKvHeads + h;
                var src = new ReadOnlySpan<byte>(codes + (long)headIdx * _codeBytes, _codeBytes);
                var outSpan = new Span<float>(dst + pos * _stride + h * _headDim, _headDim);
                codec.Decode(src, TurboQuantNormCodec.Decode(_normFormat, norms, headIdx), outSpan);
            }
        }
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) =>
        new TensorView(new TensorShape(_currentLength, _stride), DType.Float32, -1, GetKeysRef(layerIndex).DataPointer);

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) =>
        new TensorView(new TensorShape(_currentLength, _stride), DType.Float32, -1, GetValuesRef(layerIndex).DataPointer);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;

        // Null-safe: the constructor can throw after the fields are declared but before the
        // buffer arrays are assigned (e.g. an invalid headDim rejected by the codec). The
        // finalizer then runs against a partially-constructed instance — guard every array.
        FreeAll(_keyCodes);
        FreeAll(_valueCodes);
        FreeAll(_keyNorms);
        FreeAll(_valueNorms);
        nint k = _kScratch, v = _vScratch;
        Free(ref k);
        Free(ref v);
    }

    private static void FreeAll(nint[]? buffers)
    {
        if (buffers is null) return;
        for (int i = 0; i < buffers.Length; i++)
            Free(ref buffers[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Free(ref nint p)
    {
        if (p != 0) { NativeMemory.AlignedFree((void*)p); p = 0; }
    }
}
