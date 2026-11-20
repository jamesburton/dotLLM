using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache.Codecs;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// TurboQuant KV cache that is read <b>in place</b> (#1892): each K / V head vector is stored as packed Lloyd-Max codes plus one norm
/// (<see cref="TurboQuantLayout"/>), and attention decodes <c>norm * centroid[code]</c> one row at a time in the kernel. There is no
/// decoded history, no scratch, and no original-space window (<c>--cache-window &gt; 0</c> with TurboQuant is refused).
/// The scratch-path <see cref="TurboQuantKvCache"/> is the oracle: same seed, same norm format, same codes.
/// </summary>
/// <remarks>
/// Code layout per layer and side: <c>[position, kvHead, CodeUintsPerVec]</c> 32-bit words, each vector padded to a whole word
/// (not tight across vectors) with one trailing guard word. Norms: <c>[position, kvHead]</c> in the layout's norm format.
/// K and V use independent rotations (V seed = <c>seed ^ 0xD1B54A32D192ED03</c>), identical to <see cref="TurboQuantKvCache"/>.
/// </remarks>
public sealed unsafe class TurboQuantInPlaceKvCache : ITurboQuantKvCache
{
    private readonly nint[] _keyCodes;
    private readonly nint[] _valueCodes;
    private readonly nint[] _keyNorms;
    private readonly nint[] _valueNorms;
    private readonly TurboQuantCodec _codecK;
    private readonly TurboQuantCodec _codecV;
    private readonly sbyte[] _signsK;
    private readonly sbyte[] _signsV;
    private readonly TurboQuantLayout _layout;
    private readonly int _numLayers;
    private readonly int _stride;
    private readonly int _codeBytes;     // meaningful code bytes per vector (<= CodeUintsPerVec * 4)
    private readonly int _vecBytes;      // CodeUintsPerVec * 4
    private readonly int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;
    private long _normSaturations;

    /// <inheritdoc/>
    public TurboQuantLayout Layout => _layout;

    /// <inheritdoc/>
    public ReadOnlySpan<float> Centroids => _codecK.Centroids;

    /// <inheritdoc/>
    public ReadOnlySpan<sbyte> SignsK => _signsK;

    /// <inheritdoc/>
    public ReadOnlySpan<sbyte> SignsV => _signsV;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int QuantizedLength => _currentLength;

    /// <inheritdoc/>
    public int WindowLength => 0;

    /// <inheritdoc/>
    public int WindowCapacity => 0;

    /// <inheritdoc/>
    public KvCacheDType KeyDType => KvCacheDType.TurboQuant;

    /// <inheritdoc/>
    public KvCacheDType ValueDType => KvCacheDType.TurboQuant;

    /// <inheritdoc/>
    public int KeyQuantizedRowBytes => _layout.NumKvHeads * _vecBytes;

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _layout.NumKvHeads * _vecBytes;

    /// <summary>Norms that exceeded the format's range and were clamped on store.</summary>
    public long NormSaturations => _normSaturations;

    /// <summary>Total bytes of the permanent store (codes + guard words + norms, K and V, all layers); there is no scratch.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * 2 * (_layout.CodeBytesPerLayer(_maxSeqLen) + _layout.NormBytesPerLayer(_maxSeqLen));

    /// <summary>Creates an in-place TurboQuant cache for a uniform-geometry model.</summary>
    /// <param name="numLayers">Transformer layers.</param>
    /// <param name="numKvHeads">KV heads per layer.</param>
    /// <param name="headDim">Head dimension (power of two, at most 256).</param>
    /// <param name="maxSeqLen">Max cached positions.</param>
    /// <param name="bits">Bits per coordinate (1-8).</param>
    /// <param name="seed">Rotation seed (K); V derives <c>seed ^ 0xD1B54A32D192ED03</c>, as in <see cref="TurboQuantKvCache"/>.</param>
    /// <param name="normFormat">Norm storage format (default <see cref="TurboQuantLayout.DefaultNorm"/>).</param>
    /// <param name="window">Must be 0: there is no original-space window with TurboQuant.</param>
    public TurboQuantInPlaceKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen, int bits, ulong seed,
                                    TqNormFormat? normFormat = null, int window = 0)
    {
        if (window > 0)
            throw new ArgumentException("TurboQuant has no original-space window; a cache window > 0 is not supported.", nameof(window));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (headDim > 256) throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "in-place TurboQuant supports head dim <= 256.");

        _layout = new TurboQuantLayout(headDim, numKvHeads, bits, normFormat ?? TurboQuantLayout.DefaultNorm);
        _numLayers = numLayers;
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _codecK = new TurboQuantCodec(headDim, bits, seed);
        _codecV = new TurboQuantCodec(headDim, bits, seed ^ 0xD1B54A32D192ED03UL);
        _signsK = Cpu.Kernels.TurboQuantRotation.SignsToSbyte(_codecK.RotationSigns);
        _signsV = Cpu.Kernels.TurboQuantRotation.SignsToSbyte(_codecV.RotationSigns);
        _codeBytes = _codecK.CodeBytesPerVector;
        _vecBytes = _layout.CodeUintsPerVec * 4;

        _keyCodes = new nint[numLayers];
        _valueCodes = new nint[numLayers];
        _keyNorms = new nint[numLayers];
        _valueNorms = new nint[numLayers];
        nuint codeBytes = (nuint)_layout.CodeBytesPerLayer(maxSeqLen);
        nuint normBytes = (nuint)Math.Max(_layout.NormBytesPerLayer(maxSeqLen), 1);
        for (int i = 0; i < numLayers; i++)
        {
            _keyCodes[i] = AllocZeroed(codeBytes);
            _valueCodes[i] = AllocZeroed(codeBytes);
            _keyNorms[i] = AllocZeroed(normBytes);
            _valueNorms[i] = AllocZeroed(normBytes);
        }
    }

    private static nint AllocZeroed(nuint bytes)
    {
        void* p = NativeMemory.AlignedAlloc(bytes, 64);
        NativeMemory.Clear(p, bytes);
        return (nint)p;
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~TurboQuantInPlaceKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    public nint GetQuantizedKeysPtr(int layerIndex) => _keyCodes[layerIndex];

    /// <inheritdoc/>
    public nint GetQuantizedValuesPtr(int layerIndex) => _valueCodes[layerIndex];

    /// <inheritdoc/>
    public nint GetKeyNormsPtr(int layer) => _keyNorms[layer];

    /// <inheritdoc/>
    public nint GetValueNormsPtr(int layer) => _valueNorms[layer];

    /// <inheritdoc/>
    public nint GetWindowKeysPtr(int layerIndex) => 0;

    /// <inheritdoc/>
    public nint GetWindowValuesPtr(int layerIndex) => 0;

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int headDim = _layout.HeadDim, numKvHeads = _layout.NumKvHeads;
        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;
        byte* kCodes = (byte*)_keyCodes[layerIndex];
        byte* vCodes = (byte*)_valueCodes[layerIndex];
        byte* kNorms = (byte*)_keyNorms[layerIndex];
        byte* vNorms = (byte*)_valueNorms[layerIndex];
        int normBytes = _layout.NormBytes;

        int maxPos = -1;
        for (int i = 0; i < positions.Length; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;

            for (int h = 0; h < numKvHeads; h++)
            {
                int srcOff = i * _stride + h * headDim;
                long vec = (long)pos * numKvHeads + h;
                EncodeOne(_codecK, new ReadOnlySpan<float>(kSrc + srcOff, headDim), kCodes + vec * _vecBytes, kNorms + vec * normBytes);
                EncodeOne(_codecV, new ReadOnlySpan<float>(vSrc + srcOff, headDim), vCodes + vec * _vecBytes, vNorms + vec * normBytes);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    private void EncodeOne(TurboQuantCodec codec, ReadOnlySpan<float> vec, byte* codesDst, byte* normDst)
    {
        var dst = new Span<byte>(codesDst, _vecBytes);
        dst.Clear();                                  // the vector's padding bytes stay zero
        float norm = codec.Encode(vec, dst[.._codeBytes]);
        TurboQuantNormCodec.Encode(_layout.Norm, new ReadOnlySpan<float>(in norm), new Span<byte>(normDst, _layout.NormBytes), out int sat);
        _normSaturations += sat;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _stride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _stride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    private const string NoDecodedView =
        "The in-place TurboQuant cache keeps no decoded history; attention reads the codes directly.";

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException(NoDecodedView);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException(NoDecodedView);

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException(NoDecodedView);

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => throw new NotSupportedException(NoDecodedView);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength) throw new ArgumentOutOfRangeException(nameof(length));
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
        FreeAll(_keyCodes);
        FreeAll(_valueCodes);
        FreeAll(_keyNorms);
        FreeAll(_valueNorms);
    }

    private static void FreeAll(nint[]? buffers)
    {
        if (buffers is null) return;
        for (int i = 0; i < buffers.Length; i++) Free(ref buffers[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Free(ref nint p)
    {
        if (p != 0) { NativeMemory.AlignedFree((void*)p); p = 0; }
    }
}
