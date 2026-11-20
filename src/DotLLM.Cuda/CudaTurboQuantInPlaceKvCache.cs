using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// CUDA TurboQuant KV cache that is meant to be read <b>in place</b> (#1892, step 2): each K / V head vector is stored as packed
/// Lloyd-Max codes plus one norm in the <see cref="TurboQuantLayout"/> geometry shared with the CPU
/// <c>TurboQuantInPlaceKvCache</c> and the Vulkan cache. There is no decoded history, no scratch and no original-space window
/// (<c>--cache-window &gt; 0</c> with TurboQuant is refused). The scratch-dequant <see cref="CudaTurboQuantKvCache"/> stays as the oracle.
/// </summary>
/// <remarks>
/// <para><b>Buffers (for the attention kernels that bind them).</b> Per layer and side, one device buffer of codes
/// (<see cref="TurboQuantLayout.CodeBytesPerLayer"/>: <c>[position, kvHead, CodeUintsPerVec]</c> 32-bit words plus a trailing zero guard
/// word) and one of norms (<c>[position, kvHead]</c> in <see cref="NormFormat"/>; its size is rounded up to a whole 4-byte word so a
/// word-wide load of the last norm stays in bounds). Shared: the Lloyd-Max <see cref="CentroidsPtr"/> (F32, <c>1&lt;&lt;bits</c>,
/// scaled by <c>1/sqrt(headDim)</c>) and the K / V rotation sign vectors <see cref="SignsKPtr"/> / <see cref="SignsVPtr"/> (F32, +-1,
/// length headDim, independent seeds).</para>
/// <para><b>Pointer lifetime.</b> Every buffer is allocated once at construction for <c>maxSeqLen</c> rows and never grown or
/// reallocated: a pointer read from this object stays valid until <see cref="Dispose"/>. Rows beyond <see cref="CurrentLength"/> hold
/// zero codes and zero norms until encoded; <see cref="Rollback"/> only lowers the visible length (stale rows are overwritten by the
/// next <see cref="UpdateDevice"/>).</para>
/// <para><b>Attention.</b> The dense CUDA forward reads these buffers in place through <see cref="CudaTurboQuantInPlaceAttention"/>
/// (#1892 Task 6: encode, rotate Q with the K signs, <c>attention_kvtq</c> or the tensor-core <c>_tqtq</c> prefill, rotate O with
/// the V signs). Decode is eager: graph capture never engages for this type. The host <see cref="IKvCache"/> members throw.</para>
/// </remarks>
public sealed class CudaTurboQuantInPlaceKvCache : IKvCache, IPerLayerKvCache
{
    private readonly nint[] _keyCodes;
    private readonly nint[] _valueCodes;
    private readonly nint[] _keyNorms;
    private readonly nint[] _valueNorms;
    private nint _centroids;
    private nint _signsK;
    private nint _signsV;
    private readonly CudaContext? _context;

    private readonly TurboQuantLayout _layout;
    private readonly int _numLayers;
    private readonly int _stride;
    private readonly int _maxSeqLen;
    private readonly float _invSqrtD;
    private readonly long _allocatedBytes;
    private int _currentLength;
    private bool _disposed;

    /// <summary>The shared storage geometry (head dim, kv heads, MSE bits, norm format).</summary>
    public TurboQuantLayout Layout => _layout;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of layers.</summary>
    public int LayerCount => _numLayers;

    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _stride;

    /// <summary>Storage format of the per-head-vector norms.</summary>
    public TqNormFormat NormFormat => _layout.Norm;

    /// <summary>Uints per head-vector in the code store.</summary>
    public int CodeUintsPerVec => _layout.CodeUintsPerVec;

    /// <summary>Device pointer to the F32 Lloyd-Max centroids (<c>1 &lt;&lt; Layout.MseBits</c> entries, scaled by 1/sqrt(headDim)).</summary>
    public nint CentroidsPtr => _centroids;

    /// <summary>Device pointer to the K-side rotation signs (F32, +-1, length headDim): rotates the query into code space.</summary>
    public nint SignsKPtr => _signsK;

    /// <summary>Device pointer to the V-side rotation signs (F32, +-1, length headDim): rotates the attention output back.</summary>
    public nint SignsVPtr => _signsV;

    /// <summary>The 1/sqrt(headDim) the encode and rotation kernels fold in.</summary>
    public float InvSqrtD => _invSqrtD;

    /// <summary>Exact device bytes this cache allocated (codes + guard words + norms, K and V, all layers, plus centroids and signs).
    /// There is no scratch.</summary>
    public long AllocatedBytes => _allocatedBytes;

    /// <summary>Device pointer to the key code buffer of <paramref name="layerIndex"/> (see the class remarks for the layout).</summary>
    public nint GetKeyCodesPtr(int layerIndex) => _keyCodes[layerIndex];

    /// <summary>Device pointer to the value code buffer of <paramref name="layerIndex"/>.</summary>
    public nint GetValueCodesPtr(int layerIndex) => _valueCodes[layerIndex];

    /// <summary>Device pointer to the key norm buffer of <paramref name="layerIndex"/> (<c>[position, kvHead]</c>, <see cref="NormFormat"/>).</summary>
    public nint GetKeyNormsPtr(int layerIndex) => _keyNorms[layerIndex];

    /// <summary>Device pointer to the value norm buffer of <paramref name="layerIndex"/>.</summary>
    public nint GetValueNormsPtr(int layerIndex) => _valueNorms[layerIndex];

    private static long NormAllocBytes(TurboQuantLayout layout, int rows) => (layout.NormBytesPerLayer(rows) + 3) / 4 * 4;

    /// <summary>
    /// Exact device bytes a cache of this shape allocates: <c>numLayers * 2 * (CodeBytesPerLayer + norm bytes rounded to 4)</c>
    /// plus the centroids and both sign vectors. This is what the constructor admits against free VRAM.
    /// </summary>
    public static long AdmissionBytes(TurboQuantLayout layout, int numLayers, int maxSeqLen)
        => (long)numLayers * 2 * (layout.CodeBytesPerLayer(maxSeqLen) + NormAllocBytes(layout, maxSeqLen))
           + (long)sizeof(float) * ((1L << layout.MseBits) + 2L * layout.HeadDim);

    /// <summary>
    /// Steady-state device bytes per cached token (all layers, K and V): <c>numLayers * 2 * NumKvHeads * (CodeUintsPerVec * 4 + NormBytes)</c>.
    /// Excludes the one-off guard words, centroids and signs.
    /// </summary>
    public static long BytesPerToken(TurboQuantLayout layout, int numLayers)
        => (long)numLayers * 2 * layout.NumKvHeads * (layout.CodeUintsPerVec * 4L + layout.NormBytes);

    /// <summary>
    /// Creates the cache on the current context. The caller supplies the codec constants: <paramref name="centroids"/> (length
    /// 2^mseBits, scaled by 1/sqrt(d)), the K and V rotation signs (length headDim, +-1, independent seeds), and <paramref name="invSqrtD"/>.
    /// Admitted against free VRAM with the exact byte count before anything is allocated.
    /// </summary>
    /// <param name="numLayers">Transformer layers.</param>
    /// <param name="numKvHeads">KV heads per layer.</param>
    /// <param name="headDim">Head dimension (power of two, at most 256).</param>
    /// <param name="maxSeqLen">Rows allocated per layer; never grown.</param>
    /// <param name="mseBits">Bits per coordinate (1-8).</param>
    /// <param name="centroids">Lloyd-Max centroids.</param>
    /// <param name="signsK">K-side rotation signs.</param>
    /// <param name="signsV">V-side rotation signs (must differ from the K signs).</param>
    /// <param name="invSqrtD">1/sqrt(headDim).</param>
    /// <param name="normFormat">Norm storage format; default <see cref="TurboQuantLayout.DefaultNorm"/>. F32Reference and Bf16 exist for oracle A/B only.</param>
    /// <param name="window">Must be 0: there is no original-space window with TurboQuant.</param>
    /// <param name="context">The owning context, made current for the allocations and for <see cref="Dispose"/> (it may run on another thread).</param>
    /// <exception cref="ArgumentException"><paramref name="window"/> is above 0, the head dim is unsupported, or the signs are equal.</exception>
    /// <exception cref="DotLLM.Core.Configuration.DeviceMemoryAdmissionException">The cache does not fit in free VRAM.</exception>
    public unsafe CudaTurboQuantInPlaceKvCache(
        int numLayers, int numKvHeads, int headDim, int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD,
        TqNormFormat? normFormat = null, int window = 0, CudaContext? context = null)
    {
        if (window > 0)
            throw new ArgumentException("TurboQuant has no original-space window; a cache window > 0 is not supported (#1892).", nameof(window));
        TqNormFormat fmt = normFormat ?? TurboQuantLayout.DefaultNorm;
        if (fmt is not (TqNormFormat.F32Reference or TqNormFormat.F16 or TqNormFormat.Bf16))
            throw new NotSupportedException($"CUDA TurboQuant KV cache supports F32Reference, F16 and Bf16 norms; got {fmt}.");
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > 256)
            throw new ArgumentException($"headDim must be a power of two in [1,256]; got {headDim}.", nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (mseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(mseBits));
        if (centroids.Length != (1 << mseBits)) throw new ArgumentException("centroids must have 2^mseBits entries.", nameof(centroids));
        if (signsK.Length != headDim || signsV.Length != headDim) throw new ArgumentException("signs must have headDim entries.");
        if (signsK.SequenceEqual(signsV))
            throw new ArgumentException("K and V rotation signs must come from different seeds.", nameof(signsV));

        _layout = new TurboQuantLayout(headDim, numKvHeads, mseBits, fmt);
        _numLayers = numLayers;
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _invSqrtD = invSqrtD;
        _context = context;
        _allocatedBytes = AdmissionBytes(_layout, numLayers, maxSeqLen);

        // #1848: admitted against free VRAM before anything is allocated (WDDM spills an over-commit into shared memory).
        CudaAdmission.EnsureFits($"KV cache (in-place tq{mseBits}) for {maxSeqLen} tokens", _allocatedBytes);
        context?.MakeCurrent();

        nuint codeBytes = (nuint)_layout.CodeBytesPerLayer(maxSeqLen);
        nuint normBytes = (nuint)NormAllocBytes(_layout, maxSeqLen);
        _keyCodes = new nint[numLayers];
        _valueCodes = new nint[numLayers];
        _keyNorms = new nint[numLayers];
        _valueNorms = new nint[numLayers];
        try
        {
            for (int i = 0; i < numLayers; i++)
            {
                _keyCodes[i] = AllocZeroed(codeBytes);
                _valueCodes[i] = AllocZeroed(codeBytes);
                _keyNorms[i] = AllocZeroed(normBytes);
                _valueNorms[i] = AllocZeroed(normBytes);
            }
            _centroids = UploadFloats(centroids);
            _signsK = UploadFloats(signsK);
            _signsV = UploadFloats(signsV);
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    private static nint AllocZeroed(nuint bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, bytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(p, 0, bytes).ThrowOnError();
        return p;
    }

    private static unsafe nint UploadFloats(ReadOnlySpan<float> src)
    {
        nuint bytes = (nuint)(src.Length * sizeof(float));
        CudaDriverApi.cuMemAlloc_v2(out nint dev, bytes).ThrowOnError();
        fixed (float* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, bytes).ThrowOnError();
        return dev;
    }

    /// <summary>
    /// Encodes fresh device K/V (FP16, <c>[seqLen, numKvHeads * headDim]</c>) for one layer into the layer's code and norm stores
    /// with the byte-exact F16 encode kernel; norms are written in <see cref="NormFormat"/>. Positions must be contiguous ascending.
    /// Advances <see cref="CurrentLength"/>.
    /// </summary>
    public void UpdateDevice(nint keysDevice, nint valuesDevice, ReadOnlySpan<int> positions, int seqLen,
                             int layerIndex, nint stream, CudaKernels kernels)
    {
        int startPos = ValidateContiguous(positions, seqLen);
        int levelCount = 1 << _layout.MseBits;
        kernels.LaunchTurboQuantEncodeF16(keysDevice, _centroids, _signsK, _keyCodes[layerIndex], _keyNorms[layerIndex],
            seqLen, _layout.HeadDim, _layout.NumKvHeads, _layout.MseBits, _layout.CodeUintsPerVec, levelCount, startPos, _invSqrtD, stream, _layout.Norm);
        kernels.LaunchTurboQuantEncodeF16(valuesDevice, _centroids, _signsV, _valueCodes[layerIndex], _valueNorms[layerIndex],
            seqLen, _layout.HeadDim, _layout.NumKvHeads, _layout.MseBits, _layout.CodeUintsPerVec, levelCount, startPos, _invSqrtD, stream, _layout.Norm);
        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <summary>Downloads the raw stored code words of a layer (test helper). Each head-vector occupies
    /// <see cref="CodeUintsPerVec"/> uints (zero-padded). The caller synchronises the stream first.</summary>
    public unsafe void DownloadCodeUints(int layerIndex, bool values, Span<uint> destination)
    {
        _context?.MakeCurrent();
        fixed (uint* p = destination)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, (values ? _valueCodes : _keyCodes)[layerIndex], (nuint)(destination.Length * sizeof(uint))).ThrowOnError();
    }

    /// <summary>Downloads the raw stored norm bytes of a layer, in <see cref="NormFormat"/> (test helper).</summary>
    public unsafe void DownloadNormBytes(int layerIndex, bool values, Span<byte> destination)
    {
        _context?.MakeCurrent();
        fixed (byte* p = destination)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, (values ? _valueNorms : _keyNorms)[layerIndex], (nuint)destination.Length).ThrowOnError();
    }

    /// <summary>Downloads an F32 constant buffer this cache owns (<see cref="CentroidsPtr"/>, <see cref="SignsKPtr"/>, <see cref="SignsVPtr"/>); test helper.</summary>
    public unsafe void DownloadFloats(nint devicePtr, Span<float> destination)
    {
        if (devicePtr != _centroids && devicePtr != _signsK && devicePtr != _signsV)
            throw new ArgumentException("devicePtr is not one of this cache's constant buffers.", nameof(devicePtr));
        _context?.MakeCurrent();
        fixed (float* p = destination)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(destination.Length * sizeof(float))).ThrowOnError();
    }

    private int ValidateContiguous(ReadOnlySpan<int> positions, int seqLen)
    {
        if (positions.Length != seqLen) throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        int start = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] != start + i)
                throw new NotSupportedException("CudaTurboQuantInPlaceKvCache requires contiguous ascending positions.");
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

    private const string DeviceOnly = "CudaTurboQuantInPlaceKvCache is updated via UpdateDevice from the CUDA forward pass.";
    private const string NoDecodedView =
        "The in-place TurboQuant cache keeps no decoded history; the CUDA attention kernels read the codes directly (CudaTurboQuantInPlaceAttention, #1892).";

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException(DeviceOnly);
    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException(DeviceOnly);
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException(NoDecodedView);
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException(NoDecodedView);
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException(NoDecodedView);
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => throw new NotSupportedException(NoDecodedView);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _context?.MakeCurrent();
        FreeAll();
    }

    private void FreeAll()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            Free(ref _keyCodes[i]); Free(ref _valueCodes[i]); Free(ref _keyNorms[i]); Free(ref _valueNorms[i]);
        }
        Free(ref _centroids); Free(ref _signsK); Free(ref _signsV);
    }

    private static void Free(ref nint p)
    {
        if (p != 0) { CudaDriverApi.cuMemFree_v2(p); p = 0; }
    }
}
