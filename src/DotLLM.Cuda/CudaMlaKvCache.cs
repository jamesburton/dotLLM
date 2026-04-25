using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident MLA Phase A KV cache: per-layer device buffers for expanded per-head
/// <c>K_nope</c>, the shared <c>K_pe</c> (MQA-style), and per-head <c>V</c>. Mirrors
/// the CPU <c>MlaExpandedKvState</c> layout one-for-one so the MLA forward kernel
/// can write new rows into the cache and read the full history with no shape
/// translation. F32 throughout — Phase 1 keeps the entire MLA path in F32 to
/// match the CPU oracle byte-for-byte algorithmically.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a dedicated cache (not <see cref="CudaKvCache"/>)?</b> The standard
/// GQA cache assumes one head dim shared by K and V. MLA decouples them
/// (typical V2-Lite: qk=192, v=128) and adds a shared K_pe broadcast across
/// heads. Shoehorning into <see cref="CudaKvCache"/> would either bloat
/// per-head storage or leak MLA shapes through its interface. A dedicated
/// holder stays honest and leaves room for the Phase B latent cache (which
/// only stores <c>c_kv + k_pe</c>, ~7× smaller) without disturbing this
/// path.
/// </para>
/// <para>
/// <b>Per-layer layout</b>:
/// </para>
/// <list type="bullet">
///   <item><c>KNope[layer]</c>: <c>[maxSeqLen, numHeads * qkNopeHeadDim]</c> F32.</item>
///   <item><c>V[layer]</c>: <c>[maxSeqLen, numHeads * vHeadDim]</c> F32.</item>
///   <item><c>KPe[layer]</c>: <c>[maxSeqLen, qkRopeHeadDim]</c> F32 (RoPE-applied).</item>
/// </list>
/// <para>
/// <b>Lifecycle.</b> Owned by the caller. Reset on a fresh sequence
/// (<c>positions[0] == 0</c>). Each <see cref="Advance"/> moves the cached
/// length forward by <c>seqLen</c>. Single-stream / single-sequence — batching
/// or beam search would need a per-sequence instance.
/// </para>
/// </remarks>
public sealed class CudaMlaKvCache : IDisposable
{
    private readonly nint[] _kNope;
    private readonly nint[] _v;
    private readonly nint[] _kPe;
    private readonly int[] _currentLengths;

    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _numHeads;
    private readonly int _qkNopeHeadDim;
    private readonly int _vHeadDim;
    private readonly int _qkRopeHeadDim;
    // #region MLA FP16 — element size for K_nope / V / K_pe (4 = F32, 2 = F16).
    private readonly int _elementSize;
    private readonly MlaPrecision _precision;
    // #endregion

    /// <summary>Total bytes held across K_nope + V + K_pe for all layers.</summary>
    public long AllocatedBytes
    {
        get
        {
            long perTokenK = (long)_numHeads * _qkNopeHeadDim * _elementSize;
            long perTokenV = (long)_numHeads * _vHeadDim * _elementSize;
            long perTokenKpe = (long)_qkRopeHeadDim * _elementSize;
            return _numLayers * _maxSeqLen * (perTokenK + perTokenV + perTokenKpe);
        }
    }

    /// <summary>Number of transformer layers this cache holds buffers for.</summary>
    public int NumLayers => _numLayers;
    /// <summary>Maximum sequence length the cache was allocated for.</summary>
    public int MaxSeqLen => _maxSeqLen;
    /// <summary>Precision of stored cache elements (F32 or F16).</summary>
    public MlaPrecision Precision => _precision;

    /// <summary>
    /// Allocates per-layer F32 device buffers for the Phase A expanded MLA cache
    /// (back-compat constructor — original Phase A signature, F32 only).
    /// </summary>
    public CudaMlaKvCache(
        int numLayers, int maxSeqLen,
        int numHeads, int qkNopeHeadDim, int vHeadDim, int qkRopeHeadDim)
        : this(numLayers, maxSeqLen, numHeads, qkNopeHeadDim, vHeadDim, qkRopeHeadDim, MlaPrecision.F32)
    {
    }

    /// <summary>
    /// Allocates per-layer device buffers for the Phase A expanded MLA cache at
    /// the requested <paramref name="precision"/>. F16 halves the memory cost
    /// vs F32 and matches the GQA cache convention.
    /// </summary>
    public CudaMlaKvCache(
        int numLayers, int maxSeqLen,
        int numHeads, int qkNopeHeadDim, int vHeadDim, int qkRopeHeadDim,
        MlaPrecision precision)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));
        if (qkRopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkRopeHeadDim));

        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numHeads = numHeads;
        _qkNopeHeadDim = qkNopeHeadDim;
        _vHeadDim = vHeadDim;
        _qkRopeHeadDim = qkRopeHeadDim;
        _precision = precision;
        _elementSize = precision == MlaPrecision.F16 ? sizeof(ushort) : sizeof(float);

        _kNope = new nint[numLayers];
        _v = new nint[numLayers];
        _kPe = new nint[numLayers];
        _currentLengths = new int[numLayers];

        long kBytes = (long)maxSeqLen * numHeads * qkNopeHeadDim * _elementSize;
        long vBytes = (long)maxSeqLen * numHeads * vHeadDim * _elementSize;
        long kPeBytes = (long)maxSeqLen * qkRopeHeadDim * _elementSize;
        for (int i = 0; i < numLayers; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _kNope[i], (nuint)kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _v[i], (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _kPe[i], (nuint)kPeBytes).ThrowOnError();
        }
    }

    /// <summary>
    /// Resets all per-layer cached lengths to 0. Buffers are retained and overwritten
    /// on the next <see cref="Advance"/>. Call at the start of a fresh sequence
    /// (<c>positions[0] == 0</c>).
    /// </summary>
    public void Reset() => Array.Clear(_currentLengths);

    /// <summary>
    /// Returns the current cached length for the given layer.
    /// </summary>
    public int GetCurrentLength(int layerIndex) => _currentLengths[layerIndex];

    /// <summary>
    /// Advances the cached length for <paramref name="layerIndex"/> by
    /// <paramref name="tokensAdded"/>. Called by the MLA forward path after
    /// successfully writing the new K/V/K_pe rows into the cache slots
    /// <c>[currentLength..currentLength + tokensAdded)</c>.
    /// </summary>
    public void Advance(int layerIndex, int tokensAdded)
    {
        int newLen = _currentLengths[layerIndex] + tokensAdded;
        if (newLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"CudaMlaKvCache overflow on layer {layerIndex}: {newLen} > maxSeqLen={_maxSeqLen}.");
        _currentLengths[layerIndex] = newLen;
    }

    /// <summary>Device pointer to <c>K_nope[layer]</c> (F32 [maxSeqLen, numHeads*qkNopeHeadDim]).</summary>
    public nint GetKNopePtr(int layerIndex) => _kNope[layerIndex];

    /// <summary>Device pointer to <c>V[layer]</c> (F32 [maxSeqLen, numHeads*vHeadDim]).</summary>
    public nint GetVPtr(int layerIndex) => _v[layerIndex];

    /// <summary>Device pointer to shared <c>K_pe[layer]</c> (F32 [maxSeqLen, qkRopeHeadDim]).</summary>
    public nint GetKPePtr(int layerIndex) => _kPe[layerIndex];

    /// <summary>Bytes per cached K_nope row (one token, all heads).</summary>
    public long KNopeRowBytes => (long)_numHeads * _qkNopeHeadDim * _elementSize;

    /// <summary>Bytes per cached V row (one token, all heads).</summary>
    public long VRowBytes => (long)_numHeads * _vHeadDim * _elementSize;

    /// <summary>Bytes per cached K_pe row (one token, shared).</summary>
    public long KPeRowBytes => (long)_qkRopeHeadDim * _elementSize;

    /// <summary>Frees all per-layer device buffers.</summary>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_kNope[i] != 0) { CudaDriverApi.cuMemFree_v2(_kNope[i]); _kNope[i] = 0; }
            if (_v[i] != 0) { CudaDriverApi.cuMemFree_v2(_v[i]); _v[i] = 0; }
            if (_kPe[i] != 0) { CudaDriverApi.cuMemFree_v2(_kPe[i]); _kPe[i] = 0; }
        }
    }
}
