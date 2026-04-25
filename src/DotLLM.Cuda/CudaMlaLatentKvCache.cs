using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident MLA Phase B latent KV cache: per-layer device buffers for the
/// shared compressed latent <c>c_kv</c> (post-RMSNorm <c>kv_a_layernorm</c>) and
/// the shared <c>K_pe</c> (post-RoPE), with no per-head expansion. Mirrors the
/// CPU side of <see cref="DotLLM.Cpu.Kernels.MlaAttention.ExecuteLatent"/>'s
/// persistent layout: only two buffers per layer, both shared across all heads.
/// F32 throughout — Phase B keeps the full pipeline in F32 for byte-near
/// equivalence with the CPU oracle (FP16 is the next follow-up agent).
/// </summary>
/// <remarks>
/// <para>
/// <b>The Phase B win.</b> Phase A's <see cref="CudaMlaKvCache"/> stores
/// per-token, per-layer:
/// <code>
///     numHeads * qkNopeHeadDim    F32 (K_nope, e.g. 16*128 = 2048)
///   + numHeads * vHeadDim         F32 (V,      e.g. 16*128 = 2048)
///   + qkRopeHeadDim               F32 (K_pe,   e.g. 64)
///   = 4160 F32 = 16.25 KB / token / layer  (V2-Lite)
/// </code>
/// Phase B stores per-token, per-layer:
/// <code>
///     kvLoraRank                  F32 (c_kv,   e.g. 512)
///   + qkRopeHeadDim               F32 (k_pe,   e.g. 64)
///   = 576 F32 = 2.25 KB / token / layer    (V2-Lite)
/// </code>
/// On V2-Lite that is a <b>7.22× reduction</b> in cache footprint. On V2-full
/// (numHeads=128) the multiplier is even larger (~14×).
/// </para>
/// <para>
/// <b>Why a separate class (not a flag on Phase A)?</b> Phase B's per-token
/// buffer count and shape are different (2 buffers vs 3). Keeping the types
/// distinct lets each path stay honest about its on-disk layout and lets a
/// caller hold both kinds simultaneously (Phase C hybrid would expand a
/// latent cache into a Phase A view per call).
/// </para>
/// <para>
/// <b>Per-layer layout</b>:
/// </para>
/// <list type="bullet">
///   <item><c>CKv[layer]</c>: <c>[maxSeqLen, kvLoraRank]</c> F32 (post-<c>kv_a_layernorm</c>).</item>
///   <item><c>KPe[layer]</c>: <c>[maxSeqLen, qkRopeHeadDim]</c> F32 (post-RoPE).</item>
/// </list>
/// <para>
/// <b>Lifecycle.</b> Same shape as <see cref="CudaMlaKvCache"/>: caller-owned,
/// reset on a fresh sequence, <see cref="Advance"/> after a successful forward.
/// Single-stream / single-sequence — batching needs a per-sequence instance.
/// </para>
/// </remarks>
public sealed class CudaMlaLatentKvCache : IDisposable
{
    private readonly nint[] _cKv;
    private readonly nint[] _kPe;
    private readonly int[] _currentLengths;

    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _kvLoraRank;
    private readonly int _qkRopeHeadDim;

    /// <summary>Total bytes held across c_kv + k_pe for all layers.</summary>
    public long AllocatedBytes
    {
        get
        {
            long perTokenLatent = (long)_kvLoraRank * sizeof(float);
            long perTokenKpe = (long)_qkRopeHeadDim * sizeof(float);
            return _numLayers * _maxSeqLen * (perTokenLatent + perTokenKpe);
        }
    }

    /// <summary>Number of transformer layers this cache holds buffers for.</summary>
    public int NumLayers => _numLayers;
    /// <summary>Maximum sequence length the cache was allocated for.</summary>
    public int MaxSeqLen => _maxSeqLen;
    /// <summary>Per-token latent rank (= kvLoraRank).</summary>
    public int KvLoraRank => _kvLoraRank;
    /// <summary>Per-token shared rope-K dim.</summary>
    public int QkRopeHeadDim => _qkRopeHeadDim;

    /// <summary>
    /// Allocates per-layer device buffers for the Phase B latent MLA cache.
    /// </summary>
    public CudaMlaLatentKvCache(
        int numLayers, int maxSeqLen, int kvLoraRank, int qkRopeHeadDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (kvLoraRank <= 0) throw new ArgumentOutOfRangeException(nameof(kvLoraRank));
        if (qkRopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkRopeHeadDim));

        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _kvLoraRank = kvLoraRank;
        _qkRopeHeadDim = qkRopeHeadDim;

        _cKv = new nint[numLayers];
        _kPe = new nint[numLayers];
        _currentLengths = new int[numLayers];

        long latentBytes = (long)maxSeqLen * kvLoraRank * sizeof(float);
        long kPeBytes = (long)maxSeqLen * qkRopeHeadDim * sizeof(float);
        for (int i = 0; i < numLayers; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _cKv[i], (nuint)latentBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _kPe[i], (nuint)kPeBytes).ThrowOnError();
        }
    }

    /// <summary>
    /// Resets all per-layer cached lengths to 0. Buffers are retained and
    /// overwritten on the next <see cref="Advance"/>. Call at the start of a
    /// fresh sequence (<c>positions[0] == 0</c>).
    /// </summary>
    public void Reset() => Array.Clear(_currentLengths);

    /// <summary>
    /// Returns the current cached length for the given layer.
    /// </summary>
    public int GetCurrentLength(int layerIndex) => _currentLengths[layerIndex];

    /// <summary>
    /// Advances the cached length for <paramref name="layerIndex"/> by
    /// <paramref name="tokensAdded"/>. Called by the MLA Phase B forward path
    /// after successfully writing the new c_kv / K_pe rows into the cache slots
    /// <c>[currentLength..currentLength + tokensAdded)</c>.
    /// </summary>
    public void Advance(int layerIndex, int tokensAdded)
    {
        int newLen = _currentLengths[layerIndex] + tokensAdded;
        if (newLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"CudaMlaLatentKvCache overflow on layer {layerIndex}: {newLen} > maxSeqLen={_maxSeqLen}.");
        _currentLengths[layerIndex] = newLen;
    }

    /// <summary>Device pointer to <c>c_kv[layer]</c> (F32 [maxSeqLen, kvLoraRank]).</summary>
    public nint GetCKvPtr(int layerIndex) => _cKv[layerIndex];

    /// <summary>Device pointer to shared <c>K_pe[layer]</c> (F32 [maxSeqLen, qkRopeHeadDim]).</summary>
    public nint GetKPePtr(int layerIndex) => _kPe[layerIndex];

    /// <summary>Bytes per cached c_kv row (one token, shared across heads).</summary>
    public long CKvRowBytes => (long)_kvLoraRank * sizeof(float);

    /// <summary>Bytes per cached K_pe row (one token, shared across heads).</summary>
    public long KPeRowBytes => (long)_qkRopeHeadDim * sizeof(float);

    /// <summary>Frees all per-layer device buffers.</summary>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_cKv[i] != 0) { CudaDriverApi.cuMemFree_v2(_cKv[i]); _cKv[i] = 0; }
            if (_kPe[i] != 0) { CudaDriverApi.cuMemFree_v2(_kPe[i]); _kPe[i] = 0; }
        }
    }
}
