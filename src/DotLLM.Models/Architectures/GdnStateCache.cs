using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence recurrent state cache for the Gated DeltaNet (GDN) layers of a
/// Qwen3MoeHybrid model. One cache instance covers all GDN layers for a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// For each GDN layer the cache stores two buffers:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <c>conv_state</c> — the rolling K history for causal conv1d, shape
///       <c>[(DConv−1) × NKHead × DState]</c> row-major.
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>gdn_state</c> — the full associative-memory matrix state, shape
///       <c>[NVHead × DState × DState]</c> row-major. This is substantially
///       larger than a Mamba-2 vector state: with NVHead=32, DState=64,
///       each layer holds 32 × 64 × 64 × 4 B = 512 KB.
///     </description>
///   </item>
/// </list>
/// <para>
/// Buffers are unmanaged, 64-byte aligned (AVX-512 friendly), zero-initialised
/// on creation. Forward passes obtain <see cref="Span{T}"/> slices via
/// <see cref="GetConvState"/> and <see cref="GetGdnState"/> and mutate them in place.
/// </para>
/// </remarks>
public sealed unsafe class GdnStateCache : IGdnState
{
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int _numGdnLayers;
    private readonly int _convStateElements;
    private readonly int _gdnStateElements;

    // Contiguous per-layer blocks. GDN layer ordinal i occupies:
    //   conv:  _convState[i*_convStateElements .. (i+1)*_convStateElements)
    //   state: _gdnState [i*_gdnStateElements  .. (i+1)*_gdnStateElements)
    private nint _convState;
    private nint _gdnState;

    private bool _disposed;

    // Lazy ("fused") copy: while _pendingSource is set, layers with _pending[i] logically hold the SOURCE cache's layer i; the live
    // buffers of those layers are stale. The next update of such a layer reads the source for its first token instead
    // (BeginUpdate), so a checkpoint / restore of the 100+ MiB recurrent state costs no separate copy pass (#840).
    private GdnStateCache? _pendingSource;
    private bool[]? _pending;
    private int _pendingCount;

    /// <summary>Number of GDN layers covered by this cache.</summary>
    public int NumGdnLayers => _numGdnLayers;

    /// <summary>
    /// Elements per layer in the conv rolling-K buffer:
    /// <c>(DConv−1) × NKHead × DState</c>.
    /// </summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>
    /// Elements per layer in the GDN matrix state:
    /// <c>NVHead × DState × DState</c>.
    /// </summary>
    public int GdnStateElements => _gdnStateElements;

    /// <summary>
    /// Creates a new GDN state cache for the given config and layer count.
    /// All buffers are zero-initialised (zero state = no prior history).
    /// </summary>
    /// <param name="gdn">GDN hyperparameters shared by all GDN layers.</param>
    /// <param name="numGdnLayers">
    /// Count of GDN layers (blocks whose <c>HybridLayerKind</c> is
    /// <see cref="DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet"/>).
    /// </param>
    public GdnStateCache(GatedDeltaNetConfig gdn, int numGdnLayers)
    {
        if (numGdnLayers < 0) throw new ArgumentOutOfRangeException(nameof(numGdnLayers));

        _gdn = gdn;
        _numGdnLayers = numGdnLayers;
        _convStateElements = gdn.ConvStateElements; // (DConv-1) * NKHead * DState
        _gdnStateElements = gdn.StateElements;      // NVHead * DState * DState

        if (numGdnLayers == 0)
        {
            _convState = 0;
            _gdnState = 0;
            return;
        }

        long convBytes = (long)_numGdnLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numGdnLayers * _gdnStateElements * sizeof(float);

        _convState = (nint)NativeMemory.AlignedAlloc((nuint)convBytes, 64);
        _gdnState = (nint)NativeMemory.AlignedAlloc((nuint)stateBytes, 64);

        // GDN starts with zero state (empty associative memory, no K history).
        NativeMemory.Clear((void*)_convState, (nuint)convBytes);
        NativeMemory.Clear((void*)_gdnState, (nuint)stateBytes);
    }

    /// <summary>
    /// Returns the conv rolling-K state slice for GDN layer ordinal
    /// <paramref name="gdnLayerIndex"/>. Indexed by GDN-layer ordinal, not
    /// by absolute block index.
    /// </summary>
    public Span<float> GetConvState(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        if (_pendingCount != 0) MaterializeLayer(gdnLayerIndex);
        return new Span<float>(
            (float*)_convState + (long)gdnLayerIndex * _convStateElements,
            _convStateElements);
    }

    /// <summary>
    /// Returns the GDN matrix state slice for GDN layer ordinal
    /// <paramref name="gdnLayerIndex"/>. Shape: <c>[NVHead, DState, DState]</c> row-major.
    /// Indexed by GDN-layer ordinal, not by absolute block index.
    /// </summary>
    public Span<float> GetGdnState(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        if (_pendingCount != 0) MaterializeLayer(gdnLayerIndex);
        return new Span<float>(
            (float*)_gdnState + (long)gdnLayerIndex * _gdnStateElements,
            _gdnStateElements);
    }

    /// <summary>
    /// Zeroes every layer's state. Call between independent sequences.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        ClearPending();
        if (_numGdnLayers == 0) return;
        NativeMemory.Clear((void*)_convState, (nuint)((long)_numGdnLayers * _convStateElements * sizeof(float)));
        NativeMemory.Clear((void*)_gdnState, (nuint)((long)_numGdnLayers * _gdnStateElements * sizeof(float)));
    }

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numGdnLayers * (_convStateElements + _gdnStateElements) * sizeof(float);

    /// <summary>
    /// Deep-copies this cache's current contents into a freshly-allocated
    /// <see cref="GdnStateCache"/> of the same shape — a checkpoint speculative decoding can
    /// later restore via <see cref="CopyTo"/> (issue #287: rejected draft tokens' recurrent-state
    /// contribution has no rollback today; this is the primitive the fix is built on).
    /// </summary>
    public GdnStateCache Clone()
    {
        ThrowIfDisposed();
        var clone = new GdnStateCache(_gdn, _numGdnLayers);
        CopyTo(clone);
        return clone;
    }

    /// <summary>
    /// Overwrites <paramref name="destination"/>'s buffers with this cache's current contents.
    /// Both caches must share the same layer count / per-layer element counts (true for any pair
    /// obtained from the same model instance's <see cref="Qwen3HybridDenseTransformerModel.CreateSequenceState"/>
    /// or <see cref="Clone"/>).
    /// </summary>
    public void CopyTo(GdnStateCache destination)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(destination);
        destination.ThrowIfDisposed();
        if (destination._numGdnLayers != _numGdnLayers
            || destination._convStateElements != _convStateElements
            || destination._gdnStateElements != _gdnStateElements)
        {
            throw new ArgumentException(
                "Destination GdnStateCache shape does not match this cache's shape.", nameof(destination));
        }

        if (_numGdnLayers == 0) return;
        if (ReferenceEquals(destination, this)) return;
        destination.ClearPending();   // the destination is fully overwritten below

        if (_pendingCount != 0)
        {
            // This cache is lazily pointing at a source: the logical content of a pending layer is the source's layer.
            for (int i = 0; i < _numGdnLayers; i++)
            {
                var src = _pending![i] ? _pendingSource! : this;
                long cb = (long)_convStateElements * sizeof(float), sb = (long)_gdnStateElements * sizeof(float);
                Buffer.MemoryCopy((float*)src._convState + (long)i * _convStateElements, (float*)destination._convState + (long)i * _convStateElements, cb, cb);
                Buffer.MemoryCopy((float*)src._gdnState + (long)i * _gdnStateElements, (float*)destination._gdnState + (long)i * _gdnStateElements, sb, sb);
            }
            return;
        }

        long convBytes = (long)_numGdnLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numGdnLayers * _gdnStateElements * sizeof(float);
        if (convBytes > 0)
            Buffer.MemoryCopy((void*)_convState, (void*)destination._convState, convBytes, convBytes);
        if (stateBytes > 0)
            Buffer.MemoryCopy((void*)_gdnState, (void*)destination._gdnState, stateBytes, stateBytes);
    }

    // ── lazy / fused copy (checkpoint-restore without a separate copy pass, #840) ──

    /// <summary>True while at least one layer logically holds another cache's content (see <see cref="DeferCopyFrom"/>).</summary>
    public bool HasPendingSource => _pendingCount != 0;

    /// <summary>True when this cache's pending layers point at <paramref name="source"/>.</summary>
    public bool IsPendingOn(GdnStateCache source) => _pendingCount != 0 && ReferenceEquals(_pendingSource, source);

    /// <summary>
    /// Makes this cache logically equal to <paramref name="source"/> WITHOUT copying: every layer is marked pending on it and the
    /// live buffers become stale. The next <see cref="BeginUpdate"/> of a layer hands out the source's spans so the update can read
    /// them for its first step (fusing the copy into work the update does anyway); every other accessor materialises on demand.
    /// <paramref name="source"/> must stay unmodified and alive while pending layers remain; release it via
    /// <see cref="MaterializePending"/> first. A source that itself has pending layers is materialised first.
    /// </summary>
    public void DeferCopyFrom(GdnStateCache source)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        if (ReferenceEquals(source, this)) return;
        CheckShape(source);
        if (source._pendingCount != 0) source.MaterializePending();
        if (_numGdnLayers == 0) return;
        _pendingSource = source;
        _pending ??= new bool[_numGdnLayers];
        Array.Fill(_pending, true);
        _pendingCount = _numGdnLayers;
    }

    /// <summary>
    /// Prepares layer <paramref name="gdnLayerIndex"/> for an in-place update. When the layer is pending, <paramref name="srcConv"/> /
    /// <paramref name="srcGdn"/> are the SOURCE's current conv / matrix state (the caller must read its input from them and write the
    /// result into the live spans from <see cref="GetConvStateForUpdate"/> / <see cref="GetGdnStateForUpdate"/>); the layer is then
    /// no longer pending. Otherwise both are empty and the update is the usual in-place one.
    /// </summary>
    public void BeginUpdate(int gdnLayerIndex, out ReadOnlySpan<float> srcConv, out ReadOnlySpan<float> srcGdn)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers) throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        if (_pendingCount != 0 && _pending![gdnLayerIndex])
        {
            var src = _pendingSource!;
            srcConv = new ReadOnlySpan<float>((float*)src._convState + (long)gdnLayerIndex * _convStateElements, _convStateElements);
            srcGdn = new ReadOnlySpan<float>((float*)src._gdnState + (long)gdnLayerIndex * _gdnStateElements, _gdnStateElements);
            _pending[gdnLayerIndex] = false;
            if (--_pendingCount == 0) _pendingSource = null;
        }
        else { srcConv = default; srcGdn = default; }
    }

    /// <summary>Live conv state of a layer for a write (after <see cref="BeginUpdate"/>); never materialises.</summary>
    public Span<float> GetConvStateForUpdate(int gdnLayerIndex)
        => new((float*)_convState + (long)gdnLayerIndex * _convStateElements, _convStateElements);

    /// <summary>Live matrix state of a layer for a write (after <see cref="BeginUpdate"/>); never materialises.</summary>
    public Span<float> GetGdnStateForUpdate(int gdnLayerIndex)
        => new((float*)_gdnState + (long)gdnLayerIndex * _gdnStateElements, _gdnStateElements);

    /// <summary>
    /// Layer <paramref name="gdnLayerIndex"/> is about to be fully overwritten by the caller: drops its pending mark without copying.
    /// </summary>
    public void DiscardPending(int gdnLayerIndex)
    {
        if (_pendingCount != 0 && _pending![gdnLayerIndex])
        {
            _pending[gdnLayerIndex] = false;
            if (--_pendingCount == 0) _pendingSource = null;
        }
    }

    /// <summary>Layers physically copied by lazy-copy materialisation so far (a lazy copy consumed by <see cref="BeginUpdate"/> or
    /// resolved by <see cref="TryTakeSourceBuffers"/> costs none).</summary>
    public long MaterializedLayerCopies { get; private set; }

    /// <summary>
    /// <paramref name="source"/> is about to be released and this cache still lazily reads ALL of its layers from it: instead of
    /// copying, exchange buffers (this cache takes the source's, which hold exactly its logical content; the source takes the stale
    /// ones, whose content its owner no longer needs). Returns false (and does nothing) unless every layer is still pending on
    /// <paramref name="source"/>.
    /// </summary>
    public bool TryTakeSourceBuffers(GdnStateCache source)
    {
        if (_pendingCount != _numGdnLayers || _numGdnLayers == 0 || !ReferenceEquals(_pendingSource, source)) return false;
        (_convState, source._convState) = (source._convState, _convState);
        (_gdnState, source._gdnState) = (source._gdnState, _gdnState);
        ClearPending();
        return true;
    }

    /// <summary>Performs every outstanding lazy copy (after this the cache no longer depends on its former source).</summary>
    public void MaterializePending()
    {
        if (_pendingCount == 0) return;
        for (int i = 0; i < _numGdnLayers; i++)
            if (_pending![i]) MaterializeLayer(i);
    }

    private void MaterializeLayer(int i)
    {
        if (!_pending![i]) return;
        var src = _pendingSource!;
        long cb = (long)_convStateElements * sizeof(float), sb = (long)_gdnStateElements * sizeof(float);
        Buffer.MemoryCopy((float*)src._convState + (long)i * _convStateElements, (float*)_convState + (long)i * _convStateElements, cb, cb);
        Buffer.MemoryCopy((float*)src._gdnState + (long)i * _gdnStateElements, (float*)_gdnState + (long)i * _gdnStateElements, sb, sb);
        _pending[i] = false;
        MaterializedLayerCopies++;
        if (--_pendingCount == 0) _pendingSource = null;
    }

    private void ClearPending()
    {
        if (_pendingCount == 0) return;
        Array.Clear(_pending!);
        _pendingCount = 0;
        _pendingSource = null;
    }

    /// <summary>
    /// Exchanges the underlying buffers with <paramref name="other"/> (O(1)). Both must be fully materialised (no pending layers) and
    /// of the same shape. Used to capture a checkpoint without copying: the checkpoint takes the live buffers, the live cache takes
    /// the checkpoint's stale ones and defers its content from the checkpoint via <see cref="DeferCopyFrom"/>.
    /// </summary>
    public void SwapBuffersWith(GdnStateCache other)
    {
        ThrowIfDisposed(); other.ThrowIfDisposed();
        CheckShape(other);
        if (_pendingCount != 0 || other._pendingCount != 0)
            throw new InvalidOperationException("Materialise pending layers before swapping buffers.");
        (_convState, other._convState) = (other._convState, _convState);
        (_gdnState, other._gdnState) = (other._gdnState, _gdnState);
    }

    private void CheckShape(GdnStateCache other)
    {
        if (other._numGdnLayers != _numGdnLayers
            || other._convStateElements != _convStateElements
            || other._gdnStateElements != _gdnStateElements)
            throw new ArgumentException("GdnStateCache shape does not match.", nameof(other));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { NativeMemory.AlignedFree((void*)_convState); _convState = 0; }
        if (_gdnState != 0) { NativeMemory.AlignedFree((void*)_gdnState); _gdnState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GdnStateCache));
    }

    /// <summary>Finalizer — last-ditch free if the cache was not disposed.</summary>
    ~GdnStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) NativeMemory.AlignedFree((void*)_convState);
        if (_gdnState != 0) NativeMemory.AlignedFree((void*)_gdnState);
    }
}
