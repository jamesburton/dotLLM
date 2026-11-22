using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Hidden-state taps (#1920) for the CUDA GDN-hybrid dense host. See <see cref="CudaHiddenStateTaps"/> for the staging scheme. The residual
/// is the F32 <c>HiddenState</c> (the same buffer the DFlash taps read), copied asynchronously to pinned host memory at layer boundaries.
/// A tapped forward never takes the decode graph. A long prefill is tiled as usual unless a tap wants every row; a last-row tap fires in the
/// final tile. The KV entry length and the recurrent-state checkpoint are taken once for the whole forward, so an abort in any tile undoes all.
/// </summary>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel
{
    private CudaHiddenStateTaps? _tapState;
    private CudaHiddenStateTaps? _tapTs;      // the plan of the forward in flight (null: untapped)
    private bool _tapFinalTile = true;
    private bool _tapAborted;
    private ForwardAbortInfo _tapAbort;

    private CudaHiddenStateTaps TapState => _tapState ??= new CudaHiddenStateTaps(Config.NumLayers, Config.HiddenSize);

    private void GuardTapsSupported()
    {
        if (_isHeadOnly)
            throw new NotSupportedException(
                "Hidden-state taps are not supported on a head-only (partial-offload GPU head) instance: it runs layers through ForwardHead.");
    }

    /// <inheritdoc/>
    public IDisposable RegisterTap(HiddenStateTapSubscription subscription, Action<HiddenStateSample> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        GuardTapsSupported();
        return TapState.Registry.Register(subscription, onSample, null);
    }

    /// <inheritdoc/>
    public IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        GuardTapsSupported();
        return TapState.Registry.Register(subscription, null, onSample);
    }

    /// <inheritdoc/>
    public bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           out ITensor? logits, out ForwardAbortInfo abort)
    {
        logits = ForwardCoreTapped(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly: false, mtpState: null);
        if (logits is null)
        {
            abort = _tapAbort;
            return false;
        }
        abort = default;
        return true;
    }

    /// <summary>
    /// The interface default (a per-request loop; this host has no packed batch) with the request index reported as the
    /// <see cref="HiddenStateSample.SequenceId"/> of a tapped forward.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        var results = new ITensor[requests.Count];
        var ts = _tapState;
        try
        {
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                if (ts is not null) ts.SeqId = i;
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter, mtpState: null, r.LastTokenLogitsOnly);
            }
        }
        finally
        {
            if (ts is not null) ts.SeqId = 0;
        }
        return results;
    }

    private ITensor? TapAborted(ForwardAbortInfo at)
    {
        _tapAborted = true;
        _tapAbort = at;
        return null;
    }

    /// <summary>The forward with tap bookkeeping around <see cref="ForwardCoreInner"/>; <see langword="null"/> when an abortable tap aborted it.</summary>
    private ITensor? ForwardCoreTapped(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                                       bool lastTokenLogitsOnly, IMtpState? mtpState)
    {
        var ts = _tapState;
        var taps = ts?.Registry.Current;   // read once: null (nothing attached) is the whole cost of an untapped forward
        if (taps is null || tokenIds.Length == 0 || tokenIds.Length != positions.Length)
            return ForwardCoreInner(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, mtpState);

        // An MTP/DFlash capture reads the trunk hidden state itself; combining it with a tap is refused rather than half-working.
        if (mtpState is not null)
            throw new NotSupportedException("Hidden-state taps are not supported together with an MTP/DFlash state on the same forward.");
        GuardTapsSupported();

        int kvEntry = -1;
        object? ckpt = null;
        _tapAborted = false;
        _tapFinalTile = true;
        try
        {
            if (taps.HasAbortable)
            {
                kvEntry = CudaHiddenStateTaps.KvCheckpoint(kvCache);
                ckpt = CheckpointRecurrentState();
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            _context.MakeCurrent();
            ts!.Begin(taps, positions);
            _tapTs = ts;
            var result = ForwardCoreInner(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, mtpState);
            if (result is null && _tapAborted)
            {
                if (kvCache is not null && kvEntry >= 0) kvCache.Rollback(kvEntry);
                if (kvCache is CudaHybridKvCacheHandle handle)
                    _f16CacheCurrentLength = HybridKvLengthBookkeeping.SyncToCommitted(
                        _f16CacheCurrentLength, Span<int>.Empty, handle.CurrentLength);
                if (ckpt is not null) RestoreRecurrentState(ckpt);
                HiddenStateTapRegistry.PathCounters.AddAbortedForward();
            }
            return result;
        }
        finally
        {
            _tapTs = null;
            _tapAborted = false;
            _tapFinalTile = true;
            (ckpt as IDisposable)?.Dispose();
            ts!.End();
        }
    }
}
