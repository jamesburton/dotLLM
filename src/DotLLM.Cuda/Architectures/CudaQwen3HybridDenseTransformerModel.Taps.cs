using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

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

    private CudaHiddenStateTaps TapState
    {
        get
        {
            return _tapState ?? Interlocked.CompareExchange(ref _tapState, new CudaHiddenStateTaps(Config.NumLayers, Config.HiddenSize), null)
                ?? _tapState!;
        }
    }

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
        DotLLM.Core.Diagnostics.AbortableTapKvGuard.ThrowIfWindowedConfig(Config);
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
        // An abort of request i would leave requests 0..i-1 committed: refused up front (#1920 I-6), before any state changes.
        DotLLM.Core.Diagnostics.AbortableTapKvGuard.RefuseAbortableBatch(_tapState?.Registry.Current);
        var results = new ITensor[requests.Count];
        var ts = _tapState;
        int done = 0;
        try
        {
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                if (ts is not null) ts.SeqId = i;
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter, mtpState: null, r.LastTokenLogitsOnly);
                done++;
            }
        }
        catch
        {
            for (int d = 0; d < done; d++) results[d].Dispose();   // do not leak the completed requests' logits
            throw;
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

        GuardTapsSupported();

        bool abortable = taps.HasAbortable;
        int kvEntry = kvCache?.CurrentLength ?? -1;   // every tapped forward records it (free): a throwing callback must not leave KV half-advanced
        object? ckpt = null;
        _tapAborted = false;
        _tapFinalTile = true;
        try
        {
            bool pending = _rowSnapshotValidRows > 0;   // verify-row snapshots that CheckpointRecurrentState would commit
            if (abortable)
            {
                // An MTP/DFlash head is updated after the trunk and is not covered by the checkpoint: abortable + MTP is refused on every
                // backend. A read-only tap with an MTP state only reads, so it is supported (delivered after the capture).
                if (mtpState is not null)
                    throw new NotSupportedException("An abortable hidden-state tap cannot be combined with an MTP/DFlash state on this forward.");
                // CheckpointRecurrentState runs CommitPendingRows: it would commit rejected draft rows that RestoreRecurrentStateToRow still needs.
                if (pending)
                    throw new NotSupportedException(
                        "An abortable hidden-state tap cannot start while verify-row snapshots are pending (RestoreRecurrentStateToRow has not run): "
                        + "the entry checkpoint would commit the rejected draft rows. Nothing was modified.");
                kvEntry = CudaHiddenStateTaps.KvCheckpoint(kvCache, positions);
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            // Entry checkpoint: always for an abortable tap; for a read-only tap too (a throwing callback must leave the recurrent state
            // consistent with the rolled-back KV), except with an MTP state or pending verify rows, which a checkpoint cannot cover.
            if (abortable || (mtpState is null && !pending))
                ckpt = CheckpointRecurrentState();
            _context.MakeCurrent();
            ts!.Begin(taps, positions, _stream.Handle);
            _tapTs = ts;
            var result = ForwardCoreInner(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, mtpState);
            if (result is null && _tapAborted)
            {
                RestoreEntryState(kvCache, kvEntry, ckpt);
                HiddenStateTapRegistry.PathCounters.AddAbortedForward();
            }
            return result;
        }
        catch
        {
            // A throwing tap callback (or any failure on the tapped path): drain in-flight copies, then KV and recurrent state back to entry.
            try { _stream.Synchronize(); } catch (Exception) { /* the original failure wins */ }
            RestoreEntryState(kvCache, kvEntry, ckpt);
            throw;
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

    private void RestoreEntryState(IKvCache? kvCache, int kvEntry, object? ckpt)
    {
        CudaHiddenStateTaps.RollbackKvAfterFailure(kvCache, kvEntry);
        if (kvCache is CudaHybridKvCacheHandle handle)
            _f16CacheCurrentLength = HybridKvLengthBookkeeping.SyncToCommitted(
                _f16CacheCurrentLength, Span<int>.Empty, handle.CurrentLength);
        if (ckpt is not null) RestoreRecurrentState(ckpt);
    }

    /// <summary>
    /// Test hook (#1920): the model-owned F16 KV rows <c>[0, rows)</c> of every attention slot, K then V, widened to F32, for byte comparison.
    /// Empty before the first attention forward. F16 (non-quantized) KV storage only.
    /// </summary>
    internal float[] DebugReadKvRowsForTest(int rows)
    {
        if (_f16KCache is null || rows <= 0) return [];
        _context.MakeCurrent();
        _stream.Synchronize();
        int elems = rows * Config.NumKvHeads * Config.HeadDim;
        var all = new List<float>();
        var host = new ushort[elems];
        for (int slot = 0; slot < _attentionLayerCount; slot++)
        {
            foreach (nint dev in new[] { _f16KCache[slot], _f16VCache![slot] })
            {
                fixed (ushort* p = host)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(elems * sizeof(ushort))).ThrowOnError();
                foreach (ushort bits in host) all.Add((float)BitConverter.UInt16BitsToHalf(bits));
            }
        }
        return [.. all];
    }
}
