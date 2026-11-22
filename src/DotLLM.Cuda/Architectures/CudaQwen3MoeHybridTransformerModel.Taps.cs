using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Hidden-state taps (#1920) for the CUDA GDN-hybrid MoE host. See <see cref="CudaHiddenStateTaps"/> for the staging scheme: the
/// residual is the F32 <c>HiddenState</c>, copied asynchronously to pinned host memory at each subscribed site.
/// </summary>
public sealed unsafe partial class CudaQwen3MoeHybridTransformerModel
{
    private CudaHiddenStateTaps? _tapState;
    private ForwardAbortInfo _tapAbort;

    private CudaHiddenStateTaps TapState
    {
        get
        {
            return _tapState ?? Interlocked.CompareExchange(ref _tapState, new CudaHiddenStateTaps(Config.NumLayers, Config.HiddenSize), null)
                ?? _tapState!;
        }
    }

    /// <inheritdoc/>
    public IDisposable RegisterTap(HiddenStateTapSubscription subscription, Action<HiddenStateSample> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        return TapState.Registry.Register(subscription, onSample, null);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// An abort restores the recurrent (GDN + conv) state from a checkpoint taken at forward entry, which needs
    /// <see cref="IModel.SupportsRecurrentStateCheckpoint"/>. Until this host implements the checkpoint pair (#1919) the registration is refused.
    /// </remarks>
    public IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        DotLLM.Core.Diagnostics.AbortableTapKvGuard.ThrowIfWindowedConfig(Config);
        if (!((IModel)this).SupportsRecurrentStateCheckpoint)
            throw new NotSupportedException(
                "An abortable hidden-state tap on the CUDA GDN-hybrid MoE host needs recurrent-state checkpoint/restore "
                + "(IModel.SupportsRecurrentStateCheckpoint), which lands with #1919; read-only taps are supported now.");
        return TapState.Registry.Register(subscription, null, onSample);
    }

    /// <inheritdoc/>
    public bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           out ITensor? logits, out ForwardAbortInfo abort)
    {
        logits = ForwardTapped(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly: false);
        if (logits is null)
        {
            abort = _tapAbort;
            return false;
        }
        abort = default;
        return true;
    }

    /// <summary>
    /// The interface default (a per-request loop, this host has no packed batch) with the request index reported as the
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
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.LastTokenLogitsOnly);
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
        _tapAbort = at;
        return null;
    }

    /// <summary>The forward with tap bookkeeping around <see cref="ForwardCore"/>; <see langword="null"/> when an abortable tap aborted it.</summary>
    private ITensor? ForwardTapped(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                                   bool lastTokenLogitsOnly)
    {
        var ts = _tapState;
        var taps = ts?.Registry.Current;   // read once: null (nothing attached) is the whole cost of an untapped forward
        if (taps is null)
            return ForwardCore(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, taps: null);

        bool abortable = taps.HasAbortable;
        int kvEntry = kvCache?.CurrentLength ?? -1;   // every tapped forward records it (free): a throwing callback must not leave KV half-advanced
        object? ckpt = null;
        try
        {
            if (abortable)
            {
                kvEntry = TapKvCheckpoint(kvCache, positions);
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            // Entry checkpoint: always for an abortable tap; for a read-only tap too whenever the host can checkpoint (a throwing
            // callback must leave the recurrent state consistent with the rolled-back KV).
            if (abortable || ((IModel)this).SupportsRecurrentStateCheckpoint)
                ckpt = ((IModel)this).CheckpointRecurrentState();
            var result = ForwardCore(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, taps);
            if (result is null)
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
        if (ckpt is not null) ((IModel)this).RestoreRecurrentState(ckpt);
    }

    /// <summary>Takes the entry checkpoint for an abortable tap: the KV length, after proving the cache can truncate.</summary>
    private static int TapKvCheckpoint(IKvCache? kvCache, ReadOnlySpan<int> positions)
        => DotLLM.Core.Diagnostics.AbortableTapKvGuard.Checkpoint(kvCache, positions);

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
