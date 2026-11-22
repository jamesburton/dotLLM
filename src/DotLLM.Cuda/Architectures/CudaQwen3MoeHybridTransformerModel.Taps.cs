using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Hidden-state taps (#1920) for the CUDA GDN-hybrid MoE host. See <see cref="CudaHiddenStateTaps"/> for the staging scheme: the
/// residual is the F32 <c>HiddenState</c>, copied asynchronously to pinned host memory at each subscribed site.
/// </summary>
public sealed unsafe partial class CudaQwen3MoeHybridTransformerModel
{
    private CudaHiddenStateTaps? _tapState;
    private ForwardAbortInfo _tapAbort;

    private CudaHiddenStateTaps TapState => _tapState ??= new CudaHiddenStateTaps(Config.NumLayers, Config.HiddenSize);

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
        var results = new ITensor[requests.Count];
        var ts = _tapState;
        try
        {
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                if (ts is not null) ts.SeqId = i;
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.LastTokenLogitsOnly);
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

        int kvEntry = -1;
        object? ckpt = null;
        try
        {
            if (taps.HasAbortable)
            {
                kvEntry = TapKvCheckpoint(kvCache);
                ckpt = ((IModel)this).CheckpointRecurrentState();
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            var result = ForwardCore(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly, taps);
            if (result is null)
            {
                if (kvCache is not null && kvEntry >= 0) kvCache.Rollback(kvEntry);
                if (kvCache is CudaHybridKvCacheHandle handle)
                    _f16CacheCurrentLength = HybridKvLengthBookkeeping.SyncToCommitted(
                        _f16CacheCurrentLength, Span<int>.Empty, handle.CurrentLength);
                if (ckpt is not null) ((IModel)this).RestoreRecurrentState(ckpt);
                HiddenStateTapRegistry.PathCounters.AddAbortedForward();
            }
            return result;
        }
        finally
        {
            (ckpt as IDisposable)?.Dispose();
            ts!.End();
        }
    }

    /// <summary>Takes the entry checkpoint for an abortable tap: the KV length, after proving the cache can truncate.</summary>
    private static int TapKvCheckpoint(IKvCache? kvCache)
    {
        if (kvCache is null) return -1;
        int entry = kvCache.CurrentLength;
        try { kvCache.Rollback(entry); }
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
        {
            throw new NotSupportedException(
                "An abortable hidden-state tap needs a KV-cache that supports Rollback; this cache refused.", ex);
        }
        return entry;
    }
}
