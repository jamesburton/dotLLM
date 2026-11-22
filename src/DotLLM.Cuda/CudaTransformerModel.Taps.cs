using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Hidden-state taps (#1920) for the dense / GQA / MoE CUDA host. See <see cref="CudaHiddenStateTaps"/> for the staging scheme. The residual
/// is F16 (<c>Residual</c> between layers, <c>HiddenState</c> after the last, <c>ResidualF32</c> for BitNet); a Flint stream carried at a power-of-two
/// scale is unscaled on the host. A tapped forward never takes a decode graph.
/// </summary>
public sealed unsafe partial class CudaTransformerModel
{
    private CudaHiddenStateTaps? _tapState;
    private CudaHiddenStateTaps? _tapTs;   // the plan of the forward in flight (null: untapped)
    private ForwardAbortInfo _tapAbort;

    private CudaHiddenStateTaps TapState
    {
        get
        {
            // Two concurrent first registrations must share one registry (a plain ??= could drop one, whose tap would then never fire).
            return _tapState ?? Interlocked.CompareExchange(ref _tapState, new CudaHiddenStateTaps(Config.NumLayers, Config.HiddenSize), null)
                ?? _tapState!;
        }
    }

    private void GuardTapsSupported()
    {
        NotSupportedException Reject(string what) => new(
            $"Hidden-state taps are not supported on {what}: its CUDA forward is a dedicated path with its own layer loop "
            + "(only the plain dense / GQA / MoE trunk is wired).");
        if (Config.DiffusionConfig is not null) throw Reject("diffusion models");
        if (Config.BidirectionalAttention) throw Reject("bidirectional embedding towers");
        if (Config.MlaConfig is not null || _weights.MlaLayers is not null) throw Reject("MLA models");
        if (Config.Gemma3n is not null) throw Reject("Gemma-3n models");
        if (Config.NumSharedKvLayers > 0) throw Reject("shared-KV (Gemma-4) models");
        if (_isGemma4) throw Reject("Gemma-4 models");
        if (_isGemmaFamily) throw Reject("Gemma / Gemma-2 models");
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
        logits = ForwardInternal(tokenIds, positions, deviceId, kvCache, suppressGraphCapture: false, deferredHostDest: 0);
        if (logits is null)
        {
            abort = _tapAbort;
            return false;
        }
        CudaMemTrace.Mark(tokenIds.Length > 1 ? "multi-token forward" : "decode forward");
        AfterMultiTokenForward(tokenIds.Length);
        abort = default;
        return true;
    }

    private ITensor? TapAborted(ForwardAbortInfo at)
    {
        _tapAbort = at;
        return null;
    }

    /// <summary>
    /// The forward with tap bookkeeping around <see cref="ForwardInternalCore"/>. <see langword="null"/> means an abortable tap aborted it
    /// (and, as before, a deferred or captured untapped forward).
    /// </summary>
    private ITensor? ForwardInternal(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                     int deviceId, IKvCache? kvCache, bool suppressGraphCapture, nint deferredHostDest)
    {
        var ts = _tapState;
        var taps = ts?.Registry.Current;   // read once: null (nothing attached) is the whole cost of an untapped forward
        if (taps is null || _moeGraphCapturing || tokenIds.Length == 0 || tokenIds.Length != positions.Length)
            return ForwardInternalCore(tokenIds, positions, deviceId, kvCache, suppressGraphCapture, deferredHostDest);

        GuardTapsSupported();
        if (_useHighPrecisionForward && kvCache is null && _weights.MoeLayers is null)
            throw new NotSupportedException(
                "Hidden-state taps are not supported on the cacheless high-precision (F32) forward of this model.");
        if (deferredHostDest != 0)
            throw new InvalidOperationException("A tapped forward cannot defer its logits copy (the taps deliver after the forward's synchronize).");

        int kvEntry = kvCache?.CurrentLength ?? -1;   // every tapped forward records it (free): a throwing callback must not leave KV half-advanced
        try
        {
            if (taps.HasAbortable)
            {
                kvEntry = CudaHiddenStateTaps.KvCheckpoint(kvCache, positions);
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            _context.MakeCurrent();
            ts!.Begin(taps, positions, _stream.Handle);
            _tapTs = ts;
            var result = ForwardInternalCore(tokenIds, positions, deviceId, kvCache, suppressGraphCapture, deferredHostDest);
            if (result is null)
            {
                if (kvCache is not null && kvEntry >= 0) kvCache.Rollback(kvEntry);
                HiddenStateTapRegistry.PathCounters.AddAbortedForward();
            }
            return result;
        }
        catch
        {
            // A throwing tap callback (or any failure on the tapped path): drain in-flight copies, undo the KV writes, rethrow.
            try { _stream.Synchronize(); } catch (Exception) { /* the original failure wins */ }
            CudaHiddenStateTaps.RollbackKvAfterFailure(kvCache, kvEntry);
            throw;
        }
        finally
        {
            _tapTs = null;
            ts!.End();
        }
    }

    /// <summary>Whether the decode graph this forward would have taken is what a tap bypassed (so the counter never counts a forward that was eager anyway).</summary>
    private bool TapWouldHaveUsedGraph(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, IKvCache? kvCache,
                                       bool suppressGraphCapture, nint deferredHostDest, bool isMla, bool isMoe, bool isBitNet)
    {
        if (!UseGraphCapture || suppressGraphCapture || tokenIds.Length != 1 || ProfilingEnabled || isMla || _currentAdapter is not null)
            return false;
        if (!isMoe)
            return _kernels.HasKvWriteKernel && Config.NoRopeLayers is not { Count: > 0 }
                && positions[0] < (isBitNet ? BitNetGraphCaptureMaxDepth : GraphCaptureMaxDepth)
                && (kvCache is CudaKvCache || kvCache is CudaQuantizedKvCache { WindowCapacity: > 0, ReferenceDequantScratch: false });
        return !_moeGraphFailed && MoeDecodeGraphEnabled && deferredHostDest == 0 && DebugCaptureAttnLayer < 0
            && kvCache is CudaKvCache moeKv && positions[0] == moeKv.CurrentLength
            && _kernels.HasFusedRopeKvWriteKernel && CanCaptureMoeDecode();
    }

    /// <summary>
    /// <see cref="ForwardBatch"/> with a tap attached: one synchronous forward per request (the deferred pinned-logits batch has one
    /// synchronize for all of them, which a tap cannot use), each reporting its request index as the sample's sequence id.
    /// </summary>
    private IReadOnlyList<ITensor> ForwardBatchTapped(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        var ts = _tapState!;
        HiddenStateTapRegistry.PathCounters.AddPackedBatchFallback();
        var results = new ITensor[requests.Count];
        int done = 0;
        try
        {
            for (; done < requests.Count; done++)
            {
                var r = requests[done];
                ts.SeqId = done;
                results[done] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter);
            }
            return results;
        }
        catch
        {
            for (int i = 0; i < done; i++) results[i].Dispose();
            throw;
        }
        finally
        {
            ts.SeqId = 0;
        }
    }
}
