using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>Hidden-state taps (#1920) for the dense Vulkan host. See <see cref="VulkanHiddenStateTaps"/> for the staging scheme.</summary>
public sealed partial class VulkanTransformerModel
{
    private VulkanHiddenStateTaps? _tapState;
    private ForwardAbortInfo _tapAbort;

    private VulkanHiddenStateTaps TapState => _tapState ??= new VulkanHiddenStateTaps(_device, Config.NumLayers, Config.HiddenSize);

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
    public IDisposable RegisterAbortableTap(DotLLM.Core.Diagnostics.HiddenStateTapSubscription subscription,
                                            Func<DotLLM.Core.Diagnostics.HiddenStateSample, DotLLM.Core.Diagnostics.HookResult> onSample,
                                            DotLLM.Core.Attention.IKvCache cache)
    {
        DotLLM.Core.Diagnostics.AbortableTapKvGuard.ThrowIfCacheCannotAbort(cache);
        return RegisterAbortableTap(subscription, onSample);
    }

    /// <inheritdoc/>
    public bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           out ITensor? logits, out ForwardAbortInfo abort)
    {
        logits = ForwardInner(tokenIds, positions, deviceId, kvCache);
        if (logits is null)
        {
            abort = _tapAbort;
            return false;
        }
        abort = default;
        return true;
    }

    private void GuardTapsSupported()
    {
        NotSupportedException Reject(string what) => new(
            $"Hidden-state taps are not supported on {what}: its forward pass carries state across the whole trunk "
            + "(only the plain dense / GQA / MoE trunk is wired).");
        if (Config.DiffusionConfig is not null) throw Reject("diffusion models");
        if (Config.BidirectionalAttention) throw Reject("bidirectional embedding towers");
        if (Config.MlaConfig is not null) throw Reject("MLA models");
        if (Config.Gemma3n is not null) throw Reject("Gemma-3n models");
        if (Config.NumSharedKvLayers > 0) throw Reject("shared-KV (Gemma-4) models");
        if (Config.PerLayerEmbedding is not null) throw Reject("per-layer-embedding (Gemma-4) models");
        if (_weights.Layers.Length > 0 && _weights.Layers[0].Gemma4 is not null) throw Reject("Gemma-4 models");
        if (_headless || _noTokenEmbed || _firstLayer != 0) throw Reject("trimmed pipeline stages");
    }

    /// <summary>Refuses a tapped forward that takes a tail the taps do not cover (embedding, all-row head, diffusion, pipeline resume).</summary>
    private void GuardTapForward(int seqLen)
    {
        if (_embedRows != EmbedRows.None || _seedHiddenRows is not null || _pkvPhase != DiffusionKvPhase.None
            || _diffusionMaskMode != DotLLM.Core.Attention.AttentionMaskMode.Causal || (seqLen > 1 && seqLen <= _allRowLogitsLimit))
            throw new NotSupportedException(
                "Hidden-state taps are not supported on this forward (embedding output, all-row logits, diffusion or pipeline resume).");
    }

    /// <summary>
    /// One tap site: records the staging copy of <paramref name="src"/> rows, and in immediate mode submits, waits, delivers and re-opens
    /// the command buffer. Returns false (command buffer already submitted, none open) when an abortable tap aborted the forward.
    /// </summary>
    private bool TapAt(VulkanHiddenStateTaps ts, HiddenStateTapRegistry.Snapshot taps, HookPoint point, int layer,
                       VulkanDevice.Buffer src, int srcRow, ref nint cmdBuf, ReadOnlySpan<int> positions, out ForwardAbortInfo abort)
    {
        abort = default;
        int rows = ts.Rows(point, layer);
        if (rows == 0) return true;
        long off = ts.Reserve(point, layer, rows);
        ts.RecordCopy(cmdBuf, src, rows == 1 ? srcRow : 0, rows, off);   // srcRow is the row a single-row (Last) tap reads
        if (!ts.Immediate) return true;
        _submit.SubmitAndWait();
        if (!ts.DeliverImmediate(positions, out abort)) return false;
        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        return true;
    }

    /// <summary>Common tail of an aborted tapped forward: KV rollback, counters, bookkeeping. Returns null (the abort sentinel).</summary>
    private ITensor? FinishTapAbort(VulkanHiddenStateTaps ts, ForwardAbortInfo abort, IKvCache? kvCache, int kvEntry, int seqLen)
    {
        if (kvCache is not null && kvEntry >= 0) kvCache.Rollback(kvEntry);
        HiddenStateTapRegistry.PathCounters.AddAbortedForward();
        _tapAbort = abort;
        ts.End();
        DpEnd();
        if (_profActive) ProfEnd(seqLen);
        return null;
    }

    /// <summary>Undoes a failed tapped forward's KV writes (best effort on a windowed cache a read-only tap may legitimately use).</summary>
    internal static void RollbackKvAfterTapFailure(IKvCache? kvCache, int entry)
    {
        if (kvCache is null || entry < 0 || kvCache.CurrentLength <= entry) return;
        try { kvCache.Rollback(entry); }
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException) { }
    }

    /// <summary>Takes the entry checkpoint for an abortable tap: the KV length, after proving the cache can truncate.</summary>
    private static int TapKvCheckpoint(IKvCache? kvCache, ReadOnlySpan<int> positions)
    {
        int entry = DotLLM.Core.Diagnostics.AbortableTapKvGuard.Checkpoint(kvCache, positions);
        if (entry >= 0) HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
        return entry;
    }
}
