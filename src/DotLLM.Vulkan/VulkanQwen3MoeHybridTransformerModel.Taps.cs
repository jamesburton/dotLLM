using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Hidden-state taps (#1920) for the Vulkan GDN-hybrid MoE host. The default single-token decode is one fused command buffer
/// (<c>ForwardDecodeFused</c>) with no site to read from, so a tapped forward always takes the general per-layer loop (two submit-and-waits
/// per layer): every site there is read back where the GPU is idle (<see cref="VulkanHiddenStateTaps.DeliverFromDevice"/>).
/// </summary>
public sealed partial class VulkanQwen3MoeHybridTransformerModel : IHiddenStateTapHost
{
    private VulkanHiddenStateTaps? _tapState;
    private HiddenStateTapRegistry.Snapshot? _tapSnap;   // the snapshot of the forward in flight (null: untapped)
    private int _tapKvEntry = -1;
    private object? _tapCkpt;                            // recurrent-state checkpoint taken at entry (abortable taps only)
    private ForwardAbortInfo _tapAbort;

    private VulkanHiddenStateTaps TapState => _tapState ??= new VulkanHiddenStateTaps(_device, Config.NumLayers, Config.HiddenSize);

    /// <inheritdoc/>
    public IDisposable RegisterTap(HiddenStateTapSubscription subscription, Action<HiddenStateSample> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        return TapState.Registry.Register(subscription, onSample, null);
    }

    /// <inheritdoc/>
    public IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample)
    {
        ArgumentNullException.ThrowIfNull(onSample);
        DotLLM.Core.Diagnostics.AbortableTapKvGuard.ThrowIfWindowedConfig(Config);
        return TapState.Registry.Register(subscription, null, onSample);
    }

    /// <inheritdoc/>
    public bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           out ITensor? logits, out ForwardAbortInfo abort)
    {
        logits = ForwardTapped(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState: null);
        if (logits is null)
        {
            abort = _tapAbort;
            return false;
        }
        abort = default;
        return true;
    }

    /// <summary>The forward with tap bookkeeping around <see cref="ForwardCore"/>; <see langword="null"/> when an abortable tap aborted it.</summary>
    private ITensor? ForwardTapped(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                   IKvCache? kvCache, IGdnState? gdnState, IMtpState? mtpState)
    {
        var taps = _tapState?.Registry.Current;   // read once: null (nothing attached) is the whole cost of an untapped forward
        if (taps is null)
            return ForwardCore(tokenIds, positions, deviceId, kvCache, gdnState, mtpState);

        if (mtpState is not null)
            throw new NotSupportedException("Hidden-state taps are not supported together with an MTP state on the same forward.");

        _tapSnap = taps;
        _tapKvEntry = -1;
        _tapCkpt = null;
        try
        {
            if (taps.HasAbortable)
            {
                _tapKvEntry = TapKvCheckpoint(kvCache, positions);
                if (gdnState is null) _tapCkpt = CheckpointRecurrentState();
                else if (gdnState is VulkanGdnStateCache own)
                {
                    var snap = own.CloneGeometry();
                    own.CopyTo(snap);
                    _tapCkpt = snap;
                }
                HiddenStateTapRegistry.PathCounters.AddAbortCheckpoint();
            }
            return ForwardCore(tokenIds, positions, deviceId, kvCache, gdnState, mtpState);
        }
        finally
        {
            (_tapCkpt as IDisposable)?.Dispose();
            _tapCkpt = null;
            _tapSnap = null;
            _tapState?.End();
        }
    }

    /// <summary>
    /// The PreLmHead site: the final norm of the subscribed rows in a dedicated submit (so the head's own scratch is untouched and the
    /// norm is the exact one the head applies), read back and dispatched.
    /// </summary>
    private bool TapPreLmHead(int seqLen, int hiddenSize, float eps, ReadOnlySpan<int> positions, out ForwardAbortInfo abort)
    {
        abort = default;
        var ts = _tapState!;
        int rows = ts.Rows(HookPoint.PreLmHead, -1);
        if (rows == 0) return true;
        long rowBytes = (long)hiddenSize * sizeof(float);
        _submit.Begin();
        nint cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        if (rows == 1)
        {
            RecordCopyBufferRange(cmd, _state.HiddenState, _state.NormOutput,
                srcOffset: (ulong)((seqLen - 1) * rowBytes), dstOffset: 0, size: (ulong)rowBytes);
            KernelSupport.TransferToComputeBarrier(cmd);
            _kernels.RmsNorm.Record(cmd, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
                rowCount: 1, n: hiddenSize, eps: eps);
        }
        else
        {
            _kernels.RmsNorm.Record(cmd, _state.HiddenState, _weights.OutputNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
        }
        KernelSupport.ComputeToHostBarrier(cmd);
        _submit.SubmitAndWait();
        return ts.DeliverFromDevice(HookPoint.PreLmHead, -1, _state.NormOutput, 0, positions, out abort);
    }

    /// <summary>Common tail of an aborted tapped forward: KV rollback, recurrent restore, counters. Returns null (the abort sentinel).</summary>
    private ITensor? FinishTapAbort(ForwardAbortInfo abort, IKvCache? kvCache, IGdnState? gdnState)
    {
        if (kvCache is not null && _tapKvEntry >= 0) kvCache.Rollback(_tapKvEntry);
        if (_tapCkpt is VulkanGdnStateCache snap && gdnState is VulkanGdnStateCache own) snap.CopyTo(own);
        else if (_tapCkpt is not null) RestoreRecurrentState(_tapCkpt);
        HiddenStateTapRegistry.PathCounters.AddAbortedForward();
        _tapAbort = abort;
        return null;
    }

    private static int TapKvCheckpoint(IKvCache? kvCache, ReadOnlySpan<int> positions)
        => DotLLM.Core.Diagnostics.AbortableTapKvGuard.Checkpoint(kvCache, positions);
}
