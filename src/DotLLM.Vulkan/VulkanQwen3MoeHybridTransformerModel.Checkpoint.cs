using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// GDN recurrent checkpoint pair (issue #1188), the prerequisite for any speculative decoding on <c>qwen35moe</c>: a verify forward advances
/// the recurrence for every drafted token before accept / reject is known, and unlike the KV cache it has no positions to roll back.
/// Mirrors <see cref="VulkanQwen3HybridDenseTransformerModel"/>'s pooled device snapshot.
/// </summary>
public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    private VulkanGdnStateCache? _spareGdnCheckpoint;
    private bool _checkpointsClosed;

    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    /// <remarks>Pooled (one spare): disposing the returned checkpoint hands its device buffers back for the next round.</remarks>
    public object? CheckpointRecurrentState()
    {
        VulkanGdnStateCache snapshot = Interlocked.Exchange(ref _spareGdnCheckpoint, null) ?? _gdnCache.CloneGeometry();
        _gdnCache.CopyTo(snapshot);
        return new PooledGdnCheckpoint(this, snapshot);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        switch (checkpoint)
        {
            case null:
                return;
            case PooledGdnCheckpoint pooled:
                pooled.Snapshot.CopyTo(_gdnCache);
                _rowSnapValid = 0;
                return;
            case VulkanGdnStateCache snapshot:
                snapshot.CopyTo(_gdnCache);
                _rowSnapValid = 0;
                return;
            default:
                throw new ArgumentException(
                    $"{GetType().Name}.RestoreRecurrentState expects a checkpoint from CheckpointRecurrentState; got {checkpoint.GetType().Name}.",
                    nameof(checkpoint));
        }
    }

    private void CloseCheckpoints()
    {
        _checkpointsClosed = true;
        Interlocked.Exchange(ref _spareGdnCheckpoint, null)?.Dispose();
        _multiRowLogits?.Dispose();
        _multiRowLogits = null;
        _rowSnaps?.Dispose();
        _rowSnaps = null;
        _rowSnapScan?.Dispose();
    }

    private sealed class PooledGdnCheckpoint(VulkanQwen3MoeHybridTransformerModel owner, VulkanGdnStateCache snapshot) : IDisposable
    {
        private VulkanGdnStateCache? _snapshot = snapshot;

        public VulkanGdnStateCache Snapshot => _snapshot ?? throw new ObjectDisposedException(nameof(PooledGdnCheckpoint));

        public void Dispose()
        {
            var s = Interlocked.Exchange(ref _snapshot, null);
            if (s is null) return;
            if (owner._checkpointsClosed || Interlocked.CompareExchange(ref owner._spareGdnCheckpoint, s, null) is not null)
                s.Dispose();
        }
    }

    /// <summary>
    /// Largest batch whose every row gets logits (#1188); longer prefills return the last row only. Covers a K+1-row speculative verify.
    /// </summary>
    public const int MaxAllRowLogitsSeqLen = 16;

    /// <inheritdoc/>
    public int MaxAllRowLogitsLength => MaxAllRowLogitsSeqLen;

    private VulkanDevice.Buffer? _multiRowLogits;
    private int _multiRowLogitsRows;

    private VulkanDevice.Buffer EnsureMultiRowLogits(int rows, int vocab)
    {
        if (_multiRowLogits is not null && _multiRowLogitsRows >= rows)
            return _multiRowLogits;
        _multiRowLogits?.Dispose();
        _multiRowLogits = _device.AllocateHostReadback((long)rows * vocab * sizeof(float));
        _multiRowLogitsRows = rows;
        // A freed handle can be recycled into this allocation and the kernels key descriptor sets on handles.
        _kernels.InvalidateAll();
        _iqF16Prefill?.InvalidateDescriptorCache();
        return _multiRowLogits;
    }

    // ── Per-row recurrent snapshots (#473 contract, #1188): a partial rejection restores the row instead of replaying a forward ──

    private GdnScanMultiTokenSnapshotF32Kernel? _rowSnapScan;
    private Q4GdnRowSnapshots? _rowSnaps;     // Rows > 0 only while ForwardWithRecurrentSnapshots runs
    private int _rowSnapCapacity;
    private int _rowSnapValid;                // rows the last snapshot forward recorded; 0 = none

    /// <inheritdoc/>
    /// <remarks>
    /// The verify swaps the GDN scan for its bit-identical snapshot twin and copies each row's conv window (the qwen4exp host's #820
    /// machinery). Measured on Qwen3.6-35B-A3B UD-Q4_K_M (Strix, K=3): without it every partial rejection replayed the accepted rows,
    /// ~30 ms on a ~47 ms round.
    /// </remarks>
    public bool SupportsRecurrentRowSnapshots => _rowSnapScan is not null;

    /// <inheritdoc/>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, IMtpState? mtpState)
    {
        var scan = _rowSnapScan ?? throw new NotSupportedException($"{GetType().Name}: no GDN snapshot scan kernel (SupportsRecurrentRowSnapshots=false).");
        int rows = Math.Max(tokenIds.Length - 1, 0);
        EnsureRowSnapshotCapacity(scan, rows);   // before Forward records anything: a grow invalidates descriptor sets
        _rowSnaps!.Rows = rows;
        try
        {
            var logits = Forward(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState);
            _rowSnapValid = rows;
            return logits;
        }
        finally
        {
            _rowSnaps.Rows = 0;
        }
    }

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if (row == _rowSnapValid && row > 0)
            return;   // the live state IS the state after the last row
        if ((uint)row >= (uint)_rowSnapValid)
            throw new InvalidOperationException(
                $"No recurrent snapshot for row {row}: the last ForwardWithRecurrentSnapshots recorded {_rowSnapValid} row(s), or a later forward invalidated them.");
        long stateBytes = (long)_gdnCache.GdnStateElements * sizeof(float);
        long convBytes = (long)_gdnCache.ConvStateElements * sizeof(float);
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        nint cmd = ctx.CommandBuffer;
        KernelSupport.ComputeToTransferBarrier(cmd);
        for (int l = 0; l < _gdnCache.NumGdnLayers; l++)
        {
            if (stateBytes > 0)
                RecordCopyBufferRange(cmd, _rowSnaps!.Gdn[l], _gdnCache.GetGdnStateBuffer(l), (ulong)(row * stateBytes), 0, (ulong)stateBytes);
            if (convBytes > 0)
                RecordCopyBufferRange(cmd, _rowSnaps!.Conv[l], _gdnCache.GetConvStateBuffer(l), (ulong)(row * convBytes), 0, (ulong)convBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmd);
        ctx.SubmitAndWait();
        _rowSnapValid = 0;
    }

    private void EnsureRowSnapshotCapacity(GdnScanMultiTokenSnapshotF32Kernel scan, int rows)
    {
        if (_rowSnaps is not null && rows <= _rowSnapCapacity) return;
        _rowSnaps?.Dispose();
        int layers = _gdnCache.NumGdnLayers;
        long stateBytes = Math.Max((long)rows * _gdnCache.GdnStateElements * sizeof(float), 4);
        long convBytes = Math.Max((long)rows * _gdnCache.ConvStateElements * sizeof(float), 4);
        var gdn = new VulkanDevice.Buffer[layers];
        var conv = new VulkanDevice.Buffer[layers];
        for (int l = 0; l < layers; l++)
        {
            gdn[l] = _device.AllocateDeviceLocal(stateBytes);
            conv[l] = _device.AllocateDeviceLocal(convBytes);
        }
        _rowSnaps = new Q4GdnRowSnapshots { Kernel = scan, Gdn = gdn, Conv = conv };
        _rowSnapCapacity = rows;
        _rowSnapValid = 0;
        _kernels.InvalidateAll();
        _iqF16Prefill?.InvalidateDescriptorCache();
        scan.InvalidateDescriptorCache();
    }
}
