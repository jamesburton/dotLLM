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
                return;
            case VulkanGdnStateCache snapshot:
                snapshot.CopyTo(_gdnCache);
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
}
