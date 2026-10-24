using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

public sealed unsafe partial class VulkanLfm2MoeTransformerModel
{
    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    private sealed class RecurrentCheckpoint(VulkanLfm2MoeTransformerModel owner, VulkanLfm2ConvStateCache state) : IDisposable
    {
        public VulkanLfm2MoeTransformerModel Owner { get; } = owner;
        public VulkanLfm2ConvStateCache State { get; } = state;
        public void Dispose() => State.Dispose();
    }

    /// <inheritdoc/>
    public object? CheckpointRecurrentState() => new RecurrentCheckpoint(this, _defaultState.Clone());

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        if (checkpoint is null) return;
        if (checkpoint is not RecurrentCheckpoint snap || !ReferenceEquals(snap.Owner, this))
            throw new ArgumentException("Checkpoint belongs to a different model.", nameof(checkpoint));
        _defaultState.CopyFrom(snap.State);
    }

    /// <inheritdoc/>
    public bool SupportsSequencePrefixSnapshot => true;

    private sealed class PrefixSnapshot(VulkanLfm2MoeTransformerModel owner, VulkanNemotronHKvCache kv, VulkanLfm2ConvStateCache state, int length) : IDisposable
    {
        public VulkanLfm2MoeTransformerModel Owner { get; } = owner;
        public VulkanNemotronHKvCache Kv { get; } = kv;
        public VulkanLfm2ConvStateCache State { get; } = state;
        public int Length { get; } = length;
        public bool Disposed { get; private set; }
        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;
            Kv.Dispose(); State.Dispose();
        }
    }

    private VulkanLfm2ConvStateCache RequirePrefixState(IRecurrentSequenceState? state)
    {
        if (state is not ISsmState s)
            throw new ArgumentException("A per-sequence LFM2 state is required.", nameof(state));
        return ResolveState(s)!;
    }

    private VulkanNemotronHKvCache ValidatePrefixCache(IKvCache cache, int length, bool source)
    {
        if (cache is not VulkanNemotronHKvCache kv)
            throw new ArgumentException("LFM2 Vulkan snapshots require the model's F32 device KV cache.", nameof(cache));
        kv.ValidateSnapshotLayout(_device, _layout.LayerKind, Config.NumKvHeads, Config.HeadDim);
        if (length <= 0 || length > kv.MaxLength || (source && length > kv.CurrentLength))
            throw new ArgumentOutOfRangeException(nameof(length));
        return kv;
    }

    /// <inheritdoc/>
    public IDisposable? SnapshotSequencePrefix(IKvCache kvCache, IRecurrentSequenceState? state, int prefixLen)
    {
        _defaultState.ThrowIfDisposed();
        var conv = RequirePrefixState(state);
        var kv = ValidatePrefixCache(kvCache, prefixLen, source: true);
        var kvCopy = kv.SnapshotPrefix(prefixLen);
        try { return new PrefixSnapshot(this, kvCopy, conv.Clone(), prefixLen); }
        catch { kvCopy.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public void RestoreSequencePrefix(IDisposable snapshot, IKvCache kvCache, IRecurrentSequenceState? state)
    {
        _defaultState.ThrowIfDisposed();
        if (snapshot is not PrefixSnapshot snap || !ReferenceEquals(snap.Owner, this))
            throw new ArgumentException("Prefix snapshot belongs to a different model.", nameof(snapshot));
        ObjectDisposedException.ThrowIf(snap.Disposed, snapshot);
        var conv = RequirePrefixState(state);
        var kv = ValidatePrefixCache(kvCache, snap.Length, source: false);
        kv.RestorePrefixFrom(snap.Kv, snap.Length);
        _host.Q4InvalidateCaches(); // Restore may grow a lazy cache and replace its device buffers.
        conv.CopyFrom(snap.State);
    }
}
