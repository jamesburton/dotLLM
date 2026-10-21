using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

public sealed unsafe partial class Lfm2MoeTransformerModel
{
    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    private sealed class RecurrentCheckpoint(Lfm2MoeTransformerModel owner, Lfm2ConvStateCache state) : IDisposable
    {
        public Lfm2MoeTransformerModel Owner { get; } = owner;
        public Lfm2ConvStateCache State { get; } = state;
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

    private sealed class PrefixSnapshot(Lfm2MoeTransformerModel owner, Lfm2ConvStateCache state, int length, int layers) : IDisposable
    {
        public Lfm2MoeTransformerModel Owner { get; } = owner;
        public Lfm2ConvStateCache State { get; } = state;
        public int Length { get; } = length;
        public Qwen4ExpNativeBuffer[] Keys { get; } = Enumerable.Range(0, layers).Select(_ => new Qwen4ExpNativeBuffer()).ToArray();
        public Qwen4ExpNativeBuffer[] Values { get; } = Enumerable.Range(0, layers).Select(_ => new Qwen4ExpNativeBuffer()).ToArray();
        public bool Disposed { get; private set; }
        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;
            State.Dispose();
            foreach (var b in Keys) b.Dispose();
            foreach (var b in Values) b.Dispose();
        }
    }

    private Lfm2ConvStateCache RequirePrefixState(IRecurrentSequenceState? state)
        => ResolveState(state as ISsmState) ?? throw new ArgumentException("A per-sequence LFM2 state is required.", nameof(state));

    private int ValidatePrefixCache(IKvCache cache, int length, bool source)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (!cache.SupportsExactF32PrefixCopy || cache is not IPerLayerKvCache geometry)
            throw new ArgumentException("LFM2 prefix snapshots require lossless host F32 KV storage with explicit geometry.", nameof(cache));
        if (length <= 0 || length > cache.MaxLength || (source && length > cache.CurrentLength))
            throw new ArgumentOutOfRangeException(nameof(length));
        int stride = checked(Config.NumKvHeads * Config.HeadDim);
        if (geometry.LayerCount != _attentionLayerCount ||
            Enumerable.Range(0, _attentionLayerCount).Any(i => geometry.KvStrideOf(i) != stride))
            throw new ArgumentException("KV cache geometry mismatch.", nameof(cache));
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            ValidateRows(cache.GetKeysRef(i));
            ValidateRows(cache.GetValuesRef(i));
        }
        return stride;

        void ValidateRows(TensorRef rows)
        {
            if (rows.DataPointer == 0 || rows.DType != DType.Float32 || rows.DeviceId != -1 || rows.Dim1 != stride || (source && rows.Dim0 < length))
                throw new ArgumentException("LFM2 prefix snapshots require contiguous host F32 KV rows with the model's geometry.", nameof(cache));
        }
    }

    /// <inheritdoc/>
    public IDisposable? SnapshotSequencePrefix(IKvCache kvCache, IRecurrentSequenceState? state, int prefixLen)
    {
        var conv = RequirePrefixState(state);
        ArgumentNullException.ThrowIfNull(kvCache);
        if (!kvCache.SupportsExactF32PrefixCopy) return null;
        int stride = ValidatePrefixCache(kvCache, prefixLen, source: true);
        var snapshot = new PrefixSnapshot(this, conv.Clone(), prefixLen, _attentionLayerCount);
        try
        {
            int count = checked(prefixLen * stride);
            for (int i = 0; i < _attentionLayerCount; i++)
            {
                snapshot.Keys[i].EnsureExact(count); snapshot.Values[i].EnsureExact(count);
                new ReadOnlySpan<float>((void*)kvCache.GetKeysRef(i).DataPointer, count).CopyTo(snapshot.Keys[i].Slice(0, count));
                new ReadOnlySpan<float>((void*)kvCache.GetValuesRef(i).DataPointer, count).CopyTo(snapshot.Values[i].Slice(0, count));
            }
            return snapshot;
        }
        catch { snapshot.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public void RestoreSequencePrefix(IDisposable snapshot, IKvCache kvCache, IRecurrentSequenceState? state)
    {
        if (snapshot is not PrefixSnapshot snap || !ReferenceEquals(snap.Owner, this))
            throw new ArgumentException("Prefix snapshot belongs to a different model.", nameof(snapshot));
        ObjectDisposedException.ThrowIf(snap.Disposed, snapshot);
        var conv = RequirePrefixState(state);
        int stride = ValidatePrefixCache(kvCache, snap.Length, source: false);
        // Validate state lifetime before changing the destination KV cache.
        conv.ThrowIfDisposed();
        int[] positions = Enumerable.Range(0, snap.Length).ToArray();
        kvCache.Rollback(0);
        for (int i = 0; i < _attentionLayerCount; i++)
            kvCache.Update(new TensorRef(snap.Length, stride, DType.Float32, -1, (nint)snap.Keys[i].Pointer),
                new TensorRef(snap.Length, stride, DType.Float32, -1, (nint)snap.Values[i].Pointer), positions, i);
        conv.CopyFrom(snap.State);
    }
}
