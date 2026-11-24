using System.Buffers;
using System.Runtime.CompilerServices;

namespace DotLLM.Core.Diagnostics;

/// <summary>
/// The tap table of one model instance (#1920). Registration builds a new immutable <see cref="Snapshot"/> and swaps it in
/// atomically; a forward reads <see cref="Current"/> once into a local, which is <see langword="null"/> when nothing is attached,
/// so the unattached per-layer cost is one local null check and no allocation.
/// </summary>
public sealed class HiddenStateTapRegistry
{
    private readonly int _numLayers;
    private readonly object _gate = new();
    private Entry[] _entries = [];
    private volatile Snapshot? _current;

    /// <summary>Creates a registry for a model with <paramref name="numLayers"/> layers.</summary>
    public HiddenStateTapRegistry(int numLayers)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        _numLayers = numLayers;
    }

    /// <summary>The immutable table a forward should read once, or <see langword="null"/> when no tap is attached.</summary>
    public Snapshot? Current
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _current;
    }

    /// <summary>The points a tap may subscribe to.</summary>
    public static bool IsSupportedPoint(HookPoint point)
        => point is HookPoint.PostEmbedding or HookPoint.PostLayer or HookPoint.PreFinalNorm or HookPoint.PreLmHead;

    /// <summary>
    /// Validates <paramref name="subscription"/> and adds it. Exactly one of <paramref name="observer"/> and <paramref name="abortable"/> is non-null.
    /// </summary>
    /// <returns>A handle whose disposal removes the tap (idempotent).</returns>
    /// <exception cref="NotSupportedException">A point is not one of the supported ones.</exception>
    /// <exception cref="ArgumentException">No point is named, or a layer index is outside the model.</exception>
    public IDisposable Register(
        HiddenStateTapSubscription subscription,
        Action<HiddenStateSample>? observer,
        Func<HiddenStateSample, HookResult>? abortable)
    {
        ArgumentNullException.ThrowIfNull(subscription);
        if ((observer is null) == (abortable is null))
            throw new ArgumentException("Exactly one callback must be supplied.");
        if (subscription.Points is null || subscription.Points.Length == 0)
            throw new ArgumentException("A tap must name at least one HookPoint.", nameof(subscription));

        bool wantsPostLayer = false;
        foreach (var p in subscription.Points)
        {
            if (!IsSupportedPoint(p))
                throw new NotSupportedException(
                    $"HookPoint.{p} is not tappable: hidden-state taps support PostEmbedding, PostLayer, PreFinalNorm and PreLmHead "
                    + "(the other points sit inside fused kernels and are never silently ignored).");
            wantsPostLayer |= p == HookPoint.PostLayer;
        }
        if (subscription.Layers is { } layers)
        {
            foreach (int l in layers)
                if ((uint)l >= (uint)_numLayers)
                    throw new ArgumentException(
                        $"Layer index {l} is outside the model's {_numLayers} layers.", nameof(subscription));
        }

        var entry = new Entry(subscription, observer, abortable, wantsPostLayer, _numLayers);
        lock (_gate)
        {
            _entries = [.. _entries, entry];
            _current = new Snapshot(_entries, _numLayers);
        }
        return new Handle(this, entry);
    }

    private void Remove(Entry entry)
    {
        lock (_gate)
        {
            int i = Array.IndexOf(_entries, entry);
            if (i < 0) return;
            var list = new List<Entry>(_entries);
            list.RemoveAt(i);
            _entries = [.. list];
            _current = _entries.Length == 0 ? null : new Snapshot(_entries, _numLayers);
        }
    }

    private sealed class Handle(HiddenStateTapRegistry owner, Entry entry) : IDisposable
    {
        private Entry? _entry = entry;
        public void Dispose()
        {
            var e = Interlocked.Exchange(ref _entry, null);
            if (e is not null) owner.Remove(e);
        }
    }

    internal sealed class Entry
    {
        public readonly HiddenStateTapSubscription Sub;
        public readonly Action<HiddenStateSample>? Observer;
        public readonly Func<HiddenStateSample, HookResult>? Abortable;
        public readonly bool[]? PostLayerMask;   // null: every layer
        public readonly bool WantsPostLayer;
        public readonly bool WantsPostEmbedding, WantsPreFinalNorm, WantsPreLmHead;

        public Entry(HiddenStateTapSubscription sub, Action<HiddenStateSample>? observer,
                     Func<HiddenStateSample, HookResult>? abortable, bool wantsPostLayer, int numLayers)
        {
            Sub = sub; Observer = observer; Abortable = abortable; WantsPostLayer = wantsPostLayer;
            foreach (var p in sub.Points)
            {
                WantsPostEmbedding |= p == HookPoint.PostEmbedding;
                WantsPreFinalNorm |= p == HookPoint.PreFinalNorm;
                WantsPreLmHead |= p == HookPoint.PreLmHead;
            }
            if (wantsPostLayer && sub.Layers is { } layers)
            {
                PostLayerMask = new bool[numLayers];
                foreach (int l in layers) PostLayerMask[l] = true;
            }
        }

        public bool Wants(HookPoint point, int layer) => point switch
        {
            HookPoint.PostLayer => WantsPostLayer && (PostLayerMask is null || PostLayerMask[layer]),
            HookPoint.PostEmbedding => WantsPostEmbedding,
            HookPoint.PreFinalNorm => WantsPreFinalNorm,
            HookPoint.PreLmHead => WantsPreLmHead,
            _ => false,
        };
    }

    /// <summary>Immutable view of the registered taps at one instant.</summary>
    public sealed class Snapshot
    {
        private readonly Entry[] _entries;
        private readonly bool[] _postLayer;
        private readonly bool _postEmbedding, _preFinalNorm, _preLmHead;

        internal Snapshot(Entry[] entries, int numLayers)
        {
            _entries = entries;
            _postLayer = new bool[numLayers];
            foreach (var e in entries)
            {
                HasAbortable |= e.Abortable is not null;
                _postEmbedding |= e.WantsPostEmbedding;
                _preFinalNorm |= e.WantsPreFinalNorm;
                _preLmHead |= e.WantsPreLmHead;
                if (e.WantsPostLayer)
                    for (int l = 0; l < numLayers; l++)
                        _postLayer[l] |= e.PostLayerMask is null || e.PostLayerMask[l];
            }
        }

        /// <summary>True when at least one attached tap can abort the forward (the model then checkpoints at entry).</summary>
        public bool HasAbortable { get; }

        /// <summary>Whether any tap wants <paramref name="point"/> at <paramref name="layer"/> (layer is ignored for non-layer points).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Wants(HookPoint point, int layer) => point switch
        {
            HookPoint.PostLayer => (uint)layer < (uint)_postLayer.Length && _postLayer[layer],
            HookPoint.PostEmbedding => _postEmbedding,
            HookPoint.PreFinalNorm => _preFinalNorm,
            HookPoint.PreLmHead => _preLmHead,
            _ => false,
        };

        /// <summary>
        /// Whether any tap that wants (<paramref name="point"/>, <paramref name="layer"/>) asked for <see cref="TapRows.All"/>.
        /// A device backend that stages rows before delivery copies one row when this is false and the whole forward when true.
        /// </summary>
        public bool WantsAllRows(HookPoint point, int layer)
        {
            foreach (var e in _entries)
                if (e.Sub.Rows == TapRows.All && e.Wants(point, layer)) return true;
            return false;
        }

        /// <summary>
        /// Delivers the activation to every tap that wants (<paramref name="point"/>, <paramref name="layer"/>), in registration order.
        /// </summary>
        /// <param name="point">The point being tapped.</param>
        /// <param name="layer">Layer index for <see cref="HookPoint.PostLayer"/>; ignored (reported as -1) otherwise.</param>
        /// <param name="rows">Contiguous row-major <c>[rowCount x hiddenSize]</c> activation (read-only: a tap never modifies the residual).</param>
        /// <param name="rowCount">Number of rows in <paramref name="rows"/>.</param>
        /// <param name="rowOffset">Index, within the forward's token rows, of the first row of <paramref name="rows"/> (non-zero when only the LM-head rows are passed).</param>
        /// <param name="hiddenSize">Row width.</param>
        /// <param name="positions">Absolute position per row.</param>
        /// <param name="sequenceId">Sequence identifier reported in the sample.</param>
        /// <param name="step">Step counter reported in the sample.</param>
        /// <param name="abort">Where the pass was aborted, when the result is <see langword="false"/>.</param>
        /// <returns><see langword="true"/> to continue the forward; <see langword="false"/> when an abortable tap aborted it.</returns>
        /// <exception cref="InvalidOperationException">An abortable tap returned something other than Continue / AbortForward.</exception>
        public bool Dispatch(HookPoint point, int layer, ReadOnlySpan<float> rows, int rowCount, int rowOffset, int hiddenSize,
                             ReadOnlySpan<int> positions, int sequenceId, int step, out ForwardAbortInfo abort)
        {
            abort = default;
            int reportedLayer = point == HookPoint.PostLayer ? layer : -1;
            float[]? pooled = null;
            try
            {
                foreach (var e in _entries)
                {
                    if (!e.Wants(point, layer)) continue;

                    int first = e.Sub.Rows == TapRows.All ? 0 : rowCount - 1;
                    for (int r = first; r < rowCount; r++)
                    {
                        Memory<float> buf;
                        if (e.Sub.BufferProvider is { } provider)
                        {
                            buf = provider(hiddenSize);
                            if (buf.Length < hiddenSize)
                                throw new InvalidOperationException(
                                    $"The tap's BufferProvider returned {buf.Length} floats; {hiddenSize} are needed.");
                            buf = buf[..hiddenSize];
                        }
                        else
                        {
                            pooled ??= ArrayPool<float>.Shared.Rent(hiddenSize);
                            buf = pooled.AsMemory(0, hiddenSize);
                        }
                        rows.Slice(r * hiddenSize, hiddenSize).CopyTo(buf.Span);

                        var sample = new HiddenStateSample(point, reportedLayer, rowOffset + r,
                            rowOffset + r < positions.Length ? positions[rowOffset + r] : rowOffset + r, sequenceId, step, buf);
                        if (e.Observer is not null)
                        {
                            e.Observer(sample);
                            continue;
                        }

                        var result = e.Abortable!(sample);
                        if (result is HookResult.AbortForwardResult)
                        {
                            abort = new ForwardAbortInfo(reportedLayer, point);
                            return false;
                        }
                        if (result is not HookResult.ContinueResult)
                            throw new InvalidOperationException(
                                "An abortable hidden-state tap may return only HookResult.Continue or HookResult.AbortForward "
                                + "(taps are read-only; use IResidualModifier to modify the residual stream).");
                    }
                }
                return true;
            }
            finally
            {
                if (pooled is not null) ArrayPool<float>.Shared.Return(pooled);
            }
        }
    }

    /// <summary>
    /// Process-wide counters proving which path a forward took. Unattached runs never move any of them.
    /// Reset between measurements with <see cref="Reset"/> (tests).
    /// </summary>
    public static class PathCounters
    {
        private static long s_tapForwards, s_cudaGraphBypassed, s_vulkanFusedDecodeBypassed,
            s_packedBatchFallbacks, s_abortedForwards, s_abortCheckpoints, s_vulkanRowSliceBypassed;

        /// <summary>Forwards that ran with at least one tap attached.</summary>
        public static long TapForwards => Interlocked.Read(ref s_tapForwards);
        /// <summary>CUDA decode-graph launches skipped because a tap was attached.</summary>
        public static long CudaGraphBypassed => Interlocked.Read(ref s_cudaGraphBypassed);
        /// <summary>Vulkan fused-decode launches skipped because a tap was attached.</summary>
        public static long VulkanFusedDecodeBypassed => Interlocked.Read(ref s_vulkanFusedDecodeBypassed);
        /// <summary>Vulkan hybrid-dense long-prefill row-slice splits skipped because a tap was attached.</summary>
        public static long VulkanRowSliceBypassed => Interlocked.Read(ref s_vulkanRowSliceBypassed);
        /// <summary>Packed <c>ForwardBatch</c> calls that fell back to per-sequence because a tap was attached.</summary>
        public static long PackedBatchFallbacks => Interlocked.Read(ref s_packedBatchFallbacks);
        /// <summary>Forwards stopped by an abortable tap.</summary>
        public static long AbortedForwards => Interlocked.Read(ref s_abortedForwards);
        /// <summary>Recurrent-state / KV checkpoints taken at forward entry for an abortable tap.</summary>
        public static long AbortCheckpoints => Interlocked.Read(ref s_abortCheckpoints);

        /// <summary>Counts one forward that ran with a tap attached.</summary>
        public static void AddTapForward() => Interlocked.Increment(ref s_tapForwards);
        /// <summary>Counts one bypassed CUDA decode graph.</summary>
        public static void AddCudaGraphBypassed() => Interlocked.Increment(ref s_cudaGraphBypassed);
        /// <summary>Counts one bypassed Vulkan fused decode.</summary>
        public static void AddVulkanFusedDecodeBypassed() => Interlocked.Increment(ref s_vulkanFusedDecodeBypassed);
        /// <summary>Counts one bypassed Vulkan row-slice split.</summary>
        public static void AddVulkanRowSliceBypassed() => Interlocked.Increment(ref s_vulkanRowSliceBypassed);
        /// <summary>Counts one packed-batch fallback.</summary>
        public static void AddPackedBatchFallback() => Interlocked.Increment(ref s_packedBatchFallbacks);
        /// <summary>Counts one aborted forward.</summary>
        public static void AddAbortedForward() => Interlocked.Increment(ref s_abortedForwards);
        /// <summary>Counts one abort checkpoint.</summary>
        public static void AddAbortCheckpoint() => Interlocked.Increment(ref s_abortCheckpoints);

        /// <summary>Zeroes every counter.</summary>
        public static void Reset()
        {
            Interlocked.Exchange(ref s_tapForwards, 0);
            Interlocked.Exchange(ref s_cudaGraphBypassed, 0);
            Interlocked.Exchange(ref s_vulkanFusedDecodeBypassed, 0);
            Interlocked.Exchange(ref s_packedBatchFallbacks, 0);
            Interlocked.Exchange(ref s_vulkanRowSliceBypassed, 0);
            Interlocked.Exchange(ref s_abortedForwards, 0);
            Interlocked.Exchange(ref s_abortCheckpoints, 0);
        }
    }
}
