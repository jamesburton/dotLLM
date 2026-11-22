using System.Numerics.Tensors;
using DotLLM.Core.Diagnostics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Per-model hidden-state tap machinery shared by the CUDA hosts (#1920). A forward that has a tap attached calls
/// <see cref="Begin"/>; every subscribed (point, layer) site then enqueues an asynchronous device-to-host copy of the subscribed rows
/// into one pinned staging buffer on the model stream. A read-only tap's samples are delivered after the stream synchronize that
/// ends the forward (<see cref="DeliverDeferred"/>); an abortable tap, or a plan that would not fit <see cref="MaxDeferredBytes"/>,
/// synchronizes the stream at the site and delivers right there (<see cref="DeliverImmediate"/>), so a later layer is never enqueued
/// after an abort. Device data is F16 or F32 as the host stores it; it is widened to F32 on the host, and a per-site scale undoes a
/// scaled residual stream (Flint). The device is never written, and no device F32 buffer is added. With no tap attached a forward
/// reads <see cref="Registry"/><c>.Current</c> once into a local, finds it <see langword="null"/>, and does nothing else.
/// </summary>
internal sealed unsafe class CudaHiddenStateTaps : IDisposable
{
    /// <summary>Largest pinned staging buffer a deferred plan may use; a bigger plan delivers site by site instead.</summary>
    internal const long MaxDeferredBytes = 64L * 1024 * 1024;

    private const int SlotAlign = 16;

    private readonly struct Site(HookPoint point, int layer, int rows, int globalStart, long offset, int elemBytes, float scale)
    {
        public readonly HookPoint Point = point;
        public readonly int Layer = layer;
        public readonly int Rows = rows;
        public readonly int GlobalStart = globalStart;
        public readonly long Offset = offset;
        public readonly int ElemBytes = elemBytes;
        public readonly float Scale = scale;
    }

    private readonly int _hidden;
    private readonly int _numLayers;
    private readonly List<Site> _sites = [];
    private nint _pinned;
    private long _pinnedBytes;
    private float[] _host = [];
    private int[] _positions = [];

    // Per-forward plan.
    private HiddenStateTapRegistry.Snapshot? _snap;
    private int _total;       // rows of the whole forward (a tiled prefill spans several tiles)
    private long _cursor;
    private int _step;
    private bool _anyAll;

    /// <summary>Creates the tap state for a model with <paramref name="numLayers"/> layers of width <paramref name="hidden"/>.</summary>
    public CudaHiddenStateTaps(int numLayers, int hidden)
    {
        _numLayers = numLayers;
        _hidden = hidden;
        Registry = new HiddenStateTapRegistry(numLayers);
    }

    /// <summary>The registry a forward reads once.</summary>
    public HiddenStateTapRegistry Registry { get; }

    /// <summary>Sequence id reported in samples (a per-request ForwardBatch sets it per request).</summary>
    public int SeqId { get; set; }

    /// <summary>True when the current plan delivers each site right after a stream synchronize.</summary>
    public bool Immediate { get; private set; }

    /// <summary>True when any subscribed site wants every row (a tiled prefill must then run as one tile).</summary>
    public bool AnyAllRows => _anyAll;

    /// <summary>Test hook: forces the immediate plan even for read-only taps (proves the bounded-staging fallback).</summary>
    internal static bool ForceImmediateForTest { get; set; }

    /// <summary>
    /// Plans one whole forward of <paramref name="positions"/>.Length rows and sizes the pinned staging buffer. The CUDA context must be
    /// current. Call BEFORE enqueueing: the previous forward ended in a stream synchronize, so replacing the buffer is safe here.
    /// </summary>
    public void Begin(HiddenStateTapRegistry.Snapshot snap, ReadOnlySpan<int> positions)
    {
        HiddenStateTapRegistry.PathCounters.AddTapForward();
        _snap = snap;
        _total = positions.Length;
        if (_positions.Length < _total) _positions = new int[Math.Max(_total, _positions.Length * 2)];
        positions.CopyTo(_positions);
        _cursor = 0;
        _sites.Clear();
        _step++;

        long rowBytes = (long)_hidden * sizeof(float);   // upper bound: an F16 site stages half of it
        long total = 0, largest = 0;
        _anyAll = false;
        void Count(HookPoint p, int layer)
        {
            if (!snap.Wants(p, layer)) return;
            bool all = snap.WantsAllRows(p, layer);
            _anyAll |= all;
            long bytes = (all ? _total : 1) * rowBytes + SlotAlign;
            total += bytes;
            if (bytes > largest) largest = bytes;
        }
        Count(HookPoint.PostEmbedding, -1);
        for (int l = 0; l < _numLayers; l++) Count(HookPoint.PostLayer, l);
        Count(HookPoint.PreFinalNorm, -1);
        Count(HookPoint.PreLmHead, -1);

        Immediate = snap.HasAbortable || total > MaxDeferredBytes || ForceImmediateForTest;
        long need = Immediate ? largest : total;
        if (need == 0 || _pinnedBytes >= need) return;
        FreePinned();
        CudaDriverApi.cuMemHostAlloc(out _pinned, (nuint)need, 0).ThrowOnError();
        _pinnedBytes = need;
    }

    /// <summary>
    /// Whether a site at (<paramref name="point"/>, <paramref name="layer"/>) copies anything for this tile, and which rows. A Last-only
    /// site fires in the final tile only (one row); an All site copies the whole tile (a plan with an All site is never tiled).
    /// </summary>
    public bool Plan(HookPoint point, int layer, bool finalTile, int tileLen, out int srcRow, out int rows)
    {
        srcRow = 0; rows = 0;
        var snap = _snap!;
        if (!snap.Wants(point, layer)) return false;
        if (snap.WantsAllRows(point, layer))
        {
            rows = tileLen;
            return true;
        }
        if (!finalTile) return false;
        srcRow = tileLen - 1;
        rows = 1;
        return true;
    }

    /// <summary>True when a tap wants rows of <paramref name="point"/> at <paramref name="layer"/> (the cheap pre-check before any device work).</summary>
    public bool Wants(HookPoint point, int layer) => _snap!.Wants(point, layer);

    /// <summary>True when a tap wants every row at the site.</summary>
    public bool WantsAll(HookPoint point, int layer) => _snap!.WantsAllRows(point, layer);

    /// <summary>
    /// Enqueues the copy of <paramref name="rows"/> rows of a device buffer (row-major, hidden-wide, <paramref name="elemBytes"/> per element,
    /// first row at <paramref name="srcBase"/>) starting at row <paramref name="srcRow"/>. Host values are multiplied by <paramref name="scale"/>.
    /// </summary>
    public void Enqueue(HookPoint point, int layer, nint srcBase, int srcRow, int rows, int elemBytes, float scale, nint stream)
    {
        if (Immediate)
        {
            _sites.Clear();
            _cursor = 0;
        }
        long rowBytes = (long)_hidden * elemBytes;
        long offset = _cursor;
        long bytes = rows * rowBytes;
        _sites.Add(new Site(point, layer, rows, rows == 1 ? _total - 1 : 0, offset, elemBytes, scale));
        CudaDriverApi.cuMemcpyDtoHAsync_v2(_pinned + (nint)offset, srcBase + (nint)(srcRow * rowBytes), (nuint)bytes, stream).ThrowOnError();
        _cursor = (offset + bytes + SlotAlign - 1) & ~(long)(SlotAlign - 1);
    }

    /// <summary>
    /// One tap site: enqueues the copy when a tap wants it and, in immediate mode, waits and delivers at once.
    /// </summary>
    /// <returns><see langword="false"/> when an abortable tap aborted the forward (nothing later may be enqueued).</returns>
    public bool TapSite(HookPoint point, int layer, bool finalTile, int tileLen, nint srcBase, int elemBytes, float scale,
                     nint stream, out ForwardAbortInfo abort)
    {
        abort = default;
        if (!Plan(point, layer, finalTile, tileLen, out int srcRow, out int rows)) return true;
        Enqueue(point, layer, srcBase, srcRow, rows, elemBytes, scale, stream);
        return !Immediate || DeliverImmediate(stream, out abort);
    }

    /// <summary>
    /// Immediate mode: waits for the stream, then dispatches the site just enqueued.
    /// </summary>
    /// <returns><see langword="false"/> when an abortable tap aborted the forward.</returns>
    public bool DeliverImmediate(nint stream, out ForwardAbortInfo abort)
    {
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        return Dispatch(_sites[^1], out abort);
    }

    /// <summary>Deferred mode, after the forward's stream synchronize: delivers every recorded site in forward order.</summary>
    public void DeliverDeferred()
    {
        foreach (var site in _sites) Dispatch(site, out _);
    }

    private bool Dispatch(Site site, out ForwardAbortInfo abort)
    {
        int n = site.Rows * _hidden;
        if (_host.Length < n) _host = new float[Math.Max(n, _host.Length * 2)];
        var dst = _host.AsSpan(0, n);
        byte* src = (byte*)_pinned + site.Offset;
        if (site.ElemBytes == sizeof(ushort))
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, n), dst);
        else
            new ReadOnlySpan<float>(src, n).CopyTo(dst);
        if (site.Scale != 1f) TensorPrimitives.Multiply(dst, site.Scale, dst);
        return _snap!.Dispatch(site.Point, site.Layer, dst, site.Rows, site.GlobalStart, _hidden,
            _positions.AsSpan(0, _total), SeqId, _step - 1, out abort);
    }

    /// <summary>Takes the entry KV length for an abortable tap, after proving the cache can truncate (a no-op rollback is the probe).</summary>
    internal static int KvCheckpoint(DotLLM.Core.Attention.IKvCache? kvCache, ReadOnlySpan<int> positions)
        => DotLLM.Core.Diagnostics.AbortableTapKvGuard.Checkpoint(kvCache, positions);

    private static int s_logged;

    /// <summary>Writes <paramref name="message"/> to stderr the first time any model in the process asks (the bypass notices).</summary>
    internal static void LogOnce(string message)
    {
        if (Interlocked.Exchange(ref s_logged, 1) == 0)
            Console.Error.WriteLine("[dotllm] " + message);
    }

    /// <summary>Drops the plan's snapshot reference (end of forward).</summary>
    public void End()
    {
        _snap = null;
        _sites.Clear();
    }

    private void FreePinned()
    {
        if (_pinned != 0) CudaDriverApi.cuMemFreeHost(_pinned);
        _pinned = 0;
        _pinnedBytes = 0;
    }

    /// <summary>Releases the pinned staging buffer.</summary>
    public void Dispose() => FreePinned();
}
