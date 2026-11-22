using System.Buffers;
using DotLLM.Core.Diagnostics;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-model hidden-state tap machinery shared by the three Vulkan hosts (#1920). A forward that has a tap attached asks
/// <see cref="Begin"/> for a plan: every subscribed (point, layer) site gets a slice of one host-visible staging buffer, the model
/// records a device-to-staging copy of the subscribed rows at the site, and the samples are delivered after the GPU work completes
/// (<see cref="DeliverDeferred"/>, read-only taps) or at the site itself after a submit-and-wait (<see cref="DeliverImmediate"/>,
/// any abortable tap attached, or a deferred plan that would not fit <see cref="MaxDeferredBytes"/>). With no tap attached a forward
/// reads <see cref="Registry"/><c>.Current</c> once into a local, finds it <see langword="null"/>, and does nothing else.
/// </summary>
internal sealed class VulkanHiddenStateTaps
{
    /// <summary>Largest staging buffer a deferred plan may use; a bigger plan delivers site by site instead.</summary>
    internal const long MaxDeferredBytes = 64L * 1024 * 1024;

    private readonly struct Site(HookPoint point, int layer, int rows, long offset)
    {
        public readonly HookPoint Point = point;
        public readonly int Layer = layer;
        public readonly int Rows = rows;
        public readonly long Offset = offset;
    }

    private readonly VulkanDevice _device;
    private readonly int _hidden;
    private readonly int _numLayers;
    private readonly List<Site> _sites = [];
    private VulkanDevice.Buffer? _staging;
    private long _stagingBytes;
    private float[] _host = [];

    // Per-forward plan.
    private HiddenStateTapRegistry.Snapshot? _snap;
    private int _seqLen;
    private long _cursor;
    private int _step;

    /// <summary>Creates the tap state for a model with <paramref name="numLayers"/> layers of width <paramref name="hidden"/>.</summary>
    public VulkanHiddenStateTaps(VulkanDevice device, int numLayers, int hidden)
    {
        _device = device;
        _numLayers = numLayers;
        _hidden = hidden;
        Registry = new HiddenStateTapRegistry(numLayers);
    }

    /// <summary>The registry a forward reads once.</summary>
    public HiddenStateTapRegistry Registry { get; }

    /// <summary>Sequence id reported in samples (a per-sequence ForwardBatch sets it per request).</summary>
    public int SeqId { get; set; }

    /// <summary>True when the current plan delivers each site right after a submit-and-wait.</summary>
    public bool Immediate { get; private set; }

    /// <summary>The host-visible staging buffer (valid after <see cref="Begin"/>).</summary>
    public VulkanDevice.Buffer Staging => _staging!;

    /// <summary>Test hook: forces the immediate plan even for read-only taps (proves the bounded-staging fallback).</summary>
    internal static bool ForceImmediateForTest { get; set; }

    /// <summary>
    /// Plans one forward of <paramref name="seqLen"/> rows and sizes the staging buffer. Call BEFORE recording: growing the staging
    /// buffer invalidates descriptor caches, which must never happen against an open command buffer.
    /// </summary>
    /// <returns><see langword="true"/> when the staging buffer was re-allocated (the caller invalidates its descriptor caches).</returns>
    public bool Begin(HiddenStateTapRegistry.Snapshot snap, int seqLen)
    {
        HiddenStateTapRegistry.PathCounters.AddTapForward();
        _snap = snap;
        _seqLen = seqLen;
        _cursor = 0;
        _sites.Clear();
        _step++;

        long rowBytes = (long)_hidden * sizeof(float);
        long total = 0, largest = 0;
        void Count(HookPoint p, int layer)
        {
            int rows = Rows(p, layer);
            long bytes = rows * rowBytes;
            total += bytes;
            if (bytes > largest) largest = bytes;
        }
        Count(HookPoint.PostEmbedding, -1);
        for (int l = 0; l < _numLayers; l++) Count(HookPoint.PostLayer, l);
        Count(HookPoint.PreFinalNorm, -1);
        Count(HookPoint.PreLmHead, -1);

        Immediate = snap.HasAbortable || total > MaxDeferredBytes || ForceImmediateForTest;
        long need = Immediate ? largest : total;
        if (_staging is not null && _stagingBytes >= need) return false;
        if (need == 0) return false;
        _staging?.Dispose();
        _staging = _device.AllocateHostReadback(need);
        _stagingBytes = need;
        return true;
    }

    /// <summary>Rows a site copies: 0 when no tap wants it, the whole forward when any wanting tap asked for all rows, else 1 (the last).</summary>
    public int Rows(HookPoint point, int layer)
    {
        var snap = _snap!;
        if (!snap.Wants(point, layer)) return 0;
        return snap.WantsAllRows(point, layer) ? _seqLen : 1;
    }

    /// <summary>Reserves staging for a site and returns its byte offset (0 in immediate mode: one slot, reused).</summary>
    public long Reserve(HookPoint point, int layer, int rows)
    {
        if (Immediate)
        {
            _sites.Clear();
            _cursor = 0;
        }
        long offset = _cursor;
        _sites.Add(new Site(point, layer, rows, offset));
        _cursor += (long)rows * _hidden * sizeof(float);
        return offset;
    }

    /// <summary>
    /// Records a copy of <paramref name="rows"/> rows of <paramref name="src"/> (row-major, hidden-wide) starting at row
    /// <paramref name="srcRow"/> into staging at <paramref name="dstOffset"/>, fenced by full barriers on both sides.
    /// </summary>
    public void RecordCopy(nint cmd, VulkanDevice.Buffer src, int srcRow, int rows, long dstOffset)
    {
        long rowBytes = (long)_hidden * sizeof(float);
        long srcOffset = (long)srcRow * rowBytes;
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var dst = _staging!;
        _device.ActiveHazards?.OnTransfer(src.Handle, dst.Handle);
        var region = new VkBufferCopy { srcOffset = (ulong)srcOffset, dstOffset = (ulong)dstOffset, size = (ulong)(rows * rowBytes) };
        VulkanApi.vkCmdCopyBuffer(cmd, src.Handle, dst.Handle, 1, region);
        KernelSupport.TransferToHostBarrier(cmd);
        KernelSupport.ComputeTransferFullBarrier(cmd);   // a later compute write to <src> must wait for this read
    }

    /// <summary>
    /// Immediate mode, after the site's copy was submitted and waited: downloads the slot and dispatches it.
    /// </summary>
    /// <returns><see langword="false"/> when an abortable tap aborted the forward.</returns>
    public bool DeliverImmediate(ReadOnlySpan<int> positions, out ForwardAbortInfo abort)
    {
        abort = default;
        var site = _sites[^1];
        int floats = site.Rows * _hidden;
        EnsureHost(floats);
        _device.Download(_staging!, _host.AsSpan(0, floats));
        return Dispatch(site, 0, positions, out abort);
    }

    /// <summary>Deferred mode, after the forward's last submit: delivers every recorded site in forward order.</summary>
    public void DeliverDeferred(ReadOnlySpan<int> positions)
    {
        if (_sites.Count == 0) return;
        int floats = checked((int)(_cursor / sizeof(float)));
        EnsureHost(floats);
        _device.Download(_staging!, _host.AsSpan(0, floats));
        foreach (var site in _sites)
        {
            Dispatch(site, (int)(site.Offset / sizeof(float)), positions, out _);
        }
    }

    private bool Dispatch(Site site, int floatOffset, ReadOnlySpan<int> positions, out ForwardAbortInfo abort)
        => _snap!.Dispatch(site.Point, site.Layer, _host.AsSpan(floatOffset, site.Rows * _hidden), site.Rows,
            site.Rows == 1 ? _seqLen - 1 : 0, _hidden, positions, SeqId, _step - 1, out abort);

    private void EnsureHost(int floats)
    {
        if (_host.Length < floats) _host = new float[Math.Max(floats, _host.Length * 2)];
    }

    /// <summary>
    /// Starts a forward for a host whose layer loop is already synchronous (the MoE host submits and waits twice per layer): no staging
    /// plan, every site is read back right where the GPU is idle through <see cref="DeliverFromDevice"/>.
    /// </summary>
    public void BeginSynchronous(HiddenStateTapRegistry.Snapshot snap, int seqLen)
    {
        HiddenStateTapRegistry.PathCounters.AddTapForward();
        _snap = snap;
        _seqLen = seqLen;
        _cursor = 0;
        _sites.Clear();
        _step++;
        Immediate = true;
    }

    /// <summary>
    /// Synchronous site: with no GPU work in flight, copies the subscribed rows of <paramref name="src"/> to the staging buffer, reads them
    /// back and dispatches them. <paramref name="srcLastRow"/> is the row a single-row (Last) tap reads; All taps read from row 0.
    /// </summary>
    /// <returns><see langword="false"/> when an abortable tap aborted the forward.</returns>
    public bool DeliverFromDevice(HookPoint point, int layer, VulkanDevice.Buffer src, int srcLastRow,
                                  ReadOnlySpan<int> positions, out ForwardAbortInfo abort)
    {
        abort = default;
        int rows = Rows(point, layer);
        if (rows == 0) return true;
        long rowBytes = (long)_hidden * sizeof(float);
        long bytes = rows * rowBytes;
        if (_staging is null || _stagingBytes < bytes)
        {
            _staging?.Dispose();
            _staging = _device.AllocateHostReadback(bytes);
            _stagingBytes = bytes;
        }
        _device.CopyBufferRangeSynchronous(src, _staging, (ulong)((rows == 1 ? srcLastRow : 0) * rowBytes), 0, (ulong)bytes);
        int floats = rows * _hidden;
        EnsureHost(floats);
        _device.Download(_staging, _host.AsSpan(0, floats));
        return _snap!.Dispatch(point, layer, _host.AsSpan(0, floats), rows, rows == 1 ? _seqLen - 1 : 0, _hidden,
            positions, SeqId, _step - 1, out abort);
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> s_logged = new();

    /// <summary>Writes <paramref name="message"/> to stderr the first time any model in the process asks for this <paramref name="kind"/> of bypass notice.</summary>
    internal static void LogOnce(string kind, string message)
    {
        if (s_logged.TryAdd(kind, 0))
            Console.Error.WriteLine("[dotllm] " + message);
    }

    /// <summary>Drops the plan's snapshot reference (end of forward).</summary>
    public void End()
    {
        _snap = null;
        _sites.Clear();
    }

    /// <summary>Releases the staging buffer.</summary>
    public void Dispose()
    {
        _staging?.Dispose();
        _staging = null;
        _stagingBytes = 0;
    }
}
