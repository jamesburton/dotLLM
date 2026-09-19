using System.Diagnostics;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-op-category attribution for one Vulkan forward pass (issue #434).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why not host wall time alone.</b> A hybrid forward records many dispatches into one
/// command buffer per layer and submits once; a host <see cref="Stopwatch"/> around the
/// submit can only say how long the whole layer took. Attribution therefore comes from
/// <c>BOTTOM_OF_PIPE</c> timestamps written at category boundaries: the delta between
/// consecutive stamps is the GPU time of every command recorded between them, regardless of
/// where barriers sit.
/// </para>
/// <para>
/// <b>Three time axes are reported, and they are not interchangeable.</b>
/// <list type="bullet">
/// <item><description><c>gpu</c> — summed timestamp deltas, split by category. This is the
/// attribution table.</description></item>
/// <item><description><c>record</c> / <c>wait</c> — host wall time spent recording a command
/// buffer and blocked in <c>vkWaitForFences</c>. <c>wait</c> overlaps GPU execution;
/// <c>record</c> does not.</description></item>
/// <item><description><c>wall</c> — the whole forward. <c>wall − attributed</c> is the
/// host/submission remainder, and it is reported explicitly rather than folded into a bucket,
/// because a large remainder would mean the cost is submission structure and not any
/// kernel.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Split mode</b> (<see cref="SplitSubmits"/>) is an independent cross-check: instead of
/// timestamps it submits-and-waits at every category boundary and charges the elapsed host time
/// to that category. It over-counts (each split pays a full pipeline drain) but it cannot be
/// fooled by a mis-tagged query, so broad agreement between the two modes is evidence the
/// timestamp tagging is right.
/// </para>
/// <para>Zero cost when disabled: the model checks one <see langword="bool"/> per forward.</para>
/// </remarks>
internal sealed class VulkanOpProfiler : IDisposable
{
    /// <summary>Op categories. Order fixes the reporting order.</summary>
    internal enum Cat : byte
    {
        /// <summary>Anything recorded before the first mark of a submit — normally nothing.</summary>
        Other = 0,

        /// <summary>Token-embedding gather (or row copy).</summary>
        Embed,

        /// <summary>RMS norms (attention, post-attention, q/k, final).</summary>
        Norm,

        /// <summary>PrismML FWHT activation rotations.</summary>
        Hadamard,

        /// <summary>Q/gate, K, V and O projections on full-attention layers.</summary>
        ProjAttn,

        /// <summary>qkv / gate / alpha / beta / out projections on GDN layers.</summary>
        ProjGdn,

        /// <summary>Dense FFN gate / up / down projections.</summary>
        ProjFfn,

        /// <summary>SwiGLU activation.</summary>
        FfnAct,

        /// <summary>GDN pre-scan ops: decay, sigmoid, causal conv1d, SiLU, L2 norms, state copies.</summary>
        GdnPre,

        /// <summary>The GDN recurrent scan and its post-scan gate.</summary>
        GdnScan,

        /// <summary>RoPE, KV-cache update, the attention kernel, and the sigmoid output gate.</summary>
        Attention,

        /// <summary>Residual adds and the hidden-state buffer copies around them.</summary>
        Resid,

        /// <summary>
        /// The per-token / per-head <c>vkCmdCopyBuffer</c> fan-out loops: the fused Q+gate
        /// de-interleave, the GDN conv-input build and the GDN q/k/v split. These grow as
        /// O(seqLen) (and O(seqLen x numHeads) for Q+gate), so they are kept out of the
        /// projection and GDN buckets — otherwise a copy-bound prefill would read as a
        /// matmul-bound one.
        /// </summary>
        CopyFanout,

        /// <summary>Final LM-head projection.</summary>
        LmHead,
    }

    /// <summary>Reporting names, indexed by <see cref="Cat"/>.</summary>
    internal static readonly string[] CategoryNames =
    {
        "other", "embed", "norm", "hadamard", "proj_attn", "proj_gdn", "proj_ffn",
        "ffn_act", "gdn_pre", "gdn_scan", "attention", "resid", "copy_fanout", "lm_head",
    };

    // 64 layers x ~14 marks would overflow a per-forward pool, so the pool is reset and
    // collected per SUBMIT (one submit per layer here) — ~20 queries in flight at a time.
    private const int MaxQueries = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanDevice.SubmitContext _submit;

    private nint _queryPool;
    private bool _queryPoolFailed;
    private float _tsPeriodNs;
    private int _queryCount;
    private readonly byte[] _queryCats = new byte[MaxQueries];
    private readonly ulong[] _tsScratch = new ulong[MaxQueries];

    private readonly double[] _gpuMsByCat = new double[CategoryNames.Length];
    private readonly double[] _splitMsByCat = new double[CategoryNames.Length];
    private readonly Dictionary<string, int> _dispatches = new(StringComparer.Ordinal);

    private double _recordMs, _waitMs;
    private long _wallStart, _phaseStart;
    private int _seqLen, _layers;
    private bool _active;
    private bool _inSubmit;

    /// <summary>True while this profiler is attributing a forward pass.</summary>
    public bool Active => _active;

    /// <summary>
    /// When true the profiler splits the command buffer at every mark (submit + wait + re-begin)
    /// and charges host wall time instead of writing timestamps. See the class remarks.
    /// </summary>
    public bool SplitSubmits { get; init; }

    /// <summary>Creates a profiler bound to a device and the model's submit context.</summary>
    /// <param name="device">Device owning the query pool.</param>
    /// <param name="submit">The model's per-forward submit context (used by split mode only).</param>
    public VulkanOpProfiler(VulkanDevice device, VulkanDevice.SubmitContext submit)
    {
        _device = device;
        _submit = submit;
    }

    /// <summary>Starts attributing a forward pass. Call at the top of <c>Forward</c>.</summary>
    /// <param name="seqLen">Tokens in this forward.</param>
    /// <param name="layers">Model layer count (reported only).</param>
    public void BeginForward(int seqLen, int layers)
    {
        _active = true;
        _seqLen = seqLen;
        _layers = layers;
        Array.Clear(_gpuMsByCat);
        Array.Clear(_splitMsByCat);
        _dispatches.Clear();
        _recordMs = _waitMs = 0;
        _queryCount = 0;
        _inSubmit = false;
        _wallStart = _phaseStart = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Records the per-submit query-pool reset and the baseline stamp. Call immediately after
    /// <c>_submit.Begin()</c> and the opening barrier.
    /// </summary>
    /// <param name="cmdBuf">The freshly opened command buffer.</param>
    public void BeginSubmit(nint cmdBuf)
    {
        if (!_active) return;
        _phaseStart = Stopwatch.GetTimestamp();
        _inSubmit = true;
        if (SplitSubmits || !EnsureQueryPool()) return;
        VulkanApi.vkCmdResetQueryPool(cmdBuf, _queryPool, 0, MaxQueries);
        _queryCount = 0;
        // Baseline: query 0 is the start of this submit's timeline. Deltas are consumed from
        // index 1 onward, so this stamp's tag is never charged to a bucket.
        Stamp(cmdBuf, Cat.Other);
    }

    /// <summary>
    /// Category boundary. Attributes everything recorded (or, in split mode, executed) since the
    /// previous mark to <paramref name="cat"/>.
    /// </summary>
    /// <param name="cmdBuf">The open command buffer.</param>
    /// <param name="cat">Category the just-recorded work belongs to.</param>
    public void Mark(nint cmdBuf, Cat cat)
    {
        if (!_active) return;
        if (SplitSubmits)
        {
            // A full drain per boundary: the elapsed host time IS this category's GPU time,
            // plus one pipeline-drain tax that the remainder row makes visible.
            long t0 = Stopwatch.GetTimestamp();
            _submit.SubmitAndWait();
            long t1 = Stopwatch.GetTimestamp();
            _splitMsByCat[(int)cat] += Ms(t0, t1);
            _submit.Begin();
            KernelSupport.HostToComputeBarrier(_submit.CommandBuffer);
            return;
        }

        Stamp(cmdBuf, cat);
    }

    /// <summary>
    /// Charges the host record phase and closes the submit's timeline. Call immediately before
    /// the model's <c>SubmitAndWait</c>.
    /// </summary>
    /// <param name="cmdBuf">The command buffer about to be submitted.</param>
    /// <remarks>
    /// The closing stamp is tagged <see cref="Cat.Other"/>, which is what makes the accounting
    /// test discriminating: <c>other</c> collects any work recorded after the submit's last
    /// category mark, so a forgotten tail mark shows up as a non-zero <c>other</c> row rather
    /// than silently vanishing from the table.
    /// </remarks>
    public void BeforeSubmit(nint cmdBuf)
    {
        if (!_active || !_inSubmit) return;
        Stamp(cmdBuf, Cat.Other);
        long now = Stopwatch.GetTimestamp();
        _recordMs += Ms(_phaseStart, now);
        _phaseStart = now;
    }

    /// <summary>
    /// Completes a submit: charges the fence wait and reads back this submit's timestamps.
    /// Call immediately after the model's <c>SubmitAndWait</c>.
    /// </summary>
    public void AfterSubmit()
    {
        if (!_active || !_inSubmit) return;
        long now = Stopwatch.GetTimestamp();
        _waitMs += Ms(_phaseStart, now);
        _phaseStart = now;
        _inSubmit = false;
        CollectGpu();
    }

    /// <summary>Census of which kernel variant ran at which shape — proves the fast path fired.</summary>
    /// <param name="kernel">Kernel/shader identity, including the selected variant.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Reduction extent.</param>
    /// <param name="n">Token columns.</param>
    public void Note(string kernel, int m, int k, int n)
    {
        if (!_active) return;
        string key = $"{kernel} m={m} k={k} n={n}";
        _dispatches.TryGetValue(key, out int c);
        _dispatches[key] = c + 1;
    }

    /// <summary>Ends attribution and returns the report.</summary>
    /// <returns>The per-category attribution for the forward just completed.</returns>
    public VulkanOpProfileReport EndForward()
    {
        _active = false;
        double wallMs = Ms(_wallStart, Stopwatch.GetTimestamp());
        var byCat = new Dictionary<string, double>(StringComparer.Ordinal);
        double[] src = SplitSubmits ? _splitMsByCat : _gpuMsByCat;
        double total = 0;
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] <= 0) continue;
            byCat[CategoryNames[i]] = src[i];
            total += src[i];
        }

        return new VulkanOpProfileReport(
            _seqLen, _layers, wallMs, total, _recordMs, _waitMs,
            byCat, new Dictionary<string, int>(_dispatches, StringComparer.Ordinal),
            SplitSubmits ? "split" : _queryPool != 0 ? "gpu-timestamp" : "unavailable");
    }

    private bool EnsureQueryPool()
    {
        if (_queryPool != 0) return true;
        if (_queryPoolFailed) return false;
        _tsPeriodNs = _device.TimestampPeriodNs;
        if (_tsPeriodNs <= 0f)
        {
            _queryPoolFailed = true;
            return false;
        }

        var qci = new VkQueryPoolCreateInfo
        {
            sType = 11, // VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
            queryType = 2, // VK_QUERY_TYPE_TIMESTAMP
            queryCount = MaxQueries,
        };
        if (VulkanApi.vkCreateQueryPool(_device.Handle, qci, 0, out _queryPool) < 0 || _queryPool == 0)
        {
            _queryPool = 0;
            _queryPoolFailed = true;
            return false;
        }

        return true;
    }

    private void Stamp(nint cmdBuf, Cat cat)
    {
        if (_queryPool == 0 || _queryCount >= MaxQueries) return;
        _queryCats[_queryCount] = (byte)cat;
        VulkanApi.vkCmdWriteTimestamp(cmdBuf, VkPipelineStageFlags.BottomOfPipe,
            _queryPool, (uint)_queryCount++);
    }

    private unsafe void CollectGpu()
    {
        if (SplitSubmits || _queryPool == 0 || _queryCount < 2) return;
        fixed (ulong* p = _tsScratch)
        {
            // The fence is already signalled, so WAIT (0x1) | 64_BIT (0x2) returns immediately.
            if (VulkanApi.vkGetQueryPoolResults(_device.Handle, _queryPool, 0,
                    (uint)_queryCount, (nuint)(_queryCount * sizeof(ulong)),
                    (nint)p, sizeof(ulong), flags: 0x1 | 0x2) < 0)
                return;
        }

        double toMs = _tsPeriodNs / 1_000_000.0;
        for (int i = 1; i < _queryCount; i++)
            _gpuMsByCat[_queryCats[i]] += (_tsScratch[i] - _tsScratch[i - 1]) * toMs;
        _queryCount = 0;
    }

    private static double Ms(long from, long to)
        => (to - from) * 1000.0 / Stopwatch.Frequency;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_queryPool != 0)
        {
            VulkanApi.vkDestroyQueryPool(_device.Handle, _queryPool, 0);
            _queryPool = 0;
        }
    }
}

/// <summary>
/// One forward pass's per-op attribution (issue #434). See <see cref="VulkanOpProfiler"/> for
/// what each time axis means and why they are kept separate.
/// </summary>
/// <param name="SeqLen">Tokens in the profiled forward.</param>
/// <param name="Layers">Model layer count.</param>
/// <param name="WallMs">Whole-forward host wall time.</param>
/// <param name="AttributedMs">Sum of the per-category times — the denominator for shares.</param>
/// <param name="RecordMs">Host time spent recording command buffers.</param>
/// <param name="WaitMs">Host time blocked on submit fences (overlaps GPU execution).</param>
/// <param name="ByCategory">Per-category time, keyed by <see cref="VulkanOpProfiler.CategoryNames"/>.</param>
/// <param name="Dispatches">Kernel-variant census, keyed by <c>"{kernel} m= k= n="</c>.</param>
/// <param name="Mode">"gpu-timestamp", "split", or "unavailable".</param>
internal sealed record VulkanOpProfileReport(
    int SeqLen,
    int Layers,
    double WallMs,
    double AttributedMs,
    double RecordMs,
    double WaitMs,
    IReadOnlyDictionary<string, double> ByCategory,
    IReadOnlyDictionary<string, int> Dispatches,
    string Mode)
{
    /// <summary>Wall time not covered by any category — host overhead plus submission structure.</summary>
    public double UnattributedMs => WallMs - AttributedMs;

    /// <summary>Renders the attribution table, shares first.</summary>
    /// <param name="tag">Line prefix, e.g. <c>hybrid-profile</c>.</param>
    /// <returns>A multi-line report ending in a newline.</returns>
    public string Format(string tag)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"[{tag}] mode={Mode} seqLen={SeqLen} layers={Layers} " +
            $"wall_ms={WallMs:F1} attributed_ms={AttributedMs:F1} " +
            $"record_ms={RecordMs:F1} wait_ms={WaitMs:F1}");
        foreach (var kv in ByCategory.OrderByDescending(kv => kv.Value))
        {
            sb.AppendLine($"[{tag}]   {kv.Key,-12} {kv.Value,9:F1} ms  " +
                $"({kv.Value / Math.Max(AttributedMs, 1e-9) * 100.0,5:F1} % attributed, " +
                $"{kv.Value / Math.Max(WallMs, 1e-9) * 100.0,5:F1} % wall)");
        }

        sb.AppendLine($"[{tag}]   {"UNATTRIBUTED",-12} {UnattributedMs,9:F1} ms  " +
            $"({UnattributedMs / Math.Max(WallMs, 1e-9) * 100.0,5:F1} % wall)");
        foreach (var kv in Dispatches.OrderByDescending(kv => kv.Value))
            sb.AppendLine($"[{tag}]   dispatch {kv.Key} x{kv.Value}");
        return sb.ToString();
    }
}
