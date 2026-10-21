using System.Diagnostics;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Routed <c>nemotron_h_moe</c> experts on an integrated GPU, for a host that runs everything else elsewhere (issue #1083, design #1063
/// section 8c′): the iGPU cold leg of the CUDA expert offload. It computes, for a set of (expert, input row) pairs of one layer,
/// <c>down(relu²(up(x)))</c> per row, unweighted, into host-visible memory, and knows nothing about CUDA, so it runs (and is validated)
/// on any Vulkan device, e.g. Strix Halo's Radeon 8060S on its own.
/// <list type="bullet">
///   <item><b>Banks</b>: each layer's up / down expert banks are imported <b>zero-copy</b> from the host pointers (the memory-mapped GGUF
///   slices; <see cref="VulkanWeightImportPolicy"/>), so on UMA the iGPU reads the very pages the CPU leg reads. A refused import falls
///   back to a staged copy within a byte budget and says so in the log; past the budget, or for a bank type the Vulkan expert kernels
///   cannot read, the layer is refused (<see cref="Supports"/> is false and the caller keeps it on the CPU). With the GGUF mapped
///   read-only (the default) drivers refuse the import, so every bank is staged: a second copy in the same RAM
///   (<see cref="DescribeStagedBanks"/>; the first quiet window on the dev box's Arc, 2026-10-21).</item>
///   <item><b>Kernels</b>: the routed-expert kernels of <see cref="VulkanNemotronHTransformerModel"/> (its <c>MoeKernelSet</c>): the
///   indexed dp4a MMVQ (or F32) arm for small calls, the expert-grouped MMQ / coopmat prefill from
///   <see cref="VulkanNemotronHTransformerModel.MoeGroupedMinTokens"/> rows, the IQ banks of #1057. Recorded through the model's own
///   (static) recording code into this service's scratch, with top-k 1 and unit weights, so every row comes back unweighted.</item>
///   <item><b>I/O</b>: the input rows (gathered by the caller's token index) and the output rows live in host memory the device reads
///   and writes directly: an imported host allocation where the device allows it, else a persistently mapped host-visible buffer.</item>
///   <item><b>Sync</b>: <see cref="Begin"/> records and submits (no wait); <see cref="End"/> waits on the fence and copies the rows out.
///   The caller does other work (the CPU leg) in between.</item>
/// </list>
/// Not thread-safe: one submission at a time, from one thread.
/// </summary>
internal sealed unsafe class VulkanNemotronHMoeExpertService : IDisposable
{
    /// <summary>Default share of physical RAM a refused import may copy into device memory instead (on UMA that is RAM too).</summary>
    public const double DefaultStagingBudgetFraction = 0.10;

    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly VulkanDevice.SubmitContext _submit;
    private readonly VulkanNemotronHTransformerModel.MoeKernelSet _kernels;
    private readonly ReluSquaredInplaceF32Kernel _relu;
    private readonly Iq2Codebooks _iq2;
    private readonly Iq3Codebooks _iq3;
    private readonly Dictionary<int, LayerBanks> _layers = [];
    private readonly Dictionary<int, string> _refused = [];
    private readonly List<VulkanDevice.Buffer> _bankBuffers = [];
    private readonly Scratch _scratch;
    private readonly HostRows _input, _output, _indices;
    private readonly int _maxRows;

    // The submission in flight (between Begin and End), and the fence-before-read rule of End.
    private readonly FencedHostRows _fenced;
    private bool _disposed;

    private readonly record struct LayerBanks(VulkanNemotronHTransformerModel.MoeRoutedBanks Banks, int ExpertDim, NemotronHMoeWeights Host);

    /// <summary>The Vulkan device the experts run on.</summary>
    public VulkanDevice Device => _device;

    /// <summary>A short identity of the device for logs and the calibration key (name and Vulkan device type).</summary>
    public string DeviceDescription => $"{_device.DeviceName} (vk type {_device.DeviceType})";

    /// <summary>
    /// Most rows (expert, token pairs) one submission may carry: the host-visible input / output and the routed scratch are sized
    /// for it at creation. Size it with <see cref="MaxRowsFor"/>; the service does not split a larger call.
    /// </summary>
    public int MaxRows => _maxRows;

    /// <summary>Banks imported zero-copy.</summary>
    public int ImportedBanks { get; }

    /// <summary>Banks copied into device memory because the import was refused (or a Q5_0 repack).</summary>
    public int StagedBanks { get; }

    /// <summary>Bytes of the staged banks.</summary>
    public long StagedBytes { get; }

    /// <summary>Microseconds from the last <see cref="Begin"/> until <see cref="End"/> observed its fence.</summary>
    public double LastComputeUs => _fenced.LastComputeUs;

    /// <summary>Microseconds the last <see cref="End"/> blocked on the fence.</summary>
    public double LastWaitUs => _fenced.LastWaitUs;

    /// <summary>True between <see cref="Begin"/> and <see cref="End"/>.</summary>
    public bool InFlight => _fenced.InFlight;

    private VulkanNemotronHMoeExpertService(VulkanDevice device, bool ownsDevice, string spvDir,
        IReadOnlyList<(int Layer, NemotronHMoeWeights Moe)> layers, int hiddenSize, int maxRows, long stagingBudgetBytes, Action<string> log)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        _maxRows = maxRows;
        _fenced = new FencedHostRows(() => _submit!.WaitFence());   // _submit is set below, before any submission
        bool ok = false;
        try
        {
            _submit = device.CreateSubmitContext();
            _relu = ReluSquaredInplaceF32Kernel.Create(device, spvDir);
            _iq2 = Iq2Codebooks.Create(device);
            _iq3 = Iq3Codebooks.Create(device);

            // Banks: import zero-copy, else stage within the budget, else refuse the layer.
            long staged = 0;
            int importedBanks = 0, stagedBanks = 0;
            int maxExperts = 1, maxInter = 1, maxExpertDim = 1;
            using (var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024))
            {
                foreach (var (layer, moe) in layers)
                {
                    int expertDim = NemotronHMoeExpertCompute.ExpertDim(moe, hiddenSize);
                    VulkanDevice.Buffer? up = null, down = null;
                    try
                    {
                        long stagedBefore = staged;
                        bool MayStage(long bytes)
                        {
                            if (staged + bytes > stagingBudgetBytes) return false;
                            staged += bytes;
                            return true;
                        }
                        up = VulkanNemotronHWeights.UploadExpertBank(device, staging, moe.UpExpsWeight, moe.UpExpsQuantType, moe.NumExperts,
                            moe.MoeIntermediateSize, expertDim, $"blk.{layer}.ffn_up_exps", out var upQt, out _, MayStage);
                        down = VulkanNemotronHWeights.UploadExpertBank(device, staging, moe.DownExpsWeight, moe.DownExpsQuantType, moe.NumExperts,
                            expertDim, moe.MoeIntermediateSize, $"blk.{layer}.ffn_down_exps", out var downQt, out _, MayStage);
                        foreach (var b in new[] { up, down })
                        {
                            if (b.IsHostImported) importedBanks++;
                            else stagedBanks++;
                        }
                        if (staged > stagedBefore)
                            log(DescribeStagedBanks(layer, VulkanWeightImportPolicy.LastFallbackReason, staged - stagedBefore,
                                staged, stagingBudgetBytes, GgufFile.MappingAccess == MemoryMappedFileAccess.Read));
                        _bankBuffers.Add(up);
                        _bankBuffers.Add(down);
                        _layers[layer] = new LayerBanks(
                            new VulkanNemotronHTransformerModel.MoeRoutedBanks(up, upQt, down, downQt, moe.NumExperts, moe.MoeIntermediateSize),
                            expertDim, moe);
                        maxExperts = Math.Max(maxExperts, moe.NumExperts);
                        maxInter = Math.Max(maxInter, moe.MoeIntermediateSize);
                        maxExpertDim = Math.Max(maxExpertDim, expertDim);
                    }
                    catch (NotSupportedException ex)
                    {
                        up?.Dispose();
                        down?.Dispose();
                        _refused[layer] = ex.Message;
                        log($"MoE iGPU: REFUSED blk.{layer} ({moe.UpExpsQuantType} / {moe.DownExpsQuantType} banks): {ex.Message} " +
                            "Its cold experts stay on the CPU.");
                    }
                }
            }
            ImportedBanks = importedBanks;
            StagedBanks = stagedBanks;
            StagedBytes = staged;

            var iqFormats = new HashSet<MoeIqQuant>();
            foreach (var l in _layers.Values)
            {
                if (MoeIqFormats.FromQuantizationType(l.Banks.UpQt) is { } a) iqFormats.Add(a);
                if (MoeIqFormats.FromQuantizationType(l.Banks.DownQt) is { } b) iqFormats.Add(b);
            }
            _kernels = VulkanNemotronHTransformerModel.CreateMoeKernels(device, spvDir, maxExperts, iqFormats, _iq2, _iq3);
            _scratch = new Scratch(device, maxRows, maxExpertDim, maxInter, maxExperts);
            _input = HostRows.Create(device, (long)maxRows * maxExpertDim * sizeof(float), readback: false);
            _output = HostRows.Create(device, (long)maxRows * maxExpertDim * sizeof(float), readback: true);
            _indices = HostRows.Create(device, (long)maxRows * sizeof(int), readback: false);
            _scratch.MoeTopkIndices = _indices.Buffer;
            ok = true;
        }
        finally
        {
            if (!ok) DisposeCore();
        }
    }

    /// <summary>
    /// Creates the service on <paramref name="device"/> for <paramref name="layers"/>, importing their banks.
    /// </summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="ownsDevice">True to dispose <paramref name="device"/> with the service.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="layers">The MoE layers (model layer index and CPU-side weights; the banks are read in place).</param>
    /// <param name="hiddenSize">Model hidden size (the expert width of a layer without a latent).</param>
    /// <param name="maxRows">Most rows one submission carries: <see cref="MaxRowsFor"/> of the caller's block.</param>
    /// <param name="log">Sink for load notes and refusals.</param>
    /// <param name="stagingBudgetBytes">Bytes a refused import may copy instead; null = <see cref="DefaultStagingBudgetFraction"/> of RAM.</param>
    /// <returns>The service.</returns>
    public static VulkanNemotronHMoeExpertService Create(VulkanDevice device, bool ownsDevice, string spvDir,
        IReadOnlyList<(int Layer, NemotronHMoeWeights Moe)> layers, int hiddenSize, int maxRows, Action<string>? log = null,
        long? stagingBudgetBytes = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxRows, 1);
        long budget = stagingBudgetBytes
            ?? (long)(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes * DefaultStagingBudgetFraction);
        return new VulkanNemotronHMoeExpertService(device, ownsDevice, spvDir, layers, hiddenSize, maxRows, budget,
            log ?? (m => Console.Error.WriteLine($"[dotllm] {m}")));
    }

    /// <summary>
    /// Selects the integrated GPU (<see cref="VulkanIgpuDeviceSelector"/>), creates a device on it and the service over
    /// <paramref name="layers"/>; null (with the reason) when there is no usable iGPU.
    /// </summary>
    /// <param name="deviceSpec">The <c>--moe-igpu-device</c> override, or null.</param>
    /// <param name="layers">The MoE layers.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="maxRows">Most rows per submission.</param>
    /// <param name="log">Sink for load notes.</param>
    /// <param name="reason">Receives why nothing was created, or what was.</param>
    /// <param name="spvDir">SPIR-V directory; null = <see cref="VulkanModelLoader.ResolveSpvDir"/>.</param>
    /// <returns>The service, or null.</returns>
    public static VulkanNemotronHMoeExpertService? TryCreateOnIntegratedGpu(string? deviceSpec,
        IReadOnlyList<(int Layer, NemotronHMoeWeights Moe)> layers, int hiddenSize, int maxRows, Action<string>? log, out string reason,
        string? spvDir = null)
    {
        var pick = VulkanIgpuDeviceSelector.Select(VulkanDevice.EnumeratePhysicalDevices(), deviceSpec, out reason);
        if (pick is not { } info) return null;
        try
        {
            spvDir ??= VulkanModelLoader.ResolveSpvDir();
        }
        catch (InvalidOperationException ex)
        {
            reason = ex.Message;
            return null;
        }
        var device = VulkanDevice.Create(info.Index);
        try
        {
            return Create(device, ownsDevice: true, spvDir, layers, hiddenSize, maxRows, log);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The <c>maxRows</c> a caller must create the service with to submit any block of up to <paramref name="blockTokens"/> tokens of
    /// any of <paramref name="layers"/>: tokens × the <b>largest</b> top-k among them. A layer call routes each token to its own
    /// top-k experts, so its rows are at most <c>tokens · topK(layer)</c>, all of them when every expert of the call runs here.
    /// Per-layer top-k differs (Bolt's synthetic fixture has 1 to 4), so sizing by one layer's, or by a guess, under-sizes the
    /// others: 17 tokens at top-k 4 is 68 rows (#1083, the first quiet window's one failure).
    /// </summary>
    /// <param name="blockTokens">Most tokens one layer call carries (the caller's MoE block).</param>
    /// <param name="layers">The MoE layers the service will run.</param>
    /// <returns>The row capacity (at least 1).</returns>
    public static int MaxRowsFor(int blockTokens, IEnumerable<NemotronHMoeWeights> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentOutOfRangeException.ThrowIfLessThan(blockTokens, 1);
        int topK = 1;
        foreach (var m in layers) topK = Math.Max(topK, m.NumExpertsPerTok);
        return checked(blockTokens * topK);
    }

    /// <summary>
    /// The load-log line for a layer whose banks were staged because the zero-copy import was refused (#1083): what it costs (on a
    /// UMA device the staged copy is a second copy of the banks in the same RAM), how much of the staging budget is gone (past
    /// it, layers are refused and their cold experts stay on the CPU), and, when the GGUF is mapped read-only, the likely cause
    /// and the remedy: drivers refuse to import read-only pages (#508, measured on gfx1151/amdvlk; the dev box's Arc refused every
    /// bank at <c>vkAllocateMemory</c>), which <c>DOTLLM_GGUF_MAP_COW=1</c> avoids. Pure.
    /// </summary>
    /// <param name="layer">Model layer index.</param>
    /// <param name="reason">The import policy's fallback reason.</param>
    /// <param name="layerBytes">Bytes staged for this layer.</param>
    /// <param name="totalBytes">Bytes staged so far, this layer included.</param>
    /// <param name="budgetBytes">The staging budget.</param>
    /// <param name="ggufReadOnly">True when the GGUF is mapped read-only (<see cref="GgufFile.MappingAccess"/>).</param>
    /// <returns>The line.</returns>
    internal static string DescribeStagedBanks(int layer, string reason, long layerBytes, long totalBytes, long budgetBytes,
        bool ggufReadOnly)
    {
        const double MiB = 1024.0 * 1024.0;
        string line = string.Format(CultureInfo.InvariantCulture,
            "MoE iGPU: blk.{0} expert banks: the zero-copy import was refused ({1}); staged {2:F1} MiB into device memory instead, " +
            "a second copy in RAM on a UMA device ({3:F0} of {4:F0} MiB staging budget used; past it, layers stay on the CPU).",
            layer, reason, layerBytes / MiB, totalBytes / MiB, budgetBytes / MiB);
        if (ggufReadOnly)
            line += " The GGUF is mapped read-only (the default), and drivers refuse to import read-only pages (#508); " +
                "DOTLLM_GGUF_MAP_COW=1 maps it copy-on-write, which they accept without a copy.";
        return line;
    }

    /// <summary>True when the experts of <paramref name="layer"/> can run here (its banks were imported or staged).</summary>
    /// <param name="layer">Model layer index.</param>
    /// <returns>True when supported.</returns>
    public bool Supports(int layer) => _layers.ContainsKey(layer);

    /// <summary>Why <paramref name="layer"/> was refused, or null.</summary>
    /// <param name="layer">Model layer index.</param>
    /// <returns>The reason, or null.</returns>
    public string? RefusalReason(int layer) => _refused.GetValueOrDefault(layer);

    /// <summary>The expert width of a supported layer.</summary>
    /// <param name="layer">Model layer index.</param>
    /// <returns>The width.</returns>
    public int ExpertDim(int layer) => _layers[layer].ExpertDim;

    /// <summary>
    /// The iGPU rows of a schedule's groups <c>[first, end)</c> (#1083): group g is expert <c>experts[g]</c> over slot positions
    /// <c>[groupStart[g], groupStart[g + 1])</c> of <paramref name="slotIds"/>; row j is schedule position
    /// <c>groupStart[first] + j</c>, its expert is that group's, its input the slot's token (<c>slot / topK</c>). Pure.
    /// </summary>
    /// <param name="experts">Expert per group.</param>
    /// <param name="first">First group.</param>
    /// <param name="end">One past the last group.</param>
    /// <param name="groupStart">Group starts (<c>end + 1</c> entries at least).</param>
    /// <param name="slotIds">Slot ids in schedule order.</param>
    /// <param name="topK">Slots per token.</param>
    /// <param name="rowExperts">Receives each row's expert.</param>
    /// <param name="rowTokens">Receives each row's token.</param>
    /// <returns>Rows written.</returns>
    public static int BuildRows(ReadOnlySpan<int> experts, int first, int end, int* groupStart, int* slotIds, int topK,
        Span<int> rowExperts, Span<int> rowTokens)
    {
        int p0 = groupStart[first], rows = groupStart[end] - p0;
        for (int g = first; g < end; g++)
            for (int p = groupStart[g]; p < groupStart[g + 1]; p++)
            {
                rowExperts[p - p0] = experts[g];
                rowTokens[p - p0] = slotIds[p] / topK;
            }
        return rows;
    }

    /// <summary>
    /// Gathers the rows' inputs, records the experts of <paramref name="layer"/> over them and submits, without waiting.
    /// <see cref="End"/> collects the result.
    /// </summary>
    /// <param name="layer">Model layer index (must be <see cref="Supports">supported</see>).</param>
    /// <param name="rowExperts">Expert of each row.</param>
    /// <param name="rowTokens">Input row (token) of each row, into <paramref name="inRows"/>.</param>
    /// <param name="inRows">The caller's input rows, <c>[tokens, ExpertDim(layer)]</c> F32.</param>
    /// <exception cref="InvalidOperationException">A submission is already in flight, or the layer is not supported.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// No rows, more than <see cref="MaxRows"/> (the service was created for smaller blocks; see <see cref="MaxRowsFor"/>), or an
    /// expert out of range.
    /// </exception>
    public void Begin(int layer, ReadOnlySpan<int> rowExperts, ReadOnlySpan<int> rowTokens, float* inRows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _fenced.ThrowIfInFlight();
        if (!_layers.TryGetValue(layer, out var l))
            throw new InvalidOperationException($"blk.{layer} is not on the iGPU: {RefusalReason(layer) ?? "not an offloaded MoE layer"}.");
        int rows = rowExperts.Length;
        if (rows < 1 || rows > _maxRows)
            throw new ArgumentOutOfRangeException(nameof(rowExperts),
                $"{rows} rows (1..{_maxRows}): the service was created for fewer rows per submission than this call carries; " +
                "create it with MaxRowsFor(block tokens, layers).");
        if (rowTokens.Length != rows) throw new ArgumentException("rowTokens and rowExperts differ in length.", nameof(rowTokens));

        long beginTimestamp = Stopwatch.GetTimestamp();
        int ed = l.ExpertDim, numE = l.Banks.NumExperts;
        int* idx = (int*)_indices.Pointer;
        float* x = (float*)_input.Pointer;
        for (int r = 0; r < rows; r++)
        {
            int e = rowExperts[r];
            if ((uint)e >= (uint)numE) throw new ArgumentOutOfRangeException(nameof(rowExperts), $"expert {e} of blk.{layer} ({numE} experts).");
            idx[r] = e;
            new ReadOnlySpan<float>(inRows + (long)rowTokens[r] * ed, ed).CopyTo(new Span<float>(x + (long)r * ed, ed));
        }

        _submit.Begin();
        nint cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        Record(cmd, l, rows);
        KernelSupport.ComputeToHostBarrier(cmd);
        _submit.Submit();
        _fenced.Submitted(rows, ed, beginTimestamp);
    }

    /// <summary>
    /// Waits for the submission of <see cref="Begin"/> and copies its rows, <c>[rows, ExpertDim]</c>, to <paramref name="outRows"/>
    /// (row r = row r of the <see cref="Begin"/> call). Sets <see cref="LastComputeUs"/> and <see cref="LastWaitUs"/>. The fence is
    /// observed before the output is read, and even a failed wait ends the submission (<see cref="FencedHostRows.Collect"/>).
    /// </summary>
    /// <param name="outRows">Destination, at least <c>rows · ExpertDim</c> floats.</param>
    /// <returns>Rows copied.</returns>
    /// <exception cref="InvalidOperationException">Nothing is in flight.</exception>
    public int End(float* outRows) => _fenced.Collect((float*)_output.Pointer, outRows);

    /// <summary><see cref="Begin"/> then <see cref="End"/>.</summary>
    /// <param name="layer">Model layer index.</param>
    /// <param name="rowExperts">Expert of each row.</param>
    /// <param name="rowTokens">Input row of each row.</param>
    /// <param name="inRows">Input rows.</param>
    /// <param name="outRows">Output rows.</param>
    public void Compute(int layer, ReadOnlySpan<int> rowExperts, ReadOnlySpan<int> rowTokens, float* inRows, float* outRows)
    {
        Begin(layer, rowExperts, rowTokens, inRows);
        End(outRows);
    }

    // ───────────────────────── calibration (#1083, design 6) ─────────────────────────

    /// <summary>
    /// Median microseconds of an empty submission: record (two barriers), submit, fence wait. The fixed cost every iGPU leg pays.
    /// </summary>
    /// <param name="reps">Repetitions after three warm-ups.</param>
    /// <returns>Microseconds.</returns>
    public double MeasureSubmitRoundTripUs(int reps = 30)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var t = new double[reps];
        for (int i = -3; i < reps; i++)
        {
            long s = Stopwatch.GetTimestamp();
            _submit.Begin();
            nint cmd = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
            KernelSupport.ComputeToHostBarrier(cmd);
            _submit.Submit();
            _submit.WaitFence();
            if (i >= 0) t[i] = Stopwatch.GetElapsedTime(s).TotalMicroseconds;
        }
        Array.Sort(t);
        return t[reps / 2];
    }

    /// <summary>
    /// Median microseconds of one submission computing <paramref name="expert"/> of <paramref name="layer"/> for <paramref name="n"/>
    /// rows (gather, record, submit, fence, copy-out): the calibration's per-expert iGPU cost before the submit round trip is taken off.
    /// </summary>
    /// <param name="layer">A supported layer.</param>
    /// <param name="expert">Expert index.</param>
    /// <param name="n">Rows.</param>
    /// <param name="reps">Repetitions after a warm-up.</param>
    /// <returns>Microseconds.</returns>
    public double MeasureExpertUs(int layer, int expert, int n, int reps = 10)
    {
        int ed = ExpertDim(layer);
        n = Math.Clamp(n, 1, _maxRows);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * ed * sizeof(float)), 64);
        float* y = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * ed * sizeof(float)), 64);
        try
        {
            var rng = new Random(1083);
            for (long i = 0; i < (long)n * ed; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            int[] experts = new int[n], tokens = new int[n];
            Array.Fill(experts, expert);
            for (int r = 0; r < n; r++) tokens[r] = r;
            NemotronHMoeExpertCompute.Touch(_layers[layer].Host, expert);
            Compute(layer, experts, tokens, x, y);   // warm-up (descriptor sets, first-touch of the imported pages)
            var t = new double[reps];
            for (int r = 0; r < reps; r++)
            {
                long s = Stopwatch.GetTimestamp();
                Compute(layer, experts, tokens, x, y);
                t[r] = Stopwatch.GetElapsedTime(s).TotalMicroseconds;
            }
            Array.Sort(t);
            return t[reps / 2];
        }
        finally
        {
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(y);
        }
    }

    // ───────────────────────── recording ─────────────────────────

    /// <summary>
    /// Records up → relu² → down over <paramref name="rows"/> routed rows of <paramref name="l"/>: the expert-grouped prefill from
    /// <see cref="VulkanNemotronHTransformerModel.MoeGroupedMinTokens"/> rows where the banks have grouped kernels (top-k 1 and unit
    /// weights, so its scatter writes each row back unweighted), else the indexed arm, whose down projection writes the host-visible
    /// output directly.
    /// </summary>
    private void Record(nint cmd, LayerBanks l, int rows)
    {
        var k = _kernels;
        var b = l.Banks;
        int ed = l.ExpertDim;
        var path = rows >= VulkanNemotronHTransformerModel.MoeGroupedMinTokens
            ? VulkanNemotronHTransformerModel.SelectGroupedPath(k, b, ed)
            : VulkanNemotronHTransformerModel.GroupedPath.None;
        if (path != VulkanNemotronHTransformerModel.GroupedPath.None)
        {
            VulkanNemotronHTransformerModel.RecordGroupedExperts(cmd, k, _scratch, _relu, b, topK: 1, path, _input.Buffer, _output.Buffer,
                seqLen: rows, expertDim: ed);
            return;
        }

        var s = _scratch;
        VulkanNemotronHTransformerModel.RecordExpertMatmul(cmd, k, _indices.Buffer, b.UpQt, b.UpBank, _input.Buffer,
            s.MoeExpandedInputXq!, s.MoeExpandedInputXds!, s.MoeUpInter!, mDim: b.Inter, kDim: ed, rows: rows, numExperts: b.NumExperts);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _relu.Record(cmd, s.MoeUpInter!, n: rows * b.Inter);
        KernelSupport.ComputeToComputeBarrier(cmd);
        VulkanNemotronHTransformerModel.RecordExpertMatmul(cmd, k, _indices.Buffer, b.DownQt, b.DownBank, s.MoeUpInter!,
            s.MoeUpInterXq!, s.MoeUpInterXds!, _output.Buffer, mDim: ed, kDim: b.Inter, rows: rows, numExperts: b.NumExperts);
        KernelSupport.ComputeToComputeBarrier(cmd);
    }

    // ───────────────────────── buffers ─────────────────────────

    /// <summary>The service's routed-expert scratch, sized for <c>maxRows</c> rows at top-k 1.</summary>
    private sealed class Scratch : INemotronHMoeRoutedScratch, IDisposable
    {
        private readonly List<VulkanDevice.Buffer> _owned = [];

        public Scratch(VulkanDevice device, int rows, int expertDim, int inter, int numExperts)
        {
            VulkanDevice.Buffer Alloc(long bytes)
            {
                var b = device.AllocateDeviceLocal(Math.Max(bytes, 16));
                _owned.Add(b);
                return b;
            }

            var ones = new float[rows];
            Array.Fill(ones, 1.0f);
            MoeTopkWeights = Alloc((long)rows * sizeof(float));
            device.Upload(ones, MoeTopkWeights);
            MoeExpandedInput = Alloc((long)rows * expertDim * sizeof(float));
            MoeExpandedInputXq = Alloc(QuantizeQ8_1RowsKernel.PackedBytes(rows, expertDim));
            MoeExpandedInputXds = Alloc(QuantizeQ8_1RowsKernel.ScaleBytes(rows, expertDim));
            MoeUpInter = Alloc((long)rows * inter * sizeof(float));
            MoeUpInterXq = Alloc(QuantizeQ8_1RowsKernel.PackedBytes(rows, inter));
            MoeUpInterXds = Alloc(QuantizeQ8_1RowsKernel.ScaleBytes(rows, inter));
            MoeDownRows = Alloc((long)rows * expertDim * sizeof(float));
            MoeGroupCounts = Alloc((long)numExperts * sizeof(uint));
            MoeGroupOffsets = Alloc(MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint));
            MoeGroupCounters = Alloc((long)numExperts * sizeof(uint));
            MoeGroupPerm = Alloc((long)rows * sizeof(uint));
            MoeGroupInvPerm = Alloc((long)rows * sizeof(uint));
            MoeGroupDispatchArgs = Alloc(2 * MoeBuildTileListKernel.ArgsStrideBytes);
            MoeTokenXq = Alloc(QuantizeQ8_1RowsKernel.PackedBytes(rows, expertDim));
            MoeTokenXds = Alloc(QuantizeQ8_1RowsKernel.ScaleBytes(rows, expertDim));
        }

        /// <summary>The host-visible index buffer (set by the service; not owned here).</summary>
        public VulkanDevice.Buffer? MoeTopkIndices { get; set; }
        public VulkanDevice.Buffer? MoeTopkWeights { get; }
        public VulkanDevice.Buffer? MoeExpandedInput { get; }
        public VulkanDevice.Buffer? MoeExpandedInputXq { get; }
        public VulkanDevice.Buffer? MoeExpandedInputXds { get; }
        public VulkanDevice.Buffer? MoeUpInter { get; }
        public VulkanDevice.Buffer? MoeUpInterXq { get; }
        public VulkanDevice.Buffer? MoeUpInterXds { get; }
        public VulkanDevice.Buffer? MoeDownRows { get; }
        public VulkanDevice.Buffer? MoeGroupCounts { get; }
        public VulkanDevice.Buffer? MoeGroupOffsets { get; }
        public VulkanDevice.Buffer? MoeGroupCounters { get; }
        public VulkanDevice.Buffer? MoeGroupPerm { get; }
        public VulkanDevice.Buffer? MoeGroupInvPerm { get; }
        public VulkanDevice.Buffer? MoeGroupDispatchArgs { get; }
        public VulkanDevice.Buffer? MoeTokenXq { get; }
        public VulkanDevice.Buffer? MoeTokenXds { get; }

        public void Dispose()
        {
            foreach (var b in _owned) b.Dispose();
            _owned.Clear();
        }
    }

    /// <summary>
    /// Host memory the device reads or writes directly: a page-aligned host allocation imported with
    /// <c>VK_EXT_external_memory_host</c> where the device allows it (UMA: no copy at all), else a host-visible buffer kept mapped
    /// for the service's lifetime.
    /// </summary>
    private sealed class HostRows : IDisposable
    {
        private readonly VulkanDevice _device;
        private nint _hostAlloc;
        private bool _mapped;

        public VulkanDevice.Buffer Buffer { get; }
        public byte* Pointer { get; }

        private HostRows(VulkanDevice device, VulkanDevice.Buffer buffer, byte* pointer, nint hostAlloc, bool mapped)
        {
            _device = device;
            Buffer = buffer;
            Pointer = pointer;
            _hostAlloc = hostAlloc;
            _mapped = mapped;
        }

        public static HostRows Create(VulkanDevice device, long bytes, bool readback)
        {
            bytes = Math.Max(bytes, 16);
            if (device.HasExternalMemoryHost && device.PhysicalDeviceTypeValue is VkPhysicalDeviceType.IntegratedGpu)
            {
                // A whole number of import granules (and at least a page), so the import neither starts nor ends mid-granule.
                ulong align = Math.Max(4096UL, device.MinImportedHostPointerAlignment);
                long size = (long)HostImportRange.RoundUp((ulong)bytes, align);
                nint p = (nint)NativeMemory.AlignedAlloc((nuint)size, (nuint)align);
                new Span<byte>((void*)p, (int)Math.Min(size, int.MaxValue)).Clear();
                var imported = device.TryWrapHostVisible(p, size);
                if (imported is not null) return new HostRows(device, imported, (byte*)p, p, mapped: false);
                NativeMemory.AlignedFree((void*)p);
            }
            var buf = readback ? device.AllocateHostReadback(bytes) : device.Allocate(bytes);
            nint mappedPtr = device.MapMemoryWithRetry(buf.Memory, 0, (ulong)bytes, "vkMapMemory MoE iGPU rows");
            return new HostRows(device, buf, (byte*)mappedPtr, 0, mapped: true);
        }

        public void Dispose()
        {
            if (_mapped)
            {
                VulkanApi.vkUnmapMemory(_device.Handle, Buffer.Memory);
                _mapped = false;
            }
            Buffer.Dispose();
            if (_hostAlloc != 0)
            {
                NativeMemory.AlignedFree((void*)_hostAlloc);
                _hostAlloc = 0;
            }
        }
    }

    // ───────────────────────── lifetime ─────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        DisposeCore();
    }

    private void DisposeCore()
    {
        _fenced.DrainForDispose<VulkanException>();
        _input?.Dispose();
        _output?.Dispose();
        _indices?.Dispose();
        _scratch?.Dispose();
        _kernels?.Dispose();
        foreach (var b in _bankBuffers) b.Dispose();
        _bankBuffers.Clear();
        _relu?.Dispose();
        _iq2?.Dispose();
        _iq3?.Dispose();
        _submit?.Dispose();
        if (_ownsDevice) _device.Dispose();
    }
}
