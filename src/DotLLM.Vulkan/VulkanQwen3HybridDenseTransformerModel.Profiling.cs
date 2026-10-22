using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan;

/// <content>
/// Env-gated per-op profiling hooks (issue #434).
/// </content>
/// <remarks>
/// <para>
/// This model had no instrumentation at all, which made the measured 7.2x Bonsai 2 prefill
/// deficit against the PrismML llama.cpp fork unattributable — and therefore a standing
/// invitation to guess "it must be the matmul". The hooks below exist so the guess can be
/// replaced by a table.
/// </para>
/// <para>
/// <b>Environment.</b>
/// <list type="bullet">
/// <item><description><c>DOTLLM_VULKAN_HYBRID_PROFILE=1</c> — enable. GPU-timestamp mode by
/// default: BOTTOM_OF_PIPE stamps at category boundaries, plus host record/wait phase
/// accounting and an explicit unattributed remainder.</description></item>
/// <item><description><c>DOTLLM_VULKAN_HYBRID_PROFILE_SPLIT=1</c> — cross-check mode: split the
/// command buffer at every boundary and charge host wall time instead. Slower and it taxes every
/// boundary with a pipeline drain, but it cannot be fooled by a mis-tagged query.</description></item>
/// <item><description><c>DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ=&lt;n&gt;</c> — only profile forwards
/// of at least n tokens. Default 2, i.e. prefill only; set 1 to include decode steps.</description></item>
/// <item><description><c>DOTLLM_VULKAN_HYBRID_PROFILE_OUT=&lt;path&gt;</c> — also append the report
/// to a file, since <c>bench</c> renders over stderr.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Cost when disabled</b> is one <see langword="bool"/> test per forward and a null check per
/// mark site; no query pool is created and no stamp is recorded.
/// </para>
/// </remarks>
public sealed partial class VulkanQwen3HybridDenseTransformerModel
{
    /// <summary>Longest forward (tokens) recorded into a single command buffer. <c>DOTLLM_VK_FUSE_FORWARD=0</c> restores one submission per layer.</summary>
    private const int FuseMaxSeqLen = 32;

    /// <summary>
    /// Default prefill tile, in tokens (issue #1100). Without a tile a 16k-token Bonsai 2 prompt was one forward whose per-layer
    /// submissions each covered all 16k rows, and Strix lost the device (Windows GPU watchdog). <c>DOTLLM_VULKAN_PREFILL_CHUNK</c>
    /// overrides (0 = no tiling); <c>--prefill-chunk-size</c> overrides both.
    /// </summary>
    internal const int DefaultPrefillChunkTokens = 1024;

    /// <summary>Tile in force: <c>DOTLLM_VULKAN_PREFILL_CHUNK</c> when it parses to a non-negative int, else <see cref="DefaultPrefillChunkTokens"/>.</summary>
    internal static int PrefillChunkTokens { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PREFILL_CHUNK"), out int chunk) && chunk >= 0
            ? chunk
            : DefaultPrefillChunkTokens;

    /// <inheritdoc/>
    /// <remarks>Bounds each forward the engine submits (issue #1100); see <see cref="PrefillChunkTokens"/>.</remarks>
    public int PreferredPrefillChunkTokens => PrefillChunkTokens;

    /// <summary>
    /// Fixed layers per submission for a prefill longer than <see cref="FuseMaxSeqLen"/> when <c>DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT</c>
    /// is set (1 = one per layer), else 0: the submissions are then bounded by <see cref="_submitBudget"/> (issue #1100).
    /// </summary>
    private static readonly int FixedPrefillLayersPerSubmit =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT"), out int fixedLayers) && fixedLayers >= 1
            ? fixedLayers
            : 0;

    /// <summary><c>DOTLLM_VK_PREFILL_SUBMIT_LOG=1</c>: one stderr line per budgeted prefill forward (submissions, longest, fitted rates).</summary>
    private static readonly bool SubmitLogEnabled = Environment.GetEnvironmentVariable("DOTLLM_VK_PREFILL_SUBMIT_LOG") == "1";

    /// <summary>GPU-time budget and fitted cost rates for prefill submissions (issue #1100), per model: the rates are per device.</summary>
    private VulkanPrefillSubmitBudget _submitBudget = VulkanPrefillSubmitBudget.FromEnvironment();

    private bool[]? _isAttentionLayer;

    /// <summary>Test hook: the submission budget (replaceable, to force or forbid splitting).</summary>
    internal VulkanPrefillSubmitBudget SubmitBudgetForTest
    {
        get => _submitBudget;
        set => _submitBudget = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>Test hook: layer submissions the last forward or batch made.</summary>
    internal int LastForwardLayerSubmissionsForTest { get; private set; }

    /// <summary>Test hook: row-slice forwards <see cref="ForwardInRowSlices"/> ran.</summary>
    internal int RowSliceForwardsForTest { get; private set; }

    /// <summary>Weight elements of one layer, for the dense cost estimate: the FFN plus an attention-sized mixer.</summary>
    private double ParamsPerLayer
        => (3.0 * Config.HiddenSize * Config.IntermediateSize)
           + ((double)Config.HiddenSize * ((3.0 * Config.NumAttentionHeads) + (2.0 * Config.NumKvHeads)) * Config.HeadDim);

    /// <summary>
    /// Starts the layer-submission schedule of one forward (or batch) over <paramref name="rows"/> rows whose attention layers each do
    /// <paramref name="attentionGflopsPerLayer"/> GFLOPs.
    /// </summary>
    /// <param name="rows">Rows in the forward.</param>
    /// <param name="attentionGflopsPerLayer">Attention work of one attention layer.</param>
    /// <returns>The schedule.</returns>
    private SubmitSchedule BeginSubmitSchedule(int rows, double attentionGflopsPerLayer)
    {
        int fixedLayers = !FuseForwardEnabled || _prof is not null ? 1
            : rows <= FuseMaxSeqLen ? int.MaxValue
            : FixedPrefillLayersPerSubmit;
        if (_isAttentionLayer is null)
        {
            var kinds = _layout.LayerKind;
            var attention = new bool[kinds.Length];
            for (int l = 0; l < kinds.Length; l++) attention[l] = kinds[l] == HybridLayerKind.Attention;
            _isAttentionLayer = attention;
        }

        return new SubmitSchedule(this, fixedLayers, VulkanPrefillSubmitBudget.DenseGflops(ParamsPerLayer, rows), attentionGflopsPerLayer, rows);
    }

    /// <summary>
    /// Splits a long prefill whose single attention layer would exceed the submission budget at this KV depth into row slices run as
    /// consecutive forwards (issue #1100): the KV and recurrent state carry over between them exactly as between engine tiles.
    /// </summary>
    /// <returns>
    /// The last slice's logits, or null when no split is needed or allowed: no KV cache to carry the rows, all-row logits wanted,
    /// row snapshots requested, or a fixed submission override.
    /// </returns>
    private ITensor? ForwardInRowSlices(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                        IKvCache? kvCache, IGdnState? gdnState, IMtpState? mtpState)
    {
        int rows = tokenIds.Length;
        if (rows <= FuseMaxSeqLen || rows <= _allRowLogitsLimit || _rowSnapshotRequestRows > 0
            || kvCache is not VulkanNemotronHKvCache || FixedPrefillLayersPerSubmit != 0 || !FuseForwardEnabled)
            return null;
        int kvEnd = VulkanNemotronHKvCache.PositionsEnd(positions);
        int per = _submitBudget.RowsPerForward(rows, kvEnd, ParamsPerLayer, Config.NumAttentionHeads, Config.HeadDim);
        if (per >= rows) return null;
        int slices = (rows + per - 1) / per;
        int baseLen = rows / slices, extra = rows % slices;
        if (baseLen <= _allRowLogitsLimit) return null;   // a slice would return all-row logits: not this call's contract

        RowSliceForwardsForTest += slices;
        ITensor? last = null;
        int start = 0;
        for (int s = 0; s < slices; s++)
        {
            int len = baseLen + (s < extra ? 1 : 0);
            last?.Dispose();
            last = Forward(tokenIds.Slice(start, len), positions.Slice(start, len), deviceId, kvCache, gdnState, mtpState);
            start += len;
        }
        return last;
    }

    /// <summary>
    /// Where one forward's layer loop submits (issue #1100). Short forwards record every layer into one command buffer; with
    /// profiling or <c>DOTLLM_VK_FUSE_FORWARD=0</c> each layer is its own submission; <c>DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT</c> fixes the
    /// count; otherwise layers are packed up to the GPU-time budget and every submission's wall time recalibrates it.
    /// </summary>
    private struct SubmitSchedule(VulkanQwen3HybridDenseTransformerModel model, int fixedLayers, double denseGflops, double attentionGflops, int rows)
    {
        private int _lastLayer = -1;
        private double _openDense;
        private double _openAttention;
        private int _submissions;
        private double _longest;

        /// <summary>Gets a value indicating whether layers share submissions (false: every layer and the embedding submit alone).</summary>
        public readonly bool Fuse => fixedLayers != 1;

        private readonly bool Budgeted => fixedLayers == 0;

        /// <summary>Call as <paramref name="layer"/> starts recording.</summary>
        /// <param name="layer">The layer.</param>
        public void LayerStarted(int layer)
        {
            var attention = model._isAttentionLayer!;
            if (Budgeted && layer > _lastLayer)
                _lastLayer = layer + model._submitBudget.LayersInSubmission(attention, layer, denseGflops, attentionGflops) - 1;
            _openDense += denseGflops;
            if (attention[layer]) _openAttention += attentionGflops;
        }

        /// <summary>True when the command buffer must be submitted after <paramref name="layer"/>.</summary>
        /// <param name="layer">The layer just recorded.</param>
        /// <returns>Whether to submit now.</returns>
        public readonly bool SubmitAfter(int layer)
            => !Fuse || (Budgeted ? layer == _lastLayer : fixedLayers != int.MaxValue && (layer + 1) % fixedLayers == 0);

        /// <summary>Records a layer submission's measured wall time and feeds it to the budget.</summary>
        /// <param name="seconds">Wall time of the submit-and-wait.</param>
        public void Submitted(double seconds)
        {
            _submissions++;
            if (seconds > _longest) _longest = seconds;
            if (Budgeted) model._submitBudget.Observe(_openDense, _openAttention, seconds);
            _openDense = 0;
            _openAttention = 0;
        }

        /// <summary>Call once the layer loop is done.</summary>
        /// <param name="kvEnd">KV length after the forward (for the log line).</param>
        public readonly void Finish(int kvEnd)
        {
            model.LastForwardLayerSubmissionsForTest = _submissions;
            if (!Budgeted || !SubmitLogEnabled) return;
            var b = model._submitBudget;
            Console.Error.WriteLine(
                $"[vk-prefill] rows={rows} kvEnd={kvEnd} submissions={_submissions} longest={_longest * 1000:F0}ms " +
                $"budget={b.BudgetSeconds * 1000:F0}ms dense={1 / b.DenseSecondsPerGflop / 1000:F2}TF/s attn={1 / b.AttentionSecondsPerGflop / 1000:F2}TF/s");
        }
    }

    private static readonly bool FuseForwardEnabled =
        Environment.GetEnvironmentVariable("DOTLLM_VK_FUSE_FORWARD") != "0";

    private static readonly bool ProfileEnabledFromEnv =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HYBRID_PROFILE") == "1";

    private static readonly bool ProfileSplitFromEnv =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HYBRID_PROFILE_SPLIT") == "1";

    private static readonly int ProfileMinSeqLen =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ"), out int n)
        && n >= 1 ? n : 2;

    // Owned for the model's lifetime (the query pool is created once, lazily); _prof is the
    // per-forward alias that every mark site null-checks.
    private VulkanOpProfiler? _profiler;
    private VulkanOpProfiler? _prof;

    /// <summary>
    /// Test hook: force profiling on irrespective of the environment. The env reads above are
    /// <see langword="static" /> <see langword="readonly" />, so a test cannot turn profiling on
    /// by setting the variable after the type has loaded.
    /// </summary>
    internal bool ProfileOverride { get; set; }

    /// <summary>Test hook: select split-submit cross-check mode. Read once, when the profiler is created.</summary>
    internal bool ProfileSplitOverride { get; set; }

    /// <summary>The most recent completed profile, or <see langword="null"/> if none.</summary>
    internal VulkanOpProfileReport? LastProfile { get; private set; }

    private void ProfBeginForward(int seqLen)
    {
        if (!(ProfileEnabledFromEnv || ProfileOverride) || seqLen < ProfileMinSeqLen)
        {
            _prof = null;
            return;
        }

        _profiler ??= new VulkanOpProfiler(_device, _submit)
        {
            SplitSubmits = ProfileSplitFromEnv || ProfileSplitOverride,
        };
        _prof = _profiler;
        _prof.BeginForward(seqLen, Config.NumLayers);
    }

    private void ProfEndForward()
    {
        if (_prof is null) return;
        var report = _prof.EndForward();
        _prof = null;
        LastProfile = report;

        string text = report.Format("hybrid-profile");
        Console.Error.Write(text);
        if (Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HYBRID_PROFILE_OUT") is { Length: > 0 } path)
        {
            try { File.AppendAllText(path, text); }
            catch { /* diagnostics only */ }
        }
    }

    /// <summary>Opens a submit for profiling — resets the query pool and writes the baseline stamp.</summary>
    private void ProfBeginSubmit(nint cmdBuf) => _prof?.BeginSubmit(cmdBuf);

    // ── #471: MTP draft-step attribution ─────────────────────────────────────
    // A draft step (ForwardMtp with logits) is profiled on its own switch, so a trunk profile is
    // not drowned in one extra report per drafted token, and vice versa.

    private static readonly bool MtpProfileEnabledFromEnv =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MTP_PROFILE") == "1";

    /// <summary>Test hook: profile every MTP draft step irrespective of the environment.</summary>
    internal bool MtpProfileOverride { get; set; }

    /// <summary>The most recent MTP draft-step profile, or <see langword="null"/> if none.</summary>
    internal VulkanOpProfileReport? LastMtpProfile { get; private set; }

    /// <summary>
    /// Host time of the most recent profiled draft step spent after the GPU finished (reading the
    /// result back). It is part of the report's unattributed remainder.
    /// </summary>
    internal double LastMtpHostTailMs { get; private set; }

    private long _mtpProfTailStart;

    private void ProfBeginMtpStep()
    {
        if (!(MtpProfileEnabledFromEnv || MtpProfileOverride))
        {
            _prof = null;
            return;
        }

        _profiler ??= new VulkanOpProfiler(_device, _submit)
        {
            SplitSubmits = ProfileSplitFromEnv || ProfileSplitOverride,
        };
        _prof = _profiler;
        _prof.BeginForward(1, 1);
    }

    private void ProfMtpTailStart()
    {
        if (_prof is not null) _mtpProfTailStart = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private void ProfEndMtpStep()
    {
        if (_prof is null) return;
        LastMtpHostTailMs = (System.Diagnostics.Stopwatch.GetTimestamp() - _mtpProfTailStart) * 1000.0
                            / System.Diagnostics.Stopwatch.Frequency;
        var report = _prof.EndForward();
        _prof = null;
        LastMtpProfile = report;
        if (MtpProfileEnabledFromEnv)
            Console.Error.Write(report.Format("mtp-profile"));
    }

    /// <summary>Category boundary: charges everything since the previous mark to <paramref name="cat"/>.</summary>
    private void ProfMark(nint cmdBuf, VulkanOpProfiler.Cat cat) => _prof?.Mark(cmdBuf, cat);

    /// <summary>Charges the host record phase; call immediately before <c>SubmitAndWait</c>.</summary>
    private void ProfBeforeSubmit(nint cmdBuf) => _prof?.BeforeSubmit(cmdBuf);

    /// <summary>Charges the fence wait and reads back timestamps; call immediately after <c>SubmitAndWait</c>.</summary>
    private void ProfAfterSubmit() => _prof?.AfterSubmit();

    /// <summary>Records which kernel variant ran at which shape.</summary>
    private void ProfNote(string kernel, int m, int k, int n) => _prof?.Note(kernel, m, k, n);

    /// <summary>
    /// Census entry for one projection dispatch. For PQ2_0 GEMM it records the SPIR-V module the
    /// pipeline was actually built from, so the table states which variant ran rather than which
    /// one <c>PQ2_0GemmVariant.SelectFor</c> is supposed to pick.
    /// </summary>
    private void ProfNoteMatmul(
        DotLLM.Core.Configuration.QuantizationType qt, int outputDim, int inputDim, int seqLen)
    {
        bool gemv = seqLen == 1;
        string kernel = qt switch
        {
            DotLLM.Core.Configuration.QuantizationType.PQ2_0 => gemv
                ? "matmul_pq2_0_gemv"
                : $"matmul_pq2_0_gemm[{_kernels.MatMulPQ2_0Gemm.VariantName}]",
            DotLLM.Core.Configuration.QuantizationType.Q8_0 when gemv =>
                $"matmul_q8_0_gemv[{_kernels.MatMulQ8.VariantName}]",
            DotLLM.Core.Configuration.QuantizationType.Q8_0 when !gemv =>
                _kernels.MatMulQ8GemmCoopmat is not null ? "matmul_q8_0_gemm_coopmat" : "matmul_q8_0_gemm",
            DotLLM.Core.Configuration.QuantizationType.F16 when !gemv =>
                _kernels.MatMulF16GemmCoopmat is not null ? "matmul_f16_gemm_coopmat" : "matmul_f16_gemm",
            _ => $"matmul_{qt.ToString().ToLowerInvariant()}_{(gemv ? "gemv" : "gemm")}",
        };
        _prof?.Note(kernel, m: outputDim, k: inputDim, n: seqLen);
    }
}
