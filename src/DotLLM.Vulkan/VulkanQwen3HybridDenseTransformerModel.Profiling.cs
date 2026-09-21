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
            DotLLM.Core.Configuration.QuantizationType.Q8_0 when !gemv =>
                _kernels.MatMulQ8GemmCoopmat is not null ? "matmul_q8_0_gemm_coopmat" : "matmul_q8_0_gemm",
            DotLLM.Core.Configuration.QuantizationType.F16 when !gemv =>
                _kernels.MatMulF16GemmCoopmat is not null ? "matmul_f16_gemm_coopmat" : "matmul_f16_gemm",
            _ => $"matmul_{qt.ToString().ToLowerInvariant()}_{(gemv ? "gemv" : "gemm")}",
        };
        _prof?.Note(kernel, m: outputDim, k: inputDim, n: seqLen);
    }
}
