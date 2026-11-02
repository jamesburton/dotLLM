using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Speculative decoding on the Vulkan <c>nemotron_h</c> / <c>nemotron_h_moe</c> trunk (issue #1237): a DFlash / LiLiCorr drafter
/// (NVIDIA's Nemotron-3.5 Bolt drafter) in the MTP slot, exactly as on <c>qwen35</c> (#1175, #1193).
/// </summary>
/// <remarks>
/// <para><b>Exactness.</b> A batch of at most <see cref="RowExactMaxTokens"/> rows (a verify, its replay, a short prompt) runs every
/// projection through the decode GEMV over its rows (<see cref="GemvColumns"/>), the attention one row at a time through the
/// decode kernels, and the routed experts through the per-(token, slot) indexed kernels; the elementwise, norm, convolution and
/// scan kernels are per row already. Each row is then bit-identical to a decode step, so greedy speculation reproduces plain
/// decoding token for token while each weight row is still read once per verify.</para>
/// <para><b>Rollback.</b> The Mamba-2 conv/SSM state is checkpointed into a persistent device copy recorded at the start of the
/// verify (no extra submit) and restored at the start of the replay of the accepted rows.</para>
/// </remarks>
public sealed partial class VulkanNemotronHTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps, IVulkanDFlashHost
{
    /// <summary>Largest batch that runs on the decode-identical small-batch path (a DFlash block of 16).</summary>
    internal const int RowExactMaxTokens = GemvColumns.MaxColumns;

    /// <summary>Test hook: false sends small batches down the prefill kernels again (the negative control of the exactness test).</summary>
    internal static bool RowExactEnabled { get; set; } = true;

    private DFlashDrafter? _dflash;
    private VulkanDFlashBackend? _dflashGpu;
    private string _spvDir = string.Empty;
    private nint _hostTokenEmbed;
    private QuantizationType _hostTokenEmbedQt;
    private int[] _dflashSlotForLayerInput = [];
    private VulkanDevice.Buffer? _dflashTapBuffer;   // [slot][row][hidden] F32, host readback, transient
    private long _dflashTapBufferBytes;
    private float[] _dflashTapHost = [];
    private DFlashMtpState? _dflashTaps;             // set for the duration of a forward that captures taps
    private int _dflashTapRows;

    // Recurrent checkpoint: a persistent device copy of every SSM layer's conv and SSM state, filled at the start of the next
    // forward (the verify) and copied back at the start of the forward after a restore (the replay).
    private VulkanDevice.Buffer[]? _ckptConv, _ckptSsm;
    private VulkanSsmStateCache? _ckptPendingSave, _ckptPendingRestore;

    // Per-row snapshots: each SSM layer's scan inputs and conv-input rows of the last snapshot forward.
    private VulkanDevice.Buffer[]? _snapX, _snapDt, _snapB, _snapC, _snapConv;
    private VulkanSsmStateCache? _snapCache;
    private int _snapRequestRows;      // > 0 only while ForwardWithRecurrentSnapshots records
    private int _snapValidRows;        // rows the last such forward recorded; 0 = none
    private int _pendingRestoreRow = -1;
    private bool _roVerify;            // the snapshot forward in flight leaves the SSM/conv state untouched
    private bool _snapReadOnly;        // the last snapshot forward was read-only: a commit replay is owed
    private int _pendingCommitRows;    // read-only verify: rows to replay into the live state at the next forward head

    /// <summary>True when the scan keeps the state in registers (and so can run read-only).</summary>
    private bool ScanKeepsStateInRegisters
        => _mamba2ScanRegs is not null && Mamba2SelectiveScanRegsF32Kernel.IsEligible(_ssm.HeadDim, _ssm.DState);

    /// <summary>
    /// Read-only verify (#1237, default when the register scan serves the model): the verify scan reads the live state and never writes
    /// it; the accepted rows are replayed into it afterwards. No copy of the state exists. <c>DOTLLM_NEMOTRON_VERIFY_COPY=1</c> keeps the
    /// pre-verify copy instead (A/B).
    /// </summary>
    internal static bool ReadOnlyVerifyEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_NEMOTRON_VERIFY_COPY") != "1";

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <summary>Device bytes held by the attached drafter's weights and norm vectors (0 without one).</summary>
    public long DFlashWeightBytes => _dflashGpu?.WeightBytes ?? 0;

    /// <summary>Device bytes of the recurrent checkpoint (0 until the first speculative round).</summary>
    public long RecurrentCheckpointBytes
        => (_ckptConv?.Sum(b => b.Size) ?? 0) + (_ckptSsm?.Sum(b => b.Size) ?? 0);

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize,
            new VulkanDFlashBackend(this, VulkanDFlashKernels.Create(_device, _spvDir)));
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        if (drafter.BlockSize > RowExactMaxTokens)
        {
            drafter.Dispose();
            throw new NotSupportedException($"DFlash block size {drafter.BlockSize} exceeds the exact verify width {RowExactMaxTokens}.");
        }
        _dflashSlotForLayerInput = slots;
        _dflashGpu = (VulkanDFlashBackend)drafter.Backend;
        _dflash = drafter;
        // Sized now so neither a verify nor a draft grows them mid-recording.
        if (_state.EnsureCapacity(drafter.BlockSize)) InvalidateKernelCaches();
        TrySetAllRowLogitsLimit(drafter.BlockSize);
        EnsureMultiRowLogits(drafter.BlockSize, Config.VocabSize);
    }

    /// <inheritdoc/>
    public bool SupportsMtp => _dflash is not null;

    /// <summary>Draft tokens per round: the block's drafts, capped by <c>DOTLLM_DFLASH_DRAFT</c> when set.</summary>
    public int MtpPreferredDraftTokens => _dflash is null ? 0 : _dflash.DraftTokensPerRound;

    /// <summary>With a drafter attached the adaptive gate tries speculation first (#1223): a verify of K+1 rows reads each weight once.</summary>
    public long MtpGatePriorBytes => _dflash is not null ? long.MaxValue : ComputeMemoryBytes;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => _dflash is null ? null : new DFlashMtpState(_dflash);

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dflash is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(VulkanNemotronHTransformerModel)} drafts only through an attached DFlash drafter.");
        return dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
    }

    /// <inheritdoc/>
    public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(VulkanNemotronHTransformerModel)} drafts only through an attached DFlash drafter.");
        dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
        var row = dflash.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
    {
        if (adapter is not null)
            throw new NotSupportedException($"{nameof(VulkanNemotronHTransformerModel)} does not support LoRA adapters.");
        var taps = _dflash is not null ? mtpState as DFlashMtpState : null;
        if (taps is null) return Forward(tokenIds, positions, deviceId, kvCache);

        EnsureDFlashTapBuffer(tokenIds.Length);
        _dflashTaps = taps;
        _dflashTapRows = tokenIds.Length;
        ITensor result;
        try { result = Forward(tokenIds, positions, deviceId, kvCache); }
        finally { _dflashTaps = null; }
        try { FinishDFlashTaps(taps, positions, tokenIds.Length); }
        catch { result.Dispose(); throw; }
        return result;
    }

    /// <summary>Allocates the tap buffer for <paramref name="rows"/> before any recording (a new buffer invalidates descriptor caches).</summary>
    private void EnsureDFlashTapBuffer(int rows)
    {
        long bytes = (long)_dflash!.TargetLayers.Length * rows * Config.HiddenSize * sizeof(float);
        if (_dflashTapBuffer is not null && _dflashTapBufferBytes >= bytes) return;
        _dflashTapBuffer?.Dispose();
        _dflashTapBuffer = _device.AllocateHostReadback(bytes);
        _dflashTapBufferBytes = bytes;
        InvalidateKernelCaches();
    }

    /// <summary>Records the copy of the current <c>HiddenState</c> rows (the input of <paramref name="layerInput"/>) into its tap slot.</summary>
    private void RecordDFlashTap(nint cmdBuf, int layerInput)
    {
        if (_dflashTaps is null) return;
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        int rows = _dflashTapRows;
        long rowBytes = (long)Config.HiddenSize * sizeof(float);
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _dflashTapBuffer!,
            srcOffset: 0, dstOffset: (ulong)((long)slot * rows * rowBytes), size: (ulong)(rows * rowBytes));
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
    }

    /// <summary>Hands the downloaded taps to the state and injects them (after the forward's last submit).</summary>
    private unsafe void FinishDFlashTaps(DFlashMtpState state, ReadOnlySpan<int> positions, int rows)
    {
        int h = Config.HiddenSize, slots = _dflash!.TargetLayers.Length;
        int n = slots * rows * h;
        if (_dflashTapHost.Length < n) _dflashTapHost = new float[n];
        var host = _dflashTapHost.AsSpan(0, n);
        _device.Download(_dflashTapBuffer!, host);
        state.EnsureTapRows(rows);
        for (int s = 0; s < slots; s++)
            for (int r = 0; r < rows; r++)
                state.CaptureTap(r, s, host.Slice((s * rows + r) * h, h));
        state.InjectCaptured(positions);   // recorded on the drafter's command buffer; the next draft submits it
        state.TrimTapRows();
        if (rows > 4 * _dflash.BlockSize)
        {
            // A long prefill grew it: give the transient buffer back.
            _dflashTapBuffer!.Dispose();
            _dflashTapBuffer = null;
            _dflashTapBufferBytes = 0;
            _dflashTapHost = [];
            InvalidateKernelCaches();
        }
    }

    // ── Small-batch exact path ──────────────────────────────────────────

    // #1244: deterministic multi-column GEMVs (decode = 1 column) that decode each weight block once for every row; null when the shader
    // is missing, and NVFP4's when the opt-in MMVQ owns decode.
    private MatMulExactGemvMultiKernel? _exactQ8, _exactNvfp4, _exactIq2Xxs, _exactQ5_0, _exactQ4_0;

    // #1420: K-quant projections (Q4_K_M / Q6_K) decode and verify on the NR-parameterised dense-host multi-column GEMVs (one column = decode).
    private MatMulQuantGemvMultiKernel? _exactQ4K, _exactQ6K;

    /// <summary>Environment override of the exact GEMVs' rows per workgroup, <c>decode,verify</c> (tuning A/B; e.g. <c>2,4</c>).</summary>
    internal const string ExactGemvRowsEnvVar = "DOTLLM_VK_NEMOTRON_GEMV_ROWS";

    /// <summary>Test hook: false keeps the #1237 per-column GEMVs (A/B and the negative control of the multi-column tests).</summary>
    internal static bool ExactGemvEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_EXACT_GEMV") != "0";

    /// <summary>#1459 per-row-count NR tables (index = rows; 0 = default). Measured on Nano Q4_K_M, Strix, verify ms.</summary>
    private static readonly int[] ExactRowsByColumns = ParseTable(0, [0, 0, 0, 0, 3, 2, 0, 2]);
    private static readonly int[] KQuantRowsByColumns = ParseTable(1, [0, 0, 0, 0, 0, 3, 0, 2]);

    /// <summary>
    /// #1516: Q4_K has its own table since #1511 replaced its shader. The pipelined 16-byte-load Q4_K kernel has no 5- or 7-column cliff (the old
    /// <see cref="KQuantRowsByColumns"/> entries and the 5 -> 6 / 7 -> 8 padding were tuned to it), so NR stays at the default 4 for 2..12 columns and drops
    /// to 3 for 13..16 (the same crossover as the Qwen3.8-27B shapes; see the measurements in docs/SPECULATIVE.md). Q6_K keeps <see cref="KQuantRowsByColumns"/>
    /// and the padding: its shader is unchanged and still has the cliff. <c>DOTLLM_VK_NEMOTRON_Q4K_LEGACY=1</c> restores the #1459 behaviour for A/B.
    /// </summary>
    private static readonly int[] Q4KRowsByColumns = ParseTable(2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]);

    private static bool Q4KLegacyTuning => Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_Q4K_LEGACY") == "1";

    // DOTLLM_VK_NEMOTRON_PERN_TABLE="n:exactNR:kquantNR[:q4kNR];..." overrides entries (tuning sweeps; the optional fourth field is Q4_K's own table, #1516).
    private static int[] ParseTable(int which, int[] def)
    {
        var t = new int[17]; def.CopyTo(t, 0);
        if (Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_PERN_TABLE") is { Length: > 0 } e)
            foreach (var ent in e.Split(';'))
            {
                var f = ent.Split(':');
                if (f.Length is 3 or 4 && int.TryParse(f[0], out int ni) && ni is >= 2 and <= 16)
                    t[ni] = int.Parse(which == 0 ? f[1] : which == 1 ? f[2] : f.Length == 4 ? f[3] : f[2]);
            }
        return t;
    }

    /// <summary>Test/A-B hook: false restores the single NR for every row count.</summary>
    internal static bool PerRowCountTuning { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_PERN") != "0";

    /// <summary>Test/A-B hook: false keeps odd 5/7-row dispatches on their own pipelines (#1459).</summary>
    internal static bool PadOddColumns { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_PADODD") != "0";

    private void CreateExactGemvKernels(VulkanDevice device, string spvDir)
    {
        if (!ExactGemvEnabled) return;
        // Verify rows per workgroup measured on Strix at Bolt's shapes (8-row verify, ms over all layers): NVFP4 NR 1/2/4 = 34/23.5/23.2,
        // IQ2_XXS 21.5/14.9/17.2, Q8_0 (LM head + attention) best at 4. Decode is flat in NR, so it stays 1.
        int decodeRows = 1, nvfp4Rows = 4, iq2Rows = 2, q8Rows = 4;
        if (Environment.GetEnvironmentVariable(ExactGemvRowsEnvVar)?.Split(',') is [var d, var mr]
            && int.TryParse(d, out int dv) && int.TryParse(mr, out int mv))
        {
            decodeRows = dv;
            nvfp4Rows = iq2Rows = q8Rows = mv;
        }

        _exactQ8 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Q8_0, decodeRows, q8Rows);
        if (_matmulNvfp4 is not { HasMmvq: true })
            _exactNvfp4 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Nvfp4, decodeRows, nvfp4Rows);
        _exactIq2Xxs = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Iq2Xxs, decodeRows, iq2Rows);
        _exactQ5_0 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Q5_0, decodeRows, q8Rows);
        _exactQ4_0 = MatMulExactGemvMultiKernel.TryCreate(device, spvDir, ExactGemvFormat.Q4_0, decodeRows, q8Rows);
        // #1459: per-n NR for the legacy-quant / Q8_0 kernels (Nano Q4_K_M sweep); an explicit DOTLLM_VK_NEMOTRON_GEMV_ROWS wins.
        if (PerRowCountTuning && Environment.GetEnvironmentVariable(ExactGemvRowsEnvVar) is null)
            foreach (var k in new[] { _exactQ8, _exactQ5_0, _exactQ4_0 })
                if (k is not null) k.RowsByColumns = ExactRowsByColumns;
        if (PadOddColumns)
            foreach (var k in new[] { _exactQ8, _exactQ5_0, _exactQ4_0 })
                if (k is not null) k.PadOddColumns = true;
        if (Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_KQUANT_ROWS") != "0")
        {
            int kMulti = 4, kWide = 4;
            if (Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRON_KQUANT_NR")?.Split(',') is [var km, var kw]
                && int.TryParse(km, out int kmv) && int.TryParse(kw, out int kwv)) { kMulti = kmv; kWide = kwv; }
            bool q4kLegacy = Q4KLegacyTuning;
            _exactQ4K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q4_K, 2, kMulti, kWide,
                PerRowCountTuning ? (q4kLegacy ? KQuantRowsByColumns : Q4KRowsByColumns) : null);
            _exactQ6K = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_Rows, 2, kMulti, kWide, PerRowCountTuning ? KQuantRowsByColumns : null);
            if (_exactQ4K is not null) _exactQ4K.PadOddColumns = PadOddColumns && q4kLegacy;   // #1516: the pipelined shader has no 5/7 cliff; padding only adds work
            if (_exactQ6K is not null) _exactQ6K.PadOddColumns = PadOddColumns;
        }
    }

    /// <summary>Records <paramref name="cols"/> rows (1 = decode) through the exact multi-column GEMV of <paramref name="qt"/>, or returns false.</summary>
    private bool TryRecordExactGemv(nint cmdBuf, VulkanDevice.Buffer weights, QuantizationType qt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output, int outputDim, int inputDim, int cols, float scale)
    {
        var kq = qt switch { QuantizationType.Q4_K => _exactQ4K, QuantizationType.Q6_K => _exactQ6K, _ => null };
        if (kq is not null)
        {
            if (!(cols == 1 ? kq.SupportsSingleColumn && inputDim > 0 && inputDim % 256 == 0 : kq.AcceptsColumns(cols, inputDim))) return false;
            VulkanNemotronHWeights.ThrowIfScaleUnsupported("projection", scale, qt);
            kq.Record(cmdBuf, weights, input, output, outputDim, inputDim, cols);
            return true;
        }
        var kernel = qt switch
        {
            QuantizationType.Q5_0 => _exactQ5_0,
            QuantizationType.Q4_0 => _exactQ4_0,
            QuantizationType.Q8_0 => _exactQ8,
            QuantizationType.NVFP4 => _exactNvfp4,
            QuantizationType.IQ2_XXS => _exactIq2Xxs,
            _ => null,
        };
        if (kernel is null || !kernel.Accepts(cols, inputDim)) return false;
        if (qt != QuantizationType.NVFP4) VulkanNemotronHWeights.ThrowIfScaleUnsupported("projection", scale, qt);
        kernel.Record(cmdBuf, weights, input, output, outputDim, inputDim, cols, scale,
            _iq2Codebooks.Iq2XxsGrid, _iq2Codebooks.Ksigns);
        return true;
    }

    /// <summary>Records <paramref name="cols"/> rows through the decode GEMV of <paramref name="qt"/>, or returns false when it has none.</summary>
    private bool TryRecordGemvColumns(nint cmdBuf, VulkanDevice.Buffer weights, QuantizationType qt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output, int outputDim, int inputDim, int cols, float scale)
    {
        switch (qt)
        {
            case QuantizationType.NVFP4 when _matmulNvfp4 is not null:
                _matmulNvfp4.RecordGemvColumns(cmdBuf, weights, input, output, outputDim, inputDim, cols, scale);
                return true;
            case QuantizationType.Q8_0:
                VulkanNemotronHWeights.ThrowIfScaleUnsupported("projection", scale, qt);
                _matmulQ8.Record(cmdBuf, weights, input, output, outputDim, inputDim, cols);
                return true;
            case QuantizationType.IQ2_XXS:
                VulkanNemotronHWeights.ThrowIfScaleUnsupported("projection", scale, qt);
                _matmulIq2Xxs.Record(cmdBuf, weights, input, output, outputDim, inputDim, cols);
                return true;
            case QuantizationType.F32 when _matmul.CanGemvColumns(inputDim):
                VulkanNemotronHWeights.ThrowIfScaleUnsupported("projection", scale, qt);
                _matmul.RecordGemvColumns(cmdBuf, weights, input, output, outputDim, inputDim, cols);
                return true;
            default:
                return false;
        }
    }

    /// <summary>The decode attention of one query row: split-KV (Flash-Decoding) when it splits, the per-token kernel otherwise.</summary>
    private void RecordDecodeAttention(nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer output, int seqKv, int positionOffset, int numHeads, int numKvHeads, int headDim)
    {
        if (_splitKvAttention is not null && headDim <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads))
        {
            _splitKvAttention.Record(cmdBuf, q, k, v, output, seqQ: 1, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim, positionOffset: positionOffset, slidingWindow: 0);
        }
        else
        {
            _attention.Record(cmdBuf, q, k, v, output, seqQ: 1, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim, positionOffset: positionOffset, slidingWindow: 0);
        }
    }

    // ── Recurrent rollback ──────────────────────────────────────────────
    //
    // Two protocols, one pre-verify copy of the state (_ckptConv/_ckptSsm, persistent):
    //  * checkpoint + restore (+ the decoder's replay forward of the accepted rows), and
    //  * per-row snapshots (#1194's idea for Mamba-2): the verify also records each SSM layer's scan inputs (x, dt, B, C) and conv
    //    window rows; a rollback to row r copies the pre-verify SSM state back, re-runs the same scan dispatch over rows 0..r and
    //    copies row r's conv window. No second trunk forward, and no F32 state per row.
    // Every copy and replay is deferred to the head of the next forward's command buffer (restore, then save): no extra submit.

    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    public object? CheckpointRecurrentState()
    {
        var cache = _activeSsm ?? _ssmCache;
        FlushPendingRecurrentOps();   // a read-only verify's commit must land before the state is copied
        EnsureCheckpointBuffers(cache);
        _ckptPendingSave = cache;   // recorded at the start of the next forward: the verify
        return new SsmCheckpointToken(cache);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        if (checkpoint is null) return;
        if (checkpoint is not SsmCheckpointToken t)
            throw new ArgumentException($"Not a {nameof(VulkanNemotronHTransformerModel)} checkpoint.", nameof(checkpoint));
        if (_ckptPendingSave is not null)
        {
            // Never saved (no forward ran since the checkpoint): the live state is already the checkpointed one.
            _ckptPendingSave = null;
            return;
        }
        _ckptPendingRestore = t.Cache;
        if (!ReferenceEquals(t.Cache, _activeSsm ?? _ssmCache))
            FlushPendingRecurrentOps();   // not the state the next forward runs on: do it now
    }

    private sealed record SsmCheckpointToken(VulkanSsmStateCache Cache);

    /// <inheritdoc/>
    public bool SupportsRecurrentRowSnapshots => _ssmCache.NumSsmLayers > 0;

    /// <summary>Device bytes of the per-row snapshot scratch: scan inputs and conv windows of a full verify (0 until first used).</summary>
    public long RecurrentRowSnapshotBytes
        => _snapX is null ? 0 : _snapX.Sum(b => b.Size) + _snapDt!.Sum(b => b.Size) + _snapB!.Sum(b => b.Size)
            + _snapC!.Sum(b => b.Size) + _snapConv!.Sum(b => b.Size);

    /// <inheritdoc/>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                                 int deviceId, IKvCache? kvCache, IMtpState? mtpState)
    {
        if (tokenIds.Length > RowExactMaxTokens)
            throw new ArgumentException($"Per-row snapshots cover at most {RowExactMaxTokens} rows.", nameof(tokenIds));
        var cache = _activeSsm ?? _ssmCache;
        bool readOnly = ReadOnlyVerifyEnabled && ScanKeepsStateInRegisters;
        // Before any recording: allocating invalidates nothing the forward has bound, but keep it out of the command buffer anyway.
        EnsureRowSnapshotBuffers(cache);
        if (!readOnly)
        {
            EnsureCheckpointBuffers(cache);
            _ckptPendingSave = cache;
        }
        _snapRequestRows = tokenIds.Length;
        _snapValidRows = 0;
        _roVerify = readOnly;
        ITensor logits;
        try { logits = Forward(tokenIds, positions, deviceId, kvCache, adapter: null, mtpState); }
        finally { _snapRequestRows = 0; _roVerify = false; }
        _snapValidRows = tokenIds.Length;
        _snapCache = cache;
        _snapReadOnly = readOnly;
        if (readOnly) _pendingCommitRows = tokenIds.Length;   // every row accepted unless a rollback says otherwise
        return logits;
    }

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if ((uint)row >= (uint)_snapValidRows)
            throw new InvalidOperationException(
                $"No recurrent snapshot for row {row}: the last ForwardWithRecurrentSnapshots recorded {_snapValidRows} row(s), " +
                "or a later forward invalidated them.");
        if (_snapReadOnly)
        {
            _pendingCommitRows = row + 1;   // the replay advances the untouched live state through the accepted rows only
            _snapValidRows = 0;
            if (!ReferenceEquals(_snapCache, _activeSsm ?? _ssmCache)) FlushPendingRecurrentOps();
            return;
        }
        if (row == _snapValidRows - 1)
        {
            _snapValidRows = 0;
            return;   // the live state IS the state after the last row
        }
        _pendingRestoreRow = row;
        _snapValidRows = 0;
        if (!ReferenceEquals(_snapCache, _activeSsm ?? _ssmCache))
            FlushPendingRecurrentOps();
    }

    private void EnsureCheckpointBuffers(VulkanSsmStateCache cache)
    {
        int n = cache.NumSsmLayers;
        if (_ckptConv is not null && _ckptConv.Length == n) return;
        DisposeCheckpointBuffers();
        _ckptConv = new VulkanDevice.Buffer[n];
        _ckptSsm = new VulkanDevice.Buffer[n];
        for (int i = 0; i < n; i++)
        {
            _ckptConv[i] = _device.AllocateDeviceLocal(cache.GetConvStateBuffer(i).Size);
            _ckptSsm[i] = _device.AllocateDeviceLocal(cache.GetSsmStateBuffer(i).Size);
        }
    }

    private void EnsureRowSnapshotBuffers(VulkanSsmStateCache cache)
    {
        int n = cache.NumSsmLayers;
        if (_snapX is not null && _snapX.Length == n) return;
        DisposeRowSnapshotBuffers();
        int rows = RowExactMaxTokens;
        long f = sizeof(float);
        _snapX = new VulkanDevice.Buffer[n];
        _snapDt = new VulkanDevice.Buffer[n];
        _snapB = new VulkanDevice.Buffer[n];
        _snapC = new VulkanDevice.Buffer[n];
        _snapConv = new VulkanDevice.Buffer[n];
        int bcDim = _ssm.NGroup * _ssm.DState;
        for (int i = 0; i < n; i++)
        {
            _snapX[i] = _device.AllocateDeviceLocal(rows * _ssm.DInner * f);
            _snapDt[i] = _device.AllocateDeviceLocal(rows * _ssm.NHead * f);
            _snapB[i] = _device.AllocateDeviceLocal(rows * bcDim * f);
            _snapC[i] = _device.AllocateDeviceLocal(rows * bcDim * f);
            _snapConv[i] = _device.AllocateDeviceLocal((_ssm.DConv - 1 + rows) * (long)_ssm.ConvDim * f);
        }
    }

    private void DisposeCheckpointBuffers()
    {
        if (_ckptConv is not null) foreach (var b in _ckptConv) b.Dispose();
        if (_ckptSsm is not null) foreach (var b in _ckptSsm) b.Dispose();
        _ckptConv = _ckptSsm = null;
    }

    private void DisposeRowSnapshotBuffers()
    {
        foreach (var set in new[] { _snapX, _snapDt, _snapB, _snapC, _snapConv })
            if (set is not null) foreach (var b in set) b.Dispose();
        _snapX = _snapDt = _snapB = _snapC = _snapConv = null;
    }

    /// <summary>
    /// During a snapshot forward: copies SSM layer <paramref name="ordinal"/>'s conv input rows (<paramref name="convOnly"/>) or its
    /// scan inputs (x, dt, B, C) for all <paramref name="seqLen"/> rows into the layer's snapshot buffers.
    /// </summary>
    private void RecordSsmRowSnapshot(nint cmdBuf, int ordinal, int seqLen, bool convOnly)
    {
        if (_snapRequestRows == 0) return;
        long f = sizeof(float);
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        if (convOnly)
        {
            RecordCopyBufferRange(cmdBuf, _state.ConvInput, _snapConv![ordinal], 0, 0, (ulong)((_ssm.DConv - 1 + seqLen) * (long)_ssm.ConvDim * f));
        }
        else
        {
            int bcDim = _ssm.NGroup * _ssm.DState;
            RecordCopyBufferRange(cmdBuf, _state.SsmX, _snapX![ordinal], 0, 0, (ulong)(seqLen * (long)_ssm.DInner * f));
            RecordCopyBufferRange(cmdBuf, _state.DtBuf, _snapDt![ordinal], 0, 0, (ulong)(seqLen * (long)_ssm.NHead * f));
            RecordCopyBufferRange(cmdBuf, _state.SsmB, _snapB![ordinal], 0, 0, (ulong)(seqLen * (long)bcDim * f));
            RecordCopyBufferRange(cmdBuf, _state.SsmC, _snapC![ordinal], 0, 0, (ulong)(seqLen * (long)bcDim * f));
        }
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
    }

    /// <summary>Records the deferred rollback (a row replay or a checkpoint restore), then a pending checkpoint save, at a command buffer's head.</summary>
    private void RecordPendingRecurrentOps(nint cmdBuf)
    {
        if (_pendingCommitRows > 0)
        {
            // Read-only verify: advance the live state through the accepted rows (conv window, then the in-place scan replay).
            int rows = _pendingCommitRows;
            var cache = _snapCache!;
            _pendingCommitRows = 0;
            long f = sizeof(float);
            long convBytes = (_ssm.DConv - 1) * (long)_ssm.ConvDim * f;
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            for (int i = 0; i < cache.NumSsmLayers; i++)
                RecordCopyBufferRange(cmdBuf, _snapConv![i], cache.GetConvStateBuffer(i), (ulong)(rows * (long)_ssm.ConvDim * f), 0, (ulong)convBytes);
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            for (int abs = 0, i = 0; abs < Config.NumLayers; abs++)
            {
                if (_ssmLayerOrdinal[abs] < 0) continue;
                if (i > 0) KernelSupport.ComputeToComputeBarrier(cmdBuf);   // the scans share the scratch output
                RecordMamba2Scan(cmdBuf, cache.GetSsmStateBuffer(i), _snapX![i], _snapDt![i], _weights.Layers[abs].Ssm!.Value.A,
                    _snapB![i], _snapC![i], _state.SsmY, rows);
                i++;
            }
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        }
        if (_pendingRestoreRow >= 0)
        {
            int row = _pendingRestoreRow;
            var cache = _snapCache!;
            _pendingRestoreRow = -1;
            long f = sizeof(float);
            long convBytes = (_ssm.DConv - 1) * (long)_ssm.ConvDim * f;
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            for (int i = 0; i < cache.NumSsmLayers; i++)
            {
                var ssm = cache.GetSsmStateBuffer(i);
                RecordCopyBufferRange(cmdBuf, _ckptSsm![i], ssm, 0, 0, (ulong)ssm.Size);
                // Row r's conv window: conv-input rows r+1 .. r+dConv-1 (rows 0..dConv-2 are the pre-verify conv state).
                RecordCopyBufferRange(cmdBuf, _snapConv![i], cache.GetConvStateBuffer(i), (ulong)((row + 1) * (long)_ssm.ConvDim * f), 0, (ulong)convBytes);
            }
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            for (int abs = 0, i = 0; abs < Config.NumLayers; abs++)
            {
                if (_ssmLayerOrdinal[abs] < 0) continue;
                if (i > 0) KernelSupport.ComputeToComputeBarrier(cmdBuf);   // the scans share the scratch output
                RecordMamba2Scan(cmdBuf, cache.GetSsmStateBuffer(i), _snapX![i], _snapDt![i], _weights.Layers[abs].Ssm!.Value.A,
                    _snapB![i], _snapC![i], _state.SsmY, row + 1);
                i++;
            }
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        }
        if (_ckptPendingSave is null && _ckptPendingRestore is null) return;
        if (_ckptPendingRestore is { } restore)
        {
            _ckptPendingRestore = null;
            RecordStateCopies(cmdBuf, restore, toCheckpoint: false);
        }
        if (_ckptPendingSave is { } save)
        {
            _ckptPendingSave = null;
            RecordStateCopies(cmdBuf, save, toCheckpoint: true);
        }
    }

    private void RecordStateCopies(nint cmdBuf, VulkanSsmStateCache cache, bool toCheckpoint)
    {
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        for (int i = 0; i < cache.NumSsmLayers; i++)
        {
            var conv = cache.GetConvStateBuffer(i);
            var ssm = cache.GetSsmStateBuffer(i);
            if (toCheckpoint)
            {
                RecordCopyBufferRange(cmdBuf, conv, _ckptConv![i], 0, 0, (ulong)conv.Size);
                RecordCopyBufferRange(cmdBuf, ssm, _ckptSsm![i], 0, 0, (ulong)ssm.Size);
            }
            else
            {
                RecordCopyBufferRange(cmdBuf, _ckptConv![i], conv, 0, 0, (ulong)conv.Size);
                RecordCopyBufferRange(cmdBuf, _ckptSsm![i], ssm, 0, 0, (ulong)ssm.Size);
            }
        }
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
    }

    /// <summary>Runs any deferred rollback or checkpoint in its own submit.</summary>
    private void FlushPendingRecurrentOps()
    {
        if (_pendingCommitRows == 0 && _pendingRestoreRow < 0 && _ckptPendingSave is null && _ckptPendingRestore is null) return;
        _submit.Begin();
        RecordPendingRecurrentOps(_submit.CommandBuffer);
        _submit.SubmitAndWait();
    }

    /// <summary>Forgets deferred rollback / checkpoint work (the state is being reset).</summary>
    private void DropPendingRecurrentOps()
    {
        _pendingRestoreRow = -1;
        _pendingCommitRows = 0;
        _ckptPendingSave = _ckptPendingRestore = null;
        _snapValidRows = 0;
    }

    // ── IDFlashTargetOps ────────────────────────────────────────────────

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
    {
        var g = _dflashGpu!;
        if (_state.EnsureCapacity(tokens.Length)) InvalidateKernelCaches();
        nint cmd = g.Command();
        RecordEmbeddingGather(cmd, tokens);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var (buf, off) = g.Resolve(dst);
        RecordCopyBufferRange(cmd, _state.HiddenState, buf, 0, (ulong)off, (ulong)((long)tokens.Length * Config.HiddenSize * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
    }

    /// <inheritdoc/>
    unsafe void IDFlashTargetOps.ProjectForDraft(nint hidden, int rows, float* logits)
    {
        var g = _dflashGpu!;
        int h = Config.HiddenSize, vocab = Config.VocabSize;
        bool resized = _state.EnsureCapacity(rows);
        var logitsBuf = rows == 1 ? _state.Logits : EnsureMultiRowLogits(rows, vocab);
        if (resized) InvalidateKernelCaches();
        nint cmd = g.Command();
        var (buf, off) = g.Resolve(hidden);
        RecordCopyBufferRange(cmd, buf, _state.NormOutput, (ulong)off, 0, (ulong)((long)rows * h * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
        RecordMatmul(cmd, _weights.OutputWeight, _weights.OutputDeviceQuantType, _state.NormOutput, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: rows);
        g.Flush();
        _device.Download(logitsBuf, new Span<float>(logits, rows * vocab));
    }

    /// <inheritdoc/>
    unsafe void IDFlashTargetOps.EmbedHost(ReadOnlySpan<int> tokens, float* dst)
    {
        int h = Config.HiddenSize;
        long rowBytes = Cpu.Kernels.Dequantize.RowByteSize(h, _hostTokenEmbedQt);
        for (int t = 0; t < tokens.Length; t++)
        {
            if ((uint)tokens[t] >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokens));
            Cpu.Kernels.Dequantize.ToFloat32(_hostTokenEmbed + (nint)(tokens[t] * rowBytes), h, _hostTokenEmbedQt,
                new Span<float>(dst + (long)t * h, h));
        }
    }

    // ── IVulkanDFlashHost ───────────────────────────────────────────────

    VulkanDevice IVulkanDFlashHost.Device => _device;

    RmsNormF32Kernel IVulkanDFlashHost.DFlashRmsNorm => _rmsnorm;

    SwiGluF32Kernel IVulkanDFlashHost.DFlashSwiGlu => _swiglu;

    AddKernel IVulkanDFlashHost.DFlashAdd => _add;

    long IVulkanDFlashHost.DFlashUploadBytes(in DFlashDrafter.Matrix w) => VulkanNemotronHWeights.DFlashUploadBytes(w.Out, w.In, w.Qt);

    VulkanDevice.Buffer IVulkanDFlashHost.UploadDFlashMatrix(VulkanStagingBuffer stage, in DFlashDrafter.Matrix w,
        out QuantizationType deviceQt, out long bytes)
        => VulkanNemotronHWeights.UploadDFlashMatrix(_device, stage, w.Ptr, w.Qt, w.Out, w.In, out deviceQt, out bytes);

    void IVulkanDFlashHost.RecordDFlashMatmul(nint cmd, VulkanDevice.Buffer w, QuantizationType qt, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int outDim, int inDim, int rows, float scale)
        => RecordMatmul(cmd, w, qt, x, y, outputDim: outDim, inputDim: inDim, seqLen: rows, scale: scale);

    void IVulkanDFlashHost.OnDFlashBackendDisposed(VulkanDFlashBackend backend)
    {
        if (ReferenceEquals(_dflashGpu, backend)) _dflashGpu = null;
    }

    private void DisposeDFlash()
    {
        _dflash?.Dispose();
        _dflash = null;
        _dflashGpu = null;
        _dflashTapBuffer?.Dispose();
        _dflashTapBuffer = null;
        DisposeCheckpointBuffers();
        DisposeRowSnapshotBuffers();
    }
}
