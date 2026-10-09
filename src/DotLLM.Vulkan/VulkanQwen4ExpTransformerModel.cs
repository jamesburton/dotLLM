using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-sequence state of a <see cref="VulkanQwen4ExpTransformerModel"/>: the device-resident Gated-DeltaNet state of every GDN layer,
/// the host-side PLE hash window + dilated-conv history, and (when no engine <see cref="IKvCache"/> is supplied) the dense K/V cache of every QSA layer.
/// </summary>
public sealed class VulkanQwen4ExpSequenceState : IGdnState
{
    private readonly Func<VulkanNemotronHKvCache>? _kvFactory;
    private VulkanNemotronHKvCache? _kv;

    internal VulkanGdnStateCache Gdn { get; }
    internal Qwen4ExpPleState? Ple { get; }

    /// <inheritdoc/>
    public int NumGdnLayers => Gdn.NumGdnLayers;

    /// <summary>The state's own dense QSA K/V rows, allocated on first use (an engine-supplied KV cache bypasses them entirely).</summary>
    internal VulkanNemotronHKvCache OwnKv => _kv ??= _kvFactory!();

    /// <summary>Tokens consumed so far (the next position).</summary>
    public int Length { get; internal set; }

    internal VulkanQwen4ExpSequenceState(VulkanGdnStateCache gdn, Func<VulkanNemotronHKvCache> kvFactory, Qwen4ExpPleState? ple)
    {
        Gdn = gdn; _kvFactory = kvFactory; Ple = ple;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        Gdn.Reset();
        _kv?.Rollback(0);
        Ple?.Reset();
        Length = 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Gdn.Dispose();
        _kv?.Dispose();
    }
}

/// <summary>
/// Vulkan forward pass for Qwen4-Exp / Qwen3.8-Flash-Next (<c>qwen4exp</c>): 48 blocks of <c>(GDN | QSA) + 512-expert softmax MoE</c> on a
/// 4-stream gated residual, an n-gram hash embedding injected before layer 1, and a head mixer in place of the final norm (#818).
/// </summary>
/// <remarks>
/// <para>
/// <b>Composition, not a fork.</b> The Gated-DeltaNet layer, full-GQA attention layer, routed+shared MoE layer and the quant-aware
/// matmul dispatcher are the already-validated blocks of <see cref="VulkanQwen3MoeHybridTransformerModel"/>, driven through its
/// <c>Q4*</c> surface; this class adds the gated-residual plumbing (<see cref="Qwen4ExpGatedResidualKernel"/>), the sigmoid GDN output
/// gate, the host n-gram branch and the sequence state. It mirrors the CPU oracle <c>Qwen4ExpTransformerModel</c> step for step.
/// </para>
/// <para>
/// <b>V1 scope.</b> (a) Attention is DENSE: exact while the context is within the indexer budget (<c>top_k + block - 1</c> tokens,
/// 2051 on the released model); beyond that the call fails loudly instead of silently diverging from sparse QSA (V2, #819).
/// (b) The n-gram branch runs on the host at its layer: the residual is downloaded once, the CPU branch (<see cref="Qwen4ExpPleBranch"/>,
/// the oracle's own code) gathers the 16 table rows from the mmap'd table and runs key/value projections, gate and the dilated conv, and
/// the residual is uploaded back. The 28.8 GB table is registered host-only and can never be imported or staged. (c) Sequence state is per-sequence
/// (<see cref="CreateSequenceState"/>, threaded through <see cref="ForwardBatch"/>) with the QSA K/V rows in the engine KV cache (#871); no recurrent checkpoint/rollback yet. (d) No MTP block.
/// </para>
/// </remarks>
public sealed unsafe partial class VulkanQwen4ExpTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly GgufFile _gguf;
    private readonly VulkanQwen3MoeHybridTransformerModel _core;
    private readonly Qwen4ExpConfig _q4;
    private readonly VulkanQwen4ExpGrWeights[] _attnGr, _ffnGr;
    private readonly VulkanQwen4ExpGrWeights _headGr;
    private readonly VulkanQwen3MoeMoeUpload.LayerBundle[] _moe;
    private readonly Qwen4ExpPleBranch? _ple;
    private readonly int _pleLayer;
    private readonly List<nint> _owned;
    private readonly (nint Ptr, long Bytes)? _hostOnlyTable;
    private readonly Qwen4ExpGatedResidualKernel _gr;
    private readonly GroupRmsNormF32Kernel _groupRms;
    private readonly GdnPostScanGateF32Kernel _sigmoidGate;
    private readonly int _hidden, _streams, _lowRank, _vocab, _kvCapacity;
    private readonly float _eps;
    private readonly VulkanQwen4ExpSequenceState _defaultState;
    private readonly long _weightBytes;

    // Own scratch (the block input / output and every MoE / GDN / attention scratch buffer is the core's).
    private VulkanDevice.Buffer _res = null!, _xn = null!, _low = null!, _mix = null!, _gains = null!;
    private VulkanDevice.Buffer _headRes = null!;
    private int _scratchCapacity;
    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _core.ComputeMemoryBytes + _weightBytes;

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <summary>
    /// Maximum number of tokens the dense attention fallback is exact for: <c>indexer.top_k + block - 1</c> (also the KV capacity). Beyond
    /// it the forward throws <see cref="NotSupportedException"/> until sparse QSA selection exists (#819).
    /// </summary>
    public int DenseContextLimit => _kvCapacity;

    /// <summary>Test/diagnostic hook: receives <c>(name, data, rows, cols)</c> after each block (residual) (null = off, zero cost).</summary>
    internal Action<string, float[], int, int>? Trace { get; set; }

    /// <summary>Device storage type of every layer's routed gate / down / up bank (F32 = widened because no resident kernel exists for the source quant).</summary>
    internal IReadOnlyList<(Core.Configuration.QuantizationType Gate, Core.Configuration.QuantizationType Down, Core.Configuration.QuantizationType Up)> ExpertBankDeviceTypes
        => _moe.Select(m => (m.W1QuantType, m.W2QuantType, m.W3QuantType)).ToArray();

    /// <summary>Test hook (#849): how many times each routed-MoE fast path was recorded (see <c>MoePath</c>).</summary>
    internal long MoePathCount(VulkanQwen3MoeHybridTransformerModel.MoePath p) => _core.MoePathCounts[(int)p];

    /// <summary>Test hook (#876): how many times each 2..8-row fast path was recorded.</summary>
    internal long SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath p) => _core.SmallRowPathCounts[(int)p];

    /// <summary>Address of the n-gram table inside the GGUF mapping (0 when absent) - tests assert nothing was uploaded from this range.</summary>
    internal (nint Pointer, long Bytes) HostOnlyTableRange => _hostOnlyTable ?? (0, 0);

    /// <summary>Bytes of the n-gram table the model deliberately keeps off the device (0 when the checkpoint has none).</summary>
    public long HostOnlyTableBytes => _hostOnlyTable?.Bytes ?? 0;

    private VulkanQwen4ExpTransformerModel(
        VulkanDevice device, GgufFile gguf, ModelConfig config, VulkanQwen3MoeHybridTransformerModel core,
        VulkanQwen4ExpGrWeights[] attnGr, VulkanQwen4ExpGrWeights[] ffnGr, VulkanQwen4ExpGrWeights headGr,
        VulkanQwen3MoeMoeUpload.LayerBundle[] moe, Qwen4ExpPleBranch? ple, int pleLayer, List<nint> owned,
        (nint, long)? hostOnlyTable, Qwen4ExpGatedResidualKernel gr, GroupRmsNormF32Kernel groupRms,
        GdnPostScanGateF32Kernel sigmoidGate, int kvCapacity, long weightBytes)
    {
        _device = device; _gguf = gguf; Config = config; _core = core; _q4 = config.Qwen4Exp!;
        _attnGr = attnGr; _ffnGr = ffnGr; _headGr = headGr; _moe = moe; _ple = ple; _pleLayer = pleLayer;
        _owned = owned; _hostOnlyTable = hostOnlyTable; _gr = gr; _groupRms = groupRms; _sigmoidGate = sigmoidGate;
        _hidden = config.HiddenSize; _streams = _q4.HyperConnectionCount; _lowRank = _q4.HyperConnectionLowRank;
        _vocab = config.VocabSize; _eps = config.NormEpsilon; _kvCapacity = kvCapacity; _weightBytes = weightBytes;
        EnsureScratch(1);
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>Allocates a fresh sequence state (KV capacity <see cref="DenseContextLimit"/>).</summary>
    public VulkanQwen4ExpSequenceState CreateState()
        => new(_core.Q4CreateGdnState(), () => _core.Q4CreateKvCache(_kvCapacity), _ple?.CreateState());

    /// <summary>Allocates an engine KV cache for the QSA layers (capacity clamped to <see cref="DenseContextLimit"/>).</summary>
    public VulkanNemotronHKvCache CreateKvCache(int maxSeqLen) => _core.Q4CreateKvCache(Math.Min(maxSeqLen, _kvCapacity));

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => CreateState();

    private int _allRowLogitsLimit = 1;

    /// <inheritdoc/>
    public int MaxAllRowLogitsLength => _allRowLogitsLimit;

    /// <inheritdoc/>
    public bool TrySetAllRowLogitsLimit(int maxSeqLen)
    {
        if (maxSeqLen > _allRowLogitsLimit) _allRowLogitsLimit = Math.Min(maxSeqLen, _kvCapacity + 1);
        return _allRowLogitsLimit >= maxSeqLen;
    }

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    private void EnsureScratch(int seqLen)
    {
        if (seqLen <= _scratchCapacity) return;
        _res?.Dispose(); _xn?.Dispose(); _low?.Dispose(); _mix?.Dispose(); _gains?.Dispose(); _headRes?.Dispose();
        long row = (long)_streams * _hidden;
        _res = _device.AllocateDeviceLocal(seqLen * row * 4);
        _xn = _device.AllocateDeviceLocal(seqLen * row * 4);
        _low = _device.AllocateDeviceLocal((long)seqLen * _lowRank * 4);
        _mix = _device.AllocateDeviceLocal(seqLen * row * 4);
        _gains = _device.AllocateDeviceLocal((long)seqLen * _streams * 4);
        _headRes = _device.AllocateDeviceLocal(row * 4);
        _scratchCapacity = seqLen;
        _gr?.InvalidateDescriptorCache(); _groupRms?.InvalidateDescriptorCache(); _sigmoidGate?.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Diagnostic (#876): split-submit per-stage wall timing. When on, every stage of the forward (and the core's GDN / attention / MoE
    /// sub-stages, at any row count) is followed by a submit + wait and its host-observed ms accumulated; read with <see cref="TakeStageTimes"/>.
    /// Off by default; <c>DOTLLM_VULKAN_MOE_STAGE_PROFILE=1</c> turns it on at startup. Each stage carries ~50 us of sync cost.
    /// </summary>
    public static bool StageProfile
    {
        get => VulkanQwen3MoeHybridTransformerModel.StageProfileEnabled;
        set => VulkanQwen3MoeHybridTransformerModel.StageProfileEnabled = value;
    }

    /// <summary>Diagnostic (#876): the 2..8-row multi-column GEMVs (Q8_0 MMVQ, F32). On by default; <c>DOTLLM_VK_SMALLROW_GEMV=0</c> disables them at startup.</summary>
    public static bool SmallRowGemv
    {
        get => VulkanQwen3MoeHybridTransformerModel.SmallRowGemvEnabled;
        set => VulkanQwen3MoeHybridTransformerModel.SmallRowGemvEnabled = value;
    }

    /// <summary>Returns and clears the accumulated per-stage times (ms) recorded while <see cref="StageProfile"/> was on.</summary>
    public IReadOnlyDictionary<string, double> TakeStageTimes() => _core.TakeStageTimes();

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, null, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly);

    /// <summary>
    /// Per-sequence loop over the forward: each request's <see cref="SequenceForwardRequest.GdnState"/> (a
    /// <see cref="VulkanQwen4ExpSequenceState"/>) is threaded through, with the QSA K/V rows in the request's KV cache. Returns the LAST
    /// row's logits (<c>[1, vocab]</c>) per request. Interleaved sequences equal separate runs (state is per sequence).
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].GdnState is null && requests.Count >= 2)
                throw new ArgumentException(
                    $"Multi-seq ForwardBatch requires each SequenceForwardRequest to carry its own GdnState (request {i} has none): " +
                    "the model-owned default state would be shared across sequences. Allocate one with CreateSequenceState().", nameof(requests));
            if (requests[i].GdnState is not null and not VulkanQwen4ExpSequenceState)
                throw new ArgumentException(
                    $"Request {i} carries a {requests[i].GdnState!.GetType().Name}; qwen4exp on Vulkan needs a VulkanQwen4ExpSequenceState.", nameof(requests));
            if (requests[i].Adapter is not null)
                throw new NotSupportedException("LoRA adapters are not supported by the Vulkan qwen4exp model (CPU only, #845).");
        }
        var results = new List<ITensor>(requests.Count);
        try
        {
            foreach (var r in requests)
            {
                var st = (VulkanQwen4ExpSequenceState?)r.GdnState ?? _defaultState;
                results.Add(ForwardCore(r.TokenIds.Span, r.Positions.Span, deviceId, st, r.KvCache, lastTokenLogitsOnly: true));
            }
        }
        catch
        {
            foreach (var t in results) t.Dispose();
            throw;
        }
        return results;
    }

    /// <summary>
    /// Forward over a caller-owned sequence state (chunked prefill / decode). <c>positions[i]</c> must equal <c>state.Length + i</c>;
    /// the model-owned default state restarts when a call begins at position 0. Returns the LAST token's logits <c>[1, vocab]</c>.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, VulkanQwen4ExpSequenceState state)
        => ForwardCore(tokenIds, positions, deviceId, state, null, lastTokenLogitsOnly: false);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, VulkanQwen4ExpSequenceState state,
                                IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        ArgumentNullException.ThrowIfNull(state);
        VulkanNemotronHKvCache kv;
        if (kvCache is null) kv = state.OwnKv;
        else if (kvCache is VulkanNemotronHKvCache vk) kv = vk;
        else throw new ArgumentException($"qwen4exp on Vulkan needs a VulkanNemotronHKvCache (from CreateKvCache); got {kvCache.GetType().Name}.", nameof(kvCache));
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (ReferenceEquals(state, _defaultState) && positions[0] == 0 && state.Length != 0) state.Reset();
        for (int i = 0; i < T; i++)
            if (positions[i] != state.Length + i)
                throw new ArgumentException($"qwen4exp keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {state.Length + i}.");
        if (state.Length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        if (state.Length + T > _kvCapacity)
            throw new NotSupportedException(
                $"qwen4exp on Vulkan attends densely, which is exact only up to {_kvCapacity} tokens (indexer budget {_q4.IndexerTopK} + block " +
                $"{_q4.IndexerBlockSize} - 1); this call would reach {state.Length + T}. Sparse QSA block selection is issue #819.");
        if (kvCache is not null)
        {
            if (kv.CurrentLength < state.Length)
                throw new InvalidOperationException(
                    $"The KV cache holds {kv.CurrentLength} rows but the sequence state is at position {state.Length}: they must advance together.");
            if (kv.CurrentLength > state.Length) kv.Rollback(state.Length);   // stale rows after a rollback are overwritten
            if (kv.MaxLength < state.Length + T)
                throw new NotSupportedException(
                    $"The KV cache holds {kv.MaxLength} positions but this call would reach {state.Length + T}; qwen4exp on Vulkan attends densely up to {_kvCapacity} tokens.");
        }
        for (int i = 0; i < T; i++)
            if ((uint)tokenIds[i] >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {tokenIds[i]} at position {i} is out of range [0, {_vocab}).");

        bool resized = _core.Q4EnsureCapacity(T);
        EnsureScratch(T);
        if (resized) { _gr.InvalidateDescriptorCache(); _groupRms.InvalidateDescriptorCache(); _sigmoidGate.InvalidateDescriptorCache(); }
        _core.Q4UploadPositions(positions);

        var submit = _core.Q4Submit;
        var st = _core.Q4State;
        int H = _hidden, S = _streams, row = S * H;
        long rowBytes = (long)row * 4;
        nint cmd = 0;

        void Begin()
        {
            submit.Begin();
            cmd = submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
        }
        void End()
        {
            KernelSupport.ComputeToHostBarrier(cmd);
            submit.SubmitAndWait();
        }
        void Barrier() => KernelSupport.ComputeTransferFullBarrier(cmd);

        // Splitting a long prefill into per-half-layer submissions keeps each under the driver watchdog (the hybrid model does the same);
        // single-token decode runs as one command buffer except at the host PLE step.
        // Short forwards (decode, 2..8-row MTP verify) stay in ONE command buffer: each split costs a submit + fence wait (~0.1 ms x 96 per forward, #876).
        bool splitHalves = T > SplitHalvesAbove;

        // GR read: group-RMS(src) -> low-rank mix -> block input h (core NormOutput); inject gains when requested.
        void GrRead(VulkanQwen4ExpGrWeights w, VulkanDevice.Buffer src, int tokens, bool inject)
        {
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, src, _xn, 0, 0, (ulong)(tokens * rowBytes));
            Barrier();
            _core.Q4Stage("gr.copy");
            _groupRms.Record(cmd, _xn, w.Norm, tokens, S, H, _eps);
            Barrier();
            _core.Q4Stage("gr.grouprms");
            _core.Q4RecordMatmul(cmd, w.Down, w.DownQt, _xn, _low, outputDim: _lowRank, inputDim: row, seqLen: tokens);
            Barrier();
            _core.Q4Stage("gr.down");
            _gr.RecordActivateLowRank(cmd, _low, tokens * _lowRank, S);
            Barrier();
            _core.Q4Stage("gr.act");
            _core.Q4RecordMatmul(cmd, w.Up, w.UpQt, _low, _mix, outputDim: row, inputDim: _lowRank, seqLen: tokens);
            Barrier();
            _core.Q4Stage("gr.up");
            _gr.RecordMixMean(cmd, st.NormOutput, _mix, _xn, tokens, S, H);
            Barrier();
            _core.Q4Stage("gr.mixmean");
            if (inject)
            {
                _core.Q4RecordMatmul(cmd, w.Inject!, w.InjectQt, _xn, _gains, outputDim: S, inputDim: row, seqLen: tokens);
                Barrier();
                _core.Q4Stage("gr.inject_mm");
                _gr.RecordInjectGains(cmd, _gains, tokens, S);
                Barrier();
                _core.Q4Stage("gr.inject_gains");
            }
        }

        Begin();
        _core.Q4StageBegin();
        _core.Q4RecordEmbedding(cmd, tokenIds);
        Barrier();
        _core.Q4Stage("embed");
        _gr.RecordBroadcast(cmd, _res, st.HiddenState, T, S, H);
        Barrier();

        for (int il = 0; il < Config.NumLayers; il++)
        {
            // ── n-gram branch (host): R += PLE(R, token history) ──
            if (_ple is not null && il == _pleLayer)
            {
                End();
                int n = T * row;
                float[] host = System.Buffers.ArrayPool<float>.Shared.Rent(n);
                try
                {
                    _device.Download(_res, host.AsSpan(0, n));
                    _ple.Apply(tokenIds, state.Ple!, host.AsSpan(0, n));
                    _device.Upload(host.AsSpan(0, n), _res);
                }
                finally { System.Buffers.ArrayPool<float>.Shared.Return(host); }
                Begin();
            }

            // ── token mixer ──
            GrRead(_attnGr[il], _res, T, inject: true);
            if (Config.HybridLayout!.LayerKind[il] == HybridLayerKind.GatedDeltaNet)
                _core.Q4RecordGdn(cmd, il, T, _eps, state.Gdn, _sigmoidGate);
            else
                _core.Q4RecordAttention(cmd, il, T, positions, kv);
            Barrier();
            _gr.RecordWrite(cmd, _res, st.NormOutput, _gains, T, S, H);
            Barrier();
            _core.Q4Stage("gr_write_x");
            if (splitHalves) { End(); Begin(); }

            // ── MoE ──
            GrRead(_ffnGr[il], _res, T, inject: true);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.NormOutput, st.MoeSharedInput, 0, 0, (ulong)((long)T * H * 4));
            Barrier();
            _core.Q4Stage("moe_copy");
            _core.Q4RecordMoe(cmd, _moe[il], T);
            Barrier();
            _core.Q4Stage("moe_total");
            _gr.RecordWrite(cmd, _res, st.NormOutput, _gains, T, S, H);
            Barrier();
            _core.Q4Stage("gr_write_x");
            if (splitHalves) { End(); Begin(); }

            if (Trace is { } tr)
            {
                End();
                float[] dump = new float[T * row];
                _device.Download(_res, dump);
                tr($"blk.{il}.l_out", dump, T, row);
                Begin();
            }
        }

        // ── head mixer (replaces the final norm) + LM head: the LAST row, or every row when the caller opted in (perplexity) ──
        int rows = !lastTokenLogitsOnly && T <= _allRowLogitsLimit ? T : 1;
        var w0 = _core.Q4Weights;
        if (rows == 1)
        {
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, _headRes, (ulong)((T - 1) * rowBytes), 0, (ulong)rowBytes);
            Barrier();
            GrRead(_headGr, _headRes, 1, inject: false);
            _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, st.NormOutput, st.Logits,
                outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: 1);
            End();
            state.Length += T;
            var one = UnmanagedTensor.Allocate(new TensorShape(1, _vocab), DType.Float32, deviceId: -1);
            _device.Download(st.Logits, new Span<float>((void*)one.DataPointer, _vocab));
            return one;
        }

        // All rows: head mixer over all T rows, then the LM head in chunks so the device logits scratch stays small
        // (a 2K window x 248K vocab is 2 GB; the host result tensor is the only full-size allocation).
        GrRead(_headGr, _res, T, inject: false);
        const int Chunk = 32;
        EnsureHeadChunk(Chunk);
        var result = UnmanagedTensor.Allocate(new TensorShape(T, _vocab), DType.Float32, deviceId: -1);
        try
        {
            for (int c0 = 0; c0 < T; c0 += Chunk)
            {
                int n = Math.Min(Chunk, T - c0);
                if (c0 > 0) Begin();
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.NormOutput, _headIn!, (ulong)((long)c0 * H * 4), 0, (ulong)((long)n * H * 4));
                Barrier();
                _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, _headIn!, _headLogits!,
                    outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: n);
                End();
                _device.Download(_headLogits!, new Span<float>((void*)(result.DataPointer + (nint)((long)c0 * _vocab * 4)), n * _vocab));
            }
        }
        catch { result.Dispose(); throw; }
        state.Length += T;
        return result;
    }

    private static int SplitHalvesAbove =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_SPLIT_ABOVE"), out int sa) && sa >= 1 ? sa : 16;

    /// <summary>Diagnostic (#876): forwards longer than this many rows split each half-layer into its own submission (default 16).</summary>
    public static int SplitAbove { get => SplitHalvesAbove; set => SplitHalvesAbove = value; }

    /// <summary>Diagnostic (#876): smallest token count that takes the expert-grouped coopmat MoE path (default 16).</summary>
    public int MoeGroupedMinTokens
    {
        get => _core.GroupedMinTokens;
        set => _core.GroupedMinTokens = value;
    }

    /// <summary>Diagnostic (#876): smallest token count that uses the multi-row routed-MoE MMVQ variants (0 = never; default 2).</summary>
    public static int MoeMultiRowMinRows
    {
        get => VulkanQwen3MoeHybridTransformerModel.MoeMrMinRows;
        set => VulkanQwen3MoeHybridTransformerModel.MoeMrMinRows = value;
    }

    private VulkanDevice.Buffer? _headIn, _headLogits;

    private void EnsureHeadChunk(int rows)
    {
        if (_headIn is not null) return;
        _headIn = _device.AllocateDeviceLocal((long)rows * _hidden * 4);
        _headLogits = _device.AllocateDeviceLocal((long)rows * _vocab * 4);
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _defaultState.Dispose();
        foreach (var m in _moe) m.Dispose();
        foreach (var g in _attnGr) g.Dispose();
        foreach (var g in _ffnGr) g.Dispose();
        _headGr.Dispose();
        _res?.Dispose(); _xn?.Dispose(); _low?.Dispose(); _mix?.Dispose(); _gains?.Dispose(); _headRes?.Dispose();
        _headIn?.Dispose(); _headLogits?.Dispose();
        _gr.Dispose(); _groupRms.Dispose(); _sigmoidGate.Dispose();
        _core.Dispose();
        foreach (nint p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        if (_hostOnlyTable is { } h) VulkanWeightImportPolicy.UnregisterHostOnly(h.Ptr);
        GC.SuppressFinalize(this);
    }
}
