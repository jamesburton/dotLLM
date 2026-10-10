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
    private readonly Func<VulkanQwen4ExpIndexerState>? _indexerFactory;
    private VulkanNemotronHKvCache? _kv;
    private VulkanQwen4ExpIndexerState? _indexer;

    internal VulkanGdnStateCache Gdn { get; }
    internal Qwen4ExpPleState? Ple { get; }

    /// <inheritdoc/>
    public int NumGdnLayers => Gdn.NumGdnLayers;

    /// <summary>The state's own dense QSA K/V rows, allocated on first use (an engine-supplied KV cache bypasses them entirely).</summary>
    internal VulkanNemotronHKvCache OwnKv => _kv ??= _kvFactory!();

    /// <summary>The state's QSA indexer key cache (raw + pooled keys per QSA layer), allocated on first use (#819).</summary>
    internal VulkanQwen4ExpIndexerState Indexer => _indexer ??= _indexerFactory!();

    /// <summary>Tokens consumed so far (the next position).</summary>
    public int Length { get; internal set; }

    internal VulkanQwen4ExpSequenceState(VulkanGdnStateCache gdn, Func<VulkanNemotronHKvCache> kvFactory, Qwen4ExpPleState? ple,
        Func<VulkanQwen4ExpIndexerState> indexerFactory)
    {
        Gdn = gdn; _kvFactory = kvFactory; Ple = ple; _indexerFactory = indexerFactory;
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
        _indexer?.Dispose();
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
/// (<see cref="CreateSequenceState"/>, threaded through <see cref="ForwardBatch"/>) with the QSA K/V rows in the engine KV cache (#871). (d) MTP (#820): the draft head
/// (<see cref="AttachMtpHead(string)"/>) and per-row recurrent snapshots live on the model-owned state only (see <c>VulkanQwen4ExpTransformerModel.Mtp.cs</c>); the batch scheduler does not speculate.
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
    private readonly VulkanQwen4ExpPleGpu? _pleGpu;
    private readonly int _pleLayer;
    private readonly List<nint> _owned;
    private readonly (nint Ptr, long Bytes)? _hostOnlyTable;
    private readonly Qwen4ExpGatedResidualKernel _gr;
    private readonly GroupRmsNormF32Kernel _groupRms;
    internal GroupRmsNormOopF32Kernel? _groupRmsOop;
    private readonly GdnPostScanGateF32Kernel _sigmoidGate;
    private readonly int _hidden, _streams, _lowRank, _vocab, _kvCapacity;
    private readonly float _eps;
    private readonly VulkanQwen4ExpSequenceState _defaultState;
    private readonly long _weightBytes;
    private readonly VulkanQwen4ExpQsa _qsa;
    private readonly bool _hostEmbedding;
    private readonly nint _embPtr;
    private readonly QuantizationType _embQt;
    private readonly long _embRowBytes;
    private VulkanDevice.Buffer? _embStage;
    private readonly List<VulkanDevice.Buffer> _retiredStages = [];

    /// <summary>
    /// Token embedding for a batch: rows dequantised from the mmap'd table on the host (no device-resident F32 table, #819), written to a host-visible
    /// stage and copied into <c>HiddenState</c> inside the command buffer. One call per submission: the stage is rewritten by the next call.
    /// </summary>
    private void RecordEmbedding(nint cmd, ReadOnlySpan<int> tokenIds)
    {
        if (!_hostEmbedding) { _core.Q4RecordEmbedding(cmd, tokenIds); return; }
        int T = tokenIds.Length, H = _hidden;
        long bytes = (long)T * H * 4;
        if (_embStage is null || _embStage.Size < bytes)
        {
            if (_embStage is not null) _retiredStages.Add(_embStage);   // a recorded-but-unsubmitted copy may still name it
            _embStage = _device.Allocate(Math.Max(bytes, 1L << 20));
        }
        float[] host = System.Buffers.ArrayPool<float>.Shared.Rent(T * H);
        try
        {
            for (int t = 0; t < T; t++)
            {
                int id = tokenIds[t];
                if ((uint)id >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
                Dequantize.ToFloat32(_embPtr + (nint)(id * _embRowBytes), H, _embQt, host.AsSpan(t * H, H));
            }
            _device.Upload(host.AsSpan(0, T * H), _embStage);
        }
        finally { System.Buffers.ArrayPool<float>.Shared.Return(host); }
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _embStage, _core.Q4State.HiddenState, 0, 0, (ulong)bytes);
    }

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
    /// Maximum number of tokens for which QSA attention equals dense attention: <c>indexer.top_k + block - 1</c>. Beyond it the QSA layers
    /// run the indexer (block scoring, exact top-k, gather attention; #819) - the result is still exact QSA, just no longer dense.
    /// </summary>
    public int DenseContextLimit => _qsa.DenseLimit;

    /// <summary>
    /// Positions the K/V and indexer caches are sized for (the longest context this model instance serves). Default
    /// <c>min(context_length, 8192)</c>, further reduced to fit the device; <c>DOTLLM_VK_QWEN4EXP_CONTEXT</c> overrides the default.
    /// </summary>
    public int ContextCapacity => _kvCapacity;

    /// <summary>Test hook: the QSA path (kernels + indexer scratch).</summary>
    internal VulkanQwen4ExpQsa Qsa => _qsa;

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
        GdnPostScanGateF32Kernel sigmoidGate, int kvCapacity, long weightBytes, VulkanQwen4ExpQsa qsa, bool hostEmbedding,
        VulkanQwen4ExpPleGpu? pleGpu = null)
    {
        _pleGpu = pleGpu;
        _qsa = qsa;
        _hostEmbedding = hostEmbedding;
        var ed = gguf.TensorsByName[Qwen4ExpTensors.TokenEmbd];
        _embPtr = gguf.TensorDataPointer(ed);
        _embQt = ed.QuantizationType;
        _embRowBytes = Dequantize.RowByteSize(ed.Shape[0], ed.QuantizationType);
        _device = device; _gguf = gguf; Config = config; _core = core; _q4 = config.Qwen4Exp!;
        _attnGr = attnGr; _ffnGr = ffnGr; _headGr = headGr; _moe = moe; _ple = ple; _pleLayer = pleLayer;
        _owned = owned; _hostOnlyTable = hostOnlyTable; _gr = gr; _groupRms = groupRms; _sigmoidGate = sigmoidGate;
        _hidden = config.HiddenSize; _streams = _q4.HyperConnectionCount; _lowRank = _q4.HyperConnectionLowRank;
        _vocab = config.VocabSize; _eps = config.NormEpsilon; _kvCapacity = kvCapacity; _weightBytes = weightBytes;
        EnsureScratch(1);
        _core.Q4AttentionHook = _qsa;
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>
    /// Counters of the n-gram table prefetch / row-cache service (#822); <c>null</c> when the model has no n-gram branch or the service is off
    /// (<c>DOTLLM_PLE_PREFETCH=off</c> without a cache).
    /// </summary>
    public PleTableStats? PleTableStats => _ple?.Prefetcher is { } pf ? pf.Stats : null;

    /// <summary>Allocates a fresh sequence state (KV capacity <see cref="DenseContextLimit"/>).</summary>
    public VulkanQwen4ExpSequenceState CreateState()
        => new(_core.Q4CreateGdnState(), () => _core.Q4CreateKvCache(_kvCapacity), _ple?.CreateState(),
            () => new VulkanQwen4ExpIndexerState(_device, _qsa.Ordinal.Count(o => o >= 0), _kvCapacity, _q4.IndexerKeyLength, _q4.IndexerBlockSize));

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
        _core.Q4InvalidateCaches();   // freed handles can be recycled into the new buffers: no cached descriptor may name an old one
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

    /// <summary>Diagnostic (#885): with <see cref="StageProfile"/> on, charge stage deltas from GPU timestamps inside the one command buffer instead of submit+wait per stage.</summary>
    public static bool StageTimestamps
    {
        get => VulkanQwen3MoeHybridTransformerModel.StageTimestamps;
        set => VulkanQwen3MoeHybridTransformerModel.StageTimestamps = value;
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

    /// <summary>
    /// Largest number of rows one device forward processes (<c>DOTLLM_VK_PLANNED_ROWS</c>, default 1024): the per-forward scratch is planned and allocated for this
    /// many rows at load. A longer call (a whole perplexity window, an unchunked prompt) is split into chunks of this size, which equals a chunked prefill
    /// (verified bit-for-bit class: relL2 3e-7) instead of growing the scratch past the resident-memory wall after the weights filled it.
    /// </summary>
    public int MaxRowsPerForward => Qwen4ExpResidencyPlan.PlannedRows(_kvCapacity);

    private ITensor ForwardChunked(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, VulkanQwen4ExpSequenceState state,
                                   IKvCache? kvCache, bool lastTokenLogitsOnly, bool allRows, int cap)
    {
        int total = tokenIds.Length;
        bool wantAll = allRows || (!lastTokenLogitsOnly && total <= _allRowLogitsLimit);
        ITensor? result = wantAll ? UnmanagedTensor.Allocate(new TensorShape(total, _vocab), DType.Float32, deviceId: -1) : null;
        ITensor? last = null;
        try
        {
            for (int a = 0; a < total; a += cap)
            {
                int n = Math.Min(cap, total - a);
                var part = ForwardCore(tokenIds.Slice(a, n), positions.Slice(a, n), deviceId, state, kvCache, lastTokenLogitsOnly: !wantAll,
                    mtp: null, snapRows: 0, allRows: wantAll);
                if (wantAll)
                {
                    using (part)
                        new ReadOnlySpan<float>((void*)part.DataPointer, n * _vocab).CopyTo(new Span<float>((void*)(result!.DataPointer + (nint)((long)a * _vocab * 4)), n * _vocab));
                }
                else { last?.Dispose(); last = part; }
            }
        }
        catch { result?.Dispose(); last?.Dispose(); throw; }
        return wantAll ? result! : last!;
    }

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, VulkanQwen4ExpSequenceState state,
                                IKvCache? kvCache, bool lastTokenLogitsOnly, VulkanQwen4ExpMtpState? mtp = null, int snapRows = 0, bool allRows = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (mtp is null && snapRows == 0 && tokenIds.Length > MaxRowsPerForward && tokenIds.Length == positions.Length)
            return ForwardChunked(tokenIds, positions, deviceId, state, kvCache, lastTokenLogitsOnly, allRows, MaxRowsPerForward);
        if (snapRows > 0 && !ReferenceEquals(state, _defaultState))
            throw new InvalidOperationException("Row snapshots are only recorded on the model-owned state.");
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
                $"qwen4exp on Vulkan is sized for {_kvCapacity} tokens (K/V and indexer caches); this call would reach {state.Length + T}. " +
                "Raise the capacity with DOTLLM_VK_QWEN4EXP_CONTEXT (K/V costs 48 KiB per token in F32) if the device has the memory.");
        if (kvCache is null && kv.CurrentLength > state.Length) kv.Rollback(state.Length);   // rows of rejected speculative tokens
        if (kvCache is not null)
        {
            if (kv.CurrentLength < state.Length)
                throw new InvalidOperationException(
                    $"The KV cache holds {kv.CurrentLength} rows but the sequence state is at position {state.Length}: they must advance together.");
            if (kv.CurrentLength > state.Length) kv.Rollback(state.Length);   // stale rows after a rollback are overwritten
            if (kv.MaxLength < state.Length + T)
                throw new NotSupportedException(
                    $"The KV cache holds {kv.MaxLength} positions but this call would reach {state.Length + T} (model capacity {_kvCapacity}).");
        }
        for (int i = 0; i < T; i++)
            if ((uint)tokenIds[i] >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {tokenIds[i]} at position {i} is out of range [0, {_vocab}).");

        bool resized = _core.Q4EnsureCapacity(T);
        EnsureScratch(T);
        if (resized) { _gr.InvalidateDescriptorCache(); _groupRms.InvalidateDescriptorCache(); _groupRmsOop?.InvalidateDescriptorCache(); _sigmoidGate.InvalidateDescriptorCache(); }
        _core.Q4UploadPositions(positions);
        if (!VulkanQwen4ExpQsa.Enabled && state.Length + T > _qsa.DenseLimit)
            throw new NotSupportedException("The QSA indexer is disabled (DOTLLM_VK_QWEN4EXP_QSA=0): attention is dense and only exact up to " +
                                            $"{_qsa.DenseLimit} tokens, this call would reach {state.Length + T}.");
        _qsa.EnsureRows(T);                    // QSA scratch + the sequence's indexer cache (#819); allocated before any recording starts
        _qsa.Begin(state.Indexer);

        // Speculative verify (#820): record the recurrent state after each of the first T-1 rows (GDN scan twin + conv windows + n-gram state).
        _snapValid = false;
        state.Ple?.InvalidateRows();
        int recorded = snapRows > 0 ? Math.Min(snapRows, T - 1) : 0;
        Q4GdnRowSnapshots? snap = null;
        if (recorded > 0)
        {
            snap = EnsureRowSnapshots(recorded);
            snap.Rows = recorded;
            if (state.Ple is { } ps) ps.RecordRowCount = recorded;
        }

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
        // #885: ONE fence-free mid-forward submit a few layers after the host PLE step (or from the start without one): the GPU runs those layers
        // while the host records the remaining ~45 (~1.3 ms of recording that would otherwise be idle GPU time). Same queue order, so bit-identical.
        int splitAt = !splitHalves && DecodeSplitLayers > 0 && Trace is null && (!_core.Q4StageProfiling || StageTimestamps)
            ? (_ple is not null ? _pleLayer : 0) + DecodeSplitLayers : -1;

        void GrRead(VulkanQwen4ExpGrWeights w, VulkanDevice.Buffer src, int tokens, bool inject)
            => RecordGrRead(cmd, w, src, st.NormOutput, tokens, inject);

        _core.Q4DecodeMr = DecodeMoeMr;
        Begin();
        _core.Q4StageBegin();
        if (NgramPrefetch) _ple?.BeginPrefetch(tokenIds, state.Ple!);   // #822: the n-gram rows depend only on token ids - start paging the table in before layer 0
        // #885: the PLE key/value projections need only the token ids: run them on the GPU ahead of layer 0 (short forwards), so the host step
        // between layer 0 and layer 1 is just the residual-dependent remainder (no 131 MB F32 CPU GEMM).
        bool pleOnGpu = _ple is not null && _pleGpu is not null && PleOnGpu && T <= VulkanQwen4ExpPleGpu.MaxRows;
        if (pleOnGpu) { RecordPleProjections(cmd, tokenIds, state.Ple!, T); PleGpuForwards++; }
        RecordEmbedding(cmd, tokenIds);
        Barrier();
        _core.Q4Stage("embed");
        _gr.RecordBroadcast(cmd, _res, st.HiddenState, T, S, H);
        Barrier();

        for (int il = 0; il < Config.NumLayers; il++)
        {
            // ── n-gram branch (host): R += PLE(R, token history) ──
            if (pleOnGpu && il == _pleLayer)
            {
                var pg = _pleGpu!;
                int n = T * row, emb = T * pg.KeyDim, vlen = T * pg.ValueDim;
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, pg.ResidualOut, 0, 0, (ulong)((long)n * 4));
                KernelSupport.TransferToHostBarrier(cmd);
                End();
                float[] hostR = System.Buffers.ArrayPool<float>.Shared.Rent(n);
                float[] hostK = System.Buffers.ArrayPool<float>.Shared.Rent(emb);
                float[] hostV = System.Buffers.ArrayPool<float>.Shared.Rent(vlen);
                try
                {
                    long pt0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _device.Download(pg.ResidualOut, hostR.AsSpan(0, n));
                    _device.Download(pg.KeyOut, hostK.AsSpan(0, emb));
                    _device.Download(pg.ValueOut, hostV.AsSpan(0, vlen));
                    long pt1 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _ple!.ApplyProjected(tokenIds, state.Ple!, hostR.AsSpan(0, n), hostK.AsSpan(0, emb), hostV.AsSpan(0, vlen));
                    long pt2 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _device.Upload(hostR.AsSpan(0, n), pg.ResidualIn);
                    long pt3 = System.Diagnostics.Stopwatch.GetTimestamp();
                    double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                    _core.Q4AddStageMs("host.ple_download", (pt1 - pt0) * f);
                    _core.Q4AddStageMs("host.ple_apply", (pt2 - pt1) * f);
                    _core.Q4AddStageMs("host.ple_upload", (pt3 - pt2) * f);
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(hostR); System.Buffers.ArrayPool<float>.Shared.Return(hostK);
                    System.Buffers.ArrayPool<float>.Shared.Return(hostV);
                }
                Begin();
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, pg.ResidualIn, _res, 0, 0, (ulong)((long)n * 4));
                Barrier();
                _core.Q4Stage("~ple");
            }
            else if (_ple is not null && il == _pleLayer)
            {
                End();
                int n = T * row;
                float[] host = System.Buffers.ArrayPool<float>.Shared.Rent(n);
                try
                {
                    long pt0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _device.Download(_res, host.AsSpan(0, n));
                    long pt1 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _ple.Apply(tokenIds, state.Ple!, host.AsSpan(0, n));
                    long pt2 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _device.Upload(host.AsSpan(0, n), _res);
                    long pt3 = System.Diagnostics.Stopwatch.GetTimestamp();
                    double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                    _core.Q4AddStageMs("host.ple_download", (pt1 - pt0) * f);
                    _core.Q4AddStageMs("host.ple_apply", (pt2 - pt1) * f);
                    _core.Q4AddStageMs("host.ple_upload", (pt3 - pt2) * f);
                }
                finally { System.Buffers.ArrayPool<float>.Shared.Return(host); }
                Begin();
                _core.Q4Stage("~ple");
            }

            // ── token mixer ──
            GrRead(_attnGr[il], _res, T, inject: true);
            if (Config.HybridLayout!.LayerKind[il] == HybridLayerKind.GatedDeltaNet)
                _core.Q4RecordGdn(cmd, il, T, _eps, state.Gdn, _sigmoidGate, snap);
            else
                _core.Q4RecordAttention(cmd, il, T, positions, kv);
            Barrier();
            _gr.RecordWrite(cmd, _res, st.NormOutput, _gains, T, S, H);
            Barrier();
            _core.Q4Stage("gr_write_x");
            if (splitHalves) { End(); Begin(); }

            // ── MoE ──
            GrRead(_ffnGr[il], _res, T, inject: true);
            _core.Q4RecordMoe(cmd, _moe[il], T);
            Barrier();
            _core.Q4Stage("moe_total");
            _gr.RecordWrite(cmd, _res, st.NormOutput, _gains, T, S, H);
            Barrier();
            _core.Q4Stage("gr_write_x");
            if (splitHalves) { End(); Begin(); }
            if (il == splitAt) { submit.SplitSubmit(); cmd = submit.CommandBuffer; DecodeSplits++; }

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
        int rows = allRows || (!lastTokenLogitsOnly && T <= _allRowLogitsLimit) ? T : 1;
        var w0 = _core.Q4Weights;
        if (rows == 1)
        {
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, _headRes, (ulong)((T - 1) * rowBytes), 0, (ulong)rowBytes);
            Barrier();
            GrRead(_headGr, _headRes, 1, inject: false);
            _core.Q4Stage("head_gr");
            _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, st.NormOutput, st.Logits,
                outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: 1);
            _core.Q4Stage("lm_head");
            End();
            state.Length += T;
            var one = UnmanagedTensor.Allocate(new TensorShape(1, _vocab), DType.Float32, deviceId: -1);
            _device.Download(st.Logits, new Span<float>((void*)one.DataPointer, _vocab));
            return FinishForward(one, tokenIds, positions, state, mtp, recorded);
        }

        // All rows: head mixer over all T rows, then the LM head in chunks so the device logits scratch stays small
        // (a 2K window x 248K vocab is 2 GB; the host result tensor is the only full-size allocation).
        GrRead(_headGr, _res, T, inject: false);
        _core.Q4Stage("head_gr");
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
                _core.Q4Stage("lm_head");
                End();
                _device.Download(_headLogits!, new Span<float>((void*)(result.DataPointer + (nint)((long)c0 * _vocab * 4)), n * _vocab));
            }
        }
        catch { result.Dispose(); throw; }
        state.Length += T;
        return FinishForward(result, tokenIds, positions, state, mtp, recorded);
    }

    /// <summary>Common tail of a forward: closes the row recording and, when an MTP state rides along, feeds the batch to the head.</summary>
    private ITensor FinishForward(ITensor logits, ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, VulkanQwen4ExpSequenceState state,
                                  VulkanQwen4ExpMtpState? mtp, int recorded)
    {
        _qsa.Begin(null);
        if (state.Ple is { } ps) ps.RecordRowCount = 0;
        if (recorded > 0) { _snapBase = positions[0]; _snapRowsRecorded = recorded; _snapValid = true; }
        if (mtp is not null)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            try { AbsorbBatch(mtp, tokenIds, positions[0]); }
            catch { logits.Dispose(); throw; }
            _absorbTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        }
        return logits;
    }

    /// <summary>
    /// Short forwards run the n-gram key/value projections on the GPU (#885, ~4 ms of a ~57 ms decode step). On by default;
    /// <c>DOTLLM_VK_Q4E_PLE_GPU=0</c> at startup (or setting this) restores the all-host branch. Read at the top of every forward.
    /// </summary>
    public static bool PleOnGpu { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_PLE_GPU") != "0";

    /// <summary>Test hook (#885): Q8_0 GEMVs recorded through the wide MMVQ twins.</summary>
    internal long WideQ8Dispatches => _core.Q4WideDispatches;

    /// <summary>Diagnostic (#885): wide Q8_0 MMVQ twins for the gated-residual projections.</summary>
    public static bool Q8Wide
    {
        get => VulkanQwen3MoeHybridTransformerModel.Q4WideEnabled;
        set => VulkanQwen3MoeHybridTransformerModel.Q4WideEnabled = value;
    }

    /// <summary>Test hook (#885): gated-residual reads recorded on the fused path.</summary>
    internal long GrFusedReads { get; private set; }

    /// <summary>Test hook (#885): decode MoE layers that used the fused scatter + shared-gate add kernel.</summary>
    internal long CombineFusedLayers => _core.Q4CombineFusedLayers;

    /// <summary>Diagnostic (#885): fused weighted scatter + shared-expert gated add on decode.</summary>
    public static bool CombineFused
    {
        get => VulkanQwen3MoeHybridTransformerModel.Q4CombineFused;
        set => VulkanQwen3MoeHybridTransformerModel.Q4CombineFused = value;
    }

    /// <summary>Test hook (#885): MoE layers that ran on the fused single-token decode chain.</summary>
    internal long MoeFusedLayers => _core.Q4MoeFusedLayers;

    /// <summary>Diagnostic (#885): fused single-token MoE decode chain (see <see cref="VulkanQwen3MoeHybridTransformerModel.Q4MoeDecodeFused"/>).</summary>
    public static bool MoeDecodeFused
    {
        get => VulkanQwen3MoeHybridTransformerModel.Q4MoeDecodeFused;
        set => VulkanQwen3MoeHybridTransformerModel.Q4MoeDecodeFused = value;
    }

    /// <summary>
    /// Layers of work submitted ahead (fence-free) after the PLE step so the GPU runs while the host records the rest of a short forward (#885); 0 disables.
    /// <c>DOTLLM_VK_Q4E_SPLIT_LAYERS</c> at startup, default 3. Read at the top of every forward.
    /// </summary>
    public static int DecodeSplitLayers { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_SPLIT_LAYERS"), out int dsl) && dsl >= 0 ? dsl : 3;

    /// <summary>Test hook (#885): forwards that took the mid-forward submit.</summary>
    internal long DecodeSplits { get; private set; }

    /// <summary>Test hook (#885): forwards whose PLE key/value projections ran on the GPU.</summary>
    internal long PleGpuForwards { get; private set; }

    /// <summary>Gathers the table rows on the host, writes them to the device scratch and records both projections (key, value).</summary>
    private void RecordPleProjections(nint cmd, ReadOnlySpan<int> tokenIds, Qwen4ExpPleState pleState, int T)
    {
        var pg = _pleGpu!;
        int embLen = _ple!.EmbeddingLength(T);
        float[] emb = System.Buffers.ArrayPool<float>.Shared.Rent(embLen);
        try
        {
            _ple.GatherEmbeddings(tokenIds, pleState, emb.AsSpan(0, embLen));
            _device.Upload(emb.AsSpan(0, embLen), pg.Emb);
        }
        finally { System.Buffers.ArrayPool<float>.Shared.Return(emb); }
        pg.Record(_core, cmd, T);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        _core.Q4Stage("ple_proj");
    }

    /// <summary>GR read: group-RMS(src) -> low-rank mix -> block input written to <paramref name="dst"/>; inject gains when requested.</summary>
    private void RecordGrRead(nint cmd, VulkanQwen4ExpGrWeights w, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int tokens, bool inject)
    {
        int S = _streams, H = _hidden, row = S * H;
        long rowBytes = (long)row * 4;
        // #885: the residual -> scratch copy is folded into an out-of-place norm, and the inject projection (independent of the
        // low-rank chain, M = 4 rows so latency-bound) runs concurrently with the down projection instead of after the mix-mean.
        bool fast = _groupRmsOop is not null && GrFused;
        if (fast) GrFusedReads++;
        if (fast)
        {
            _groupRmsOop!.Record(cmd, src, w.Norm, _xn, tokens, S, H, _eps);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            _core.Q4Stage("gr.grouprms");
        }
        else
        {
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, src, _xn, 0, 0, (ulong)(tokens * rowBytes));
            KernelSupport.ComputeTransferFullBarrier(cmd);
            _core.Q4Stage("gr.copy");
            _groupRms.Record(cmd, _xn, w.Norm, tokens, S, H, _eps);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            _core.Q4Stage("gr.grouprms");
        }
        // fast path + inject: quantize xn once, then record the down GEMV and the (latency-bound, 4-row) inject GEMV with no barrier between them
        bool q8 = fast && tokens == 1 && w.DownQt == QuantizationType.Q8_0 && w.UpQt == QuantizationType.Q8_0;
        bool xqReady = q8 && _core.Q4PrepareQ8Row(cmd, _xn, row);
        if (!(xqReady && _core.Q4RecordQ8Wide(cmd, kSplit: true, w.Down, _low, _lowRank, row)))
            _core.Q4RecordMatmul(cmd, w.Down, w.DownQt, _xn, _low, outputDim: _lowRank, inputDim: row, seqLen: tokens, xqReady: xqReady);
        if (fast && inject) _core.Q4RecordMatmul(cmd, w.Inject!, w.InjectQt, _xn, _gains, outputDim: S, inputDim: row, seqLen: tokens);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        _core.Q4Stage("gr.down");
        _gr.RecordActivateLowRank(cmd, _low, tokens * _lowRank, S);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        _core.Q4Stage("gr.act");
        if (!(q8 && _lowRank % 32 == 0 && _core.Q4PrepareQ8Row(cmd, _low, _lowRank) && _core.Q4RecordQ8Wide(cmd, kSplit: false, w.Up, _mix, row, _lowRank)))
            _core.Q4RecordMatmul(cmd, w.Up, w.UpQt, _low, _mix, outputDim: row, inputDim: _lowRank, seqLen: tokens);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        _core.Q4Stage("gr.up");
        _gr.RecordMixMean(cmd, dst, _mix, _xn, tokens, S, H);
        if (inject && fast) _gr.RecordInjectGains(cmd, _gains, tokens, S);   // independent of the mix-mean: same barrier slot
        KernelSupport.ComputeTransferFullBarrier(cmd);
        _core.Q4Stage("gr.mixmean");
        if (inject && !fast)
        {
            _core.Q4RecordMatmul(cmd, w.Inject!, w.InjectQt, _xn, _gains, outputDim: S, inputDim: row, seqLen: tokens);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            _core.Q4Stage("gr.inject_mm");
            _gr.RecordInjectGains(cmd, _gains, tokens, S);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            _core.Q4Stage("gr.inject_gains");
        }
    }

    /// <summary>
    /// Gated-residual reads fold the residual copy into an out-of-place group norm and overlap the inject projection with the low-rank
    /// chain (#885). Bit-identical to the unfused sequence. On by default; <c>DOTLLM_VK_Q4E_GR_FUSED=0</c> (or setting this) restores it.
    /// </summary>
    public static bool GrFused { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_GR_FUSED") != "0";

    /// <summary>
    /// Start paging the n-gram table rows of a chunk in at the top of the forward (#822) and, for MTP, while the draft steps run. On by default;
    /// <c>DOTLLM_VK_Q4E_PREFETCH=0</c> turns it off at startup (A/B diagnostic - it never changes a result, only when the table pages become resident).
    /// </summary>
    public static bool NgramPrefetch { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_PREFETCH") != "0";

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

    /// <summary>
    /// Single-token decode uses the multi-row routed-MoE MMVQs (#885; 56.7 -> 52.5 ms per 1-row forward on UD-Q4_K_XL). On by default;
    /// <c>DOTLLM_VK_Q4E_DECODE_MR=0</c> at startup (or setting this) restores the one-row kernels. Read at the top of every forward.
    /// </summary>
    public static bool DecodeMoeMr { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_DECODE_MR") != "0";

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
        // Host-readback on purpose: a device-local-only buffer is read back through a freshly allocated staging buffer on every call, which cost
        // ~25 ms per 5-row verify (measured, #820); a host-visible one is mapped directly, like the core's 1-row logits buffer.
        _headLogits = _device.AllocateHostReadback((long)rows * _vocab * 4);
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _mtp?.Dispose();
        _ple?.Prefetcher?.Dispose();
        _pleGpu?.Dispose();
        _defaultState.Dispose();
        foreach (var m in _moe) m.Dispose();
        foreach (var g in _attnGr) g.Dispose();
        foreach (var g in _ffnGr) g.Dispose();
        _headGr.Dispose();
        _res?.Dispose(); _xn?.Dispose(); _low?.Dispose(); _mix?.Dispose(); _gains?.Dispose(); _headRes?.Dispose();
        _headIn?.Dispose(); _headLogits?.Dispose();
        _snapGdn?.Dispose(); _snapKernel?.Dispose();
        _gr.Dispose(); _groupRms.Dispose(); _groupRmsOop?.Dispose(); _sigmoidGate.Dispose();
        _core.Q4AttentionHook = null;
        _qsa.Dispose();
        _embStage?.Dispose();
        foreach (var b in _retiredStages) b.Dispose();
        _core.Dispose();
        foreach (nint p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        if (_hostOnlyTable is { } h) VulkanWeightImportPolicy.UnregisterHostOnly(h.Ptr);
        GC.SuppressFinalize(this);
    }
}
