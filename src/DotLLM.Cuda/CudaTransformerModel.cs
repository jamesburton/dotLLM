using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-accelerated transformer forward pass using CUDA. All operations execute on a single
/// CUDA stream with no host synchronization until the final logits D2H transfer.
/// Mirrors <see cref="TransformerModel"/> structure but uses cuBLAS GEMM/GEMV and custom PTX kernels.
/// </summary>
public sealed unsafe class CudaTransformerModel : IModel
{
    private readonly CudaWeights _weights;
    private readonly CudaForwardState _state;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly GgufFile? _gguf;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <summary>Non-null when model weights exceed available VRAM. Caller should display after loading.</summary>
    public string? VramWarning { get; }

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers (embedding + LM head only).</summary>
    internal int DebugMaxLayers { get; set; }

    /// <summary>Debug: override RoPE type. -1 = use model's type (default).</summary>
    internal int DebugRopeTypeOverride { get; set; } = -1;

    /// <summary>Debug: skip bias add operations.</summary>
    internal bool DebugSkipBias { get; set; }

    /// <summary>
    /// When true, <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// brackets the kernel-launch sequence with CUDA events so <see cref="LastGpuLaunchMs"/>
    /// reports the GPU-side wallclock between the first and last kernel of the forward pass.
    /// Wall time is measured by the caller; <c>wall - LastGpuLaunchMs</c> bounds the
    /// host-dispatch + sync + D2H overhead. Off by default — events themselves are cheap
    /// but the read after sync adds a little host-side serialisation.
    /// </summary>
    internal bool ProfilingEnabled { get; set; }

    /// <summary>GPU wallclock (ms) of the most recent <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/> when <see cref="ProfilingEnabled"/> is on. NaN otherwise.</summary>
    internal float LastGpuLaunchMs { get; private set; } = float.NaN;

    /// <summary>Per-category GPU time (ms) for the most recent profiled forward, indexed by <see cref="ProfileCategory"/>.</summary>
    internal float[] LastCategoryMs => _categoryMsLast;

    /// <summary>
    /// When set, single-token decode steps are captured into a CUDA Graph the first time
    /// they are seen, then replayed via <c>cuGraphLaunch</c> on each subsequent step.
    /// Collapses ~400 per-step kernel submissions into one stream packet — typically
    /// 2-3× decode speedup on Windows / WDDM where each launch costs ~22 µs.
    /// <para>
    /// Both the standard <see cref="CudaKvCache"/> and the mixed-precision
    /// <see cref="CudaQuantizedKvCache"/> (with <c>WindowCapacity &gt; 0</c>) decode
    /// paths are graph-capable. Pure-quantized configs (<c>WindowCapacity == 0</c>)
    /// stay on the eager path. Prefill (seqLen &gt; 1) always stays eager. The graph is
    /// invalidated when the kvCache identity changes or when <see cref="DebugMaxLayers"/>
    /// flips between calls.
    /// </para>
    /// <para>
    /// <b>Default ON</b> (post re-bench on RTX 3060: graph is never slower across
    /// SmolLM-135M / Qwen3-4B / Qwen3-8B). Override with the
    /// <c>DOTLLM_DISABLE_GRAPH_CAPTURE=1</c> env var (matches the
    /// <c>DOTLLM_DISABLE_*</c> kernel-feature convention) or by explicitly
    /// assigning <c>false</c>. When the model is constructed without the
    /// kv-write fusion kernel (<see cref="CudaKernels.HasKvWriteKernel"/>),
    /// the default silently falls back to eager and a one-line warning is
    /// emitted — speculative decoding (multi-token decode) and prefill
    /// (multi-token forward) always fall through to the eager path at
    /// runtime regardless of this flag.
    /// </para>
    /// </summary>
    public bool UseGraphCapture { get; set; }

    /// <summary>Env-var override for the default-on graph-capture path. Set
    /// <c>DOTLLM_DISABLE_GRAPH_CAPTURE=1</c> to force eager decode regardless
    /// of capability. Test/benchmark hook follows the same convention as
    /// <see cref="CudaKernels.DisablePreQ8_1"/>, etc.</summary>
    public static bool DisableGraphCapture { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_GRAPH_CAPTURE") == "1";

    private nint _evtStart;
    private nint _evtEnd;

    // ── CUDA Graphs decode-capture state ──
    // Two device-resident ints feed graph-baked kernel pointers so seq_kv and the
    // RoPE/attention position offset can grow per replay without re-instantiation:
    //   _decodePosDevice   — int  (= absolute decode position; seq_kv = pos + 1)
    //   _decodeSeqKvDevice — int  (= seq_kv = pos + 1)
    // Host bumps both via cuMemcpyHtoD_v2 (~1 µs each) before each cuGraphLaunch.
    private nint _decodePosDevice;
    private nint _decodeSeqKvDevice;
    private nint _decodeGraph;       // cuGraph handle (intermediate, freed after instantiate)
    private nint _decodeGraphExec;   // cuGraphExec handle (the launchable instance)
    // KvCache the graph was captured against (may be CudaKvCache or CudaQuantizedKvCache);
    // invalidate if it changes. Stored as object since both implementations are graph-capable
    // but go through different launch sequences.
    private object? _decodeGraphKvCache;
    private int _decodeGraphLayerCount;       // DebugMaxLayers snapshot at capture time

    // ── MLA / MoE per-model state (lazy-allocated, populated only when the
    //    model declares the matching config) ──
    // RoPE cos/sin tables for MLA's decoupled rope sub-dimension. Shape:
    // [maxSeqLen, qkRopeHeadDim/2] each, F32. Allocated on first MLA forward.
    private nint _mlaRopeCosF32;
    private nint _mlaRopeSinF32;
    private int _mlaRopeMaxSeqLen;
    // FP16 MLA scratch + KV cache (Phase A expanded layout). The KV cache is
    // owned by the model since the standard kvCache parameter shape doesn't
    // match MLA's decoupled K_nope / V / K_pe storage. Reset when positions[0]==0.
    private CudaMlaScratchF16? _mlaScratchF16;
    private CudaMlaKvCache? _mlaKvCache;
    // F32 staging buffers for the F16↔F32 conversion at the MoE FFN boundary
    // (the routed-MoE kernel takes F32 in/out; the rest of the model is F16).
    private nint _moeStagingInF32;
    private nint _moeStagingOutF32;
    private int _moeStagingCapacityElems;
    private CudaMoeScratch? _moeScratch;

    // ── per-category profiling state (only allocated when ProfilingEnabled is set) ──
    internal const int ProfileCategoryCount = 12;
    private nint[]? _profEvents;        // event pool, allocated lazily
    private byte[]? _profEventCategory; // category id of the interval ENDING at event[i] (i>0)
    private int _profEventCursor;
    private readonly float[] _categoryMsLast = new float[ProfileCategoryCount];

    /// <summary>Buckets used by per-category profiling. Order matches <see cref="LastCategoryMs"/> indices.</summary>
    internal enum ProfileCategory : byte
    {
        Embed = 0,
        QkvProj = 1,
        RopeAndExtras = 2,   // bias adds + QK norms + RoPE
        KvUpdate = 3,
        Attention = 4,
        OProj = 5,
        Norm = 6,            // initial rmsnorm + every fused-add-rmsnorm + final rmsnorm
        MlpUp = 7,           // gate + up projections
        Swiglu = 8,
        MlpDown = 9,
        LmHead = 10,
        Convert = 11,        // FP16 logits → FP32 + final residual add for the last layer
    }

    private CudaTransformerModel(
        ModelConfig config, CudaWeights weights, CudaForwardState state,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context,
        CudaKernels kernels, GgufFile? gguf, int deviceId,
        float ropeTheta, int ropeDim, int ropeType, string? vramWarning)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _gguf = gguf;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        VramWarning = vramWarning;
        _ropeType = ropeType;

        // Default-on graph capture when capable. Re-bench on RTX 3060
        // (post pre-Q8_1 + MMVQ-large default-ON) shows graph never regresses
        // across SmolLM-135M / Qwen3-4B / Qwen3-8B. Env-var override:
        // DOTLLM_DISABLE_GRAPH_CAPTURE=1.
        // The runtime path-selection in Forward() additionally gates on
        // single-token decode, no profiling, and a graph-capable kvCache,
        // so prefill / speculative-verify / pure-quant configs naturally
        // fall through to eager. Here we only suppress the default-on
        // when the underlying kv-write fusion kernel isn't loaded.
        if (DisableGraphCapture)
        {
            UseGraphCapture = false;
        }
        else if (!kernels.HasKvWriteKernel)
        {
            UseGraphCapture = false;
            Console.Error.WriteLine(
                "[dotLLM.Cuda] Graph-capture default disabled: kv-write fusion kernel " +
                "not available — falling back to eager decode.");
        }
        else
        {
            UseGraphCapture = true;
        }
    }

    /// <summary>
    /// Loads a transformer model onto the GPU from an opened GGUF file.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. If null, auto-detects from assembly location.</param>
    public static CudaTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config,
                                                       int deviceId = 0, string? ptxDir = null)
    {
        // Load CPU weights (mmap references only, no heavy allocation)
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        // VRAM estimate uses GGUF quant info — cheaper than walking cpuWeights.
        long estimatedWeightBytes = 0;
        foreach (var t in gguf.TensorsByName.Values)
        {
            int innerDim = t.Shape[0];
            long outerDim = (long)t.Shape.ElementCount / innerDim;
            estimatedWeightBytes += Cpu.Kernels.Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
        }

        return LoadFromCpuWeights(cpuWeights, config, gguf, deviceId, ptxDir, estimatedWeightBytes);
    }

    /// <summary>
    /// Loads a transformer model onto the GPU from an opened HuggingFace-convention
    /// safetensors source (single-file or multi-shard). Same arch coverage as
    /// <see cref="LoadFromGguf"/> for the Transformer family; MLA/Mamba3 not yet
    /// ported to CUDA and will throw at forward time if attempted.
    /// </summary>
    /// <param name="file">Opened safetensors source; caller retains ownership.</param>
    /// <param name="config">Model configuration parsed from <c>config.json</c>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null auto-detects.</param>
    public static CudaTransformerModel LoadFromSafetensors(ISafetensorsTensorSource file,
                                                              ModelConfig config,
                                                              int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        // Safetensors path produces a TransformerWeights that owns bf16→F32 upcast
        // allocations; do NOT call RepackWeights (R4 interleaving is a CPU-SIMD
        // concern, not a GPU one) — CudaWeights.LoadFromGguf reads the raw tensor
        // pointers and uploads them. The misleading method name stays for now; the
        // underlying flow is source-agnostic.
        var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);

        // VRAM estimate: skip for the safetensors path for now — TransformerWeights
        // doesn't expose per-tensor byte sizes cheaply, and the CPU pre-load above
        // already either succeeded in RAM or failed with an explicit error. Follow-up:
        // add a TransformerWeights.EstimatedDeviceBytes helper.
        return LoadFromCpuWeights(cpuWeights, config, gguf: null, deviceId, ptxDir,
                                  estimatedWeightBytes: 0);
    }

    /// <summary>
    /// Shared CUDA init + upload used by both the GGUF and safetensors entrypoints.
    /// Creates the CUDA context/stream/cuBLAS, loads the PTX module, emits the VRAM
    /// warning if the estimate exceeds free device memory, and uploads the already-
    /// loaded <see cref="TransformerWeights"/> to the GPU.
    /// </summary>
    private static CudaTransformerModel LoadFromCpuWeights(
        TransformerWeights cpuWeights, ModelConfig config, GgufFile? gguf,
        int deviceId, string? ptxDir, long estimatedWeightBytes)
    {
        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        string? vramWarning = null;
        if (estimatedWeightBytes > 0
            && CudaDriverApi.cuMemGetInfo_v2(out nuint freeBefore, out nuint totalVram) == 0
            && totalVram > 0 && estimatedWeightBytes > (long)freeBefore)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long freeMb = (long)freeBefore / (1024 * 1024);
            long totalMb = (long)totalVram / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed available VRAM ({freeMb}/{totalMb} MB free). " +
                          $"Performance will be degraded due to PCIe memory paging. " +
                          $"Consider a smaller model or quantization format.";
        }

        var weights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle);

        var state = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new CudaTransformerModel(config, weights, state, stream, cublas, context,
            kernels, gguf, deviceId, ropeTheta, ropeDim, ropeType, vramWarning);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// Records an event on the stream tagged with the given category. The interval
    /// (previous event → this event) is attributed to <paramref name="cat"/> when
    /// per-category timings are aggregated after stream sync. No-op when profiling
    /// is disabled — kept tight enough for the JIT to drop it from the hot path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void MarkProfile(ProfileCategory cat)
    {
        if (!ProfilingEnabled) return;
        EnsureProfileCapacity(_profEventCursor + 1);
        CudaDriverApi.cuEventRecord(_profEvents![_profEventCursor], _stream.Handle).ThrowOnError();
        _profEventCategory![_profEventCursor] = (byte)cat;
        _profEventCursor++;
    }

    private void EnsureProfileCapacity(int needed)
    {
        if (_profEvents != null && _profEvents.Length >= needed) return;

        int newCap = Math.Max(needed, _profEvents?.Length * 2 ?? 512);
        var newEvents = new nint[newCap];
        var newCats = new byte[newCap];
        int oldLen = _profEvents?.Length ?? 0;
        if (_profEvents != null) Array.Copy(_profEvents, newEvents, oldLen);
        if (_profEventCategory != null) Array.Copy(_profEventCategory, newCats, oldLen);
        for (int i = oldLen; i < newCap; i++)
            CudaDriverApi.cuEventCreate(out newEvents[i], CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
        _profEvents = newEvents;
        _profEventCategory = newCats;
    }

    /// <summary>
    /// Lazy-allocate the per-model MLA state (RoPE cos/sin tables sized for the
    /// configured max sequence length, FP16 scratch, F16 KV cache). Reset the KV
    /// cache when <paramref name="positions"/>[0] == 0 (fresh sequence).
    /// </summary>
    private void EnsureMlaState(ReadOnlySpan<int> positions)
    {
        var mla = Config.MlaConfig
            ?? throw new InvalidOperationException("EnsureMlaState called without MlaConfig.");

        int maxSeq = Config.MaxSequenceLength > 0 ? Config.MaxSequenceLength : 4096;

        // RoPE tables: F32 cos/sin, [maxSeq, qkRope/2] each.
        if (_mlaRopeCosF32 == 0 || _mlaRopeMaxSeqLen < maxSeq)
        {
            if (_mlaRopeCosF32 != 0) CudaDriverApi.cuMemFree_v2(_mlaRopeCosF32);
            if (_mlaRopeSinF32 != 0) CudaDriverApi.cuMemFree_v2(_mlaRopeSinF32);

            int half = mla.QkRopeHeadDim / 2;
            long elems = (long)maxSeq * half;
            long bytes = elems * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out _mlaRopeCosF32, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _mlaRopeSinF32, (nuint)bytes).ThrowOnError();

            var cosArr = new float[elems];
            var sinArr = new float[elems];
            DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(maxSeq, mla.QkRopeHeadDim, mla.RopeTheta, cosArr, sinArr);
            unsafe
            {
                fixed (float* cp = cosArr) CudaDriverApi.cuMemcpyHtoD_v2(_mlaRopeCosF32, (nint)cp, (nuint)bytes).ThrowOnError();
                fixed (float* sp = sinArr) CudaDriverApi.cuMemcpyHtoD_v2(_mlaRopeSinF32, (nint)sp, (nuint)bytes).ThrowOnError();
            }
            _mlaRopeMaxSeqLen = maxSeq;
        }

        _mlaScratchF16 ??= new CudaMlaScratchF16();

        if (_mlaKvCache is null)
        {
            _mlaKvCache = new CudaMlaKvCache(
                numLayers: Config.NumLayers,
                maxSeqLen: maxSeq,
                numHeads: Config.NumAttentionHeads,
                qkNopeHeadDim: mla.QkNopeHeadDim,
                vHeadDim: mla.VHeadDim,
                qkRopeHeadDim: mla.QkRopeHeadDim,
                precision: MlaPrecision.F16);
        }

        // Fresh sequence — clear cached lengths so layer N's cache write goes to row 0.
        if (positions.Length > 0 && positions[0] == 0)
            _mlaKvCache.Reset();
    }

    /// <summary>
    /// Lazy-allocate F32 staging buffers for the F16↔F32 conversion at the
    /// MoE FFN boundary. Two buffers of [seqLen × hiddenSize] floats each;
    /// reused across layers and forward calls.
    /// </summary>
    private void EnsureMoeStaging(int seqLen, int hiddenSize)
    {
        int needed = seqLen * hiddenSize;
        if (_moeStagingCapacityElems >= needed) return;

        if (_moeStagingInF32 != 0) CudaDriverApi.cuMemFree_v2(_moeStagingInF32);
        if (_moeStagingOutF32 != 0) CudaDriverApi.cuMemFree_v2(_moeStagingOutF32);
        long bytes = (long)needed * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _moeStagingInF32, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _moeStagingOutF32, (nuint)bytes).ThrowOnError();
        _moeStagingCapacityElems = needed;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        bool isMla = _weights.MlaLayers is not null;
        bool isMoe = _weights.MoeLayers is not null;

        // CUDA Graphs decode fast-path: single-token decode for both the standard
        // FP16 KV-cache and the quantized cache (when a mixed-precision FP16 window
        // is configured). Captures on first invocation, replays via cuGraphLaunch
        // thereafter. Falls through to eager for prefill, multi-token decode
        // (speculative verify), and pure-quantized configs (windowCapacity == 0).
        // MLA / MoE are not graph-capable today (MoE has host-side bucketing; MLA's
        // absorbed kernel uses dynamic shmem) — they fall through to eager.
        if (UseGraphCapture
            && tokenIds.Length == 1
            && _kernels.HasKvWriteKernel
            && !ProfilingEnabled            // event injection between launches breaks capture
            && !isMla && !isMoe)
        {
            if (kvCache is CudaKvCache stdKv)
                return ForwardDecodeGraph(tokenIds, positions, deviceId, stdKv);
            if (kvCache is CudaQuantizedKvCache qKv && qKv.WindowCapacity > 0)
                return ForwardDecodeGraphQuantized(tokenIds, positions, deviceId, qKv);
        }

        _context.MakeCurrent();
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); // FP16 element size

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        _state.EnsureCapacity(seqLen);

        if (ProfilingEnabled)
        {
            if (_evtStart == 0)
            {
                CudaDriverApi.cuEventCreate(out _evtStart, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _evtEnd, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
            }
            CudaDriverApi.cuEventRecord(_evtStart, s).ThrowOnError();
        }

        // 1. Upload tokenIds + positions to device
        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // 2. Embedding lookup → FP16 HiddenState
        _kernels.LaunchEmbeddingLookup(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState,
            seqLen, hiddenSize, s);
        MarkProfile(ProfileCategory.Embed);

        // 3. Layer 0 setup: copy hidden→residual; on the GQA path also
        //    pre-RmsNorm into NormOutput. The MLA path skips the pre-norm —
        //    CudaMlaAttention.ForwardF16 applies its own input RMSNorm internally,
        //    and consumes the raw hidden state from Residual.
        long hiddenBytes = (long)seqLen * hiddenSize * h;
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
        if (!isMla)
        {
            _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);
        }
        MarkProfile(ProfileCategory.Norm);

        // Lazy-allocate MLA / MoE per-model state on first invocation that needs it.
        if (isMla)
            EnsureMlaState(positions);
        if (isMoe)
            EnsureMoeStaging(seqLen, hiddenSize);

        // 4. Transformer layers — FP16 activations, cuBLAS GEMM for prefill, quantized GEMV for decode,
        //    FusedAddRmsNorm at residual junctions to avoid FP16 truncation.
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,   // skip all layers (embedding + LM head only)
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        // When skipping all layers, treat embedding output as final hidden state
        if (numLayers == 0)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // ── ATTENTION BLOCK ──
            // MLA path: CudaMlaAttention.ForwardF16 absorbs the entire QKV/RoPE/
            // KV-update/Attention/OProj sequence into one helper. Reads raw hidden
            // from Residual (does its own input RMSNorm internally), writes
            // post-O_proj F16 output into NormOutput (no residual add — that
            // happens in the shared FusedAddRmsNorm step below).
            if (isMla)
            {
                ref readonly var mlaLayer = ref _weights.MlaLayers![layer];
                float scale = 1.0f / MathF.Sqrt(mlaLayer.QkNopeHeadDim + mlaLayer.QkRopeHeadDim);
                scale *= Config.MlaConfig!.ComputeYarnSoftmaxScaleMultiplier();
                _mlaScratchF16!.EnsureCapacity(seqLen, in mlaLayer);
                CudaMlaAttention.ForwardF16(
                    hiddenF16: _state.Residual,
                    outputF16: _state.NormOutput,
                    seqLen: seqLen,
                    positionOffset: positions[0],
                    layer: in mlaLayer,
                    kvCache: _mlaKvCache!,
                    layerIndex: layer,
                    ropeCosF32: _mlaRopeCosF32,
                    ropeSinF32: _mlaRopeSinF32,
                    rmsNormEps: eps,
                    softmaxScale: scale,
                    scratch: _mlaScratchF16,
                    cublasHandle: cublasH,
                    kernels: _kernels,
                    stream: s);
                _mlaKvCache!.Advance(layer, seqLen);
                MarkProfile(ProfileCategory.QkvProj);
                MarkProfile(ProfileCategory.RopeAndExtras);
                MarkProfile(ProfileCategory.KvUpdate);
                MarkProfile(ProfileCategory.Attention);
                MarkProfile(ProfileCategory.OProj);

                // Residual + FfnNorm. The MLA loader uploads its FfnNormWeight as F32
                // (the FP16 RMSNorm helper inside CudaMlaAttention.ForwardF16 takes an
                // F32 weight). LaunchFusedAddRmsNorm expects F16 — use the F16 sibling
                // already uploaded into _weights.Layers[layer].FfnNormWeight, which
                // shares the same source data via UploadNormWeight's F32→F16 cast.
                _kernels.LaunchFusedAddRmsNorm(
                    _state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
                MarkProfile(ProfileCategory.Norm);

                goto FfnBlock;
            }

            // Q/K/V projections. For decode (seqLen=1) with a pre-packed quantized
            // QKV weight, one fused GEMV produces [n_q|n_kv|n_kv] contiguously in
            // _state.QkvPacked; downstream kernels take pointers so they can read
            // slices directly. For prefill or mixed-quant fallback, keep the 3-call
            // path. Aliases qPtr/kPtr/vPtr hide the path choice from the rest of
            // the layer body.
            nint qPtr, kPtr, vPtr;
            bool fusedQkv = seqLen == 1 && lw.QkvPacked != 0;
            if (fusedQkv)
            {
                if (_kernels.HasMmq(lw.QkvPackedQuantType))
                {
                    nint scratch = MaybePreQuantize(_state.NormOutput, lw.QInputDim, s);
                    _kernels.LaunchQuantizedGemvMmq(lw.QkvPacked, lw.QkvPackedQuantType,
                        _state.NormOutput, _state.QkvPacked,
                        lw.QkvPackedOutputDim, lw.QInputDim, scratch, s);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(lw.QkvPacked, lw.QkvPackedQuantType,
                        _state.NormOutput, _state.QkvPacked,
                        lw.QkvPackedOutputDim, lw.QInputDim, s);
                }
                qPtr = _state.QkvPacked;
                kPtr = _state.QkvPacked + (nint)((long)lw.QOutputDim * h);
                vPtr = _state.QkvPacked + (nint)((long)(lw.QOutputDim + lw.KOutputDim) * h);
            }
            else
            {
                Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
                Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);
                qPtr = _state.Q;
                kPtr = _state.K;
                vPtr = _state.V;
            }
            MarkProfile(ProfileCategory.QkvProj);

            // Optional biases (FP16)
            if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(qPtr, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(kPtr, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(vPtr, lw.VBias, lw.VOutputDim, seqLen, s);

            // Optional QK-norms (FP16)
            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(qPtr, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(kPtr, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            // RoPE + KV-cache write. For decode (seqLen=1) against a standard
            // CudaKvCache with the fused kernel available, fold both into a single
            // launch — saves 2 launches/layer (rope + 2× cuMemcpyDtoDAsync) → 1.
            int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
            bool useFusedRopeKv = seqLen == 1
                && kvCache is CudaKvCache
                && _kernels.HasFusedRopeKvWriteKernel;

            if (useFusedRopeKv)
            {
                var cudaKvCache = (CudaKvCache)kvCache!;
                cudaKvCache.FusedRopeAndUpdateDevice(
                    qPtr, kPtr, vPtr,
                    _state.PositionsDevice, positions[0],
                    layer,
                    numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType,
                    s, _kernels);
                MarkProfile(ProfileCategory.RopeAndExtras);
                int seqKv = cudaKvCache.CurrentLength;
                MarkProfile(ProfileCategory.KvUpdate);

                _kernels.LaunchAttention(qPtr, cudaKvCache.GetKeysPtr(layer),
                    cudaKvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else
            {
                // Eager fallback path (prefill seqLen>1, quantized KV, or no fused kernel).
                _kernels.LaunchRoPE(qPtr, kPtr, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);
                MarkProfile(ProfileCategory.RopeAndExtras);

                if (kvCache is CudaQuantizedKvCache cudaQKvCache)
                {
                    cudaQKvCache.UpdateDevice(kPtr, vPtr, positions, seqLen, layer, s, _kernels);
                    int seqKv = cudaQKvCache.CurrentLength;

                    // Dequant quantized region + copy window → scratch, then regular attention
                    var (kCachePtr, vCachePtr) = cudaQKvCache.PrepareAttentionScratch(layer, s, _kernels);
                    MarkProfile(ProfileCategory.KvUpdate);
                    _kernels.LaunchAttention(qPtr, kCachePtr, vCachePtr, _state.AttnOutput,
                        seqLen, seqKv, numHeads, numKvHeads, headDim,
                        positions[0], slidingWindow, s);
                }
                else if (kvCache is CudaKvCache cudaKvCache)
                {
                    cudaKvCache.UpdateDevice(kPtr, vPtr, positions, seqLen, layer, s);
                    int seqKv = cudaKvCache.CurrentLength;
                    MarkProfile(ProfileCategory.KvUpdate);

                    _kernels.LaunchAttention(qPtr, cudaKvCache.GetKeysPtr(layer),
                        cudaKvCache.GetValuesPtr(layer), _state.AttnOutput,
                        seqLen, seqKv, numHeads, numKvHeads, headDim,
                        positions[0], slidingWindow, s);
                }
                else
                {
                    MarkProfile(ProfileCategory.KvUpdate);
                    _kernels.LaunchAttention(qPtr, kPtr, vPtr, _state.AttnOutput,
                        seqLen, seqLen, numHeads, numKvHeads, headDim,
                        0, slidingWindow, s);
                }
            }
            MarkProfile(ProfileCategory.Attention);

            // O projection → NormOutput
            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.OProj);

            // ── FUSED: attention residual + FFN norm ──
            // residual = residual + NormOutput (via FP32), NormOutput = rmsnorm(new_residual, ffnNormWeight)
            _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);
            MarkProfile(ProfileCategory.Norm);

        FfnBlock:
            // ── FFN BLOCK (NormOutput has FFN-normalized input) ──
            // MoE path: CudaMoeFfn.Forward takes F32 in/out so we stage via the
            // model-owned F32 conversion buffers. Routed top-k + per-expert SwiGLU +
            // optional shared-expert sum all happen inside the helper.
            CudaMoeLayerWeights? moeLayer = isMoe ? _weights.MoeLayers![layer] : null;
            if (moeLayer is not null)
            {
                _moeScratch ??= new CudaMoeScratch();
                _moeScratch.EnsureCapacity(seqLen, moeLayer);
                int hiddenElems = seqLen * hiddenSize;
                _kernels.LaunchConvertF16ToF32(_state.NormOutput, _moeStagingInF32, hiddenElems, s);
                CudaMoeFfn.Forward(
                    hiddenF32: _moeStagingInF32,
                    outputF32: _moeStagingOutF32,
                    seqLen: seqLen,
                    weights: moeLayer,
                    scratch: _moeScratch,
                    cublasHandle: cublasH,
                    kernels: _kernels,
                    stream: s);
                _kernels.LaunchConvertF32ToF16(_moeStagingOutF32, _state.NormOutput, hiddenElems, s);
                MarkProfile(ProfileCategory.MlpUp);
                MarkProfile(ProfileCategory.Swiglu);
                MarkProfile(ProfileCategory.MlpDown);
                goto EndOfLayer;
            }

            // Gate/Up projections — fused into a single GEMV when packed weights
            // are available (decode-only). Same packing strategy as QKV.
            nint gatePtr, upPtr;
            bool fusedGateUp = seqLen == 1 && lw.GateUpPacked != 0;
            if (fusedGateUp)
            {
                if (_kernels.HasMmq(lw.GateUpPackedQuantType))
                {
                    nint scratch = MaybePreQuantize(_state.NormOutput, lw.GateInputDim, s);
                    _kernels.LaunchQuantizedGemvMmq(lw.GateUpPacked, lw.GateUpPackedQuantType,
                        _state.NormOutput, _state.GateUpPacked,
                        lw.GateUpPackedOutputDim, lw.GateInputDim, scratch, s);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(lw.GateUpPacked, lw.GateUpPackedQuantType,
                        _state.NormOutput, _state.GateUpPacked,
                        lw.GateUpPackedOutputDim, lw.GateInputDim, s);
                }
                gatePtr = _state.GateUpPacked;
                upPtr = _state.GateUpPacked + (nint)((long)lw.GateOutputDim * h);
            }
            else
            {
                Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
                Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);
                gatePtr = _state.FfnGate;
                upPtr = _state.FfnUp;
            }

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(gatePtr, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(upPtr, lw.UpBias, lw.UpOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.MlpUp);

            // SwiGLU (FP16)
            _kernels.LaunchSwiGLU(gatePtr, upPtr, _state.SiluOutput,
                intermediateSize, seqLen, s);
            MarkProfile(ProfileCategory.Swiglu);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.MlpDown);

        EndOfLayer:
            // ── FUSED: FFN residual + next layer's attention norm ──
            // GQA path needs the next layer's AttnNorm pre-applied to NormOutput
            // (FusedAddRmsNorm's combined add + norm). MLA does its own input
            // RMSNorm internally, so for an MLA-next-layer we only need the
            // residual update; the next iteration reads raw hidden from Residual.
            if (layer < numLayers - 1)
            {
                if (isMla)
                {
                    _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.Residual,
                        seqLen * hiddenSize, s);
                }
                else
                {
                    ref readonly var nextLw = ref _weights.Layers[layer + 1];
                    _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                        hiddenSize, eps, seqLen, s);
                }
            }
            else
            {
                // Last processed layer: plain add → HiddenState for final norm
                _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                    seqLen * hiddenSize, s);
            }
            MarkProfile(ProfileCategory.Norm);
        }

        // 5. Final RmsNorm (last token only)
        nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);
        _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
            hiddenSize, eps, 1, s);
        MarkProfile(ProfileCategory.Norm);

        // 6. LM head (last token only) → FP16 logits, then convert to FP32
        Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
            _state.NormOutput, _state.LogitsF16,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);
        MarkProfile(ProfileCategory.LmHead);

        _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
        MarkProfile(ProfileCategory.Convert);

        if (ProfilingEnabled)
            CudaDriverApi.cuEventRecord(_evtEnd, s).ThrowOnError();

        // 7. Stream sync (single sync point for entire forward pass)
        _stream.Synchronize();

        if (ProfilingEnabled)
        {
            CudaDriverApi.cuEventElapsedTime(out float gpuMs, _evtStart, _evtEnd).ThrowOnError();
            LastGpuLaunchMs = gpuMs;

            // Walk per-category events: interval (prev → events[i]) is attributed to category[i].
            Array.Clear(_categoryMsLast);
            nint prev = _evtStart;
            for (int i = 0; i < _profEventCursor; i++)
            {
                CudaDriverApi.cuEventElapsedTime(out float ms, prev, _profEvents![i]).ThrowOnError();
                _categoryMsLast[_profEventCategory![i]] += ms;
                prev = _profEvents[i];
            }
            _profEventCursor = 0;
        }

        // 8. D2H copy FP32 logits to CPU UnmanagedTensor
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// CUDA Graphs decode replay path. Mirrors the eager <c>Forward</c> body for
    /// the seqLen=1 / standard <see cref="CudaKvCache"/> case, but with two structural
    /// changes that make it graph-replay-safe:
    /// <list type="number">
    /// <item><description>The attention launch goes through <see cref="CudaKernels.LaunchAttentionDyn"/>,
    /// which reads <c>seq_kv</c> and <c>position_offset</c> from device-resident ints
    /// (<see cref="_decodeSeqKvDevice"/> / <see cref="_decodePosDevice"/>). Host bumps
    /// these via <c>cuMemcpyHtoD</c> before each <c>cuGraphLaunch</c> — the kernel arg
    /// pointer values are baked into the graph but the values they reference are not.</description></item>
    /// <item><description>The KV-cache write goes through <see cref="CudaKvCache.UpdateDeviceSingleDevicePos"/>,
    /// which fires a <c>kv_write_one_f16</c> kernel that computes <c>dst = base + posPtr[0] * stride</c>
    /// device-side. The eager path's host-computed <c>cuMemcpyDtoDAsync</c> destination
    /// would be baked into the graph and clobber the same row each replay.</description></item>
    /// </list>
    /// Token id and RoPE position already pass through device buffers (TokenIdsDevice,
    /// PositionsDevice), so those uploads land in stable graph-baked pointers and don't
    /// need any kernel changes. The final logits D2H is issued AFTER the graph launch
    /// (and after the stream sync) so it stays a normal sync memcpy.
    /// </summary>
    private ITensor ForwardDecodeGraph(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                        int deviceId, CudaKvCache kvCache)
    {
        _context.MakeCurrent();
        int vocabSize = Config.VocabSize;

        _state.EnsureCapacity(1);
        EnsureGraphScalarBuffers();

        nint s = _stream.Handle;
        int pos = positions[0];
        int seqKv = pos + 1;

        // ── Per-step host inputs uploaded BEFORE the graph launch ──
        // These land in stable device buffers; the graph reads them via baked-in
        // pointers. Each is one 4-byte cuMemcpyHtoD (~1 µs on WDDM).
        unsafe
        {
            int tok = tokenIds[0];
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)(&tok), sizeof(int)).ThrowOnError();
            int p = positions[0];
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)(&p), sizeof(int)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(_decodePosDevice, (nint)(&pos), sizeof(int)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(_decodeSeqKvDevice, (nint)(&seqKv), sizeof(int)).ThrowOnError();
        }

        // Capture the graph the first time we see a graph-eligible decode call against
        // this kvCache. Re-capture if the kvCache identity OR the layer count changed
        // (DebugMaxLayers can flip between calls in tests).
        int effectiveLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        if (_decodeGraphExec == 0
            || !ReferenceEquals(_decodeGraphKvCache, kvCache)
            || _decodeGraphLayerCount != effectiveLayers)
        {
            DisposeDecodeGraph();
            CaptureDecodeGraph(kvCache, effectiveLayers);
            _decodeGraphKvCache = kvCache;
            _decodeGraphLayerCount = effectiveLayers;
        }

        // Replay: single packet submission.
        CudaDriverApi.cuGraphLaunch(_decodeGraphExec, s).ThrowOnError();
        _stream.Synchronize();

        // Update host-side KV length so the next eager call (or sampler stop check) sees
        // the right value. The graph already wrote into the cache at posPtr[0].
        kvCache.AdvanceLengthForGraphDecode(seqKv);

        // D2H final logits.
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// CUDA Graphs decode replay path for the mixed-precision quantized KV cache.
    /// Mirrors <see cref="ForwardDecodeGraph"/> but routes the KV-cache update and
    /// attention-scratch preparation through <see cref="CudaQuantizedKvCache"/>'s
    /// graph-friendly variants, which read the absolute decode position from
    /// <see cref="_decodePosDevice"/> and predicate quantize-on-evict device-side.
    /// </summary>
    private ITensor ForwardDecodeGraphQuantized(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                                  int deviceId, CudaQuantizedKvCache kvCache)
    {
        _context.MakeCurrent();
        int vocabSize = Config.VocabSize;

        _state.EnsureCapacity(1);
        EnsureGraphScalarBuffers();

        nint s = _stream.Handle;
        int pos = positions[0];
        int seqKv = pos + 1;

        unsafe
        {
            int tok = tokenIds[0];
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)(&tok), sizeof(int)).ThrowOnError();
            int p = positions[0];
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)(&p), sizeof(int)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(_decodePosDevice, (nint)(&pos), sizeof(int)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(_decodeSeqKvDevice, (nint)(&seqKv), sizeof(int)).ThrowOnError();
        }

        int effectiveLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        if (_decodeGraphExec == 0
            || !ReferenceEquals(_decodeGraphKvCache, kvCache)
            || _decodeGraphLayerCount != effectiveLayers)
        {
            DisposeDecodeGraph();
            CaptureDecodeGraphQuantized(kvCache, effectiveLayers);
            _decodeGraphKvCache = kvCache;
            _decodeGraphLayerCount = effectiveLayers;
        }

        CudaDriverApi.cuGraphLaunch(_decodeGraphExec, s).ThrowOnError();
        _stream.Synchronize();

        kvCache.AdvanceLengthForGraphDecode(seqKv);

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// Captures the decode forward into a CUDA Graph by running the same kernel sequence
    /// as the eager path, but on a stream that has <c>cuStreamBeginCapture</c> active.
    /// All API calls that would normally enqueue work on the stream are added to the
    /// graph instead of executing. <c>cuStreamEndCapture</c> returns the topology-only
    /// graph, which we instantiate into an executable graph and cache.
    /// </summary>
    private void CaptureDecodeGraph(CudaKvCache kvCache, int numLayers)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        const int seqLen = 1;
        const int h = sizeof(ushort);

        nint s = _stream.Handle;

        CudaDriverApi.cuStreamBeginCapture_v2(s, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();

        try
        {
            // --- Same kernel sequence as the eager Forward(seqLen=1) path ---

            // Embedding lookup → FP16 HiddenState
            _kernels.LaunchEmbeddingLookup(
                _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
                _state.TokenIdsDevice, _state.HiddenState,
                seqLen, hiddenSize, s);

            // Layer 0 setup: copy hidden→residual, RmsNorm→NormOutput
            long hiddenBytes = (long)seqLen * hiddenSize * h;
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
            _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

            if (numLayers == 0)
                CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();

            for (int layer = 0; layer < numLayers; layer++)
            {
                ref readonly var lw = ref _weights.Layers[layer];

                Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
                Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

                if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
                if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

                if (lw.QNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
                if (lw.KNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

                int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
                _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);

                // KV-cache update via device-resident position; replaces the eager
                // path's cuMemcpyDtoDAsync (which would bake the dst address).
                kvCache.UpdateDeviceSingleDevicePos(_state.K, _state.V, layer, _decodePosDevice, s, _kernels);

                // Attention with device-resident seq_kv / position_offset.
                _kernels.LaunchAttentionDyn(_state.Q, kvCache.GetKeysPtr(layer),
                    kvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, _decodeSeqKvDevice, numHeads, numKvHeads, headDim,
                    _decodePosDevice, slidingWindow, s);

                Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
                if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);

                Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
                Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

                if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
                if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

                _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput, intermediateSize, seqLen, s);

                Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
                if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

                if (layer < numLayers - 1)
                {
                    ref readonly var nextLw = ref _weights.Layers[layer + 1];
                    _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                        hiddenSize, eps, seqLen, s);
                }
                else
                {
                    _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                        seqLen * hiddenSize, s);
                }
            }

            nint lastHidden = _state.HiddenState; // seqLen=1
            _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
                hiddenSize, eps, 1, s);

            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1);

            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
        }
        catch
        {
            // Capture must always be ended (or aborted) — otherwise the stream is
            // left in capturing state and all subsequent ops fail.
            CudaDriverApi.cuStreamEndCapture(s, out _);
            throw;
        }

        CudaDriverApi.cuStreamEndCapture(s, out _decodeGraph).ThrowOnError();
        CudaDriverApi.cuGraphInstantiateWithFlags(out _decodeGraphExec, _decodeGraph, 0).ThrowOnError();
    }

    /// <summary>
    /// Quantized-KV variant of <see cref="CaptureDecodeGraph"/>. Identical kernel
    /// sequence except the per-layer KV-cache update goes through
    /// <see cref="CudaQuantizedKvCache.UpdateDeviceForGraph"/> (FP16 ring write +
    /// predicated quantize-on-evict) and the attention reads from
    /// <see cref="CudaQuantizedKvCache.PrepareAttentionScratchForGraph"/>'s scratch
    /// buffers (predicated dequant + window scatter). Both ops are driven by the
    /// existing device-resident <see cref="_decodePosDevice"/> counter so no new
    /// device-side state is required.
    /// </summary>
    private void CaptureDecodeGraphQuantized(CudaQuantizedKvCache kvCache, int numLayers)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        const int seqLen = 1;
        const int h = sizeof(ushort);

        nint s = _stream.Handle;

        CudaDriverApi.cuStreamBeginCapture_v2(s, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();

        try
        {
            _kernels.LaunchEmbeddingLookup(
                _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
                _state.TokenIdsDevice, _state.HiddenState,
                seqLen, hiddenSize, s);

            long hiddenBytes = (long)seqLen * hiddenSize * h;
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
            _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

            if (numLayers == 0)
                CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();

            for (int layer = 0; layer < numLayers; layer++)
            {
                ref readonly var lw = ref _weights.Layers[layer];

                Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
                Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

                if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
                if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

                if (lw.QNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
                if (lw.KNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

                int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
                _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);

                // KV-cache update (FP16 ring write + predicated quantize-on-evict),
                // device-side eviction state.
                kvCache.UpdateDeviceForGraph(_state.K, _state.V, layer, _decodePosDevice, s, _kernels);

                // Dequant the quantized region + scatter the window into the FP16 attention
                // scratch — both predicated, both reading position from _decodePosDevice.
                var (kCachePtr, vCachePtr) =
                    kvCache.PrepareAttentionScratchForGraph(layer, _decodePosDevice, s, _kernels);

                // Attention with device-resident seq_kv / position_offset.
                _kernels.LaunchAttentionDyn(_state.Q, kCachePtr, vCachePtr, _state.AttnOutput,
                    seqLen, _decodeSeqKvDevice, numHeads, numKvHeads, headDim,
                    _decodePosDevice, slidingWindow, s);

                Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
                if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);

                Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
                Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

                if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
                if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

                _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput, intermediateSize, seqLen, s);

                Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
                if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

                if (layer < numLayers - 1)
                {
                    ref readonly var nextLw = ref _weights.Layers[layer + 1];
                    _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                        hiddenSize, eps, seqLen, s);
                }
                else
                {
                    _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                        seqLen * hiddenSize, s);
                }
            }

            nint lastHidden = _state.HiddenState; // seqLen=1
            _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
                hiddenSize, eps, 1, s);

            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1);

            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
        }
        catch
        {
            CudaDriverApi.cuStreamEndCapture(s, out _);
            throw;
        }

        CudaDriverApi.cuStreamEndCapture(s, out _decodeGraph).ThrowOnError();
        CudaDriverApi.cuGraphInstantiateWithFlags(out _decodeGraphExec, _decodeGraph, 0).ThrowOnError();
    }

    private void EnsureGraphScalarBuffers()
    {
        if (_decodePosDevice == 0)
            CudaDriverApi.cuMemAlloc_v2(out _decodePosDevice, sizeof(int)).ThrowOnError();
        if (_decodeSeqKvDevice == 0)
            CudaDriverApi.cuMemAlloc_v2(out _decodeSeqKvDevice, sizeof(int)).ThrowOnError();
    }

    private void DisposeDecodeGraph()
    {
        if (_decodeGraphExec != 0) { CudaDriverApi.cuGraphExecDestroy(_decodeGraphExec); _decodeGraphExec = 0; }
        if (_decodeGraph != 0) { CudaDriverApi.cuGraphDestroy(_decodeGraph); _decodeGraph = 0; }
        _decodeGraphKvCache = null;
        _decodeGraphLayerCount = 0;
    }

    /// <summary>
    /// Pre-quantizes the input vector to INT8 (Q8_1) into <see cref="CudaForwardState.PreQ8_1Scratch"/>
    /// when the pre-Q8_1 kernel is available, the scratch is large enough, inputDim is a
    /// 32-element multiple, AND inputDim ≥ <see cref="CudaKernels.MmvqLargeKThreshold"/>. Returns
    /// the scratch pointer for the GEMV launcher to consume, or 0 when the on-the-fly Stage 1
    /// path should be used.
    /// <para>
    /// The k threshold is the same one that gates MMVQ-large dispatch: below 1024 elements the
    /// MMQ-4-rows path's in-kernel Stage 1 already amortizes across 4 rows, so the extra
    /// pre-quant launch overhead (~22 µs on WDDM eager) outweighs the saving and SmolLM-class
    /// models regress slightly. At k≥1024 MMVQ-large runs Stage 1 once per output row (n× across
    /// the GEMV) — pre-quantization eliminates that and unlocks the structural win on
    /// Qwen3-class models.
    /// </para>
    /// </summary>
    private nint MaybePreQuantize(nint input, int inputDim, nint stream)
    {
        if (!_kernels.HasPreQ8_1) return 0;
        if (_state.PreQ8_1Scratch == 0) return 0;
        if (inputDim > _state.PreQ8_1ScratchK) return 0;
        if ((inputDim & 31) != 0) return 0;
        if (inputDim < CudaKernels.MmvqLargeKThreshold) return 0;

        _kernels.LaunchQuantizeXToQ8_1(input, _state.PreQ8_1Scratch, inputDim, stream);
        return _state.PreQ8_1Scratch;
    }

    /// <summary>
    /// Dispatches projection as cuBLAS HGEMM (prefill) or quantized/cuBLAS GEMV (decode).
    /// For quantized weights with no persistent FP16 copy (<paramref name="fp16Weight"/> == 0),
    /// dequantizes on-the-fly into <see cref="CudaForwardState.DequantScratch"/> before calling cuBLAS.
    /// </summary>
    private void Project(nint quantWeight, QuantizationType qt, nint fp16Weight,
                          nint input, nint output, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;

        if (seqLen > 1) // Prefill: cuBLAS HGEMM
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                // Quantized: dequant into scratch, then GEMM
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (quantWeight != 0 && _kernels.HasMmq(qt))
        {
            // Decode: MMQ-style fused dequant+matmul (dp4a) — faster than the FP fmuladd kernel.
            // Routes Q4_K, Q5_K, Q6_K through the dp4a path; the rest fall through to the
            // legacy FP-fmuladd kernel below. Use the pre-Q8_1 scratch when available and
            // when inputDim fits — eliminates per-block redundant Stage 1 work
            // (especially material for the MMVQ-large variant).
            nint scratch = MaybePreQuantize(input, inputDim, s);
            _kernels.LaunchQuantizedGemvMmq(quantWeight, qt, input, output, outputDim, inputDim, scratch, s);
        }
        else if (quantWeight != 0 && CudaKernels.HasQuantizedGemv(qt)) // Decode: quantized GEMV
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, s);
        }
        else // Decode fallback: cuBLAS GEMV (F16/F32 weights or unsupported quant)
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.GemvF16(_cublas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    /// <summary>
    /// Creates a <see cref="CudaKvCache"/> for this model.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for the cache.</param>
    public CudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new CudaKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
    }

    /// <summary>
    /// Creates a KV-cache with optional quantization for this model.
    /// Returns <see cref="CudaQuantizedKvCache"/> when quantization is configured,
    /// otherwise a standard <see cref="CudaKvCache"/>.
    /// </summary>
    public Core.Attention.IKvCache CreateKvCache(int maxSeqLen, Core.Configuration.KvCacheConfig config)
    {
        _context.MakeCurrent();
        if (!config.IsQuantized)
            return new CudaKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
        return new CudaQuantizedKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen, config);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        DisposeDecodeGraph();
        if (_decodePosDevice != 0) { CudaDriverApi.cuMemFree_v2(_decodePosDevice); _decodePosDevice = 0; }
        if (_decodeSeqKvDevice != 0) { CudaDriverApi.cuMemFree_v2(_decodeSeqKvDevice); _decodeSeqKvDevice = 0; }
        if (_evtStart != 0) CudaDriverApi.cuEventDestroy_v2(_evtStart);
        if (_evtEnd != 0) CudaDriverApi.cuEventDestroy_v2(_evtEnd);
        if (_profEvents != null)
        {
            for (int i = 0; i < _profEvents.Length; i++)
                if (_profEvents[i] != 0) CudaDriverApi.cuEventDestroy_v2(_profEvents[i]);
        }
        if (_mlaRopeCosF32 != 0) { CudaDriverApi.cuMemFree_v2(_mlaRopeCosF32); _mlaRopeCosF32 = 0; }
        if (_mlaRopeSinF32 != 0) { CudaDriverApi.cuMemFree_v2(_mlaRopeSinF32); _mlaRopeSinF32 = 0; }
        if (_moeStagingInF32 != 0) { CudaDriverApi.cuMemFree_v2(_moeStagingInF32); _moeStagingInF32 = 0; }
        if (_moeStagingOutF32 != 0) { CudaDriverApi.cuMemFree_v2(_moeStagingOutF32); _moeStagingOutF32 = 0; }
        _mlaScratchF16?.Dispose();
        _mlaKvCache?.Dispose();
        _moeScratch?.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
