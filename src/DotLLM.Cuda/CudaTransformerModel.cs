using System.Runtime.CompilerServices;
using System.Buffers;
using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
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
    private readonly TransformerWeights? _cpuWeights;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;
    private readonly bool _useHighPrecisionForward;

    // ── Gemma-4 (DiffusionGemma AR) per-attention-type rope params ──
    // Sliding (SWA) layers use _ropeTheta/_ropeDim (full NeoX rotation over the
    // sliding head dim). Global (full-attention) layers use _gemma4GlobalRopeTheta
    // and a PARTIAL NeoX rotation of _gemma4GlobalRotatedPairs pairs, coupling
    // (i, i + GlobalHeadDim/2), with the freq base over the full GlobalHeadDim.
    // Computed once in the ctor; mirror of TransformerModel's GetLayerRope.
    private readonly bool _isGemma4;
    private readonly float _gemma4GlobalRopeTheta;
    private readonly int _gemma4GlobalRotatedPairs;
    private readonly float _gemma4FinalSoftcap;

    // Launch-ceiling profiling (env DOTLLM_PROFILE_LAUNCH=1).
    // Measures CPU-side kernel-dispatch time (queue all launches, no GPU wait) vs the full
    // forward including the single _stream.Synchronize(). Zero-cost when the env var is unset.
    private static readonly bool s_profileLaunch =
        Environment.GetEnvironmentVariable("DOTLLM_PROFILE_LAUNCH") == "1";
    private long _profDispatchTicks;   // accumulated dispatch-only ticks (decode steps)
    private long _profTotalTicks;      // accumulated total forward ticks (decode steps)
    private int _profSteps;            // counted steady-state decode steps
    private int _profWarmupSkipped;    // warmup decode steps skipped before accumulation
    private const int ProfWarmupSteps = 10; // skip first N decode steps (lazy alloc/JIT)
    private static readonly bool s_i2sA8Decode =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_I2S_A8") == "1";

    // LoRA adapter staging — set by the 5-arg Forward overload. Read in the layer
    // loop to gate fused decode kernels off and apply the per-projection delta.
    private ILoraAdapter? _currentAdapter;
    private CudaLoraWeights? _cudaLora;

    /// <summary>
    /// Steady-state status of the single-token decode CUDA graph (last decode step). Diagnostic only.
    /// </summary>
    public CudaDecodeGraphState DecodeGraphState { get; private set; } = CudaDecodeGraphState.None;

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

    /// <summary>Disable the F32 activation correctness path for IQ4-family models.</summary>
    public static bool EnableHighPrecisionIQuants { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ENABLE_HIGH_PRECISION_IQUANTS") == "1";

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

    // F32 gemma4 dual-FFN scratch: dense-branch result, MoE-branch result, and the
    // custom-router input (rms(attn_out)·RouterScale·1/√H). Sized [seqLen × hidden].
    private nint _gemma4DenseF32;
    private nint _gemma4MoeF32;
    private nint _gemma4RouterInF32;
    private int _gemma4ScratchCapacityElems;

    // Scratch for the per-projection activation Q8_0 round-trip (matches the CPU
    // oracle's on-the-fly activation quantization). Sized to the largest projection
    // input the gemma4 forward feeds (hidden or the attn-output width).
    private nint _gemma4ActScratchF32;
    private int _gemma4ActScratchElems;

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
        CudaKernels kernels, GgufFile? gguf, TransformerWeights? cpuWeights, int deviceId,
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

        // ── Gemma-4 per-attention-type rope (global layers: partial NeoX) ──
        _isGemma4 = weights.Gemma4Layers is not null;
        if (_isGemma4 && config.GlobalRoPEConfig is RoPEConfig gcfg)
        {
            int baseDim = gcfg.DimensionCount > 0
                ? gcfg.DimensionCount
                : (config.GlobalHeadDim ?? config.HeadDim);
            float prf = config.PartialRotaryFactor ?? 1.0f;
            int rotated = (int)MathF.Floor(prf * baseDim);
            rotated &= ~1;                       // round down to even (RoPE rotates pairs)
            if (rotated < 2) rotated = 2;
            rotated = Math.Min(rotated, config.GlobalHeadDim ?? config.HeadDim);
            _gemma4GlobalRotatedPairs = rotated / 2;
            _gemma4GlobalRopeTheta = gcfg.Theta;
        }
        _gemma4FinalSoftcap = config.FinalLogitSoftcap ?? 0f;

        _useHighPrecisionForward = ShouldUseHighPrecisionForward(weights);
        // The gemma4 forward runs the F32 path and uses the host LM head (the tied
        // vocab x hidden table is too large to expand to F32 device scratch), so it
        // also needs the CPU weights retained.
        if (_useHighPrecisionForward || _isGemma4)
        {
            _cpuWeights = cpuWeights;
        }
        else
        {
            _cpuWeights = null;
            cpuWeights?.Dispose();
        }

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
        // GPU-only path: skip the F32 host dequant of the per-expert MoE 3D
        // tensors. Saves ~2.2 GB host RAM per V2-Lite Q4_K_M MoE layer
        // (the GPU loader takes the raw view; the CPU MoeSwiGluMlp oracle
        // is never called on a CudaTransformerModel anyway).
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);

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
    /// Test-only factory that wires a CUDA model around an already-built CPU
    /// <see cref="TransformerWeights"/>. Mirrors
    /// <see cref="TransformerModel.BuildFromPrebuiltWeights(TransformerWeights, ModelConfig, ThreadingConfig?)"/>
    /// so synthetic fixtures can pin CPU-vs-CUDA parity without a GGUF / safetensors
    /// file. Caller retains ownership of every host pointer carried by
    /// <paramref name="cpuWeights"/>; <see cref="Dispose"/> only releases the device
    /// allocations created by the loader.
    /// </summary>
    /// <param name="cpuWeights">Already-built weight bundle (host F32 pointers OK).</param>
    /// <param name="config">Matching model configuration.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null auto-detects.</param>
    internal static CudaTransformerModel BuildFromPrebuiltWeights(
        TransformerWeights cpuWeights, ModelConfig config,
        int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(config);
        // Skip the VRAM-estimate path — synthetic fixtures are tiny and we
        // don't have a GGUF tensor manifest to walk; passing 0 disables the
        // warning emission inside LoadFromCpuWeights.
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

        // Gemma-4 full-attention (global) layers may use a distinct head dim and
        // KV-head count (GlobalHeadDim / NumGlobalKvHeads). Size the Q/K/V scratch
        // for the LARGER of the sliding and global layer types so a single
        // allocation covers both; per-layer dispatch uses the layer's own dims.
        int stateHeadDim = Math.Max(config.HeadDim, config.GlobalHeadDim ?? config.HeadDim);
        int stateKvHeads = Math.Max(config.NumKvHeads, config.NumGlobalKvHeads ?? config.NumKvHeads);

        // BitNet's residual stream exceeds FP16 range in deep layers, so it carries the
        // residual in FP32 (overflow→NaN otherwise).
        bool useFp32Residual = config.Architecture == Architecture.BitNet;

        // G1: default the FP16-accumulate prefill GEMM on for GeForce Ampere with quantized weights —
        // validated quality-safe (decode-mode perplexity ±0.000%, prefill −0.079% on Llama-3.2-1B Q8_0)
        // and ~1.06–1.22× faster whole-prefill there. Off elsewhere: datacenter Ampere doesn't throttle
        // FP32 accumulate (no win), and 16F decode-GEMV for non-quant weights is unvalidated (quant
        // decode is integer, so 16F never touches it). DOTLLM_CUDA_GEMM_16F overrides ("1"/"0").
        // BitNet excluded: its prefill dequantizes I2_S → F16 and routes through CudaGemm.LinearF16,
        // but its activations/residual exceed FP16 range (the model carries the residual in FP32 for
        // exactly this reason), so FP16-accumulate prefill GEMM is unvalidated and risks overflow.
        var device = CudaDevice.GetDevice(deviceId);
        bool geForceAmpere = device.ComputeCapabilityMajor == 8
            && device.Name.Contains("GeForce", StringComparison.OrdinalIgnoreCase);
        bool quantizedWeights = weights.Layers.Length > 0
            && weights.Layers[0].QQuantType is not (QuantizationType.F32 or QuantizationType.F16);
        CudaGemm.ConfigureDefault(
            geForceAmpere && quantizedWeights && config.Architecture != Architecture.BitNet);

        var state = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, stateKvHeads,
            stateHeadDim, config.IntermediateSize, config.VocabSize, useFp32Residual);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        // Translate the public RoPEType enum (Norm=0, NeoX=2) to the CUDA
        // kernel's encoding (Norm=0, NeoX=1). Direct cast was a long-standing
        // bug — passing NeoX=2 fell into the kernel's "anything but 1 → GPT-J
        // interleaved" branch, silently mis-rotating every Qwen/Phi forward.
        int ropeType = CudaKernels.ToCudaRopeType(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new CudaTransformerModel(config, weights, state, stream, cublas, context,
            kernels, gguf, cpuWeights, deviceId, ropeTheta, ropeDim, ropeType, vramWarning);
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

    /// <summary>
    /// Lazily (re)allocates the gemma4 dual-FFN F32 scratch (dense result, MoE
    /// result, custom-router input), each sized <c>[seqLen × hidden]</c>. Also
    /// ensures the shared MoE staging input buffer (<see cref="_moeStagingInF32"/>)
    /// exists — the gemma4 MoE branch reuses it as the expert input.
    /// </summary>
    private void EnsureGemma4Staging(int seqLen, int hiddenSize)
    {
        EnsureMoeStaging(seqLen, hiddenSize);
        int needed = seqLen * hiddenSize;
        if (_gemma4ScratchCapacityElems >= needed) return;

        if (_gemma4DenseF32 != 0) CudaDriverApi.cuMemFree_v2(_gemma4DenseF32);
        if (_gemma4MoeF32 != 0) CudaDriverApi.cuMemFree_v2(_gemma4MoeF32);
        if (_gemma4RouterInF32 != 0) CudaDriverApi.cuMemFree_v2(_gemma4RouterInF32);
        long bytes = (long)needed * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _gemma4DenseF32, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _gemma4MoeF32, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _gemma4RouterInF32, (nuint)bytes).ThrowOnError();
        _gemma4ScratchCapacityElems = needed;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        bool isMla = _weights.MlaLayers is not null;
        bool isMoe = _weights.MoeLayers is not null;
        bool isBitNet = Config.Architecture == Architecture.BitNet;

        // Gemma-4 (DiffusionGemma AR) — dedicated F32 forward. AR only (no KV cache
        // on CUDA yet; the cacheless path covers single-shot / short-context decode).
        if (_isGemma4)
        {
            if (kvCache is not null)
                throw new NotSupportedException(
                    "CUDA gemma4 forward is cacheless (autoregressive, no KV cache yet). "
                    + "Pass kvCache: null. KV-cache support is a follow-up.");
            return ForwardGemma4(tokenIds, positions, deviceId);
        }

        if (_useHighPrecisionForward && kvCache is null && !isMla && !isMoe)
        {
            return ForwardHighPrecision(tokenIds, positions, deviceId);
        }

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
            && !isMla && !isMoe
            && !isBitNet                    // dev's generic capture body omits BitNet's FP32 residual, Sub-LN and relu² — replaying it on BitNet produces garbage
            && _currentAdapter is null)     // the captured body has no ApplyLoraDeltaDevice call, so an active adapter would be silently dropped on every decoded token
        {
            if (kvCache is CudaKvCache stdKv)
            {
                // Guard against silent corruption: if speculative-verify rolled the
                // cache back, the captured cuGraphExec was instantiated against a
                // device-side write-pos counter that is now stale relative to the
                // actual cache state. Today speculative decoding is restricted to
                // greedy (non-graph), so this branch only fires if a future change
                // wires both on simultaneously without invalidating the graph.
                if (stdKv.WasRolledBack)
                    throw new InvalidOperationException(
                        "CudaKvCache.Rollback was called on this cache; the captured " +
                        "cuGraphExec is no longer valid. Either disable UseGraphCapture " +
                        "before the rollback or invalidate the graph and recapture.");
                return ForwardDecodeGraph(tokenIds, positions, deviceId, stdKv);
            }
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

        // Profiling is only meaningful for the single-token decode step.
        bool profile = s_profileLaunch && seqLen == 1;
        long dispatchStartTs = profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

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

        // Decode CUDA-graph capture/replay is handled by dev's dedicated
        // ForwardDecodeGraph / ForwardDecodeGraphQuantized early-return path
        // (dispatched near the top of this method when UseGraphCapture is on).
        // This straight-through body runs without graph wrapping; it serves prefill,
        // BitNet I2_S decode, and adapter-active decode (all graph-ineligible).

        // 2. Embedding lookup → FP16 HiddenState
        _kernels.LaunchEmbeddingLookup(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState,
            seqLen, hiddenSize, s);
        MarkProfile(ProfileCategory.Embed);

        // FP32 residual stream for BitNet (its residual magnitude exceeds FP16's ~65504 ceiling).
        bool fp32Res = _state.ResidualF32 != 0 && isBitNet;

        // 3. Layer 0 setup: copy hidden→residual; on the GQA path also
        //    pre-RmsNorm into NormOutput. The MLA path skips the pre-norm —
        //    CudaMlaAttention.ForwardF16 applies its own input RMSNorm internally,
        //    and consumes the raw hidden state from Residual. BitNet carries the
        //    residual in FP32 (LaunchCopyF16ToF32 into ResidualF32).
        long hiddenBytes = (long)seqLen * hiddenSize * h;
        if (fp32Res)
            _kernels.LaunchCopyF16ToF32(_state.HiddenState, _state.ResidualF32, seqLen * hiddenSize, s);
        else
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

        // When skipping all layers, treat embedding output as final hidden state.
        // (HiddenState already holds the embedding; with FP16 residual we restore from the
        // residual copy, with FP32 residual HiddenState is already correct so nothing to do.)
        if (numLayers == 0 && !fp32Res)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // When a LoRA adapter is active, every fused decode kernel below is bypassed.
            // Declared at the top of the loop (before the MLA `goto FfnBlock`) so it is
            // definitely assigned on every path that reaches the FFN gate/up gating.
            bool adapterActive = _currentAdapter is not null;

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

            // adapterActive (declared at the top of this layer loop) gates the fused
            // decode kernels below so the LoRA delta lands against the SAME intermediate
            // buffers the CPU path uses (see TransformerModel.cs). When no adapter is
            // active this is byte-for-byte the existing decode path.

            // ── ATTENTION BLOCK (NormOutput has normalized input) ──
            // Q/K/V projections. Decode dispatch order:
            //  1. I2_S (BitNet) fused 3-way GEMV → _state.Q/K/V (no adapter, no A8 mode)
            //  2. Pre-packed quantized QKV (dev) → one fused GEMV into _state.QkvPacked
            //  3. Fallback: three separate Project calls → _state.Q/K/V
            // Aliases qPtr/kPtr/vPtr hide the path choice from the rest of the layer body.
            // An active adapter forces the unfused (_state.Q/K/V) path so LoRA deltas land
            // on the same intermediate buffers as the CPU path.
            nint qPtr, kPtr, vPtr;
            bool fusedI2SQkv = !adapterActive && !s_i2sA8Decode
                && CanFuseI2SDecode(seqLen, lw.QQuantType, lw.KQuantType, lw.VQuantType,
                    lw.QInputDim, lw.KInputDim, lw.VInputDim);
            bool fusedQkv = !adapterActive && !fusedI2SQkv && seqLen == 1 && lw.QkvPacked != 0;
            if (fusedI2SQkv)
            {
                _kernels.LaunchI2_SGemv3F16In(
                    lw.QQuant, lw.KQuant, lw.VQuant, _state.NormOutput,
                    _state.Q, _state.K, _state.V,
                    lw.QOutputDim, lw.KOutputDim, lw.VOutputDim, lw.QInputDim, s);
                qPtr = _state.Q;
                kPtr = _state.K;
                vPtr = _state.V;
            }
            else if (fusedQkv)
            {
                if (_kernels.HasMmq(lw.QkvPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

            // LoRA delta (q/k/v): y += scale · (NormOutput · B) · A. Applied AFTER bias and
            // BEFORE QK-norm + RoPE, mirroring TransformerModel.cs:298-300. x = NormOutput
            // (the projection input), y = Q/K/V (the raw projection output). No-op for
            // projections the adapter does not target (ApplyLoraDeltaDevice early-returns).
            if (adapterActive)
            {
                ApplyLoraDeltaDevice(layer, "q_proj", _state.NormOutput, _state.Q, seqLen);
                ApplyLoraDeltaDevice(layer, "k_proj", _state.NormOutput, _state.K, seqLen);
                ApplyLoraDeltaDevice(layer, "v_proj", _state.NormOutput, _state.V, seqLen);
            }

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
                else if (kvCache is CudaTurboQuantKvCache cudaTqKvCache)
                {
                    // TurboQuant: encode fresh FP16 K/V → codes, then dequant the live range into the
                    // shared FP16 scratch the attention kernel reads (same shape as a plain F16 cache).
                    cudaTqKvCache.UpdateDevice(kPtr, vPtr, positions, seqLen, layer, s, _kernels);
                    int seqKv = cudaTqKvCache.CurrentLength;
                    var (kCachePtr, vCachePtr) = cudaTqKvCache.PrepareAttentionScratch(layer, s, _kernels);
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

            // Optional attention Sub-LN (BitNet): RMSNorm over the attention output [numHeads·headDim]
            // before the output projection. No-op for non-BitNet models (weight == 0).
            // When an adapter is active we MUST take the unfused path so that, at delta
            // time, _state.AttnOutput holds the Sub-LN'd attention output — the exact
            // O-projection input the CPU uses (TransformerModel.cs:354-368: Sub-LN is
            // applied in place into attnOut, then the O GEMM reads attnOut).
            bool fusedAttnSubNormO = !adapterActive && !s_i2sA8Decode && CanFuseI2SNormDecode(
                seqLen, lw.AttnSubNormWeight, lw.OQuantType, lw.OInputDim, numHeads * headDim);
            if (fusedAttnSubNormO)
            {
                _kernels.LaunchI2_SGemvNormF16In(
                    lw.OQuant, _state.AttnOutput, lw.AttnSubNormWeight, _state.NormOutput,
                    lw.OOutputDim, lw.OInputDim, eps, s);
            }

            if (lw.AttnSubNormWeight != 0 && !fusedAttnSubNormO)
                _kernels.LaunchRmsNorm(_state.AttnOutput, lw.AttnSubNormWeight, _state.AttnOutput,
                    numHeads * headDim, eps, seqLen, s);

            // O projection → NormOutput
            if (!fusedAttnSubNormO)
                Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.OProj);

            // LoRA delta (o_proj): y += scale · (AttnOutput · B) · A. Mirrors
            // TransformerModel.cs:374 — x is the Sub-LN'd attention output (AttnOutput,
            // normed in place above on the unfused path), y is the O-projection output
            // (NormOutput). Applied after the O bias and BEFORE the fused residual add.
            if (adapterActive)
                ApplyLoraDeltaDevice(layer, "o_proj", _state.AttnOutput, _state.NormOutput, seqLen);

            // ── FUSED: attention residual + FFN norm ──
            // residual = residual + NormOutput, NormOutput = rmsnorm(new_residual, ffnNormWeight).
            // BitNet carries the residual in FP32 to avoid FP16 overflow.
            if (fp32Res)
                _kernels.LaunchFusedAddRmsNormF32Res(_state.ResidualF32, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
            else
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

            // Gate/Up projections. Decode dispatch order mirrors Q/K/V:
            //  1. I2_S (BitNet) fused 2-way GEMV → _state.FfnGate/FfnUp
            //  2. Pre-packed quantized Gate/Up (dev) → one fused GEMV into _state.GateUpPacked
            //  3. Fallback: two separate Project calls → _state.FfnGate/FfnUp
            // An active adapter forces the unfused path so LoRA deltas land on _state.FfnGate/FfnUp.
            nint gatePtr, upPtr;
            bool fusedI2SGateUp = !adapterActive && !s_i2sA8Decode
                && CanFuseI2SDecode(seqLen, lw.GateQuantType, lw.UpQuantType, lw.GateInputDim, lw.UpInputDim);
            bool fusedGateUp = !adapterActive && !fusedI2SGateUp && seqLen == 1 && lw.GateUpPacked != 0;
            if (fusedI2SGateUp)
            {
                _kernels.LaunchI2_SGemv2F16In(
                    lw.GateQuant, lw.UpQuant, _state.NormOutput,
                    _state.FfnGate, _state.FfnUp,
                    lw.GateOutputDim, lw.UpOutputDim, lw.GateInputDim, s);
                gatePtr = _state.FfnGate;
                upPtr = _state.FfnUp;
            }
            else if (fusedGateUp)
            {
                if (_kernels.HasMmq(lw.GateUpPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

            // LoRA delta (gate/up): y += scale · (NormOutput · B) · A. Mirrors the CPU path —
            // x = NormOutput (the FFN-normed input), y = FfnGate/FfnUp (raw projection outputs).
            // Applied after the gate/up biases and BEFORE the GLU activation below. The adapter-
            // active path forces the unfused projections so gatePtr/upPtr alias _state.FfnGate/FfnUp.
            if (adapterActive)
            {
                ApplyLoraDeltaDevice(layer, "gate_proj", _state.NormOutput, _state.FfnGate, seqLen);
                ApplyLoraDeltaDevice(layer, "up_proj", _state.NormOutput, _state.FfnUp, seqLen);
            }

            // Gated activation (FP16). BitNet b1.58 uses squared-ReLU GLU followed by a Sub-LN
            // RMSNorm; the un-normalized relu(gate)²·up intermediate overflows FP16, so when the
            // Sub-LN weight is present we fuse activation + RMSNorm (large value kept in FP32,
            // only the normalized O(1) result hits FP16). Otherwise dispatch the plain activation.
            if (Config.ActivationFunction == ActivationFunction.ReluSquared && lw.FfnSubNormWeight != 0)
            {
                _kernels.LaunchReLU2GluRmsNorm(gatePtr, upPtr, lw.FfnSubNormWeight,
                    _state.SiluOutput, intermediateSize, eps, seqLen, s);
            }
            else if (Config.ActivationFunction == ActivationFunction.ReluSquared)
            {
                _kernels.LaunchReLU2(gatePtr, upPtr, _state.SiluOutput,
                    intermediateSize, seqLen, s);

                // Optional FFN Sub-LN (BitNet) for the non-fused fallback. No-op when weight == 0.
                if (lw.FfnSubNormWeight != 0)
                    _kernels.LaunchRmsNorm(_state.SiluOutput, lw.FfnSubNormWeight, _state.SiluOutput,
                        intermediateSize, eps, seqLen, s);
            }
            else
            {
                _kernels.LaunchSwiGLU(gatePtr, upPtr, _state.SiluOutput,
                    intermediateSize, seqLen, s);
            }
            MarkProfile(ProfileCategory.Swiglu);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.MlpDown);

            // LoRA delta (down_proj): y += scale · (SiluOutput · B) · A. Mirrors the CPU path —
            // x = SiluOutput (the post-(SwiGLU/ReLU²) GLU, possibly FFN-Sub-LN'd, down-projection
            // input), y = NormOutput (the down output). Applied after the down bias, BEFORE the
            // fused residual add. Not reached on the MoE path (it jumps to EndOfLayer below).
            if (adapterActive)
                ApplyLoraDeltaDevice(layer, "down_proj", _state.SiluOutput, _state.NormOutput, seqLen);

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
                    // MLA does its own input RMSNorm; only update the residual here.
                    _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.Residual,
                        seqLen * hiddenSize, s);
                }
                else
                {
                    ref readonly var nextLw = ref _weights.Layers[layer + 1];
                    // BitNet carries the residual in FP32 to avoid FP16 overflow.
                    if (fp32Res)
                        _kernels.LaunchFusedAddRmsNormF32Res(_state.ResidualF32, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                            hiddenSize, eps, seqLen, s);
                    else
                        _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                            hiddenSize, eps, seqLen, s);
                }
            }
            else
            {
                // Last processed layer: add residual + FFN output for the final norm.
                if (fp32Res)
                    // Accumulate into the FP32 residual (the final norm reads it directly, avoiding an
                    // FP16 truncation that would overflow for BitNet's large final residual).
                    _kernels.LaunchAddF32F16(_state.ResidualF32, _state.NormOutput, _state.ResidualF32,
                        seqLen * hiddenSize, s);
                else
                    _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                        seqLen * hiddenSize, s);
            }
            MarkProfile(ProfileCategory.Norm);
        }

        // 5. Final RmsNorm (last token only). For BitNet, read the FP32 residual directly so the
        // large final residual is never truncated to FP16.
        if (fp32Res && numLayers > 0)
        {
            nint lastResF32 = _state.ResidualF32 + (nint)((long)(seqLen - 1) * hiddenSize * sizeof(float));
            _kernels.LaunchRmsNormF32InF16W(lastResF32, _weights.OutputNormWeight, _state.NormOutput,
                hiddenSize, eps, 1, s);
        }
        else
        {
            nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);
            _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
                hiddenSize, eps, 1, s);
        }
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

        // Decode CUDA-graph capture for this straight-through body is not used here;
        // dev's ForwardDecodeGraph early-return path handles graph replay separately.
        if (seqLen == 1)
            DecodeGraphState = CudaDecodeGraphState.Off;

        // Capture dispatch-only time (all launches queued, before GPU wait).
        long dispatchEndTs = profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

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

        if (profile)
        {
            long totalEndTs = System.Diagnostics.Stopwatch.GetTimestamp();
            if (_profWarmupSkipped < ProfWarmupSteps)
            {
                _profWarmupSkipped++;
            }
            else
            {
                _profDispatchTicks += dispatchEndTs - dispatchStartTs;
                _profTotalTicks += totalEndTs - dispatchStartTs;
                _profSteps++;
            }
        }

        // 8. D2H copy FP32 logits to CPU UnmanagedTensor
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is null)
            return Forward(tokenIds, positions, deviceId, kvCache);

        if (!ReferenceEquals(_cudaLora?.Source, adapter))
        {
            _cudaLora?.Dispose();
            _cudaLora = CudaLoraWeights.Stage(adapter, Config, _kernels, _stream.Handle);
        }

        _currentAdapter = adapter;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _currentAdapter = null;
        }
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

                nint qPtr, kPtr, vPtr;
                if (lw.QkvPacked != 0)
                {
                    if (_kernels.HasMmq(lw.QkvPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

                if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(qPtr, lw.QBias, lw.QOutputDim, seqLen, s);
                if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(kPtr, lw.KBias, lw.KOutputDim, seqLen, s);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(vPtr, lw.VBias, lw.VOutputDim, seqLen, s);

                if (lw.QNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(qPtr, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
                if (lw.KNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(kPtr, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

                int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
                _kernels.LaunchRoPE(qPtr, kPtr, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);

                // KV-cache update via device-resident position; replaces the eager
                // path's cuMemcpyDtoDAsync (which would bake the dst address).
                kvCache.UpdateDeviceSingleDevicePos(kPtr, vPtr, layer, _decodePosDevice, s, _kernels);

                // Attention with device-resident seq_kv / position_offset.
                _kernels.LaunchAttentionDyn(qPtr, kvCache.GetKeysPtr(layer),
                    kvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, _decodeSeqKvDevice, numHeads, numKvHeads, headDim,
                    _decodePosDevice, slidingWindow, s);

                Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
                if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);

                nint gatePtr, upPtr;
                if (lw.GateUpPacked != 0)
                {
                    if (_kernels.HasMmq(lw.GateUpPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

                _kernels.LaunchSwiGLU(gatePtr, upPtr, _state.SiluOutput, intermediateSize, seqLen, s);

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

                nint qPtr, kPtr, vPtr;
                if (lw.QkvPacked != 0)
                {
                    if (_kernels.HasMmq(lw.QkvPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

                if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(qPtr, lw.QBias, lw.QOutputDim, seqLen, s);
                if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(kPtr, lw.KBias, lw.KOutputDim, seqLen, s);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(vPtr, lw.VBias, lw.VOutputDim, seqLen, s);

                if (lw.QNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(qPtr, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
                if (lw.KNormWeight != 0)
                    _kernels.LaunchPerHeadRmsNorm(kPtr, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

                int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
                _kernels.LaunchRoPE(qPtr, kPtr, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);

                // KV-cache update (FP16 ring write + predicated quantize-on-evict),
                // device-side eviction state.
                kvCache.UpdateDeviceForGraph(kPtr, vPtr, layer, _decodePosDevice, s, _kernels);

                // Dequant the quantized region + scatter the window into the FP16 attention
                // scratch — both predicated, both reading position from _decodePosDevice.
                var (kCachePtr, vCachePtr) =
                    kvCache.PrepareAttentionScratchForGraph(layer, _decodePosDevice, s, _kernels);

                // Attention with device-resident seq_kv / position_offset.
                _kernels.LaunchAttentionDyn(qPtr, kCachePtr, vCachePtr, _state.AttnOutput,
                    seqLen, _decodeSeqKvDevice, numHeads, numKvHeads, headDim,
                    _decodePosDevice, slidingWindow, s);

                Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
                if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);

                nint gatePtr, upPtr;
                if (lw.GateUpPacked != 0)
                {
                    if (_kernels.HasMmq(lw.GateUpPackedQuantType) && !CudaKernels.ForceDirectGemv)
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

                _kernels.LaunchSwiGLU(gatePtr, upPtr, _state.SiluOutput, intermediateSize, seqLen, s);

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

    private static bool ShouldUseHighPrecisionForward(CudaWeights weights)
    {
        if (!EnableHighPrecisionIQuants) return false;
        if (IsIQuant(weights.OutputQuantType))
            return true;

        foreach (ref readonly var lw in weights.Layers.AsSpan())
        {
            if (IsIQuant(lw.QQuantType)
                || IsIQuant(lw.KQuantType)
                || IsIQuant(lw.VQuantType)
                || IsIQuant(lw.OQuantType)
                || IsIQuant(lw.GateQuantType)
                || IsIQuant(lw.UpQuantType)
                || IsIQuant(lw.DownQuantType))
                return true;
        }

        return false;

        // IQ-family detector covers IQ4_NL / IQ4_XS plus the 2-bit IQ2_* family
        // (IQ2_XXS / IQ2_XS / IQ2_S — IQ2_S also stores IQ2_M file-type tensors).
        // Higher precision-loss formats benefit from the dequant→F32→dot fallback.
        static bool IsIQuant(QuantizationType qt) =>
            qt is QuantizationType.IQ4_NL or QuantizationType.IQ4_XS
                or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS or QuantizationType.IQ2_S;
    }

    private ITensor ForwardHighPrecision(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
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
        nint s = _stream.Handle;

        _state.EnsureCapacity(seqLen);

        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        _kernels.LaunchEmbeddingLookupF32(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenStateF32,
            seqLen, hiddenSize, s);

        long hiddenBytesF32 = (long)seqLen * hiddenSize * sizeof(float);
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.ResidualF32, _state.HiddenStateF32,
            (nuint)hiddenBytesF32, s).ThrowOnError();

        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        if (numLayers > 0)
            RmsNormF32WithWeight(_state.ResidualF32, _weights.Layers[0].AttnNormWeight,
                _cpuWeights?.Layers[0].AttnNormWeight,
                _state.NormOutputF32, hiddenSize, eps, seqLen, s);

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            ProjectF32(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutputF32, _state.QF32,
                lw.QOutputDim, lw.QInputDim, seqLen);
            ProjectF32(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutputF32, _state.KF32,
                lw.KOutputDim, lw.KInputDim, seqLen);
            ProjectF32(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutputF32, _state.VF32,
                lw.VOutputDim, lw.VInputDim, seqLen);

            if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF32(_state.QF32, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF32(_state.KF32, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF32(_state.VF32, lw.VBias, lw.VOutputDim, seqLen, s);

            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNormF32(_state.QF32, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNormF32(_state.KF32, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
            _kernels.LaunchRoPEF32(_state.QF32, _state.KF32, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, effectiveRopeType, s);

            _kernels.LaunchAttentionF32(_state.QF32, _state.KF32, _state.VF32, _state.AttnOutputF32,
                seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);

            ProjectF32(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutputF32, _state.NormOutputF32,
                lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAddF32(_state.NormOutputF32, lw.OBias, lw.OOutputDim, seqLen, s);

            _kernels.LaunchAddF32(_state.ResidualF32, _state.NormOutputF32, _state.ResidualF32,
                seqLen * hiddenSize, s);

            RmsNormF32WithWeight(_state.ResidualF32, lw.FfnNormWeight,
                _cpuWeights?.Layers[layer].FfnNormWeight, _state.NormOutputF32,
                hiddenSize, eps, seqLen, s);

            ProjectF32(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutputF32, _state.FfnGateF32,
                lw.GateOutputDim, lw.GateInputDim, seqLen);
            ProjectF32(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutputF32, _state.FfnUpF32,
                lw.UpOutputDim, lw.UpInputDim, seqLen);
            if (lw.GateBias != 0) _kernels.LaunchBiasAddF32(_state.FfnGateF32, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAddF32(_state.FfnUpF32, lw.UpBias, lw.UpOutputDim, seqLen, s);

            _kernels.LaunchSwiGLUF32(_state.FfnGateF32, _state.FfnUpF32, _state.SiluOutputF32,
                intermediateSize, seqLen, s);

            ProjectF32(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutputF32, _state.NormOutputF32,
                lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAddF32(_state.NormOutputF32, lw.DownBias, lw.DownOutputDim, seqLen, s);

            _kernels.LaunchAddF32(_state.ResidualF32, _state.NormOutputF32, _state.ResidualF32,
                seqLen * hiddenSize, s);

            if (layer < numLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                RmsNormF32WithWeight(_state.ResidualF32, nextLw.AttnNormWeight,
                    _cpuWeights?.Layers[layer + 1].AttnNormWeight, _state.NormOutputF32,
                    hiddenSize, eps, seqLen, s);
            }
        }

        nint lastHidden = _state.ResidualF32 + (nint)((seqLen - 1) * hiddenSize * sizeof(float));
        RmsNormF32WithWeight(lastHidden, _weights.OutputNormWeight, _cpuWeights?.OutputNormWeight, _state.NormOutputF32,
            hiddenSize, eps, 1, s);

        _stream.Synchronize();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        if (_cpuWeights is not null)
        {
            float[] normHost = ArrayPool<float>.Shared.Rent(hiddenSize);
            try
            {
                fixed (float* pNorm = normHost)
                {
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)pNorm, _state.NormOutputF32,
                        (nuint)(hiddenSize * sizeof(float))).ThrowOnError();
                    ProjectCpuLmHead(_cpuWeights.OutputWeight, _cpuWeights.OutputQuantType,
                        pNorm, (float*)result.DataPointer, _cpuWeights.OutputOutputDim,
                        _cpuWeights.OutputInputDim);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(normHost);
            }
        }
        else
        {
            // Keep the large LM head on the existing quantized/FP16 projection path:
            // expanding vocab x hidden to F32 would require several GiB of scratch.
            _kernels.LaunchConvertF32ToF16(_state.NormOutputF32, _state.NormOutput, hiddenSize, s);
            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1);
            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
            _stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
                (nuint)(vocabSize * sizeof(float))).ThrowOnError();
        }
        return result;
    }

    /// <summary>
    /// Gemma-4 (DiffusionGemma AR) forward — dedicated F32 path mirroring the CPU
    /// <c>TransformerModel.RunGemma4Layer</c> and the Vulkan
    /// <c>RecordGemma4Attention</c>/<c>RecordGemma4Ffn</c>. Cacheless (autoregressive,
    /// single-shot / short-context). Covers per-layer dual head dim/KV/rope, V-from-K
    /// (copy K→V, skip V matmul), per-head QK-norm, weight-less V-norm, attn scale 1.0
    /// (Q pre-scaled by sqrt(headDim) so the kernel's 1/sqrt(headDim) cancels), partial/
    /// dual RoPE, dual parallel dense-GeGLU + MoE-GeGLU FFN with the five norms, the
    /// custom router (1/√H folded into RouterScale), per-expert down scale (folded into
    /// the F32 down bank at load), layer_output_scale, and the final-logit softcap.
    /// </summary>
    private ITensor ForwardGemma4(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
        if (!_kernels.HasGemma4Kernels)
            throw new InvalidOperationException(
                "Gemma4 F32 helper kernels not available. Compile native/kernels/gemma4_f32.cu to "
                + "PTX (native/build.{sh,ps1}) and ensure gemma4_f32.ptx ships under runtimes/.../ptx.");
        if (!_kernels.HasMoeKernels)
            throw new InvalidOperationException(
                "MoE kernels not available (required for the gemma4 expert path). Compile "
                + "native/kernels/moe_ffn.cu to PTX.");

        _context.MakeCurrent();
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        nint s = _stream.Handle;

        _state.EnsureCapacity(seqLen);
        EnsureGemma4Staging(seqLen, hiddenSize);
        _moeScratch ??= new CudaMoeScratch();

        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // Embedding lookup (× sqrt(hidden) baked into the GGUF EmbeddingScale path
        // — gemma ties + scales the embed). LaunchEmbeddingLookupF32 applies the
        // tok_embd table directly; the sqrt(hidden) scale is applied here.
        _kernels.LaunchEmbeddingLookupF32(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenStateF32,
            seqLen, hiddenSize, s);
        float embedScale = Config.EmbeddingScale ?? 1.0f;
        if (embedScale != 1.0f)
            _kernels.LaunchScaleInplaceF32(_state.HiddenStateF32, seqLen * hiddenSize, embedScale, s);

        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            var g4 = _weights.Gemma4Layers![layer]!;
            var moeW = _weights.MoeLayers![layer]!;

            // ResidualF32 holds the layer input (== HiddenStateF32 at entry).
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.ResidualF32, _state.HiddenStateF32,
                (nuint)((long)seqLen * hiddenSize * sizeof(float)), s).ThrowOnError();

            RunGemma4AttentionF32(layer, in lw, g4, seqLen, eps, s);
            // attn_out = post_attention_norm(O) + residual ; leaves attn_out in HiddenStateF32.
            _kernels.LaunchRmsNormF32(_state.NormOutputF32, g4.PostAttnNorm, _state.NormOutputF32,
                hiddenSize, eps, seqLen, s);
            _kernels.LaunchAddF32(_state.ResidualF32, _state.NormOutputF32, _state.HiddenStateF32,
                seqLen * hiddenSize, s);

            RunGemma4FfnF32(layer, in lw, g4, moeW, seqLen, eps, s);
        }

        // Final RMSNorm (last token only) + LM head + softcap.
        nint lastHidden = _state.HiddenStateF32 + (nint)((long)(seqLen - 1) * hiddenSize * sizeof(float));
        RmsNormF32WithWeight(lastHidden, _weights.OutputNormWeight, _cpuWeights?.OutputNormWeight,
            _state.NormOutputF32, hiddenSize, eps, 1, s);

        _stream.Synchronize();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        float softcap = _gemma4FinalSoftcap;

        if (_cpuWeights is not null)
        {
            float[] normHost = ArrayPool<float>.Shared.Rent(hiddenSize);
            try
            {
                fixed (float* pNorm = normHost)
                {
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)pNorm, _state.NormOutputF32,
                        (nuint)(hiddenSize * sizeof(float))).ThrowOnError();
                    ProjectCpuLmHead(_cpuWeights.OutputWeight, _cpuWeights.OutputQuantType,
                        pNorm, (float*)result.DataPointer, _cpuWeights.OutputOutputDim,
                        _cpuWeights.OutputInputDim);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(normHost);
            }

            // Final-logit soft-capping on the host output: c * tanh(x / c).
            if (softcap > 0f)
            {
                float* logits = (float*)result.DataPointer;
                float inv = 1.0f / softcap;
                for (int i = 0; i < vocabSize; i++)
                    logits[i] = softcap * MathF.Tanh(logits[i] * inv);
            }
        }
        else
        {
            _kernels.LaunchConvertF32ToF16(_state.NormOutputF32, _state.NormOutput, hiddenSize, s);
            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1);
            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
            if (softcap > 0f)
                _kernels.LaunchSoftcapInplaceF32(_state.LogitsF32, vocabSize, softcap, s);
            _stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
                (nuint)(vocabSize * sizeof(float))).ThrowOnError();
        }
        return result;
    }

    /// <summary>
    /// Gemma-4 attention (F32). Reads the layer input from <c>ResidualF32</c>,
    /// writes the o_proj output into <c>NormOutputF32</c> (the shared
    /// post-attention norm + residual run in the caller). Per-layer dual head dim /
    /// KV / rope, V-from-K, per-head QK-norm, weight-less V-norm, scale 1.0.
    /// </summary>
    private void RunGemma4AttentionF32(int layer, in CudaLayerWeights lw,
        CudaGemma4LayerWeights g4, int seqLen, float eps, nint s)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int headDim = Config.GetLayerHeadDim(layer);
        int numKvHeads = Config.NumGlobalKvHeads is int gk && Config.IsFullAttentionLayer(layer)
            ? gk : Config.NumKvHeads;

        // attn_norm(input) → NormOutputF32
        _kernels.LaunchRmsNormF32(_state.ResidualF32, g4.AttnNorm, _state.NormOutputF32,
            hiddenSize, eps, seqLen, s);

        // Q, K projections (raw — K captured before k-norm/rope for V-from-K).
        ProjectF32Gemma4(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutputF32, _state.QF32,
            lw.QOutputDim, lw.QInputDim, seqLen);
        ProjectF32Gemma4(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutputF32, _state.KF32,
            lw.KOutputDim, lw.KInputDim, seqLen);

        // V branch: V-from-K (global, V-less) copies the raw K projection into V;
        // else V = wv · normIn.
        if (g4.VFromK)
        {
            long kvBytes = (long)seqLen * numKvHeads * headDim * sizeof(float);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.VF32, _state.KF32, (nuint)kvBytes, s).ThrowOnError();
        }
        else
        {
            ProjectF32Gemma4(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutputF32, _state.VF32,
                lw.VOutputDim, lw.VInputDim, seqLen);
        }

        // Per-head Q/K RMSNorm (× learned weight); weight-less V RMSNorm (unit gamma).
        _kernels.LaunchPerHeadRmsNormF32(_state.QF32, g4.QNorm, eps, numHeads, headDim, seqLen, s);
        _kernels.LaunchPerHeadRmsNormF32(_state.KF32, g4.KNorm, eps, numKvHeads, headDim, seqLen, s);
        _kernels.LaunchRmsNormWeightlessF32(_state.VF32, _state.VF32, headDim, eps,
            seqLen * numKvHeads, s);

        // RoPE on Q and K (V not roped). Global layers: partial NeoX (pair (i, i+headDim/2),
        // freq base over the full head dim). Sliding layers: full NeoX rotation.
        if (Config.IsFullAttentionLayer(layer) && _gemma4GlobalRotatedPairs > 0
            && Config.GlobalHeadDim is int)
        {
            _kernels.LaunchRoPEF32PartialNeoX(_state.QF32, _state.KF32, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _gemma4GlobalRotatedPairs,
                _gemma4GlobalRopeTheta, s);
        }
        else
        {
            // Sliding layers use the primary (sliding) rope schedule, full NeoX.
            _kernels.LaunchRoPEF32(_state.QF32, _state.KF32, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta,
                CudaKernels.ToCudaRopeType(RoPEType.NeoX), s);
        }

        // Attention scale = 1.0 (q/k-norm make Q,K unit). The F32 attention kernel
        // hardcodes 1/sqrt(headDim); pre-scale Q by sqrt(headDim) so it cancels.
        _kernels.LaunchScaleInplaceF32(_state.QF32, seqLen * numHeads * headDim,
            MathF.Sqrt((float)headDim), s);

        int slidingWindow = GetGemmaLayerSlidingWindow(layer);
        _kernels.LaunchAttentionF32(_state.QF32, _state.KF32, _state.VF32, _state.AttnOutputF32,
            seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);

        // o_proj → NormOutputF32.
        ProjectF32Gemma4(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutputF32, _state.NormOutputF32,
            lw.OOutputDim, lw.OInputDim, seqLen);
    }

    /// <summary>
    /// Gemma-4 dual parallel FFN (F32). Reads attn_out from <c>HiddenStateF32</c>;
    /// runs the dense GeGLU branch and the MoE GeGLU branch, combines them
    /// (rms(dense+moe)·post_ffw_norm + attn_out), then applies layer_output_scale.
    /// Leaves the layer output in <c>HiddenStateF32</c>.
    /// </summary>
    private void RunGemma4FfnF32(int layer, in CudaLayerWeights lw,
        CudaGemma4LayerWeights g4, CudaMoeLayerWeights moeW, int seqLen, float eps, nint s)
    {
        int hiddenSize = Config.HiddenSize;
        int denseInterm = lw.GateOutputDim; // dense ("shared expert") FF width

        // ── Dense branch → Gemma4DenseF32 ──
        // cur_mlp = rms(attn_out)*ffn_norm ; geglu(gate, up) ; down ; rms*post_ffw_norm_1
        _kernels.LaunchRmsNormF32(_state.HiddenStateF32, g4.FfnNorm, _state.NormOutputF32,
            hiddenSize, eps, seqLen, s);
        ProjectF32Gemma4(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutputF32, _state.FfnGateF32,
            lw.GateOutputDim, lw.GateInputDim, seqLen);
        ProjectF32Gemma4(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutputF32, _state.FfnUpF32,
            lw.UpOutputDim, lw.UpInputDim, seqLen);
        _kernels.LaunchGeGLUTanhF32(_state.FfnGateF32, _state.FfnUpF32, _state.SiluOutputF32,
            denseInterm, seqLen, s);
        ProjectF32Gemma4(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutputF32, _gemma4DenseF32,
            lw.DownOutputDim, lw.DownInputDim, seqLen);
        _kernels.LaunchRmsNormF32(_gemma4DenseF32, g4.PostFfwNorm1, _gemma4DenseF32,
            hiddenSize, eps, seqLen, s);

        // ── MoE branch → Gemma4MoeF32 ──
        // Expert input = rms(attn_out)*pre_ffw_norm_2 (into the shared MoE staging buffer).
        _kernels.LaunchRmsNormF32(_state.HiddenStateF32, g4.PreFfwNorm2, _moeStagingInF32,
            hiddenSize, eps, seqLen, s);
        // Custom-router input = rms(attn_out)*RouterScale (RouterScale carries 1/√H).
        _kernels.LaunchRmsNormF32(_state.HiddenStateF32, g4.RouterScaleDevice, _gemma4RouterInF32,
            hiddenSize, eps, seqLen, s);
        // Gemma-4 MoE: custom router (separate router input) + GeGLU experts +
        // per-expert down scale (pre-folded into the F32 down bank) + renorm clamp.
        CudaGemma4Ffn.ForwardMoe(
            expertInF32: _moeStagingInF32,
            routerInF32: _gemma4RouterInF32,
            outputF32: _gemma4MoeF32,
            seqLen: seqLen,
            weights: moeW,
            scratch: _moeScratch!,
            cublasHandle: _cublas.Handle,
            kernels: _kernels,
            stream: s);
        _kernels.LaunchRmsNormF32(_gemma4MoeF32, g4.PostFfwNorm2, _gemma4MoeF32,
            hiddenSize, eps, seqLen, s);

        // ── Combine: cur = rms(dense + moe)*post_ffw_norm + attn_out, then ×layer_output_scale ──
        _kernels.LaunchAddF32(_gemma4DenseF32, _gemma4MoeF32, _state.NormOutputF32,
            seqLen * hiddenSize, s);
        _kernels.LaunchRmsNormF32(_state.NormOutputF32, g4.PostFfwNorm, _state.NormOutputF32,
            hiddenSize, eps, seqLen, s);
        _kernels.LaunchAddF32(_state.HiddenStateF32, _state.NormOutputF32, _state.HiddenStateF32,
            seqLen * hiddenSize, s);
        if (g4.LayerOutputScale != 1.0f)
            _kernels.LaunchScaleInplaceF32(_state.HiddenStateF32, seqLen * hiddenSize,
                g4.LayerOutputScale, s);
    }

    private int GetGemmaLayerSlidingWindow(int layer)
    {
        var perLayer = Config.PerLayerSlidingWindow;
        if (perLayer is not null && (uint)layer < (uint)perLayer.Count)
            return perLayer[layer] ?? 0;
        return Config.SlidingWindowSize ?? 0;
    }

    private static void ProjectCpuLmHead(nint weights, QuantizationType qt, float* input, float* output,
                                          int outputDim, int inputDim)
    {
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(inputDim, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(inputDim);
        try
        {
            var row = rowBuf.AsSpan(0, inputDim);
            var x = new ReadOnlySpan<float>(input, inputDim);
            for (int i = 0; i < outputDim; i++)
            {
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32(weights + i * (nint)rowBytes,
                    inputDim, qt, row);
                output[i] = TensorPrimitives.Dot(row, x);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    private void RmsNormF32WithWeight(nint inputF32, nint weightF16, float[]? hostWeightF32, nint outputF32,
                                      int hiddenSize, float eps, int rows, nint stream)
    {
        if (hostWeightF32 is not null)
        {
            fixed (float* pWeight = hostWeightF32)
            {
                CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.DequantScratchF32, (nint)pWeight,
                    (nuint)(hiddenSize * sizeof(float)), stream).ThrowOnError();
            }
        }
        else
        {
            _kernels.LaunchConvertF16ToF32(weightF16, _state.DequantScratchF32, hiddenSize, stream);
        }
        _kernels.LaunchRmsNormF32(inputF32, _state.DequantScratchF32, outputF32,
            hiddenSize, eps, rows, stream);
    }

    /// <summary>
    /// Ensures the gemma4 activation round-trip scratch holds at least
    /// <paramref name="elems"/> F32 values.
    /// </summary>
    private void EnsureGemma4ActScratch(int elems)
    {
        if (_gemma4ActScratchElems >= elems) return;
        if (_gemma4ActScratchF32 != 0) CudaDriverApi.cuMemFree_v2(_gemma4ActScratchF32);
        CudaDriverApi.cuMemAlloc_v2(out _gemma4ActScratchF32, (nuint)((long)elems * sizeof(float))).ThrowOnError();
        _gemma4ActScratchElems = elems;
    }

    /// <summary>
    /// Gemma4 projection that reproduces the CPU oracle's per-projection ACTIVATION
    /// quantization. The CPU <c>TransformerModel.Gemm</c> for Q8_0 weights quantizes
    /// the F32 activation to Q8_0 before the int8 dot; the F32 cuBLAS path here would
    /// otherwise keep the activation in full precision and drift enough that a logit
    /// pokes over the CPU-parity tolerance. For Q8_0 weights we copy the input to
    /// scratch, round-trip it through Q8_0 (FP16 scale, round-nearest-even), then
    /// GEMM — within reduction-order of the CPU. Other quant types fall through to
    /// the plain F32 projection (their CPU activation-quant round-trips can be added
    /// if a future fixture's experts push the worst logit over tolerance).
    /// </summary>
    private void ProjectF32Gemma4(nint quantWeight, QuantizationType qt, nint fp16Weight,
                                   nint inputF32, nint outputF32, int outputDim, int inputDim, int seqLen)
    {
        if (qt == QuantizationType.Q8_0 && (inputDim & 31) == 0)
        {
            nint s = _stream.Handle;
            int elems = seqLen * inputDim;
            EnsureGemma4ActScratch(elems);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_gemma4ActScratchF32, inputF32,
                (nuint)((long)elems * sizeof(float)), s).ThrowOnError();
            _kernels.LaunchQuantizeActivationQ8_0RoundtripF32(_gemma4ActScratchF32, inputDim, seqLen, s);
            ProjectF32(quantWeight, qt, fp16Weight, _gemma4ActScratchF32, outputF32, outputDim, inputDim, seqLen);
            return;
        }
        ProjectF32(quantWeight, qt, fp16Weight, inputF32, outputF32, outputDim, inputDim, seqLen);
    }

    private void ProjectF32(nint quantWeight, QuantizationType qt, nint fp16Weight,
                             nint inputF32, nint outputF32, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;
        nint weightF32 = _state.DequantScratchF32;

        if (quantWeight != 0)
        {
            if (qt is QuantizationType.IQ4_NL or QuantizationType.IQ4_XS or QuantizationType.Q5_K
                    or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS or QuantizationType.IQ2_S)
            {
                _kernels.LaunchDequantToF32(quantWeight, qt, weightF32,
                    outputDim * inputDim, s);
            }
            else
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                _kernels.LaunchConvertF16ToF32(_state.DequantScratch, weightF32,
                    outputDim * inputDim, s);
            }
        }
        else
        {
            _kernels.LaunchConvertF16ToF32(fp16Weight, weightF32,
                outputDim * inputDim, s);
        }

        CudaGemm.LinearF32(_cublas.Handle, inputF32, weightF32, outputF32,
            seqLen, inputDim, outputDim, s);
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
                if (qt == QuantizationType.I2_S)
                    _kernels.LaunchDequantI2_SToF16(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                else
                    _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                        outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (CanUseI2SA8Project(qt, seqLen, outputDim, inputDim)) // Decode: I2_S W2A8 GEMV
        {
            _kernels.LaunchQuantizeF16ToI8AbsMax(input, _state.A8Input, _state.A8InvScale, inputDim, s);
            _kernels.LaunchI2_SGemvA8DeviceScale(
                quantWeight, _state.A8Input, _state.A8OutputF32, outputDim, inputDim, _state.A8InvScale, s);
            _kernels.LaunchConvertF32ToF16(_state.A8OutputF32, output, outputDim, s);
        }
        else if (quantWeight != 0 && qt == QuantizationType.I2_S) // Decode: I2_S ternary GEMV
        {
            _kernels.LaunchI2_SGemvF16In(quantWeight, input, output, outputDim, inputDim, s);
        }
        else if (quantWeight != 0 && _kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
        {
            // Decode: MMQ-style fused dequant+matmul (dp4a) — faster than the FP fmuladd kernel.
            // Routes Q4_K, Q5_K, Q6_K through the dp4a path; the rest fall through to the
            // legacy FP-fmuladd kernel below. Use the pre-Q8_1 scratch when available and
            // when inputDim fits — eliminates per-block redundant Stage 1 work
            // (especially material for the MMVQ-large variant).
            nint scratch = MaybePreQuantize(input, inputDim, s);
            _kernels.LaunchQuantizedGemvMmq(quantWeight, qt, input, output, outputDim, inputDim, scratch, s);
        }
        else if (quantWeight != 0 && _kernels.HasQuantizedGemvKernel(qt)) // Decode: quantized GEMV
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
    /// Applies the LoRA delta for one (layer, proj) site on device:
    /// <c>yDev += scale · (A[outputDim, rank] · (B[rank, inputDim] · xDev))</c>.
    /// All operands are FP16 on device; intermediate tmp is <see cref="CudaForwardState.LoraTmp"/>.
    /// For decode (seqLen == 1): two GEMVs via cuBLAS.
    /// For prefill (seqLen &gt; 1): two batched GEMMs via cuBLAS.
    /// Early-returns without error when <paramref name="layer"/>/<paramref name="proj"/> is
    /// not covered by the staged adapter.
    /// </summary>
    /// <param name="layer">Zero-based transformer layer index.</param>
    /// <param name="proj">Canonical projection name (e.g. <c>q_proj</c>).</param>
    /// <param name="xDev">Device pointer to the FP16 input [seqLen, inputDim].</param>
    /// <param name="yDev">Device pointer to the FP16 output [seqLen, outputDim] to accumulate into.</param>
    /// <param name="seqLen">Number of tokens in the current step (1 = decode, &gt;1 = prefill).</param>
    private void ApplyLoraDeltaDevice(int layer, string proj, nint xDev, nint yDev, int seqLen)
    {
        if (_cudaLora is null)
            return;
        if (!_cudaLora.TryGet(layer, proj, out nint aF16, out nint bF16, out int inputDim, out int outputDim))
            return;

        int rank = _cudaLora.Rank;
        if (rank > CudaForwardState.MaxLoraRank)
            throw new InvalidOperationException(
                $"LoRA rank {rank} exceeds the device delta rank cap ({CudaForwardState.MaxLoraRank}). " +
                $"Reduce adapter rank or rebuild with a higher MaxLoraRank.");

        float scale = _cudaLora.Scale;
        nint tmp = _state.LoraTmp;  // [seqLen, MaxLoraRank] FP16
        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        if (seqLen == 1)
        {
            // Decode path — two GEMVs.
            // Step 1: tmp[rank] = B[rank, inputDim] · x[inputDim]  (overwrites tmp)
            CudaGemm.GemvF16(cublasH, bF16, xDev, tmp, rank, inputDim, s);

            // Step 2: y[outputDim] += scale · A[outputDim, rank] · tmp[rank]
            CudaGemm.GemvF16Accum(cublasH, aF16, tmp, yDev, outputDim, rank, scale, s);
        }
        else
        {
            // Prefill path — two batched GEMMs.
            // Step 1: tmp[seqLen, rank] = X[seqLen, inputDim] × B^T  (overwrites tmp)
            CudaGemm.LinearF16(cublasH, xDev, bF16, tmp, seqLen, inputDim, rank, s);

            // Step 2: Y[seqLen, outputDim] += scale · tmp[seqLen, rank] × A^T
            CudaGemm.LinearF16Accum(cublasH, tmp, aF16, yDev, seqLen, rank, outputDim, scale, s);
        }
    }

    private static bool CanFuseI2SDecode(
        int seqLen,
        QuantizationType qt0, QuantizationType qt1,
        int inputDim0, int inputDim1)
        => seqLen == 1
           && qt0 == QuantizationType.I2_S
           && qt1 == QuantizationType.I2_S
           && inputDim0 == inputDim1;

    private static bool CanFuseI2SDecode(
        int seqLen,
        QuantizationType qt0, QuantizationType qt1, QuantizationType qt2,
        int inputDim0, int inputDim1, int inputDim2)
        => seqLen == 1
           && qt0 == QuantizationType.I2_S
           && qt1 == QuantizationType.I2_S
           && qt2 == QuantizationType.I2_S
           && inputDim0 == inputDim1
           && inputDim0 == inputDim2;

    private static bool CanFuseI2SNormDecode(
        int seqLen,
        nint normWeight,
        QuantizationType qt,
        int inputDim,
        int normDim)
        => seqLen == 1
           && normWeight != 0
           && qt == QuantizationType.I2_S
           && inputDim == normDim;

    private bool CanUseI2SA8Project(QuantizationType qt, int seqLen, int outputDim, int inputDim)
        => s_i2sA8Decode
           && seqLen == 1
           && qt == QuantizationType.I2_S
           && Config.Architecture == Architecture.BitNet
           && outputDim != Config.VocabSize
           && inputDim <= Math.Max(Config.HiddenSize, Config.IntermediateSize)
           && outputDim <= Math.Max(Config.HiddenSize, Config.IntermediateSize);

    /// <summary>
    /// Creates a <see cref="CudaKvCache"/> for this model.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for the cache.</param>
    public CudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new CudaKvCache(Core.Attention.KvGeometry.FromConfig(Config), maxSeqLen);
    }

    /// <summary>
    /// Creates a KV-cache with optional quantization for this model.
    /// Returns <see cref="CudaQuantizedKvCache"/> when quantization is configured,
    /// otherwise a standard <see cref="CudaKvCache"/>.
    /// </summary>
    public Core.Attention.IKvCache CreateKvCache(int maxSeqLen, Core.Configuration.KvCacheConfig config)
    {
        _context.MakeCurrent();
        var geom = Core.Attention.KvGeometry.FromConfig(Config);
        if (!config.IsQuantized)
            return new CudaKvCache(geom, maxSeqLen);
        return new CudaQuantizedKvCache(geom, maxSeqLen, config);
    }

    /// <summary>
    /// Creates a GPU-resident TurboQuant (MSE-stage) KV-cache on this model's CUDA context. The caller
    /// supplies the codec constants (the Cuda project does not depend on the Engine codec):
    /// <paramref name="centroids"/> (2^mseBits, scaled by 1/√d), per-K/V RHT sign sets (length headDim,
    /// ±1), and <paramref name="invSqrtD"/>. Uniform geometry, headDim a power of two ≤ 256; eager decode.
    /// </summary>
    public CudaTurboQuantKvCache CreateTurboQuantKvCache(
        int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
    {
        _context.MakeCurrent();
        return new CudaTurboQuantKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim,
            maxSeqLen, mseBits, centroids, signsK, signsV, invSqrtD);
    }

    /// <summary>
    /// Prints mean CPU-side dispatch ms/token vs total ms/token over the profiled
    /// steady-state decode steps. No-op unless DOTLLM_PROFILE_LAUNCH=1.
    /// </summary>
    private void ReportLaunchProfile()
    {
        if (!s_profileLaunch || _profSteps == 0)
            return;

        double freq = System.Diagnostics.Stopwatch.Frequency;
        double dispatchMs = (_profDispatchTicks / freq) * 1000.0 / _profSteps;
        double totalMs = (_profTotalTicks / freq) * 1000.0 / _profSteps;
        double ratio = totalMs > 0 ? dispatchMs / totalMs : 0;
        Console.Error.WriteLine(
            $"[DOTLLM_PROFILE_LAUNCH] decode steps={_profSteps} " +
            $"dispatch={dispatchMs:F3} ms/token  total={totalMs:F3} ms/token  " +
            $"ratio(dispatch/total)={ratio:F3}");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        ReportLaunchProfile();
        _cudaLora?.Dispose();
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
        if (_gemma4DenseF32 != 0) { CudaDriverApi.cuMemFree_v2(_gemma4DenseF32); _gemma4DenseF32 = 0; }
        if (_gemma4MoeF32 != 0) { CudaDriverApi.cuMemFree_v2(_gemma4MoeF32); _gemma4MoeF32 = 0; }
        if (_gemma4RouterInF32 != 0) { CudaDriverApi.cuMemFree_v2(_gemma4RouterInF32); _gemma4RouterInF32 = 0; }
        if (_gemma4ActScratchF32 != 0) { CudaDriverApi.cuMemFree_v2(_gemma4ActScratchF32); _gemma4ActScratchF32 = 0; }
        _mlaScratchF16?.Dispose();
        _mlaKvCache?.Dispose();
        _moeScratch?.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _cpuWeights?.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
