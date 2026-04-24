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

    private nint _evtStart;
    private nint _evtEnd;

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

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
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

        // 3. Layer 0 setup: copy hidden→residual, RmsNorm→NormOutput
        long hiddenBytes = (long)seqLen * hiddenSize * h;
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
            hiddenSize, eps, seqLen, s);
        MarkProfile(ProfileCategory.Norm);

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

            // ── ATTENTION BLOCK (NormOutput has normalized input) ──

            // Q/K/V projections: prefill → cuBLAS HGEMM, decode → quantized GEMV
            Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);
            MarkProfile(ProfileCategory.QkvProj);

            // Optional biases (FP16)
            if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

            // Optional QK-norms (FP16)
            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            // RoPE (FP16, in-place on Q and K)
            int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
            _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim,
                _ropeDim, _ropeTheta, effectiveRopeType, s);
            MarkProfile(ProfileCategory.RopeAndExtras);

            // KV-cache update + Attention (FP16)
            if (kvCache is CudaQuantizedKvCache cudaQKvCache)
            {
                cudaQKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s, _kernels);
                int seqKv = cudaQKvCache.CurrentLength;

                // Dequant quantized region + copy window → scratch, then regular attention
                var (kPtr, vPtr) = cudaQKvCache.PrepareAttentionScratch(layer, s, _kernels);
                MarkProfile(ProfileCategory.KvUpdate);
                _kernels.LaunchAttention(_state.Q, kPtr, vPtr, _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else if (kvCache is CudaKvCache cudaKvCache)
            {
                cudaKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = cudaKvCache.CurrentLength;
                MarkProfile(ProfileCategory.KvUpdate);

                _kernels.LaunchAttention(_state.Q, cudaKvCache.GetKeysPtr(layer),
                    cudaKvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else
            {
                MarkProfile(ProfileCategory.KvUpdate);
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim,
                    0, slidingWindow, s);
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

            // ── FFN BLOCK (NormOutput has FFN-normalized input) ──

            // Gate/Up projections
            Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.MlpUp);

            // SwiGLU (FP16)
            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                intermediateSize, seqLen, s);
            MarkProfile(ProfileCategory.Swiglu);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);
            MarkProfile(ProfileCategory.MlpDown);

            // ── FUSED: FFN residual + next layer's attention norm ──
            if (layer < numLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
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
        if (_evtStart != 0) CudaDriverApi.cuEventDestroy_v2(_evtStart);
        if (_evtEnd != 0) CudaDriverApi.cuEventDestroy_v2(_evtEnd);
        if (_profEvents != null)
        {
            for (int i = 0; i < _profEvents.Length; i++)
                if (_profEvents[i] != 0) CudaDriverApi.cuEventDestroy_v2(_profEvents[i]);
        }
        _state.Dispose();
        _weights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
