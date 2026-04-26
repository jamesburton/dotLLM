using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

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
    private readonly GgufFile _gguf;
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

    // ── per-category profiling enum (kept so cherry-picked MarkProfile calls compile) ──
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
        CudaKernels kernels, GgufFile gguf, int deviceId,
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

        // Initialize CUDA
        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        // Resolve PTX directory
        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        // Check VRAM before loading — warn if model likely exceeds available memory.
        // Estimate: sum of quantized byte sizes for all GGUF tensors.
        long estimatedWeightBytes = 0;
        foreach (var t in gguf.TensorsByName.Values)
        {
            int innerDim = t.Shape[0];
            long outerDim = (long)t.Shape.ElementCount / innerDim;
            estimatedWeightBytes += Cpu.Kernels.Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
        }

        string? vramWarning = null;
        if (CudaDriverApi.cuMemGetInfo_v2(out nuint freeBefore, out nuint totalVram) == 0
            && totalVram > 0 && estimatedWeightBytes > (long)freeBefore)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long freeMb = (long)freeBefore / (1024 * 1024);
            long totalMb = (long)totalVram / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed available VRAM ({freeMb}/{totalMb} MB free). " +
                          $"Performance will be degraded due to PCIe memory paging. " +
                          $"Consider a smaller model or quantization format.";
        }

        // Upload weights to GPU
        var weights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle);

        // Create scratch buffers
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

    // Placeholder: profiling + graph-capture infrastructure does not exist in
    // this PR's base (it lands separately). MarkProfile is a no-op so the
    // cherry-picked MLA / MoE dispatch keeps its call sites intact.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void MarkProfile(ProfileCategory cat) { _ = cat; }

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

        // Note: CUDA Graphs decode fast-path + profiling lands in a separate
        // PR (cuda-infra). MLA / MoE would not be graph-capable anyway because
        // MoE does host-side bucketing and MLA's absorbed kernel uses dynamic
        // shmem — both fall through to the eager path here.

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

                // Residual + FfnNorm. MLA's FfnNormWeight comes from CudaMlaLayerWeights
                // (uploaded by CudaMlaWeightsLoader.LoadLayerF16). Same kernel as the GQA
                // path; only the norm-weight pointer differs.
                _kernels.LaunchFusedAddRmsNorm(
                    _state.Residual, _state.NormOutput, mlaLayer.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
                MarkProfile(ProfileCategory.Norm);

                goto FfnBlock;
            }

            // Q/K/V projections: prefill → cuBLAS HGEMM, decode → quantized GEMV
            Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

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

            // KV-cache update + Attention (FP16)
            if (kvCache is CudaQuantizedKvCache cudaQKvCache)
            {
                cudaQKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s, _kernels);
                int seqKv = cudaQKvCache.CurrentLength;

                // Dequant quantized region + copy window → scratch, then regular attention
                var (kPtr, vPtr) = cudaQKvCache.PrepareAttentionScratch(layer, s, _kernels);
                _kernels.LaunchAttention(_state.Q, kPtr, vPtr, _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else if (kvCache is CudaKvCache cudaKvCache)
            {
                cudaKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = cudaKvCache.CurrentLength;

                _kernels.LaunchAttention(_state.Q, cudaKvCache.GetKeysPtr(layer),
                    cudaKvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim,
                    0, slidingWindow, s);
            }

            // O projection → NormOutput
            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            // ── FUSED: attention residual + FFN norm ──
            // residual = residual + NormOutput (via FP32), NormOutput = rmsnorm(new_residual, ffnNormWeight)
            _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

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

            // Gate/Up projections
            Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            // SwiGLU (FP16)
            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                intermediateSize, seqLen, s);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

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
        }

        // 5. Final RmsNorm (last token only)
        nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);
        _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
            hiddenSize, eps, 1, s);

        // 6. LM head (last token only) → FP16 logits, then convert to FP32
        Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
            _state.NormOutput, _state.LogitsF16,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);
        _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);

        // 7. Stream sync (single sync point for entire forward pass)
        _stream.Synchronize();

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
