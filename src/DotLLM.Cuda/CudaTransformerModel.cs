using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
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
    private static readonly bool s_cudaGraphDecode =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_GRAPH") != "0";
    private static readonly bool s_i2sA8Decode =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_I2S_A8") == "1";
    private CudaDecodeGraph? _decodeGraph;
    private CudaKvCache? _decodeGraphCache;
    private bool _decodeGraphDisabled;

    // LoRA adapter staging — set by the 5-arg Forward overload.
#pragma warning disable CS0414 // TODO(Task 3): _currentAdapter will be read in the layer loop when delta application lands.
    private ILoraAdapter? _currentAdapter;
#pragma warning restore CS0414
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

        // Create scratch buffers. BitNet's residual stream exceeds FP16 range in deep layers,
        // so it carries the residual in FP32 (overflow→NaN otherwise).
        bool useFp32Residual = config.Architecture == Architecture.BitNet;
        var state = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize, useFp32Residual);

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

        // Profiling is only meaningful for the single-token decode step.
        bool profile = s_profileLaunch && seqLen == 1;
        long dispatchStartTs = profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        _state.EnsureCapacity(seqLen);

        // 1. Upload tokenIds + positions to device
        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        bool graphEligible = CanUseDecodeGraph(seqLen, kvCache);
        bool graphLaunched = false;
        bool graphCapture = false;
        bool graphReplayCompatible = false;

        if (graphEligible)
        {
            var cudaKvCache = (CudaKvCache)kvCache!;
            if (!ReferenceEquals(_decodeGraphCache, cudaKvCache))
            {
                _decodeGraph?.Dispose();
                _decodeGraph = null;
                _decodeGraphCache = cudaKvCache;
            }

            if (_decodeGraph?.IsCaptured == true)
            {
                _decodeGraph.Launch(s);
                graphLaunched = true;
            }
        }

        if (!graphLaunched)
        {
        DispatchAgain:
            graphCapture = false;
            graphReplayCompatible = false;
            if (graphEligible && !_decodeGraphDisabled && _decodeGraph?.IsCaptured != true)
            {
                _decodeGraph ??= new CudaDecodeGraph();
                try
                {
                    _decodeGraph.Begin(s);
                    graphCapture = true;
                    graphReplayCompatible = true;
                }
                catch (CudaException)
                {
                    _decodeGraph.Dispose();
                    _decodeGraph = null;
                    _decodeGraphDisabled = true;
                }
            }

            try
            {

        // 2. Embedding lookup → FP16 HiddenState
        _kernels.LaunchEmbeddingLookup(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState,
            seqLen, hiddenSize, s);

        // FP32 residual stream for BitNet (its residual magnitude exceeds FP16's ~65504 ceiling).
        bool fp32Res = _state.ResidualF32 != 0;

        // 3. Layer 0 setup: seed residual from embedding, RmsNorm→NormOutput
        long hiddenBytes = (long)seqLen * hiddenSize * h;
        if (fp32Res)
            _kernels.LaunchCopyF16ToF32(_state.HiddenState, _state.ResidualF32, seqLen * hiddenSize, s);
        else
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
            hiddenSize, eps, seqLen, s);

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

            // ── ATTENTION BLOCK (NormOutput has normalized input) ──

            // Q/K/V projections: prefill → cuBLAS HGEMM, decode → quantized GEMV
            if (!s_i2sA8Decode && CanFuseI2SDecode(seqLen, lw.QQuantType, lw.KQuantType, lw.VQuantType,
                    lw.QInputDim, lw.KInputDim, lw.VInputDim))
            {
                _kernels.LaunchI2_SGemv3F16In(
                    lw.QQuant, lw.KQuant, lw.VQuant, _state.NormOutput,
                    _state.Q, _state.K, _state.V,
                    lw.QOutputDim, lw.KOutputDim, lw.VOutputDim, lw.QInputDim, s);
            }
            else
            {
                Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
                Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);
            }

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
                if (seqLen == 1)
                    cudaKvCache.UpdateDevicePositioned(
                        _state.K, _state.V, positions, seqLen, layer, s, _kernels, _state.PositionsDevice);
                else
                    cudaKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = graphReplayCompatible && seqLen == 1
                    ? cudaKvCache.MaxLength
                    : cudaKvCache.CurrentLength;

                if (seqLen == 1)
                    _kernels.LaunchAttentionPos(_state.Q, cudaKvCache.GetKeysPtr(layer),
                        cudaKvCache.GetValuesPtr(layer), _state.AttnOutput, _state.PositionsDevice,
                        seqLen, seqKv, numHeads, numKvHeads, headDim,
                        slidingWindow, s);
                else
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

            // Optional attention Sub-LN (BitNet): RMSNorm over the attention output [numHeads·headDim]
            // before the output projection. No-op for non-BitNet models (weight == 0).
            bool fusedAttnSubNormO = !s_i2sA8Decode && CanFuseI2SNormDecode(
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

            // ── FUSED: attention residual + FFN norm ──
            // residual = residual + NormOutput, NormOutput = rmsnorm(new_residual, ffnNormWeight)
            if (fp32Res)
                _kernels.LaunchFusedAddRmsNormF32Res(_state.ResidualF32, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
            else
                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK (NormOutput has FFN-normalized input) ──

            // Gate/Up projections
            if (!s_i2sA8Decode && CanFuseI2SDecode(seqLen, lw.GateQuantType, lw.UpQuantType,
                    lw.GateInputDim, lw.UpInputDim))
            {
                _kernels.LaunchI2_SGemv2F16In(
                    lw.GateQuant, lw.UpQuant, _state.NormOutput,
                    _state.FfnGate, _state.FfnUp,
                    lw.GateOutputDim, lw.UpOutputDim, lw.GateInputDim, s);
            }
            else
            {
                Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
                Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);
            }

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            // Gated activation (FP16). BitNet b1.58 uses squared-ReLU GLU followed by a Sub-LN
            // RMSNorm; the un-normalized relu(gate)²·up intermediate overflows FP16, so when the
            // Sub-LN weight is present we fuse activation + RMSNorm (large value kept in FP32,
            // only the normalized O(1) result hits FP16). Otherwise dispatch the plain activation.
            if (Config.ActivationFunction == ActivationFunction.ReluSquared && lw.FfnSubNormWeight != 0)
            {
                _kernels.LaunchReLU2GluRmsNorm(_state.FfnGate, _state.FfnUp, lw.FfnSubNormWeight,
                    _state.SiluOutput, intermediateSize, eps, seqLen, s);
            }
            else
            {
                if (Config.ActivationFunction == ActivationFunction.ReluSquared)
                    _kernels.LaunchReLU2(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                        intermediateSize, seqLen, s);
                else
                    _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                        intermediateSize, seqLen, s);

                // Optional FFN Sub-LN (BitNet) for the non-fused fallback. No-op when weight == 0.
                if (lw.FfnSubNormWeight != 0)
                    _kernels.LaunchRmsNorm(_state.SiluOutput, lw.FfnSubNormWeight, _state.SiluOutput,
                        intermediateSize, eps, seqLen, s);
            }

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // ── FUSED: FFN residual + next layer's attention norm ──
            if (layer < numLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                if (fp32Res)
                    _kernels.LaunchFusedAddRmsNormF32Res(_state.ResidualF32, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                        hiddenSize, eps, seqLen, s);
                else
                    _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                        hiddenSize, eps, seqLen, s);
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

        // 6. LM head (last token only) → FP16 logits, then convert to FP32
        Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
            _state.NormOutput, _state.LogitsF16,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);
        _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
            }
            catch
            {
                if (graphCapture)
                    _decodeGraph?.Abort(s);
                throw;
            }

            if (graphCapture && _decodeGraph?.TryEnd(s) != true)
            {
                _decodeGraph?.Dispose();
                _decodeGraph = null;
                _decodeGraphDisabled = true;
                goto DispatchAgain;
            }
            if (graphCapture)
            {
                _decodeGraph!.Launch(s);
            }
        }

        // Record decode-graph status for diagnostics (single-token decode only; prefill is N/A).
        if (seqLen == 1)
        {
            DecodeGraphState = !s_cudaGraphDecode ? CudaDecodeGraphState.Off
                : graphLaunched ? CudaDecodeGraphState.Replayed
                : _decodeGraphDisabled ? CudaDecodeGraphState.Fallback
                : graphCapture ? CudaDecodeGraphState.Captured
                : CudaDecodeGraphState.Ineligible;
        }

        // Capture dispatch-only time (all launches queued, before GPU wait).
        long dispatchEndTs = profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // 7. Stream sync (single sync point for entire forward pass)
        _stream.Synchronize();

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

    private bool CanUseDecodeGraph(int seqLen, IKvCache? kvCache)
        => s_cudaGraphDecode
           && !_decodeGraphDisabled
           && seqLen == 1
           && kvCache is CudaKvCache
           && Config.Architecture == Architecture.BitNet
           && DebugMaxLayers == 0
           && DebugRopeTypeOverride < 0
           && !DebugSkipBias;

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
        _decodeGraph?.Dispose();
        _cudaLora?.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
