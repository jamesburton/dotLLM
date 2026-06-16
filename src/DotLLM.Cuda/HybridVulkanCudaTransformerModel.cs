using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

namespace DotLLM.Cuda;

/// <summary>
/// Hybrid Vulkan+CUDA transformer model: first N layers run on the Vulkan iGPU
/// (any vendor, FP32 activations), remaining layers run on the CUDA eGPU
/// (FP16 activations, cuBLAS GEMM/GEMV). The hidden-state boundary transfer
/// (Vulkan → host FP32 → CUDA FP16) is a synchronous H2D copy.
/// </summary>
/// <remarks>
/// <para>
/// Intended deployment scenario: AMD/Intel iGPU (Arc, RDNA) runs the first N
/// transformer layers via Vulkan; an NVIDIA dGPU runs the remaining layers plus
/// the final RMSNorm and LM head via CUDA. The VRAM split lets the iGPU's
/// shared system memory absorb early layers while the eGPU handles the
/// numerically-intensive second half.
/// </para>
/// <para>
/// SYNC WARNING: The CUDA layer loop (Phase 3) replicates logic from
/// <c>CudaTransformerModel.Forward</c> and the Phase 1 embedding + Vulkan
/// layer loop replicates logic from <c>VulkanTransformerModel.Forward</c>.
/// Bug fixes to attention, FFN, or norm logic may need to be applied in all
/// three locations.
/// </para>
/// <para>
/// M1 scope: standard GQA (no MLA / MoE / graph-capture / profiling). Those
/// paths are follow-up work once the basic layer-split machinery is validated.
/// </para>
/// </remarks>
public sealed unsafe class HybridVulkanCudaTransformerModel : IModel
{
    // ── Vulkan resources (Phase 1) ──
    private readonly VulkanTransformerModel _vulkanModel;
    private readonly int _numVulkanLayers;

    // ── CUDA resources (Phase 3) ──
    private readonly CudaWeights _cudaWeights;
    private readonly CudaForwardState _cudaState;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly int _cudaDeviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _cudaState.AllocatedBytes;

    /// <summary>Number of transformer layers running on the Vulkan device.</summary>
    public int NumVulkanLayers => _numVulkanLayers;

    private HybridVulkanCudaTransformerModel(
        ModelConfig config,
        VulkanTransformerModel vulkanModel, int numVulkanLayers,
        CudaWeights cudaWeights, CudaForwardState cudaState,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context,
        CudaKernels kernels, int cudaDeviceId,
        float ropeTheta, int ropeDim, int ropeType)
    {
        Config = config;
        _vulkanModel = vulkanModel;
        _numVulkanLayers = numVulkanLayers;
        _cudaWeights = cudaWeights;
        _cudaState = cudaState;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _cudaDeviceId = cudaDeviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
    }

    /// <summary>
    /// Loads a hybrid Vulkan+CUDA transformer model from an opened GGUF file.
    /// The Vulkan device is selected by the standard env-var rules
    /// (e.g. <c>DOTLLM_VULKAN_DEVICE_VENDOR</c>).
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="numVulkanLayers">
    /// Number of layers to run on the Vulkan iGPU (1 ≤ N &lt; config.NumLayers).
    /// </param>
    /// <param name="cudaDeviceId">CUDA GPU device ordinal (0-based).</param>
    /// <param name="spvDir">
    /// SPIR-V shader directory for the Vulkan model. Null auto-detects from
    /// <c>AppContext.BaseDirectory/spv/</c>.
    /// </param>
    /// <param name="ptxDir">
    /// PTX kernel directory for the CUDA model. Null auto-detects from
    /// <c>AppContext.BaseDirectory/ptx/</c>.
    /// </param>
    public static HybridVulkanCudaTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int numVulkanLayers,
        int cudaDeviceId = 0, string? spvDir = null, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (numVulkanLayers <= 0 || numVulkanLayers >= config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(numVulkanLayers),
                $"numVulkanLayers must be between 1 and {config.NumLayers - 1}. " +
                $"Use VulkanTransformerModel for pure Vulkan or CudaTransformerModel for pure CUDA.");

        // 1. Load Vulkan model for the first N layers only.
        //    config with { NumLayers = N } restricts weight upload to N layers.
        var vulkanConfig = config with { NumLayers = numVulkanLayers };
        var vulkanModel = VulkanTransformerModel.LoadFromGguf(gguf, vulkanConfig, spvDir);

        // 2. Load CPU weights for CUDA upload (full model).
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        return CreateFromWeights(config, vulkanModel, numVulkanLayers,
            cpuWeights, cudaDeviceId, ptxDir);
    }

    /// <summary>
    /// Test-only factory that wires a Vulkan+CUDA hybrid model around
    /// already-built <see cref="TransformerWeights"/>. Mirrors the structure of
    /// <see cref="HybridTransformerModel.BuildFromPrebuiltWeights"/>. Caller
    /// retains ownership of the host weight pointers.
    /// </summary>
    /// <param name="cpuWeights">Already-built weight bundle (host F32 pointers).</param>
    /// <param name="config">Full model configuration.</param>
    /// <param name="numVulkanLayers">
    /// Number of layers to run on Vulkan (1 ≤ N &lt; config.NumLayers).
    /// </param>
    /// <param name="vulkanDevice">
    /// Vulkan device to use for the first N layers.
    /// The caller retains ownership (not disposed by this model).
    /// </param>
    /// <param name="cudaDeviceId">CUDA GPU device ordinal (0-based).</param>
    /// <param name="spvDir">SPIR-V shader directory.</param>
    /// <param name="ptxDir">PTX kernel directory. Null auto-detects.</param>
    internal static HybridVulkanCudaTransformerModel BuildFromPrebuiltWeights(
        TransformerWeights cpuWeights, ModelConfig config,
        int numVulkanLayers, VulkanDevice vulkanDevice,
        int cudaDeviceId = 0, string? spvDir = null, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(vulkanDevice);
        if (numVulkanLayers <= 0 || numVulkanLayers >= config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(numVulkanLayers),
                $"numVulkanLayers must be between 1 and {config.NumLayers - 1}.");

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");

        // Load Vulkan model for N layers (device not owned — caller retains it).
        var vulkanConfig = config with { NumLayers = numVulkanLayers };
        var vulkanModel = VulkanTransformerModel.BuildFromPrebuiltWeights(
            vulkanDevice, vulkanConfig, cpuWeights, spvDir);

        // Repack CPU weights for CUDA upload (idempotent).
        cpuWeights.RepackWeights();

        return CreateFromWeights(config, vulkanModel, numVulkanLayers,
            cpuWeights, cudaDeviceId, ptxDir);
    }

    private static HybridVulkanCudaTransformerModel CreateFromWeights(
        ModelConfig config, VulkanTransformerModel vulkanModel, int numVulkanLayers,
        TransformerWeights cpuWeights, int cudaDeviceId, string? ptxDir)
    {
        // Initialize CUDA.
        var context = CudaContext.Create(cudaDeviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        // Upload full model weights to CUDA (norm + LM head required for Phase 3 final).
        // numGpuLayers = -1 means all layers + output norm + LM head.
        var cudaWeights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle,
            numGpuLayers: -1);

        // CUDA scratch activations (same sizing as CudaTransformerModel).
        var cudaState = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        // RoPE config (same translation as CudaTransformerModel and HybridTransformerModel).
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = CudaKernels.ToCudaRopeType(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new HybridVulkanCudaTransformerModel(
            config, vulkanModel, numVulkanLayers,
            cudaWeights, cudaState, stream, cublas, context, kernels,
            cudaDeviceId, ropeTheta, ropeDim, ropeType);
    }

    /// <summary>
    /// Creates a <see cref="VulkanCudaKvCache"/> sized for this model's layer split.
    /// The Vulkan cache holds <see cref="NumVulkanLayers"/> FP32 per-layer KV tables;
    /// the CUDA cache holds <c>Config.NumLayers - NumVulkanLayers</c> FP16 tables.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for both sub-caches.</param>
    public VulkanCudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        int numCudaLayers = Config.NumLayers - _numVulkanLayers;
        return new VulkanCudaKvCache(
            _vulkanModel.CreateKvCache(maxSeqLen),
            new CudaKvCache(numCudaLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen),
            _numVulkanLayers);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// Three-phase forward pass:
    /// <list type="number">
    /// <item><description>
    /// Phase 1 — Vulkan: embedding lookup + layers 0..<see cref="NumVulkanLayers"/>-1.
    /// Output is the FP32 hidden state downloaded via a staging buffer.
    /// </description></item>
    /// <item><description>
    /// Phase 2 — Transfer: host FP32 → CUDA device FP32 → CUDA device FP16.
    /// </description></item>
    /// <item><description>
    /// Phase 3 — CUDA: layers <see cref="NumVulkanLayers"/>..N-1 (standard GQA,
    /// no MLA/MoE) + final RMSNorm + LM head. Returns host FP32 logits.
    /// </description></item>
    /// </list>
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        var vulkanCudaCache = kvCache as VulkanCudaKvCache;

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

        // ════════════════════════════════════════════════════════
        //  PHASE 1: Vulkan — Embedding + Layers 0.._numVulkanLayers-1
        // ════════════════════════════════════════════════════════

        // Run the Vulkan forward pass for the first N layers.
        // Logits returned here are discarded — only the hidden state matters.
        using var vulkanLogits = _vulkanModel.Forward(tokenIds, positions, deviceId: -1,
            kvCache: vulkanCudaCache?.VulkanCache);

        // Download the post-layer-N hidden state from Vulkan device memory.
        // DownloadHiddenState uses a Vulkan staging buffer internally; result is host FP32.
        using var vulkanHiddenTensor = _vulkanModel.DownloadHiddenState(seqLen);

        // ════════════════════════════════════════════════════════
        //  PHASE 2: Boundary Transfer (Vulkan host FP32 → CUDA FP16)
        // ════════════════════════════════════════════════════════

        _context.MakeCurrent();
        _cudaState.EnsureCapacity(seqLen);

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        int transferElems = seqLen * hiddenSize;
        long fp32Bytes = (long)transferElems * sizeof(float);
        long hiddenFp16Bytes = (long)transferElems * h;
        int totalLayers = Config.NumLayers;

        // Allocate a temporary device FP32 buffer, upload host FP32 from the Vulkan
        // tensor, then convert FP32→FP16 on-device into _cudaState.HiddenState.
        // The buffer is freed AFTER stream sync (it is read asynchronously).
        nint tempF32Device;
        CudaDriverApi.cuMemAlloc_v2(out tempF32Device, (nuint)fp32Bytes).ThrowOnError();

        // Synchronous H2D upload (blocking until data is queued; the convert kernel runs async).
        CudaDriverApi.cuMemcpyHtoD_v2(tempF32Device, vulkanHiddenTensor.DataPointer,
            (nuint)fp32Bytes).ThrowOnError();

        // Device FP32 → FP16 (async, on the CUDA stream).
        _kernels.LaunchConvertF32ToF16(tempF32Device, _cudaState.HiddenState, transferElems, s);

        // ════════════════════════════════════════════════════════
        //  PHASE 3: CUDA — Layers _numVulkanLayers..N-1 + Final Norm + LM Head
        // ════════════════════════════════════════════════════════

        // Upload positions to device (required by LaunchRoPE in the layer loop below).
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_cudaState.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // Layer 0 of CUDA phase (= global layer _numVulkanLayers):
        // copy hidden→residual, then RmsNorm into NormOutput.
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_cudaState.Residual, _cudaState.HiddenState,
            (nuint)hiddenFp16Bytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_cudaState.HiddenState, _cudaWeights.Layers[_numVulkanLayers].AttnNormWeight,
            _cudaState.NormOutput, hiddenSize, eps, seqLen, s);

        // CUDA layer loop — standard GQA forward for layers _numVulkanLayers..totalLayers-1.
        // KV-cache: CudaCache uses 0-based indices (global layer - _numVulkanLayers).
        var cudaKvCache = vulkanCudaCache?.CudaCache;
        for (int layer = _numVulkanLayers; layer < totalLayers; layer++)
        {
            ref readonly var lw = ref _cudaWeights.Layers[layer];
            int cacheLayer = layer - _numVulkanLayers; // 0-based index for CudaCache

            // ── ATTENTION BLOCK ──
            Project(lw.QQuant, lw.QQuantType, lw.Q, _cudaState.NormOutput, _cudaState.Q,
                lw.QOutputDim, lw.QInputDim, seqLen, s, cublasH);
            Project(lw.KQuant, lw.KQuantType, lw.K, _cudaState.NormOutput, _cudaState.K,
                lw.KOutputDim, lw.KInputDim, seqLen, s, cublasH);
            Project(lw.VQuant, lw.VQuantType, lw.V, _cudaState.NormOutput, _cudaState.V,
                lw.VOutputDim, lw.VInputDim, seqLen, s, cublasH);

            if (lw.QBias != 0) _kernels.LaunchBiasAdd(_cudaState.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0) _kernels.LaunchBiasAdd(_cudaState.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0) _kernels.LaunchBiasAdd(_cudaState.V, lw.VBias, lw.VOutputDim, seqLen, s);

            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_cudaState.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_cudaState.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            _kernels.LaunchRoPE(_cudaState.Q, _cudaState.K, _cudaState.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, _ropeType, s);

            // KV-cache update (uses cacheLayer for 0-based CUDA cache indexing)
            if (cudaKvCache is not null)
            {
                cudaKvCache.UpdateDevice(_cudaState.K, _cudaState.V, positions, seqLen, cacheLayer, s);
                int seqKv = cudaKvCache.CurrentLength;
                _kernels.LaunchAttention(_cudaState.Q, cudaKvCache.GetKeysPtr(cacheLayer),
                    cudaKvCache.GetValuesPtr(cacheLayer), _cudaState.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_cudaState.Q, _cudaState.K, _cudaState.V, _cudaState.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);
            }

            // O projection → NormOutput
            Project(lw.OQuant, lw.OQuantType, lw.O, _cudaState.AttnOutput, _cudaState.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen, s, cublasH);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_cudaState.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            // Fused attention residual + FFN norm
            _kernels.LaunchFusedAddRmsNorm(_cudaState.Residual, _cudaState.NormOutput,
                lw.FfnNormWeight, _cudaState.NormOutput, hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK ──
            Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _cudaState.NormOutput, _cudaState.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, seqLen, s, cublasH);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _cudaState.NormOutput, _cudaState.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen, s, cublasH);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_cudaState.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_cudaState.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            _kernels.LaunchSwiGLU(_cudaState.FfnGate, _cudaState.FfnUp, _cudaState.SiluOutput,
                intermediateSize, seqLen, s);

            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _cudaState.SiluOutput, _cudaState.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen, s, cublasH);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_cudaState.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // Fused FFN residual + next-layer setup
            if (layer < totalLayers - 1)
            {
                // Not last layer: FusedAddRmsNorm pre-applies next layer's AttnNorm
                ref readonly var nextLw = ref _cudaWeights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_cudaState.Residual, _cudaState.NormOutput,
                    nextLw.AttnNormWeight, _cudaState.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last layer: plain add → HiddenState for final norm
                _kernels.LaunchAdd(_cudaState.Residual, _cudaState.NormOutput, _cudaState.HiddenState,
                    seqLen * hiddenSize, s);
            }
        }

        // ── Final RMSNorm + LM Head (last token only) ──
        nint lastHidden = _cudaState.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);
        _kernels.LaunchRmsNorm(lastHidden, _cudaWeights.OutputNormWeight, _cudaState.NormOutput,
            hiddenSize, eps, 1, s);

        Project(_cudaWeights.OutputWeightQuant, _cudaWeights.OutputQuantType, _cudaWeights.OutputWeight,
            _cudaState.NormOutput, _cudaState.LogitsF16,
            _cudaWeights.OutputOutputDim, _cudaWeights.OutputInputDim, 1, s, cublasH);

        _kernels.LaunchConvertF16ToF32(_cudaState.LogitsF16, _cudaState.LogitsF32, vocabSize, s);

        // Single sync for all CUDA kernel submissions.
        // tempF32Device is freed after sync — safe because all async reads are done.
        _stream.Synchronize();
        CudaDriverApi.cuMemFree_v2(tempF32Device).ThrowOnError();

        // D2H: FP32 logits → host UnmanagedTensor.
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _cudaState.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    // ── Private helpers ──

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Project(nint quantWeight, QuantizationType qt, nint fp16Weight,
                         nint input, nint output, int outputDim, int inputDim,
                         int seqLen, nint stream, nint cublasHandle)
    {
        if (seqLen > 1)
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _cudaState.DequantScratch,
                    outputDim * inputDim, stream);
                w = _cudaState.DequantScratch;
            }
            CudaGemm.LinearF16(cublasHandle, input, w, output, seqLen, inputDim, outputDim, stream);
        }
        else if (quantWeight != 0 && _kernels.HasQuantizedGemvKernel(qt))
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, stream);
        }
        else
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _cudaState.DequantScratch,
                    outputDim * inputDim, stream);
                w = _cudaState.DequantScratch;
            }
            CudaGemm.GemvF16(cublasHandle, w, input, output, outputDim, inputDim, stream);
        }
    }

    /// <summary>Releases all Vulkan, CUDA, and host resources owned by this model.</summary>
    public void Dispose()
    {
        _vulkanModel.Dispose();
        _cudaWeights.Dispose();
        _cudaState.Dispose();
        _stream.Dispose();
        _cublas.Dispose();
        _context.Dispose();
    }
}
