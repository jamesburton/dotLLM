using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Two-device CUDA pipeline-parallel transformer (layer-spanning): layers <c>[0..SplitLayer)</c> run on the
/// first CUDA device and layers <c>[SplitLayer..L)</c> plus the final norm + LM head on the second. The
/// hidden state crosses the boundary device0 → host (FP32) → device1, so a model can span more VRAM than any
/// single device holds.
/// </summary>
/// <remarks>
/// <para>
/// This is the CUDA→CUDA analogue of <c>VulkanPipelineTransformerModel</c> (Vulkan→Vulkan) and the
/// two-context sibling of <see cref="HybridVulkanCudaTransformerModel"/> (Vulkan→CUDA). Each stage is an
/// independent <see cref="CudaPipelineStage"/> bound to its own <see cref="CudaContext"/>; stage 1 holds a
/// windowed device-weight upload (<see cref="CudaWeights.LoadFromGguf"/> with <c>firstLayer = SplitLayer</c>),
/// so only its slice of the model occupies device-1 VRAM. <see cref="CudaWeights.LoadFromGguf"/> already skips
/// the output norm + LM head for a non-final window (<c>firstLayer + layerCount &lt; NumLayers</c>) and loads
/// them for the final window; stage 1 additionally skips the token-embedding table
/// (<c>skipTokenEmbed</c> — it is only ever seeded from stage 0's hidden state and never gathers). So stage 0
/// carries embedding + its layers and stage 1 carries its layers + norm + head, each shedding the non-layer
/// weights it cannot use — the same per-stage trim as the Vulkan pipeline split (#368; CUDA sibling #123).
/// </para>
/// <para>
/// The per-stage CUDA layer loop reuses the proven sequence from
/// <see cref="HybridVulkanCudaTransformerModel"/>'s CUDA phase (QKV projection → optional bias / QK-norm →
/// RoPE → KV-cache update → attention → O-proj → fused-add-RMSNorm → SwiGLU FFN). The only per-stage
/// differences are the head (stage 0 reads the embedding table; stage 1 uploads the host FP32 hidden and
/// converts to FP16) and the tail (stage 0 downloads the post-last-layer FP32 hidden; stage 1 applies the
/// final norm + LM head and downloads FP32 logits).
/// </para>
/// <para>
/// M-scope: standard dense / GQA causal forward (no MLA / MoE / gemma4 / graph-capture). Those follow
/// <see cref="HybridVulkanCudaTransformerModel"/>'s M1 scope and are follow-up work once the basic
/// layer-split machinery is validated on dual T4.
/// </para>
/// </remarks>
public sealed unsafe class CudaPipelineTransformerModel : IModel
{
    private readonly CudaPipelineStage _stage0; // embedding + layers [0..SplitLayer)
    private readonly CudaPipelineStage _stage1; // layers [SplitLayer..L) + final norm + LM head
    private readonly int _splitLayer;
    private readonly GgufFile? _gguf;            // owned only by the LoadFromGguf path
    private readonly TransformerWeights? _ownedCpuWeights; // owned only by the LoadFromGguf path

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _stage0.AllocatedBytes + _stage1.AllocatedBytes;

    /// <summary>Global layer index at which the second pipeline stage begins.</summary>
    public int SplitLayer => _splitLayer;

    /// <summary>Test seam (#123): stage 0 (embedding + layers <c>[0..SplitLayer)</c>).</summary>
    internal CudaPipelineStage Stage0 => _stage0;

    /// <summary>Test seam (#123): stage 1 (layers <c>[SplitLayer..L)</c> + norm + head; no embed table).</summary>
    internal CudaPipelineStage Stage1 => _stage1;

    private CudaPipelineTransformerModel(
        ModelConfig config, CudaPipelineStage stage0, CudaPipelineStage stage1, int splitLayer,
        GgufFile? gguf, TransformerWeights? ownedCpuWeights)
    {
        Config = config;
        _stage0 = stage0;
        _stage1 = stage1;
        _splitLayer = splitLayer;
        _gguf = gguf;
        _ownedCpuWeights = ownedCpuWeights;
    }

    /// <summary>
    /// Loads a two-device CUDA pipeline model from an opened GGUF, binding stage 0 to CUDA device
    /// <paramref name="device0Id"/> and stage 1 to <paramref name="device1Id"/>. The model owns the CPU
    /// weights (mmap) and the <paramref name="gguf"/> handle and disposes them with itself.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (held alive for the weights' mmap; disposed by this model).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="splitLayer">Global layer index where stage 1 begins (1 ≤ split &lt; NumLayers).</param>
    /// <param name="device0Id">CUDA device ordinal for stage 0 (0-based).</param>
    /// <param name="device1Id">CUDA device ordinal for stage 1 (0-based).</param>
    /// <param name="ptxDir">PTX kernel directory. Null auto-detects from <c>AppContext.BaseDirectory/ptx/</c>.</param>
    public static CudaPipelineTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int splitLayer,
        int device0Id, int device1Id, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ValidateSplit(splitLayer, config.NumLayers);
        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");

        // Shared host weights (full model) feed both windowed device uploads. The GPU-only load skips the
        // F32 host dequant of per-expert MoE tensors (matches CudaTransformerModel.LoadFromGguf).
        TransformerWeights? cpuWeights = null;
        CudaPipelineStage? stage0 = null;
        try
        {
            cpuWeights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);
            cpuWeights.RepackWeights(); // idempotent; matches the CUDA upload contract
            (stage0, var stage1) = BuildStages(config, cpuWeights, splitLayer, device0Id, device1Id, ptxDir);
            return new CudaPipelineTransformerModel(config, stage0, stage1, splitLayer, gguf, cpuWeights);
        }
        catch
        {
            stage0?.Dispose();
            cpuWeights?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Test-only factory that wires a two-device CUDA pipeline around already-built CPU
    /// <see cref="TransformerWeights"/> (synthetic fixtures). Mirrors
    /// <c>VulkanPipelineTransformerModel.BuildFromPrebuiltWeights</c>. The caller retains ownership of
    /// every host pointer carried by <paramref name="cpuWeights"/>; <see cref="Dispose"/> releases only the
    /// device allocations created by the two stage loaders.
    /// </summary>
    internal static CudaPipelineTransformerModel BuildFromPrebuiltWeights(
        TransformerWeights cpuWeights, ModelConfig config, int splitLayer,
        int device0Id = 0, int device1Id = 1, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(config);
        ValidateSplit(splitLayer, config.NumLayers);
        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");

        cpuWeights.RepackWeights(); // idempotent
        CudaPipelineStage? stage0 = null;
        try
        {
            (stage0, var stage1) = BuildStages(config, cpuWeights, splitLayer, device0Id, device1Id, ptxDir);
            // ownedCpuWeights = null: caller retains the host weights (test fixture owns them).
            return new CudaPipelineTransformerModel(config, stage0, stage1, splitLayer, gguf: null, ownedCpuWeights: null);
        }
        catch
        {
            stage0?.Dispose();
            throw;
        }
    }

    private static (CudaPipelineStage Stage0, CudaPipelineStage Stage1) BuildStages(
        ModelConfig config, TransformerWeights cpuWeights, int splitLayer,
        int device0Id, int device1Id, string ptxDir)
    {
        // Stage 0: embedding + layers [0..splitLayer). firstLayer=0, layerCount=splitLayer →
        //          (0 + splitLayer) < NumLayers ⇒ CudaWeights treats it as a non-final window and skips the
        //          output norm + LM head. isFinalStage=false.
        // Stage 1: layers [splitLayer..L) + final norm + LM head. firstLayer=splitLayer, layerCount=L-split →
        //          (splitLayer + (L-split)) == NumLayers ⇒ output norm + LM head are uploaded. isFinalStage=true.
        //          skipTokenEmbed: stage 1 is only entered via EnqueueFromHidden (never gathers), so the
        //          vocab × hidden embedding table is not uploaded to device 1 (#123, Vulkan sibling #368).
        CudaPipelineStage? stage0 = null;
        try
        {
            stage0 = CudaPipelineStage.Build(
                config, cpuWeights, device0Id, ptxDir,
                firstLayer: 0, layerCount: splitLayer, isFinalStage: false);
            var stage1 = CudaPipelineStage.Build(
                config, cpuWeights, device1Id, ptxDir,
                firstLayer: splitLayer, layerCount: config.NumLayers - splitLayer, isFinalStage: true,
                skipTokenEmbed: true);
            return (stage0, stage1);
        }
        catch
        {
            stage0?.Dispose();
            throw;
        }
    }

    private static void ValidateSplit(int splitLayer, int numLayers)
    {
        if (splitLayer <= 0 || splitLayer >= numLayers)
            throw new ArgumentOutOfRangeException(nameof(splitLayer),
                $"splitLayer must be between 1 and {numLayers - 1}; use CudaTransformerModel for a single device.");
    }

    /// <summary>Creates a composite KV-cache: stage-0 layers on device 0, stage-1 layers on device 1.</summary>
    public IKvCache CreateKvCache(int maxSeqLen)
    {
        CudaKvCache? kv0 = null;
        try
        {
            kv0 = _stage0.CreateKvCache(maxSeqLen);
            var kv1 = _stage1.CreateKvCache(maxSeqLen);
            return new CudaPipelineKvCache(kv0, kv1, _splitLayer);
        }
        catch
        {
            kv0?.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// Stage 0 (embedding + layers <c>[0..SplitLayer)</c>) runs on CUDA device 0; its post-last-layer hidden
    /// state is downloaded to host FP32 and fed to stage 1 (layers <c>[SplitLayer..L)</c> + final norm + LM
    /// head) on CUDA device 1. A <see cref="CudaPipelineKvCache"/> supplies each stage's device-local
    /// KV-cache; pass null for a cacheless prefill.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");
        var pipeline = kvCache as CudaPipelineKvCache;
        if (kvCache is not null && pipeline is null)
            throw new ArgumentException(
                $"Expected a {nameof(CudaPipelineKvCache)} from {nameof(CreateKvCache)}, got {kvCache.GetType().Name}.",
                nameof(kvCache));

        int seqLen = tokenIds.Length;

        // Stage 0 on device 0: embedding + layers [0..SplitLayer). Leaves the FP32 hidden state on its stream.
        _stage0.EnqueueFromEmbedding(tokenIds, positions, seqLen, pipeline?.Stage0);
        using ITensor hidden = _stage0.DownloadHiddenStateF32(seqLen); // device0 → host FP32

        // Stage 1 on device 1: resume from the host hidden rows, run layers [SplitLayer..L) + norm + LM head.
        _stage1.EnqueueFromHidden(hidden.DataPointer, positions, seqLen, pipeline?.Stage1); // host → device1
        return _stage1.FinishLogits();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _stage1.Dispose();
        _stage0.Dispose();
        _ownedCpuWeights?.Dispose();
        _gguf?.Dispose();
    }
}

/// <summary>
/// One CUDA device's resources + layer loop for a <see cref="CudaPipelineTransformerModel"/> stage. Owns a
/// dedicated <see cref="CudaContext"/>, stream, cuBLAS handle, PTX kernels, a windowed
/// <see cref="CudaWeights"/> upload, and the FP16 activation scratch. The per-layer body mirrors
/// <see cref="HybridVulkanCudaTransformerModel"/>'s CUDA phase exactly.
/// </summary>
internal sealed unsafe class CudaPipelineStage : IDisposable
{
    private readonly ModelConfig _config;
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaKernels _kernels;
    private readonly CudaWeights _weights;
    private readonly CudaForwardState _state;
    private readonly int _layerCount;       // layers resident on THIS device (window size)
    private readonly bool _isFinalStage;    // true ⇒ applies final norm + LM head, returns logits
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    // Persistent device FP32 staging buffer for the host→device boundary upload (stage 1 only). Grown
    // on demand (power-of-2) and reused across forwards to avoid per-call cuMemAlloc/cuMemFree.
    private nint _tempF32Device;
    private long _tempF32Capacity; // bytes

    public long AllocatedBytes => _state.AllocatedBytes;

    /// <summary>
    /// Test seam for the stage VRAM trim (#123): whether this stage holds the token-embedding table on
    /// device. A silent re-upload regression is invisible to the logit-parity tests, so the trim tests
    /// assert on this directly.
    /// </summary>
    internal bool HasTokenEmbed => _weights.TokenEmbedDevice != 0;

    private CudaPipelineStage(
        ModelConfig config, CudaContext context, CudaStream stream, CudaCublasHandle cublas,
        CudaKernels kernels, CudaWeights weights, CudaForwardState state,
        int layerCount, bool isFinalStage, float ropeTheta, int ropeDim, int ropeType)
    {
        _config = config;
        _context = context;
        _stream = stream;
        _cublas = cublas;
        _kernels = kernels;
        _weights = weights;
        _state = state;
        _layerCount = layerCount;
        _isFinalStage = isFinalStage;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
    }

    /// <summary>
    /// Builds a stage on CUDA device <paramref name="deviceId"/> from shared host weights, uploading only
    /// the window <c>[firstLayer .. firstLayer+layerCount)</c>. The output norm + LM head are uploaded iff
    /// the window reaches the last layer (see <see cref="CudaWeights.LoadFromGguf"/>). Pass
    /// <paramref name="skipTokenEmbed"/> for a non-first stage: the embedding table is not uploaded and
    /// <see cref="EnqueueFromEmbedding"/> throws — the stage must be entered via
    /// <see cref="EnqueueFromHidden"/>.
    /// </summary>
    public static CudaPipelineStage Build(
        ModelConfig config, TransformerWeights cpuWeights, int deviceId, string ptxDir,
        int firstLayer, int layerCount, bool isFinalStage, bool skipTokenEmbed = false)
    {
        CudaContext? context = null;
        CudaStream? stream = null;
        CudaCublasHandle? cublas = null;
        CudaKernels? kernels = null;
        CudaWeights? weights = null;
        CudaForwardState? state = null;
        try
        {
            // Create the context first — it becomes current on the calling thread, so the stream / cuBLAS /
            // PTX module / device uploads below all bind to THIS device.
            context = CudaContext.Create(deviceId);
            context.MakeCurrent();
            stream = CudaStream.Create();
            cublas = CudaCublasHandle.Create();
            cublas.SetStream(stream);
            kernels = new CudaKernels(ptxDir);

            weights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle,
                numGpuLayers: layerCount, firstLayer: firstLayer, skipTokenEmbed: skipTokenEmbed);

            state = new CudaForwardState(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
                config.HeadDim, config.IntermediateSize, config.VocabSize);

            int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
            if (ropeDim == 0) ropeDim = config.HeadDim;
            float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
            int ropeType = CudaKernels.ToCudaRopeType(config.RoPEConfig?.Type ?? RoPEType.Norm);

            return new CudaPipelineStage(config, context, stream, cublas, kernels, weights, state,
                layerCount, isFinalStage, ropeTheta, ropeDim, ropeType);
        }
        catch
        {
            state?.Dispose();
            weights?.Dispose();
            // #383 review follow-up: kernels was previously left as a non-nullable `var` local,
            // out of scope here — never disposed on failure. Hoisted to a nullable local above so
            // all 12 factory sites in this issue dispose kernels uniformly on the catch path.
            kernels?.Dispose();
            cublas?.Dispose();
            stream?.Dispose();
            context?.Dispose();
            throw;
        }
    }

    /// <summary>Creates a device-local FP16 KV-cache (this stage's layer count) bound to this stage's context.</summary>
    public CudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new CudaKvCache(_layerCount, _config.NumKvHeads, _config.HeadDim, maxSeqLen, _context);
    }

    // ── Entry: first stage (embedding lookup) ──

    /// <summary>
    /// Enqueues this stage starting from the token embedding: upload tokens + positions, embedding lookup,
    /// then the layer loop. Used by the first pipeline stage. Leaves the post-last-layer FP32 hidden state
    /// queued (downloaded by <see cref="DownloadHiddenStateF32"/>).
    /// </summary>
    public void EnqueueFromEmbedding(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int seqLen,
                                     CudaKvCache? kvCache)
    {
        if (_weights.TokenEmbedDevice == 0)
            throw new InvalidOperationException(
                "This stage was built without the token-embedding table (skipTokenEmbed) and can only " +
                "resume from a previous stage's hidden state via " + nameof(EnqueueFromHidden) + ".");
        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint s = _stream.Handle;

        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        _kernels.LaunchEmbeddingLookup(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState,
            seqLen, _config.HiddenSize, s);

        RunLayers(positions, seqLen, kvCache);
    }

    // ── Entry: second stage (resume from host hidden) ──

    /// <summary>
    /// Enqueues this stage starting from a host FP32 hidden state: upload positions, H2D the hidden into a
    /// persistent FP32 staging buffer, convert to FP16, then the layer loop. Used by the second pipeline
    /// stage. The host buffer at <paramref name="hostHiddenF32Ptr"/> is fully consumed by the synchronous
    /// H2D before this returns, so the caller may dispose it on return.
    /// </summary>
    public void EnqueueFromHidden(nint hostHiddenF32Ptr, ReadOnlySpan<int> positions, int seqLen,
                                  CudaKvCache? kvCache)
    {
        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint s = _stream.Handle;

        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        int transferElems = seqLen * _config.HiddenSize;
        long fp32Bytes = (long)transferElems * sizeof(float);
        EnsureTempF32Capacity(fp32Bytes);

        // Synchronous H2D — fully drains the host hidden buffer into device FP32, then async F32→F16.
        CudaDriverApi.cuMemcpyHtoD_v2(_tempF32Device, hostHiddenF32Ptr, (nuint)fp32Bytes).ThrowOnError();
        _kernels.LaunchConvertF32ToF16(_tempF32Device, _state.HiddenState, transferElems, s);

        RunLayers(positions, seqLen, kvCache);
    }

    // ── Shared layer loop (mirrors HybridVulkanCudaTransformerModel.EnqueueCudaPhase) ──

    private void RunLayers(ReadOnlySpan<int> positions, int seqLen, CudaKvCache? kvCache)
    {
        int hiddenSize = _config.HiddenSize;
        int numHeads = _config.NumAttentionHeads;
        int numKvHeads = _config.NumKvHeads;
        int headDim = _config.HeadDim;
        int intermediateSize = _config.IntermediateSize;
        int vocabSize = _config.VocabSize;
        float eps = _config.NormEpsilon;
        int slidingWindow = _config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); // FP16 element size

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;
        long hiddenFp16Bytes = (long)seqLen * hiddenSize * h;

        // Layer 0 of this stage: residual = hidden; pre-RMSNorm into NormOutput using the local layer-0
        // attention norm. (For the second stage this is the GLOBAL split-layer's attn norm — Layers[] is
        // 0-based over the uploaded window.)
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
            (nuint)hiddenFp16Bytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight,
            _state.NormOutput, hiddenSize, eps, seqLen, s);

        for (int local = 0; local < _layerCount; local++)
        {
            ref readonly var lw = ref _weights.Layers[local];
            int cacheLayer = local; // per-stage KV cache is 0-based over this stage's layers

            // ── ATTENTION BLOCK ──
            Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q,
                lw.QOutputDim, lw.QInputDim, seqLen, s, cublasH);
            Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K,
                lw.KOutputDim, lw.KInputDim, seqLen, s, cublasH);
            Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V,
                lw.VOutputDim, lw.VInputDim, seqLen, s, cublasH);

            if (lw.QBias != 0) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, _ropeType, s);

            if (kvCache is not null)
            {
                kvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, cacheLayer, s);
                int seqKv = kvCache.CurrentLength;
                _kernels.LaunchAttention(_state.Q, kvCache.GetKeysPtr(cacheLayer),
                    kvCache.GetValuesPtr(cacheLayer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);
            }

            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen, s, cublasH);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput,
                lw.FfnNormWeight, _state.NormOutput, hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK ──
            Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, seqLen, s, cublasH);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen, s, cublasH);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                intermediateSize, seqLen, s);

            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen, s, cublasH);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            if (local < _layerCount - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[local + 1];
                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput,
                    nextLw.AttnNormWeight, _state.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last layer of this stage: HiddenState = Residual + FFN-out (the raw residual stream, no
                // next-layer norm). This is exactly the hidden state handed to the next stage.
                _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                    seqLen * hiddenSize, s);
            }
        }

        if (_isFinalStage)
        {
            // ── Final RMSNorm + LM Head (last token only) → FP32 logits ──
            nint lastHidden = _state.HiddenState + (nint)((long)(seqLen - 1) * hiddenSize * h);
            _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
                hiddenSize, eps, 1, s);

            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1, s, cublasH);

            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
        }
        else
        {
            // Non-final stage: convert the full [seqLen × hidden] FP16 hidden to FP32 for the host handoff.
            _kernels.LaunchConvertF16ToF32(_state.HiddenState, _state.HiddenStateF32, seqLen * hiddenSize, s);
        }
    }

    // ── Tails ──

    /// <summary>
    /// (Non-final stage) Drains the stream and copies the post-last-layer FP32 hidden state
    /// (<c>[seqLen × hidden]</c>) back to a freshly-allocated host tensor for the next stage.
    /// </summary>
    public ITensor DownloadHiddenStateF32(int seqLen)
    {
        _context.MakeCurrent();
        _stream.Synchronize();

        var shape = new TensorShape(seqLen, _config.HiddenSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.HiddenStateF32,
            (nuint)((long)seqLen * _config.HiddenSize * sizeof(float))).ThrowOnError();
        return result;
    }

    /// <summary>
    /// (Final stage) Drains the stream and copies the FP32 logits (<c>[1 × vocab]</c>) back to a
    /// freshly-allocated host tensor.
    /// </summary>
    public ITensor FinishLogits()
    {
        _context.MakeCurrent();
        int vocabSize = _config.VocabSize;
        _stream.Synchronize();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();
        return result;
    }

    private void EnsureTempF32Capacity(long bytes)
    {
        if (bytes <= _tempF32Capacity) return;

        long newCapacity = 1;
        while (newCapacity < bytes) newCapacity <<= 1;

        if (_tempF32Device != 0)
            CudaDriverApi.cuMemFree_v2(_tempF32Device).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _tempF32Device, (nuint)newCapacity).ThrowOnError();
        _tempF32Capacity = newCapacity;
    }

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
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, stream);
                w = _state.DequantScratch;
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
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, stream);
                w = _state.DequantScratch;
            }
            CudaGemm.GemvF16(cublasHandle, w, input, output, outputDim, inputDim, stream);
        }
    }

    public void Dispose()
    {
        _context.MakeCurrent();
        if (_tempF32Device != 0)
        {
            CudaDriverApi.cuMemFree_v2(_tempF32Device);
            _tempF32Device = 0;
        }
        _weights.Dispose();
        _state.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
