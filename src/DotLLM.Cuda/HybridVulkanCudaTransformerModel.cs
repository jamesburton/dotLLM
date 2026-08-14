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

    // Persistent device FP32 staging buffer for the Vulkan→CUDA boundary upload.
    // Grown on demand (power-of-2) and reused across forwards to avoid a per-call
    // cuMemAlloc/cuMemFree on the hot path. Required by the pipelined path: the
    // synchronous H2D fills it before each EnqueueCudaPhase returns, so a single
    // buffer is safe (CUDA(i-1) is synced before CUDA(i) reuses it).
    private nint _tempF32Device;
    private long _tempF32Capacity; // in bytes

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
        // #383: `vulkanModel` and `cpuWeights` are each created before CreateFromWeights'
        // (CUDA-side) resources exist, so CreateFromWeights' own try/catch can't reach them —
        // if it throws, this method must dispose whichever of these it already created.
        var vulkanConfig = config with { NumLayers = numVulkanLayers };
        var vulkanModel = VulkanTransformerModel.LoadFromGguf(gguf, vulkanConfig, spvDir);
        try
        {
            // 2. Load CPU weights for CUDA upload (full model).
            var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            try
            {
                return CreateFromWeights(config, vulkanModel, numVulkanLayers,
                    cpuWeights, cudaDeviceId, ptxDir);
            }
            catch
            {
                cpuWeights.Dispose();
                throw;
            }
        }
        catch
        {
            vulkanModel.Dispose();
            throw;
        }
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
        // #383: `vulkanModel` is created before CreateFromWeights' (CUDA-side) resources exist,
        // so CreateFromWeights' own try/catch can't reach it — if it throws, this method must
        // dispose the vulkanModel it already created. `cpuWeights` stays caller-owned (per this
        // method's doc) — never disposed here, on success or failure.
        var vulkanConfig = config with { NumLayers = numVulkanLayers };
        var vulkanModel = VulkanTransformerModel.BuildFromPrebuiltWeights(
            vulkanDevice, vulkanConfig, cpuWeights, spvDir);
        try
        {
            // Repack CPU weights for CUDA upload (idempotent).
            cpuWeights.RepackWeights();

            return CreateFromWeights(config, vulkanModel, numVulkanLayers,
                cpuWeights, cudaDeviceId, ptxDir);
        }
        catch
        {
            vulkanModel.Dispose();
            throw;
        }
    }

    private static HybridVulkanCudaTransformerModel CreateFromWeights(
        ModelConfig config, VulkanTransformerModel vulkanModel, int numVulkanLayers,
        TransformerWeights cpuWeights, int cudaDeviceId, string? ptxDir)
    {
        // Initialize CUDA. #383: context creation cannot leak on its own throw (nothing
        // allocated yet), so it stays outside the try/catch — everything created from here on
        // is disposed on any failure before rethrowing. This method does NOT own
        // `vulkanModel`/`cpuWeights` — both callers (LoadFromGguf, BuildFromPrebuiltWeights) are
        // responsible for those.
        var context = CudaContext.Create(cudaDeviceId);
        CudaStream? stream = null;
        CudaCublasHandle? cublas = null;
        CudaKernels? kernels = null;
        CudaWeights? cudaWeights = null;
        CudaForwardState? cudaState = null;
        try
        {
            stream = CudaStream.Create();
            cublas = CudaCublasHandle.Create();
            cublas.SetStream(stream);

            ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
            kernels = new CudaKernels(ptxDir);

            // Upload only the CUDA-resident layers to CUDA VRAM. Layers 0..numVulkanLayers-1
            // are handled by Vulkan; no need to pay for them in CUDA VRAM.
            // firstLayer skips the Vulkan slice; numGpuLayers = L-V covers the rest + output norm/LM head.
            // skipTokenEmbed: the embedding gather happens on the Vulkan side, so the CUDA phase never
            // reads the table — don't pay vocab × hidden of CUDA VRAM for it (#123).
            int numCudaOnlyLayers = config.NumLayers - numVulkanLayers;
            cudaWeights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle,
                numGpuLayers: numCudaOnlyLayers, firstLayer: numVulkanLayers, skipTokenEmbed: true);

            // CUDA scratch activations (same sizing as CudaTransformerModel).
            cudaState = new CudaForwardState(
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
        catch
        {
            cudaState?.Dispose();
            cudaWeights?.Dispose();
            kernels?.Dispose();
            cublas?.Dispose();
            stream?.Dispose();
            // CudaContext.Create (above) makes the context current on THIS thread, and this
            // catch runs synchronously on the same thread — no MakeCurrent() call is needed
            // here (matches #368's convention).
            context.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a <see cref="HybridVulkanCudaKvCache"/> sized for this model's layer split.
    /// The Vulkan cache holds <see cref="NumVulkanLayers"/> FP32 per-layer KV tables;
    /// the CUDA cache holds <c>Config.NumLayers - NumVulkanLayers</c> FP16 tables.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for both sub-caches.</param>
    public HybridVulkanCudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        int numCudaLayers = Config.NumLayers - _numVulkanLayers;
        return new HybridVulkanCudaKvCache(
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
        var vulkanCudaCache = kvCache as HybridVulkanCudaKvCache;
        int seqLen = tokenIds.Length;

        // ── PHASE 1: Vulkan (embedding + layers 0..V-1), host download ──
        using var vulkanHiddenTensor = RunVulkanPhase(tokenIds, positions, vulkanCudaCache);

        // ── PHASE 2+3: CUDA enqueue (async on the stream), then finish (sync + DtoH) ──
        EnqueueCudaPhase(vulkanHiddenTensor.DataPointer, positions, seqLen, vulkanCudaCache);
        return FinishCudaPhase();
    }

    /// <summary>
    /// Pipelined batched forward over <paramref name="requests"/> independent
    /// streams (sequences). Overlaps the Vulkan stage of stream <c>i</c> with the
    /// in-flight CUDA stage of stream <c>i-1</c>: while the CUDA eGPU consumes
    /// stream <c>i-1</c>'s activations, the Vulkan iGPU computes stream <c>i</c>'s
    /// early layers. This fills the pipeline bubble inherent to a 2-device
    /// layer split and is the M3 throughput win for the batched / multi-stream /
    /// multi-turn serving path.
    /// </summary>
    /// <param name="requests">
    /// Per-stream inputs: token ids, positions, and the stream's own KV cache.
    /// Each request is decoded independently (one forward each); the KV caches
    /// must be distinct per stream.
    /// </param>
    /// <returns>
    /// One logits tensor (<c>[1, vocabSize]</c>, host FP32) per request, in input
    /// order. Caller owns disposal of each.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The overlap is cross-device: the two GPUs touch disjoint scratch
    /// (<c>_cudaState</c> is CUDA-only, the Vulkan model's scratch is Vulkan-only)
    /// and per-stream KV caches, so the shared single scratch on each device is
    /// safe. CUDA work is stream-ordered on one stream, so CUDA(i) is enqueued
    /// only after CUDA(i-1) has been synced — making the persistent staging
    /// buffers safe to reuse without a double-buffer ring.
    /// </para>
    /// <para>
    /// The Vulkan→CUDA handoff routes through host RAM (the synchronous
    /// <c>cuMemcpyHtoD</c> fully consumes the host hidden buffer before
    /// <see cref="EnqueueCudaPhase"/> returns). The original M3 plan gated the
    /// CUDA stage with an imported Vulkan semaphore
    /// (<c>cuWaitExternalSemaphoresAsync</c>); on this Intel-Arc-iGPU +
    /// NVIDIA-3060-eGPU box that cross-API import is blocked by CUDA's
    /// same-adapter (UUID/LUID) requirement (see the M3 interop smoke tests).
    /// Host-pipelining delivers the same overlap because, per
    /// <c>offload-partitioning-strategy.md</c>, the cross-API wait would only
    /// have hidden the &lt;1%-of-compute handoff term; the dominant win is
    /// serial→pipelined, captured here without it.
    /// </para>
    /// <para>
    /// <c>requests.Count == 1</c> degenerates to the synchronous serial path
    /// (Vulkan → enqueue → finish), so single-stream batch=1 decode is neutral.
    /// </para>
    /// </remarks>
    public ITensor[] ForwardBatchedPipelined(IReadOnlyList<PipelinedRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        int n = requests.Count;
        if (n == 0) return [];

        var results = new ITensor[n];

        // Software pipeline, depth 1: Vulkan(i) ∥ CUDA(i-1).
        //
        //   Vulkan(0) ; Enqueue(0)
        //   for i in 1..n-1:  Vulkan(i)  [overlaps in-flight CUDA(i-1)]
        //                     Finish(i-1) ; Enqueue(i)
        //   Finish(n-1)
        //
        // The host blocks inside Vulkan(i) on the iGPU fence while the eGPU runs
        // CUDA(i-1) — that wait is the overlap window.

        var first = requests[0];
        var hiddenCurrent = RunVulkanPhase(first.TokenIds, first.Positions, AsHybridCache(first.KvCache));
        EnqueueCudaPhase(hiddenCurrent.DataPointer, first.Positions, first.TokenIds.Length,
            AsHybridCache(first.KvCache));

        for (int i = 1; i < n; i++)
        {
            var req = requests[i];
            // Vulkan stage for stream i — runs on the iGPU while CUDA(i-1) is in flight on the eGPU.
            var hiddenNext = RunVulkanPhase(req.TokenIds, req.Positions, AsHybridCache(req.KvCache));

            // Drain CUDA(i-1): sync + read its logits. Its scratch/staging are now free.
            results[i - 1] = FinishCudaPhase();
            hiddenCurrent.Dispose();
            hiddenCurrent = hiddenNext;

            // Enqueue CUDA(i) (synchronous H2D consumes hiddenCurrent, then async kernels).
            EnqueueCudaPhase(hiddenCurrent.DataPointer, req.Positions, req.TokenIds.Length,
                AsHybridCache(req.KvCache));
        }

        results[n - 1] = FinishCudaPhase();
        hiddenCurrent.Dispose();
        return results;
    }

    private static HybridVulkanCudaKvCache? AsHybridCache(IKvCache? kvCache)
        => kvCache as HybridVulkanCudaKvCache;

    // ── Phase helpers (shared by serial Forward and pipelined batched forward) ──

    /// <summary>
    /// Runs the Vulkan stage (embedding + layers 0..V-1) for one sequence and
    /// downloads the boundary hidden state to host FP32. Blocks the host on the
    /// Vulkan fence — that host wait is what overlaps in-flight CUDA work in the
    /// pipelined path.
    /// </summary>
    private ITensor RunVulkanPhase(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                   HybridVulkanCudaKvCache? vulkanCudaCache)
    {
        using var vulkanLogits = _vulkanModel.Forward(tokenIds, positions, deviceId: -1,
            kvCache: vulkanCudaCache?.VulkanCache);
        return _vulkanModel.DownloadHiddenState(tokenIds.Length);
    }

    /// <summary>
    /// Enqueues the CUDA stage (boundary upload + layers V..L-1 + final norm + LM
    /// head) on the CUDA stream <b>without synchronising</b>. The host FP32
    /// hidden state at <paramref name="hostHiddenPtr"/> is fully consumed by the
    /// synchronous H2D before this returns, so the caller may overwrite it
    /// (e.g. dispose the Vulkan tensor) once this method returns. The CUDA
    /// kernels run asynchronously; call <see cref="FinishCudaPhase"/> to drain.
    /// </summary>
    private void EnqueueCudaPhase(nint hostHiddenPtr, ReadOnlySpan<int> positions, int seqLen,
                                  HybridVulkanCudaKvCache? vulkanCudaCache)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); // FP16 element size

        _context.MakeCurrent();
        _cudaState.EnsureCapacity(seqLen);

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        int transferElems = seqLen * hiddenSize;
        long fp32Bytes = (long)transferElems * sizeof(float);
        long hiddenFp16Bytes = (long)transferElems * h;
        int totalLayers = Config.NumLayers;

        EnsureTempF32Capacity(fp32Bytes);

        // Synchronous H2D upload — fully drains hostHiddenPtr into device memory,
        // so the caller may free/overwrite the host buffer once this returns.
        CudaDriverApi.cuMemcpyHtoD_v2(_tempF32Device, hostHiddenPtr, (nuint)fp32Bytes).ThrowOnError();

        // Device FP32 → FP16 (async, on the CUDA stream).
        _kernels.LaunchConvertF32ToF16(_tempF32Device, _cudaState.HiddenState, transferElems, s);

        // Upload positions to device (required by LaunchRoPE in the layer loop below).
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_cudaState.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // Layer 0 of CUDA phase (= global layer _numVulkanLayers, local CUDA index 0).
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_cudaState.Residual, _cudaState.HiddenState,
            (nuint)hiddenFp16Bytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_cudaState.HiddenState, _cudaWeights.Layers[0].AttnNormWeight,
            _cudaState.NormOutput, hiddenSize, eps, seqLen, s);

        var cudaKvCache = vulkanCudaCache?.CudaCache;
        for (int layer = _numVulkanLayers; layer < totalLayers; layer++)
        {
            int localLayer = layer - _numVulkanLayers; // 0-based for both CudaWeights and CudaCache
            ref readonly var lw = ref _cudaWeights.Layers[localLayer];
            int cacheLayer = localLayer;

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

            Project(lw.OQuant, lw.OQuantType, lw.O, _cudaState.AttnOutput, _cudaState.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen, s, cublasH);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_cudaState.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

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

            if (layer < totalLayers - 1)
            {
                ref readonly var nextLw = ref _cudaWeights.Layers[localLayer + 1];
                _kernels.LaunchFusedAddRmsNorm(_cudaState.Residual, _cudaState.NormOutput,
                    nextLw.AttnNormWeight, _cudaState.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                _kernels.LaunchAdd(_cudaState.Residual, _cudaState.NormOutput, _cudaState.HiddenState,
                    seqLen * hiddenSize, s);
            }
        }

        // ── Final RMSNorm + LM Head (last token only) ──
        nint lastHidden = _cudaState.HiddenState + (nint)((long)(seqLen - 1) * hiddenSize * h);
        _kernels.LaunchRmsNorm(lastHidden, _cudaWeights.OutputNormWeight, _cudaState.NormOutput,
            hiddenSize, eps, 1, s);

        Project(_cudaWeights.OutputWeightQuant, _cudaWeights.OutputQuantType, _cudaWeights.OutputWeight,
            _cudaState.NormOutput, _cudaState.LogitsF16,
            _cudaWeights.OutputOutputDim, _cudaWeights.OutputInputDim, 1, s, cublasH);

        _kernels.LaunchConvertF16ToF32(_cudaState.LogitsF16, _cudaState.LogitsF32, vocabSize, s);
    }

    /// <summary>
    /// Drains the CUDA stream from a prior <see cref="EnqueueCudaPhase"/>: blocks
    /// until all enqueued work completes, then copies the FP32 logits back to a
    /// freshly-allocated host tensor. After this returns the CUDA scratch and the
    /// persistent staging buffer are free for the next enqueue.
    /// </summary>
    private ITensor FinishCudaPhase()
    {
        int vocabSize = Config.VocabSize;
        _stream.Synchronize();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _cudaState.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();
        return result;
    }

    /// <summary>
    /// Ensures <see cref="_tempF32Device"/> holds at least <paramref name="bytes"/>,
    /// growing it (power-of-2) and freeing the old allocation when necessary.
    /// </summary>
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
        _context.MakeCurrent();
        if (_tempF32Device != 0)
        {
            CudaDriverApi.cuMemFree_v2(_tempF32Device);
            _tempF32Device = 0;
        }
        _vulkanModel.Dispose();
        _cudaWeights.Dispose();
        _cudaState.Dispose();
        _stream.Dispose();
        _cublas.Dispose();
        _context.Dispose();
    }
}

/// <summary>
/// One stream's inputs for <see cref="HybridVulkanCudaTransformerModel.ForwardBatchedPipelined"/>.
/// Each request is decoded independently with its own KV cache; the pipelined
/// scheduler overlaps the Vulkan stage of one request with the CUDA stage of the
/// previous one.
/// </summary>
public sealed class PipelinedRequest
{
    /// <summary>Token ids for this forward pass (decode: a single token).</summary>
    public required int[] TokenIds { get; init; }

    /// <summary>Absolute positions matching <see cref="TokenIds"/>.</summary>
    public required int[] Positions { get; init; }

    /// <summary>
    /// This stream's own KV cache (a <see cref="HybridVulkanCudaKvCache"/>), or
    /// <c>null</c> for a stateless (no-cache) forward. Must be distinct per stream.
    /// </summary>
    public IKvCache? KvCache { get; init; }
}
