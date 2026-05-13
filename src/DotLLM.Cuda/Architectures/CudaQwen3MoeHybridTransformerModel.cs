using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the Qwen3MoeHybrid (<c>qwen35moe</c>) model. F32 activations
/// throughout — mirrors <see cref="Qwen3MoeHybridTransformerModel"/> on the GPU to deliver
/// initial-correctness parity with the CPU oracle for the Qwen3.6-35B-A3B-UD-Q6_K_XL GGUF.
/// An FP16 fast-path is left to follow-up optimisation.
/// </summary>
/// <remarks>
/// <para>
/// Each of the 40 layers has a token-mixing sub-layer (GDN recurrence or full GQA attention)
/// followed by a sparse MoE SwiGLU FFN. Layer kind for every index comes from
/// <see cref="ModelConfig.HybridLayout"/>; full-attention layers are placed every
/// <see cref="GatedDeltaNetConfig.FullAttnInterval"/> steps (e.g. 4, 8, 12, ... 40 for interval = 4).
/// </para>
/// <para>
/// Memory footprint at Qwen3.6-35B-A3B-UD-Q6_K_XL scale is dominated by the routed expert
/// raw-quant uploads (~25-30 GB for 128 experts × 3 projection matrices × 40 layers at Q6_K).
/// This is fine on A6000 / H100; will OOM on a 24 GB card. The shared-expert F32 weights
/// (small) are uploaded as F32 device buffers.
/// </para>
/// </remarks>
public sealed unsafe class CudaQwen3MoeHybridTransformerModel : IModel
{
    private readonly CudaQwen3MoeHybridForwardState _state;
    private readonly CudaGdnStateCache _gdnCache;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly GgufFile? _gguf;
    private readonly int _deviceId;

    // Per-layer device-side weight pointers — loaded once, alive for model lifetime.
    private readonly DeviceLayer[] _layers;

    // Output stage: token embedding (shared with lm_head when output.weight is missing) and
    // the final RMSNorm gain + lm_head projection.
    private readonly nint _tokenEmbedDevice;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly long _tokenEmbedBytes;
    private readonly nint _outputNormDevice; // F32 [hiddenSize]
    private readonly nint _outputDevice;     // lm_head raw quant bytes (may alias _tokenEmbedDevice)
    private readonly QuantizationType _outputQt;
    private readonly int _outputOutputDim;   // vocab size
    private readonly int _outputInputDim;    // hidden size
    private readonly bool _ownsOutputDevice; // false when aliased to embed

    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _gdnLayerOrdinal;

    private readonly float _ropeTheta;
    private readonly int _ropeDim;

    // Model-owned device F16 scratch for on-the-fly weight dequant in the prefill path
    // (seqLen > 1). Holds the dequantised weight tile that cuBLAS HGEMM then consumes —
    // mirrors the dense CudaTransformerModel.Project() prefill branch which dequants
    // quantised weights into an F16 scratch and runs F16 GEMM. Sized to the largest single
    // weight tile we ever GEMM with (`maxTileFloats` halves in element count → halves in
    // bytes vs the previous F32 scratch since each F16 element is 2 bytes vs 4). The
    // routed-expert dequant has its own dedicated scratch in _state.MoeW{1,2,3}Scratch.
    // Decode-time (seqLen == 1) projections bypass this entirely via the quantised GEMV
    // path (LaunchQuantizedGemv / LaunchQuantizedGemvMmq / LaunchQuantizedGemvF32In).
    private nint _dequantScratchF16Weight;
    private long _dequantScratchElems;

    // Lazily allocated F16 activation staging buffers for the decode/prefill F16 GEMV/GEMM
    // path. Activations on this model live in F32, but the quantised GEMV kernels and
    // cuBLAS HGEMM consume F16. We stage F32→F16 on the way in and F16→F32 on the way out.
    // Sized in F16 *elements* (each = 2 bytes). Grown lazily to the widest (seqLen × K_max)
    // and (seqLen × M_max) seen across calls.
    private nint _activF16InScratch;       // input staging: seqLen × K halfs
    private long _activF16InScratchElems;
    private nint _activF16OutScratch;      // output staging: seqLen × M halfs
    private long _activF16OutScratchElems;

    // F32 token embedding scratch used when the embedding table is not in F32/F16/Q8_0
    // (LaunchEmbeddingLookupF32 only supports those three formats). At Q6_K we pre-dequant
    // the entire table to a device-side F32 buffer at load time. Zero when the lookup kernel
    // can read the raw embedding table directly.
    private readonly nint _embedF32Device;
    private readonly bool _ownsEmbedF32;

    // Per-model MoE scratch reused across layers and forward calls. Owned by the model,
    // disposed in Dispose(). Sized once for (numExperts, topK, intermediate, hidden).
    private readonly CudaMoeScratch _moeScratch;

    // Per-attention-layer F32 KV cache. Sized lazily on first kvCache-enabled Forward
    // call from kvCache.MaxLength. Slot index per absolute layer comes from
    // _kvSlotForLayer; non-attention layers map to -1. Each entry is a device F32
    // buffer [maxSeqLen, kvElems].
    private nint[]? _f32KCache;
    private nint[]? _f32VCache;
    private int _f32CacheMaxSeqLen;        // current allocated capacity
    private int _f32CacheCurrentLength;    // mirrors IKvCache.CurrentLength for the model-internal cache

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — matches the sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private CudaQwen3MoeHybridTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        DeviceLayer[] layers,
        nint tokenEmbedDevice, QuantizationType tokenEmbedQt, long tokenEmbedBytes,
        nint embedF32Device, bool ownsEmbedF32,
        nint outputNormDevice,
        nint outputDevice, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        bool ownsOutputDevice,
        int[] kvSlotForLayer, int attentionLayerCount,
        float ropeTheta, int ropeDim,
        CudaQwen3MoeHybridForwardState state, CudaGdnStateCache gdnCache,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context, CudaKernels kernels,
        int deviceId,
        long dequantScratchElems, nint dequantScratchDevice,
        CudaMoeScratch moeScratch)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _tokenEmbedDevice = tokenEmbedDevice;
        _tokenEmbedQt = tokenEmbedQt;
        _tokenEmbedBytes = tokenEmbedBytes;
        _embedF32Device = embedF32Device;
        _ownsEmbedF32 = ownsEmbedF32;
        _outputNormDevice = outputNormDevice;
        _outputDevice = outputDevice;
        _outputQt = outputQt;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _ownsOutputDevice = ownsOutputDevice;
        _layout = config.HybridLayout!;
        _gdn = config.GdnConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _state = state;
        _gdnCache = gdnCache;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _deviceId = deviceId;
        _dequantScratchElems = dequantScratchElems;
        _dequantScratchF16Weight = dequantScratchDevice;
        _moeScratch = moeScratch;

        _gdnLayerOrdinal = new int[config.NumLayers];
        int gdnOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _gdnLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet
                ? gdnOrdinal++
                : -1;
        }
    }

    /// <summary>
    /// Loads a Qwen3MoeHybrid model from an opened GGUF file onto the given CUDA device.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for the model's lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX. Null auto-detects.</param>
    public static CudaQwen3MoeHybridTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"CudaQwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout!;
        int hiddenSize = config.HiddenSize;

        // ── Token embedding ──
        var embDesc = tensors["token_embd.weight"];
        long embRowBytes = Dequantize.RowByteSize(hiddenSize, embDesc.QuantizationType);
        long embTotalBytes = embRowBytes * config.VocabSize;
        nint tokenEmbedDevice = AllocDevice(embTotalBytes);
        CopyHtoD(tokenEmbedDevice, dataBase + (nint)embDesc.DataOffset, embTotalBytes);

        // If the embed quant type is not directly supported by LaunchEmbeddingLookupF32
        // (F32 / F16 / Q8_0 only), pre-dequant the whole table to a model-owned F32 buffer.
        // The Q6_K_XL variant ships the embed at Q6_K, so this path is the common one for
        // the target checkpoint.
        nint embedF32Device = 0;
        bool ownsEmbedF32 = false;
        bool needsEmbedF32 =
            embDesc.QuantizationType != QuantizationType.F32 &&
            embDesc.QuantizationType != QuantizationType.F16 &&
            embDesc.QuantizationType != QuantizationType.Q8_0;
        if (needsEmbedF32)
        {
            long totalElems = (long)config.VocabSize * hiddenSize;
            embedF32Device = AllocDevice(totalElems * sizeof(float));
            ownsEmbedF32 = true;
            // Host-side full-table dequant then H2D — once per load.
            float[] embedF32Host = new float[totalElems];
            Dequantize.ToFloat32(dataBase + (nint)embDesc.DataOffset, totalElems,
                embDesc.QuantizationType, embedF32Host);
            fixed (float* pHost = embedF32Host)
            {
                CopyHtoD(embedF32Device, (nint)pHost, totalElems * sizeof(float));
            }
        }

        // ── Output norm (always F32 [hiddenSize], dequant on host then H2D) ──
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormHost = new float[hiddenSize];
        Dequantize.ToFloat32(dataBase + (nint)outNormDesc.DataOffset, hiddenSize,
            outNormDesc.QuantizationType, outputNormHost);
        nint outputNormDevice = AllocDevice((long)hiddenSize * sizeof(float));
        fixed (float* p = outputNormHost)
        {
            CopyHtoD(outputNormDevice, (nint)p, (long)hiddenSize * sizeof(float));
        }

        // ── lm_head ──
        // When "output.weight" is missing, the model ties the lm_head with the token
        // embedding (Qwen3 follows this convention for most checkpoints). In that case
        // the device pointer is aliased — no separate upload, and Dispose() must not
        // free it twice.
        nint outputDevice;
        QuantizationType outputQt;
        int outputOutputDim;
        int outputInputDim;
        bool ownsOutputDevice;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            long outRowBytes = Dequantize.RowByteSize(outDesc.Shape[0], outDesc.QuantizationType);
            long outTotalBytes = outRowBytes * outDesc.Shape[1];
            outputDevice = AllocDevice(outTotalBytes);
            CopyHtoD(outputDevice, dataBase + (nint)outDesc.DataOffset, outTotalBytes);
            outputQt = outDesc.QuantizationType;
            outputInputDim = outDesc.Shape[0];
            outputOutputDim = outDesc.Shape[1];
            ownsOutputDevice = true;
        }
        else
        {
            outputDevice = tokenEmbedDevice;
            outputQt = embDesc.QuantizationType;
            outputInputDim = embDesc.Shape[0];
            outputOutputDim = embDesc.Shape[1];
            ownsOutputDevice = false;
        }

        // ── RoPE config ──
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        // ── Per-layer load ──
        var layers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        var owned = new List<nint>(); // CPU-side mmap pointers the GGUF loader may own
        long maxTileFloats = 0;

        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayerDevice(i, dataBase, tensors, config, owned, ref maxTileFloats);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        // Largest weight tile we ever GEMM with. Used as the F16 weight dequant scratch
        // for the prefill (seqLen > 1) cuBLAS HGEMM path. The lm_head dominates
        // (vocab × hidden), then the routed-expert dequant which has its own scratch in
        // _state. Decode (seqLen == 1) bypasses this scratch entirely via the quantised
        // GEMV kernels (LaunchQuantizedGemv / LaunchQuantizedGemvMmq /
        // LaunchQuantizedGemvF32In).
        maxTileFloats = Math.Max(maxTileFloats, (long)outputOutputDim * outputInputDim);
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        // ── GDN ordinal count + state cache + scratch state ──
        int gdnLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet) gdnLayerCount++;

        var gdn = config.GdnConfig!.Value;
        var moe = config.Moe!;
        // allocFullExpertDequantScratch:false — the GPU dispatcher dequants per-active-expert
        // on each MoE forward via CudaMoeFfn (grouped GEMV when available). Pre-allocating the
        // full [3 × numExperts × intermediate × hidden] F32 scratch alone is ~3.2 GiB at
        // qwen35moe shapes and OOMs any sub-A6000 GPU. Leave the scratch fields at zero.
        var state = new CudaQwen3MoeHybridForwardState(
            hiddenSize: hiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * gdn.NKHead + gdn.NVHead) * gdn.DState,
            dConv: gdn.DConv,
            nVHead: gdn.NVHead,
            nKHead: gdn.NKHead,
            dState: gdn.DState,
            moeNumExperts: moe.NumExperts,
            moeIntermediate: moe.MoeIntermediateSize,
            allocFullExpertDequantScratch: false);
        var gdnCache = new CudaGdnStateCache(gdn, gdnLayerCount);
        var moeScratch = new CudaMoeScratch();

        return new CudaQwen3MoeHybridTransformerModel(
            config, gguf, layers,
            tokenEmbedDevice, embDesc.QuantizationType, embTotalBytes,
            embedF32Device, ownsEmbedF32,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, gdnCache, stream, cublas, context, kernels, deviceId,
            maxTileFloats, dequantScratchDevice, moeScratch);
    }

    /// <summary>
    /// Builds a Qwen3MoeHybrid CUDA model in <i>layer-by-layer harness mode</i>: all
    /// global state (token embedding, output norm, lm_head, RoPE config, F32 activation
    /// scratch, GDN state cache, dequant scratch sized for the widest weight tile across
    /// the model) is uploaded, but every per-layer
    /// <see cref="DeviceLayer"/> slot is left zeroed. Per-layer weights are then streamed in
    /// and out of device memory one layer at a time via
    /// <see cref="LoadSingleLayerWeightsFromGguf"/> /
    /// <see cref="FreeSingleLayerWeights"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Purpose: enables CPU↔CUDA per-layer parity testing against the full real GGUF
    /// (~30 GiB Q6_K_XL) on a consumer 12 GiB GPU. At qwen35moe scale the routed-expert
    /// fused tensors dominate device residency (~6 GiB per layer); holding 40 layers on
    /// device at once is impossible. The harness loops one layer at a time:
    /// load → set input → run body → compare → free.
    /// </para>
    /// <para>
    /// <b>What is uploaded eagerly.</b> Everything that must remain alive across all 40
    /// layer iterations (token embedding for the layer-0 input shortcut, lm_head and
    /// output_norm for the projection sanity check, all activation scratch).
    /// <b>What is NOT uploaded.</b> Any <c>blk.NN.*</c> tensor — those flow through
    /// <see cref="LoadSingleLayerWeightsFromGguf"/>.
    /// </para>
    /// <para>
    /// <b>Dequant scratch sizing.</b> Walks every <c>blk.NN.*</c> projection descriptor up
    /// front and sizes <c>_dequantScratchF16Weight</c> to the widest tile encountered — so
    /// any subsequent per-layer load can GEMM (prefill HGEMM path) into a pre-allocated
    /// scratch without reallocation. Cost: <c>O(numTensors)</c> descriptor inspection, no
    /// byte uploads.
    /// </para>
    /// </remarks>
    /// <param name="gguf">Opened GGUF file. Must remain alive for the model's lifetime.</param>
    /// <param name="config">Extracted model configuration.</param>
    /// <param name="deviceId">CUDA device ordinal.</param>
    /// <param name="ptxDir">Optional PTX kernel directory override.</param>
    internal static CudaQwen3MoeHybridTransformerModel LoadFromGgufForLayerByLayerHarness(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"CudaQwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout!;
        int hiddenSize = config.HiddenSize;

        // ── Token embedding (same path as LoadFromGguf) ──
        var embDesc = tensors["token_embd.weight"];
        long embRowBytes = Dequantize.RowByteSize(hiddenSize, embDesc.QuantizationType);
        long embTotalBytes = embRowBytes * config.VocabSize;
        nint tokenEmbedDevice = AllocDevice(embTotalBytes);
        CopyHtoD(tokenEmbedDevice, dataBase + (nint)embDesc.DataOffset, embTotalBytes);

        nint embedF32Device = 0;
        bool ownsEmbedF32 = false;
        bool needsEmbedF32 =
            embDesc.QuantizationType != QuantizationType.F32 &&
            embDesc.QuantizationType != QuantizationType.F16 &&
            embDesc.QuantizationType != QuantizationType.Q8_0;
        if (needsEmbedF32)
        {
            long totalElems = (long)config.VocabSize * hiddenSize;
            embedF32Device = AllocDevice(totalElems * sizeof(float));
            ownsEmbedF32 = true;
            float[] embedF32Host = new float[totalElems];
            Dequantize.ToFloat32(dataBase + (nint)embDesc.DataOffset, totalElems,
                embDesc.QuantizationType, embedF32Host);
            fixed (float* pHost = embedF32Host)
            {
                CopyHtoD(embedF32Device, (nint)pHost, totalElems * sizeof(float));
            }
        }

        // ── Output norm ──
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormHost = new float[hiddenSize];
        Dequantize.ToFloat32(dataBase + (nint)outNormDesc.DataOffset, hiddenSize,
            outNormDesc.QuantizationType, outputNormHost);
        nint outputNormDevice = AllocDevice((long)hiddenSize * sizeof(float));
        fixed (float* p = outputNormHost)
        {
            CopyHtoD(outputNormDevice, (nint)p, (long)hiddenSize * sizeof(float));
        }

        // ── lm_head ──
        nint outputDevice;
        QuantizationType outputQt;
        int outputOutputDim;
        int outputInputDim;
        bool ownsOutputDevice;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            long outRowBytes = Dequantize.RowByteSize(outDesc.Shape[0], outDesc.QuantizationType);
            long outTotalBytes = outRowBytes * outDesc.Shape[1];
            outputDevice = AllocDevice(outTotalBytes);
            CopyHtoD(outputDevice, dataBase + (nint)outDesc.DataOffset, outTotalBytes);
            outputQt = outDesc.QuantizationType;
            outputInputDim = outDesc.Shape[0];
            outputOutputDim = outDesc.Shape[1];
            ownsOutputDevice = true;
        }
        else
        {
            outputDevice = tokenEmbedDevice;
            outputQt = embDesc.QuantizationType;
            outputInputDim = embDesc.Shape[0];
            outputOutputDim = embDesc.Shape[1];
            ownsOutputDevice = false;
        }

        // ── RoPE config (same as LoadFromGguf) ──
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        // ── Per-layer descriptor walk to compute max tile + kvSlot ordinals ──
        // No bytes uploaded; just metadata inspection. Mirrors what LoadGdnLayerDevice /
        // LoadFullAttnLayerDevice / UploadMoeLayer call UpdateMaxTile on, so a subsequent
        // per-layer load can GEMM into a pre-allocated scratch without resizing.
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        long maxTileFloats = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
            string prefix = $"blk.{i}";
            // Token-mixing tile sizes — GDN or full-attn.
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet)
            {
                foreach (string suffix in new[]
                {
                    "attn_qkv.weight", "attn_gate.weight",
                    "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
                })
                {
                    var d = tensors[$"{prefix}.{suffix}"];
                    UpdateMaxTile(ref maxTileFloats, (long)d.Shape[0] * d.Shape[1]);
                }
            }
            else
            {
                foreach (string suffix in new[]
                {
                    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                })
                {
                    var d = tensors[$"{prefix}.{suffix}"];
                    UpdateMaxTile(ref maxTileFloats, (long)d.Shape[0] * d.Shape[1]);
                }
            }
            // MoE tile sizes — per-expert routed gate/up/down, all with [I, hidden] or [hidden, I].
            // UploadMoeLayer only updates with W1 (I × hidden) and W2 (hidden × I); UploadMoeLayerFromHost
            // does the same. Mirror the smaller set so the budget matches the production loader exactly.
            if (tensors.TryGetValue($"{prefix}.ffn_gate_exps.weight", out var gateExps))
                UpdateMaxTile(ref maxTileFloats, (long)gateExps.Shape[0] * gateExps.Shape[1]);
            if (tensors.TryGetValue($"{prefix}.ffn_down_exps.weight", out var downExps))
                UpdateMaxTile(ref maxTileFloats, (long)downExps.Shape[0] * downExps.Shape[1]);
        }
        // Include the lm_head tile. Scratch holds the dequantised F16 weight tile for
        // the prefill HGEMM path; decode goes through the quantised GEMV kernels and
        // doesn't touch it. 2 bytes per element vs the previous 4-byte F32 layout.
        maxTileFloats = Math.Max(maxTileFloats, (long)outputOutputDim * outputInputDim);
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        // ── GDN ordinal count + state cache + activation scratch ──
        int gdnLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet) gdnLayerCount++;

        var gdn = config.GdnConfig!.Value;
        var moe = config.Moe!;
        var state = new CudaQwen3MoeHybridForwardState(
            hiddenSize: hiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * gdn.NKHead + gdn.NVHead) * gdn.DState,
            dConv: gdn.DConv,
            nVHead: gdn.NVHead,
            nKHead: gdn.NKHead,
            dState: gdn.DState,
            moeNumExperts: moe.NumExperts,
            moeIntermediate: moe.MoeIntermediateSize,
            allocFullExpertDequantScratch: false);
        var gdnCache = new CudaGdnStateCache(gdn, gdnLayerCount);
        var moeScratch = new CudaMoeScratch();

        // Per-layer device slots — left zeroed; populated lazily by
        // LoadSingleLayerWeightsFromGguf.
        var layers = new DeviceLayer[config.NumLayers];

        return new CudaQwen3MoeHybridTransformerModel(
            config, gguf, layers,
            tokenEmbedDevice, embDesc.QuantizationType, embTotalBytes,
            embedF32Device, ownsEmbedF32,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, gdnCache, stream, cublas, context, kernels, deviceId,
            maxTileFloats, dequantScratchDevice, moeScratch);
    }

    /// <summary>
    /// Loads the weight bundle for layer <paramref name="layerIdx"/> into the
    /// <see cref="DeviceLayer"/> slot held by this model. The model must have been built via
    /// <see cref="LoadFromGgufForLayerByLayerHarness"/>; the slot must be empty (every device
    /// pointer in <see cref="DeviceLayer"/> is zero, the state immediately after harness
    /// construction or after <see cref="FreeSingleLayerWeights"/>).
    /// </summary>
    /// <param name="layerIdx">Zero-based absolute layer index.</param>
    internal void LoadSingleLayerWeightsFromGguf(int layerIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_gguf is null)
            throw new InvalidOperationException(
                "LoadSingleLayerWeightsFromGguf requires the model to have been built via " +
                "LoadFromGgufForLayerByLayerHarness (with a retained GgufFile).");
        if ((uint)layerIdx >= (uint)_layers.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));
        if (_layers[layerIdx].AttnNormWeightDevice != 0)
            throw new InvalidOperationException(
                $"Layer {layerIdx} is already loaded. Call FreeSingleLayerWeights({layerIdx}) first.");

        _context.MakeCurrent();
        nint dataBase = _gguf.DataBasePointer;
        var tensors = _gguf.TensorsByName;
        long maxTileFloats = _dequantScratchElems; // already sized to the global max
        _layers[layerIdx] = LoadLayerDevice(layerIdx, dataBase, tensors, Config,
            owned: new List<nint>(), ref maxTileFloats);
    }

    /// <summary>
    /// Frees the device weight bundle held for layer <paramref name="layerIdx"/> and
    /// zeroes its <see cref="DeviceLayer"/> slot, releasing the routed-expert raw-quant
    /// allocations (which dominate per-layer device footprint at Q6_K_XL scale).
    /// </summary>
    /// <param name="layerIdx">Zero-based absolute layer index.</param>
    internal void FreeSingleLayerWeights(int layerIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layerIdx >= (uint)_layers.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));

        _context.MakeCurrent();
        FreeLayer(ref _layers[layerIdx]);
        // FreeLayer leaves the struct fields zeroed (every FreeIfNonZero sets ref to 0)
        // but doesn't reset the optional sub-structs (Gdn / FullAttn). Clear them so a
        // subsequent LoadSingleLayerWeightsFromGguf can detect an empty slot.
        _layers[layerIdx] = default;
    }

    /// <summary>
    /// Runs a single layer (<paramref name="layerIdx"/>) of the Qwen3MoeHybrid block in
    /// isolation: copies <paramref name="hostInput"/> (a CPU-captured hidden-state row
    /// trace, shape <c>[seqLen, hiddenSize]</c>, F32 row-major) into device hidden state,
    /// runs <see cref="RunSingleLayerBody"/> for the given layer, and copies the result
    /// back into <paramref name="hostOutput"/>. The GDN recurrent state for this layer's
    /// ordinal is zeroed before the call so the run starts from a clean state — matching
    /// the start of a CPU full-forward where every layer's GDN state begins at zero.
    /// </summary>
    /// <param name="layerIdx">Zero-based absolute layer index.</param>
    /// <param name="hostInput">Input hidden state <c>[seqLen, hiddenSize]</c> F32 row-major.</param>
    /// <param name="positions">Per-token absolute positions (length seqLen).</param>
    /// <param name="kvCache">Optional KV cache. Pass <c>null</c> for the no-cache prefill fast path — recommended for parity testing to dodge the F32 KV cache sidecar.</param>
    /// <param name="hostOutput">Output hidden state <c>[seqLen, hiddenSize]</c> F32 row-major. Filled on return.</param>
    internal void RunIsolatedLayerFromHostInput(int layerIdx,
        ReadOnlySpan<float> hostInput, ReadOnlySpan<int> positions, IKvCache? kvCache,
        Span<float> hostOutput)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layerIdx >= (uint)_layers.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));
        if (_layers[layerIdx].AttnNormWeightDevice == 0)
            throw new InvalidOperationException(
                $"Layer {layerIdx} weights are not loaded. Call LoadSingleLayerWeightsFromGguf first.");

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int seqLen = positions.Length;
        long hiddenElems = (long)seqLen * hiddenSize;

        if (hostInput.Length != hiddenElems)
            throw new ArgumentException(
                $"hostInput length {hostInput.Length} != seqLen*hiddenSize = {hiddenElems}.",
                nameof(hostInput));
        if (hostOutput.Length != hiddenElems)
            throw new ArgumentException(
                $"hostOutput length {hostOutput.Length} != seqLen*hiddenSize = {hiddenElems}.",
                nameof(hostOutput));

        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint streamH = _stream.Handle;
        long hiddenBytes = hiddenElems * sizeof(float);

        // Reset the GDN state for this layer's ordinal — only matters for GDN layers, but
        // calling Reset() (which zeros all layers) is fine because the harness runs one
        // layer at a time and never depends on cross-call state.
        _gdnCache.Reset();

        // H2D copy hidden input.
        fixed (float* pIn = hostInput)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.HiddenState, (nint)pIn,
                (nuint)hiddenBytes, streamH).ThrowOnError();
        }
        // H2D copy positions (required by full-attn RoPE kernel).
        fixed (int* pPos = positions)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.PositionsDevice, (nint)pPos,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }

        RunSingleLayerBody(layerIdx, seqLen, positions, hiddenSize,
            numHeads, numKvHeads, headDim, eps, kvCache);

        // D2H copy hidden output.
        _stream.Synchronize();
        fixed (float* pOut = hostOutput)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pOut, _state.HiddenState,
                (nuint)hiddenBytes).ThrowOnError();
        }
    }

    /// <summary>
    /// Applies the final RMSNorm + lm_head projection to a CPU-captured hidden state and
    /// returns the resulting logits. Used by the layer-by-layer harness to verify the
    /// output stage in isolation. Matches the tail of
    /// <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>
    /// after the per-layer loop.
    /// </summary>
    /// <param name="hostInputHidden">Input hidden state <c>[seqLen, hiddenSize]</c> F32.</param>
    /// <param name="seqLen">Number of rows in <paramref name="hostInputHidden"/>.</param>
    /// <param name="hostOutputLogits">Output logits <c>[seqLen, vocabSize]</c> F32. Filled on return.</param>
    internal void RunOutputProjectionFromHostInput(
        ReadOnlySpan<float> hostInputHidden, int seqLen, Span<float> hostOutputLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        long hiddenElems = (long)seqLen * hiddenSize;
        long logitElems = (long)seqLen * vocabSize;
        if (hostInputHidden.Length != hiddenElems)
            throw new ArgumentException(
                $"hostInputHidden length {hostInputHidden.Length} != seqLen*hiddenSize = {hiddenElems}.",
                nameof(hostInputHidden));
        if (hostOutputLogits.Length != logitElems)
            throw new ArgumentException(
                $"hostOutputLogits length {hostOutputLogits.Length} != seqLen*vocabSize = {logitElems}.",
                nameof(hostOutputLogits));

        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint streamH = _stream.Handle;
        long hiddenBytes = hiddenElems * sizeof(float);

        fixed (float* pIn = hostInputHidden)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.HiddenState, (nint)pIn,
                (nuint)hiddenBytes, streamH).ThrowOnError();
        }
        _kernels.LaunchRmsNormF32(_state.HiddenState, _outputNormDevice, _state.HiddenState,
            hiddenSize, eps, seqLen, streamH);
        Gemm(_outputDevice, _outputQt, _state.HiddenState, _state.Logits,
             _outputOutputDim, _outputInputDim, seqLen);

        _stream.Synchronize();
        fixed (float* pOut = hostOutputLogits)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pOut, _state.Logits,
                (nuint)(logitElems * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>
    /// Reads the device's current free / total memory via <c>cuMemGetInfo</c>. Used by the
    /// layer-by-layer harness for VRAM bookkeeping at layer boundaries.
    /// </summary>
    /// <returns>(usedBytes, totalBytes) on the current CUDA device.</returns>
    internal (long Used, long Total) GetDeviceMemoryUsage()
    {
        _context.MakeCurrent();
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total).ThrowOnError();
        return ((long)(total - free), (long)total);
    }

    /// <summary>
    /// Embedding lookup helper for the layer-by-layer harness: copies the F32 embedding
    /// rows for the given token IDs into <paramref name="hostOutput"/>. Mirrors the first
    /// step of <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>
    /// so the harness can compare CUDA's layer-0 input against the CPU dump <c>token_embd</c>.
    /// </summary>
    /// <param name="tokenIds">Token IDs to look up.</param>
    /// <param name="hostOutput">Output buffer <c>[tokenIds.Length, hiddenSize]</c> F32.</param>
    internal void LookupEmbeddingsToHost(ReadOnlySpan<int> tokenIds, Span<float> hostOutput)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int hiddenSize = Config.HiddenSize;
        int seqLen = tokenIds.Length;
        long elems = (long)seqLen * hiddenSize;
        if (hostOutput.Length != elems)
            throw new ArgumentException(
                $"hostOutput length {hostOutput.Length} != seqLen*hiddenSize = {elems}.",
                nameof(hostOutput));

        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint streamH = _stream.Handle;
        fixed (int* pIds = tokenIds)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.TokenIdsDevice, (nint)pIds,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }
        if (_embedF32Device != 0)
        {
            _kernels.LaunchEmbeddingLookupF32(_embedF32Device, QuantizationType.F32,
                _state.TokenIdsDevice, _state.HiddenState, seqLen, hiddenSize, streamH);
        }
        else
        {
            _kernels.LaunchEmbeddingLookupF32(_tokenEmbedDevice, _tokenEmbedQt,
                _state.TokenIdsDevice, _state.HiddenState, seqLen, hiddenSize, streamH);
        }

        _stream.Synchronize();
        fixed (float* pOut = hostOutput)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pOut, _state.HiddenState,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>
    /// Builds a CUDA Qwen3MoeHybrid model from caller-owned, pre-dequantised F32 host weight
    /// pointers — the CUDA parallel of
    /// <see cref="Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(ModelConfig, Qwen3MoeLayerWeights[], float[], nint, QuantizationType, nint, QuantizationType, int, int)"/>.
    /// Uploads every weight to device memory and returns a fully self-contained model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Ownership.</b> Unlike <see cref="LoadFromGguf"/> which mmap-aliases the GGUF
    /// data region, this constructor <i>copies</i> every host buffer onto the device.
    /// The returned model OWNS every device allocation and frees them in
    /// <see cref="Dispose"/>. Host-side ownership of the input pointers stays with the
    /// caller — they may be freed as soon as this method returns.
    /// </para>
    /// <para>
    /// <b>F32 throughout.</b> Every projection is uploaded as raw F32 row-major bytes;
    /// <see cref="BuildRoutedMoeLayerWeights"/> automatically selects
    /// <see cref="MoePrecision.F32"/> when the routed-expert quant types are F32, avoiding
    /// the lossy F32→F16→F32 round-trip the quantized dispatch would otherwise apply.
    /// </para>
    /// <para>
    /// <b>Routed-expert layout.</b> The synthetic CPU fixture stores per-expert
    /// projections as <c>nint[]</c> of discontiguous F32 allocations (one per expert).
    /// We pack each expert's bytes contiguously into ONE fused device buffer per
    /// projection, in expert-major order, matching the GGUF mmap stride convention that
    /// <see cref="BuildRoutedMoeLayerWeights"/> already consumes (per-expert offset
    /// <c>= e * RowByteSize(M*K, F32) = e * M * K * 4</c>).
    /// </para>
    /// </remarks>
    /// <param name="config">Model configuration. Must have <see cref="Architecture.Qwen3MoeHybrid"/>.</param>
    /// <param name="layers">Per-layer weight bundle, length must equal <c>config.NumLayers</c>.</param>
    /// <param name="outputNormWeight">Final RMSNorm gain [hiddenSize] F32.</param>
    /// <param name="tokenEmbedWeight">Host F32 [vocab, hidden] token-embedding table.</param>
    /// <param name="tokenEmbedQuantType">Must be <see cref="QuantizationType.F32"/>.</param>
    /// <param name="outputWeight">Host F32 [vocab, hidden] lm_head — may alias <paramref name="tokenEmbedWeight"/> for tied embeddings.</param>
    /// <param name="outputQuantType">Must be <see cref="QuantizationType.F32"/>.</param>
    /// <param name="outputOutputDim">lm_head output dim (vocab size).</param>
    /// <param name="outputInputDim">lm_head input dim (hidden size).</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX. Null auto-detects.</param>
    internal static CudaQwen3MoeHybridTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        Qwen3MoeLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);
        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"CudaQwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));
        if (layers.Length != config.NumLayers)
            throw new ArgumentException(
                $"layers length {layers.Length} != config.NumLayers {config.NumLayers}.", nameof(layers));
        if (tokenEmbedQuantType != QuantizationType.F32)
            throw new ArgumentException(
                $"BuildFromPrebuiltWeights expects F32 tokenEmbedQuantType, got {tokenEmbedQuantType}.",
                nameof(tokenEmbedQuantType));
        if (outputQuantType != QuantizationType.F32)
            throw new ArgumentException(
                $"BuildFromPrebuiltWeights expects F32 outputQuantType, got {outputQuantType}.",
                nameof(outputQuantType));

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        var layout = config.HybridLayout!;
        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        // ── Token embedding (F32 host → F32 device) ──
        long embTotalBytes = (long)vocabSize * hiddenSize * sizeof(float);
        nint tokenEmbedDevice = AllocDevice(embTotalBytes);
        CopyHtoD(tokenEmbedDevice, tokenEmbedWeight, embTotalBytes);

        // ── Output norm — F32 [hiddenSize] from managed array → device ──
        nint outputNormDevice = AllocDevice((long)hiddenSize * sizeof(float));
        unsafe
        {
            fixed (float* p = outputNormWeight)
            {
                CopyHtoD(outputNormDevice, (nint)p, (long)hiddenSize * sizeof(float));
            }
        }

        // ── lm_head ──
        // Detect aliasing: when the caller passed the same host pointer for embed and
        // lm_head we mirror the GGUF tied-embedding path (no second device alloc), so
        // Dispose() doesn't double-free.
        nint outputDevice;
        bool ownsOutputDevice;
        if (outputWeight == tokenEmbedWeight)
        {
            outputDevice = tokenEmbedDevice;
            ownsOutputDevice = false;
        }
        else
        {
            long outBytes = (long)outputOutputDim * outputInputDim * sizeof(float);
            outputDevice = AllocDevice(outBytes);
            CopyHtoD(outputDevice, outputWeight, outBytes);
            ownsOutputDevice = true;
        }

        // ── RoPE config (mirrors LoadFromGguf) ──
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new ArgumentException(
                $"rope_dim={ropeDim} must be even for pair-wise rotation.", nameof(config));
        if (ropeDim > config.HeadDim)
            throw new ArgumentException(
                $"rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.", nameof(config));
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        // ── Per-layer device upload (host pointers → fresh device allocations) ──
        var deviceLayers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        long maxTileFloats = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            deviceLayers[i] = UploadLayerFromHost(i, layers[i], config, ref maxTileFloats);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        // Account for the lm_head tile in the dequant scratch sizing. F16 element width
        // (2 bytes) — see _dequantScratchF16Weight field comment for the rewire rationale.
        maxTileFloats = Math.Max(maxTileFloats, (long)outputOutputDim * outputInputDim);
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        // ── GDN ordinal count + state cache + scratch state ──
        int gdnLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet) gdnLayerCount++;

        var gdn = config.GdnConfig!.Value;
        var moe = config.Moe!;
        var state = new CudaQwen3MoeHybridForwardState(
            hiddenSize: hiddenSize,
            vocabSize: vocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * gdn.NKHead + gdn.NVHead) * gdn.DState,
            dConv: gdn.DConv,
            nVHead: gdn.NVHead,
            nKHead: gdn.NKHead,
            dState: gdn.DState,
            moeNumExperts: moe.NumExperts,
            moeIntermediate: moe.MoeIntermediateSize,
            allocFullExpertDequantScratch: false);
        var gdnCache = new CudaGdnStateCache(gdn, gdnLayerCount);
        var moeScratch = new CudaMoeScratch();

        return new CudaQwen3MoeHybridTransformerModel(
            config, gguf: null, deviceLayers,
            tokenEmbedDevice, QuantizationType.F32, embTotalBytes,
            embedF32Device: 0, ownsEmbedF32: false,
            outputNormDevice,
            outputDevice, QuantizationType.F32, outputOutputDim, outputInputDim, ownsOutputDevice,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, gdnCache, stream, cublas, context, kernels, deviceId,
            maxTileFloats, dequantScratchDevice, moeScratch);
    }

    /// <summary>
    /// Uploads one <see cref="Qwen3MoeLayerWeights"/> bundle to device memory and returns
    /// a fully-populated <see cref="DeviceLayer"/>. Mirrors <see cref="LoadLayerDevice"/>
    /// but reads from caller-owned host F32 pointers instead of an mmap'd GGUF region.
    /// </summary>
    private static DeviceLayer UploadLayerFromHost(
        int layerIdx, Qwen3MoeLayerWeights host,
        ModelConfig config, ref long maxTileFloats)
    {
        int hiddenSize = config.HiddenSize;
        var layout = config.HybridLayout!;

        nint attnNormDevice = UploadF32Array(host.AttnNormWeight);
        nint postAttnNormDevice = UploadF32Array(host.PostAttnNormWeight);

        DeviceGdn? gdnDev = null;
        DeviceFullAttn? attnDev = null;
        switch (layout.LayerKind[layerIdx])
        {
            case HybridLayerKind.GatedDeltaNet:
                if (host.Gdn is null)
                    throw new ArgumentException(
                        $"Layer {layerIdx} is GDN in HybridLayout but Qwen3MoeLayerWeights.Gdn is null.",
                        nameof(host));
                gdnDev = UploadGdnLayer(host.Gdn, ref maxTileFloats);
                break;
            case HybridLayerKind.Attention:
                if (host.FullAttn is null)
                    throw new ArgumentException(
                        $"Layer {layerIdx} is Attention in HybridLayout but Qwen3MoeLayerWeights.FullAttn is null.",
                        nameof(host));
                attnDev = UploadFullAttnLayer(host.FullAttn, ref maxTileFloats);
                break;
            default:
                throw new InvalidOperationException(
                    $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3MoeHybrid.");
        }

        DeviceMoe moeDev = UploadMoeLayerFromHost(host.Moe, hiddenSize, ref maxTileFloats);

        return new DeviceLayer
        {
            AttnNormWeightDevice = attnNormDevice,
            PostAttnNormWeightDevice = postAttnNormDevice,
            Gdn = gdnDev,
            FullAttn = attnDev,
            Moe = moeDev,
            // MoeHost is kept for descriptor metadata only — the actual weight bytes live on
            // device. Hand back the same host bundle the caller built so any downstream
            // diagnostic introspection sees consistent dim metadata.
            MoeHost = host.Moe,
        };
    }

    private static DeviceGdn UploadGdnLayer(GdnTokenMixingWeights gdn, ref long maxTileFloats)
    {
        // Quantised-projection upload: each projection's raw bytes already lay out in the
        // declared quant format ([M*K] elements packed via Dequantize.RowByteSize). The
        // Gemm() dispatcher reads QkvQt / GateQt / ... and routes through the matching
        // CUDA branch (decode-direct quantised GEMV, prefill F16-dequant + cuBLAS HGEMM,
        // or cuBLAS LinearF32 for the F32 fast path).
        nint qkvDevice = UploadProjectionPtr(gdn.QkvWeight, gdn.QkvOutputDim, gdn.QkvInputDim, gdn.QkvQuantType);
        nint gateDevice = UploadProjectionPtr(gdn.GateWeight, gdn.GateOutputDim, gdn.GateInputDim, gdn.GateQuantType);
        nint alphaDevice = UploadProjectionPtr(gdn.AlphaWeight, gdn.AlphaOutputDim, gdn.AlphaInputDim, gdn.AlphaQuantType);
        nint betaDevice = UploadProjectionPtr(gdn.BetaWeight, gdn.BetaOutputDim, gdn.BetaInputDim, gdn.BetaQuantType);
        nint outDevice = UploadProjectionPtr(gdn.OutWeight, gdn.OutOutputDim, gdn.OutInputDim, gdn.OutQuantType);

        nint conv1dWeightDevice = UploadF32Array(gdn.Conv1dWeight);
        nint conv1dBiasDevice = UploadF32Array(gdn.Conv1dBias);
        nint aDevice = UploadF32Array(gdn.A);
        nint dtBiasDevice = UploadF32Array(gdn.DtBias);
        nint ssmNormDevice = UploadF32Array(gdn.SsmNormWeight);

        UpdateMaxTile(ref maxTileFloats, (long)gdn.QkvInputDim * gdn.QkvOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)gdn.GateInputDim * gdn.GateOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)gdn.AlphaInputDim * gdn.AlphaOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)gdn.BetaInputDim * gdn.BetaOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)gdn.OutInputDim * gdn.OutOutputDim);

        return new DeviceGdn
        {
            QkvDevice = qkvDevice, QkvQt = gdn.QkvQuantType,
            QkvInputDim = gdn.QkvInputDim, QkvOutputDim = gdn.QkvOutputDim,

            GateDevice = gateDevice, GateQt = gdn.GateQuantType,
            GateInputDim = gdn.GateInputDim, GateOutputDim = gdn.GateOutputDim,

            AlphaDevice = alphaDevice, AlphaQt = gdn.AlphaQuantType,
            AlphaInputDim = gdn.AlphaInputDim, AlphaOutputDim = gdn.AlphaOutputDim,

            BetaDevice = betaDevice, BetaQt = gdn.BetaQuantType,
            BetaInputDim = gdn.BetaInputDim, BetaOutputDim = gdn.BetaOutputDim,

            Conv1dWeightDevice = conv1dWeightDevice,
            Conv1dBiasDevice = conv1dBiasDevice,
            ADevice = aDevice,
            DtBiasDevice = dtBiasDevice,
            SsmNormDevice = ssmNormDevice,

            OutDevice = outDevice, OutQt = gdn.OutQuantType,
            OutInputDim = gdn.OutInputDim, OutOutputDim = gdn.OutOutputDim,
        };
    }

    private static DeviceFullAttn UploadFullAttnLayer(Qwen3FullAttnWeights attn, ref long maxTileFloats)
    {
        // Quant-aware upload: see UploadGdnLayer for rationale; the QQt/KQt/VQt/OQt fields
        // drive Gemm() dispatch in the per-layer body.
        nint qDevice = UploadProjectionPtr(attn.QWeight, attn.QOutputDim, attn.QInputDim, attn.QQuantType);
        nint kDevice = UploadProjectionPtr(attn.KWeight, attn.KOutputDim, attn.KInputDim, attn.KQuantType);
        nint vDevice = UploadProjectionPtr(attn.VWeight, attn.VOutputDim, attn.VInputDim, attn.VQuantType);
        nint oDevice = UploadProjectionPtr(attn.OWeight, attn.OOutputDim, attn.OInputDim, attn.OQuantType);

        nint qNormDevice = UploadF32Array(attn.QNormWeight);
        nint kNormDevice = UploadF32Array(attn.KNormWeight);

        UpdateMaxTile(ref maxTileFloats, (long)attn.QInputDim * attn.QOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)attn.KInputDim * attn.KOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)attn.VInputDim * attn.VOutputDim);
        UpdateMaxTile(ref maxTileFloats, (long)attn.OInputDim * attn.OOutputDim);

        return new DeviceFullAttn
        {
            QDevice = qDevice, QQt = attn.QQuantType,
            QInputDim = attn.QInputDim, QOutputDim = attn.QOutputDim,

            KDevice = kDevice, KQt = attn.KQuantType,
            KInputDim = attn.KInputDim, KOutputDim = attn.KOutputDim,

            VDevice = vDevice, VQt = attn.VQuantType,
            VInputDim = attn.VInputDim, VOutputDim = attn.VOutputDim,

            ODevice = oDevice, OQt = attn.OQuantType,
            OInputDim = attn.OInputDim, OOutputDim = attn.OOutputDim,

            NumKvHeads = attn.NumKvHeads,
            QNormDevice = qNormDevice,
            KNormDevice = kNormDevice,
        };
    }

    /// <summary>
    /// Packs per-expert discontiguous F32 host pointers (one nint per expert) into a
    /// single contiguous device buffer per projection, matching the GGUF fused-expert
    /// stride convention <see cref="BuildRoutedMoeLayerWeights"/> consumes. F32 router
    /// gate, shared-expert F32 projections, and the optional shared-expert sigmoid gate
    /// are uploaded one-shot from their managed-array hosts.
    /// </summary>
    private static DeviceMoe UploadMoeLayerFromHost(MoeLayerWeights moe, int hiddenSize, ref long maxTileFloats)
    {
        int E = moe.NumExperts;
        int I = moe.IntermediateSize;
        if (moe.W1.Length != E || moe.W2.Length != E || moe.W3.Length != E)
            throw new ArgumentException(
                $"MoeLayerWeights expert pointer arrays must each have NumExperts={E} entries " +
                $"(W1={moe.W1.Length}, W2={moe.W2.Length}, W3={moe.W3.Length}).", nameof(moe));

        // Router gate — F32 [E, hidden] managed array.
        nint gateRouterDevice = UploadF32Array(moe.Gate);

        // Routed experts — concatenate W1/W3 ([I, hidden]) and W2 ([hidden, I]) into fused
        // device buffers. Per-expert byte stride = M*K*4 (F32). The forward path in
        // BuildRoutedMoeLayerWeights pulls per-expert pointers at e * stride.
        // W1 = gate proj, W3 = up proj, W2 = down proj (verified against the CPU oracle's
        // MoeSwiGluMlp.ExecuteRoutedFromAssignments mapping at
        // Qwen3MoeHybridTransformerModel.cs:1006-1008).
        long gateBytesPerExpert = (long)I * hiddenSize * sizeof(float);
        long upBytesPerExpert = gateBytesPerExpert;
        long downBytesPerExpert = (long)hiddenSize * I * sizeof(float);

        nint gateExpsDevice = AllocDevice((long)E * gateBytesPerExpert);
        nint upExpsDevice = AllocDevice((long)E * upBytesPerExpert);
        nint downExpsDevice = AllocDevice((long)E * downBytesPerExpert);
        for (int e = 0; e < E; e++)
        {
            CopyHtoD(gateExpsDevice + (nint)(e * gateBytesPerExpert), moe.W1[e], gateBytesPerExpert);
            CopyHtoD(upExpsDevice + (nint)(e * upBytesPerExpert), moe.W3[e], upBytesPerExpert);
            CopyHtoD(downExpsDevice + (nint)(e * downBytesPerExpert), moe.W2[e], downBytesPerExpert);
        }

        UpdateMaxTile(ref maxTileFloats, (long)I * hiddenSize);
        UpdateMaxTile(ref maxTileFloats, (long)hiddenSize * I);

        // Shared experts — F32 [sI, hidden] / [hidden, sI] managed pointers.
        int numShared = moe.NumSharedExperts;
        if (numShared > 1)
            throw new NotSupportedException(
                $"CUDA Qwen3MoeHybrid shared-expert path handles at most one shared expert; " +
                $"got {numShared}. Extend ForwardSharedExpertF32 to loop and accumulate.");
        nint[] sharedGateDevice = new nint[numShared];
        nint[] sharedUpDevice = new nint[numShared];
        nint[] sharedDownDevice = new nint[numShared];
        long sharedFloats = (long)moe.SharedIntermediateSize * hiddenSize;
        for (int s = 0; s < numShared; s++)
        {
            sharedGateDevice[s] = UploadF32Ptr(moe.SharedGateProj[s], sharedFloats);
            sharedUpDevice[s] = UploadF32Ptr(moe.SharedUpProj[s], sharedFloats);
            sharedDownDevice[s] = UploadF32Ptr(moe.SharedDownProj[s], sharedFloats);
        }

        nint sharedExpertGateDevice = 0;
        if (moe.SharedExpertGate is not null)
            sharedExpertGateDevice = UploadF32Array(moe.SharedExpertGate);

        return new DeviceMoe
        {
            GateRouterDevice = gateRouterDevice,
            NumExperts = E,
            NumExpertsPerTok = moe.NumExpertsPerTok,
            IntermediateSize = I,
            NormTopKProb = moe.NormTopKProb,

            GateExpsDevice = gateExpsDevice, GateExpsQt = QuantizationType.F32,
            GateExpsMDim = I, GateExpsKDim = hiddenSize,

            UpExpsDevice = upExpsDevice, UpExpsQt = QuantizationType.F32,
            UpExpsMDim = I, UpExpsKDim = hiddenSize,

            DownExpsDevice = downExpsDevice, DownExpsQt = QuantizationType.F32,
            DownExpsMDim = hiddenSize, DownExpsKDim = I,

            SharedGateDevice = sharedGateDevice,
            SharedUpDevice = sharedUpDevice,
            SharedDownDevice = sharedDownDevice,
            SharedIntermediateSize = moe.SharedIntermediateSize,
            SharedExpertGateDevice = sharedExpertGateDevice,
        };
    }

    /// <summary>Allocates a device F32 buffer and copies <paramref name="elemCount"/> floats from <paramref name="hostF32Ptr"/>.</summary>
    private static nint UploadF32Ptr(nint hostF32Ptr, long elemCount)
    {
        long bytes = elemCount * sizeof(float);
        nint device = AllocDevice(bytes);
        CopyHtoD(device, hostF32Ptr, bytes);
        return device;
    }

    /// <summary>
    /// Allocates a device buffer sized for an <c>[m, k]</c> row-major weight matrix in the
    /// declared quantisation format and copies the raw bytes from host. The byte count is
    /// <c>RowByteSize(k, qt) * m</c> — so K-quants (block size 256) and Q8_0 (block size 32)
    /// use their packed block byte layout, F16 uses 2 bytes/elem, F32 uses 4. The Gemm()
    /// dispatcher (see file header) then routes each call by the matching DeviceGdn /
    /// DeviceFullAttn QuantType field to the correct CUDA branch.
    /// </summary>
    /// <param name="hostPtr">Host base pointer of the row-major weight matrix, <c>RowByteSize(k, qt) * m</c> bytes.</param>
    /// <param name="m">Output dim (rows).</param>
    /// <param name="k">Input/contraction dim (columns per row of un-quantised view).</param>
    /// <param name="qt">Quantisation format of the row bytes.</param>
    private static nint UploadProjectionPtr(nint hostPtr, int m, int k, QuantizationType qt)
    {
        long bytes = Dequantize.RowByteSize(k, qt) * m;
        nint device = AllocDevice(bytes);
        CopyHtoD(device, hostPtr, bytes);
        return device;
    }

    /// <summary>Allocates a device F32 buffer and copies a managed <c>float[]</c> into it.</summary>
    private static nint UploadF32Array(float[] hostArray)
    {
        long bytes = (long)hostArray.Length * sizeof(float);
        nint device = AllocDevice(bytes);
        unsafe
        {
            fixed (float* p = hostArray)
            {
                CopyHtoD(device, (nint)p, bytes);
            }
        }
        return device;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-layer loaders (host → device upload of raw quant bytes)
    // ──────────────────────────────────────────────────────────────────────

    private static DeviceLayer LoadLayerDevice(
        int layerIdx, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, List<nint> owned, ref long maxTileFloats)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        var layout = config.HybridLayout!;

        // Norms — F32 [hiddenSize].
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        nint attnNormDevice = UploadF32Tensor(dataBase, attnNormDesc, hiddenSize);
        var postNormDesc = tensors[$"{prefix}.post_attention_norm.weight"];
        nint postAttnNormDevice = UploadF32Tensor(dataBase, postNormDesc, hiddenSize);

        DeviceGdn? gdnDev = null;
        DeviceFullAttn? attnDev = null;
        switch (layout.LayerKind[layerIdx])
        {
            case HybridLayerKind.GatedDeltaNet:
                gdnDev = LoadGdnLayerDevice(prefix, dataBase, tensors, config, ref maxTileFloats);
                break;
            case HybridLayerKind.Attention:
                attnDev = LoadFullAttnLayerDevice(prefix, dataBase, tensors, config,
                    layout.HeadCountKv[layerIdx], ref maxTileFloats);
                break;
            default:
                throw new InvalidOperationException(
                    $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3MoeHybrid.");
        }

        // Routed-expert raw-quant view + shared expert F32 weights (small).
        MoeLayerWeights moeHost = TransformerWeights.LoadDeepSeekMoeLayer(
            layerIdx, dataBase, tensors, config, owned, skipRoutedF32Only: true);
        DeviceMoe moeDev = UploadMoeLayer(moeHost, hiddenSize, ref maxTileFloats);

        return new DeviceLayer
        {
            AttnNormWeightDevice = attnNormDevice,
            PostAttnNormWeightDevice = postAttnNormDevice,
            Gdn = gdnDev,
            FullAttn = attnDev,
            Moe = moeDev,
            MoeHost = moeHost,
        };
    }

    private static DeviceGdn LoadGdnLayerDevice(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, ref long maxTileFloats)
    {
        var gdn = config.GdnConfig!.Value;
        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;

        var qkvDesc = tensors[$"{prefix}.attn_qkv.weight"];
        var gateDesc = tensors[$"{prefix}.attn_gate.weight"];
        var alphaDesc = tensors[$"{prefix}.ssm_alpha.weight"];
        var betaDesc = tensors[$"{prefix}.ssm_beta.weight"];
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
        var aDesc = tensors[$"{prefix}.ssm_a"];
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"];
        var ssmNormDesc = tensors[$"{prefix}.ssm_norm.weight"];
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];

        // Quantized projections — upload raw bytes.
        nint qkvDevice = UploadRawTensor(dataBase, qkvDesc);
        nint gateDevice = UploadRawTensor(dataBase, gateDesc);
        nint alphaDevice = UploadRawTensor(dataBase, alphaDesc);
        nint betaDevice = UploadRawTensor(dataBase, betaDesc);
        nint outDevice = UploadRawTensor(dataBase, outDesc);

        // Conv1d weight — F32 [DConv, convDim]; CPU oracle host-dequants then we H2D.
        nint conv1dWeightDevice = UploadF32Tensor(dataBase, conv1dWDesc, gdn.DConv * convDim);
        // Conv bias is zero-filled (GDN has no conv bias tensor).
        nint conv1dBiasDevice = AllocDevice((long)convDim * sizeof(float));
        CudaDriverApi.cuMemsetD8_v2(conv1dBiasDevice, 0, (nuint)((long)convDim * sizeof(float)))
            .ThrowOnError();

        // Small F32 scalars — A, dt_bias, ssm_norm.
        nint aDevice = UploadF32Tensor(dataBase, aDesc, gdn.NVHead);
        nint dtBiasDevice = UploadF32Tensor(dataBase, dtBDesc, gdn.NVHead);
        nint ssmNormDevice = UploadF32Tensor(dataBase, ssmNormDesc, gdn.DState);

        UpdateMaxTile(ref maxTileFloats, (long)qkvDesc.Shape[0] * qkvDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)gateDesc.Shape[0] * gateDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)alphaDesc.Shape[0] * alphaDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)betaDesc.Shape[0] * betaDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)outDesc.Shape[0] * outDesc.Shape[1]);

        return new DeviceGdn
        {
            QkvDevice = qkvDevice, QkvQt = qkvDesc.QuantizationType,
            QkvInputDim = qkvDesc.Shape[0], QkvOutputDim = qkvDesc.Shape[1],

            GateDevice = gateDevice, GateQt = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0], GateOutputDim = gateDesc.Shape[1],

            AlphaDevice = alphaDevice, AlphaQt = alphaDesc.QuantizationType,
            AlphaInputDim = alphaDesc.Shape[0], AlphaOutputDim = alphaDesc.Shape[1],

            BetaDevice = betaDevice, BetaQt = betaDesc.QuantizationType,
            BetaInputDim = betaDesc.Shape[0], BetaOutputDim = betaDesc.Shape[1],

            Conv1dWeightDevice = conv1dWeightDevice,
            Conv1dBiasDevice = conv1dBiasDevice,
            ADevice = aDevice,
            DtBiasDevice = dtBiasDevice,
            SsmNormDevice = ssmNormDevice,

            OutDevice = outDevice, OutQt = outDesc.QuantizationType,
            OutInputDim = outDesc.Shape[0], OutOutputDim = outDesc.Shape[1],
        };
    }

    private static DeviceFullAttn LoadFullAttnLayerDevice(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, int numKvHeads, ref long maxTileFloats)
    {
        var q = tensors[$"{prefix}.attn_q.weight"];
        var k = tensors[$"{prefix}.attn_k.weight"];
        var v = tensors[$"{prefix}.attn_v.weight"];
        var o = tensors[$"{prefix}.attn_output.weight"];

        int expectedQGateOut = 2 * config.NumAttentionHeads * config.HeadDim;
        if (q.Shape[1] != expectedQGateOut)
        {
            throw new InvalidDataException(
                $"{prefix}.attn_q.weight has output dim {q.Shape[1]} but qwen35moe expects " +
                $"{expectedQGateOut} = 2 * {config.NumAttentionHeads} * {config.HeadDim} (Q+Gate fused).");
        }

        nint qDevice = UploadRawTensor(dataBase, q);
        nint kDevice = UploadRawTensor(dataBase, k);
        nint vDevice = UploadRawTensor(dataBase, v);
        nint oDevice = UploadRawTensor(dataBase, o);

        nint qNormDevice = UploadF32Tensor(dataBase, tensors[$"{prefix}.attn_q_norm.weight"], config.HeadDim);
        nint kNormDevice = UploadF32Tensor(dataBase, tensors[$"{prefix}.attn_k_norm.weight"], config.HeadDim);

        UpdateMaxTile(ref maxTileFloats, (long)q.Shape[0] * q.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)k.Shape[0] * k.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)v.Shape[0] * v.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)o.Shape[0] * o.Shape[1]);

        return new DeviceFullAttn
        {
            QDevice = qDevice, QQt = q.QuantizationType,
            QInputDim = q.Shape[0], QOutputDim = q.Shape[1],

            KDevice = kDevice, KQt = k.QuantizationType,
            KInputDim = k.Shape[0], KOutputDim = k.Shape[1],

            VDevice = vDevice, VQt = v.QuantizationType,
            VInputDim = v.Shape[0], VOutputDim = v.Shape[1],

            ODevice = oDevice, OQt = o.QuantizationType,
            OInputDim = o.Shape[0], OOutputDim = o.Shape[1],

            NumKvHeads = numKvHeads,
            QNormDevice = qNormDevice,
            KNormDevice = kNormDevice,
        };
    }

    /// <summary>
    /// Uploads the per-layer MoE weights to device. Routed experts are stored as raw
    /// quant bytes (host mmap → device). Shared expert F32 dequants are uploaded as F32.
    /// The router gate.weight is F32 (already produced by the CPU loader).
    /// </summary>
    private static DeviceMoe UploadMoeLayer(MoeLayerWeights moe, int hiddenSize, ref long maxTileFloats)
    {
        // Router gate weight [numExperts, hiddenSize] F32 — small, upload directly.
        long routerFloats = (long)moe.NumExperts * hiddenSize;
        nint gateDevice = AllocDevice(routerFloats * sizeof(float));
        fixed (float* pGate = moe.Gate)
        {
            CopyHtoD(gateDevice, (nint)pGate, routerFloats * sizeof(float));
        }

        // Routed expert raw quant views — single contiguous H2D per fused-experts tensor.
        long gateExpsBytes = Dequantize.RowByteSize(moe.GateExpsKDim, moe.GateExpsRawQt) *
                             moe.GateExpsMDim * moe.NumExperts;
        long upExpsBytes = Dequantize.RowByteSize(moe.UpExpsKDim, moe.UpExpsRawQt) *
                           moe.UpExpsMDim * moe.NumExperts;
        long downExpsBytes = Dequantize.RowByteSize(moe.DownExpsKDim, moe.DownExpsRawQt) *
                             moe.DownExpsMDim * moe.NumExperts;

        nint gateExpsDevice = AllocDevice(gateExpsBytes);
        nint upExpsDevice = AllocDevice(upExpsBytes);
        nint downExpsDevice = AllocDevice(downExpsBytes);
        CopyHtoD(gateExpsDevice, moe.GateExpsRaw, gateExpsBytes);
        CopyHtoD(upExpsDevice, moe.UpExpsRaw, upExpsBytes);
        CopyHtoD(downExpsDevice, moe.DownExpsRaw, downExpsBytes);

        UpdateMaxTile(ref maxTileFloats, (long)moe.GateExpsMDim * moe.GateExpsKDim);
        UpdateMaxTile(ref maxTileFloats, (long)moe.DownExpsMDim * moe.DownExpsKDim);

        // Shared experts: F32 pointers from the CPU loader → device F32.
        // Qwen3.6-A3B ships exactly one shared expert (verified from the Q6_K_XL GGUF
        // metadata: qwen35moe.expert_shared_feed_forward_length = 512, single tensor).
        // ForwardSharedExpertF32 below only walks index 0 — if a future variant ships
        // multiple shared experts the cross-expert accumulation would be silently dropped.
        // Fail loud at load.
        int numShared = moe.NumSharedExperts;
        if (numShared > 1)
            throw new NotSupportedException(
                $"CUDA Qwen3MoeHybrid shared-expert path handles at most one shared expert; " +
                $"got {numShared}. Extend ForwardSharedExpertF32 to loop and accumulate.");
        nint[] sharedGateDevice = new nint[numShared];
        nint[] sharedUpDevice = new nint[numShared];
        nint[] sharedDownDevice = new nint[numShared];
        long sharedHiddenIntFloats = (long)moe.SharedIntermediateSize * hiddenSize;
        for (int s = 0; s < numShared; s++)
        {
            sharedGateDevice[s] = AllocDevice(sharedHiddenIntFloats * sizeof(float));
            CopyHtoD(sharedGateDevice[s], moe.SharedGateProj[s], sharedHiddenIntFloats * sizeof(float));
            sharedUpDevice[s] = AllocDevice(sharedHiddenIntFloats * sizeof(float));
            CopyHtoD(sharedUpDevice[s], moe.SharedUpProj[s], sharedHiddenIntFloats * sizeof(float));
            sharedDownDevice[s] = AllocDevice(sharedHiddenIntFloats * sizeof(float));
            CopyHtoD(sharedDownDevice[s], moe.SharedDownProj[s], sharedHiddenIntFloats * sizeof(float));
        }

        // Optional Qwen1.5-MoE shared-expert sigmoid gate [hiddenSize] F32.
        nint sharedExpertGateDevice = 0;
        if (moe.SharedExpertGate is not null)
        {
            sharedExpertGateDevice = AllocDevice((long)hiddenSize * sizeof(float));
            fixed (float* pGateShared = moe.SharedExpertGate)
            {
                CopyHtoD(sharedExpertGateDevice, (nint)pGateShared, (long)hiddenSize * sizeof(float));
            }
        }

        return new DeviceMoe
        {
            GateRouterDevice = gateDevice,
            NumExperts = moe.NumExperts,
            NumExpertsPerTok = moe.NumExpertsPerTok,
            IntermediateSize = moe.IntermediateSize,
            NormTopKProb = moe.NormTopKProb,

            GateExpsDevice = gateExpsDevice, GateExpsQt = moe.GateExpsRawQt,
            GateExpsMDim = moe.GateExpsMDim, GateExpsKDim = moe.GateExpsKDim,

            UpExpsDevice = upExpsDevice, UpExpsQt = moe.UpExpsRawQt,
            UpExpsMDim = moe.UpExpsMDim, UpExpsKDim = moe.UpExpsKDim,

            DownExpsDevice = downExpsDevice, DownExpsQt = moe.DownExpsRawQt,
            DownExpsMDim = moe.DownExpsMDim, DownExpsKDim = moe.DownExpsKDim,

            SharedGateDevice = sharedGateDevice,
            SharedUpDevice = sharedUpDevice,
            SharedDownDevice = sharedDownDevice,
            SharedIntermediateSize = moe.SharedIntermediateSize,
            SharedExpertGateDevice = sharedExpertGateDevice,
        };
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Forward dispatch
    // ──────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);

        nint streamH = _stream.Handle;

        // ── Upload tokenIds + positions to device ──
        fixed (int* tokenPtr = tokenIds)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }
        fixed (int* posPtr = positions)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }

        // ── Embedding lookup → HiddenState (F32) ──
        if (_embedF32Device != 0)
        {
            // Pre-dequanted F32 table — use the F32-format lookup path.
            _kernels.LaunchEmbeddingLookupF32(_embedF32Device, QuantizationType.F32,
                _state.TokenIdsDevice, _state.HiddenState, seqLen, hiddenSize, streamH);
        }
        else
        {
            _kernels.LaunchEmbeddingLookupF32(_tokenEmbedDevice, _tokenEmbedQt,
                _state.TokenIdsDevice, _state.HiddenState, seqLen, hiddenSize, streamH);
        }

        // ── Per-layer body ──
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            RunSingleLayerBody(layer, seqLen, positions, hiddenSize,
                numHeads, numKvHeads, headDim, eps, kvCache);
        }

        // ── Final norm + lm_head ──
        _kernels.LaunchRmsNormF32(_state.HiddenState, _outputNormDevice, _state.HiddenState,
            hiddenSize, eps, seqLen, streamH);
        Gemm(_outputDevice, _outputQt, _state.HiddenState, _state.Logits,
             _outputOutputDim, _outputInputDim, seqLen);

        // Sync and D2H the logits.
        _stream.Synchronize();

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.Logits,
            (nuint)((long)seqLen * vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// Runs a single transformer-block body for the layer at <paramref name="layerIdx"/>:
    /// reads the current hidden state from <see cref="CudaQwen3MoeHybridForwardState.HiddenState"/>,
    /// applies attn_norm, the token-mixing sub-layer (GDN or full-attention) with first
    /// residual add, post_attn_norm, the MoE FFN, and the second residual add — then writes
    /// the updated hidden back to the same <c>HiddenState</c> buffer in-place. Mirrors
    /// the body of the per-layer loop in <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>;
    /// extracted so the layer-by-layer parity harness can drive one layer at a time on top
    /// of a CPU-captured activation trace.
    /// </summary>
    /// <remarks>
    /// The caller is responsible for ensuring <see cref="CudaQwen3MoeHybridForwardState.EnsureCapacity"/>
    /// has been called with at least <paramref name="seqLen"/>, that
    /// <see cref="DeviceLayer"/> at <paramref name="layerIdx"/> has its weight pointers
    /// populated (e.g. via <c>LoadFromGguf</c> or <c>LoadSingleLayerWeightsFromGguf</c>),
    /// and that <c>_state.PositionsDevice</c> has been written with the F32-attn position
    /// values when this layer is a full-attention layer.
    /// </remarks>
    private void RunSingleLayerBody(int layerIdx, int seqLen, ReadOnlySpan<int> positions,
        int hiddenSize, int numHeads, int numKvHeads, int headDim, float eps, IKvCache? kvCache)
    {
        nint streamH = _stream.Handle;
        long hiddenBytes = (long)seqLen * hiddenSize * sizeof(float);
        var kinds = _layout.LayerKind;
        ref readonly DeviceLayer lw = ref _layers[layerIdx];

        // 1. Token mixing — residual = hidden; normOut = RmsNorm(hidden, attn_norm).
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
            (nuint)hiddenBytes, streamH).ThrowOnError();
        _kernels.LaunchRmsNormF32(_state.HiddenState, lw.AttnNormWeightDevice, _state.NormOutput,
            hiddenSize, eps, seqLen, streamH);

        if (kinds[layerIdx] == HybridLayerKind.GatedDeltaNet)
        {
            ForwardGdnBody(lw.Gdn!.Value, layerIdx, seqLen, hiddenSize, eps);
        }
        else
        {
            ForwardFullAttnBody(lw.FullAttn!.Value, layerIdx, seqLen, positions,
                numHeads, numKvHeads, headDim, eps, kvCache);
        }

        // 2. First residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);

        // 3. MoE FFN — residual = hidden; normOut = RmsNorm(hidden, post_attn_norm).
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
            (nuint)hiddenBytes, streamH).ThrowOnError();
        _kernels.LaunchRmsNormF32(_state.HiddenState, lw.PostAttnNormWeightDevice, _state.NormOutput,
            hiddenSize, eps, seqLen, streamH);

        // Capture the normed FFN input for the shared expert — CudaMoeFfn overwrites
        // NormOutput with the routed sum and the shared MLP reads the same normed
        // tensor as input. We reuse _state.GateScratch (sized seqLen * qElems floats);
        // for qwen35moe qElems = 16 * 256 = 4096 >= hiddenSize = 2048, so it fits.
        // The full-attn body owns this buffer but the MoE runs AFTER attention each
        // layer, so the buffer is free at this point.
        if (lw.Moe.SharedGateDevice is { Length: > 0 })
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.GateScratch, _state.NormOutput,
                (nuint)hiddenBytes, streamH).ThrowOnError();
        }

        ForwardMoeBody(lw, seqLen, hiddenSize);

        // 4. Second residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  GDN token-mixing body
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// GDN (Gated DeltaNet) token-mixing forward. Reads pre-normed activations from
    /// <see cref="CudaQwen3MoeHybridForwardState.NormOutput"/> and writes the ssm_out
    /// projection back to the same buffer. Advances the per-layer GDN conv and
    /// associative-memory state in place.
    /// </summary>
    private void ForwardGdnBody(
        in DeviceGdn gdnW, int absoluteLayerIdx, int seqLen, int hiddenSize, float eps)
    {
        nint streamH = _stream.Handle;
        int nVHead = _gdn.NVHead;
        int nKHead = _gdn.NKHead;
        int dState = _gdn.DState;
        int dConv = _gdn.DConv;
        int convDim = (2 * nKHead + nVHead) * dState;
        int vDim = nVHead * dState;
        int kDim = nKHead * dState;
        int gdnOrdinal = _gdnLayerOrdinal[absoluteLayerIdx];

        nint normOut = _state.NormOutput;
        nint qkvBuf = _state.GdnQkvBuf;
        nint zBuf = _state.GdnZBuf;
        nint alphaBuf = _state.GdnAlphaBuf;
        nint betaBuf = _state.GdnBetaBuf;
        nint qBuf = _state.GdnQBuf;
        nint kBuf = _state.GdnKBuf;
        nint vBuf = _state.GdnVBuf;
        nint gdnOut = _state.GdnOut;
        nint convInput = _state.GdnConvInput;

        // ── 1. Projections from the normed input ──
        Gemm(gdnW.QkvDevice, gdnW.QkvQt, normOut, qkvBuf,
             gdnW.QkvOutputDim, gdnW.QkvInputDim, seqLen);
        Gemm(gdnW.GateDevice, gdnW.GateQt, normOut, zBuf,
             gdnW.GateOutputDim, gdnW.GateInputDim, seqLen);
        Gemm(gdnW.AlphaDevice, gdnW.AlphaQt, normOut, alphaBuf,
             gdnW.AlphaOutputDim, gdnW.AlphaInputDim, seqLen);
        Gemm(gdnW.BetaDevice, gdnW.BetaQt, normOut, betaBuf,
             gdnW.BetaOutputDim, gdnW.BetaInputDim, seqLen);

        // ── 2. Decay g and write-gate beta ──
        // g[t, vh] = exp(softplus(alpha[t, vh] + dt_bias[vh]) * A[vh])
        // beta = sigmoid(beta_proj)
        if (_kernels.HasGdnDecayF32)
        {
            _kernels.LaunchGdnDecayF32(alphaBuf, gdnW.DtBiasDevice, gdnW.ADevice,
                seqLen, nVHead, streamH);
        }
        else
        {
            LaunchGdnDecayHostFallback(alphaBuf, gdnW.DtBiasDevice, gdnW.ADevice, seqLen, nVHead);
        }
        if (_kernels.HasElementwiseF32)
        {
            _kernels.LaunchSigmoidF32(betaBuf, (long)seqLen * nVHead, streamH);
        }
        else
        {
            LaunchSigmoidHostFallback(betaBuf, seqLen * nVHead);
        }

        // ── 3. Conv1d on QKV concat ──
        // Fill ConvInput: [(DConv-1) conv-state rows | seqLen qkvBuf rows].
        nint convStateDev = _gdnCache.GetConvStatePtr(gdnOrdinal);
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convInput, convStateDev,
            (nuint)convStateBytes, streamH).ThrowOnError();
        long qkvRowsBytes = (long)seqLen * convDim * sizeof(float);
        nint convInputQkvOff = convInput + (nint)convStateBytes;
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convInputQkvOff, qkvBuf,
            (nuint)qkvRowsBytes, streamH).ThrowOnError();

        // Conv1d → qkvBuf (in place), then SiLU. The Conv1d kernel currently does
        // NOT apply SiLU itself; we apply it as a separate elementwise pass.
        _kernels.LaunchConv1dCausalF32(convInput, gdnW.Conv1dWeightDevice, gdnW.Conv1dBiasDevice,
            qkvBuf, dConv, convDim, seqLen, streamH);
        if (_kernels.HasElementwiseF32)
        {
            _kernels.LaunchSiluF32(qkvBuf, (long)seqLen * convDim, streamH);
        }
        else
        {
            LaunchSiluHostFallback(qkvBuf, (long)seqLen * convDim);
        }

        // Save the trailing (DConv-1) rows of convInput back to the rolling state.
        nint trailRowsSrc = convInput + (nint)((long)seqLen * convDim * sizeof(float));
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convStateDev, trailRowsSrc,
            (nuint)convStateBytes, streamH).ThrowOnError();

        // ── 4. De-interleave Q/K/V from conv output, L2-normalise Q and K per head ──
        // Layout per token: [Q (kDim) | K (kDim) | V (vDim)].
        long rowBytes = (long)convDim * sizeof(float);
        long kDimBytes = (long)kDim * sizeof(float);
        long vDimBytes = (long)vDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint srcRow = qkvBuf + (nint)(t * rowBytes);
            nint qDst = qBuf + (nint)(t * kDimBytes);
            nint kDst = kBuf + (nint)(t * kDimBytes);
            nint vDst = vBuf + (nint)(t * vDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(qDst, srcRow, (nuint)kDimBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, srcRow + (nint)kDimBytes, (nuint)kDimBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, srcRow + (nint)(2 * kDimBytes), (nuint)vDimBytes, streamH).ThrowOnError();
        }

        // Per-head L2 normalisation (seqLen * nKHead heads of length dState in each of Q and K).
        _kernels.LaunchL2NormalizeHeadsF32(qBuf, seqLen * nKHead, dState, 1e-6f, streamH);
        _kernels.LaunchL2NormalizeHeadsF32(kBuf, seqLen * nKHead, dState, 1e-6f, streamH);

        // ── 5. GDN scan — single-token kernel driven by host loop ──
        nint gdnStateDev = _gdnCache.GetGdnStatePtr(gdnOrdinal);
        long qStepBytes = (long)kDim * sizeof(float);
        long kStepBytes = qStepBytes;
        long vStepBytes = (long)vDim * sizeof(float);
        long gStepBytes = (long)nVHead * sizeof(float);
        long betaStepBytes = gStepBytes;
        long outStepBytes = vStepBytes;
        for (int t = 0; t < seqLen; t++)
        {
            nint qT = qBuf + (nint)(t * qStepBytes);
            nint kT = kBuf + (nint)(t * kStepBytes);
            nint vT = vBuf + (nint)(t * vStepBytes);
            nint gT = alphaBuf + (nint)(t * gStepBytes);
            nint betaT = betaBuf + (nint)(t * betaStepBytes);
            nint outT = gdnOut + (nint)(t * outStepBytes);
            _kernels.LaunchGdnScanStepF32(gdnStateDev, qT, kT, vT, gT, betaT, outT,
                nVHead, nKHead, dState, streamH);
        }

        // ── 6. Per-head RMSNorm(out, ssm_norm) * silu(z) gating ──
        // ssm_norm weight [dState] is broadcast across all (seqLen * nVHead) heads.
        // Reuse the existing RMSNorm-F32 kernel as a per-head normaliser:
        //   rows = seqLen * nVHead, hiddenSize = dState. The weight is dState floats
        //   shared across all rows — exactly what LaunchRmsNormF32 does.
        _kernels.LaunchRmsNormF32(gdnOut, gdnW.SsmNormDevice, gdnOut,
            dState, eps, seqLen * nVHead, streamH);
        // gdnOut *= silu(z). silu(z) = z * sigmoid(z); SwiGLU(gate=z, up=gdnOut) = silu(z)*up.
        _kernels.LaunchSwiGLUF32(zBuf, gdnOut, gdnOut, vDim, seqLen, streamH);

        // ── 7. ssm_out projection into NormOutput ──
        Gemm(gdnW.OutDevice, gdnW.OutQt, gdnOut, normOut,
             gdnW.OutOutputDim, gdnW.OutInputDim, seqLen);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Full GQA attention body
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Full GQA attention (qwen35moe variant) — Q+Gate fused projection, QK-norm, partial-rotary
    /// RoPE (NeoX pair pattern), GQA SDPA, and a post-attention sigmoid(gate) elementwise
    /// product before the O projection. Reads from <see cref="CudaQwen3MoeHybridForwardState.NormOutput"/>
    /// and writes back to the same buffer.
    /// </summary>
    private void ForwardFullAttnBody(
        in DeviceFullAttn attn, int layer, int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, float eps, IKvCache? kvCache)
    {
        nint streamH = _stream.Handle;
        int qElems = numHeads * headDim;
        int qgElems = 2 * qElems;
        int kvElems = numKvHeads * headDim;

        nint normOut = _state.NormOutput;
        nint qgBuf = _state.QGateScratch;
        nint q = _state.QScratch;
        nint k = _state.KScratch;
        nint v = _state.VScratch;
        nint gate = _state.GateScratch;
        nint attnOut = _state.AttnOutput;

        // ── 1. Fused Q+Gate projection ──
        Gemm(attn.QDevice, attn.QQt, normOut, qgBuf, attn.QOutputDim, attn.QInputDim, seqLen);
        DumpDevice2D($"blk.{layer}.fa_qg", qgBuf, seqLen, qgElems);

        // ── 2. De-interleave QG → Q and Gate. Per-token layout:
        //       [Q_h0(headDim), Gate_h0(headDim), Q_h1(headDim), Gate_h1(headDim), ...]
        //    Each head is 2*headDim contiguous floats with Q first, Gate second.
        long perTokenQgBytes = (long)qgElems * sizeof(float);
        long perTokenQBytes = (long)qElems * sizeof(float);
        long perHeadBytes = (long)headDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint qgRow = qgBuf + (nint)(t * perTokenQgBytes);
            nint qRow = q + (nint)(t * perTokenQBytes);
            nint gRow = gate + (nint)(t * perTokenQBytes);
            for (int h = 0; h < numHeads; h++)
            {
                nint qgHead = qgRow + (nint)(h * 2 * perHeadBytes);
                nint qHead = qRow + (nint)(h * perHeadBytes);
                nint gHead = gRow + (nint)(h * perHeadBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(qHead, qgHead, (nuint)perHeadBytes, streamH).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(gHead, qgHead + (nint)perHeadBytes,
                    (nuint)perHeadBytes, streamH).ThrowOnError();
            }
        }
        DumpDevice2D($"blk.{layer}.fa_q_split", q, seqLen, numHeads * headDim);
        DumpDevice2D($"blk.{layer}.fa_gate_split", gate, seqLen, numHeads * headDim);

        // ── 3. K and V projections ──
        Gemm(attn.KDevice, attn.KQt, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen);
        Gemm(attn.VDevice, attn.VQt, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen);
        DumpDevice2D($"blk.{layer}.fa_k", k, seqLen, numKvHeads * headDim);
        DumpDevice2D($"blk.{layer}.fa_v", v, seqLen, numKvHeads * headDim);

        // ── 4. Per-head QK-norm. Treat Q as [seqLen*numHeads, headDim] and apply RMSNorm
        //       with the F32 attn_q_norm weight broadcast across rows (same weight, same
        //       headDim). Same for K with attn_k_norm and seqLen*numKvHeads rows.
        _kernels.LaunchRmsNormF32(q, attn.QNormDevice, q,
            headDim, eps, seqLen * numHeads, streamH);
        _kernels.LaunchRmsNormF32(k, attn.KNormDevice, k,
            headDim, eps, seqLen * numKvHeads, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postnorm", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postnorm", k, seqLen, numKvHeads * headDim);

        // ── 5. RoPE — partial-rotary NeoX (HuggingFace rotate_half) over the first ropeDim
        //    of each head. The CUDA kernel `rope_f32.cu` encodes the pair pattern as:
        //    rope_type == 1 → NeoX split halves (pair i, i + halfRope), anything else →
        //    GPT-J interleaved (2i, 2i+1). That convention is pinned by
        //    CudaKernelComparisonTests.RoPEF32_NeoX_MatchesCpuReference. The C# enum value
        //    RoPEType.NeoX = 2 must NOT be passed straight through.
        _kernels.LaunchRoPEF32(q, k, _state.PositionsDevice,
            seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, 1, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postrope", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postrope", k, seqLen, numKvHeads * headDim);

        // ── 6. Attention (GQA with causal mask). Two paths:
        //   - kvCache == null: F32 kernel walks the freshly-projected K and V as the
        //     entire context (prefill chunk, no cache).
        //   - kvCache != null: write the new K/V rows into the model-private F32 KV
        //     cache at their absolute positions, then run attention over the full
        //     cached context [0, currentLength). positionOffset = positions[0] sets the
        //     causal mask anchor so position p only attends to keys at index ≤ p.
        if (kvCache is not null)
        {
            EnsureF32KvCache(kvCache.MaxLength, numKvHeads, headDim);
            int slot = _kvSlotForLayer[layer];
            if (slot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} is not a full-attention layer but ForwardFullAttnBody was invoked.");
            WriteF32KvRows(slot, k, v, positions, numKvHeads, headDim);

            int positionOffset = positions[0];
            int seqKv = _f32CacheCurrentLength;
            _kernels.LaunchAttentionF32(q, _f32KCache![slot], _f32VCache![slot], attnOut,
                seqLen, seqKv, numHeads, numKvHeads, headDim,
                positionOffset: positionOffset, slidingWindow: 0, streamH);
        }
        else
        {
            _kernels.LaunchAttentionF32(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeads, headDim,
                positionOffset: 0, slidingWindow: 0, streamH);
        }
        DumpDevice2D($"blk.{layer}.fa_attnout_pregate", attnOut, seqLen, qElems);

        // ── 7. attnOut *= sigmoid(gate). ──
        if (_kernels.HasElementwiseF32)
        {
            _kernels.LaunchSigmoidMulF32(attnOut, gate, (long)seqLen * qElems, streamH);
        }
        else
        {
            LaunchSigmoidMulHostFallback(attnOut, gate, (long)seqLen * qElems);
        }
        DumpDevice2D($"blk.{layer}.fa_attnout_postgate", attnOut, seqLen, qElems);

        // ── 8. Output projection ──
        Gemm(attn.ODevice, attn.OQt, attnOut, _state.NormOutput,
             attn.OOutputDim, attn.OInputDim, seqLen);
    }

    /// <summary>
    /// Debug helper: D2H-copy a contiguous F32 device buffer and forward it to TensorDump.
    /// Compiled away to a single env-var check when DOTLLM_TENSOR_DUMP is unset.
    /// </summary>
    private void DumpDevice2D(string name, nint devPtr, int d0, int d1)
    {
        if (!DotLLM.Models.Architectures.TensorDump.Enabled) return;
        long n = (long)d0 * d1;
        if (n <= 0) return;
        _stream.Synchronize();
        float[] host = new float[n];
        fixed (float* pHost = host)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pHost, devPtr, (nuint)(n * sizeof(float))).ThrowOnError();
            DotLLM.Models.Architectures.TensorDump.Dump2D(name, pHost, d0, d1);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-attention-layer F32 KV cache (model-private)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Lazily allocates the per-attention-layer F32 K and V buffers. Called only from
    /// the full-attn path when the caller passes a non-null <see cref="IKvCache"/>.
    /// Each buffer is sized to <c>maxSeqLen × (numKvHeads × headDim) × sizeof(float)</c>
    /// — at qwen35moe (16 KV heads × 256 = 4096 floats per token × 2 cache halves × 10
    /// attention layers × maxSeqLen) the total scales linearly with maxSeqLen. At the
    /// model's full 262144 context this is ~80 GiB — the caller is expected to size
    /// the cache to something reasonable for their hardware.
    /// </summary>
    /// <remarks>
    /// We don't reuse the standard <see cref="CudaKvCache"/> because it stores FP16
    /// keys/values; the F32 attention path (<see cref="CudaKernels.LaunchAttentionF32"/>)
    /// expects F32 packed [seqKv, kvStride]. Bridging via per-row F32→F16→F32 converts
    /// would dominate the attention cost; an F32 sidecar is cleaner.
    /// </remarks>
    private void EnsureF32KvCache(int maxSeqLen, int numKvHeads, int headDim)
    {
        if (_f32KCache is not null && maxSeqLen <= _f32CacheMaxSeqLen) return;

        // Free any existing buffers (resize).
        if (_f32KCache is not null)
        {
            for (int i = 0; i < _f32KCache.Length; i++)
            {
                if (_f32KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f32KCache[i]);
                if (_f32VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f32VCache[i]);
            }
        }

        _f32KCache = new nint[_attentionLayerCount];
        _f32VCache = new nint[_attentionLayerCount];
        long bytesPerLayer = (long)maxSeqLen * numKvHeads * headDim * sizeof(float);
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _f32KCache[i] = AllocDevice(bytesPerLayer);
            _f32VCache[i] = AllocDevice(bytesPerLayer);
        }
        _f32CacheMaxSeqLen = maxSeqLen;
        _f32CacheCurrentLength = 0;
    }

    /// <summary>
    /// Writes the freshly-projected K and V rows for one attention layer into the
    /// per-layer F32 KV cache at their absolute positions. Updates the cache's
    /// current-length counter to <c>max(positions) + 1</c> on a per-call basis (the
    /// same length is shared across all attention layers — they always advance in
    /// lockstep because every layer runs once per forward).
    /// </summary>
    /// <remarks>
    /// Contiguous-positions fast path issues a single bulk <c>cuMemcpyDtoDAsync</c>
    /// per K/V buffer — the common case for prefill and sequential decode. The
    /// non-contiguous fallback (e.g. partial-cache reuse after rollback) issues one
    /// DtoD per row.
    /// </remarks>
    private void WriteF32KvRows(int layerSlot, nint kSrc, nint vSrc,
                                 ReadOnlySpan<int> positions, int numKvHeads, int headDim)
    {
        nint streamH = _stream.Handle;
        int seqLen = positions.Length;
        long rowBytes = (long)numKvHeads * headDim * sizeof(float);

        bool contiguous = seqLen > 0;
        int maxPos = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            int p = positions[i];
            if ((uint)p >= (uint)_f32CacheMaxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {p} at index {i} exceeds F32 KV cache capacity {_f32CacheMaxSeqLen}.");
            if (p > maxPos) maxPos = p;
            if (i > 0 && positions[i] != positions[i - 1] + 1) contiguous = false;
        }

        nint kBase = _f32KCache![layerSlot];
        nint vBase = _f32VCache![layerSlot];

        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint kDst = kBase + (nint)((long)positions[0] * rowBytes);
            nint vDst = vBase + (nint)((long)positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)bulkBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)bulkBytes, streamH).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                nint kDst = kBase + (nint)((long)positions[i] * rowBytes);
                nint vDst = vBase + (nint)((long)positions[i] * rowBytes);
                nint kS = kSrc + (nint)((long)i * rowBytes);
                nint vS = vSrc + (nint)((long)i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kS, (nuint)rowBytes, streamH).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vS, (nuint)rowBytes, streamH).ThrowOnError();
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _f32CacheCurrentLength)
            _f32CacheCurrentLength = newLength;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  MoE FFN body
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// MoE SwiGLU FFN forward (qwen35moe variant). Reads pre-normed activations from
    /// <see cref="CudaQwen3MoeHybridForwardState.NormOutput"/> and writes the routed +
    /// shared-expert output back to the same buffer. Entirely on-device — no host
    /// round-trip.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Routed expert pipeline runs through <see cref="CudaMoeFfn"/> with a per-call
    /// adapter that builds per-expert raw-quant pointers from the fused-experts base
    /// pointer + the <c>e * RowByteSize(M*K, qt)</c> stride convention used by the CPU
    /// oracle (<see cref="MoeSwiGluMlp"/>) and the GGUF mmap layout. The adapter
    /// declares <c>NumSharedExperts = 0</c> — <see cref="CudaMoeFfn"/> has a single
    /// <c>Precision</c> field that would force the F32-resident shared experts down
    /// the quantized dequant path. We run the shared expert separately below via
    /// direct cuBLAS F32 GEMM, summed (with optional sigmoid gating for the Qwen1.5
    /// shared-expert-gate variant) into the routed output.
    /// </para>
    /// <para>
    /// The CudaMoeFfn dispatcher syncs the stream once for the topk DtoH copy — for
    /// decode (seqLen=1, K=8) that's <c>8 × sizeof(int) = 32 bytes</c> and is the only
    /// host round-trip per MoE layer. Routed expert dispatch then runs via the
    /// Phase-B grouped quantized GEMV (Q6_K / Q8_0 supported) collapsing all
    /// <c>K_active</c> gate / up projections into two single-launch kernels each.
    /// </para>
    /// </remarks>
    private void ForwardMoeBody(in DeviceLayer lw, int seqLen, int hiddenSize)
    {
        // CudaMoeFfn zero-clears outputF32 BEFORE it reads hiddenF32 (line ~80 of
        // CudaMoeFfn.cs), so we cannot pass the same pointer for both. The normed input
        // lives in _state.NormOutput; we route the routed-expert sum into _state.HiddenState
        // (currently a duplicate of _state.Residual — the residual was just copied from
        // HiddenState in the outer Forward loop) and copy it back to NormOutput on exit so
        // the final residual add at step 4 reads the right buffer.
        nint streamH = _stream.Handle;
        nint moeInput = _state.NormOutput;
        nint moeOutput = _state.HiddenState;
        long hiddenBytes = (long)seqLen * hiddenSize * sizeof(float);

        var weights = BuildRoutedMoeLayerWeights(lw.Moe, hiddenSize);
        CudaMoeFfn.Forward(
            hiddenF32: moeInput,
            outputF32: moeOutput,
            seqLen: seqLen,
            weights: weights,
            scratch: _moeScratch,
            cublasHandle: _cublas.Handle,
            kernels: _kernels,
            stream: streamH);

        // Shared expert(s) — F32 device-resident; one MLP per token, summed into moeOutput.
        // The shared-expert input is the captured pre-MoE normed tensor in
        // _state.GateScratch (see the outer Forward loop).
        if (lw.Moe.SharedGateDevice is { Length: > 0 })
        {
            ForwardSharedExpertF32(lw.Moe, seqLen, hiddenSize, moeOutput);
        }

        // Stage the routed+shared sum back into _state.NormOutput so the caller's
        // step-4 residual add (Residual + NormOutput → HiddenState) sees the right
        // buffer. Single D2D copy of seqLen × hiddenSize × 4B floats (8 KB per token
        // at hiddenSize=2048) — overhead is dominated by the GEMM work.
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.NormOutput, moeOutput,
            (nuint)hiddenBytes, streamH).ThrowOnError();
    }

    /// <summary>
    /// Builds a <see cref="CudaMoeLayerWeights"/> view over the routed experts only.
    /// Per-expert pointers slice into the fused <c>GateExpsDevice</c> /
    /// <c>UpExpsDevice</c> / <c>DownExpsDevice</c> raw-quant tensors at byte offset
    /// <c>e * RowByteSize(M × K, qt)</c> — the same stride
    /// <see cref="Qwen3MoeHybridTransformerModel"/> uses on the CPU side
    /// (verified against <see cref="MoeSwiGluMlp.ExecuteRoutedFromAssignments"/>).
    /// Shared experts are intentionally omitted (NumSharedExperts=0); the caller runs
    /// them via the F32 helper below.
    /// </summary>
    private CudaMoeLayerWeights BuildRoutedMoeLayerWeights(in DeviceMoe moe, int hiddenSize)
    {
        int E = moe.NumExperts;
        long gateBytesPerExpert = Dequantize.RowByteSize((long)moe.GateExpsMDim * moe.GateExpsKDim, moe.GateExpsQt);
        long upBytesPerExpert = Dequantize.RowByteSize((long)moe.UpExpsMDim * moe.UpExpsKDim, moe.UpExpsQt);
        long downBytesPerExpert = Dequantize.RowByteSize((long)moe.DownExpsMDim * moe.DownExpsKDim, moe.DownExpsQt);

        var gateProj = new nint[E];
        var upProj = new nint[E];
        var downProj = new nint[E];
        for (int e = 0; e < E; e++)
        {
            gateProj[e] = moe.GateExpsDevice + (nint)(e * gateBytesPerExpert);
            upProj[e] = moe.UpExpsDevice + (nint)(e * upBytesPerExpert);
            downProj[e] = moe.DownExpsDevice + (nint)(e * downBytesPerExpert);
        }

        // F32-only synthetic / parity-test path: when every routed projection is F32 the
        // fused per-expert tensors are just packed row-major F32 matrices, so we drop the
        // dequant round-trip and let CudaMoeFfn dispatch through MoePrecision.F32 — direct
        // cuBLAS LinearF32 per expert. This is the path the CPU-vs-CUDA parity tests
        // exercise; the Quantized path stays the default for real GGUF Q*_K checkpoints.
        bool allRoutedF32 = moe.GateExpsQt == QuantizationType.F32
                            && moe.UpExpsQt == QuantizationType.F32
                            && moe.DownExpsQt == QuantizationType.F32;
        var precision = allRoutedF32 ? MoePrecision.F32 : MoePrecision.Quantized;

        return new CudaMoeLayerWeights(
            numExperts: E,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            moeIntermediateSize: moe.IntermediateSize,
            normTopKProb: moe.NormTopKProb,
            router: moe.GateRouterDevice,
            gateProj: gateProj,
            upProj: upProj,
            downProj: downProj,
            // Shared experts handled separately — see ForwardSharedExpertF32.
            numSharedExperts: 0,
            sharedIntermediateSize: 0,
            sharedGateProj: null,
            sharedUpProj: null,
            sharedDownProj: null,
            sharedExpertGate: 0,
            precision: precision,
            gateProjQuantType: moe.GateExpsQt,
            upProjQuantType: moe.UpExpsQt,
            downProjQuantType: moe.DownExpsQt,
            sharedGateProjQuantType: QuantizationType.F32,
            sharedUpProjQuantType: QuantizationType.F32,
            sharedDownProjQuantType: QuantizationType.F32);
    }

    /// <summary>
    /// Adds the shared-expert SwiGLU MLP contribution to <c>_state.NormOutput</c>.
    /// Reads the original layer input from <c>_state.HiddenState</c> (which already
    /// holds the post-attention residual — see <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>'s
    /// outer loop), but we don't have direct access to it here — instead the caller
    /// guarantees <c>_state.NormOutput</c> was written by RmsNorm(hiddenState, post_attn_norm)
    /// and CudaMoeFfn has overwritten it with the routed output. We need the
    /// post-attn-norm-ed input, so we keep a copy in <c>_state.Residual</c> before
    /// invoking the MoE — see the outer Forward loop. Actually the outer loop already
    /// keeps <c>_state.Residual</c> as the pre-norm value, so we must dedicate a
    /// separate scratch buffer for the shared-expert input. Simpler approach: take
    /// the normed input from <c>_state.NormOutput</c> BEFORE CudaMoeFfn overwrites it
    /// by capturing it in <c>_state.GateScratch</c> just before the MoE call. The
    /// outer loop has been updated accordingly.
    /// </summary>
    /// <remarks>
    /// At Qwen3.6-A3B scale: SharedIntermediateSize=512, hiddenSize=2048. So the
    /// shared MLP is a 2048→512 gate + 2048→512 up + SwiGLU + 512→2048 down. Tiny vs
    /// the routed work; F32 cuBLAS GEMV is fine.
    /// </remarks>
    private void ForwardSharedExpertF32(in DeviceMoe moe, int seqLen, int hiddenSize, nint outF32)
    {
        nint streamH = _stream.Handle;
        int sI = moe.SharedIntermediateSize;
        // The normed input we want is in _state.GateScratch (captured by the outer loop
        // before the routed MoE overwrote its output). See Forward() body.
        nint sharedInput = _state.GateScratch;
        // Per-token scratch buffers reused — these are sized to seqLen * sI, but
        // _state.GdnQkvBuf is conv-sized (seqLen * convDim); for qwen35moe convDim
        // includes 2*NKHead*DState + NVHead*DState = (2 + groupRatio) * groups * DState.
        // The smaller candidates are _state.GdnZBuf (seqLen * NVHead*DState = seqLen * 4096
        // for our config) which dwarfs sI*seqLen. Just allocate dedicated buffers — at
        // qwen35moe sI=512 we'd want sI*seqLen * 4B per scratch, but for the typical decode
        // (seqLen=1) that's 2 KB. To avoid permanent allocation when shared experts are
        // absent we lazily allocate three small per-call scratches under the moe-scratch
        // umbrella; for now we share with the routed _moeScratch instance's shared-* buffers.
        // CudaMoeScratch only allocates these when the bundle declares numSharedExperts > 0,
        // which we set to 0 above; so use a local re-EnsureCapacity flag... simplest path:
        // allocate the four shared buffers on this object as model-private once.
        EnsureSharedExpertScratch(seqLen, hiddenSize, sI);

        long sIBytes = (long)seqLen * sI * sizeof(float);

        // 1. Optional sigmoid gate: scale = sigmoid(hidden @ gate_logit)
        bool hasGate = moe.SharedExpertGateDevice != 0;

        // 2. gate proj + up proj (F32 GEMMs).
        CudaGemm.LinearF32(_cublas.Handle, sharedInput, moe.SharedGateDevice[0],
            _sharedGateBuf, seqLen, hiddenSize, sI, streamH);
        CudaGemm.LinearF32(_cublas.Handle, sharedInput, moe.SharedUpDevice[0],
            _sharedUpBuf, seqLen, hiddenSize, sI, streamH);

        // 3. SwiGLU element-wise: silu(gate) * up → silu buffer.
        _kernels.LaunchSwiGLUF32(_sharedGateBuf, _sharedUpBuf, _sharedSiluBuf,
            sI, seqLen, streamH);

        // 4. down proj into a dedicated buffer, then sum into outF32.
        CudaGemm.LinearF32(_cublas.Handle, _sharedSiluBuf, moe.SharedDownDevice[0],
            _sharedDownBuf, seqLen, sI, hiddenSize, streamH);

        if (hasGate)
        {
            // sigmoid_logit = hidden · gate (per token) — compute per-token sigmoid scale,
            // then a scaled add into outF32.
            _kernels.LaunchMoeSigmoidLogitF32(
                sharedInput, moe.SharedExpertGateDevice, _sharedScaleBuf,
                seqLen, hiddenSize, streamH);
            _kernels.LaunchMoeAxpyScaledPerTokenF32(
                outF32, _sharedDownBuf, _sharedScaleBuf, seqLen, hiddenSize, streamH);
        }
        else
        {
            _kernels.LaunchMoeAxpyUnweightedF32(outF32, _sharedDownBuf, seqLen, hiddenSize, streamH);
        }
    }

    // ── Shared-expert per-call scratch (lazy, F32) ──
    // Allocated by EnsureSharedExpertScratch on first MoE forward that has a shared
    // expert. Sized to (capSeqLen × max(hidden, sI)) so all three reuse one alloc class
    // but each gets its own buffer for clarity.
    private nint _sharedGateBuf;     // [seqLen, sI]
    private nint _sharedUpBuf;       // [seqLen, sI]
    private nint _sharedSiluBuf;     // [seqLen, sI]
    private nint _sharedDownBuf;     // [seqLen, hidden]
    private nint _sharedScaleBuf;    // [seqLen]
    private int _sharedScratchSeqLen;
    private int _sharedScratchSI;
    private int _sharedScratchHidden;

    private void EnsureSharedExpertScratch(int seqLen, int hiddenSize, int sI)
    {
        if (seqLen <= _sharedScratchSeqLen
            && sI == _sharedScratchSI
            && hiddenSize == _sharedScratchHidden)
            return;

        FreeIfNonZero(ref _sharedGateBuf);
        FreeIfNonZero(ref _sharedUpBuf);
        FreeIfNonZero(ref _sharedSiluBuf);
        FreeIfNonZero(ref _sharedDownBuf);
        FreeIfNonZero(ref _sharedScaleBuf);

        int cap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(seqLen, 1));
        _sharedGateBuf = AllocDevice((long)cap * sI * sizeof(float));
        _sharedUpBuf = AllocDevice((long)cap * sI * sizeof(float));
        _sharedSiluBuf = AllocDevice((long)cap * sI * sizeof(float));
        _sharedDownBuf = AllocDevice((long)cap * hiddenSize * sizeof(float));
        _sharedScaleBuf = AllocDevice((long)cap * sizeof(float));
        _sharedScratchSeqLen = cap;
        _sharedScratchSI = sI;
        _sharedScratchHidden = hiddenSize;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Gemm dispatcher — quantised-direct GEMV (decode) / HGEMM-after-F16-dequant (prefill)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-layer F32-in / F32-out projection dispatcher. Routes through the smallest-VRAM
    /// path that produces correct output for the given quant type and batch shape:
    /// <list type="bullet">
    ///   <item><term>F32 weights</term><description>cuBLAS <see cref="CudaGemm.LinearF32"/>
    ///     directly on the raw weight pointer; no dequant.</description></item>
    ///   <item><term>Decode (<paramref name="seqLen"/> == 1), Q8_0</term><description>
    ///     <see cref="CudaKernels.LaunchQuantizedGemvF32In"/> — single-launch F32-in/F32-out
    ///     fused dequant+GEMV. No staging buffers needed.</description></item>
    ///   <item><term>Decode (<paramref name="seqLen"/> == 1), supported Q-type</term>
    ///     <description>Stage input F32→F16 → run MMQ/MMVQ-large
    ///     (<c>LaunchQuantizedGemvMmq</c>) or the legacy per-row
    ///     <c>LaunchQuantizedGemv</c> in F16 → stage output F16→F32. Per-call F16 staging
    ///     is tiny (≤ <c>max(M,K)</c> halfs). Mirrors the dense
    ///     <c>CudaTransformerModel.Project()</c> decode branch.</description>
    ///   </item>
    ///   <item><term>Prefill (<paramref name="seqLen"/> &gt; 1)</term><description>Dequantise
    ///     the weight tile to <see cref="_dequantScratchF16Weight"/> (F16) → stage input F32→F16 →
    ///     cuBLAS HGEMM → stage output F16→F32. Mirrors the dense prefill branch. F16 weight
    ///     scratch is <c>maxTileFloats × 2 B</c>, half the bytes of the previous F32 scratch.
    ///     </description></item>
    ///   <item><term>F16 weights</term><description>Decode goes via the F16→F16 GEMV path
    ///     directly (no dequant). Prefill copies the F16 weight to the dequant scratch (no
    ///     conversion) and runs HGEMM.</description></item>
    /// </list>
    /// The big <c>_dequantScratchF16Weight</c> persistent allocation is only touched on
    /// prefill — decode-time projections never expand a weight tile to dense memory.
    /// </summary>
    /// <param name="weight">Device pointer to raw weight bytes (quant-format or F16/F32).</param>
    /// <param name="qt">Quantization type of the weight.</param>
    /// <param name="x">Device F32 input pointer [seqLen, K].</param>
    /// <param name="y">Device F32 output pointer [seqLen, M].</param>
    /// <param name="m">Output dimension (rows of the weight matrix).</param>
    /// <param name="k">Input dimension (cols of the weight matrix).</param>
    /// <param name="seqLen">Number of input rows.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)
    {
        nint streamH = _stream.Handle;

        if (qt == QuantizationType.F32)
        {
            // Direct F32 — no dequant.
            CudaGemm.LinearF32(_cublas.Handle, x, weight, y, seqLen, k, m, streamH);
            return;
        }

        if (seqLen == 1)
        {
            // ── Decode path: quantised-direct GEMV ──
            // Special-cased Q8_0 has a single-launch F32-in/F32-out kernel that fuses
            // dequant + matmul + the F32 cast — avoids two staging launches.
            if (qt == QuantizationType.Q8_0)
            {
                _kernels.LaunchQuantizedGemvF32In(weight, x, y, m, k, streamH);
                return;
            }

            // General decode path: F32→F16 input stage → F16 GEMV (quantised-direct) →
            // F16→F32 output stage. Per-call F16 staging is <= max(m, k) halfs (~kilobytes
            // for hidden ≤ 4096) vs the multi-GB F32 weight expansion the previous path used.
            if (qt == QuantizationType.F16
                || _kernels.HasMmq(qt)
                || _kernels.HasQuantizedGemvKernel(qt))
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);

                if (qt == QuantizationType.F16)
                {
                    // F16 weight + F16 input → cuBLAS GEMV (HGEMM-1xK).
                    CudaGemm.GemvF16(_cublas.Handle, weight, _activF16InScratch,
                        _activF16OutScratch, m, k, streamH);
                }
                else if (_kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
                {
                    // Prefer MMQ/MMVQ-large where available. preqScratch=0 keeps this
                    // path stateless — we don't share a pre-Q8_1 buffer across the MoE
                    // model's projections (the dense path does, but the MoE residual
                    // stream layout doesn't pool an equivalent scratch). The on-the-fly
                    // Stage 1 path inside the kernel runs instead — adds ~µs per call,
                    // dwarfed by the VRAM win.
                    _kernels.LaunchQuantizedGemvMmq(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, preqScratch: 0, streamH);
                }
                else
                {
                    // Legacy per-row quantised GEMV — covers Q5_0 and anything MMQ doesn't.
                    _kernels.LaunchQuantizedGemv(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, streamH);
                }

                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }
        }

        // ── Prefill (seqLen > 1) and decode fallback ──
        // Dequant the weight tile to F16, stage input F32→F16, cuBLAS HGEMM, stage
        // output F16→F32. The F16 weight scratch is the model-owned _dequantScratchF16Weight
        // (sized at load to maxTileFloats halfs). All quant types that LaunchDequantToF16
        // covers route through here — same kernel coverage as the dense prefill path.
        long totalElems = (long)m * k;
        int totalElemsI = checked((int)totalElems);
        // LaunchConvertF32ToF16 / LaunchConvertF16ToF32 take `int n` — checked casts will
        // surface an OverflowException if a future caller ever uses seqLen × {k,m} > 2^31
        // (e.g. lm_head at vocab≈152k with seqLen > ~14k). Realistic contexts stay well
        // under that — no need to tile today, but make the failure mode explicit.
        int activInElems  = checked((int)((long)seqLen * k));
        int activOutElems = checked((int)((long)seqLen * m));
        EnsureActivF16InScratch(activInElems);
        EnsureActivF16OutScratch(activOutElems);

        // F16-weight branch: no dequant needed — copy is implicit (LaunchDequantToF16
        // handles QuantizationType.F16 as a DtoD copy already).
        _kernels.LaunchDequantToF16(weight, qt, _dequantScratchF16Weight, totalElemsI, streamH);
        _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, activInElems, streamH);
        CudaGemm.LinearF16(_cublas.Handle, _activF16InScratch, _dequantScratchF16Weight,
            _activF16OutScratch, seqLen, k, m, streamH);
        _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, activOutElems, streamH);
    }

    private void EnsureActivF16InScratch(long halfs)
    {
        if (halfs <= _activF16InScratchElems) return;
        FreeIfNonZero(ref _activF16InScratch);
        _activF16InScratch = AllocDevice(halfs * sizeof(ushort));
        _activF16InScratchElems = halfs;
    }

    private void EnsureActivF16OutScratch(long halfs)
    {
        if (halfs <= _activF16OutScratchElems) return;
        FreeIfNonZero(ref _activF16OutScratch);
        _activF16OutScratch = AllocDevice(halfs * sizeof(ushort));
        _activF16OutScratchElems = halfs;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Host fallbacks — temporary CPU paths used while waiting on CUDA kernels.
    //  Each one mirrors the CPU oracle exactly and is gated behind a TODO.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// TODO(CUDA-KERNEL): Replace with LaunchGdnDecayF32 (fused softplus+exp+A*).
    /// Computes g[t, vh] = exp(softplus(alpha[t, vh] + dt_bias[vh]) * A[vh]) in place on alphaBuf.
    /// </summary>
    private void LaunchGdnDecayHostFallback(nint alphaBufDev, nint dtBiasDev, nint aDev,
                                            int seqLen, int nVHead)
    {
        _stream.Synchronize();
        int n = seqLen * nVHead;
        float[] alphaHost = new float[n];
        float[] dtBiasHost = new float[nVHead];
        float[] aHost = new float[nVHead];
        fixed (float* pAlpha = alphaHost)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pAlpha, alphaBufDev,
                (nuint)((long)n * sizeof(float))).ThrowOnError();
        }
        fixed (float* pDt = dtBiasHost)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pDt, dtBiasDev,
                (nuint)((long)nVHead * sizeof(float))).ThrowOnError();
        }
        fixed (float* pA = aHost)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, aDev,
                (nuint)((long)nVHead * sizeof(float))).ThrowOnError();
        }
        for (int t = 0; t < seqLen; t++)
        {
            int off = t * nVHead;
            for (int vh = 0; vh < nVHead; vh++)
            {
                float alpha = alphaHost[off + vh] + dtBiasHost[vh];
                float sp = MathF.Log(1f + MathF.Exp(alpha));
                alphaHost[off + vh] = MathF.Exp(sp * aHost[vh]);
            }
        }
        fixed (float* pAlpha = alphaHost)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(alphaBufDev, (nint)pAlpha,
                (nuint)((long)n * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>
    /// TODO(CUDA-KERNEL): Replace with LaunchSigmoidF32 (in-place).
    /// </summary>
    private void LaunchSigmoidHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] host = new float[elems];
        fixed (float* p = host)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, bufDev,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
        for (long i = 0; i < elems; i++)
        {
            host[i] = 1f / (1f + MathF.Exp(-host[i]));
        }
        fixed (float* p = host)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)p,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>
    /// TODO(CUDA-KERNEL): Replace with LaunchSiluF32 (in-place).
    /// </summary>
    private void LaunchSiluHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] host = new float[elems];
        fixed (float* p = host)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, bufDev,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
        for (long i = 0; i < elems; i++)
        {
            float x = host[i];
            host[i] = x * (1f / (1f + MathF.Exp(-x)));
        }
        fixed (float* p = host)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)p,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>
    /// TODO(CUDA-KERNEL): Replace with LaunchSigmoidMulF32 (a *= sigmoid(b), elementwise).
    /// </summary>
    private void LaunchSigmoidMulHostFallback(nint aDev, nint bDev, long elems)
    {
        _stream.Synchronize();
        float[] aHost = new float[elems];
        float[] bHost = new float[elems];
        fixed (float* pa = aHost)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pa, aDev,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
        fixed (float* pb = bHost)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pb, bDev,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
        for (long i = 0; i < elems; i++)
        {
            float bi = bHost[i];
            aHost[i] *= 1f / (1f + MathF.Exp(-bi));
        }
        fixed (float* pa = aHost)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(aDev, (nint)pa,
                (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Disposal
    // ──────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Per-layer weight free.
        for (int i = 0; i < _layers.Length; i++)
        {
            FreeLayer(ref _layers[i]);
        }

        FreeIfNonZero(ref _dequantScratchF16Weight);
        FreeIfNonZero(ref _activF16InScratch);
        FreeIfNonZero(ref _activF16OutScratch);

        if (_ownsEmbedF32)
        {
            nint p = _embedF32Device;
            if (p != 0) CudaDriverApi.cuMemFree_v2(p);
        }

        // Output stage. Output may alias the embedding table — only free when we own it.
        nint outNormPtr = _outputNormDevice;
        if (outNormPtr != 0) CudaDriverApi.cuMemFree_v2(outNormPtr);
        if (_ownsOutputDevice)
        {
            nint outPtr = _outputDevice;
            if (outPtr != 0) CudaDriverApi.cuMemFree_v2(outPtr);
        }
        nint embPtr = _tokenEmbedDevice;
        if (embPtr != 0) CudaDriverApi.cuMemFree_v2(embPtr);

        // Shared-expert scratch (lazily allocated).
        FreeIfNonZero(ref _sharedGateBuf);
        FreeIfNonZero(ref _sharedUpBuf);
        FreeIfNonZero(ref _sharedSiluBuf);
        FreeIfNonZero(ref _sharedDownBuf);
        FreeIfNonZero(ref _sharedScaleBuf);

        // Per-attention-layer F32 KV cache.
        if (_f32KCache is not null)
        {
            for (int i = 0; i < _f32KCache.Length; i++)
            {
                if (_f32KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f32KCache[i]);
                if (_f32VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f32VCache[i]);
            }
            _f32KCache = null;
            _f32VCache = null;
        }

        _moeScratch.Dispose();
        _state.Dispose();
        _gdnCache.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();

        GC.SuppressFinalize(this);
    }

    private static void FreeLayer(ref DeviceLayer layer)
    {
        FreeIfNonZero(ref layer.AttnNormWeightDevice);
        FreeIfNonZero(ref layer.PostAttnNormWeightDevice);

        if (layer.Gdn is { } gdn)
        {
            FreeIfNonZero(ref gdn.QkvDevice);
            FreeIfNonZero(ref gdn.GateDevice);
            FreeIfNonZero(ref gdn.AlphaDevice);
            FreeIfNonZero(ref gdn.BetaDevice);
            FreeIfNonZero(ref gdn.Conv1dWeightDevice);
            FreeIfNonZero(ref gdn.Conv1dBiasDevice);
            FreeIfNonZero(ref gdn.ADevice);
            FreeIfNonZero(ref gdn.DtBiasDevice);
            FreeIfNonZero(ref gdn.SsmNormDevice);
            FreeIfNonZero(ref gdn.OutDevice);
            layer.Gdn = gdn;
        }
        if (layer.FullAttn is { } attn)
        {
            FreeIfNonZero(ref attn.QDevice);
            FreeIfNonZero(ref attn.KDevice);
            FreeIfNonZero(ref attn.VDevice);
            FreeIfNonZero(ref attn.ODevice);
            FreeIfNonZero(ref attn.QNormDevice);
            FreeIfNonZero(ref attn.KNormDevice);
            layer.FullAttn = attn;
        }
        var moe = layer.Moe;
        FreeIfNonZero(ref moe.GateRouterDevice);
        FreeIfNonZero(ref moe.GateExpsDevice);
        FreeIfNonZero(ref moe.UpExpsDevice);
        FreeIfNonZero(ref moe.DownExpsDevice);
        FreeIfNonZero(ref moe.SharedExpertGateDevice);
        if (moe.SharedGateDevice is not null)
        {
            for (int s = 0; s < moe.SharedGateDevice.Length; s++)
            {
                if (moe.SharedGateDevice[s] != 0) CudaDriverApi.cuMemFree_v2(moe.SharedGateDevice[s]);
                if (moe.SharedUpDevice[s] != 0) CudaDriverApi.cuMemFree_v2(moe.SharedUpDevice[s]);
                if (moe.SharedDownDevice[s] != 0) CudaDriverApi.cuMemFree_v2(moe.SharedDownDevice[s]);
            }
        }
        layer.Moe = moe;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Static helpers
    // ──────────────────────────────────────────────────────────────────────

    private static nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return ptr;
    }

    private static void CopyHtoD(nint dst, nint src, long bytes)
    {
        CudaDriverApi.cuMemcpyHtoD_v2(dst, src, (nuint)bytes).ThrowOnError();
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            CudaDriverApi.cuMemFree_v2(ptr);
            ptr = 0;
        }
    }

    /// <summary>
    /// Dequantizes a small tensor to F32 on the host then uploads to device.
    /// Used for norms, A, dt_bias, ssm_norm, conv1d_weight (host-dequant is cheap for
    /// these shapes).
    /// </summary>
    private static nint UploadF32Tensor(nint dataBase, GgufTensorDescriptor desc, int expectedElems)
    {
        float[] host = new float[expectedElems];
        Dequantize.ToFloat32(dataBase + (nint)desc.DataOffset, expectedElems,
            desc.QuantizationType, host);
        nint device = AllocDevice((long)expectedElems * sizeof(float));
        fixed (float* p = host)
        {
            CopyHtoD(device, (nint)p, (long)expectedElems * sizeof(float));
        }
        return device;
    }

    /// <summary>
    /// Uploads a tensor's raw quantized bytes to device unmodified. The output device
    /// pointer holds the same byte representation as the source mmap region; dequant
    /// happens at GEMM time via <see cref="Gemm"/>.
    /// </summary>
    private static nint UploadRawTensor(nint dataBase, GgufTensorDescriptor desc)
    {
        int innerDim = desc.Shape[0];
        long outerDim = desc.Shape.ElementCount / innerDim;
        long bytes = Dequantize.RowByteSize(innerDim, desc.QuantizationType) * outerDim;
        nint device = AllocDevice(bytes);
        CopyHtoD(device, dataBase + (nint)desc.DataOffset, bytes);
        return device;
    }

    private static void UpdateMaxTile(ref long max, long candidate)
    {
        if (candidate > max) max = candidate;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-layer device-side bundles
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Per-layer device pointers (norms + token-mixing + MoE).</summary>
    internal struct DeviceLayer
    {
        public nint AttnNormWeightDevice;
        public nint PostAttnNormWeightDevice;
        public DeviceGdn? Gdn;
        public DeviceFullAttn? FullAttn;
        public DeviceMoe Moe;
        /// <summary>
        /// Host-side MoE bundle retained for the on-the-fly raw-quant view of the routed
        /// experts. The CUDA forward needs the descriptor metadata even when the actual
        /// bytes live on device — this keeps the bookkeeping out of <see cref="DeviceMoe"/>.
        /// </summary>
        public MoeLayerWeights MoeHost;
    }

    /// <summary>Device-side GDN token-mixing weights.</summary>
    internal struct DeviceGdn
    {
        public nint QkvDevice;
        public QuantizationType QkvQt;
        public int QkvInputDim;
        public int QkvOutputDim;

        public nint GateDevice;
        public QuantizationType GateQt;
        public int GateInputDim;
        public int GateOutputDim;

        public nint AlphaDevice;
        public QuantizationType AlphaQt;
        public int AlphaInputDim;
        public int AlphaOutputDim;

        public nint BetaDevice;
        public QuantizationType BetaQt;
        public int BetaInputDim;
        public int BetaOutputDim;

        public nint Conv1dWeightDevice;
        public nint Conv1dBiasDevice;
        public nint ADevice;
        public nint DtBiasDevice;
        public nint SsmNormDevice;

        public nint OutDevice;
        public QuantizationType OutQt;
        public int OutInputDim;
        public int OutOutputDim;
    }

    /// <summary>Device-side full-attention weights.</summary>
    internal struct DeviceFullAttn
    {
        public nint QDevice;
        public QuantizationType QQt;
        public int QInputDim;
        public int QOutputDim;

        public nint KDevice;
        public QuantizationType KQt;
        public int KInputDim;
        public int KOutputDim;

        public nint VDevice;
        public QuantizationType VQt;
        public int VInputDim;
        public int VOutputDim;

        public nint ODevice;
        public QuantizationType OQt;
        public int OInputDim;
        public int OOutputDim;

        public int NumKvHeads;
        public nint QNormDevice;
        public nint KNormDevice;
    }

    /// <summary>Device-side MoE weights — router + routed-expert raw views + shared experts.</summary>
    internal struct DeviceMoe
    {
        public nint GateRouterDevice;     // F32 [numExperts, hiddenSize]
        public int NumExperts;
        public int NumExpertsPerTok;
        public int IntermediateSize;
        public bool NormTopKProb;

        // Routed experts — single fused tensor per (gate/up/down) holding all experts.
        public nint GateExpsDevice;
        public QuantizationType GateExpsQt;
        public int GateExpsMDim;
        public int GateExpsKDim;

        public nint UpExpsDevice;
        public QuantizationType UpExpsQt;
        public int UpExpsMDim;
        public int UpExpsKDim;

        public nint DownExpsDevice;
        public QuantizationType DownExpsQt;
        public int DownExpsMDim;
        public int DownExpsKDim;

        // Shared experts — F32 device pointers.
        public nint[] SharedGateDevice;
        public nint[] SharedUpDevice;
        public nint[] SharedDownDevice;
        public int SharedIntermediateSize;
        public nint SharedExpertGateDevice; // Qwen1.5-MoE sigmoid gate; 0 when absent.
    }
}
