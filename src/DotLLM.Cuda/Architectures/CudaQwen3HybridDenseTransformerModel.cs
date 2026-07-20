using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the Qwen3HybridDense (<c>qwen35</c>) model — e.g. PrismML's
/// Bonsai-27B. F32 activations throughout — mirrors
/// <c>DotLLM.Models.Architectures.Qwen3HybridDenseTransformerModel</c> (CPU) on the GPU.
/// Adapted from <see cref="CudaQwen3MoeHybridTransformerModel"/>: the GDN and
/// full-attention token-mixing sub-layers are byte-for-byte identical (shared
/// <see cref="HybridLayerLayout"/>/<see cref="GatedDeltaNetConfig"/> infrastructure); the
/// only structural difference is the FFN sub-layer — dense SwiGLU (this class) instead
/// of sparse MoE routing.
/// </summary>
/// <remarks>
/// <para>
/// Each of the <c>numLayers</c> layers has a token-mixing sub-layer (GDN recurrence or
/// full GQA attention) followed by a dense SwiGLU FFN. Layer kind for every index comes
/// from <see cref="ModelConfig.HybridLayout"/>; full-attention layers are placed every
/// <see cref="GatedDeltaNetConfig.FullAttnInterval"/> steps (Bonsai-27B: interval = 4 over
/// 64 layers → 16 full-attention layers, 48 GDN layers).
/// </para>
/// <para>
/// Unlike <see cref="CudaQwen3MoeHybridTransformerModel"/>'s <c>Gemm</c> dispatcher, this
/// class's <see cref="Gemm"/> has explicit I2_S / PQ2_0 branches — Bonsai-27B ships PQ2_0
/// ternary weights, so the ternary GEMV kernels (<see cref="CudaKernels.LaunchPQ2_0GemvF16In"/>
/// / <see cref="CudaKernels.LaunchDequantPQ2_0ToF16"/>) must be reachable from every
/// projection site (GDN, attention, and dense FFN).
/// </para>
/// </remarks>
public sealed unsafe class CudaQwen3HybridDenseTransformerModel : IModel
{
    private readonly CudaQwen3HybridDenseForwardState _state;
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
    private readonly nint _outputNormDevice; // F32 [hiddenSize]
    private readonly nint _outputDevice;     // lm_head raw quant bytes (may alias _tokenEmbedDevice)
    private readonly QuantizationType _outputQt;
    private readonly int _outputOutputDim;   // vocab size
    private readonly int _outputInputDim;    // hidden size
    private readonly bool _ownsOutputDevice; // false when aliased to embed

    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int _intermediateSize;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _gdnLayerOrdinal;

    private readonly float _ropeTheta;
    private readonly int _ropeDim;

    // Model-owned device F16 scratch for on-the-fly weight dequant in the prefill path
    // (seqLen > 1). See CudaQwen3MoeHybridTransformerModel's field doc for the full
    // rationale — identical convention here.
    private nint _dequantScratchF16Weight;

    // Lazily allocated F16 activation staging buffers for the decode/prefill F16 GEMV/GEMM
    // path. Activations live in F32; the quantised GEMV kernels and cuBLAS HGEMM consume F16.
    private nint _activF16InScratch;
    private long _activF16InScratchElems;
    private nint _activF16OutScratch;
    private long _activF16OutScratchElems;

    // Host-side per-row embedding lookup (NOT a full-table GPU pre-dequant — see the
    // LoadFromGguf remarks for why). Points at the mmap'd GGUF data region backing
    // token_embd.weight; each Forward call dequantizes only its `seqLen` rows on the CPU
    // and H2D-copies the tiny result.
    private readonly nint _embedDataBase;
    private readonly ulong _embedDataOffset;
    private readonly long _embedRowBytes;

    // Per-attention-layer F16 KV cache. Sized lazily on first kvCache-enabled Forward call.
    // See CudaQwen3MoeHybridTransformerModel's field doc for the full rationale.
    private nint[]? _f16KCache;
    private nint[]? _f16VCache;
    private int _f16CacheMaxSeqLen;
    private int _f16CacheCurrentLength;

    private nint _f16KvWriteStaging;
    private long _f16KvWriteStagingElems;
    private nint _f32KvReadStagingK;
    private nint _f32KvReadStagingV;
    private long _f32KvReadStagingElems;

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — matches the sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>
    /// Creates a length-only <see cref="IKvCache"/> handle sized to <paramref name="maxSeqLen"/>.
    /// K/V storage is owned internally by this model (a per-attention-layer F16 device
    /// cache) — the returned handle only communicates the capacity to
    /// <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>.
    /// </summary>
    public CudaHybridKvCacheHandle CreateKvCache(int maxSeqLen) => new(maxSeqLen);

    private CudaQwen3HybridDenseTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        DeviceLayer[] layers,
        nint tokenEmbedDevice, QuantizationType tokenEmbedQt,
        nint embedDataBase, ulong embedDataOffset, long embedRowBytes,
        nint outputNormDevice,
        nint outputDevice, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        bool ownsOutputDevice,
        int[] kvSlotForLayer, int attentionLayerCount,
        float ropeTheta, int ropeDim,
        CudaQwen3HybridDenseForwardState state, CudaGdnStateCache gdnCache,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context, CudaKernels kernels,
        int deviceId,
        nint dequantScratchDevice)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _tokenEmbedDevice = tokenEmbedDevice;
        _tokenEmbedQt = tokenEmbedQt;
        _embedDataBase = embedDataBase;
        _embedDataOffset = embedDataOffset;
        _embedRowBytes = embedRowBytes;
        _outputNormDevice = outputNormDevice;
        _outputDevice = outputDevice;
        _outputQt = outputQt;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _ownsOutputDevice = ownsOutputDevice;
        _layout = config.HybridLayout!;
        _gdn = config.GdnConfig!.Value;
        _intermediateSize = config.IntermediateSize;
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
        _dequantScratchF16Weight = dequantScratchDevice;

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
    /// Loads a Qwen3HybridDense model from an opened GGUF file onto the given CUDA device.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for the model's lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX. Null auto-detects.</param>
    public static CudaQwen3HybridDenseTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"CudaQwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", nameof(config));

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
        // Raw quant bytes are uploaded (small — e.g. ~340 MB at Bonsai's PQ2_0 packing for a
        // 248320-vocab table) for the lm_head-tied case. Unlike
        // CudaQwen3MoeHybridTransformerModel (built for A6000/H100-class VRAM budgets), this
        // model targets a 12 GB consumer card, and Bonsai's huge vocab makes a full-table F32
        // pre-dequant (~5 GB) plus the prefill dequant scratch (~2.5 GB, sized to the largest
        // single tile — the lm_head) push total VRAM demand past 12 GB. That silently trips
        // WDDM's host-RAM paging fallback (confirmed via
        // `Get-Counter '\GPU Process Memory(*)\Shared Usage'` showing multi-GB "shared usage"
        // for the process) — a hang-that-isn't-a-hang, not a deadlock: every GPU memory access
        // becomes a PCIe round-trip once oversubscribed. So the embedding LOOKUP (as opposed to
        // the lm_head projection) is done as a per-call host-side row dequant in Forward()
        // instead — only `seqLen` rows are ever needed, never all 248320.
        var embDesc = tensors["token_embd.weight"];
        long embRowBytes = Dequantize.RowByteSize(hiddenSize, embDesc.QuantizationType);
        long embTotalBytes = embRowBytes * config.VocabSize;
        nint tokenEmbedDevice = AllocDevice(embTotalBytes);
        CopyHtoD(tokenEmbedDevice, dataBase + (nint)embDesc.DataOffset, embTotalBytes);

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

        // ── lm_head (tied to token embedding when output.weight is absent) ──
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
                $"Qwen3HybridDense rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        // ── Per-layer load ──
        var layers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        long maxTileFloats = 0;

        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayerDevice(i, dataBase, tensors, config, ref maxTileFloats);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        maxTileFloats = Math.Max(maxTileFloats, (long)outputOutputDim * outputInputDim);
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        var gdn = config.GdnConfig!.Value;
        var state = new CudaQwen3HybridDenseForwardState(
            hiddenSize: hiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * gdn.NKHead + gdn.NVHead) * gdn.DState,
            dConv: gdn.DConv,
            nVHead: gdn.NVHead,
            nKHead: gdn.NKHead,
            dState: gdn.DState,
            intermediateSize: config.IntermediateSize);

        int gdnLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet) gdnLayerCount++;
        var gdnCache = new CudaGdnStateCache(gdn, gdnLayerCount);

        return new CudaQwen3HybridDenseTransformerModel(
            config, gguf, layers,
            tokenEmbedDevice, embDesc.QuantizationType,
            dataBase, embDesc.DataOffset, embRowBytes,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, gdnCache, stream, cublas, context, kernels, deviceId,
            dequantScratchDevice);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-layer loaders (host → device upload of raw quant bytes)
    // ──────────────────────────────────────────────────────────────────────

    private static DeviceLayer LoadLayerDevice(
        int layerIdx, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, ref long maxTileFloats)
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
                    $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3HybridDense.");
        }

        // Dense SwiGLU FFN — ffn_gate.weight, ffn_up.weight, ffn_down.weight (no MoE
        // routing; no "_exps" suffix, confirmed against the real Ternary-Bonsai-27B-Q2_0.gguf).
        var gateDesc = tensors[$"{prefix}.ffn_gate.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down.weight"];
        nint gateDevice = UploadRawTensor(dataBase, gateDesc);
        nint upDevice = UploadRawTensor(dataBase, upDesc);
        nint downDevice = UploadRawTensor(dataBase, downDesc);
        UpdateMaxTile(ref maxTileFloats, (long)gateDesc.Shape[0] * gateDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)upDesc.Shape[0] * upDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)downDesc.Shape[0] * downDesc.Shape[1]);

        return new DeviceLayer
        {
            AttnNormWeightDevice = attnNormDevice,
            PostAttnNormWeightDevice = postAttnNormDevice,
            Gdn = gdnDev,
            FullAttn = attnDev,

            GateWeight = gateDevice, GateQt = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0], GateOutputDim = gateDesc.Shape[1],

            UpWeight = upDevice, UpQt = upDesc.QuantizationType,
            UpInputDim = upDesc.Shape[0], UpOutputDim = upDesc.Shape[1],

            DownWeight = downDevice, DownQt = downDesc.QuantizationType,
            DownInputDim = downDesc.Shape[0], DownOutputDim = downDesc.Shape[1],
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

        nint qkvDevice = UploadRawTensor(dataBase, qkvDesc);
        nint gateDevice = UploadRawTensor(dataBase, gateDesc);
        nint alphaDevice = UploadRawTensor(dataBase, alphaDesc);
        nint betaDevice = UploadRawTensor(dataBase, betaDesc);
        nint outDevice = UploadRawTensor(dataBase, outDesc);

        nint conv1dWeightDevice = UploadF32Tensor(dataBase, conv1dWDesc, gdn.DConv * convDim);
        nint conv1dBiasDevice = AllocDevice((long)convDim * sizeof(float));
        CudaDriverApi.cuMemsetD8_v2(conv1dBiasDevice, 0, (nuint)((long)convDim * sizeof(float)))
            .ThrowOnError();

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
                $"{prefix}.attn_q.weight has output dim {q.Shape[1]} but qwen35 expects " +
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

        // Host-side per-row embedding lookup — see LoadFromGguf's remarks on why this isn't
        // a GPU-resident full-table dequant. Dequantize each token's row (any quant type)
        // into a small host buffer, then a single bulk H2D copy into HiddenState.
        float[] embedHost = new float[(long)seqLen * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            nint rowSrc = _embedDataBase + (nint)(_embedDataOffset + (ulong)tokenIds[t] * (ulong)_embedRowBytes);
            Dequantize.ToFloat32(rowSrc, hiddenSize, _tokenEmbedQt,
                embedHost.AsSpan(t * hiddenSize, hiddenSize));
        }
        fixed (float* pEmbedHost = embedHost)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.HiddenState, (nint)pEmbedHost,
                (nuint)((long)seqLen * hiddenSize * sizeof(float)), streamH).ThrowOnError();
        }

        for (int layer = 0; layer < _layers.Length; layer++)
        {
            RunSingleLayerBody(layer, seqLen, positions, hiddenSize,
                numHeads, numKvHeads, headDim, eps, kvCache);
        }

        if (DebugTrace) { _stream.Synchronize(); Console.Error.WriteLine("[hybrid-debug] all layers done, starting lm_head"); Console.Error.Flush(); }
        _kernels.LaunchRmsNormF32(_state.HiddenState, _outputNormDevice, _state.HiddenState,
            hiddenSize, eps, seqLen, streamH);
        Gemm(_outputDevice, _outputQt, _state.HiddenState, _state.Logits,
             _outputOutputDim, _outputInputDim, seqLen);
        if (DebugTrace) { _stream.Synchronize(); Console.Error.WriteLine("[hybrid-debug] lm_head done"); Console.Error.Flush(); }

        _stream.Synchronize();

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.Logits,
            (nuint)((long)seqLen * vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    private void RunSingleLayerBody(int layerIdx, int seqLen, ReadOnlySpan<int> positions,
        int hiddenSize, int numHeads, int numKvHeads, int headDim, float eps, IKvCache? kvCache)
    {
        nint streamH = _stream.Handle;
        long hiddenBytes = (long)seqLen * hiddenSize * sizeof(float);
        var kinds = _layout.LayerKind;
        ref readonly DeviceLayer lw = ref _layers[layerIdx];

        bool debug = DebugTrace;
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} start ({kinds[layerIdx]})"); Console.Error.Flush(); }

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
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} token-mixing done"); Console.Error.Flush(); }

        // 2. First residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);

        // 3. Dense FFN — residual = hidden; normOut = RmsNorm(hidden, post_attn_norm).
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
            (nuint)hiddenBytes, streamH).ThrowOnError();
        _kernels.LaunchRmsNormF32(_state.HiddenState, lw.PostAttnNormWeightDevice, _state.NormOutput,
            hiddenSize, eps, seqLen, streamH);

        ForwardDenseFfnBody(lw, seqLen, hiddenSize);
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} ffn done"); Console.Error.Flush(); }

        // 4. Second residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);
    }

    private static readonly bool DebugTrace =
        Environment.GetEnvironmentVariable("DOTLLM_HYBRID_DEBUG") == "1";

    // ──────────────────────────────────────────────────────────────────────
    //  GDN token-mixing body — verbatim from CudaQwen3MoeHybridTransformerModel
    //  (architecture-agnostic to the FFN kind).
    // ──────────────────────────────────────────────────────────────────────

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
        nint convStateDev = _gdnCache.GetConvStatePtr(gdnOrdinal);
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convInput, convStateDev,
            (nuint)convStateBytes, streamH).ThrowOnError();
        long qkvRowsBytes = (long)seqLen * convDim * sizeof(float);
        nint convInputQkvOff = convInput + (nint)convStateBytes;
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convInputQkvOff, qkvBuf,
            (nuint)qkvRowsBytes, streamH).ThrowOnError();

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

        nint trailRowsSrc = convInput + (nint)((long)seqLen * convDim * sizeof(float));
        CudaDriverApi.cuMemcpyDtoDAsync_v2(convStateDev, trailRowsSrc,
            (nuint)convStateBytes, streamH).ThrowOnError();

        // ── 4. De-interleave Q/K/V from conv output, L2-normalise Q and K per head ──
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
        _kernels.LaunchRmsNormF32(gdnOut, gdnW.SsmNormDevice, gdnOut,
            dState, eps, seqLen * nVHead, streamH);
        _kernels.LaunchSwiGLUF32(zBuf, gdnOut, gdnOut, vDim, seqLen, streamH);

        // ── 7. ssm_out projection into NormOutput ──
        Gemm(gdnW.OutDevice, gdnW.OutQt, gdnOut, normOut,
             gdnW.OutOutputDim, gdnW.OutInputDim, seqLen);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Full GQA attention body — verbatim from CudaQwen3MoeHybridTransformerModel.
    // ──────────────────────────────────────────────────────────────────────

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

        // ── 2. De-interleave QG → Q and Gate ──
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

        // ── 4. Per-head QK-norm ──
        _kernels.LaunchRmsNormF32(q, attn.QNormDevice, q,
            headDim, eps, seqLen * numHeads, streamH);
        _kernels.LaunchRmsNormF32(k, attn.KNormDevice, k,
            headDim, eps, seqLen * numKvHeads, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postnorm", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postnorm", k, seqLen, numKvHeads * headDim);

        // ── 5. RoPE — partial-rotary NeoX ──
        _kernels.LaunchRoPEF32(q, k, _state.PositionsDevice,
            seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, 1, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postrope", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postrope", k, seqLen, numKvHeads * headDim);

        // ── 6. Attention (GQA with causal mask) ──
        if (kvCache is not null)
        {
            EnsureF16KvCache(kvCache.MaxLength, numKvHeads, headDim);
            int slot = _kvSlotForLayer[layer];
            if (slot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} is not a full-attention layer but ForwardFullAttnBody was invoked.");
            WriteF16KvRows(slot, k, v, positions, numKvHeads, headDim);

            int positionOffset = positions[0];
            int seqKv = _f16CacheCurrentLength;
            int kvLiveElems = seqKv * kvElems;

            EnsureF32KvReadStaging(seqKv, kvElems);
            _kernels.LaunchConvertF16ToF32(_f16KCache![slot], _f32KvReadStagingK, kvLiveElems, streamH);
            _kernels.LaunchConvertF16ToF32(_f16VCache![slot], _f32KvReadStagingV, kvLiveElems, streamH);

            _kernels.LaunchAttentionF32(q, _f32KvReadStagingK, _f32KvReadStagingV, attnOut,
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
    //  Per-attention-layer F16 KV cache (model-private) — verbatim.
    // ──────────────────────────────────────────────────────────────────────

    private void EnsureF16KvCache(int maxSeqLen, int numKvHeads, int headDim)
    {
        if (_f16KCache is not null && maxSeqLen <= _f16CacheMaxSeqLen) return;

        if (_f16KCache is not null)
        {
            for (int i = 0; i < _f16KCache.Length; i++)
            {
                if (_f16KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f16KCache[i]);
                if (_f16VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f16VCache[i]);
            }
        }

        _f16KCache = new nint[_attentionLayerCount];
        _f16VCache = new nint[_attentionLayerCount];
        long bytesPerLayer = (long)maxSeqLen * numKvHeads * headDim * sizeof(ushort);
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _f16KCache[i] = AllocDevice(bytesPerLayer);
            _f16VCache[i] = AllocDevice(bytesPerLayer);
        }
        _f16CacheMaxSeqLen = maxSeqLen;
        _f16CacheCurrentLength = 0;
    }

    private void EnsureF16KvWriteStaging(int seqLen, int kvElems)
    {
        long needed = (long)seqLen * kvElems;
        if (needed <= _f16KvWriteStagingElems) return;

        long grown = _f16KvWriteStagingElems == 0 ? 256L : _f16KvWriteStagingElems;
        while (grown < needed) grown *= 2;

        FreeIfNonZero(ref _f16KvWriteStaging);
        _f16KvWriteStaging = AllocDevice(grown * sizeof(ushort));
        _f16KvWriteStagingElems = grown;
    }

    private void EnsureF32KvReadStaging(int seqKv, int kvElems)
    {
        long needed = (long)seqKv * kvElems;
        if (needed <= _f32KvReadStagingElems) return;

        long grown = _f32KvReadStagingElems == 0 ? 256L : _f32KvReadStagingElems;
        while (grown < needed) grown *= 2;

        FreeIfNonZero(ref _f32KvReadStagingK);
        FreeIfNonZero(ref _f32KvReadStagingV);
        _f32KvReadStagingK = AllocDevice(grown * sizeof(float));
        _f32KvReadStagingV = AllocDevice(grown * sizeof(float));
        _f32KvReadStagingElems = grown;
    }

    private void WriteF16KvRows(int layerSlot, nint kSrcF32, nint vSrcF32,
                                 ReadOnlySpan<int> positions, int numKvHeads, int headDim)
    {
        nint streamH = _stream.Handle;
        int seqLen = positions.Length;
        int kvElems = numKvHeads * headDim;
        long rowBytes = (long)kvElems * sizeof(ushort);
        int totalElems = seqLen * kvElems;

        bool contiguous = seqLen > 0;
        int maxPos = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            int p = positions[i];
            if ((uint)p >= (uint)_f16CacheMaxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {p} at index {i} exceeds F16 KV cache capacity {_f16CacheMaxSeqLen}.");
            if (p > maxPos) maxPos = p;
            if (i > 0 && positions[i] != positions[i - 1] + 1) contiguous = false;
        }

        EnsureF16KvWriteStaging(seqLen, kvElems);
        nint stagingF16 = _f16KvWriteStaging;
        nint kBase = _f16KCache![layerSlot];
        nint vBase = _f16VCache![layerSlot];

        _kernels.LaunchConvertF32ToF16(kSrcF32, stagingF16, totalElems, streamH);
        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint kDst = kBase + (nint)((long)positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, stagingF16, (nuint)bulkBytes, streamH).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                nint kDst = kBase + (nint)((long)positions[i] * rowBytes);
                nint kS = stagingF16 + (nint)((long)i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kS, (nuint)rowBytes, streamH).ThrowOnError();
            }
        }

        _kernels.LaunchConvertF32ToF16(vSrcF32, stagingF16, totalElems, streamH);
        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint vDst = vBase + (nint)((long)positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, stagingF16, (nuint)bulkBytes, streamH).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                nint vDst = vBase + (nint)((long)positions[i] * rowBytes);
                nint vS = stagingF16 + (nint)((long)i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vS, (nuint)rowBytes, streamH).ThrowOnError();
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _f16CacheCurrentLength)
            _f16CacheCurrentLength = newLength;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Dense SwiGLU FFN body (replaces CudaQwen3MoeHybridTransformerModel's
    //  ForwardMoeBody).
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dense SwiGLU FFN forward. Reads pre-normed activations from
    /// <see cref="CudaQwen3HybridDenseForwardState.NormOutput"/> and writes the FFN output
    /// back to the same buffer. Entirely on-device — no host round-trip.
    /// </summary>
    private void ForwardDenseFfnBody(in DeviceLayer lw, int seqLen, int hiddenSize)
    {
        nint streamH = _stream.Handle;
        nint normOut = _state.NormOutput;
        nint ffnGate = _state.FfnGate;
        nint ffnUp = _state.FfnUp;
        nint siluOut = _state.SiluOutput;

        Gemm(lw.GateWeight, lw.GateQt, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
        Gemm(lw.UpWeight, lw.UpQt, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);
        _kernels.LaunchSwiGLUF32(ffnGate, ffnUp, siluOut, _intermediateSize, seqLen, streamH);
        Gemm(lw.DownWeight, lw.DownQt, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Gemm dispatcher — quantised-direct GEMV (decode) / HGEMM-after-F16-dequant (prefill)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-layer F32-in / F32-out projection dispatcher. Adapted from
    /// <see cref="CudaQwen3MoeHybridTransformerModel.Gemm"/> with explicit I2_S / PQ2_0
    /// branches added — Bonsai-27B ships PQ2_0 ternary weights end-to-end (GDN projections,
    /// attention projections, and the dense FFN), so the ternary GEMV/dequant kernels must
    /// be reachable from every call site. The MoE hybrid's Gemm lacks these branches; that
    /// is a pre-existing, separate gap (tracked, not fixed here — out of scope for this
    /// dense-architecture addition).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)
    {
        nint streamH = _stream.Handle;

        if (qt == QuantizationType.F32)
        {
            CudaGemm.LinearF32(_cublas.Handle, x, weight, y, seqLen, k, m, streamH);
            return;
        }

        if (seqLen == 1)
        {
            if (qt == QuantizationType.Q8_0)
            {
                _kernels.LaunchQuantizedGemvF32In(weight, x, y, m, k, streamH);
                return;
            }

            if (qt == QuantizationType.I2_S)
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);
                _kernels.LaunchI2_SGemvF16In(weight, _activF16InScratch, _activF16OutScratch, m, k, streamH);
                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }

            if (qt == QuantizationType.PQ2_0)
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);
                _kernels.LaunchPQ2_0GemvF16In(weight, _activF16InScratch, _activF16OutScratch, m, k, streamH);
                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }

            if (qt == QuantizationType.F16
                || _kernels.HasMmq(qt)
                || _kernels.HasQuantizedGemvKernel(qt))
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);

                if (qt == QuantizationType.F16)
                {
                    CudaGemm.GemvF16(_cublas.Handle, weight, _activF16InScratch,
                        _activF16OutScratch, m, k, streamH);
                }
                else if (_kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
                {
                    _kernels.LaunchQuantizedGemvMmq(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, preqScratch: 0, streamH);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, streamH);
                }

                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }
        }

        // ── Prefill (seqLen > 1) and decode fallback ──
        long totalElems = (long)m * k;
        int totalElemsI = checked((int)totalElems);
        int activInElems = checked((int)((long)seqLen * k));
        int activOutElems = checked((int)((long)seqLen * m));
        EnsureActivF16InScratch(activInElems);
        EnsureActivF16OutScratch(activOutElems);

        if (qt == QuantizationType.I2_S)
            _kernels.LaunchDequantI2_SToF16(weight, _dequantScratchF16Weight, m, k, streamH);
        else if (qt == QuantizationType.PQ2_0)
            _kernels.LaunchDequantPQ2_0ToF16(weight, _dequantScratchF16Weight, m, k, streamH);
        else
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
    //  Verbatim from CudaQwen3MoeHybridTransformerModel.
    // ──────────────────────────────────────────────────────────────────────

    private void LaunchGdnDecayHostFallback(nint alphaBufDev, nint dtBiasDev, nint aDev,
        int seqLen, int nVHead)
    {
        _stream.Synchronize();
        float[] alpha = new float[seqLen * nVHead];
        float[] dtBias = new float[nVHead];
        float[] a = new float[nVHead];
        fixed (float* pAlpha = alpha)
        fixed (float* pDtBias = dtBias)
        fixed (float* pA = a)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pAlpha, alphaBufDev, (nuint)(alpha.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pDtBias, dtBiasDev, (nuint)(dtBias.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, aDev, (nuint)(a.Length * sizeof(float))).ThrowOnError();

            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < nVHead; h++)
                {
                    int idx = t * nVHead + h;
                    float x = alpha[idx] + dtBias[h];
                    float softplus = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
                    alpha[idx] = MathF.Exp(softplus * a[h]);
                }
            }

            CudaDriverApi.cuMemcpyHtoD_v2(alphaBufDev, (nint)pAlpha, (nuint)(alpha.Length * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSigmoidHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] buf = new float[elems];
        fixed (float* pBuf = buf)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pBuf, bufDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                buf[i] = 1f / (1f + MathF.Exp(-buf[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)pBuf, (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSiluHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] buf = new float[elems];
        fixed (float* pBuf = buf)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pBuf, bufDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                buf[i] = buf[i] / (1f + MathF.Exp(-buf[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)pBuf, (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSigmoidMulHostFallback(nint aDev, nint bDev, long elems)
    {
        _stream.Synchronize();
        float[] a = new float[elems];
        float[] b = new float[elems];
        fixed (float* pA = a)
        fixed (float* pB = b)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, aDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pB, bDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                a[i] *= 1f / (1f + MathF.Exp(-b[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(aDev, (nint)pA, (nuint)(elems * sizeof(float))).ThrowOnError();
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

        for (int i = 0; i < _layers.Length; i++)
        {
            FreeLayer(ref _layers[i]);
        }

        FreeIfNonZero(ref _dequantScratchF16Weight);
        FreeIfNonZero(ref _activF16InScratch);
        FreeIfNonZero(ref _activF16OutScratch);

        nint outNormPtr = _outputNormDevice;
        if (outNormPtr != 0) CudaDriverApi.cuMemFree_v2(outNormPtr);
        if (_ownsOutputDevice)
        {
            nint outPtr = _outputDevice;
            if (outPtr != 0) CudaDriverApi.cuMemFree_v2(outPtr);
        }
        nint embPtr = _tokenEmbedDevice;
        if (embPtr != 0) CudaDriverApi.cuMemFree_v2(embPtr);

        if (_f16KCache is not null)
        {
            for (int i = 0; i < _f16KCache.Length; i++)
            {
                if (_f16KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f16KCache[i]);
                if (_f16VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f16VCache[i]);
            }
            _f16KCache = null;
            _f16VCache = null;
        }
        FreeIfNonZero(ref _f16KvWriteStaging);
        FreeIfNonZero(ref _f32KvReadStagingK);
        FreeIfNonZero(ref _f32KvReadStagingV);

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
        FreeIfNonZero(ref layer.GateWeight);
        FreeIfNonZero(ref layer.UpWeight);
        FreeIfNonZero(ref layer.DownWeight);
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

    /// <summary>Per-layer device pointers (norms + token-mixing + dense FFN).</summary>
    internal struct DeviceLayer
    {
        public nint AttnNormWeightDevice;
        public nint PostAttnNormWeightDevice;
        public DeviceGdn? Gdn;
        public DeviceFullAttn? FullAttn;

        public nint GateWeight;
        public QuantizationType GateQt;
        public int GateInputDim;
        public int GateOutputDim;

        public nint UpWeight;
        public QuantizationType UpQt;
        public int UpInputDim;
        public int UpOutputDim;

        public nint DownWeight;
        public QuantizationType DownQt;
        public int DownInputDim;
        public int DownOutputDim;
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
}
