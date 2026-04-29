using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Per-layer GPU weight pointers. All linear projections stored as FP16 on device.
/// </summary>
internal readonly struct CudaLayerWeights
{
    // FP16 dequantized weights on device [outputDim, inputDim] (for prefill GEMM)
    public readonly nint Q, K, V, O, Gate, Up, Down;
    // Original quantized weights on device (for decode quantized GEMV)
    public readonly nint QQuant, KQuant, VQuant, OQuant, GateQuant, UpQuant, DownQuant;
    public readonly QuantizationType QQuantType, KQuantType, VQuantType, OQuantType;
    public readonly QuantizationType GateQuantType, UpQuantType, DownQuantType;
    public readonly int QOutputDim, QInputDim, KOutputDim, KInputDim;
    public readonly int VOutputDim, VInputDim, OOutputDim, OInputDim;
    public readonly int GateOutputDim, GateInputDim, UpOutputDim, UpInputDim;
    public readonly int DownOutputDim, DownInputDim;

    // Norm weights on device (FP16)
    public readonly nint AttnNormWeight, FfnNormWeight;
    public readonly nint QNormWeight, KNormWeight; // 0 when absent

    // Bias on device (FP16, 0 when absent)
    public readonly nint QBias, KBias, VBias, OBias;
    public readonly nint GateBias, UpBias, DownBias;

    // ── Fused Q/K/V projection weight (decode-only, single-call quantized GEMV) ──
    // Packed along the N (output) dim: rows 0..QOutputDim-1 = Q, then K, then V.
    // Each row is independently quantized along K, so byte-concatenating the three
    // quantized tensors yields a valid single weight in the SAME layout — every
    // existing per-row GEMV kernel works unchanged with N = QOutputDim+2*KvOutputDim.
    // 0 when fusion is not possible (mixed quant types, or quant kernel missing).
    public readonly nint QkvPacked;
    public readonly QuantizationType QkvPackedQuantType;
    public readonly int QkvPackedOutputDim; // QOutputDim + KOutputDim + VOutputDim

    // ── Fused Gate/Up projection weight (decode-only) ──
    // Packed along N: rows 0..GateOutputDim-1 = Gate, then Up.
    public readonly nint GateUpPacked;
    public readonly QuantizationType GateUpPackedQuantType;
    public readonly int GateUpPackedOutputDim; // GateOutputDim + UpOutputDim

    public CudaLayerWeights(
        nint q, int qOut, int qIn, nint k, int kOut, int kIn,
        nint v, int vOut, int vIn, nint o, int oOut, int oIn,
        nint gate, int gateOut, int gateIn, nint up, int upOut, int upIn,
        nint down, int downOut, int downIn,
        nint attnNorm, nint ffnNorm,
        nint qNorm, nint kNorm,
        nint qBias, nint kBias, nint vBias, nint oBias,
        nint gateBias, nint upBias, nint downBias,
        nint qQuant, QuantizationType qQt, nint kQuant, QuantizationType kQt,
        nint vQuant, QuantizationType vQt, nint oQuant, QuantizationType oQt,
        nint gateQuant, QuantizationType gateQt, nint upQuant, QuantizationType upQt,
        nint downQuant, QuantizationType downQt,
        nint qkvPacked, QuantizationType qkvPackedQt, int qkvPackedOut,
        nint gateUpPacked, QuantizationType gateUpPackedQt, int gateUpPackedOut)
    {
        Q = q; QOutputDim = qOut; QInputDim = qIn;
        K = k; KOutputDim = kOut; KInputDim = kIn;
        V = v; VOutputDim = vOut; VInputDim = vIn;
        O = o; OOutputDim = oOut; OInputDim = oIn;
        Gate = gate; GateOutputDim = gateOut; GateInputDim = gateIn;
        Up = up; UpOutputDim = upOut; UpInputDim = upIn;
        Down = down; DownOutputDim = downOut; DownInputDim = downIn;
        AttnNormWeight = attnNorm; FfnNormWeight = ffnNorm;
        QNormWeight = qNorm; KNormWeight = kNorm;
        QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
        GateBias = gateBias; UpBias = upBias; DownBias = downBias;
        QQuant = qQuant; QQuantType = qQt; KQuant = kQuant; KQuantType = kQt;
        VQuant = vQuant; VQuantType = vQt; OQuant = oQuant; OQuantType = oQt;
        GateQuant = gateQuant; GateQuantType = gateQt;
        UpQuant = upQuant; UpQuantType = upQt;
        DownQuant = downQuant; DownQuantType = downQt;
        QkvPacked = qkvPacked; QkvPackedQuantType = qkvPackedQt; QkvPackedOutputDim = qkvPackedOut;
        GateUpPacked = gateUpPacked; GateUpPackedQuantType = gateUpPackedQt; GateUpPackedOutputDim = gateUpPackedOut;
    }
}

/// <summary>
/// Manages all model weights on GPU. Uploads from GGUF mmap, dequantizes to FP16 on device.
/// </summary>
internal sealed class CudaWeights : IDisposable
{
    public CudaLayerWeights[] Layers { get; }

    /// <summary>
    /// Per-layer MLA weights for DeepSeek-V2/V3. Non-null iff
    /// <c>config.MlaConfig is not null</c>; entries are populated for layers
    /// whose CPU side carries an <c>Mla</c> bundle (today: every layer in pure
    /// MLA models). When non-null, the GQA Q/K/V/O slots in the matching
    /// <see cref="Layers"/> entry are zeroed and the forward dispatcher routes
    /// through <see cref="CudaMlaAttention.ForwardF16"/>.
    /// </summary>
    public CudaMlaLayerWeights[]? MlaLayers { get; }

    /// <summary>
    /// Per-layer MoE weights for Mixtral / Qwen-MoE / DeepSeek MoE. Non-null
    /// iff <c>config.Moe is not null</c>; entries are non-null for routed-MoE
    /// layers and null for dense layers (Qwen3-MoE alternates per
    /// <see cref="MoeConfig.IsMoeLayer"/>). When the entry is non-null the
    /// dense FFN slots in the matching <see cref="Layers"/> entry are zeroed
    /// and the forward dispatcher routes through
    /// <see cref="CudaMoeFfn.Forward"/>.
    /// </summary>
    public CudaMoeLayerWeights?[]? MoeLayers { get; }

    public nint TokenEmbedDevice { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public nint OutputNormWeight { get; }
    public nint OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }
    public nint OutputWeightQuant { get; }
    public QuantizationType OutputQuantType { get; }

    private readonly List<nint> _allAllocations = new();

    private CudaWeights(CudaLayerWeights[] layers, nint tokenEmbed, QuantizationType tokenEmbedQt,
                          nint outputNorm, nint outputWeight, int outputOutDim, int outputInDim,
                          nint outputWeightQuant, QuantizationType outputQt,
                          List<nint> allocs,
                          CudaMlaLayerWeights[]? mlaLayers,
                          CudaMoeLayerWeights?[]? moeLayers)
    {
        Layers = layers;
        TokenEmbedDevice = tokenEmbed;
        TokenEmbedQuantType = tokenEmbedQt;
        OutputNormWeight = outputNorm;
        OutputWeight = outputWeight;
        OutputOutputDim = outputOutDim;
        OutputInputDim = outputInDim;
        OutputWeightQuant = outputWeightQuant;
        OutputQuantType = outputQt;
        _allAllocations = allocs;
        MlaLayers = mlaLayers;
        MoeLayers = moeLayers;
    }

    /// <summary>
    /// Uploads weights from CPU (GGUF mmap) to GPU. Quantized weights are
    /// dequantized to FP16 on-device to avoid transferring the larger FP16 data over PCIe.
    /// </summary>
    /// <param name="cpuWeights">CPU-side weights (mmap'd from GGUF).</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="kernels">Loaded PTX kernels for dequantization.</param>
    /// <param name="stream">CUDA stream for async uploads.</param>
    /// <param name="numGpuLayers">Number of layers to upload. -1 = all layers.
    /// When less than total layers (hybrid mode), output norm and LM head are skipped
    /// since the CPU handles final projection.</param>
    public static CudaWeights LoadFromGguf(TransformerWeights cpuWeights, ModelConfig config,
                                              CudaKernels kernels, nint stream,
                                              int numGpuLayers = -1)
    {
        int layerCount = numGpuLayers < 0
            ? config.NumLayers
            : Math.Min(numGpuLayers, config.NumLayers);
        bool isHybrid = layerCount < config.NumLayers;

        var allocs = new List<nint>();

        // Token embeddings — upload in original format if a per-row embedding lookup
        // kernel exists for it (saves the FP16 expansion of a vocab×hidden table).
        // Otherwise dequant the entire table to FP16 at load time (one-time cost,
        // costs vocab×hidden×2 bytes of VRAM — 1.16 GiB on Qwen3-8B Q4_K_M).
        // K-quant variants need hidden % 256 == 0; HasEmbeddingLookup gates this.
        nint tokenEmbed;
        var tokenEmbedQt = cpuWeights.TokenEmbedQuantType;
        // Env-var escape hatch (matches DOTLLM_DISABLE_MMQ_* convention) — forces
        // the legacy bulk-dequant path even when a per-row kernel exists. Used
        // for A/B perf comparison and as a fallback if a per-row kernel ever
        // misbehaves on a new model.
        bool disablePerRowEmbed = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_EMBED_ROWLOOKUP") == "1";
        if (!disablePerRowEmbed && kernels.HasEmbeddingLookup(tokenEmbedQt, config.HiddenSize))
        {
            long embedBytes = Dequantize.RowByteSize(config.HiddenSize, tokenEmbedQt) * config.VocabSize;
            tokenEmbed = AllocAndUpload(cpuWeights.TokenEmbedWeight, embedBytes, allocs);
        }
        else
        {
            // No per-row kernel (e.g. Q4_0, or K-quant with hidden not a multiple
            // of 256) — dequant the entire table to FP16 once at load.
            tokenEmbed = UploadAndDequant(cpuWeights.TokenEmbedWeight, tokenEmbedQt,
                config.VocabSize, config.HiddenSize, allocs, kernels, stream);
            tokenEmbedQt = QuantizationType.F16;
        }

        // Output norm + LM head: skip in hybrid mode (CPU handles final norm + LM head)
        nint outputNorm = 0;
        nint outputWeight = 0;
        nint outputWeightQuant = 0;

        if (!isHybrid)
        {
            // Output norm (float[] → FP16)
            outputNorm = UploadNormWeight(cpuWeights.OutputNormWeight, allocs, kernels, stream);

            // LM head — too large for the per-projection dequant scratch (vocabSize × hiddenSize).
            // Create a persistent FP16 copy unless the runtime has a loaded
            // quantized GEMV implementation for this type.
            bool lmHeadHasGemv = kernels.HasLoadedQuantizedGemv(cpuWeights.OutputQuantType);
            outputWeight = (!IsQuantized(cpuWeights.OutputQuantType) || !lmHeadHasGemv)
                ? UploadAndDequant(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                    cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs, kernels, stream)
                : 0;
        }

        // Per-layer weights — skip persistent FP16 copies only for types with loaded
        // quantized GEMV kernels. These can dequant on-the-fly into
        // a scratch buffer for prefill GEMM, and use the GEMV kernel directly for decode.
        // All other types keep a persistent FP16 copy.
        // In hybrid mode, only upload the first layerCount layers.
        var layers = new CudaLayerWeights[layerCount];

        // MLA / MoE side-tables. Non-null iff the model declares the matching config.
        // Per-layer entries are populated for layers whose CPU side carries an Mla / Moe
        // bundle (Qwen3-MoE-style alternating layouts leave non-MoE layers null).
        bool hasMla = config.MlaConfig is not null;
        bool hasMoe = config.Moe is not null;
        var mlaLayers = hasMla ? new CudaMlaLayerWeights[layerCount] : null;
        var moeLayers = hasMoe ? new CudaMoeLayerWeights?[layerCount] : null;

        for (int i = 0; i < layerCount; i++)
        {
            ref readonly var lw = ref cpuWeights.Layers[i];

            // MLA layers do NOT carry GQA Q/K/V tensors — those slots are zero on
            // the CPU side. Skip the GQA upload path entirely; CudaMlaWeightsLoader
            // owns the q_a/q_b/kv_a/kv_b/o uploads. Norms still come from the
            // shared CPU layer (AttnNorm/FfnNorm); MLA's internal AttnNorm pointer
            // duplicates these tiny buffers for kernel-call convenience.
            bool isMlaLayer = lw.Mla is not null;
            // MoE layers do NOT carry dense gate/up/down tensors — those slots are
            // zero on the CPU side. The MoE loader uploads per-expert projections
            // into separate device allocations.
            bool isMoeLayer = lw.Moe is not null;

            nint q = 0, k = 0, v = 0, o = 0;
            nint qQuant = 0, kQuant = 0, vQuant = 0, oQuant = 0;
            nint qkvPacked = 0; QuantizationType qkvPackedQt = QuantizationType.F16; int qkvPackedOut = 0;
            nint qBias = 0, kBias = 0, vBias = 0, oBias = 0;
            nint qNorm = 0, kNorm = 0;
            if (!isMlaLayer)
            {
                q = SkipFp16(lw.QQuantType, kernels) ? 0 : UploadAndDequant(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs, kernels, stream);
                k = SkipFp16(lw.KQuantType, kernels) ? 0 : UploadAndDequant(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs, kernels, stream);
                v = SkipFp16(lw.VQuantType, kernels) ? 0 : UploadAndDequant(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs, kernels, stream);
                o = SkipFp16(lw.OQuantType, kernels) ? 0 : UploadAndDequant(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs, kernels, stream);

                // ── Upload raw quantized Q/K/V weights ──
                // When fusion is possible (shared quant type + input dim, GEMV kernel exists),
                // allocate ONE packed device buffer and upload Q/K/V directly into it at the
                // appropriate row offsets. The per-tensor pointers (qQuant/kQuant/vQuant) are
                // then slices into the packed buffer — bit-identical layout, zero data copy,
                // and the per-tensor row-iterating consumers (Project, ProjectGpu, MMQ GEMV)
                // work unchanged because they only read `outputDim` rows starting at the
                // given pointer. Saves ~`(qOut+kOut+vOut)*rowBytes` per layer of VRAM that
                // was previously double-stored. Only the packed allocation is in `allocs`.
                if (!CudaKernels.DisablePackedQkv)
                {
                    (qkvPacked, qkvPackedQt, qkvPackedOut,
                     qQuant, kQuant, vQuant) = TryUploadPackedThree(
                        lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim,
                        lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim,
                        lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim,
                        allocs, kernels);
                }
                if (qkvPacked == 0)
                {
                    // Fusion not possible — fall back to per-tensor uploads (separate allocations).
                    qQuant = UploadQuantized(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs);
                    kQuant = UploadQuantized(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs);
                    vQuant = UploadQuantized(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs);
                }

                oQuant = UploadQuantized(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs);

                qBias = UploadBias(lw.QBias, allocs, kernels, stream);
                kBias = UploadBias(lw.KBias, allocs, kernels, stream);
                vBias = UploadBias(lw.VBias, allocs, kernels, stream);
                oBias = UploadBias(lw.OBias, allocs, kernels, stream);
                qNorm = lw.QNormWeight is not null ? UploadNormWeight(lw.QNormWeight, allocs, kernels, stream) : 0;
                kNorm = lw.KNormWeight is not null ? UploadNormWeight(lw.KNormWeight, allocs, kernels, stream) : 0;
            }

            nint gate = 0, up = 0, down = 0;
            nint gateQuant = 0, upQuant = 0, downQuant = 0;
            nint gateUpPacked = 0; QuantizationType gateUpPackedQt = QuantizationType.F16; int gateUpPackedOut = 0;
            nint gateBias = 0, upBias = 0, downBias = 0;
            if (!isMoeLayer)
            {
                gate = SkipFp16(lw.GateQuantType, kernels) ? 0 : UploadAndDequant(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs, kernels, stream);
                up = SkipFp16(lw.UpQuantType, kernels) ? 0 : UploadAndDequant(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs, kernels, stream);
                down = SkipFp16(lw.DownQuantType, kernels) ? 0 : UploadAndDequant(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs, kernels, stream);

                // ── Upload raw quantized Gate/Up weights (same packing strategy as Q/K/V) ──
                if (!CudaKernels.DisablePackedGateUp)
                {
                    (gateUpPacked, gateUpPackedQt, gateUpPackedOut,
                     gateQuant, upQuant) = TryUploadPackedTwo(
                        lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim,
                        lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim,
                        allocs, kernels);
                }
                if (gateUpPacked == 0)
                {
                    gateQuant = UploadQuantized(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs);
                    upQuant = UploadQuantized(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs);
                }

                downQuant = UploadQuantized(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs);

                gateBias = UploadBias(lw.GateBias, allocs, kernels, stream);
                upBias = UploadBias(lw.UpBias, allocs, kernels, stream);
                downBias = UploadBias(lw.DownBias, allocs, kernels, stream);
            }

            nint attnNorm = UploadNormWeight(lw.AttnNormWeight, allocs, kernels, stream);
            nint ffnNorm = UploadNormWeight(lw.FfnNormWeight, allocs, kernels, stream);

            layers[i] = new CudaLayerWeights(
                q, lw.QOutputDim, lw.QInputDim, k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim, o, lw.OOutputDim, lw.OInputDim,
                gate, lw.GateOutputDim, lw.GateInputDim, up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                attnNorm, ffnNorm, qNorm, kNorm,
                qBias, kBias, vBias, oBias, gateBias, upBias, downBias,
                qQuant, lw.QQuantType, kQuant, lw.KQuantType,
                vQuant, lw.VQuantType, oQuant, lw.OQuantType,
                gateQuant, lw.GateQuantType, upQuant, lw.UpQuantType,
                downQuant, lw.DownQuantType,
                qkvPacked, qkvPackedQt, qkvPackedOut,
                gateUpPacked, gateUpPackedQt, gateUpPackedOut);

            if (isMlaLayer)
            {
                // GGUF source: raw quant view present → upload Q4_K bytes directly
                // (~140 MB for V2-Lite, vs ~1.4 GB F16). Safetensors source: no
                // raw view → fall back to F32→F16 cast as before.
                mlaLayers![i] = lw.Mla!.HasRawQuantView
                    ? CudaMlaWeightsLoader.LoadLayerQuant(lw, config.HiddenSize, lw.OQuantType, allocs)
                    : CudaMlaWeightsLoader.LoadLayerF16(lw, config.HiddenSize, allocs);
            }
            if (isMoeLayer)
            {
                // GGUF source with raw quant view → upload Q4_K bytes per expert
                // (~26 GB at full V2-Lite Q4_K_M scale; the next perf milestone
                // is grouped-GEMM compaction). Safetensors source → F32 path.
                moeLayers![i] = lw.Moe!.HasRawQuantView
                    ? CudaMoeWeightsLoader.LoadLayerQuant(lw, allocs)
                    : CudaMoeWeightsLoader.LoadLayer(lw, allocs);
            }
        }

        // Sync to ensure all uploads are complete
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        // LM head quantized copy for decode (skip in hybrid mode — CPU handles LM head)
        if (!isHybrid)
        {
            outputWeightQuant = UploadQuantized(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs);
        }

        return new CudaWeights(layers, tokenEmbed, tokenEmbedQt,
            outputNorm, outputWeight, cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim,
            outputWeightQuant, cpuWeights.OutputQuantType, allocs,
            mlaLayers, moeLayers);
    }

    /// <summary>Upload raw quantized weight bytes to GPU (no dequant). For decode quantized GEMV.</summary>
    private static nint UploadQuantized(nint hostPtr, QuantizationType qt,
                                          int outputDim, int inputDim, List<nint> allocs)
    {
        if (qt is QuantizationType.F16 or QuantizationType.F32)
            return 0; // Non-quantized weights don't need a separate quantized copy

        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        return AllocAndUpload(hostPtr, quantBytes, allocs);
    }

    /// <summary>
    /// Allocates a single packed device buffer and uploads three CPU quantized weight
    /// tensors directly into it via H2D copies at row offsets along the N (output)
    /// dimension. Returns the packed handle plus three sub-pointers (Q/K/V) that
    /// alias into the packed buffer — bit-identical to the layout of three separate
    /// per-tensor uploads, but stored ONCE. Returns all-zero on failure (caller falls
    /// back to per-tensor uploads).
    /// <para>
    /// Conditions for packing (must all hold):
    ///  - all three weights are quantized (skip F16/F32 — those don't get a quant copy)
    ///  - all three share the same quantization type
    ///  - that type has a loaded quantized-GEMV kernel (the only consumer of these pointers)
    ///  - the input dim (K) matches across all three
    /// </para>
    /// <para>
    /// VRAM saving vs. previous "separate + DtoD-pack" approach: eliminates the
    /// duplicate copy that held the three tensors a second time (~Q+K+V row-bytes
    /// per layer). Q4_K_M Qwen3-8B (k=4096, 36 layers): saves ~2.5 GB.
    /// </para>
    /// <para>
    /// Why this is safe: every consumer of <c>QQuant</c>/<c>KQuant</c>/<c>VQuant</c>
    /// (Project, ProjectGpu, MMQ GEMV) iterates exactly <c>outputDim</c> rows starting
    /// at the given pointer. Slicing into the packed buffer is invisible to the kernel
    /// — it sees the same row layout as a standalone per-tensor allocation.
    /// </para>
    /// </summary>
    /// <returns>Tuple of (packed handle, packed quant type, packed output dim,
    /// q-slice pointer, k-slice pointer, v-slice pointer). All zero on failure.</returns>
    private static (nint Packed, QuantizationType PackedQt, int PackedOut,
                    nint QSlice, nint KSlice, nint VSlice) TryUploadPackedThree(
        nint qHost, QuantizationType qQt, int qOut, int qIn,
        nint kHost, QuantizationType kQt, int kOut, int kIn,
        nint vHost, QuantizationType vQt, int vOut, int vIn,
        List<nint> allocs, CudaKernels kernels)
    {
        if (qQt is QuantizationType.F16 or QuantizationType.F32) return default;
        if (qQt != kQt || qQt != vQt) return default;
        if (!kernels.HasLoadedQuantizedGemv(qQt)) return default;
        if (qIn != kIn || qIn != vIn) return default;

        long rowBytes = Dequantize.RowByteSize(qIn, qQt);
        long qBytes = rowBytes * qOut;
        long kBytes = rowBytes * kOut;
        long vBytes = rowBytes * vOut;
        long totalBytes = qBytes + kBytes + vBytes;

        AllocOrThrowWithContext(totalBytes, "QkvPacked", out nint packed);
        allocs.Add(packed);
        // Upload each tensor's bytes directly into its slice — no intermediate alloc,
        // no D2D copy. Slices are NOT in `allocs` (they alias `packed`).
        nint qSlice = packed;
        nint kSlice = packed + (nint)qBytes;
        nint vSlice = packed + (nint)(qBytes + kBytes);
        MemcpyHtoDOrThrowWithContext(qSlice, qHost, qBytes, "QkvPacked.Q");
        MemcpyHtoDOrThrowWithContext(kSlice, kHost, kBytes, "QkvPacked.K");
        MemcpyHtoDOrThrowWithContext(vSlice, vHost, vBytes, "QkvPacked.V");
        return (packed, qQt, qOut + kOut + vOut, qSlice, kSlice, vSlice);
    }

    /// <summary>
    /// Two-tensor variant of <see cref="TryUploadPackedThree"/>. Used to fuse
    /// Gate + Up MLP projections into a single decode-time GEMV with one
    /// shared quantized weight buffer.
    /// </summary>
    private static (nint Packed, QuantizationType PackedQt, int PackedOut,
                    nint ASlice, nint BSlice) TryUploadPackedTwo(
        nint aHost, QuantizationType aQt, int aOut, int aIn,
        nint bHost, QuantizationType bQt, int bOut, int bIn,
        List<nint> allocs, CudaKernels kernels)
    {
        if (aQt is QuantizationType.F16 or QuantizationType.F32) return default;
        if (aQt != bQt) return default;
        if (!kernels.HasLoadedQuantizedGemv(aQt)) return default;
        if (aIn != bIn) return default;

        long rowBytes = Dequantize.RowByteSize(aIn, aQt);
        long aBytes = rowBytes * aOut;
        long bBytes = rowBytes * bOut;
        long totalBytes = aBytes + bBytes;

        AllocOrThrowWithContext(totalBytes, "GateUpPacked", out nint packed);
        allocs.Add(packed);
        nint aSlice = packed;
        nint bSlice = packed + (nint)aBytes;
        MemcpyHtoDOrThrowWithContext(aSlice, aHost, aBytes, "GateUpPacked.Gate");
        MemcpyHtoDOrThrowWithContext(bSlice, bHost, bBytes, "GateUpPacked.Up");
        return (packed, aQt, aOut + bOut, aSlice, bSlice);
    }

    /// <summary>Upload quantized weight to GPU, then dequantize to FP16 on device.</summary>
    private static nint UploadAndDequant(nint hostPtr, QuantizationType qt,
                                           int outputDim, int inputDim,
                                           List<nint> allocs, CudaKernels kernels, nint stream)
    {
        int totalElements = outputDim * inputDim;

        if (qt == QuantizationType.F16)
        {
            // Already FP16 — just upload
            long bytes = (long)totalElements * sizeof(ushort);
            return AllocAndUpload(hostPtr, bytes, allocs);
        }

        if (qt == QuantizationType.F32)
        {
            // Upload F32, convert to F16 on device
            long f32Bytes = (long)totalElements * sizeof(float);
            nint devF32 = AllocAndUpload(hostPtr, f32Bytes, allocs);
            long f16Bytes = (long)totalElements * sizeof(ushort);
            CudaDriverApi.cuMemAlloc_v2(out nint devF16, (nuint)f16Bytes).ThrowOnError();
            allocs.Add(devF16);
            kernels.LaunchConvertF32ToF16(devF32, devF16, totalElements, stream);
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            allocs.Remove(devF32);
            CudaDriverApi.cuMemFree_v2(devF32);
            return devF16;
        }

        // Quantized: upload raw bytes, dequant to FP16 on device
        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        nint devQuant = AllocAndUpload(hostPtr, quantBytes, allocs);

        long fp16Bytes = (long)totalElements * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16, (nuint)fp16Bytes).ThrowOnError();
        allocs.Add(devFp16);

        kernels.LaunchDequantToF16(devQuant, qt, devFp16, totalElements, stream);
        return devFp16;
    }

    /// <summary>Upload float[] norm weight → FP16 on device (F32→F16 conversion via GPU kernel).</summary>
    private static unsafe nint UploadNormWeight(float[] weight, List<nint> allocs,
                                                  CudaKernels kernels, nint stream)
    {
        int n = weight.Length;

        // Upload F32 to temp buffer, then convert to FP16 on device
        long f32Bytes = (long)n * sizeof(float);
        long f16Bytes = (long)n * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint devF32, (nuint)f32Bytes).ThrowOnError();
        allocs.Add(devF32);
        fixed (float* ptr = weight)
            CudaDriverApi.cuMemcpyHtoD_v2(devF32, (nint)ptr, (nuint)f32Bytes).ThrowOnError();

        CudaDriverApi.cuMemAlloc_v2(out nint devF16, (nuint)f16Bytes).ThrowOnError();
        allocs.Add(devF16);
        kernels.LaunchConvertF32ToF16(devF32, devF16, n, stream);

        return devF16;
    }

    /// <summary>Upload optional float[] bias → FP16 on device. Returns 0 if bias is null.</summary>
    private static nint UploadBias(float[]? bias, List<nint> allocs,
                                     CudaKernels kernels, nint stream)
    {
        if (bias is null) return 0;
        return UploadNormWeight(bias, allocs, kernels, stream);
    }

    private static bool IsQuantized(QuantizationType qt) =>
        qt is not QuantizationType.F16 and not QuantizationType.F32;

    /// <summary>
    /// Whether to skip the persistent FP16 copy for this quant type.
    /// Only skip when we have BOTH a loaded custom quantized GEMV kernel (for decode)
    /// AND a dequant-to-F16 kernel (for on-the-fly prefill GEMM via scratch buffer).
    /// Types without a loaded custom GEMV keep persistent FP16
    /// because the scratch buffer approach requires cuBLAS fallback.
    /// </summary>
    private static bool SkipFp16(QuantizationType qt, CudaKernels kernels) =>
        kernels.HasLoadedQuantizedGemv(qt);

    /// <summary>Allocate device memory and copy host data.</summary>
    private static nint AllocAndUpload(nint hostPtr, long bytes, List<nint> allocs)
    {
        AllocOrThrowWithContext(bytes, "weight upload", out nint devPtr);
        allocs.Add(devPtr);
        MemcpyHtoDOrThrowWithContext(devPtr, hostPtr, bytes, "weight upload");
        return devPtr;
    }

    /// <summary>
    /// Allocates device memory; on failure (typically OOM), augments the exception
    /// with VRAM context (free / total) and the requested size. Used by the packed
    /// weight allocators where running out of VRAM is the primary suspected failure.
    /// </summary>
    private static void AllocOrThrowWithContext(long bytes, string label, out nint devPtr)
    {
        int rc = CudaDriverApi.cuMemAlloc_v2(out devPtr, (nuint)bytes);
        if (rc == 0) return;
        // Best-effort mem probe for diagnostics; ignore probe failure.
        nuint free = 0, total = 0;
        _ = CudaDriverApi.cuMemGetInfo_v2(out free, out total);
        throw new InvalidOperationException(
            $"CUDA OOM allocating {label} ({bytes / (1024.0 * 1024.0):F1} MiB requested). " +
            $"Free VRAM: {free / (1024.0 * 1024.0):F1} MiB / {total / (1024.0 * 1024.0):F1} MiB total. " +
            $"Underlying cuMemAlloc rc={rc}.");
    }

    /// <summary>
    /// Synchronous H2D copy that augments OOM-class failures with VRAM context.
    /// CUDA can defer page commits until first write, so an alloc may succeed and
    /// the subsequent memcpy reports OOM.
    /// </summary>
    private static void MemcpyHtoDOrThrowWithContext(nint devPtr, nint hostPtr, long bytes, string label)
    {
        int rc = CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes);
        if (rc == 0) return;
        nuint free = 0, total = 0;
        _ = CudaDriverApi.cuMemGetInfo_v2(out free, out total);
        throw new InvalidOperationException(
            $"CUDA H2D failure for {label} ({bytes / (1024.0 * 1024.0):F1} MiB). " +
            $"Free VRAM: {free / (1024.0 * 1024.0):F1} MiB / {total / (1024.0 * 1024.0):F1} MiB total. " +
            $"Underlying cuMemcpyHtoD rc={rc} (typically rc=2 → OOM via deferred page commit).");
    }

    public void Dispose()
    {
        foreach (nint ptr in _allAllocations)
        {
            if (ptr != 0)
                CudaDriverApi.cuMemFree_v2(ptr);
        }
        _allAllocations.Clear();
    }
}
