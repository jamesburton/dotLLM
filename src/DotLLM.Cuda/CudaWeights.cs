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
    // BitNet Sub-LN weights on device (FP16). 0 when absent (non-BitNet models).
    public readonly nint AttnSubNormWeight, FfnSubNormWeight;

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
        nint attnSubNorm, nint ffnSubNorm,
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
        AttnSubNormWeight = attnSubNorm; FfnSubNormWeight = ffnSubNorm;
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

    /// <summary>
    /// Per-layer Gemma-4 (DiffusionGemma AR) extras. Non-null iff
    /// <c>config.Gemma4DualFfn</c>; one entry per layer (all gemma4 layers are
    /// MoE layers). When non-null the forward routes through the gemma4 F32 path
    /// (V-from-K, weight-less V-norm, partial rope, dual dense+MoE GeGLU FFN, the
    /// five norms, custom router, per-expert down scale, layer_output_scale).
    /// The companion experts live in <see cref="MoeLayers"/>; the dense
    /// gate/up/down + norms live in the matching <see cref="Layers"/> entry.
    /// </summary>
    public CudaGemma4LayerWeights?[]? Gemma4Layers { get; }

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
                          CudaMoeLayerWeights?[]? moeLayers,
                          CudaGemma4LayerWeights?[]? gemma4Layers)
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
        Gemma4Layers = gemma4Layers;
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
    /// <param name="firstLayer">
    /// First layer index (0-based) in <paramref name="cpuWeights"/> to upload.
    /// Layers <c>firstLayer..(firstLayer+layerCount-1)</c> are uploaded.
    /// The resulting <see cref="Layers"/> array is always 0-based regardless of
    /// <paramref name="firstLayer"/>. Used by the Vulkan+CUDA split to avoid uploading
    /// the Vulkan-resident layers to CUDA VRAM. Default 0 = upload from the beginning.
    /// </param>
    /// <param name="skipTokenEmbed">
    /// When <c>true</c>, the token-embedding table is not uploaded and <see cref="TokenEmbedDevice"/>
    /// stays 0. Used by a non-first pipeline stage (<c>CudaPipelineTransformerModel</c>), which is only
    /// ever seeded from a previous stage's hidden state and never gathers embeddings — saves the
    /// vocab × hidden table (FP16 when bulk-dequanted, raw-quant when a per-row lookup kernel exists)
    /// plus the transient upload. The owning stage must never launch an embedding lookup.
    /// </param>
    /// <param name="onHostTensorUploaded">
    /// Optional direct-to-device streaming hook. When non-null, it is invoked with each
    /// per-layer linear-projection HOST pointer (Q/K/V/O and dense Gate/Up/Down) right
    /// after that tensor's synchronous host→device copy completes, so the caller can free
    /// the host scratch buffer immediately instead of holding the whole host weight set
    /// until upload finishes — roughly halving the transient CPU-RAM peak. The callback is
    /// expected to free only its own owned host allocations and ignore mmap views (see
    /// <see cref="TransformerWeights.TryReleaseOwnedHostAllocation"/>). It is NOT invoked for
    /// the token-embedding table or LM head (which may alias each other via tied embeddings)
    /// nor for MoE / MLA / Gemma-4 layers (uploaded by dedicated loaders). Null (the default)
    /// preserves the legacy batch behavior: all host buffers stay resident until the caller
    /// disposes <paramref name="cpuWeights"/>. The caller MUST pass null whenever it retains
    /// <paramref name="cpuWeights"/> for a CPU-side forward.
    /// </param>
    /// <param name="skipOutputHead">
    /// When <c>true</c>, the output norm + LM head (and the head's quantized decode copy) are not
    /// uploaded even though this window reaches the last layer, exactly as if the window stopped
    /// short of it. For a caller that applies the head elsewhere — the layer-cycling perplexity
    /// windows, which always run the head on the host so that logits are produced for every row
    /// (issue #395) — the head is otherwise pure dead VRAM on the one window that happens to contain
    /// the final layer, enough to OOM the last window of a cycle whose earlier windows all fit.
    /// Measured per-window on Llama-3.2-1B-Q8_0 via <c>cuMemGetInfo_v2</c> around the same window
    /// built both ways: <b>268 MiB</b>, matching the <c>vocab x hidden</c> arithmetic for the raw
    /// quantized copy (no FP16 copy is made when a GEMV kernel is loaded for the head's quant type).
    /// It scales with <c>vocab x hidden</c>, so it is materially larger on a 27-30B model. Default
    /// <c>false</c> preserves the existing "final window owns the head" behavior.
    /// </param>
    public static CudaWeights LoadFromGguf(TransformerWeights cpuWeights, ModelConfig config,
                                              CudaKernels kernels, nint stream,
                                              int numGpuLayers = -1, int firstLayer = 0,
                                              bool skipTokenEmbed = false,
                                              Action<nint>? onHostTensorUploaded = null,
                                              bool skipOutputHead = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstLayer);
        int layerCount = numGpuLayers < 0
            ? config.NumLayers - firstLayer
            : Math.Min(numGpuLayers, config.NumLayers - firstLayer);
        // isHybrid: this upload does NOT own the output norm + LM head, because either it covers
        // only a contiguous slice of the full model (the caller owns the tail), or the caller has
        // said outright that it applies the head itself (skipOutputHead).
        bool isHybrid = (firstLayer + layerCount) < config.NumLayers || skipOutputHead;

        var allocs = new List<nint>();

        // Token embeddings — upload in original format if a per-row embedding lookup
        // kernel exists for it (saves the FP16 expansion of a vocab×hidden table).
        // Otherwise dequant the entire table to FP16 at load time (one-time cost,
        // costs vocab×hidden×2 bytes of VRAM — 1.16 GiB on Qwen3-8B Q4_K_M).
        // K-quant variants need hidden % 256 == 0; HasEmbeddingLookup gates this.
        // A non-first pipeline stage never gathers (seeded from a previous stage's
        // hidden state), so it skips the table entirely — TokenEmbedDevice stays 0.
        nint tokenEmbed = 0;
        var tokenEmbedQt = cpuWeights.TokenEmbedQuantType;
        // Env-var escape hatch (matches DOTLLM_DISABLE_MMQ_* convention) — forces
        // the legacy bulk-dequant path even when a per-row kernel exists. Used
        // for A/B perf comparison and as a fallback if a per-row kernel ever
        // misbehaves on a new model.
        bool disablePerRowEmbed = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_EMBED_ROWLOOKUP") == "1";
        if (skipTokenEmbed)
        {
            // Nothing to upload.
        }
        else if (!disablePerRowEmbed && kernels.HasEmbeddingLookup(tokenEmbedQt, config.HiddenSize))
        {
            long embedBytes = Dequantize.RowByteSize(config.HiddenSize, tokenEmbedQt) * config.VocabSize;
            tokenEmbed = AllocAndUpload(cpuWeights.TokenEmbedWeight, embedBytes, allocs);
        }
        else
        {
            // No per-row kernel (e.g. Q4_0, or K-quant with hidden not a multiple
            // of 256) — dequant the entire table to FP16 once at load.
            tokenEmbed = UploadAndDequant(cpuWeights.TokenEmbedWeight, tokenEmbedQt,
                config.VocabSize, config.HiddenSize, allocs, kernels, stream, "token embedding table");
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
                    cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs, kernels, stream, "LM head")
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
        // Gemma-4 is a dual-FFN MoE: every layer carries BOTH a dense gate/up/down
        // (the "shared expert") AND a 128-expert MoE. The dense slots must still be
        // uploaded (unlike pure-MoE Qwen/DeepSeek layers, which zero them), so the
        // dense-FFN upload is gated on `isMoeLayer && !isGemma4` below.
        bool hasGemma4 = config.Gemma4DualFfn;
        var mlaLayers = hasMla ? new CudaMlaLayerWeights[layerCount] : null;
        var moeLayers = hasMoe ? new CudaMoeLayerWeights?[layerCount] : null;
        var gemma4Layers = hasGemma4 ? new CudaGemma4LayerWeights?[layerCount] : null;

        for (int i = 0; i < layerCount; i++)
        {
            // cpuWeights.Layers is indexed globally; layers array is 0-based (local to this CUDA slice).
            int globalLayer = firstLayer + i;
            ref readonly var lw = ref cpuWeights.Layers[globalLayer];

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
            // Gemma-4 layers ARE MoE layers but ALSO carry a dense gate/up/down
            // ("shared expert") that must be uploaded into the standard dense slots.
            bool isGemma4Layer = lw.Gemma4 is not null;
            // V-from-K (gemma4 global layers): no attn_v.weight — the V slot is 0
            // on the CPU side and the forward copies the raw K projection into V.
            bool vFromK = isGemma4Layer && lw.Gemma4!.VFromK;

            nint q = 0, k = 0, v = 0, o = 0;
            nint qQuant = 0, kQuant = 0, vQuant = 0, oQuant = 0;
            nint qkvPacked = 0; QuantizationType qkvPackedQt = QuantizationType.F16; int qkvPackedOut = 0;
            nint qBias = 0, kBias = 0, vBias = 0, oBias = 0;
            nint qNorm = 0, kNorm = 0;
            if (!isMlaLayer)
            {
                q = SkipFp16(lw.QQuantType, kernels) ? 0 : UploadAndDequant(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs, kernels, stream, $"layer {globalLayer} Q projection");
                k = SkipFp16(lw.KQuantType, kernels) ? 0 : UploadAndDequant(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs, kernels, stream, $"layer {globalLayer} K projection");
                // V-from-K (gemma4 global layers): no attn_v.weight — leave V slots 0;
                // the gemma4 forward copies the raw K projection into V.
                v = (vFromK || SkipFp16(lw.VQuantType, kernels)) ? 0 : UploadAndDequant(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs, kernels, stream, $"layer {globalLayer} V projection");
                o = SkipFp16(lw.OQuantType, kernels) ? 0 : UploadAndDequant(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs, kernels, stream, $"layer {globalLayer} O projection");

                // ── Upload raw quantized Q/K/V weights ──
                // When fusion is possible (shared quant type + input dim, GEMV kernel exists),
                // allocate ONE packed device buffer and upload Q/K/V directly into it at the
                // appropriate row offsets. The per-tensor pointers (qQuant/kQuant/vQuant) are
                // then slices into the packed buffer — bit-identical layout, zero data copy,
                // and the per-tensor row-iterating consumers (Project, ProjectGpu, MMQ GEMV)
                // work unchanged because they only read `outputDim` rows starting at the
                // given pointer. Saves ~`(qOut+kOut+vOut)*rowBytes` per layer of VRAM that
                // was previously double-stored. Only the packed allocation is in `allocs`.
                // Skip packing on V-from-K layers — there is no V tensor to pack.
                if (!CudaKernels.DisablePackedQkv && !vFromK)
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
                    vQuant = vFromK ? 0 : UploadQuantized(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs);
                }

                oQuant = UploadQuantized(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs);

                qBias = UploadBias(lw.QBias, allocs, kernels, stream);
                kBias = UploadBias(lw.KBias, allocs, kernels, stream);
                vBias = UploadBias(lw.VBias, allocs, kernels, stream);
                oBias = UploadBias(lw.OBias, allocs, kernels, stream);
                qNorm = lw.QNormWeight is not null ? UploadNormWeight(lw.QNormWeight, allocs, kernels, stream) : 0;
                kNorm = lw.KNormWeight is not null ? UploadNormWeight(lw.KNormWeight, allocs, kernels, stream) : 0;

                // Direct-to-device streaming: every host→device copy of the attention
                // projections above used the SYNCHRONOUS cuMemcpyHtoD_v2 (via AllocAndUpload /
                // UploadQuantized / TryUploadPackedThree), which blocks until the transfer is
                // complete. The on-device dequant kernels queued on `stream` read the uploaded
                // DEVICE buffers only — never these host pointers — so the host scratch is safe
                // to free now, before the final cuStreamSynchronize. Each owned host buffer is
                // read exactly once in this block (F32 upcasts via UploadAndDequant; I2_S via the
                // packed/quantized upload), so freeing here cannot race a later read. The callback
                // frees only owned allocations and ignores mmap views. V is 0 on V-from-K layers,
                // which the callback treats as a no-op.
                if (onHostTensorUploaded is not null)
                {
                    onHostTensorUploaded(lw.QWeight);
                    onHostTensorUploaded(lw.KWeight);
                    onHostTensorUploaded(lw.VWeight);
                    onHostTensorUploaded(lw.OWeight);
                }
            }

            nint gate = 0, up = 0, down = 0;
            nint gateQuant = 0, upQuant = 0, downQuant = 0;
            nint gateUpPacked = 0; QuantizationType gateUpPackedQt = QuantizationType.F16; int gateUpPackedOut = 0;
            nint gateBias = 0, upBias = 0, downBias = 0;
            // Gemma-4 layers ARE MoE layers but carry a dense gate/up/down ("shared
            // expert") that must be uploaded into the standard dense slots. Run the
            // dense upload for non-MoE layers AND for gemma4 layers.
            if (!isMoeLayer || isGemma4Layer)
            {
                gate = SkipFp16(lw.GateQuantType, kernels) ? 0 : UploadAndDequant(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs, kernels, stream, $"layer {globalLayer} Gate projection");
                up = SkipFp16(lw.UpQuantType, kernels) ? 0 : UploadAndDequant(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs, kernels, stream, $"layer {globalLayer} Up projection");
                down = SkipFp16(lw.DownQuantType, kernels) ? 0 : UploadAndDequant(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs, kernels, stream, $"layer {globalLayer} Down projection");

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

                // Stream-free the dense FFN host scratch (same safety argument as the
                // attention block above). Restricted to genuinely-dense layers: a Gemma-4
                // layer is `isMoeLayer` yet also uploads dense slots, but it retains its CPU
                // weights for the host LM head, so `onHostTensorUploaded` is null for it and
                // this never runs. Pure-MoE layers zero these slots and are handled by the
                // MoE loader, so they are excluded by `!isMoeLayer`.
                if (onHostTensorUploaded is not null && !isMoeLayer)
                {
                    onHostTensorUploaded(lw.GateWeight);
                    onHostTensorUploaded(lw.UpWeight);
                    onHostTensorUploaded(lw.DownWeight);
                }
            }

            nint attnNorm = UploadNormWeight(lw.AttnNormWeight, allocs, kernels, stream);
            nint ffnNorm = UploadNormWeight(lw.FfnNormWeight, allocs, kernels, stream);

            // BitNet Sub-LN weights (F32 → FP16). 0 when absent (non-BitNet models).
            nint attnSubNorm = lw.AttnSubNormWeight is not null ? UploadNormWeight(lw.AttnSubNormWeight, allocs, kernels, stream) : 0;
            nint ffnSubNorm = lw.FfnSubNormWeight is not null ? UploadNormWeight(lw.FfnSubNormWeight, allocs, kernels, stream) : 0;

            layers[i] = new CudaLayerWeights(
                q, lw.QOutputDim, lw.QInputDim, k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim, o, lw.OOutputDim, lw.OInputDim,
                gate, lw.GateOutputDim, lw.GateInputDim, up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                attnNorm, ffnNorm, qNorm, kNorm,
                attnSubNorm, ffnSubNorm,
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
            if (isGemma4Layer)
            {
                // Gemma-4 dual-FFN MoE: host-dequant the fused gate_up bank into
                // separate F32 gate/up per-expert banks + the down bank with the
                // per-expert down scale folded in, plus the gemma4 per-layer extras
                // (five F32 norms, router scale with 1/√H folded, layer_output_scale,
                // V-from-K flag). The experts reuse the F32 CudaMoeFfn routed path
                // (GeGLU substituted for SwiGLU by the gemma4 FFN helper).
                var (extras, g4Moe) = CudaGemma4WeightsLoader.LoadLayer(lw, config, allocs);
                gemma4Layers![i] = extras;
                moeLayers![i] = g4Moe;
            }
            else if (isMoeLayer)
            {
                // BitNet-ternary (I2_S) MoE (identity-MoTE, issue #246): per-expert packed-trit
                // banks + per-expert absmean scales + per-expert FFN Sub-LN, dispatched through
                // CudaMoeFfn's BitNetI2S precision branch. Checked FIRST — a BitNet-MoE layer's
                // W1/W2/W3 F32 arrays are empty (Array.Empty<nint>()) and its raw-quant-view
                // fields are unset, so falling through to LoadLayer/LoadLayerQuant below would
                // throw or upload garbage.
                if (lw.Moe!.IsBitNetI2S)
                {
                    moeLayers![i] = CudaMoeWeightsLoader.LoadLayerBitNetI2S(lw, allocs, config.NormEpsilon);
                }
                else
                {
                    // GGUF source with raw quant view → upload Q4_K bytes per expert
                    // (~26 GB at full V2-Lite Q4_K_M scale; the next perf milestone
                    // is grouped-GEMM compaction). Safetensors source → F32 path.
                    moeLayers![i] = lw.Moe!.HasRawQuantView
                        ? CudaMoeWeightsLoader.LoadLayerQuant(lw, allocs)
                        : CudaMoeWeightsLoader.LoadLayer(lw, allocs);
                }
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
            mlaLayers, moeLayers, gemma4Layers);
    }

    /// <summary>Upload raw quantized weight bytes to GPU (no dequant). For decode quantized GEMV.</summary>
    private static nint UploadQuantized(nint hostPtr, QuantizationType qt,
                                          int outputDim, int inputDim, List<nint> allocs)
    {
        if (qt is QuantizationType.F16 or QuantizationType.F32)
            return 0; // Non-quantized weights don't need a separate quantized copy

        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        if (qt == QuantizationType.I2_S)
            quantBytes += 4; // include the trailing per-tensor float32 scale (RowByteSize excludes it).
                             // The I2_S GEMV/dequant kernels read the scale from the tensor tail at
                             // byte offset n·k/4, so the device copy must include these +4 bytes.
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
                                           List<nint> allocs, CudaKernels kernels, nint stream,
                                           string tensorLabel)
    {
        // 64-bit: `outputDim * inputDim` overflows int for a tensor of >2^31 elements
        // (e.g. a 256k-vocab x 16384-hidden LM head = 4.2e9). Every byte-size below already
        // widens to long; the element COUNT itself was the remaining 32-bit product.
        long totalElements = (long)outputDim * inputDim;

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
            // The kernel API takes an int element count; fail loudly rather than
            // silently truncating a >2^31-element tensor.
            kernels.LaunchConvertF32ToF16(devF32, devF16, checked((int)totalElements), stream);
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            allocs.Remove(devF32);
            CudaDriverApi.cuMemFree_v2(devF32);
            return devF16;
        }

        if (qt == QuantizationType.I2_S)
        {
            // I2_S (BitNet ternary): the per-tensor float32 scale lives at the tensor tail
            // (byte offset n·k/4), not per block — so the tail offset must be derived from
            // (outputDim, inputDim), which the generic element-count dequant API cannot do.
            // Upload the packed body + trailing scale (+4), then dequant via the I2_S kernel.
            long packedBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim + 4;
            nint devI2s = AllocAndUpload(hostPtr, packedBytes, allocs);

            long i2sFp16Bytes = (long)totalElements * sizeof(ushort);
            CudaDriverApi.cuMemAlloc_v2(out nint devI2sFp16, (nuint)i2sFp16Bytes).ThrowOnError();
            allocs.Add(devI2sFp16);

            kernels.LaunchDequantI2_SToF16(devI2s, devI2sFp16, outputDim, inputDim, stream);
            return devI2sFp16;
        }

        // No dedicated handling above (no native GEMV/MMQ kernel for this type) — about to
        // fall back to a full, model-lifetime-resident dequant. Gated: see
        // CudaKernels.EnsureQuantExpansionAllowed for why this defaults to a hard failure.
        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        long fp16Bytes = (long)totalElements * sizeof(ushort);
        CudaKernels.EnsureQuantExpansionAllowed(qt, tensorLabel, quantBytes, fp16Bytes);

        // Quantized: upload raw bytes, dequant to FP16 on device
        nint devQuant = AllocAndUpload(hostPtr, quantBytes, allocs);

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16, (nuint)fp16Bytes).ThrowOnError();
        allocs.Add(devFp16);

        kernels.LaunchDequantToF16(devQuant, qt, devFp16, checked((int)totalElements), stream);
        // Free the transient raw-quant upload once the dequant kernel has consumed it — it is
        // never read again (mirrors the F32 branch above, which already frees its own
        // transient upload the same way).
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        allocs.Remove(devQuant);
        CudaDriverApi.cuMemFree_v2(devQuant);
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
        kernels.HasLoadedQuantizedGemv(qt)
            || qt == QuantizationType.I2_S; // I2_S: decode GEMV + on-the-fly prefill dequant (LaunchDequantI2_SToF16)

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
