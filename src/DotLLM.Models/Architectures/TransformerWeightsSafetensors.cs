using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;
using static DotLLM.Models.Architectures.SafetensorsTensorResolver;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Loads <see cref="TransformerWeights"/> from a HuggingFace-convention
/// <see cref="SafetensorsFile"/>. Mirrors <see cref="TransformerWeights.LoadFromGguf"/>
/// but reads the HF tensor naming scheme
/// (<c>model.layers.{i}.self_attn.q_proj.weight</c>,
/// <c>model.embed_tokens.weight</c>, <c>lm_head.weight</c>, …).
/// </summary>
/// <remarks>
/// <para>
/// Stored F32 tensors are wired as zero-copy <c>nint</c> handles into the
/// mmap view. BF16 tensors are upcast into 64-byte-aligned
/// <see cref="NativeMemory.AlignedAlloc"/> scratch (a copy-at-load cost, but
/// the only way to feed existing F32 SIMD kernels). Any owned scratch
/// allocations are tracked by <see cref="TransformerWeights"/> and released
/// by its <see cref="TransformerWeights.Dispose"/>.
/// </para>
/// <para>
/// <b>tie_word_embeddings.</b> When the HF config declares tied embeddings
/// and <c>lm_head.weight</c> is physically absent from the safetensors file,
/// the LM-head pointer aliases <c>model.embed_tokens.weight</c>. The
/// resulting <c>TransformerWeights</c> treats that alias as a plain pointer
/// with no extra ownership — the mmap anchor keeps it alive.
/// </para>
/// </remarks>
internal static class TransformerWeightsSafetensorsLoader
{
    /// <summary>
    /// Resolves every transformer weight tensor from <paramref name="file"/>
    /// against the HF naming scheme for the architectures in
    /// <paramref name="config"/>. Throws on missing required tensors.
    /// </summary>
    public static TransformerWeights Load(ISafetensorsTensorSource file, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        var owned = new List<nint>();
        try
        {
            // Token embedding
            var (embPtr, embQt, embM, embK) = ResolveLinear(file, "model.embed_tokens.weight", owned);
            if (embM != config.VocabSize || embK != config.HiddenSize)
                throw new InvalidDataException(
                    $"model.embed_tokens.weight shape [{embM},{embK}] does not match config [vocab={config.VocabSize}, hidden={config.HiddenSize}].");

            var layers = new TransformerLayerWeights[config.NumLayers];
            bool isDeepSeekMla = config.Architecture
                                   is DotLLM.Core.Configuration.Architecture.DeepSeekV2
                                   or DotLLM.Core.Configuration.Architecture.DeepSeekV3
                                 && config.MlaConfig is not null;
            for (int i = 0; i < config.NumLayers; i++)
            {
                layers[i] = isDeepSeekMla
                    ? LoadDeepSeekMlaLayer(i, file, config, owned)
                    : LoadLayer(i, file, config, owned);
            }

            // Final RMSNorm
            float[] outputNorm = ResolveNorm(file, "model.norm.weight", config.HiddenSize);

            // LM head — may be tied to embeddings
            nint outPtr;
            QuantizationType outQt;
            int outM, outK;
            if (file.TensorsByName.ContainsKey("lm_head.weight"))
            {
                (outPtr, outQt, outM, outK) = ResolveLinear(file, "lm_head.weight", owned);
            }
            else
            {
                // Tied: alias the embedding matrix. lm_head is logically [vocab, hidden]
                // and so is the embedding, so the shape/pointer line up directly.
                outPtr = embPtr;
                outQt = embQt;
                outM = embM;
                outK = embK;
            }
            if (outM != config.VocabSize || outK != config.HiddenSize)
                throw new InvalidDataException(
                    $"lm_head.weight shape [{outM},{outK}] does not match config [vocab={config.VocabSize}, hidden={config.HiddenSize}].");

            return TransformerWeights.CreateFromSafetensors(
                tokenEmbedWeight: embPtr, tokenEmbedQt: embQt,
                vocabSize: config.VocabSize, hiddenSize: config.HiddenSize,
                layers: layers,
                outputNormWeight: outputNorm,
                outputWeight: outPtr, outputQt: outQt, outputM: outM, outputK: outK,
                ownedAllocations: owned);
        }
        catch
        {
            // Roll back any allocations we made before rethrowing.
            foreach (var p in owned)
                unsafe { NativeMemory.AlignedFree((void*)p); }
            throw;
        }
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Pre-attention RMSNorm + all attention projections (Llama-style GQA
        // or Phi-3 fused-QKV, auto-selected by tensor presence; optional
        // Qwen2 biases; optional Qwen3 QK-norms).
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);
        var attn = AttentionTensorLoader.Load(AttentionVariant.Gqa, file, config, layerIdx, owned);

        // Post-attention (pre-FFN) RMSNorm
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // FFN projections — HF SwiGLU names: gate_proj, up_proj, down_proj.
        // Phi-3 convention fuses gate+up into `mlp.gate_up_proj.weight` of
        // shape [2*intermediate, hidden] (row-major, gate rows [0..I),
        // up rows [I..2I)). Split per-layer into two owned F32 allocations
        // when the fused form is present; otherwise fall through to per-
        // tensor resolution (Llama/Mistral/Qwen convention).
        nint gatePtr, upPtr, downPtr;
        QuantizationType gateQt, upQt, downQt;
        int gateM, gateK, upM, upK, downM, downK;
        string fusedGateUpName = $"{prefix}.mlp.gate_up_proj.weight";
        if (file.TensorsByName.ContainsKey(fusedGateUpName))
        {
            SplitFusedProjection(
                file, fusedGateUpName,
                new[] { config.IntermediateSize, config.IntermediateSize }, hiddenSize, owned,
                out var gateUpPtrs);
            gatePtr = gateUpPtrs[0]; upPtr = gateUpPtrs[1];
            gateQt = upQt = QuantizationType.F32;
            gateM = upM = config.IntermediateSize;
            gateK = upK = hiddenSize;
        }
        else
        {
            (gatePtr, gateQt, gateM, gateK) = ResolveLinear(file, $"{prefix}.mlp.gate_proj.weight", owned);
            (upPtr, upQt, upM, upK) = ResolveLinear(file, $"{prefix}.mlp.up_proj.weight", owned);
            ValidateProjectionShape(gateM, gateK, config.IntermediateSize, hiddenSize, $"{prefix}.mlp.gate_proj.weight");
            ValidateProjectionShape(upM, upK, config.IntermediateSize, hiddenSize, $"{prefix}.mlp.up_proj.weight");
        }
        (downPtr, downQt, downM, downK) = ResolveLinear(file, $"{prefix}.mlp.down_proj.weight", owned);
        ValidateProjectionShape(downM, downK, hiddenSize, config.IntermediateSize, $"{prefix}.mlp.down_proj.weight");

        return new TransformerLayerWeights(
            attnNorm,
            attn.QWeight, attn.QQuantType, attn.QOutputDim, attn.QInputDim,
            attn.KWeight, attn.KQuantType, attn.KOutputDim, attn.KInputDim,
            attn.VWeight, attn.VQuantType, attn.VOutputDim, attn.VInputDim,
            attn.OWeight, attn.OQuantType, attn.OOutputDim, attn.OInputDim,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            attn.QBias, attn.KBias, attn.VBias, attn.OBias,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: attn.QNormWeight, kNormWeight: attn.KNormWeight);
    }

    /// <summary>
    /// Loads one transformer layer for a DeepSeek-V2 / DeepSeek-V3 checkpoint.
    /// Routes the attention projections through the MLA-specific tensor
    /// naming (<c>q_a_proj</c> / <c>q_b_proj</c> or monolithic <c>q_proj</c>,
    /// <c>kv_a_proj_with_mqa</c>, <c>kv_b_proj</c>, their layernorms, and
    /// <c>o_proj</c>). The FFN side currently loads a Llama-style dense
    /// SwiGLU — the DeepSeek MoE branch lands with the MoE foundation PR.
    /// All MLA tensors are coerced to F32 via
    /// <see cref="SafetensorsTensorResolver.ResolveLinearAsF32"/>; the scalar
    /// MLA kernel consumes F32 row-major throughout.
    /// </summary>
    private static TransformerLayerWeights LoadDeepSeekMlaLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var mlaCfg = config.MlaConfig
                     ?? throw new InvalidOperationException(
                         "LoadDeepSeekMlaLayer called but ModelConfig.MlaConfig is null.");

        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int qkNope = mlaCfg.QkNopeHeadDim;
        int qkRope = mlaCfg.QkRopeHeadDim;
        int qkHead = qkNope + qkRope;
        int vHead = mlaCfg.VHeadDim;
        int qLoraRank = mlaCfg.QLoraRank;
        int kvLoraRank = mlaCfg.KvLoraRank;
        int qTotalOut = numHeads * qkHead;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInputDim = numHeads * vHead;

        // Pre-attention RMSNorm (standard Llama-style input_layernorm).
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);

        // Q path: LoRA-factored (V2 full, V3) or monolithic (V2-Lite). The
        // kernel decides which path to take based on qLoraRank; we pass zero
        // pointers for the unused set.
        nint qAProj = 0, qBProj = 0, qProj = 0;
        float[]? qALayernorm = null;
        if (qLoraRank > 0)
        {
            (qAProj, _, int qAm, int qAk) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_a_proj.weight", owned);
            ValidateProjectionShape(qAm, qAk, qLoraRank, hiddenSize,
                $"{prefix}.self_attn.q_a_proj.weight");
            qALayernorm = ResolveNorm(file, $"{prefix}.self_attn.q_a_layernorm.weight", qLoraRank);
            (qBProj, _, int qBm, int qBk) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_b_proj.weight", owned);
            ValidateProjectionShape(qBm, qBk, qTotalOut, qLoraRank,
                $"{prefix}.self_attn.q_b_proj.weight");
        }
        else
        {
            (qProj, _, int qM, int qK) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_proj.weight", owned);
            ValidateProjectionShape(qM, qK, qTotalOut, hiddenSize,
                $"{prefix}.self_attn.q_proj.weight");
        }

        // KV path: always LoRA-factored. kv_a_proj_with_mqa emits
        // [kvLoraRank + qkRopeHeadDim] per token — the first kvLoraRank rows
        // feed kv_a_layernorm then kv_b_proj, the last qkRopeHeadDim rows are
        // the MQA-shared rope-K. No separate LayerNorm on the rope-K side.
        int kvADim = kvLoraRank + qkRope;
        (nint kvAProj, _, int kvaM, int kvaK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", owned);
        ValidateProjectionShape(kvaM, kvaK, kvADim, hiddenSize,
            $"{prefix}.self_attn.kv_a_proj_with_mqa.weight");
        float[] kvALayernorm = ResolveNorm(
            file, $"{prefix}.self_attn.kv_a_layernorm.weight", kvLoraRank);
        (nint kvBProj, _, int kvbM, int kvbK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_b_proj.weight", owned);
        ValidateProjectionShape(kvbM, kvbK, kvBOut, kvLoraRank,
            $"{prefix}.self_attn.kv_b_proj.weight");

        // Output projection: hidden ← n_heads * v_head_dim. Kept in the
        // existing O slot (not MLA-specific) because the forward path still
        // applies bias (if any) through the same AddBias logic.
        var (oPtr, oQt, oM, oK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.o_proj.weight", owned);
        ValidateProjectionShape(oM, oK, hiddenSize, oInputDim,
            $"{prefix}.self_attn.o_proj.weight");
        float[]? oBias = ResolveOptionalBias(file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        var mla = new MlaLayerWeights(
            qAProj: qAProj, qALayernormWeight: qALayernorm, qBProj: qBProj, qProj: qProj,
            kvAProjWithMqa: kvAProj, kvALayernormWeight: kvALayernorm, kvBProj: kvBProj,
            numHeads: numHeads,
            qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
            qLoraRank: qLoraRank, kvLoraRank: kvLoraRank);

        // Post-attention RMSNorm (shared with Llama convention).
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // Dense FFN (Llama SwiGLU convention). DeepSeek-V2/V3 interleaves
        // dense MLP (first_k_dense_replace layers) with MoE (rest) — only
        // the dense path is wired in this foundation PR. The MoE FFN branch
        // and its layer-level routing land with the MoE foundation PR.
        var (gatePtr, gateQt, gateM, gateK) = ResolveLinear(
            file, $"{prefix}.mlp.gate_proj.weight", owned);
        var (upPtr, upQt, upM, upK) = ResolveLinear(
            file, $"{prefix}.mlp.up_proj.weight", owned);
        var (downPtr, downQt, downM, downK) = ResolveLinear(
            file, $"{prefix}.mlp.down_proj.weight", owned);
        ValidateProjectionShape(gateM, gateK, config.IntermediateSize, hiddenSize,
            $"{prefix}.mlp.gate_proj.weight");
        ValidateProjectionShape(upM, upK, config.IntermediateSize, hiddenSize,
            $"{prefix}.mlp.up_proj.weight");
        ValidateProjectionShape(downM, downK, hiddenSize, config.IntermediateSize,
            $"{prefix}.mlp.down_proj.weight");

        return new TransformerLayerWeights(
            attnNorm,
            qWeight: 0, qQuantType: QuantizationType.F32, qOutputDim: 0, qInputDim: 0,
            kWeight: 0, kQuantType: QuantizationType.F32, kOutputDim: 0, kInputDim: 0,
            vWeight: 0, vQuantType: QuantizationType.F32, vOutputDim: 0, vInputDim: 0,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias: null, kBias: null, vBias: null, oBias: oBias,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: null, kNormWeight: null,
            mla: mla);
    }

}
