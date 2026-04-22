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
        int headDim = config.HeadDim;
        int qOut = config.NumAttentionHeads * headDim;
        int kvOut = config.NumKvHeads * headDim;

        // Input (pre-attention) RMSNorm
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);

        // Q / K / V / O projections.
        // Phi-3 convention fuses QKV into a single `self_attn.qkv_proj.weight`
        // of shape [qOut + 2*kvOut, hidden] (row-major, Q top → K → V). Split
        // per-layer into three owned F32 allocations so downstream forward
        // path sees the standard Q/K/V slots. Falls through to per-tensor
        // resolution when the fused tensor is absent (vanilla Llama/Mistral/
        // Qwen convention).
        nint qPtr, kPtr, vPtr, oPtr;
        QuantizationType qQt, kQt, vQt, oQt;
        int qM, qK, kM, kK, vM, vK, oM, oK;
        string fusedQkvName = $"{prefix}.self_attn.qkv_proj.weight";
        if (file.TensorsByName.ContainsKey(fusedQkvName))
        {
            SplitFusedProjection(
                file, fusedQkvName, new[] { qOut, kvOut, kvOut }, hiddenSize, owned,
                out var qkvPtrs);
            qPtr = qkvPtrs[0]; kPtr = qkvPtrs[1]; vPtr = qkvPtrs[2];
            qQt = kQt = vQt = QuantizationType.F32;
            qM = qOut; kM = kvOut; vM = kvOut;
            qK = kK = vK = hiddenSize;
        }
        else
        {
            (qPtr, qQt, qM, qK) = ResolveLinear(file, $"{prefix}.self_attn.q_proj.weight", owned);
            (kPtr, kQt, kM, kK) = ResolveLinear(file, $"{prefix}.self_attn.k_proj.weight", owned);
            (vPtr, vQt, vM, vK) = ResolveLinear(file, $"{prefix}.self_attn.v_proj.weight", owned);
            ValidateProjectionShape(qM, qK, qOut, hiddenSize, $"{prefix}.self_attn.q_proj.weight");
            ValidateProjectionShape(kM, kK, kvOut, hiddenSize, $"{prefix}.self_attn.k_proj.weight");
            ValidateProjectionShape(vM, vK, kvOut, hiddenSize, $"{prefix}.self_attn.v_proj.weight");
        }
        (oPtr, oQt, oM, oK) = ResolveLinear(file, $"{prefix}.self_attn.o_proj.weight", owned);
        ValidateProjectionShape(oM, oK, hiddenSize, qOut, $"{prefix}.self_attn.o_proj.weight");

        // Optional projection biases (Qwen2 has q/k/v biases; Llama does not)
        float[]? qBias = ResolveOptionalBias(file, $"{prefix}.self_attn.q_proj.bias", qOut);
        float[]? kBias = ResolveOptionalBias(file, $"{prefix}.self_attn.k_proj.bias", kvOut);
        float[]? vBias = ResolveOptionalBias(file, $"{prefix}.self_attn.v_proj.bias", kvOut);
        float[]? oBias = ResolveOptionalBias(file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        // Optional QK-norms (Qwen3 per-head RMSNorm). Not emitted by vanilla HF
        // Llama/Mistral/Qwen2. Qwen3 names them {q_norm,k_norm}.weight.
        float[]? qNorm = ResolveOptionalNorm(file, $"{prefix}.self_attn.q_norm.weight", headDim);
        float[]? kNorm = ResolveOptionalNorm(file, $"{prefix}.self_attn.k_norm.weight", headDim);

        // Post-attention (pre-FFN) RMSNorm
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // FFN — dense (Llama/Mistral/Qwen), Mixtral-convention MoE, or
        // Qwen-MoE-convention MoE (possibly interleaved with dense layers via
        // decoder_sparse_step / mlp_only_layers).
        if (config.Moe is not null)
        {
            MoeLayerWeights? moe = null;
            bool useRoutedMoE = config.Architecture switch
            {
                // Mixtral: every layer is MoE.
                DotLLM.Core.Configuration.Architecture.Mixtral => true,
                // Qwen-MoE: per-layer decision based on decoder_sparse_step
                // and mlp_only_layers. A "dense" Qwen-MoE layer uses the
                // standard Llama-style mlp.{gate,up,down}_proj names — fall
                // through to the dense path below.
                DotLLM.Core.Configuration.Architecture.QwenMoe => config.Moe.IsMoeLayer(layerIdx),
                _ => true,
            };

            if (useRoutedMoE)
            {
                moe = config.Architecture switch
                {
                    DotLLM.Core.Configuration.Architecture.QwenMoe => LoadQwenMoeLayer(layerIdx, file, config, owned),
                    DotLLM.Core.Configuration.Architecture.GraniteMoe => LoadGraniteMoeLayer(layerIdx, file, config, owned),
                    _ => LoadMixtralMoeLayer(layerIdx, file, config, owned),
                };
                return new TransformerLayerWeights(
                    attnNorm,
                    qPtr, qQt, qM, qK,
                    kPtr, kQt, kM, kK,
                    vPtr, vQt, vM, vK,
                    oPtr, oQt, oM, oK,
                    ffnNorm,
                    gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                    upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                    downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                    qBias, kBias, vBias, oBias,
                    gateBias: null, upBias: null, downBias: null,
                    qNormWeight: qNorm, kNormWeight: kNorm,
                    moe: moe);
            }
            // Otherwise: Qwen-MoE interleaved DENSE layer — fall through to
            // the Llama-style dense SwiGLU resolution below.
        }

        // Dense FFN — HF SwiGLU names: gate_proj, up_proj, down_proj.
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
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias, kBias, vBias, oBias,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: qNorm, kNormWeight: kNorm);
    }

    /// <summary>
    /// Loads one transformer layer for a DeepSeek-V2 / DeepSeek-V3 checkpoint.
    /// Routes the attention projections through the MLA-specific tensor
    /// naming (<c>q_a_proj</c> / <c>q_b_proj</c> or monolithic <c>q_proj</c>,
    /// <c>kv_a_proj_with_mqa</c>, <c>kv_b_proj</c>, their layernorms, and
    /// <c>o_proj</c>), and routes the FFN either through a Llama-style dense
    /// SwiGLU (first <c>first_k_dense_replace</c> layers) or the DeepSeek
    /// MoE branch (plural <c>mlp.shared_experts.{k}.*</c>, no sigmoid gate).
    /// All MLA tensors are coerced to F32 via
    /// <see cref="ResolveLinearAsF32"/>; the scalar MLA kernel consumes F32
    /// row-major throughout.
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

        // FFN: DeepSeek interleaves dense MLP (first K layers) and MoE (rest).
        // ExtractMoeConfig folds first_k_dense_replace into MlpOnlyLayers so
        // IsMoeLayer() already resolves this correctly.
        if (config.Moe is not null && config.Moe.IsMoeLayer(layerIdx))
        {
            var moe = LoadQwenMoeLayer(layerIdx, file, config, owned);
            return new TransformerLayerWeights(
                attnNorm,
                qWeight: 0, qQuantType: QuantizationType.F32, qOutputDim: 0, qInputDim: 0,
                kWeight: 0, kQuantType: QuantizationType.F32, kOutputDim: 0, kInputDim: 0,
                vWeight: 0, vQuantType: QuantizationType.F32, vOutputDim: 0, vInputDim: 0,
                oPtr, oQt, oM, oK,
                ffnNorm,
                gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                qBias: null, kBias: null, vBias: null, oBias: oBias,
                gateBias: null, upBias: null, downBias: null,
                qNormWeight: null, kNormWeight: null,
                moe: moe,
                mla: mla);
        }

        // Dense FFN (first_k_dense_replace prefix): Llama SwiGLU convention.
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
            moe: null,
            mla: mla);
    }

    /// <summary>
    /// Loads Qwen-MoE-convention MoE weights for one transformer layer:
    /// <c>model.layers.{i}.mlp.gate.weight</c> and
    /// <c>model.layers.{i}.mlp.experts.{j}.{gate_proj,up_proj,down_proj}.weight</c>
    /// — math-identical to Mixtral but with HF Llama-style tensor names.
    /// When <see cref="MoeConfig.SharedExpertIntermediateSize"/> is set the
    /// parallel shared-expert branch (<c>mlp.shared_expert.*</c>) and
    /// optionally the <c>mlp.shared_expert_gate.weight</c> sigmoid gate are
    /// resolved too. Everything lands in F32 via
    /// <see cref="ResolveLinearAsF32"/> so the kernel is uniform in dtype.
    /// </summary>
    private static MoeLayerWeights LoadQwenMoeLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe
                  ?? throw new InvalidOperationException("LoadQwenMoeLayer called with null Moe config.");

        string prefix = $"model.layers.{layerIdx}.mlp";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = moe.MoeIntermediateSize;
        int numExperts = moe.NumExperts;

        // Router gate — F32 [E, H].
        float[] gate = ResolveDense2D(file, $"{prefix}.gate.weight", numExperts, hiddenSize);

        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            // w1 ≡ gate_proj: [intermediate, hidden]
            (w1[e], _, int w1M, int w1K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.gate_proj.weight", owned);
            ValidateProjectionShape(w1M, w1K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.gate_proj.weight");
            // w3 ≡ up_proj: [intermediate, hidden]
            (w3[e], _, int w3M, int w3K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.up_proj.weight", owned);
            ValidateProjectionShape(w3M, w3K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.up_proj.weight");
            // w2 ≡ down_proj: [hidden, intermediate]
            (w2[e], _, int w2M, int w2K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.down_proj.weight", owned);
            ValidateProjectionShape(w2M, w2K, hiddenSize, intermediateSize,
                $"{prefix}.experts.{e}.down_proj.weight");
        }

        // Shared expert(s). Two naming conventions:
        //   - Qwen1.5-MoE-A2.7B: singular mlp.shared_expert.{gate,up,down}_proj
        //     (always exactly one shared expert; optionally gated by
        //     mlp.shared_expert_gate.weight).
        //   - DeepSeek-V2/V3: plural mlp.shared_experts.{k}.{gate,up,down}_proj
        //     (n_shared_experts >= 1, summed, no gate).
        // We resolve whichever set of tensors the file actually contains; the
        // kernel sees a uniform pointer-array API. If the config flags a shared
        // expert but the tensors are absent, we silently fall back to routed-only.
        nint[] sharedGate = Array.Empty<nint>();
        nint[] sharedUp = Array.Empty<nint>();
        nint[] sharedDown = Array.Empty<nint>();
        int sharedIntermediate = 0;
        float[]? sharedExpertGate = null;
        if (moe.SharedExpertIntermediateSize is int sharedI)
        {
            int numShared = moe.NumSharedExperts;
            // Detect the tensor-name convention. Prefer plural (DeepSeek) when
            // present — this is the forward-compatible format. Fall back to
            // singular (Qwen1.5-MoE) when only that exists.
            bool hasPlural = numShared >= 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_experts.0.gate_proj.weight");
            bool hasSingular = numShared == 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_expert.gate_proj.weight");

            if (hasPlural)
            {
                sharedIntermediate = sharedI;
                sharedGate = new nint[numShared];
                sharedUp = new nint[numShared];
                sharedDown = new nint[numShared];
                for (int k = 0; k < numShared; k++)
                {
                    (sharedGate[k], _, int sgM, int sgK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.gate_proj.weight", owned);
                    ValidateProjectionShape(sgM, sgK, sharedI, hiddenSize,
                        $"{prefix}.shared_experts.{k}.gate_proj.weight");
                    (sharedUp[k], _, int suM, int suK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.up_proj.weight", owned);
                    ValidateProjectionShape(suM, suK, sharedI, hiddenSize,
                        $"{prefix}.shared_experts.{k}.up_proj.weight");
                    (sharedDown[k], _, int sdM, int sdK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.down_proj.weight", owned);
                    ValidateProjectionShape(sdM, sdK, hiddenSize, sharedI,
                        $"{prefix}.shared_experts.{k}.down_proj.weight");
                }
            }
            else if (hasSingular)
            {
                sharedIntermediate = sharedI;
                sharedGate = new nint[1];
                sharedUp = new nint[1];
                sharedDown = new nint[1];
                (sharedGate[0], _, int sgM, int sgK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.gate_proj.weight", owned);
                ValidateProjectionShape(sgM, sgK, sharedI, hiddenSize,
                    $"{prefix}.shared_expert.gate_proj.weight");
                (sharedUp[0], _, int suM, int suK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.up_proj.weight", owned);
                ValidateProjectionShape(suM, suK, sharedI, hiddenSize,
                    $"{prefix}.shared_expert.up_proj.weight");
                (sharedDown[0], _, int sdM, int sdK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.down_proj.weight", owned);
                ValidateProjectionShape(sdM, sdK, hiddenSize, sharedI,
                    $"{prefix}.shared_expert.down_proj.weight");

                // Optional sigmoid gate — HF stores it as [1, hiddenSize] (a plain
                // Linear(hidden -> 1, bias=False)). ElementCount == hiddenSize, so
                // ResolveNorm slots in cleanly.
                string gateName = $"{prefix}.shared_expert_gate.weight";
                if (moe.HasSharedExpertGate && file.TensorsByName.ContainsKey(gateName))
                {
                    sharedExpertGate = ResolveNorm(file, gateName, hiddenSize);
                }
            }
            // else: config declared a shared branch but the file has neither
            // plural nor singular tensors — silently fall back to routed-only
            // (sharedIntermediate stays 0, arrays stay empty).
        }

        return new MoeLayerWeights(
            gate: gate,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: sharedGate,
            sharedUpProj: sharedUp,
            sharedDownProj: sharedDown,
            sharedIntermediateSize: sharedIntermediate,
            sharedExpertGate: sharedExpertGate);
    }

    /// <summary>
    /// Loads Mixtral-convention MoE weights for one transformer layer:
    /// <c>model.layers.{i}.block_sparse_moe.gate.weight</c> and
    /// <c>model.layers.{i}.block_sparse_moe.experts.{j}.(w1|w2|w3).weight</c>.
    /// Router gate is resolved into a managed <c>float[]</c> (tiny —
    /// numExperts × hiddenSize). Per-expert weights are F32 pointers; bf16/
    /// F16 tensors are upcast at load time into 64-byte-aligned scratch and
    /// registered in <paramref name="owned"/>.
    /// </summary>
    private static MoeLayerWeights LoadMixtralMoeLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe
                  ?? throw new InvalidOperationException("LoadMixtralMoeLayer called with null Moe config.");

        string prefix = $"model.layers.{layerIdx}.block_sparse_moe";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = moe.MoeIntermediateSize;
        int numExperts = moe.NumExperts;

        // Router gate — F32 [E, H].
        float[] gate = ResolveDense2D(file, $"{prefix}.gate.weight", numExperts, hiddenSize);

        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            // w1 (gate_proj): [intermediate, hidden]
            (w1[e], _, int w1M, int w1K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w1.weight", owned);
            ValidateProjectionShape(w1M, w1K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.w1.weight");
            // w3 (up_proj): [intermediate, hidden]
            (w3[e], _, int w3M, int w3K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w3.weight", owned);
            ValidateProjectionShape(w3M, w3K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.w3.weight");
            // w2 (down_proj): [hidden, intermediate]
            (w2[e], _, int w2M, int w2K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w2.weight", owned);
            ValidateProjectionShape(w2M, w2K, hiddenSize, intermediateSize,
                $"{prefix}.experts.{e}.w2.weight");
        }

        return new MoeLayerWeights(
            gate: gate,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize);
    }

    /// <summary>
    /// Loads Granite-3.x-convention MoE weights for one transformer layer.
    /// Unlike Mixtral / Qwen-MoE which store each expert's projections as
    /// individual tensors, Granite packs ALL experts of one layer into three
    /// fused rank-3 tensors:
    /// <list type="bullet">
    ///   <item><c>mlp.block_sparse_moe.router.layer.weight [E, H]</c> — router gate.</item>
    ///   <item><c>mlp.block_sparse_moe.input_linear.weight [E, 2*I, H]</c> —
    ///     per-expert w1 (gate_proj) in rows [0..I), w3 (up_proj) in rows
    ///     [I..2*I). One expert = a flat [2*I, H] slab.</item>
    ///   <item><c>mlp.block_sparse_moe.output_linear.weight [E, H, I]</c> —
    ///     per-expert w2 (down_proj), already a [H, I] slab per expert.</item>
    /// </list>
    /// The kernel (<see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp"/>) requires
    /// per-expert F32 row-major pointers. We therefore allocate one F32
    /// buffer per expert per matrix (w1/w2/w3) and upcast from the fused BF16
    /// source — mmap'd zero-copy is not viable because (a) kernels expect
    /// F32, and (b) pointing mid-way into a BF16 tensor would skip the
    /// dtype-conversion layer. Allocations are registered in
    /// <paramref name="owned"/> for deterministic cleanup.
    /// </summary>
    private static unsafe MoeLayerWeights LoadGraniteMoeLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe
                  ?? throw new InvalidOperationException("LoadGraniteMoeLayer called with null Moe config.");

        string prefix = $"model.layers.{layerIdx}.block_sparse_moe";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = moe.MoeIntermediateSize;
        int numExperts = moe.NumExperts;

        // Router gate — fused [E, H] but shape-compatible with the flat
        // [numExperts, hiddenSize] router gate expected by the MoE kernel.
        // ResolveDense2D already upcasts BF16/F16 to F32 into a managed array.
        float[] gate = ResolveDense2D(file, $"{prefix}.router.layer.weight", numExperts, hiddenSize);

        // input_linear: [E, 2*I, H]. Per expert e:
        //   rows [0..I)       = w1 (gate_proj)  — shape [I, H]
        //   rows [I..2*I)     = w3 (up_proj)    — shape [I, H]
        string inputName = $"{prefix}.input_linear.weight";
        if (!file.TensorsByName.TryGetValue(inputName, out var inputDesc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{inputName}'.");
        if (inputDesc.Shape.Length != 3
            || inputDesc.Shape[0] != numExperts
            || inputDesc.Shape[1] != 2 * intermediateSize
            || inputDesc.Shape[2] != hiddenSize)
            throw new InvalidDataException(
                $"Tensor '{inputName}' shape [{string.Join(',', inputDesc.Shape)}] "
                + $"does not match expected [{numExperts},{2 * intermediateSize},{hiddenSize}].");
        nint inputSrc = file.GetTensorPointer(inputName);

        // output_linear: [E, H, I]. Per expert e: shape [H, I] = w2 slab.
        string outputName = $"{prefix}.output_linear.weight";
        if (!file.TensorsByName.TryGetValue(outputName, out var outputDesc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{outputName}'.");
        if (outputDesc.Shape.Length != 3
            || outputDesc.Shape[0] != numExperts
            || outputDesc.Shape[1] != hiddenSize
            || outputDesc.Shape[2] != intermediateSize)
            throw new InvalidDataException(
                $"Tensor '{outputName}' shape [{string.Join(',', outputDesc.Shape)}] "
                + $"does not match expected [{numExperts},{hiddenSize},{intermediateSize}].");
        nint outputSrc = file.GetTensorPointer(outputName);

        long inputPerExpert = (long)(2 * intermediateSize) * hiddenSize;  // elements
        long outputPerExpert = (long)hiddenSize * intermediateSize;       // elements
        long w1Elements = (long)intermediateSize * hiddenSize;
        long w3Elements = (long)intermediateSize * hiddenSize;

        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            // Source byte offsets into the fused tensors. Element type drives
            // the stride: for BF16/F16 the dtype is 2 bytes/element; for F32
            // it's 4. We compute via pointer casts per-dtype to avoid a
            // bytes-based math bug.
            long inputExpertStart = e * inputPerExpert;      // start of expert slab (elements)
            long w1Start = inputExpertStart;                 // first I rows
            long w3Start = inputExpertStart + w1Elements;    // next I rows
            long outputExpertStart = e * outputPerExpert;

            w1[e] = AllocPartAsF32(inputSrc, inputDesc.DType, w1Start, w1Elements, owned, inputName);
            w3[e] = AllocPartAsF32(inputSrc, inputDesc.DType, w3Start, w3Elements, owned, inputName);
            w2[e] = AllocPartAsF32(outputSrc, outputDesc.DType, outputExpertStart, outputPerExpert, owned, outputName);
        }

        return new MoeLayerWeights(
            gate: gate,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: Array.Empty<nint>(),
            sharedUpProj: Array.Empty<nint>(),
            sharedDownProj: Array.Empty<nint>(),
            sharedIntermediateSize: 0,
            sharedExpertGate: null);
    }

}
