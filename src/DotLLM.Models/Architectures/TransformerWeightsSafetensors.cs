using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
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
            bool isBitNet = config.Architecture is DotLLM.Core.Configuration.Architecture.BitNet;
            for (int i = 0; i < config.NumLayers; i++)
            {
                layers[i] = isDeepSeekMla
                    ? LoadDeepSeekMlaLayer(i, file, config, owned)
                    : isBitNet
                        ? LoadBitNetLayer(i, file, config, owned)
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
        bool isGemma = config.IsGemmaArchitecture;

        // Pre-attention RMSNorm + all attention projections (Llama-style GQA
        // or Phi-3 fused-QKV, auto-selected by tensor presence; optional
        // Qwen2 biases; optional Qwen3 / Gemma QK-norms).
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);
        var attn = AttentionTensorLoader.Load(AttentionVariant.Gqa, file, config, layerIdx, owned);

        // Per-layer norm weights:
        //  - Standard (Llama/…): `post_attention_layernorm` is the PRE-FFN norm
        //    and there is no post-attn / post-ffn sublayer norm (two-norm layout).
        //  - Gemma (four-norm layout): `post_attention_layernorm` runs on the
        //    attention sublayer output before the residual add, `pre_feedforward_layernorm`
        //    is the pre-FFN norm, and `post_feedforward_layernorm` runs on the FFN
        //    sublayer output before its residual add. Gemma also stores every RMSNorm
        //    weight as an offset from 1.0, absorbed here by adding 1.0 at load.
        float[] ffnNorm;
        float[]? postAttnNorm = null;
        float[]? postFfnNorm = null;
        if (isGemma)
        {
            postAttnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);
            ffnNorm = ResolveNorm(file, $"{prefix}.pre_feedforward_layernorm.weight", hiddenSize);
            postFfnNorm = ResolveNorm(file, $"{prefix}.post_feedforward_layernorm.weight", hiddenSize);

            // (1 + w) RMSNorm absorption — applied to EVERY Gemma RMSNorm weight
            // (input, post-attn, pre-ffn, post-ffn, and the per-head Q/K norms)
            // so the existing RMSNorm kernel runs unchanged.
            GemmaAbsorbOnePlusWeight(attnNorm);
            GemmaAbsorbOnePlusWeight(postAttnNorm);
            GemmaAbsorbOnePlusWeight(ffnNorm);
            GemmaAbsorbOnePlusWeight(postFfnNorm);
            GemmaAbsorbOnePlusWeight(attn.QNormWeight);
            GemmaAbsorbOnePlusWeight(attn.KNormWeight);
        }
        else
        {
            // Post-attention (pre-FFN) RMSNorm — standard two-norm layout.
            ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);
        }

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
                // Gemma 4 MoE: every layer is MoE (no dense/MoE interleaving in
                // the DiffusionGemma text tower). Uses the Qwen-MoE tensor names
                // (mlp.gate + mlp.experts.{j}.{gate,up,down}_proj).
                DotLLM.Core.Configuration.Architecture.Gemma4 => config.Moe.IsMoeLayer(layerIdx),
                // DiffusionGemma: identical Gemma-4 MoE text tower (same Qwen-MoE
                // expert tensor names); the diffusion decode seam is independent of
                // weight loading.
                DotLLM.Core.Configuration.Architecture.DiffusionGemma => config.Moe.IsMoeLayer(layerIdx),
                _ => true,
            };

            if (useRoutedMoE)
            {
                moe = config.Architecture switch
                {
                    DotLLM.Core.Configuration.Architecture.QwenMoe => LoadQwenMoeLayer(layerIdx, file, config, owned),
                    DotLLM.Core.Configuration.Architecture.Gemma4 => LoadQwenMoeLayer(layerIdx, file, config, owned),
                    DotLLM.Core.Configuration.Architecture.DiffusionGemma => LoadQwenMoeLayer(layerIdx, file, config, owned),
                    DotLLM.Core.Configuration.Architecture.GraniteMoe => LoadGraniteMoeLayer(layerIdx, file, config, owned),
                    _ => LoadMixtralMoeLayer(layerIdx, file, config, owned),
                };
                return new TransformerLayerWeights(
                    attnNorm,
                    attn.QWeight, attn.QQuantType, attn.QOutputDim, attn.QInputDim,
                    attn.KWeight, attn.KQuantType, attn.KOutputDim, attn.KInputDim,
                    attn.VWeight, attn.VQuantType, attn.VOutputDim, attn.VInputDim,
                    attn.OWeight, attn.OQuantType, attn.OOutputDim, attn.OInputDim,
                    ffnNorm,
                    gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                    upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                    downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                    attn.QBias, attn.KBias, attn.VBias, attn.OBias,
                    gateBias: null, upBias: null, downBias: null,
                    qNormWeight: attn.QNormWeight, kNormWeight: attn.KNormWeight,
                    moe: moe,
                    mla: null,
                    postAttnNormWeight: postAttnNorm, postFfnNormWeight: postFfnNorm);
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
            qNormWeight: attn.QNormWeight, kNormWeight: attn.KNormWeight,
            moe: null,
            mla: null,
            postAttnNormWeight: postAttnNorm, postFfnNormWeight: postFfnNorm);
    }

    /// <summary>
    /// Loads one BitNet b1.58 transformer layer from an HF safetensors
    /// checkpoint. Unlike <see cref="LoadLayer"/> (which keeps linears at
    /// F32/F16), every linear projection here is quantized to ternary
    /// <see cref="QuantizationType.I2_S"/> at load via
    /// <see cref="DotLLM.Cpu.Kernels.BitNetQuantize.QuantizeToI2S"/> — mirroring
    /// what the GGUF BitNet path receives pre-quantized from Microsoft's
    /// converter. The two BitNet Sub-LN norms
    /// (<c>self_attn.attn_sub_norm.weight</c> over the attention output before
    /// <c>o_proj</c>, and <c>mlp.ffn_sub_norm.weight</c> over the gated
    /// intermediate before <c>down_proj</c>) are wired into
    /// <see cref="TransformerLayerWeights.AttnSubNormWeight"/> /
    /// <see cref="TransformerLayerWeights.FfnSubNormWeight"/> so the forward
    /// pass applies Sub-LN exactly as it does for GGUF. The squared-ReLU FFN is
    /// selected by <see cref="ModelConfig.ActivationFunction"/> = ReluSquared.
    /// </summary>
    private static TransformerLayerWeights LoadBitNetLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = config.IntermediateSize;
        int qDim = config.NumAttentionHeads * config.HeadDim;
        int kvDim = config.NumKvHeads * config.HeadDim;

        // Pre-attention RMSNorm.
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);

        // Attention projections — quantized to ternary I2_S.
        var (qPtr, qQt, qM, qK) = ResolveLinearAsI2S(file, $"{prefix}.self_attn.q_proj.weight", owned);
        var (kPtr, kQt, kM, kK) = ResolveLinearAsI2S(file, $"{prefix}.self_attn.k_proj.weight", owned);
        var (vPtr, vQt, vM, vK) = ResolveLinearAsI2S(file, $"{prefix}.self_attn.v_proj.weight", owned);
        var (oPtr, oQt, oM, oK) = ResolveLinearAsI2S(file, $"{prefix}.self_attn.o_proj.weight", owned);
        ValidateProjectionShape(qM, qK, qDim, hiddenSize, $"{prefix}.self_attn.q_proj.weight");
        ValidateProjectionShape(kM, kK, kvDim, hiddenSize, $"{prefix}.self_attn.k_proj.weight");
        ValidateProjectionShape(vM, vK, kvDim, hiddenSize, $"{prefix}.self_attn.v_proj.weight");
        ValidateProjectionShape(oM, oK, hiddenSize, qDim, $"{prefix}.self_attn.o_proj.weight");

        // BitNet attention Sub-LN — RMSNorm over the attention output [hidden]
        // before o_proj.
        float[] attnSubNorm = ResolveNorm(file, $"{prefix}.self_attn.attn_sub_norm.weight", hiddenSize);

        // Pre-FFN RMSNorm.
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // BitNet FFN Sub-LN — RMSNorm over the gated intermediate [intermediate]
        // before down_proj.
        float[] ffnSubNorm = ResolveNorm(file, $"{prefix}.mlp.ffn_sub_norm.weight", intermediateSize);

        // Dense SwiGLU-shaped FFN projections — quantized to ternary I2_S.
        var (gatePtr, gateQt, gateM, gateK) = ResolveLinearAsI2S(file, $"{prefix}.mlp.gate_proj.weight", owned);
        var (upPtr, upQt, upM, upK) = ResolveLinearAsI2S(file, $"{prefix}.mlp.up_proj.weight", owned);
        var (downPtr, downQt, downM, downK) = ResolveLinearAsI2S(file, $"{prefix}.mlp.down_proj.weight", owned);
        ValidateProjectionShape(gateM, gateK, intermediateSize, hiddenSize, $"{prefix}.mlp.gate_proj.weight");
        ValidateProjectionShape(upM, upK, intermediateSize, hiddenSize, $"{prefix}.mlp.up_proj.weight");
        ValidateProjectionShape(downM, downK, hiddenSize, intermediateSize, $"{prefix}.mlp.down_proj.weight");

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
            qBias: null, kBias: null, vBias: null, oBias: null,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: null, kNormWeight: null,
            moe: null,
            mla: null,
            postAttnNormWeight: null, postFfnNormWeight: null,
            gemma4: null,
            attnSubNormWeight: attnSubNorm, ffnSubNormWeight: ffnSubNorm);
    }

    /// <summary>
    /// Resolves a rank-2 projection weight and quantizes it to ternary
    /// <see cref="QuantizationType.I2_S"/> for BitNet. The source tensor is
    /// first upcast to F32 (BF16/F16 → owned scratch, F32 → zero-copy mmap),
    /// then <see cref="DotLLM.Cpu.Kernels.BitNetQuantize.QuantizeToI2S"/>
    /// packs it into a fresh 64-byte-aligned buffer of
    /// <c>m*k/4 + sizeof(float)</c> bytes (registered in
    /// <paramref name="owned"/>). Any temporary F32 upcast buffer is freed
    /// once the packed I2_S buffer exists. The element count (<c>m*k</c>) must
    /// be a multiple of 128 — always true for real BitNet dims.
    /// </summary>
    private static unsafe (nint ptr, QuantizationType qt, int m, int k) ResolveLinearAsI2S(
        ISafetensorsTensorSource file, string name, List<nint> owned)
    {
        // Upcast to F32 first (temporary — freed below once packed).
        var temp = new List<nint>();
        var (f32Ptr, _, m, k) = ResolveLinearAsF32(file, name, temp);
        try
        {
            long count = (long)m * k;
            if (count % 128 != 0)
                throw new InvalidDataException(
                    $"BitNet linear '{name}' element count {count} (shape [{m},{k}]) is not a "
                    + "multiple of 128; I2_S ternary quantization requires 128-element blocks.");

            long packedBytes = count / 4;
            nuint destBytes = checked((nuint)(packedBytes + sizeof(float)));
            nint dst = (nint)NativeMemory.AlignedAlloc(destBytes, 64);
            owned.Add(dst);

            BitNetQuantize.QuantizeToI2S(
                new ReadOnlySpan<float>((void*)f32Ptr, checked((int)count)),
                count,
                new Span<byte>((void*)dst, checked((int)destBytes)));

            return (dst, QuantizationType.I2_S, m, k);
        }
        finally
        {
            // Free any temporary F32 upcast scratch (BF16/F16 source). An F32
            // source is a zero-copy mmap view and won't be in `temp`.
            foreach (var p in temp)
                NativeMemory.AlignedFree((void*)p);
        }
    }

    /// <summary>
    /// Applies Gemma's <c>(1 + w)</c> RMSNorm-weight convention in place: adds
    /// 1.0 to every element so the standard RMSNorm kernel (which multiplies by
    /// <c>w</c>) reproduces Gemma's <c>(1 + w)</c> scaling without a special-case
    /// kernel. No-op when <paramref name="weights"/> is null (absent QK-norm).
    /// </summary>
    private static void GemmaAbsorbOnePlusWeight(float[]? weights)
    {
        if (weights is null) return;
        for (int i = 0; i < weights.Length; i++)
            weights[i] += 1.0f;
    }

    /// <summary>
    /// Loads one transformer layer for a DeepSeek-V2 / DeepSeek-V3 checkpoint.
    /// Delegates the attention projections to
    /// <see cref="AttentionTensorLoader"/> with the MLA variant (LoRA-
    /// factored <c>q_a_proj</c>/<c>q_b_proj</c> or monolithic <c>q_proj</c>;
    /// LoRA-factored <c>kv_a_proj_with_mqa</c>/<c>kv_b_proj</c> with shared
    /// rope-K; <c>o_proj</c>; all coerced to F32). Routes the FFN either
    /// through a Llama-style dense SwiGLU (first
    /// <c>first_k_dense_replace</c> layers) or the DeepSeek MoE branch
    /// (plural <c>mlp.shared_experts.{k}.*</c>, no sigmoid gate).
    /// </summary>
    private static TransformerLayerWeights LoadDeepSeekMlaLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Pre-attention RMSNorm (standard Llama-style input_layernorm) + all
        // MLA-specific Q/KV/O projections (LoRA-factored on V2/V3, monolithic
        // Q on V2-Lite). Coerces every projection to F32 — the scalar MLA
        // kernel consumes F32 row-major throughout.
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);
        var attn = AttentionTensorLoader.Load(AttentionVariant.Mla, file, config, layerIdx, owned);
        var mla = attn.Mla!; // Guaranteed non-null for AttentionVariant.Mla.

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
                attn.OWeight, attn.OQuantType, attn.OOutputDim, attn.OInputDim,
                ffnNorm,
                gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                qBias: null, kBias: null, vBias: null, oBias: attn.OBias,
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
            attn.OWeight, attn.OQuantType, attn.OOutputDim, attn.OInputDim,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias: null, kBias: null, vBias: null, oBias: attn.OBias,
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

        // Shared expert(s). Three naming conventions in the wild:
        //   - Qwen1.5-MoE-A2.7B: singular mlp.shared_expert.{gate,up,down}_proj
        //     (always exactly one shared expert; optionally gated by
        //     mlp.shared_expert_gate.weight).
        //   - DeepSeek-V2/V3: a SINGLE fused MLP at
        //     mlp.shared_experts.{gate,up,down}_proj (plural, NO numeric index)
        //     with intermediate_size = moe_intermediate * n_shared_experts.
        //     HfConfigExtractor represents this as NumSharedExperts=1,
        //     SharedExpertIntermediateSize=moe_intermediate*n_shared_experts.
        //   - Indexed-plural (forward-compat) mlp.shared_experts.{k}.{gate,up,down}_proj
        //     — not used by any shipped checkpoint we've seen but kept as a
        //     fallback for future model families.
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
            // Detect the tensor-name convention.
            // Priority: DeepSeek fused-plural (no index) first — that's the
            // actually-shipped format for DeepSeek-V2/V3. Then indexed-plural.
            // Then singular (Qwen1.5-MoE).
            bool hasFusedPlural = numShared == 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_experts.gate_proj.weight");
            bool hasIndexedPlural = !hasFusedPlural
                && numShared >= 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_experts.0.gate_proj.weight");
            bool hasSingular = !hasFusedPlural && !hasIndexedPlural
                && numShared == 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_expert.gate_proj.weight");

            if (hasFusedPlural)
            {
                // DeepSeek-V2/V3: single fused-intermediate shared MLP. Matches
                // HF's DeepseekV2MoE.__init__ which builds one DeepseekV2MLP
                // with intermediate_size = moe_intermediate * n_shared_experts.
                sharedIntermediate = sharedI;
                sharedGate = new nint[1];
                sharedUp = new nint[1];
                sharedDown = new nint[1];
                (sharedGate[0], _, int sgM, int sgK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_experts.gate_proj.weight", owned);
                ValidateProjectionShape(sgM, sgK, sharedI, hiddenSize,
                    $"{prefix}.shared_experts.gate_proj.weight");
                (sharedUp[0], _, int suM, int suK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_experts.up_proj.weight", owned);
                ValidateProjectionShape(suM, suK, sharedI, hiddenSize,
                    $"{prefix}.shared_experts.up_proj.weight");
                (sharedDown[0], _, int sdM, int sdK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_experts.down_proj.weight", owned);
                ValidateProjectionShape(sdM, sdK, hiddenSize, sharedI,
                    $"{prefix}.shared_experts.down_proj.weight");
            }
            else if (hasIndexedPlural)
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
