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
            for (int i = 0; i < config.NumLayers; i++)
            {
                layers[i] = LoadLayer(i, file, config, owned);
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

}
