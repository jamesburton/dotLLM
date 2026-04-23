using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Identifies the attention tensor layout used by a transformer architecture.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Gqa"/> covers vanilla GQA/MHA/MQA attention where Q/K/V/O are
/// stored as four independent tensors under <c>self_attn.(q|k|v|o)_proj</c>,
/// plus the Phi-3 convention that row-fuses Q/K/V into a single
/// <c>self_attn.qkv_proj</c> tensor. The fused form is auto-detected by the
/// loader based on which tensors are present in the file, so a single
/// <see cref="Gqa"/> variant covers Llama / Mistral / Qwen (2 and 3) /
/// Phi-3 / Granite-MoE / Mixtral / Qwen-MoE (which all share the same
/// Llama-style attention layout; only the FFN differs).
/// </para>
/// <para>
/// <see cref="Mla"/> covers DeepSeek-V2 / V3 Multi-head Latent Attention:
/// LoRA-factored Q (<c>q_a_proj</c> / <c>q_a_layernorm</c> / <c>q_b_proj</c>)
/// or monolithic <c>q_proj</c> (V2-Lite), LoRA-factored KV with a shared
/// rope-K channel (<c>kv_a_proj_with_mqa</c> / <c>kv_a_layernorm</c> /
/// <c>kv_b_proj</c>), and a standard <c>o_proj</c>. All MLA tensors are
/// coerced to F32 at load time because the current MLA kernel is F32-only.
/// </para>
/// </remarks>
internal enum AttentionVariant
{
    /// <summary>
    /// Vanilla Llama-style GQA (also MHA/MQA) attention. Auto-detects the
    /// Phi-3 fused <c>qkv_proj</c> layout and splits it at load time.
    /// </summary>
    Gqa,

    /// <summary>
    /// DeepSeek-V2/V3 Multi-head Latent Attention. Expects
    /// <see cref="ModelConfig.MlaConfig"/> to be non-null.
    /// </summary>
    Mla,
}

/// <summary>
/// Populated attention slot for a single transformer layer — the subset of
/// <see cref="TransformerLayerWeights"/> that the per-architecture
/// attention loader owns. Returned by
/// <see cref="AttentionTensorLoader.Load"/> and consumed by
/// <see cref="TransformerWeightsSafetensorsLoader"/> when building the
/// per-layer weight struct.
/// </summary>
/// <remarks>
/// <para>
/// For the GQA variants the Q/K/V/O slots are populated and <see cref="Mla"/>
/// is null. For MLA the Q/K/V weights/dims are all zero (the forward path
/// takes the MLA branch), the O slot is populated, and <see cref="Mla"/>
/// carries the MLA-specific projections + hyperparameters.
/// </para>
/// <para>
/// All pointer fields alias either (a) zero-copy mmap views into the
/// safetensors file (F32 / F16 tensors that the downstream kernels accept
/// as-is) or (b) 64-byte-aligned owned F32 scratch registered in the
/// caller's <c>owned</c> list (BF16 upcasts + fused-QKV splits + MLA
/// all-F32 coercion). The struct itself owns nothing — lifetime tracking
/// is unchanged.
/// </para>
/// </remarks>
internal readonly record struct AttentionLayerTensors(
    // Q projection
    nint QWeight, QuantizationType QQuantType, int QOutputDim, int QInputDim,
    // K projection
    nint KWeight, QuantizationType KQuantType, int KOutputDim, int KInputDim,
    // V projection
    nint VWeight, QuantizationType VQuantType, int VOutputDim, int VInputDim,
    // O projection
    nint OWeight, QuantizationType OQuantType, int OOutputDim, int OInputDim,
    // Optional biases
    float[]? QBias,
    float[]? KBias,
    float[]? VBias,
    float[]? OBias,
    // Optional QK-norms (Qwen3)
    float[]? QNormWeight,
    float[]? KNormWeight,
    // MLA bundle (non-null iff variant is Mla)
    MlaLayerWeights? Mla);

/// <summary>
/// Shared tensor-name resolver for per-layer transformer attention weights.
/// Extracts the repeated ~30-60-line blocks that previously lived inline in
/// the per-architecture layer loaders (Llama GQA, Phi-3 fused-QKV,
/// DeepSeek MLA, Granite-MoE GQA). See PLANS.md §P1.1.
/// </summary>
/// <remarks>
/// Bit-identical with the pre-extraction code: same resolution order, same
/// validation errors, same BF16→F32 / F16 handling, same owned-allocation
/// bookkeeping.
/// </remarks>
internal static class AttentionTensorLoader
{
    /// <summary>
    /// Resolves the attention projections for <paramref name="layerIdx"/>
    /// under the HF tensor-naming convention implied by
    /// <paramref name="variant"/>.
    /// </summary>
    /// <param name="variant">Attention layout (GQA separate/fused or MLA).</param>
    /// <param name="file">Safetensors source (single-shard or multi-shard view).</param>
    /// <param name="config">Model config — supplies head dims, MLA sub-config, etc.</param>
    /// <param name="layerIdx">Zero-based layer index; used to build the
    /// <c>model.layers.{L}</c> prefix.</param>
    /// <param name="owned">Caller's owned-allocation list; this loader appends
    /// every 64-byte-aligned F32 scratch buffer it creates (BF16 upcasts,
    /// fused-QKV splits, MLA all-F32 coercions) so the parent
    /// <see cref="TransformerWeights"/> can free them on dispose.</param>
    public static AttentionLayerTensors Load(
        AttentionVariant variant,
        ISafetensorsTensorSource file,
        ModelConfig config,
        int layerIdx,
        List<nint> owned)
    {
        return variant switch
        {
            AttentionVariant.Gqa => LoadGqa(file, config, layerIdx, owned),
            AttentionVariant.Mla => LoadMla(file, config, layerIdx, owned),
            _ => throw new ArgumentOutOfRangeException(nameof(variant), variant,
                "Unknown attention variant."),
        };
    }

    /// <summary>
    /// GQA/MHA/MQA path: vanilla separate <c>q_proj</c>/<c>k_proj</c>/
    /// <c>v_proj</c> tensors OR a single Phi-3-style fused <c>qkv_proj</c>
    /// that we split into three owned F32 slabs at load time. Optional
    /// projection biases (Qwen2) and optional per-head QK-norms (Qwen3) are
    /// read when present.
    /// </summary>
    private static AttentionLayerTensors LoadGqa(
        ISafetensorsTensorSource file, ModelConfig config, int layerIdx, List<nint> owned)
    {
        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int headDim = config.HeadDim;
        int qOut = config.NumAttentionHeads * headDim;
        int kvOut = config.NumKvHeads * headDim;

        // Q / K / V projections. Phi-3 convention fuses QKV into a single
        // `self_attn.qkv_proj.weight` of shape [qOut + 2*kvOut, hidden]
        // (row-major, Q top → K → V). Split per-layer into three owned F32
        // allocations so downstream forward path sees the standard Q/K/V
        // slots. Falls through to per-tensor resolution when the fused
        // tensor is absent (vanilla Llama/Mistral/Qwen convention).
        nint qPtr, kPtr, vPtr;
        QuantizationType qQt, kQt, vQt;
        int qM, qK, kM, kK, vM, vK;
        string fusedQkvName = $"{prefix}.self_attn.qkv_proj.weight";
        if (file.TensorsByName.ContainsKey(fusedQkvName))
        {
            SafetensorsTensorResolver.SplitFusedProjection(
                file, fusedQkvName, new[] { qOut, kvOut, kvOut }, hiddenSize, owned,
                out var qkvPtrs);
            qPtr = qkvPtrs[0]; kPtr = qkvPtrs[1]; vPtr = qkvPtrs[2];
            qQt = kQt = vQt = QuantizationType.F32;
            qM = qOut; kM = kvOut; vM = kvOut;
            qK = kK = vK = hiddenSize;
        }
        else
        {
            (qPtr, qQt, qM, qK) = SafetensorsTensorResolver.ResolveLinear(file, $"{prefix}.self_attn.q_proj.weight", owned);
            (kPtr, kQt, kM, kK) = SafetensorsTensorResolver.ResolveLinear(file, $"{prefix}.self_attn.k_proj.weight", owned);
            (vPtr, vQt, vM, vK) = SafetensorsTensorResolver.ResolveLinear(file, $"{prefix}.self_attn.v_proj.weight", owned);
            SafetensorsTensorResolver.ValidateProjectionShape(qM, qK, qOut, hiddenSize, $"{prefix}.self_attn.q_proj.weight");
            SafetensorsTensorResolver.ValidateProjectionShape(kM, kK, kvOut, hiddenSize, $"{prefix}.self_attn.k_proj.weight");
            SafetensorsTensorResolver.ValidateProjectionShape(vM, vK, kvOut, hiddenSize, $"{prefix}.self_attn.v_proj.weight");
        }

        var (oPtr, oQt, oM, oK) = SafetensorsTensorResolver.ResolveLinear(file, $"{prefix}.self_attn.o_proj.weight", owned);
        SafetensorsTensorResolver.ValidateProjectionShape(oM, oK, hiddenSize, qOut, $"{prefix}.self_attn.o_proj.weight");

        // Optional projection biases (Qwen2 has q/k/v biases; Llama does not)
        float[]? qBias = SafetensorsTensorResolver.ResolveOptionalBias(file, $"{prefix}.self_attn.q_proj.bias", qOut);
        float[]? kBias = SafetensorsTensorResolver.ResolveOptionalBias(file, $"{prefix}.self_attn.k_proj.bias", kvOut);
        float[]? vBias = SafetensorsTensorResolver.ResolveOptionalBias(file, $"{prefix}.self_attn.v_proj.bias", kvOut);
        float[]? oBias = SafetensorsTensorResolver.ResolveOptionalBias(file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        // Optional QK-norms (Qwen3 per-head RMSNorm). Not emitted by vanilla
        // HF Llama/Mistral/Qwen2. Qwen3 names them {q_norm,k_norm}.weight.
        float[]? qNorm = SafetensorsTensorResolver.ResolveOptionalNorm(file, $"{prefix}.self_attn.q_norm.weight", headDim);
        float[]? kNorm = SafetensorsTensorResolver.ResolveOptionalNorm(file, $"{prefix}.self_attn.k_norm.weight", headDim);

        return new AttentionLayerTensors(
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            qBias, kBias, vBias, oBias,
            QNormWeight: qNorm, KNormWeight: kNorm,
            Mla: null);
    }

    /// <summary>
    /// DeepSeek-V2 / V3 MLA path. Routes through the MLA-specific tensor
    /// naming (<c>q_a_proj</c> / <c>q_b_proj</c> + <c>q_a_layernorm</c> or
    /// monolithic <c>q_proj</c>; <c>kv_a_proj_with_mqa</c> with shared
    /// rope-K; <c>kv_a_layernorm</c>; <c>kv_b_proj</c>; <c>o_proj</c>).
    /// All MLA tensors are coerced to F32 via
    /// <see cref="SafetensorsTensorResolver.ResolveLinearAsF32"/> — the
    /// scalar MLA kernel consumes F32 row-major throughout.
    /// </summary>
    private static AttentionLayerTensors LoadMla(
        ISafetensorsTensorSource file, ModelConfig config, int layerIdx, List<nint> owned)
    {
        var mlaCfg = config.MlaConfig
                     ?? throw new InvalidOperationException(
                         "AttentionTensorLoader.LoadMla called but ModelConfig.MlaConfig is null.");

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

        // Q path: LoRA-factored (V2 full, V3) or monolithic (V2-Lite). The
        // kernel decides which path to take based on qLoraRank; we pass
        // zero pointers for the unused set.
        nint qAProj = 0, qBProj = 0, qProj = 0;
        float[]? qALayernorm = null;
        if (qLoraRank > 0)
        {
            (qAProj, _, int qAm, int qAk) = SafetensorsTensorResolver.ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_a_proj.weight", owned);
            SafetensorsTensorResolver.ValidateProjectionShape(qAm, qAk, qLoraRank, hiddenSize,
                $"{prefix}.self_attn.q_a_proj.weight");
            qALayernorm = SafetensorsTensorResolver.ResolveNorm(
                file, $"{prefix}.self_attn.q_a_layernorm.weight", qLoraRank);
            (qBProj, _, int qBm, int qBk) = SafetensorsTensorResolver.ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_b_proj.weight", owned);
            SafetensorsTensorResolver.ValidateProjectionShape(qBm, qBk, qTotalOut, qLoraRank,
                $"{prefix}.self_attn.q_b_proj.weight");
        }
        else
        {
            (qProj, _, int qM, int qK) = SafetensorsTensorResolver.ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_proj.weight", owned);
            SafetensorsTensorResolver.ValidateProjectionShape(qM, qK, qTotalOut, hiddenSize,
                $"{prefix}.self_attn.q_proj.weight");
        }

        // KV path: always LoRA-factored. kv_a_proj_with_mqa emits
        // [kvLoraRank + qkRopeHeadDim] per token — the first kvLoraRank
        // rows feed kv_a_layernorm then kv_b_proj, the last qkRopeHeadDim
        // rows are the MQA-shared rope-K. No separate LayerNorm on the
        // rope-K side.
        int kvADim = kvLoraRank + qkRope;
        (nint kvAProj, _, int kvaM, int kvaK) = SafetensorsTensorResolver.ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", owned);
        SafetensorsTensorResolver.ValidateProjectionShape(kvaM, kvaK, kvADim, hiddenSize,
            $"{prefix}.self_attn.kv_a_proj_with_mqa.weight");
        float[] kvALayernorm = SafetensorsTensorResolver.ResolveNorm(
            file, $"{prefix}.self_attn.kv_a_layernorm.weight", kvLoraRank);
        (nint kvBProj, _, int kvbM, int kvbK) = SafetensorsTensorResolver.ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_b_proj.weight", owned);
        SafetensorsTensorResolver.ValidateProjectionShape(kvbM, kvbK, kvBOut, kvLoraRank,
            $"{prefix}.self_attn.kv_b_proj.weight");

        // Output projection: hidden ← n_heads * v_head_dim. Kept in the
        // existing O slot (not MLA-specific) because the forward path
        // still applies bias (if any) through the same AddBias logic.
        var (oPtr, oQt, oM, oK) = SafetensorsTensorResolver.ResolveLinearAsF32(
            file, $"{prefix}.self_attn.o_proj.weight", owned);
        SafetensorsTensorResolver.ValidateProjectionShape(oM, oK, hiddenSize, oInputDim,
            $"{prefix}.self_attn.o_proj.weight");
        float[]? oBias = SafetensorsTensorResolver.ResolveOptionalBias(
            file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        var mla = new MlaLayerWeights(
            qAProj: qAProj, qALayernormWeight: qALayernorm, qBProj: qBProj, qProj: qProj,
            kvAProjWithMqa: kvAProj, kvALayernormWeight: kvALayernorm, kvBProj: kvBProj,
            numHeads: numHeads,
            qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
            qLoraRank: qLoraRank, kvLoraRank: kvLoraRank);

        return new AttentionLayerTensors(
            QWeight: 0, QQuantType: QuantizationType.F32, QOutputDim: 0, QInputDim: 0,
            KWeight: 0, KQuantType: QuantizationType.F32, KOutputDim: 0, KInputDim: 0,
            VWeight: 0, VQuantType: QuantizationType.F32, VOutputDim: 0, VInputDim: 0,
            OWeight: oPtr, OQuantType: oQt, OOutputDim: oM, OInputDim: oK,
            QBias: null, KBias: null, VBias: null, OBias: oBias,
            QNormWeight: null, KNormWeight: null,
            Mla: mla);
    }
}
