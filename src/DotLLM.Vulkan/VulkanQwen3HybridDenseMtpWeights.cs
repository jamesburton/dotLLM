using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident weights for the trailing Multi-Token Prediction (MTP / "NextN") head of a
/// <c>qwen35</c> checkpoint (issue #435) — the Vulkan mirror of
/// <see cref="MtpHeadWeights"/> and of CUDA's <c>CudaMtpHeadWeights</c>.
/// </summary>
/// <remarks>
/// <para>
/// The MTP block is structurally a normal full-attention decoder layer plus four <c>nextn.*</c>
/// tensors, so the attention / dense-FFN halves reuse the very same upload helpers the trunk's own
/// layers go through. Nothing here is MTP-specific except <c>eh_proj</c>, the two extra norms, and
/// the three optional head-local tensors.
/// </para>
/// <para>
/// <b>The head-local tensors are optional, and which way they fall matters for correctness on a
/// Hadamard-folded checkpoint.</b> When <c>nextn.embed_tokens</c> / <c>nextn.shared_head_head</c>
/// are absent (this is the case for <c>Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0</c>), the MTP head falls
/// back to the trunk's own <c>token_embd.weight</c> and <c>output.weight</c> — both of which are in
/// the checkpoint's <c>prism.hadamard.*</c> fold set, so the fallback path must apply the same
/// rotations the trunk's own embedding lookup and lm_head do. A head-local tensor is not folded and
/// takes the unrotated activation. <see cref="UsesTrunkEmbedding"/> and
/// <see cref="UsesTrunkLmHead"/> are what the forward path branches on.
/// </para>
/// </remarks>
internal sealed class VulkanQwen3HybridDenseMtpWeights : IDisposable
{
    /// <summary><c>attn_norm.weight</c> of the MTP block.</summary>
    public required VulkanDevice.Buffer AttnNormWeight { get; init; }

    /// <summary><c>post_attention_norm.weight</c> of the MTP block.</summary>
    public required VulkanDevice.Buffer PostAttnNormWeight { get; init; }

    /// <summary>The MTP block's gated full-attention projections.</summary>
    public required VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers Attention { get; init; }

    /// <summary>The MTP block's dense SwiGLU FFN triple.</summary>
    public required VulkanQwen3HybridDenseWeights.DenseFfnLayerBuffers Ffn { get; init; }

    /// <summary><c>nextn.eh_proj.weight</c> — maps <c>concat(enorm(embed), hnorm(hidden))</c> back to hiddenSize.</summary>
    public required VulkanDevice.Buffer EhProjWeight { get; init; }
    public required QuantizationType EhProjDeviceQuantType { get; init; }
    public required int EhProjInputDim { get; init; }
    public required int EhProjOutputDim { get; init; }

    /// <summary><c>nextn.enorm.weight</c> — RMSNorm weight for the predicted-from token's embedding.</summary>
    public required VulkanDevice.Buffer EnormWeight { get; init; }

    /// <summary><c>nextn.hnorm.weight</c> — RMSNorm weight for the incoming trunk hidden state.</summary>
    public required VulkanDevice.Buffer HnormWeight { get; init; }

    /// <summary>Optional head-local <c>nextn.embed_tokens.weight</c>, widened to F32. Null ⇒ use the trunk table.</summary>
    public VulkanDevice.Buffer? EmbedTokensWeight { get; init; }

    /// <summary>Optional head-local <c>nextn.shared_head_head.weight</c>. Null ⇒ use the trunk LM head.</summary>
    public VulkanDevice.Buffer? SharedHeadHeadWeight { get; init; }
    public QuantizationType SharedHeadHeadDeviceQuantType { get; init; }
    public int SharedHeadHeadInputDim { get; init; }
    public int SharedHeadHeadOutputDim { get; init; }

    /// <summary>Optional head-local <c>nextn.shared_head_norm.weight</c>. Null ⇒ use the trunk <c>output_norm</c>.</summary>
    public VulkanDevice.Buffer? SharedHeadNormWeight { get; init; }

    /// <summary>Total device bytes uploaded for the head.</summary>
    public required long AllocatedBytes { get; init; }

    /// <summary>True when the head embeds through the trunk's (possibly Hadamard-latent) <c>token_embd.weight</c>.</summary>
    public bool UsesTrunkEmbedding => EmbedTokensWeight is null;

    /// <summary>True when the head projects logits through the trunk's (possibly folded) <c>output.weight</c>.</summary>
    public bool UsesTrunkLmHead => SharedHeadHeadWeight is null;

    /// <summary>
    /// Uploads every MTP-head tensor. <paramref name="cpuHead"/> is the CPU loader's already-parsed
    /// head record — the same source of truth the trunk's own layers are uploaded from.
    /// </summary>
    public static VulkanQwen3HybridDenseMtpWeights Upload(
        VulkanDevice device, ModelConfig config, MtpHeadWeights cpuHead)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuHead);

        var layer = cpuHead.Layer;
        var attnW = layer.FullAttn
            ?? throw new InvalidDataException(
                "MTP head has no full-attention weights — the MTP block must be a full-attention layer.");

        long stagingBytes = ComputeMaxStagingBytes(config, cpuHead, attnW);
        using var staging = VulkanStagingBuffer.Create(device, stagingBytes);

        long total = 0;

        var attnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, layer.AttnNormWeight);
        var postAttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, layer.PostAttnNormWeight);
        total += ((long)layer.AttnNormWeight.Length + layer.PostAttnNormWeight.Length) * sizeof(float);

        var attn = VulkanQwen3MoeHybridWeights.UploadFullAttnLayer(device, staging, attnW, out long attnBytes);
        total += attnBytes;

        var ffn = VulkanQwen3HybridDenseWeights.UploadDenseFfnLayer(device, staging, layer, out long ffnBytes);
        total += ffnBytes;

        var ehProj = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            cpuHead.EhProjWeight, cpuHead.EhProjQuantType,
            cpuHead.EhProjOutputDim, cpuHead.EhProjInputDim,
            forceF32: false, out var ehProjQt, out long ehProjBytes);
        total += ehProjBytes;

        var enorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cpuHead.EnormWeight);
        var hnorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cpuHead.HnormWeight);
        total += ((long)cpuHead.EnormWeight.Length + cpuHead.HnormWeight.Length) * sizeof(float);

        // Head-local embedding table, when the checkpoint ships one. Widened to F32 so the lookup
        // is the same vkCmdCopyBuffer row copy the trunk's unpacked path uses; a head-local table
        // at Bonsai's vocabulary would be far too large for that, but no released checkpoint ships
        // one — Bonsai 2 MTP falls back to the trunk table, which stays packed.
        VulkanDevice.Buffer? embedTokens = null;
        if (cpuHead.EmbedTokensWeight is { } embedPtr)
        {
            embedTokens = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
                embedPtr, cpuHead.EmbedTokensQuantType, config.VocabSize, config.HiddenSize,
                forceF32: true, out _, out long embedBytes);
            total += embedBytes;
        }

        VulkanDevice.Buffer? sharedHead = null;
        QuantizationType sharedHeadQt = default;
        if (cpuHead.SharedHeadHeadWeight is { } sharedHeadPtr)
        {
            sharedHead = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
                sharedHeadPtr, cpuHead.SharedHeadHeadQuantType, config.VocabSize, config.HiddenSize,
                forceF32: false, out sharedHeadQt, out long sharedHeadBytes);
            total += sharedHeadBytes;
        }

        VulkanDevice.Buffer? sharedHeadNorm = null;
        if (cpuHead.SharedHeadNormWeight is { } shn)
        {
            sharedHeadNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, shn);
            total += (long)shn.Length * sizeof(float);
        }

        return new VulkanQwen3HybridDenseMtpWeights
        {
            AttnNormWeight = attnNorm,
            PostAttnNormWeight = postAttnNorm,
            Attention = attn,
            Ffn = ffn,
            EhProjWeight = ehProj,
            EhProjDeviceQuantType = ehProjQt,
            EhProjInputDim = cpuHead.EhProjInputDim,
            EhProjOutputDim = cpuHead.EhProjOutputDim,
            EnormWeight = enorm,
            HnormWeight = hnorm,
            EmbedTokensWeight = embedTokens,
            SharedHeadHeadWeight = sharedHead,
            SharedHeadHeadDeviceQuantType = sharedHeadQt,
            SharedHeadHeadInputDim = config.HiddenSize,
            SharedHeadHeadOutputDim = config.VocabSize,
            SharedHeadNormWeight = sharedHeadNorm,
            AllocatedBytes = total,
        };
    }

    private static long ComputeMaxStagingBytes(
        ModelConfig config, MtpHeadWeights head, Qwen3FullAttnWeights attnW)
    {
        var layer = head.Layer;
        long max = 64;
        max = Math.Max(max, (long)layer.AttnNormWeight.Length * sizeof(float));
        max = Math.Max(max, (long)layer.PostAttnNormWeight.Length * sizeof(float));
        max = Math.Max(max, (long)head.EnormWeight.Length * sizeof(float));
        max = Math.Max(max, (long)head.HnormWeight.Length * sizeof(float));
        if (head.SharedHeadNormWeight is { } shn)
            max = Math.Max(max, (long)shn.Length * sizeof(float));

        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(attnW.QOutputDim, attnW.QInputDim, attnW.QQuantType));
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(attnW.KOutputDim, attnW.KInputDim, attnW.KQuantType));
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(attnW.VOutputDim, attnW.VInputDim, attnW.VQuantType));
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(attnW.OOutputDim, attnW.OInputDim, attnW.OQuantType));
        max = Math.Max(max, (long)attnW.QNormWeight.Length * sizeof(float));
        max = Math.Max(max, (long)attnW.KNormWeight.Length * sizeof(float));

        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(layer.GateOutputDim, layer.GateInputDim, layer.GateQuantType));
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(layer.UpOutputDim, layer.UpInputDim, layer.UpQuantType));
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(layer.DownOutputDim, layer.DownInputDim, layer.DownQuantType));

        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(head.EhProjOutputDim, head.EhProjInputDim, head.EhProjQuantType));

        // forceF32 widening for a head-local embed table needs a full F32-sized staging window.
        if (head.EmbedTokensWeight is not null)
            max = Math.Max(max, (long)config.VocabSize * config.HiddenSize * sizeof(float));
        if (head.SharedHeadHeadWeight is not null)
            max = Math.Max(max,
                VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(config.VocabSize, config.HiddenSize, head.SharedHeadHeadQuantType));

        return max;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        AttnNormWeight.Dispose();
        PostAttnNormWeight.Dispose();
        Attention.Dispose();
        Ffn.Dispose();
        EhProjWeight.Dispose();
        EnormWeight.Dispose();
        HnormWeight.Dispose();
        EmbedTokensWeight?.Dispose();
        SharedHeadHeadWeight?.Dispose();
        SharedHeadNormWeight?.Dispose();
    }
}
