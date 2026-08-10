using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer device-resident weight buffers for the Qwen3HybridDense
/// (<c>qwen35</c>) model — Gated-DeltaNet / full-attention token mixing plus a
/// <b>dense</b> SwiGLU FFN.
/// </summary>
/// <remarks>
/// <para>
/// The token-mixing half is byte-for-byte the same problem as
/// <see cref="VulkanQwen3MoeHybridWeights"/> — the CPU weight records say so
/// explicitly (<see cref="Qwen3HybridDenseLayerWeights"/>: "the only structural
/// difference from the MoE hybrid is the FFN sublayer"). Rather than clone the
/// GDN / full-attention upload paths and the quant-on-device policy matrix, this
/// class reuses <see cref="VulkanQwen3MoeHybridWeights"/>'s
/// <see cref="VulkanQwen3MoeHybridWeights.GdnLayerBuffers"/> /
/// <see cref="VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers"/> and its
/// <c>internal static</c> upload helpers. Those two structs are the shared
/// representation of a hybrid token-mixing sublayer, not MoE-specific state.
/// </para>
/// <para>
/// The dense FFN weights <i>are</i> uploaded here, unlike the MoE hybrid's routed
/// expert banks (which stay on the host and stream per forward). A dense
/// gate/up/down triple is the same order of magnitude as the attention
/// projections, so there is no streaming/resident policy to make.
/// </para>
/// </remarks>
internal sealed class VulkanQwen3HybridDenseWeights : IDisposable
{
    /// <summary>Per-layer dense SwiGLU FFN buffers (<c>ffn_gate</c> / <c>ffn_up</c> / <c>ffn_down</c>).</summary>
    internal readonly struct DenseFfnLayerBuffers
    {
        public readonly VulkanDevice.Buffer GateWeight;
        public readonly VulkanDevice.Buffer UpWeight;
        public readonly VulkanDevice.Buffer DownWeight;
        public readonly QuantizationType GateDeviceQuantType;
        public readonly QuantizationType UpDeviceQuantType;
        public readonly QuantizationType DownDeviceQuantType;
        public readonly int GateInputDim, GateOutputDim;
        public readonly int UpInputDim, UpOutputDim;
        public readonly int DownInputDim, DownOutputDim;

        public DenseFfnLayerBuffers(
            VulkanDevice.Buffer gate, QuantizationType gateQt, int gateK, int gateM,
            VulkanDevice.Buffer up, QuantizationType upQt, int upK, int upM,
            VulkanDevice.Buffer down, QuantizationType downQt, int downK, int downM)
        {
            GateWeight = gate; GateDeviceQuantType = gateQt; GateInputDim = gateK; GateOutputDim = gateM;
            UpWeight = up; UpDeviceQuantType = upQt; UpInputDim = upK; UpOutputDim = upM;
            DownWeight = down; DownDeviceQuantType = downQt; DownInputDim = downK; DownOutputDim = downM;
        }

        public void Dispose()
        {
            GateWeight.Dispose();
            UpWeight.Dispose();
            DownWeight.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;
        public readonly VulkanDevice.Buffer PostAttnNormWeight;
        public readonly HybridLayerKind Kind;
        public readonly VulkanQwen3MoeHybridWeights.GdnLayerBuffers? Gdn;
        public readonly VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers? Attention;
        public readonly DenseFfnLayerBuffers Ffn;

        public LayerBuffers(
            VulkanDevice.Buffer attnNorm, VulkanDevice.Buffer postAttnNorm,
            HybridLayerKind kind,
            VulkanQwen3MoeHybridWeights.GdnLayerBuffers? gdn,
            VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers? attn,
            DenseFfnLayerBuffers ffn)
        {
            AttnNormWeight = attnNorm; PostAttnNormWeight = postAttnNorm;
            Kind = kind; Gdn = gdn; Attention = attn; Ffn = ffn;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            PostAttnNormWeight.Dispose();
            Gdn?.Dispose();
            Attention?.Dispose();
            Ffn.Dispose();
        }
    }

    private readonly LayerBuffers[] _layers;
    public LayerBuffers[] Layers => _layers;

    public VulkanDevice.Buffer TokenEmbedding { get; }
    public VulkanDevice.Buffer OutputNormWeight { get; }
    public VulkanDevice.Buffer OutputWeight { get; }
    public QuantizationType OutputDeviceQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    public long AllocatedBytes { get; }

    private VulkanQwen3HybridDenseWeights(
        LayerBuffers[] layers,
        VulkanDevice.Buffer tokenEmbedding,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, QuantizationType outputQt,
        int outputOutputDim, int outputInputDim,
        long allocatedBytes)
    {
        _layers = layers;
        TokenEmbedding = tokenEmbedding;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputDeviceQuantType = outputQt;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads every weight the dense hybrid forward needs: per-layer norms, the
    /// GDN or full-attention token-mixing sublayer, the dense FFN triple, plus the
    /// global token embedding, output norm and LM head.
    /// </summary>
    public static VulkanQwen3HybridDenseWeights Upload(
        VulkanDevice device,
        ModelConfig config,
        Qwen3HybridDenseLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        nint outputWeight, QuantizationType outputQt, int outputOutputDim, int outputInputDim)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);

        var layout = config.HybridLayout!;
        long totalBytes = 0;

        long stagingBytes = ComputeMaxStagingBytes(config, cpuLayers, outputNormWeight,
            outputOutputDim, outputInputDim, outputQt);
        using var staging = VulkanStagingBuffer.Create(device, stagingBytes);

        // Token embedding always dequantises to F32 — the embedding gather uses
        // vkCmdCopyBuffer byte offsets and needs a contiguous F32 layout.
        var tokenEmbed = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            tokenEmbedWeight, tokenEmbedQt, config.VocabSize, config.HiddenSize,
            forceF32: true, out _, out long tokenEmbedBytes);
        totalBytes += tokenEmbedBytes;

        var layers = new LayerBuffers[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var lw = cpuLayers[i];
            var attnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, lw.AttnNormWeight);
            var postAttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, lw.PostAttnNormWeight);
            totalBytes += ((long)lw.AttnNormWeight.Length + lw.PostAttnNormWeight.Length) * sizeof(float);

            var ffn = UploadDenseFfnLayer(device, staging, lw, out long ffnBytes);
            totalBytes += ffnBytes;

            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet)
            {
                var gdn = VulkanQwen3MoeHybridWeights.UploadGdnLayer(device, staging, lw.Gdn!, out long gdnBytes);
                totalBytes += gdnBytes;
                layers[i] = new LayerBuffers(attnNorm, postAttnNorm, HybridLayerKind.GatedDeltaNet, gdn, null, ffn);
            }
            else
            {
                var attn = VulkanQwen3MoeHybridWeights.UploadFullAttnLayer(device, staging, lw.FullAttn!, out long attnBytes);
                totalBytes += attnBytes;
                layers[i] = new LayerBuffers(attnNorm, postAttnNorm, HybridLayerKind.Attention, null, attn, ffn);
            }
        }

        var outputNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, outputNormWeight);
        totalBytes += (long)outputNormWeight.Length * sizeof(float);

        var outputW = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            outputWeight, outputQt, outputOutputDim, outputInputDim,
            forceF32: false, out var outputDeviceQt, out long outputBytes);
        totalBytes += outputBytes;

        return new VulkanQwen3HybridDenseWeights(layers, tokenEmbed, outputNorm,
            outputW, outputDeviceQt, outputOutputDim, outputInputDim, totalBytes);
    }

    private static DenseFfnLayerBuffers UploadDenseFfnLayer(
        VulkanDevice device, VulkanStagingBuffer staging,
        Qwen3HybridDenseLayerWeights lw, out long uploadedBytes)
    {
        var gate = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim,
            forceF32: false, out var gateQt, out long gateBytes);
        var up = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim,
            forceF32: false, out var upQt, out long upBytes);
        var down = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
            lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim,
            forceF32: false, out var downQt, out long downBytes);

        uploadedBytes = gateBytes + upBytes + downBytes;
        return new DenseFfnLayerBuffers(
            gate, gateQt, lw.GateInputDim, lw.GateOutputDim,
            up, upQt, lw.UpInputDim, lw.UpOutputDim,
            down, downQt, lw.DownInputDim, lw.DownOutputDim);
    }

    private static long ComputeMaxStagingBytes(
        ModelConfig config, Qwen3HybridDenseLayerWeights[] cpuLayers, float[] outputNormWeight,
        int outputOutputDim, int outputInputDim, QuantizationType outputQt)
    {
        long max = (long)config.VocabSize * config.HiddenSize * sizeof(float);
        max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(outputOutputDim, outputInputDim, outputQt));
        max = Math.Max(max, (long)outputNormWeight.Length * sizeof(float));

        for (int i = 0; i < cpuLayers.Length; i++)
        {
            var lw = cpuLayers[i];
            max = Math.Max(max, (long)lw.AttnNormWeight.Length * sizeof(float));
            max = Math.Max(max, (long)lw.PostAttnNormWeight.Length * sizeof(float));

            max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(lw.GateOutputDim, lw.GateInputDim, lw.GateQuantType));
            max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(lw.UpOutputDim, lw.UpInputDim, lw.UpQuantType));
            max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(lw.DownOutputDim, lw.DownInputDim, lw.DownQuantType));

            if (lw.Gdn is { } g)
            {
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(g.QkvOutputDim, g.QkvInputDim, g.QkvQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(g.GateOutputDim, g.GateInputDim, g.GateQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(g.AlphaOutputDim, g.AlphaInputDim, g.AlphaQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(g.BetaOutputDim, g.BetaInputDim, g.BetaQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(g.OutOutputDim, g.OutInputDim, g.OutQuantType));
                max = Math.Max(max, (long)g.Conv1dWeight.Length * sizeof(float));
                max = Math.Max(max, (long)g.Conv1dBias.Length * sizeof(float));
                max = Math.Max(max, (long)g.SsmNormWeight.Length * sizeof(float));
                max = Math.Max(max, (long)g.A.Length * sizeof(float));
                max = Math.Max(max, (long)g.DtBias.Length * sizeof(float));
            }
            if (lw.FullAttn is { } a)
            {
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(a.QOutputDim, a.QInputDim, a.QQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(a.KOutputDim, a.KInputDim, a.KQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(a.VOutputDim, a.VInputDim, a.VQuantType));
                max = Math.Max(max, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(a.OOutputDim, a.OInputDim, a.OQuantType));
                max = Math.Max(max, (long)a.QNormWeight.Length * sizeof(float));
                max = Math.Max(max, (long)a.KNormWeight.Length * sizeof(float));
            }
        }
        return Math.Max(max, 64);
    }

    public void Dispose()
    {
        TokenEmbedding.Dispose();
        OutputNormWeight.Dispose();
        OutputWeight.Dispose();
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Dispose();
    }
}
