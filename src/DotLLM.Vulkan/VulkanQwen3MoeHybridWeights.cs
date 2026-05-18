using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer device-resident weight buffers for the Qwen3MoeHybrid model. Mirrors
/// <see cref="VulkanNemotronHWeights"/> in policy (Q8_0 / Q4_K / Q5_K / Q6_K /
/// F16 / BF16 kept native when contraction-axis aligned; everything else
/// dequantised to F32 at upload), but with the GDN / full-attention sub-layer
/// split that Qwen3MoeHybrid imposes.
/// </summary>
/// <remarks>
/// Routed MoE expert weights are NOT uploaded here — they are too large at
/// Qwen3.6-35B-A3B scale (~120 GB if fully dequantised on-device) and are
/// streamed per-layer per forward by <see cref="VulkanQwen3MoeMoeUpload"/>.
/// The CPU-side <see cref="MoeLayerWeights"/> retain the raw quant view that
/// the streamer dequantises on demand.
/// </remarks>
internal sealed class VulkanQwen3MoeHybridWeights : IDisposable
{
    /// <summary>Per-GDN-layer weight buffers (token-mixing sub-layer).</summary>
    internal readonly struct GdnLayerBuffers
    {
        public readonly VulkanDevice.Buffer QkvWeight;
        public readonly VulkanDevice.Buffer GateWeight;
        public readonly VulkanDevice.Buffer AlphaWeight;
        public readonly VulkanDevice.Buffer BetaWeight;
        public readonly VulkanDevice.Buffer Conv1dWeight;
        public readonly VulkanDevice.Buffer Conv1dBias;
        public readonly VulkanDevice.Buffer SsmNormWeight;
        public readonly VulkanDevice.Buffer OutWeight;

        public readonly QuantizationType QkvDeviceQuantType;
        public readonly QuantizationType GateDeviceQuantType;
        public readonly QuantizationType AlphaDeviceQuantType;
        public readonly QuantizationType BetaDeviceQuantType;
        public readonly QuantizationType OutDeviceQuantType;

        public readonly int QkvInputDim, QkvOutputDim;
        public readonly int GateInputDim, GateOutputDim;
        public readonly int AlphaInputDim, AlphaOutputDim;
        public readonly int BetaInputDim, BetaOutputDim;
        public readonly int OutInputDim, OutOutputDim;

        // Device-resident A and dt_bias vectors fed to the gdn_decay_f32 shader:
        // g[t, vh] = exp(softplus(alpha[t, vh] + DtBiasDevice[vh]) * ADevice[vh]).
        // Both are tiny (nVHead floats — 64 bytes at qwen35moe) but uploaded
        // once at model load so the GDN forward stays fully GPU-resident.
        // The host-side originals are also kept for tests / fallback diagnostics.
        public readonly VulkanDevice.Buffer ADevice;
        public readonly VulkanDevice.Buffer DtBiasDevice;
        public readonly float[] AHost;
        public readonly float[] DtBiasHost;

        public GdnLayerBuffers(
            VulkanDevice.Buffer qkv, QuantizationType qkvQt, int qkvK, int qkvM,
            VulkanDevice.Buffer gate, QuantizationType gateQt, int gateK, int gateM,
            VulkanDevice.Buffer alpha, QuantizationType alphaQt, int alphaK, int alphaM,
            VulkanDevice.Buffer beta, QuantizationType betaQt, int betaK, int betaM,
            VulkanDevice.Buffer conv1dWeight, VulkanDevice.Buffer conv1dBias,
            VulkanDevice.Buffer ssmNormWeight,
            VulkanDevice.Buffer outWeight, QuantizationType outQt, int outK, int outM,
            VulkanDevice.Buffer aDevice, VulkanDevice.Buffer dtBiasDevice,
            float[] aHost, float[] dtBiasHost)
        {
            QkvWeight = qkv; QkvDeviceQuantType = qkvQt; QkvInputDim = qkvK; QkvOutputDim = qkvM;
            GateWeight = gate; GateDeviceQuantType = gateQt; GateInputDim = gateK; GateOutputDim = gateM;
            AlphaWeight = alpha; AlphaDeviceQuantType = alphaQt; AlphaInputDim = alphaK; AlphaOutputDim = alphaM;
            BetaWeight = beta; BetaDeviceQuantType = betaQt; BetaInputDim = betaK; BetaOutputDim = betaM;
            Conv1dWeight = conv1dWeight; Conv1dBias = conv1dBias;
            SsmNormWeight = ssmNormWeight;
            OutWeight = outWeight; OutDeviceQuantType = outQt; OutInputDim = outK; OutOutputDim = outM;
            ADevice = aDevice; DtBiasDevice = dtBiasDevice;
            AHost = aHost; DtBiasHost = dtBiasHost;
        }

        public void Dispose()
        {
            QkvWeight.Dispose();
            GateWeight.Dispose();
            AlphaWeight.Dispose();
            BetaWeight.Dispose();
            Conv1dWeight.Dispose();
            Conv1dBias.Dispose();
            SsmNormWeight.Dispose();
            OutWeight.Dispose();
            ADevice.Dispose();
            DtBiasDevice.Dispose();
        }
    }

    /// <summary>Per-full-attention-layer weight buffers.</summary>
    internal readonly struct FullAttnLayerBuffers
    {
        public readonly VulkanDevice.Buffer QWeight;   // fused Q+Gate
        public readonly VulkanDevice.Buffer KWeight;
        public readonly VulkanDevice.Buffer VWeight;
        public readonly VulkanDevice.Buffer OWeight;
        public readonly VulkanDevice.Buffer QNormWeight;
        public readonly VulkanDevice.Buffer KNormWeight;
        public readonly QuantizationType QDeviceQuantType;
        public readonly QuantizationType KDeviceQuantType;
        public readonly QuantizationType VDeviceQuantType;
        public readonly QuantizationType ODeviceQuantType;
        public readonly int QInputDim, QOutputDim;
        public readonly int KInputDim, KOutputDim;
        public readonly int VInputDim, VOutputDim;
        public readonly int OInputDim, OOutputDim;
        public readonly int NumKvHeads;

        public FullAttnLayerBuffers(
            VulkanDevice.Buffer q, QuantizationType qQt, int qK, int qM,
            VulkanDevice.Buffer k, QuantizationType kQt, int kK, int kM,
            VulkanDevice.Buffer v, QuantizationType vQt, int vK, int vM,
            VulkanDevice.Buffer o, QuantizationType oQt, int oK, int oM,
            VulkanDevice.Buffer qNorm, VulkanDevice.Buffer kNorm,
            int numKvHeads)
        {
            QWeight = q; QDeviceQuantType = qQt; QInputDim = qK; QOutputDim = qM;
            KWeight = k; KDeviceQuantType = kQt; KInputDim = kK; KOutputDim = kM;
            VWeight = v; VDeviceQuantType = vQt; VInputDim = vK; VOutputDim = vM;
            OWeight = o; ODeviceQuantType = oQt; OInputDim = oK; OOutputDim = oM;
            QNormWeight = qNorm; KNormWeight = kNorm;
            NumKvHeads = numKvHeads;
        }

        public void Dispose()
        {
            QWeight.Dispose(); KWeight.Dispose(); VWeight.Dispose(); OWeight.Dispose();
            QNormWeight.Dispose(); KNormWeight.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;
        public readonly VulkanDevice.Buffer PostAttnNormWeight;
        public readonly HybridLayerKind Kind;
        public readonly GdnLayerBuffers? Gdn;
        public readonly FullAttnLayerBuffers? Attention;

        public LayerBuffers(
            VulkanDevice.Buffer attnNorm, VulkanDevice.Buffer postAttnNorm,
            HybridLayerKind kind, GdnLayerBuffers? gdn, FullAttnLayerBuffers? attn)
        {
            AttnNormWeight = attnNorm; PostAttnNormWeight = postAttnNorm;
            Kind = kind; Gdn = gdn; Attention = attn;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            PostAttnNormWeight.Dispose();
            Gdn?.Dispose();
            Attention?.Dispose();
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

    private VulkanQwen3MoeHybridWeights(
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
    /// Uploads token-mixing weights (norms, GDN per-layer, full-attn per-layer)
    /// plus the global token embedding and LM head. Routed MoE expert banks are
    /// handled separately by <see cref="VulkanQwen3MoeMoeUpload"/>.
    /// </summary>
    public static VulkanQwen3MoeHybridWeights Upload(
        VulkanDevice device,
        ModelConfig config,
        Qwen3MoeLayerWeights[] cpuLayers,
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

        // Stage to the largest single matrix we will upload. Compute the bound
        // up-front so we only allocate one staging buffer (re-used across every
        // upload below — same pattern as VulkanNemotronHWeights).
        long stagingBytes = ComputeMaxStagingBytes(config, cpuLayers, outputNormWeight,
            outputOutputDim, outputInputDim, outputQt, tokenEmbedQt);
        using var staging = device.Allocate(stagingBytes);

        // Token embedding always dequantises to F32 — the embedding gather uses
        // vkCmdCopyBuffer byte offsets and needs a contiguous F32 layout.
        var tokenEmbed = UploadProjectionMatrix(device, staging,
            tokenEmbedWeight, tokenEmbedQt, config.VocabSize, config.HiddenSize,
            forceF32: true, out _, out long tokenEmbedBytes);
        totalBytes += tokenEmbedBytes;

        var layers = new LayerBuffers[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var lw = cpuLayers[i];
            var attnNorm = UploadFloatArray(device, staging, lw.AttnNormWeight);
            var postAttnNorm = UploadFloatArray(device, staging, lw.PostAttnNormWeight);
            totalBytes += ((long)lw.AttnNormWeight.Length + lw.PostAttnNormWeight.Length) * sizeof(float);

            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet)
            {
                var gdn = UploadGdnLayer(device, staging, lw.Gdn!, out long gdnBytes);
                totalBytes += gdnBytes;
                layers[i] = new LayerBuffers(attnNorm, postAttnNorm, HybridLayerKind.GatedDeltaNet, gdn, null);
            }
            else
            {
                var attn = UploadFullAttnLayer(device, staging, lw.FullAttn!, out long attnBytes);
                totalBytes += attnBytes;
                layers[i] = new LayerBuffers(attnNorm, postAttnNorm, HybridLayerKind.Attention, null, attn);
            }
        }

        var outputNorm = UploadFloatArray(device, staging, outputNormWeight);
        totalBytes += (long)outputNormWeight.Length * sizeof(float);

        var outputW = UploadProjectionMatrix(device, staging,
            outputWeight, outputQt, outputOutputDim, outputInputDim,
            forceF32: false, out var outputDeviceQt, out long outputBytes);
        totalBytes += outputBytes;

        return new VulkanQwen3MoeHybridWeights(layers, tokenEmbed, outputNorm,
            outputW, outputDeviceQt, outputOutputDim, outputInputDim, totalBytes);
    }

    private static GdnLayerBuffers UploadGdnLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        GdnTokenMixingWeights gdnW, out long uploadedBytes)
    {
        uploadedBytes = 0;
        var qkv = UploadProjectionMatrix(device, staging, gdnW.QkvWeight, gdnW.QkvQuantType,
            gdnW.QkvOutputDim, gdnW.QkvInputDim, forceF32: false, out var qkvQt, out long qkvBytes);
        var gate = UploadProjectionMatrix(device, staging, gdnW.GateWeight, gdnW.GateQuantType,
            gdnW.GateOutputDim, gdnW.GateInputDim, forceF32: false, out var gateQt, out long gateBytes);
        var alpha = UploadProjectionMatrix(device, staging, gdnW.AlphaWeight, gdnW.AlphaQuantType,
            gdnW.AlphaOutputDim, gdnW.AlphaInputDim, forceF32: false, out var alphaQt, out long alphaBytes);
        var beta = UploadProjectionMatrix(device, staging, gdnW.BetaWeight, gdnW.BetaQuantType,
            gdnW.BetaOutputDim, gdnW.BetaInputDim, forceF32: false, out var betaQt, out long betaBytes);
        var conv1dWeight = UploadFloatArray(device, staging, gdnW.Conv1dWeight);
        var conv1dBias = UploadFloatArray(device, staging, gdnW.Conv1dBias);
        var ssmNorm = UploadFloatArray(device, staging, gdnW.SsmNormWeight);
        var outBuf = UploadProjectionMatrix(device, staging, gdnW.OutWeight, gdnW.OutQuantType,
            gdnW.OutOutputDim, gdnW.OutInputDim, forceF32: false, out var outQt, out long outBytes);

        // A and dt_bias feed the gdn_decay_f32 shader — tiny (nVHead floats
        // each, 64 bytes at qwen35moe) but uploaded once at load so the GDN
        // forward never needs to push them per call.
        var aDevice = UploadFloatArray(device, staging, gdnW.A);
        var dtBiasDevice = UploadFloatArray(device, staging, gdnW.DtBias);

        uploadedBytes = qkvBytes + gateBytes + alphaBytes + betaBytes + outBytes
            + (long)gdnW.Conv1dWeight.Length * sizeof(float)
            + (long)gdnW.Conv1dBias.Length * sizeof(float)
            + (long)gdnW.SsmNormWeight.Length * sizeof(float)
            + (long)gdnW.A.Length * sizeof(float)
            + (long)gdnW.DtBias.Length * sizeof(float);

        return new GdnLayerBuffers(
            qkv, qkvQt, gdnW.QkvInputDim, gdnW.QkvOutputDim,
            gate, gateQt, gdnW.GateInputDim, gdnW.GateOutputDim,
            alpha, alphaQt, gdnW.AlphaInputDim, gdnW.AlphaOutputDim,
            beta, betaQt, gdnW.BetaInputDim, gdnW.BetaOutputDim,
            conv1dWeight, conv1dBias,
            ssmNorm,
            outBuf, outQt, gdnW.OutInputDim, gdnW.OutOutputDim,
            aDevice: aDevice,
            dtBiasDevice: dtBiasDevice,
            aHost: gdnW.A,
            dtBiasHost: gdnW.DtBias);
    }

    private static FullAttnLayerBuffers UploadFullAttnLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        Qwen3FullAttnWeights attnW, out long uploadedBytes)
    {
        var q = UploadProjectionMatrix(device, staging, attnW.QWeight, attnW.QQuantType,
            attnW.QOutputDim, attnW.QInputDim, forceF32: false, out var qQt, out long qBytes);
        var k = UploadProjectionMatrix(device, staging, attnW.KWeight, attnW.KQuantType,
            attnW.KOutputDim, attnW.KInputDim, forceF32: false, out var kQt, out long kBytes);
        var v = UploadProjectionMatrix(device, staging, attnW.VWeight, attnW.VQuantType,
            attnW.VOutputDim, attnW.VInputDim, forceF32: false, out var vQt, out long vBytes);
        var o = UploadProjectionMatrix(device, staging, attnW.OWeight, attnW.OQuantType,
            attnW.OOutputDim, attnW.OInputDim, forceF32: false, out var oQt, out long oBytes);
        var qNorm = UploadFloatArray(device, staging, attnW.QNormWeight);
        var kNorm = UploadFloatArray(device, staging, attnW.KNormWeight);
        uploadedBytes = qBytes + kBytes + vBytes + oBytes
            + (long)(attnW.QNormWeight.Length + attnW.KNormWeight.Length) * sizeof(float);

        return new FullAttnLayerBuffers(
            q, qQt, attnW.QInputDim, attnW.QOutputDim,
            k, kQt, attnW.KInputDim, attnW.KOutputDim,
            v, vQt, attnW.VInputDim, attnW.VOutputDim,
            o, oQt, attnW.OInputDim, attnW.OOutputDim,
            qNorm, kNorm,
            attnW.NumKvHeads);
    }

    // ── Quant-on-device policy ──────────────────────────────────────────────
    // Same matrix as VulkanNemotronHWeights — keep raw blocks on device when
    // the contraction axis is aligned to the format's group size; otherwise
    // dequantise to F32 at upload.

    private static bool KeepQ8(QuantizationType qt, int k) => qt == QuantizationType.Q8_0 && (k % 32) == 0;
    private static bool KeepQ4K(QuantizationType qt, int k) => qt == QuantizationType.Q4_K && (k % 256) == 0;
    private static bool KeepQ5K(QuantizationType qt, int k) => qt == QuantizationType.Q5_K && (k % 256) == 0;
    private static bool KeepQ6K(QuantizationType qt, int k) => qt == QuantizationType.Q6_K && (k % 256) == 0;
    private static bool KeepIq2Xxs(QuantizationType qt, int k) => qt == QuantizationType.IQ2_XXS && (k % 256) == 0;
    private static bool KeepIq2Xs(QuantizationType qt, int k) => qt == QuantizationType.IQ2_XS && (k % 256) == 0;
    private static bool KeepIq2S(QuantizationType qt, int k) => qt == QuantizationType.IQ2_S && (k % 256) == 0;
    private static bool KeepIq3Xxs(QuantizationType qt, int k) => qt == QuantizationType.IQ3_XXS && (k % 256) == 0;
    private static bool KeepIq3S(QuantizationType qt, int k) => qt == QuantizationType.IQ3_S && (k % 256) == 0;
    private static bool KeepF16(QuantizationType qt, int k) => qt == QuantizationType.F16 && (k & 1) == 0;
    private static bool KeepBf16(QuantizationType qt, int k) => qt == QuantizationType.BF16 && (k & 1) == 0;

    private static bool KeepNative(QuantizationType qt, int k)
        => KeepQ8(qt, k) || KeepQ4K(qt, k) || KeepQ5K(qt, k) || KeepQ6K(qt, k)
        || KeepIq2Xxs(qt, k) || KeepIq2Xs(qt, k) || KeepIq2S(qt, k)
        || KeepIq3Xxs(qt, k) || KeepIq3S(qt, k)
        || KeepF16(qt, k) || KeepBf16(qt, k);

    private static QuantizationType DeviceQuantTypeFor(QuantizationType qt, int k)
    {
        if (KeepQ8(qt, k)) return QuantizationType.Q8_0;
        if (KeepQ4K(qt, k)) return QuantizationType.Q4_K;
        if (KeepQ5K(qt, k)) return QuantizationType.Q5_K;
        if (KeepQ6K(qt, k)) return QuantizationType.Q6_K;
        if (KeepIq2Xxs(qt, k)) return QuantizationType.IQ2_XXS;
        if (KeepIq2Xs(qt, k)) return QuantizationType.IQ2_XS;
        if (KeepIq2S(qt, k)) return QuantizationType.IQ2_S;
        if (KeepIq3Xxs(qt, k)) return QuantizationType.IQ3_XXS;
        if (KeepIq3S(qt, k)) return QuantizationType.IQ3_S;
        if (KeepF16(qt, k)) return QuantizationType.F16;
        if (KeepBf16(qt, k)) return QuantizationType.BF16;
        return QuantizationType.F32;
    }

    private static long ProjectionUploadBytes(int outputDim, int inputDim, QuantizationType qt)
    {
        if (KeepNative(qt, inputDim))
        {
            QuantizationType keep = DeviceQuantTypeFor(qt, inputDim);
            return Dequantize.RowByteSize(inputDim, keep) * outputDim;
        }
        return (long)outputDim * inputDim * sizeof(float);
    }

    private static long ComputeMaxStagingBytes(
        ModelConfig config, Qwen3MoeLayerWeights[] cpuLayers, float[] outputNormWeight,
        int outputOutputDim, int outputInputDim, QuantizationType outputQt, QuantizationType tokenEmbedQt)
    {
        long max = (long)config.VocabSize * config.HiddenSize * sizeof(float);
        max = Math.Max(max, ProjectionUploadBytes(outputOutputDim, outputInputDim, outputQt));
        max = Math.Max(max, (long)outputNormWeight.Length * sizeof(float));

        for (int i = 0; i < cpuLayers.Length; i++)
        {
            var lw = cpuLayers[i];
            max = Math.Max(max, (long)lw.AttnNormWeight.Length * sizeof(float));
            max = Math.Max(max, (long)lw.PostAttnNormWeight.Length * sizeof(float));

            if (lw.Gdn is { } g)
            {
                max = Math.Max(max, ProjectionUploadBytes(g.QkvOutputDim, g.QkvInputDim, g.QkvQuantType));
                max = Math.Max(max, ProjectionUploadBytes(g.GateOutputDim, g.GateInputDim, g.GateQuantType));
                max = Math.Max(max, ProjectionUploadBytes(g.AlphaOutputDim, g.AlphaInputDim, g.AlphaQuantType));
                max = Math.Max(max, ProjectionUploadBytes(g.BetaOutputDim, g.BetaInputDim, g.BetaQuantType));
                max = Math.Max(max, ProjectionUploadBytes(g.OutOutputDim, g.OutInputDim, g.OutQuantType));
                max = Math.Max(max, (long)g.Conv1dWeight.Length * sizeof(float));
                max = Math.Max(max, (long)g.Conv1dBias.Length * sizeof(float));
                max = Math.Max(max, (long)g.SsmNormWeight.Length * sizeof(float));
                max = Math.Max(max, (long)g.A.Length * sizeof(float));
                max = Math.Max(max, (long)g.DtBias.Length * sizeof(float));
            }
            if (lw.FullAttn is { } a)
            {
                max = Math.Max(max, ProjectionUploadBytes(a.QOutputDim, a.QInputDim, a.QQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.KOutputDim, a.KInputDim, a.KQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.VOutputDim, a.VInputDim, a.VQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.OOutputDim, a.OInputDim, a.OQuantType));
                max = Math.Max(max, (long)a.QNormWeight.Length * sizeof(float));
                max = Math.Max(max, (long)a.KNormWeight.Length * sizeof(float));
            }
        }
        return Math.Max(max, 64);
    }

    /// <summary>
    /// Uploads one projection matrix from an unmanaged source pointer. Keeps
    /// the source quant bytes verbatim on device when the contraction axis
    /// alignment permits, otherwise dequantises to F32 on the host before
    /// upload.
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadProjectionMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool forceF32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;

        if (!forceF32 && KeepNative(qt, inputDim))
        {
            QuantizationType keepQt = DeviceQuantTypeFor(qt, inputDim);
            long rowBytes = Dequantize.RowByteSize(inputDim, keepQt);
            long bytes = rowBytes * outputDim;

            var buf = device.AllocateDeviceLocal(bytes);
            VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
                .ThrowOnError("vkMapMemory VulkanQwen3MoeHybridWeights.UploadProjectionMatrix raw quant");
            try
            {
                new ReadOnlySpan<byte>((void*)srcPtr, checked((int)bytes))
                    .CopyTo(new Span<byte>((void*)mapped, checked((int)bytes)));
            }
            finally
            {
                VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
            }
            device.CopyBufferSynchronous(staging, buf, (ulong)bytes);

            deviceQuantType = keepQt;
            uploadedBytes = bytes;
            return buf;
        }

        long fpBytes = elems * sizeof(float);
        var fpBuf = device.AllocateDeviceLocal(fpBytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)fpBytes, 0, out nint fpMapped)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeHybridWeights.UploadProjectionMatrix F32");
        try
        {
            float* dst = (float*)fpMapped;
            if (qt == QuantizationType.F32)
            {
                new ReadOnlySpan<float>((void*)srcPtr, checked((int)elems))
                    .CopyTo(new Span<float>(dst, checked((int)elems)));
            }
            else if (qt == QuantizationType.F16)
            {
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((void*)srcPtr, checked((int)elems)),
                    new Span<float>(dst, checked((int)elems)));
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(inputDim, qt);
                for (int row = 0; row < outputDim; row++)
                {
                    nint rowSrc = srcPtr + (nint)((long)row * rowBytes);
                    Dequantize.ToFloat32(rowSrc, inputDim, qt,
                        new Span<float>(dst + (long)row * inputDim, inputDim));
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, fpBuf, (ulong)fpBytes);

        deviceQuantType = QuantizationType.F32;
        uploadedBytes = fpBytes;
        return fpBuf;
    }

    private static unsafe VulkanDevice.Buffer UploadFloatArray(
        VulkanDevice device, VulkanDevice.Buffer staging, float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeHybridWeights.UploadFloatArray");
        try
        {
            data.AsSpan().CopyTo(new Span<float>((void*)mapped, data.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
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
