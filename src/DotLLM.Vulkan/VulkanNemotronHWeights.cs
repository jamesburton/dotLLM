using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer weight buffers on a Vulkan device for the NemotronH hybrid model. Mirrors
/// <see cref="VulkanWeights"/> but with three mutually-exclusive sub-bundles per layer
/// (SSM / Attention / FFN) gated by <see cref="HybridLayerKind"/>, plus the two model-wide
/// tensors (token embedding, output norm + LM head) that every NemotronH model carries.
/// </summary>
/// <remarks>
/// <para>
/// <b>F32-only constraint (first cut, issue #5):</b> every projection weight is dequantised
/// to F32 at upload, regardless of the source quant type. The Vulkan path consumes them via
/// <see cref="DotLLM.Vulkan.Kernels.MatMulF32Kernel"/>. The Q8_0 fast path used by
/// <see cref="VulkanWeights"/> is intentionally not wired here yet — quant support is a
/// follow-up so this commit can focus on getting the SSM + hybrid-dispatch wiring correct.
/// </para>
/// <para>
/// Norm weights and small per-head vectors (ssm_a, ssm_d, ssm_dt.bias, ssm_norm.weight,
/// ssm_conv1d.weight, ssm_conv1d.bias) are uploaded as F32 device buffers — they are
/// already dequantised to <c>float[]</c> on the CPU side at GGUF load time.
/// </para>
/// </remarks>
internal sealed class VulkanNemotronHWeights : IDisposable
{
    /// <summary>Per-layer SSM weight buffers (Mamba2 sub-layer).</summary>
    internal readonly struct SsmLayerBuffers
    {
        public readonly VulkanDevice.Buffer InWeight;       // [in_proj_dim, hidden]
        public readonly VulkanDevice.Buffer Conv1dWeight;   // [d_conv, conv_dim] (channel-major)
        public readonly VulkanDevice.Buffer Conv1dBias;     // [conv_dim]
        public readonly VulkanDevice.Buffer A;              // [n_head]
        public readonly VulkanDevice.Buffer D;              // [n_head]
        public readonly VulkanDevice.Buffer DtBias;         // [n_head]
        public readonly VulkanDevice.Buffer NormWeight;     // [d_inner] (group-norm weight, n_group * group_dim)
        public readonly VulkanDevice.Buffer OutWeight;      // [hidden, d_inner]

        public readonly int InInputDim;
        public readonly int InOutputDim;
        public readonly int OutInputDim;
        public readonly int OutOutputDim;

        public SsmLayerBuffers(
            VulkanDevice.Buffer inWeight, int inInputDim, int inOutputDim,
            VulkanDevice.Buffer conv1dWeight, VulkanDevice.Buffer conv1dBias,
            VulkanDevice.Buffer a, VulkanDevice.Buffer d, VulkanDevice.Buffer dtBias,
            VulkanDevice.Buffer normWeight,
            VulkanDevice.Buffer outWeight, int outInputDim, int outOutputDim)
        {
            InWeight = inWeight; InInputDim = inInputDim; InOutputDim = inOutputDim;
            Conv1dWeight = conv1dWeight;
            Conv1dBias = conv1dBias;
            A = a; D = d; DtBias = dtBias;
            NormWeight = normWeight;
            OutWeight = outWeight; OutInputDim = outInputDim; OutOutputDim = outOutputDim;
        }

        public void Dispose()
        {
            InWeight.Dispose();
            Conv1dWeight.Dispose();
            Conv1dBias.Dispose();
            A.Dispose(); D.Dispose(); DtBias.Dispose();
            NormWeight.Dispose();
            OutWeight.Dispose();
        }
    }

    /// <summary>Per-layer attention weight buffers (GQA sub-layer).</summary>
    internal readonly struct AttentionLayerBuffers
    {
        public readonly VulkanDevice.Buffer Q;
        public readonly VulkanDevice.Buffer K;
        public readonly VulkanDevice.Buffer V;
        public readonly VulkanDevice.Buffer O;
        public readonly int QOutputDim, QInputDim;
        public readonly int KOutputDim, KInputDim;
        public readonly int VOutputDim, VInputDim;
        public readonly int OOutputDim, OInputDim;
        public readonly int NumKvHeads;

        public AttentionLayerBuffers(
            VulkanDevice.Buffer q, int qM, int qK,
            VulkanDevice.Buffer k, int kM, int kK,
            VulkanDevice.Buffer v, int vM, int vK,
            VulkanDevice.Buffer o, int oM, int oK,
            int numKvHeads)
        {
            Q = q; QOutputDim = qM; QInputDim = qK;
            K = k; KOutputDim = kM; KInputDim = kK;
            V = v; VOutputDim = vM; VInputDim = vK;
            O = o; OOutputDim = oM; OInputDim = oK;
            NumKvHeads = numKvHeads;
        }

        public void Dispose()
        {
            Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
        }
    }

    /// <summary>Per-layer FFN weight buffers (squared-ReLU MLP, no gate).</summary>
    internal readonly struct FfnLayerBuffers
    {
        public readonly VulkanDevice.Buffer Up;
        public readonly VulkanDevice.Buffer Down;
        public readonly int UpOutputDim, UpInputDim;
        public readonly int DownOutputDim, DownInputDim;

        public FfnLayerBuffers(
            VulkanDevice.Buffer up, int upM, int upK,
            VulkanDevice.Buffer down, int downM, int downK)
        {
            Up = up; UpOutputDim = upM; UpInputDim = upK;
            Down = down; DownOutputDim = downM; DownInputDim = downK;
        }

        public void Dispose()
        {
            Up.Dispose(); Down.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;
        public readonly HybridLayerKind Kind;
        public readonly SsmLayerBuffers? Ssm;
        public readonly AttentionLayerBuffers? Attention;
        public readonly FfnLayerBuffers? Ffn;

        public LayerBuffers(VulkanDevice.Buffer attnNormWeight, HybridLayerKind kind,
            SsmLayerBuffers? ssm, AttentionLayerBuffers? attention, FfnLayerBuffers? ffn)
        {
            AttnNormWeight = attnNormWeight;
            Kind = kind;
            Ssm = ssm;
            Attention = attention;
            Ffn = ffn;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Ssm?.Dispose();
            Attention?.Dispose();
            Ffn?.Dispose();
        }
    }

    private readonly LayerBuffers[] _layers;

    public LayerBuffers[] Layers => _layers;
    public VulkanDevice.Buffer TokenEmbedding { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    public VulkanDevice.Buffer OutputNormWeight { get; }
    public VulkanDevice.Buffer OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    public long AllocatedBytes { get; private set; }

    private VulkanNemotronHWeights(
        LayerBuffers[] layers,
        VulkanDevice.Buffer tokenEmbedding, int vocabSize, int hiddenSize,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, int outputOutputDim, int outputInputDim,
        long allocatedBytes)
    {
        _layers = layers;
        TokenEmbedding = tokenEmbedding;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads a NemotronH model's weights to the Vulkan device. All projection weights are
    /// dequantised to F32 in this first cut — see class remarks.
    /// </summary>
    public static VulkanNemotronHWeights Upload(
        VulkanDevice device,
        ModelConfig config,
        NemotronHLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType,
        int outputOutputDim, int outputInputDim)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);

        var layout = config.HybridLayout!;
        long totalBytes = 0;

        // Size the staging buffer to the largest single matrix we will upload (in its
        // F32 form, since every matrix is dequantised).
        long stagingBytes = ComputeMaxStagingBytes(config, cpuLayers, outputNormWeight,
            outputOutputDim, outputInputDim);
        using var staging = device.Allocate(stagingBytes);

        // Token embedding [vocab, hidden] — dequantised on upload.
        var tokenEmbed = UploadProjectionMatrix(device, staging,
            tokenEmbedWeight, tokenEmbedQuantType, config.VocabSize, config.HiddenSize,
            out long tokenEmbedBytes);
        totalBytes += tokenEmbedBytes;

        var layers = new LayerBuffers[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var lw = cpuLayers[i];
            var attnNorm = UploadFloatArray(device, staging, lw.AttnNormWeight);
            totalBytes += (long)lw.AttnNormWeight.Length * sizeof(float);

            switch (layout.LayerKind[i])
            {
                case HybridLayerKind.Ssm:
                    {
                        var ssmW = lw.Ssm!;
                        var ssm = UploadSsmLayer(device, staging, ssmW, out long ssmBytes);
                        totalBytes += ssmBytes;
                        layers[i] = new LayerBuffers(attnNorm, HybridLayerKind.Ssm, ssm, null, null);
                        break;
                    }
                case HybridLayerKind.Attention:
                    {
                        var attnW = lw.Attention!;
                        var attn = UploadAttentionLayer(device, staging, attnW, out long attnBytes);
                        totalBytes += attnBytes;
                        layers[i] = new LayerBuffers(attnNorm, HybridLayerKind.Attention, null, attn, null);
                        break;
                    }
                case HybridLayerKind.Ffn:
                    {
                        var ffnW = lw.Ffn!;
                        var ffn = UploadFfnLayer(device, staging, ffnW, out long ffnBytes);
                        totalBytes += ffnBytes;
                        layers[i] = new LayerBuffers(attnNorm, HybridLayerKind.Ffn, null, null, ffn);
                        break;
                    }
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {layout.LayerKind[i]} at layer {i}.");
            }
        }

        var outputNorm = UploadFloatArray(device, staging, outputNormWeight);
        totalBytes += (long)outputNormWeight.Length * sizeof(float);

        var outputW = UploadProjectionMatrix(device, staging,
            outputWeight, outputQuantType, outputOutputDim, outputInputDim,
            out long outputBytes);
        totalBytes += outputBytes;

        return new VulkanNemotronHWeights(layers,
            tokenEmbed, config.VocabSize, config.HiddenSize,
            outputNorm, outputW, outputOutputDim, outputInputDim,
            totalBytes);
    }

    private static long ComputeMaxStagingBytes(
        ModelConfig config, NemotronHLayerWeights[] cpuLayers, float[] outputNormWeight,
        int outputOutputDim, int outputInputDim)
    {
        long max = 0;
        max = Math.Max(max, (long)config.VocabSize * config.HiddenSize * sizeof(float));
        max = Math.Max(max, (long)outputOutputDim * outputInputDim * sizeof(float));
        max = Math.Max(max, (long)outputNormWeight.Length * sizeof(float));

        for (int i = 0; i < cpuLayers.Length; i++)
        {
            var lw = cpuLayers[i];
            max = Math.Max(max, (long)lw.AttnNormWeight.Length * sizeof(float));

            if (lw.Ssm is { } s)
            {
                max = Math.Max(max, (long)s.InOutputDim * s.InInputDim * sizeof(float));
                max = Math.Max(max, (long)s.OutOutputDim * s.OutInputDim * sizeof(float));
                max = Math.Max(max, (long)s.Conv1dWeight.Length * sizeof(float));
                max = Math.Max(max, (long)s.Conv1dBias.Length * sizeof(float));
                max = Math.Max(max, (long)s.A.Length * sizeof(float));
                max = Math.Max(max, (long)s.D.Length * sizeof(float));
                max = Math.Max(max, (long)s.DtBias.Length * sizeof(float));
                max = Math.Max(max, (long)s.NormWeight.Length * sizeof(float));
            }
            if (lw.Attention is { } a)
            {
                max = Math.Max(max, (long)a.QOutputDim * a.QInputDim * sizeof(float));
                max = Math.Max(max, (long)a.KOutputDim * a.KInputDim * sizeof(float));
                max = Math.Max(max, (long)a.VOutputDim * a.VInputDim * sizeof(float));
                max = Math.Max(max, (long)a.OOutputDim * a.OInputDim * sizeof(float));
            }
            if (lw.Ffn is { } f)
            {
                max = Math.Max(max, (long)f.UpOutputDim * f.UpInputDim * sizeof(float));
                max = Math.Max(max, (long)f.DownOutputDim * f.DownInputDim * sizeof(float));
            }
        }
        return Math.Max(max, 64);
    }

    private static SsmLayerBuffers UploadSsmLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        NemotronHSsmWeights ssmW, out long uploadedBytes)
    {
        uploadedBytes = 0;
        var inBuf = UploadProjectionMatrix(device, staging,
            ssmW.InWeight, ssmW.InQuantType, ssmW.InOutputDim, ssmW.InInputDim,
            out long inBytes);
        uploadedBytes += inBytes;

        var conv1dWeight = UploadFloatArray(device, staging, ssmW.Conv1dWeight);
        uploadedBytes += (long)ssmW.Conv1dWeight.Length * sizeof(float);
        var conv1dBias = UploadFloatArray(device, staging, ssmW.Conv1dBias);
        uploadedBytes += (long)ssmW.Conv1dBias.Length * sizeof(float);
        var a = UploadFloatArray(device, staging, ssmW.A);
        uploadedBytes += (long)ssmW.A.Length * sizeof(float);
        var d = UploadFloatArray(device, staging, ssmW.D);
        uploadedBytes += (long)ssmW.D.Length * sizeof(float);
        var dtBias = UploadFloatArray(device, staging, ssmW.DtBias);
        uploadedBytes += (long)ssmW.DtBias.Length * sizeof(float);
        var norm = UploadFloatArray(device, staging, ssmW.NormWeight);
        uploadedBytes += (long)ssmW.NormWeight.Length * sizeof(float);

        var outBuf = UploadProjectionMatrix(device, staging,
            ssmW.OutWeight, ssmW.OutQuantType, ssmW.OutOutputDim, ssmW.OutInputDim,
            out long outBytes);
        uploadedBytes += outBytes;

        return new SsmLayerBuffers(
            inBuf, ssmW.InInputDim, ssmW.InOutputDim,
            conv1dWeight, conv1dBias,
            a, d, dtBias,
            norm,
            outBuf, ssmW.OutInputDim, ssmW.OutOutputDim);
    }

    private static AttentionLayerBuffers UploadAttentionLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        NemotronHAttentionWeights attnW, out long uploadedBytes)
    {
        uploadedBytes = 0;
        var q = UploadProjectionMatrix(device, staging,
            attnW.QWeight, attnW.QQuantType, attnW.QOutputDim, attnW.QInputDim, out long qBytes);
        var k = UploadProjectionMatrix(device, staging,
            attnW.KWeight, attnW.KQuantType, attnW.KOutputDim, attnW.KInputDim, out long kBytes);
        var v = UploadProjectionMatrix(device, staging,
            attnW.VWeight, attnW.VQuantType, attnW.VOutputDim, attnW.VInputDim, out long vBytes);
        var o = UploadProjectionMatrix(device, staging,
            attnW.OWeight, attnW.OQuantType, attnW.OOutputDim, attnW.OInputDim, out long oBytes);
        uploadedBytes += qBytes + kBytes + vBytes + oBytes;

        return new AttentionLayerBuffers(
            q, attnW.QOutputDim, attnW.QInputDim,
            k, attnW.KOutputDim, attnW.KInputDim,
            v, attnW.VOutputDim, attnW.VInputDim,
            o, attnW.OOutputDim, attnW.OInputDim,
            attnW.NumKvHeads);
    }

    private static FfnLayerBuffers UploadFfnLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        NemotronHFfnWeights ffnW, out long uploadedBytes)
    {
        var up = UploadProjectionMatrix(device, staging,
            ffnW.UpWeight, ffnW.UpQuantType, ffnW.UpOutputDim, ffnW.UpInputDim, out long upBytes);
        var down = UploadProjectionMatrix(device, staging,
            ffnW.DownWeight, ffnW.DownQuantType, ffnW.DownOutputDim, ffnW.DownInputDim, out long downBytes);
        uploadedBytes = upBytes + downBytes;
        return new FfnLayerBuffers(
            up, ffnW.UpOutputDim, ffnW.UpInputDim,
            down, ffnW.DownOutputDim, ffnW.DownInputDim);
    }

    /// <summary>
    /// Uploads one projection matrix from an unmanaged source pointer, dequantising to F32
    /// when the source quant type is not already F32.
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadProjectionMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;
        long bytes = elems * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanNemotronHWeights.UploadProjectionMatrix staging");
        try
        {
            float* dst = (float*)mapped;
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

        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        uploadedBytes = bytes;
        return buf;
    }

    private static unsafe VulkanDevice.Buffer UploadFloatArray(
        VulkanDevice device, VulkanDevice.Buffer staging, float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanNemotronHWeights.UploadFloatArray staging");
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
