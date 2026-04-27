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
/// <b>Two-mode storage.</b> Q8_0 source projections are kept on device as raw 34-byte
/// blocks when the contraction dim is a multiple of 32 (the Q8_0 group size) — the
/// forward pass dispatches them through the existing
/// <see cref="DotLLM.Vulkan.Kernels.MatMulQ8_0Kernel"/> /
/// <see cref="DotLLM.Vulkan.Kernels.MatMulQ8_0GemmKernel"/>, mirroring the
/// <see cref="VulkanWeights"/> path. F16 and Q4_K/Q5_K/Q6_K/Q5_0 are still dequantised
/// to F32 at upload (no kernel in tree); F32 sources are uploaded verbatim. Norm weights
/// and the small per-head SSM vectors (<c>ssm_a</c>, <c>ssm_d</c>, <c>ssm_dt.bias</c>,
/// <c>ssm_norm.weight</c>, <c>ssm_conv1d.weight</c>, <c>ssm_conv1d.bias</c>) are always
/// uploaded as F32 device buffers — they are already dequantised to <c>float[]</c> on
/// the CPU side at GGUF load time.
/// </para>
/// <para>
/// <b>Token embedding.</b> Dequantised to F32 at upload regardless of source quant type
/// — the embedding gather uses <c>vkCmdCopyBuffer</c> with row-major byte offsets, which
/// only works with a contiguous F32 layout. This matches the
/// <see cref="VulkanWeights"/> convention.
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

        public readonly QuantizationType InDeviceQuantType;
        public readonly QuantizationType OutDeviceQuantType;

        public readonly int InInputDim;
        public readonly int InOutputDim;
        public readonly int OutInputDim;
        public readonly int OutOutputDim;

        public SsmLayerBuffers(
            VulkanDevice.Buffer inWeight, QuantizationType inDeviceQt, int inInputDim, int inOutputDim,
            VulkanDevice.Buffer conv1dWeight, VulkanDevice.Buffer conv1dBias,
            VulkanDevice.Buffer a, VulkanDevice.Buffer d, VulkanDevice.Buffer dtBias,
            VulkanDevice.Buffer normWeight,
            VulkanDevice.Buffer outWeight, QuantizationType outDeviceQt, int outInputDim, int outOutputDim)
        {
            InWeight = inWeight; InDeviceQuantType = inDeviceQt;
            InInputDim = inInputDim; InOutputDim = inOutputDim;
            Conv1dWeight = conv1dWeight;
            Conv1dBias = conv1dBias;
            A = a; D = d; DtBias = dtBias;
            NormWeight = normWeight;
            OutWeight = outWeight; OutDeviceQuantType = outDeviceQt;
            OutInputDim = outInputDim; OutOutputDim = outOutputDim;
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
        public readonly QuantizationType QDeviceQuantType;
        public readonly QuantizationType KDeviceQuantType;
        public readonly QuantizationType VDeviceQuantType;
        public readonly QuantizationType ODeviceQuantType;
        public readonly int QOutputDim, QInputDim;
        public readonly int KOutputDim, KInputDim;
        public readonly int VOutputDim, VInputDim;
        public readonly int OOutputDim, OInputDim;
        public readonly int NumKvHeads;

        public AttentionLayerBuffers(
            VulkanDevice.Buffer q, QuantizationType qQt, int qM, int qK,
            VulkanDevice.Buffer k, QuantizationType kQt, int kM, int kK,
            VulkanDevice.Buffer v, QuantizationType vQt, int vM, int vK,
            VulkanDevice.Buffer o, QuantizationType oQt, int oM, int oK,
            int numKvHeads)
        {
            Q = q; QDeviceQuantType = qQt; QOutputDim = qM; QInputDim = qK;
            K = k; KDeviceQuantType = kQt; KOutputDim = kM; KInputDim = kK;
            V = v; VDeviceQuantType = vQt; VOutputDim = vM; VInputDim = vK;
            O = o; ODeviceQuantType = oQt; OOutputDim = oM; OInputDim = oK;
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
        public readonly QuantizationType UpDeviceQuantType;
        public readonly QuantizationType DownDeviceQuantType;
        public readonly int UpOutputDim, UpInputDim;
        public readonly int DownOutputDim, DownInputDim;

        public FfnLayerBuffers(
            VulkanDevice.Buffer up, QuantizationType upQt, int upM, int upK,
            VulkanDevice.Buffer down, QuantizationType downQt, int downM, int downK)
        {
            Up = up; UpDeviceQuantType = upQt; UpOutputDim = upM; UpInputDim = upK;
            Down = down; DownDeviceQuantType = downQt; DownOutputDim = downM; DownInputDim = downK;
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
    public QuantizationType OutputDeviceQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    public long AllocatedBytes { get; private set; }

    private VulkanNemotronHWeights(
        LayerBuffers[] layers,
        VulkanDevice.Buffer tokenEmbedding, int vocabSize, int hiddenSize,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, QuantizationType outputDeviceQt,
        int outputOutputDim, int outputInputDim,
        long allocatedBytes)
    {
        _layers = layers;
        TokenEmbedding = tokenEmbedding;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputDeviceQuantType = outputDeviceQt;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads a NemotronH model's weights to the Vulkan device. Q8_0 projection sources
    /// are kept on device as raw Q8_0 blocks when the contraction dim is a multiple of 32;
    /// every other source dtype is dequantised to F32 on upload (see class remarks).
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
        // on-device byte form — Q8_0 blocks for kept-Q8_0, F32 elsewhere).
        long stagingBytes = ComputeMaxStagingBytes(config, cpuLayers, outputNormWeight,
            outputOutputDim, outputInputDim, outputQuantType);
        using var staging = device.Allocate(stagingBytes);

        // Token embedding [vocab, hidden] — always dequantised on upload (the embedding
        // gather uses byte-offset vkCmdCopyBuffer and needs a contiguous F32 layout).
        var tokenEmbed = UploadProjectionMatrix(device, staging,
            tokenEmbedWeight, tokenEmbedQuantType, config.VocabSize, config.HiddenSize,
            forceF32: true,
            out _, out long tokenEmbedBytes);
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
            forceF32: false,
            out var outputDeviceQt, out long outputBytes);
        totalBytes += outputBytes;

        return new VulkanNemotronHWeights(layers,
            tokenEmbed, config.VocabSize, config.HiddenSize,
            outputNorm, outputW, outputDeviceQt, outputOutputDim, outputInputDim,
            totalBytes);
    }

    /// <summary>True iff a Q8_0 source projection can be kept on device as raw Q8_0
    /// blocks — gated on the input dim being a multiple of the Q8_0 group size (32).</summary>
    private static bool KeepQ8OnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q8_0 && (inputDim % 32) == 0;

    /// <summary>True iff a Q4_K source projection can be kept on device as raw Q4_K
    /// super-blocks — gated on the input dim being a multiple of the Q4_K super-block
    /// size (256). Phase 1 of K-quant work; Q5_K / Q6_K follow-up tickets.</summary>
    private static bool KeepQ4KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q4_K && (inputDim % 256) == 0;

    /// <summary>True iff the source projection is a quantised format with a Vulkan
    /// kernel (Q8_0 or Q4_K) AND the contraction axis is aligned to that format's
    /// group size — i.e. the raw blocks can stay on device verbatim.</summary>
    private static bool KeepQuantOnDevice(QuantizationType qt, int inputDim)
        => KeepQ8OnDevice(qt, inputDim) || KeepQ4KOnDevice(qt, inputDim);

    /// <summary>Returns the on-device byte size for one projection matrix in its
    /// chosen storage form — Q8_0 / Q4_K row-stride bytes when kept quantised,
    /// otherwise F32.</summary>
    private static long ProjectionUploadBytes(int outputDim, int inputDim, QuantizationType qt)
    {
        if (KeepQ8OnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q8_0) * outputDim;
        if (KeepQ4KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q4_K) * outputDim;
        return (long)outputDim * inputDim * sizeof(float);
    }

    private static long ComputeMaxStagingBytes(
        ModelConfig config, NemotronHLayerWeights[] cpuLayers, float[] outputNormWeight,
        int outputOutputDim, int outputInputDim, QuantizationType outputQt)
    {
        long max = 0;
        // Token embedding is always dequantised → F32 staging size.
        max = Math.Max(max, (long)config.VocabSize * config.HiddenSize * sizeof(float));
        max = Math.Max(max, ProjectionUploadBytes(outputOutputDim, outputInputDim, outputQt));
        max = Math.Max(max, (long)outputNormWeight.Length * sizeof(float));

        for (int i = 0; i < cpuLayers.Length; i++)
        {
            var lw = cpuLayers[i];
            max = Math.Max(max, (long)lw.AttnNormWeight.Length * sizeof(float));

            if (lw.Ssm is { } s)
            {
                max = Math.Max(max, ProjectionUploadBytes(s.InOutputDim, s.InInputDim, s.InQuantType));
                max = Math.Max(max, ProjectionUploadBytes(s.OutOutputDim, s.OutInputDim, s.OutQuantType));
                max = Math.Max(max, (long)s.Conv1dWeight.Length * sizeof(float));
                max = Math.Max(max, (long)s.Conv1dBias.Length * sizeof(float));
                max = Math.Max(max, (long)s.A.Length * sizeof(float));
                max = Math.Max(max, (long)s.D.Length * sizeof(float));
                max = Math.Max(max, (long)s.DtBias.Length * sizeof(float));
                max = Math.Max(max, (long)s.NormWeight.Length * sizeof(float));
            }
            if (lw.Attention is { } a)
            {
                max = Math.Max(max, ProjectionUploadBytes(a.QOutputDim, a.QInputDim, a.QQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.KOutputDim, a.KInputDim, a.KQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.VOutputDim, a.VInputDim, a.VQuantType));
                max = Math.Max(max, ProjectionUploadBytes(a.OOutputDim, a.OInputDim, a.OQuantType));
            }
            if (lw.Ffn is { } f)
            {
                max = Math.Max(max, ProjectionUploadBytes(f.UpOutputDim, f.UpInputDim, f.UpQuantType));
                max = Math.Max(max, ProjectionUploadBytes(f.DownOutputDim, f.DownInputDim, f.DownQuantType));
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
            forceF32: false,
            out var inDeviceQt, out long inBytes);
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
            forceF32: false,
            out var outDeviceQt, out long outBytes);
        uploadedBytes += outBytes;

        return new SsmLayerBuffers(
            inBuf, inDeviceQt, ssmW.InInputDim, ssmW.InOutputDim,
            conv1dWeight, conv1dBias,
            a, d, dtBias,
            norm,
            outBuf, outDeviceQt, ssmW.OutInputDim, ssmW.OutOutputDim);
    }

    private static AttentionLayerBuffers UploadAttentionLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        NemotronHAttentionWeights attnW, out long uploadedBytes)
    {
        uploadedBytes = 0;
        var q = UploadProjectionMatrix(device, staging,
            attnW.QWeight, attnW.QQuantType, attnW.QOutputDim, attnW.QInputDim,
            forceF32: false, out var qQt, out long qBytes);
        var k = UploadProjectionMatrix(device, staging,
            attnW.KWeight, attnW.KQuantType, attnW.KOutputDim, attnW.KInputDim,
            forceF32: false, out var kQt, out long kBytes);
        var v = UploadProjectionMatrix(device, staging,
            attnW.VWeight, attnW.VQuantType, attnW.VOutputDim, attnW.VInputDim,
            forceF32: false, out var vQt, out long vBytes);
        var o = UploadProjectionMatrix(device, staging,
            attnW.OWeight, attnW.OQuantType, attnW.OOutputDim, attnW.OInputDim,
            forceF32: false, out var oQt, out long oBytes);
        uploadedBytes += qBytes + kBytes + vBytes + oBytes;

        return new AttentionLayerBuffers(
            q, qQt, attnW.QOutputDim, attnW.QInputDim,
            k, kQt, attnW.KOutputDim, attnW.KInputDim,
            v, vQt, attnW.VOutputDim, attnW.VInputDim,
            o, oQt, attnW.OOutputDim, attnW.OInputDim,
            attnW.NumKvHeads);
    }

    private static FfnLayerBuffers UploadFfnLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        NemotronHFfnWeights ffnW, out long uploadedBytes)
    {
        var up = UploadProjectionMatrix(device, staging,
            ffnW.UpWeight, ffnW.UpQuantType, ffnW.UpOutputDim, ffnW.UpInputDim,
            forceF32: false, out var upQt, out long upBytes);
        var down = UploadProjectionMatrix(device, staging,
            ffnW.DownWeight, ffnW.DownQuantType, ffnW.DownOutputDim, ffnW.DownInputDim,
            forceF32: false, out var downQt, out long downBytes);
        uploadedBytes = upBytes + downBytes;
        return new FfnLayerBuffers(
            up, upQt, ffnW.UpOutputDim, ffnW.UpInputDim,
            down, downQt, ffnW.DownOutputDim, ffnW.DownInputDim);
    }

    /// <summary>
    /// Uploads one projection matrix from an unmanaged source pointer. When the source
    /// is Q8_0 and <paramref name="forceF32"/> is false and the contraction dim is a
    /// multiple of 32, the raw Q8_0 block bytes are copied verbatim and
    /// <paramref name="deviceQuantType"/> is <see cref="QuantizationType.Q8_0"/>; the
    /// caller must dispatch the matmul through a Q8_0 kernel. Otherwise the source is
    /// dequantised to F32 before upload and <paramref name="deviceQuantType"/> is
    /// <see cref="QuantizationType.F32"/>.
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadProjectionMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool forceF32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;

        if (!forceF32 && KeepQuantOnDevice(qt, inputDim))
        {
            // Raw quant-block upload — same on-device byte layout as VulkanWeights so
            // the matching matmul kernel (Q8_0 or Q4_K) reads it directly.
            QuantizationType keepQt = KeepQ8OnDevice(qt, inputDim)
                ? QuantizationType.Q8_0
                : QuantizationType.Q4_K;
            long rowBytes = Dequantize.RowByteSize(inputDim, keepQt);
            long bytes = rowBytes * outputDim;

            var buf = device.AllocateDeviceLocal(bytes);
            VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
                .ThrowOnError("vkMapMemory VulkanNemotronHWeights.UploadProjectionMatrix staging (raw quant)");
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

        // F32 dequantised upload — covers F32 source, F16 source, every K-quant /
        // Q5_0 (no Vulkan kernel for those yet), and the forceF32-token-embedding
        // path that always lands here.
        long fpBytes = elems * sizeof(float);
        var fpBuf = device.AllocateDeviceLocal(fpBytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)fpBytes, 0, out nint fpMapped)
            .ThrowOnError("vkMapMemory VulkanNemotronHWeights.UploadProjectionMatrix staging");
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
