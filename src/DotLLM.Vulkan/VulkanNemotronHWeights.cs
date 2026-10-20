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
/// <see cref="VulkanWeights"/> path. Q4_K / Q5_K / Q6_K source projections are kept
/// on device when the contraction dim is a multiple of 256 (their super-block size)
/// and dispatched via the matching K-quant kernels. Q5_0 is kept packed (#568); F16 is still dequantised
/// to F32 at upload; F32 sources are uploaded verbatim. Norm weights
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

        /// <summary>Per-tensor ssm_in / ssm_out output scales (#1038; 1 when the GGUF has none).</summary>
        public readonly float InScale;
        public readonly float OutScale;

        public SsmLayerBuffers(
            VulkanDevice.Buffer inWeight, QuantizationType inDeviceQt, int inInputDim, int inOutputDim,
            VulkanDevice.Buffer conv1dWeight, VulkanDevice.Buffer conv1dBias,
            VulkanDevice.Buffer a, VulkanDevice.Buffer d, VulkanDevice.Buffer dtBias,
            VulkanDevice.Buffer normWeight,
            VulkanDevice.Buffer outWeight, QuantizationType outDeviceQt, int outInputDim, int outOutputDim,
            float inScale = 1.0f, float outScale = 1.0f)
        {
            InScale = inScale; OutScale = outScale;
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

    /// <summary>
    /// nemotron_h_moe routed-MoE FFN layer (#375/#402): sigmoid router + selection bias, ungated relu-squared routed expert banks
    /// (up, down), and one ungated relu-squared shared expert. Expert banks are resident in a quantized form that has an
    /// indexed matmul kernel: Q8_0 and Q5_1 as stored, Q5_0 repacked to Q5_1 at upload (bit-exact, <see cref="VulkanWeights.ConvertQ5_0BlocksToQ5_1"/>).
    /// </summary>
    internal readonly struct MoeLayerBuffers
    {
        public readonly VulkanDevice.Buffer Router;          // [numExperts, hidden] in RouterDeviceQuantType
        public readonly QuantizationType RouterDeviceQuantType;
        public readonly VulkanDevice.Buffer SelectionBias;   // [numExperts] F32 (exp_probs_b; selection only)
        public readonly VulkanDevice.Buffer UpBank;          // [numExperts, moeInter, hidden]
        public readonly QuantizationType UpBankQuantType;
        public readonly VulkanDevice.Buffer DownBank;        // [numExperts, hidden, moeInter]
        public readonly QuantizationType DownBankQuantType;
        public readonly VulkanDevice.Buffer SharedUp;        // [sharedInter, hidden] dense
        public readonly QuantizationType SharedUpDeviceQuantType;
        public readonly VulkanDevice.Buffer SharedDown;      // [hidden, sharedInter] dense
        public readonly QuantizationType SharedDownDeviceQuantType;
        public readonly int NumExperts, NumExpertsPerTok, MoeIntermediateSize, SharedIntermediateSize;
        public readonly bool NormalizeWeights;
        public readonly float WeightsScale;

        /// <summary>LatentMoE width (#1045, Nemotron 3.5 Bolt), or 0 when the routed experts read the full hidden state.</summary>
        public readonly int LatentSize;
        /// <summary>ffn_latent_down [latent, hidden] dense projection; null when <see cref="LatentSize"/> is 0.</summary>
        public readonly VulkanDevice.Buffer? LatentDown;
        public readonly QuantizationType LatentDownDeviceQuantType;
        /// <summary>ffn_latent_up [hidden, latent] dense projection; null when <see cref="LatentSize"/> is 0.</summary>
        public readonly VulkanDevice.Buffer? LatentUp;
        public readonly QuantizationType LatentUpDeviceQuantType;

        /// <summary>Whether the routed experts run in the LatentMoE bottleneck.</summary>
        public bool HasLatent => LatentSize > 0;

        public MoeLayerBuffers(
            VulkanDevice.Buffer router, QuantizationType routerQt, VulkanDevice.Buffer selectionBias,
            VulkanDevice.Buffer upBank, QuantizationType upQt, VulkanDevice.Buffer downBank, QuantizationType downQt,
            VulkanDevice.Buffer sharedUp, QuantizationType sharedUpQt, VulkanDevice.Buffer sharedDown, QuantizationType sharedDownQt,
            int numExperts, int topK, int moeInter, int sharedInter, bool normalize, float scale,
            int latentSize = 0,
            VulkanDevice.Buffer? latentDown = null, QuantizationType latentDownQt = QuantizationType.F32,
            VulkanDevice.Buffer? latentUp = null, QuantizationType latentUpQt = QuantizationType.F32)
        {
            Router = router; RouterDeviceQuantType = routerQt; SelectionBias = selectionBias;
            UpBank = upBank; UpBankQuantType = upQt; DownBank = downBank; DownBankQuantType = downQt;
            SharedUp = sharedUp; SharedUpDeviceQuantType = sharedUpQt; SharedDown = sharedDown; SharedDownDeviceQuantType = sharedDownQt;
            NumExperts = numExperts; NumExpertsPerTok = topK; MoeIntermediateSize = moeInter; SharedIntermediateSize = sharedInter;
            NormalizeWeights = normalize; WeightsScale = scale;
            LatentSize = latentSize;
            LatentDown = latentDown; LatentDownDeviceQuantType = latentDownQt;
            LatentUp = latentUp; LatentUpDeviceQuantType = latentUpQt;
        }

        public void Dispose()
        {
            Router.Dispose(); SelectionBias.Dispose();
            UpBank.Dispose(); DownBank.Dispose();
            SharedUp.Dispose(); SharedDown.Dispose();
            LatentDown?.Dispose(); LatentUp?.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;
        public readonly HybridLayerKind Kind;
        public readonly SsmLayerBuffers? Ssm;
        public readonly AttentionLayerBuffers? Attention;
        public readonly FfnLayerBuffers? Ffn;
        /// <summary>Non-null for an Ffn-kind layer that carries a router (nemotron_h_moe); <see cref="Ffn"/> is null then.</summary>
        public readonly MoeLayerBuffers? Moe;

        public LayerBuffers(VulkanDevice.Buffer attnNormWeight, HybridLayerKind kind,
            SsmLayerBuffers? ssm, AttentionLayerBuffers? attention, FfnLayerBuffers? ffn,
            MoeLayerBuffers? moe = null)
        {
            AttnNormWeight = attnNormWeight;
            Kind = kind;
            Ssm = ssm;
            Attention = attention;
            Ffn = ffn;
            Moe = moe;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Ssm?.Dispose();
            Attention?.Dispose();
            Ffn?.Dispose();
            Moe?.Dispose();
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
        VulkanWeightImportPolicy.Reset();
        using var staging = VulkanStagingBuffer.Create(device, stagingBytes);

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
                case HybridLayerKind.Ffn when lw.Moe is { } moeW:
                    {
                        var moe = UploadMoeLayer(device, staging, moeW, config.HiddenSize, out long moeBytes);
                        totalBytes += moeBytes;
                        layers[i] = new LayerBuffers(attnNorm, HybridLayerKind.Ffn, null, null, null, moe);
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

    /// <summary>True iff a Q2_K source projection can be kept on device as raw Q2_K
    /// super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepQ2KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q2_K && (inputDim % 256) == 0;

    /// <summary>True iff a Q3_K source projection can be kept on device as raw Q3_K
    /// super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepQ3KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q3_K && (inputDim % 256) == 0;

    /// <summary>True iff a Q5_0 source projection can be kept on device as raw 22-byte Q5_0
    /// blocks — gated on the contraction dim being a multiple of the legacy group size (32).
    /// Uses the same F32-in GEMV/GEMM kernels as the dense path (#344, #568).</summary>
    private static bool KeepQ5_0OnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q5_0 && (inputDim % 32) == 0 && !ExpandQ5_0ToF32;

    // DOTLLM_VK_NEMOTRONH_Q5_0_F32=1 restores the pre-#568 behaviour (expand Q5_0 to F32 at upload) for A/B.
    private static readonly bool ExpandQ5_0ToF32 =
        Environment.GetEnvironmentVariable("DOTLLM_VK_NEMOTRONH_Q5_0_F32") == "1";

    /// <summary>True iff a Q4_K source projection can be kept on device as raw Q4_K
    /// super-blocks — gated on the input dim being a multiple of the Q4_K super-block
    /// size (256). Phase 1 of K-quant work.</summary>
    private static bool KeepQ4KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q4_K && (inputDim % 256) == 0;

    /// <summary>True iff a Q5_K source projection can be kept on device as raw Q5_K
    /// super-blocks — gated on the input dim being a multiple of the Q5_K super-block
    /// size (256). Phase 1 sibling of <see cref="KeepQ4KOnDevice"/>.</summary>
    private static bool KeepQ5KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q5_K && (inputDim % 256) == 0;

    /// <summary>True iff a Q6_K source projection can be kept on device as raw Q6_K
    /// super-blocks — gated on the input dim being a multiple of the Q6_K super-block
    /// size (256). Phase 1 sibling of <see cref="KeepQ4KOnDevice"/> completing the
    /// K-quant matmul kernel coverage (Q4_K / Q5_K / Q6_K).</summary>
    private static bool KeepQ6KOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.Q6_K && (inputDim % 256) == 0;

    /// <summary>True iff an IQ4_NL source projection can be kept on device as raw
    /// 18-byte blocks — gated on the input dim being a multiple of 32.</summary>
    /// <summary>True iff an NVFP4 projection can stay packed on device (36-byte blocks of 64, #1038).</summary>
    private static bool KeepNvfp4OnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.NVFP4 && (inputDim % QuantFormat.Nvfp4GroupSize) == 0;

    /// <summary>
    /// Refuses a per-tensor output scale on a projection whose device storage has no kernel that applies
    /// it — only the NVFP4 matmul takes one (#1038).
    /// </summary>
    internal static void ThrowIfScaleUnsupported(string what, float scale, QuantizationType deviceQt)
    {
        if (scale != 1.0f && deviceQt != QuantizationType.NVFP4)
            throw new NotSupportedException(
                $"Nemotron-H {what} carries a per-tensor scale ({scale}) but is stored as {deviceQt} on Vulkan; " +
                "only the NVFP4 matmul applies X.scale (#1038). Use --device cpu.");
    }

    private static bool KeepIq4NlOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ4_NL && (inputDim % 32) == 0;

    /// <summary>True iff an IQ4_XS source projection can be kept on device as raw
    /// 136-byte super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepIq4XsOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ4_XS && (inputDim % 256) == 0;
    /// <summary>True iff an IQ2_XXS source projection (66 bytes / 256 elements).</summary>
    private static bool KeepIq2XxsOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ2_XXS && (inputDim % 256) == 0;

    /// <summary>True iff an IQ2_XS source projection (74 bytes / 256 elements).</summary>
    private static bool KeepIq2XsOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ2_XS && (inputDim % 256) == 0;

    /// <summary>True iff an IQ2_S source projection (82 bytes / 256 elements).
    /// Also covers MOSTLY_IQ2_M file-type tensors.</summary>
    private static bool KeepIq2SOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ2_S && (inputDim % 256) == 0;

    /// <summary>True iff an IQ3_XXS source projection can be kept on device as raw
    /// 98-byte super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepIq3XxsOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ3_XXS && (inputDim % 256) == 0;

    /// <summary>True iff an IQ3_S source projection can be kept on device as raw
    /// 110-byte super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepIq3SOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ3_S && (inputDim % 256) == 0;

    /// <summary>True iff an IQ1_S source projection can be kept on device as raw
    /// 50-byte super-blocks — gated on the input dim being a multiple of 256.</summary>
    private static bool KeepIq1SOnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.IQ1_S && (inputDim % 256) == 0;

    /// <summary>True iff an F16 source projection can be kept on device as raw 2-byte
    /// F16 elements — gated on the contraction dim being a multiple of 2. Phase 8 of
    /// the native matmul work.</summary>
    private static bool KeepF16OnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.F16 && (inputDim & 1) == 0;

    /// <summary>True iff a BF16 source projection can be kept on device as raw 2-byte
    /// BF16 elements — gated on the contraction dim being a multiple of 2. Phase 8
    /// sibling of <see cref="KeepF16OnDevice"/>.</summary>
    private static bool KeepBf16OnDevice(QuantizationType qt, int inputDim)
        => qt == QuantizationType.BF16 && (inputDim & 1) == 0;

    /// <summary>True iff the source projection is a supported on-device dtype (Q8_0, the
    /// K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, the IQ family IQ1_S/IQ2_*/IQ3_*/IQ4_*, F16 or
    /// BF16) AND the contraction axis is aligned to that format's group size — i.e. the raw
    /// bytes can stay on device verbatim. The disjunction below is the authority; keep this
    /// list in step with it. Narrower than the dense path's
    /// <c>VulkanWeights.DeviceQuantTypeFor</c>, which also keeps I2_S and PQ2_0.</summary>
    private static bool KeepQuantOnDevice(QuantizationType qt, int inputDim)
        => KeepQ8OnDevice(qt, inputDim)
        || KeepQ5_0OnDevice(qt, inputDim)
        || KeepQ2KOnDevice(qt, inputDim)
        || KeepQ3KOnDevice(qt, inputDim)
        || KeepQ4KOnDevice(qt, inputDim)
        || KeepQ5KOnDevice(qt, inputDim)
        || KeepQ6KOnDevice(qt, inputDim)
        || KeepNvfp4OnDevice(qt, inputDim)
        || KeepIq4NlOnDevice(qt, inputDim)
        || KeepIq4XsOnDevice(qt, inputDim)
        || KeepIq2XxsOnDevice(qt, inputDim)
        || KeepIq2XsOnDevice(qt, inputDim)
        || KeepIq2SOnDevice(qt, inputDim)
        || KeepIq3XxsOnDevice(qt, inputDim)
        || KeepIq3SOnDevice(qt, inputDim)
        || KeepIq1SOnDevice(qt, inputDim)
        || KeepF16OnDevice(qt, inputDim)
        || KeepBf16OnDevice(qt, inputDim);

    /// <summary>Returns the storage quant type the projection will land at on device
    /// for the given source/contraction-axis pair. F32 means it'll be dequantised at
    /// upload.</summary>
    private static QuantizationType DeviceQuantTypeFor(QuantizationType qt, int inputDim)
    {
        if (KeepQ8OnDevice(qt, inputDim)) return QuantizationType.Q8_0;
        if (KeepQ5_0OnDevice(qt, inputDim)) return QuantizationType.Q5_0;
        if (KeepQ2KOnDevice(qt, inputDim)) return QuantizationType.Q2_K;
        if (KeepQ3KOnDevice(qt, inputDim)) return QuantizationType.Q3_K;
        if (KeepQ4KOnDevice(qt, inputDim)) return QuantizationType.Q4_K;
        if (KeepQ5KOnDevice(qt, inputDim)) return QuantizationType.Q5_K;
        if (KeepQ6KOnDevice(qt, inputDim)) return QuantizationType.Q6_K;
        if (KeepNvfp4OnDevice(qt, inputDim)) return QuantizationType.NVFP4;
        if (KeepIq4NlOnDevice(qt, inputDim)) return QuantizationType.IQ4_NL;
        if (KeepIq4XsOnDevice(qt, inputDim)) return QuantizationType.IQ4_XS;
        if (KeepIq2XxsOnDevice(qt, inputDim)) return QuantizationType.IQ2_XXS;
        if (KeepIq2XsOnDevice(qt, inputDim)) return QuantizationType.IQ2_XS;
        if (KeepIq2SOnDevice(qt, inputDim)) return QuantizationType.IQ2_S;
        if (KeepIq3XxsOnDevice(qt, inputDim)) return QuantizationType.IQ3_XXS;
        if (KeepIq3SOnDevice(qt, inputDim)) return QuantizationType.IQ3_S;
        if (KeepIq1SOnDevice(qt, inputDim)) return QuantizationType.IQ1_S;
        if (KeepF16OnDevice(qt, inputDim)) return QuantizationType.F16;
        if (KeepBf16OnDevice(qt, inputDim)) return QuantizationType.BF16;
        return QuantizationType.F32;
    }

    /// <summary>Returns the on-device byte size for one projection matrix in its
    /// chosen storage form — Q-format / F16 / BF16 row-stride bytes when kept native,
    /// otherwise F32.</summary>
    private static long ProjectionUploadBytes(int outputDim, int inputDim, QuantizationType qt)
    {
        if (KeepQ8OnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q8_0) * outputDim;
        if (KeepQ5_0OnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q5_0) * outputDim;
        if (KeepQ2KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q2_K) * outputDim;
        if (KeepQ3KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q3_K) * outputDim;
        if (KeepQ4KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q4_K) * outputDim;
        if (KeepQ5KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q5_K) * outputDim;
        if (KeepQ6KOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q6_K) * outputDim;
        if (KeepIq4NlOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ4_NL) * outputDim;
        if (KeepIq4XsOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ4_XS) * outputDim;
        if (KeepIq2XxsOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_XXS) * outputDim;
        if (KeepIq2XsOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_XS) * outputDim;
        if (KeepIq2SOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_S) * outputDim;
        if (KeepIq3XxsOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ3_XXS) * outputDim;
        if (KeepIq3SOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ3_S) * outputDim;
        if (KeepIq1SOnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ1_S) * outputDim;
        if (KeepF16OnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.F16) * outputDim;
        if (KeepBf16OnDevice(qt, inputDim))
            return Dequantize.RowByteSize(inputDim, QuantizationType.BF16) * outputDim;
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
            if (lw.Moe is { } m)
            {
                int h = config.HiddenSize;
                max = Math.Max(max, ProjectionUploadBytes(m.NumExperts, h, m.GateInpQuantType));
                max = Math.Max(max, (long)m.SelectionBias.Length * sizeof(float));
                max = Math.Max(max, ProjectionUploadBytes(m.SharedIntermediateSize, h, m.UpShexpQuantType));
                max = Math.Max(max, ProjectionUploadBytes(h, m.SharedIntermediateSize, m.DownShexpQuantType));
                if (m.HasLatent)
                {
                    max = Math.Max(max, ProjectionUploadBytes(m.LatentSize, h, m.LatentDownQuantType));
                    max = Math.Max(max, ProjectionUploadBytes(h, m.LatentSize, m.LatentUpQuantType));
                }
                // Expert banks stream row by row (UploadRows), so one destination row is the only hard requirement.
                max = Math.Max(max, Dequantize.RowByteSize(Math.Max(h, m.MoeIntermediateSize), QuantizationType.Q5_1));
            }
        }
        return Math.Max(max, 64);
    }

    private static SsmLayerBuffers UploadSsmLayer(
        VulkanDevice device, VulkanStagingBuffer staging,
        NemotronHSsmWeights ssmW, out long uploadedBytes)
    {
        // #1038: only the NVFP4 kernel multiplies by a per-tensor X.scale; a scale on any other device
        // storage would run unscaled. Checked before anything is allocated.
        ThrowIfScaleUnsupported("ssm_in", ssmW.InScale, DeviceQuantTypeFor(ssmW.InQuantType, ssmW.InInputDim));
        ThrowIfScaleUnsupported("ssm_out", ssmW.OutScale, DeviceQuantTypeFor(ssmW.OutQuantType, ssmW.OutInputDim));
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
            outBuf, outDeviceQt, ssmW.OutInputDim, ssmW.OutOutputDim,
            ssmW.InScale, ssmW.OutScale);
    }

    private static AttentionLayerBuffers UploadAttentionLayer(
        VulkanDevice device, VulkanStagingBuffer staging,
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

    /// <summary>
    /// Uploads one nemotron_h_moe MoE layer. The routed banks stay quantized (the F32 expansion of the published files is
    /// ~100 GB), in a form an indexed matmul kernel reads: Q8_0, Q5_1 and the K-quants Q4_K / Q5_K / Q6_K as stored (#1045, the
    /// Nemotron 3.5 Bolt AD-Q4_K_M / AD-Q6_K banks), Q5_0 exactly repacked to Q5_1.
    /// Any other expert quantization throws instead of silently widening (IQ4_NL / Q4_0 / MXFP4 banks: see #407, #402).
    /// </summary>
    private static VulkanDevice.Buffer UploadExpertBank(
        VulkanDevice device, VulkanStagingBuffer staging,
        nint srcPtr, QuantizationType srcQt, int numExperts, int mDim, int kDim, string what,
        out QuantizationType deviceQt, out long bytes)
    {
        long rows = (long)numExperts * mDim;
        long srcRowBytes = Dequantize.RowByteSize(kDim, srcQt);
        if (kDim % 32 != 0)
            throw new NotSupportedException($"nemotron_h_moe {what}: K={kDim} is not a multiple of 32.");

        switch (srcQt)
        {
            case QuantizationType.Q4_K:
            case QuantizationType.Q5_K:
            case QuantizationType.Q6_K:
                // K-quant super-blocks span 256 elements; every indexed K-quant kernel (MMVQ and F32-in) assumes whole blocks per row.
                if (kDim % 256 != 0)
                    throw new NotSupportedException($"nemotron_h_moe {what} is {srcQt} with K={kDim}, not a multiple of the 256-element K-quant block.");
                goto case QuantizationType.Q8_0;
            case QuantizationType.Q8_0:
            case QuantizationType.Q5_1:
            {
                deviceQt = srcQt;
                bytes = srcRowBytes * rows;
                if (VulkanWeightImportPolicy.TryImport(device, srcPtr, bytes, out var imported))
                    return imported!;
                var buf = device.AllocateDeviceLocal(bytes);
                staging.UploadBytes(srcPtr, bytes, buf);
                VulkanWeightImportPolicy.NoteStaged(srcPtr, bytes);
                return buf;
            }
            case QuantizationType.Q5_0:
            {
                deviceQt = QuantizationType.Q5_1;
                long dstRowBytes = Dequantize.RowByteSize(kDim, QuantizationType.Q5_1);
                bytes = dstRowBytes * rows;
                var buf = device.AllocateDeviceLocal(bytes);
                int blocksPerRow = kDim / 32;
                staging.UploadRows(rows, dstRowBytes, buf, 0, (chunkPtr, firstRow, rowCount) =>
                {
                    nint srcFirst = srcPtr + (nint)(firstRow * srcRowBytes);
                    System.Threading.Tasks.Parallel.For(0, rowCount, r =>
                        VulkanWeights.ConvertQ5_0BlocksToQ5_1(srcFirst + (nint)(r * srcRowBytes), chunkPtr + (nint)(r * dstRowBytes), blocksPerRow));
                });
                VulkanWeightImportPolicy.NoteStaged(srcPtr, rows * srcRowBytes, "repacked_q5_0_to_q5_1");
                return buf;
            }
            default:
                throw new NotSupportedException(
                    $"nemotron_h_moe {what} is {srcQt}: the Vulkan expert-indexed kernels cover Q8_0, Q5_1, Q5_0 (repacked to Q5_1) and Q4_K / Q5_K / Q6_K banks only. " +
                    "Widening the banks to F32 would need ~100 GB. IQ4_NL / Q4_0 / MXFP4 expert banks are tracked in #407 and #402; use --device cpu.");
        }
    }

    private static MoeLayerBuffers UploadMoeLayer(
        VulkanDevice device, VulkanStagingBuffer staging, NemotronHMoeWeights moe, int hidden, out long uploadedBytes)
    {
        int numE = moe.NumExperts;
        int inter = moe.MoeIntermediateSize;
        int sharedInter = moe.SharedIntermediateSize;
        // LatentMoE (#1045): the routed banks are [latent -> inter] / [inter -> latent]; the router and shared expert keep the hidden.
        int expertDim = moe.HasLatent ? moe.LatentSize : hidden;

        var router = UploadProjectionMatrix(device, staging, moe.GateInpWeight, moe.GateInpQuantType, numE, hidden,
            forceF32: false, out var routerQt, out long routerBytes);
        var bias = UploadFloatArray(device, staging, moe.SelectionBias);
        var up = UploadExpertBank(device, staging, moe.UpExpsWeight, moe.UpExpsQuantType, numE, inter, expertDim,
            "ffn_up_exps", out var upQt, out long upBytes);
        var down = UploadExpertBank(device, staging, moe.DownExpsWeight, moe.DownExpsQuantType, numE, expertDim, inter,
            "ffn_down_exps", out var downQt, out long downBytes);
        var sharedUp = UploadProjectionMatrix(device, staging, moe.UpShexpWeight, moe.UpShexpQuantType, sharedInter, hidden,
            forceF32: false, out var sharedUpQt, out long sharedUpBytes);
        var sharedDown = UploadProjectionMatrix(device, staging, moe.DownShexpWeight, moe.DownShexpQuantType, hidden, sharedInter,
            forceF32: false, out var sharedDownQt, out long sharedDownBytes);

        VulkanDevice.Buffer? latentDown = null, latentUp = null;
        QuantizationType latentDownQt = QuantizationType.F32, latentUpQt = QuantizationType.F32;
        long latentBytes = 0;
        if (moe.HasLatent)
        {
            latentDown = UploadProjectionMatrix(device, staging, moe.LatentDownWeight, moe.LatentDownQuantType, moe.LatentSize, hidden,
                forceF32: false, out latentDownQt, out long latentDownBytes);
            latentUp = UploadProjectionMatrix(device, staging, moe.LatentUpWeight, moe.LatentUpQuantType, hidden, moe.LatentSize,
                forceF32: false, out latentUpQt, out long latentUpBytes);
            latentBytes = latentDownBytes + latentUpBytes;
        }

        uploadedBytes = routerBytes + (long)moe.SelectionBias.Length * sizeof(float) + upBytes + downBytes + sharedUpBytes + sharedDownBytes
                      + latentBytes;
        return new MoeLayerBuffers(router, routerQt, bias, up, upQt, down, downQt,
            sharedUp, sharedUpQt, sharedDown, sharedDownQt,
            numE, moe.NumExpertsPerTok, inter, sharedInter, moe.NormalizeWeights, moe.WeightsScale,
            moe.LatentSize, latentDown, latentDownQt, latentUp, latentUpQt);
    }

    private static FfnLayerBuffers UploadFfnLayer(
        VulkanDevice device, VulkanStagingBuffer staging,
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
        VulkanDevice device, VulkanStagingBuffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool forceF32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;
        long sourceBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;

        if (!forceF32 && KeepQuantOnDevice(qt, inputDim))
        {
            // Raw quant-block upload — same on-device byte layout as VulkanWeights so
            // the matching matmul kernel (Q8_0 / Q4_K / Q5_K / Q6_K) reads it directly.
            QuantizationType keepQt = DeviceQuantTypeFor(qt, inputDim);
            long rowBytes = Dequantize.RowByteSize(inputDim, keepQt);
            long bytes = rowBytes * outputDim;

            deviceQuantType = keepQt;
            uploadedBytes = bytes;

            // #508: the device image IS the source bytes, so alias the mmap'd pages
            // instead of copying them. Staging is the fallback.
            if (VulkanWeightImportPolicy.TryImport(device, srcPtr, bytes, out var imported))
                return imported!;

            var buf = device.AllocateDeviceLocal(bytes);
            staging.UploadBytes(srcPtr, bytes, buf);
            VulkanWeightImportPolicy.NoteStaged(srcPtr, bytes);
            return buf;
        }

        deviceQuantType = QuantizationType.F32;
        uploadedBytes = elems * sizeof(float);

        // An F32 source reaches the device byte-for-byte even on the "widening" arm —
        // there is nothing to widen — so it is an import candidate too (#508).
        if (qt == QuantizationType.F32
            && VulkanWeightImportPolicy.TryImport(device, srcPtr, uploadedBytes, out var importedF32))
            return importedF32!;

        VulkanWeightImportPolicy.NoteStaged(
            srcPtr, sourceBytes,
            qt == QuantizationType.F32 ? null : "not_source_bytes");

        // F32 dequantised upload — covers F32 source, F16 source, every K-quant /
        // Q5_0 (no Vulkan kernel for those yet), and the forceF32-token-embedding
        // path that always lands here.
        long fpBytes = elems * sizeof(float);
        var fpBuf = device.AllocateDeviceLocal(fpBytes);

        if (qt == QuantizationType.F32)
        {
            staging.UploadBytes(srcPtr, fpBytes, fpBuf);
        }
        else if (qt == QuantizationType.F16)
        {
            staging.UploadRows(outputDim, (long)inputDim * sizeof(float), fpBuf, 0,
                (chunkPtr, firstRow, rowCount) =>
                {
                    long elemOff = firstRow * inputDim;
                    long n = (long)rowCount * inputDim;
                    System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                        new ReadOnlySpan<Half>((void*)(srcPtr + (nint)(elemOff * 2)), checked((int)n)),
                        new Span<float>((void*)chunkPtr, checked((int)n)));
                });
        }
        else
        {
            long rowBytes = Dequantize.RowByteSize(inputDim, qt);
            staging.UploadRows(outputDim, (long)inputDim * sizeof(float), fpBuf, 0,
                (chunkPtr, firstRow, rowCount) =>
                {
                    float* dst = (float*)chunkPtr;
                    for (int i = 0; i < rowCount; i++)
                    {
                        nint rowSrc = srcPtr + (nint)((firstRow + i) * rowBytes);
                        Dequantize.ToFloat32(rowSrc, inputDim, qt,
                            new Span<float>(dst + (long)i * inputDim, inputDim));
                    }
                });
        }

        deviceQuantType = QuantizationType.F32;
        uploadedBytes = fpBytes;
        return fpBuf;
    }

    private static unsafe VulkanDevice.Buffer UploadFloatArray(
        VulkanDevice device, VulkanStagingBuffer staging, float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        staging.UploadFloats(data, buf);
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
