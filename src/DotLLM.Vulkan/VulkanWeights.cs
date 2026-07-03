using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer weight buffers on a Vulkan device. Mirrors
/// <c>DotLLM.Cuda.CudaWeights</c> but with a two-mode storage model:
/// Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, F16, and BF16 matrices are kept on device in
/// their source byte layout when <c>dequantToFp32=false</c> (default) and the
/// relevant Vulkan kernel supports the contraction shape. Unsupported shapes
/// fall back to dequantised F32 upload. Bias and norm weights are always FP32
/// device buffers (tiny, kernels consume FP32).
/// </summary>
internal sealed class VulkanWeights : IDisposable
{
    /// <summary>
    /// Per-layer device-resident MoE (Mixtral / Qwen-MoE) weight bundle.
    /// Per-expert weights are <i>packed</i> into one contiguous device bank
    /// per projection so either the indexed-matmul path or the grouped F16
    /// coopmat path can address any expert via a single descriptor binding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Three banks per layer for the routed top-k experts:
    /// <list type="bullet">
    ///   <item><c>W1Bank</c> (<i>gate_proj</i>): <c>[numExperts, intermediate, hidden]</c></item>
    ///   <item><c>W2Bank</c> (<i>down_proj</i>): <c>[numExperts, hidden, intermediate]</c></item>
    ///   <item><c>W3Bank</c> (<i>up_proj</i>):   <c>[numExperts, intermediate, hidden]</c></item>
    /// </list>
    /// Plus the router gate <c>[numExperts, hidden]</c>.
    /// </para>
    /// <para>
    /// Shared experts (DeepSeek-V2/V3 ungated branch) are stored as <i>separate
    /// per-expert buffers</i>, not packed into a single bank. The per-shared-
    /// expert matmuls go through the standard <c>matmul_f32</c> kernel which
    /// reads its weight buffer from offset 0 — packing all shared experts into
    /// one bank would require either a per-expert sub-buffer (the kernel API
    /// takes a whole <c>VulkanDevice.Buffer</c>, not a sub-range) or a new
    /// weight-offset push constant on the matmul kernel. Shared experts are
    /// few (typically 1..2) and small, so per-expert buffers keep the wiring
    /// simple while costing one extra buffer per shared expert per layer.
    /// Qwen1.5-MoE's per-token sigmoid gate is uploaded as a one-row matrix
    /// and consumed by the fused sigmoid-gated add kernel when present.
    /// </para>
    /// </remarks>
    internal readonly struct MoeLayerBuffers
    {
        public readonly VulkanDevice.Buffer Gate;       // [numExperts, hidden]
        public readonly VulkanDevice.Buffer W1Bank;     // [numExperts, intermediate, hidden]
        public readonly VulkanDevice.Buffer W2Bank;     // [numExperts, hidden, intermediate]
        public readonly VulkanDevice.Buffer W3Bank;     // [numExperts, intermediate, hidden]
        public readonly QuantizationType W1DeviceQuantType;
        public readonly QuantizationType W2DeviceQuantType;
        public readonly QuantizationType W3DeviceQuantType;

        // Device-side storage type for the router gate. Q8_0 when the source carried a
        // Q8_0 overlay (and hidden % 32 == 0); F32 otherwise — same two-mode policy as
        // VulkanWeights.UploadMatrix, dispatched by VulkanTransformerModel.RecordMatmul.
        public readonly QuantizationType GateDeviceQuantType;

        // Shared-expert weights (DeepSeek-V2/V3 ungated convention). Each
        // array has one entry per shared expert; null when no shared experts
        // are present on this layer. Stored as separate buffers (NOT packed)
        // because the matmul kernel reads its weight buffer from offset 0.
        public readonly VulkanDevice.Buffer[]? SharedW1;     // [sharedIntermediate, hidden]
        public readonly VulkanDevice.Buffer[]? SharedW2;     // [hidden, sharedIntermediate]
        public readonly VulkanDevice.Buffer[]? SharedW3;     // [sharedIntermediate, hidden]

        // Device-side storage type for the per-shared-expert projections. All three
        // (SharedW1/W2/W3) share one quant type — the upload either keeps everything
        // Q8_0 (when the overlay is set and contraction axes are multiples of 32) or
        // dequantises everything to F32. F32 when no shared expert is present.
        public readonly QuantizationType SharedW1DeviceQuantType;
        public readonly QuantizationType SharedW2DeviceQuantType;
        public readonly QuantizationType SharedW3DeviceQuantType;

        // Optional Qwen1.5-MoE sigmoid gate weight for the shared-expert
        // branch (HF: <c>mlp.shared_expert_gate.weight</c>, [hidden]). Stored
        // as a [1, hidden] buffer so the existing F32 matmul kernel (M=1) can
        // produce per-token gate logits in one dispatch. Null on
        // DeepSeek-V2/V3 (ungated shared experts) and on Mixtral-style
        // routed-only layers.
        public readonly VulkanDevice.Buffer? SharedExpertGate;
        public readonly QuantizationType SharedExpertGateDeviceQuantType;

        public readonly int NumExperts;
        public readonly int NumExpertsPerTok;
        public readonly int HiddenSize;
        public readonly int IntermediateSize;
        public readonly bool NormTopKProb;
        public readonly int SharedIntermediateSize;
        public readonly int NumSharedExperts;

        public MoeLayerBuffers(
            VulkanDevice.Buffer gate, QuantizationType gateDeviceQt,
            VulkanDevice.Buffer w1, VulkanDevice.Buffer w2, VulkanDevice.Buffer w3,
            QuantizationType w1DeviceQt, QuantizationType w2DeviceQt, QuantizationType w3DeviceQt,
            int numExperts, int numExpertsPerTok,
            int hiddenSize, int intermediateSize, bool normTopKProb,
            VulkanDevice.Buffer[]? sharedW1, VulkanDevice.Buffer[]? sharedW2, VulkanDevice.Buffer[]? sharedW3,
            QuantizationType sharedW1DeviceQt, QuantizationType sharedW2DeviceQt, QuantizationType sharedW3DeviceQt,
            int sharedIntermediateSize, int numSharedExperts,
            VulkanDevice.Buffer? sharedExpertGate, QuantizationType sharedExpertGateDeviceQt)
        {
            Gate = gate;
            GateDeviceQuantType = gateDeviceQt;
            W1Bank = w1;
            W2Bank = w2;
            W3Bank = w3;
            W1DeviceQuantType = w1DeviceQt;
            W2DeviceQuantType = w2DeviceQt;
            W3DeviceQuantType = w3DeviceQt;
            NumExperts = numExperts;
            NumExpertsPerTok = numExpertsPerTok;
            HiddenSize = hiddenSize;
            IntermediateSize = intermediateSize;
            NormTopKProb = normTopKProb;
            SharedW1 = sharedW1;
            SharedW2 = sharedW2;
            SharedW3 = sharedW3;
            SharedW1DeviceQuantType = sharedW1DeviceQt;
            SharedW2DeviceQuantType = sharedW2DeviceQt;
            SharedW3DeviceQuantType = sharedW3DeviceQt;
            SharedIntermediateSize = sharedIntermediateSize;
            NumSharedExperts = numSharedExperts;
            SharedExpertGate = sharedExpertGate;
            SharedExpertGateDeviceQuantType = sharedExpertGateDeviceQt;
        }

        public void Dispose()
        {
            Gate.Dispose();
            W1Bank.Dispose();
            W2Bank.Dispose();
            W3Bank.Dispose();
            if (SharedW1 is not null)
                for (int i = 0; i < SharedW1.Length; i++) SharedW1[i].Dispose();
            if (SharedW2 is not null)
                for (int i = 0; i < SharedW2.Length; i++) SharedW2[i].Dispose();
            if (SharedW3 is not null)
                for (int i = 0; i < SharedW3.Length; i++) SharedW3[i].Dispose();
            SharedExpertGate?.Dispose();
        }
    }

    /// <summary>
    /// Per-layer device-resident MLA (DeepSeek-V2/V3) weight bundle. All
    /// projection buffers are F32 row-major, mirroring
    /// <see cref="MlaLayerWeights"/>. The CPU loader stores
    /// <c>kv_a_proj_with_mqa</c> as a fused <c>[kvLoraRank + qkRopeHeadDim,
    /// hidden]</c> matrix; the Vulkan upload splits it row-wise into two
    /// device buffers so the latent path can RMSNorm just the kvLoraRank
    /// portion (the existing rmsnorm kernel doesn't support a stride),
    /// while the rope-K portion goes straight to the RoPE kernel.
    /// </summary>
    internal readonly struct MlaLayerBuffers
    {
        // Q path — exactly one of (QAProj+QBProj) / (QProj) is non-null.
        public readonly VulkanDevice.Buffer? QAProj;
        public readonly VulkanDevice.Buffer? QALayernormWeight;
        public readonly VulkanDevice.Buffer? QBProj;
        public readonly VulkanDevice.Buffer? QProj;

        // KV path — KvAProjWithMqa split row-wise on upload:
        //   KvALatentProj = first kvLoraRank rows  (→ kv latent bottleneck)
        //   KvAKPeProj    = last qkRopeHeadDim rows (→ MQA-shared rope-K)
        public readonly VulkanDevice.Buffer KvALatentProj;
        public readonly VulkanDevice.Buffer KvAKPeProj;
        public readonly VulkanDevice.Buffer KvALayernormWeight;
        public readonly VulkanDevice.Buffer KvBProj;

        // Hyperparameters carried for forward-path convenience.
        public readonly int NumHeads;
        public readonly int QkNopeHeadDim;
        public readonly int QkRopeHeadDim;
        public readonly int VHeadDim;
        public readonly int QLoraRank;
        public readonly int KvLoraRank;
        public readonly int HiddenSize;

        public MlaLayerBuffers(
            VulkanDevice.Buffer? qAProj, VulkanDevice.Buffer? qALayernorm, VulkanDevice.Buffer? qBProj,
            VulkanDevice.Buffer? qProj,
            VulkanDevice.Buffer kvALatentProj, VulkanDevice.Buffer kvAKPeProj,
            VulkanDevice.Buffer kvALayernorm, VulkanDevice.Buffer kvBProj,
            int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
            int qLoraRank, int kvLoraRank, int hiddenSize)
        {
            QAProj = qAProj;
            QALayernormWeight = qALayernorm;
            QBProj = qBProj;
            QProj = qProj;
            KvALatentProj = kvALatentProj;
            KvAKPeProj = kvAKPeProj;
            KvALayernormWeight = kvALayernorm;
            KvBProj = kvBProj;
            NumHeads = numHeads;
            QkNopeHeadDim = qkNopeHeadDim;
            QkRopeHeadDim = qkRopeHeadDim;
            VHeadDim = vHeadDim;
            QLoraRank = qLoraRank;
            KvLoraRank = kvLoraRank;
            HiddenSize = hiddenSize;
        }

        public int QkHeadDim => QkNopeHeadDim + QkRopeHeadDim;
        public int QTotal => NumHeads * QkHeadDim;
        public int KvBOutputDim => NumHeads * (QkNopeHeadDim + VHeadDim);

        public void Dispose()
        {
            QAProj?.Dispose();
            QALayernormWeight?.Dispose();
            QBProj?.Dispose();
            QProj?.Dispose();
            KvALatentProj.Dispose();
            KvAKPeProj.Dispose();
            KvALayernormWeight.Dispose();
            KvBProj.Dispose();
        }
    }

    /// <summary>
    /// Per-layer Gemma-4 MoE extras: the dual-parallel-FFN norms, custom-router
    /// channel scale, per-layer output scale, and the V-from-K flag. Mirrors the
    /// CPU <c>Gemma4LayerWeights</c>. Non-null only on Gemma-4 (<c>gemma4</c> /
    /// DiffusionGemma) layers; null on every other architecture, where the
    /// standard dense/MoE FFN graph runs. The dense FFN (Gate/Up/Down on
    /// <see cref="LayerBuffers"/>) and the routed experts (<see cref="LayerBuffers.Moe"/>)
    /// both run in parallel on a Gemma-4 layer — neither stubs the other.
    /// </summary>
    internal readonly struct Gemma4LayerBuffers
    {
        /// <summary>MoE branch pre-norm <c>pre_ffw_norm_2</c> [hidden] — RMSNorm'd attn_out fed to the experts.</summary>
        public readonly VulkanDevice.Buffer PreFfwNorm2;
        /// <summary>Dense branch post-norm <c>post_ffw_norm_1</c> [hidden] — applied to the dense MLP output.</summary>
        public readonly VulkanDevice.Buffer PostFfwNorm1;
        /// <summary>MoE branch post-norm <c>post_ffw_norm_2</c> [hidden] — applied to the MoE output.</summary>
        public readonly VulkanDevice.Buffer PostFfwNorm2;
        /// <summary>Combined post-norm <c>post_ffw_norm</c> [hidden] — wraps (dense + MoE) before the residual add.</summary>
        public readonly VulkanDevice.Buffer PostFfwNorm;
        /// <summary>Custom-router channel scale <c>ffn_gate_inp.scale</c> [hidden] — multiplies the scaled-RMS router input.</summary>
        public readonly VulkanDevice.Buffer RouterScale;
        /// <summary>Per-layer output scale <c>layer_output_scale</c> — single scalar applied as the LAST per-layer op.</summary>
        public readonly float LayerOutputScale;
        /// <summary>True on a V-less (global/full-attention) layer where V branches off the RAW K projection — the forward copies K→V and the V projection slot is unused.</summary>
        public readonly bool VFromK;
        /// <summary>
        /// Per-expert down-projection scale <c>ffn_down_exps.scale</c> [numExperts] as an
        /// F32 device buffer. Non-null ONLY on the QUANTIZED expert path, where the down
        /// (Q5_1) indexed-matmul shader folds the scale into its output (op #14). On the
        /// F32 host-dequant path the scale is pre-folded into the W2 weight at upload, so
        /// this is null and the weighted-scatter carries no per-expert scale.
        /// </summary>
        public readonly VulkanDevice.Buffer? DownExpertScale;

        public Gemma4LayerBuffers(
            VulkanDevice.Buffer preFfwNorm2, VulkanDevice.Buffer postFfwNorm1,
            VulkanDevice.Buffer postFfwNorm2, VulkanDevice.Buffer postFfwNorm,
            VulkanDevice.Buffer routerScale, float layerOutputScale, bool vFromK,
            VulkanDevice.Buffer? downExpertScale = null)
        {
            PreFfwNorm2 = preFfwNorm2;
            PostFfwNorm1 = postFfwNorm1;
            PostFfwNorm2 = postFfwNorm2;
            PostFfwNorm = postFfwNorm;
            RouterScale = routerScale;
            LayerOutputScale = layerOutputScale;
            VFromK = vFromK;
            DownExpertScale = downExpertScale;
        }

        public void Dispose()
        {
            PreFfwNorm2.Dispose();
            PostFfwNorm1.Dispose();
            PostFfwNorm2.Dispose();
            PostFfwNorm.Dispose();
            RouterScale.Dispose();
            DownExpertScale?.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;

        // Q/K/V/QBias/KBias/VBias are unused (default) on MLA layers — see
        // <see cref="Mla"/>. The dense FFN block (FfnNorm/Gate/Up/Down) is
        // shared with the standard transformer path.
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

        public readonly VulkanDevice.Buffer? QBias, KBias, VBias, OBias;

        /// <summary>
        /// Per-head Q/K RMSNorm weights [headDim] (Gemma-4, Qwen3). Applied per
        /// head as <c>rmsnorm(rowCount = seqLen*numHeads, n = headDim)</c> before
        /// RoPE. Null when the architecture has no QK-norm.
        /// </summary>
        public readonly VulkanDevice.Buffer? QNormWeight, KNormWeight;

        /// <summary>
        /// Gemma four-norm layout: optional post-attention RMSNorm applied to
        /// the attention sublayer output BEFORE the residual add. Null on
        /// non-Gemma architectures (standard two-norm layout).
        /// </summary>
        public readonly VulkanDevice.Buffer? PostAttnNormWeight;

        /// <summary>
        /// Gemma four-norm layout: optional post-FFN RMSNorm applied to the
        /// FFN sublayer output BEFORE the residual add. Null on non-Gemma
        /// architectures.
        /// </summary>
        public readonly VulkanDevice.Buffer? PostFfnNormWeight;

        /// <summary>
        /// BitNet b1.58 Sub-LN: optional RMSNorm over the attention output
        /// [hiddenSize] applied BEFORE the output projection. Null on every
        /// non-BitNet architecture (no extra norm).
        /// </summary>
        public readonly VulkanDevice.Buffer? AttnSubNormWeight;

        /// <summary>
        /// BitNet b1.58 Sub-LN: optional RMSNorm over the gated FFN intermediate
        /// [intermediateSize] applied BEFORE the down projection. Null on every
        /// non-BitNet architecture.
        /// </summary>
        public readonly VulkanDevice.Buffer? FfnSubNormWeight;

        /// <summary>
        /// Non-null when the layer uses MLA attention (DeepSeek-V2/V3).
        /// Forward routes through <c>RecordMlaLayer</c> and the Q/K/V slots
        /// above are unused (zero buffers).
        /// </summary>
        public readonly MlaLayerBuffers? Mla;

        /// <summary>
        /// Non-null when the layer uses a MoE FFN (Mixtral, Qwen-MoE).
        /// Forward routes the FFN through <c>RecordMoeLayer</c> and the
        /// dense Gate/Up/Down slots above are unused (zero buffers).
        /// </summary>
        public readonly MoeLayerBuffers? Moe;

        /// <summary>
        /// Non-null when the layer is a Gemma-4 MoE dual-FFN layer. Carries the
        /// extra FFN norms, custom-router scale, layer-output scale and V-from-K
        /// flag (see <see cref="Gemma4LayerBuffers"/>). On a Gemma-4 layer BOTH
        /// the dense Gate/Up/Down slots AND <see cref="Moe"/> are populated and
        /// run in parallel — the forward routes through <c>RecordGemma4Layer</c>.
        /// </summary>
        public readonly Gemma4LayerBuffers? Gemma4;

        public readonly VulkanDevice.Buffer FfnNormWeight;

        public readonly VulkanDevice.Buffer Gate;
        public readonly VulkanDevice.Buffer Up;
        public readonly VulkanDevice.Buffer Down;
        public readonly QuantizationType GateDeviceQuantType;
        public readonly QuantizationType UpDeviceQuantType;
        public readonly QuantizationType DownDeviceQuantType;
        public readonly int GateOutputDim, GateInputDim;
        public readonly int UpOutputDim, UpInputDim;
        public readonly int DownOutputDim, DownInputDim;

        public readonly VulkanDevice.Buffer? GateBias, UpBias, DownBias;

        public LayerBuffers(
            VulkanDevice.Buffer attnNorm,
            VulkanDevice.Buffer q, QuantizationType qQt, int qM, int qK,
            VulkanDevice.Buffer k, QuantizationType kQt, int kM, int kK,
            VulkanDevice.Buffer v, QuantizationType vQt, int vM, int vK,
            VulkanDevice.Buffer o, QuantizationType oQt, int oM, int oK,
            VulkanDevice.Buffer? qBias, VulkanDevice.Buffer? kBias, VulkanDevice.Buffer? vBias, VulkanDevice.Buffer? oBias,
            VulkanDevice.Buffer ffnNorm,
            VulkanDevice.Buffer gate, QuantizationType gateQt, int gateM, int gateK,
            VulkanDevice.Buffer up, QuantizationType upQt, int upM, int upK,
            VulkanDevice.Buffer down, QuantizationType downQt, int downM, int downK,
            VulkanDevice.Buffer? gateBias, VulkanDevice.Buffer? upBias, VulkanDevice.Buffer? downBias,
            VulkanDevice.Buffer? postAttnNorm = null,
            VulkanDevice.Buffer? postFfnNorm = null,
            VulkanDevice.Buffer? attnSubNorm = null,
            VulkanDevice.Buffer? ffnSubNorm = null,
            MlaLayerBuffers? mla = null,
            MoeLayerBuffers? moe = null,
            Gemma4LayerBuffers? gemma4 = null,
            VulkanDevice.Buffer? qNorm = null,
            VulkanDevice.Buffer? kNorm = null)
        {
            AttnNormWeight = attnNorm;
            QNormWeight = qNorm;
            KNormWeight = kNorm;
            PostAttnNormWeight = postAttnNorm;
            PostFfnNormWeight = postFfnNorm;
            AttnSubNormWeight = attnSubNorm;
            FfnSubNormWeight = ffnSubNorm;
            Q = q; QDeviceQuantType = qQt; QOutputDim = qM; QInputDim = qK;
            K = k; KDeviceQuantType = kQt; KOutputDim = kM; KInputDim = kK;
            V = v; VDeviceQuantType = vQt; VOutputDim = vM; VInputDim = vK;
            O = o; ODeviceQuantType = oQt; OOutputDim = oM; OInputDim = oK;
            QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
            FfnNormWeight = ffnNorm;
            Gate = gate; GateDeviceQuantType = gateQt; GateOutputDim = gateM; GateInputDim = gateK;
            Up = up; UpDeviceQuantType = upQt; UpOutputDim = upM; UpInputDim = upK;
            Down = down; DownDeviceQuantType = downQt; DownOutputDim = downM; DownInputDim = downK;
            GateBias = gateBias; UpBias = upBias; DownBias = downBias;
            Mla = mla;
            Moe = moe;
            Gemma4 = gemma4;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
            QBias?.Dispose(); KBias?.Dispose(); VBias?.Dispose(); OBias?.Dispose();
            QNormWeight?.Dispose(); KNormWeight?.Dispose();
            PostAttnNormWeight?.Dispose();
            PostFfnNormWeight?.Dispose();
            AttnSubNormWeight?.Dispose();
            FfnSubNormWeight?.Dispose();
            FfnNormWeight.Dispose();
            Gate.Dispose(); Up.Dispose(); Down.Dispose();
            GateBias?.Dispose(); UpBias?.Dispose(); DownBias?.Dispose();
            Mla?.Dispose();
            Moe?.Dispose();
            Gemma4?.Dispose();
        }
    }

    private readonly VulkanDevice _device;
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

    private VulkanWeights(
        VulkanDevice device,
        VulkanDevice.Buffer tokenEmbed, int vocabSize, int hiddenSize,
        LayerBuffers[] layers,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, QuantizationType outputDeviceQt, int outputM, int outputK,
        long allocatedBytes)
    {
        _device = device;
        TokenEmbedding = tokenEmbed;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        _layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputDeviceQuantType = outputDeviceQt;
        OutputOutputDim = outputM;
        OutputInputDim = outputK;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads the given CPU-resident <see cref="TransformerWeights"/> to the
    /// Vulkan device as immutable device-local buffers.
    /// </summary>
    /// <param name="device">Vulkan device to upload to.</param>
    /// <param name="weights">CPU-resident weights (mmap-backed).</param>
    /// <param name="numLayers">Number of transformer layers to upload.</param>
    /// <param name="dequantToFp32">
    /// When <c>false</c> (default) Q8_0 matrices are uploaded as raw Q8_0
    /// blocks and the forward pass dispatches them through the quantised
    /// Q8_0 matmul kernels. When <c>true</c> every matrix is dequantised to
    /// FP32 at upload — the legacy scaffold path, kept as a fallback for
    /// environments where the Q8_0 kernels regress.
    /// </param>
    /// <param name="firstLayer">
    /// Index of the first <see cref="TransformerWeights.Layers"/> entry to upload (default 0). Together
    /// with <paramref name="numLayers"/> this selects the half-open window
    /// <c>[firstLayer .. firstLayer+numLayers)</c> — used to load only a pipeline stage's slice of layers
    /// onto a given device for cross-device layer-spanning (pipeline parallelism). The returned
    /// <see cref="Layers"/> are indexed locally (0-based) within the window.
    /// </param>
    /// <remarks>
    /// <para>
    /// Staging is sized to fit the largest single matrix in its <i>widest</i>
    /// on-host form (FP32 for non-Q8_0 matrices or when <paramref name="dequantToFp32"/>
    /// is true; raw Q8_0 bytes for Q8_0 matrices when kept on device).
    /// </para>
    /// </remarks>
    public static VulkanWeights Upload(
        VulkanDevice device, TransformerWeights weights, int numLayers,
        bool dequantToFp32 = false, int firstLayer = 0)
    {
        if (firstLayer < 0 || firstLayer + numLayers > weights.Layers.Length)
            throw new ArgumentOutOfRangeException(nameof(firstLayer),
                $"Layer window [{firstLayer}..{firstLayer + numLayers}) is outside [0..{weights.Layers.Length}).");

        long totalBytes = 0;
        ResetUploadCounters();

        // Size the reusable staging buffer to the largest single weight upload
        // (in its on-device byte form) across the uploaded layer window.
        long stagingBytes = ComputeMaxUploadBytes(weights, numLayers, dequantToFp32, firstLayer);
        using var staging = device.Allocate(stagingBytes);

        // Token embedding table: [vocabSize, hiddenSize]. Uploaded once as a
        // device-local F32 buffer so VulkanTransformerModel.Forward can gather
        // per-token rows via vkCmdCopyBuffer onto the shared command buffer —
        // no per-forward host→device write. Quantised tables (Q8_0, F16, etc.)
        // are dequantised to F32 at construction time; keeping them as raw
        // Q8_0 blocks on device would need a GPU gather-and-dequant kernel,
        // which is out of scope for this change.
        var tokenEmbed = UploadMatrix(device, staging,
            weights.TokenEmbedWeight, weights.TokenEmbedQuantType,
            weights.VocabSize, weights.HiddenSize,
            dequantToFp32: true,
            out _, out long tokenEmbedBytes);
        totalBytes += tokenEmbedBytes;

        // Upload an arbitrary layer window [firstLayer .. firstLayer+numLayers): the local LayerBuffers
        // index is 0..numLayers-1, sourced from the global layer firstLayer+i. firstLayer=0 (the default)
        // is the single-device / first-pipeline-stage case; firstLayer>0 lets a second pipeline stage hold
        // only its slice of layers for cross-device layer-spanning (pipeline parallelism).
        var layerBuffers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[firstLayer + i];

            // Gemma-4 MoE dual-FFN layer: both the dense Gate/Up/Down AND the
            // routed experts run in parallel, V branches off the raw K projection
            // on V-less (global) layers, and four extra FFN norms + a per-layer
            // output scale apply. Detected by the loader-resolved Gemma4 extras.
            bool isGemma4 = lw.Gemma4 is not null;
            bool vFromK = lw.Gemma4?.VFromK ?? false;

            var attnNorm = UploadNormVec(device, staging, lw.AttnNormWeight);
            totalBytes += (long)lw.AttnNormWeight.Length * sizeof(float);

            // Per-head Q/K RMSNorm weights [headDim] (Gemma-4, Qwen3). Null when
            // the architecture has no QK-norm (UploadOptionalVec returns null).
            var qNorm = UploadOptionalVec(device, staging, lw.QNormWeight);
            var kNorm = UploadOptionalVec(device, staging, lw.KNormWeight);
            if (lw.QNormWeight is not null) totalBytes += (long)lw.QNormWeight.Length * sizeof(float);
            if (lw.KNormWeight is not null) totalBytes += (long)lw.KNormWeight.Length * sizeof(float);

            // Gemma four-norm layout: optional post-attention RMSNorm weight
            // applied to the attention output before the residual add. Null on
            // non-Gemma architectures (UploadOptionalVec returns null).
            var postAttnNorm = UploadOptionalVec(device, staging, lw.PostAttnNormWeight);
            if (lw.PostAttnNormWeight is not null)
                totalBytes += (long)lw.PostAttnNormWeight.Length * sizeof(float);

            // MLA layers carry their projections in lw.Mla; the standard
            // Q/K/V slots are zeroed by the loader. Replace each with a
            // 64-byte stub so the LayerBuffers contract still holds — the
            // forward pass never dispatches a matmul against them.
            VulkanDevice.Buffer q, k, v;
            QuantizationType qDeviceQt, kDeviceQt, vDeviceQt;
            long qBytes, kBytes, vBytes;
            VulkanDevice.Buffer? qBias, kBias, vBias;
            if (lw.Mla is not null)
            {
                q = device.AllocateDeviceLocal(64);
                k = device.AllocateDeviceLocal(64);
                v = device.AllocateDeviceLocal(64);
                qDeviceQt = kDeviceQt = vDeviceQt = QuantizationType.F32;
                qBytes = kBytes = vBytes = 0;
                qBias = kBias = vBias = null;
            }
            else
            {
                q = UploadMatrix(device, staging, lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim,
                    dequantToFp32, out qDeviceQt, out qBytes);
                k = UploadMatrix(device, staging, lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim,
                    dequantToFp32, out kDeviceQt, out kBytes);
                if (vFromK)
                {
                    // V-less global layer: no attn_v weight; the forward copies
                    // the raw K projection into V. Stub the slot (never matmul'd).
                    v = device.AllocateDeviceLocal(64);
                    vDeviceQt = QuantizationType.F32;
                    vBytes = 0;
                }
                else
                {
                    v = UploadMatrix(device, staging, lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim,
                        dequantToFp32, out vDeviceQt, out vBytes);
                }
                qBias = UploadOptionalVec(device, staging, lw.QBias);
                kBias = UploadOptionalVec(device, staging, lw.KBias);
                vBias = UploadOptionalVec(device, staging, lw.VBias);
            }
            var o = UploadMatrix(device, staging, lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim,
                dequantToFp32, out var oDeviceQt, out long oBytes);
            var oBias = UploadOptionalVec(device, staging, lw.OBias);

            MlaLayerBuffers? mla = null;
            if (lw.Mla is not null)
            {
                mla = UploadMlaLayer(device, staging, lw.Mla, weights.HiddenSize, out long mlaBytes);
                totalBytes += mlaBytes;
            }

            var ffnNorm = UploadNormVec(device, staging, lw.FfnNormWeight);
            totalBytes += (long)lw.FfnNormWeight.Length * sizeof(float);

            // Gemma four-norm layout: optional post-FFN RMSNorm weight applied
            // to the FFN output before the residual add. Null on non-Gemma.
            // For Gemma-4 this is forced null — the combined post_ffw_norm is
            // applied INSIDE RecordGemma4Ffn (g4.PostFfwNorm), so the shared
            // residual-#2 post-FFN norm must NOT also fire (double-norm).
            var postFfnNorm = isGemma4 ? null : UploadOptionalVec(device, staging, lw.PostFfnNormWeight);
            if (!isGemma4 && lw.PostFfnNormWeight is not null)
                totalBytes += (long)lw.PostFfnNormWeight.Length * sizeof(float);

            // BitNet b1.58 Sub-LN: optional RMSNorm weights applied to the attention
            // output (before o_proj) and the gated FFN intermediate (before ffn_down).
            // Null on every non-BitNet architecture (UploadOptionalVec returns null).
            var attnSubNorm = UploadOptionalVec(device, staging, lw.AttnSubNormWeight);
            if (lw.AttnSubNormWeight is not null)
                totalBytes += (long)lw.AttnSubNormWeight.Length * sizeof(float);
            var ffnSubNorm = UploadOptionalVec(device, staging, lw.FfnSubNormWeight);
            if (lw.FfnSubNormWeight is not null)
                totalBytes += (long)lw.FfnSubNormWeight.Length * sizeof(float);

            // MoE layers replace the dense Gate/Up/Down with per-expert
            // banks (lw.Moe). Stub the dense slots with 64-byte buffers so
            // the LayerBuffers contract still holds — the forward pass
            // never dispatches a matmul against them on MoE layers.
            VulkanDevice.Buffer gate, up, down;
            QuantizationType gateDeviceQt, upDeviceQt, downDeviceQt;
            long gateBytes, upBytes, downBytes;
            VulkanDevice.Buffer? gateBias, upBias, downBias;
            if (lw.Moe is not null && !isGemma4)
            {
                gate = device.AllocateDeviceLocal(64);
                up = device.AllocateDeviceLocal(64);
                down = device.AllocateDeviceLocal(64);
                gateDeviceQt = upDeviceQt = downDeviceQt = QuantizationType.F32;
                gateBytes = upBytes = downBytes = 0;
                gateBias = upBias = downBias = null;
            }
            else
            {
                gate = UploadMatrix(device, staging, lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim,
                    dequantToFp32, out gateDeviceQt, out gateBytes);
                up = UploadMatrix(device, staging, lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim,
                    dequantToFp32, out upDeviceQt, out upBytes);
                down = UploadMatrix(device, staging, lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim,
                    dequantToFp32, out downDeviceQt, out downBytes);
                gateBias = UploadOptionalVec(device, staging, lw.GateBias);
                upBias = UploadOptionalVec(device, staging, lw.UpBias);
                downBias = UploadOptionalVec(device, staging, lw.DownBias);
            }

            MoeLayerBuffers? moe = null;
            Gemma4LayerBuffers? gemma4 = null;
            if (lw.Moe is not null)
            {
                if (isGemma4)
                {
                    // Gemma-4 experts. PREFERRED: keep the fused gate_up (Q4_K) and
                    // down (Q5_1) banks QUANTIZED on device — repacked into contiguous
                    // gate/up/down banks the Q4_K / Q5_1 indexed-MoE matmul shaders
                    // dequant per-row (memory-viable for the real 26B, whose F32
                    // experts are tens of GB). FALLBACK: host-dequant to F32 banks
                    // (the existing path) when the quant mix is unsupported.
                    VulkanDevice.Buffer? downScaleBuf = null;
                    if (Gemma4ExpertsKeepQuantized(lw.Moe))
                    {
                        moe = UploadGemma4MoeLayerQuantized(device, lw.Moe, lw.Gemma4!,
                            out var dsBuf, out long moeBytes);
                        downScaleBuf = dsBuf;
                        totalBytes += moeBytes;
                    }
                    else
                    {
                        moe = UploadGemma4MoeLayer(device, lw.Moe, lw.Gemma4!, out long moeBytes);
                        totalBytes += moeBytes;
                    }

                    var g4 = lw.Gemma4!;
                    // Pre-fold 1/sqrt(hidden) into the router channel scale so the
                    // custom-router input is a plain rmsnorm(attn_out, RouterScale·invSqrtH)
                    // dispatch on device (CPU does rms·(1/√H)·ffn_gate_inp_s).
                    float invSqrtH = 1.0f / MathF.Sqrt(weights.HiddenSize);
                    float[] routerScaleScaled = new float[g4.RouterScale.Length];
                    for (int j = 0; j < routerScaleScaled.Length; j++)
                        routerScaleScaled[j] = g4.RouterScale[j] * invSqrtH;

                    gemma4 = new Gemma4LayerBuffers(
                        UploadNormVec(device, staging, g4.PreFfwNorm2),
                        UploadNormVec(device, staging, g4.PostFfwNorm1),
                        UploadNormVec(device, staging, g4.PostFfwNorm2),
                        UploadNormVec(device, staging, g4.PostFfwNorm),
                        UploadNormVec(device, staging, routerScaleScaled),
                        g4.LayerOutputScale,
                        g4.VFromK,
                        downScaleBuf);
                    totalBytes += (long)(g4.PreFfwNorm2.Length + g4.PostFfwNorm1.Length
                        + g4.PostFfwNorm2.Length + g4.PostFfwNorm.Length + g4.RouterScale.Length) * sizeof(float);
                }
                else
                {
                    moe = UploadMoeLayer(device, lw.Moe, out long moeBytes);
                    totalBytes += moeBytes;
                }
            }

            layerBuffers[i] = new LayerBuffers(
                attnNorm,
                q, qDeviceQt, lw.QOutputDim, lw.QInputDim,
                k, kDeviceQt, lw.KOutputDim, lw.KInputDim,
                v, vDeviceQt, lw.VOutputDim, lw.VInputDim,
                o, oDeviceQt, lw.OOutputDim, lw.OInputDim,
                qBias, kBias, vBias, oBias,
                ffnNorm,
                gate, gateDeviceQt, lw.GateOutputDim, lw.GateInputDim,
                up, upDeviceQt, lw.UpOutputDim, lw.UpInputDim,
                down, downDeviceQt, lw.DownOutputDim, lw.DownInputDim,
                gateBias, upBias, downBias,
                postAttnNorm, postFfnNorm,
                attnSubNorm, ffnSubNorm,
                mla, moe, gemma4,
                qNorm, kNorm);

            totalBytes += qBytes + kBytes + vBytes + oBytes
                + gateBytes + upBytes + downBytes;
        }

        var outputNorm = UploadNormVec(device, staging, weights.OutputNormWeight);
        totalBytes += (long)weights.OutputNormWeight.Length * sizeof(float);

        var outputWeight = UploadMatrix(device, staging,
            weights.OutputWeight, weights.OutputQuantType,
            weights.OutputOutputDim, weights.OutputInputDim,
            dequantToFp32,
            out var outputDeviceQt, out long outputBytes);
        totalBytes += outputBytes;

        return new VulkanWeights(
            device, tokenEmbed, weights.VocabSize, weights.HiddenSize,
            layerBuffers,
            outputNorm, outputWeight, outputDeviceQt,
            weights.OutputOutputDim, weights.OutputInputDim,
            totalBytes);
    }

    /// <summary>
    /// Diagnostic counter — number of weight matrices that took the
    /// <c>VK_EXT_external_memory_host</c> zero-copy path on the most recent
    /// <see cref="Upload"/> call. Reset to zero at the start of each upload.
    /// Reported by the benchmark + test harness; not used at runtime.
    /// </summary>
    public static int LastUploadZeroCopyMatrices { get; private set; }

    /// <summary>
    /// Diagnostic counter — number of weight matrices that took the staging
    /// copy path on the most recent <see cref="Upload"/> call. Sum of this
    /// plus <see cref="LastUploadZeroCopyMatrices"/> is the total number of
    /// raw-quant-block matrices uploaded; F32-dequant matrices are counted
    /// separately as staging (they cannot be zero-copy imported).
    /// </summary>
    public static int LastUploadStagingMatrices { get; private set; }

    /// <summary>
    /// Diagnostic counter — total bytes that took the zero-copy path on the
    /// most recent <see cref="Upload"/> call. The microbench compares this
    /// against <c>AllocatedBytes</c> to confirm the path actually fired.
    /// </summary>
    public static long LastUploadZeroCopyBytes { get; private set; }

    /// <summary>
    /// Set <c>DOTLLM_VULKAN_DISABLE_HOST_IMPORT=1</c> in the environment to
    /// force the staging-copy path even when the driver supports
    /// <c>VK_EXT_external_memory_host</c>. Used by parity tests to verify
    /// that the zero-copy import produces bit-identical kernel output, and
    /// by the microbench to measure the staging baseline.
    /// </summary>
    private static bool IsHostImportDisabled() =>
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT") == "1";

    /// <summary>
    /// Attempts to wrap <paramref name="srcPtr"/> + <paramref name="bytes"/>
    /// in a host-imported <c>VkBuffer</c> via
    /// <see cref="VulkanDevice.TryWrapHostVisible"/>. Returns true on success
    /// (and increments the diagnostic counters); false otherwise — caller
    /// falls back to staging.
    /// </summary>
    private static bool TryZeroCopyImport(
        VulkanDevice device, nint srcPtr, long bytes,
        out VulkanDevice.Buffer? buf)
    {
        buf = null;
        if (!device.HasExternalMemoryHost)
        {
            LastUploadFallbackReason = "feature_absent";
            return false;
        }
        if (IsHostImportDisabled())
        {
            LastUploadFallbackReason = "env_disabled";
            return false;
        }
        if (srcPtr == 0)
        {
            LastUploadFallbackReason = "null_src";
            return false;
        }

        var wrapped = device.TryWrapHostVisible(srcPtr, bytes);
        if (wrapped is null)
        {
            LastUploadFallbackReason = "import_rejected";
            return false;
        }

        LastUploadZeroCopyMatrices++;
        LastUploadZeroCopyBytes += bytes;
        buf = wrapped;
        return true;
    }

    /// <summary>
    /// Last reason the most recent <see cref="UploadMatrix"/> call fell back
    /// from the zero-copy path to staging. Diagnostic only. Values include
    /// "feature_absent" (driver does not expose VK_EXT_external_memory_host),
    /// "env_disabled" (DOTLLM_VULKAN_DISABLE_HOST_IMPORT=1), "null_src"
    /// (source pointer is null), "import_rejected" (driver rejected the
    /// vkAllocateMemory import). Empty string when the most recent call took
    /// the zero-copy path or when no fallback decision has been made.
    /// </summary>
    public static string LastUploadFallbackReason { get; private set; } = string.Empty;

    /// <summary>Resets the per-upload diagnostic counters. Called from
    /// <see cref="Upload"/> at the start of each call.</summary>
    private static void ResetUploadCounters()
    {
        LastUploadZeroCopyMatrices = 0;
        LastUploadStagingMatrices = 0;
        LastUploadZeroCopyBytes = 0;
        LastUploadFallbackReason = string.Empty;
    }

    /// <summary>Returns true when the matrix will be kept on device as Q8_0 blocks.</summary>
    private static bool KeepQ8OnDevice(QuantizationType qt, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q8_0;

    /// <summary>Returns true when the matrix will be kept on device as native F16
    /// (2 bytes per element). Gated on the contraction axis being a multiple of 2
    /// (each storage uint holds two F16 elements via <c>unpackHalf2x16</c>).
    /// Phase 8 of the K-quant / native-float work — unblocks BF16 / F16 SafeTensors
    /// loads that previously had to expand to F32 at upload, doubling VRAM.</summary>
    private static bool KeepF16OnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.F16 && (inputDim & 1) == 0;

    /// <summary>Returns true when the matrix will be kept on device as native BF16
    /// (2 bytes per element). Gated on the contraction axis being a multiple of 2.
    /// BF16 expand on read: shift-left-16 + reinterpret-as-F32 in the matmul shader.
    /// Phase 8 sibling of <see cref="KeepF16OnDevice"/>.</summary>
    private static bool KeepBf16OnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.BF16 && (inputDim & 1) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q2_K super-blocks
    /// (84 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q2_K super-block size (256).</summary>
    private static bool KeepQ2KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q2_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q3_K super-blocks
    /// (110 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q3_K super-block size (256).</summary>
    private static bool KeepQ3KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q3_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q4_K super-blocks
    /// (144 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q4_K super-block size (256). Phase 1 of the K-quant work.</summary>
    private static bool KeepQ4KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q4_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q5_K super-blocks
    /// (176 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q5_K super-block size (256). Phase 1 sibling of <see cref="KeepQ4KOnDevice"/>.</summary>
    private static bool KeepQ5KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q5_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q6_K super-blocks
    /// (210 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q6_K super-block size (256). Phase 1 sibling of <see cref="KeepQ4KOnDevice"/>
    /// completing the K-quant matmul kernel coverage (Q4_K / Q5_K / Q6_K).</summary>
    private static bool KeepQ6KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q6_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ4_NL blocks
    /// (18 bytes per 32 elements). Gated on the contraction axis being a multiple of
    /// the IQ4_NL block size (32).</summary>
    private static bool KeepIq4NlOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ4_NL && (inputDim % 32) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ4_XS super-blocks
    /// (136 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the IQ4_XS super-block size (256).</summary>
    private static bool KeepIq4XsOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ4_XS && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ1_S super-blocks
    /// (50 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the IQ1_S super-block size (256). The smallest GGUF quant — ~1.5-1.7 bpw.</summary>
    private static bool KeepIq1SOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ1_S && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as raw I2_S (BitNet b1.58
    /// ternary): m·(K/4) packed bytes + one trailing per-tensor float32 scale. Gated on the
    /// contraction axis being a multiple of the I2_S block size (128), which the GEMV/GEMM
    /// kernels require.</summary>
    private static bool KeepI2SOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.I2_S && (inputDim % 128) == 0;

    /// <summary>Returns the on-device storage quant type for a projection: Q8_0 / Q4_K /
    /// Q5_K / Q6_K / IQ4_NL / IQ4_XS / F16 / BF16 / F32 depending on the source and the
    /// alignment constraints.</summary>
    /// <summary>Returns true when the matrix will be kept on device as IQ2_XXS super-blocks
    /// (66 bytes per 256 elements). Gated on the contraction axis being a multiple of 256.</summary>
    private static bool KeepIq2XxsOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ2_XXS && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ2_XS super-blocks
    /// (74 bytes per 256 elements). Gated on the contraction axis being a multiple of 256.</summary>
    private static bool KeepIq2XsOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ2_XS && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ2_S super-blocks
    /// (82 bytes per 256 elements). Also covers MOSTLY_IQ2_M file-type tensors.
    /// Gated on the contraction axis being a multiple of 256.</summary>
    private static bool KeepIq2SOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ2_S && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ3_XXS super-blocks
    /// (98 bytes per 256 elements). Gated on the contraction axis being a multiple of 256.</summary>
    private static bool KeepIq3XxsOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ3_XXS && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as IQ3_S super-blocks
    /// (110 bytes per 256 elements). Gated on the contraction axis being a multiple of 256.</summary>
    private static bool KeepIq3SOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.IQ3_S && (inputDim % 256) == 0;

    /// <summary>Returns the on-device storage quant type for a projection: Q8_0 / Q4_K /
    /// Q5_K / Q6_K / IQ2_XXS / IQ2_XS / IQ2_S / F16 / BF16 / F32 depending on the source
    /// Q5_K / Q6_K / IQ4_NL / IQ4_XS / IQ1_S / F16 / BF16 / F32 depending on the source
    /// and the alignment constraints.</summary>
    /// <summary>Returns the on-device storage quant type for a projection: Q8_0 / Q2_K /
    /// Q3_K / Q4_K / Q5_K / Q6_K / F16 / BF16 / F32 depending on the source and the
    /// alignment constraints.</summary>
    private static QuantizationType DeviceQuantTypeFor(
        QuantizationType srcQt, int inputDim, bool dequantToFp32)
    {
        if (KeepQ8OnDevice(srcQt, dequantToFp32)) return QuantizationType.Q8_0;
        if (KeepQ2KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q2_K;
        if (KeepQ3KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q3_K;
        if (KeepQ4KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q4_K;
        if (KeepQ5KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q5_K;
        if (KeepQ6KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q6_K;
        if (KeepIq4NlOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ4_NL;
        if (KeepIq4XsOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ4_XS;
        if (KeepIq2XxsOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ2_XXS;
        if (KeepIq2XsOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ2_XS;
        if (KeepIq2SOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ2_S;
        if (KeepIq3XxsOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ3_XXS;
        if (KeepIq3SOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ3_S;
        if (KeepIq1SOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.IQ1_S;
        if (KeepI2SOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.I2_S;
        if (KeepF16OnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.F16;
        if (KeepBf16OnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.BF16;
        return QuantizationType.F32;
    }

    private static long ComputeMaxUploadBytes(
        TransformerWeights weights, int numLayers, bool dequantToFp32, int firstLayer = 0)
    {
        long max = 0;
        max = Math.Max(max, UploadBytes(weights.VocabSize, weights.HiddenSize, weights.TokenEmbedQuantType, dequantToFp32: true));
        max = Math.Max(max, UploadBytes(weights.OutputOutputDim, weights.OutputInputDim, weights.OutputQuantType, dequantToFp32));
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[firstLayer + i];
            max = Math.Max(max, UploadBytes(lw.QOutputDim, lw.QInputDim, lw.QQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.KOutputDim, lw.KInputDim, lw.KQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.VOutputDim, lw.VInputDim, lw.VQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.OOutputDim, lw.OInputDim, lw.OQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.GateOutputDim, lw.GateInputDim, lw.GateQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.UpOutputDim, lw.UpInputDim, lw.UpQuantType, dequantToFp32));
            max = Math.Max(max, UploadBytes(lw.DownOutputDim, lw.DownInputDim, lw.DownQuantType, dequantToFp32));

            // MLA projections are F32 row-major (loader upcasts F16/BF16 at load).
            if (lw.Mla is not null)
            {
                int hidden = weights.HiddenSize;
                int qTotal = lw.Mla.NumHeads * (lw.Mla.QkNopeHeadDim + lw.Mla.QkRopeHeadDim);
                int kvBOut = lw.Mla.NumHeads * (lw.Mla.QkNopeHeadDim + lw.Mla.VHeadDim);
                if (lw.Mla.QLoraRank > 0)
                {
                    max = Math.Max(max, (long)lw.Mla.QLoraRank * hidden * sizeof(float));
                    max = Math.Max(max, (long)qTotal * lw.Mla.QLoraRank * sizeof(float));
                }
                else
                {
                    max = Math.Max(max, (long)qTotal * hidden * sizeof(float));
                }
                max = Math.Max(max, (long)lw.Mla.KvLoraRank * hidden * sizeof(float));
                max = Math.Max(max, (long)lw.Mla.QkRopeHeadDim * hidden * sizeof(float));
                max = Math.Max(max, (long)kvBOut * lw.Mla.KvLoraRank * sizeof(float));
            }
        }
        return max;
    }

    private static long UploadBytes(int outputDim, int inputDim, QuantizationType qt, bool dequantToFp32)
    {
        long elems = (long)outputDim * inputDim;
        if (KeepQ8OnDevice(qt, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q8_0) * outputDim;
        if (KeepQ2KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q2_K) * outputDim;
        if (KeepQ3KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q3_K) * outputDim;
        if (KeepQ4KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q4_K) * outputDim;
        if (KeepQ5KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q5_K) * outputDim;
        if (KeepQ6KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q6_K) * outputDim;
        if (KeepIq4NlOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ4_NL) * outputDim;
        if (KeepIq4XsOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ4_XS) * outputDim;
        if (KeepIq2XxsOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_XXS) * outputDim;
        if (KeepIq2XsOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_XS) * outputDim;
        if (KeepIq2SOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ2_S) * outputDim;
        if (KeepIq3XxsOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ3_XXS) * outputDim;
        if (KeepIq3SOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ3_S) * outputDim;
        if (KeepIq1SOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.IQ1_S) * outputDim;
        if (KeepI2SOnDevice(qt, inputDim, dequantToFp32))
            // m·(K/4) packed bytes + one trailing per-tensor float32 scale.
            return Dequantize.RowByteSize(inputDim, QuantizationType.I2_S) * outputDim + sizeof(float);
        if (KeepF16OnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.F16) * outputDim;
        if (KeepBf16OnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.BF16) * outputDim;
        return elems * sizeof(float);
    }

    /// <summary>
    /// Uploads a single weight matrix. When <paramref name="dequantToFp32"/> is false and
    /// the source is a quantised format with a matching Vulkan kernel (Q8_0 / Q4_K / Q5_K)
    /// and the contraction axis satisfies the kernel's group-size constraint, the raw
    /// block bytes are copied to device memory verbatim and the returned
    /// <paramref name="deviceQuantType"/> reflects the source format. Otherwise the source
    /// is dequantised to FP32 before upload and <paramref name="deviceQuantType"/> is
    /// <see cref="QuantizationType.F32"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Zero-copy fast path</b> (<c>VK_EXT_external_memory_host</c>): when the matrix is
    /// kept as raw quant blocks AND <c>device.HasExternalMemoryHost</c> is true AND the
    /// mmap'd source pointer satisfies the driver's
    /// <c>minImportedHostPointerAlignment</c> (after page-rounding the start +
    /// rounding-up the size), the buffer is imported directly from the mmap'd GGUF page
    /// range — no staging copy, no double-counted physical RAM on a unified-memory APU.
    /// llama.cpp does not do this today on the Vulkan path; this is dotLLM
    /// differentiation on Strix Halo and similar UMA iGPUs (see
    /// <c>.planning/notes/gaia-lemonade-research.md</c> §6 H3).
    /// </para>
    /// <para>
    /// On any failure of the zero-copy path (extension absent, driver rejects import,
    /// alignment unsolvable) this method silently falls through to the staging-copy
    /// upload below — the import is opportunistic, not load-bearing.
    /// </para>
    /// </remarks>
    private static unsafe VulkanDevice.Buffer UploadMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool dequantToFp32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;

        // Raw quant-block upload — keeps the GGUF on-disk byte layout intact on device so
        // the matmul_q8_0 / matmul_q2_k / matmul_q3_k / matmul_q4_k / matmul_q5_k / matmul_q6_k kernels can read it
        // directly. Mirrors the CPU path's mmap-backed layout.
        QuantizationType keepQt = DeviceQuantTypeFor(qt, inputDim, dequantToFp32);
        if (keepQt != QuantizationType.F32)
        {
            long rowBytes = Dequantize.RowByteSize(inputDim, keepQt);
            long bytes = rowBytes * outputDim;
            // I2_S carries one trailing per-tensor float32 scale after all packed
            // rows (at offset m·K/4); the kernels read it from the buffer tail.
            if (keepQt == QuantizationType.I2_S) bytes += sizeof(float);

            // Zero-copy import attempt. Opt out via env var so the staging path
            // can be exercised on a host that does support the extension —
            // also the parity-test escape hatch.
            if (TryZeroCopyImport(device, srcPtr, bytes, out var importedBuf))
            {
                deviceQuantType = keepQt;
                uploadedBytes = bytes;
                return importedBuf!;
            }

            var buf = device.AllocateDeviceLocal(bytes);
            VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
                .ThrowOnError("vkMapMemory VulkanWeights.UploadMatrix staging (raw quant)");
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

            LastUploadStagingMatrices++;
            deviceQuantType = keepQt;
            uploadedBytes = bytes;
            return buf;
        }

        // FP32 dequantised upload.
        long fpBytes = elems * sizeof(float);
        var fpBuf = device.AllocateDeviceLocal(fpBytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)fpBytes, 0, out nint fpMapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadMatrix staging");
        try
        {
            float* d = (float*)fpMapped;
            if (qt == QuantizationType.F32)
            {
                new ReadOnlySpan<float>((void*)srcPtr, checked((int)elems))
                    .CopyTo(new Span<float>(d, checked((int)elems)));
            }
            else if (qt == QuantizationType.I2_S)
            {
                // I2_S has a single per-tensor scale at the tail (offset m·K/4), so it must
                // be dequantised over the whole tensor at once — a per-row loop would read
                // the scale from mid-matrix packed bytes. Reached only when the raw I2_S
                // path is bypassed (dequant forced, e.g. lm_head, or K % 128 != 0).
                Dequantize.ToFloat32(srcPtr, elems, qt, new Span<float>(d, checked((int)elems)));
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(inputDim, qt);
                for (int row = 0; row < outputDim; row++)
                {
                    nint rowSrc = srcPtr + (nint)(row * rowBytes);
                    Dequantize.ToFloat32(rowSrc, inputDim, qt,
                        new Span<float>(d + (long)row * inputDim, inputDim));
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

    private static unsafe VulkanDevice.Buffer UploadNormVec(
        VulkanDevice device, VulkanDevice.Buffer staging, float[] normWeight)
    {
        long bytes = (long)normWeight.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadNormVec staging");
        try
        {
            normWeight.AsSpan().CopyTo(new Span<float>((void*)mapped, normWeight.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }

        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
    }

    private static VulkanDevice.Buffer? UploadOptionalVec(
        VulkanDevice device, VulkanDevice.Buffer staging, float[]? vec)
    {
        if (vec is null) return null;
        return UploadNormVec(device, staging, vec);
    }

    /// <summary>
    /// Uploads the MLA-specific projection weights for one layer. The CPU
    /// loader hands us F32 row-major pointers; we upload them as device-local
    /// FP32 buffers (no Q8_0 path on MLA — the kernels are F32 only). The
    /// fused <c>kv_a_proj_with_mqa</c> tensor of shape
    /// <c>[kvLoraRank + qkRopeHeadDim, hidden]</c> is split row-wise into a
    /// dense latent projection (rows <c>[0, kvLoraRank)</c>) and a dense
    /// rope-K projection (rows <c>[kvLoraRank, kvLoraRank+qkRopeHeadDim)</c>)
    /// so the forward path can RMSNorm just the latent slice without a
    /// stride-aware kernel.
    /// </summary>
    private static unsafe MlaLayerBuffers UploadMlaLayer(
        VulkanDevice device, VulkanDevice.Buffer staging,
        MlaLayerWeights mla, int hiddenSize, out long uploadedBytes)
    {
        uploadedBytes = 0;
        int qTotal = mla.NumHeads * (mla.QkNopeHeadDim + mla.QkRopeHeadDim);
        int kvBOut = mla.NumHeads * (mla.QkNopeHeadDim + mla.VHeadDim);

        VulkanDevice.Buffer? qAProj = null, qBProj = null, qProj = null, qALayernorm = null;
        if (mla.QLoraRank > 0)
        {
            qAProj = UploadFp32Matrix(device, staging, mla.QAProj, mla.QLoraRank, hiddenSize, out long qABytes);
            qBProj = UploadFp32Matrix(device, staging, mla.QBProj, qTotal, mla.QLoraRank, out long qBBytes);
            qALayernorm = UploadNormVec(device, staging, mla.QALayernormWeight!);
            uploadedBytes += qABytes + qBBytes + (long)mla.QLoraRank * sizeof(float);
        }
        else
        {
            qProj = UploadFp32Matrix(device, staging, mla.QProj, qTotal, hiddenSize, out long qPBytes);
            uploadedBytes += qPBytes;
        }

        // Split kv_a_proj_with_mqa row-wise. Rows are contiguous in row-major
        // [output_dim, input_dim] storage, so the latent block sits at byte
        // offset 0 and the rope-K block at kvLoraRank * hidden * 4.
        long latentRowsBytes = (long)mla.KvLoraRank * hiddenSize * sizeof(float);
        var kvALatent = UploadFp32Matrix(device, staging,
            mla.KvAProjWithMqa, mla.KvLoraRank, hiddenSize, out long latentBytes);
        nint kPePtr = mla.KvAProjWithMqa + (nint)latentRowsBytes;
        var kvAKPe = UploadFp32Matrix(device, staging,
            kPePtr, mla.QkRopeHeadDim, hiddenSize, out long kPeBytes);
        uploadedBytes += latentBytes + kPeBytes;

        var kvALayernorm = UploadNormVec(device, staging, mla.KvALayernormWeight);
        uploadedBytes += (long)mla.KvLoraRank * sizeof(float);

        var kvBProj = UploadFp32Matrix(device, staging,
            mla.KvBProj, kvBOut, mla.KvLoraRank, out long kvBBytes);
        uploadedBytes += kvBBytes;

        return new MlaLayerBuffers(
            qAProj, qALayernorm, qBProj, qProj,
            kvALatent, kvAKPe, kvALayernorm, kvBProj,
            mla.NumHeads, mla.QkNopeHeadDim, mla.QkRopeHeadDim, mla.VHeadDim,
            mla.QLoraRank, mla.KvLoraRank, hiddenSize);
    }

    /// <summary>
    /// Uploads the MoE-specific weights for one layer. The router gate goes into its own
    /// buffer; per-routed-expert <c>W1</c>/<c>W2</c>/<c>W3</c> are <i>packed</i> into one
    /// contiguous F32 device bank per projection so the indexed matmul kernel can address
    /// any expert via a single descriptor binding plus a per-row index lookup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Quant policy.</b> The router gate, the per-shared-expert gate/up/down, and the
    /// optional Qwen1.5-MoE shared-expert sigmoid gate honour the optional Q8_0 overlay on
    /// <see cref="MoeLayerWeights"/> (see <see cref="MoeLayerWeights.GateQ8Ptr"/> et al.):
    /// when the overlay is set and the contraction axis is a multiple of 32 the raw Q8_0
    /// blocks are uploaded verbatim and the forward pass dispatches via
    /// <c>matmul_q8_0</c> / <c>matmul_q8_0_gemm</c>. Otherwise (production loaders never
    /// set the overlay today, so this is the default) the F32 source is uploaded and the
    /// forward pass uses <c>matmul_f32</c>. The per-routed-expert bank tensors stay F32 in
    /// every mode — the indexed-matmul kernel is F32-only in tree, no Q8_0 variant exists.
    /// </para>
    /// </remarks>
    private static unsafe MoeLayerBuffers UploadMoeLayer(
        VulkanDevice device, MoeLayerWeights moe, out long uploadedBytes)
    {
        uploadedBytes = 0;
        int hidden = moe.HiddenSize;
        int interm = moe.IntermediateSize;
        int numE = moe.NumExperts;
        int numShared = moe.NumSharedExperts;
        int sharedI = moe.SharedIntermediateSize;
        bool hasShared = moe.HasSharedExpert;

        // Two-mode byte sizes for the quant-overlayable projections (gate, routed
        // expert banks, per-shared-expert gate/up/down, shared-expert sigmoid gate).
        // `*KeepQuant` is true when the overlay declares a supported quant format (Q8_0
        // or Q4_K) AND the contraction axis is aligned to that format's group size.
        bool gateKeepQuant = MoeOverlayKeepsQuantized(moe.GateQuantTypeOverlay, hidden);
        long gateBytes = MoeOverlayUploadBytes(moe.GateQuantTypeOverlay, numE, hidden);

        QuantizationType routedW1Qt = MoeRoutedRawDeviceQuantType(device, moe.GateExpsRaw, moe.GateExpsRawQt, moe.GateExpsMDim, moe.GateExpsKDim, interm, hidden);
        QuantizationType routedW2Qt = MoeRoutedRawDeviceQuantType(device, moe.DownExpsRaw, moe.DownExpsRawQt, moe.DownExpsMDim, moe.DownExpsKDim, hidden, interm);
        QuantizationType routedW3Qt = MoeRoutedRawDeviceQuantType(device, moe.UpExpsRaw, moe.UpExpsRawQt, moe.UpExpsMDim, moe.UpExpsKDim, interm, hidden);
        long perExpertW1Bytes = routedW1Qt != QuantizationType.F32
            ? MoeOverlayUploadBytes(routedW1Qt, interm, hidden)
            : (long)interm * hidden * sizeof(float);
        long perExpertW2Bytes = routedW2Qt != QuantizationType.F32
            ? MoeOverlayUploadBytes(routedW2Qt, hidden, interm)
            : (long)hidden * interm * sizeof(float);
        long perExpertW3Bytes = routedW3Qt != QuantizationType.F32
            ? MoeOverlayUploadBytes(routedW3Qt, interm, hidden)
            : (long)interm * hidden * sizeof(float);

        bool sharedW1KeepQuant = hasShared && MoeOverlayKeepsQuantized(moe.SharedExpertProjQuantTypeOverlay, hidden);
        bool sharedW2KeepQuant = hasShared && MoeOverlayKeepsQuantized(moe.SharedExpertProjQuantTypeOverlay, sharedI);
        bool sharedW3KeepQuant = sharedW1KeepQuant;
        long perSharedW1Bytes = hasShared
            ? MoeOverlayUploadBytes(moe.SharedExpertProjQuantTypeOverlay, sharedI, hidden)
            : 0;
        long perSharedW2Bytes = hasShared
            ? MoeOverlayUploadBytes(moe.SharedExpertProjQuantTypeOverlay, hidden, sharedI)
            : 0;
        long perSharedW3Bytes = perSharedW1Bytes;

        // Stage sized to the largest single per-expert matrix OR the gate row,
        // whichever is bigger. The bank-pack copies one expert at a time so
        // we never need to stage the full bank at once. Shared-expert
        // matrices are sized off sharedIntermediate which may be larger than
        // the routed intermediate (DeepSeek-V2-Lite: shared==3*moe_intermediate),
        // so include them in the staging bound.
        long stageBytes = Math.Max(gateBytes, Math.Max(perExpertW1Bytes, perExpertW2Bytes));
        if (hasShared)
            stageBytes = Math.Max(stageBytes, Math.Max(perSharedW1Bytes, perSharedW2Bytes));
        using var stage = device.Allocate(stageBytes);

        // ── Router gate ──────────────────────────────────────────────
        VulkanDevice.Buffer gate;
        QuantizationType gateDeviceQt;
        if (gateKeepQuant)
        {
            // Raw quant-block upload — same on-device byte layout as VulkanWeights so the
            // matching matmul kernel (Q8_0 or Q4_K) reads it directly.
            gate = device.AllocateDeviceLocal(gateBytes);
            UploadRawBytes(device, stage, moe.GateQ8Ptr, gateBytes, gate);
            gateDeviceQt = moe.GateQuantTypeOverlay;
        }
        else
        {
            gate = device.AllocateDeviceLocal(gateBytes);
            VulkanApi.vkMapMemory(device.Handle, stage.Memory, 0, (ulong)gateBytes, 0, out nint gateMapped)
                .ThrowOnError("vkMapMemory UploadMoeLayer gate");
            try
            {
                moe.Gate.AsSpan().CopyTo(new Span<float>((void*)gateMapped, moe.Gate.Length));
            }
            finally
            {
                VulkanApi.vkUnmapMemory(device.Handle, stage.Memory);
            }
            device.CopyBufferSynchronous(stage, gate, (ulong)gateBytes);
            gateDeviceQt = QuantizationType.F32;
        }
        uploadedBytes += gateBytes;

        // ── Bank packing (per-routed-expert, F32 only) ───────────────
        long w1BankBytes = perExpertW1Bytes * numE;
        long w2BankBytes = perExpertW2Bytes * numE;
        long w3BankBytes = perExpertW3Bytes * numE;
        var w1Bank = device.AllocateDeviceLocal(w1BankBytes);
        var w2Bank = device.AllocateDeviceLocal(w2BankBytes);
        var w3Bank = device.AllocateDeviceLocal(w3BankBytes);

        for (int e = 0; e < numE; e++)
        {
            if (routedW1Qt != QuantizationType.F32)
                UploadRawBankSlot(device, stage, moe.GateExpsRaw + (nint)((long)e * perExpertW1Bytes), perExpertW1Bytes, w1Bank, (long)e * perExpertW1Bytes);
            else
                UploadExpertBankSlot(device, stage, moe.W1[e], perExpertW1Bytes, w1Bank, (long)e * perExpertW1Bytes);

            if (routedW2Qt != QuantizationType.F32)
                UploadRawBankSlot(device, stage, moe.DownExpsRaw + (nint)((long)e * perExpertW2Bytes), perExpertW2Bytes, w2Bank, (long)e * perExpertW2Bytes);
            else
                UploadExpertBankSlot(device, stage, moe.W2[e], perExpertW2Bytes, w2Bank, (long)e * perExpertW2Bytes);

            if (routedW3Qt != QuantizationType.F32)
                UploadRawBankSlot(device, stage, moe.UpExpsRaw + (nint)((long)e * perExpertW3Bytes), perExpertW3Bytes, w3Bank, (long)e * perExpertW3Bytes);
            else
                UploadExpertBankSlot(device, stage, moe.W3[e], perExpertW3Bytes, w3Bank, (long)e * perExpertW3Bytes);
        }
        uploadedBytes += w1BankBytes + w2BankBytes + w3BankBytes;

        // ── Shared-expert per-expert buffers (separate buffers, NOT a packed bank — the
        //    matmul kernel reads its weight buffer from offset 0). Each shared expert
        //    gets its own three device buffers in the same quant mode the overlay
        //    selects: Q8_0 / Q4_K raw blocks dispatched via the matching kernel, or F32
        //    dispatched via matmul_f32. Mixed quant/F32 across W1/W2/W3 IS allowed here
        //    on a per-axis basis — the contraction axes differ (W1/W3 contract along
        //    hidden, W2 along sharedIntermediate) so a single overlay quant type with
        //    per-axis MoeOverlayKeepsQuantized gating is the right granularity (e.g. an
        //    overlay declaring Q4_K survives on W1/W3 if hidden % 256 == 0 but falls back
        //    to F32 on W2 if sharedI % 256 != 0). ─────────────────────────────────────
        VulkanDevice.Buffer[]? sharedW1 = null, sharedW2 = null, sharedW3 = null;
        QuantizationType sharedW1Qt = QuantizationType.F32;
        QuantizationType sharedW2Qt = QuantizationType.F32;
        QuantizationType sharedW3Qt = QuantizationType.F32;
        if (hasShared)
        {
            sharedW1 = new VulkanDevice.Buffer[numShared];
            sharedW2 = new VulkanDevice.Buffer[numShared];
            sharedW3 = new VulkanDevice.Buffer[numShared];
            sharedW1Qt = sharedW1KeepQuant ? moe.SharedExpertProjQuantTypeOverlay : QuantizationType.F32;
            sharedW2Qt = sharedW2KeepQuant ? moe.SharedExpertProjQuantTypeOverlay : QuantizationType.F32;
            sharedW3Qt = sharedW3KeepQuant ? moe.SharedExpertProjQuantTypeOverlay : QuantizationType.F32;

            for (int s = 0; s < numShared; s++)
            {
                sharedW1[s] = device.AllocateDeviceLocal(perSharedW1Bytes);
                sharedW2[s] = device.AllocateDeviceLocal(perSharedW2Bytes);
                sharedW3[s] = device.AllocateDeviceLocal(perSharedW3Bytes);

                if (sharedW1KeepQuant)
                    UploadRawBytes(device, stage, moe.SharedGateProjQ8Ptrs![s], perSharedW1Bytes, sharedW1[s]);
                else
                    UploadExpertBankSlot(device, stage, moe.SharedGateProj[s], perSharedW1Bytes, sharedW1[s], 0);

                if (sharedW2KeepQuant)
                    UploadRawBytes(device, stage, moe.SharedDownProjQ8Ptrs![s], perSharedW2Bytes, sharedW2[s]);
                else
                    UploadExpertBankSlot(device, stage, moe.SharedDownProj[s], perSharedW2Bytes, sharedW2[s], 0);

                if (sharedW3KeepQuant)
                    UploadRawBytes(device, stage, moe.SharedUpProjQ8Ptrs![s], perSharedW3Bytes, sharedW3[s]);
                else
                    UploadExpertBankSlot(device, stage, moe.SharedUpProj[s], perSharedW3Bytes, sharedW3[s], 0);
            }
            uploadedBytes += (long)numShared * (perSharedW1Bytes + perSharedW2Bytes + perSharedW3Bytes);
        }

        // Optional Qwen1.5-MoE per-token sigmoid gate. Uploaded as a [1, hidden] device
        // buffer so the matmul kernel (M=1) can produce per-token gate logits in one
        // dispatch. Honours the quant overlay (contraction axis = hidden).
        VulkanDevice.Buffer? sharedExpertGate = null;
        QuantizationType sharedExpertGateDeviceQt = QuantizationType.F32;
        if (moe.SharedExpertGate is not null)
        {
            bool sgKeepQuant = MoeOverlayKeepsQuantized(moe.SharedExpertGateQuantTypeOverlay, hidden);
            if (sgKeepQuant)
            {
                long sgBytes = Dequantize.RowByteSize(hidden, moe.SharedExpertGateQuantTypeOverlay); // M=1
                sharedExpertGate = device.AllocateDeviceLocal(sgBytes);
                UploadRawBytes(device, stage, moe.SharedExpertGateQ8Ptr, sgBytes, sharedExpertGate);
                sharedExpertGateDeviceQt = moe.SharedExpertGateQuantTypeOverlay;
                uploadedBytes += sgBytes;
            }
            else
            {
                sharedExpertGate = UploadNormVec(device, stage, moe.SharedExpertGate);
                sharedExpertGateDeviceQt = QuantizationType.F32;
                uploadedBytes += (long)moe.SharedExpertGate.Length * sizeof(float);
            }
        }

        return new MoeLayerBuffers(gate, gateDeviceQt, w1Bank, w2Bank, w3Bank,
            routedW1Qt,
            routedW2Qt,
            routedW3Qt,
            moe.NumExperts, moe.NumExpertsPerTok,
            moe.HiddenSize, moe.IntermediateSize, moe.NormTopKProb,
            sharedW1, sharedW2, sharedW3,
            sharedW1Qt, sharedW2Qt, sharedW3Qt,
            sharedIntermediateSize: hasShared ? sharedI : 0,
            numSharedExperts: hasShared ? numShared : 0,
            sharedExpertGate: sharedExpertGate,
            sharedExpertGateDeviceQt: sharedExpertGateDeviceQt);
    }

    /// <summary>
    /// Uploads a Gemma-4 MoE layer's experts by HOST-dequantising the fused
    /// <c>gate_up</c> bank (Q4_K) into two F32 banks (gate → W1, up → W3) and the
    /// <c>down</c> bank (Q5_1) into an F32 W2 bank PRE-SCALED per expert by
    /// <c>ffn_down_exps.scale[e]</c> (folds Gemma-4 op #14 into the weight, so the
    /// downstream weighted-scatter need not carry the per-expert scale). The
    /// router gate stays F32. Result is a standard F32 <see cref="MoeLayerBuffers"/>
    /// the existing <c>moe_indexed_matmul_f32</c> kernel runs unchanged.
    ///
    /// <para>The fused tensor stores, per expert, a <c>[2*Ie, hidden]</c> slab:
    /// rows <c>[0, Ie)</c> = gate, rows <c>[Ie, 2*Ie)</c> = up, with a per-expert
    /// stride of <see cref="Gemma4LayerWeights.GateUpExpsRowBytes"/>. The CPU loader
    /// pre-offsets <see cref="MoeLayerWeights.UpExpsRaw"/> to the up rows, so gate
    /// and up share the same per-expert stride from their respective bases.</para>
    ///
    /// <para><b>Memory:</b> F32 dequant-on-load is the correctness / synthetic-fixture
    /// path. The real 26B (128 experts × Ie × hidden F32 per layer) is impractical
    /// this way — that is what the deferred Q4_K + Q5_1 indexed-MoE matmul shaders
    /// (keep experts quantized on device) are for.</para>
    /// </summary>
    private static unsafe MoeLayerBuffers UploadGemma4MoeLayer(
        VulkanDevice device, MoeLayerWeights moe, Gemma4LayerWeights g4, out long uploadedBytes)
    {
        uploadedBytes = 0;
        int hidden = moe.HiddenSize;
        int interm = moe.IntermediateSize;
        int numE = moe.NumExperts;

        long perGateUpElems = (long)interm * hidden;   // W1 (gate) / W3 (up) per expert
        long perDownElems = (long)hidden * interm;     // W2 (down) per expert
        long perGateUpBytes = perGateUpElems * sizeof(float);
        long perDownBytes = perDownElems * sizeof(float);
        long gateRouterBytes = (long)numE * hidden * sizeof(float);

        long stageBytes = Math.Max(gateRouterBytes, Math.Max(perGateUpBytes, perDownBytes));
        using var stage = device.Allocate(stageBytes);

        // ── Router gate (F32 [numExperts, hidden]) ───────────────────
        var gate = device.AllocateDeviceLocal(gateRouterBytes);
        VulkanApi.vkMapMemory(device.Handle, stage.Memory, 0, (ulong)gateRouterBytes, 0, out nint gateMapped)
            .ThrowOnError("vkMapMemory UploadGemma4MoeLayer gate");
        try
        {
            moe.Gate.AsSpan().CopyTo(new Span<float>((void*)gateMapped, moe.Gate.Length));
        }
        finally { VulkanApi.vkUnmapMemory(device.Handle, stage.Memory); }
        device.CopyBufferSynchronous(stage, gate, (ulong)gateRouterBytes);
        uploadedBytes += gateRouterBytes;

        // ── Expert banks (F32), one expert slice at a time ───────────
        var w1Bank = device.AllocateDeviceLocal(perGateUpBytes * numE);   // gate
        var w3Bank = device.AllocateDeviceLocal(perGateUpBytes * numE);   // up
        var w2Bank = device.AllocateDeviceLocal(perDownBytes * numE);     // down (pre-scaled)

        for (int e = 0; e < numE; e++)
        {
            nint gateSrc = moe.GateExpsRaw + (nint)(e * g4.GateUpExpsRowBytes);
            nint upSrc = moe.UpExpsRaw + (nint)(e * g4.GateUpExpsRowBytes);
            nint downSrc = moe.DownExpsRaw + (nint)(e * g4.DownExpsRowBytes);

            DequantAndUploadSlot(device, stage, gateSrc, perGateUpElems, moe.GateExpsRawQt,
                scale: 1.0f, w1Bank, (long)e * perGateUpBytes);
            DequantAndUploadSlot(device, stage, upSrc, perGateUpElems, moe.UpExpsRawQt,
                scale: 1.0f, w3Bank, (long)e * perGateUpBytes);
            // Fold the per-expert down scale into the weight (op #14).
            DequantAndUploadSlot(device, stage, downSrc, perDownElems, moe.DownExpsRawQt,
                scale: g4.DownExpertScale[e], w2Bank, (long)e * perDownBytes);
        }
        uploadedBytes += (perGateUpBytes * 2 + perDownBytes) * numE;

        return new MoeLayerBuffers(gate, QuantizationType.F32, w1Bank, w2Bank, w3Bank,
            QuantizationType.F32, QuantizationType.F32, QuantizationType.F32,
            moe.NumExperts, moe.NumExpertsPerTok,
            moe.HiddenSize, moe.IntermediateSize, moe.NormTopKProb,
            sharedW1: null, sharedW2: null, sharedW3: null,
            QuantizationType.F32, QuantizationType.F32, QuantizationType.F32,
            sharedIntermediateSize: 0, numSharedExperts: 0,
            sharedExpertGate: null, sharedExpertGateDeviceQt: QuantizationType.F32);
    }

    /// <summary>
    /// True when a Gemma-4 MoE layer's experts can stay QUANTIZED on device: the
    /// fused <c>gate_up</c> raw bank is Q4_K (and the contraction axis <c>hidden</c>
    /// is a multiple of 256) and the <c>down</c> raw bank is Q5_1, Q5_0 or Q8_0 (and
    /// the expert FF width is a multiple of 32). Q5_0 downs are repacked bit-exactly
    /// to Q5_1 at upload; Q8_0 downs stay Q8_0 with the per-expert scale folded into
    /// each block's fp16 <c>d</c> (real unsloth Q4_K_M conversions ship Q5_0/Q8_0
    /// downs, not Q5_1). When false the caller falls back to the F32 host-dequant
    /// path (<see cref="UploadGemma4MoeLayer"/>).
    /// </summary>
    private static bool Gemma4ExpertsKeepQuantized(MoeLayerWeights moe)
        => moe.GateExpsRaw != 0 && moe.UpExpsRaw != 0 && moe.DownExpsRaw != 0
        && moe.GateExpsRawQt == QuantizationType.Q4_K
        && moe.UpExpsRawQt == QuantizationType.Q4_K
        && moe.DownExpsRawQt is QuantizationType.Q5_1 or QuantizationType.Q5_0 or QuantizationType.Q8_0
        && (moe.HiddenSize % 256) == 0
        && (moe.IntermediateSize % 32) == 0;

    /// <summary>
    /// Uploads a Gemma-4 MoE layer's experts KEPT QUANTIZED on device. The fused
    /// <c>gate_up</c> Q4_K bank is REPACKED into two contiguous Q4_K banks — W1 (gate,
    /// rows <c>[0, Ie)</c> of each expert slab) and W3 (up, rows <c>[Ie, 2*Ie)</c>) —
    /// and the <c>down</c> bank is copied contiguously into W2: Q5_1 verbatim, Q5_0
    /// repacked bit-exactly to Q5_1 (<see cref="ConvertQ5_0BlocksToQ5_1"/>), Q8_0
    /// verbatim with the per-expert scale folded into the block scales
    /// (<see cref="CopyQ8_0BlocksScaled"/>). For the Q5_1-dispatched banks the
    /// per-expert down scale <c>ffn_down_exps.scale[e]</c> is uploaded as an F32
    /// [numExperts] buffer and folded by the Q5_1 indexed-matmul shader (NOT
    /// pre-multiplied into the weight); Q8_0 banks are pre-folded and their dispatch
    /// ignores that buffer. The router gate stays F32.
    ///
    /// <para>This is the memory-viable path for the real 26B: experts occupy their
    /// GGUF-quantized footprint (~Q4_K + Q5_1) instead of being dequantised to F32
    /// at load (tens of GB). The matching <c>moe_indexed_matmul_q4_k_f32</c> /
    /// <c>moe_indexed_matmul_q5_1_f32</c> shaders dequant per-row in the inner loop.</para>
    ///
    /// <para>The fused tensor stores, per expert, a <c>[2*Ie, hidden]</c> Q4_K slab at
    /// per-expert stride <see cref="Gemma4LayerWeights.GateUpExpsRowBytes"/>; gate is the
    /// first <c>Ie</c> rows (<see cref="MoeLayerWeights.GateExpsRaw"/>), up the next
    /// <c>Ie</c> rows (<see cref="MoeLayerWeights.UpExpsRaw"/>, pre-offset by the loader).
    /// Each contiguous bank uses a per-expert stride of <c>Ie * rowBytes</c>.</para>
    /// </summary>
    private static unsafe MoeLayerBuffers UploadGemma4MoeLayerQuantized(
        VulkanDevice device, MoeLayerWeights moe, Gemma4LayerWeights g4,
        out VulkanDevice.Buffer downScaleBuffer, out long uploadedBytes)
    {
        uploadedBytes = 0;
        int hidden = moe.HiddenSize;
        int interm = moe.IntermediateSize;   // Ie
        int numE = moe.NumExperts;

        // Per-expert quantized slab byte sizes (one projection each):
        //   gate/up: Ie rows of hidden Q4_K       → Ie * rowBytes(Q4_K, hidden)
        //   down:    hidden rows of Ie (src type)  → hidden * rowBytes(srcQt, Ie)
        // Q5_0 downs are converted to Q5_1 on the way through staging (bit-exact:
        // d·(q−16) = d·q + m with m = −16·d, same 5-bit payload), so the DEVICE bank
        // size uses the device type, not the source type.
        QuantizationType downSrcQt = moe.DownExpsRawQt;
        QuantizationType downDevQt = downSrcQt == QuantizationType.Q5_0 ? QuantizationType.Q5_1 : downSrcQt;
        long gateUpRowBytes = Dequantize.RowByteSize(hidden, QuantizationType.Q4_K);
        long perGateUpBytes = (long)interm * gateUpRowBytes;
        long downSrcRowBytes = Dequantize.RowByteSize(interm, downSrcQt);
        long perDownSrcBytes = (long)hidden * downSrcRowBytes;
        long downDevRowBytes = Dequantize.RowByteSize(interm, downDevQt);
        long perDownDevBytes = (long)hidden * downDevRowBytes;
        long gateRouterBytes = (long)numE * hidden * sizeof(float);

        // The fused gate_up per-expert stride must equal 2 * gate (gate + up halves).
        if (g4.GateUpExpsRowBytes != perGateUpBytes * 2)
            throw new InvalidOperationException(
                $"Gemma-4 fused gate_up stride mismatch: GateUpExpsRowBytes={g4.GateUpExpsRowBytes} "
                + $"!= 2*{perGateUpBytes} (Ie={interm}, hidden={hidden}, rowBytes={gateUpRowBytes}).");
        if (g4.DownExpsRowBytes != perDownSrcBytes)
            throw new InvalidOperationException(
                $"Gemma-4 down stride mismatch: DownExpsRowBytes={g4.DownExpsRowBytes} != {perDownSrcBytes} "
                + $"(hidden={hidden}, Ie={interm}, srcQt={downSrcQt}, rowBytes={downSrcRowBytes}).");

        long stageBytes = Math.Max(gateRouterBytes, Math.Max(perGateUpBytes, perDownDevBytes));
        using var stage = device.Allocate(stageBytes);

        // ── Router gate (F32 [numExperts, hidden]) ───────────────────
        var gate = device.AllocateDeviceLocal(gateRouterBytes);
        VulkanApi.vkMapMemory(device.Handle, stage.Memory, 0, (ulong)gateRouterBytes, 0, out nint gateMapped)
            .ThrowOnError("vkMapMemory UploadGemma4MoeLayerQuantized gate");
        try
        {
            moe.Gate.AsSpan().CopyTo(new Span<float>((void*)gateMapped, moe.Gate.Length));
        }
        finally { VulkanApi.vkUnmapMemory(device.Handle, stage.Memory); }
        device.CopyBufferSynchronous(stage, gate, (ulong)gateRouterBytes);
        uploadedBytes += gateRouterBytes;

        // ── Quantized expert banks (Q4_K gate/up; Q5_1 or Q8_0 down) ─
        var w1Bank = device.AllocateDeviceLocal(perGateUpBytes * numE);   // gate Q4_K
        var w3Bank = device.AllocateDeviceLocal(perGateUpBytes * numE);   // up   Q4_K
        var w2Bank = device.AllocateDeviceLocal(perDownDevBytes * numE);  // down Q5_1/Q8_0

        long downBlocks = (long)hidden * (interm / 32);
        for (int e = 0; e < numE; e++)
        {
            nint gateSrc = moe.GateExpsRaw + (nint)(e * g4.GateUpExpsRowBytes);
            nint upSrc = moe.UpExpsRaw + (nint)(e * g4.GateUpExpsRowBytes);
            nint downSrc = moe.DownExpsRaw + (nint)(e * g4.DownExpsRowBytes);

            UploadRawBankSlot(device, stage, gateSrc, perGateUpBytes, w1Bank, (long)e * perGateUpBytes);
            UploadRawBankSlot(device, stage, upSrc, perGateUpBytes, w3Bank, (long)e * perGateUpBytes);
            switch (downSrcQt)
            {
                case QuantizationType.Q5_1:
                    UploadRawBankSlot(device, stage, downSrc, perDownDevBytes, w2Bank, (long)e * perDownDevBytes);
                    break;
                case QuantizationType.Q5_0:
                    // Bit-exact Q5_0 → Q5_1 repack; rides the validated Q5_1 shader
                    // (per-expert scale folded by the shader via downScaleBuffer).
                    UploadQ5_0SlotAsQ5_1(device, stage, downSrc, downBlocks, w2Bank, (long)e * perDownDevBytes);
                    break;
                case QuantizationType.Q8_0:
                    // Q8_0 rides the generic Q8_0 indexed-matmul kernel, which has no
                    // scale plumbing — fold ffn_down_exps.scale[e] (op #14) into each
                    // block's fp16 d here (≤2⁻¹¹ relative rounding, inside the gemma4
                    // GPU parity envelope). The forward MUST NOT apply the scale again.
                    UploadQ8_0SlotScaled(device, stage, downSrc, downBlocks, g4.DownExpertScale[e],
                        w2Bank, (long)e * perDownDevBytes);
                    break;
                default:
                    throw new InvalidOperationException($"Unexpected gemma4 down quant type {downSrcQt}.");
            }
        }
        uploadedBytes += (perGateUpBytes * 2 + perDownDevBytes) * numE;

        // ── Per-expert down scale (folded by the Q5_1 down shader, op #14).
        // For Q8_0 down banks the scale is already folded into the block scales
        // above and the Q8_0 dispatch ignores this buffer — kept for diagnostics. ──
        downScaleBuffer = UploadNormVec(device, stage, g4.DownExpertScale);
        uploadedBytes += (long)numE * sizeof(float);

        return new MoeLayerBuffers(gate, QuantizationType.F32, w1Bank, w2Bank, w3Bank,
            QuantizationType.Q4_K, downDevQt, QuantizationType.Q4_K,
            moe.NumExperts, moe.NumExpertsPerTok,
            moe.HiddenSize, moe.IntermediateSize, moe.NormTopKProb,
            sharedW1: null, sharedW2: null, sharedW3: null,
            QuantizationType.F32, QuantizationType.F32, QuantizationType.F32,
            sharedIntermediateSize: 0, numSharedExperts: 0,
            sharedExpertGate: null, sharedExpertGateDeviceQt: QuantizationType.F32);
    }

    /// <summary>
    /// Dequantises <paramref name="elems"/> contiguous elements at
    /// <paramref name="src"/> (format <paramref name="qt"/>) to F32 directly into
    /// the staging buffer, optionally multiplying every element by
    /// <paramref name="scale"/>, then copies the slice into <paramref name="bank"/>
    /// at <paramref name="dstOffsetBytes"/>. Used by the Gemma-4 F32 expert upload.
    /// </summary>
    private static unsafe void DequantAndUploadSlot(
        VulkanDevice device, VulkanDevice.Buffer stage,
        nint src, long elems, QuantizationType qt, float scale,
        VulkanDevice.Buffer bank, long dstOffsetBytes)
    {
        long bytes = elems * sizeof(float);
        VulkanApi.vkMapMemory(device.Handle, stage.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.DequantAndUploadSlot");
        try
        {
            var dst = new Span<float>((void*)mapped, checked((int)elems));
            Dequantize.ToFloat32(src, elems, qt, dst);
            if (scale != 1.0f)
                for (int i = 0; i < dst.Length; i++) dst[i] *= scale;
        }
        finally { VulkanApi.vkUnmapMemory(device.Handle, stage.Memory); }
        device.CopyBufferRangeSynchronous(stage, bank, srcOffset: 0, dstOffset: (ulong)dstOffsetBytes, size: (ulong)bytes);
    }

    /// <summary>True iff a Q8_0 MoE overlay can be kept on device as raw Q8_0 blocks —
    /// gated on the contraction-axis dim being a multiple of the Q8_0 group size (32).</summary>
    private static bool MoeOverlayKeepsQ8(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q8_0 && (contractionDim % 32) == 0;

    private static bool MoeRoutedRawKeepsQ8(
        nint raw, QuantizationType qt,
        int rawM, int rawK,
        int expectedM, int expectedK)
        => raw != 0
        && qt == QuantizationType.Q8_0
        && rawM == expectedM
        && rawK == expectedK
        && (expectedK % 32) == 0;

    private static QuantizationType MoeRoutedRawDeviceQuantType(
        VulkanDevice device,
        nint raw, QuantizationType qt,
        int rawM, int rawK,
        int expectedM, int expectedK)
    {
        if (MoeRoutedRawKeepsQ8(raw, qt, rawM, rawK, expectedM, expectedK))
            return QuantizationType.Q8_0;

        // Strategy C path: keep routed F16 expert banks raw only when the
        // coopmat grouped matmul can consume them. Otherwise upload F32 so
        // the existing indexed path remains the fallback.
        if (raw != 0
            && qt == QuantizationType.F16
            && device.HasCooperativeMatrix
            && rawM == expectedM
            && rawK == expectedK
            && (expectedK % 32) == 0)
            return QuantizationType.F16;

        return QuantizationType.F32;
    }

    /// <summary>True iff a Q4_K MoE overlay can be kept on device as raw Q4_K super-blocks
    /// — gated on the contraction-axis dim being a multiple of the Q4_K super-block size
    /// (256). Phase 1 of the K-quant work.</summary>
    private static bool MoeOverlayKeepsQ4K(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q4_K && (contractionDim % 256) == 0;

    /// <summary>True iff a Q5_K MoE overlay can be kept on device as raw Q5_K super-blocks
    /// — gated on the contraction-axis dim being a multiple of the Q5_K super-block size
    /// (256). Phase 1 sibling of <see cref="MoeOverlayKeepsQ4K"/>.</summary>
    private static bool MoeOverlayKeepsQ5K(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q5_K && (contractionDim % 256) == 0;

    /// <summary>True iff a Q6_K MoE overlay can be kept on device as raw Q6_K super-blocks
    /// — gated on the contraction-axis dim being a multiple of the Q6_K super-block size
    /// (256). Phase 1 sibling of <see cref="MoeOverlayKeepsQ4K"/> completing the K-quant
    /// MoE coverage.</summary>
    private static bool MoeOverlayKeepsQ6K(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q6_K && (contractionDim % 256) == 0;

    /// <summary>True iff an F16 MoE overlay can be kept on device as raw 2-byte F16
    /// elements — gated on the contraction-axis dim being a multiple of 2. Phase 8
    /// sibling of <see cref="MoeOverlayKeepsQ4K"/>.</summary>
    private static bool MoeOverlayKeepsF16(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.F16 && (contractionDim & 1) == 0;

    /// <summary>True iff a BF16 MoE overlay can be kept on device as raw 2-byte BF16
    /// elements — gated on the contraction-axis dim being a multiple of 2. Phase 8
    /// sibling of <see cref="MoeOverlayKeepsF16"/>.</summary>
    private static bool MoeOverlayKeepsBf16(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.BF16 && (contractionDim & 1) == 0;

    /// <summary>True iff the MoE overlay is one of the supported native dtypes
    /// (Q8_0 / Q4_K / Q5_K / Q6_K / F16 / BF16) AND the contraction axis is aligned
    /// to that format's group size — i.e. raw bytes can be kept on device verbatim
    /// and dispatched through the matching matmul kernel.</summary>
    private static bool MoeOverlayKeepsQuantized(QuantizationType qt, int contractionDim)
        => MoeOverlayKeepsQ8(qt, contractionDim)
        || MoeOverlayKeepsQ4K(qt, contractionDim)
        || MoeOverlayKeepsQ5K(qt, contractionDim)
        || MoeOverlayKeepsQ6K(qt, contractionDim)
        || MoeOverlayKeepsF16(qt, contractionDim)
        || MoeOverlayKeepsBf16(qt, contractionDim);

    /// <summary>Returns the on-device byte size for an MoE projection in its chosen
    /// storage form — raw Q-format / F16 / BF16 row-stride bytes when the overlay
    /// says so, otherwise F32.</summary>
    private static long MoeOverlayUploadBytes(
        QuantizationType qt, int outputDim, int contractionDim)
    {
        if (MoeOverlayKeepsQ8(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q8_0) * outputDim;
        if (MoeOverlayKeepsQ4K(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q4_K) * outputDim;
        if (MoeOverlayKeepsQ5K(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q5_K) * outputDim;
        if (MoeOverlayKeepsQ6K(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q6_K) * outputDim;
        if (MoeOverlayKeepsF16(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.F16) * outputDim;
        if (MoeOverlayKeepsBf16(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.BF16) * outputDim;
        return (long)outputDim * contractionDim * sizeof(float);
    }

    /// <summary>Copies <paramref name="bytes"/> raw bytes from <paramref name="srcPtr"/>
    /// through <paramref name="staging"/> into the device-local <paramref name="dst"/>.</summary>
    private static unsafe void UploadRawBytes(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long bytes, VulkanDevice.Buffer dst)
    {
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadMoeLayer raw");
        try
        {
            new ReadOnlySpan<byte>((void*)srcPtr, checked((int)bytes))
                .CopyTo(new Span<byte>((void*)mapped, checked((int)bytes)));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, dst, (ulong)bytes);
    }

    private static unsafe void UploadRawBankSlot(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long bytes, VulkanDevice.Buffer bank, long dstOffset)
    {
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadRawBankSlot");
        try
        {
            new ReadOnlySpan<byte>((void*)srcPtr, checked((int)bytes))
                .CopyTo(new Span<byte>((void*)mapped, checked((int)bytes)));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferRangeSynchronous(staging, bank, srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);
    }

    /// <summary>
    /// Converts <paramref name="blockCount"/> Q5_0 blocks (22 B: fp16 <c>d</c>, 4 B <c>qh</c>,
    /// 16 B <c>qs</c>; value = <c>d·(q−16)</c>) to Q5_1 blocks (24 B: fp16 <c>d</c>, fp16
    /// <c>m</c>, <c>qh</c>, <c>qs</c>; value = <c>d·q + m</c>) with <c>d' = d</c> and
    /// <c>m' = −16·d</c>. The 5-bit payload is copied verbatim and <c>−16·d</c> is exactly
    /// representable in fp16 (pure exponent shift), so both blocks dequantise to identical
    /// F32 values — a Q5_0 bank can ride the Q5_1 kernels with no dedicated Q5_0 shader.
    /// </summary>
    internal static unsafe void ConvertQ5_0BlocksToQ5_1(nint src, nint dst, long blockCount)
    {
        byte* s = (byte*)src;
        byte* d = (byte*)dst;
        for (long b = 0; b < blockCount; b++)
        {
            Half scale = *(Half*)s;
            *(Half*)d = scale;
            *(Half*)(d + 2) = (Half)(-16f * (float)scale);
            new ReadOnlySpan<byte>(s + 2, 20).CopyTo(new Span<byte>(d + 4, 20)); // qh + qs verbatim
            s += 22;
            d += 24;
        }
    }

    /// <summary>
    /// Copies <paramref name="blockCount"/> Q8_0 blocks (34 B: fp16 <c>d</c> + 32×int8)
    /// multiplying each block's <c>d</c> by <paramref name="scale"/> — folds the Gemma-4
    /// per-expert <c>ffn_down_exps.scale</c> (op #14) into the weight so the scale-less
    /// generic Q8_0 indexed-matmul kernel can serve the down projection. The fp16
    /// re-rounding of <c>d·scale</c> is ≤ 2⁻¹¹ relative.
    /// </summary>
    internal static unsafe void CopyQ8_0BlocksScaled(nint src, nint dst, long blockCount, float scale)
    {
        byte* s = (byte*)src;
        byte* d = (byte*)dst;
        for (long b = 0; b < blockCount; b++)
        {
            *(Half*)d = (Half)((float)*(Half*)s * scale);
            new ReadOnlySpan<byte>(s + 2, 32).CopyTo(new Span<byte>(d + 2, 32));
            s += 34;
            d += 34;
        }
    }

    /// <summary>
    /// Stages one per-expert Q5_0 down slab converted to Q5_1
    /// (<see cref="ConvertQ5_0BlocksToQ5_1"/>) and copies it into
    /// <paramref name="bank"/> at <paramref name="dstOffset"/>.
    /// </summary>
    private static unsafe void UploadQ5_0SlotAsQ5_1(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long blockCount, VulkanDevice.Buffer bank, long dstOffset)
    {
        long bytes = blockCount * 24; // Q5_1 block size
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadQ5_0SlotAsQ5_1");
        try
        {
            ConvertQ5_0BlocksToQ5_1(srcPtr, mapped, blockCount);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferRangeSynchronous(staging, bank, srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);
    }

    /// <summary>
    /// Stages one per-expert Q8_0 down slab with the per-expert scale folded into the
    /// block scales (<see cref="CopyQ8_0BlocksScaled"/>) and copies it into
    /// <paramref name="bank"/> at <paramref name="dstOffset"/>.
    /// </summary>
    private static unsafe void UploadQ8_0SlotScaled(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long blockCount, float scale, VulkanDevice.Buffer bank, long dstOffset)
    {
        long bytes = blockCount * 34; // Q8_0 block size
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadQ8_0SlotScaled");
        try
        {
            CopyQ8_0BlocksScaled(srcPtr, mapped, blockCount, scale);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferRangeSynchronous(staging, bank, srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);
    }

    /// <summary>
    /// Uploads one per-expert F32 matrix from an unmanaged source pointer
    /// into a slot of a packed bank buffer at <paramref name="dstOffset"/>.
    /// </summary>
    private static unsafe void UploadExpertBankSlot(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long bytes, VulkanDevice.Buffer bank, long dstOffset)
    {
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadExpertBankSlot");
        try
        {
            int elems = checked((int)(bytes / sizeof(float)));
            new ReadOnlySpan<float>((void*)srcPtr, elems)
                .CopyTo(new Span<float>((void*)mapped, elems));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferRangeSynchronous(staging, bank, srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);
    }

    /// <summary>
    /// Uploads a contiguous F32 row-major matrix from an unmanaged pointer to
    /// a device-local buffer via the supplied staging buffer. Used by the MLA
    /// path where every projection is F32 (no quant path on MLA today).
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadFp32Matrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, int outputDim, int inputDim, out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;
        long bytes = elems * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadFp32Matrix staging");
        try
        {
            new ReadOnlySpan<float>((void*)srcPtr, checked((int)elems))
                .CopyTo(new Span<float>((void*)mapped, checked((int)elems)));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        uploadedBytes = bytes;
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
