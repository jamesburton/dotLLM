using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer weight buffers on a Vulkan device. Mirrors
/// <c>DotLLM.Cuda.CudaWeights</c> but with a two-mode storage model:
/// Q8_0 matrices are kept on device as raw 34-byte blocks when
/// <c>dequantToFp32=false</c> (default) so the <c>matmul_q8_0</c> /
/// <c>matmul_q8_0_gemm</c> kernels can read them directly — 4× less VRAM
/// and 4× less per-forward bandwidth vs the dequantised F32 path.
/// F16 and other quant types are still dequantised to FP32 at upload time
/// (those kernels don't exist yet); passing <c>dequantToFp32=true</c> forces
/// the legacy all-F32 path as a fallback. Bias and norm weights are always
/// FP32 device buffers (tiny, kernels consume FP32).
/// </summary>
internal sealed class VulkanWeights : IDisposable
{
    /// <summary>
    /// Per-layer device-resident MoE (Mixtral / Qwen-MoE) weight bundle.
    /// Per-expert weights are <i>packed</i> into one contiguous F32 device
    /// bank per projection so the indexed-matmul kernel can address any
    /// expert via a single descriptor binding plus a per-row index lookup.
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
    /// Qwen1.5-MoE's per-token sigmoid gate is intentionally NOT wired here —
    /// the upload guard rejects layers carrying a <c>SharedExpertGate</c>
    /// until a dedicated sigmoid + scalar-multiply kernel pair lands.
    /// </para>
    /// </remarks>
    internal readonly struct MoeLayerBuffers
    {
        public readonly VulkanDevice.Buffer Gate;       // [numExperts, hidden]
        public readonly VulkanDevice.Buffer W1Bank;     // [numExperts, intermediate, hidden]
        public readonly VulkanDevice.Buffer W2Bank;     // [numExperts, hidden, intermediate]
        public readonly VulkanDevice.Buffer W3Bank;     // [numExperts, intermediate, hidden]

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
            MlaLayerBuffers? mla = null,
            MoeLayerBuffers? moe = null)
        {
            AttnNormWeight = attnNorm;
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
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
            QBias?.Dispose(); KBias?.Dispose(); VBias?.Dispose(); OBias?.Dispose();
            FfnNormWeight.Dispose();
            Gate.Dispose(); Up.Dispose(); Down.Dispose();
            GateBias?.Dispose(); UpBias?.Dispose(); DownBias?.Dispose();
            Mla?.Dispose();
            Moe?.Dispose();
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
    /// <remarks>
    /// <para>
    /// Staging is sized to fit the largest single matrix in its <i>widest</i>
    /// on-host form (FP32 for non-Q8_0 matrices or when <paramref name="dequantToFp32"/>
    /// is true; raw Q8_0 bytes for Q8_0 matrices when kept on device).
    /// </para>
    /// </remarks>
    public static VulkanWeights Upload(
        VulkanDevice device, TransformerWeights weights, int numLayers,
        bool dequantToFp32 = false)
    {
        long totalBytes = 0;

        // Size the reusable staging buffer to the largest single weight upload
        // (in its on-device byte form).
        long stagingBytes = ComputeMaxUploadBytes(weights, numLayers, dequantToFp32);
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

        var layerBuffers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];

            var attnNorm = UploadNormVec(device, staging, lw.AttnNormWeight);
            totalBytes += (long)lw.AttnNormWeight.Length * sizeof(float);

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
                v = UploadMatrix(device, staging, lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim,
                    dequantToFp32, out vDeviceQt, out vBytes);
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

            // MoE layers replace the dense Gate/Up/Down with per-expert
            // banks (lw.Moe). Stub the dense slots with 64-byte buffers so
            // the LayerBuffers contract still holds — the forward pass
            // never dispatches a matmul against them on MoE layers.
            VulkanDevice.Buffer gate, up, down;
            QuantizationType gateDeviceQt, upDeviceQt, downDeviceQt;
            long gateBytes, upBytes, downBytes;
            VulkanDevice.Buffer? gateBias, upBias, downBias;
            if (lw.Moe is not null)
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
            if (lw.Moe is not null)
            {
                moe = UploadMoeLayer(device, lw.Moe, out long moeBytes);
                totalBytes += moeBytes;
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
                mla, moe);

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

    /// <summary>Returns true when the matrix will be kept on device as Q8_0 blocks.</summary>
    private static bool KeepQ8OnDevice(QuantizationType qt, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q8_0;

    /// <summary>Returns true when the matrix will be kept on device as Q4_K super-blocks
    /// (144 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q4_K super-block size (256). Phase 1 of the K-quant work — Q6_K is the
    /// remaining follow-up ticket and still dequantises to F32 at upload.</summary>
    private static bool KeepQ4KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q4_K && (inputDim % 256) == 0;

    /// <summary>Returns true when the matrix will be kept on device as Q5_K super-blocks
    /// (176 bytes per 256 elements). Gated on the contraction axis being a multiple of
    /// the Q5_K super-block size (256). Phase 1 sibling of <see cref="KeepQ4KOnDevice"/>.</summary>
    private static bool KeepQ5KOnDevice(QuantizationType qt, int inputDim, bool dequantToFp32)
        => !dequantToFp32 && qt == QuantizationType.Q5_K && (inputDim % 256) == 0;

    /// <summary>Returns the on-device storage quant type for a projection: Q8_0 / Q4_K /
    /// Q5_K / F32 depending on the source and the alignment constraints.</summary>
    private static QuantizationType DeviceQuantTypeFor(
        QuantizationType srcQt, int inputDim, bool dequantToFp32)
    {
        if (KeepQ8OnDevice(srcQt, dequantToFp32)) return QuantizationType.Q8_0;
        if (KeepQ4KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q4_K;
        if (KeepQ5KOnDevice(srcQt, inputDim, dequantToFp32)) return QuantizationType.Q5_K;
        return QuantizationType.F32;
    }

    private static long ComputeMaxUploadBytes(
        TransformerWeights weights, int numLayers, bool dequantToFp32)
    {
        long max = 0;
        max = Math.Max(max, UploadBytes(weights.VocabSize, weights.HiddenSize, weights.TokenEmbedQuantType, dequantToFp32: true));
        max = Math.Max(max, UploadBytes(weights.OutputOutputDim, weights.OutputInputDim, weights.OutputQuantType, dequantToFp32));
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];
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
        if (KeepQ4KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q4_K) * outputDim;
        if (KeepQ5KOnDevice(qt, inputDim, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q5_K) * outputDim;
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
    private static unsafe VulkanDevice.Buffer UploadMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool dequantToFp32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;

        // Raw quant-block upload — keeps the GGUF on-disk byte layout intact on device so
        // the matmul_q8_0 / matmul_q4_k / matmul_q5_k kernels can read it directly.
        // Mirrors the CPU path's mmap-backed layout.
        QuantizationType keepQt = DeviceQuantTypeFor(qt, inputDim, dequantToFp32);
        if (keepQt != QuantizationType.F32)
        {
            long rowBytes = Dequantize.RowByteSize(inputDim, keepQt);
            long bytes = rowBytes * outputDim;

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

        // Two-mode byte sizes for the quant-overlayable projections (gate, per-shared-
        // expert gate/up/down, shared-expert sigmoid gate). The per-routed-expert banks
        // are always F32 since the Vulkan indexed-matmul kernel is F32-only.
        // `*KeepQuant` is true when the overlay declares a supported quant format (Q8_0
        // or Q4_K) AND the contraction axis is aligned to that format's group size.
        bool gateKeepQuant = MoeOverlayKeepsQuantized(moe.GateQuantTypeOverlay, hidden);
        long gateBytes = MoeOverlayUploadBytes(moe.GateQuantTypeOverlay, numE, hidden);

        long perExpertW1Bytes = (long)interm * hidden * sizeof(float);
        long perExpertW2Bytes = (long)hidden * interm * sizeof(float);
        long perExpertW3Bytes = perExpertW1Bytes;

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
            UploadExpertBankSlot(device, stage, moe.W1[e], perExpertW1Bytes, w1Bank, (long)e * perExpertW1Bytes);
            UploadExpertBankSlot(device, stage, moe.W2[e], perExpertW2Bytes, w2Bank, (long)e * perExpertW2Bytes);
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
            moe.NumExperts, moe.NumExpertsPerTok,
            moe.HiddenSize, moe.IntermediateSize, moe.NormTopKProb,
            sharedW1, sharedW2, sharedW3,
            sharedW1Qt, sharedW2Qt, sharedW3Qt,
            sharedIntermediateSize: hasShared ? sharedI : 0,
            numSharedExperts: hasShared ? numShared : 0,
            sharedExpertGate: sharedExpertGate,
            sharedExpertGateDeviceQt: sharedExpertGateDeviceQt);
    }

    /// <summary>True iff a Q8_0 MoE overlay can be kept on device as raw Q8_0 blocks —
    /// gated on the contraction-axis dim being a multiple of the Q8_0 group size (32).</summary>
    private static bool MoeOverlayKeepsQ8(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q8_0 && (contractionDim % 32) == 0;

    /// <summary>True iff a Q4_K MoE overlay can be kept on device as raw Q4_K super-blocks
    /// — gated on the contraction-axis dim being a multiple of the Q4_K super-block size
    /// (256). Phase 1 of the K-quant work; Q6_K follow-up ticket.</summary>
    private static bool MoeOverlayKeepsQ4K(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q4_K && (contractionDim % 256) == 0;

    /// <summary>True iff a Q5_K MoE overlay can be kept on device as raw Q5_K super-blocks
    /// — gated on the contraction-axis dim being a multiple of the Q5_K super-block size
    /// (256). Phase 1 sibling of <see cref="MoeOverlayKeepsQ4K"/>.</summary>
    private static bool MoeOverlayKeepsQ5K(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q5_K && (contractionDim % 256) == 0;

    /// <summary>True iff the MoE overlay is one of the supported quantised formats
    /// (Q8_0 / Q4_K / Q5_K) AND the contraction axis is aligned to that format's group
    /// size — i.e. raw blocks can be kept on device verbatim and dispatched through the
    /// matching matmul kernel.</summary>
    private static bool MoeOverlayKeepsQuantized(QuantizationType qt, int contractionDim)
        => MoeOverlayKeepsQ8(qt, contractionDim)
        || MoeOverlayKeepsQ4K(qt, contractionDim)
        || MoeOverlayKeepsQ5K(qt, contractionDim);

    /// <summary>Returns the on-device byte size for an MoE projection in its chosen
    /// storage form — raw Q8_0 / Q4_K / Q5_K row-stride bytes when the overlay says so,
    /// otherwise F32.</summary>
    private static long MoeOverlayUploadBytes(
        QuantizationType qt, int outputDim, int contractionDim)
    {
        if (MoeOverlayKeepsQ8(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q8_0) * outputDim;
        if (MoeOverlayKeepsQ4K(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q4_K) * outputDim;
        if (MoeOverlayKeepsQ5K(qt, contractionDim))
            return Dequantize.RowByteSize(contractionDim, QuantizationType.Q5_K) * outputDim;
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
