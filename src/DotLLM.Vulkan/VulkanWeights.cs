using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer F32 weight buffers on a Vulkan device. Mirrors
/// <c>DotLLM.Cuda.CudaWeights</c> but with a simpler storage model:
/// all weights are dequantized to FP32 at load time (the Vulkan kernel set
/// is F32-only in this wave — no quantized GEMV yet). Bias tensors are
/// uploaded as FP32 buffers; norm weights become FP32 device buffers.
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
    /// Three banks per layer:
    /// <list type="bullet">
    ///   <item><c>W1Bank</c> (<i>gate_proj</i>): <c>[numExperts, intermediate, hidden]</c></item>
    ///   <item><c>W2Bank</c> (<i>down_proj</i>): <c>[numExperts, hidden, intermediate]</c></item>
    ///   <item><c>W3Bank</c> (<i>up_proj</i>):   <c>[numExperts, intermediate, hidden]</c></item>
    /// </list>
    /// Plus the router gate <c>[numExperts, hidden]</c>. Mixtral-convention
    /// renormalisation (<c>NormTopKProb=true</c>) is hard-wired in this
    /// pass — the base loader does not yet surface this flag from the
    /// model config and Mixtral is the only MoE family supported here.
    /// Shared experts (DeepSeek-V2/V3) are out of scope.
    /// </para>
    /// </remarks>
    internal readonly struct MoeLayerBuffers
    {
        public readonly VulkanDevice.Buffer Gate;       // [numExperts, hidden]
        public readonly VulkanDevice.Buffer W1Bank;     // [numExperts, intermediate, hidden]
        public readonly VulkanDevice.Buffer W2Bank;     // [numExperts, hidden, intermediate]
        public readonly VulkanDevice.Buffer W3Bank;     // [numExperts, intermediate, hidden]

        public readonly int NumExperts;
        public readonly int NumExpertsPerTok;
        public readonly int HiddenSize;
        public readonly int IntermediateSize;

        /// <summary>
        /// When <c>true</c>, the top-k softmax kernel divides the selected
        /// weights by their sum before they reach the weighted-scatter
        /// combine. Hard-coded to <c>true</c> for Mixtral — matches the
        /// CPU <c>MoeSwiGluMlp</c> reference, which always renormalises.
        /// </summary>
        public readonly bool NormTopKProb;

        public MoeLayerBuffers(
            VulkanDevice.Buffer gate, VulkanDevice.Buffer w1, VulkanDevice.Buffer w2, VulkanDevice.Buffer w3,
            int numExperts, int numExpertsPerTok,
            int hiddenSize, int intermediateSize, bool normTopKProb)
        {
            Gate = gate;
            W1Bank = w1;
            W2Bank = w2;
            W3Bank = w3;
            NumExperts = numExperts;
            NumExpertsPerTok = numExpertsPerTok;
            HiddenSize = hiddenSize;
            IntermediateSize = intermediateSize;
            NormTopKProb = normTopKProb;
        }

        public void Dispose()
        {
            Gate.Dispose();
            W1Bank.Dispose();
            W2Bank.Dispose();
            W3Bank.Dispose();
        }
    }

    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;

        public readonly VulkanDevice.Buffer Q;
        public readonly VulkanDevice.Buffer K;
        public readonly VulkanDevice.Buffer V;
        public readonly VulkanDevice.Buffer O;
        public readonly int QOutputDim, QInputDim;
        public readonly int KOutputDim, KInputDim;
        public readonly int VOutputDim, VInputDim;
        public readonly int OOutputDim, OInputDim;

        public readonly VulkanDevice.Buffer? QBias, KBias, VBias, OBias;

        public readonly VulkanDevice.Buffer FfnNormWeight;

        public readonly VulkanDevice.Buffer Gate;
        public readonly VulkanDevice.Buffer Up;
        public readonly VulkanDevice.Buffer Down;
        public readonly int GateOutputDim, GateInputDim;
        public readonly int UpOutputDim, UpInputDim;
        public readonly int DownOutputDim, DownInputDim;

        public readonly VulkanDevice.Buffer? GateBias, UpBias, DownBias;

        /// <summary>
        /// Non-null when the layer uses a MoE FFN (Mixtral, Qwen-MoE).
        /// Forward routes the FFN through <c>RunMoeLayer</c> and the
        /// dense Gate/Up/Down slots above are unused (stub buffers).
        /// </summary>
        public readonly MoeLayerBuffers? Moe;

        public LayerBuffers(
            VulkanDevice.Buffer attnNorm,
            VulkanDevice.Buffer q, int qM, int qK,
            VulkanDevice.Buffer k, int kM, int kK,
            VulkanDevice.Buffer v, int vM, int vK,
            VulkanDevice.Buffer o, int oM, int oK,
            VulkanDevice.Buffer? qBias, VulkanDevice.Buffer? kBias, VulkanDevice.Buffer? vBias, VulkanDevice.Buffer? oBias,
            VulkanDevice.Buffer ffnNorm,
            VulkanDevice.Buffer gate, int gateM, int gateK,
            VulkanDevice.Buffer up, int upM, int upK,
            VulkanDevice.Buffer down, int downM, int downK,
            VulkanDevice.Buffer? gateBias, VulkanDevice.Buffer? upBias, VulkanDevice.Buffer? downBias,
            MoeLayerBuffers? moe = null)
        {
            AttnNormWeight = attnNorm;
            Q = q; QOutputDim = qM; QInputDim = qK;
            K = k; KOutputDim = kM; KInputDim = kK;
            V = v; VOutputDim = vM; VInputDim = vK;
            O = o; OOutputDim = oM; OInputDim = oK;
            QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
            FfnNormWeight = ffnNorm;
            Gate = gate; GateOutputDim = gateM; GateInputDim = gateK;
            Up = up; UpOutputDim = upM; UpInputDim = upK;
            Down = down; DownOutputDim = downM; DownInputDim = downK;
            GateBias = gateBias; UpBias = upBias; DownBias = downBias;
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
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    public long AllocatedBytes { get; private set; }

    private VulkanWeights(
        VulkanDevice device,
        VulkanDevice.Buffer tokenEmbed, int vocabSize, int hiddenSize,
        LayerBuffers[] layers,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, int outputM, int outputK,
        long allocatedBytes)
    {
        _device = device;
        TokenEmbedding = tokenEmbed;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        _layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputOutputDim = outputM;
        OutputInputDim = outputK;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads the given CPU-resident <see cref="TransformerWeights"/> to the
    /// Vulkan device. All quantized weights are dequantized to FP32 row-by-row
    /// into a pooled scratch buffer before upload; this keeps the host memory
    /// footprint bounded at one row per upload even when the whole model
    /// wouldn't fit dequantized in RAM.
    /// </summary>
    public static VulkanWeights Upload(VulkanDevice device, TransformerWeights weights, int numLayers)
    {
        long totalBytes = 0;

        // Token embedding table: [vocabSize, hiddenSize] FP32.
        var tokenEmbed = UploadMatrix(device, weights.TokenEmbedWeight, weights.TokenEmbedQuantType,
            weights.VocabSize, weights.HiddenSize);
        totalBytes += (long)weights.VocabSize * weights.HiddenSize * sizeof(float);

        var layerBuffers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];

            var attnNorm = UploadNormVec(device, lw.AttnNormWeight);

            var q = UploadMatrix(device, lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim);
            var k = UploadMatrix(device, lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim);
            var v = UploadMatrix(device, lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim);
            var o = UploadMatrix(device, lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim);

            var qBias = UploadOptionalVec(device, lw.QBias);
            var kBias = UploadOptionalVec(device, lw.KBias);
            var vBias = UploadOptionalVec(device, lw.VBias);
            var oBias = UploadOptionalVec(device, lw.OBias);

            var ffnNorm = UploadNormVec(device, lw.FfnNormWeight);

            // MoE layers replace the dense Gate/Up/Down with per-expert
            // banks (lw.Moe). Stub the dense slots with 64-byte buffers so
            // the LayerBuffers contract still holds — the forward pass
            // never dispatches a matmul against them on MoE layers.
            VulkanDevice.Buffer gate, up, down;
            VulkanDevice.Buffer? gateBias, upBias, downBias;
            MoeLayerBuffers? moe = null;
            if (lw.Moe is not null)
            {
                gate = device.Allocate(64);
                up = device.Allocate(64);
                down = device.Allocate(64);
                gateBias = upBias = downBias = null;
                // Shared experts (DeepSeek-V2/V3, Qwen1.5-MoE shared-expert
                // branch) are not represented in the base MoeLayerWeights
                // type yet — when they land, add a guard here.
                moe = UploadMoeLayer(device, lw.Moe, normTopKProb: true, out long moeBytes);
                totalBytes += moeBytes;
            }
            else
            {
                gate = UploadMatrix(device, lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim);
                up = UploadMatrix(device, lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim);
                down = UploadMatrix(device, lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim);
                gateBias = UploadOptionalVec(device, lw.GateBias);
                upBias = UploadOptionalVec(device, lw.UpBias);
                downBias = UploadOptionalVec(device, lw.DownBias);

                totalBytes += (long)lw.GateOutputDim * lw.GateInputDim * sizeof(float);
                totalBytes += (long)lw.UpOutputDim * lw.UpInputDim * sizeof(float);
                totalBytes += (long)lw.DownOutputDim * lw.DownInputDim * sizeof(float);
            }

            layerBuffers[i] = new LayerBuffers(
                attnNorm,
                q, lw.QOutputDim, lw.QInputDim,
                k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim,
                o, lw.OOutputDim, lw.OInputDim,
                qBias, kBias, vBias, oBias,
                ffnNorm,
                gate, lw.GateOutputDim, lw.GateInputDim,
                up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                gateBias, upBias, downBias,
                moe);

            totalBytes += (long)lw.QOutputDim * lw.QInputDim * sizeof(float);
            totalBytes += (long)lw.KOutputDim * lw.KInputDim * sizeof(float);
            totalBytes += (long)lw.VOutputDim * lw.VInputDim * sizeof(float);
            totalBytes += (long)lw.OOutputDim * lw.OInputDim * sizeof(float);
        }

        var outputNorm = UploadNormVec(device, weights.OutputNormWeight);
        var outputWeight = UploadMatrix(device, weights.OutputWeight, weights.OutputQuantType,
            weights.OutputOutputDim, weights.OutputInputDim);
        totalBytes += (long)weights.OutputOutputDim * weights.OutputInputDim * sizeof(float);

        return new VulkanWeights(
            device, tokenEmbed, weights.VocabSize, weights.HiddenSize,
            layerBuffers,
            outputNorm, outputWeight, weights.OutputOutputDim, weights.OutputInputDim,
            totalBytes);
    }

    private static VulkanDevice.Buffer UploadMatrix(VulkanDevice device, nint srcPtr, QuantizationType qt,
        int outputDim, int inputDim)
    {
        long elems = (long)outputDim * inputDim;
        var buf = device.Allocate(elems * sizeof(float));

        if (qt == QuantizationType.F32)
        {
            // Direct upload from mmap.
            unsafe
            {
                var srcSpan = new ReadOnlySpan<float>((void*)srcPtr, checked((int)elems));
                device.Upload(srcSpan, buf);
            }
            return buf;
        }

        // Dequantize row-by-row into a pooled scratch array (bounded host footprint).
        float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(inputDim);
        try
        {
            long rowBytes = Dequantize.RowByteSize(inputDim, qt);
            // Map once, write all rows, unmap. Faster than the generic
            // VulkanDevice.Upload helper which maps/unmaps per call.
            UploadRowsDequantized(device, buf, srcPtr, outputDim, inputDim, qt, rowBytes, scratch);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
        return buf;
    }

    private static unsafe void UploadRowsDequantized(
        VulkanDevice device, VulkanDevice.Buffer dst,
        nint srcPtr, int outputDim, int inputDim,
        QuantizationType qt, long rowBytes, float[] scratch)
    {
        long totalBytes = (long)outputDim * inputDim * sizeof(float);
        DotLLM.Vulkan.Interop.VulkanApi.vkMapMemory(device.Handle, dst.Memory, 0, (ulong)totalBytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadMatrix");
        try
        {
            float* d = (float*)mapped;
            for (int row = 0; row < outputDim; row++)
            {
                nint rowSrc = srcPtr + (nint)(row * rowBytes);
                Dequantize.ToFloat32(rowSrc, inputDim, qt, scratch.AsSpan(0, inputDim));
                new ReadOnlySpan<float>(scratch, 0, inputDim)
                    .CopyTo(new Span<float>(d + (long)row * inputDim, inputDim));
            }
        }
        finally
        {
            DotLLM.Vulkan.Interop.VulkanApi.vkUnmapMemory(device.Handle, dst.Memory);
        }
    }

    /// <summary>
    /// Uploads the MoE-specific weights for one layer. The router gate
    /// goes into its own buffer; per-expert <c>W1</c>/<c>W2</c>/<c>W3</c>
    /// are <i>packed</i> into one contiguous F32 device bank per
    /// projection so the indexed matmul kernel can address any expert via
    /// a single descriptor binding plus a per-row index lookup. Each bank
    /// is mapped once and every expert is memcpy'd into its slot — no
    /// staging buffer needed since all device buffers on this base are
    /// host-visible host-coherent.
    /// </summary>
    private static unsafe MoeLayerBuffers UploadMoeLayer(
        VulkanDevice device, MoeLayerWeights moe, bool normTopKProb, out long uploadedBytes)
    {
        uploadedBytes = 0;
        int hidden = moe.HiddenSize;
        int interm = moe.IntermediateSize;
        int numE = moe.NumExperts;

        long gateBytes = (long)numE * hidden * sizeof(float);
        long perExpertW1Bytes = (long)interm * hidden * sizeof(float);
        long perExpertW2Bytes = (long)hidden * interm * sizeof(float);
        long perExpertW3Bytes = perExpertW1Bytes;

        // ── Router gate ──────────────────────────────────────────────
        var gate = device.Allocate(gateBytes);
        device.Upload(moe.Gate.AsSpan(), gate);
        uploadedBytes += gateBytes;

        // ── Bank packing (per-expert) ────────────────────────────────
        long w1BankBytes = perExpertW1Bytes * numE;
        long w2BankBytes = perExpertW2Bytes * numE;
        long w3BankBytes = perExpertW3Bytes * numE;
        var w1Bank = device.Allocate(w1BankBytes);
        var w2Bank = device.Allocate(w2BankBytes);
        var w3Bank = device.Allocate(w3BankBytes);

        PackExpertBank(device, w1Bank, moe.W1, perExpertW1Bytes, numE);
        PackExpertBank(device, w2Bank, moe.W2, perExpertW2Bytes, numE);
        PackExpertBank(device, w3Bank, moe.W3, perExpertW3Bytes, numE);
        uploadedBytes += w1BankBytes + w2BankBytes + w3BankBytes;

        return new MoeLayerBuffers(gate, w1Bank, w2Bank, w3Bank,
            moe.NumExperts, moe.NumExpertsPerTok,
            moe.HiddenSize, moe.IntermediateSize, normTopKProb);
    }

    /// <summary>
    /// Packs <paramref name="numExperts"/> per-expert F32 matrices (each
    /// <paramref name="perExpertBytes"/> bytes pointed at by
    /// <paramref name="srcPtrs"/>) into a single contiguous device bank.
    /// One mapMemory call covers the whole bank, then each expert is
    /// memcpy'd into <c>bank[e]</c> at offset <c>e * perExpertBytes</c>.
    /// </summary>
    private static unsafe void PackExpertBank(
        VulkanDevice device, VulkanDevice.Buffer bank, nint[] srcPtrs,
        long perExpertBytes, int numExperts)
    {
        long totalBytes = perExpertBytes * numExperts;
        VulkanApi.vkMapMemory(device.Handle, bank.Memory, 0, (ulong)totalBytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.PackExpertBank");
        try
        {
            byte* dst = (byte*)mapped;
            for (int e = 0; e < numExperts; e++)
            {
                System.Buffer.MemoryCopy(
                    (void*)srcPtrs[e],
                    dst + (long)e * perExpertBytes,
                    perExpertBytes,
                    perExpertBytes);
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, bank.Memory);
        }
    }

    private static VulkanDevice.Buffer UploadNormVec(VulkanDevice device, float[] normWeight)
    {
        var buf = device.Allocate((long)normWeight.Length * sizeof(float));
        device.Upload(normWeight.AsSpan(), buf);
        return buf;
    }

    private static VulkanDevice.Buffer? UploadOptionalVec(VulkanDevice device, float[]? vec)
    {
        if (vec is null) return null;
        var buf = device.Allocate((long)vec.Length * sizeof(float));
        device.Upload(vec.AsSpan(), buf);
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
