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
            VulkanDevice.Buffer? gateBias, VulkanDevice.Buffer? upBias, VulkanDevice.Buffer? downBias)
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
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
            QBias?.Dispose(); KBias?.Dispose(); VBias?.Dispose(); OBias?.Dispose();
            FfnNormWeight.Dispose();
            Gate.Dispose(); Up.Dispose(); Down.Dispose();
            GateBias?.Dispose(); UpBias?.Dispose(); DownBias?.Dispose();
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

            var gate = UploadMatrix(device, lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim);
            var up = UploadMatrix(device, lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim);
            var down = UploadMatrix(device, lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim);

            var gateBias = UploadOptionalVec(device, lw.GateBias);
            var upBias = UploadOptionalVec(device, lw.UpBias);
            var downBias = UploadOptionalVec(device, lw.DownBias);

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
                gateBias, upBias, downBias);

            totalBytes += (long)lw.QOutputDim * lw.QInputDim * sizeof(float);
            totalBytes += (long)lw.KOutputDim * lw.KInputDim * sizeof(float);
            totalBytes += (long)lw.VOutputDim * lw.VInputDim * sizeof(float);
            totalBytes += (long)lw.OOutputDim * lw.OInputDim * sizeof(float);
            totalBytes += (long)lw.GateOutputDim * lw.GateInputDim * sizeof(float);
            totalBytes += (long)lw.UpOutputDim * lw.UpInputDim * sizeof(float);
            totalBytes += (long)lw.DownOutputDim * lw.DownInputDim * sizeof(float);
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
