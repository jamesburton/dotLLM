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
    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;

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
            VulkanDevice.Buffer? gateBias, VulkanDevice.Buffer? upBias, VulkanDevice.Buffer? downBias)
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

            var q = UploadMatrix(device, staging, lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim,
                dequantToFp32, out var qDeviceQt, out long qBytes);
            var k = UploadMatrix(device, staging, lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim,
                dequantToFp32, out var kDeviceQt, out long kBytes);
            var v = UploadMatrix(device, staging, lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim,
                dequantToFp32, out var vDeviceQt, out long vBytes);
            var o = UploadMatrix(device, staging, lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim,
                dequantToFp32, out var oDeviceQt, out long oBytes);

            var qBias = UploadOptionalVec(device, staging, lw.QBias);
            var kBias = UploadOptionalVec(device, staging, lw.KBias);
            var vBias = UploadOptionalVec(device, staging, lw.VBias);
            var oBias = UploadOptionalVec(device, staging, lw.OBias);

            var ffnNorm = UploadNormVec(device, staging, lw.FfnNormWeight);
            totalBytes += (long)lw.FfnNormWeight.Length * sizeof(float);

            var gate = UploadMatrix(device, staging, lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim,
                dequantToFp32, out var gateDeviceQt, out long gateBytes);
            var up = UploadMatrix(device, staging, lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim,
                dequantToFp32, out var upDeviceQt, out long upBytes);
            var down = UploadMatrix(device, staging, lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim,
                dequantToFp32, out var downDeviceQt, out long downBytes);

            var gateBias = UploadOptionalVec(device, staging, lw.GateBias);
            var upBias = UploadOptionalVec(device, staging, lw.UpBias);
            var downBias = UploadOptionalVec(device, staging, lw.DownBias);

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
                gateBias, upBias, downBias);

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
        }
        return max;
    }

    private static long UploadBytes(int outputDim, int inputDim, QuantizationType qt, bool dequantToFp32)
    {
        long elems = (long)outputDim * inputDim;
        if (KeepQ8OnDevice(qt, dequantToFp32))
            return Dequantize.RowByteSize(inputDim, QuantizationType.Q8_0) * outputDim;
        return elems * sizeof(float);
    }

    /// <summary>
    /// Uploads a single weight matrix. When <paramref name="dequantToFp32"/> is false and
    /// <paramref name="qt"/> is Q8_0 the raw Q8_0 block bytes are copied to device memory
    /// and the returned <paramref name="deviceQuantType"/> is <see cref="QuantizationType.Q8_0"/>.
    /// Otherwise the source is dequantised to FP32 before upload and
    /// <paramref name="deviceQuantType"/> is <see cref="QuantizationType.F32"/>.
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim,
        bool dequantToFp32,
        out QuantizationType deviceQuantType,
        out long uploadedBytes)
    {
        long elems = (long)outputDim * inputDim;

        if (KeepQ8OnDevice(qt, dequantToFp32))
        {
            // Raw Q8_0 blob upload — mirrors the CPU path's mmap-backed layout.
            long rowBytes = Dequantize.RowByteSize(inputDim, QuantizationType.Q8_0);
            long bytes = rowBytes * outputDim;

            var buf = device.AllocateDeviceLocal(bytes);
            VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
                .ThrowOnError("vkMapMemory VulkanWeights.UploadMatrix staging (Q8_0)");
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

            deviceQuantType = QuantizationType.Q8_0;
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

    public void Dispose()
    {
        TokenEmbedding.Dispose();
        OutputNormWeight.Dispose();
        OutputWeight.Dispose();
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Dispose();
    }
}
