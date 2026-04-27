using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer weight buffers on a Vulkan device for the Mamba-3 SISO model. Mirrors
/// <see cref="VulkanNemotronHWeights"/>'s upload/lifetime pattern but with a single
/// per-layer projection bundle (the Mamba-3 mixer tensors) rather than the three
/// hybrid sub-bundles of NemotronH.
/// </summary>
/// <remarks>
/// <para>
/// <b>F32-only constraint (first cut, issue #5):</b> every weight is consumed as F32 by
/// the Vulkan kernels. The CPU-side <see cref="Mamba3WeightLoader"/> already requires F32
/// source tensors (see <c>Mamba3WeightLoader.ResolveRequired</c>), so the upload path here
/// is a straight host-to-device copy — no dequantisation, no Q8_0 fast path. Quantised
/// Mamba-3 ingest is a follow-up.
/// </para>
/// <para>
/// <b>SISO only.</b> <see cref="Upload(VulkanDevice, ModelConfig, Mamba3Weights)"/> rejects
/// MIMO checkpoints (<see cref="Mamba3Config.IsMimo"/> is <c>true</c>) at the upload boundary
/// so callers fail fast — the Mamba-3 SISO orchestrator does not consume <c>mimo_x</c> /
/// <c>mimo_z</c> / <c>mimo_o</c> and the MIMO scan kernel is wired in by a follow-up
/// commit (see <see cref="DotLLM.Vulkan.Kernels.Mamba3CanonicalSsdMimoF32Kernel"/>).
/// </para>
/// </remarks>
internal sealed class VulkanMamba3Weights : IDisposable
{
    /// <summary>Per-layer Mamba-3 mixer weight buffers (SISO).</summary>
    /// <remarks>
    /// <para>
    /// Holds device-local buffers for every projection plus tiny host-side mirrors
    /// (<see cref="BNormHost"/>, <see cref="CNormHost"/>, <see cref="BBiasHost"/>,
    /// <see cref="CBiasHost"/>, <see cref="DtBiasHost"/>) for the four small tensors
    /// the per-token CPU preprocessing block reads. Keeping these as managed
    /// <c>float[]</c> mirrors saves a per-layer device-local-buffer download path
    /// during the host prep step.
    /// </para>
    /// </remarks>
    internal sealed class LayerBuffers : IDisposable
    {
        public required VulkanDevice.Buffer Norm { get; init; }
        public required VulkanDevice.Buffer InProj { get; init; }
        public required VulkanDevice.Buffer OutProj { get; init; }
        public required VulkanDevice.Buffer BNorm { get; init; }
        public required VulkanDevice.Buffer CNorm { get; init; }
        public required VulkanDevice.Buffer BBias { get; init; }
        public required VulkanDevice.Buffer CBias { get; init; }
        public required VulkanDevice.Buffer D { get; init; }
        public required VulkanDevice.Buffer DtBias { get; init; }

        public required int InProjOutputDim { get; init; }
        public required int InProjInputDim { get; init; }
        public required int OutProjOutputDim { get; init; }
        public required int OutProjInputDim { get; init; }

        // Host-side mirrors of the tiny tensors consumed by the per-token CPU prep.
        public required float[] BNormHost { get; init; }
        public required float[] CNormHost { get; init; }
        public required float[] BBiasHost { get; init; }
        public required float[] CBiasHost { get; init; }
        public required float[] DtBiasHost { get; init; }

        public void Dispose()
        {
            Norm.Dispose();
            InProj.Dispose();
            OutProj.Dispose();
            BNorm.Dispose();
            CNorm.Dispose();
            BBias.Dispose();
            CBias.Dispose();
            D.Dispose();
            DtBias.Dispose();
        }
    }

    private readonly LayerBuffers[] _layers;

    public LayerBuffers[] Layers => _layers;

    public VulkanDevice.Buffer TokenEmbedding { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    public VulkanDevice.Buffer FinalNormWeight { get; }
    public VulkanDevice.Buffer LmHead { get; }
    public int LmHeadOutputDim { get; }
    public int LmHeadInputDim { get; }

    public long AllocatedBytes { get; }

    private VulkanMamba3Weights(
        LayerBuffers[] layers,
        VulkanDevice.Buffer tokenEmbedding, int vocabSize, int hiddenSize,
        VulkanDevice.Buffer finalNorm,
        VulkanDevice.Buffer lmHead, int lmHeadOutputDim, int lmHeadInputDim,
        long allocatedBytes)
    {
        _layers = layers;
        TokenEmbedding = tokenEmbedding;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        FinalNormWeight = finalNorm;
        LmHead = lmHead;
        LmHeadOutputDim = lmHeadOutputDim;
        LmHeadInputDim = lmHeadInputDim;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads a Mamba-3 SISO model's weights to the Vulkan device. Every tensor handle
    /// in <paramref name="weights"/> must be populated and F32 — <see cref="Mamba3WeightLoader"/>
    /// already enforces F32 at load time, so this is a straight memcpy upload.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The supplied <paramref name="config"/> describes a MIMO checkpoint
    /// (<c>nRank &gt; 1</c>). MIMO scan wires in via a follow-up commit; the current path
    /// is SISO only.
    /// </exception>
    public static VulkanMamba3Weights Upload(
        VulkanDevice device, ModelConfig config, Mamba3Weights weights)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);

        Mamba3Config m3 = config.Mamba3Config
            ?? throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated for VulkanMamba3Weights.",
                nameof(config));

        if (m3.IsMimo)
            throw new NotSupportedException(
                "MIMO scan wires in via follow-up commit; current path is SISO only.");

        if (weights.Report.HasMissingRequired)
            throw new InvalidDataException(
                $"Mamba-3 weights are incomplete ({weights.Report.MissingRequiredCount} required tensors missing).");

        int numLayers = config.NumLayers;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;
        int dInner = m3.DInner;
        int dState = m3.StateSize;
        int nHead = m3.NumHeads;
        int dInProj = m3.InputProjectionDim;

        // Largest single matrix we will upload, in its F32 form. Used to size the
        // staging buffer once and reuse it across every device-local copy.
        long maxBytes = ComputeMaxStagingBytes(numLayers, hidden, vocab, dInner, dState, nHead, dInProj);
        using var staging = device.Allocate(maxBytes);

        long totalBytes = 0;

        // Token embedding [vocab, hidden].
        var tokenEmbed = UploadTensor(device, staging, weights.TokenEmbedding, (long)vocab * hidden, out long tokenBytes);
        totalBytes += tokenBytes;

        // Final norm [hidden].
        var finalNorm = UploadTensor(device, staging, weights.FinalNorm, hidden, out long fnBytes);
        totalBytes += fnBytes;

        // LM head [vocab, hidden]. Whether tied or not, the weight loader gives us a
        // populated handle; we always upload a separate device buffer so the model
        // doesn't need to know about tying.
        var lmHead = UploadTensor(device, staging, weights.LmHead, (long)vocab * hidden, out long lmBytes);
        totalBytes += lmBytes;

        var layers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];

            var norm = UploadTensor(device, staging, lw.Norm, hidden, out long normBytes);
            var inProj = UploadTensor(device, staging, lw.InProj, (long)dInProj * hidden, out long inProjBytes);
            var outProj = UploadTensor(device, staging, lw.OutProj, (long)hidden * dInner, out long outProjBytes);
            var bNorm = UploadTensor(device, staging, lw.BNorm, dState, out long bNormBytes);
            var cNorm = UploadTensor(device, staging, lw.CNorm, dState, out long cNormBytes);
            // Bias shape on disk is [n_head, 1, d_state] — element count nHead * dState.
            var bBias = UploadTensor(device, staging, lw.BBias, (long)nHead * dState, out long bBiasBytes);
            var cBias = UploadTensor(device, staging, lw.CBias, (long)nHead * dState, out long cBiasBytes);
            var d = UploadTensor(device, staging, lw.D, nHead, out long dBytes);
            var dtBias = UploadTensor(device, staging, lw.DtBias, nHead, out long dtBytes);

            totalBytes += normBytes + inProjBytes + outProjBytes + bNormBytes + cNormBytes
                        + bBiasBytes + cBiasBytes + dBytes + dtBytes;

            var lb = new LayerBuffers
            {
                Norm = norm,
                InProj = inProj, InProjOutputDim = dInProj, InProjInputDim = hidden,
                OutProj = outProj, OutProjOutputDim = hidden, OutProjInputDim = dInner,
                BNorm = bNorm, CNorm = cNorm,
                BBias = bBias, CBias = cBias,
                D = d, DtBias = dtBias,
                // Host-side mirrors of the small tensors the per-token CPU prep reads.
                BNormHost = SnapshotHost(lw.BNorm, dState),
                CNormHost = SnapshotHost(lw.CNorm, dState),
                BBiasHost = SnapshotHost(lw.BBias, nHead * dState),
                CBiasHost = SnapshotHost(lw.CBias, nHead * dState),
                DtBiasHost = SnapshotHost(lw.DtBias, nHead),
            };
            layers[i] = lb;
        }

        return new VulkanMamba3Weights(layers,
            tokenEmbed, vocab, hidden,
            finalNorm,
            lmHead, lmHeadOutputDim: vocab, lmHeadInputDim: hidden,
            totalBytes);
    }

    private static long ComputeMaxStagingBytes(
        int numLayers, int hidden, int vocab, int dInner, int dState, int nHead, int dInProj)
    {
        long max = 0;
        max = Math.Max(max, (long)vocab * hidden * sizeof(float));            // token embed / lm_head
        max = Math.Max(max, (long)dInProj * hidden * sizeof(float));          // in_proj
        max = Math.Max(max, (long)hidden * dInner * sizeof(float));           // out_proj
        max = Math.Max(max, (long)nHead * dState * sizeof(float));            // bias
        max = Math.Max(max, (long)hidden * sizeof(float));                    // norms
        max = Math.Max(max, (long)dState * sizeof(float));                    // bc_norm
        max = Math.Max(max, (long)nHead * sizeof(float));                     // d, dt_bias
        return Math.Max(max, 64);
    }

    /// <summary>
    /// Copies the F32 tensor pointed to by <paramref name="handle"/> into a fresh managed
    /// array of length <paramref name="elements"/>. Used for the tiny host-side mirrors of
    /// per-layer tensors that the per-token CPU preprocessing block consumes.
    /// </summary>
    private static unsafe float[] SnapshotHost(Mamba3TensorHandle handle, int elements)
    {
        if (!handle.IsPopulated)
            throw new InvalidOperationException("Mamba-3 tensor handle is not populated.");
        if (handle.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"Mamba-3 tensor dtype {handle.SourceDType} is not yet supported (expected F32).");
        var copy = new float[elements];
        new ReadOnlySpan<float>((void*)handle.Pointer, elements).CopyTo(copy);
        return copy;
    }

    private static unsafe VulkanDevice.Buffer UploadTensor(
        VulkanDevice device, VulkanDevice.Buffer staging,
        Mamba3TensorHandle handle, long expectedElements, out long uploadedBytes)
    {
        if (!handle.IsPopulated)
            throw new InvalidOperationException(
                "Mamba-3 tensor handle is not populated — check Mamba3Weights.Report.");
        if (handle.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"Mamba-3 tensor dtype {handle.SourceDType} is not yet supported (expected F32).");

        long bytes = expectedElements * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanMamba3Weights.UploadTensor staging");
        try
        {
            new ReadOnlySpan<float>((void*)handle.Pointer, checked((int)expectedElements))
                .CopyTo(new Span<float>((void*)mapped, checked((int)expectedElements)));
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
        FinalNormWeight.Dispose();
        LmHead.Dispose();
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Dispose();
    }
}
