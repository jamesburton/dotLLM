using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer weight buffers on a Vulkan device for the Mamba-3 model (SISO and MIMO).
/// Mirrors <see cref="VulkanNemotronHWeights"/>'s upload/lifetime pattern but with a
/// single per-layer projection bundle (the Mamba-3 mixer tensors) rather than the three
/// hybrid sub-bundles of NemotronH.
/// </summary>
/// <remarks>
/// <para>
/// <b>Two-mode storage.</b> The matmul-target projections (<c>in_proj</c>, <c>out_proj</c>,
/// <c>lm_head</c>) honour the optional Q8_0 overlay on <see cref="Mamba3Weights"/> /
/// <see cref="Mamba3LayerQuantOverlay"/>: when the overlay is set and the contraction axis
/// is a multiple of 32 (the Q8_0 group size) the raw Q8_0 blocks are uploaded verbatim and
/// the forward pass dispatches them through the existing
/// <see cref="DotLLM.Vulkan.Kernels.MatMulQ8_0Kernel"/> /
/// <see cref="DotLLM.Vulkan.Kernels.MatMulQ8_0GemmKernel"/> kernels, mirroring the standard
/// transformer at <see cref="VulkanWeights"/>. Otherwise — and on every production load
/// path today, since <see cref="Mamba3WeightLoader"/> emits only F32 handles — the F32
/// source is uploaded verbatim and the forward pass uses <c>matmul_f32</c>.
/// </para>
/// <para>
/// <b>Token embedding.</b> Always uploaded as F32 regardless of any overlay — the
/// embedding gather uses <c>vkCmdCopyBuffer</c> with row-major byte offsets, which only
/// works with a contiguous F32 layout. Same convention as <see cref="VulkanWeights"/> and
/// <see cref="VulkanNemotronHWeights"/>.
/// </para>
/// <para>
/// <b>Small per-layer tensors.</b> Norms (<c>Norm</c>, <c>BNorm</c>, <c>CNorm</c>,
/// <c>FinalNorm</c>), biases (<c>BBias</c>, <c>CBias</c>, <c>DtBias</c>), per-head decay
/// (<c>D</c>), and the MIMO per-rank weights (<c>mimo_z</c>, <c>mimo_o</c>) are always
/// F32. They are never quantised in production GGUFs (and the overlay schema does not
/// expose Q8_0 slots for them).
/// </para>
/// <para>
/// <b>SISO and MIMO.</b> Both checkpoint flavours land here. SISO uploads the canonical
/// 9 per-layer mixer tensors; MIMO additionally uploads the per-rank gate / output
/// contraction weights (<c>mimo_z</c>, <c>mimo_o</c>) and lays <c>B_bias</c>/<c>C_bias</c>
/// out as the rank-expanded <c>[num_heads, mimo_rank, state_size]</c> form the canonical
/// MIMO scan expects. The canonical kernel folds <c>mimo_x</c> (V's per-rank expansion)
/// into the rank-summed K·V state update inside <c>ExecuteMimo</c>, so the Vulkan side
/// does not consume <c>mimo_x</c> directly — it lives on the CPU loader for compatibility
/// with canonical checkpoints but is not uploaded.
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

        // MIMO-only per-rank weights (null on a SISO layer). The MIMO scan kernel needs
        // both bound when nRank > 1 — the canonical kernel folds mimo_x into the rank-
        // summed state update, so it is intentionally NOT mirrored on the device.
        public VulkanDevice.Buffer? MimoZ { get; init; }
        public VulkanDevice.Buffer? MimoO { get; init; }

        public required int InProjOutputDim { get; init; }
        public required int InProjInputDim { get; init; }
        public required int OutProjOutputDim { get; init; }
        public required int OutProjInputDim { get; init; }

        // Device-side storage type per matmul-target projection. <see cref="QuantizationType.Q8_0"/>
        // when the source carried a Q8_0 overlay AND the contraction axis is a multiple of 32;
        // <see cref="QuantizationType.F32"/> otherwise. The forward pass branches on this to
        // choose the matmul kernel (<c>matmul_q8_0[_gemm]</c> vs <c>matmul_f32</c>) — same
        // routing as <see cref="VulkanNemotronHTransformerModel"/>.
        public QuantizationType InProjDeviceQuantType { get; init; }
        public QuantizationType OutProjDeviceQuantType { get; init; }

        // Host-side mirrors of the tiny tensors consumed by the per-token CPU prep.
        // BBiasHost / CBiasHost are sized to nHead * effectiveRank * dState — for SISO
        // that is nHead * dState; for MIMO it is nHead * mimoRank * dState laid out
        // [H, R, N] row-major (matches Mamba3Block.ForwardMimo's bias indexing).
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
            MimoZ?.Dispose();
            MimoO?.Dispose();
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

    /// <summary>Device-side storage type for <see cref="LmHead"/>. <see cref="QuantizationType.Q8_0"/>
    /// when the source carried a Q8_0 overlay AND <c>hidden_size % 32 == 0</c>;
    /// <see cref="QuantizationType.F32"/> otherwise.</summary>
    public QuantizationType LmHeadDeviceQuantType { get; }

    public long AllocatedBytes { get; }

    private VulkanMamba3Weights(
        LayerBuffers[] layers,
        VulkanDevice.Buffer tokenEmbedding, int vocabSize, int hiddenSize,
        VulkanDevice.Buffer finalNorm,
        VulkanDevice.Buffer lmHead, QuantizationType lmHeadDeviceQt,
        int lmHeadOutputDim, int lmHeadInputDim,
        long allocatedBytes)
    {
        _layers = layers;
        TokenEmbedding = tokenEmbedding;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        FinalNormWeight = finalNorm;
        LmHead = lmHead;
        LmHeadDeviceQuantType = lmHeadDeviceQt;
        LmHeadOutputDim = lmHeadOutputDim;
        LmHeadInputDim = lmHeadInputDim;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads a Mamba-3 model's weights (SISO or MIMO) to the Vulkan device. Every F32
    /// tensor handle in <paramref name="weights"/> must be populated —
    /// <see cref="Mamba3WeightLoader"/> already enforces F32 at load time. The optional
    /// Q8_0 overlay (<see cref="Mamba3Weights.LmHeadQ8Ptr"/> et al.) is honoured for the
    /// matmul-target projections when the contraction axis is a multiple of 32; the F32
    /// handle is still required (the CPU oracle reads it) but is not uploaded when the
    /// overlay is consumed.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The supplied <paramref name="config"/> requests a non-positive MIMO rank.
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

        bool isMimo = m3.IsMimo;
        int mimoRank = isMimo ? m3.MimoRank : 1;
        if (mimoRank < 1)
            throw new ArgumentException(
                $"Mamba3Config.MimoRank must be >= 1 (got {mimoRank}).", nameof(config));

        if (weights.Report.HasMissingRequired)
            throw new InvalidDataException(
                $"Mamba-3 weights are incomplete ({weights.Report.MissingRequiredCount} required tensors missing).");

        int numLayers = config.NumLayers;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;
        int dInner = m3.DInner;
        int dState = m3.StateSize;
        int nHead = m3.NumHeads;
        int headDim = m3.HeadDim;
        int dInProj = m3.InputProjectionDim;

        // Per-layer rank-aware bias element count. SISO: H * 1 * N == H * N. MIMO:
        // H * R * N (canonical [H, R, N] layout, validated by the loader).
        int bcBiasElems = nHead * mimoRank * dState;
        // mimo_z / mimo_o per-layer element count when MIMO is on.
        int mimoElems = nHead * mimoRank * headDim;

        // Largest single matrix we will upload, in its F32 form. Used to size the
        // staging buffer once and reuse it across every device-local copy.
        long maxBytes = ComputeMaxStagingBytes(
            numLayers, hidden, vocab, dInner, dState, nHead, dInProj, bcBiasElems, mimoElems);
        using var staging = device.Allocate(maxBytes);

        long totalBytes = 0;

        // Token embedding [vocab, hidden]. Always F32 — the embedding gather uses
        // byte-offset vkCmdCopyBuffer which needs a contiguous F32 layout. Same convention
        // as VulkanWeights / VulkanNemotronHWeights.
        var tokenEmbed = UploadTensor(device, staging, weights.TokenEmbedding, (long)vocab * hidden, out long tokenBytes);
        totalBytes += tokenBytes;

        // Final norm [hidden].
        var finalNorm = UploadTensor(device, staging, weights.FinalNorm, hidden, out long fnBytes);
        totalBytes += fnBytes;

        // LM head [vocab, hidden]. Whether tied or not, the weight loader gives us a
        // populated handle; we always upload a separate device buffer so the model doesn't
        // need to know about tying. Honours the Q8_0 overlay when present and
        // hidden % 32 == 0; otherwise falls back to the F32 source upload.
        bool lmKeepQ8 = KeepQ8OnDevice(weights.LmHeadQuantTypeOverlay, hidden) && weights.LmHeadQ8Ptr != 0;
        VulkanDevice.Buffer lmHead;
        QuantizationType lmHeadDeviceQt;
        long lmBytes;
        if (lmKeepQ8)
        {
            lmBytes = Dequantize.RowByteSize(hidden, QuantizationType.Q8_0) * vocab;
            lmHead = device.AllocateDeviceLocal(lmBytes);
            UploadRawBytes(device, staging, weights.LmHeadQ8Ptr, lmBytes, lmHead);
            lmHeadDeviceQt = QuantizationType.Q8_0;
        }
        else
        {
            lmHead = UploadTensor(device, staging, weights.LmHead, (long)vocab * hidden, out lmBytes);
            lmHeadDeviceQt = QuantizationType.F32;
        }
        totalBytes += lmBytes;

        // Per-layer overlays — null on production load paths (production loaders never set
        // them); tests populate one entry per layer to drive the Q8_0 matmul kernels.
        Mamba3LayerQuantOverlay[]? overlays = weights.LayerOverlays;
        if (overlays is not null && overlays.Length != numLayers)
            throw new ArgumentException(
                $"Mamba3Weights.LayerOverlays length {overlays.Length} != NumLayers {numLayers}.",
                nameof(weights));

        var layers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];
            Mamba3LayerQuantOverlay? layerOv = overlays?[i];

            var norm = UploadTensor(device, staging, lw.Norm, hidden, out long normBytes);

            // in_proj: contraction axis = hiddenSize. Honours Q8_0 overlay when set AND
            // hidden % 32 == 0; otherwise falls back to F32 source upload.
            VulkanDevice.Buffer inProj;
            QuantizationType inProjDeviceQt;
            long inProjBytes;
            bool inProjKeepQ8 = layerOv is not null
                && KeepQ8OnDevice(layerOv.InProjQuantTypeOverlay, hidden)
                && layerOv.InProjQ8Ptr != 0;
            if (inProjKeepQ8)
            {
                inProjBytes = Dequantize.RowByteSize(hidden, QuantizationType.Q8_0) * dInProj;
                inProj = device.AllocateDeviceLocal(inProjBytes);
                UploadRawBytes(device, staging, layerOv!.InProjQ8Ptr, inProjBytes, inProj);
                inProjDeviceQt = QuantizationType.Q8_0;
            }
            else
            {
                inProj = UploadTensor(device, staging, lw.InProj, (long)dInProj * hidden, out inProjBytes);
                inProjDeviceQt = QuantizationType.F32;
            }

            // out_proj: contraction axis = dInner. Honours Q8_0 overlay when set AND
            // dInner % 32 == 0.
            VulkanDevice.Buffer outProj;
            QuantizationType outProjDeviceQt;
            long outProjBytes;
            bool outProjKeepQ8 = layerOv is not null
                && KeepQ8OnDevice(layerOv.OutProjQuantTypeOverlay, dInner)
                && layerOv.OutProjQ8Ptr != 0;
            if (outProjKeepQ8)
            {
                outProjBytes = Dequantize.RowByteSize(dInner, QuantizationType.Q8_0) * hidden;
                outProj = device.AllocateDeviceLocal(outProjBytes);
                UploadRawBytes(device, staging, layerOv!.OutProjQ8Ptr, outProjBytes, outProj);
                outProjDeviceQt = QuantizationType.Q8_0;
            }
            else
            {
                outProj = UploadTensor(device, staging, lw.OutProj, (long)hidden * dInner, out outProjBytes);
                outProjDeviceQt = QuantizationType.F32;
            }
            var bNorm = UploadTensor(device, staging, lw.BNorm, dState, out long bNormBytes);
            var cNorm = UploadTensor(device, staging, lw.CNorm, dState, out long cNormBytes);
            // Bias shape on disk: SISO [n_head, 1, d_state] (element count H·N), MIMO
            // [n_head, mimo_rank, d_state] (element count H·R·N). Element count is the
            // only thing the upload path cares about — the rank-expanded slot ordering
            // is preserved verbatim by the row-major copy.
            var bBias = UploadTensor(device, staging, lw.BBias, bcBiasElems, out long bBiasBytes);
            var cBias = UploadTensor(device, staging, lw.CBias, bcBiasElems, out long cBiasBytes);
            var d = UploadTensor(device, staging, lw.D, nHead, out long dBytes);
            var dtBias = UploadTensor(device, staging, lw.DtBias, nHead, out long dtBytes);

            totalBytes += normBytes + inProjBytes + outProjBytes + bNormBytes + cNormBytes
                        + bBiasBytes + cBiasBytes + dBytes + dtBytes;

            VulkanDevice.Buffer? mimoZ = null;
            VulkanDevice.Buffer? mimoO = null;
            if (isMimo)
            {
                mimoZ = UploadTensor(device, staging, lw.MimoZ, mimoElems, out long mzBytes);
                mimoO = UploadTensor(device, staging, lw.MimoO, mimoElems, out long moBytes);
                totalBytes += mzBytes + moBytes;
                // mimo_x is intentionally not uploaded — the canonical MIMO scan folds
                // its V-rank expansion into the rank-summed K·V state update inside
                // ExecuteMimo (mirrored by Mamba3CanonicalSsdMimoF32Kernel).
            }

            var lb = new LayerBuffers
            {
                Norm = norm,
                InProj = inProj, InProjOutputDim = dInProj, InProjInputDim = hidden,
                InProjDeviceQuantType = inProjDeviceQt,
                OutProj = outProj, OutProjOutputDim = hidden, OutProjInputDim = dInner,
                OutProjDeviceQuantType = outProjDeviceQt,
                BNorm = bNorm, CNorm = cNorm,
                BBias = bBias, CBias = cBias,
                D = d, DtBias = dtBias,
                MimoZ = mimoZ, MimoO = mimoO,
                // Host-side mirrors of the small tensors the per-token CPU prep reads.
                BNormHost = SnapshotHost(lw.BNorm, dState),
                CNormHost = SnapshotHost(lw.CNorm, dState),
                BBiasHost = SnapshotHost(lw.BBias, bcBiasElems),
                CBiasHost = SnapshotHost(lw.CBias, bcBiasElems),
                DtBiasHost = SnapshotHost(lw.DtBias, nHead),
            };
            layers[i] = lb;
        }

        return new VulkanMamba3Weights(layers,
            tokenEmbed, vocab, hidden,
            finalNorm,
            lmHead, lmHeadDeviceQt, lmHeadOutputDim: vocab, lmHeadInputDim: hidden,
            totalBytes);
    }

    /// <summary>True iff a Q8_0 overlay can be kept on device as raw Q8_0 blocks — gated
    /// on the contraction dim being a multiple of the Q8_0 group size (32). When the
    /// constraint fails the upload silently falls back to the F32 source instead.</summary>
    private static bool KeepQ8OnDevice(QuantizationType qt, int contractionDim)
        => qt == QuantizationType.Q8_0 && (contractionDim % 32) == 0;

    /// <summary>Copies <paramref name="bytes"/> raw bytes from <paramref name="srcPtr"/>
    /// through <paramref name="staging"/> into the device-local <paramref name="dst"/>.
    /// Same on-device byte layout as <see cref="VulkanWeights"/> so the existing
    /// <c>matmul_q8_0</c> / <c>matmul_q8_0_gemm</c> kernels can read it directly.</summary>
    private static unsafe void UploadRawBytes(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, long bytes, VulkanDevice.Buffer dst)
    {
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanMamba3Weights raw");
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

    private static long ComputeMaxStagingBytes(
        int numLayers, int hidden, int vocab, int dInner, int dState, int nHead, int dInProj,
        int bcBiasElems, int mimoElems)
    {
        long max = 0;
        max = Math.Max(max, (long)vocab * hidden * sizeof(float));            // token embed / lm_head
        max = Math.Max(max, (long)dInProj * hidden * sizeof(float));          // in_proj
        max = Math.Max(max, (long)hidden * dInner * sizeof(float));           // out_proj
        max = Math.Max(max, (long)bcBiasElems * sizeof(float));               // B_bias / C_bias (rank-aware)
        max = Math.Max(max, (long)hidden * sizeof(float));                    // norms
        max = Math.Max(max, (long)dState * sizeof(float));                    // bc_norm
        max = Math.Max(max, (long)nHead * sizeof(float));                     // d, dt_bias
        max = Math.Max(max, (long)mimoElems * sizeof(float));                 // mimo_z / mimo_o
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
