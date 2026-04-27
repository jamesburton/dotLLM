using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for the NVIDIA NemotronH hybrid model. Each layer is
/// gated on <see cref="HybridLayerLayout.LayerKind"/> and dispatches one of three sub-layer
/// bodies (Mamba2 SSM / GQA Attention / squared-ReLU FFN) between a pre-norm RMSNorm and a
/// residual add.
/// </summary>
/// <remarks>
/// <para>
/// Projection weights honour their source quant type at upload: Q8_0 sources are kept on
/// device as raw Q8_0 blocks when the contraction dim is a multiple of 32 and dispatched
/// through <see cref="MatMulQ8_0Kernel"/> (decode) or <see cref="MatMulQ8_0GemmKernel"/> /
/// <see cref="MatMulQ8_0GemmCoopmatKernel"/> (prefill) — same routing as
/// <see cref="VulkanTransformerModel"/>. Every other dtype (F32, F16, K-quants) is
/// dequantised to F32 at upload and dispatched through <see cref="MatMulF32Kernel"/>. The 6
/// SSM kernels (silu_inplace, conv1d_causal, mamba2_selective_scan, ssm_d_skip,
/// group_rmsnorm, relu_squared_inplace) all have parity tests in
/// <c>tests/DotLLM.Tests.Unit/Vulkan/</c>; this model only orchestrates them.
/// </para>
/// <para>
/// This class is intentionally separate from <see cref="VulkanTransformerModel"/> — they
/// share kernels but not orchestration. The MoE/MLA-bearing standard model has its own
/// scratch shapes; the SSM-bearing hybrid path has its own (Zxbcdt, ConvInput, …).
/// </para>
/// </remarks>
public sealed class VulkanNemotronHTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanNemotronHWeights _weights;
    private readonly VulkanNemotronHForwardState _state;
    private readonly VulkanSsmStateCache _ssmCache;

    // Hybrid layout (per-layer kind), SSM config + ordinal map.
    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;
    private readonly int[] _ssmLayerOrdinal;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;

    // Kernels.
    private readonly MatMulF32Kernel _matmul;
    private readonly MatMulQ8_0Kernel _matmulQ8;
    private readonly MatMulQ8_0GemmKernel _matmulQ8Gemm;
    // Coopmat-accelerated Q8_0 prefill kernel — created opportunistically; null on devices
    // without VK_KHR_cooperative_matrix support, in which case the scalar Q8_0 GEMM is used.
    private readonly MatMulQ8_0GemmCoopmatKernel? _matmulQ8GemmCoopmat;
    // Q4_K_M matmul kernels — Phase 1 of K-quant work. Always created; the dispatcher
    // in RecordMatmul branches on the device-side QuantizationType per call.
    private readonly MatMulQ4KGemvF32Kernel _matmulQ4K;
    private readonly MatMulQ4KGemmF32Kernel _matmulQ4KGemm;
    // Q5_K_M matmul kernels — Phase 1 sibling of Q4_K. Always created.
    private readonly MatMulQ5KGemvF32Kernel _matmulQ5K;
    private readonly MatMulQ5KGemmF32Kernel _matmulQ5KGemm;
    // Q6_K_M matmul kernels — Phase 1 sibling of Q4_K / Q5_K, completing the
    // K-quant matmul kernel coverage. Always created.
    private readonly MatMulQ6KGemvF32Kernel _matmulQ6K;
    private readonly MatMulQ6KGemmF32Kernel _matmulQ6KGemm;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    private readonly SwiGluF32Kernel _swiglu;
    private readonly AddKernel _add;
    private readonly BiasAddF32Kernel _biasAdd;
    private readonly Conv1dCausalF32Kernel _conv1dCausal;
    private readonly SiluInplaceF32Kernel _siluInplace;
    private readonly Mamba2SelectiveScanF32Kernel _mamba2Scan;
    private readonly SsmDSkipF32Kernel _ssmDSkip;
    private readonly GroupRmsNormF32Kernel _groupRmsNorm;
    private readonly ReluSquaredInplaceF32Kernel _reluSquared;
    private readonly SsmSplitXbcF32Kernel _ssmSplitXbc;

    private readonly VulkanDevice.SubmitContext _submit;
    private readonly bool _ownsDevice;

    private readonly int _ropeDim;
    private readonly float _ropeTheta;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes + _ssmCache.AllocatedBytes;

    /// <summary>Number of attention layers (KV slots).</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>Creates a sparse <see cref="VulkanNemotronHKvCache"/> sized for this model.</summary>
    public VulkanNemotronHKvCache CreateKvCache(int maxSeqLen)
        => new(_device, _kvSlotForLayer, _attentionLayerCount,
               Config.NumKvHeads, Config.HeadDim, maxSeqLen);

    private VulkanNemotronHTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config,
        VulkanNemotronHWeights weights,
        VulkanNemotronHForwardState state,
        VulkanSsmStateCache ssmCache,
        int[] ssmLayerOrdinal,
        int[] kvSlotForLayer, int attentionLayerCount,
        MatMulF32Kernel matmul, MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        MatMulQ4KGemvF32Kernel matmulQ4K, MatMulQ4KGemmF32Kernel matmulQ4KGemm,
        MatMulQ5KGemvF32Kernel matmulQ5K, MatMulQ5KGemmF32Kernel matmulQ5KGemm,
        MatMulQ6KGemvF32Kernel matmulQ6K, MatMulQ6KGemmF32Kernel matmulQ6KGemm,
        RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, SwiGluF32Kernel swiglu, AddKernel add, BiasAddF32Kernel biasAdd,
        Conv1dCausalF32Kernel conv1dCausal, SiluInplaceF32Kernel siluInplace,
        Mamba2SelectiveScanF32Kernel mamba2Scan, SsmDSkipF32Kernel ssmDSkip,
        GroupRmsNormF32Kernel groupRmsNorm, ReluSquaredInplaceF32Kernel reluSquared,
        SsmSplitXbcF32Kernel ssmSplitXbc,
        VulkanDevice.SubmitContext submit,
        int ropeDim, float ropeTheta)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _state = state;
        _ssmCache = ssmCache;
        _layout = config.HybridLayout!;
        _ssm = config.SsmConfig!.Value;
        _ssmLayerOrdinal = ssmLayerOrdinal;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;

        _matmul = matmul;
        _matmulQ8 = matmulQ8;
        _matmulQ8Gemm = matmulQ8Gemm;
        _matmulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        _matmulQ4K = matmulQ4K;
        _matmulQ4KGemm = matmulQ4KGemm;
        _matmulQ5K = matmulQ5K;
        _matmulQ5KGemm = matmulQ5KGemm;
        _matmulQ6K = matmulQ6K;
        _matmulQ6KGemm = matmulQ6KGemm;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _swiglu = swiglu;
        _add = add;
        _biasAdd = biasAdd;
        _conv1dCausal = conv1dCausal;
        _siluInplace = siluInplace;
        _mamba2Scan = mamba2Scan;
        _ssmDSkip = ssmDSkip;
        _groupRmsNorm = groupRmsNorm;
        _reluSquared = reluSquared;
        _ssmSplitXbc = ssmSplitXbc;

        _submit = submit;
        _ropeDim = ropeDim;
        _ropeTheta = ropeTheta;
    }

    /// <summary>
    /// Builds a Vulkan NemotronH model from caller-owned, pre-built <see cref="NemotronHLayerWeights"/> —
    /// used by the synthetic-fixture parity test. The caller retains ownership of every
    /// unmanaged pointer (token embed, output, plus every projection inside <paramref name="cpuLayers"/>).
    /// </summary>
    internal static VulkanNemotronHTransformerModel BuildFromPrebuiltWeights(
        VulkanDevice device,
        ModelConfig config,
        NemotronHLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputM, int outputK,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(spvDir);

        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"VulkanNemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));

        var layout = config.HybridLayout!;
        var ssm = config.SsmConfig!.Value;

        // Build per-layer ordinals for the SSM cache and the sparse KV map.
        var ssmLayerOrdinal = new int[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int ssmOrdinal = 0;
        int attentionLayerCount = 0;
        int maxIntermediate = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            ssmLayerOrdinal[i] = layout.LayerKind[i] == HybridLayerKind.Ssm ? ssmOrdinal++ : -1;
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;

            if (layout.LayerKind[i] == HybridLayerKind.Ffn)
            {
                var ffn = cpuLayers[i].Ffn!;
                if (ffn.UpOutputDim > maxIntermediate) maxIntermediate = ffn.UpOutputDim;
            }
        }
        if (maxIntermediate == 0) maxIntermediate = config.HiddenSize;

        // Validate RoPE config when there are attention layers.
        int ropeDim = config.RoPEConfig?.DimensionCount ?? 0;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        if (attentionLayerCount > 0)
        {
            if (ropeDim <= 0 || (ropeDim & 1) != 0)
                throw new ArgumentException(
                    $"NemotronH attention layers require an even rope_dim > 0 (got {ropeDim}).", nameof(config));
            if (ropeDim > config.HeadDim)
                throw new ArgumentException(
                    $"rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.", nameof(config));
        }

        // Upload weights and allocate scratch.
        var weights = VulkanNemotronHWeights.Upload(device, config, cpuLayers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQt, outputWeight, outputQt, outputM, outputK);

        var state = new VulkanNemotronHForwardState(device,
            hiddenSize: config.HiddenSize,
            maxIntermediateSize: maxIntermediate,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            ssm: ssm,
            initialSeqLen: 1);

        var ssmCache = new VulkanSsmStateCache(device, ssm, ssmOrdinal);

        // Create kernels.
        var matmul = MatMulF32Kernel.Create(device, spvDir);
        // Q8_0 matmul kernels are always created — projections that aren't kept on device as
        // Q8_0 (i.e. uploaded as F32) simply never dispatch through them, but the dispatch
        // router needs them bound on every device because mixed-quant configs (some Q8_0
        // SSM in_proj, F16 attention Q, …) are common in real GGUFs.
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        // Optional coopmat prefill GEMM — null on devices without KHR_cooperative_matrix.
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* Kernel threw: no usable tile shape. Stay on scalar. */ }
        }
        // Q4_K_M GEMV + GEMM — Phase 1 of K-quant work. Always created.
        var matmulQ4K = MatMulQ4KGemvF32Kernel.Create(device, spvDir);
        var matmulQ4KGemm = MatMulQ4KGemmF32Kernel.Create(device, spvDir);
        // Q5_K_M GEMV + GEMM — Phase 1 sibling of Q4_K. Always created.
        var matmulQ5K = MatMulQ5KGemvF32Kernel.Create(device, spvDir);
        var matmulQ5KGemm = MatMulQ5KGemmF32Kernel.Create(device, spvDir);
        // Q6_K_M GEMV + GEMM — Phase 1 sibling of Q4_K / Q5_K. Always created.
        var matmulQ6K = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
        var matmulQ6KGemm = MatMulQ6KGemmF32Kernel.Create(device, spvDir);
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);
        var biasAdd = BiasAddF32Kernel.Create(device, spvDir);
        var conv1dCausal = Conv1dCausalF32Kernel.Create(device, spvDir);
        var siluInplace = SiluInplaceF32Kernel.Create(device, spvDir);
        var mamba2Scan = Mamba2SelectiveScanF32Kernel.Create(device, spvDir);
        var ssmDSkip = SsmDSkipF32Kernel.Create(device, spvDir);
        var groupRmsNorm = GroupRmsNormF32Kernel.Create(device, spvDir);
        var reluSquared = ReluSquaredInplaceF32Kernel.Create(device, spvDir);
        var ssmSplitXbc = SsmSplitXbcF32Kernel.Create(device, spvDir);

        var submit = device.CreateSubmitContext();

        return new VulkanNemotronHTransformerModel(
            device, ownsDevice: false,
            config, weights, state, ssmCache,
            ssmLayerOrdinal, kvSlotForLayer, attentionLayerCount,
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            matmulQ4K, matmulQ4KGemm,
            matmulQ5K, matmulQ5KGemm,
            matmulQ6K, matmulQ6KGemm,
            rmsnorm, rope, attention, swiglu, add, biasAdd,
            conv1dCausal, siluInplace, mamba2Scan, ssmDSkip, groupRmsNorm, reluSquared,
            ssmSplitXbc,
            submit,
            ropeDim, ropeTheta);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");

        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        bool scratchResized = _state.EnsureCapacity(seqLen);
        if (scratchResized)
            InvalidateKernelCaches();

        ValidateTokenIds(tokenIds);
        UploadPositions(positions);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        _state.ResetHiddenSlot();
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // Pre-sublayer RMSNorm: HiddenState → NormOutput. Shared across all three kinds.
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            switch (kinds[layer])
            {
                case HybridLayerKind.Ssm:
                    RecordSsmLayer(cmdBuf, layer, lw.Ssm!.Value, seqLen, eps);
                    break;
                case HybridLayerKind.Attention:
                    RecordAttentionLayer(cmdBuf, layer, lw.Attention!.Value, seqLen, positions,
                        numHeads, numKvHeads, headDim, kvCache);
                    break;
                case HybridLayerKind.Ffn:
                    RecordFfnLayer(cmdBuf, lw.Ffn!.Value, seqLen);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {kinds[layer]} at layer {layer}.");
            }

            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Residual add: NewHidden = OldHidden + NormOutput. OldHidden aliases Residual
            // (same slot); the add writes into AddScratch (alternate slot); we then rotate.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            _state.RotateHiddenSlot();

            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // Final RMSNorm + LM head on the last token only.
        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: 1);

        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        return result;
    }

    /// <summary>
    /// Records the 12-step SSM sub-layer composed of the 6 SSM kernels plus reused matmuls.
    /// Mirrors the CPU oracle <c>NemotronHTransformerModel.ForwardSsmBody</c> step-for-step:
    /// (1) ssm_in matmul, (2) build conv_input by concatenating cached state with z's xBC slice,
    /// (3) conv1d_causal + (4) silu, (5) save the new state, (6) dt = dt slice + dtBias,
    /// (7) split xBC into x/B/C, (8) selective scan, (9) y += x*D, (10) silu(z)*y,
    /// (11) group rmsnorm, (12) ssm_out matmul into NormOutput.
    /// </summary>
    private void RecordSsmLayer(
        nint cmdBuf, int absoluteLayerIndex, VulkanNemotronHWeights.SsmLayerBuffers ssmW,
        int seqLen, float eps)
    {
        int dInner = _ssm.DInner;
        int dConv = _ssm.DConv;
        int nHead = _ssm.NHead;
        int headDim = _ssm.HeadDim;
        int dState = _ssm.DState;
        int nGroup = _ssm.NGroup;
        int convDim = _ssm.ConvDim;
        int groupDim = dInner / nGroup;
        int inProjDim = _ssm.InputProjectionDim;
        int bcDim = nGroup * dState;
        int dtOffset = 2 * dInner + 2 * nGroup * dState;

        int ssmOrdinal = _ssmLayerOrdinal[absoluteLayerIndex];
        var convStateBuf = _ssmCache.GetConvStateBuffer(ssmOrdinal);
        var ssmStateBuf = _ssmCache.GetSsmStateBuffer(ssmOrdinal);

        // 1. ssm_in matmul: NormOutput[seqLen, hidden] @ InWeight^T → Zxbcdt[seqLen, inProjDim]
        RecordMatmul(cmdBuf, ssmW.InWeight, ssmW.InDeviceQuantType, _state.NormOutput, _state.Zxbcdt,
            outputDim: ssmW.InOutputDim, inputDim: ssmW.InInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. ConvInput = concat(conv_state[(d_conv-1)*conv_dim], xBC rows from Zxbcdt)
        //    Step (a): copy cached conv_state into ConvInput[0 .. (d_conv-1)*conv_dim).
        //    Step (b): for each token t copy Zxbcdt[t, dInner..dInner+convDim] into
        //              ConvInput[((d_conv-1)+t)*convDim .. ].
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
        if (convStateBytes > 0)
        {
            RecordCopyBufferRange(cmdBuf, convStateBuf, _state.ConvInput,
                srcOffset: 0, dstOffset: 0, size: (ulong)convStateBytes);
        }
        long inProjRowBytes = (long)inProjDim * sizeof(float);
        long convDimBytes = (long)convDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong srcOff = (ulong)((long)t * inProjRowBytes + dInner * sizeof(float));
            ulong dstOff = (ulong)(((long)(dConv - 1) + t) * convDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.Zxbcdt, _state.ConvInput,
                srcOffset: srcOff, dstOffset: dstOff, size: (ulong)convDimBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // 3. Conv1d causal → XBC.
        _conv1dCausal.Record(cmdBuf, _state.ConvInput, ssmW.Conv1dWeight, ssmW.Conv1dBias, _state.XBC,
            dConv: dConv, channels: convDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. SiLU on XBC in place.
        _siluInplace.Record(cmdBuf, _state.XBC, n: seqLen * convDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 5. Save state: copy last (d_conv-1) rows of ConvInput (pre-SiLU values) back into
        //    conv_state. The CPU oracle reads from indices [seqLen, seqLen+1, …, seqLen+dConv-2]
        //    of ConvInput, i.e. the rows numbered seqLen..(seqLen+dConv-2). Copy
        //    contiguous (d_conv-1)*convDim bytes starting at byte-offset seqLen*convDimBytes.
        if (convStateBytes > 0)
        {
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            ulong saveSrcOffset = (ulong)((long)seqLen * convDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.ConvInput, convStateBuf,
                srcOffset: saveSrcOffset, dstOffset: 0, size: (ulong)convStateBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }

        // 6. dt = Zxbcdt[:, dtOffset..dtOffset+nHead] + DtBias.
        //    Per-token copy Zxbcdt slice → DtBuf, then BiasAdd in place.
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long dtRowBytes = (long)nHead * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong srcOff = (ulong)((long)t * inProjRowBytes + dtOffset * sizeof(float));
            ulong dstOff = (ulong)((long)t * dtRowBytes);
            RecordCopyBufferRange(cmdBuf, _state.Zxbcdt, _state.DtBuf,
                srcOffset: srcOff, dstOffset: dstOff, size: (ulong)dtRowBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _biasAdd.Record(cmdBuf, _state.DtBuf, ssmW.DtBias, seqLen, nHead);
        // The mamba2Scan call below is the consumer of DtBuf; the split kernel below
        // doesn't touch DtBuf, so it can run concurrently. The compute→compute barrier
        // after the split kernel covers both the bias_add → scan and split → scan deps.

        // 7. Split XBC[t, :] into SsmX[t, 0..dInner], SsmB[t, 0..bcDim], SsmC[t, 0..bcDim].
        //    XBC's row is laid out as [x | B | C]. One fused compute dispatch
        //    replaces the previous per-token loop of 3 vkCmdCopyBuffer regions
        //    (one each for x, B, C) — same math, no transfer↔compute stage
        //    transition, dispatch count drops from O(3·seqLen) to 1 per SSM
        //    layer. Bit-equal to the per-token-copy path (pure F32 strided
        //    load/store, no FP arithmetic).
        _ssmSplitXbc.Record(cmdBuf, _state.XBC, _state.SsmX, _state.SsmB, _state.SsmC,
            seqLen: seqLen, dInner: dInner, bcDim: bcDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 8. Mamba2 selective scan: state, SsmX, DtBuf, A, SsmB, SsmC -> SsmY.
        _mamba2Scan.Record(cmdBuf, ssmStateBuf, _state.SsmX, _state.DtBuf, ssmW.A,
            _state.SsmB, _state.SsmC, _state.SsmY,
            nHead: nHead, headDim: headDim, dState: dState, nGroup: nGroup, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 9. SsmY += SsmX * D
        _ssmDSkip.Record(cmdBuf, _state.SsmY, _state.SsmX, ssmW.D,
            seqLen: seqLen, nHead: nHead, headDim: headDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 10. Extract z = Zxbcdt[t, 0..dInner] into SsmZ, then SwiGLU(SsmZ, SsmY) → SsmY.
        //     Per-token copy of dInner-wide slice from row offset 0.
        long ssmXRowBytes = (long)dInner * sizeof(float);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        for (int t = 0; t < seqLen; t++)
        {
            ulong srcOff = (ulong)((long)t * inProjRowBytes);
            ulong dstOff = (ulong)((long)t * ssmXRowBytes);
            RecordCopyBufferRange(cmdBuf, _state.Zxbcdt, _state.SsmZ,
                srcOffset: srcOff, dstOffset: dstOff, size: (ulong)ssmXRowBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // SwiGLU computes silu(gate) * up — feed z as gate, SsmY as up. The kernel
        // requires distinct buffers, so we route the result back through SsmZ then
        // copy SsmZ → SsmY for the next steps. (SwiGLU into SsmY would alias 'up'.)
        _swiglu.Record(cmdBuf, _state.SsmZ, _state.SsmY, _state.SsmZ, n: seqLen * dInner);
        // Barrier the SwiGLU shader_write before the device-to-device copy.
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long ssmYTotalBytes = (long)seqLen * ssmXRowBytes;
        RecordCopyBufferRange(cmdBuf, _state.SsmZ, _state.SsmY,
            srcOffset: 0, dstOffset: 0, size: (ulong)ssmYTotalBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // 11. Group RMSNorm on SsmY in place.
        _groupRmsNorm.Record(cmdBuf, _state.SsmY, ssmW.NormWeight,
            seqLen: seqLen, nGroup: nGroup, groupDim: groupDim, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 12. ssm_out matmul: SsmY @ OutWeight^T → NormOutput.
        RecordMatmul(cmdBuf, ssmW.OutWeight, ssmW.OutDeviceQuantType, _state.SsmY, _state.NormOutput,
            outputDim: ssmW.OutOutputDim, inputDim: ssmW.OutInputDim, seqLen: seqLen);
    }

    /// <summary>
    /// Records the GQA attention sub-layer for one layer. Reads from <c>NormOutput</c> (the
    /// post-pre-norm activation) and writes the o_proj result back into <c>NormOutput</c>.
    /// </summary>
    private void RecordAttentionLayer(
        nint cmdBuf, int absoluteLayerIndex, VulkanNemotronHWeights.AttentionLayerBuffers attnW,
        int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int kvStride = numKvHeads * headDim;

        // Q/K/V projections — read NormOutput, write Q/K/V.
        RecordMatmul(cmdBuf, attnW.Q, attnW.QDeviceQuantType, _state.NormOutput, _state.Q,
            outputDim: attnW.QOutputDim, inputDim: attnW.QInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, attnW.K, attnW.KDeviceQuantType, _state.NormOutput, _state.K,
            outputDim: attnW.KOutputDim, inputDim: attnW.KInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, attnW.V, attnW.VDeviceQuantType, _state.NormOutput, _state.V,
            outputDim: attnW.VOutputDim, inputDim: attnW.VInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Partial RoPE on Q/K (only the first _ropeDim dims of each head).
        _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
            seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
            headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
            variant: RopeF32Kernel.Variant.Norm);

        VulkanDevice.Buffer kSrc, vSrc;
        int seqKv;
        int positionOffset;
        if (kvCache is VulkanNemotronHKvCache vkCache)
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, absoluteLayerIndex);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kSrc = vkCache.GetKeysBuffer(absoluteLayerIndex);
            vSrc = vkCache.GetValuesBuffer(absoluteLayerIndex);
            seqKv = vkCache.CurrentLength;
            positionOffset = positions[0];
        }
        else
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            kSrc = _state.K;
            vSrc = _state.V;
            seqKv = seqLen;
            positionOffset = 0;
        }

        _attention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
            seqQ: seqLen, seqKv: seqKv,
            numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            positionOffset: positionOffset, slidingWindow: 0);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Output projection → NormOutput (mirrors the GQA contract for the residual add).
        RecordMatmul(cmdBuf, attnW.O, attnW.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
            outputDim: attnW.OOutputDim, inputDim: attnW.OInputDim, seqLen: seqLen);
    }

    /// <summary>
    /// Records the squared-ReLU FFN sub-layer for one layer. Up matmul → ReluSquaredInplace
    /// → Down matmul into NormOutput.
    /// </summary>
    private void RecordFfnLayer(
        nint cmdBuf, VulkanNemotronHWeights.FfnLayerBuffers ffnW, int seqLen)
    {
        int intermediateSize = ffnW.UpOutputDim;

        RecordMatmul(cmdBuf, ffnW.Up, ffnW.UpDeviceQuantType, _state.NormOutput, _state.FfnIntermediate,
            outputDim: ffnW.UpOutputDim, inputDim: ffnW.UpInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _reluSquared.Record(cmdBuf, _state.FfnIntermediate, n: seqLen * intermediateSize);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, ffnW.Down, ffnW.DownDeviceQuantType, _state.FfnIntermediate, _state.NormOutput,
            outputDim: ffnW.DownOutputDim, inputDim: ffnW.DownInputDim, seqLen: seqLen);
    }

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _matmulQ8.InvalidateDescriptorCache();
        _matmulQ8Gemm.InvalidateDescriptorCache();
        _matmulQ8GemmCoopmat?.InvalidateDescriptorCache();
        _matmulQ4K.InvalidateDescriptorCache();
        _matmulQ4KGemm.InvalidateDescriptorCache();
        _matmulQ5K.InvalidateDescriptorCache();
        _matmulQ5KGemm.InvalidateDescriptorCache();
        _matmulQ6K.InvalidateDescriptorCache();
        _matmulQ6KGemm.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _rope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
        _swiglu.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
        _biasAdd.InvalidateDescriptorCache();
        _conv1dCausal.InvalidateDescriptorCache();
        _siluInplace.InvalidateDescriptorCache();
        _mamba2Scan.InvalidateDescriptorCache();
        _ssmDSkip.InvalidateDescriptorCache();
        _groupRmsNorm.InvalidateDescriptorCache();
        _reluSquared.InvalidateDescriptorCache();
        _ssmSplitXbc.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Dispatches a matmul for a single linear projection: chooses
    /// <see cref="MatMulQ8_0Kernel"/> (decode-path GEMV) when the device-side weight is
    /// Q8_0 and <paramref name="seqLen"/>==1, the batched <see cref="MatMulQ8_0GemmKernel"/>
    /// (or its coopmat variant when available) when Q8_0 and <paramref name="seqLen"/>&gt;1,
    /// and <see cref="MatMulF32Kernel"/> for every non-Q8_0 weight.
    /// </summary>
    /// <remarks>
    /// All Q8_0 kernels require <paramref name="inputDim"/> to be a multiple of 32 (the
    /// Q8_0 group size). The upload path (<see cref="VulkanNemotronHWeights"/>) only keeps
    /// Q8_0 sources on device when that constraint holds — otherwise the source is
    /// dequantised to F32 at upload and lands here as F32, sidestepping the kernel
    /// alignment requirement entirely.
    /// </remarks>
    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantizationType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (weightQt == QuantizationType.Q8_0)
        {
            if (seqLen == 1)
            {
                _matmulQ8.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else if (_matmulQ8GemmCoopmat is not null)
            {
                _matmulQ8GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ8Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q4_K)
        {
            if (seqLen == 1)
            {
                _matmulQ4K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ4KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q5_K)
        {
            // Q5_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM. Same
            // alignment requirement as Q4_K (inputDim % 256 == 0, enforced by upload).
            if (seqLen == 1)
            {
                _matmulQ5K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ5KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q6_K)
        {
            // Q6_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM. Same
            // alignment as Q4_K / Q5_K (inputDim % 256 == 0, enforced by upload path).
            if (seqLen == 1)
            {
                _matmulQ6K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ6KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else
        {
            _matmul.Record(cmdBuf, weights, input, output, outputDim, inputDim, seqLen);
        }
    }

    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    private void ValidateTokenIds(ReadOnlySpan<int> tokenIds)
    {
        int vocab = Config.VocabSize;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
        }
    }

    private void RecordEmbeddingGather(nint cmdBuf, ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        long rowBytes = (long)hiddenSize * sizeof(float);
        var srcBuf = _weights.TokenEmbedding.Handle;
        var dstBuf = _state.HiddenState.Handle;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            var region = new VkBufferCopy
            {
                srcOffset = (ulong)((long)id * rowBytes),
                dstOffset = (ulong)((long)t * rowBytes),
                size = (ulong)rowBytes,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, srcBuf, dstBuf, 1, region);
        }
    }

    private void UploadPositions(ReadOnlySpan<int> positions)
    {
        var posBytes = MemoryMarshal.AsBytes(positions);
        _device.Upload(posBytes, _state.PositionsBuffer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _ssmCache.Dispose();

        _ssmSplitXbc.Dispose();
        _reluSquared.Dispose();
        _groupRmsNorm.Dispose();
        _ssmDSkip.Dispose();
        _mamba2Scan.Dispose();
        _siluInplace.Dispose();
        _conv1dCausal.Dispose();
        _biasAdd.Dispose();
        _add.Dispose();
        _swiglu.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _matmulQ6KGemm.Dispose();
        _matmulQ6K.Dispose();
        _matmulQ5KGemm.Dispose();
        _matmulQ5K.Dispose();
        _matmulQ4KGemm.Dispose();
        _matmulQ4K.Dispose();
        _matmulQ8GemmCoopmat?.Dispose();
        _matmulQ8Gemm.Dispose();
        _matmulQ8.Dispose();
        _matmul.Dispose();

        if (_ownsDevice)
            _device.Dispose();
    }
}
