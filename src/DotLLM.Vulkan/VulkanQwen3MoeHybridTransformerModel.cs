using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for the Qwen3MoeHybrid architecture
/// (Gated DeltaNet recurrence + sparse MoE FFN — 40 layers, every fourth
/// layer is full GQA attention). Mirrors the verified CPU reference in
/// <see cref="Qwen3MoeHybridTransformerModel"/> step-for-step at the
/// command-buffer level so the tensor-dump parity rig can validate.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> Token-mixing (GDN / full-attn) runs fully on device through
/// the SSM/GQA kernel set plus the six GDN-specific kernels (L2-normalize-heads,
/// scan-step, multi-token scan, post-scan gate, sigmoid-gate-mul, gdn_decay,
/// sigmoid-inplace). MoE routed experts default to streaming (re-uploaded
/// every forward — correctness-first, fits any model size); set
/// <c>DOTLLM_VK_MOE_RESIDENT=1</c> to opt in to per-layer resident caching.
/// Resident mode auto-detects uniformly Q6_K source banks at upload time
/// and keeps them on device as raw Q6_K blocks (≈25 GB at qwen35moe-A3B
/// scale on a 128 GB Strix Halo unified-memory host — fits) dispatching
/// through <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ6_KF32Kernel"/>.
/// Non-Q6_K source banks (or mixed-quant layers) fall back to F32 dequant +
/// upload — fits when the model is small enough that ≈4× expansion stays
/// under device-memory bounds; at Qwen3.6-35B-A3B scale (256 experts × 40
/// layers × 3 matrices × ~1M elems) the fully-F32 resident layout would
/// consume ~120 GB and would NOT fit, so the Q6_K-resident path is the
/// only resident option for Qwen3.6-A3B-UD-Q6_K_XL.
/// </para>
/// <para>
/// <b>Submission boundaries.</b> Two submissions per layer × 40 layers + a
/// final submission for the LM head. The previous per-GDN-layer mid-body
/// submit/wait (from the host-side decay+sigmoid path) has been removed by
/// the on-device gdn_decay_f32 + sigmoid_inplace_f32 fusion.
/// </para>
/// <para>
/// <b>Bit-parity targets.</b> Every dispatch path keeps the same FP32
/// rounding order as the CPU reference. New shaders are documented in
/// <c>native/vulkan/shaders/gdn_*.comp</c> with their parity invariants;
/// transcendental kernels (decay / sigmoid) target ≤4 ULP drift.
/// </para>
/// </remarks>
public sealed class VulkanQwen3MoeHybridTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly GgufFile? _gguf;

    // The CPU model retains the GGUF mmap and the raw quant views consumed by
    // VulkanQwen3MoeMoeUpload on every forward — keeping it alive for the
    // lifetime of the Vulkan model is mandatory.
    private readonly Qwen3MoeHybridTransformerModel? _cpuModel;

    // Per-layer device-resident weights for the token-mixing path. MoE
    // weights live on the CPU side (as Qwen3MoeLayerWeights[].Moe) and are
    // uploaded on demand per layer per forward — see _moeLayerBuffersSlot.
    private readonly VulkanQwen3MoeHybridWeights _weights;
    private readonly Qwen3MoeLayerWeights[] _cpuLayers;
    private readonly VulkanQwen3MoeHybridForwardState _state;
    private readonly VulkanGdnStateCache _gdnCache;
    private readonly VulkanQwen3MoeHybridKernels _kernels;

    // Hybrid layout: kind per layer + sparse KV-slot mapping for attention layers only.
    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _gdnLayerOrdinal;

    // RoPE precomputed tables; uploaded once into device buffers shared across
    // every attention layer.
    private readonly int _ropeDim;
    private readonly float _ropeTheta;

    private readonly VulkanDevice.SubmitContext _submit;

    // Per-layer resident MoE bundles. When `_residentMoeEnabled` is true
    // (opt-in via DOTLLM_VK_MOE_RESIDENT=1), each layer's routed experts are
    // uploaded once on first use and retained for the lifetime of the model
    // — eliminating the dequant + host→device upload cost from subsequent
    // forwards. The default is streaming-mode (re-upload every forward),
    // which is the correctness-first path: the routed banks currently
    // dequantise to F32 (see VulkanQwen3MoeMoeUpload remarks), so a fully
    // resident layout for Qwen3.6-A3B at qwen35moe scale (256 experts × 40
    // layers × 3 matrices, ~120 GB F32) cannot fit in unified memory on any
    // current single-device host. Resident mode is only safe for smaller
    // models or once a quantized MoE matmul shader lands (follow-up
    // Priority 3) — hence the explicit opt-in.
    private readonly VulkanQwen3MoeMoeUpload.LayerBundle?[] _residentMoeBundles;
    private readonly bool _residentMoeEnabled;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes =>
        _state.AllocatedBytes + _weights.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>Creates a sparse <see cref="VulkanNemotronHKvCache"/> sized for this model.</summary>
    public VulkanNemotronHKvCache CreateKvCache(int maxSeqLen)
        => new(_device, _kvSlotForLayer, _attentionLayerCount,
               Config.NumKvHeads, Config.HeadDim, maxSeqLen);

    private VulkanQwen3MoeHybridTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config,
        GgufFile? gguf,
        Qwen3MoeHybridTransformerModel? cpuModel,
        Qwen3MoeLayerWeights[] cpuLayers,
        VulkanQwen3MoeHybridWeights weights,
        VulkanQwen3MoeHybridForwardState state,
        VulkanGdnStateCache gdnCache,
        VulkanQwen3MoeHybridKernels kernels,
        int[] kvSlotForLayer, int attentionLayerCount,
        int[] gdnLayerOrdinal,
        int ropeDim, float ropeTheta)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _gguf = gguf;
        _cpuModel = cpuModel;
        _cpuLayers = cpuLayers;
        _weights = weights;
        _state = state;
        _gdnCache = gdnCache;
        _kernels = kernels;
        _layout = config.HybridLayout!;
        _gdn = config.GdnConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _gdnLayerOrdinal = gdnLayerOrdinal;
        _ropeDim = ropeDim;
        _ropeTheta = ropeTheta;

        _submit = device.CreateSubmitContext();

        _residentMoeEnabled =
            string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT"), "1", StringComparison.Ordinal);
        _residentMoeBundles = new VulkanQwen3MoeMoeUpload.LayerBundle?[cpuLayers.Length];
    }

    /// <summary>
    /// Loads the Qwen3MoeHybrid model from a GGUF file onto a Vulkan device.
    /// Token-mixing weights upload immediately; MoE routed experts stay on
    /// the host (as raw quant views inside <see cref="Qwen3MoeLayerWeights.Moe"/>)
    /// and are streamed to the GPU on demand per layer.
    /// </summary>
    public static VulkanQwen3MoeHybridTransformerModel BuildFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(spvDir);

        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"VulkanQwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));

        // Reuse the CPU loader to derive Qwen3MoeLayerWeights[] (which holds the
        // raw quant view of routed experts and the small F32 norm/conv vectors).
        // The CPU model owns dispose of the GgufFile mmap; we keep it alive too.
        var cpuModel = Qwen3MoeHybridTransformerModel.LoadFromGguf(gguf, config);
        var cpuLayers = ExtractCpuLayers(cpuModel);
        var outputNormWeight = ExtractOutputNormWeight(cpuModel);
        var (tokenEmbedPtr, tokenEmbedQt) = ExtractTokenEmbed(cpuModel);
        var (outputPtr, outputQt, outputM, outputK) = ExtractOutput(cpuModel);

        var layout = config.HybridLayout!;
        var gdn = config.GdnConfig!.Value;

        var kvSlotForLayer = new int[config.NumLayers];
        var gdnLayerOrdinal = new int[config.NumLayers];
        int attentionLayerCount = 0;
        int gdnOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (layout.LayerKind[i] == HybridLayerKind.Attention)
            {
                kvSlotForLayer[i] = attentionLayerCount++;
                gdnLayerOrdinal[i] = -1;
            }
            else
            {
                kvSlotForLayer[i] = -1;
                gdnLayerOrdinal[i] = gdnOrdinal++;
            }
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        if (attentionLayerCount > 0)
        {
            if ((ropeDim & 1) != 0)
                throw new InvalidDataException(
                    $"Qwen3MoeHybrid rope_dim={ropeDim} must be even for pair-wise rotation.");
            if (ropeDim > config.HeadDim)
                throw new InvalidDataException(
                    $"Qwen3MoeHybrid rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        }

        // Upload all token-mixing weights (norm, GDN per-layer, full-attn per-layer,
        // token embedding, output norm, LM head). Routed MoE banks are NOT uploaded
        // here — they live on host and stream per layer in the forward pass.
        var weights = VulkanQwen3MoeHybridWeights.Upload(device, config, cpuLayers, outputNormWeight,
            tokenEmbedPtr, tokenEmbedQt, outputPtr, outputQt, outputM, outputK);

        var state = new VulkanQwen3MoeHybridForwardState(device, config, gdn, initialSeqLen: 1);
        var gdnCache = new VulkanGdnStateCache(device, gdn, gdnOrdinal);

        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir);

        return new VulkanQwen3MoeHybridTransformerModel(
            device, ownsDevice: false,
            config, gguf, cpuModel, cpuLayers, weights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, gdnLayerOrdinal,
            ropeDim, ropeTheta);
    }

    // ── CPU-model accessors (we share the CPU loader; reach into its layers) ─

    private static Qwen3MoeLayerWeights[] ExtractCpuLayers(Qwen3MoeHybridTransformerModel m)
    {
        // The CPU model holds `_layers` privately. Surface it via reflection — the
        // alternative is plumbing a public accessor through DotLLM.Models, which
        // would widen the public API for a single internal consumer.
        var fi = typeof(Qwen3MoeHybridTransformerModel)
            .GetField("_layers", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException("Qwen3MoeHybridTransformerModel._layers field missing.");
        return (Qwen3MoeLayerWeights[])fi.GetValue(m)!;
    }

    private static float[] ExtractOutputNormWeight(Qwen3MoeHybridTransformerModel m)
    {
        var fi = typeof(Qwen3MoeHybridTransformerModel)
            .GetField("_outputNormWeight", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return (float[])fi.GetValue(m)!;
    }

    private static (nint ptr, QuantizationType qt) ExtractTokenEmbed(Qwen3MoeHybridTransformerModel m)
    {
        var t = typeof(Qwen3MoeHybridTransformerModel);
        var ptr = (nint)t.GetField("_tokenEmbedWeight", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        var qt = (QuantizationType)t.GetField("_tokenEmbedQuantType", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        return (ptr, qt);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) ExtractOutput(Qwen3MoeHybridTransformerModel m)
    {
        var t = typeof(Qwen3MoeHybridTransformerModel);
        var ptr = (nint)t.GetField("_outputWeight", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        var qt = (QuantizationType)t.GetField("_outputQuantType", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        var outDim = (int)t.GetField("_outputOutputDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        var inDim = (int)t.GetField("_outputInputDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(m)!;
        return (ptr, qt, outDim, inDim);
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
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        bool resized = _state.EnsureCapacity(seqLen);
        if (resized) _kernels.InvalidateAll();

        UploadPositions(positions);

        var kinds = _layout.LayerKind;

        // ── 1. Token embedding (single submission) ────────────────────────────
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // ── 2. Per-layer body. Two submissions per layer: one for the
        //      token-mixing path, one for the MoE FFN (which needs a host
        //      dequant + upload of the routed experts in between). ────────────
        for (int layer = 0; layer < _cpuLayers.Length; layer++)
        {
            var lw = _cpuLayers[layer];
            ref readonly var layerBuf = ref _weights.Layers[layer];

            // ── 2a. Token-mixing submission ─────────────────────────────────
            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmdBuf);

            // Snapshot hidden → residual (HiddenState aliases the residual slot
            // in the ping-pong; we use a dedicated explicit copy for clarity at
            // the cost of one extra device copy per layer — bit-identical and
            // simpler than the rotate-slot trick in NemotronH).
            RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            if (kinds[layer] == HybridLayerKind.GatedDeltaNet)
            {
                RecordGdnLayer(cmdBuf, layer, layerBuf.Gdn!.Value, seqLen, eps);
            }
            else
            {
                RecordFullAttnLayer(cmdBuf, layer, layerBuf.Attention!.Value, seqLen, positions,
                    numHeads, numKvHeads, headDim, kvCache);
            }
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // First residual add: HiddenState = Residual + NormOutput (token-mixing output).
            //   AddScratch is reused later as MoE intermediates; here it just receives the sum
            //   so we can copy it back into HiddenState in one transfer.
            _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
                seqLen * hiddenSize);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();

            // ── 2b. MoE submission. Resolve this layer's routed experts:
            //        either fetch a resident bundle (opt-in) or upload fresh
            //        and dispose after the layer (default — safe for any
            //        model size). When resident-mode is on, the upload also
            //        opts into a Q6_K-resident bank when the source allows
            //        (~25 GB Q6_K vs ~120 GB F32 at qwen35moe-A3B scale —
            //        the only way the resident layout fits on a 128 GB
            //        Strix Halo unified-memory host). ──────────────────────
            VulkanQwen3MoeMoeUpload.LayerBundle moeBuf;
            bool disposeAfterLayer;
            if (_residentMoeEnabled)
            {
                // Lazily upload on first use; retained for the life of the
                // model after that. See _residentMoeBundles field docstring
                // for the device-memory caveat — DOTLLM_VK_MOE_RESIDENT=1
                // is opt-in for models that fit.
                moeBuf = _residentMoeBundles[layer]
                    ?? (_residentMoeBundles[layer] = VulkanQwen3MoeMoeUpload.UploadLayer(
                        _device, lw.Moe, hiddenSize, residentQuant: true));
                disposeAfterLayer = false;
            }
            else
            {
                moeBuf = VulkanQwen3MoeMoeUpload.UploadLayer(_device, lw.Moe, hiddenSize);
                disposeAfterLayer = true;
            }

            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmdBuf);

            // Second residual snapshot (HiddenState now holds the updated activations).
            RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            RecordMoeLayer(cmdBuf, moeBuf, layerBuf.PostAttnNormWeight, seqLen, hiddenSize, eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Second residual add.
            _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
                seqLen * hiddenSize);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();

            // In streaming mode, free this layer's transient banks before
            // moving to the next layer. In resident mode the bundle is kept
            // alive on _residentMoeBundles and only disposed at model Dispose.
            if (disposeAfterLayer) moeBuf.Dispose();
        }

        // ── 3. Final norm + LM head (single submission, last token only) ──────
        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.RmsNorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // ── 4. Download logits ─────────────────────────────────────────────────
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        return result;
    }

    // ── Token-mixing path: Gated DeltaNet ────────────────────────────────────

    /// <summary>
    /// Records the GDN token-mixing forward for one layer. Mirrors the CPU
    /// <c>Qwen3MoeHybridTransformerModel.ForwardGdnBody</c>:
    /// (1) project QKV / gate / alpha / beta; (2) decay g = exp(softplus(α+dt)·A) +
    /// sigmoid(β) — fused into a small fixup kernel; (3) Conv1d + SiLU on the
    /// QKV concat; (4) de-interleave Q/K/V, L2-normalise Q and K; (5) seqLen
    /// dispatches of GdnScanStep advancing the state; (6) per-head RMSNorm
    /// + silu(z) gate via the fused post-scan kernel; (7) ssm_out projection
    /// back into NormOutput.
    /// </summary>
    private unsafe void RecordGdnLayer(
        nint cmdBuf, int absoluteLayerIdx, VulkanQwen3MoeHybridWeights.GdnLayerBuffers gdnW,
        int seqLen, float eps)
    {
        int nVHead = _gdn.NVHead;
        int nKHead = _gdn.NKHead;
        int dState = _gdn.DState;
        int dConv = _gdn.DConv;
        int convDim = (2 * nKHead + nVHead) * dState;
        int vDim = nVHead * dState;
        int kDim = nKHead * dState;
        int gdnOrdinal = _gdnLayerOrdinal[absoluteLayerIdx];

        var convStateBuf = _gdnCache.GetConvStateBuffer(gdnOrdinal);
        var gdnStateBuf = _gdnCache.GetGdnStateBuffer(gdnOrdinal);

        // ── 1. Projections ───────────────────────────────────────────────────
        RecordMatmul(cmdBuf, gdnW.QkvWeight, gdnW.QkvDeviceQuantType,
            _state.NormOutput, _state.GdnQkvBuf,
            outputDim: gdnW.QkvOutputDim, inputDim: gdnW.QkvInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.GateWeight, gdnW.GateDeviceQuantType,
            _state.NormOutput, _state.GdnZBuf,
            outputDim: gdnW.GateOutputDim, inputDim: gdnW.GateInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.AlphaWeight, gdnW.AlphaDeviceQuantType,
            _state.NormOutput, _state.GdnAlphaBuf,
            outputDim: gdnW.AlphaOutputDim, inputDim: gdnW.AlphaInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.BetaWeight, gdnW.BetaDeviceQuantType,
            _state.NormOutput, _state.GdnBetaBuf,
            outputDim: gdnW.BetaOutputDim, inputDim: gdnW.BetaInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 2. Fused on-device decay g and sigmoid(β) ─────────────────────────
        // gdn_decay_f32 fuses (alpha + dt_bias) → softplus → * A → exp into one
        // dispatch over GdnAlphaBuf, then sigmoid_inplace_f32 maps the β
        // projection to the write-gate. Together they replace the previous
        // ComputeDecayAndBetaOnHost roundtrip — eliminating a D2H/H2D pair
        // plus a mid-layer submit/wait per GDN layer (30 GDN layers per forward
        // at qwen35moe-Q35B-A3B scale).
        _kernels.GdnDecay.Record(cmdBuf, _state.GdnAlphaBuf, gdnW.DtBiasDevice, gdnW.ADevice,
            seqLen: seqLen, nVHead: nVHead);
        _kernels.SigmoidInplace.Record(cmdBuf, _state.GdnBetaBuf, n: seqLen * nVHead);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 3. Build conv input + Conv1d + SiLU ───────────────────────────────
        // ConvInput = [convState (DConv-1 rows) | qkvBuf (seqLen rows)]
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
        if (convStateBytes > 0)
        {
            RecordCopyBufferRange(cmdBuf, convStateBuf, _state.GdnConvInput,
                srcOffset: 0, dstOffset: 0, size: (ulong)convStateBytes);
        }
        long convDimBytes = (long)convDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong srcOff = (ulong)((long)t * convDimBytes);
            ulong dstOff = (ulong)(((long)(dConv - 1) + t) * convDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, _state.GdnConvInput,
                srcOffset: srcOff, dstOffset: dstOff, size: (ulong)convDimBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.Conv1dCausal.Record(cmdBuf, _state.GdnConvInput, gdnW.Conv1dWeight, gdnW.Conv1dBias,
            _state.GdnQkvBuf, dConv: dConv, channels: convDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _kernels.SiluInplace.Record(cmdBuf, _state.GdnQkvBuf, n: seqLen * convDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Save the trailing (dConv-1) rows of ConvInput back to convState.
        // The CPU reference reads from rows seqLen..(seqLen+dConv-2) of the
        // pre-SiLU ConvInput (NOT the convolved output). Same offset pattern
        // as VulkanNemotronH SSM forward.
        if (convStateBytes > 0)
        {
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            ulong saveSrc = (ulong)((long)seqLen * convDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.GdnConvInput, convStateBuf,
                srcOffset: saveSrc, dstOffset: 0, size: (ulong)convStateBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }

        // ── 4. De-interleave Q/K/V and L2-normalise Q and K ──────────────────
        // GdnQkvBuf layout per token: [Q(kDim) | K(kDim) | V(vDim)]
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long kDimBytes = (long)kDim * sizeof(float);
        long vDimBytes = (long)vDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong rowBase = (ulong)((long)t * convDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, _state.GdnQBuf,
                srcOffset: rowBase, dstOffset: (ulong)((long)t * kDimBytes), size: (ulong)kDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, _state.GdnKBuf,
                srcOffset: rowBase + (ulong)kDimBytes, dstOffset: (ulong)((long)t * kDimBytes), size: (ulong)kDimBytes);
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, _state.GdnVBuf,
                srcOffset: rowBase + (ulong)(2 * kDimBytes), dstOffset: (ulong)((long)t * vDimBytes), size: (ulong)vDimBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnQBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnKBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 5. GDN scan — single multi-token dispatch ────────────────────────
        // GdnScanMultiToken walks the seqLen loop INSIDE the shader, mutating
        // the per-sequence state matrix between tokens. Replaces the previous
        // host-driven O(seqLen) per-token dispatch + 6 D2D copies per token.
        // Same bit-parity guarantees as the per-token shader, by construction.
        _kernels.GdnScanMultiToken.Record(cmdBuf,
            state: gdnStateBuf,
            q: _state.GdnQBuf, k: _state.GdnKBuf, v: _state.GdnVBuf,
            g: _state.GdnAlphaBuf, beta: _state.GdnBetaBuf,
            output: _state.GdnOut,
            seqLen: seqLen, nVHead: nVHead, nKHead: nKHead, dState: dState);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 6. Per-head RMSNorm × silu(z) gate (fused) ───────────────────────
        _kernels.GdnPostScanGate.Record(cmdBuf,
            gdnOut: _state.GdnOut, z: _state.GdnZBuf, ssmNormWeight: gdnW.SsmNormWeight,
            seqLen: seqLen, nVHead: nVHead, dState: dState, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 7. ssm_out projection back into NormOutput ───────────────────────
        RecordMatmul(cmdBuf, gdnW.OutWeight, gdnW.OutDeviceQuantType,
            _state.GdnOut, _state.NormOutput,
            outputDim: gdnW.OutOutputDim, inputDim: gdnW.OutInputDim, seqLen: seqLen);
    }

    // ── Token-mixing path: full GQA attention ────────────────────────────────

    /// <summary>
    /// Records the full-attention forward for one layer (every fourth layer
    /// at qwen35moe interval=4). Q+Gate are fused in <c>attn_q</c> at output
    /// width <c>2 * nQ * headDim</c>; we de-interleave per head before
    /// QK-norm, RoPE and attention.
    /// </summary>
    private unsafe void RecordFullAttnLayer(
        nint cmdBuf, int absoluteLayerIdx, VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers attnW,
        int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int qElems = numHeads * headDim;
        int qgElems = 2 * qElems;
        int kvStride = numKvHeads * headDim;

        // 1. Fused Q+Gate projection.
        RecordMatmul(cmdBuf, attnW.QWeight, attnW.QDeviceQuantType,
            _state.NormOutput, _state.QGateScratch,
            outputDim: attnW.QOutputDim, inputDim: attnW.QInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. De-interleave per head into Q and Gate scratch buffers.
        //    Per token row: [Q_h0, Gate_h0, Q_h1, Gate_h1, ...] each headDim wide.
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long headBytes = (long)headDim * sizeof(float);
        long qRowBytes = (long)qElems * sizeof(float);
        long qgRowBytes = (long)qgElems * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong qgRowBase = (ulong)((long)t * qgRowBytes);
            ulong qRowBase = (ulong)((long)t * qRowBytes);
            for (int h = 0; h < numHeads; h++)
            {
                ulong qgHeadOff = qgRowBase + (ulong)(h * 2 * headBytes);
                ulong qHeadOff = qRowBase + (ulong)(h * headBytes);
                RecordCopyBufferRange(cmdBuf, _state.QGateScratch, _state.Q,
                    srcOffset: qgHeadOff, dstOffset: qHeadOff, size: (ulong)headBytes);
                RecordCopyBufferRange(cmdBuf, _state.QGateScratch, _state.GateScratch,
                    srcOffset: qgHeadOff + (ulong)headBytes, dstOffset: qHeadOff, size: (ulong)headBytes);
            }
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // 3. K and V projections.
        RecordMatmul(cmdBuf, attnW.KWeight, attnW.KDeviceQuantType,
            _state.NormOutput, _state.K,
            outputDim: attnW.KOutputDim, inputDim: attnW.KInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, attnW.VWeight, attnW.VDeviceQuantType,
            _state.NormOutput, _state.V,
            outputDim: attnW.VOutputDim, inputDim: attnW.VInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. QK-norm — per-head RMSNorm with attn_q_norm / attn_k_norm weights.
        //    Reshape as [seqLen * numHeads, headDim] rows for the RMSNorm kernel.
        _kernels.RmsNorm.Record(cmdBuf, _state.Q, attnW.QNormWeight, _state.Q,
            rowCount: seqLen * numHeads, n: headDim, eps: Config.NormEpsilon);
        _kernels.RmsNorm.Record(cmdBuf, _state.K, attnW.KNormWeight, _state.K,
            rowCount: seqLen * numKvHeads, n: headDim, eps: Config.NormEpsilon);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 5. RoPE — NeoX pair pattern over the first ropeDim of each head.
        //    NOTE: the CPU reference flags this as UNVERIFIED for qwen35moe;
        //    we mirror its choice (NeoX) so device output matches CPU output.
        _kernels.Rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
            seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
            headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
            variant: RopeF32Kernel.Variant.NeoX);

        // 6. Attention.
        VulkanDevice.Buffer kSrc, vSrc;
        int seqKv, positionOffset;
        if (kvCache is VulkanNemotronHKvCache vkCache)
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, absoluteLayerIdx);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kSrc = vkCache.GetKeysBuffer(absoluteLayerIdx);
            vSrc = vkCache.GetValuesBuffer(absoluteLayerIdx);
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

        if (_kernels.FlashAttention is not null && seqLen > 1 && headDim <= VulkanFlashAttentionF32Kernel.MaxHeadDim)
        {
            _kernels.FlashAttention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else
        {
            _kernels.Attention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 7. Apply sigmoid(gate) element-wise to attention output.
        _kernels.SigmoidGateMul.Record(cmdBuf, _state.AttnOutput, _state.GateScratch,
            nTotal: seqLen * qElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 8. Output projection.
        RecordMatmul(cmdBuf, attnW.OWeight, attnW.ODeviceQuantType,
            _state.AttnOutput, _state.NormOutput,
            outputDim: attnW.OOutputDim, inputDim: attnW.OInputDim, seqLen: seqLen);
    }

    // ── MoE FFN ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Records the routed-MoE SwiGLU FFN dispatch using the per-layer banks
    /// uploaded by <see cref="VulkanQwen3MoeMoeUpload.UploadLayer"/>. Mirrors
    /// the routed path of <see cref="VulkanTransformerModel"/>'s
    /// <c>RecordMoeLayer</c> and folds in the optional Qwen1.5-MoE-style
    /// sigmoid-gated shared expert.
    /// </summary>
    private unsafe void RecordMoeLayer(
        nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW,
        VulkanDevice.Buffer postAttnNormWeight, int seqLen, int hidden, float eps)
    {
        int interm = moeW.IntermediateSize;
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int expandedRows = seqLen * topK;

        // 1. Router gate logits.
        RecordMatmul(cmdBuf, moeW.Gate, QuantizationType.F32,
            _state.NormOutput, _state.MoeRouterLogits,
            outputDim: numE, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. Top-k softmax.
        _kernels.MoeTopkSoftmax.Record(cmdBuf,
            _state.MoeRouterLogits, _state.MoeTopkIndices, _state.MoeTopkWeights,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 3. Broadcast NormOutput[seqLen, hidden] → MoeExpandedInput[seqLen*topK, hidden].
        _kernels.MoeBroadcast.Record(cmdBuf,
            _state.NormOutput, _state.MoeExpandedInput,
            seqLen: seqLen, topK: topK, hidden: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. Indexed expert matmuls. Two paths share the same buffer
        //    contract (bank/x/indices/y) and the same shape (m, k, n,
        //    numExperts) — only the dequant differs:
        //       F32 banks  → MoeIndexedMatmul (plain F32 dot)
        //       Q6_K banks → MoeIndexedMatmulQ6K (per-row Q6_K dequant in
        //                    the inner loop)
        //    See VulkanQwen3MoeMoeUpload remarks for the residency caveat.
        RecordIndexedMoeMatmul(cmdBuf, moeW,
            moeW.W1Bank, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeGateInter,
            m: interm, k: hidden, n: expandedRows, numExperts: numE);
        RecordIndexedMoeMatmul(cmdBuf, moeW,
            moeW.W3Bank, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeUpInter,
            m: interm, k: hidden, n: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 5. SwiGLU: silu(gate) * up
        _kernels.SwiGlu.Record(cmdBuf, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter,
            n: expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 6. Indexed down matmul.
        RecordIndexedMoeMatmul(cmdBuf, moeW,
            moeW.W2Bank, _state.MoeSiluInter, _state.MoeTopkIndices, _state.MoeDownRows,
            m: hidden, k: interm, n: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 7. Weighted scatter into NormOutput.
        _kernels.MoeWeightedScatter.Record(cmdBuf,
            _state.MoeDownRows, _state.MoeTopkWeights, _state.NormOutput,
            seqLen: seqLen, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 8. Shared-expert branch (Qwen1.5-MoE sigmoid-gated convention).
        if (moeW.HasSharedExpert)
        {
            RecordSharedExpert(cmdBuf, moeW, postAttnNormWeight, seqLen, hidden, eps);
        }
    }

    /// <summary>
    /// Per-bank-quant-type dispatcher for a routed-expert indexed matmul. Bank
    /// storage type is recorded once at upload time on
    /// <see cref="VulkanQwen3MoeMoeUpload.LayerBundle.BankQuantType"/>; this
    /// helper picks the matching kernel so the caller stays oblivious to
    /// whether the layer is F32-streaming or Q6_K-resident.
    /// </summary>
    private void RecordIndexedMoeMatmul(
        nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        switch (moeW.BankQuantType)
        {
            case QuantizationType.Q6_K:
                _kernels.MoeIndexedMatmulQ6K.Record(cmdBuf, bank, x, indices, y,
                    m: m, k: k, n: n, numExperts: numExperts);
                break;
            case QuantizationType.F32:
                _kernels.MoeIndexedMatmul.Record(cmdBuf, bank, x, indices, y,
                    m: m, k: k, n: n, numExperts: numExperts);
                break;
            default:
                // Other quant types aren't wired through the resident-bank
                // upload path yet — UploadLayer falls back to F32, so we
                // should never see them here. Defensive throw catches a
                // future upload-side regression that introduces a new bank
                // quant type without updating this dispatch site.
                throw new InvalidOperationException(
                    $"Unsupported MoE bank quant type: {moeW.BankQuantType}. " +
                    "Add a kernel dispatch arm and an upload-side branch in " +
                    "VulkanQwen3MoeMoeUpload.UploadLayer.");
        }
    }

    /// <summary>
    /// Records the optional Qwen1.5-MoE shared-expert branch: SwiGLU MLP over
    /// the same RMSNormed hidden state, with a per-token sigmoid gate folding
    /// the output into <c>NormOutput</c> via <see cref="MoeSigmoidGatedAddF32Kernel"/>.
    /// </summary>
    private void RecordSharedExpert(
        nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW,
        VulkanDevice.Buffer postAttnNormWeight, int seqLen, int hidden, float eps)
    {
        int sharedI = moeW.SharedIntermediateSize;
        int sharedInterElems = seqLen * sharedI;

        // Shared input = the same RMSNormed hidden state we used for the routed
        // path. The routed scatter overwrote NormOutput so we re-derive it.
        _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, postAttnNormWeight, _state.MoeSharedInput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Shared expert gate/up matmuls share the input.
        RecordMatmul(cmdBuf, moeW.SharedGate!, QuantizationType.F32,
            _state.MoeSharedInput, _state.MoeSharedGate,
            outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
        RecordMatmul(cmdBuf, moeW.SharedUp!, QuantizationType.F32,
            _state.MoeSharedInput, _state.MoeSharedUp,
            outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _kernels.SwiGlu.Record(cmdBuf, _state.MoeSharedGate, _state.MoeSharedUp, _state.MoeSharedSilu,
            n: sharedInterElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, moeW.SharedDown!, QuantizationType.F32,
            _state.MoeSharedSilu, _state.MoeSharedSumA,
            outputDim: hidden, inputDim: sharedI, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        if (moeW.SharedExpertGate is not null)
        {
            // gateLogits[t] = SharedExpertGate[1, hidden] @ MoeSharedInput[t, :].
            RecordMatmul(cmdBuf, moeW.SharedExpertGate, QuantizationType.F32,
                _state.MoeSharedInput, _state.MoeSharedGateLogits,
                outputDim: 1, inputDim: hidden, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _kernels.MoeSigmoidGatedAdd.Record(cmdBuf,
                output: _state.NormOutput, b: _state.MoeSharedSumA, gateLogits: _state.MoeSharedGateLogits,
                seqLen: seqLen, hiddenSize: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
            // Plain add into NormOutput via a ping-pong destination.
            _kernels.Add.Record(cmdBuf, _state.NormOutput, _state.MoeSharedSumA, _state.MoeSharedSumB,
                seqLen * hidden);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.MoeSharedSumB, _state.NormOutput,
                0, 0, (ulong)((long)seqLen * hidden * sizeof(float)));
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
    }

    // ── Matmul dispatcher (mirrors VulkanNemotronHTransformerModel.RecordMatmul) ─

    /// <summary>
    /// Per-quant-type matmul dispatcher. Routes Q8_0 / Q2_K / Q3_K / Q4_K / Q5_K
    /// / Q6_K / F16 / BF16 / F32 weights through the matching kernel selected by
    /// the device storage type recorded at upload time.
    /// </summary>
    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantizationType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        switch (weightQt)
        {
            case QuantizationType.Q8_0:
                if (seqLen == 1)
                    _kernels.MatMulQ8.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ8GemmCoopmat is not null)
                    _kernels.MatMulQ8GemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ8Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q2_K:
                if (seqLen == 1)
                    _kernels.MatMulQ2K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulQ2KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q3_K:
                if (seqLen == 1)
                    _kernels.MatMulQ3K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulQ3KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q4_K:
                if (seqLen == 1)
                    _kernels.MatMulQ4K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulQ4KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q5_K:
                if (seqLen == 1)
                    _kernels.MatMulQ5K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulQ5KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q6_K:
                if (seqLen == 1)
                    _kernels.MatMulQ6K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulQ6KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ4_NL:
                if (seqLen == 1)
                    _kernels.MatMulIq4Nl.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq4NlGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ4_XS:
                if (seqLen == 1)
                    _kernels.MatMulIq4Xs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq4XsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ2_XXS:
                if (seqLen == 1)
                    _kernels.MatMulIq2Xxs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq2XxsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ2_XS:
                if (seqLen == 1)
                    _kernels.MatMulIq2Xs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq2XsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ2_S:
                if (seqLen == 1)
                    _kernels.MatMulIq2S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq2SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ1_S:
                if (seqLen == 1)
                    _kernels.MatMulIq1S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq1SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.F16:
                if (seqLen == 1)
                    _kernels.MatMulF16.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulF16GemmCoopmat is not null)
                    _kernels.MatMulF16GemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulF16Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.BF16:
                if (seqLen == 1)
                    _kernels.MatMulBf16.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulBf16Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            default:
                _kernels.MatMul.Record(cmdBuf, weights, input, output, outputDim, inputDim, seqLen);
                break;
        }
    }

    // ── Plumbing ─────────────────────────────────────────────────────────────

    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    private static void CopyTokenRow(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        int t, int rowElems)
    {
        long rowBytes = (long)rowElems * sizeof(float);
        var region = new VkBufferCopy
        {
            srcOffset = (ulong)((long)t * rowBytes),
            dstOffset = 0,
            size = (ulong)rowBytes,
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    private unsafe void RecordEmbeddingGather(nint cmdBuf, ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        long rowBytes = (long)hiddenSize * sizeof(float);
        var srcBuf = _weights.TokenEmbedding.Handle;
        var dstBuf = _state.HiddenState.Handle;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
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
        // Resident MoE bundles outlive single forwards in the default mode;
        // free them here before kernels and the device go away.
        for (int i = 0; i < _residentMoeBundles.Length; i++)
        {
            _residentMoeBundles[i]?.Dispose();
            _residentMoeBundles[i] = null;
        }
        _state.Dispose();
        _weights.Dispose();
        _gdnCache.Dispose();
        _kernels.Dispose();
        // Disposing the CPU model frees its NormWeight / DequantizeF32 native
        // allocations and detaches it from the GgufFile. The GgufFile itself
        // is owned by the caller (BuildFromGguf parameter) so we don't dispose
        // it here.
        _cpuModel?.Dispose();
        if (_ownsDevice) _device.Dispose();
    }
}
