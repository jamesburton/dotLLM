using DotLLM.Core.Lora;
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
/// End-to-end Vulkan forward pass for the Qwen3HybridDense architecture
/// (GGUF <c>qwen35</c> — Gated DeltaNet recurrence interleaved with full GQA
/// attention, plus a <b>dense</b> SwiGLU FFN on every layer). Mirrors the verified
/// CPU reference in <see cref="Qwen3HybridDenseTransformerModel"/> step-for-step
/// at the command-buffer level.
/// </summary>
/// <remarks>
/// <para>
/// <b>Relationship to the MoE hybrid.</b> This is the sibling of
/// <see cref="VulkanQwen3MoeHybridTransformerModel"/>; the CPU weight record says
/// the two differ only in the FFN sublayer (<see cref="Qwen3HybridDenseLayerWeights"/>:
/// "the only structural difference from the MoE hybrid is the FFN sublayer").
/// The token-mixing recording (<c>RecordGdnLayer</c> / <c>RecordFullAttnLayer</c>)
/// is therefore the same graph, and the same
/// <see cref="VulkanQwen3MoeHybridKernels"/> bundle and
/// <see cref="VulkanNemotronHKvCache"/> / <see cref="VulkanGdnStateCache"/> serve
/// both. The MoE-only kernels in that bundle are created but never dispatched here
/// — a handful of unused pipelines, which is cheaper than forking the bundle.
/// </para>
/// <para>
/// <b>Submission boundaries.</b> <b>One</b> submission per layer, unlike the MoE
/// hybrid's two. The MoE host needs a mid-layer submit because routed experts
/// require a host dequant + upload between token mixing and the FFN; a dense FFN
/// is fully device-resident, so the whole layer records into one command buffer.
/// </para>
/// <para>
/// <b>MTP ("NextN").</b> Implemented (issue #435): a <c>qwen35</c> checkpoint carrying an MTP
/// block gets <see cref="SupportsMtp"/>, <see cref="CreateMtpState(int)"/> and
/// <see cref="ForwardMtp"/>, mirroring the CPU reference's <c>ForwardMtpCore</c> step for step,
/// so <c>MtpSpeculativeDecoder</c> drives this model with no backend-specific branch of its own.
/// </para>
/// </remarks>
public sealed partial class VulkanQwen3HybridDenseTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly GgufFile? _gguf;

    // The CPU model retains the GGUF mmap that every device-resident weight was
    // uploaded from, plus the dequantised F32 norm arrays. Keeping it alive for
    // the lifetime of the Vulkan model is mandatory.
    private readonly Qwen3HybridDenseTransformerModel? _cpuModel;

    private readonly VulkanQwen3HybridDenseWeights _weights;
    private readonly VulkanQwen3HybridDenseForwardState _state;

    // PrismML Hadamard fold (prism.hadamard.*) — null outside the Bonsai 2 family; every rotation
    // site below is a no-op when it is null.
    private readonly VulkanHadamardRotation? _hadamard;

    // Compute-dispatch gather for a token-embedding table kept packed on the device; null when the
    // table was widened to F32 and the vkCmdCopyBuffer row copy applies.
    private readonly Pq2_0EmbedGatherF32Kernel? _embedGather;
    private readonly VulkanGdnStateCache _gdnCache;
    private readonly VulkanQwen3MoeHybridKernels _kernels;

    // MTP ("NextN") head — issue #435. Null for every checkpoint without a nextn.* tensor group,
    // which is the overwhelming majority; SupportsMtp and every MTP member below key off it.
    private readonly VulkanQwen3HybridDenseMtpWeights? _mtpHead;
    private MtpScratch? _mtpScratch;
    private VulkanMtpState? _lastMtpState;

    // Lazily-grown [rows, vocab] logits buffer for the all-row LM head (see MaxAllRowLogitsSeqLen).
    private VulkanDevice.Buffer? _multiRowLogits;
    private int _multiRowLogitsRows;

    // Host staging for the MTP pre-final-norm hidden capture (issue #435). Grown on demand.
    private float[] _mtpCaptureScratch = [];

    // Host copy of output_norm.weight for the post-norm MTP capture; downloaded on first use.
    private float[] _outputNormHost = [];

    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _gdnLayerOrdinal;

    private readonly int _ropeDim;
    private readonly float _ropeTheta;

    private readonly VulkanDevice.SubmitContext _submit;

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

    /// <summary>
    /// Creates a fresh per-sequence <see cref="VulkanGdnStateCache"/> sized for this
    /// model's GDN-layer count. The scheduler / multi-seq dispatcher should allocate
    /// one per active sequence and pass it via <see cref="SequenceForwardRequest.GdnState"/>;
    /// without that, multi-seq dispatch leaks recurrent state across sequences.
    /// </summary>
    public VulkanGdnStateCache CreateGdnStateCache()
        => new(_device, _gdn, _gdnCache.NumGdnLayers);

    private VulkanQwen3HybridDenseTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config,
        GgufFile? gguf,
        Qwen3HybridDenseTransformerModel? cpuModel,
        VulkanQwen3HybridDenseWeights weights,
        VulkanQwen3HybridDenseForwardState state,
        VulkanGdnStateCache gdnCache,
        VulkanQwen3MoeHybridKernels kernels,
        int[] kvSlotForLayer, int attentionLayerCount,
        int[] gdnLayerOrdinal,
        int ropeDim, float ropeTheta,
        VulkanHadamardRotation? hadamard,
        Pq2_0EmbedGatherF32Kernel? embedGather,
        VulkanQwen3HybridDenseMtpWeights? mtpHead)
    {
        _mtpHead = mtpHead;
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _gguf = gguf;
        _cpuModel = cpuModel;
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
        _hadamard = hadamard;
        _embedGather = embedGather;

        _submit = device.CreateSubmitContext();
    }

    /// <summary>
    /// Loads the Qwen3HybridDense model from a GGUF file onto a Vulkan device.
    /// Reuses the CPU loader for tensor-name mapping (so the <c>qwen35</c> naming
    /// quirks — fused Q+gate, GDN layers with no <c>attn_output.weight</c> — live
    /// in one place), then uploads every weight to the device.
    /// </summary>
    /// <param name="device">Vulkan device. Not owned; the caller disposes it.</param>
    /// <param name="gguf">Source GGUF file. Must outlive the returned model.</param>
    /// <param name="config">Model configuration extracted from <paramref name="gguf"/>.</param>
    /// <param name="spvDir">Directory containing compiled SPIR-V shaders.</param>
    public static VulkanQwen3HybridDenseTransformerModel BuildFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(spvDir);

        ValidateConfig(config, nameof(config));

        var cpuModel = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);
        var cpuLayers = ExtractCpuLayers(cpuModel);
        var outputNormWeight = ExtractOutputNormWeight(cpuModel);
        var (tokenEmbedPtr, tokenEmbedQt) = ExtractTokenEmbed(cpuModel);
        var (outputPtr, outputQt, outputM, outputK) = ExtractOutput(cpuModel);

        return Build(device, ownsDevice: false, config, gguf, cpuModel,
            cpuLayers, outputNormWeight,
            tokenEmbedPtr, tokenEmbedQt, outputPtr, outputQt, outputM, outputK, spvDir);
    }

    /// <summary>
    /// Builds a Vulkan Qwen3HybridDense model from caller-owned, pre-built
    /// <see cref="Qwen3HybridDenseLayerWeights"/> — for synthetic-fixture parity
    /// tests that bypass the GGUF loader. The caller retains ownership of every
    /// unmanaged pointer.
    /// </summary>
    internal static VulkanQwen3HybridDenseTransformerModel BuildFromPrebuiltWeights(
        VulkanDevice device,
        ModelConfig config,
        Qwen3HybridDenseLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputM, int outputK,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);
        ArgumentNullException.ThrowIfNull(spvDir);

        ValidateConfig(config, nameof(config));
        if (cpuLayers.Length != config.NumLayers)
            throw new ArgumentException(
                $"cpuLayers length {cpuLayers.Length} != config.NumLayers {config.NumLayers}.", nameof(cpuLayers));

        return Build(device, ownsDevice: false, config, gguf: null, cpuModel: null,
            cpuLayers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQt, outputWeight, outputQt, outputM, outputK, spvDir);
    }

    private static void ValidateConfig(ModelConfig config, string paramName)
    {
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"VulkanQwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                paramName);
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", paramName);
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", paramName);
    }

    private static VulkanQwen3HybridDenseTransformerModel Build(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, GgufFile? gguf, Qwen3HybridDenseTransformerModel? cpuModel,
        Qwen3HybridDenseLayerWeights[] cpuLayers, float[] outputNormWeight,
        nint tokenEmbedPtr, QuantizationType tokenEmbedQt,
        nint outputPtr, QuantizationType outputQt, int outputM, int outputK,
        string spvDir)
    {
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
                    $"Qwen3HybridDense rope_dim={ropeDim} must be even for pair-wise rotation.");
            if (ropeDim > config.HeadDim)
                throw new InvalidDataException(
                    $"Qwen3HybridDense rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        }

        var weights = VulkanQwen3HybridDenseWeights.Upload(device, config, cpuLayers, outputNormWeight,
            tokenEmbedPtr, tokenEmbedQt, outputPtr, outputQt, outputM, outputK);

        var state = new VulkanQwen3HybridDenseForwardState(device, config, gdn, initialSeqLen: 1);
        var gdnCache = new VulkanGdnStateCache(device, gdn, gdnOrdinal);
        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir, config.HeadDim);

        var hadamard = config.HadamardFold is { } fold
            ? VulkanHadamardRotation.Create(device, spvDir, fold, gdn)
            : null;

        var embedGather = weights.TokenEmbeddingQuantType == QuantizationType.PQ2_0
            ? Pq2_0EmbedGatherF32Kernel.Create(device, spvDir)
            : null;

        // MTP head (issue #435). Only the GGUF path can carry one: the prebuilt-weights entry
        // point takes trunk layers only, so a fixture built that way simply gets null here.
        var mtpHead = cpuModel is not null && ExtractMtpHead(cpuModel) is { } cpuMtp
            ? VulkanQwen3HybridDenseMtpWeights.Upload(device, config, cpuMtp)
            : null;

        if (mtpHead is not null && config.HadamardFold is { } mtpFold)
        {
            // The MTP block's OWN projections are assumed unfolded — that is what Bonsai 2's MTP
            // checkpoint ships (blk.{NumLayers}.* is absent from prism.hadamard.weight_names, and
            // its matmuls are Q8_0 rather than PQ2_0) and it is what the CPU reference computes.
            // Refuse loudly rather than silently mis-compute if a future checkpoint folds them.
            string mtpPrefix = $"blk.{config.NumLayers}";
            foreach (string suffix in new[] { "attn_qkv.weight", "attn_gate.weight", "attn_q.weight",
                                              "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                                              "ffn_down.weight", "nextn.eh_proj.weight" })
            {
                if (mtpFold.IsFolded($"{mtpPrefix}.{suffix}"))
                    throw new NotSupportedException(
                        $"Checkpoint folds the MTP block's own weight '{mtpPrefix}.{suffix}' " +
                        "(prism.hadamard.weight_names). Neither the CPU reference nor this Vulkan host " +
                        "rotates the MTP block's own projections — see issue #435.");
            }
        }

        return new VulkanQwen3HybridDenseTransformerModel(
            device, ownsDevice,
            config, gguf, cpuModel, weights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, gdnLayerOrdinal,
            ropeDim, ropeTheta, hadamard, embedGather, mtpHead);
    }

    // ── CPU-model accessors (we share the CPU loader; reach into its layers) ─
    // Same reflection approach as VulkanQwen3MoeHybridTransformerModel /
    // VulkanNemotronHTransformerModel: the alternative is widening DotLLM.Models'
    // public API for a single internal consumer.

    private static T GetPrivateField<T>(Qwen3HybridDenseTransformerModel m, string name)
    {
        var fi = typeof(Qwen3HybridDenseTransformerModel)
            .GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException($"Qwen3HybridDenseTransformerModel.{name} field missing.");
        return (T)fi.GetValue(m)!;
    }

    private static Qwen3HybridDenseLayerWeights[] ExtractCpuLayers(Qwen3HybridDenseTransformerModel m)
        => GetPrivateField<Qwen3HybridDenseLayerWeights[]>(m, "_layers");

    private static float[] ExtractOutputNormWeight(Qwen3HybridDenseTransformerModel m)
        => GetPrivateField<float[]>(m, "_outputNormWeight");

    private static (nint ptr, QuantizationType qt) ExtractTokenEmbed(Qwen3HybridDenseTransformerModel m)
        => (GetPrivateField<nint>(m, "_tokenEmbedWeight"),
            GetPrivateField<QuantizationType>(m, "_tokenEmbedQuantType"));

    private static MtpHeadWeights? ExtractMtpHead(Qwen3HybridDenseTransformerModel m)
        => GetPrivateFieldOrNull<MtpHeadWeights>(m, "_mtpHead");

    private static T? GetPrivateFieldOrNull<T>(Qwen3HybridDenseTransformerModel m, string name)
        where T : class
    {
        var fi = typeof(Qwen3HybridDenseTransformerModel)
            .GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException($"Qwen3HybridDenseTransformerModel.{name} field missing.");
        return fi.GetValue(m) as T;
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) ExtractOutput(
        Qwen3HybridDenseTransformerModel m)
        => (GetPrivateField<nint>(m, "_outputWeight"),
            GetPrivateField<QuantizationType>(m, "_outputQuantType"),
            GetPrivateField<int>(m, "_outputOutputDim"),
            GetPrivateField<int>(m, "_outputInputDim"));

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null, gdnState: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => Forward(tokenIds, positions, deviceId, kvCache, gdnState: null);

    /// <summary>
    /// Runs a forward pass with optional KV-cache (for the GQA layers) and optional
    /// per-sequence GDN recurrent state (for the GDN layers). When
    /// <paramref name="gdnState"/> is <see langword="null"/>, falls back to the
    /// model-owned default cache — safe only for single-sequence dispatch.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, IGdnState? gdnState)
        => Forward(tokenIds, positions, deviceId, kvCache, gdnState, mtpState: null);

    /// <inheritdoc/>
    /// <remarks>
    /// MTP (issues #253 / #435): when <paramref name="mtpState"/> is a <see cref="VulkanMtpState"/>
    /// on a model with <see cref="SupportsMtp"/> true, this call additionally downloads the trunk's
    /// pre-final-norm hidden state — one row per input position — into that state, so a subsequent
    /// <see cref="ForwardMtp"/> can seed the MTP head from a hidden state the trunk actually
    /// produced. The capture is a pure side effect: the returned logits are identical either way.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
    {
        if (adapter is not null)
            throw new NotSupportedException(
                "VulkanQwen3HybridDenseTransformerModel does not support LoRA adapters.");
        return Forward(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState: mtpState);
    }

    /// <summary>
    /// Full forward with optional KV-cache, per-sequence GDN state and MTP hidden capture.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, IGdnState? gdnState, IMtpState? mtpState)
    {
        if (mtpState is not null && mtpState is not VulkanMtpState)
            throw new ArgumentException(
                $"VulkanQwen3HybridDenseTransformerModel requires a VulkanMtpState; got {mtpState.GetType().Name}.",
                nameof(mtpState));

        VulkanGdnStateCache gdnCache;
        if (gdnState is null)
        {
            gdnCache = _gdnCache;
        }
        else if (gdnState is VulkanGdnStateCache vk)
        {
            if (vk.NumGdnLayers != _gdnCache.NumGdnLayers)
                throw new ArgumentException(
                    $"GdnState NumGdnLayers ({vk.NumGdnLayers}) does not match model GDN-layer count ({_gdnCache.NumGdnLayers}).",
                    nameof(gdnState));
            gdnCache = vk;
        }
        else
        {
            throw new ArgumentException(
                $"VulkanQwen3HybridDenseTransformerModel requires a VulkanGdnStateCache; got {gdnState.GetType().Name}.",
                nameof(gdnState));
        }

        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");
        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int intermediateSize = Config.IntermediateSize;
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

        // Rows the LM head will cover — see MaxAllRowLogitsSeqLen. Resolved HERE, before any
        // recording, because EnsureMultiRowLogits may reallocate and therefore invalidate every
        // kernel's descriptor cache. That path calls vkResetDescriptorPool, which frees sets an
        // already-recorded dispatch still references (DescriptorSetCache's own remarks document
        // exactly this hazard), so it must never run against an open command buffer.
        int headRows = seqLen <= MaxAllRowLogitsSeqLen ? seqLen : 1;

        bool resized = _state.EnsureCapacity(seqLen);
        if (resized)
        {
            _kernels.InvalidateAll();
            // The FWHT kernel keeps its own handle-keyed descriptor cache, and a freed buffer handle
            // can be recycled into the new scratch allocation — which would bind the stale set.
            _hadamard?.InvalidateDescriptorCache();
            _embedGather?.InvalidateDescriptorCache();
        }

        var logitsBuf = headRows == 1 ? _state.Logits : EnsureMultiRowLogits(headRows, vocabSize);

        UploadPositions(positions);

        var kinds = _layout.LayerKind;

        ProfBeginForward(seqLen);

        // ── 1. Token embedding (single submission) ────────────────────────────
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        ProfBeginSubmit(cmdBuf);
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfBeforeSubmit(cmdBuf);
        _submit.SubmitAndWait();
        ProfAfterSubmit();

        // ── 2. Per-layer body — ONE submission per layer. Unlike the MoE hybrid
        //      there is no host round-trip between token mixing and the FFN, so
        //      both sublayers record into the same command buffer. ─────────────
        long hiddenRowBytes = (long)hiddenSize * sizeof(float);
        for (int layer = 0; layer < kinds.Length; layer++)
        {
            ref readonly var layerBuf = ref _weights.Layers[layer];

            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmdBuf);
            ProfBeginSubmit(cmdBuf);

            // ── 2a. Token mixing ────────────────────────────────────────────
            RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

            if (kinds[layer] == HybridLayerKind.GatedDeltaNet)
            {
                RecordGdnLayer(cmdBuf, layer, layerBuf.Gdn!.Value, seqLen, eps, gdnCache);
            }
            else
            {
                RecordFullAttnLayer(cmdBuf, layer, layerBuf.Attention!.Value, seqLen, positions,
                    numHeads, numKvHeads, headDim, kvCache);
            }
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // First residual add: HiddenState = Residual + NormOutput.
            _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
                seqLen * hiddenSize);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

            // ── 2b. Dense SwiGLU FFN ────────────────────────────────────────
            // Post-norm → gate/up → SwiGLU → down → residual add. Mirrors
            // Qwen3HybridDenseTransformerModel.ForwardDenseFfnBody.
            //
            // Both HiddenState and the FFN's residual snapshot are fanned out
            // from AddScratch rather than chaining HiddenState → Residual.
            // Chaining would be a transfer-write followed by a transfer-read of
            // the same buffer inside one command buffer, and the barrier
            // vocabulary here has no TRANSFER→TRANSFER edge (only
            // TransferToCompute, which does not order a later transfer read).
            // The MoE hybrid sibling never hits this because its two sublayers
            // are separated by a submit; a dense FFN needs no host round-trip,
            // so this host records the whole layer into one command buffer.
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

            RecordDenseFfn(cmdBuf, layerBuf.Ffn, seqLen, intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
                seqLen * hiddenSize);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);
            ProfBeforeSubmit(cmdBuf);
            _submit.SubmitAndWait();
            ProfAfterSubmit();
        }

        // ── 2c. MTP hidden capture (issues #435, #469) ───────────────────────
        // The head consumes llama.cpp's `h_nextn`: the hidden state AFTER output_norm, for every
        // position. The device final norm below only covers the rows that get logits (the last
        // row of a long prefill), so the rows are normalised on the host with the same RMSNorm
        // the CPU reference uses. A pure side effect on the MTP state.
        VulkanMtpState? mtpCapture = _mtpHead is not null ? mtpState as VulkanMtpState : null;
        if (mtpCapture is not null)
        {
            int captureElems = checked(seqLen * hiddenSize);
            if (_mtpCaptureScratch.Length < captureElems)
                _mtpCaptureScratch = new float[captureElems];
            var rows = _mtpCaptureScratch.AsSpan(0, captureElems);
            _device.Download(_state.HiddenState, rows);
            if (_outputNormHost.Length == 0)
            {
                _outputNormHost = new float[hiddenSize];
                _device.Download(_weights.OutputNormWeight, _outputNormHost);
            }
            for (int r = 0; r < seqLen; r++)
            {
                var row = rows.Slice(r * hiddenSize, hiddenSize);
                DotLLM.Cpu.Kernels.RmsNorm.Execute(row, _outputNormHost, eps, row);
            }
            mtpCapture.SetCapturedRows(rows, seqLen);
        }

        // ── 3. Final norm + LM head (single submission) ───────────────────────
        // Rows: the IModel contract is [seq, vocab], but running a 248k-row LM head over a
        // 2048-token prefill would both dominate prefill time and produce a 2 GB logits tensor
        // no caller reads. So the head runs over every row for a SHORT batch — which is exactly
        // the speculative verify-batch regime MtpSpeculativeDecoder needs, and which it indexes
        // row-by-row — and stays last-row-only above that. See MaxAllRowLogitsSeqLen.
        long headSrcOffset = (long)(seqLen - headRows) * hiddenRowBytes;

        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        ProfBeginSubmit(cmdBuf);

        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)headSrcOffset, dstOffset: 0, size: (ulong)((long)headRows * hiddenRowBytes));
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        _kernels.RmsNorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: headRows, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        var headIn = _state.NormOutput;
        if (_hadamard is { } headRot)
        {
            headIn = _state.HadamardScratch!;
            headRot.RecordForward(cmdBuf, _state.NormOutput, headIn, headRows, _weights.OutputInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            headIn, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: headRows);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.LmHead);
        ProfBeforeSubmit(cmdBuf);
        _submit.SubmitAndWait();
        ProfAfterSubmit();

        // ── 4. Download logits ───────────────────────────────────────────────
        var shape = new TensorShape(headRows, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, headRows * vocabSize);
            _device.Download(logitsBuf, dest);
        }

        ProfEndForward();

        if (mtpCapture is not null)
            AbsorbMtp(_mtpHead!, mtpCapture, tokenIds, positions);

        return result;
    }

    /// <summary>
    /// Runs the MTP head over every token of a trunk batch, without logits, so its KV-cache holds
    /// the whole sequence — llama.cpp's <c>draft-mtp</c> <c>process()</c> (issue #469). Token
    /// <c>i</c> pairs with the trunk hidden state of the previous position: the carried row for
    /// <c>i == 0</c>, captured row <c>i - 1</c> otherwise. Mirrors the CPU reference.
    /// </summary>
    private void AbsorbMtp(VulkanQwen3HybridDenseMtpWeights mtpHead, VulkanMtpState state,
                           ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions)
    {
        BindMtpState(state);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (i == 0) state.SetPendingFromCarry();
            else state.SetPendingFromCapturedRow(i - 1);
            ForwardMtpCore(mtpHead, state, tokenIds[i], positions[i], computeLogits: false);
        }
        state.SeedFromCapturedRow(tokenIds.Length - 1);
    }

    // A previously-disposed state's buffer handles can be recycled into this one's; invalidate the
    // kernels' descriptor caches once per new state (belt and braces alongside #467's eviction).
    private void BindMtpState(VulkanMtpState mtp)
    {
        if (ReferenceEquals(_lastMtpState, mtp))
            return;
        _kernels.InvalidateAll();
        _hadamard?.InvalidateDescriptorCache();
        _embedGather?.InvalidateDescriptorCache();
        _lastMtpState = mtp;
    }

    /// <summary>
    /// Batch length up to which the LM head runs over <em>every</em> row, so <c>Forward</c> honours
    /// the <see cref="IModel"/> <c>[seq, vocab]</c> contract. Sized to
    /// <see cref="MtpDefaultMaxDraftSteps"/>: an MTP verify batch is K+1 rows (the last token plus
    /// K drafts) and needs a logit row per position, while a real prefill is orders of magnitude longer and only
    /// ever has its last row read.
    /// </summary>
    public const int MaxAllRowLogitsSeqLen = MtpDefaultMaxDraftSteps;

    /// <inheritdoc/>
    /// <remarks>
    /// The honest declaration of the deviation documented on <see cref="MaxAllRowLogitsSeqLen"/>.
    /// Without it, a caller that decides "does this backend return all rows?" from a short probe
    /// forward gets <see langword="true"/> here and then indexes rows that do not exist at real
    /// context lengths.
    /// </remarks>
    public int MaxAllRowLogitsLength => MaxAllRowLogitsSeqLen;

    /// <summary>Lazily (re)allocates the multi-row logits buffer for <paramref name="rows"/> x <paramref name="vocab"/>.</summary>
    private VulkanDevice.Buffer EnsureMultiRowLogits(int rows, int vocab)
    {
        if (_multiRowLogits is not null && _multiRowLogitsRows >= rows)
            return _multiRowLogits;

        _multiRowLogits?.Dispose();
        _multiRowLogits = _device.AllocateHostReadback((long)rows * vocab * sizeof(float));
        _multiRowLogitsRows = rows;
        // A freed buffer handle can be recycled into this allocation, and every kernel here keys
        // its descriptor sets on the handle — invalidate all three caches so no stale set survives
        // into a dispatch that now means a different buffer. The FWHT and embed-gather kernels are
        // included even though neither binds this buffer: the handle we just freed could equally be
        // recycled into one of THEIR buffers on a later allocation.
        _kernels.InvalidateAll();
        _hadamard?.InvalidateDescriptorCache();
        _embedGather?.InvalidateDescriptorCache();
        return _multiRowLogits;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Re-zeroes the model-owned Gated-DeltaNet state cache used by every forward that
    /// does not carry a caller-supplied per-sequence state container. Callers treating
    /// each forward as an independent sequence (perplexity windows) must call this
    /// between sequences — see issue #261.
    /// </remarks>
    public void ResetSequenceState() => _gdnCache.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => CreateGdnStateCache();

    /// <summary>
    /// Per-sequence <c>ForwardBatch</c>. Mirrors
    /// <see cref="VulkanQwen3MoeHybridTransformerModel.ForwardBatch"/>: the GDN scan
    /// threads one sequence's recurrent state through tokens in order and cannot
    /// share a dispatch across sequences, so this loops the per-seq
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, IGdnState?)"/>
    /// using each request's own <see cref="SequenceForwardRequest.GdnState"/>.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "VulkanQwen3HybridDenseTransformerModel.ForwardBatch does not support LoRA " +
                    "adapters (no Qwen3HybridDense LoRA path today). Re-issue the request without " +
                    "an adapter.");
        }

        // Multi-seq dispatch without per-seq GDN state would silently corrupt the
        // model-owned recurrent state across sequences. Fail loudly instead.
        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].GdnState is null)
                    throw new ArgumentException(
                        $"Multi-seq ForwardBatch requires each SequenceForwardRequest to carry " +
                        $"its own GdnState (request index {i} has GdnState=null). The " +
                        "model-owned VulkanGdnStateCache is shared across all calls into this " +
                        "model instance, so a null slot in a multi-seq batch would leak GDN " +
                        "recurrent state across sequences. Construct one VulkanGdnStateCache " +
                        "per active sequence and assign it via SequenceForwardRequest.GdnState.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.GdnState);
        }
        return results;
    }

    // ── Dense SwiGLU FFN ─────────────────────────────────────────────────────

    /// <summary>
    /// Records the dense FFN for one layer, reading the post-FFN-norm activations
    /// from <c>NormOutput</c> and writing the down-projection back into it — the
    /// device mirror of <c>Qwen3HybridDenseTransformerModel.ForwardDenseFfnBody</c>.
    /// </summary>
    private void RecordDenseFfn(
        nint cmdBuf, in VulkanQwen3HybridDenseWeights.DenseFfnLayerBuffers ffn,
        int seqLen, int intermediateSize)
    {
        // ffn_gate and ffn_up are both folded and share this input — one rotation serves both.
        var ffnIn = _state.NormOutput;
        if (_hadamard is { } ffnRot)
        {
            ffnIn = _state.HadamardScratch!;
            ffnRot.RecordForward(cmdBuf, _state.NormOutput, ffnIn, seqLen, ffn.GateInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, ffn.GateWeight, ffn.GateDeviceQuantType,
            ffnIn, _state.FfnGate,
            outputDim: ffn.GateOutputDim, inputDim: ffn.GateInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, ffn.UpWeight, ffn.UpDeviceQuantType,
            ffnIn, _state.FfnUp,
            outputDim: ffn.UpOutputDim, inputDim: ffn.UpInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjFfn);

        _kernels.SwiGlu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.FfnSilu,
            n: checked(seqLen * intermediateSize));
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.FfnAct);

        // Reusing the scratch is safe: the gate/up rotation above has been consumed by both GEMMs.
        var downIn = _state.FfnSilu;
        if (_hadamard is { } downRot)
        {
            downIn = _state.HadamardScratch!;
            downRot.RecordForward(cmdBuf, _state.FfnSilu, downIn, seqLen, ffn.DownInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, ffn.DownWeight, ffn.DownDeviceQuantType,
            downIn, _state.NormOutput,
            outputDim: ffn.DownOutputDim, inputDim: ffn.DownInputDim, seqLen: seqLen);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjFfn);
    }

    // ── Token-mixing path: Gated DeltaNet ────────────────────────────────────

    /// <summary>
    /// Records the GDN token-mixing forward for one layer. Identical graph to
    /// <see cref="VulkanQwen3MoeHybridTransformerModel"/>'s — the two architectures
    /// share the token-mixing sublayer verbatim. Mirrors the CPU
    /// <c>Qwen3HybridDenseTransformerModel.ForwardGdnBody</c>.
    /// </summary>
    private void RecordGdnLayer(
        nint cmdBuf, int absoluteLayerIdx, VulkanQwen3MoeHybridWeights.GdnLayerBuffers gdnW,
        int seqLen, float eps, VulkanGdnStateCache gdnCache)
    {
        int nVHead = _gdn.NVHead;
        int nKHead = _gdn.NKHead;
        int dState = _gdn.DState;
        int dConv = _gdn.DConv;
        int convDim = (2 * nKHead + nVHead) * dState;
        int vDim = nVHead * dState;
        int kDim = nKHead * dState;
        int gdnOrdinal = _gdnLayerOrdinal[absoluteLayerIdx];

        var convStateBuf = gdnCache.GetConvStateBuffer(gdnOrdinal);
        var gdnStateBuf = gdnCache.GetGdnStateBuffer(gdnOrdinal);

        // ── 1. Projections ───────────────────────────────────────────────────
        // Only attn_qkv and attn_gate are folded; ssm_alpha and ssm_beta below deliberately keep
        // reading the UNROTATED NormOutput, which is why the rotation goes to a separate buffer.
        var gdnProjIn = _state.NormOutput;
        if (_hadamard is { } gdnRot)
        {
            gdnProjIn = _state.HadamardScratch!;
            gdnRot.RecordForward(cmdBuf, _state.NormOutput, gdnProjIn, seqLen, gdnW.QkvInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, gdnW.QkvWeight, gdnW.QkvDeviceQuantType,
            gdnProjIn, _state.GdnQkvBuf,
            outputDim: gdnW.QkvOutputDim, inputDim: gdnW.QkvInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.GateWeight, gdnW.GateDeviceQuantType,
            gdnProjIn, _state.GdnZBuf,
            outputDim: gdnW.GateOutputDim, inputDim: gdnW.GateInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.AlphaWeight, gdnW.AlphaDeviceQuantType,
            _state.NormOutput, _state.GdnAlphaBuf,
            outputDim: gdnW.AlphaOutputDim, inputDim: gdnW.AlphaInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.BetaWeight, gdnW.BetaDeviceQuantType,
            _state.NormOutput, _state.GdnBetaBuf,
            outputDim: gdnW.BetaOutputDim, inputDim: gdnW.BetaInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjGdn);

        // ── 2. Fused on-device decay g and sigmoid(β) ─────────────────────────
        _kernels.GdnDecay.Record(cmdBuf, _state.GdnAlphaBuf, gdnW.DtBiasDevice, gdnW.ADevice,
            seqLen: seqLen, nVHead: nVHead);
        _kernels.SigmoidInplace.Record(cmdBuf, _state.GdnBetaBuf, n: seqLen * nVHead);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.GdnPre);

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
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.CopyFanout);
        ProfNote("copy_gdn_conv_input", m: convDim, k: dConv, n: seqLen);

        _kernels.Conv1dCausal.Record(cmdBuf, _state.GdnConvInput, gdnW.Conv1dWeight, gdnW.Conv1dBias,
            _state.GdnQkvBuf, dConv: dConv, channels: convDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _kernels.SiluInplace.Record(cmdBuf, _state.GdnQkvBuf, n: seqLen * convDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.GdnPre);

        // Save the trailing (dConv-1) rows of the PRE-SiLU ConvInput back to
        // convState — same offset pattern as the MoE hybrid and VulkanNemotronH.
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
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.CopyFanout);
        ProfNote("copy_gdn_qkv_split", m: kDim, k: vDim, n: seqLen);

        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnQBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnKBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.GdnPre);

        // ── 5. GDN scan — single multi-token dispatch ────────────────────────
        _kernels.GdnScanMultiToken.Record(cmdBuf,
            state: gdnStateBuf,
            q: _state.GdnQBuf, k: _state.GdnKBuf, v: _state.GdnVBuf,
            g: _state.GdnAlphaBuf, beta: _state.GdnBetaBuf,
            output: _state.GdnOut,
            seqLen: seqLen, nVHead: nVHead, nKHead: nKHead, dState: dState);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.GdnScanCore);   // #445 sub-bucket
        ProfNote("gdn_scan_multitoken", m: nVHead, k: dState, n: seqLen);

        // ── 6. Per-head RMSNorm × silu(z) gate (fused) ───────────────────────
        _kernels.GdnPostScanGate.Record(cmdBuf,
            gdnOut: _state.GdnOut, z: _state.GdnZBuf, ssmNormWeight: gdnW.SsmNormWeight,
            seqLen: seqLen, nVHead: nVHead, dState: dState, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.GdnPostGate);   // #445 sub-bucket

        // ── 7. ssm_out projection back into NormOutput ───────────────────────
        // The one site taking the value-head permutation: the fold was computed in grouped
        // [dState, rep, nKHead] order while the scan emits tiled order.
        var ssmOutIn = _state.GdnOut;
        if (_hadamard is { } outRot)
        {
            ssmOutIn = _state.HadamardScratch!;
            outRot.RecordForward(cmdBuf, _state.GdnOut, ssmOutIn, seqLen, gdnW.OutInputDim,
                permuteGdnValueHeads: true);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, gdnW.OutWeight, gdnW.OutDeviceQuantType,
            ssmOutIn, _state.NormOutput,
            outputDim: gdnW.OutOutputDim, inputDim: gdnW.OutInputDim, seqLen: seqLen);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjGdn);
    }

    // ── Token-mixing path: full GQA attention ────────────────────────────────

    /// <summary>
    /// Records the full-attention forward for one layer. Q+Gate are fused in
    /// <c>attn_q</c> at output width <c>2 * nQ * headDim</c>; we de-interleave per
    /// head before QK-norm, RoPE and attention.
    /// </summary>
    private void RecordFullAttnLayer(
        nint cmdBuf, int absoluteLayerIdx, VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers attnW,
        int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int qElems = numHeads * headDim;
        int qgElems = 2 * qElems;

        // 1. Fused Q+Gate projection.
        // attn_q / attn_k / attn_v are all folded and share this input, so one rotation serves the
        // three. HadamardScratch is untouched between here and the K/V projections below.
        var attnProjIn = _state.NormOutput;
        if (_hadamard is { } attnRot)
        {
            attnProjIn = _state.HadamardScratch!;
            attnRot.RecordForward(cmdBuf, _state.NormOutput, attnProjIn, seqLen, attnW.QInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, attnW.QWeight, attnW.QDeviceQuantType,
            attnProjIn, _state.QGateScratch,
            outputDim: attnW.QOutputDim, inputDim: attnW.QInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);

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
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.CopyFanout);
        ProfNote("copy_qgate_deinterleave", m: numHeads, k: headDim, n: seqLen);

        // 3. K and V projections.
        RecordMatmul(cmdBuf, attnW.KWeight, attnW.KDeviceQuantType,
            attnProjIn, _state.K,
            outputDim: attnW.KOutputDim, inputDim: attnW.KInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, attnW.VWeight, attnW.VDeviceQuantType,
            attnProjIn, _state.V,
            outputDim: attnW.VOutputDim, inputDim: attnW.VInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);

        // 4. QK-norm — per-head RMSNorm with attn_q_norm / attn_k_norm weights.
        _kernels.RmsNorm.Record(cmdBuf, _state.Q, attnW.QNormWeight, _state.Q,
            rowCount: seqLen * numHeads, n: headDim, eps: Config.NormEpsilon);
        _kernels.RmsNorm.Record(cmdBuf, _state.K, attnW.KNormWeight, _state.K,
            rowCount: seqLen * numKvHeads, n: headDim, eps: Config.NormEpsilon);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        // 5. RoPE — NeoX pair pattern over the first ropeDim of each head, mirroring
        //    the CPU reference's choice so device output matches CPU output.
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
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnRope);      // #445 sub-bucket
            vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, absoluteLayerIdx);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnKvUpdate);  // #445 sub-bucket
            kSrc = vkCache.GetKeysBuffer(absoluteLayerIdx);
            vSrc = vkCache.GetValuesBuffer(absoluteLayerIdx);
            seqKv = vkCache.CurrentLength;
            positionOffset = positions[0];
        }
        else
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnRope);      // #445 sub-bucket
            kSrc = _state.K;
            vSrc = _state.V;
            seqKv = seqLen;
            positionOffset = 0;
        }

        if (_kernels.SplitKvAttention is not null && seqLen == 1
            && headDim <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads))
        {
            // Decode: split the KV range across many workgroups (Flash-Decoding).
            // Engages from seqKv >= 17 with the shipping heuristic (issue #331);
            // only seqKv <= 16 falls through to the per-token kernel.
            ProfNote("attn_splitkv", m: numHeads, k: headDim, n: seqKv);
            _kernels.SplitKvAttention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else if (_kernels.FlashAttention is not null && seqLen > 1 && headDim <= _kernels.FlashAttention.SupportedMaxHeadDim)
        {
            ProfNote("attn_flash", m: numHeads, k: headDim, n: seqKv);
            _kernels.FlashAttention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else
        {
            ProfNote("attn_naive", m: numHeads, k: headDim, n: seqKv);
            _kernels.Attention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnCore);      // #445 sub-bucket

        // 7. Apply sigmoid(gate) element-wise to attention output.
        _kernels.SigmoidGateMul.Record(cmdBuf, _state.AttnOutput, _state.GateScratch,
            nTotal: seqLen * qElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnGate);      // #445 sub-bucket

        // 8. Output projection.
        // Shares the 6144 sign vector with ssm_out but takes NO value-head permutation.
        var oProjIn = _state.AttnOutput;
        if (_hadamard is { } oRot)
        {
            oProjIn = _state.HadamardScratch!;
            oRot.RecordForward(cmdBuf, _state.AttnOutput, oProjIn, seqLen, attnW.OInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, attnW.OWeight, attnW.ODeviceQuantType,
            oProjIn, _state.NormOutput,
            outputDim: attnW.OOutputDim, inputDim: attnW.OInputDim, seqLen: seqLen);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);
    }

    // ── Recurrent-state checkpoint (issue #287 / #435) ───────────────────────

    /// <inheritdoc/>
    /// <remarks>
    /// Needed by speculative decoding: the batched verify forward advances the Gated-DeltaNet
    /// recurrence for every drafted token before accept/reject is known, and a pure sequential
    /// recurrence has no position addressing to undo a rejected token's contribution the way
    /// <see cref="IKvCache"/> rollback does. Without this pair, MTP self-speculation on Vulkan
    /// would silently corrupt the trunk's GDN state on every partial rejection — the acceptance
    /// rate would look healthy while the output drifted away from greedy decoding.
    /// </remarks>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    /// <remarks>
    /// Snapshots are pooled (one spare): a speculative decoder takes one per round, and allocating
    /// a fresh set of per-layer device buffers each time (96 on Bonsai 2) cost more than the copy.
    /// Disposing the returned checkpoint hands its buffers back for the next round.
    /// </remarks>
    public object? CheckpointRecurrentState()
    {
        VulkanGdnStateCache snapshot = Interlocked.Exchange(ref _spareGdnCheckpoint, null)
            ?? _gdnCache.CloneGeometry();
        _gdnCache.CopyTo(snapshot);
        return new PooledGdnCheckpoint(this, snapshot);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        switch (checkpoint)
        {
            case null:
                return;
            case PooledGdnCheckpoint pooled:
                pooled.Snapshot.CopyTo(_gdnCache);
                return;
            case VulkanGdnStateCache snapshot:
                snapshot.CopyTo(_gdnCache);
                return;
            default:
                throw new ArgumentException(
                    $"{GetType().Name}.RestoreRecurrentState expects a checkpoint from CheckpointRecurrentState; " +
                    $"got {checkpoint.GetType().Name}.",
                    nameof(checkpoint));
        }
    }

    private VulkanGdnStateCache? _spareGdnCheckpoint;
    private bool _disposed;

    /// <summary>A pooled GDN snapshot; disposing returns its buffers to the owning model.</summary>
    private sealed class PooledGdnCheckpoint(VulkanQwen3HybridDenseTransformerModel owner, VulkanGdnStateCache snapshot)
        : IDisposable
    {
        private VulkanGdnStateCache? _snapshot = snapshot;

        public VulkanGdnStateCache Snapshot
            => _snapshot ?? throw new ObjectDisposedException(nameof(PooledGdnCheckpoint));

        public void Dispose()
        {
            var s = Interlocked.Exchange(ref _snapshot, null);
            if (s is null) return;
            if (owner._disposed || Interlocked.CompareExchange(ref owner._spareGdnCheckpoint, s, null) is not null)
                s.Dispose();
        }
    }

    // ── MTP ("NextN") self-speculative decoding — issue #435 ─────────────────

    /// <inheritdoc/>
    public bool SupportsMtp => _mtpHead is not null;

    /// <summary>
    /// Default MTP KV-cache depth, matching the CPU and CUDA hosts so a state created here holds
    /// the same number of autoregressive draft steps.
    /// </summary>
    public const int MtpDefaultMaxDraftSteps = 16;

    /// <inheritdoc/>
    /// <remarks>
    /// Sized for the MTP head's own attention — the MTP block is a normal full-attention layer —
    /// with a device-resident, position-indexed KV-cache of <see cref="MtpDefaultMaxSequenceLength"/>
    /// positions (issue #469).
    /// </remarks>
    public IMtpState? CreateMtpState() => CreateMtpState(MtpDefaultMaxSequenceLength);

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
    {
        if (_mtpHead is null)
            return null;
        return new VulkanMtpState(_device,
            hiddenSize: Config.HiddenSize,
            numKvHeads: _mtpHead.Attention.NumKvHeads,
            headDim: Config.HeadDim,
            maxSteps: maxSequenceLength);
    }

    /// <summary>
    /// Default MTP KV-cache depth, in sequence positions, for <see cref="CreateMtpState()"/> —
    /// the head's cache is indexed by position and absorbs the whole sequence (issue #469).
    /// </summary>
    public const int MtpDefaultMaxSequenceLength = 4096;

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (_mtpHead is not { } mtpHead)
            throw new NotSupportedException(
                $"{nameof(VulkanQwen3HybridDenseTransformerModel)} has no MTP head loaded (SupportsMtp=false).");
        if (state is not VulkanMtpState mtp)
            throw new ArgumentException(
                $"VulkanQwen3HybridDenseTransformerModel requires a VulkanMtpState; got {state.GetType().Name}.",
                nameof(state));
        if ((uint)tokenId >= (uint)Config.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"Token id {tokenId} is out of range.");
        if ((uint)position >= (uint)Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"Position {position} exceeds max sequence length {Config.MaxSequenceLength}.");

        BindMtpState(mtp);
        return ForwardMtpCore(mtpHead, mtp, tokenId, position, computeLogits: true)!;
    }

    /// <summary>
    /// Lazily-allocated device scratch for <see cref="ForwardMtpCore"/> — one row each, since the
    /// MTP head always runs a single token per call. Everything wider (Q/K/V, the FFN triple) is
    /// borrowed from the trunk's own forward scratch, which is sized for at least one row and is
    /// never live across an <c>ForwardMtp</c> call (the trunk re-gathers its embeddings at the
    /// start of every <c>Forward</c>).
    /// </summary>
    private sealed class MtpScratch : IDisposable
    {
        public required VulkanDevice.Buffer ENorm { get; init; }       // [hidden]
        public required VulkanDevice.Buffer HNorm { get; init; }       // [hidden]
        public required VulkanDevice.Buffer Concat { get; init; }      // [2 * hidden]
        public required VulkanDevice.Buffer Cur { get; init; }         // [hidden]
        public required VulkanDevice.Buffer Residual { get; init; }    // [hidden]
        public required VulkanDevice.Buffer Normed { get; init; }      // [hidden]
        public required VulkanDevice.Buffer AddOut { get; init; }      // [hidden]
        public required VulkanDevice.Buffer NormedHead { get; init; }  // [hidden]

        public static MtpScratch Allocate(VulkanDevice device, int hiddenSize)
        {
            long h = (long)hiddenSize * sizeof(float);
            return new MtpScratch
            {
                ENorm = device.AllocateDeviceLocal(h),
                HNorm = device.AllocateDeviceLocal(h),
                Concat = device.AllocateDeviceLocal(2 * h),
                Cur = device.AllocateDeviceLocal(h),
                Residual = device.AllocateDeviceLocal(h),
                Normed = device.AllocateDeviceLocal(h),
                AddOut = device.AllocateDeviceLocal(h),
                NormedHead = device.AllocateDeviceLocal(h),
            };
        }

        public void Dispose()
        {
            ENorm.Dispose(); HNorm.Dispose(); Concat.Dispose(); Cur.Dispose();
            Residual.Dispose(); Normed.Dispose(); AddOut.Dispose(); NormedHead.Dispose();
        }
    }

    /// <summary>
    /// Records one MTP head autoregressive draft step. Operation order mirrors the CPU reference
    /// (<c>Qwen3HybridDenseTransformerModel.ForwardMtpCore</c>) exactly:
    /// <list type="number">
    ///   <item><c>h_norm = RMSNorm(pendingHidden, nextn.hnorm)</c>, <c>e_norm = RMSNorm(embed(tokenId), nextn.enorm)</c>.</item>
    ///   <item><c>cur = eh_proj @ concat(e_norm, h_norm)</c> — this is the attention sub-block's residual.</item>
    ///   <item>Gated full attention over the MTP head's OWN tiny KV-cache (seqQ=1, seqKv=step+1), residual-added.</item>
    ///   <item>Dense SwiGLU FFN, residual-added — the result seeds this state's NEXT call.</item>
    ///   <item><c>shared_head_norm</c> (or the trunk's <c>output_norm</c>) then <c>shared_head_head</c>
    ///         (or the trunk's own LM head) to logits.</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// <b>Hadamard fold.</b> The MTP block's own projections are never rotated (validated at load —
    /// they are absent from <c>prism.hadamard.weight_names</c>). The two FALLBACK tensors are a
    /// different matter: the trunk's <c>token_embd.weight</c> is Hadamard-latent and its
    /// <c>output.weight</c> is folded, so when the checkpoint ships no head-local
    /// <c>nextn.embed_tokens</c> / <c>nextn.shared_head_head</c> — which is what Bonsai 2's MTP
    /// pack does — those two sites take the same rotations the trunk's own embedding lookup and
    /// lm_head take. Getting this wrong is silent: the draft tokens come out plausible but
    /// uncorrelated, and the acceptance rate collapses to noise.
    /// </remarks>
    private ITensor? ForwardMtpCore(VulkanQwen3HybridDenseMtpWeights mtpHead, VulkanMtpState state,
                                    int tokenId, int position, bool computeLogits)
    {
        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = mtpHead.Attention.NumKvHeads;
        int headDim = Config.HeadDim;
        int qElems = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        int intermediateSize = mtpHead.Ffn.GateOutputDim;
        float eps = Config.NormEpsilon;

        // Position-indexed head KV-cache (issue #469): slot p holds the pair (h_{p-1}, x_p).
        if (state.CurrentLength > position)
            state.Rollback(position);
        else if (state.CurrentLength < position)
            throw new InvalidOperationException(
                $"MTP step at position {position} but the MTP KV-cache only covers {state.CurrentLength} " +
                "positions. Every trunk Forward of the sequence must pass the MTP state so the head " +
                "absorbs it (prefill included).");
        int step = position;
        if (step >= state.MaxSteps)
            throw new InvalidOperationException(
                $"VulkanMtpState KV-cache exhausted at position {position} (MaxSteps={state.MaxSteps}). " +
                "Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");

        _state.EnsureCapacity(1);
        _mtpScratch ??= MtpScratch.Allocate(_device, hiddenSize);
        var sc = _mtpScratch;

        long hiddenRowBytes = (long)hiddenSize * sizeof(float);
        var attnW = mtpHead.Attention;

        // RoPE reads the position from the shared positions buffer.
        Span<int> posOne = stackalloc int[1];
        posOne[0] = position;
        if (computeLogits) ProfBeginMtpStep();
        UploadPositions(posOne);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        ProfBeginSubmit(cmdBuf);

        // ── 1. Embed the predicted-from token into HiddenState row 0 ─────────
        if (mtpHead.EmbedTokensWeight is { } headEmbed)
        {
            // Head-local table: plain F32 row copy, and NOT Hadamard-latent.
            var region = new VkBufferCopy
            {
                srcOffset = (ulong)((long)tokenId * hiddenRowBytes),
                dstOffset = 0,
                size = (ulong)hiddenRowBytes,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, headEmbed.Handle, _state.HiddenState.Handle, 1, region);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
        else
        {
            // Trunk table — reuse the trunk's own gather, which already handles both the packed
            // PQ2_0 dispatch and the inverse Hadamard rotation a latent table needs.
            Span<int> oneToken = stackalloc int[1];
            oneToken[0] = tokenId;
            RecordEmbeddingGather(cmdBuf, oneToken);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Embed);

        // ── 2. enorm / hnorm, concatenated ───────────────────────────────────
        _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, mtpHead.EnormWeight, sc.ENorm,
            rowCount: 1, n: hiddenSize, eps: eps);
        _kernels.RmsNorm.Record(cmdBuf, state.PendingHidden, mtpHead.HnormWeight, sc.HNorm,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        RecordCopyBufferRange(cmdBuf, sc.ENorm, sc.Concat, 0, 0, (ulong)hiddenRowBytes);
        RecordCopyBufferRange(cmdBuf, sc.HNorm, sc.Concat, 0, (ulong)hiddenRowBytes, (ulong)hiddenRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        // ── 3. cur = eh_proj @ concat — the attention sub-block's residual ────
        RecordMatmul(cmdBuf, mtpHead.EhProjWeight, mtpHead.EhProjDeviceQuantType,
            sc.Concat, sc.Cur,
            outputDim: mtpHead.EhProjOutputDim, inputDim: mtpHead.EhProjInputDim, seqLen: 1);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.MtpEhProj);
        RecordCopyBufferRange(cmdBuf, sc.Cur, sc.Residual, 0, 0, (ulong)hiddenRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        // ── 4. Attention sub-block (seqQ=1 against the head's own KV-cache) ───
        _kernels.RmsNorm.Record(cmdBuf, sc.Cur, mtpHead.AttnNormWeight, sc.Normed,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        RecordMatmul(cmdBuf, attnW.QWeight, attnW.QDeviceQuantType,
            sc.Normed, _state.QGateScratch,
            outputDim: attnW.QOutputDim, inputDim: attnW.QInputDim, seqLen: 1);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);

        // De-interleave the fused Q+Gate row: [Q_h0, Gate_h0, Q_h1, Gate_h1, ...].
        long headBytes = (long)headDim * sizeof(float);
        for (int h = 0; h < numHeads; h++)
        {
            ulong qgHeadOff = (ulong)(h * 2 * headBytes);
            ulong qHeadOff = (ulong)(h * headBytes);
            RecordCopyBufferRange(cmdBuf, _state.QGateScratch, _state.Q, qgHeadOff, qHeadOff, (ulong)headBytes);
            RecordCopyBufferRange(cmdBuf, _state.QGateScratch, _state.GateScratch,
                qgHeadOff + (ulong)headBytes, qHeadOff, (ulong)headBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.CopyFanout);

        RecordMatmul(cmdBuf, attnW.KWeight, attnW.KDeviceQuantType,
            sc.Normed, _state.K,
            outputDim: attnW.KOutputDim, inputDim: attnW.KInputDim, seqLen: 1);
        RecordMatmul(cmdBuf, attnW.VWeight, attnW.VDeviceQuantType,
            sc.Normed, _state.V,
            outputDim: attnW.VOutputDim, inputDim: attnW.VInputDim, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);

        _kernels.RmsNorm.Record(cmdBuf, _state.Q, attnW.QNormWeight, _state.Q,
            rowCount: numHeads, n: headDim, eps: eps);
        _kernels.RmsNorm.Record(cmdBuf, _state.K, attnW.KNormWeight, _state.K,
            rowCount: numKvHeads, n: headDim, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        _kernels.Rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
            seqLen: 1, numHeads: numHeads, numKvHeads: numKvHeads,
            headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
            variant: RopeF32Kernel.Variant.NeoX);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnRope);

        // Append this step's K/V into the MTP head's own tiny cache, then attend causally over
        // everything drafted so far in this round — NOT the trunk's KV-cache.
        long kvRowBytes = (long)kvStride * sizeof(float);
        ulong kvSlot = (ulong)((long)step * kvRowBytes);
        RecordCopyBufferRange(cmdBuf, _state.K, state.KeyCache, 0, kvSlot, (ulong)kvRowBytes);
        RecordCopyBufferRange(cmdBuf, _state.V, state.ValueCache, 0, kvSlot, (ulong)kvRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnKvUpdate);

        _kernels.Attention.Record(cmdBuf, _state.Q, state.KeyCache, state.ValueCache, _state.AttnOutput,
            seqQ: 1, seqKv: step + 1,
            numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            positionOffset: step, slidingWindow: 0);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnCore);

        _kernels.SigmoidGateMul.Record(cmdBuf, _state.AttnOutput, _state.GateScratch, nTotal: qElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.AttnGate);

        RecordMatmul(cmdBuf, attnW.OWeight, attnW.ODeviceQuantType,
            _state.AttnOutput, sc.Cur,
            outputDim: attnW.OOutputDim, inputDim: attnW.OInputDim, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjAttn);

        _kernels.Add.Record(cmdBuf, sc.Residual, sc.Cur, sc.AddOut, hiddenSize);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, sc.AddOut, sc.Residual, 0, 0, (ulong)hiddenRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        // ── 5. Dense SwiGLU FFN sub-layer ────────────────────────────────────
        _kernels.RmsNorm.Record(cmdBuf, sc.AddOut, mtpHead.PostAttnNormWeight, sc.Normed,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        RecordMatmul(cmdBuf, mtpHead.Ffn.GateWeight, mtpHead.Ffn.GateDeviceQuantType,
            sc.Normed, _state.FfnGate,
            outputDim: mtpHead.Ffn.GateOutputDim, inputDim: mtpHead.Ffn.GateInputDim, seqLen: 1);
        RecordMatmul(cmdBuf, mtpHead.Ffn.UpWeight, mtpHead.Ffn.UpDeviceQuantType,
            sc.Normed, _state.FfnUp,
            outputDim: mtpHead.Ffn.UpOutputDim, inputDim: mtpHead.Ffn.UpInputDim, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjFfn);

        _kernels.SwiGlu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.FfnSilu, n: intermediateSize);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.FfnAct);

        RecordMatmul(cmdBuf, mtpHead.Ffn.DownWeight, mtpHead.Ffn.DownDeviceQuantType,
            _state.FfnSilu, sc.Cur,
            outputDim: mtpHead.Ffn.DownOutputDim, inputDim: mtpHead.Ffn.DownInputDim, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.ProjFfn);

        _kernels.Add.Record(cmdBuf, sc.Residual, sc.Cur, sc.AddOut, hiddenSize);

        if (!computeLogits)
        {
            // Absorb: only the KV row mattered; the caller seeds the next step from a trunk row.
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();
            state.Advance();
            return null;
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        // ── 6. Shared LM head ────────────────────────────────────────────────
        var headNormWeight = mtpHead.SharedHeadNormWeight ?? _weights.OutputNormWeight;
        _kernels.RmsNorm.Record(cmdBuf, sc.AddOut, headNormWeight, sc.NormedHead,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Norm);

        // The next chained draft step pairs this step's hidden state with the token it predicts.
        // llama.cpp chains the head's `h_nextn` — AFTER shared_head_norm (issue #469).
        RecordCopyBufferRange(cmdBuf, sc.NormedHead, state.PendingHidden, 0, 0, (ulong)hiddenRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Resid);

        VulkanDevice.Buffer headWeight;
        QuantizationType headQt;
        int headOutputDim, headInputDim;
        if (mtpHead.SharedHeadHeadWeight is { } localHead)
        {
            headWeight = localHead;
            headQt = mtpHead.SharedHeadHeadDeviceQuantType;
            headOutputDim = mtpHead.SharedHeadHeadOutputDim;
            headInputDim = mtpHead.SharedHeadHeadInputDim;
        }
        else
        {
            headWeight = _weights.OutputWeight;
            headQt = _weights.OutputDeviceQuantType;
            headOutputDim = _weights.OutputOutputDim;
            headInputDim = _weights.OutputInputDim;
        }

        var headIn = sc.NormedHead;
        // Only the TRUNK lm_head is folded; a head-local shared_head_head is not.
        if (mtpHead.UsesTrunkLmHead && _hadamard is { } mtpHeadRot)
        {
            headIn = _state.HadamardScratch!;
            mtpHeadRot.RecordForward(cmdBuf, sc.NormedHead, headIn, 1, headInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
        }

        RecordMatmul(cmdBuf, headWeight, headQt, headIn, _state.Logits,
            outputDim: headOutputDim, inputDim: headInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        ProfMark(cmdBuf, VulkanOpProfiler.Cat.LmHead);
        ProfBeforeSubmit(cmdBuf);
        _submit.SubmitAndWait();
        ProfAfterSubmit();

        state.Advance();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        ProfMtpTailStart();
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        ProfEndMtpStep();
        return result;
    }

    // ── Matmul dispatch ──────────────────────────────────────────────────────

    /// <summary>
    /// Dispatches the GEMV (<paramref name="seqLen"/> == 1) or GEMM variant for the
    /// weight's on-device quantization. Same policy table as
    /// <see cref="VulkanQwen3MoeHybridTransformerModel"/>.
    /// </summary>
    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantizationType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (_prof is not null) ProfNoteMatmul(weightQt, outputDim, inputDim, seqLen);

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
            case QuantizationType.IQ3_XXS:
                if (seqLen == 1)
                    _kernels.MatMulIq3Xxs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq3XxsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ3_S:
                if (seqLen == 1)
                    _kernels.MatMulIq3S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq3SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.IQ1_S:
                if (seqLen == 1)
                    _kernels.MatMulIq1S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else
                    _kernels.MatMulIq1SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.PQ2_0:
                // PQ2_0 (PrismML Bonsai ternary): 128-element group alignment, enforced upload-side
                // by VulkanQwen3MoeHybridWeights.KeepPQ2_0. Each group carries its own fp16 scale,
                // read and applied in-shader. Widening this to F32 instead is what used to make
                // Bonsai 2 27B ask for ~108 GB of device-local memory.
                // #446/#470: the kernel choice is PQ2_0SmallNDispatch's, not a bare
                // seqLen == 1 test -- 2-8 token verify batches go to the multi-column GEMV,
                // which reads the weights once for all of them; the 128x128 GEMM tile costs
                // 4-6 single-token GEMVs even at n = 2.
                PQ2_0SmallNDispatch.Record(cmdBuf, _kernels.MatMulPQ2_0, _kernels.MatMulPQ2_0Gemm,
                    weights, input, output, m: outputDim, k: inputDim, n: seqLen);
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

    private void RecordEmbeddingGather(nint cmdBuf, ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;

        // Packed PQ2_0 table: gather + dequantize as a compute dispatch. The widened F32 form of
        // Bonsai 2's token_embd is 5.08 GB, over Vulkan's 4 GiB maxStorageBufferRange, so this is
        // the only way the table can be resident at all.
        if (_embedGather is { } gather)
        {
            _device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tokenIds), _state.TokenIdsBuffer!);
            KernelSupport.HostToComputeBarrier(cmdBuf);
            gather.Record(cmdBuf, _weights.TokenEmbedding, _state.TokenIdsBuffer!, _state.HiddenState,
                nTokens: tokenIds.Length, hidden: hiddenSize, vocabSize: Config.VocabSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Embed);

            if (_hadamard is { } packedEmbRot)
            {
                packedEmbRot.RecordInverseInPlace(cmdBuf, _state.HiddenState, tokenIds.Length, hiddenSize);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
            }
            return;
        }

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

        ProfMark(cmdBuf, VulkanOpProfiler.Cat.Embed);

        // A Hadamard-latent embedding table stores rotated rows, so restore the primal basis right
        // after the lookup. Note the INVERSE order — rotation then signs — which is the opposite of
        // every folded-weight site. In place is safe here: without the permute each workgroup reads
        // only the block it writes.
        if (_hadamard is { } embRot)
        {
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            embRot.RecordInverseInPlace(cmdBuf, _state.HiddenState, tokenIds.Length, hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            ProfMark(cmdBuf, VulkanOpProfiler.Cat.Hadamard);
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
        if (_disposed) return;
        _disposed = true;
        Interlocked.Exchange(ref _spareGdnCheckpoint, null)?.Dispose();
        // Before _device: the profiler owns a query pool on it.
        _profiler?.Dispose();
        _profiler = null;
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _gdnCache.Dispose();
        _hadamard?.Dispose();
        _embedGather?.Dispose();
        _mtpHead?.Dispose();
        _mtpScratch?.Dispose();
        _multiRowLogits?.Dispose();
        _kernels.Dispose();
        // Frees the CPU model's dequantised norm arrays and detaches it from the
        // GgufFile. The GgufFile itself is caller-owned.
        _cpuModel?.Dispose();
        if (_ownsDevice) _device.Dispose();
    }
}
