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
/// <b>Not yet covered.</b> The MTP ("NextN") head — <see cref="Qwen3HybridDenseTransformerModel.ForwardMtp"/>
/// on CPU and the CUDA equivalent — is not implemented here. A <c>qwen35</c>
/// checkpoint carrying an MTP block loads and generates normally; only
/// MTP-accelerated speculative decoding is unavailable on Vulkan.
/// </para>
/// </remarks>
public sealed class VulkanQwen3HybridDenseTransformerModel : IModel
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
    private readonly VulkanGdnStateCache _gdnCache;
    private readonly VulkanQwen3MoeHybridKernels _kernels;

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
        int ropeDim, float ropeTheta)
    {
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
        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir);

        return new VulkanQwen3HybridDenseTransformerModel(
            device, ownsDevice,
            config, gguf, cpuModel, weights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, gdnLayerOrdinal,
            ropeDim, ropeTheta);
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
    {
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

            // ── 2a. Token mixing ────────────────────────────────────────────
            RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

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
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            // ── 2b. Dense SwiGLU FFN ────────────────────────────────────────
            // Snapshot the post-token-mixing hidden state as the FFN residual,
            // then post-norm → gate/up → SwiGLU → down → residual add. Mirrors
            // Qwen3HybridDenseTransformerModel.ForwardDenseFfnBody.
            RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            RecordDenseFfn(cmdBuf, layerBuf.Ffn, seqLen, intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
                seqLen * hiddenSize);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
                0, 0, (ulong)((long)seqLen * hiddenRowBytes));
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();
        }

        // ── 3. Final norm + LM head (single submission, last token only) ──────
        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        long lastRowOffset = (long)(seqLen - 1) * hiddenRowBytes;
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)hiddenRowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.RmsNorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // ── 4. Download logits ───────────────────────────────────────────────
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        return result;
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
        RecordMatmul(cmdBuf, ffn.GateWeight, ffn.GateDeviceQuantType,
            _state.NormOutput, _state.FfnGate,
            outputDim: ffn.GateOutputDim, inputDim: ffn.GateInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, ffn.UpWeight, ffn.UpDeviceQuantType,
            _state.NormOutput, _state.FfnUp,
            outputDim: ffn.UpOutputDim, inputDim: ffn.UpInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _kernels.SwiGlu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.FfnSilu,
            n: checked(seqLen * intermediateSize));
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, ffn.DownWeight, ffn.DownDeviceQuantType,
            _state.FfnSilu, _state.NormOutput,
            outputDim: ffn.DownOutputDim, inputDim: ffn.DownInputDim, seqLen: seqLen);
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

        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnQBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnKBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── 5. GDN scan — single multi-token dispatch ────────────────────────
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
        _kernels.RmsNorm.Record(cmdBuf, _state.Q, attnW.QNormWeight, _state.Q,
            rowCount: seqLen * numHeads, n: headDim, eps: Config.NormEpsilon);
        _kernels.RmsNorm.Record(cmdBuf, _state.K, attnW.KNormWeight, _state.K,
            rowCount: seqLen * numKvHeads, n: headDim, eps: Config.NormEpsilon);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

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

        if (_kernels.SplitKvAttention is not null && seqLen == 1
            && headDim <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads))
        {
            _kernels.SplitKvAttention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else if (_kernels.FlashAttention is not null && seqLen > 1 && headDim <= VulkanFlashAttentionF32Kernel.MaxHeadDim)
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
        _state.Dispose();
        _weights.Dispose();
        _gdnCache.Dispose();
        _kernels.Dispose();
        // Frees the CPU model's dequantised norm arrays and detaches it from the
        // GgufFile. The GgufFile itself is caller-owned.
        _cpuModel?.Dispose();
        if (_ownsDevice) _device.Dispose();
    }
}
