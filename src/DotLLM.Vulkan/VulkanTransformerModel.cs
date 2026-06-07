using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end F32 Vulkan forward pass for Llama-family transformer models.
/// Implements <see cref="IModel"/> using only the six wave-1/wave-2 Vulkan
/// compute kernels: <see cref="MatMulF32Kernel"/>, <see cref="RmsNormF32Kernel"/>,
/// <see cref="RopeF32Kernel"/>, <see cref="AttentionF32Kernel"/>,
/// <see cref="SwiGluF32Kernel"/>, plus <see cref="AddKernel"/> for residuals.
/// </summary>
/// <remarks>
/// <para>
/// Scope: F32-only. Quantised weights are dequantised to FP32 at
/// construction time via <see cref="VulkanWeights.Upload"/>. Llama-family
/// dense and Mixtral-convention MoE FFNs are supported; MLA attention
/// and SSM layers are rejected at load time. DeepSeek-V2/V3 shared-expert
/// MoE is not yet wired in.
/// </para>
/// <para>
/// The forward pass is synchronous: every kernel dispatch in the chain
/// ends in <c>vkQueueWaitIdle</c> (inherited from the wave-1 kernels). That
/// is correct but slow; fence-based pipelining is deferred to the
/// perf-polish wave per the scaffold discipline.
/// </para>
/// <para>
/// Architectural parallel with <c>DotLLM.Cuda.CudaTransformerModel</c>:
/// upload weights once at construction, reuse a single
/// <see cref="VulkanForwardState"/> for scratch, and drive every linear
/// projection through one <c>matmul_f32</c> call — no prefill / decode
/// split because there is no quantised GEMV kernel yet. Logits come back
/// as a single <see cref="UnmanagedTensor"/> of shape <c>[1, vocabSize]</c>
/// matching the CUDA return convention.
/// </para>
/// </remarks>
public sealed class VulkanTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanWeights _weights;
    private readonly VulkanForwardState _state;

    // Kernels — one instance each, pipelines are reused across all launches.
    private readonly MatMulF32Kernel _matmul;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    private readonly SwiGluF32Kernel _swiglu;
    private readonly AddKernel _add;

    // MoE kernel set — constructed only when the model carries a MoE layer.
    private readonly MoeTopKSoftmaxF32Kernel? _moeTopkSoftmax;
    private readonly MoeIndexedMatmulF32Kernel? _moeIndexedMatmul;
    private readonly MoeWeightedScatterF32Kernel? _moeWeightedScatter;
    private readonly MoeBroadcastF32Kernel? _moeBroadcast;

    private readonly TransformerWeights _cpuWeights; // retained for embedding lookup
    private readonly GgufFile? _gguf;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly RopeF32Kernel.Variant _ropeVariant;
    private readonly int _slidingWindow;
    private readonly bool _ownsDevice;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes;

    /// <summary>Creates a <see cref="VulkanKvCache"/> sized for this model.</summary>
    public VulkanKvCache CreateKvCache(int maxSeqLen)
        => new(_device, Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);

    private VulkanTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, VulkanWeights weights, TransformerWeights cpuWeights,
        VulkanForwardState state,
        MatMulF32Kernel matmul, RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, SwiGluF32Kernel swiglu, AddKernel add,
        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax,
        MoeIndexedMatmulF32Kernel? moeIndexedMatmul,
        MoeWeightedScatterF32Kernel? moeWeightedScatter,
        MoeBroadcastF32Kernel? moeBroadcast,
        GgufFile? gguf,
        float ropeTheta, int ropeDim, RopeF32Kernel.Variant ropeVariant, int slidingWindow)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _cpuWeights = cpuWeights;
        _state = state;
        _matmul = matmul;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _swiglu = swiglu;
        _add = add;
        _moeTopkSoftmax = moeTopkSoftmax;
        _moeIndexedMatmul = moeIndexedMatmul;
        _moeWeightedScatter = moeWeightedScatter;
        _moeBroadcast = moeBroadcast;
        _gguf = gguf;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeVariant = ropeVariant;
        _slidingWindow = slidingWindow;
    }

    /// <summary>
    /// Loads a model from an opened GGUF file onto a new Vulkan device.
    /// The caller owns the returned model; disposing it tears down the
    /// device, pipelines, and weight buffers.
    /// </summary>
    /// <param name="gguf">Opened GGUF file. Must remain alive for the model's lifetime.</param>
    /// <param name="config">Model configuration extracted from the GGUF metadata.</param>
    /// <param name="spvDir">
    /// Directory containing the compiled Vulkan SPIR-V blobs. When null,
    /// falls back to <c>spv/</c> next to the running assembly (matches the
    /// MSBuild <c>Content</c> copy pattern used by the Vulkan project).
    /// </param>
    public static VulkanTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model onto an existing <see cref="VulkanDevice"/>. The device
    /// is NOT disposed when the model is disposed — the caller retains
    /// ownership. Useful when the device is shared with other Vulkan
    /// components (e.g. a diagnostic hook that wants to launch its own
    /// kernels on the same queue).
    /// </summary>
    public static VulkanTransformerModel LoadFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf);
    }

    /// <summary>
    /// Loads a model from an opened HuggingFace-convention safetensors file
    /// onto a new Vulkan device. The caller owns the returned model; disposing
    /// it tears down the device, pipelines, and weight buffers. The safetensors
    /// file must remain alive (its mmap is referenced by the weight pointers).
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        SafetensorsFile file, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf: null);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model from an opened HuggingFace-convention safetensors file
    /// onto an existing <see cref="VulkanDevice"/>. The device is NOT disposed
    /// when the model is disposed.
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        VulkanDevice device, SafetensorsFile file, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf: null);
    }

    private static VulkanTransformerModel BuildModel(
        VulkanDevice device, bool ownsDevice, ModelConfig config,
        TransformerWeights cpuWeights, string spvDir, GgufFile? gguf)
    {
        var weights = VulkanWeights.Upload(device, cpuWeights, config.NumLayers);

        // MoE forward scratch is sized from the layer that carries the MoE
        // bundle (any/all of them — they share the per-layer MoE shape).
        // When no layer carries one, leave the MoE dims at zero so
        // VulkanForwardState skips allocation entirely.
        int moeNumExperts = 0, moeTopK = 0, moeIntermediate = 0;
        int moeSharedIntermediate = 0, moeNumSharedExperts = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (weights.Layers[i].Moe is { } m)
            {
                moeNumExperts = Math.Max(moeNumExperts, m.NumExperts);
                moeTopK = Math.Max(moeTopK, m.NumExpertsPerTok);
                moeIntermediate = Math.Max(moeIntermediate, m.IntermediateSize);
                if (m.NumSharedExperts > 0)
                {
                    moeSharedIntermediate = Math.Max(moeSharedIntermediate, m.SharedIntermediateSize);
                    moeNumSharedExperts = Math.Max(moeNumSharedExperts, m.NumSharedExperts);
                }
            }
        }

        var state = new VulkanForwardState(device,
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            initialSeqLen: 1,
            moeNumExperts: moeNumExperts,
            moeTopK: moeTopK,
            moeIntermediateSize: moeIntermediate,
            moeSharedIntermediateSize: moeSharedIntermediate,
            moeNumSharedExperts: moeNumSharedExperts);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);

        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax = null;
        MoeIndexedMatmulF32Kernel? moeIndexedMatmul = null;
        MoeWeightedScatterF32Kernel? moeWeightedScatter = null;
        MoeBroadcastF32Kernel? moeBroadcast = null;
        if (moeNumExperts > 0)
        {
            moeTopkSoftmax = MoeTopKSoftmaxF32Kernel.Create(device, spvDir);
            moeIndexedMatmul = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
            moeWeightedScatter = MoeWeightedScatterF32Kernel.Create(device, spvDir);
            moeBroadcast = MoeBroadcastF32Kernel.Create(device, spvDir);
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        var ropeVariant = ropeType == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm;

        int slidingWindow = config.SlidingWindowSize ?? 0;

        return new VulkanTransformerModel(
            device, ownsDevice,
            config, weights, cpuWeights, state,
            matmul, rmsnorm, rope, attention, swiglu, add,
            moeTopkSoftmax, moeIndexedMatmul, moeWeightedScatter, moeBroadcast,
            gguf,
            ropeTheta, ropeDim, ropeVariant, slidingWindow);
    }

    private static void RejectUnsupportedArchitecture(ModelConfig config)
    {
        if (config.MlaConfig is not null)
            throw new NotSupportedException("MLA (DeepSeek-V2/V3) is not supported on the Vulkan backend yet.");

        // MoE / HybridLayout / SsmConfig / Mamba3Config guards live with the
        // chains that introduce those ModelConfig properties — they will be
        // wired into RejectUnsupportedArchitecture by those chains' Vulkan
        // follow-up PRs. F32-dense routing is the only path supported here.
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
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        _state.EnsureCapacity(seqLen);

        // 1. Embedding lookup on the host: copy one row per token from the already-FP32 embedding
        //    table into a scratch buffer, then upload. (A kernel would be nicer; cost is 576*4*seqLen
        //    bytes — ~2.3 KB/token for SmolLM — which is trivial for prompts we care about.)
        UploadEmbeddings(tokenIds);

        // 2. Upload positions for RoPE.
        UploadPositions(positions);

        // 3. Main transformer forward.
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // Copy hidden -> residual (for later add). Host-side copy via mapped memory.
            CopyDeviceBuffer(_state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));

            // ── Attention block ──────────────────────────────────────────

            // a. RMSNorm attn_in
            _rmsnorm.Launch(_state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);

            // b. Q/K/V projections (matmul [M,K] x [N,K]^T -> [N,M])
            _matmul.Launch(lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            _matmul.Launch(lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            _matmul.Launch(lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

            // c. Optional biases.
            if (lw.QBias is { } qb) AddBiasRows(_state.Q, qb, lw.QOutputDim, seqLen);
            if (lw.KBias is { } kb) AddBiasRows(_state.K, kb, lw.KOutputDim, seqLen);
            if (lw.VBias is { } vb) AddBiasRows(_state.V, vb, lw.VOutputDim, seqLen);

            // d. RoPE on Q and K.
            _rope.Launch(_state.Q, _state.K, _state.PositionsBuffer,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);

            // e. Attention: either over the current seqLen window (no cache)
            //    or over the cached sequence (prefill fills the cache, decode
            //    reads from it).
            VulkanDevice.Buffer kSrc, vSrc;
            int seqKv;
            int positionOffset;
            if (kvCache is VulkanKvCache vkCache)
            {
                vkCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer);
                kSrc = vkCache.GetKeysBuffer(layer);
                vSrc = vkCache.GetValuesBuffer(layer);
                seqKv = vkCache.CurrentLength;
                positionOffset = positions[0];
            }
            else
            {
                kSrc = _state.K;
                vSrc = _state.V;
                seqKv = seqLen;
                positionOffset = 0;
            }

            _attention.Launch(_state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: _slidingWindow);

            // f. Output projection -> NormOutput (reuse slot).
            _matmul.Launch(lw.O, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias is { } ob) AddBiasRows(_state.NormOutput, ob, lw.OOutputDim, seqLen);

            // g. Residual add: hidden = residual + attn_out (written to NormOutput above).
            //    Add kernel forbids output aliasing; write into AddScratch, then swap.
            _add.Launch(_state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            CopyDeviceBuffer(_state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));

            // Refresh residual (copy of the post-attention hidden state) for the FFN residual.
            CopyDeviceBuffer(_state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));

            // ── FFN block ───────────────────────────────────────────────
            if (lw.Moe is { } moeW)
            {
                // MoE FFN — writes the combined output into NormOutput.
                RunMoeLayer(moeW, lw.FfnNormWeight, seqLen, hiddenSize, eps);
            }
            else
            {
                // h. RMSNorm ffn_in
                _rmsnorm.Launch(_state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);

                // i. Gate/Up projections
                _matmul.Launch(lw.Gate, _state.NormOutput, _state.FfnGate,
                    lw.GateOutputDim, lw.GateInputDim, seqLen);
                _matmul.Launch(lw.Up, _state.NormOutput, _state.FfnUp,
                    lw.UpOutputDim, lw.UpInputDim, seqLen);
                if (lw.GateBias is { } gb) AddBiasRows(_state.FfnGate, gb, lw.GateOutputDim, seqLen);
                if (lw.UpBias is { } ub) AddBiasRows(_state.FfnUp, ub, lw.UpOutputDim, seqLen);

                // j. SwiGLU: silu(gate) * up -> SiluOutput
                _swiglu.Launch(_state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);

                // k. Down projection -> NormOutput
                _matmul.Launch(lw.Down, _state.SiluOutput, _state.NormOutput,
                    lw.DownOutputDim, lw.DownInputDim, seqLen);
                if (lw.DownBias is { } db) AddBiasRows(_state.NormOutput, db, lw.DownOutputDim, seqLen);
            }

            // l. Residual add: hidden = residual + ffn_out
            _add.Launch(_state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            CopyDeviceBuffer(_state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));
        }

        // 4. Final RMSNorm on the last token only, then LM head.
        //    Copy just the last token's hidden state into a single-row view via NormOutput.
        CopyLastTokenSlice(_state.HiddenState, _state.NormOutput, seqLen, hiddenSize);
        _rmsnorm.Launch(_state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);

        _matmul.Launch(_weights.OutputWeight, _state.NormOutput, _state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);

        // 5. Return logits as a host-resident UnmanagedTensor [1, vocabSize].
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
    /// Runs the MoE FFN sub-layer using the Vulkan MoE kernel set —
    /// rmsnorm → gate matmul → top-k softmax → broadcast → indexed
    /// matmul (W1 / W3) → SwiGLU → indexed matmul (W2) → weighted
    /// scatter. Mirrors the per-(token, slot) accumulation order of
    /// <c>MoeSwiGluMlp.Execute</c> on the CPU, so the result is
    /// numerically parity with the CPU reference modulo F32 rounding.
    /// Writes the combined per-token output into <see cref="VulkanForwardState.NormOutput"/>.
    /// </summary>
    private unsafe void RunMoeLayer(
        VulkanWeights.MoeLayerBuffers moeW, VulkanDevice.Buffer ffnNormWeight,
        int seqLen, int hiddenSize, float eps)
    {
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int interm = moeW.IntermediateSize;
        int nExpanded = seqLen * topK;

        // 1. RMSNorm into NormOutput.
        _rmsnorm.Launch(_state.HiddenState, ffnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hiddenSize, eps: eps);

        // 2. Router gate matmul -> MoeRouterLogits [seqLen, numExperts].
        _matmul.Launch(moeW.Gate, _state.NormOutput, _state.MoeRouterLogits!,
            m: numE, k: hiddenSize, n: seqLen);

        // 3. Top-k softmax: writes MoeTopkIndices (int) and MoeTopkWeights.
        _moeTopkSoftmax!.Launch(
            _state.MoeRouterLogits!, _state.MoeTopkIndices!, _state.MoeTopkWeights!,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);

        // 4. Broadcast NormOutput[seqLen, hidden] → MoeExpandedInput[seqLen*topK, hidden].
        //    Each token's row is replicated topK times. One compute dispatch
        //    replaces the seqLen × topK loop of vkCmdCopyBuffer regions the
        //    previous implementation issued — same math (no FP arithmetic,
        //    bit-exact), no transfer↔compute stage transitions, dispatch
        //    count drops from O(seqLen·topK) to 1 per MoE layer.
        _moeBroadcast!.Launch(_state.NormOutput, _state.MoeExpandedInput!,
            seqLen: seqLen, topK: topK, hidden: hiddenSize);

        // 5. Indexed matmul W1 (gate_proj): ExpandedInput → GateInter.
        _moeIndexedMatmul!.Launch(
            moeW.W1Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeGateInter!,
            m: interm, k: hiddenSize, n: nExpanded, numExperts: numE);

        // 6. Indexed matmul W3 (up_proj): ExpandedInput → UpInter.
        _moeIndexedMatmul.Launch(
            moeW.W3Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeUpInter!,
            m: interm, k: hiddenSize, n: nExpanded, numExperts: numE);

        // 7. SwiGLU: silu(GateInter) * UpInter → SiluInter.
        _swiglu.Launch(_state.MoeGateInter!, _state.MoeUpInter!, _state.MoeSiluInter!,
            n: nExpanded * interm);

        // 8. Indexed matmul W2 (down_proj): SiluInter → DownRows [seqLen*topK, hidden].
        _moeIndexedMatmul.Launch(
            moeW.W2Bank, _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeDownRows!,
            m: hiddenSize, k: interm, n: nExpanded, numExperts: numE);

        // 9. Weighted scatter: collapse [seqLen*topK, hidden] into [seqLen, hidden],
        //    overwriting NormOutput with the combined per-token MoE output so the
        //    residual-add tail runs unchanged.
        _moeWeightedScatter!.Launch(
            _state.MoeDownRows!, _state.MoeTopkWeights!, _state.NormOutput,
            seqLen: seqLen, topK: topK, hiddenSize: hiddenSize);

        // 10. Shared-expert branch (DeepSeek-V2/V3 ungated). Each shared expert
        //     runs a dense SwiGLU MLP on the per-token hidden state and the
        //     outputs are summed into the routed result. Skipped when the
        //     layer has no shared experts (Mixtral / Qwen3-MoE without shared).
        if (moeW.NumSharedExperts > 0)
        {
            RunMoeSharedExperts(moeW, ffnNormWeight, seqLen, hiddenSize, eps);
        }
    }

    /// <summary>
    /// Runs the shared-expert branch of a DeepSeek-V2/V3-style MoE layer:
    /// for each shared expert run a dense SwiGLU MLP over the per-token
    /// normalised hidden state, sum the outputs, and add the sum into the
    /// routed-MoE result already in <c>_state.NormOutput</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The routed-MoE scatter has overwritten <c>NormOutput</c> with the
    /// routed sum already, so we re-derive the normalised hidden state from
    /// <c>HiddenState</c> via a fresh rmsnorm into
    /// <see cref="VulkanForwardState.MoeSharedInput"/> — a dedicated buffer
    /// that pins the shared-expert input across every iteration. That keeps
    /// the SumA / SumB pair available as pure ping-pong accumulator slots.
    /// </para>
    /// <para>
    /// Accumulation: shared expert 0's down-projection writes directly into
    /// SumA. For each subsequent expert we matmul into MoeSharedDown and add
    /// (running-sum, MoeSharedDown) into the alternating ping-pong side.
    /// After all shared experts the running sum is folded into NormOutput via
    /// the unused ping-pong slot and a host-mapped memcpy lands the result
    /// back in <c>NormOutput</c> so the caller's residual-add contract is
    /// preserved.
    /// </para>
    /// <para>
    /// Each kernel here is invoked via the standard Launch API
    /// (one-time-submit command buffer + <c>vkQueueWaitIdle</c>), matching
    /// the rest of the Vulkan forward pass — no explicit
    /// compute-to-compute barriers are needed because the queue wait already
    /// publishes the prior dispatch's writes.
    /// </para>
    /// </remarks>
    private unsafe void RunMoeSharedExperts(
        VulkanWeights.MoeLayerBuffers moeW, VulkanDevice.Buffer ffnNormWeight,
        int seqLen, int hiddenSize, float eps)
    {
        int numShared = moeW.NumSharedExperts;
        int sharedI = moeW.SharedIntermediateSize;
        int sharedInterElems = seqLen * sharedI;
        long hiddenBytes = (long)seqLen * hiddenSize * sizeof(float);

        // Re-derive the normalised hidden state. NormOutput is occupied by
        // the routed-MoE result; HiddenState still holds the pre-FFN residual.
        var sharedInput = _state.MoeSharedInput!;
        _rmsnorm.Launch(_state.HiddenState, ffnNormWeight, sharedInput,
            rowCount: seqLen, n: hiddenSize, eps: eps);

        // SumA / SumB ping-pong; activeSum tracks the slot currently holding
        // the running shared-expert sum. Expert 0 writes directly into SumA;
        // subsequent experts compute their down-output into MoeSharedDown and
        // we add (activeSum + MoeSharedDown) → the OTHER side, alternating.
        VulkanDevice.Buffer activeSum = _state.MoeSharedSumA!;

        for (int s = 0; s < numShared; s++)
        {
            // gate / up matmuls share sharedInput; SwiGLU then fuses them.
            _matmul.Launch(moeW.SharedW1![s], sharedInput, _state.MoeSharedGate!,
                m: sharedI, k: hiddenSize, n: seqLen);
            _matmul.Launch(moeW.SharedW3![s], sharedInput, _state.MoeSharedUp!,
                m: sharedI, k: hiddenSize, n: seqLen);

            _swiglu.Launch(_state.MoeSharedGate!, _state.MoeSharedUp!, _state.MoeSharedSilu!,
                n: sharedInterElems);

            if (s == 0)
            {
                // First expert seeds the running sum directly into SumA.
                _matmul.Launch(moeW.SharedW2![s], _state.MoeSharedSilu!, _state.MoeSharedSumA!,
                    m: hiddenSize, k: sharedI, n: seqLen);
                activeSum = _state.MoeSharedSumA!;
            }
            else
            {
                // Per-expert down output → MoeSharedDown, then ping-pong add
                // into the slot opposite activeSum.
                _matmul.Launch(moeW.SharedW2![s], _state.MoeSharedSilu!, _state.MoeSharedDown!,
                    m: hiddenSize, k: sharedI, n: seqLen);

                var sumDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
                    ? _state.MoeSharedSumB!
                    : _state.MoeSharedSumA!;
                _add.Launch(activeSum, _state.MoeSharedDown!, sumDst, seqLen * hiddenSize);
                activeSum = sumDst;
            }
        }

        // Fold the running shared sum into NormOutput. We need a destination
        // buffer different from both NormOutput (read) and activeSum (read);
        // pick the OTHER ping-pong slot, then copy it back into NormOutput so
        // the caller's residual-add reads the merged result.
        VulkanDevice.Buffer foldDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
            ? _state.MoeSharedSumB!
            : _state.MoeSharedSumA!;
        _add.Launch(_state.NormOutput, activeSum, foldDst, seqLen * hiddenSize);

        // Host-mapped device-to-device copy back into NormOutput so the
        // residual-add tail downstream sees the merged routed + shared sum.
        // Matches the rest of the file's mapMemory + memcpy idiom — all
        // buffers are host-visible host-coherent and the prior Launch ended
        // in vkQueueWaitIdle, so reads are already published.
        CopyDeviceBuffer(foldDst, _state.NormOutput, hiddenBytes);
    }

    /// <summary>
    /// Copies <paramref name="byteCount"/> bytes from <paramref name="src"/> to
    /// <paramref name="dst"/> via host-mapped pointers. Scaffold path — the
    /// proper implementation issues <c>vkCmdCopyBuffer</c> on the compute
    /// queue, but while all buffers are host-visible host-coherent this is
    /// both correct and serialised against prior kernel launches (each of
    /// which ends in <c>vkQueueWaitIdle</c>).
    /// </summary>
    private unsafe void CopyDeviceBuffer(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long byteCount)
    {
        VulkanApi.vkMapMemory(_device.Handle, src.Memory, 0, (ulong)byteCount, 0, out nint srcMapped)
            .ThrowOnError("vkMapMemory CopyDeviceBuffer.src");
        VulkanApi.vkMapMemory(_device.Handle, dst.Memory, 0, (ulong)byteCount, 0, out nint dstMapped)
            .ThrowOnError("vkMapMemory CopyDeviceBuffer.dst");
        try
        {
            System.Buffer.MemoryCopy((void*)srcMapped, (void*)dstMapped, byteCount, byteCount);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, dst.Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, src.Memory);
        }
    }

    private unsafe void CopyLastTokenSlice(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int seqLen, int hiddenSize)
    {
        long rowBytes = (long)hiddenSize * sizeof(float);
        long srcOffset = (long)(seqLen - 1) * rowBytes;

        VulkanApi.vkMapMemory(_device.Handle, src.Memory, 0, (ulong)(srcOffset + rowBytes), 0, out nint srcMapped)
            .ThrowOnError("vkMapMemory CopyLastTokenSlice.src");
        VulkanApi.vkMapMemory(_device.Handle, dst.Memory, 0, (ulong)rowBytes, 0, out nint dstMapped)
            .ThrowOnError("vkMapMemory CopyLastTokenSlice.dst");
        try
        {
            System.Buffer.MemoryCopy((byte*)srcMapped + srcOffset, (void*)dstMapped, rowBytes, rowBytes);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, dst.Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, src.Memory);
        }
    }

    /// <summary>
    /// Adds a per-feature bias vector to every row of a
    /// <c>[seqLen, outputDim]</c> FP32 output buffer. Implemented in-place on
    /// the host via mapped memory — biases are tiny (hidden_size scale), and
    /// adding a dedicated "bias_add" compute kernel is out of scope for the
    /// correctness wave.
    /// </summary>
    private unsafe void AddBiasRows(VulkanDevice.Buffer output, VulkanDevice.Buffer bias, int outputDim, int seqLen)
    {
        long biasBytes = (long)outputDim * sizeof(float);
        long outBytes = biasBytes * seqLen;

        VulkanApi.vkMapMemory(_device.Handle, output.Memory, 0, (ulong)outBytes, 0, out nint outMapped)
            .ThrowOnError("vkMapMemory AddBiasRows.output");
        VulkanApi.vkMapMemory(_device.Handle, bias.Memory, 0, (ulong)biasBytes, 0, out nint biasMapped)
            .ThrowOnError("vkMapMemory AddBiasRows.bias");
        try
        {
            float* o = (float*)outMapped;
            float* b = (float*)biasMapped;
            for (int t = 0; t < seqLen; t++)
            {
                for (int i = 0; i < outputDim; i++)
                    o[t * outputDim + i] += b[i];
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, bias.Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, output.Memory);
        }
    }

    /// <summary>
    /// Resolves each token ID into its FP32 embedding row and packs the
    /// result into <see cref="VulkanForwardState.HiddenState"/>. Does a
    /// row-by-row dequant when the table was Q8_0 / F16 / other (GGUF often
    /// quantises the embedding table alongside the weights).
    /// </summary>
    private unsafe void UploadEmbeddings(ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        int vocab = Config.VocabSize;
        int seqLen = tokenIds.Length;
        var qt = _cpuWeights.TokenEmbedQuantType;

        long rowBytes = (long)hiddenSize * sizeof(float);
        VulkanApi.vkMapMemory(_device.Handle, _state.HiddenState.Memory, 0, (ulong)(seqLen * rowBytes), 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadEmbeddings");
        try
        {
            float* dst = (float*)mapped;

            if (qt == QuantizationType.F32)
            {
                // Direct memcpy from mmap.
                float* src = (float*)_cpuWeights.TokenEmbedWeight;
                for (int t = 0; t < seqLen; t++)
                {
                    int id = tokenIds[t];
                    if ((uint)id >= (uint)vocab)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
                    new ReadOnlySpan<float>(src + (long)id * hiddenSize, hiddenSize)
                        .CopyTo(new Span<float>(dst + (long)t * hiddenSize, hiddenSize));
                }
            }
            else
            {
                // Dequantize one row per token into mapped hidden-state region.
                long tableRowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                for (int t = 0; t < seqLen; t++)
                {
                    int id = tokenIds[t];
                    if ((uint)id >= (uint)vocab)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
                    nint rowPtr = _cpuWeights.TokenEmbedWeight + (nint)(id * tableRowBytes);
                    Dequantize.ToFloat32(rowPtr, hiddenSize, qt,
                        new Span<float>(dst + (long)t * hiddenSize, hiddenSize));
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, _state.HiddenState.Memory);
        }
    }

    private unsafe void UploadPositions(ReadOnlySpan<int> positions)
    {
        // The Allocate in EnsureCapacity already sized PositionsBuffer for seqLen;
        // delegate the mapped copy to device.Upload via a raw byte span.
        var posBytes = MemoryMarshal.AsBytes(positions);
        _device.Upload(posBytes, _state.PositionsBuffer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _state.Dispose();
        _weights.Dispose();

        _moeBroadcast?.Dispose();
        _moeWeightedScatter?.Dispose();
        _moeIndexedMatmul?.Dispose();
        _moeTopkSoftmax?.Dispose();

        _add.Dispose();
        _swiglu.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _matmul.Dispose();

        _cpuWeights.Dispose();
        if (_ownsDevice)
            _device.Dispose();
    }
}
