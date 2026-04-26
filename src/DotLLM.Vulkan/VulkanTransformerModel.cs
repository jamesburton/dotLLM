using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

using QuantType = DotLLM.Core.Configuration.QuantizationType;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for Llama-family transformer models.
/// Implements <see cref="IModel"/> using the wave-1/wave-2 Vulkan compute
/// kernels: <see cref="MatMulF32Kernel"/> plus the Q8_0 matmul kernels
/// (<see cref="MatMulQ8_0Kernel"/> for decode-path GEMV,
/// <see cref="MatMulQ8_0GemmKernel"/> for batched prefill),
/// <see cref="RmsNormF32Kernel"/>, <see cref="RopeF32Kernel"/>,
/// <see cref="AttentionF32Kernel"/>, <see cref="SwiGluF32Kernel"/>, and
/// <see cref="AddKernel"/> for residuals.
/// </summary>
/// <remarks>
/// <para>
/// Q8_0 weights stay on device as 34-byte blocks and are consumed directly
/// by the Q8_0 matmul kernels — 4× less VRAM and 4× less bytes-per-forward
/// on the weight read vs the legacy dequantise-at-load path. Other quant
/// types (F16, K-quants) still dequantise to FP32 at load. All non-matmul
/// kernels remain F32; only the weight storage changes. The model assumes
/// a pure-Transformer Llama-family architecture — MLA, MoE, and SSM layers
/// are rejected at load time.
/// </para>
/// <para>
/// Forward pass is fence-pipelined: a single persistent command buffer
/// records every kernel dispatch + inter-kernel pipeline barrier for the
/// whole forward, submits once per forward, and waits on a single fence
/// before downloading logits. Legacy synchronous kernel launches (one
/// <c>vkQueueWaitIdle</c> per kernel) are only used by the standalone
/// unit tests.
/// </para>
/// <para>
/// Architectural parallel with <c>DotLLM.Cuda.CudaTransformerModel</c>:
/// upload weights once at construction, reuse a single
/// <see cref="VulkanForwardState"/> for scratch. Each linear projection
/// dispatches through <see cref="RecordMatmul"/> which picks
/// <c>matmul_q8_0</c> / <c>matmul_q8_0_gemm</c> / <c>matmul_f32</c> based on
/// the weight's device-side quant type and <c>seqLen</c>. Logits come back
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
    private readonly MatMulQ8_0Kernel _matmulQ8;
    private readonly MatMulQ8_0GemmKernel _matmulQ8Gemm;
    // Optional: coopmat Q8_0 GEMM for prefill (seqLen>1) on devices that
    // advertise VK_KHR_cooperative_matrix. ~3.8× over the scalar GEMM on AMD
    // RDNA3.5 iGPU at Llama-3 4096² N=64 (790 vs 209 GFLOPS). Null on devices
    // without coopmat — the router falls back to _matmulQ8Gemm then.
    private readonly MatMulQ8_0GemmCoopmatKernel? _matmulQ8GemmCoopmat;
    // Optional decode-path fusion of rmsnorm + Q8_0 GEMV. Eliminates one
    // dispatch + one barrier per attn-norm/Q proj and per ffn-norm/Gate proj
    // (60 dispatches per decode at 30 layers). Null when the SPV is missing
    // or when the model's hidden size exceeds the shader's on-chip cap;
    // router falls back to the standalone (rmsnorm + matmul_q8_0) pair.
    private readonly RmsNormMatmulQ8_0FusedKernel? _rmsnormMatmulQ8Fused;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    private readonly SwiGluF32Kernel _swiglu;
    private readonly AddKernel _add;
    // Per-feature bias add. Replaces the host-mapped fallback that used to
    // split the forward into multiple submits whenever Phi-3 / Qwen3 /
    // DeepSeek-V2 layers carried biases — now the whole forward stays in
    // one submit regardless of bias presence.
    private readonly BiasAddF32Kernel _biasAdd;
    // MLA (DeepSeek-V2/V3) — null when the model carries no MLA layer.
    // The post-projection attention loop (per-head SDPA with Q_nope/Q_pe
    // split + MQA-shared K_pe), the decoupled-rope rotation on Q_pe + K_pe,
    // and the per-head split of kv_b_proj's fused output into K_nope/V.
    private readonly AttentionMlaF32Kernel? _mlaAttention;
    private readonly RopeMlaF32Kernel? _mlaRope;
    private readonly MlaKvSplitF32Kernel? _mlaKvSplit;
    // MLA softmax scale = (YaRN mscale²) / sqrt(qk_head_dim). Folded once at
    // construction since the kernel takes scale as a push constant.
    private readonly float _mlaScale;
    private readonly int _mlaQkNopeHeadDim;
    private readonly int _mlaQkRopeHeadDim;
    private readonly int _mlaVHeadDim;
    private readonly int _mlaNumHeads;
    private readonly float _mlaRopeTheta;
    // MoE (Mixtral / Qwen-MoE) — null when the model carries no MoE layer.
    private readonly MoeTopKSoftmaxF32Kernel? _moeTopkSoftmax;
    private readonly MoeIndexedMatmulF32Kernel? _moeIndexedMatmul;
    // Tiled (shared-memory) variant of the indexed matmul. Wins on prefill at
    // large N (seqLen * topK ≥ 32) by amortising the x-row load across a
    // TILE_M-wide output tile; the scalar variant remains for decode (small N)
    // where the GEMV-style scalar dispatch wins.
    private readonly MoeIndexedMatmulTiledF32Kernel? _moeIndexedMatmulTiled;
    private readonly MoeWeightedScatterF32Kernel? _moeWeightedScatter;
    private readonly MoeBroadcastF32Kernel? _moeBroadcast;
    // Optional Qwen1.5-MoE per-token sigmoid gate fold for the shared-expert
    // branch. Null when no MoE layer exists OR when no MoE layer carries a
    // SharedExpertGate weight (DeepSeek-V2/V3, Mixtral). Allocated alongside
    // the other MoE kernels so the gated path is wired wherever it might
    // fire across the per-layer mix.
    private readonly MoeSigmoidGatedAddF32Kernel? _moeSigmoidGatedAdd;

    // Persistent command buffer + fence used by Forward. One SubmitContext
    // per model — reset+begin at the start of each forward, submit+wait at
    // the end. Bias host-side steps split the forward into multiple submits
    // but each submit still batches many dispatches behind one fence.
    private readonly VulkanDevice.SubmitContext _submit;

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

    /// <summary>
    /// Creates a per-layer MLA (DeepSeek-V2/V3) KV-cache sized for this
    /// model. Throws when the model is not an MLA model (no <c>MlaConfig</c>
    /// at construction).
    /// </summary>
    public MlaVulkanKvCache CreateMlaKvCache(int maxSeqLen)
    {
        if (Config.MlaConfig is null)
            throw new InvalidOperationException(
                "CreateMlaKvCache requires a model with MlaConfig set; this model has none.");
        return new MlaVulkanKvCache(_device, Config.NumLayers, maxSeqLen,
            _mlaNumHeads, _mlaQkNopeHeadDim, _mlaVHeadDim, _mlaQkRopeHeadDim);
    }

    private VulkanTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, VulkanWeights weights, TransformerWeights cpuWeights,
        VulkanForwardState state,
        MatMulF32Kernel matmul, MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        RmsNormMatmulQ8_0FusedKernel? rmsnormMatmulQ8Fused,
        RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, SwiGluF32Kernel swiglu, AddKernel add,
        BiasAddF32Kernel biasAdd,
        AttentionMlaF32Kernel? mlaAttention, RopeMlaF32Kernel? mlaRope, MlaKvSplitF32Kernel? mlaKvSplit,
        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax, MoeIndexedMatmulF32Kernel? moeIndexedMatmul,
        MoeIndexedMatmulTiledF32Kernel? moeIndexedMatmulTiled,
        MoeWeightedScatterF32Kernel? moeWeightedScatter, MoeBroadcastF32Kernel? moeBroadcast,
        MoeSigmoidGatedAddF32Kernel? moeSigmoidGatedAdd,
        VulkanDevice.SubmitContext submit,
        GgufFile? gguf,
        float ropeTheta, int ropeDim, RopeF32Kernel.Variant ropeVariant, int slidingWindow,
        int mlaNumHeads, int mlaQkNopeHeadDim, int mlaQkRopeHeadDim, int mlaVHeadDim,
        float mlaScale, float mlaRopeTheta)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _cpuWeights = cpuWeights;
        _state = state;
        _matmul = matmul;
        _matmulQ8 = matmulQ8;
        _matmulQ8Gemm = matmulQ8Gemm;
        _matmulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        _rmsnormMatmulQ8Fused = rmsnormMatmulQ8Fused;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _swiglu = swiglu;
        _add = add;
        _biasAdd = biasAdd;
        _mlaAttention = mlaAttention;
        _mlaRope = mlaRope;
        _mlaKvSplit = mlaKvSplit;
        _moeTopkSoftmax = moeTopkSoftmax;
        _moeIndexedMatmul = moeIndexedMatmul;
        _moeIndexedMatmulTiled = moeIndexedMatmulTiled;
        _moeWeightedScatter = moeWeightedScatter;
        _moeBroadcast = moeBroadcast;
        _moeSigmoidGatedAdd = moeSigmoidGatedAdd;
        _submit = submit;
        _gguf = gguf;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeVariant = ropeVariant;
        _slidingWindow = slidingWindow;
        _mlaNumHeads = mlaNumHeads;
        _mlaQkNopeHeadDim = mlaQkNopeHeadDim;
        _mlaQkRopeHeadDim = mlaQkRopeHeadDim;
        _mlaVHeadDim = mlaVHeadDim;
        _mlaScale = mlaScale;
        _mlaRopeTheta = mlaRopeTheta;
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
    /// Loads a model from a HuggingFace-convention safetensors source onto a
    /// new Vulkan device. Mirrors <see cref="TransformerModel.LoadFromSafetensors(ISafetensorsTensorSource, ModelConfig)"/>
    /// but produces a Vulkan-backed model. Used by tests and tooling that
    /// build synthetic fixtures (no GGUF roundtrip).
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, string? spvDir = null)
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

    private static VulkanTransformerModel BuildModel(
        VulkanDevice device, bool ownsDevice, ModelConfig config,
        TransformerWeights cpuWeights, string spvDir, GgufFile? gguf)
    {
        // Q8_0 matrices stay on device as 34-byte blocks — the forward pass
        // below dispatches them through the Q8_0 GEMV / GEMM kernels. Other
        // quant types are still dequantised to FP32 at upload.
        var weights = VulkanWeights.Upload(device, cpuWeights, config.NumLayers);

        // MoE detection: any layer with non-null Moe in CPU weights. We
        // don't gate on config.Moe because Mixtral/Qwen-MoE configs may
        // mark "MoE everywhere" while DeepSeek-V2 first_k_dense_replace
        // makes only the tail layers MoE — any-layer check is the
        // conservative trigger.
        bool hasMoe = false;
        int moeNumExperts = 0, moeTopK = 0, moeIntermediate = 0;
        int moeSharedIntermediate = 0, moeNumSharedExperts = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            ref readonly var lwTmp = ref cpuWeights.Layers[i];
            if (lwTmp.Moe is not null)
            {
                hasMoe = true;
                moeNumExperts = Math.Max(moeNumExperts, lwTmp.Moe.NumExperts);
                moeTopK = Math.Max(moeTopK, lwTmp.Moe.NumExpertsPerTok);
                moeIntermediate = Math.Max(moeIntermediate, lwTmp.Moe.IntermediateSize);
                if (lwTmp.Moe.HasSharedExpert)
                {
                    moeSharedIntermediate = Math.Max(moeSharedIntermediate, lwTmp.Moe.SharedIntermediateSize);
                    moeNumSharedExperts = Math.Max(moeNumSharedExperts, lwTmp.Moe.NumSharedExperts);
                }
            }
        }

        bool hasMla = config.MlaConfig is not null;
        int mlaNumHeads = hasMla ? config.NumAttentionHeads : 0;
        int mlaQkNope = hasMla ? config.MlaConfig!.QkNopeHeadDim : 0;
        int mlaQkRope = hasMla ? config.MlaConfig!.QkRopeHeadDim : 0;
        int mlaVHead = hasMla ? config.MlaConfig!.VHeadDim : 0;
        int mlaQLora = hasMla ? config.MlaConfig!.QLoraRank : 0;
        int mlaKvLora = hasMla ? config.MlaConfig!.KvLoraRank : 0;
        float mlaScale = 0f, mlaRopeTheta = 0f;
        if (hasMla)
        {
            int qkHeadDim = mlaQkNope + mlaQkRope;
            float yarnMul = config.MlaConfig!.ComputeYarnSoftmaxScaleMultiplier();
            mlaScale = yarnMul / MathF.Sqrt(qkHeadDim);
            mlaRopeTheta = config.MlaConfig!.RopeTheta;
        }

        var state = new VulkanForwardState(device,
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            initialSeqLen: 1,
            mlaNumHeads: mlaNumHeads,
            mlaQkNopeHeadDim: mlaQkNope,
            mlaQkRopeHeadDim: mlaQkRope,
            mlaVHeadDim: mlaVHead,
            mlaQLoraRank: mlaQLora,
            mlaKvLoraRank: mlaKvLora,
            moeNumExperts: moeNumExperts,
            moeTopK: moeTopK,
            moeIntermediateSize: moeIntermediate,
            moeSharedIntermediateSize: moeSharedIntermediate,
            moeNumSharedExperts: moeNumSharedExperts);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        // Optional coopmat prefill GEMM — 3.8× over scalar on AMD RDNA3.5 at
        // Llama-3 4096² N=64. Null on devices without KHR_cooperative_matrix;
        // router falls back to the scalar GEMM. Tolerance: abs 5e-3 / rel 5e-3
        // end-to-end (looser than the 1e-4 / 1e-3 of the scalar path because
        // KHR_coopmat only offers F16 operands — see the coopmat kernel tests).
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* Kernel threw: no usable tile shape. Stay on scalar. */ }
        }
        // Optional decode-path fusion of rmsnorm + Q8_0 GEMV. Older builds
        // without the fused SPV stay working — TryCreate returns null and
        // the router falls back to the standalone pair.
        RmsNormMatmulQ8_0FusedKernel? rmsnormMatmulQ8Fused =
            RmsNormMatmulQ8_0FusedKernel.TryCreate(device, spvDir);

        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);
        var biasAdd = BiasAddF32Kernel.Create(device, spvDir);

        AttentionMlaF32Kernel? mlaAttention = null;
        RopeMlaF32Kernel? mlaRope = null;
        MlaKvSplitF32Kernel? mlaKvSplit = null;
        if (hasMla)
        {
            mlaAttention = AttentionMlaF32Kernel.Create(device, spvDir);
            mlaRope = RopeMlaF32Kernel.Create(device, spvDir);
            mlaKvSplit = MlaKvSplitF32Kernel.Create(device, spvDir);
        }

        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax = null;
        MoeIndexedMatmulF32Kernel? moeIndexedMatmul = null;
        MoeIndexedMatmulTiledF32Kernel? moeIndexedMatmulTiled = null;
        MoeWeightedScatterF32Kernel? moeWeightedScatter = null;
        MoeBroadcastF32Kernel? moeBroadcast = null;
        MoeSigmoidGatedAddF32Kernel? moeSigmoidGatedAdd = null;
        if (hasMoe)
        {
            moeTopkSoftmax = MoeTopKSoftmaxF32Kernel.Create(device, spvDir);
            moeIndexedMatmul = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
            moeIndexedMatmulTiled = MoeIndexedMatmulTiledF32Kernel.Create(device, spvDir);
            moeWeightedScatter = MoeWeightedScatterF32Kernel.Create(device, spvDir);
            moeBroadcast = MoeBroadcastF32Kernel.Create(device, spvDir);
            moeSigmoidGatedAdd = MoeSigmoidGatedAddF32Kernel.Create(device, spvDir);
        }

        var submit = device.CreateSubmitContext();

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        var ropeVariant = ropeType == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm;

        int slidingWindow = config.SlidingWindowSize ?? 0;

        return new VulkanTransformerModel(
            device, ownsDevice,
            config, weights, cpuWeights, state,
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            rmsnormMatmulQ8Fused,
            rmsnorm, rope, attention, swiglu, add,
            biasAdd,
            mlaAttention, mlaRope, mlaKvSplit,
            moeTopkSoftmax, moeIndexedMatmul, moeIndexedMatmulTiled, moeWeightedScatter, moeBroadcast,
            moeSigmoidGatedAdd,
            submit,
            gguf,
            ropeTheta, ropeDim, ropeVariant, slidingWindow,
            mlaNumHeads, mlaQkNope, mlaQkRope, mlaVHead,
            mlaScale, mlaRopeTheta);
    }

    private static void RejectUnsupportedArchitecture(ModelConfig config)
    {
        if (config.HybridLayout is not null || config.SsmConfig is not null || config.Mamba3Config is not null)
            throw new NotSupportedException("Hybrid SSM / Mamba architectures are not supported on the Vulkan backend yet.");
        // MLA: latent / hybrid cache modes are CPU-only for now; the Vulkan
        // path runs the Phase A expanded cache (MlaVulkanKvCache) which
        // matches the CPU expanded path. Reject the latent flags so callers
        // don't silently get a different attention math.
        if (config.MlaConfig is { UseLatentCache: true } or { UseHybridMlaCache: true })
            throw new NotSupportedException(
                "MLA latent / hybrid KV-cache modes are not supported on the Vulkan backend yet; use the default expanded cache.");
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

        bool scratchResized = _state.EnsureCapacity(seqLen);

        // Descriptor sets cache buffer handles. When scratch is re-allocated
        // every cached set becomes stale and must be dropped — otherwise the
        // next dispatch binds a dangling VkBuffer. In steady-state decode
        // (seqLen = 1 after the initial prefill) scratch never grows, so the
        // cache stays warm across forwards.
        if (scratchResized)
            InvalidateKernelCaches();

        // 1. Validate token IDs (done host-side; cheap), then upload only
        //    positions host→device. The embedding table is device-local and
        //    populated once at construction; per-token rows are gathered into
        //    HiddenState via vkCmdCopyBuffer recorded on the same command
        //    buffer (see RecordEmbeddingGather below).
        ValidateTokenIds(tokenIds);
        UploadPositions(positions);

        // 2. Begin the single per-forward command buffer and record the
        //    whole transformer. Bias-add host steps split the forward into
        //    multiple submits (one per distinct set of biases we need to
        //    pause for); everything else stays inside the pipelined path.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        // Canonicalise the hidden-slot rotation to slot 0 so the embedding
        // gather below writes into the same physical buffer every forward
        // (keeps kernel descriptor-set caches warm across decode steps).
        _state.ResetHiddenSlot();

        // Gather one embedding row per token from the device-local
        // TokenEmbedding buffer into HiddenState[t, :]. The first consumer
        // is the first RMSNorm's COMPUTE read on HiddenState — hidden/residual
        // now alias (no TRANSFER copy in between) so a TRANSFER→COMPUTE
        // barrier is all we need.
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            ref readonly var cpuLw = ref _cpuWeights.Layers[layer];

            // Pre-attention residual snapshot: Residual aliases HiddenState
            // (same physical buffer), so no copy is needed. The barrier from
            // the previous layer's final residual add (or the embedding
            // gather's TRANSFER→COMPUTE on layer 0) has already made the
            // hidden-state writes visible to this rmsnorm.

            if (lw.Mla is { } mlaW)
            {
                // MLA (DeepSeek-V2/V3) attention block — projection ladder +
                // decoupled RoPE + per-head SDPA. Writes the post-o_proj
                // result into _state.NormOutput (mirrors the GQA path's
                // contract so the shared residual-add code below works
                // unchanged).
                RecordMlaLayer(cmdBuf, layer, mlaW, lw, seqLen, eps,
                    positions, kvCache);
            }
            else
            {

            // Attn RMSNorm + Q projection — fused into one dispatch when
            // available (decode + Q8_0 + hidden ≤ shader cap). The fused
            // shader writes BOTH the normalised hidden state (for K/V to
            // read) AND the Q matmul output. Falls back to the standalone
            // pair on prefill, non-Q8_0 weights, or oversized hidden.
            if (!TryRecordFusedRmsNormMatmul(cmdBuf,
                    _state.HiddenState, lw.AttnNormWeight,
                    lw.Q, lw.QDeviceQuantType,
                    _state.NormOutput, _state.Q,
                    lw.QOutputDim, lw.QInputDim, seqLen, eps))
            {
                _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                RecordMatmul(cmdBuf, lw.Q, lw.QDeviceQuantType, _state.NormOutput, _state.Q,
                    lw.QOutputDim, lw.QInputDim, seqLen);
            }
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // K/V projections — read the normalised hidden state written above.
            RecordMatmul(cmdBuf, lw.K, lw.KDeviceQuantType, _state.NormOutput, _state.K,
                lw.KOutputDim, lw.KInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.V, lw.VDeviceQuantType, _state.NormOutput, _state.V,
                lw.VOutputDim, lw.VInputDim, seqLen);

            // Optional QKV biases — kernel path keeps the whole forward in
            // one submit. Each bias add writes a different output buffer
            // (Q / K / V are independent), so no inter-bias barrier needed.
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.QBias is not null) _biasAdd.Record(cmdBuf, _state.Q, lw.QBias, seqLen, lw.QOutputDim);
            if (lw.KBias is not null) _biasAdd.Record(cmdBuf, _state.K, lw.KBias, seqLen, lw.KOutputDim);
            if (lw.VBias is not null) _biasAdd.Record(cmdBuf, _state.V, lw.VBias, seqLen, lw.VOutputDim);
            if (lw.QBias is not null || lw.KBias is not null || lw.VBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // RoPE on Q and K
            _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);

            // Attention input buffers: either the uncached K/V window or the full KV cache.
            VulkanDevice.Buffer kSrc, vSrc;
            int seqKv;
            int positionOffset;
            if (kvCache is VulkanKvCache vkCache)
            {
                // RoPE writes K; attention (via the cache buffers) reads K.
                // Barrier the RoPE → KV copy, then the KV copy → attention.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
                KernelSupport.TransferToComputeBarrier(cmdBuf);
                kSrc = vkCache.GetKeysBuffer(layer);
                vSrc = vkCache.GetValuesBuffer(layer);
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
                positionOffset: positionOffset, slidingWindow: _slidingWindow);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Output projection → NormOutput (reuse slot).
            RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.OBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.OBias, seqLen, lw.OOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }
            }  // end of GQA branch (else of MLA)

            // Residual add #1: AddScratch = Residual + NormOutput. The add
            // reads from HiddenState (which aliases Residual — same slot)
            // and writes to AddScratch (the alternate slot). After the
            // rotate, HiddenState = old AddScratch and AddScratch = old
            // HiddenState — no copies, just a label swap. The single
            // ComputeToComputeBarrier covers the shader_write→shader_read
            // ordering the FFN rmsnorm needs to see the new hidden state.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            _state.RotateHiddenSlot();
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Pre-FFN residual snapshot: Residual aliases HiddenState (same
            // slot); no copy needed.

            if (lw.Moe is { } moeW)
            {
                // MoE FFN replaces the dense Gate/Up/Down with a sparse
                // top-k expert dispatch. Writes the post-MoE result into
                // _state.NormOutput so the shared residual-add below
                // works unchanged.
                RecordMoeLayer(cmdBuf, moeW, lw, seqLen, eps);
            }
            else
            {

            // FFN RMSNorm + Gate projection — fused when available
            // (mirrors the attn-norm + Q fusion above). Up reads the
            // normalised hidden state written by the fused dispatch.
            if (!TryRecordFusedRmsNormMatmul(cmdBuf,
                    _state.HiddenState, lw.FfnNormWeight,
                    lw.Gate, lw.GateDeviceQuantType,
                    _state.NormOutput, _state.FfnGate,
                    lw.GateOutputDim, lw.GateInputDim, seqLen, eps))
            {
                _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                RecordMatmul(cmdBuf, lw.Gate, lw.GateDeviceQuantType, _state.NormOutput, _state.FfnGate,
                    lw.GateOutputDim, lw.GateInputDim, seqLen);
            }
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Up projection — reads the normalised hidden state.
            RecordMatmul(cmdBuf, lw.Up, lw.UpDeviceQuantType, _state.NormOutput, _state.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen);

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.GateBias is not null) _biasAdd.Record(cmdBuf, _state.FfnGate, lw.GateBias, seqLen, lw.GateOutputDim);
            if (lw.UpBias is not null) _biasAdd.Record(cmdBuf, _state.FfnUp, lw.UpBias, seqLen, lw.UpOutputDim);
            if (lw.GateBias is not null || lw.UpBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // SwiGLU
            _swiglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Down projection
            RecordMatmul(cmdBuf, lw.Down, lw.DownDeviceQuantType, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen);

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.DownBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.DownBias, seqLen, lw.DownOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }
            }  // end of dense-FFN branch (else of MoE)

            // Residual add #2: AddScratch = Residual + NormOutput; then rotate
            // the slot so the new hidden state lives in the buffer we just
            // wrote. See residual add #1 comment above for why no copy is
            // needed.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            _state.RotateHiddenSlot();

            // COMPUTE→COMPUTE between layers — next iteration's first op is
            // the attention RMSNorm, which reads the freshly-rotated
            // HiddenState written by the add.
            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // 3. Final RMSNorm on the last token only, then LM head.
        //    The last hidden state was just written by the final layer's
        //    residual add (compute shader). The following single-row copy
        //    runs in TRANSFER, so we need a compute→transfer barrier — a
        //    plain ComputeToComputeBarrier does NOT synchronise transfer
        //    reads against prior compute writes.
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
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: 1);

        // 4. COMPUTE→HOST barrier for the vocab-row download that follows, submit, wait.
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

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

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _matmulQ8.InvalidateDescriptorCache();
        _matmulQ8Gemm.InvalidateDescriptorCache();
        _matmulQ8GemmCoopmat?.InvalidateDescriptorCache();
        _rmsnormMatmulQ8Fused?.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _rope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
        _swiglu.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
        _biasAdd.InvalidateDescriptorCache();
        _mlaAttention?.InvalidateDescriptorCache();
        _mlaRope?.InvalidateDescriptorCache();
        _mlaKvSplit?.InvalidateDescriptorCache();
        _moeTopkSoftmax?.InvalidateDescriptorCache();
        _moeIndexedMatmul?.InvalidateDescriptorCache();
        _moeIndexedMatmulTiled?.InvalidateDescriptorCache();
        _moeWeightedScatter?.InvalidateDescriptorCache();
        _moeBroadcast?.InvalidateDescriptorCache();
        _moeSigmoidGatedAdd?.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Dispatches a matmul for a single linear projection: chooses
    /// <see cref="MatMulQ8_0Kernel"/> (decode-path GEMV) when the device-side
    /// weight is Q8_0 and <paramref name="seqLen"/>==1, the batched
    /// <see cref="MatMulQ8_0GemmKernel"/> when Q8_0 and <paramref name="seqLen"/>&gt;1,
    /// and <see cref="MatMulF32Kernel"/> for every non-Q8_0 weight.
    /// </summary>
    /// <remarks>
    /// All Q8_0 kernels require <paramref name="inputDim"/> to be a multiple
    /// of 32 (the Q8_0 group size). Llama-family projections satisfy this by
    /// construction; the Q8_0 kernels still validate at dispatch so a
    /// surprise non-aligned model fails loud.
    /// </remarks>
    /// <summary>
    /// Attempts to dispatch a fused (rmsnorm → Q8_0 matmul) pair as a single
    /// dispatch with one barrier instead of two. Returns false when the fast
    /// path is unavailable (no fused SPV, non-Q8_0 weight, prefill, or hidden
    /// size beyond the shader's on-chip cap) — the caller must record the
    /// standalone (rmsnorm + matmul) pair as a fallback.
    /// </summary>
    /// <remarks>
    /// On success the fused dispatch:
    ///   1. Computes rmsnorm of <paramref name="hidden"/> with
    ///      <paramref name="normWeight"/>, writing the normalised values to
    ///      <paramref name="normOutput"/> (so downstream non-fused matmuls
    ///      like K, V, Up still see the normalised hidden state).
    ///   2. Computes <c>matmulOutput[m] = sum_k weight[m,k] * normalised[k]</c>
    ///      using on-chip shared memory for the dot product.
    /// Caller is responsible for the post-dispatch barrier — same shape as a
    /// standalone matmul.
    /// </remarks>
    private bool TryRecordFusedRmsNormMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer hidden, VulkanDevice.Buffer normWeight,
        VulkanDevice.Buffer weights, QuantType weightQt,
        VulkanDevice.Buffer normOutput, VulkanDevice.Buffer matmulOutput,
        int outputDim, int inputDim, int seqLen, float eps)
    {
        if (_rmsnormMatmulQ8Fused is null) return false;
        // Opt-out switch — default is fused-on. On RDNA3.5 the fused path
        // wins by ~3-5% in median paired-run min latency and is more
        // resilient to dispatch-time contention. Set the env var to "1"
        // to bypass on hardware where fusion regresses (vendor A/B).
        if (Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_FUSED_RMSNORM_MATMUL") == "1") return false;
        if (seqLen != 1) return false;
        if (weightQt != QuantType.Q8_0) return false;
        if (!RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(inputDim)) return false;

        _rmsnormMatmulQ8Fused.Record(cmdBuf, hidden, normWeight, weights,
            normOutput, matmulOutput,
            m: outputDim, k: inputDim, eps: eps);
        return true;
    }

    /// <summary>
    /// Records the MLA (DeepSeek-V2/V3) attention block for one layer:
    /// rmsnorm → Q path (LoRA-factored or monolithic) → KV path (latent
    /// rmsnorm + kv_b expansion + per-head split) → decoupled-rope on
    /// Q_pe + shared K_pe → optional KV-cache write → per-head SDPA →
    /// o_proj. Writes the post-o_proj result into <c>_state.NormOutput</c>
    /// so the shared residual-add downstream sees the same contract as the
    /// GQA path.
    /// </summary>
    /// <remarks>
    /// All MLA projections are F32 (no Q8_0 path on MLA today; the loader
    /// upcasts F16/BF16 at load). The matmul router still uses
    /// <see cref="RecordMatmul"/> and lands on <c>matmul_f32</c> uniformly.
    /// </remarks>
    private void RecordMlaLayer(
        nint cmdBuf, int layer, VulkanWeights.MlaLayerBuffers mlaW,
        in VulkanWeights.LayerBuffers lw, int seqLen, float eps,
        ReadOnlySpan<int> positions, IKvCache? kvCache)
    {
        int qkHeadDim = mlaW.QkHeadDim;
        int hidden = mlaW.HiddenSize;

        // Pre-attention RMSNorm: HiddenState → NormOutput.
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Q path ────────────────────────────────────────────────────
        // LoRA: NormOutput → MlaQLatent → (rmsnorm with QALayernormWeight)
        //       → MlaQLatentNorm → MlaQ.
        // Monolithic: NormOutput → MlaQ.
        if (mlaW.QLoraRank > 0)
        {
            _matmul.Record(cmdBuf, mlaW.QAProj!, _state.NormOutput, _state.MlaQLatent!,
                m: mlaW.QLoraRank, k: hidden, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _rmsnorm.Record(cmdBuf, _state.MlaQLatent!, mlaW.QALayernormWeight!, _state.MlaQLatentNorm!,
                rowCount: seqLen, n: mlaW.QLoraRank, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, mlaW.QBProj!, _state.MlaQLatentNorm!, _state.MlaQ!,
                m: mlaW.QTotal, k: mlaW.QLoraRank, n: seqLen);
        }
        else
        {
            _matmul.Record(cmdBuf, mlaW.QProj!, _state.NormOutput, _state.MlaQ!,
                m: mlaW.QTotal, k: hidden, n: seqLen);
        }

        // ── KV path (latent + rope-K split) ──────────────────────────
        // Two parallel matmuls off the same NormOutput: first kvLoraRank
        // rows of kv_a_proj_with_mqa → MlaKvLatent (independent), last
        // qkRopeHeadDim rows → MlaKPe (independent — also independent from
        // the Q matmul above). All three writes complete before the
        // single barrier below.
        _matmul.Record(cmdBuf, mlaW.KvALatentProj, _state.NormOutput, _state.MlaKvLatent!,
            m: mlaW.KvLoraRank, k: hidden, n: seqLen);
        _matmul.Record(cmdBuf, mlaW.KvAKPeProj, _state.NormOutput, _state.MlaKPe!,
            m: mlaW.QkRopeHeadDim, k: hidden, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // RMSNorm the latent slice (rope-K is left untouched).
        _rmsnorm.Record(cmdBuf, _state.MlaKvLatent!, mlaW.KvALayernormWeight, _state.MlaKvLatentNorm!,
            rowCount: seqLen, n: mlaW.KvLoraRank, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // kv_b expansion: latent_norm → MlaKvBExpanded
        // Then split per-head into MlaKNope and MlaV.
        _matmul.Record(cmdBuf, mlaW.KvBProj, _state.MlaKvLatentNorm!, _state.MlaKvBExpanded!,
            m: mlaW.KvBOutputDim, k: mlaW.KvLoraRank, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _mlaKvSplit!.Record(cmdBuf, _state.MlaKvBExpanded!, _state.MlaKNope!, _state.MlaV!,
            seqLen: seqLen, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, vHeadDim: mlaW.VHeadDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Decoupled RoPE on Q_pe (per head) and shared K_pe ────────
        _mlaRope!.Record(cmdBuf, _state.MlaQ!, _state.MlaKPe!, _state.PositionsBuffer,
            seqLen: seqLen, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, qkRopeHeadDim: mlaW.QkRopeHeadDim,
            theta: _mlaRopeTheta);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── KV-cache update + attention ──────────────────────────────
        VulkanDevice.Buffer kNopeSrc, vSrc, kPeSrc;
        int seqKv;
        int positionOffset;
        if (kvCache is MlaVulkanKvCache mlaCache)
        {
            // Cache the new K_nope / V / K_pe rows; attention then reads
            // the full cached window.
            mlaCache.RecordUpdate(cmdBuf, _state.MlaKNope!, _state.MlaV!, _state.MlaKPe!,
                positions, seqLen, layer);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kNopeSrc = mlaCache.GetKNopeBuffer(layer);
            vSrc = mlaCache.GetVBuffer(layer);
            kPeSrc = mlaCache.GetKPeBuffer(layer);
            seqKv = mlaCache.CurrentLength;
            positionOffset = positions[0];
        }
        else
        {
            kNopeSrc = _state.MlaKNope!;
            vSrc = _state.MlaV!;
            kPeSrc = _state.MlaKPe!;
            seqKv = seqLen;
            positionOffset = 0;
        }

        _mlaAttention!.Record(cmdBuf, _state.MlaQ!, kNopeSrc, vSrc, kPeSrc, _state.MlaAttnOutput!,
            seqQ: seqLen, seqKv: seqKv, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, qkRopeHeadDim: mlaW.QkRopeHeadDim,
            vHeadDim: mlaW.VHeadDim,
            positionOffset: positionOffset, scale: _mlaScale);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── o_proj → NormOutput (mirrors GQA contract for residual add) ─
        RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.MlaAttnOutput!, _state.NormOutput,
            lw.OOutputDim, lw.OInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        if (lw.OBias is not null)
        {
            _biasAdd.Record(cmdBuf, _state.NormOutput, lw.OBias, seqLen, lw.OOutputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
    }

    /// <summary>
    /// Records the MoE (Mixtral / Qwen-MoE) FFN block for one layer:
    /// rmsnorm → router gate matmul → top-k softmax → broadcast hidden
    /// to per-(token, slot) → indexed gate / up matmuls (W1, W3) →
    /// SwiGLU → indexed down matmul (W2) → weighted scatter back to
    /// per-token output. Writes the post-MoE result into <c>_state.NormOutput</c>
    /// so the shared residual-add downstream sees the same contract as
    /// the dense FFN path.
    /// </summary>
    /// <remarks>
    /// All projection weights are F32 (the loader upcasts F16/BF16 at
    /// load) so every matmul lands on the F32 plain kernel. Per-expert
    /// banks are addressed via <c>moe_indexed_matmul_f32</c> with the
    /// per-row expert index sourced from <c>moe_topk_softmax_f32</c>'s
    /// indices buffer — no host sync between the router and the expert
    /// matmuls.
    /// </remarks>
    private unsafe void RecordMoeLayer(
        nint cmdBuf, VulkanWeights.MoeLayerBuffers moeW,
        in VulkanWeights.LayerBuffers lw, int seqLen, float eps)
    {
        int hidden = moeW.HiddenSize;
        int interm = moeW.IntermediateSize;
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int expandedRows = seqLen * topK;

        // 1. Pre-FFN RMSNorm: HiddenState → NormOutput.
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. Router gate matmul: Gate @ NormOutput → MoeRouterLogits.
        _matmul.Record(cmdBuf, moeW.Gate, _state.NormOutput, _state.MoeRouterLogits!,
            m: numE, k: hidden, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 3. Top-k softmax: writes MoeTopkIndices (int) and MoeTopkWeights.
        _moeTopkSoftmax!.Record(cmdBuf,
            _state.MoeRouterLogits!, _state.MoeTopkIndices!, _state.MoeTopkWeights!,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        // Broadcast (compute) reads NormOutput, writes MoeExpandedInput; the
        // indexed matmul downstream reads MoeExpandedInput plus topk
        // indices/weights. A single compute→compute barrier covers both
        // RMSNorm-output → broadcast-read on NormOutput and topk-write →
        // matmul-read on the indices/weights.
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. Broadcast NormOutput[seqLen, hidden] → MoeExpandedInput[seqLen*topK, hidden].
        //    Each token's row gets replicated topK times so each (t, slot)
        //    shares the same input but consumes a different expert. One
        //    compute dispatch replaces the seqLen × topK loop of
        //    vkCmdCopyBuffer regions the previous implementation issued —
        //    same math, no transfer↔compute stage transition, dispatch
        //    count drops from O(seqLen·topK) to 1 per MoE layer.
        _moeBroadcast!.Record(cmdBuf,
            _state.NormOutput, _state.MoeExpandedInput!,
            seqLen: seqLen, topK: topK, hidden: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 5. Indexed expert matmuls: gate (W1) and up (W3) project the
        //    expanded input through the experts selected by topk indices.
        RecordMoeIndexedMatmul(cmdBuf,
            moeW.W1Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeGateInter!,
            m: interm, k: hidden, n: expandedRows, numExperts: numE);
        RecordMoeIndexedMatmul(cmdBuf,
            moeW.W3Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeUpInter!,
            m: interm, k: hidden, n: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 6. SwiGLU pointwise: silu(gate) * up.
        _swiglu.Record(cmdBuf, _state.MoeGateInter!, _state.MoeUpInter!, _state.MoeSiluInter!,
            n: expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 7. Indexed down matmul (W2): silu_intermediate → MoeDownRows.
        RecordMoeIndexedMatmul(cmdBuf,
            moeW.W2Bank, _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeDownRows!,
            m: hidden, k: interm, n: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 8. Weighted scatter: combine each token's topK expert outputs into
        //    NormOutput, scaled by the routing weights.
        _moeWeightedScatter!.Record(cmdBuf,
            _state.MoeDownRows!, _state.MoeTopkWeights!, _state.NormOutput,
            seqLen: seqLen, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 9. Shared-expert branch (DeepSeek-V2/V3 ungated). Each shared expert
        //    runs a dense SwiGLU MLP on the per-token hidden state and the
        //    outputs are summed into the routed result. Skipped when the
        //    layer has no shared experts (Mixtral / Qwen3-MoE without shared).
        if (moeW.NumSharedExperts > 0)
        {
            RecordMoeSharedExperts(cmdBuf, moeW, lw.FfnNormWeight, seqLen, hidden, eps);
        }
    }

    /// <summary>
    /// Records the shared-expert branch of a DeepSeek-V2/V3-style MoE layer:
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
    /// the unused ping-pong slot and a device-to-device copy lands the result
    /// back in <c>NormOutput</c> so the caller's residual-add contract is
    /// preserved.
    /// </para>
    /// </remarks>
    private unsafe void RecordMoeSharedExperts(
        nint cmdBuf, VulkanWeights.MoeLayerBuffers moeW,
        VulkanDevice.Buffer ffnNormWeight, int seqLen, int hidden, float eps)
    {
        int numShared = moeW.NumSharedExperts;
        int sharedI = moeW.SharedIntermediateSize;
        int hiddenElems = seqLen * hidden;
        int sharedInterElems = seqLen * sharedI;

        // Re-derive the normalised hidden state. NormOutput is occupied by
        // the routed-MoE result; HiddenState still holds the pre-FFN residual.
        var sharedInput = _state.MoeSharedInput!;
        _rmsnorm.Record(cmdBuf, _state.HiddenState, ffnNormWeight, sharedInput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // SumA / SumB ping-pong; activeSum tracks the slot currently holding
        // the running shared-expert sum. Expert 0 writes directly into SumA;
        // subsequent experts compute their down-output into MoeSharedDown and
        // we add (activeSum + MoeSharedDown) → the OTHER side, alternating.
        VulkanDevice.Buffer activeSum = _state.MoeSharedSumA!;

        for (int s = 0; s < numShared; s++)
        {
            // gate / up matmuls share sharedInput; SwiGLU then fuses them.
            _matmul.Record(cmdBuf, moeW.SharedW1![s], sharedInput, _state.MoeSharedGate!,
                m: sharedI, k: hidden, n: seqLen);
            _matmul.Record(cmdBuf, moeW.SharedW3![s], sharedInput, _state.MoeSharedUp!,
                m: sharedI, k: hidden, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _swiglu.Record(cmdBuf, _state.MoeSharedGate!, _state.MoeSharedUp!, _state.MoeSharedSilu!,
                n: sharedInterElems);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            if (s == 0)
            {
                // First expert seeds the running sum directly into SumA.
                _matmul.Record(cmdBuf, moeW.SharedW2![s], _state.MoeSharedSilu!, _state.MoeSharedSumA!,
                    m: hidden, k: sharedI, n: seqLen);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                activeSum = _state.MoeSharedSumA!;
            }
            else
            {
                // Per-expert down output → MoeSharedDown, then ping-pong add
                // into the slot opposite activeSum.
                _matmul.Record(cmdBuf, moeW.SharedW2![s], _state.MoeSharedSilu!, _state.MoeSharedDown!,
                    m: hidden, k: sharedI, n: seqLen);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

                var sumDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
                    ? _state.MoeSharedSumB!
                    : _state.MoeSharedSumA!;
                _add.Record(cmdBuf, activeSum, _state.MoeSharedDown!, sumDst, hiddenElems);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                activeSum = sumDst;
            }
        }

        // Fold the running shared sum into NormOutput. Two paths:
        //  - DeepSeek-V2/V3 (no gate): plain add via a ping-pong destination,
        //    then a vkCmdCopyBuffer back into NormOutput (the existing add
        //    kernel cannot self-write).
        //  - Qwen1.5-MoE (sigmoid gate): compute per-token gate logits via a
        //    1×hidden matmul against SharedExpertGate, then apply sigmoid +
        //    weighted-add into NormOutput in place via the fused kernel.
        //    No ping-pong / copy needed — the gated kernel writes NormOutput
        //    directly, with sigmoid(logit_t) folded into the per-token scale.
        if (moeW.SharedExpertGate is not null)
        {
            // gateLogits[t] = SharedExpertGate[1, hidden] @ MoeSharedInput[t, :].
            // The post-FFN-RMSNorm hidden state is the right input here — it
            // mirrors MoeSwiGluMlp.ExecuteCoreGrouped which receives the
            // already-RMSNormed hidden as `hidden` and computes the gate
            // logit against that same buffer.
            _matmul.Record(cmdBuf, moeW.SharedExpertGate, sharedInput, _state.MoeSharedGateLogits!,
                m: 1, k: hidden, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _moeSigmoidGatedAdd!.Record(cmdBuf,
                output: _state.NormOutput, b: activeSum, gateLogits: _state.MoeSharedGateLogits!,
                seqLen: seqLen, hiddenSize: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
            VulkanDevice.Buffer foldDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
                ? _state.MoeSharedSumB!
                : _state.MoeSharedSumA!;
            _add.Record(cmdBuf, _state.NormOutput, activeSum, foldDst, hiddenElems);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            var foldRegion = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = 0,
                size = (ulong)hiddenElems * sizeof(float),
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, foldDst.Handle, _state.NormOutput.Handle, 1, foldRegion);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
    }

    /// <summary>
    /// Routes between the scalar and tiled (shared-memory) variants of the
    /// MoE indexed expert matmul based on dispatch shape. Tiled wins on
    /// prefill at large N (each token's x-row is reloaded TILE_M times by
    /// the scalar variant — the tile amortises that), scalar wins on decode
    /// where N is tiny and a TILE_M-wide cooperative load is mostly idle.
    /// </summary>
    /// <remarks>
    /// Heuristic: use the tiled kernel when <c>n ≥ TiledMinRows</c> AND
    /// <c>m % TILE_M == 0</c>. The first guard keeps decode (N ≤ 16, e.g.
    /// 1 token × topK=8 = 8 expanded rows) on the scalar fast path. The
    /// second avoids the shader's tail-bounds path on prefill — the tile
    /// kernel handles ragged m correctly via its in-shader bounds checks,
    /// but on a divisible m we get the cleanest dispatch shape with no
    /// branching in the inner loop. The threshold is conservative; a
    /// future perf-wave should re-tune from device benchmarks.
    /// </remarks>
    private const int TiledMinRows = 32;

    private void RecordMoeIndexedMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        bool useTiled = _moeIndexedMatmulTiled is not null
            && n >= TiledMinRows
            && (m % MoeIndexedMatmulTiledF32Kernel.TileM) == 0;

        if (useTiled)
        {
            _moeIndexedMatmulTiled!.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
        }
        else
        {
            _moeIndexedMatmul!.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
        }
    }

    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (weightQt == QuantType.Q8_0)
        {
            if (seqLen == 1)
            {
                _matmulQ8.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else if (_matmulQ8GemmCoopmat is not null)
            {
                // Prefill path on coopmat-capable devices — ~3.8× over scalar
                // at Llama-3 prefill shapes. See MatMulQ8_0GemmCoopmatKernel.
                _matmulQ8GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ8Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else
        {
            _matmul.Record(cmdBuf, weights, input, output,
                outputDim, inputDim, seqLen);
        }
    }

    /// <summary>
    /// Records a single-region device-to-device <c>vkCmdCopyBuffer</c>. The
    /// residual-shuffle copies that used to run here were eliminated via
    /// hidden-slot rotation (<see cref="VulkanForwardState.RotateHiddenSlot"/>).
    /// Remaining callers: the last-row extraction before the LM-head RMSNorm
    /// (offset copy of one row), the embedding gather, and the KV-cache
    /// update — none of which can be turned into a label swap.
    /// </summary>
    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    /// <summary>
    /// Validates every token id is in range <c>[0, vocabSize)</c>. Separated
    /// from <see cref="RecordEmbeddingGather"/> so the check happens before
    /// we begin recording the command buffer — a bad id throws cleanly
    /// without leaving the submit context half-written.
    /// </summary>
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

    /// <summary>
    /// Records N device-local <c>vkCmdCopyBuffer</c> calls (one per input
    /// token) that gather per-token rows from the already-resident
    /// <see cref="VulkanWeights.TokenEmbedding"/> buffer into
    /// <see cref="VulkanForwardState.HiddenState"/>. The embedding table was
    /// dequantised to F32 and uploaded to device-local VRAM at construction
    /// time (see <see cref="VulkanWeights.Upload"/>), so the only
    /// per-forward cost here is <c>seqLen</c> cheap on-device copy commands
    /// — no host-mapped write, no host→device transfer bandwidth.
    /// </summary>
    /// <remarks>
    /// Vulkan's <c>vkCmdCopyBuffer</c> does accept a regions array, but the
    /// current P/Invoke surface takes a single region (matching the
    /// KV-cache-update path in <see cref="VulkanKvCache.RecordUpdate"/>).
    /// For <c>seqLen=1</c> decode this is one call; for prefill it's
    /// <c>promptLen</c> calls, still dwarfed by the per-layer matmul cost.
    /// </remarks>
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
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();

        _moeSigmoidGatedAdd?.Dispose();
        _moeBroadcast?.Dispose();
        _moeWeightedScatter?.Dispose();
        _moeIndexedMatmulTiled?.Dispose();
        _moeIndexedMatmul?.Dispose();
        _moeTopkSoftmax?.Dispose();
        _mlaKvSplit?.Dispose();
        _mlaRope?.Dispose();
        _mlaAttention?.Dispose();
        _biasAdd.Dispose();
        _add.Dispose();
        _swiglu.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _rmsnormMatmulQ8Fused?.Dispose();
        _matmulQ8GemmCoopmat?.Dispose();
        _matmulQ8Gemm.Dispose();
        _matmulQ8.Dispose();
        _matmul.Dispose();

        _cpuWeights.Dispose();
        if (_ownsDevice)
            _device.Dispose();
    }
}
