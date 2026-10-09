using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
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
/// Resident mode auto-detects uniformly Q6_K or Q4_K source banks at upload
/// time and keeps them on device as raw quant blocks (≈25 GB Q6_K, or
/// smaller still for Q4_K, at qwen35moe-A3B scale on a 128 GB Strix Halo
/// unified-memory host — fits) dispatching through
/// <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ6_KF32Kernel"/> or
/// <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ4_KF32Kernel"/>
/// respectively. Other source quants (or mixed-quant layers) fall back to
/// F32 dequant + upload — fits when the model is small enough that ≈4×
/// expansion stays under device-memory bounds; at Qwen3.6-35B-A3B scale
/// (256 experts × 40 layers × 3 matrices × ~1M elems) the fully-F32
/// resident layout would consume ~120 GB and would NOT fit, so the
/// Q6_K/Q4_K-resident paths are the only resident options at that scale
/// (Q4_K is what the cached UD-Q4_K_XL checkpoint actually uses).
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
public sealed partial class VulkanQwen3MoeHybridTransformerModel : IModel
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
    /// <summary>#849: [numExperts] of 1.0f - the Q5_1 indexed kernels take a per-expert output scale (Gemma-4 ffn_down_exps.scale); routed
    /// banks here have none, so they get the identity. Null for synthetic fixtures without an MoE config.</summary>
    private readonly VulkanDevice.Buffer? _moeUnitScale;

    /// <summary>Which routed-MoE fast paths were RECORDED (test hook: proves a branch was taken, not merely that a kernel exists).</summary>
    internal enum MoePath { GroupedPrefill, GroupedLegacyDown, MmvqDown, MmvqLegacyDown, FusedDecode }
    internal readonly long[] MoePathCounts = new long[Enum.GetValues<MoePath>().Length];

    /// <summary>Record-time counters for the 2..8-row fast paths (#876): prove a fast path ran by counting, not by IsSupported.</summary>
    internal enum SmallRowPath { F32Multi, F16Multi, Q8Multi, MoeQ4KMr, MoeQ5_1Mr }
    internal readonly long[] SmallRowPathCounts = new long[Enum.GetValues<SmallRowPath>().Length];
    private void CountSmallRow(SmallRowPath p) => SmallRowPathCounts[(int)p]++;
    private void CountMoePath(MoePath p) => MoePathCounts[(int)p]++;
    private static bool IsLegacyQuant(QuantizationType qt) => qt is QuantizationType.Q5_1 or QuantizationType.Q8_0;

    // #383: opt-in dp4a indexed-matmul MMQ for Q4_K-resident gate/up banks —
    // see the constructor assignment for rollout rationale.
    private readonly bool _moeIndexedMmqEnabled;

    // CPU/GPU per-layer MoE expert placement (#370, llama.cpp `--n-cpu-moe`
    // shorthand equivalent): _cpuMoeLayer[i] == true means layer i's routed
    // expert compute runs entirely on the CPU (Cpu/Kernels/MoeSwiGluMlp.cs)
    // against the raw GGUF quant view — no GPU bank is EVER uploaded for
    // that layer, so device memory is actually reduced, not just deferred.
    // Dense/attention weights stay GPU-resident regardless (repo-wide
    // "device placement always explicit" rule — CLAUDE.md). Layer selection
    // is uniform-per-layer (v1 simplification the issue accepts): the first
    // <see cref="NCpuMoeLayers"/> layers by index, matching llama.cpp's
    // "-ncmoe N: keep the MoE weights of the first N layers in the CPU".
    private readonly bool[] _cpuMoeLayer;

    /// <summary>Number of layers (from layer 0) whose routed MoE experts are CPU-placed.</summary>
    public int NCpuMoeLayers { get; }

    /// <summary>
    /// Rough device-memory bytes NOT allocated because of CPU-placed layers —
    /// each CPU-placed layer's three routed banks (W1/W2/W3) at the
    /// streaming-F32 sizing (the default, non-resident upload cost every
    /// forward would otherwise pay). Q6_K-resident mode would save
    /// proportionally less (~25% of this estimate) per layer; this property
    /// reports the F32-streaming baseline since that's the default policy
    /// CPU-offload is most valuable against.
    /// </summary>
    public long EstimatedCpuOffloadVramSavedBytes { get; }

    /// <summary>
    /// Resolves the CPU-MoE-offload layer count: an explicit
    /// <paramref name="nCpuMoeLayers"/> &gt;= 0 wins; otherwise falls back to
    /// the <c>DOTLLM_N_CPU_MOE</c> environment variable (default 0 — no
    /// offload, fully GPU-resident/streaming, matching pre-#370 behaviour).
    /// </summary>
    private static int ResolveNCpuMoeLayers(int nCpuMoeLayers)
    {
        if (nCpuMoeLayers >= 0) return nCpuMoeLayers;
        string? raw = Environment.GetEnvironmentVariable("DOTLLM_N_CPU_MOE");
        return int.TryParse(raw, out int n) && n > 0 ? n : 0;
    }

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
    /// model's GDN-layer count. The scheduler / multi-seq dispatcher should
    /// allocate one of these per active sequence and pass it via
    /// <see cref="SequenceForwardRequest.GdnState"/>; without that, multi-seq
    /// dispatch leaks recurrent state across sequences.
    /// </summary>
    public VulkanGdnStateCache CreateGdnStateCache()
        => new(_device, _gdn, _gdnCache.NumGdnLayers);

    /// <summary>
    /// Resident (device-local, packed-quant) MoE banks versus per-layer transient upload. The transient path measured 0.09 tok/s
    /// decode on Qwen3.6-35B-A3B Q4_K_M (3.25 tok/s prefill) against 13-18 / 23-75 tok/s resident (#635), so resident is the default
    /// whenever the GGUF payload (+15% headroom for KV, scratch and bank re-packing) fits the device's resident capacity (device-local heap, plus the GTT heaps on a UMA iGPU - #812: on the 512 MB BIOS split the 68 GiB device-local heap alone made 122B fall to the 0.01 tok/s transient path). <c>DOTLLM_VK_MOE_RESIDENT</c>
    /// =1 forces it on, =0 forces it off. Synthetic fixtures (no GGUF) keep the transient path.
    /// </summary>
    private static bool ResolveResidentMoe(VulkanDevice device, GgufFile? gguf)
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT");
        if (env == "1") return true;
        if (env == "0" || gguf is null) return false;
        return gguf.DataSectionLength * 1.15 < device.ResidentCapacityBytes();
    }

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
        int ropeDim, float ropeTheta,
        int nCpuMoeLayers)
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

        _residentMoeEnabled = ResolveResidentMoe(device, gguf);
        if (config.Moe is { } moeCfg)
        {
            _moeUnitScale = device.AllocateDeviceLocal((long)moeCfg.NumExperts * sizeof(float));
            float[] ones = new float[moeCfg.NumExperts];
            Array.Fill(ones, 1f);
            device.Upload(ones, _moeUnitScale);
        }
        _residentMoeBundles = new VulkanQwen3MoeMoeUpload.LayerBundle?[cpuLayers.Length];
        // #383/#633: dp4a indexed-matmul MMQ for Q4_K-resident gate/up (+ Q5_K down) banks. Default-on after real-model validation
        // (Qwen3.6-35B-A3B Q4_K_M, Strix Halo: pp128 23 -> 75, tg 13.3 -> 18.1 tok/s, PPL within noise); DOTLLM_VK_MOE_INDEXED_MMQ=0 opts out.
        _moeIndexedMmqEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_INDEXED_MMQ"), "0", StringComparison.Ordinal);   // default-on (#633); =0 opts out

        int n = Math.Clamp(nCpuMoeLayers, 0, cpuLayers.Length);
        NCpuMoeLayers = n;
        _cpuMoeLayer = new bool[cpuLayers.Length];
        long savedBytes = 0;
        int hiddenSize = config.HiddenSize;
        for (int i = 0; i < n; i++)
        {
            _cpuMoeLayer[i] = true;
            var m = cpuLayers[i].Moe;
            long w1Elems = (long)m.IntermediateSize * hiddenSize;
            long w2Elems = (long)hiddenSize * m.IntermediateSize;
            long perExpertBytes = (2 * w1Elems + w2Elems) * sizeof(float);
            savedBytes += perExpertBytes * m.NumExperts;
        }
        EstimatedCpuOffloadVramSavedBytes = savedBytes;
    }

    /// <summary>
    /// Loads the Qwen3MoeHybrid model from a GGUF file onto a Vulkan device.
    /// Token-mixing weights upload immediately; MoE routed experts stay on
    /// the host (as raw quant views inside <see cref="Qwen3MoeLayerWeights.Moe"/>)
    /// and are streamed to the GPU on demand per layer.
    /// </summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="gguf">Source GGUF file.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="spvDir">Directory containing compiled SPIR-V shaders.</param>
    /// <param name="nCpuMoeLayers">
    /// CPU/GPU expert placement (#370, llama.cpp <c>--n-cpu-moe</c> shorthand
    /// equivalent): the first <paramref name="nCpuMoeLayers"/> layers (by
    /// index) run their routed MoE expert compute on the CPU instead of
    /// uploading a GPU bank — trading decode/prefill throughput for reduced
    /// device memory. Negative (default) falls back to the
    /// <c>DOTLLM_N_CPU_MOE</c> environment variable (0 when unset — no
    /// offload, identical to pre-#370 behaviour). Clamped to
    /// <c>[0, config.NumLayers]</c>.
    /// </param>
    public static VulkanQwen3MoeHybridTransformerModel BuildFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir,
        int nCpuMoeLayers = -1)
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

        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir, config.HeadDim);

        var model = new VulkanQwen3MoeHybridTransformerModel(
            device, ownsDevice: false,
            config, gguf, cpuModel, cpuLayers, weights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, gdnLayerOrdinal,
            ropeDim, ropeTheta, ResolveNCpuMoeLayers(nCpuMoeLayers));
        model._iqF16Prefill = CreateIqF16Prefill(device, spvDir, config, kernels);
        return model;
    }

    /// <summary>
    /// Builds a Vulkan Qwen3MoeHybrid model from caller-owned, pre-built
    /// <see cref="Qwen3MoeLayerWeights"/> — used by synthetic-fixture parity tests
    /// that bypass the GGUF loader. The caller retains ownership of every
    /// unmanaged pointer (token embed, output, plus every projection inside
    /// <paramref name="cpuLayers"/>, including routed MoE expert banks).
    /// </summary>
    /// <remarks>
    /// Mirrors the signature of <see cref="VulkanNemotronHTransformerModel"/>'s
    /// <c>BuildFromPrebuiltWeights</c>. The constructed Vulkan model holds
    /// <see langword="null"/> for both the <c>gguf</c> and <c>cpuModel</c> slots —
    /// disposal frees only device-side weights, forward scratch, the GDN state
    /// cache, and kernels; weight memory belongs to the caller.
    /// </remarks>
    internal static VulkanQwen3MoeHybridTransformerModel BuildFromPrebuiltWeights(
        VulkanDevice device,
        ModelConfig config,
        Qwen3MoeLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputM, int outputK,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        string spvDir,
        int nCpuMoeLayers = -1)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);
        ArgumentNullException.ThrowIfNull(spvDir);

        // Qwen4Exp (#818) reuses this class as its GDN / full-attention / MoE building-block set (VulkanQwen4ExpTransformerModel).
        if (config.Architecture is not (Architecture.Qwen3MoeHybrid or Architecture.Qwen4Exp))
            throw new ArgumentException(
                $"VulkanQwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));
        if (cpuLayers.Length != config.NumLayers)
            throw new ArgumentException(
                $"cpuLayers length {cpuLayers.Length} != config.NumLayers {config.NumLayers}.", nameof(cpuLayers));

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
                throw new ArgumentException(
                    $"Qwen3MoeHybrid rope_dim={ropeDim} must be even for pair-wise rotation.", nameof(config));
            if (ropeDim > config.HeadDim)
                throw new ArgumentException(
                    $"Qwen3MoeHybrid rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.", nameof(config));
        }

        // Upload token-mixing weights (norm, GDN per-layer, full-attn per-layer,
        // token embedding, output norm, LM head). Routed MoE banks stay on host
        // inside cpuLayers[*].Moe and stream per layer in the forward pass — same
        // policy as BuildFromGguf.
        var weights = VulkanQwen3MoeHybridWeights.Upload(device, config, cpuLayers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQt, outputWeight, outputQt, outputM, outputK);

        var state = new VulkanQwen3MoeHybridForwardState(device, config, gdn, initialSeqLen: 1);
        var gdnCache = new VulkanGdnStateCache(device, gdn, gdnOrdinal);

        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir, config.HeadDim);

        var model = new VulkanQwen3MoeHybridTransformerModel(
            device, ownsDevice: false,
            config, gguf: null, cpuModel: null, cpuLayers, weights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, gdnLayerOrdinal,
            ropeDim, ropeTheta, ResolveNCpuMoeLayers(nCpuMoeLayers));
        model._iqF16Prefill = CreateIqF16Prefill(device, spvDir, config, kernels);
        return model;
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

    // Coarse env-gated prefill profiler (DOTLLM_VULKAN_MOE_PREFILL_PROFILE=1), mirroring
    // VulkanTransformerModel's per-category profiler but at per-layer-submission
    // granularity (2a token-mixing vs 2b MoE FFN) since this model already issues one
    // SubmitAndWait per phase per layer -- no extra mid-command-buffer splits needed.
    private static readonly bool MoePrefillProfileEnabled =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MOE_PREFILL_PROFILE") == "1";
    // Per-STAGE split-submit timing inside the routed-MoE prefill layer (DOTLLM_VULKAN_MOE_STAGE_PROFILE=1): after each stage the
    // command buffer is submitted and waited, so the stage's wall time is attributed to it (sync cost ~50 us per stage is included).
    // Diagnostic only; the totals are printed next to the coarse profile.
    private static bool MoeStageProfileEnabled =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MOE_STAGE_PROFILE") == "1";
    /// <summary>Runtime switch for the split-submit stage timing (#876 probe; also on at startup via the env var). When on, token-mixing stages are timed at 1 row too.</summary>
    internal static bool StageProfileEnabled { get => MoeStageProfileEnabled; set => MoeStageProfileEnabled = value; }
    /// <summary>Accumulated per-stage wall ms since the last <see cref="TakeStageTimes"/>.</summary>
    internal Dictionary<string, double> TakeStageTimes() { var d = new Dictionary<string, double>(_moeStageMs); _moeStageMs.Clear(); return d; }
    internal void Q4Stage(string name) => MoeStage(name);
    internal void Q4StageBegin() => MoeStageBegin();
    private readonly Dictionary<string, double> _moeStageMs = new();
    private long _moeStageLast;

    private void MoeStageBegin()
    {
        if (!MoeStageProfileEnabled) return;
        _moeStageLast = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    /// <summary>Token-mixing stage marker: only meaningful on the per-layer-submit prefill path (never inside the fused decode buffer).</summary>
    private void TmStage(string name, int seqLen)
    {
        if (MoeStageProfileEnabled) MoeStage(name);
    }

    private void MoeStage(string name)
    {
        if (!MoeStageProfileEnabled) return;
        KernelSupport.ComputeToHostBarrier(_submit.CommandBuffer);
        _submit.SubmitAndWait();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        double ms = (now - _moeStageLast) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _moeStageMs[name] = _moeStageMs.GetValueOrDefault(name) + ms;
        _submit.Begin();
        KernelSupport.HostToComputeBarrier(_submit.CommandBuffer);
        _moeStageLast = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private double _profAttnMs;
    private double _profMoeMs;
    private double _profEmbedMs;
    private double _profHeadMs;

    private static double ProfElapsedMs(ref long lastTicks)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        double ms = (now - lastTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        lastTicks = now;
        return ms;
    }

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
    /// model-owned default cache — safe only for single-sequence dispatch from a
    /// freshly-constructed model. Multi-seq batched dispatch via
    /// <see cref="ForwardBatch"/> supplies a fresh per-seq
    /// <see cref="VulkanGdnStateCache"/> for each request to keep recurrent state
    /// isolated.
    /// </summary>
    /// <param name="tokenIds">Input token IDs.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for the returned tensor.</param>
    /// <param name="kvCache">Optional per-seq KV-cache for the GQA layers.</param>
    /// <param name="gdnState">
    /// Optional per-seq GDN recurrent state container. Must be a
    /// <see cref="VulkanGdnStateCache"/> sized for this model's GDN-layer count.
    /// </param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, IGdnState? gdnState)
    {
        // Resolve per-seq GDN state container; model-owned _gdnCache is the
        // backwards-compat fallback for single-seq Forward callers.
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
                $"VulkanQwen3MoeHybridTransformerModel requires a VulkanGdnStateCache; got {gdnState.GetType().Name}.",
                nameof(gdnState));
        }

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
        if (resized) { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }

        UploadPositions(positions);

        var kinds = _layout.LayerKind;
        bool profActive = MoePrefillProfileEnabled && seqLen > 1;
        if (profActive) { _profAttnMs = _profMoeMs = _profEmbedMs = _profHeadMs = 0; }
        long profT0 = profActive ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // Single-token decode with resident fast-path banks in every layer: the whole forward is ONE command buffer (see ForwardDecodeFused).
        if (seqLen == 1 && FuseDecodeEnabled && CanFuseMoeDecode(hiddenSize))
            return ForwardDecodeFused(tokenIds, positions, gdnCache, kvCache, kinds, hiddenSize, vocabSize, numHeads, numKvHeads, headDim, eps);

        // ── 1. Token embedding (single submission) ────────────────────────────
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _submit.SubmitAndWait();
        if (profActive) { _profEmbedMs += ProfElapsedMs(ref profT0); }

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
            MoeStageBegin();
            KernelSupport.HostToComputeBarrier(cmdBuf);
            RecordMoeHybridTokenMixing(cmdBuf, layer, layerBuf, kinds, seqLen, hiddenSize, eps, positions, gdnCache, kvCache, numHeads, numKvHeads, headDim);
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();
            if (profActive) { _profAttnMs += ProfElapsedMs(ref profT0); }

            // ── 2b. MoE submission. Per-layer CPU/GPU expert placement (#370,
            //        `DOTLLM_N_CPU_MOE` / explicit nCpuMoeLayers): layers
            //        0..N-1 route their routed-expert compute entirely
            //        through the CPU (never uploading a GPU bank for that
            //        layer — VRAM saved, not just deferred); remaining
            //        layers keep the existing resident/streaming GPU path.
            if (_cpuMoeLayer[layer])
            {
                RunCpuPlacedMoeLayer(layer, layerBuf, seqLen, hiddenSize, eps);
            }
            else
            {
                RunGpuPlacedMoeLayer(layer, lw, layerBuf, seqLen, hiddenSize, eps);
            }
            if (profActive) { _profMoeMs += ProfElapsedMs(ref profT0); }
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
        if (profActive)
        {
            _profHeadMs += ProfElapsedMs(ref profT0);
            double total = _profEmbedMs + _profAttnMs + _profMoeMs + _profHeadMs;
            Console.Error.WriteLine(
                $"[moe-prefill-profile] seqLen={seqLen} layers={_cpuLayers.Length} total_ms={total:F1}  " +
                $"embed={_profEmbedMs:F1}ms({_profEmbedMs / total * 100:F1}%)  " +
                $"attn(2a)={_profAttnMs:F1}ms({_profAttnMs / total * 100:F1}%)  " +
                $"moe(2b)={_profMoeMs:F1}ms({_profMoeMs / total * 100:F1}%)  " +
                $"head={_profHeadMs:F1}ms({_profHeadMs / total * 100:F1}%)");
            if (MoeStageProfileEnabled && _moeStageMs.Count > 0)
            {
                Console.Error.WriteLine("[moe-stage-profile] " + string.Join("  ",
                    _moeStageMs.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value:F1}ms")));
                _moeStageMs.Clear();
            }
        }

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

    /// <inheritdoc/>
    /// <remarks>
    /// Re-zeroes the model-owned Gated-DeltaNet state cache used by every forward that does not carry a caller-supplied
    /// per-sequence state container. Callers that treat each forward as an independent sequence
    /// (perplexity windows) must call this between sequences — see issue #261.
    /// </remarks>
    public void ResetSequenceState() => _gdnCache.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => CreateGdnStateCache();

    /// <summary>
    /// Phase 5f mirror — Qwen3MoeHybrid <c>ForwardBatch</c> override. Loops per-seq
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, IGdnState?)"/>
    /// using each request's caller-supplied <see cref="SequenceForwardRequest.GdnState"/>
    /// so multi-sequence dispatch keeps GDN recurrent state isolated across sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why no per-layer batched fusion (yet).</b> Every Qwen3MoeHybrid layer is
    /// either a GDN layer (per-token recurrent associative-memory scan that must
    /// thread one sequence's state through tokens in order) or a full-attention
    /// layer wrapped in MoE FFN. The GDN scan cannot share a dispatch across
    /// sequences. MoE routing produces a per-token expert mask that varies across
    /// tokens / sequences, so the indexed-expert matmuls cannot batch at the
    /// <c>seqLen = Σ N_i</c> granularity the dense host uses. The surrounding
    /// matmuls (RMSNorm, projections in / out of the scan) <i>could</i> in
    /// principle batch across sequences within a single seqLen=Σ row stack, but
    /// the win is bounded by Amdahl on the per-seq scan + MoE dispatch — left as
    /// a future workstream (DOTLLM-NN). lm_head fan-out is the same story: it
    /// could batch across simple seqs, but those seqs still need their own GDN
    /// state through every preceding layer, so the per-seq Forward loop already
    /// dispatches lm_head N times anyway.
    /// </para>
    /// <para>
    /// <b>Falling back to model-owned state.</b> Single-seq requests with a null
    /// <see cref="SequenceForwardRequest.GdnState"/> delegate to the per-seq Forward
    /// overload that pre-dates the per-seq state plumbing — the model-owned
    /// <see cref="VulkanGdnStateCache"/> is used as the fallback. This keeps the
    /// shape compatible with the existing single-seq Forward callers and tests.
    /// Multi-seq requests with a null GdnState would silently share the model-owned
    /// cache (corrupting state) — those throw with a clear diagnostic to surface
    /// the misuse loudly.
    /// </para>
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        // Qwen3MoeHybrid has no LoRA path today — reject adapter-bearing requests up front.
        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "VulkanQwen3MoeHybridTransformerModel.ForwardBatch does not support LoRA " +
                    "adapters (no Qwen3MoeHybrid LoRA path today). Re-issue the request without " +
                    "an adapter.");
        }

        // Multi-seq dispatch without per-seq GDN state would silently corrupt the
        // model-owned recurrent state across sequences. Detect this misuse before
        // running anything and emit a diagnostic that names the missing slot.
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
        int seqLen, float eps, VulkanGdnStateCache gdnCache, GdnPostScanGateF32Kernel? postScanGateOverride = null)
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
        bool gdnXq = (seqLen == 1 || SmallRowQ8(seqLen)) && gdnW.QkvDeviceQuantType == QuantizationType.Q8_0 && gdnW.GateDeviceQuantType == QuantizationType.Q8_0
            && gdnW.QkvInputDim == gdnW.GateInputDim && TryPrepareQ8Activations(cmdBuf, _state.NormOutput, gdnW.QkvInputDim, seqLen);
        RecordMatmul(cmdBuf, gdnW.QkvWeight, gdnW.QkvDeviceQuantType,
            _state.NormOutput, _state.GdnQkvBuf,
            outputDim: gdnW.QkvOutputDim, inputDim: gdnW.QkvInputDim, seqLen: seqLen, xqReady: gdnXq);
        RecordMatmul(cmdBuf, gdnW.GateWeight, gdnW.GateDeviceQuantType,
            _state.NormOutput, _state.GdnZBuf,
            outputDim: gdnW.GateOutputDim, inputDim: gdnW.GateInputDim, seqLen: seqLen, xqReady: gdnXq);
        RecordMatmul(cmdBuf, gdnW.AlphaWeight, gdnW.AlphaDeviceQuantType,
            _state.NormOutput, _state.GdnAlphaBuf,
            outputDim: gdnW.AlphaOutputDim, inputDim: gdnW.AlphaInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.BetaWeight, gdnW.BetaDeviceQuantType,
            _state.NormOutput, _state.GdnBetaBuf,
            outputDim: gdnW.BetaOutputDim, inputDim: gdnW.BetaInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("gdn_proj", seqLen);

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
        TmStage("gdn_decay", seqLen);

        // ── 3. Conv1d + SiLU ────────────────────────────────────────────────────
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
        long convDimBytes = (long)convDim * sizeof(float);
        // Issue #695: for real prefills (>= 8 rows) one fused pass reads the conv state and the qkv rows directly (no [state | qkv]
        // concatenation copy), applies SiLU, and writes GdnConvInput; the new conv state is the last (dConv-1) qkv rows. Bit-identical to the
        // copy + conv + SiLU chain below, which short forwards (decode, verify) and DOTLLM_VK_GDN_CONV_FUSED=0 keep.
        var convOut = _state.GdnQkvBuf;
        if (_kernels.GdnConvSilu is { } fusedConv && seqLen >= 8 && dConv >= 2 && dConv <= GdnConvSiluF32Kernel.MaxConvWidth)
        {
            fusedConv.Record(cmdBuf, convStateBuf, _state.GdnQkvBuf, gdnW.Conv1dWeight, gdnW.Conv1dBias, _state.GdnConvInput,
                dConv: dConv, channels: convDim, seqLen: seqLen);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, convStateBuf,
                srcOffset: (ulong)((long)(seqLen - (dConv - 1)) * convDimBytes), dstOffset: 0, size: (ulong)convStateBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            convOut = _state.GdnConvInput;
            TmStage("gdn_conv_fused", seqLen);
        }
        else
        {
            // ConvInput = [convState (DConv-1 rows) | qkvBuf (seqLen rows)]
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            if (convStateBytes > 0)
            {
                RecordCopyBufferRange(cmdBuf, convStateBuf, _state.GdnConvInput,
                    srcOffset: 0, dstOffset: 0, size: (ulong)convStateBytes);
            }
            // The qkv rows are contiguous in both buffers, so the whole [seqLen, convDim] block is one copy (was seqLen copies).
            RecordCopyBufferRange(cmdBuf, _state.GdnQkvBuf, _state.GdnConvInput,
                srcOffset: 0, dstOffset: (ulong)((long)(dConv - 1) * convDimBytes), size: (ulong)((long)seqLen * convDimBytes));
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            _kernels.Conv1dCausal.Record(cmdBuf, _state.GdnConvInput, gdnW.Conv1dWeight, gdnW.Conv1dBias,
                _state.GdnQkvBuf, dConv: dConv, channels: convDim, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            TmStage("gdn_conv", seqLen);

            _kernels.SiluInplace.Record(cmdBuf, _state.GdnQkvBuf, n: seqLen * convDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            TmStage("gdn_silu", seqLen);

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

        }

        // ── 4. De-interleave Q/K/V and L2-normalise Q and K ──────────────────
        // GdnQkvBuf layout per token: [Q(kDim) | K(kDim) | V(vDim)]
        if (_kernels.GdnQkvSplit is { } qkvSplit)
        {
            qkvSplit.RecordGdnQkvSplit(cmdBuf, convOut, _state.GdnQBuf, _state.GdnKBuf, _state.GdnVBuf, seqLen, kDim, vDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        long kDimBytes = (long)kDim * sizeof(float);
        long vDimBytes = (long)vDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            ulong rowBase = (ulong)((long)t * convDimBytes);
            RecordCopyBufferRange(cmdBuf, convOut, _state.GdnQBuf,
                srcOffset: rowBase, dstOffset: (ulong)((long)t * kDimBytes), size: (ulong)kDimBytes);
            RecordCopyBufferRange(cmdBuf, convOut, _state.GdnKBuf,
                srcOffset: rowBase + (ulong)kDimBytes, dstOffset: (ulong)((long)t * kDimBytes), size: (ulong)kDimBytes);
            RecordCopyBufferRange(cmdBuf, convOut, _state.GdnVBuf,
                srcOffset: rowBase + (ulong)(2 * kDimBytes), dstOffset: (ulong)((long)t * vDimBytes), size: (ulong)vDimBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        }

        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnQBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnKBuf, totalHeads: seqLen * nKHead, dState: dState, eps: 1e-6f);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("gdn_split_l2", seqLen);

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
        TmStage("gdn_scan", seqLen);

        // ── 6. Per-head RMSNorm × silu(z) gate (fused) ───────────────────────
        // postScanGateOverride: Qwen4-Exp gates with sigmoid(z) where Qwen3.5/3.6 gate with silu(z) (the shipped kernel).
        (postScanGateOverride ?? _kernels.GdnPostScanGate).Record(cmdBuf,
            gdnOut: _state.GdnOut, z: _state.GdnZBuf, ssmNormWeight: gdnW.SsmNormWeight,
            seqLen: seqLen, nVHead: nVHead, dState: dState, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("gdn_postgate", seqLen);

        // ── 7. ssm_out projection back into NormOutput ───────────────────────
        RecordMatmul(cmdBuf, gdnW.OutWeight, gdnW.OutDeviceQuantType,
            _state.GdnOut, _state.NormOutput,
            outputDim: gdnW.OutOutputDim, inputDim: gdnW.OutInputDim, seqLen: seqLen);
        TmStage("gdn_outproj", seqLen);
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
        bool attnXq = (seqLen == 1 || SmallRowQ8(seqLen)) && attnW.QDeviceQuantType == QuantizationType.Q8_0 && attnW.KDeviceQuantType == QuantizationType.Q8_0
            && attnW.VDeviceQuantType == QuantizationType.Q8_0 && attnW.QInputDim == attnW.KInputDim && attnW.QInputDim == attnW.VInputDim
            && TryPrepareQ8Activations(cmdBuf, _state.NormOutput, attnW.QInputDim, seqLen);
        RecordMatmul(cmdBuf, attnW.QWeight, attnW.QDeviceQuantType,
            _state.NormOutput, _state.QGateScratch,
            outputDim: attnW.QOutputDim, inputDim: attnW.QInputDim, seqLen: seqLen, xqReady: attnXq);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("attn_qproj", seqLen);

        // 2. De-interleave per head into Q and Gate scratch buffers.
        //    Per token row: [Q_h0, Gate_h0, Q_h1, Gate_h1, ...] each headDim wide.
        if (_kernels.QGateDeinterleave is { } qgSplit)
        {
            qgSplit.RecordQGateDeinterleave(cmdBuf, _state.QGateScratch, _state.Q, _state.GateScratch, seqLen, numHeads, headDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
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
        }

        // 3. K and V projections.
        RecordMatmul(cmdBuf, attnW.KWeight, attnW.KDeviceQuantType,
            _state.NormOutput, _state.K,
            outputDim: attnW.KOutputDim, inputDim: attnW.KInputDim, seqLen: seqLen, xqReady: attnXq);
        RecordMatmul(cmdBuf, attnW.VWeight, attnW.VDeviceQuantType,
            _state.NormOutput, _state.V,
            outputDim: attnW.VOutputDim, inputDim: attnW.VInputDim, seqLen: seqLen, xqReady: attnXq);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("attn_kvproj_deint", seqLen);

        // 4. QK-norm — per-head RMSNorm with attn_q_norm / attn_k_norm weights.
        //    Reshape as [seqLen * numHeads, headDim] rows for the RMSNorm kernel.
        _kernels.RmsNorm.Record(cmdBuf, _state.Q, attnW.QNormWeight, _state.Q,
            rowCount: seqLen * numHeads, n: headDim, eps: Config.NormEpsilon);
        _kernels.RmsNorm.Record(cmdBuf, _state.K, attnW.KNormWeight, _state.K,
            rowCount: seqLen * numKvHeads, n: headDim, eps: Config.NormEpsilon);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("attn_qknorm", seqLen);

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

        if (_kernels.SplitKvAttention is not null && seqLen == 1
            && headDim <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads))
        {
            // Decode: split the KV range across many workgroups (Flash-Decoding).
            // Engages from seqKv >= 17 with the shipping heuristic (issue #331);
            // only seqKv <= 16 falls through to the per-token kernel.
            // Real-GGUF e2e split-KV coverage exists as a test
            // (VulkanSplitDecodeMoeParityTests, issue #331) but is BLOCKED on this
            // box by issue #356 (descriptor-cache overflow in the streaming-F32
            // shared-expert matmul, unrelated to split-KV). Until #356 lands this
            // architecture ships on the shared kernel's CPU-oracle parity plus
            // synthetic-weight forward tests, like Nemotron-H.
            _kernels.SplitKvAttention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else if (_kernels.FlashAttentionCoopmat is not null && seqLen > 1 && headDim <= _kernels.FlashAttentionCoopmat.SupportedMaxHeadDim)
        {
            _kernels.FlashAttentionCoopmat.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: 0);
        }
        else if (_kernels.FlashAttention is not null && seqLen > 1 && headDim <= _kernels.FlashAttention.SupportedMaxHeadDim)
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
        TmStage("attn_rope_core", seqLen);

        // 7. Apply sigmoid(gate) element-wise to attention output.
        _kernels.SigmoidGateMul.Record(cmdBuf, _state.AttnOutput, _state.GateScratch,
            nTotal: seqLen * qElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        TmStage("attn_sigmul", seqLen);

        // 8. Output projection.
        RecordMatmul(cmdBuf, attnW.OWeight, attnW.ODeviceQuantType,
            _state.AttnOutput, _state.NormOutput,
            outputDim: attnW.OOutputDim, inputDim: attnW.OInputDim, seqLen: seqLen);
    }

    // ── MoE FFN ──────────────────────────────────────────────────────────────

    private static readonly bool FuseDecodeEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_FUSE_FORWARD"), "0", StringComparison.Ordinal);

    // Latched once every layer's resident bundle exists and takes RecordMoeDecodeFast (the bundles are uploaded lazily by the first
    // forwards, which therefore take the per-layer-submission path); false/unset re-checks until all bundles are present.
    private bool _fuseDecodeOk;

    private bool CanFuseMoeDecode(int hiddenSize)
    {
        if (_fuseDecodeOk) return true;
        if (!_residentMoeEnabled) return false;
        for (int l = 0; l < _cpuLayers.Length; l++)
        {
            if (_cpuMoeLayer[l]) return false;
            var b = _residentMoeBundles[l];
            if (b is null || !CanRecordMoeDecodeFast(b, 1, hiddenSize)) return false;
        }
        return _fuseDecodeOk = true;
    }

    /// <summary>
    /// Single-token forward recorded into ONE command buffer. The per-layer path submits twice per layer (~80 fence round trips per token) and
    /// the GPU idles during each host turnaround (the dense hybrid model measured 3 of 18 ms/token from exactly this: Tev1-4B 55 -> 63 tok/s).
    /// Everything here is device-resident (resident expert banks, GPU top-k), so no host work is needed between layers.
    /// </summary>
    private ITensor ForwardDecodeFused(
        ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, VulkanGdnStateCache gdnCache, IKvCache? kvCache,
        HybridLayerKind[] kinds, int hiddenSize, int vocabSize, int numHeads, int numKvHeads, int headDim, float eps)
    {
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        for (int layer = 0; layer < _cpuLayers.Length; layer++)
        {
            ref readonly var layerBuf = ref _weights.Layers[layer];
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            RecordMoeHybridTokenMixing(cmdBuf, layer, layerBuf, kinds, 1, hiddenSize, eps, positions, gdnCache, kvCache, numHeads, numKvHeads, headDim);
            KernelSupport.ComputeTransferFullBarrier(cmdBuf);
            RecordMoeDecodeFast(cmdBuf, _residentMoeBundles[layer]!, layerBuf, hiddenSize, eps);
        }

        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        long rowBytes = (long)hiddenSize * sizeof(float);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput, srcOffset: 0, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _kernels.RmsNorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput, rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType, _state.NormOutput, _state.Logits,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var result = UnmanagedTensor.Allocate(new TensorShape(1, vocabSize), DType.Float32, deviceId: -1);
        unsafe
        {
            _device.Download(_state.Logits, new Span<float>((void*)result.DataPointer, vocabSize));
        }
        return result;
    }

    /// <summary>
    /// Records one layer's token-mixing half (residual snapshot, attn-norm, GDN or full attention, first residual add into HiddenState)
    /// into <paramref name="cmdBuf"/>. Shared by the per-layer-submission path and the fused single-token decode path.
    /// </summary>
    private void RecordMoeHybridTokenMixing(
        nint cmdBuf, int layer, in VulkanQwen3MoeHybridWeights.LayerBuffers layerBuf, HybridLayerKind[] kinds,
        int seqLen, int hiddenSize, float eps, ReadOnlySpan<int> positions, VulkanGdnStateCache gdnCache, IKvCache? kvCache,
        int numHeads, int numKvHeads, int headDim)
    {

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
        TmStage("tm_copy_norm", seqLen);

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
        TmStage(kinds[layer] == HybridLayerKind.GatedDeltaNet ? "tm_gdn_total" : "tm_attn_total", seqLen);

        // First residual add: HiddenState = Residual + NormOutput (token-mixing output).
        //   AddScratch is reused later as MoE intermediates; here it just receives the sum
        //   so we can copy it back into HiddenState in one transfer.
        _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize);
    }

    /// <summary>
    /// Runs one GPU-placed layer's MoE FFN: the existing resident/streaming
    /// indexed-matmul path, unchanged from pre-#370 behaviour.
    /// </summary>
    private void RunGpuPlacedMoeLayer(
        int layer, Qwen3MoeLayerWeights lw,
        in VulkanQwen3MoeHybridWeights.LayerBuffers layerBuf,
        int seqLen, int hiddenSize, float eps)
    {
        // Resolve this layer's routed experts: either fetch a resident bundle
        // (opt-in) or upload fresh and dispose after the layer (default —
        // safe for any model size). When resident-mode is on, the upload
        // also opts into a resident-quant bank (Q6_K or Q4_K) when the
        // source allows (~25 GB Q6_K, or smaller for Q4_K, vs ~120 GB F32 at
        // qwen35moe-A3B scale — the only way the resident layout fits on a
        // 128 GB Strix Halo unified-memory host).
        VulkanQwen3MoeMoeUpload.LayerBundle moeBuf;
        bool disposeAfterLayer;
        if (_residentMoeEnabled)
        {
            // Lazily upload on first use; retained for the life of the
            // model after that. See _residentMoeBundles field docstring
            // for the device-memory caveat — DOTLLM_VK_MOE_RESIDENT=1
            // is opt-in for models that fit.
            bool firstUpload = _residentMoeBundles[layer] is null;
            moeBuf = _residentMoeBundles[layer]
                ?? (_residentMoeBundles[layer] = VulkanQwen3MoeMoeUpload.UploadLayer(
                    _device, lw.Moe, hiddenSize, residentQuant: true));
            if (firstUpload && layer == 0)
            {
                // One-line diagnostic so DOTLLM_VK_MOE_RESIDENT=1 runs make
                // the chosen bank storage type observable (#371) — this is
                // the only place that knows whether the resident-quant
                // overlay engaged or silently fell back to F32.
                Console.Error.WriteLine(
                    $"[dotLLM] Vulkan resident-MoE bank quant types: gate={moeBuf.W1QuantType} " +
                    $"up={moeBuf.W3QuantType} down={moeBuf.W2QuantType} " +
                    $"(source: gate={lw.Moe.GateExpsRawQt}, up={lw.Moe.UpExpsRawQt}, down={lw.Moe.DownExpsRawQt})");
            }
            disposeAfterLayer = false;
        }
        else
        {
            moeBuf = VulkanQwen3MoeMoeUpload.UploadLayer(_device, lw.Moe, hiddenSize);
            disposeAfterLayer = true;
        }

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        if (CanRecordMoeDecodeFast(moeBuf, seqLen, hiddenSize))
        {
            RecordMoeDecodeFast(cmdBuf, moeBuf, layerBuf, hiddenSize, eps);
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();
            if (disposeAfterLayer) moeBuf.Dispose();
            return;
        }

        // Second residual snapshot (HiddenState now holds the updated activations).
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
            0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMoeLayer(cmdBuf, moeBuf, layerBuf.PostAttnNormWeight, seqLen, hiddenSize, eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Second residual add (straight into HiddenState: it is not an input of the add).
        _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // In streaming mode, free this layer's transient banks before
        // moving to the next layer. In resident mode the bundle is kept
        // alive on _residentMoeBundles and only disposed at model Dispose.
        if (disposeAfterLayer) moeBuf.Dispose();
    }

    private static readonly bool MoeDecodeFastEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_DECODE_FUSED"), "0", StringComparison.Ordinal);

    /// <summary>
    /// True when the single-token MoE layer can take <see cref="RecordMoeDecodeFast"/>: resident Q4_K gate/up + Q5_K/Q6_K down with the
    /// indexed MMVQ kernels, a sigmoid-gated shared expert, and the fused norm/SwiGLU + quantize kernels.
    /// </summary>
    private bool CanRecordMoeDecodeFast(VulkanQwen3MoeMoeUpload.LayerBundle moeW, int seqLen, int hidden)
        => MoeDecodeFastEnabled && seqLen == 1
            && _kernels.RmsNormQuantizeFused is not null && _kernels.SwiGluQuantizeFused is not null
            && _kernels.MoeMmvqQ4K is not null
            && moeW.W1QuantType == QuantizationType.Q4_K && moeW.W3QuantType == QuantizationType.Q4_K
            // K-quant down only: the legacy Q5_1 / Q8_0 down banks (#849) take the general MMVQ path in RecordMoeLayer. The fused single-token
            // layer is not exercised by any Q5_1 / Q8_0 fixture (qwen4exp has no post-attention norm and never enters it), so it is not widened blind.
            && moeW.W2QuantType is QuantizationType.Q5_K or QuantizationType.Q6_K
            && DownMmvqFits(moeW.W2QuantType, moeW.IntermediateSize)
            && moeW.HasSharedExpert && moeW.SharedExpertGate is not null
            && (hidden % 256) == 0;

    /// <summary>
    /// True when the down bank of quant <paramref name="qt"/> with input width <paramref name="interm"/> has an MMVQ decode kernel. K-quants
    /// need K % 256 == 0 (super-block); the legacy Q5_1 / Q8_0 banks (#849) need only K % 32 == 0, which is what lets the real qwen4exp
    /// file (intermediate 640) take the fast decode path.
    /// </summary>
    private bool DownMmvqFits(QuantizationType qt, int interm) => qt switch
    {
        QuantizationType.Q5_K => _kernels.MoeMmvqQ5K is not null && (interm % 256) == 0,
        QuantizationType.Q6_K => _kernels.MoeMmvqQ6K is not null && (interm % 256) == 0,
        QuantizationType.Q5_1 => _kernels.MoeMmvqQ5_1 is not null && _moeUnitScale is not null && (interm % 32) == 0,
        QuantizationType.Q8_0 => _kernels.MoeMmvqQ8_0 is not null && (interm % 32) == 0,
        _ => false,
    };

    /// <summary>Records the down-projection MMVQ for <paramref name="qt"/> (Q5_1 passes the identity expert scale).</summary>
    private void RecordDownMmvq(nint cmdBuf, QuantizationType qt, VulkanQwen3MoeMoeUpload.LayerBundle moeW, int hidden, int interm, int n, int numE, bool multiRow = false)
    {
        switch (qt)
        {
            case QuantizationType.Q5_K:
                _kernels.MoeMmvqQ5K!.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                    _state.MoeTopkIndices, _state.MoeDownRows, m: hidden, k: interm, n: n, numExperts: numE);
                break;
            case QuantizationType.Q6_K:
                _kernels.MoeMmvqQ6K!.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                    _state.MoeTopkIndices, _state.MoeDownRows, m: hidden, k: interm, n: n, numExperts: numE);
                break;
            case QuantizationType.Q5_1:
                if (multiRow && _kernels.MoeMmvqQ5_1Mr is { } q51Mr && (hidden % q51Mr.RowsPerGroup) == 0)
                {
                    CountSmallRow(SmallRowPath.MoeQ5_1Mr);
                    q51Mr.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                        _state.MoeTopkIndices, _state.MoeDownRows, _moeUnitScale!, m: hidden, k: interm, n: n, numExperts: numE);
                    break;
                }
                _kernels.MoeMmvqQ5_1!.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                    _state.MoeTopkIndices, _state.MoeDownRows, _moeUnitScale!, m: hidden, k: interm, n: n, numExperts: numE);
                break;
            case QuantizationType.Q8_0:
                _kernels.MoeMmvqQ8_0!.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                    _state.MoeTopkIndices, _state.MoeDownRows, m: hidden, k: interm, n: n, numExperts: numE);
                break;
            default:
                throw new InvalidOperationException($"No MMVQ down kernel for {qt}.");
        }
    }

    /// <summary>
    /// Single-token MoE layer with the dependent chain compressed (issue #647): the shared expert is independent of the routed experts, so its
    /// matmuls ride the same barrier phases (no re-derived RMSNorm: the routed scatter lands in NormOutput only after every NormOutput reader
    /// is done); the broadcast + Q8_1 quantize collapse into the norm (one row, indexed MMVQ reads it for all topK slots); SwiGLU is fused with
    /// the down-projection quantize; the final add writes HiddenState directly. 7 barriers / 16 dispatches instead of ~17 / 22, ~13 us each on gfx1151.
    /// </summary>
    private void RecordMoeDecodeFast(
        nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW,
        in VulkanQwen3MoeHybridWeights.LayerBuffers layerBuf, int hidden, float eps)
    {
        int interm = moeW.IntermediateSize;
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int sharedI = moeW.SharedIntermediateSize;

        CountMoePath(MoePath.FusedDecode);
        // Phase 0: residual snapshot (transfer) alongside norm + quantize of the single row (compute).
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual, 0, 0, (ulong)((long)hidden * sizeof(float)));
        _kernels.RmsNormQuantizeFused!.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
            _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: hidden, eps: eps);
        KernelSupport.ComputeAndTransferToComputeBarrier(cmdBuf);

        // Phase 1: router + the three shared-expert matmuls that read NormOutput.
        RecordMatmul(cmdBuf, moeW.Gate, QuantizationType.F32, _state.NormOutput, _state.MoeRouterLogits,
            outputDim: numE, inputDim: hidden, seqLen: 1);
        // Q8_0 raw gate/up (decode only) read the row the fused norm already quantized into MoeExpandedInputXq/Xds.
        if (moeW.SharedGateQ8 is not null && moeW.SharedUpQ8 is not null && _kernels.MatMulQ8Mmvq is not null)
        {
            RecordMatmul(cmdBuf, moeW.SharedGateQ8, QuantizationType.Q8_0, _state.NormOutput, _state.MoeSharedGate,
                outputDim: sharedI, inputDim: hidden, seqLen: 1, xqReady: true);
            RecordMatmul(cmdBuf, moeW.SharedUpQ8, QuantizationType.Q8_0, _state.NormOutput, _state.MoeSharedUp,
                outputDim: sharedI, inputDim: hidden, seqLen: 1, xqReady: true);
        }
        else
        {
            RecordMatmul(cmdBuf, moeW.SharedGate!, moeW.SharedQuantType, _state.NormOutput, _state.MoeSharedGate,
                outputDim: sharedI, inputDim: hidden, seqLen: 1);
            RecordMatmul(cmdBuf, moeW.SharedUp!, moeW.SharedQuantType, _state.NormOutput, _state.MoeSharedUp,
                outputDim: sharedI, inputDim: hidden, seqLen: 1);
        }
        RecordMatmul(cmdBuf, moeW.SharedExpertGate!, QuantizationType.F32, _state.NormOutput, _state.MoeSharedGateLogits,
            outputDim: 1, inputDim: hidden, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Phase 2: top-k routing + shared SwiGLU.
        _kernels.MoeTopkSoftmax.Record(cmdBuf, _state.MoeRouterLogits, _state.MoeTopkIndices, _state.MoeTopkWeights,
            seqLen: 1, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        _kernels.SwiGlu.Record(cmdBuf, _state.MoeSharedGate, _state.MoeSharedUp, _state.MoeSharedSilu, n: sharedI);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Phase 3: routed gate/up (one activation row broadcast to the topK slots) + shared down.
        _kernels.MoeMmvqQ4K!.Record(cmdBuf, moeW.W1Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
            _state.MoeTopkIndices, _state.MoeGateInter, m: interm, k: hidden, n: topK, numExperts: numE, xDiv: topK);
        _kernels.MoeMmvqQ4K.Record(cmdBuf, moeW.W3Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
            _state.MoeTopkIndices, _state.MoeUpInter, m: interm, k: hidden, n: topK, numExperts: numE, xDiv: topK);
        RecordMatmul(cmdBuf, moeW.SharedDown!, moeW.SharedQuantType, _state.MoeSharedSilu, _state.MoeSharedSumA,
            outputDim: hidden, inputDim: sharedI, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Phase 4: SwiGLU fused with the Q8_1 quantize feeding the down projection (topK rows of interm are one contiguous run of 32-blocks).
        _kernels.SwiGluQuantizeFused!.Record(cmdBuf, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter,
            _state.MoeSiluInterXq, _state.MoeSiluInterXds, n: topK * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Phase 5: routed down.
        RecordDownMmvq(cmdBuf, moeW.W2QuantType, moeW, hidden, interm, topK, numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Phase 6: weighted scatter into NormOutput (all its readers finished in phases 1-2), shared sigmoid-gated add, residual add.
        _kernels.MoeWeightedScatter.Record(cmdBuf, _state.MoeDownRows, _state.MoeTopkWeights, _state.NormOutput,
            seqLen: 1, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _kernels.MoeSigmoidGatedAdd.Record(cmdBuf,
            output: _state.NormOutput, b: _state.MoeSharedSumA, gateLogits: _state.MoeSharedGateLogits,
            seqLen: 1, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.HiddenState, hidden);
    }

    /// <summary>
    /// Runs one CPU-placed layer's MoE FFN (#370): the post-attn-normed
    /// hidden state is computed on the GPU as usual, then downloaded, run
    /// through <see cref="MoeSwiGluMlp"/> on the CPU against the layer's raw
    /// GGUF quant-view weight pointers (identical routing/GEMM path the
    /// pure-CPU <c>Qwen3MoeHybridTransformerModel</c> uses), and re-uploaded
    /// in place — no GPU expert bank is EVER allocated for this layer.
    /// Dense/attention compute for the layer (already recorded in the 2a
    /// submission before this is called) stays fully on GPU, matching the
    /// repo's "device placement always explicit" rule.
    /// </summary>
    private void RunCpuPlacedMoeLayer(
        int layer, in VulkanQwen3MoeHybridWeights.LayerBuffers layerBuf,
        int seqLen, int hiddenSize, float eps)
    {
        // ── GPU half: residual snapshot + post-attn RMSNorm → NormOutput. ──
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.Residual,
            0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, layerBuf.PostAttnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // ── Host round trip: download, run the CPU MoE kernel in place, upload. ──
        int elemCount = seqLen * hiddenSize;
        float[] hostBuf = ArrayPool<float>.Shared.Rent(elemCount);
        try
        {
            var hostSpan = hostBuf.AsSpan(0, elemCount);
            _device.Download(_state.NormOutput, hostSpan);
            RunMoeLayerOnCpu(_cpuLayers[layer].Moe, seqLen, hiddenSize, hostSpan);
            _device.Upload(hostSpan, _state.NormOutput);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hostBuf);
        }

        // ── GPU half: residual add + copy back into HiddenState. ──
        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        _kernels.Add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch,
            seqLen * hiddenSize);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.AddScratch, _state.HiddenState,
            0, 0, (ulong)((long)seqLen * hiddenSize * sizeof(float)));
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();
    }

    /// <summary>
    /// Runs one layer's MoE SwiGLU FFN entirely on the CPU, in place over
    /// <paramref name="normOut"/>. Calls the exact same
    /// <see cref="MoeSwiGluMlp.Route"/> / <see cref="MoeSwiGluMlp.ExecuteRoutedFromAssignments"/>
    /// pair — against the same raw-quant-view weight pointers — that the
    /// pure-CPU <c>Qwen3MoeHybridTransformerModel.ForwardMoeBody</c> uses, so
    /// output is bit-identical to running the whole model on CPU for this
    /// layer. LoRA is not threaded through here — Vulkan-side LoRA is a
    /// separate delta system; CPU-placed layers under an active LoRA
    /// adapter are a known v1 gap (#370 ledger).
    /// </summary>
    private static unsafe void RunMoeLayerOnCpu(MoeLayerWeights moe, int seqLen, int hiddenSize, Span<float> normOut)
    {
        int numExperts = moe.NumExperts;
        int numExpertsPerTok = moe.NumExpertsPerTok;
        int intermediate = moe.IntermediateSize;
        int totalAssignments = seqLen * numExpertsPerTok;

        int[] assignExpert = ArrayPool<int>.Shared.Rent(totalAssignments);
        float[] assignWeight = ArrayPool<float>.Shared.Rent(totalAssignments);
        int[] bucketCursors = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] bucketTokens = ArrayPool<int>.Shared.Rent(totalAssignments);
        int[] bucketSlots = ArrayPool<int>.Shared.Rent(totalAssignments);
        int[] uniqueExperts = ArrayPool<int>.Shared.Rent(numExperts);
        try
        {
            int uniqueCount = MoeSwiGluMlp.Route(
                hidden: normOut,
                gateWeights: moe.Gate,
                assignExpert: assignExpert.AsSpan(0, totalAssignments),
                assignWeight: assignWeight.AsSpan(0, totalAssignments),
                bucketCursors: bucketCursors.AsSpan(0, numExperts + 1),
                bucketTokens: bucketTokens.AsSpan(0, totalAssignments),
                bucketSlots: bucketSlots.AsSpan(0, totalAssignments),
                uniqueExperts: uniqueExperts.AsSpan(0, numExperts),
                numExperts: numExperts,
                numExpertsPerTok: numExpertsPerTok,
                hiddenSize: hiddenSize,
                seqLen: seqLen,
                normTopKProb: moe.NormTopKProb);

            ReadOnlySpan<float> sharedGateSpan = moe.SharedExpertGate is not null
                ? moe.SharedExpertGate.AsSpan()
                : ReadOnlySpan<float>.Empty;

            bool useRawQuantView = moe.HasRawQuantView;
            nint gateBase = useRawQuantView ? moe.GateExpsRaw : 0;
            nint upBase = useRawQuantView ? moe.UpExpsRaw : 0;
            nint downBase = useRawQuantView ? moe.DownExpsRaw : 0;
            QuantizationType gateQt = useRawQuantView ? moe.GateExpsRawQt : QuantizationType.F32;
            QuantizationType upQt = useRawQuantView ? moe.UpExpsRawQt : QuantizationType.F32;
            QuantizationType downQt = useRawQuantView ? moe.DownExpsRawQt : QuantizationType.F32;

            // Per-expert byte stride into the fused gate/up/down tensors — the
            // slice for expert e is at base + e * RowByteSize(M*K, qt), i.e.
            // M * RowByteSize(K, qt) for valid quant data. Mirrors
            // Qwen3MoeHybridTransformerModel.ForwardMoeBody exactly.
            long gateRowBytes = useRawQuantView
                ? Dequantize.RowByteSize((long)intermediate * hiddenSize, gateQt) : 0;
            long upRowBytes = useRawQuantView
                ? Dequantize.RowByteSize((long)intermediate * hiddenSize, upQt) : 0;
            long downRowBytes = useRawQuantView
                ? Dequantize.RowByteSize((long)hiddenSize * intermediate, downQt) : 0;

            ReadOnlySpan<nint> gateF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W1;
            ReadOnlySpan<nint> upF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W3;
            ReadOnlySpan<nint> downF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W2;

            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                hidden: normOut,
                gateExpsRawBase: gateBase, gateExpsQt: gateQt, gateExpsRowBytes: gateRowBytes, gateExpsF32Ptrs: gateF32Ptrs,
                upExpsRawBase: upBase, upExpsQt: upQt, upExpsRowBytes: upRowBytes, upExpsF32Ptrs: upF32Ptrs,
                downExpsRawBase: downBase, downExpsQt: downQt, downExpsRowBytes: downRowBytes, downExpsF32Ptrs: downF32Ptrs,
                assignExpert: assignExpert.AsSpan(0, totalAssignments),
                assignWeight: assignWeight.AsSpan(0, totalAssignments),
                bucketCursors: bucketCursors.AsSpan(0, numExperts + 1),
                bucketTokens: bucketTokens.AsSpan(0, totalAssignments),
                bucketSlots: bucketSlots.AsSpan(0, totalAssignments),
                uniqueExperts: uniqueExperts.AsSpan(0, numExperts),
                uniqueExpertCount: uniqueCount,
                output: normOut,
                numExperts: numExperts,
                numExpertsPerTok: numExpertsPerTok,
                hiddenSize: hiddenSize,
                intermediateSize: intermediate,
                seqLen: seqLen,
                sharedGateProj: moe.SharedGateProj,
                sharedUpProj: moe.SharedUpProj,
                sharedDownProj: moe.SharedDownProj,
                sharedIntermediateSize: moe.SharedIntermediateSize,
                sharedExpertGate: sharedGateSpan,
                loraAdapter: null,
                loraLayer: -1,
                threadPool: null);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(assignExpert);
            ArrayPool<float>.Shared.Return(assignWeight);
            ArrayPool<int>.Shared.Return(bucketCursors);
            ArrayPool<int>.Shared.Return(bucketTokens);
            ArrayPool<int>.Shared.Return(bucketSlots);
            ArrayPool<int>.Shared.Return(uniqueExperts);
        }
    }

    private MoeGroupedMatmulKQuantCoopmatKernel? GroupedDownKernel(QuantizationType qt) => qt switch
    {
        QuantizationType.Q5_K => _kernels.MoeGroupedQ5K,
        QuantizationType.Q6_K => _kernels.MoeGroupedQ6K,
        _ => null,
    };

    private MoeGroupedMatmulLegacyQuantCoopmatKernel? GroupedLegacyDownKernel(QuantizationType qt) => qt switch
    {
        QuantizationType.Q5_1 => _kernels.MoeGroupedQ5_1,
        QuantizationType.Q8_0 => _kernels.MoeGroupedQ8_0,
        _ => null,
    };

    /// <summary>
    /// True when the grouped coopmat prefill path has a down kernel for <paramref name="qt"/> at input width <paramref name="interm"/>:
    /// K-quants need K % 256 == 0; the legacy Q5_1 / Q8_0 kernels (#849) stage two 32-blocks per round, so K % 64 == 0 (intermediate 640 is fine).
    /// </summary>
    private bool GroupedDownFits(QuantizationType qt, int interm)
        => GroupedDownKernel(qt) is not null ? (interm % 256) == 0
            : GroupedLegacyDownKernel(qt) is not null && _moeUnitScale is not null && (interm % MoeGroupedMatmulLegacyQuantCoopmatKernel.KGroup) == 0;

    private static readonly int GroupedMinTokensDefault =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_GROUPED_MIN_TOKENS"), out int g) && g > 0 ? g : 16;

    /// <summary>Smallest token count that takes the expert-grouped coopmat MoE path; per instance so the qwen4exp wrapper can lower it (#876).</summary>
    internal int GroupedMinTokens { get; set; } = GroupedMinTokensDefault;

    /// <summary>
    /// Grouped-by-expert routed FFN (issue #637). Buffer reuse: MoeExpandedInput (broadcast rows) -> packed rows in MoeDownRows ->
    /// gate/up into MoeGateInter/MoeUpInter (packed order) -> SwiGLU -> grouped down into MoeExpandedInput (dead by now) -> ungroup into
    /// MoeDownRows in the original row order, which the weighted scatter then consumes unchanged.
    /// </summary>
    private void RecordGroupedExperts(nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW, int seqLen, int hidden, int interm, int numE, int expandedRows,
        bool fusedGlue, int topK)
    {
        _kernels.MoeExpertOffsets!.Record(cmdBuf, _state.MoeTopkIndices, _state.MoeGroupCounts, _state.MoeGroupOffsets, _state.MoeGroupCounters,
            rows: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("expert_offsets");
        if (fusedGlue)
            _kernels.MoeExpandGatherGroup!.Record(cmdBuf, _state.NormOutput, _state.MoeTopkIndices, _state.MoeGroupOffsets,
                _state.MoeGroupCounters, _state.MoeDownRows, _state.MoeGroupPerm, _state.MoeGroupInvPerm,
                rows: expandedRows, hidden: hidden, numExperts: numE, topK: topK);
        else
            _kernels.MoeExpandGroupByExpert!.Record(cmdBuf, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeGroupOffsets,
                _state.MoeGroupCounters, _state.MoeDownRows, _state.MoeGroupPerm, rows: expandedRows, hidden: hidden, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("expand_group");

        // Launch only the (expert, 16-row tile) pairs that exist. The legacy grid (every expert x every possible row tile; a token routes
        // to an expert at most once, so no expert owns more than seqLen rows) launched ~30x more workgroups than there was work and the
        // early-out workgroups alone cost ~100 ms of a 512-token prefill. DOTLLM_VK_MOE_INDIRECT_TILES=0 restores it.
        // A legacy-quant down kernel only exists for 16-row tiles, so when gate/up would be the 32-row row-pair form the layer's gate/up
        // drops to the 16-row Q4_K kernel to share one tile list (the tile list is rebuilt per layer).
        var legacyDown = GroupedLegacyDownKernel(moeW.W2QuantType);
        CountMoePath(MoePath.GroupedPrefill);
        if (legacyDown is not null) CountMoePath(MoePath.GroupedLegacyDown);
        var gateUpKernel = legacyDown is not null && _kernels.MoeGroupedQ4K16 is not null ? _kernels.MoeGroupedQ4K16 : _kernels.MoeGroupedQ4K!;
        var downKernel = GroupedDownKernel(moeW.W2QuantType);
        int downMTiles = downKernel?.MTiles(hidden) ?? legacyDown!.MTiles(hidden);
        var tileBuild = _kernels.MoeBuildTileList;
        if (tileBuild is not null)
        {
            tileBuild.Record(cmdBuf, _state.MoeGroupOffsets, _state.MoeGroupDispatchArgs, numE,
                gateUpKernel.MTiles(interm), downMTiles, tileRows: gateUpKernel.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(cmdBuf);
            if (MoeStageProfileEnabled)
            {
                MoeStage("tile_list_build");
                Span<float> raw = stackalloc float[6];
                _device.Download(_state.MoeGroupDispatchArgs, raw);   // uint triples reinterpreted as float bits
                _moeStageMs["tiles(count,not ms)"] = _moeStageMs.GetValueOrDefault("tiles(count,not ms)") + BitConverter.SingleToUInt32Bits(raw[1]);
                _moeStageLast = System.Diagnostics.Stopwatch.GetTimestamp();
            }
            gateUpKernel.RecordIndirect(cmdBuf, moeW.W1Bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeGateInter,
                _state.MoeGroupDispatchArgs, 0, m: interm, k: hidden, rows: expandedRows, numExperts: numE);
            gateUpKernel.RecordIndirect(cmdBuf, moeW.W3Bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeUpInter,
                _state.MoeGroupDispatchArgs, 0, m: interm, k: hidden, rows: expandedRows, numExperts: numE);
        }
        else
        {
            gateUpKernel.Record(cmdBuf, moeW.W1Bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeGateInter,
                m: interm, k: hidden, rows: expandedRows, numExperts: numE, maxRowsPerExpert: seqLen);
            gateUpKernel.Record(cmdBuf, moeW.W3Bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeUpInter,
                m: interm, k: hidden, rows: expandedRows, numExperts: numE, maxRowsPerExpert: seqLen);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("grouped_gate_up");

        _kernels.SwiGlu.Record(cmdBuf, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter, n: expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("swiglu");

        if (legacyDown is not null)
        {
            // The legacy kernels take a (never applied here) per-expert scale buffer in the same binding slot.
            if (tileBuild is not null)
                legacyDown.RecordIndirect(cmdBuf, moeW.W2Bank, _state.MoeSiluInter, _state.MoeGroupOffsets, _state.MoeExpandedInput, _moeUnitScale!,
                    applyScale: false, _state.MoeGroupDispatchArgs, MoeBuildTileListKernel.ArgsStrideBytes, m: hidden, k: interm, rows: expandedRows, numExperts: numE);
            else
                legacyDown.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInter, _state.MoeGroupOffsets, _state.MoeExpandedInput, _moeUnitScale!,
                    applyScale: false, m: hidden, k: interm, rows: expandedRows, numExperts: numE, maxRowsPerExpert: seqLen);
        }
        else if (tileBuild is not null)
            downKernel!.RecordIndirect(cmdBuf, moeW.W2Bank, _state.MoeSiluInter, _state.MoeGroupOffsets, _state.MoeExpandedInput,
                _state.MoeGroupDispatchArgs, MoeBuildTileListKernel.ArgsStrideBytes, m: hidden, k: interm, rows: expandedRows, numExperts: numE);
        else
            downKernel!.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInter, _state.MoeGroupOffsets, _state.MoeExpandedInput,
                m: hidden, k: interm, rows: expandedRows, numExperts: numE, maxRowsPerExpert: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("grouped_down");
        if (!fusedGlue)
        {
            _kernels.MoeUngroupScatter!.Record(cmdBuf, _state.MoeExpandedInput, _state.MoeGroupPerm, _state.MoeDownRows, rows: expandedRows, hidden: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MoeStage("ungroup");
        }
    }

    /// <summary>
    /// Records the routed-MoE SwiGLU FFN dispatch using the per-layer banks
    /// uploaded by <see cref="VulkanQwen3MoeMoeUpload.UploadLayer"/>. Mirrors
    /// the routed path of <see cref="VulkanTransformerModel"/>'s
    /// <c>RecordMoeLayer</c> and folds in the optional Qwen1.5-MoE-style
    /// sigmoid-gated shared expert.
    /// </summary>
    private unsafe void RecordMoeLayer(
        nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle moeW,
        VulkanDevice.Buffer? postAttnNormWeight, int seqLen, int hidden, float eps)
    {
        int interm = moeW.IntermediateSize;
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int expandedRows = seqLen * topK;
        MoeStageBegin();

        // 1. Router gate logits.
        RecordMatmul(cmdBuf, moeW.Gate, QuantizationType.F32,
            _state.NormOutput, _state.MoeRouterLogits,
            outputDim: numE, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("router");

        // 2. Top-k softmax.
        _kernels.MoeTopkSoftmax.Record(cmdBuf,
            _state.MoeRouterLogits, _state.MoeTopkIndices, _state.MoeTopkWeights,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("topk");

        // Issue #637: prefill batches group the routed rows by expert and run each expert's weights through a coopmat GEMM once per
        // 16-row tile (the indexed kernels below re-read an expert's weights for every routed row). Resident Q4_K gate/up + Q5_K/Q6_K down only (UD-Q4_K_M mixes both down types).
        bool grouped = seqLen >= GroupedMinTokens
            && _kernels.MoeGroupedQ4K is not null && GroupedDownFits(moeW.W2QuantType, interm)
            && _kernels.MoeExpertOffsets is not null && _kernels.MoeExpandGroupByExpert is not null && _kernels.MoeUngroupScatter is not null
            && moeW.W1QuantType == QuantizationType.Q4_K && moeW.W3QuantType == QuantizationType.Q4_K
            && (hidden % 256) == 0;
        // Fused glue: gather the token rows straight into expert order (no broadcast pass) and combine straight from the grouped down
        // output (no ungroup pass). DOTLLM_VK_MOE_FUSED_GLUE=0 restores broadcast + expand + ungroup + scatter.
        bool fusedGlue = grouped && _kernels.MoeExpandGatherGroup is not null && _kernels.MoeWeightedScatterGrouped is not null
            && (hidden & 3) == 0;
        if (!fusedGlue)
        {
            // 3. Broadcast NormOutput[seqLen, hidden] → MoeExpandedInput[seqLen*topK, hidden].
            _kernels.MoeBroadcast.Record(cmdBuf,
                _state.NormOutput, _state.MoeExpandedInput,
                seqLen: seqLen, topK: topK, hidden: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MoeStage("broadcast");
        }
        if (grouped)
        {
            RecordGroupedExperts(cmdBuf, moeW, seqLen, hidden, interm, numE, expandedRows, fusedGlue, topK);
        }
        else
        {
        // 4. Indexed expert matmuls. All paths share the same buffer contract
        //    (bank/x/indices/y) and the same shape (m, k, n, numExperts) —
        //    only the dequant differs, and each bank picks its OWN kernel
        //    independently via its own resolved quant type (#372):
        //       F32 banks  → MoeIndexedMatmul (plain F32 dot)
        //       Q6_K/Q4_K/Q5_K banks → the matching per-row-dequant kernel
        //    See VulkanQwen3MoeMoeUpload remarks for the residency caveat.
        //
        // #383: when both gate/up banks are Q4_K-resident and the dp4a MMQ kernel
        // is available, quantize MoeExpandedInput to Q8_1 ONCE (gate and up share
        // the same activation row + K=hidden) and route both through the dp4a
        // indexed matmul instead of the scalar per-element kernel. Down (K=
        // intermediate, a different activation buffer) isn't covered by this pass
        // — it stays on its own resolved-bank kernel (Q5_K for the cached
        // UD-Q4_K_XL checkpoint, no MMQ variant wired for that bank yet).
        bool decodeMmvq = seqLen < GroupedMinTokens;
        bool useGateUpMmq = _moeIndexedMmqEnabled
            && _kernels.MoeIndexedMatmulQ4KMmq is not null
            && moeW.W1QuantType == QuantizationType.Q4_K
            && moeW.W3QuantType == QuantizationType.Q4_K
            && (hidden % MoeIndexedMatmulQ4KMmqKernel.Q4_KGroupSize) == 0;
        if (useGateUpMmq)
        {
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmdBuf,
                _state.MoeExpandedInput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
                n: expandedRows, k: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            // Decode-sized batches: the coalesced subgroup-per-cell MMVQ GEMV instead of the one-thread-per-cell MMQ.
            var gateUpMmvq = decodeMmvq ? _kernels.MoeMmvqQ4K : null;
            // #876: 2..15-token steps use the multi-row variant (bit-equal, NR output rows per workgroup).
            if (gateUpMmvq is not null && MoeMrMinRows > 0 && seqLen >= MoeMrMinRows && SmallRowGemvEnabled && _kernels.MoeMmvqQ4KMr is { } q4Mr && (interm % q4Mr.RowsPerGroup) == 0)
            { gateUpMmvq = q4Mr; CountSmallRow(SmallRowPath.MoeQ4KMr); }
            if (gateUpMmvq is not null)
            {
                gateUpMmvq.Record(cmdBuf,
                    moeW.W1Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
                    _state.MoeTopkIndices, _state.MoeGateInter,
                    m: interm, k: hidden, n: expandedRows, numExperts: numE);
                gateUpMmvq.Record(cmdBuf,
                    moeW.W3Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
                    _state.MoeTopkIndices, _state.MoeUpInter,
                    m: interm, k: hidden, n: expandedRows, numExperts: numE);
            }
            else
            {
            _kernels.MoeIndexedMatmulQ4KMmq!.Record(cmdBuf,
                moeW.W1Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
                _state.MoeTopkIndices, _state.MoeGateInter,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
            _kernels.MoeIndexedMatmulQ4KMmq!.Record(cmdBuf,
                moeW.W3Bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds,
                _state.MoeTopkIndices, _state.MoeUpInter,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
            }
        }
        else
        {
            RecordIndexedMoeMatmul(cmdBuf, moeW.W1QuantType,
                moeW.W1Bank, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeGateInter,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
            RecordIndexedMoeMatmul(cmdBuf, moeW.W3QuantType,
                moeW.W3Bank, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeUpInter,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("gate_up");

        // 5. SwiGLU: silu(gate) * up
        _kernels.SwiGlu.Record(cmdBuf, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter,
            n: expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("swiglu");

        // 6. Indexed down matmul. #383 follow-up: same dp4a swap as gate/up, for the
        // Q5_K-resident down bank (K=intermediate, MoeSiluInter as input — a
        // different activation buffer than gate/up's, so its own quantize pass).
        bool downMmvq = decodeMmvq && _kernels.QuantizeQ8_1RowsActivations is not null && DownMmvqFits(moeW.W2QuantType, interm);
        bool useDownMmq = _moeIndexedMmqEnabled
            && _kernels.MoeIndexedMatmulQ5KMmq is not null
            && moeW.W2QuantType == QuantizationType.Q5_K
            && (interm % MoeIndexedMatmulQ5KMmqKernel.Q5_KGroupSize) == 0;
        if (downMmvq)
        {
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmdBuf,
                _state.MoeSiluInter, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                n: expandedRows, k: interm);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            CountMoePath(MoePath.MmvqDown);
            if (IsLegacyQuant(moeW.W2QuantType)) CountMoePath(MoePath.MmvqLegacyDown);
            RecordDownMmvq(cmdBuf, moeW.W2QuantType, moeW, hidden, interm, expandedRows, numE, multiRow: MoeMrMinRows > 0 && seqLen >= MoeMrMinRows && SmallRowGemvEnabled);
        }
        else if (useDownMmq)
        {
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmdBuf,
                _state.MoeSiluInter, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                n: expandedRows, k: interm);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _kernels.MoeIndexedMatmulQ5KMmq!.Record(cmdBuf,
                moeW.W2Bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds,
                _state.MoeTopkIndices, _state.MoeDownRows,
                m: hidden, k: interm, n: expandedRows, numExperts: numE);
        }
        else
        {
            RecordIndexedMoeMatmul(cmdBuf, moeW.W2QuantType,
                moeW.W2Bank, _state.MoeSiluInter, _state.MoeTopkIndices, _state.MoeDownRows,
                m: hidden, k: interm, n: expandedRows, numExperts: numE);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("down");

        }

        // 7. Weighted scatter into NormOutput.
        if (fusedGlue)
            _kernels.MoeWeightedScatterGrouped!.Record(cmdBuf,
                _state.MoeExpandedInput, _state.MoeGroupInvPerm, _state.MoeTopkWeights, _state.NormOutput,
                seqLen: seqLen, topK: topK, hiddenSize: hidden);
        else
            _kernels.MoeWeightedScatter.Record(cmdBuf,
                _state.MoeDownRows, _state.MoeTopkWeights, _state.NormOutput,
                seqLen: seqLen, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("weighted_scatter");

        // 8. Shared-expert branch (Qwen1.5-MoE sigmoid-gated convention).
        if (moeW.HasSharedExpert)
        {
            RecordSharedExpert(cmdBuf, moeW, postAttnNormWeight, seqLen, hidden, eps);
            MoeStage("shared_expert");
        }
    }

    /// <summary>
    /// Per-bank-quant-type dispatcher for a routed-expert indexed matmul. Each
    /// bank's storage type is resolved independently at upload time (#372,
    /// see <see cref="VulkanQwen3MoeMoeUpload.LayerBundle.W1QuantType"/> /
    /// <c>W2QuantType</c> / <c>W3QuantType</c>); the caller passes the
    /// specific bank's own quant type here so this helper stays oblivious to
    /// whether the layer is F32-streaming or resident, and whether sibling
    /// banks share its quant type or not.
    /// </summary>
    private void RecordIndexedMoeMatmul(
        nint cmdBuf, QuantizationType bankQuantType,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        switch (bankQuantType)
        {
            case QuantizationType.Q6_K:
                _kernels.MoeIndexedMatmulQ6K.Record(cmdBuf, bank, x, indices, y,
                    m: m, k: k, n: n, numExperts: numExperts);
                break;
            case QuantizationType.Q4_K:
                _kernels.MoeIndexedMatmulQ4K.Record(cmdBuf, bank, x, indices, y,
                    m: m, k: k, n: n, numExperts: numExperts);
                break;
            case QuantizationType.Q5_K:
                _kernels.MoeIndexedMatmulQ5K.Record(cmdBuf, bank, x, indices, y,
                    m: m, k: k, n: n, numExperts: numExperts);
                break;
            case QuantizationType.Q5_1:
                (_kernels.MoeIndexedMatmulQ5_1 ?? throw new InvalidOperationException("moe_indexed_matmul_q5_1_f32.spv missing."))
                    .Record(cmdBuf, bank, x, indices, y, _moeUnitScale!, m: m, k: k, n: n, numExperts: numExperts);
                break;
            case QuantizationType.Q8_0:
                (_kernels.MoeIndexedMatmulQ8_0 ?? throw new InvalidOperationException("moe_indexed_matmul_q8_0_f32.spv missing."))
                    .Record(cmdBuf, bank, x, indices, y, m: m, k: k, n: n, numExperts: numExperts);
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
                    $"Unsupported MoE bank quant type: {bankQuantType}. " +
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
        VulkanDevice.Buffer? postAttnNormWeight, int seqLen, int hidden, float eps)
    {
        int sharedI = moeW.SharedIntermediateSize;
        int sharedInterElems = seqLen * sharedI;

        // Shared input = the same RMSNormed hidden state we used for the routed
        // path. The routed scatter overwrote NormOutput so we re-derive it.
        // A null norm weight means the caller (Qwen4-Exp: the gated-residual read already produced the block input, there is no
        // post-attention norm) pre-staged the shared input in MoeSharedInput before the router ran.
        if (postAttnNormWeight is not null)
        {
            _kernels.RmsNorm.Record(cmdBuf, _state.HiddenState, postAttnNormWeight, _state.MoeSharedInput,
                rowCount: seqLen, n: hidden, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        MoeStage("shx_norm");

        // Shared expert gate/up matmuls share the input.
        RecordMatmul(cmdBuf, moeW.SharedGate!, moeW.SharedQuantType,
            _state.MoeSharedInput, _state.MoeSharedGate,
            outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
        RecordMatmul(cmdBuf, moeW.SharedUp!, moeW.SharedQuantType,
            _state.MoeSharedInput, _state.MoeSharedUp,
            outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("shx_gateup");

        _kernels.SwiGlu.Record(cmdBuf, _state.MoeSharedGate, _state.MoeSharedUp, _state.MoeSharedSilu,
            n: sharedInterElems);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("shx_swiglu");

        RecordMatmul(cmdBuf, moeW.SharedDown!, moeW.SharedQuantType,
            _state.MoeSharedSilu, _state.MoeSharedSumA,
            outputDim: hidden, inputDim: sharedI, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("shx_down");

        if (moeW.SharedExpertGate is not null)
        {
            if (seqLen > 1 && (hidden & 3) == 0 && _kernels.MoeSharedGateAdd is { } fusedGate)
            {
                // Issue #693: gate logit (a 1 x hidden dot per token) + sigmoid + gated add in ONE pass. The M = 1 F32 GEMM below launched a
                // 64-workgroup grid (~0.5 ms/layer at 2048 tokens) and the scalar gated add another ~1 ms/layer.
                fusedGate.Record(cmdBuf, output: _state.NormOutput, b: _state.MoeSharedSumA, x: _state.MoeSharedInput,
                    gateWeight: moeW.SharedExpertGate, seqLen: seqLen, hiddenSize: hidden);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                return;
            }

            // gateLogits[t] = SharedExpertGate[1, hidden] @ MoeSharedInput[t, :].
            RecordMatmul(cmdBuf, moeW.SharedExpertGate, QuantizationType.F32,
                _state.MoeSharedInput, _state.MoeSharedGateLogits,
                outputDim: 1, inputDim: hidden, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MoeStage("shx_gatelogit");

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
    /// <summary>IQ1/IQ2/IQ3 prefill: dequantise to F16 scratch + F16 coopmat GEMM (#621). Null when disabled or unsupported.</summary>
    private IqF16PrefillMatmul? _iqF16Prefill;

    private static IqF16PrefillMatmul? CreateIqF16Prefill(VulkanDevice device, string spvDir, ModelConfig config, VulkanQwen3MoeHybridKernels kernels)
    {
        long hidden = config.HiddenSize;
        return IqF16PrefillMatmul.TryCreate(device, spvDir, kernels.MatMulF16GemmCoopmat, 4L * hidden * hidden);
    }

    /// <summary>
    /// Quantizes the single decode row <paramref name="input"/> to Q8_1 into the MoE expanded-input scratch (idle outside the routed
    /// gate/up section) so the dp4a Q8_0 MMVQ GEMV can read it; one quantize serves every Q8_0 projection that shares the input
    /// (pass <c>xqReady</c> to <see cref="RecordMatmul"/>). False when the kernels are unavailable or the row does not fit the scratch.
    /// </summary>
    private bool TryPrepareQ8Activations(nint cmdBuf, VulkanDevice.Buffer input, int k, int n = 1)
    {
        if (_kernels.MatMulQ8Mmvq is null || _kernels.QuantizeQ8_1RowsActivations is null || (k & 31) != 0) return false;
        if (QuantizeQ8_1RowsKernel.PackedBytes(n, k) > _state.MoeExpandedInputXq.Size
            || QuantizeQ8_1RowsKernel.ScaleBytes(n, k) > _state.MoeExpandedInputXds.Size) return false;
        _kernels.QuantizeQ8_1RowsActivations.Record(cmdBuf, input, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: n, k: k);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        return true;
    }

    /// <summary>Runtime switch for the 2..8-row multi-column GEMVs (#876); <c>DOTLLM_VK_SMALLROW_GEMV=0</c> at startup disables (A/B and diagnostics).</summary>
    internal static bool SmallRowGemvEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_SMALLROW_GEMV") != "0";

    /// <summary>Smallest token count that takes the multi-row routed-MoE MMVQ variants (#876); 0 = never (A/B). Default 2 leaves single-token decode on the proven kernels (DOTLLM_VK_MOE_MR_MIN_ROWS=1 opts decode in: -9% 1-row forward on the real qwen4exp file).</summary>
    internal static int MoeMrMinRows { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_MR_MIN_ROWS"), out int mrMin) && mrMin >= 0 ? mrMin : 2;

    /// <summary>True for 2..8-row forwards that have the Q8_0 multi-column MMVQ GEMV (#876).</summary>
    private bool SmallRowQ8(int seqLen)
        => SmallRowGemvEnabled && seqLen >= 2 && MatMulQ8_0MmvqMultiKernel.Accepts(seqLen, 32) && _kernels.MatMulQ8MmvqMulti is not null
           && _kernels.QuantizeQ8_1RowsActivations is not null;

    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantizationType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen, bool xqReady = false)
    {
        // IQ1/IQ2/IQ3 prefill: dequant to F16 scratch + coopmat GEMM instead of the scalar tiled GEMM (#621).
        if (seqLen > 1 && _iqF16Prefill is not null
            && _iqF16Prefill.TryRecord(cmdBuf, weightQt, weights, input, output, outputDim, inputDim, seqLen))
            return;

        switch (weightQt)
        {
            case QuantizationType.Q8_0:
                if (seqLen == 1 && _kernels.MatMulQ8Mmvq is not null && (xqReady || TryPrepareQ8Activations(cmdBuf, input, inputDim)))
                    _kernels.MatMulQ8Mmvq.Record(cmdBuf, weights, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, output, m: outputDim, k: inputDim);
                else if (seqLen == 1)
                    _kernels.MatMulQ8.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (SmallRowQ8(seqLen) && (inputDim & 31) == 0 && SmallRowQ8Prepare(cmdBuf, input, inputDim, seqLen, xqReady))
                {
                    CountSmallRow(SmallRowPath.Q8Multi);
                    _kernels.MatMulQ8MmvqMulti!.Record(cmdBuf, weights, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, output,
                        m: outputDim, k: inputDim, n: seqLen);
                }   // #876: 2..8 rows read the weights once instead of the 128x128 coopmat GEMM
                else if (_kernels.MatMulQ8GemmCoopmat is not null)
                    _kernels.MatMulQ8GemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ8Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q2_K:
                if (seqLen == 1)
                    _kernels.MatMulQ2K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ2KGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulQ2KGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ2KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q3_K:
                if (seqLen == 1)
                    _kernels.MatMulQ3K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ3KGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulQ3KGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ3KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q4_K:
                if (seqLen == 1)
                    _kernels.MatMulQ4K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ4KGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulQ4KGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ4KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q5_K:
                if (seqLen == 1)
                    _kernels.MatMulQ5K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ5KGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulQ5KGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulQ5KGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.Q6_K:
                if (seqLen == 1)
                    _kernels.MatMulQ6K.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (_kernels.MatMulQ6KGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulQ6KGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
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
                else if (_kernels.MatMulIq4XsGemmCoopmat is not null && (inputDim % 256) == 0)
                    _kernels.MatMulIq4XsGemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
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
                // Shares VulkanQwen3MoeHybridWeights.KeepPQ2_0 with the dense hybrid model, so the
                // dispatch has to match: a weights-side keep-packed arm with no matching kernel arm
                // here would reach the default and throw.
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
                else if (SmallRowGemvEnabled && _kernels.MatMulF16Multi is { } f16Multi && MatMulF16GemvMultiKernel.Accepts(seqLen, inputDim))
                { CountSmallRow(SmallRowPath.F16Multi); f16Multi.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen); }   // #876
                else if (_kernels.MatMulF16GemmCoopmat is not null)
                    _kernels.MatMulF16GemmCoopmat.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                else
                    _kernels.MatMulF16Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            case QuantizationType.BF16:
                if (seqLen == 1)
                    _kernels.MatMulBf16.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                else if (seqLen <= MatMulBf16GemvMultiF32Kernel.MaxColumns && _kernels.MatMulBf16Multi is { } bf16Multi)
                    bf16Multi.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);   // #706: thin BF16 projections at 2..8 rows
                else
                    _kernels.MatMulBf16Gemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
                break;
            default:
                if (SmallRowGemvEnabled && weightQt == QuantizationType.F32 && _kernels.MatMulF32Multi is { } f32Multi && MatMulF32GemvMultiKernel.Accepts(seqLen, inputDim))
                { CountSmallRow(SmallRowPath.F32Multi); f32Multi.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen); }   // #876
                else
                    _kernels.MatMul.Record(cmdBuf, weights, input, output, outputDim, inputDim, seqLen);
                break;
        }
    }

    /// <summary>
    /// Quantizes the <paramref name="n"/> activation rows for the multi-column Q8_0 GEMV. The shared scratch may still be read by the
    /// previous matmul, so a barrier precedes the quantize unless the caller already quantized this input (<paramref name="xqReady"/>).
    /// </summary>
    private bool SmallRowQ8Prepare(nint cmdBuf, VulkanDevice.Buffer input, int k, int n, bool xqReady)
    {
        if (xqReady) return true;
        if (QuantizeQ8_1RowsKernel.PackedBytes(n, k) > _state.MoeExpandedInputXq.Size
            || QuantizeQ8_1RowsKernel.ScaleBytes(n, k) > _state.MoeExpandedInputXds.Size) return false;
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        return TryPrepareQ8Activations(cmdBuf, input, k, n);
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
        var rows = _weights.TokenEmbeddingRows;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
            rows.RecordRowCopy(cmdBuf, id, _state.HiddenState, (long)t * rowBytes);
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
        _moeUnitScale?.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _gdnCache.Dispose();
        _iqF16Prefill?.Dispose();
        _kernels.Dispose();
        // Disposing the CPU model frees its NormWeight / DequantizeF32 native
        // allocations and detaches it from the GgufFile. The GgufFile itself
        // is owned by the caller (BuildFromGguf parameter) so we don't dispose
        // it here.
        _cpuModel?.Dispose();
        if (_ownsDevice) _device.Dispose();
    }
}
