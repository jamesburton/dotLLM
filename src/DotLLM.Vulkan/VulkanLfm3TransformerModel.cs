using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan LFM3 (<c>lfm3</c>) model (issue #1562). Same layer algebra as the CPU <see cref="Lfm3TransformerModel"/>:
/// <c>x += post_operator_norm(mixer(operator_norm(x))); x += post_ffn_norm(ffn(ffn_norm(x)))</c> with a gated short-conv
/// (<c>L_cache = 7</c>), sliding-window GQA or global GQA mixer (QK RMS-norm, NeoX RoPE, per-head sigmoid gate) and a dense SwiGLU FFN.
/// The recording machinery (kernel registry, scratch state, matmul / embedding recorders) is borrowed from
/// <see cref="VulkanQwen3MoeHybridTransformerModel"/> through a layer-less host, as <see cref="VulkanLfm2MoeTransformerModel"/> does.
/// </summary>
/// <remarks>
/// <para><b>RoPE.</b> Sliding layers rotate all 64 NeoX pairs with theta 1e4. Global layers rotate only the first 32 pairs, but with the
/// frequency exponent over the FULL head dim (<c>2i/128</c>) and the pair partner at <c>i + 64</c>: the Gemma-4 partial-global convention
/// the plain RoPE kernel already supports (<c>ropeDim = 64, freqDim = 128, neoxPairOffset = 64</c>).</para>
/// <para><b>Weights</b> are uploaded in their GGUF quantization (Q4_0 stays packed when native kernels exist; otherwise the exact Q8_0
/// transcode used by the LFM2 model).</para>
/// </remarks>
public sealed unsafe partial class VulkanLfm3TransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly Lfm3TransformerModel _cpu;      // keeps the GGUF mmap alive and owns the weight views
    private readonly VulkanQwen3MoeHybridTransformerModel _host;
    private readonly SigmoidGateMulHeadsF32Kernel _gateMul;
    private readonly VulkanFlashAttentionGqaKernel? _gqaFlash;   // GQA flash prefill with window-start tile skipping; null = per-head fallback
    private readonly LayerDev[] _layers;
    private readonly HybridLayerLayout _layout;
    private readonly int[] _convOrdinal;
    private readonly int _convLayerCount;
    private readonly int _lCache;
    private readonly float _localTheta, _globalTheta;
    private readonly int _globalRotPairs;
    private readonly VulkanLfm2ConvStateCache _defaultState;
    private VulkanLfm2ConvStateCache? _activeState;
    private readonly List<nint> _ownedHost;
    private long _deviceWeightBytes;
    private VulkanDevice.Buffer? _ffnGate, _ffnUp, _convProj, _attnGate;
    private int _scratchCapacity;
    private VulkanDevice.Buffer? _multiRowLogits;
    private int _multiRowLogitsRows;
    private int _allRowLogitsLimit;
    private long _seenKvBufferEpoch = -1;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _host.ComputeMemoryBytes + _deviceWeightBytes + _defaultState.AllocatedBytes;

    /// <summary>Number of attention layers (global + sliding): the sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _host.AttentionLayerCount;

    /// <summary>
    /// Creates the sparse (attention-layers-only), lazily allocated KV cache this model expects. It may grow up to the context window.
    /// </summary>
    /// <param name="maxSeqLen">Initial logical capacity in positions (capped at the context window).</param>
    /// <returns>The new cache.</returns>
    public VulkanNemotronHKvCache CreateKvCache(int maxSeqLen)
        => _host.CreateKvCache(Math.Min(maxSeqLen, Config.MaxSequenceLength), VulkanKvStorage.F32, lazy: true,
                               growthLimit: Config.MaxSequenceLength);

    private sealed class LayerDev
    {
        public required VulkanDevice.Buffer OperatorNorm, FfnNorm;
        public VulkanDevice.Buffer? PostOperatorNorm, PostFfnNorm;
        // conv
        public VulkanDevice.Buffer? InProj, ConvW, OutProj;
        // attention
        public VulkanDevice.Buffer? Q, K, V, O, QNorm, KNorm, AttnGate;
        public int NumKvHeads;
        public int Window;                        // 0 = global
        public bool IsAttention;
        // dense FFN
        public VulkanDevice.Buffer? Gate, Up, Down;
        public QuantizationType InProjQt, OutProjQt, QQt, KQt, VQt, OQt, AttnGateQt, GateQt, UpQt, DownQt;
        public int GateOut;
    }

    private VulkanLfm3TransformerModel(VulkanDevice device, ModelConfig config, Lfm3TransformerModel cpu,
        VulkanQwen3MoeHybridTransformerModel host, SigmoidGateMulHeadsF32Kernel gateMul, VulkanFlashAttentionGqaKernel? gqaFlash, LayerDev[] layers, int[] convOrdinal, int convLayers,
        float localTheta, float globalTheta, int globalRotPairs, long deviceBytes, List<nint> ownedHost)
    {
        _device = device;
        Config = config;
        _cpu = cpu;
        _host = host;
        _gateMul = gateMul;
        _gqaFlash = gqaFlash;
        _layers = layers;
        _layout = config.HybridLayout!;
        _convOrdinal = convOrdinal;
        _convLayerCount = convLayers;
        _lCache = config.ShortConvCacheLength;
        _localTheta = localTheta;
        _globalTheta = globalTheta;
        _globalRotPairs = globalRotPairs;
        _deviceWeightBytes = deviceBytes;
        _ownedHost = ownedHost;
        _defaultState = new VulkanLfm2ConvStateCache(device, convLayers, _lCache - 1, config.HiddenSize);
    }

    // ── Loading ──────────────────────────────────────────────────────────────

    /// <summary>Builds the Vulkan model from an opened GGUF (which must outlive the model).</summary>
    public static VulkanLfm3TransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Lfm3)
            throw new ArgumentException($"VulkanLfm3TransformerModel requires Architecture.Lfm3, got {config.Architecture}.", nameof(config));
        if (!Lfm2ShortConvF32Kernel.IsSupportedOn(spvDir))
            throw new FileNotFoundException($"lfm2_shortconv_f32.spv not found in {spvDir}.");
        if (config.ShortConvCacheLength > 8)
            throw new NotSupportedException($"lfm2_shortconv_f32 supports L_cache <= 8; the model has {config.ShortConvCacheLength}.");

        var cpu = Lfm3TransformerModel.LoadFromGguf(gguf, config);
        int hidden = config.HiddenSize, hd = config.HeadDim;
        var owned = new List<nint>();
        long bytes = 0;

        var (embPtr, embQt) = cpu.TokenEmbed;
        var (outPtr, outQt, outM, outK) = cpu.Output;
        var head = VulkanQwen3MoeHybridWeights.UploadHeadOnly(device, config, cpu.OutputNormWeight, embPtr, embQt, outPtr, outQt, outM, outK);
        bytes += head.AllocatedBytes;

        // The Gemma-4 style partial rotation expresses LFM3's global layers only when every rotated pair has freq factor 1 and the
        // tail is identity; anything else (a checkpoint with rescaled frequencies) is refused rather than rotated wrongly.
        int rotPairs = cpu.GlobalRotatedPairs;
        float globalTheta = config.GlobalRoPEConfig?.Theta ?? 1_000_000f, localTheta = config.RoPEConfig?.Theta ?? 10_000f;
        for (int i = 0; i < hd / 2; i++)
        {
            float expected = i < rotPairs ? MathF.Pow(globalTheta, -2f * i / hd) : 0f;
            if (MathF.Abs(cpu.InvFreqGlobal[i] - expected) > 1e-6f * MathF.Max(1f, MathF.Abs(expected)))
                throw new NotSupportedException($"LFM3 Vulkan: rope_freqs.weight is not a 1.0-then-identity table (pair {i}); the partial NeoX rope kernel cannot express it.");
        }

        // The layer-less host sizes its MoE scratch from config.Moe; LFM3 has none, so hand it a one-expert placeholder it never uses.
        var hostConfig = config with { Moe = new MoeConfig { NumExperts = 1, NumExpertsPerTok = 1, MoeIntermediateSize = 1 } };
        var host = VulkanQwen3MoeHybridTransformerModel.CreateLayerlessHost(device, hostConfig, spvDir, head,
            cpu.KvSlotForLayer, cpu.AttentionLayerCount, hd, localTheta);
        host.Q8GateUpMmvq = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM2_Q8_GATEUP_MMVQ") != "0";
        bool native = host.SupportsNativeQ4_0;
        var gateMul = SigmoidGateMulHeadsF32Kernel.Create(device, spvDir);
        var slidingFlash = VulkanTransformerModel.IsFlashAttentionDisabled() ? null
            : VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32, allowCoopmat: !VulkanTransformerModel.IsCoopmatAttentionDisabled());

        var layers = new LayerDev[config.NumLayers];
        var layout = config.HybridLayout!;
        using var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024);
        VulkanWeightImportPolicy.Reset();

        for (int i = 0; i < config.NumLayers; i++)
        {
            var cl = cpu.LayerWeights[i];
            var ld = new LayerDev
            {
                OperatorNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.OperatorNorm),
                FfnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.FfnNorm),
                PostOperatorNorm = cl.PostOperatorNorm is null ? null : VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.PostOperatorNorm),
                PostFfnNorm = cl.PostFfnNorm is null ? null : VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.PostFfnNorm),
                Window = cl.Window ?? 0,
            };
            bytes += (long)(cl.OperatorNorm.Length + cl.FfnNorm.Length + (cl.PostOperatorNorm?.Length ?? 0) + (cl.PostFfnNorm?.Length ?? 0)) * sizeof(float);

            if (layout.LayerKind[i] == HybridLayerKind.ShortConv)
            {
                ld.InProj = UploadProjection(device, staging, cl.InProj.Ptr, cl.InProj.Qt, cl.InProj.OutDim, cl.InProj.InDim, owned, native, out ld.InProjQt, ref bytes);
                ld.OutProj = UploadProjection(device, staging, cl.OutProj.Ptr, cl.OutProj.Qt, cl.OutProj.OutDim, cl.OutProj.InDim, owned, native, out ld.OutProjQt, ref bytes);
                // CPU keeps the taps tap-major ([L][H]); the shader wants channel-major (GGUF layout): w[c*L + k].
                int L = config.ShortConvCacheLength;
                var wT = cl.ConvWeightT!;
                var w = new float[L * hidden];
                for (int c = 0; c < hidden; c++)
                    for (int k = 0; k < L; k++)
                        w[c * L + k] = wT[k * hidden + c];
                ld.ConvW = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, w);
                bytes += (long)w.Length * sizeof(float);
            }
            else
            {
                var a = cl.Attn!;
                ld.IsAttention = true;
                ld.Q = UploadProjection(device, staging, a.QWeight, a.QQuantType, a.QOutputDim, a.QInputDim, owned, native, out ld.QQt, ref bytes);
                ld.K = UploadProjection(device, staging, a.KWeight, a.KQuantType, a.KOutputDim, a.KInputDim, owned, native, out ld.KQt, ref bytes);
                ld.V = UploadProjection(device, staging, a.VWeight, a.VQuantType, a.VOutputDim, a.VInputDim, owned, native, out ld.VQt, ref bytes);
                ld.O = UploadProjection(device, staging, a.OWeight, a.OQuantType, a.OOutputDim, a.OInputDim, owned, native, out ld.OQt, ref bytes);
                ld.AttnGate = UploadProjection(device, staging, cl.AttnGate.Ptr, cl.AttnGate.Qt, cl.AttnGate.OutDim, cl.AttnGate.InDim, owned, native, out ld.AttnGateQt, ref bytes);
                ld.QNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, a.QNormWeight);
                ld.KNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, a.KNormWeight);
                ld.NumKvHeads = a.NumKvHeads;
                bytes += (long)(a.QNormWeight.Length + a.KNormWeight.Length) * sizeof(float);
            }

            ld.Gate = UploadProjection(device, staging, cl.DenseGate.Ptr, cl.DenseGate.Qt, cl.DenseGate.OutDim, cl.DenseGate.InDim, owned, native, out ld.GateQt, ref bytes);
            ld.Up = UploadProjection(device, staging, cl.DenseUp.Ptr, cl.DenseUp.Qt, cl.DenseUp.OutDim, cl.DenseUp.InDim, owned, native, out ld.UpQt, ref bytes);
            ld.Down = UploadProjection(device, staging, cl.DenseDown.Ptr, cl.DenseDown.Qt, cl.DenseDown.OutDim, cl.DenseDown.InDim, owned, native, out ld.DownQt, ref bytes);
            ld.GateOut = cl.DenseGate.OutDim;
            layers[i] = ld;
        }

        return new VulkanLfm3TransformerModel(device, config, cpu, host, gateMul, slidingFlash, layers, cpu.ConvOrdinalForLayer, cpu.ConvLayerCount,
            localTheta, globalTheta, rotPairs, bytes, owned);
    }

    private static nint AllocPages(long bytes)
    {
        long rounded = (bytes + 4095) & ~4095L;
        return (nint)NativeMemory.AlignedAlloc((nuint)rounded, 4096);
    }

    /// <summary>
    /// Uploads a projection in its source quantization. A Q4_0 source stays packed (4.5 bpw) when native kernels exist; otherwise it is
    /// transcoded exactly to Q8_0 into a page-aligned host buffer kept alive for the model's lifetime (the upload may import it).
    /// </summary>
    private static VulkanDevice.Buffer UploadProjection(VulkanDevice device, VulkanStagingBuffer staging, nint src, QuantizationType qt,
        int outDim, int inDim, List<nint> owned, bool native, out QuantizationType deviceQt, ref long bytes)
    {
        nint ptr = src;
        if (qt == QuantizationType.Q4_0 && native && inDim % 32 == 0)
        {
            var raw = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, src, qt, outDim, inDim, out long rawBytes);
            deviceQt = QuantizationType.Q4_0;
            bytes += rawBytes;
            return raw;
        }
        if (qt == QuantizationType.Q4_0)
        {
            if (inDim % 32 != 0) throw new NotSupportedException($"Q4_0 projection with K={inDim} is not a multiple of 32.");
            long blocks = (long)outDim * (inDim / 32);
            ptr = AllocPages(blocks * 34);
            owned.Add(ptr);
            Q4_0ToQ8_0.Transcode(src, ptr, blocks);
            qt = QuantizationType.Q8_0;
        }
        var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging, ptr, qt, outDim, inDim, forceF32: false, out deviceQt, out long up);
        bytes += up;
        return buf;
    }

    // ── IModel ───────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, null, null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, kvCache, null);

    /// <summary>Forward with a caller-supplied per-sequence conv state (null = model-owned default).</summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ISsmState? ssmState)
        => ForwardCore(tokenIds, positions, deviceId, kvCache, ResolveState(ssmState));

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is not null)
            throw new NotSupportedException("VulkanLfm3TransformerModel does not support LoRA adapters.");
        return ForwardCore(tokenIds, positions, deviceId, kvCache, null);
    }

    private VulkanLfm2ConvStateCache? ResolveState(ISsmState? s)
    {
        if (s is null) return null;
        if (s is VulkanLfm2ConvStateCache c && c.NumSsmLayers == _convLayerCount &&
            c.StateSlots == _lCache - 1 && c.HiddenSize == Config.HiddenSize && ReferenceEquals(c.Device, _device))
        {
            c.ThrowIfDisposed();
            return c;
        }
        throw new ArgumentException($"VulkanLfm3TransformerModel requires a {nameof(VulkanLfm2ConvStateCache)} with {_convLayerCount} layers; got {s.GetType().Name}.", nameof(s));
    }

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState()
        => new VulkanLfm2ConvStateCache(_device, _convLayerCount, _lCache - 1, Config.HiddenSize);

    /// <inheritdoc/>
    public bool TrySetAllRowLogitsLimit(int maxSeqLen)
    {
        if (maxSeqLen > _allRowLogitsLimit) _allRowLogitsLimit = maxSeqLen;
        return _allRowLogitsLimit >= maxSeqLen;
    }

    /// <inheritdoc/>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            if (r.Adapter is not null)
                throw new NotSupportedException("VulkanLfm3TransformerModel does not support LoRA adapters.");
            if (requests.Count > 1 && r.SsmState is null)
                throw new ArgumentException($"ForwardBatch with {requests.Count} requests requires a per-seq SsmState; request[{i}] has none.");
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }

    private static void Fb(nint cmd) => KernelSupport.ComputeTransferFullBarrier(cmd);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, VulkanLfm2ConvStateCache? state)
    {
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        for (int i = 0; i < positions.Length; i++)
            if ((uint)positions[i] >= (uint)Config.MaxSequenceLength)
                throw new ArgumentOutOfRangeException(nameof(positions), $"Position {positions[i]} at index {i} exceeds max sequence length {Config.MaxSequenceLength}.");

        var prev = _activeState;
        _activeState = state ?? _defaultState;
        try
        {
            return ForwardInRowSlices(tokenIds, positions, deviceId, kvCache) ?? Run(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _activeState = prev;
        }
    }

    /// <summary>Per-submission GPU-time budget; its rates are refitted from every multi-row submission (issue #1649).</summary>
    private VulkanPrefillSubmitBudget _submitBudget = VulkanPrefillSubmitBudget.FromEnvironment();

    /// <summary>Test hook: replaces the submission budget (a tiny one forces row slicing at small prompts).</summary>
    internal VulkanPrefillSubmitBudget SubmitBudgetForTest { get => _submitBudget; set => _submitBudget = value; }

    /// <summary><c>DOTLLM_VK_LFM3_ATTN_CHUNK=0</c> disables query-row slicing of long prefills (A/B and bisection).</summary>
    private static readonly bool RowSlicingEnabled = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM3_ATTN_CHUNK") != "0";

    /// <summary>Test hook: row-slice forwards <see cref="ForwardInRowSlices"/> ran.</summary>
    internal int RowSliceForwardsForTest { get; private set; }

    /// <summary>Weight elements of one layer, for the dense cost estimate: the FFN plus an attention-sized mixer.</summary>
    private double ParamsPerLayer
        => (3.0 * Config.HiddenSize * _layers[0].GateOut) + (4.0 * Config.HiddenSize * Config.HiddenSize);

    /// <summary>Attention GFLOPs of one layer: the causal cost, with the KV span capped at the sliding window.</summary>
    private double AttentionGflopsOf(int window, int rows, int kvEnd)
    {
        double g = VulkanPrefillSubmitBudget.AttentionGflops(VulkanPrefillSubmitBudget.EffectiveRows(rows), kvEnd,
            Config.NumAttentionHeads, Config.HeadDim);
        if (window <= 0) return g;
        double meanKv = Math.Max(1.0, kvEnd - ((rows - 1) / 2.0));
        return g * Math.Min(window, meanKv) / meanKv;
    }

    /// <summary>
    /// Splits a long prefill whose single attention layer would exceed the submission budget at this KV depth into query-row slices
    /// run as consecutive forwards (issue #1649): KV and conv state carry over exactly as between engine tiles, so no single
    /// one-layer submission outlives the GPU watchdog (28672 rows lost the device at ~2.5 s per submit).
    /// </summary>
    /// <returns>The last slice's logits, or null when no split is needed or possible.</returns>
    private ITensor? ForwardInRowSlices(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        int rows = tokenIds.Length;
        if (!RowSlicingEnabled || rows <= VulkanPrefillSubmitBudget.MinRowsPerForward * 2 || rows <= _allRowLogitsLimit
            || kvCache is not VulkanNemotronHKvCache)
            return null;
        int kvEnd = VulkanNemotronHKvCache.PositionsEnd(positions);
        int kvStart0 = kvEnd - rows;
        int first = _submitBudget.NextSliceRows(rows, kvStart0, ParamsPerLayer, Config.NumAttentionHeads, Config.HeadDim);
        if (first >= rows || first <= _allRowLogitsLimit) return null;

        // Progressive slicing: each slice is sized at its own KV depth with the rates the previous slices refitted, so the shallow
        // start runs in long slices and only the deep tail is cut short (the uniform sizing of #1649 was ~25 % slower at 28K).
        ITensor? last = null;
        int start = 0, previous = 0;
        while (start < rows)
        {
            int remaining = rows - start;
            int len = start == 0 ? first
                : _submitBudget.NextSliceRows(remaining, kvStart0 + start, ParamsPerLayer, Config.NumAttentionHeads, Config.HeadDim, previous);
            len = Math.Clamp(len, 1, remaining);
            // Every slice must stay on the last-row-logits path (a slice short enough for all-row logits would download them all).
            if (len <= _allRowLogitsLimit) len = Math.Min(remaining, _allRowLogitsLimit + 1);
            RowSliceForwardsForTest++;
            last?.Dispose();
            last = Run(tokenIds.Slice(start, len), positions.Slice(start, len), deviceId, kvCache);
            start += len;
            previous = len;
        }
        return last;
    }

    private ITensor Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        int T = tokenIds.Length;
        int H = Config.HiddenSize, V = Config.VocabSize;
        float eps = Config.NormEpsilon;
        var st = _host.Q4State;
        var kern = _host.Q4Kernels;
        var wts = _host.Q4Weights;
        bool allRows = T > 1 && _allRowLogitsLimit >= T;

        bool resized = _host.Q4EnsureCapacity(T);
        resized |= EnsureScratch(T);
        if (allRows) resized |= EnsureMultiRow(T, V);
        if (VulkanNemotronHKvCache.PrepareForForward(kvCache, VulkanNemotronHKvCache.PositionsEnd(positions), ref _seenKvBufferEpoch))
            resized = true;

        if (resized) { _host.Q4InvalidateCaches(); _gateMul.InvalidateDescriptorCache(); _gqaFlash?.InvalidateDescriptorCache(); }
        _host.Q4UploadPositions(positions);

        var submit = _host.Q4Submit;
        submit.Begin();
        nint cmd = submit.CommandBuffer;
        // GPU-timestamp stage profile (#885 tooling; DOTLLM_VULKAN_MOE_STAGE_PROFILE=1 DOTLLM_VK_STAGE_TS=1): stamps inside the real
        // command buffers, no extra submits. The split-submit mode is not supported here (it would re-begin the buffer under `cmd`).
        bool prof = StageProfiling;
        if (prof) _host.Q4StageBegin();
        KernelSupport.HostToComputeBarrier(cmd);
        _host.Q4RecordEmbedding(cmd, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmd);
        if (prof) _host.Q4Stage("embed");

        ulong hiddenBytes = (ulong)((long)T * H * sizeof(float));
        int kvEndObs = VulkanNemotronHKvCache.PositionsEnd(positions);
        double openDense = 0, openAttn = 0;
        long openStart = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var ld = _layers[layer];

            Fb(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.HiddenState, st.Residual, 0, 0, hiddenBytes);
            Fb(cmd);
            kern.RmsNorm.Record(cmd, st.HiddenState, ld.OperatorNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            Fb(cmd);
            if (prof) _host.Q4Stage("norm+resid");

            if (ld.IsAttention) RecordAttention(cmd, ld, layer, T, positions, kvCache, prof);
            else { RecordConv(cmd, ld, _convOrdinal[layer], T); if (prof) _host.Q4Stage("conv"); }

            if (ld.PostOperatorNorm is not null)
            {
                Fb(cmd);
                kern.RmsNorm.Record(cmd, st.NormOutput, ld.PostOperatorNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            }
            Fb(cmd);
            kern.Add.Record(cmd, st.Residual, st.NormOutput, st.HiddenState, T * H);
            Fb(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.HiddenState, st.Residual, 0, 0, hiddenBytes);
            Fb(cmd);
            kern.RmsNorm.Record(cmd, st.HiddenState, ld.FfnNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            Fb(cmd);
            if (prof) _host.Q4Stage("norm+resid");

            RecordDense(cmd, ld, T);
            if (prof) _host.Q4Stage("ffn");

            if (ld.PostFfnNorm is not null)
            {
                Fb(cmd);
                kern.RmsNorm.Record(cmd, st.NormOutput, ld.PostFfnNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            }
            Fb(cmd);
            kern.Add.Record(cmd, st.Residual, st.NormOutput, st.HiddenState, T * H);
            if (prof) _host.Q4Stage("norm+resid");

            // Windows TDR: a single submission must finish within ~2 s. A long prefill of the whole stack does not, so a multi-row
            // forward is cut into several submissions.
            openDense += VulkanPrefillSubmitBudget.DenseGflops(ParamsPerLayer, VulkanPrefillSubmitBudget.EffectiveRows(T));
            if (ld.IsAttention) openAttn += AttentionGflopsOf(ld.Window, T, kvEndObs);
            if (T > 1 && (layer + 1) % LayersPerSubmit(T) == 0 && layer + 1 < _layers.Length)
            {
                Fb(cmd);
                submit.SubmitAndWait();
                _submitBudget.Observe(openDense, openAttn, System.Diagnostics.Stopwatch.GetElapsedTime(openStart).TotalSeconds,
                    attentionMeanKv: Math.Max(1.0, kvEndObs - ((T - 1) / 2.0)));
                openDense = openAttn = 0;
                openStart = System.Diagnostics.Stopwatch.GetTimestamp();
                submit.Begin();
                cmd = submit.CommandBuffer;
                if (prof) _host.Q4Stage("~resubmit");   // absorbs the host round trip so it is not charged to the next stage
            }
        }

        // Final norm + tied LM head.
        Fb(cmd);
        int rows = allRows ? T : 1;
        ulong rowBytes = (ulong)((long)rows * H * sizeof(float));
        ulong srcOff = allRows ? 0UL : (ulong)((long)(T - 1) * H * sizeof(float));
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.HiddenState, st.NormOutput, srcOff, 0, rowBytes);
        KernelSupport.TransferToComputeBarrier(cmd);
        kern.RmsNorm.Record(cmd, st.NormOutput, wts.OutputNormWeight, st.NormOutput, rowCount: rows, n: H, eps: eps);
        Fb(cmd);
        var logitsBuf = allRows ? _multiRowLogits! : st.Logits;
        _host.Q4RecordMatmul(cmd, wts.OutputWeight, wts.OutputDeviceQuantType, st.NormOutput, logitsBuf,
            wts.OutputOutputDim, wts.OutputInputDim, rows);
        if (prof) _host.Q4Stage("head");
        KernelSupport.ComputeToHostBarrier(cmd);
        submit.SubmitAndWait();
        if (prof) ReportStages(T);

        var result = UnmanagedTensor.Allocate(new TensorShape(rows, V), DType.Float32, deviceId: -1);
        _device.Download(logitsBuf, new Span<float>((void*)result.DataPointer, rows * V));
        return result;
    }

    /// <summary>Stage profiling is on only in GPU-timestamp mode (the split-submit mode would re-begin the command buffer under us).</summary>
    private bool StageProfiling => _host.Q4StageProfiling && VulkanQwen3MoeHybridTransformerModel.StageTimestamps;

    private readonly Dictionary<string, double> _stageMs = new();
    private int _stageForwards;

    /// <summary>Prints the per-stage GPU ms of this forward (prefill) or of the last 32 forwards (decode), largest first.</summary>
    private void ReportStages(int T)
    {
        foreach (var kv in _host.TakeStageTimes()) _stageMs[kv.Key] = _stageMs.GetValueOrDefault(kv.Key) + kv.Value;
        _stageForwards++;
        if (T > 1 || _stageForwards % 32 == 0)
        {
            int div = T > 1 ? 1 : 32;
            double total = _stageMs.Values.Sum();
            Console.Error.WriteLine($"[lfm3-stage] T={T} avg over {div} forward(s), total {total / div:F1} ms: " + string.Join("  ",
                _stageMs.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value / div:F2}ms")));
            _stageMs.Clear();
            _stageForwards = 0;
        }
    }

    private static int LayersPerSubmit(int T) => T >= 4096 ? 1 : T >= 1536 ? 2 : 4;

    private bool EnsureScratch(int T)
    {
        if (_ffnGate is not null && _scratchCapacity >= T) return false;
        _ffnGate?.Dispose(); _ffnUp?.Dispose(); _convProj?.Dispose(); _attnGate?.Dispose();
        int cap = 1;
        while (cap < T) cap *= 2;
        long f = Config.IntermediateSize;
        _ffnGate = _device.AllocateDeviceLocal((long)cap * f * sizeof(float));
        _ffnUp = _device.AllocateDeviceLocal((long)cap * f * sizeof(float));
        _convProj = _device.AllocateDeviceLocal((long)cap * 3 * Config.HiddenSize * sizeof(float));
        _attnGate = _device.AllocateDeviceLocal((long)cap * Config.NumAttentionHeads * sizeof(float));
        _scratchCapacity = cap;
        return true;
    }

    private bool EnsureMultiRow(int rows, int vocab)
    {
        if (_multiRowLogits is not null && _multiRowLogitsRows >= rows) return false;
        _multiRowLogits?.Dispose();
        _multiRowLogits = _device.AllocateHostReadback((long)rows * vocab * sizeof(float));
        _multiRowLogitsRows = rows;
        return true;
    }

    private void RecordConv(nint cmd, LayerDev ld, int ordinal, int T)
    {
        int H = Config.HiddenSize;
        var st = _host.Q4State;
        var conv = _host.Q4Kernels.Lfm2ShortConv!;
        var state = _activeState!.GetBuffer(ordinal);
        var proj = _convProj!;            // [T, 3*H]
        var y = st.AttnOutput;            // [T, H]
        _host.Q4RecordMatmul(cmd, ld.InProj!, ld.InProjQt, st.NormOutput, proj, 3 * H, H, T);
        Fb(cmd);
        conv.Record(cmd, state, proj, ld.ConvW!, y, _lCache, H, T);
        Fb(cmd);
        conv.RecordStateUpdate(cmd, state, proj, ld.ConvW!, y, _lCache, H, T);
        Fb(cmd);
        _host.Q4RecordMatmul(cmd, ld.OutProj!, ld.OutProjQt, y, st.NormOutput, H, H, T);
    }

    /// <summary>A/B switch (<c>DOTLLM_VK_LFM3_SLIDING_OPT=0</c>): restores the pre-#1572 behaviour (full-range split-KV, per-head flash) for sliding layers.</summary>
    internal static readonly bool SlidingOptDisabled = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM3_SLIDING_OPT") == "0";

    /// <summary>A/B switch (<c>DOTLLM_VK_LFM3_GLOBAL_GQA=0</c>): global-attention prefills go back to the per-head coopmat flash kernel.</summary>
    internal static readonly bool GlobalGqaDisabled = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM3_GLOBAL_GQA") == "0";

    /// <summary>Counters proving which attention path ran (sliding decodes with a clipped KV range; sliding / global prefills on the GQA flash kernel).</summary>
    internal static long SlidingClippedDecodes, SlidingFlashPrefills, GlobalFlashPrefills;

    /// <summary>
    /// First KV position a sliding-window layer still reads after a forward of <paramref name="tokens"/> rows starting at absolute
    /// position <paramref name="positionOffset"/>: the LAST row (position <c>pos = positionOffset + tokens - 1</c>) attends
    /// <c>[pos - window + 1, pos]</c>. Used to clip the single-row split-KV pass; 0 for global layers or while the context still fits.
    /// </summary>
    internal static int SlidingKvStart(int positionOffset, int tokens, int window)
        => window > 0 ? Math.Max(0, positionOffset + tokens - window) : 0;

    private void RecordAttention(nint cmd, LayerDev ld, int layer, int T, ReadOnlySpan<int> positions, IKvCache? kvCache, bool prof = false)
    {
        var st = _host.Q4State;
        var kern = _host.Q4Kernels;
        int H = Config.HiddenSize;
        int nh = Config.NumAttentionHeads, nkv = ld.NumKvHeads, hd = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int window = ld.Window;

        _host.Q4RecordMatmul(cmd, ld.Q!, ld.QQt, st.NormOutput, st.Q, nh * hd, H, T);
        _host.Q4RecordMatmul(cmd, ld.K!, ld.KQt, st.NormOutput, st.K, nkv * hd, H, T);
        _host.Q4RecordMatmul(cmd, ld.V!, ld.VQt, st.NormOutput, st.V, nkv * hd, H, T);
        _host.Q4RecordMatmul(cmd, ld.AttnGate!, ld.AttnGateQt, st.NormOutput, _attnGate!, nh, H, T);
        Fb(cmd);
        kern.RmsNorm.Record(cmd, st.Q, ld.QNorm!, st.Q, rowCount: T * nh, n: hd, eps: eps);
        kern.RmsNorm.Record(cmd, st.K, ld.KNorm!, st.K, rowCount: T * nkv, n: hd, eps: eps);
        Fb(cmd);
        if (window > 0)
            kern.Rope.Record(cmd, st.Q, st.K, st.PositionsBuffer, seqLen: T, numHeads: nh, numKvHeads: nkv,
                headDim: hd, ropeDim: hd, theta: _localTheta, variant: RopeF32Kernel.Variant.NeoX);
        else
            kern.Rope.Record(cmd, st.Q, st.K, st.PositionsBuffer, seqLen: T, numHeads: nh, numKvHeads: nkv,
                headDim: hd, ropeDim: 2 * _globalRotPairs, theta: _globalTheta, variant: RopeF32Kernel.Variant.NeoX,
                freqDim: hd, neoxPairOffset: hd / 2);

        if (prof) _host.Q4Stage("attn_proj");
        VulkanDevice.Buffer kSrc, vSrc;
        int seqKv, posOff;
        if (kvCache is VulkanNemotronHKvCache vk)
        {
            Fb(cmd);
            vk.RecordUpdate(cmd, st.K, st.V, positions, T, layer, null);
            Fb(cmd);
            kSrc = vk.GetKeysBuffer(layer);
            vSrc = vk.GetValuesBuffer(layer);
            seqKv = vk.CurrentLength;
            posOff = positions[0];
        }
        else
        {
            Fb(cmd);
            kSrc = st.K; vSrc = st.V; seqKv = T; posOff = 0;
        }
        if (prof) _host.Q4Stage("attn_kvwrite");

        // Windowed layers only need the last `window` keys: decode clips the split-KV range to them (kvStart), and the GQA flash
        // kernel skips whole KV tiles before each workgroup's window start. The per-head coopmat / scalar flash kernels mask but
        // still walk every tile, so a 32K prefill of the 5 sliding layers would cost as much as the 6 global ones.
        // Global layers (#1572 remainder): the per-head coopmat kernel re-reads every K/V tile once per query head (4x for 16/4 GQA)
        // and pays ~19 barriers per 32-key tile; the GQA-grouped kernel reads each tile once per KV head with Q and O in registers.
        int kvStart = SlidingOptDisabled ? 0 : SlidingKvStart(posOff, T, window);
        var gqaFlash = T > 1 && (window > 0 ? !SlidingOptDisabled : !GlobalGqaDisabled) ? _gqaFlash : null;
        if (kvStart > 0) SlidingClippedDecodes++;
        if (gqaFlash is not null) { if (window > 0) SlidingFlashPrefills++; else GlobalFlashPrefills++; }
        if (kern.SplitKvAttention is not null && T == 1 && hd <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv - kvStart, nh))
            kern.SplitKvAttention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: window, kvStart: kvStart);
        else if (gqaFlash is not null && gqaFlash.CanDispatch(hd, nh, nkv))
            gqaFlash.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: window);
        else if (kern.FlashAttentionCoopmat is not null && T > 1 && hd <= kern.FlashAttentionCoopmat.SupportedMaxHeadDim)
            kern.FlashAttentionCoopmat.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: window);
        else if (kern.FlashAttention is not null && T > 1 && hd <= kern.FlashAttention.SupportedMaxHeadDim)
            kern.FlashAttention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: window);
        else
            kern.Attention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: window);
        if (prof) _host.Q4Stage(window > 0 ? "attn_core_sliding" : "attn_core_global");
        Fb(cmd);
        _gateMul.Record(cmd, st.AttnOutput, _attnGate!, T * nh * hd, hd);
        Fb(cmd);
        _host.Q4RecordMatmul(cmd, ld.O!, ld.OQt, st.AttnOutput, st.NormOutput, H, nh * hd, T);
        if (prof) _host.Q4Stage("attn_out");
    }

    private void RecordDense(nint cmd, LayerDev ld, int T)
    {
        var st = _host.Q4State;
        int H = Config.HiddenSize, F = ld.GateOut;
        _host.Q4RecordMatmul(cmd, ld.Gate!, ld.GateQt, st.NormOutput, _ffnGate!, F, H, T);
        _host.Q4RecordMatmul(cmd, ld.Up!, ld.UpQt, st.NormOutput, _ffnUp!, F, H, T);
        Fb(cmd);
        _host.Q4Kernels.SwiGlu.Record(cmd, _ffnGate!, _ffnUp!, _ffnGate!, n: T * F);
        Fb(cmd);
        _host.Q4RecordMatmul(cmd, ld.Down!, ld.DownQt, _ffnGate!, st.NormOutput, H, F, T);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _defaultState.Dispose();
        foreach (var l in _layers)
        {
            l.OperatorNorm.Dispose(); l.FfnNorm.Dispose(); l.PostOperatorNorm?.Dispose(); l.PostFfnNorm?.Dispose();
            l.InProj?.Dispose(); l.ConvW?.Dispose(); l.OutProj?.Dispose();
            l.Q?.Dispose(); l.K?.Dispose(); l.V?.Dispose(); l.O?.Dispose(); l.QNorm?.Dispose(); l.KNorm?.Dispose(); l.AttnGate?.Dispose();
            l.Gate?.Dispose(); l.Up?.Dispose(); l.Down?.Dispose();
        }
        _ffnGate?.Dispose(); _ffnUp?.Dispose(); _convProj?.Dispose(); _attnGate?.Dispose(); _multiRowLogits?.Dispose();
        _gateMul.Dispose();
        _gqaFlash?.Dispose();
        _host.Dispose();
        foreach (nint p in _ownedHost) NativeMemory.AlignedFree((void*)p);
        _ownedHost.Clear();
        _cpu.Dispose();
    }
}
