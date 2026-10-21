using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan LFM2.5-24B-A2B (<c>lfm2moe</c>) model (issue #1015). Same layer algebra as <see cref="Lfm2MoeTransformerModel"/>:
/// <c>x += mixer(operator_norm(x)); x += ffn(ffn_norm(x))</c> with a gated short-conv or GQA(+QK-norm, NeoX RoPE) mixer and a dense or
/// sigmoid+bias top-k SwiGLU MoE FFN. The recording machinery (kernel registry, scratch state, matmul / routed-MoE / embedding recorders)
/// is borrowed from <see cref="VulkanQwen3MoeHybridTransformerModel"/> through a layer-less host.
/// </summary>
/// <remarks>
/// <para><b>Q4_0 weights.</b> Packed dense and routed kernels keep Q4_0 at 4.5 bits per weight when integer-dot-product kernels are available.
/// Dense prefill uses cooperative matrices when supported, otherwise the portable packed GEMM. Without the required kernels, or with
/// <c>DOTLLM_VK_Q4_0_NATIVE=0</c>, Q4_0 is transcoded exactly to Q8_0 (8.5 bits per weight, identical decoded weights).</para>
/// </remarks>
public sealed unsafe class VulkanLfm2MoeTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly GgufFile? _gguf;
    private readonly Lfm2MoeTransformerModel _cpu;      // keeps the GGUF mmap + raw routed-expert views alive
    private readonly VulkanQwen3MoeHybridTransformerModel _host;
    private readonly LayerDev[] _layers;
    private readonly HybridLayerLayout _layout;
    private readonly int[] _convOrdinal;
    private readonly int _convLayerCount;
    private readonly int _lCache;
    private readonly int _ropeDim;
    private readonly float _ropeTheta;
    private readonly VulkanLfm2ConvStateCache _defaultState;
    private VulkanLfm2ConvStateCache? _activeState;
    private readonly List<nint> _ownedHost = new();
    private long _deviceWeightBytes;

    private VulkanDevice.Buffer? _denseGate, _denseUp;
    private VulkanDevice.Buffer? _convProj;
    private int _denseCapacity;
    private VulkanDevice.Buffer? _multiRowLogits;
    private int _multiRowLogitsRows;
    private int _allRowLogitsLimit;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _host.ComputeMemoryBytes + _deviceWeightBytes + _defaultState.AllocatedBytes
        + (_convProj?.Size ?? 0);

    /// <summary>Number of full-attention layers.</summary>
    public int AttentionLayerCount => _host.AttentionLayerCount;

    /// <summary>Creates the sparse (attention-layers-only) KV cache this model expects.</summary>
    public VulkanNemotronHKvCache CreateKvCache(int maxSeqLen) => _host.CreateKvCache(maxSeqLen);

    /// <summary>Per-stage wall ms via split-submit (<c>DOTLLM_VK_LFM2_STAGE_PROFILE=1</c>): distorts decode (a fence round trip per stage).</summary>
    public static bool StageProfile { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM2_STAGE_PROFILE") == "1";

    /// <summary>GPU-timestamp stage profile (<c>DOTLLM_VK_LFM2_STAGE_PROFILE=ts</c>): one BOTTOM_OF_PIPE stamp per stage in the real submission structure;
    /// per-forward breakdown on stderr (decode: averaged over 32 forwards).</summary>
    public static bool StageTs { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM2_STAGE_PROFILE") == "ts";

    private sealed class LayerDev
    {
        public required VulkanDevice.Buffer OperatorNorm, FfnNorm;
        // conv
        public VulkanDevice.Buffer? InProj, ConvW, OutProj;
        // attention
        public VulkanDevice.Buffer? Q, K, V, O, QNorm, KNorm;
        public int NumKvHeads;
        // dense FFN
        public VulkanDevice.Buffer? Gate, Up, Down;
        public QuantizationType InProjQt, OutProjQt, QQt, KQt, VQt, OQt, GateQt, UpQt, DownQt;
        public int GateOut;
        // routed MoE
        public VulkanQwen3MoeMoeUpload.LayerBundle? Moe;
    }

    private VulkanLfm2MoeTransformerModel(VulkanDevice device, ModelConfig config, GgufFile? gguf, Lfm2MoeTransformerModel cpu,
        VulkanQwen3MoeHybridTransformerModel host, LayerDev[] layers, int[] convOrdinal, int convLayers, int ropeDim, float ropeTheta,
        long deviceBytes, List<nint> ownedHost)
    {
        _device = device;
        Config = config;
        _gguf = gguf;
        _cpu = cpu;
        _host = host;
        _layers = layers;
        _layout = config.HybridLayout!;
        _convOrdinal = convOrdinal;
        _convLayerCount = convLayers;
        _lCache = config.ShortConvCacheLength;
        _ropeDim = ropeDim;
        _ropeTheta = ropeTheta;
        _deviceWeightBytes = deviceBytes;
        _ownedHost = ownedHost;
        _defaultState = new VulkanLfm2ConvStateCache(device, convLayers, _lCache - 1, config.HiddenSize);
    }

    // ── Loading ──────────────────────────────────────────────────────────────

    /// <summary>Builds the Vulkan model from an opened GGUF (which must outlive the model).</summary>
    public static VulkanLfm2MoeTransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Lfm2Moe)
            throw new ArgumentException($"VulkanLfm2MoeTransformerModel requires Architecture.Lfm2Moe, got {config.Architecture}.", nameof(config));
        if (!Lfm2ShortConvF32Kernel.IsSupportedOn(spvDir))
            throw new FileNotFoundException($"lfm2_shortconv_f32.spv not found in {spvDir}.");

        var cpu = Lfm2MoeTransformerModel.LoadFromGguf(gguf, config);
        int hidden = config.HiddenSize;
        var owned = new List<nint>();
        long bytes = 0;

        var (embPtr, embQt) = cpu.TokenEmbed;
        var (outPtr, outQt, outM, outK) = cpu.Output;
        var head = VulkanQwen3MoeHybridWeights.UploadHeadOnly(device, config, cpu.OutputNormWeight, embPtr, embQt, outPtr, outQt, outM, outK);
        bytes += head.AllocatedBytes;

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        var host = VulkanQwen3MoeHybridTransformerModel.CreateLayerlessHost(device, config, spvDir, head,
            cpu.KvSlotForLayer, cpu.AttentionLayerCount, ropeDim, ropeTheta);
        host.Q8GateUpMmvq = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM2_Q8_GATEUP_MMVQ") != "0";
        // #1018: Q4_0 stays packed (4.5 bpw) when the native kernels exist; DOTLLM_VK_Q4_0_NATIVE=0 restores the exact Q8_0 transcode (8.5 bpw).
        bool native = host.SupportsNativeQ4_0;
        Console.Error.WriteLine($"LFM2-MoE Vulkan: Q4_0 weights {(native ? "NATIVE (packed 4.5 bpw)" : "transcoded to Q8_0 (8.5 bpw)")}");
        host.Q8GroupedGateUp = true;

        var layers = new LayerDev[config.NumLayers];
        var layout = config.HybridLayout!;
        using var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024);
        VulkanWeightImportPolicy.Reset();

        // Reusable host scratch for the transcoded routed banks (freed after each layer's upload, which copies through staging).
        long bankQ8Bytes = 0;
        foreach (var l in cpu.LayerWeights)
            if (!native && l.Moe is { } m) bankQ8Bytes = Math.Max(bankQ8Bytes, Q4_0ToQ8_0.Q8Bytes((long)m.NumExperts * m.IntermediateSize * hidden / 32 * 18));
        nint bankG = 0, bankU = 0, bankD = 0;
        if (bankQ8Bytes > 0)
        {
            bankG = AllocPages(bankQ8Bytes); bankU = AllocPages(bankQ8Bytes); bankD = AllocPages(bankQ8Bytes);
        }

        try
        {
            for (int i = 0; i < config.NumLayers; i++)
            {
                var cl = cpu.LayerWeights[i];
                var ld = new LayerDev
                {
                    OperatorNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.OperatorNorm),
                    FfnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, cl.FfnNorm),
                };
                bytes += (long)(cl.OperatorNorm.Length + cl.FfnNorm.Length) * sizeof(float);

                if (layout.LayerKind[i] == HybridLayerKind.ShortConv)
                {
                    ld.InProj = UploadQ8(device, staging, cl.InProj.Ptr, cl.InProj.Qt, cl.InProj.OutDim, cl.InProj.InDim, owned, native, out ld.InProjQt, ref bytes);
                    ld.OutProj = UploadQ8(device, staging, cl.OutProj.Ptr, cl.OutProj.Qt, cl.OutProj.OutDim, cl.OutProj.InDim, owned, native, out ld.OutProjQt, ref bytes);
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
                    ld.Q = UploadQ8(device, staging, a.QWeight, a.QQuantType, a.QOutputDim, a.QInputDim, owned, native, out ld.QQt, ref bytes);
                    ld.K = UploadQ8(device, staging, a.KWeight, a.KQuantType, a.KOutputDim, a.KInputDim, owned, native, out ld.KQt, ref bytes);
                    ld.V = UploadQ8(device, staging, a.VWeight, a.VQuantType, a.VOutputDim, a.VInputDim, owned, native, out ld.VQt, ref bytes);
                    ld.O = UploadQ8(device, staging, a.OWeight, a.OQuantType, a.OOutputDim, a.OInputDim, owned, native, out ld.OQt, ref bytes);
                    ld.QNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, a.QNormWeight);
                    ld.KNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, a.KNormWeight);
                    ld.NumKvHeads = a.NumKvHeads;
                    bytes += (long)(a.QNormWeight.Length + a.KNormWeight.Length) * sizeof(float);
                }

                if (cl.Moe is { } moe)
                {
                    long perBank = (long)moe.NumExperts * moe.IntermediateSize * hidden / 32 * 18;   // Q4_0 source bytes of one bank
                    if (moe.GateExpsRawQt != QuantizationType.Q4_0 || moe.UpExpsRawQt != QuantizationType.Q4_0 || moe.DownExpsRawQt != QuantizationType.Q4_0)
                        throw new NotSupportedException(
                            $"LFM2-MoE Vulkan: layer {i} routed banks are {moe.GateExpsRawQt}/{moe.UpExpsRawQt}/{moe.DownExpsRawQt}; only Q4_0 is supported today.");
                    if (native)
                    {
                        ld.Moe = VulkanQwen3MoeMoeUpload.UploadLayer(device, moe, hidden, residentQuant: true);
                        if (ld.Moe.W1QuantType != QuantizationType.Q4_0 || ld.Moe.W2QuantType != QuantizationType.Q4_0 || ld.Moe.W3QuantType != QuantizationType.Q4_0)
                            throw new InvalidOperationException(
                                $"LFM2-MoE layer {i}: expected Q4_0-resident banks, got {ld.Moe.W1QuantType}/{ld.Moe.W3QuantType}/{ld.Moe.W2QuantType}.");
                        bytes += 3 * perBank + (long)moe.NumExperts * hidden * sizeof(float);
                    }
                    else
                    {
                        Q4_0ToQ8_0.Transcode(moe.GateExpsRaw, bankG, perBank / 18);
                        Q4_0ToQ8_0.Transcode(moe.UpExpsRaw, bankU, perBank / 18);
                        Q4_0ToQ8_0.Transcode(moe.DownExpsRaw, bankD, perBank / 18);
                        var m8 = moe.WithRawBanks(bankG, QuantizationType.Q8_0, bankU, QuantizationType.Q8_0, bankD, QuantizationType.Q8_0);
                        ld.Moe = VulkanQwen3MoeMoeUpload.UploadLayer(device, m8, hidden, residentQuant: true);
                        if (ld.Moe.W1QuantType != QuantizationType.Q8_0 || ld.Moe.W2QuantType != QuantizationType.Q8_0 || ld.Moe.W3QuantType != QuantizationType.Q8_0)
                            throw new InvalidOperationException(
                                $"LFM2-MoE layer {i}: expected Q8_0-resident banks, got {ld.Moe.W1QuantType}/{ld.Moe.W3QuantType}/{ld.Moe.W2QuantType}.");
                        bytes += 3 * Q4_0ToQ8_0.Q8Bytes(perBank) + (long)moe.NumExperts * hidden * sizeof(float);
                    }
                }
                else
                {
                    ld.Gate = UploadQ8(device, staging, cl.DenseGate.Ptr, cl.DenseGate.Qt, cl.DenseGate.OutDim, cl.DenseGate.InDim, owned, native, out ld.GateQt, ref bytes);
                    ld.Up = UploadQ8(device, staging, cl.DenseUp.Ptr, cl.DenseUp.Qt, cl.DenseUp.OutDim, cl.DenseUp.InDim, owned, native, out ld.UpQt, ref bytes);
                    ld.Down = UploadQ8(device, staging, cl.DenseDown.Ptr, cl.DenseDown.Qt, cl.DenseDown.OutDim, cl.DenseDown.InDim, owned, native, out ld.DownQt, ref bytes);
                    ld.GateOut = cl.DenseGate.OutDim;
                }
                layers[i] = ld;
            }
        }
        finally
        {
            foreach (nint p in new[] { bankG, bankU, bankD })
                if (p != 0) NativeMemory.AlignedFree((void*)p);
        }

        int[] convOrd = cpu.ConvOrdinalForLayer;
        return new VulkanLfm2MoeTransformerModel(device, config, gguf, cpu, host, layers, convOrd, cpu.ConvLayerCount, ropeDim, ropeTheta, bytes, owned);
    }

    private static nint AllocPages(long bytes)
    {
        long rounded = (bytes + 4095) & ~4095L;
        return (nint)NativeMemory.AlignedAlloc((nuint)rounded, 4096);
    }

    /// <summary>
    /// Uploads a projection as Q8_0. A Q4_0 source is transcoded exactly into a page-aligned host buffer that is kept alive for the model's
    /// lifetime (the upload may import the host pointer instead of copying it); any other quant type is uploaded as the loader would.
    /// </summary>
    private static VulkanDevice.Buffer UploadQ8(VulkanDevice device, VulkanStagingBuffer staging, nint src, QuantizationType qt,
        int outDim, int inDim, List<nint> owned, bool native, out QuantizationType deviceQt, ref long bytes)
    {
        nint ptr = src;
        if (qt == QuantizationType.Q4_0 && native && inDim % 32 == 0)
        {
            // #1018: keep the 18-byte blocks verbatim (zero-copy mmap import when the policy allows it).
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

    private VulkanLfm2ConvStateCache? ResolveState(ISsmState? s)
    {
        if (s is null) return null;
        if (s is VulkanLfm2ConvStateCache c && c.NumSsmLayers == _convLayerCount) return c;
        throw new ArgumentException($"VulkanLfm2MoeTransformerModel requires a {nameof(VulkanLfm2ConvStateCache)} with {_convLayerCount} layers; got {s.GetType().Name}.", nameof(s));
    }

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => false;

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
                throw new NotSupportedException("VulkanLfm2MoeTransformerModel does not support LoRA adapters.");
            if (requests.Count > 1 && r.SsmState is null)
                throw new ArgumentException($"ForwardBatch with {requests.Count} requests requires a per-seq SsmState; request[{i}] has none.");
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }

    private static void Fb(nint cmd) => KernelSupport.ComputeTransferFullBarrier(cmd);

    // ── Stage profiling (split-submit, GPU-synchronous wall per stage; DOTLLM_VK_LFM2_STAGE_PROFILE=1) ──────────

    private readonly Dictionary<string, double> _stageMs = new();
    private long _stageLast;

    /// <summary>Returns and clears the accumulated per-stage wall ms.</summary>
    public Dictionary<string, double> TakeStageTimes()
    {
        var copy = new Dictionary<string, double>(_stageMs);
        _stageMs.Clear();
        return copy;
    }

    private const int TsMax = 4096;
    private nint _tsPool;
    private int _tsCount;
    private readonly string[] _tsLabels = new string[TsMax];
    private readonly ulong[] _tsScratch = new ulong[TsMax];
    private int _tsForwards;

    private void TsBegin(nint cmd)
    {
        if (_tsPool == 0)
        {
            var qci = new VkQueryPoolCreateInfo { sType = 11, queryType = 2, queryCount = TsMax };
            if (VulkanApi.vkCreateQueryPool(_device.Handle, qci, 0, out _tsPool) < 0) { _tsPool = 0; return; }
        }
        _tsCount = 0;
        VulkanApi.vkCmdResetQueryPool(cmd, _tsPool, 0, TsMax);
        TsStamp(cmd, "~begin");
    }

    private void TsStamp(nint cmd, string label)
    {
        if (_tsPool == 0 || _tsCount >= TsMax) return;
        _tsLabels[_tsCount] = label;
        VulkanApi.vkCmdWriteTimestamp(cmd, VkPipelineStageFlags.BottomOfPipe, _tsPool, (uint)_tsCount++);
    }

    private unsafe void TsCollect(int T)
    {
        int n = _tsCount;
        _tsCount = 0;
        if (_tsPool == 0 || n < 2) return;
        fixed (ulong* p = _tsScratch)
        {
            if (VulkanApi.vkGetQueryPoolResults(_device.Handle, _tsPool, 0, (uint)n, (nuint)(n * sizeof(ulong)), (nint)p, sizeof(ulong), flags: 0x1 | 0x2) < 0) return;
        }
        double toMs = _device.TimestampPeriodNs / 1_000_000.0;
        double total = (_tsScratch[n - 1] - _tsScratch[0]) * toMs;
        for (int i = 1; i < n; i++)
            _stageMs[_tsLabels[i]] = _stageMs.GetValueOrDefault(_tsLabels[i]) + (_tsScratch[i] - _tsScratch[i - 1]) * toMs;
        _stageMs["~total"] = _stageMs.GetValueOrDefault("~total") + total;
        _tsForwards++;
        if (T > 1 || _tsForwards % 32 == 0)
        {
            int div = T > 1 ? 1 : 32;
            Console.Error.WriteLine($"[lfm2-stage-ts] T={T} avg over {div} forward(s): " + string.Join("  ",
                _stageMs.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value / div:F3}ms")));
            _stageMs.Clear();
        }
    }

    private nint Stage(nint cmd, string name)
    {
        if (StageTs) { TsStamp(cmd, name); return cmd; }
        if (!StageProfile) return cmd;
        var submit = _host.Q4Submit;
        Fb(cmd);
        submit.SubmitAndWait();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_stageLast != 0)
            _stageMs[name] = _stageMs.GetValueOrDefault(name) + (now - _stageLast) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        submit.Begin();
        _stageLast = System.Diagnostics.Stopwatch.GetTimestamp();
        return submit.CommandBuffer;
    }

    // ── Forward ──────────────────────────────────────────────────────────────

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
            return Run(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _activeState = prev;
        }
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
        resized |= EnsureDense(T);
        if (allRows) resized |= EnsureMultiRow(T, V);
        if (resized) _host.Q4InvalidateCaches();
        _host.Q4UploadPositions(positions);

        long hp0 = HostProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        var submit = _host.Q4Submit;
        submit.Begin();
        nint cmd = submit.CommandBuffer;
        _stageLast = 0;
        KernelSupport.HostToComputeBarrier(cmd);
        _host.Q4RecordEmbedding(cmd, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmd);
        if (StageProfile) { _stageLast = System.Diagnostics.Stopwatch.GetTimestamp(); }
        if (StageTs) TsBegin(cmd);
        if (_host.Q4StageProfiling) _host.Q4StageBegin();   // routed-MoE sub-stage profile (DOTLLM_VULKAN_MOE_STAGE_PROFILE=1 [+ DOTLLM_VK_STAGE_TS=1])
        cmd = Stage(cmd, "embed");

        ulong hiddenBytes = (ulong)((long)T * H * sizeof(float));
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var ld = _layers[layer];
            bool isConv = _layout.LayerKind[layer] == HybridLayerKind.ShortConv;

            Fb(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.HiddenState, st.Residual, 0, 0, hiddenBytes);
            Fb(cmd);
            kern.RmsNorm.Record(cmd, st.HiddenState, ld.OperatorNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            Fb(cmd);
            cmd = Stage(cmd, "norm_in");

            if (isConv) cmd = RecordConv(cmd, ld, _convOrdinal[layer], T);
            else cmd = RecordAttention(cmd, ld, layer, T, positions, kvCache);

            Fb(cmd);
            kern.Add.Record(cmd, st.Residual, st.NormOutput, st.HiddenState, T * H);
            Fb(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.HiddenState, st.Residual, 0, 0, hiddenBytes);
            Fb(cmd);
            kern.RmsNorm.Record(cmd, st.HiddenState, ld.FfnNorm, st.NormOutput, rowCount: T, n: H, eps: eps);
            Fb(cmd);
            cmd = Stage(cmd, "norm_ffn");

            if (ld.Moe is { } moe)
            {
                _host.LfmRecordMoe(cmd, moe, ld.FfnNorm, T);
                cmd = submit.CommandBuffer;   // the host's split-submit stage profile re-begins the buffer
                cmd = Stage(cmd, "moe_total");
            }
            else
            {
                cmd = RecordDense(cmd, ld, T);
            }

            Fb(cmd);
            kern.Add.Record(cmd, st.Residual, st.NormOutput, st.HiddenState, T * H);
            cmd = Stage(cmd, "res_add");

            // Windows TDR: a single submission must finish within ~2 s. A 512-token prefill of the whole stack does not, so a multi-row
            // forward is cut into one submission per layer.
            if (T > 1 && (layer + 1) % LayersPerSubmit(T) == 0 && layer + 1 < _layers.Length && !StageProfile)
            {
                Fb(cmd);
                submit.SubmitAndWait();   // SplitSubmit is limited to one use per forward; per-layer waits are noise next to a prefill layer
                submit.Begin();
                cmd = submit.CommandBuffer;
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
        cmd = Stage(cmd, "lm_head");
        KernelSupport.ComputeToHostBarrier(cmd);
        long hp1 = HostProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        submit.SubmitAndWait();
        long hp2 = HostProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (StageTs) TsCollect(T);
        if (_host.Q4StageProfiling) ReportMoeSubStages(T);

        var result = UnmanagedTensor.Allocate(new TensorShape(rows, V), DType.Float32, deviceId: -1);
        _device.Download(logitsBuf, new Span<float>((void*)result.DataPointer, rows * V));
        if (HostProfile && T == 1) HostProfileTick(hp0, hp1, hp2, System.Diagnostics.Stopwatch.GetTimestamp());
        return result;
    }

    /// <summary>Host-side split of a decode forward (<c>DOTLLM_VK_LFM2_HOST_PROFILE=1</c>): command recording, submit + GPU wait, logits download; printed per 32 tokens.</summary>
    public static bool HostProfile { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_LFM2_HOST_PROFILE") == "1";
    private double _hpRecord, _hpWait, _hpDownload;
    private int _hpN;

    private void HostProfileTick(long t0, long t1, long t2, long t3)
    {
        double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _hpRecord += (t1 - t0) * f; _hpWait += (t2 - t1) * f; _hpDownload += (t3 - t2) * f;
        if (++_hpN % 32 == 0)
        {
            Console.Error.WriteLine($"[lfm2-host] decode avg over 32: record={_hpRecord / 32:F3}ms  submit+wait={_hpWait / 32:F3}ms  download={_hpDownload / 32:F3}ms");
            _hpRecord = _hpWait = _hpDownload = 0;
        }
    }

    private readonly Dictionary<string, double> _moeSub = new();
    private int _moeSubForwards;

    private void ReportMoeSubStages(int T)
    {
        foreach (var kv in _host.TakeStageTimes())
            _moeSub[kv.Key] = _moeSub.GetValueOrDefault(kv.Key) + kv.Value;
        _moeSubForwards++;
        if (T > 1 || _moeSubForwards % 32 == 0)
        {
            int div = T > 1 ? 1 : 32;
            Console.Error.WriteLine($"[lfm2-moe-sub] T={T} sum over 38 MoE layers, avg over {div} forward(s): " + string.Join("  ",
                _moeSub.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value / div:F3}ms")));
            _moeSub.Clear();
        }
    }

    /// <summary>Layers recorded per submission on a multi-row forward (TDR budget: ~2 s per submission; a layer is ~15 ms at 512 rows).</summary>
    private static int LayersPerSubmit(int T) => T >= 4096 ? 1 : T >= 1536 ? 2 : 4;

    private bool EnsureDense(int T)
    {
        if (_denseGate is not null && _denseCapacity >= T) return false;
        _denseGate?.Dispose(); _denseUp?.Dispose(); _convProj?.Dispose();
        int cap = 1;
        while (cap < T) cap *= 2;
        long f = Config.IntermediateSize;
        _denseGate = _device.AllocateDeviceLocal((long)cap * f * sizeof(float));
        _denseUp = _device.AllocateDeviceLocal((long)cap * f * sizeof(float));
        // Conv emits [B | C | X], independent of routing top-k. MoE scratch can be smaller than 3H.
        _convProj = _device.AllocateDeviceLocal((long)cap * 3 * Config.HiddenSize * sizeof(float));
        _denseCapacity = cap;
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

    private nint RecordConv(nint cmd, LayerDev ld, int ordinal, int T)
    {
        int H = Config.HiddenSize;
        var st = _host.Q4State;
        var conv = _host.Q4Kernels.Lfm2ShortConv!;
        var state = _activeState!.GetBuffer(ordinal);
        var proj = _convProj!;            // [T, 3*H], independent of routing top-k
        var y = st.AttnOutput;            // [T, H]
        _host.Q4RecordMatmul(cmd, ld.InProj!, ld.InProjQt, st.NormOutput, proj, 3 * H, H, T);
        cmd = Stage(cmd, "conv_inproj");
        Fb(cmd);
        conv.Record(cmd, state, proj, ld.ConvW!, y, _lCache, H, T);
        Fb(cmd);
        conv.RecordStateUpdate(cmd, state, proj, ld.ConvW!, y, _lCache, H, T);
        Fb(cmd);
        cmd = Stage(cmd, "conv_core");
        _host.Q4RecordMatmul(cmd, ld.OutProj!, ld.OutProjQt, y, st.NormOutput, H, H, T);
        cmd = Stage(cmd, "conv_outproj");
        return cmd;
    }

    private nint RecordAttention(nint cmd, LayerDev ld, int layer, int T, ReadOnlySpan<int> positions, IKvCache? kvCache)
    {
        var st = _host.Q4State;
        var kern = _host.Q4Kernels;
        int H = Config.HiddenSize;
        int nh = Config.NumAttentionHeads, nkv = ld.NumKvHeads, hd = Config.HeadDim;
        float eps = Config.NormEpsilon;

        _host.Q4RecordMatmul(cmd, ld.Q!, ld.QQt, st.NormOutput, st.Q, nh * hd, H, T);
        _host.Q4RecordMatmul(cmd, ld.K!, ld.KQt, st.NormOutput, st.K, nkv * hd, H, T);
        _host.Q4RecordMatmul(cmd, ld.V!, ld.VQt, st.NormOutput, st.V, nkv * hd, H, T);
        Fb(cmd);
        cmd = Stage(cmd, "attn_qkv");
        kern.RmsNorm.Record(cmd, st.Q, ld.QNorm!, st.Q, rowCount: T * nh, n: hd, eps: eps);
        kern.RmsNorm.Record(cmd, st.K, ld.KNorm!, st.K, rowCount: T * nkv, n: hd, eps: eps);
        Fb(cmd);
        kern.Rope.Record(cmd, st.Q, st.K, st.PositionsBuffer, seqLen: T, numHeads: nh, numKvHeads: nkv,
            headDim: hd, ropeDim: _ropeDim, theta: _ropeTheta, variant: RopeF32Kernel.Variant.NeoX);
        cmd = Stage(cmd, "attn_norm_rope");

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

        if (kern.SplitKvAttention is not null && T == 1 && hd <= VulkanSplitKvAttentionKernel.MaxHeadDim
            && VulkanSplitKvAttentionKernel.WouldSplit(seqKv, nh))
            kern.SplitKvAttention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: 0);
        else if (kern.FlashAttentionCoopmat is not null && T > 1 && hd <= kern.FlashAttentionCoopmat.SupportedMaxHeadDim)
            kern.FlashAttentionCoopmat.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: 0);
        else if (kern.FlashAttention is not null && T > 1 && hd <= kern.FlashAttention.SupportedMaxHeadDim)
            kern.FlashAttention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: 0);
        else
            kern.Attention.Record(cmd, st.Q, kSrc, vSrc, st.AttnOutput, seqQ: T, seqKv: seqKv, numHeads: nh, numKvHeads: nkv,
                headDim: hd, positionOffset: posOff, slidingWindow: 0);
        Fb(cmd);
        cmd = Stage(cmd, "attn_core");
        _host.Q4RecordMatmul(cmd, ld.O!, ld.OQt, st.AttnOutput, st.NormOutput, H, nh * hd, T);
        cmd = Stage(cmd, "attn_oproj");
        return cmd;
    }

    private nint RecordDense(nint cmd, LayerDev ld, int T)
    {
        var st = _host.Q4State;
        int H = Config.HiddenSize, F = ld.GateOut;
        _host.Q4RecordMatmul(cmd, ld.Gate!, ld.GateQt, st.NormOutput, _denseGate!, F, H, T);
        _host.Q4RecordMatmul(cmd, ld.Up!, ld.UpQt, st.NormOutput, _denseUp!, F, H, T);
        Fb(cmd);
        _host.Q4Kernels.SwiGlu.Record(cmd, _denseGate!, _denseUp!, _denseGate!, n: T * F);
        Fb(cmd);
        _host.Q4RecordMatmul(cmd, ld.Down!, ld.DownQt, _denseGate!, st.NormOutput, H, F, T);
        return Stage(cmd, "ffn_dense");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _defaultState.Dispose();
        foreach (var l in _layers)
        {
            l.OperatorNorm.Dispose(); l.FfnNorm.Dispose();
            l.InProj?.Dispose(); l.ConvW?.Dispose(); l.OutProj?.Dispose();
            l.Q?.Dispose(); l.K?.Dispose(); l.V?.Dispose(); l.O?.Dispose(); l.QNorm?.Dispose(); l.KNorm?.Dispose();
            l.Gate?.Dispose(); l.Up?.Dispose(); l.Down?.Dispose();
            l.Moe?.Dispose();
        }
        _denseGate?.Dispose(); _denseUp?.Dispose(); _convProj?.Dispose(); _multiRowLogits?.Dispose();
        _host.Dispose();
        foreach (nint p in _ownedHost) NativeMemory.AlignedFree((void*)p);
        _ownedHost.Clear();
        _cpu.Dispose();
    }
}

/// <summary>
/// Exact Q4_0 to Q8_0 block transcode: <c>q8[j] = nibble_j - 8</c> with the unchanged f16 scale (Q4_0 value <c>(n - 8) * d</c> is an int8 times
/// <c>d</c>, so the Q8_0 block reconstructs the identical F32 weights).
/// </summary>
internal static unsafe class Q4_0ToQ8_0
{
    /// <summary>Q8_0 bytes for the same blocks as <paramref name="q4Bytes"/> Q4_0 bytes.</summary>
    public static long Q8Bytes(long q4Bytes) => q4Bytes / 18 * 34;

    /// <summary>Transcodes <paramref name="blocks"/> Q4_0 blocks at <paramref name="src"/> into Q8_0 blocks at <paramref name="dst"/>.</summary>
    public static void Transcode(nint src, nint dst, long blocks)
    {
        const long Chunk = 8192;
        long chunks = (blocks + Chunk - 1) / Chunk;
        byte* s0 = (byte*)src;
        byte* d0 = (byte*)dst;
        Parallel.For(0, chunks, c =>
        {
            long b0 = c * Chunk, b1 = Math.Min(blocks, b0 + Chunk);
            for (long b = b0; b < b1; b++)
            {
                byte* s = s0 + b * 18;
                sbyte* d = (sbyte*)(d0 + b * 34);
                *(ushort*)(d0 + b * 34) = *(ushort*)s;
                byte* qs = s + 2;
                for (int j = 0; j < 16; j++)
                {
                    byte v = qs[j];
                    d[2 + j] = (sbyte)((v & 0x0F) - 8);
                    d[2 + j + 16] = (sbyte)((v >> 4) - 8);
                }
            }
        });
    }
}
