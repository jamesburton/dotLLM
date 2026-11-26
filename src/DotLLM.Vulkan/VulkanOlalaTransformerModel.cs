using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

/// <summary>Receives <c>(name, data, rows, cols)</c> activations from <see cref="VulkanOlalaTransformerModel"/>'s trace hook (tests / bisection).</summary>
internal delegate void OlalaVkTraceSink(string name, float[] data, int rows, int cols);

/// <summary>Test / measurement switches of the Olala Vulkan forward. The defaults are the shipped configuration.</summary>
/// <param name="ActivationF32">
/// Build the F32-activation-STORAGE reference arm (the <c>_f32act</c> shaders). A control for the precision sign-off measurements, never the
/// default: CLAUDE.md precision rule 1 forbids F32 activation storage without the user's sign-off.
/// </param>
/// <param name="MaxChunk">Tokens per submission (prefill chunk), bounds the activation scratch.</param>
internal sealed record OlalaVulkanOptions(bool ActivationF32 = false, int MaxChunk = 256);

/// <summary>
/// Vulkan forward of OVHcloud Olala (<c>olala</c>, issue #1983; CPU reference: <see cref="OlalaTransformerModel"/>, semantics in
/// <c>docs/OLALA.md</c>). 36 blocks of <c>geodesic(mixer) -> geodesic(latent MoE + shared expert)</c>; the mixer is a Mamba-3 MIMO layer (<c>M</c>) or
/// a differential tensor-product-attention layer (<c>V</c>).
/// </summary>
/// <remarks>
/// <para><b>Precision (user decisions 2026-11-25).</b> Stored activations are BF16 (Olala's activations reach 1e9..1e14, FP16 would overflow);
/// every kernel accumulates and reduces in F32 registers / shared memory and rounds to BF16 once on store. The recurrent state (SSM matrix, rotary
/// angle, trapezoid K and V, token-shift K/V) is F32 (<see cref="OlalaStatePolicy"/>). The attention KV cache is BF16. Weights run quantized
/// in place (Q8_0 / Q6_K / BF16 read by the shader, never widened; the router is F32). No FP16 anywhere on the activation path, no quantized
/// (FP16-scaled) activation matmul.</para>
/// <para>The model owns its sequence state like the CPU model: positions must continue it, and a call at position 0 restarts it. One command
/// buffer per chunk of <see cref="OlalaVulkanOptions.MaxChunk"/> tokens (decode = one submission per token).</para>
/// </remarks>
public sealed unsafe class VulkanOlalaTransformerModel : IModel
{
    private sealed class Mat
    {
        public required VulkanDevice.Buffer Buf;
        public required QuantizationType Qt;
        public required int In, Out;
    }

    private sealed class Layer
    {
        public required int Index;
        public required bool IsSsm;
        public required float GeoMixScale, GeoMixBias, GeoMlpScale, GeoMlpBias;
        public required Mat MixerOut;
        // MoE
        public required VulkanDevice.Buffer RouterW, ExpertBias;
        public required Mat LatentDown, LatentUp, SharedUp, SharedDown;
        public required Mat UpExps, DownExps;   // In = per-expert K, Out = per-expert rows
        // M
        public Mat? SsmIn, SsmInDyn;
        public VulkanDevice.Buffer? Params;     // packed F32 constants, see olala_mamba3_step.glsl
        public VulkanDevice.Buffer? SsmState;   // F32 recurrent state
        // V
        public Mat? Q, WaK, WaV, WbK, WbV, ShiftK, ShiftV, Lambda, Gate;
        public VulkanDevice.Buffer? QNorm, KNorm, Scaler, LastKv;   // LastKv: F32 token-shift carry [2, nk*hd]
        public VulkanDevice.Buffer? KCache, VCache;                 // BF16 [pos, nk, hd]
    }

    private readonly VulkanDevice _device;
    private readonly string _spvDir;
    private readonly GgufFile _gguf;
    private readonly OlalaConfig _o;
    private readonly OlalaVulkanOptions _opt;
    private readonly Layer[] _layers;
    private readonly Mat _embed, _output;
    private readonly int _hidden, _vocab, _heads, _hd, _nq, _nk, _group, _snr, _rank, _lat, _ff, _topk, _experts;
    private readonly int _P, _N, _R, _A, _chunk;
    private readonly uint _window;
    private readonly float _eps, _softcap;
    private readonly int _actBytes;
    private readonly Dictionary<string, OlalaKernel> _kernels = new();
    private readonly VulkanDevice.Buffer _dummy;
    private readonly List<VulkanDevice.Buffer> _owned = new();
    private readonly VulkanDevice.SubmitContext _submit;
    private int _kvCapacity;
    private int _length;
    private long _deviceBytes;

    // activation scratch (sized for _chunk tokens)
    private VulkanDevice.Buffer _res = null!, _mix = null!, _zxdt = null!, _dyn = null!, _ssmY = null!;
    private VulkanDevice.Buffer _q = null!, _ak = null!, _av = null!, _bk = null!, _bv = null!, _sk = null!, _sv = null!, _lam = null!, _gate = null!, _attnY = null!;
    private VulkanDevice.Buffer _x0 = null!, _h = null!, _routedLat = null!, _routed = null!, _sh = null!, _sel = null!, _selW = null!, _ids = null!;
    private VulkanDevice.Buffer? _logits;
    private long _logitsBytes;

    // barrier tracking
    private readonly HashSet<nint> _pendingW = new(), _pendingR = new();
    private nint _cmd;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _deviceBytes;

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <summary>Tokens consumed by the current sequence.</summary>
    public int Length => _length;

    /// <summary>Test/diagnostic hook: receives named intermediates (null = off). Setting it splits each block into several submissions.</summary>
    internal OlalaVkTraceSink? Trace { get; set; }

    /// <summary>Test/diagnostic: post-compile VGPR / LDS / scratch (spill) bytes of every pipeline built so far (VK_AMD_shader_info; empty when the device lacks it).</summary>
    internal IReadOnlyList<string> ShaderStatistics()
    {
        var lines = new List<string>();
        if (!_device.HasShaderInfoAmd) return lines;
        foreach (var k in _kernels.Values)
        {
            var st = _device.GetShaderStatisticsAmd(k.PipelineHandle);
            lines.Add($"{k.Name}: vgprs {st.resourceUsage.numUsedVgprs}/{st.numAvailableVgprs}, lds {st.resourceUsage.ldsUsageSizeInBytes} B, scratch {st.resourceUsage.scratchMemUsageInBytes} B/lane, wg {st.computeWorkGroupSizeX}");
        }
        return lines;
    }

    /// <summary>Activation storage element size in bytes (2 = BF16, the shipped configuration).</summary>
    internal int ActivationElementBytes => _actBytes;

    /// <summary>Allocated recurrent-state buffers: (kind, element count, bytes). Kinds mirror <see cref="OlalaStatePolicy"/>. Guard-test input.</summary>
    internal IReadOnlyList<(string Kind, long Elements, long Bytes)> StateBuffers => ComputeStateLayout(_o, Config.HeadDim);

    /// <summary>Bytes of the attention K/V cache currently allocated (BF16).</summary>
    internal long KvCacheBytes => _layers.Sum(l => (l.KCache?.Size ?? 0) + (l.VCache?.Size ?? 0));

    private VulkanOlalaTransformerModel(VulkanDevice device, string spvDir, GgufFile gguf, ModelConfig config, OlalaVulkanOptions opt,
                                        Layer[] layers, Mat embed, Mat output, long weightBytes)
    {
        _device = device; _spvDir = spvDir; _gguf = gguf; Config = config; _opt = opt; _layers = layers; _embed = embed; _output = output;
        _o = config.Olala!;
        _hidden = config.HiddenSize; _vocab = config.VocabSize; _hd = config.HeadDim;
        _heads = _o.NumSsmHeads; _P = _o.MambaHeadDim; _N = _o.StateSize; _R = _o.MimoRank; _A = _o.NumRopeAngles;
        _nk = _o.NumNoiseHeads; _nq = _o.NumSignalHeads + _o.NumNoiseHeads; _group = _nq / _nk; _snr = _group - 1; _rank = _o.TpaRank;
        _lat = _o.MoeLatentSize > 0 ? _o.MoeLatentSize : _hidden; _ff = _o.ExpertFfLength; _topk = _o.ExpertUsedCount; _experts = _o.ExpertCount;
        _window = (uint)Math.Min(config.SlidingWindowSize ?? int.MaxValue, int.MaxValue);
        _eps = config.NormEpsilon; _softcap = config.AttnLogitSoftcap ?? 0f;
        _actBytes = opt.ActivationF32 ? 4 : 2;
        _chunk = Math.Max(1, opt.MaxChunk);
        _deviceBytes = weightBytes;
        _dummy = Own(device.AllocateDeviceLocal(256));
        _submit = device.CreateSubmitContext();
        AllocateScratch();
        ResetSequenceState();
    }

    private VulkanDevice.Buffer Own(VulkanDevice.Buffer b) { _owned.Add(b); return b; }

    private VulkanDevice.Buffer Scratch(long elements)
    {
        var b = Own(_device.AllocateDeviceLocal(Math.Max(16, elements * _actBytes)));
        _deviceBytes += b.Size;
        return b;
    }

    private void AllocateScratch()
    {
        int T = _chunk;
        _res = Scratch((long)T * _hidden); _mix = Scratch((long)T * _hidden);
        _zxdt = Scratch((long)T * _heads * (2 * _P + 3)); _dyn = Scratch((long)T * (2 * _R * _N + _A)); _ssmY = Scratch((long)T * _heads * _P);
        int gateW = _nk * _snr * _hd;
        _q = Scratch((long)T * _nq * _hd);
        _ak = Scratch((long)T * _nk * _rank); _av = Scratch((long)T * _nk * _rank);
        _bk = Scratch((long)T * _rank * _hd); _bv = Scratch((long)T * _rank * _hd);
        _sk = Scratch((long)T * _nk); _sv = Scratch((long)T * _nk); _lam = Scratch((long)T * _nk);
        _gate = Scratch((long)T * gateW); _attnY = Scratch((long)T * gateW);
        _x0 = Scratch((long)T * _lat); _h = Scratch((long)T * _topk * _ff); _routedLat = Scratch((long)T * _lat);
        _routed = Scratch((long)T * _hidden); _sh = Scratch((long)T * _o.SharedExpertFfLength);
        _sel = Own(_device.AllocateDeviceLocal(Math.Max(16, (long)T * _topk * 4))); _deviceBytes += _sel.Size;
        _selW = Scratch((long)T * _topk);
        _ids = Own(_device.Allocate(Math.Max(16, (long)T * 4)));
    }

    // ───────────────────────────── state layout ─────────────────────────────

    /// <summary>
    /// The recurrent-state buffers an Olala model of this geometry allocates, with element counts: the single source of truth shared by the
    /// allocator and the F32-policy guard test (<see cref="OlalaStatePolicy"/>).
    /// </summary>
    internal static List<(string Kind, long Elements, long Bytes)> ComputeStateLayout(OlalaConfig o, int headDim)
    {
        int H = o.NumSsmHeads, P = o.MambaHeadDim, N = o.StateSize, R = o.MimoRank, A = o.NumRopeAngles;
        int ssm = o.SsmLayerCount, attn = o.AttentionLayerCount;
        long Bytes(OlalaStateStorage s, long e) => e * OlalaStatePolicy.ElementBytes(s);
        return
        [
            ("ssm", (long)ssm * H * N * P, Bytes(OlalaStatePolicy.Ssm, (long)ssm * H * N * P)),
            ("angle", (long)ssm * H * A, Bytes(OlalaStatePolicy.Angle, (long)ssm * H * A)),
            ("trapezoidK", (long)ssm * R * H * N, Bytes(OlalaStatePolicy.Trapezoid, (long)ssm * R * H * N)),
            ("trapezoidV", (long)ssm * H * P, Bytes(OlalaStatePolicy.Trapezoid, (long)ssm * H * P)),
            ("tokenShift", (long)attn * 2 * o.NumNoiseHeads * headDim, Bytes(OlalaStatePolicy.TokenShift, (long)attn * 2 * o.NumNoiseHeads * headDim)),
        ];
    }

    /// <summary>Actual device-buffer bytes of the recurrent state of THIS model instance (what was really allocated), per kind.</summary>
    internal IReadOnlyDictionary<string, long> AllocatedStateBytes
    {
        get
        {
            long ssm = 0, rest = 0, shift = 0;
            foreach (var l in _layers)
            {
                if (l.SsmState is { } s) { ssm += s.Size; }
                if (l.LastKv is { } k) shift += k.Size;
            }
            _ = rest;
            return new Dictionary<string, long> { ["mamba"] = ssm, ["tokenShift"] = shift };
        }
    }

    /// <inheritdoc/>
    public void ResetSequenceState()
    {
        foreach (var l in _layers)
        {
            if (l.SsmState is { } s) _device.FillBufferSynchronous(s, 0);
            if (l.LastKv is { } k) _device.FillBufferSynchronous(k, 0);
        }
        _length = 0;
    }

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Builds the Vulkan model from an opened GGUF (which must outlive the model; weights are imported in place where the policy allows).</summary>
    public static VulkanOlalaTransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
        => BuildFromGguf(device, gguf, config, spvDir, new OlalaVulkanOptions());

    internal static VulkanOlalaTransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir, OlalaVulkanOptions opt)
    {
        ArgumentNullException.ThrowIfNull(device); ArgumentNullException.ThrowIfNull(gguf); ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Olala || config.Olala is not { } o)
            throw new ArgumentException("VulkanOlalaTransformerModel requires an Architecture.Olala configuration.", nameof(config));
        if (!device.HasSubgroupArithmetic)
            throw new NotSupportedException("The Olala Vulkan kernels need subgroup arithmetic (VK_KHR_shader_subgroup_arithmetic); this device does not report it.");
        if (!device.NumericFeatures.StorageBuffer16BitAccess)
            throw new NotSupportedException("The Olala Vulkan kernels store BF16 activations through 16-bit storage buffer access; this device does not report storageBuffer16BitAccess.");
        var tensors = gguf.TensorsByName;
        Nvfp4Support.RejectUnappliedScales(tensors, consumedScales: null, "The olala Vulkan loader");
        var problems = OlalaTensors.FindProblems(tensors, config);
        if (problems.Count > 0)
            throw new InvalidDataException("olala tensor table does not match the model contract: " + string.Join("; ", problems.Take(8)));
        if (config.HeadDim % 8 != 0 || config.HeadDim > 128)
            throw new NotSupportedException($"Olala Vulkan attention supports head_dim a multiple of 8 and <= 128, got {config.HeadDim}.");
        if (o.MambaHeadDim > 64 || o.MimoRank > 8 || o.StateSize > 128 || o.StateSize % 2 != 0 || o.NumRopeAngles > 64 || o.NumRopeAngles > o.StateSize / 2)
            throw new NotSupportedException($"Olala Vulkan Mamba step supports head dim <= 64, rank <= 8, an even state size <= 128 (the state column is split across two lanes), angles <= min(64, state/2); got head dim {o.MambaHeadDim}, rank {o.MimoRank}, state {o.StateSize}, angles {o.NumRopeAngles}.");
        if (o.NumSignalHeads + o.NumNoiseHeads != 4 * o.NumNoiseHeads)
            throw new NotSupportedException("Olala Vulkan attention supports 4 query heads per kv head (3 signal + 1 noise).");
        // olala_moe_route keeps the token row and the scores in fixed shared arrays (sx4[1024] = 4096 floats read as vec4, sig/scored[1024], chosen[16]).
        if (config.HiddenSize > 4096 || config.HiddenSize % 4 != 0 || o.ExpertCount > 1024 || o.ExpertUsedCount < 1 || o.ExpertUsedCount > 16 || o.ExpertUsedCount > o.ExpertCount)
            throw new NotSupportedException($"Olala Vulkan router supports hidden <= 4096 (a multiple of 4: the F32 router is read as vec4), experts <= 1024 and 1 <= top-k <= min(16, experts); got hidden {config.HiddenSize}, " +
                                            $"experts {o.ExpertCount}, top-k {o.ExpertUsedCount}.");

        long bytes = 0;
        VulkanWeightImportPolicy.Reset();
        using var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024);

        Mat Weight(string name, bool allowF32 = false)
        {
            var d = tensors[name];
            var qt = d.QuantizationType;
            int inDim = (int)d.Shape[0];
            bool ok = qt switch
            {
                QuantizationType.Q8_0 => inDim % 32 == 0,
                QuantizationType.Q6_K => inDim % 256 == 0,
                QuantizationType.BF16 => inDim % 2 == 0,
                QuantizationType.F32 => allowF32,
                _ => false,
            };
            if (!ok)
                throw new NotSupportedException($"olala tensor {name} is {qt} with K = {inDim}: the Vulkan forward reads Q8_0 / Q6_K / BF16 weights in place " +
                                                "(and the F32 router) and refuses to widen anything else (CLAUDE.md precision rule 2).");
            long outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
            if (d.Shape.Rank > 2) outDim *= d.Shape[2];   // expert banks: rows of all experts back to back
            var buf = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), qt, (int)outDim, inDim, out long up);
            bytes += up;
            return new Mat { Buf = buf, Qt = qt, In = inDim, Out = (int)(d.Shape.Rank > 1 ? d.Shape[1] : 1) };
        }
        float[] F32(string name)
        {
            var d = tensors[name];
            if (d.QuantizationType != QuantizationType.F32)
                throw new InvalidDataException($"olala tensor {name} is {d.QuantizationType}; the contract keeps it F32.");
            long n = 1; for (int i = 0; i < d.Shape.Rank; i++) n *= d.Shape[i];
            return new ReadOnlySpan<float>((void*)gguf.TensorDataPointer(d), (int)n).ToArray();
        }
        VulkanDevice.Buffer Floats(float[] a) { bytes += (long)a.Length * 4; return VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, a); }

        var embed = Weight(OlalaTensors.TokenEmbd);
        if (embed.Qt is not (QuantizationType.Q8_0 or QuantizationType.BF16))
            throw new NotSupportedException($"olala token_embd is {embed.Qt}; the Vulkan embedding gather reads Q8_0 or BF16.");
        Mat output = tensors.ContainsKey(OlalaTensors.Output) ? Weight(OlalaTensors.Output) : embed;

        var layers = new Layer[config.NumLayers];
        int hidden = config.HiddenSize, hd = config.HeadDim;
        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => OlalaTensors.Block(il, s);
            var router = tensors[B("ffn_gate_inp.weight")];
            if (router.QuantizationType != QuantizationType.F32)
                throw new NotSupportedException($"olala router {B("ffn_gate_inp.weight")} is {router.QuantizationType}; the contract keeps the router F32.");
            var up = Weight(B("ffn_up_exps.weight")); var down = Weight(B("ffn_down_exps.weight"));
            // Weight() reports Out = shape[1] (rows of ONE expert): that is what the MoE kernels index by.
            var layer = new Layer
            {
                Index = il, IsSsm = o.IsSsmLayer(il),
                GeoMixScale = F32(B("geo_mixer_scale"))[0], GeoMixBias = F32(B("geo_mixer_bias"))[0],
                GeoMlpScale = F32(B("geo_mlp_scale"))[0], GeoMlpBias = F32(B("geo_mlp_bias"))[0],
                MixerOut = Weight(B("attn_output.weight")),
                /* F32 router read in place (zero-copy import where the policy allows), never narrowed */
                RouterW = Weight(B("ffn_gate_inp.weight"), allowF32: true).Buf, ExpertBias = Floats(F32(B("exp_probs_b"))),
                LatentDown = Weight(B("ffn_latent_down.weight")), LatentUp = Weight(B("ffn_latent_up.weight")),
                SharedUp = Weight(B("ffn_up_shexp.weight")), SharedDown = Weight(B("ffn_down_shexp.weight")),
                UpExps = up, DownExps = down,
            };
            if (layer.IsSsm)
            {
                layer.SsmIn = Weight(B("ssm_in.weight")); layer.SsmInDyn = Weight(B("ssm_in_dyn.weight"));
                var prm = new List<float>();
                foreach (var n in new[] { "ssm_b_norm.weight", "ssm_c_norm.weight", "ssm_b_bias", "ssm_c_bias", "ssm_dt_bias", "ssm_d", "ssm_mimo_x", "ssm_mimo_z", "ssm_mimo_o" })
                    prm.AddRange(F32(B(n)));
                layer.Params = Floats(prm.ToArray());
            }
            else
            {
                layer.Q = Weight(B("attn_q.weight")); layer.WaK = Weight(B("attn_wa_k.weight")); layer.WaV = Weight(B("attn_wa_v.weight"));
                layer.WbK = Weight(B("attn_wb_k.weight")); layer.WbV = Weight(B("attn_wb_v.weight"));
                layer.ShiftK = Weight(B("attn_shift_k.weight")); layer.ShiftV = Weight(B("attn_shift_v.weight"));
                layer.Lambda = Weight(B("attn_lambda.weight")); layer.Gate = Weight(B("attn_gate.weight"));
                layer.QNorm = Floats(F32(B("attn_q_norm.weight"))); layer.KNorm = Floats(F32(B("attn_k_norm.weight")));
                layer.Scaler = Floats(F32(B("attn_softmax_scaler")));
            }
            layers[il] = layer;
        }
        _ = hidden; _ = hd;

        var model = new VulkanOlalaTransformerModel(device, spvDir, gguf, config, opt, layers, embed, output, bytes);
        model.AllocateState();
        return model;
    }

    private void AllocateState()
    {
        foreach (var l in _layers)
        {
            if (l.IsSsm)
            {
                long floats = (long)_heads * _N * _P + (long)_heads * _A + (long)_R * _heads * _N + (long)_heads * _P;
                l.SsmState = Own(_device.AllocateDeviceLocal(floats * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Ssm)));
                _deviceBytes += l.SsmState.Size;
            }
            else
            {
                l.LastKv = Own(_device.AllocateDeviceLocal(2L * _nk * _hd * OlalaStatePolicy.ElementBytes(OlalaStatePolicy.TokenShift)));
                _deviceBytes += l.LastKv.Size;
            }
        }
        ResetSequenceState();
        EnsureKvCapacity(Math.Min(Config.MaxSequenceLength, 1024));
    }

    private void EnsureKvCapacity(int needed)
    {
        if (needed <= _kvCapacity) return;
        int cap = Math.Min(Config.MaxSequenceLength, Math.Max(needed, _kvCapacity == 0 ? needed : _kvCapacity * 2));
        if (cap < needed) throw new ArgumentOutOfRangeException(nameof(needed), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        long bytes = (long)cap * _nk * _hd * 2;
        foreach (var l in _layers)
        {
            if (l.IsSsm) continue;
            var k = _device.AllocateDeviceLocal(bytes); var v = _device.AllocateDeviceLocal(bytes);
            _deviceBytes += k.Size + v.Size;
            if (l.KCache is { } ok)
            {
                // Preserve the history on growth (copy through the device, no host round trip).
                long used = (long)_kvCapacity * _nk * _hd * 2;
                _device.CopyBufferRangeSynchronous(ok, k, 0, 0, (ulong)used);
                _device.CopyBufferRangeSynchronous(l.VCache!, v, 0, 0, (ulong)used);
                _deviceBytes -= ok.Size + l.VCache!.Size;
                _owned.Remove(ok); _owned.Remove(l.VCache!);
                ok.Dispose(); l.VCache!.Dispose();
            }
            l.KCache = Own(k); l.VCache = Own(v);
        }
        _kvCapacity = cap;
    }

    // ───────────────────────────── kernels & recording ─────────────────────────────

    private OlalaKernel Kernel(string key, string name, int buffers, int pushWords, params uint[] spec)
    {
        string full = name + (_opt.ActivationF32 ? "_f32act" : "");
        string k = key + "|" + full;
        if (!_kernels.TryGetValue(k, out var kern))
            _kernels[k] = kern = new OlalaKernel(_device, _spvDir, full, buffers, pushWords, spec);
        return kern;
    }

    private static string Wt(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => "q8_0", QuantizationType.Q6_K => "q6_k", QuantizationType.BF16 => "bf16",
        _ => throw new NotSupportedException($"olala Vulkan has no kernel for {qt}."),
    };

    /// <summary>Records one dispatch, inserting a compute barrier first when it conflicts with a still-pending write (RAW / WAW / WAR).</summary>
    private void Run(OlalaKernel k, uint writeMask, ReadOnlySpan<uint> push, uint gx, uint gy, uint gz, params VulkanDevice.Buffer[] bufs)
    {
        bool conflict = false;
        for (int i = 0; i < bufs.Length && !conflict; i++)
        {
            nint h = bufs[i].Handle;
            if (ReferenceEquals(bufs[i], _dummy)) continue;
            bool write = ((writeMask >> i) & 1u) != 0;
            if (_pendingW.Contains(h) || (write && _pendingR.Contains(h))) conflict = true;
        }
        if (conflict)
        {
            KernelSupport.ComputeToComputeBarrier(_cmd);
            _pendingW.Clear(); _pendingR.Clear();
        }
        for (int i = 0; i < bufs.Length; i++)
        {
            if (ReferenceEquals(bufs[i], _dummy)) continue;
            if (((writeMask >> i) & 1u) != 0) _pendingW.Add(bufs[i].Handle); else _pendingR.Add(bufs[i].Handle);
        }
        k.Record(_cmd, bufs, push, gx, gy, gz);
    }

    private static uint F(float f) => BitConverter.SingleToUInt32Bits(f);

    /// <summary><c>y[T, M] = epilogue(W[M, K] x[T, K])</c>; epi 0 none, 1 relu^2, 2 add <paramref name="aux"/>; <paramref name="f32Out"/> stores logits to F32.</summary>
    private void Gemv(Mat w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int T, int epi = 0, VulkanDevice.Buffer? aux = null, bool f32Out = false, long xOff = 0)
    {
        int nc = T == 1 ? 1 : 8;
        var k = Kernel($"gemv|{nc}|{epi}|{(f32Out ? 1 : 0)}|{Wt(w.Qt)}", "olala_gemv_" + Wt(w.Qt), 5, 4, (uint)nc, (uint)epi, f32Out ? 1u : 0u);
        uint M = (uint)w.Out;
        uint gx = Math.Min(M, 32768u), gy = (M + 32767u) / 32768u, gz = (uint)((T + nc - 1) / nc);
        var yAct = f32Out ? _dummy : y;
        var yf = f32Out ? y : _dummy;
        Run(k, (1u << 2) | (1u << 4), [M, (uint)w.In, (uint)T, (uint)xOff], gx, gy, gz, w.Buf, x, yAct, aux ?? _dummy, yf);
    }

    private void Geodesic(VulkanDevice.Buffer x, VulkanDevice.Buffer g, int T, int layerIndex, float scale, float bias)
    {
        var k = Kernel("geo", "olala_geodesic", 2, 4);
        Run(k, 1u << 0, [(uint)_hidden, (uint)layerIndex + 1, F(scale), F(bias)], (uint)T, 1, 1, x, g);
    }

    /// <summary>
    /// Opt-in split-submit stage profile (null = off, zero cost): when set, every <see cref="Sp"/> mark submits and waits, then adds the host
    /// time since the previous mark to <c>label</c> -> (ms, count). Each mark costs one submit/fence sync (~50 us), so subtract that x count.
    /// </summary>
    internal Dictionary<string, (double Ms, int Count)>? StageProfile { get; set; }
    private long _spT0;

    private void Sp(string label)
    {
        if (StageProfile is null) return;
        _submit.SubmitAndWait();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        StageProfile.TryGetValue(label, out var v);
        StageProfile[label] = (v.Ms + System.Diagnostics.Stopwatch.GetElapsedTime(_spT0, now).TotalMilliseconds, v.Count + 1);
        _pendingW.Clear(); _pendingR.Clear();
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        _spT0 = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private void Flush()
    {
        _submit.SubmitAndWait();
        _pendingW.Clear(); _pendingR.Clear();
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
    }

    private float[] DownloadAct(VulkanDevice.Buffer b, long count)
    {
        var res = new float[count];
        if (_actBytes == 4) { _device.Download(b, res); return res; }
        var raw = new float[(count + 1) / 2];
        _device.Download(b, raw);
        var u = MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
        for (long i = 0; i < count; i++) res[i] = BitConverter.UInt32BitsToSingle((uint)u[(int)i] << 16);
        return res;
    }

    private void TraceBuf(string name, VulkanDevice.Buffer b, int rows, int cols)
    {
        if (Trace is null) return;
        Flush();
        Trace(name, DownloadAct(b, (long)rows * cols), rows, cols);
    }

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, lastOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, lastOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, lastTokenLogitsOnly);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, bool lastOnly)
    {
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (positions[0] == 0 && _length != 0) ResetSequenceState();
        for (int i = 0; i < T; i++)
            if (positions[i] != _length + i)
                throw new ArgumentException($"olala keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {_length + i}.");
        if (_length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        for (int i = 0; i < T; i++)
            if ((uint)tokenIds[i] >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {tokenIds[i]} at position {i} is out of range [0, {_vocab}).");
        EnsureKvCapacity(_length + T);

        int rows = lastOnly ? 1 : T;
        var result = UnmanagedTensor.Allocate(new TensorShape(rows, _vocab), DType.Float32, deviceId);
        var dst = new Span<float>((void*)result.DataPointer, rows * _vocab);
        try
        {
            for (int c0 = 0; c0 < T; c0 += _chunk)
            {
                int Tc = Math.Min(_chunk, T - c0);
                bool lastChunk = c0 + Tc == T;
                int wantRows = lastOnly ? (lastChunk ? 1 : 0) : Tc;
                RunChunk(tokenIds.Slice(c0, Tc), _length + c0, wantRows, lastOnly, dst.Slice(lastOnly ? 0 : c0 * _vocab, wantRows * _vocab));
            }
        }
        catch { result.Dispose(); throw; }
        _length += T;
        return result;
    }

    private void RunChunk(ReadOnlySpan<int> ids, int basePos, int logitRows, bool lastOnly, Span<float> logitsDst)
    {
        int Tc = ids.Length;
        _device.Upload(MemoryMarshal.AsBytes(ids), _ids);
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(_cmd);
        _pendingW.Clear(); _pendingR.Clear();
        _spT0 = System.Diagnostics.Stopwatch.GetTimestamp();

        var embK = Kernel("emb", "olala_embed_" + Wt(_embed.Qt), 3, 2);
        Run(embK, 1u << 2, [(uint)_hidden, (uint)_vocab], (uint)Tc, 1, 1, _embed.Buf, _ids, _res);
        Sp("embed");
        TraceBuf("embed", _res, Tc, _hidden);

        foreach (var l in _layers)
        {
            if (l.IsSsm) RecordSsm(l, Tc);
            else RecordAttention(l, Tc, basePos);
            TraceBuf($"blk.{l.Index}.mixer_out", _mix, Tc, _hidden);
            Geodesic(_res, _mix, Tc, l.Index, l.GeoMixScale, l.GeoMixBias);
            Sp("geodesic");
            TraceBuf($"blk.{l.Index}.post_mixer", _res, Tc, _hidden);
            RecordMoe(l, Tc);
            TraceBuf($"blk.{l.Index}.mlp_out", _mix, Tc, _hidden);
            Geodesic(_res, _mix, Tc, l.Index, l.GeoMlpScale, l.GeoMlpBias);
            Sp("geodesic");
            TraceBuf($"blk.{l.Index}.l_out", _res, Tc, _hidden);
        }

        VulkanDevice.Buffer? logits = null;
        if (logitRows > 0)
        {
            long need = (long)logitRows * _vocab * 4;
            if (_logits is null || _logitsBytes < need)
            {
                _logits?.Dispose();
                _logits = _device.Allocate(need); _logitsBytes = need;
            }
            logits = _logits;
            long xOff = lastOnly ? (long)(Tc - 1) * _hidden : 0;
            Gemv(_output, _res, logits, logitRows, f32Out: true, xOff: xOff);
            Sp("lm_head");
        }
        _submit.SubmitAndWait();
        if (logits is not null) _device.Download(logits, logitsDst);
        _pendingW.Clear(); _pendingR.Clear();
    }

    private void RecordSsm(Layer l, int T)
    {
        Gemv(l.SsmIn!, _res, _zxdt, T); Sp("M ssm_in");
        Gemv(l.SsmInDyn!, _res, _dyn, T); Sp("M ssm_in_dyn");
        var k = Kernel("mamba", "olala_mamba3_step", 5, 7, (uint)_N, (uint)_R);
        Run(k, (1u << 3) | (1u << 4), [(uint)T, (uint)_heads, (uint)_P, (uint)_N, (uint)_R, (uint)_A, F(_eps)], (uint)_heads, 1, 1,
            _zxdt, _dyn, l.Params!, l.SsmState!, _ssmY);
        Sp("M mamba3_step");
        TraceBuf($"blk.{l.Index}.ssm_y", _ssmY, T, _heads * _P);
        Gemv(l.MixerOut, _ssmY, _mix, T); Sp("M out_proj");
    }

    private void RecordAttention(Layer l, int T, int basePos)
    {
        Gemv(l.Q!, _res, _q, T);
        Gemv(l.WaK!, _res, _ak, T); Gemv(l.WaV!, _res, _av, T);
        Gemv(l.WbK!, _res, _bk, T); Gemv(l.WbV!, _res, _bv, T);
        Gemv(l.ShiftK!, _res, _sk, T); Gemv(l.ShiftV!, _res, _sv, T);
        Gemv(l.Lambda!, _res, _lam, T);
        Gemv(l.Gate!, _res, _gate, T);
        Sp("V proj gemvs (9)");
        var kv = Kernel("tpakv", "olala_tpa_kv", 10, 7);
        uint[] pc0 = [(uint)T, (uint)basePos, (uint)_nk, (uint)_hd, (uint)_rank, 0u, F(_eps)];
        Run(kv, (1u << 8) | (1u << 9), pc0, (uint)_nk, (uint)T, 1, _ak, _av, _bk, _bv, _sk, _sv, l.KNorm!, l.LastKv!, l.KCache!, l.VCache!);
        uint[] pc1 = [(uint)T, (uint)basePos, (uint)_nk, (uint)_hd, (uint)_rank, 1u, F(_eps)];
        Run(kv, (1u << 7), pc1, (uint)_nk, 1, 1, _ak, _av, _bk, _bv, _sk, _sv, l.KNorm!, l.LastKv!, l.KCache!, l.VCache!);
        Sp("V tpa_kv");
        var at = Kernel("tpaattn", "olala_tpa_attn", 8, 9);
        Run(at, 1u << 7, [(uint)T, (uint)basePos, (uint)_nk, (uint)_hd, (uint)_group, _window, F(_softcap), F(_o.GateBias), F(_eps)],
            (uint)_nk, (uint)T, 1, _q, l.KCache!, l.VCache!, _lam, _gate, l.QNorm!, l.Scaler!, _attnY);
        Sp("V tpa_attn");
        TraceBuf($"blk.{l.Index}.attn_y", _attnY, T, _nk * _snr * _hd);
        Gemv(l.MixerOut, _attnY, _mix, T); Sp("V out_proj");
    }

    private void RecordMoe(Layer l, int T)
    {
        var route = Kernel("route", "olala_moe_route", 5, 4);
        Run(route, (1u << 3) | (1u << 4), [(uint)_hidden, (uint)_experts, (uint)_topk, F(_o.ExpertWeightsScale)], (uint)T, 1, 1,
            l.RouterW, _res, l.ExpertBias, _sel, _selW);
        Sp("moe route");
        Gemv(l.LatentDown, _res, _x0, T); Sp("moe latent_down");
        var up = Kernel("moeup|" + Wt(l.UpExps.Qt), "olala_moe_up_" + Wt(l.UpExps.Qt), 4, 3);
        Run(up, 1u << 2, [(uint)_ff, (uint)_lat, (uint)_topk], (uint)_ff, (uint)(T * _topk), 1, l.UpExps.Buf, _x0, _h, _sel);
        Sp("moe up_exps");
        var down = Kernel("moedown|" + Wt(l.DownExps.Qt), "olala_moe_down_" + Wt(l.DownExps.Qt), 5, 3);
        Run(down, 1u << 2, [(uint)_lat, (uint)_ff, (uint)_topk], (uint)_lat, (uint)T, 1, l.DownExps.Buf, _h, _routedLat, _sel, _selW);
        Sp("moe down_exps");
        Gemv(l.LatentUp, _routedLat, _routed, T); Sp("moe latent_up");
        Gemv(l.SharedUp, _res, _sh, T, epi: 1); Sp("moe shared_up");
        Gemv(l.SharedDown, _sh, _mix, T, epi: 2, aux: _routed); Sp("moe shared_down");
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <summary>
    /// Engine-facing KV-cache handle. The model owns its sequence state (recurrent state + BF16 K/V), like the CPU model, so the handle only
    /// reports the sequence length; it holds no tensors.
    /// </summary>
    /// <param name="maxLength">Requested capacity (informational).</param>
    public IKvCache CreateKvCacheHandle(int maxLength) => new OlalaKvHandle(this, Math.Min(maxLength, Config.MaxSequenceLength));

    private sealed class OlalaKvHandle(VulkanOlalaTransformerModel model, int max) : IKvCache
    {
        public int CurrentLength => model._length;
        public int MaxLength => max;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public ITensor GetValues(int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException("olala owns its K/V state.");
        public void Rollback(int length)
        {
            if (length != model._length) throw new NotSupportedException("olala cannot roll its recurrent state back (prefix snapshots are not implemented, #1969).");
        }
        public bool CanRollbackAfterAppend(int minPos, int maxPos) => false;
        public bool SupportsAbortRollback => false;
        public void Dispose() { }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var k in _kernels.Values) k.Dispose();
        _kernels.Clear();
        _submit.Dispose();
        _logits?.Dispose();
        foreach (var b in _owned) b.Dispose();
        _owned.Clear();
        foreach (var l in _layers) DisposeLayer(l);
        _embed.Buf.Dispose();
        if (!ReferenceEquals(_output, _embed)) _output.Buf.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void DisposeLayer(Layer l)
    {
        foreach (var m in new[] { l.MixerOut, l.LatentDown, l.LatentUp, l.SharedUp, l.SharedDown, l.UpExps, l.DownExps, l.SsmIn, l.SsmInDyn,
                                  l.Q, l.WaK, l.WaV, l.WbK, l.WbV, l.ShiftK, l.ShiftV, l.Lambda, l.Gate })
            m?.Buf.Dispose();
        foreach (var b in new[] { l.RouterW, l.ExpertBias, l.Params, l.QNorm, l.KNorm, l.Scaler })
            b?.Dispose();
    }
}
