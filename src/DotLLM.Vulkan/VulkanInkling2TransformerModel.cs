using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

/// <summary>Receives <c>(name, data, rows, cols)</c> activations from <see cref="VulkanInkling2TransformerModel"/>'s trace hook (tests / bisection).</summary>
internal delegate void Inkling2VkTraceSink(string name, float[] data, int rows, int cols);

/// <summary>Storage format of the Inkling 2 K/V cache on the GPU (user ruling 2026-11-26: F16 default, q8_0 evaluated).</summary>
public enum Inkling2KvFormat
{
    /// <summary>IEEE half (2 bytes per element). The default.</summary>
    F16,

    /// <summary>ggml q8_0 blocks (fp16 scale + 32 int8 = 34 bytes per 32 elements).</summary>
    Q8_0,
}

/// <summary>Storage format of the Inkling 2 Vulkan activations (stored scratch buffers; accumulation is F32 in every arm). BF16 is the shipped default.</summary>
public enum Inkling2ActFormat
{
    /// <summary>bfloat16 (the shipped default, user ruling 2026-11-26).</summary>
    Bf16,

    /// <summary>IEEE half: measurement arm of the activation-precision study (epic #2025); overflows to infinity above 65,504.</summary>
    F16,

    /// <summary>F32: measurement arm / control (CLAUDE.md precision rule 1 sign-off needed to ship).</summary>
    F32,
}

/// <summary>Switches of the Inkling 2 Vulkan forward. The defaults are the shipped configuration.</summary>
/// <param name="ActivationF32">
/// Build the F32-activation-STORAGE reference arm (the <c>_f32act</c> shaders): a control that is compared against the CPU reference without llama.cpp's noise,
/// never the default (CLAUDE.md precision rule 1).
/// </param>
/// <param name="Kv">K/V cache format.</param>
/// <param name="MaxChunk">Tokens per submission (prefill chunk). Local layers keep a ring of <c>window + MaxChunk</c> positions (see remarks of the model).</param>
/// <param name="ActivationF16">Build the F16-activation-storage MEASUREMENT arm (the <c>_f16act</c> shaders; activation-precision study, epic #2025). Ignored when <paramref name="ActivationF32"/> is set.</param>
/// <param name="F32ImageChunks">
/// Measurement arm: a prefill chunk that carries image rows runs with F32 activation scratch (a second scratch set, allocated on the first such chunk) while text chunks and decode keep the base format.
/// Nothing activation-typed is carried between chunks (the conv state is BF16 in every arm, the KV is F16), so the switch is per chunk.
/// </param>
public sealed record Inkling2VulkanOptions(bool ActivationF32 = false, Inkling2KvFormat Kv = Inkling2KvFormat.F16, int MaxChunk = 128, bool ActivationF16 = false, bool F32ImageChunks = false)
{
    /// <summary>The base activation storage format of this configuration.</summary>
    public Inkling2ActFormat BaseAct => ActivationF32 ? Inkling2ActFormat.F32 : ActivationF16 ? Inkling2ActFormat.F16 : Inkling2ActFormat.Bf16;

    /// <summary>Options from <c>DOTLLM_INKLING2_KV</c> (<c>f16|q8_0</c>); unset keeps the defaults, a bad value throws.</summary>
    public static Inkling2VulkanOptions FromEnvironment()
    {
        var o = new Inkling2VulkanOptions();
        string? kv = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_KV");
        if (!string.IsNullOrEmpty(kv))
            o = o with
            {
                Kv = kv.ToLowerInvariant() switch
                {
                    "f16" => Inkling2KvFormat.F16,
                    "q8_0" or "q8" => Inkling2KvFormat.Q8_0,
                    _ => throw new ArgumentException($"DOTLLM_INKLING2_KV='{kv}': expected f16 or q8_0."),
                },
            };
        string? ch = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_VK_CHUNK");
        if (!string.IsNullOrEmpty(ch)) o = o with { MaxChunk = int.Parse(ch) };
        // Measurement arms (opt-in; the default stays BF16): DOTLLM_INKLING2_ACT=bf16|f16|f32, DOTLLM_INKLING2_ACT_F32_IMAGE_CHUNKS=1.
        string? act = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_ACT");
        if (!string.IsNullOrEmpty(act))
            o = act.ToLowerInvariant() switch
            {
                "bf16" => o,
                "f16" => o with { ActivationF16 = true },
                "f32" => o with { ActivationF32 = true },
                _ => throw new ArgumentException($"DOTLLM_INKLING2_ACT='{act}': expected bf16, f16 or f32."),
            };
        if (Environment.GetEnvironmentVariable("DOTLLM_INKLING2_ACT_F32_IMAGE_CHUNKS") == "1") o = o with { F32ImageChunks = true };
        return o;
    }
}

/// <summary>
/// Vulkan forward of Thinking Machines Inkling 2 (<c>inkling2_loop</c>, Inkling-2-Mini; issue #2034, epic #2025). CPU reference:
/// <see cref="Inkling2TransformerModel"/>; layer-by-layer semantics and the oracle in <c>docs/INKLING2.md</c>.
/// </summary>
/// <remarks>
/// <para><b>Precision (user rulings 2026-11-26).</b> Stored activations are BF16 and every kernel accumulates and reduces in F32 registers / shared memory,
/// rounding once on store. KV is F16 (q8_0 evaluated, <see cref="Inkling2KvFormat"/>); the short-conv state is BF16; the loop-controller matrices are BF16
/// (converted once at load from the GGUF's F32: 57 MB). Quantized weights run quantized: NVFP4 expert banks (with their per-expert F32 scale) and Q8_0 matrices
/// are read in place by the kernels (decoded in registers); the router, norm gains, queries, clips and conv taps are the GGUF's own F32 vectors, uploaded
/// verbatim. Nothing is widened. The only F32 stored on the device are those weight vectors and the logits.</para>
/// <para><b>Ring KV.</b> Local (sliding-window) layers keep <c>window + MaxChunk</c> positions per cache slot (1024 + 128 = 1152), flat in the sequence length.
/// The 1024-position window needs <c>window + T - 1</c> slots for a prefill chunk of T tokens: a chunk writes all its K/V before attending, so a ring of exactly
/// <c>window</c> positions would let a later token of the chunk overwrite a key an earlier token still reads. Global layers are linear and grow by doubling.
/// Each of the 48 cache slots (24 physical layers + 2 extra passes of the 12-layer core) has its own K/V and four conv states.</para>
/// <para>The model owns its sequence state like the CPU model: positions must continue it, and a call at position 0 restarts it. One command buffer per chunk
/// of <see cref="Inkling2VulkanOptions.MaxChunk"/> tokens.</para>
/// </remarks>
public sealed unsafe class VulkanInkling2TransformerModel : IModel, IImageInputModel, IImageEncoderProvider
{
    private sealed class Mat
    {
        public required VulkanDevice.Buffer Buf;
        public required QuantizationType Qt;
        public required int In, Out;
    }

    private sealed class LoopW
    {
        public required VulkanDevice.Buffer MsgQuery;           // BF16 [4][H]
        public required Mat[] Down, Gain, Forget;
    }

    private sealed class Layer
    {
        public required bool Local;
        public required int Extent;
        public required Mat Wq, Wk, Wv, Wo, Gate, Rel, Sink;
        public Mat? KConvProj, VConvProj;
        public required VulkanDevice.Buffer QClip, RelClip, RelBias, KConv, VConv, PostConv, FfnPostConv, FfnResQuery, FfnNorm, ExpBias, Router;
        public VulkanDevice.Buffer? AttnNorm, QInpNorm, KInpNorm, VInpNorm, AttnResQuery;
        public required VulkanDevice.Buffer GateUp, Down;
        public required QuantizationType GateUpQt, DownQt;
        public VulkanDevice.Buffer? GateUpScale, DownScale;
        public LoopW? Loop;
    }

    private sealed class Slot
    {
        public required VulkanDevice.Buffer KConv, VConv, AttnConv, FfnConv;
        public required bool Local;
        public VulkanDevice.Buffer? K, V;
        public int Capacity;   // positions
    }

    private readonly VulkanDevice _device;
    private readonly string _spvDir;
    private readonly string? _spvOverride;
    private readonly GgufFile _gguf;
    private readonly Inkling2Config _c;
    private readonly Inkling2VulkanOptions _opt;
    private readonly Layer[] _layers;
    private readonly Slot[] _slots;
    private readonly Mat _embedMat, _output;
    private readonly VulkanDevice.Buffer _tokEmbNorm, _outputNorm, _outputResQuery;
    private readonly int _H, _vocab, _nh, _nkv, _hd, _nPhys, _chunk, _ff, _topk, _experts, _qW, _kvW, _relLen, _ringCap, _maxBlocks;
    private readonly float _eps;
    private readonly Dictionary<string, OlalaKernel> _kernels = new();
    private readonly VulkanDevice.Buffer _dummy;
    private readonly List<VulkanDevice.Buffer> _owned = new();
    private readonly VulkanDevice.SubmitContext _submit;
    private int _length;
    private long _deviceBytes, _weightBytes, _controllerBytes, _convBytes;

    // activation scratch (sized for _chunk tokens)
    private VulkanDevice.Buffer _x0 = null!, _x1 = null!, _pre = null!, _bank0 = null!, _bank1 = null!, _blk = null!, _mixed = null!, _mixed2 = null!, _msg = null!;
    private VulkanDevice.Buffer _n0 = null!, _n1 = null!, _n2 = null!, _gateIn = null!, _g64 = null!, _gainPre = null!, _forgetPre = null!;
    private VulkanDevice.Buffer _q = null!, _kRaw = null!, _vRaw = null!, _kc = null!, _vc = null!, _rel = null!, _sink = null!, _gateB = null!, _kDyn = null!, _vDyn = null!;
    private VulkanDevice.Buffer _attn = null!, _attnOut = null!, _ffnInp = null!, _mn = null!, _moe = null!, _h = null!, _sel = null!, _selW = null!, _ids = null!;
    private VulkanDevice.Buffer? _logits;
    private long _logitsBytes;
    // activation format of the scratch currently in use; sets of the other formats are kept once built (F32ImageChunks switches per chunk)
    private Inkling2ActFormat _act;
    private readonly Dictionary<Inkling2ActFormat, VulkanDevice.Buffer[]> _scratchSets = new();
    private readonly Dictionary<Inkling2ActFormat, long> _scratchSetBytes = new();
    // image-embedding injection (#2062), allocated on the first image prefill: F16 rows of the chunk's image positions (compact) and the per-token row selector.
    private VulkanDevice.Buffer? _imgRows, _imgSel;
    private ushort[]? _imgRowsHost;
    private uint[]? _imgSelHost;

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

    /// <summary>The options this model was built with.</summary>
    public Inkling2VulkanOptions Options => _opt;

    /// <summary>Test/diagnostic hook: receives named intermediates (null = off). Setting it splits the forward into several submissions.</summary>
    internal Inkling2VkTraceSink? Trace { get; set; }

    /// <summary>How the weights reached the device (DirectStorage / zero-copy import / staged host bounce), for the PR's direct-transfer evidence.</summary>
    public string WeightLoadReport { get; private set; } = "";

    /// <summary>Bytes of the loop-controller matrices as stored on the device (BF16; the GGUF's F32 original is 113.7 MB).</summary>
    public long LoopControllerBytes => _controllerBytes;

    /// <summary>Bytes of the four short-conv states of all slots (BF16; constant in the sequence length).</summary>
    public long ConvStateBytes => _convBytes;

    /// <summary>Allocated K/V bytes right now (local slots: the flat ring; global slots: the current capacity).</summary>
    public long KvBytes => _slots.Sum(s => (s.K?.Size ?? 0) + (s.V?.Size ?? 0));

    /// <summary>K/V bytes per cached position of one slot (K and V together): the marginal cost of a token on a GLOBAL slot.</summary>
    public long KvBytesPerPosition => 2L * KvRowBytes * _nkv;

    private long KvRowBytes => _opt.Kv == Inkling2KvFormat.F16 ? _hd * 2L : (_hd / 32) * 34L;

    /// <summary>Number of global (linear-KV) cache slots.</summary>
    public int GlobalSlots => _slots.Count(s => !s.Local);

    /// <summary>Number of local (ring-KV) cache slots.</summary>
    public int LocalSlots => _slots.Count(s => s.Local);

    /// <summary>Positions a local slot keeps (window + chunk slack).</summary>
    public int RingPositions => _ringCap;

    private VulkanInkling2TransformerModel(VulkanDevice device, string spvDir, GgufFile gguf, ModelConfig config, Inkling2VulkanOptions opt, Layer[] layers,
                                           Mat embed, Mat output, VulkanDevice.Buffer tokEmbNorm, VulkanDevice.Buffer outputNorm, VulkanDevice.Buffer outputResQuery,
                                           long weightBytes, long controllerBytes)
    {
        _device = device; _spvDir = spvDir; _gguf = gguf; Config = config; _opt = opt; _layers = layers; _embedMat = embed; _output = output;
        _tokEmbNorm = tokEmbNorm; _outputNorm = outputNorm; _outputResQuery = outputResQuery;
        _spvOverride = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");
        _c = config.Inkling2!;
        _H = config.HiddenSize; _vocab = config.VocabSize; _nh = config.NumAttentionHeads; _nkv = config.NumKvHeads; _hd = config.HeadDim;
        _nPhys = config.NumLayers; _eps = config.NormEpsilon; _chunk = Math.Max(1, opt.MaxChunk); _act = opt.BaseAct;
        _ff = _c.ExpertFfLength; _topk = _c.ExpertUsedCount; _experts = _c.ExpertCount; _qW = _nh * _hd; _kvW = _nkv * _hd; _relLen = _c.RelativeLength;
        _ringCap = _c.SlidingWindow + _chunk;
        _weightBytes = weightBytes; _controllerBytes = controllerBytes; _deviceBytes = weightBytes;
        _maxBlocks = CountBlockSlots();
        _dummy = Own(device.AllocateDeviceLocal(256));
        _submit = device.CreateSubmitContext();
        _slots = new Slot[_c.CacheSlots(_nPhys)];
        AllocateScratch();
        AllocateSlots();
    }

    private VulkanDevice.Buffer Own(VulkanDevice.Buffer b) { _owned.Add(b); return b; }

    private VulkanDevice.Buffer Scratch(long elements)
    {
        var b = Own(_device.AllocateDeviceLocal(Math.Max(16, elements * ActBytes)));
        _deviceBytes += b.Size;
        return b;
    }

    private int ActBytes => _act == Inkling2ActFormat.F32 ? 4 : 2;

    /// <summary>Kernel-name suffix of the current activation format (empty = BF16).</summary>
    private string ActSuffix => _act switch { Inkling2ActFormat.F32 => "_f32act", Inkling2ActFormat.F16 => "_f16act", _ => "" };

    /// <summary>Device bytes of the activation scratch of one format (0 until that set is built).</summary>
    public long ActivationScratchBytes(Inkling2ActFormat f) => _scratchSetBytes.TryGetValue(f, out long b) ? b : 0;

    /// <summary>The activation format of the scratch of the last chunk.</summary>
    public Inkling2ActFormat CurrentActivationFormat => _act;

    private ref VulkanDevice.Buffer ScratchField(int i)
    {
        switch (i)
        {
            case 0: return ref _x0; case 1: return ref _x1; case 2: return ref _pre; case 3: return ref _bank0; case 4: return ref _bank1; case 5: return ref _blk;
            case 6: return ref _mixed; case 7: return ref _mixed2; case 8: return ref _msg; case 9: return ref _n0; case 10: return ref _n1; case 11: return ref _n2;
            case 12: return ref _gateIn; case 13: return ref _g64; case 14: return ref _gainPre; case 15: return ref _forgetPre; case 16: return ref _q; case 17: return ref _kRaw;
            case 18: return ref _vRaw; case 19: return ref _kc; case 20: return ref _vc; case 21: return ref _rel; case 22: return ref _sink; case 23: return ref _gateB;
            case 24: return ref _kDyn; case 25: return ref _vDyn; case 26: return ref _attn; case 27: return ref _attnOut; case 28: return ref _ffnInp; case 29: return ref _mn;
            case 30: return ref _moe; case 31: return ref _h; case 32: return ref _selW;   // _sel (int ids) and _ids (token ids) are not activation-typed: shared by every set
            default: throw new ArgumentOutOfRangeException(nameof(i));
        }
    }
    private const int ScratchFieldCount = 33;

    /// <summary>Makes <paramref name="f"/> the activation format of the following chunk: swaps in that format's scratch set (building it on first use) and kernel family.</summary>
    private void UseActivationFormat(Inkling2ActFormat f)
    {
        if (f == _act) return;
        var cur = new VulkanDevice.Buffer[ScratchFieldCount];
        for (int i = 0; i < ScratchFieldCount; i++) cur[i] = ScratchField(i);
        _scratchSets[_act] = cur;
        _act = f;
        if (_scratchSets.TryGetValue(f, out var set)) { for (int i = 0; i < ScratchFieldCount; i++) ScratchField(i) = set[i]; }
        else AllocateScratch();
    }

    private int CountBlockSlots()
    {
        int bs = _c.AttnResBlockSize, start = _c.LoopStart, end = _c.LoopEnd, passes = _c.LoopPasses;
        int pre = 0, core = 0, post = 0;
        int coreStart = passes > 1 ? start : _nPhys;
        for (int il = 0; il < coreStart; il++) if (il % bs == 0) pre++;
        if (passes > 1)
        {
            for (int ip = start; ip < end; ip++) if (ip % bs == 0) core++;
            for (int il = end; il < _nPhys; il++) if (il % bs == 0) post++;
        }
        return pre + passes * core + post;
    }

    private void AllocateScratch()
    {
        long before = _deviceBytes;
        AllocateScratchBuffers();
        _scratchSetBytes[_act] = _deviceBytes - before;
    }

    private void AllocateScratchBuffers()
    {
        int T = _chunk, H = _H;
        _x0 = Scratch((long)T * H); _x1 = Scratch((long)T * H); _pre = Scratch((long)T * H); _bank0 = Scratch((long)T * H); _bank1 = Scratch((long)T * H);
        _blk = Scratch((long)_maxBlocks * T * H); _mixed = Scratch(3L * T * H); _mixed2 = Scratch((long)T * H); _msg = Scratch(4L * T * H);
        _n0 = Scratch((long)T * H); _n1 = Scratch((long)T * H); _n2 = Scratch((long)T * H);
        _gateIn = Scratch(2L * T * H); _g64 = Scratch((long)T * _c.LoopGateRank); _gainPre = Scratch((long)T * H); _forgetPre = Scratch((long)T * H);
        _q = Scratch((long)T * _qW); _kRaw = Scratch((long)T * _kvW); _vRaw = Scratch((long)T * _kvW); _kc = Scratch((long)T * _kvW); _vc = Scratch((long)T * _kvW);
        _rel = Scratch((long)T * _nh * _relLen); _sink = Scratch((long)T * _nh); _gateB = Scratch((long)T * _qW);
        _kDyn = Scratch((long)T * _c.ConvKernel * _nkv); _vDyn = Scratch((long)T * _c.ConvKernel * _nkv);
        _attn = Scratch((long)T * _qW); _attnOut = Scratch((long)T * H); _ffnInp = Scratch((long)T * H); _mn = Scratch((long)T * H); _moe = Scratch((long)T * H);
        _h = Scratch((long)T * _topk * _ff);
        if (_sel is null) { _sel = Own(_device.AllocateDeviceLocal(Math.Max(16, (long)T * _topk * 4))); _deviceBytes += _sel.Size; }
        _selW = Scratch((long)T * _topk);
        _ids ??= Own(_device.Allocate(Math.Max(16, (long)T * 4)));
    }

    private void AllocateSlots()
    {
        int core = _c.CoreLayers;
        for (int s = 0; s < _slots.Length; s++)
        {
            int ip = s < _nPhys ? s : _c.LoopStart + (s - _nPhys) % core;   // physical layer that owns the slot
            bool local = _c.IsLocalLayer[ip];
            VulkanDevice.Buffer St(int channels)
            {
                var b = Own(_device.AllocateDeviceLocal((long)channels * (_c.ConvKernel - 1) * 2));
                _deviceBytes += b.Size; _convBytes += b.Size;
                return b;
            }
            var slot = new Slot { Local = local, KConv = St(_kvW), VConv = St(_kvW), AttnConv = St(_H), FfnConv = St(_H) };
            if (local) { Reserve(slot, _ringCap); }
            else Reserve(slot, Math.Min(Config.MaxSequenceLength, 1024));
            _slots[s] = slot;
        }
        ResetSequenceState();
    }

    /// <summary>Gives <paramref name="slot"/> a K/V capacity of <paramref name="positions"/>, preserving the history already stored (device copy, no host round trip).</summary>
    private void Reserve(Slot slot, int positions)
    {
        if (positions <= slot.Capacity) return;
        long bytes = (long)positions * _nkv * KvRowBytes;
        var k = _device.AllocateDeviceLocal(bytes); var v = _device.AllocateDeviceLocal(bytes);
        _deviceBytes += k.Size + v.Size;
        if (slot.K is { } ok)
        {
            long used = (long)slot.Capacity * _nkv * KvRowBytes;
            _device.CopyBufferRangeSynchronous(ok, k, 0, 0, (ulong)used);
            _device.CopyBufferRangeSynchronous(slot.V!, v, 0, 0, (ulong)used);
            _deviceBytes -= ok.Size + slot.V!.Size;
            ok.Dispose(); slot.V!.Dispose();
        }
        slot.K = k; slot.V = v; slot.Capacity = positions;
    }

    /// <inheritdoc/>
    public void ResetSequenceState()
    {
        foreach (var s in _slots)
        {
            _device.FillBufferSynchronous(s.KConv, 0); _device.FillBufferSynchronous(s.VConv, 0);
            _device.FillBufferSynchronous(s.AttnConv, 0); _device.FillBufferSynchronous(s.FfnConv, 0);
        }
        _length = 0;
    }

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Builds the Vulkan model from an opened GGUF (which must outlive the model; weights are imported in place where the policy allows).</summary>
    public static VulkanInkling2TransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
        => BuildFromGguf(device, gguf, config, spvDir, Inkling2VulkanOptions.FromEnvironment());

    /// <summary>Builds the Vulkan model with explicit options.</summary>
    public static VulkanInkling2TransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir, Inkling2VulkanOptions opt)
    {
        ArgumentNullException.ThrowIfNull(device); ArgumentNullException.ThrowIfNull(gguf); ArgumentNullException.ThrowIfNull(config); ArgumentNullException.ThrowIfNull(opt);
        if (config.Architecture != Architecture.Inkling2 || config.Inkling2 is not { } c)
            throw new ArgumentException("VulkanInkling2TransformerModel requires an Architecture.Inkling2 configuration.", nameof(config));
        if (!device.HasSubgroupArithmetic)
            throw new NotSupportedException("The Inkling 2 Vulkan kernels need subgroup arithmetic (VK_KHR_shader_subgroup_arithmetic); this device does not report it.");
        // inkling2_attn reduces each key's score over 16 lanes with subgroupShuffleXor (offsets 8..1), and wgSum keeps at most 32 subgroup partials:
        // both need a subgroup of at least 16 lanes. 0 = the probe did not run (not a known-bad width).
        if (device.SubgroupSize != 0 && device.SubgroupSize < 16)
            throw new NotSupportedException($"The Inkling 2 Vulkan kernels need a subgroup of at least 16 lanes; this device reports {device.SubgroupSize}.");
        if (!device.NumericFeatures.StorageBuffer16BitAccess)
            throw new NotSupportedException("The Inkling 2 Vulkan kernels store BF16 activations through 16-bit storage buffer access; this device does not report storageBuffer16BitAccess.");
        if (config.HeadDim != 128 || config.NumAttentionHeads != 3 * config.NumKvHeads || c.RelativeLength != 16 || c.ConvKernel != 4 || config.HiddenSize > 4096
            || config.HiddenSize % 8 != 0 || c.ExpertUsedCount > 16 || c.ExpertCount > 1024)
            throw new NotSupportedException($"The Inkling 2 Vulkan kernels are written for head_dim 128, 3 query heads per kv head, relative length 16, 4 conv taps and hidden <= 4096; got head_dim {config.HeadDim}, " +
                                            $"{config.NumAttentionHeads} q / {config.NumKvHeads} kv heads, relative length {c.RelativeLength}, {c.ConvKernel} taps, hidden {config.HiddenSize}.");
        if (opt.MaxChunk < 1) throw new ArgumentOutOfRangeException(nameof(opt), "MaxChunk must be >= 1.");

        var tensors = gguf.TensorsByName;
        var problems = Inkling2Tensors.FindProblems(tensors, config, out var consumedScales);
        if (problems.Count > 0)
            throw new InvalidDataException("inkling2 tensor table does not match the model contract: " + string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));
        Nvfp4Support.RejectUnappliedScales(tensors, consumedScales, "The inkling2 Vulkan loader");

        long bytes = 0, controllerBytes = 0;
        VulkanWeightImportPolicy.Reset();
        using var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024);

        Mat Weight(string name)
        {
            var d = tensors[name];
            var qt = d.QuantizationType;
            int inDim = (int)d.Shape[0];
            bool ok = qt switch { QuantizationType.Q8_0 => inDim % 32 == 0, QuantizationType.NVFP4 => inDim % 64 == 0, _ => false };
            if (!ok)
                throw new NotSupportedException($"inkling2 tensor {name} is {qt} with K = {inDim}: the Vulkan forward reads Q8_0 / NVFP4 weights in place and refuses to widen anything else (CLAUDE.md precision rule 2).");
            long outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
            if (d.Shape.Rank > 2) outDim *= d.Shape[2];   // expert banks: the rows of all experts back to back
            var buf = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), qt, (int)outDim, inDim, out long up);
            bytes += up;
            return new Mat { Buf = buf, Qt = qt, In = inDim, Out = (int)(d.Shape.Rank > 1 ? d.Shape[1] : 1) };
        }
        // An F32 tensor (norm gain, query, clip, conv taps, router, expert scale): uploaded verbatim from the mapped file, never narrowed or widened.
        VulkanDevice.Buffer F32(string name)
        {
            var d = tensors[name];
            if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"inkling2 tensor {name} is {d.QuantizationType}; the contract keeps it F32.");
            long n = 1; for (int i = 0; i < d.Shape.Rank; i++) n *= d.Shape[i];
            var buf = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), QuantizationType.F32, 1, checked((int)n), out long up);
            bytes += up;
            return buf;
        }
        // A loop-controller matrix: F32 in the GGUF, BF16 on the device (user ruling 2026-11-26), converted once through a transient native buffer.
        Mat Controller(string name)
        {
            var d = tensors[name];
            if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"inkling2 tensor {name} is {d.QuantizationType}; the contract keeps it F32.");
            int inDim = (int)d.Shape[0], outDim = d.Shape.Rank > 1 ? (int)d.Shape[1] : 1;
            long count = (long)inDim * outDim;
            var src = new ReadOnlySpan<float>((void*)gguf.TensorDataPointer(d), checked((int)count));
            long nb = count * 2;
            nint tmp = (nint)NativeMemory.AlignedAlloc((nuint)nb, 64);
            try
            {
                var u = new Span<ushort>((void*)tmp, checked((int)count));
                for (int i = 0; i < src.Length; i++) u[i] = Inkling2ConvState.ToBf16(src[i]);
                var buf = device.AllocateDeviceLocal(nb);
                staging.UploadBytes(tmp, nb, buf);
                bytes += nb; controllerBytes += nb;
                return new Mat { Buf = buf, Qt = QuantizationType.BF16, In = inDim, Out = outDim };
            }
            finally { NativeMemory.AlignedFree((void*)tmp); }
        }

        var embed = Weight(Inkling2Tensors.TokenEmbd);
        if (embed.Qt != QuantizationType.Q8_0) throw new NotSupportedException($"inkling2 token_embd is {embed.Qt}; the Vulkan embedding gather reads Q8_0.");
        var outputD = tensors[Inkling2Tensors.Output];
        if (outputD.QuantizationType != QuantizationType.Q8_0) throw new NotSupportedException($"inkling2 output.weight is {outputD.QuantizationType}; the Vulkan LM head reads Q8_0.");
        var output = Weight(Inkling2Tensors.Output);
        output.Out = config.VocabSize;   // only the unpadded rows are computed

        var layers = new Layer[config.NumLayers];
        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => Inkling2Tensors.Block(il, s);
            bool local = c.IsLocalLayer[il];
            var gu = Weight(B("ffn_gate_up_exps.weight")); var dn = Weight(B("ffn_down_exps.weight"));
            var layer = new Layer
            {
                Local = local, Extent = local ? c.SlidingWindow : c.RelativeExtent,
                Wq = Weight(B("attn_q.weight")), Wk = Weight(B("attn_k.weight")), Wv = Weight(B("attn_v.weight")), Wo = Weight(B("attn_output.weight")),
                Gate = Weight(B("attn_gate.weight")), Rel = Weight(B("attn_rel.weight")), Sink = Weight(B("attn_sink_proj.weight")),
                QClip = F32(B("attn_q_clip.weight")), RelClip = F32(B("attn_rel_clip.weight")), RelBias = F32(B("attn_rel_bias.weight")),
                KConv = F32(B("attn_k_conv.weight")), VConv = F32(B("attn_v_conv.weight")), PostConv = F32(B("attn_post_conv.weight")),
                FfnPostConv = F32(B("ffn_post_conv.weight")), FfnResQuery = F32(B("ffn_res_query.weight")), FfnNorm = F32(B("ffn_norm.weight")),
                ExpBias = F32(B("exp_probs_b.bias")), Router = F32(B("ffn_gate_inp.weight")),
                GateUp = gu.Buf, GateUpQt = gu.Qt, Down = dn.Buf, DownQt = dn.Qt,
            };
            if (tensors.ContainsKey(B("ffn_gate_up_exps.scale"))) layer.GateUpScale = F32(B("ffn_gate_up_exps.scale"));
            if (tensors.ContainsKey(B("ffn_down_exps.scale"))) layer.DownScale = F32(B("ffn_down_exps.scale"));
            if (local) layer.AttnNorm = F32(B("attn_norm.weight"));
            else
            {
                layer.QInpNorm = F32(B("attn_q_inp_norm.weight")); layer.KInpNorm = F32(B("attn_k_inp_norm.weight")); layer.VInpNorm = F32(B("attn_v_inp_norm.weight"));
                layer.KConvProj = Weight(B("attn_k_conv_proj.weight")); layer.VConvProj = Weight(B("attn_v_conv_proj.weight"));
            }
            if (il > 0) layer.AttnResQuery = F32(B("attn_res_query.weight"));
            if (c.LoopPasses > 1 && il >= c.LoopStart && il < c.LoopEnd)
            {
                var down = new Mat[4]; var gain = new Mat[4]; var forget = new Mat[4];
                for (int k = 0; k < 4; k++)
                {
                    string kind = Inkling2Tensors.CtrlKinds[k];
                    down[k] = Controller(B($"loop_{kind}_ctrl_down.weight"));
                    gain[k] = Controller(B($"loop_{kind}_ctrl_gain.weight"));
                    forget[k] = Controller(B($"loop_{kind}_ctrl_forget.weight"));
                }
                layer.Loop = new LoopW { MsgQuery = Controller(B("loop_msg_query.weight")).Buf, Down = down, Gain = gain, Forget = forget };
            }
            layers[il] = layer;
        }

        var tokNorm = F32(Inkling2Tensors.TokenEmbdNorm); var outNorm = F32(Inkling2Tensors.OutputNorm); var outRes = F32(Inkling2Tensors.OutputResQuery);
        var model = new VulkanInkling2TransformerModel(device, spvDir, gguf, config, opt, layers, embed, output, tokNorm, outNorm, outRes, bytes, controllerBytes);
        model.WeightLoadReport = $"direct(DirectStorage) {VulkanWeightImportPolicy.DirectTensorCount} tensors / {VulkanWeightImportPolicy.DirectBytes / 1e6:F1} MB; imported(zero-copy) {VulkanWeightImportPolicy.ImportedTensorCount} / {VulkanWeightImportPolicy.ImportedBytes / 1e6:F1} MB; " +
                                 $"staged(host bounce) {VulkanWeightImportPolicy.StagedTensorCount} / {VulkanWeightImportPolicy.StagedBytes / 1e6:F1} MB; controller BF16 narrowing copies {controllerBytes / 1e6:F1} MB (always staged); last fallback: '{VulkanWeightImportPolicy.LastFallbackReason}'";
        return model;
    }

    // ───────────────────────────── kernels & recording ─────────────────────────────

    private OlalaKernel Kernel(string key, string name, int buffers, int pushWords, bool hasF32Twin = true, params uint[] spec)
    {
        string full = name + (hasF32Twin ? ActSuffix : "");
        string k = key + "|" + full;
        if (!_kernels.TryGetValue(k, out var kern))
        {
            string dir = _spvDir;
            // Test-only: a directory holding mutated copies of individual shaders (the tracked .spv are never touched).
            if (!string.IsNullOrEmpty(_spvOverride) && File.Exists(Path.Combine(_spvOverride, full + ".spv"))) dir = _spvOverride;
            _kernels[k] = kern = new OlalaKernel(_device, dir, full, buffers, pushWords, spec);
        }
        return kern;
    }

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

    private static string Wt(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => "q8_0", QuantizationType.BF16 => "bf16",
        _ => throw new NotSupportedException($"inkling2 Vulkan has no dense GEMV for {qt}."),
    };

    internal const int PrefillQ8Cols = 16, PrefillQ8Rows = 2;

    /// <summary><c>y[T, M] = W[M, K] x[T, K]</c> through the Olala GEMV (Q8_0 or BF16 weights read in place, BF16 activations, F32 accumulation); <paramref name="f32Out"/> stores F32 logits.</summary>
    private void Gemv(Mat w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int T, long xOff = 0, bool f32Out = false)
    {
        bool q8Tile = T > 1 && w.Qt == QuantizationType.Q8_0;
        int nc = T == 1 ? 1 : q8Tile ? PrefillQ8Cols : 8;
        if ((xOff & 7) != 0 || (w.In & 7) != 0) throw new InvalidOperationException($"inkling2 gemv needs xOff and K to be multiples of 8 (xOff {xOff}, K {w.In}).");
        int rows = q8Tile ? PrefillQ8Rows : 1;
        var k = rows > 1
            ? Kernel($"gemv|{nc}|{(f32Out ? 1 : 0)}|{Wt(w.Qt)}|r{rows}", "olala_gemv_" + Wt(w.Qt), 5, 4, true, (uint)nc, 0u, f32Out ? 1u : 0u, (uint)rows)
            : Kernel($"gemv|{nc}|{(f32Out ? 1 : 0)}|{Wt(w.Qt)}", "olala_gemv_" + Wt(w.Qt), 5, 4, true, (uint)nc, 0u, f32Out ? 1u : 0u);
        uint M = (uint)w.Out;
        uint groups = (M + (uint)rows - 1) / (uint)rows;
        uint gx = Math.Min(groups, 32768u), gy = (groups + 32767u) / 32768u, gz = (uint)((T + nc - 1) / nc);
        var yAct = f32Out ? _dummy : y;
        var yf = f32Out ? y : _dummy;
        Run(k, (1u << 2) | (1u << 4), [M, (uint)w.In, (uint)T, (uint)xOff], gx, gy, gz, w.Buf, x, yAct, _dummy, yf);
    }

    private void Rms(VulkanDevice.Buffer x, long inOff, VulkanDevice.Buffer? gamma, VulkanDevice.Buffer y, long outOff, int T, float eps)
    {
        var k = Kernel("rms", "inkling2_rmsnorm", 3, 5);
        Run(k, 1u << 2, [(uint)_H, (uint)inOff, (uint)outOff, gamma is null ? 0u : 1u, F(eps)], (uint)T, 1, 1, x, gamma ?? _dummy, y);
    }

    /// <summary>Depth mix of <paramref name="cur"/> (row 0 at <paramref name="curOff"/>) against the stored block stack: output <c>[nq][T][H]</c> in <paramref name="dst"/>.</summary>
    private void Mix(IReadOnlyList<int> stack, VulkanDevice.Buffer cur, long curOff, VulkanDevice.Buffer query, bool queryBf16, int nq, int T, VulkanDevice.Buffer dst)
    {
        if (stack.Count + 1 > 6) throw new NotSupportedException($"inkling2 Vulkan depth mix supports up to 5 stored blocks, got {stack.Count}.");
        var k = Kernel("mix|" + (queryBf16 ? 1 : 0), "inkling2_mix", 4, 13, true, queryBf16 ? 1u : 0u);
        var pc = new uint[13];
        pc[0] = (uint)_H; pc[1] = (uint)T; pc[2] = (uint)nq; pc[3] = (uint)stack.Count; pc[4] = (uint)((long)_chunk * _H); pc[5] = (uint)curOff; pc[6] = F(1e-6f);
        for (int i = 0; i < stack.Count; i++) pc[7 + i] = (uint)stack[i];
        Run(k, 1u << 3, pc, (uint)T, 1, 1, _blk, cur, query, dst);
    }

    /// <summary>The loop controller <paramref name="ctrl"/> on <paramref name="cur"/>, followed by the gain-ed RMS norm: <paramref name="dst"/> = norm((cur - tanh(forget) cur) + tanh(gain) msg).</summary>
    private void Control(VulkanDevice.Buffer cur, long curOff, LoopW loop, int ctrl, VulkanDevice.Buffer gamma, VulkanDevice.Buffer dst, int T)
    {
        long msgOff = (long)ctrl * T * _H;
        var kin = Kernel("ctrlin", "inkling2_ctrl_in", 3, 4);
        Run(kin, 1u << 2, [(uint)_H, (uint)curOff, (uint)msgOff, F(1e-6f)], (uint)T, 1, 1, cur, _msg, _gateIn);
        Gemv(loop.Down[ctrl], _gateIn, _g64, T);
        Gemv(loop.Gain[ctrl], _g64, _gainPre, T);
        Gemv(loop.Forget[ctrl], _g64, _forgetPre, T);
        var kout = Kernel("ctrlout", "inkling2_ctrl_out", 6, 4);
        Run(kout, 1u << 5, [(uint)_H, (uint)curOff, (uint)msgOff, F(_eps)], (uint)T, 1, 1, cur, _msg, _gainPre, _forgetPre, gamma, dst);
    }

    private void Conv(VulkanDevice.Buffer x, VulkanDevice.Buffer w, VulkanDevice.Buffer? dyn, VulkanDevice.Buffer state, VulkanDevice.Buffer? add, VulkanDevice.Buffer y, int channels, int T)
    {
        var k = Kernel($"conv|{(dyn is null ? 0 : 1)}|{(add is null ? 0 : 1)}", "inkling2_conv", 6, 5, true, dyn is null ? 0u : 1u, add is null ? 0u : 1u);
        Run(k, (1u << 3) | (1u << 5), [(uint)channels, (uint)T, (uint)_nkv, 0u, 0u], (uint)((channels + 63) / 64), 1, 1, x, w, dyn ?? _dummy, state, add ?? _dummy, y);
    }

    // ───────────────────────────── trace / flush ─────────────────────────────

    private void Flush()
    {
        _submit.SubmitAndWait();
        _pendingW.Clear(); _pendingR.Clear();
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
    }

    private float[] DownloadAct(VulkanDevice.Buffer b, long count, long elemOffset = 0)
    {
        var res = new float[count];
        long total = elemOffset + count;
        if (ActBytes == 4)
        {
            var all = new float[total];
            _device.Download(b, all);
            Array.Copy(all, elemOffset, res, 0, count);
            return res;
        }
        var raw = new float[(total + 1) / 2];
        _device.Download(b, raw);
        var u = MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
        if (_act == Inkling2ActFormat.F16)
            for (long i = 0; i < count; i++) res[i] = (float)BitConverter.UInt16BitsToHalf(u[(int)(elemOffset + i)]);
        else
            for (long i = 0; i < count; i++) res[i] = BitConverter.UInt32BitsToSingle((uint)u[(int)(elemOffset + i)] << 16);
        return res;
    }

    private void TraceBuf(string name, VulkanDevice.Buffer b, int rows, int cols, long elemOffset = 0)
    {
        if (Trace is null) return;
        Flush();
        Trace(name, DownloadAct(b, (long)rows * cols, elemOffset), rows, cols);
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

    /// <summary>Builds the Vulkan ViT encoder (#2062 part B) on this model's device; <c>DOTLLM_VISION_DEVICE=cpu</c> is handled by the callers.</summary>
    public IImageEncoder CreateImageEncoder(string mmprojPath, ImageOverLimit overLimit)
        => VulkanInkling2VisionEncoder.Load(_device, mmprojPath, _spvDir, overLimit);

    /// <summary>
    /// Forward with image embeddings (issue #2062): the F16 rows of every <see cref="ImageSpan"/> replace the token embeddings at the positions they cover and
    /// bypass <c>token_embd_norm</c> (the <c>inkling2_embed_rows</c> kernel rounds them once to the BF16 activation type, after the token-embedding kernel has run);
    /// the token ids there are ignored. Spans are in absolute sequence positions, so a chunked prefill passes the same list for every chunk. The model owns its sequence
    /// state, so <paramref name="kvCache"/> is only the engine's handle and is not read. There is no LoRA or MTP/DSpark state: the callers refuse an adapter with images
    /// and the engine does not draft for an image request.
    /// </summary>
    public ITensor ForwardWithImages(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IReadOnlyList<ImageSpan> images, IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        ArgumentNullException.ThrowIfNull(images);
        for (int i = 0; i < images.Count; i++)
            if (images[i].Embeddings.Dimension != _H)
                throw new ArgumentException($"image embeddings are {images[i].Embeddings.Dimension}-wide, the model hidden size is {_H}.", nameof(images));
        return ForwardCore(tokenIds, positions, deviceId, lastTokenLogitsOnly, images);
    }

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, bool lastOnly, IReadOnlyList<ImageSpan>? images = null)
    {
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (positions[0] == 0 && _length != 0) ResetSequenceState();
        for (int i = 0; i < T; i++)
            if (positions[i] != _length + i)
                throw new ArgumentException($"inkling2 keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {_length + i}.");
        if (_length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        for (int i = 0; i < T; i++)
            if ((uint)tokenIds[i] >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {tokenIds[i]} at position {i} is out of range [0, {_vocab}).");
        foreach (var s in _slots)
            if (!s.Local && _length + T > s.Capacity) Reserve(s, (int)Math.Min(Config.MaxSequenceLength, Math.Max(_length + T, (long)s.Capacity * 2)));

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
                RunChunk(tokenIds.Slice(c0, Tc), _length + c0, wantRows, lastOnly, dst.Slice(lastOnly ? 0 : c0 * _vocab, wantRows * _vocab), images);
            }
        }
        catch { result.Dispose(); throw; }
        _length += T;
        return result;
    }

    /// <summary>
    /// Gathers the F16 rows of the image positions inside <c>[basePos, basePos + Tc)</c> into a compact host staging buffer and uploads it with the per-token
    /// selector (0xFFFFFFFF = token row). Returns false when no position of the chunk is an image position (the chunk then runs the unchanged text path).
    /// </summary>
    private bool StageImageRows(IReadOnlyList<ImageSpan> images, int basePos, int Tc)
    {
        int H = _H;
        _imgSelHost ??= new uint[_chunk];
        _imgRowsHost ??= new ushort[(long)_chunk * H];   // reused host staging (the encoder's F16 payload is copied, never widened)
        int n = 0;
        for (int t = 0; t < Tc; t++)
        {
            _imgSelHost[t] = 0xFFFFFFFFu;
            int pos = basePos + t;
            for (int i = 0; i < images.Count; i++)
            {
                var sp = images[i];
                if (pos < sp.Start || pos >= sp.End) continue;
                MemoryMarshal.Cast<Half, ushort>(sp.Embeddings.Rows.Slice((pos - sp.Start) * H, H)).CopyTo(_imgRowsHost.AsSpan(n * H, H));
                _imgSelHost[t] = (uint)n++;
                break;
            }
        }
        if (n == 0) return false;
        if (_imgRows is null)
        {
            _imgRows = Own(_device.Allocate((long)_chunk * H * 2));
            _imgSel = Own(_device.Allocate(Math.Max(16, (long)_chunk * 4)));
            _deviceBytes += _imgRows.Size + _imgSel.Size;
        }
        _device.Upload(MemoryMarshal.AsBytes(_imgRowsHost.AsSpan(0, n * H)), _imgRows);
        _device.Upload(MemoryMarshal.AsBytes(_imgSelHost.AsSpan(0, Tc)), _imgSel!);
        return true;
    }

    private VulkanDevice.Buffer PickOther(VulkanDevice.Buffer cur) => ReferenceEquals(cur, _x0) ? _x1 : _x0;

    private void RunChunk(ReadOnlySpan<int> ids, int basePos, int logitRows, bool lastOnly, Span<float> logitsDst, IReadOnlyList<ImageSpan>? images = null)
    {
        int Tc = ids.Length, H = _H;
        _device.Upload(MemoryMarshal.AsBytes(ids), _ids);
        bool hasImageRows = images is { Count: > 0 } && StageImageRows(images, basePos, Tc);
        UseActivationFormat(_opt.F32ImageChunks && hasImageRows ? Inkling2ActFormat.F32 : _opt.BaseAct);
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(_cmd);
        _pendingW.Clear(); _pendingR.Clear();

        var embK = Kernel("emb", "inkling2_embed", 4, 3);
        Run(embK, 1u << 3, [(uint)H, (uint)_vocab, F(_eps)], (uint)Tc, 1, 1, _embedMat.Buf, _ids, _tokEmbNorm, _x0);
        if (hasImageRows)
        {
            var rowsK = Kernel("embrows", "inkling2_embed_rows", 3, 1);
            Run(rowsK, 1u << 2, [(uint)H], (uint)Tc, 1, 1, _imgRows!, _imgSel!, _x0);
        }
        TraceBuf("embed", _x0, Tc, H);

        var stack = new List<int>();
        int nextSlot = 0;
        int start = _c.LoopStart, end = _c.LoopEnd, passes = _c.LoopPasses;
        int coreStart = passes > 1 ? start : _nPhys;
        var cur = _x0;

        for (int il = 0; il < coreStart; il++)
        {
            // The last prelude layer's output is the start state of every pass: keep it in its own buffer.
            var outBuf = passes > 1 && il == coreStart - 1 ? _pre : PickOther(cur);
            RecordLayer(il, il, cur, outBuf, null, Tc, basePos, stack, ref nextSlot);
            cur = outBuf;
            TraceBuf($"pre.l{il}", cur, Tc, H);
        }

        if (passes > 1)
        {
            var prelude = cur;
            int preludeBlocks = stack.Count;
            int core = end - start;
            var bank = new List<int>();
            VulkanDevice.Buffer? bankOut = null;
            for (int pass = 0; pass < passes; pass++)
            {
                if (pass > 0)
                {
                    bank.Clear();
                    for (int i = preludeBlocks + 1; i < stack.Count; i++) bank.Add(stack[i]);
                    stack.RemoveRange(preludeBlocks, stack.Count - preludeBlocks);
                    bankOut = cur;
                    cur = prelude;
                }
                for (int ip = start; ip < end; ip++)
                {
                    int slot = pass == 0 ? ip : _nPhys + (pass - 1) * core + (ip - start);
                    LoopW? loop = null;
                    if (pass > 0)
                    {
                        loop = _layers[ip].Loop!;
                        Mix(bank, bankOut!, 0, loop.MsgQuery, queryBf16: true, 4, Tc, _msg);
                    }
                    // The output of the last core layer of a pass (except the final pass) is the next pass's bank state: its own buffer.
                    VulkanDevice.Buffer outBuf = ip == end - 1 && pass < passes - 1 ? (pass == 0 ? _bank0 : _bank1) : PickOther(cur);
                    if (ReferenceEquals(outBuf, cur)) outBuf = PickOther(cur);
                    RecordLayer(slot, ip, cur, outBuf, pass > 0 ? loop : null, Tc, basePos, stack, ref nextSlot);
                    cur = outBuf;
                    TraceBuf($"pass{pass}.l{ip}", cur, Tc, H);
                }
            }
            // bankOut for pass p+1 is the buffer pass p ended in; bank0 / bank1 alternate so the previous pass's bank survives its successor's layers.
            for (int il = end; il < _nPhys; il++)
            {
                var outBuf = PickOther(cur);
                RecordLayer(il, il, cur, outBuf, null, Tc, basePos, stack, ref nextSlot);
                cur = outBuf;
                TraceBuf($"post.l{il}", cur, Tc, H);
            }
        }

        // Output: depth mix with output_res_query, gain-ed RMS norm, LM head (unpadded rows only), final logit softcap.
        Mix(stack, cur, 0, _outputResQuery, queryBf16: false, 1, Tc, _mixed2);
        Rms(_mixed2, 0, _outputNorm, _n0, 0, Tc, _eps);
        TraceBuf("result_norm", _n0, Tc, H);

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
            long xOff = lastOnly ? (long)(Tc - 1) * H : 0;
            Gemv(_output, _n0, logits, logitRows, xOff, f32Out: true);
            float cap = _c.FinalLogitSoftcap;
            if (cap != 0f)
            {
                var sc = Kernel("softcap", "inkling2_softcap", 1, 2, false);
                uint n = (uint)(logitRows * _vocab);
                Run(sc, 1u << 0, [n, F(cap)], (n + 255u) / 256u, 1, 1, logits);
            }
        }
        _submit.SubmitAndWait();
        if (logits is not null) _device.Download(logits, logitsDst);
        _pendingW.Clear(); _pendingR.Clear();
    }

    /// <summary>
    /// One layer visit (llama.cpp <c>graph::build_layer</c>; CPU: <c>ForwardLayer</c>): <paramref name="cur"/> in, <paramref name="outBuf"/> out. The block stack and
    /// <paramref name="nextSlot"/> follow the CPU's: a layer that starts an attention-residual block pushes the RMS-normalised copy of its input.
    /// </summary>
    private void RecordLayer(int slotIx, int ip, VulkanDevice.Buffer cur, VulkanDevice.Buffer outBuf, LoopW? loop, int T, int pos0, List<int> stack, ref int nextSlot)
    {
        var lw = _layers[ip];
        var ss = _slots[slotIx];
        int H = _H;
        bool hasMsg = loop is not null;
        long plane = (long)T * H;

        // Attention inputs: layer 0 reads the embedding; later layers mix the earlier blocks with cur (one query for local layers, Q / K / V for global ones).
        VulkanDevice.Buffer in0 = cur, in1 = cur, in2 = cur;
        long off0 = 0, off1 = 0, off2 = 0;
        if (ip > 0)
        {
            int nq = lw.Local ? 1 : 3;
            Mix(stack, cur, 0, lw.AttnResQuery!, queryBf16: false, nq, T, _mixed);
            in0 = in1 = in2 = _mixed;
            off0 = 0; off1 = lw.Local ? 0 : plane; off2 = lw.Local ? 0 : 2 * plane;
        }
        VulkanDevice.Buffer n0, n1, n2;
        if (!hasMsg)
        {
            if (lw.Local) { Rms(in0, off0, lw.AttnNorm, _n0, 0, T, _eps); n0 = n1 = n2 = _n0; }
            else
            {
                Rms(in0, off0, lw.QInpNorm, _n0, 0, T, _eps); Rms(in1, off1, lw.KInpNorm, _n1, 0, T, _eps); Rms(in2, off2, lw.VInpNorm, _n2, 0, T, _eps);
                n0 = _n0; n1 = _n1; n2 = _n2;
            }
        }
        else
        {
            Control(in0, off0, loop!, 0, lw.Local ? lw.AttnNorm! : lw.QInpNorm!, _n0, T);
            Control(in1, off1, loop!, 1, lw.Local ? lw.AttnNorm! : lw.KInpNorm!, _n1, T);
            Control(in2, off2, loop!, 2, lw.Local ? lw.AttnNorm! : lw.VInpNorm!, _n2, T);
            n0 = _n0; n1 = _n1; n2 = _n2;
        }

        // ── self-attention ──
        Gemv(lw.Wq, n0, _q, T); Gemv(lw.Wk, n1, _kRaw, T); Gemv(lw.Wv, n2, _vRaw, T);
        Gemv(lw.Rel, n0, _rel, T); Gemv(lw.Sink, n0, _sink, T); Gemv(lw.Gate, n0, _gateB, T);
        if (!lw.Local) { Gemv(lw.KConvProj!, n1, _kDyn, T); Gemv(lw.VConvProj!, n2, _vDyn, T); }
        Conv(_kRaw, lw.KConv, lw.Local ? null : _kDyn, ss.KConv, null, _kc, _kvW, T);
        Conv(_vRaw, lw.VConv, lw.Local ? null : _vDyn, ss.VConv, null, _vc, _kvW, T);

        bool kvq8 = _opt.Kv == Inkling2KvFormat.Q8_0;
        uint ringCap = lw.Local ? (uint)_ringCap : 0u;
        var prep = Kernel($"qkprep|{(kvq8 ? 1 : 0)}", "inkling2_qkprep", 8, 12, true, kvq8 ? 1u : 0u);
        Run(prep, (1u << 0) | (1u << 1) | (1u << 6) | (1u << 7),
            [(uint)T, (uint)pos0, ringCap, lw.Local ? 0u : 1u, F(_c.QkClip), F(_c.RelativeClip), F(_eps), F(_c.LogScaleFloor), F(_c.LogScaleAlpha), (uint)_nh, (uint)_nkv, (uint)_relLen],
            (uint)(_nh + 2 * _nkv + _nh), (uint)T, 1, _q, _rel, _kc, _vc, lw.QClip, lw.RelClip, ss.K!, ss.V!);

        var attn = Kernel($"attn|{(kvq8 ? 1 : 0)}", "inkling2_attn", 8, 8, true, kvq8 ? 1u : 0u);
        Run(attn, 1u << 7, [(uint)T, (uint)pos0, (uint)_nh, (uint)_nkv, lw.Local ? 1u : 0u, (uint)_c.SlidingWindow, ringCap, (uint)lw.Extent],
            (uint)_nkv, (uint)T, 1, _q, _rel, _sink, _gateB, ss.K!, ss.V!, lw.RelBias, _attn);
        Gemv(lw.Wo, _attn, _attnOut, T);

        // The first layer of each block starts a new residual stream (its input joins the mixed blocks); the others add the residual (fused into the conv).
        if (ip % _c.AttnResBlockSize == 0)
        {
            Conv(_attnOut, lw.PostConv, null, ss.AttnConv, null, _ffnInp, H, T);
            int slotId = nextSlot++;
            Rms(cur, 0, null, _blk, (long)slotId * _chunk * H, T, 1e-6f);
            stack.Add(slotId);
        }
        else
        {
            Conv(_attnOut, lw.PostConv, null, ss.AttnConv, cur, _ffnInp, H, T);
        }

        // ── MoE ──
        Mix(stack, _ffnInp, 0, lw.FfnResQuery, queryBf16: false, 1, T, _mixed2);
        if (hasMsg) Control(_mixed2, 0, loop!, 3, lw.FfnNorm, _mn, T);
        else Rms(_mixed2, 0, lw.FfnNorm, _mn, 0, T, _eps);

        var route = Kernel("route", "olala_moe_route", 5, 4);
        Run(route, (1u << 3) | (1u << 4), [(uint)H, (uint)_experts, (uint)_topk, F(_c.ExpertWeightsScale)], (uint)T, 1, 1, lw.Router, _mn, lw.ExpBias, _sel, _selW);

        string gt = lw.GateUpQt == QuantizationType.NVFP4 ? "nvfp4" : "q8_0", dt = lw.DownQt == QuantizationType.NVFP4 ? "nvfp4" : "q8_0";
        var up = Kernel($"moeup|{gt}|{(lw.GateUpScale is null ? 0 : 1)}", "inkling2_moe_gate_up_" + gt, 5, 5, true, lw.GateUpScale is null ? 0u : 1u);
        Run(up, 1u << 2, [(uint)_ff, (uint)H, (uint)_topk, F(_c.SituBeta), F(_c.SituLinearBeta)], (uint)_ff, (uint)(T * _topk), 1, lw.GateUp, _mn, _h, _sel, lw.GateUpScale ?? _dummy);
        var down = Kernel($"moedown|{dt}|{(lw.DownScale is null ? 0 : 1)}", "inkling2_moe_down_" + dt, 6, 3, true, lw.DownScale is null ? 0u : 1u);
        Run(down, 1u << 2, [(uint)H, (uint)_ff, (uint)_topk], (uint)H, (uint)T, 1, lw.Down, _h, _moe, _sel, _selW, lw.DownScale ?? _dummy);

        Conv(_moe, lw.FfnPostConv, null, ss.FfnConv, _ffnInp, outBuf, H, T);
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <summary>Engine-facing KV-cache handle. The model owns its sequence state (KV + conv state), like the CPU model, so the handle only reports the sequence length.</summary>
    /// <param name="maxLength">Requested capacity (informational).</param>
    public IKvCache CreateKvCacheHandle(int maxLength) => new Inkling2KvHandle(this, Math.Min(maxLength, Config.MaxSequenceLength));

    private sealed class Inkling2KvHandle(VulkanInkling2TransformerModel model, int max) : IKvCache
    {
        public int CurrentLength => model._length;
        public int MaxLength => max;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public ITensor GetValues(int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException("inkling2 owns its K/V state.");
        public void Rollback(int length)
        {
            if (length != model._length) throw new NotSupportedException("inkling2 cannot roll its recurrent conv state back (prefix snapshots are not implemented).");
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
        foreach (var s in _slots) { s.K?.Dispose(); s.V?.Dispose(); }
        foreach (var l in _layers) DisposeLayer(l);
        _embedMat.Buf.Dispose(); _output.Buf.Dispose();
        _tokEmbNorm.Dispose(); _outputNorm.Dispose(); _outputResQuery.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void DisposeLayer(Layer l)
    {
        foreach (var m in new[] { l.Wq, l.Wk, l.Wv, l.Wo, l.Gate, l.Rel, l.Sink, l.KConvProj, l.VConvProj }) m?.Buf.Dispose();
        foreach (var b in new[] { l.QClip, l.RelClip, l.RelBias, l.KConv, l.VConv, l.PostConv, l.FfnPostConv, l.FfnResQuery, l.FfnNorm, l.ExpBias, l.Router, l.AttnNorm, l.QInpNorm,
                                  l.KInpNorm, l.VInpNorm, l.AttnResQuery, l.GateUp, l.Down, l.GateUpScale, l.DownScale })
            b?.Dispose();
        if (l.Loop is { } lp)
        {
            lp.MsgQuery.Dispose();
            foreach (var m in lp.Down.Concat(lp.Gain).Concat(lp.Forget)) m.Buf.Dispose();
        }
    }
}
