using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>Receives <c>(name, devicePointer, rows, cols)</c> of a BF16 activation from the trace hook of <see cref="CudaOlalaTransformerModel"/> (tests / bisection).</summary>
internal delegate void CudaOlalaTraceSink(string name, nint deviceBf16, int rows, int cols);

/// <summary>
/// CUDA forward of OVHcloud Olala (<c>olala</c>, Olala-v1.0-mini-7A1B; issue #1984, epic #1969): 36 blocks of
/// <c>geodesic(mixer) → geodesic(latent MoE + shared expert)</c>; the mixer is a Mamba-3 MIMO layer (<c>M</c>) or differential
/// tensor-product attention (<c>V</c>). Semantics and the CPU reference: <c>docs/OLALA.md</c>, <see cref="OlalaTransformerModel"/>.
/// </summary>
/// <remarks>
/// <para><b>Precision (James, 2026-11-25).</b> Stored activations are BF16 (Olala activations reach ~1e9..1e14: FP16 overflows, F32
/// storage needs sign-off); accumulation is F32 in registers. The recurrent state (SSM matrix, rotary phase, trapezoid K/V, token-shift
/// K/V) is F32 (<see cref="OlalaStatePolicy"/>, sized by <see cref="CudaOlalaStateLayout"/>). Weights run quantized in place: Q8_0, Q6_K,
/// BF16 and the F32 router are read by the kernels directly, never widened. The router scores, top-K and routing weights stay F32
/// (6 floats per token) and are never rounded to BF16.</para>
/// <para>The model owns its sequence state, like the CPU model: positions must continue it, a call at position 0 restarts it, and an
/// external <see cref="IKvCache"/> is ignored (a <see cref="CudaHybridKvCacheHandle"/> is only a length handle).</para>
/// <para>Weights are uploaded from the memory-mapped GGUF with <c>cuMemcpyHtoD</c> from pageable memory, like the other CUDA loaders
/// (a disk-to-device direct route is a platform gap tracked under CLAUDE.md rule 3, not added here).</para>
/// </remarks>
public sealed unsafe class CudaOlalaTransformerModel : IModel
{
    /// <summary>Default tokens per internal prefill chunk (bounds the scratch buffers; the recurrence and the K/V history continue across chunks).</summary>
    internal const int DefaultPrefillChunk = 256;

    /// <summary>Tokens per internal prefill chunk (tests lower it to exercise the chunk boundaries).</summary>
    internal int ChunkTokens { get; set; } = DefaultPrefillChunk;

    private readonly record struct Mat(nint Ptr, OlalaWeightKind Kind, int In, int Out);

    private sealed class MoeW
    {
        public Mat LatentDown, LatentUp, SharedUp, SharedDown;
        public nint Router, ExpertBias;                    // F32 [E][H], F32 [E]
        public nint UpExps, DownExps;
        public OlalaWeightKind UpKind, DownKind;
        public long UpExpertBytes, DownExpertBytes;
    }

    private sealed class Blk
    {
        public int Index;
        public bool IsSsm;
        public float GeoMixScale, GeoMixBias, GeoMlpScale, GeoMlpBias;
        public Mat MixerOut;
        public MoeW Moe = null!;
        // M
        public Mat SsmIn, SsmInDyn;
        public nint BNorm, CNorm, BBias, CBias, DtBias, D, MimoX, MimoZ, MimoO;
        public nint Ssm, Angle, TrapK, TrapV;              // F32 state
        // V
        public Mat Q, WaK, WaV, WbK, WbV, ShiftK, ShiftV, Lambda, Gate;
        public nint QNorm, KNorm, Scaler;
        public nint LastK, LastV;                          // F32 token-shift carry
        public nint KCache, VCache;                        // BF16 history [capacity][kvW]
        public long Capacity;
    }

    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaOlalaKernels _k;
    private readonly OlalaConfig _o;
    private readonly Blk[] _blocks;
    private readonly List<nint> _allocs = [];
    private readonly Dictionary<string, (nint Ptr, long Bytes)> _scratch = [];
    private Mat _output;
    private nint _embed;
    private OlalaWeightKind _embedKind;
    private long _weightBytes, _stateBytes, _historyBytes;
    private int _length;
    private bool _disposed;
    private int _actBytes = CudaOlalaStateLayout.ActivationBytes;   // 2 (BF16); 4 only in the F32 measurement-control arm
    private readonly int _hidden, _vocab, _window;
    private readonly float _eps, _softcap;
    private readonly GpuSpillMonitor _spillMonitor = new(() => "CudaOlalaTransformerModel");

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Test/diagnostic hook: receives BF16 activations (null = off, zero cost).</summary>
    internal CudaOlalaTraceSink? Trace { get; set; }

    /// <summary>Tokens consumed by the model-owned sequence state.</summary>
    internal int Length => _length;

    /// <summary>Bytes of resident weights.</summary>
    public long DeviceWeightBytes => _weightBytes;

    /// <summary>Bytes of the F32 recurrent state plus the token-shift carry (constant in the sequence length).</summary>
    internal long RecurrentStateBytes => _stateBytes;

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _scratch.Values.Sum(b => b.Bytes) + _stateBytes + _historyBytes;

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    private CudaOlalaTransformerModel(ModelConfig config, CudaContext context, CudaStream stream, CudaOlalaKernels kernels)
    {
        Config = config; _context = context; _stream = stream; _k = kernels;
        _o = config.Olala!;
        _blocks = new Blk[config.NumLayers];
        _hidden = config.HiddenSize; _vocab = config.VocabSize;
        _window = config.SlidingWindowSize ?? int.MaxValue;
        _eps = config.NormEpsilon; _softcap = config.AttnLogitSoftcap ?? 0f;
    }

    /// <summary>Creates a length-only handle (the model owns its K/V and recurrent storage).</summary>
    public CudaHybridKvCacheHandle CreateKvCache(int maxSeqLen) => new(maxSeqLen);

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Loads an Olala GGUF (Q8_0 / Q6_K / BF16 / F32 tensors) onto a CUDA device. The GGUF must outlive the model.</summary>
    public static CudaOlalaTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
        => Load(gguf, config, deviceId, ptxDir, f32Activations: false);

    /// <summary>
    /// MEASUREMENT CONTROL ONLY (tests): the same forward with F32 stored activations (<c>olala_f32act.ptx</c>). Internal, never reached from the loader
    /// or the CLI; F32 activation storage is not approved for the shipped path (it needs sign-off).
    /// </summary>
    internal static CudaOlalaTransformerModel LoadF32ActivationControl(GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
        => Load(gguf, config, deviceId, ptxDir, f32Activations: true);

    private static CudaOlalaTransformerModel Load(GgufFile gguf, ModelConfig config, int deviceId, string? ptxDir, bool f32Activations)
    {
        ArgumentNullException.ThrowIfNull(gguf); ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Olala || config.Olala is not { } o)
            throw new ArgumentException("CudaOlalaTransformerModel requires an Architecture.Olala configuration.", nameof(config));
        ValidateGeometry(config, o);
        Nvfp4Support.RejectUnappliedScales(gguf.TensorsByName, consumedScales: null, "The CUDA olala loader");
        var problems = OlalaTensors.FindProblems(gguf.TensorsByName, config);
        if (problems.Count > 0)
            throw new InvalidDataException("olala tensor table does not match the model contract: " + string.Join("; ", problems.Take(8))
                                           + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));
        ptxDir = CudaKernels.ResolveAndValidatePtxDirectory(ptxDir);
        string ptxName = f32Activations ? "olala_f32act.ptx" : "olala.ptx";
        if (!File.Exists(Path.Combine(ptxDir, ptxName))) throw new FileNotFoundException($"CUDA Olala requires native/ptx/{ptxName}.");

        CudaContext? context = null; CudaStream? stream = null; CudaOlalaKernels? kernels = null; CudaOlalaTransformerModel? model = null;
        try
        {
            context = CudaContext.Create(deviceId); stream = CudaStream.Create();
            kernels = new CudaOlalaKernels(ptxDir, f32Activations);
            model = new CudaOlalaTransformerModel(config, context, stream, kernels) { _actBytes = f32Activations ? 4 : CudaOlalaStateLayout.ActivationBytes };
            model.Upload(gguf);
            return model;
        }
        catch
        {
            if (model is not null) model.Dispose();
            else { kernels?.Dispose(); stream?.Dispose(); context?.Dispose(); }
            throw;
        }
    }

    /// <summary>Geometry limits of the kernels (runtime-sized otherwise).</summary>
    internal static void ValidateGeometry(ModelConfig config, OlalaConfig o)
    {
        if (o.StateSize > 128 || o.StateSize % 2 != 0) throw new NotSupportedException($"CUDA olala supports state size N <= 128 (got {o.StateSize}).");
        if (o.MambaHeadDim > 64) throw new NotSupportedException($"CUDA olala supports mamba head dim P <= 64 (got {o.MambaHeadDim}).");
        if (o.MimoRank > 4) throw new NotSupportedException($"CUDA olala supports MIMO rank R <= 4 (got {o.MimoRank}).");
        if (o.NumRopeAngles > 64 || o.NumRopeAngles > o.StateSize / 2) throw new NotSupportedException($"CUDA olala: unsupported rope angle count {o.NumRopeAngles}.");
        if (config.HeadDim > 128) throw new NotSupportedException($"CUDA olala supports head_dim <= 128 (got {config.HeadDim}).");
        if (o.ExpertUsedCount > 16) throw new NotSupportedException($"CUDA olala supports <= 16 experts per token (got {o.ExpertUsedCount}).");
        if (o.SharedExpertFfLength <= 0) throw new NotSupportedException("CUDA olala requires the shared expert.");
        if (o.MoeLatentSize <= 0) throw new NotSupportedException("CUDA olala requires the latent MoE (moe_latent_size > 0).");
    }

    private static OlalaWeightKind KindOf(string name, QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => OlalaWeightKind.Q8_0,
        QuantizationType.Q6_K => OlalaWeightKind.Q6_K,
        QuantizationType.BF16 => OlalaWeightKind.BF16,
        QuantizationType.F32 => OlalaWeightKind.F32,
        _ => throw new NotSupportedException($"CUDA olala reads Q8_0 / Q6_K / BF16 / F32 weights in place; tensor {name} is {qt}. "
                                             + "Widening it is not allowed (CLAUDE.md precision rule 2); use the safe2 Q8_0 or the BF16 GGUF."),
    };

    private nint UploadBytes(nint host, long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        _allocs.Add(p); _weightBytes += bytes;
        CudaDriverApi.cuMemcpyHtoD_v2(p, host, (nuint)bytes).ThrowOnError();
        return p;
    }

    private nint Alloc(long bytes, bool zero, bool state = false)
    {
        bytes = Math.Max(bytes, 16);
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        _allocs.Add(p);
        if (state) _stateBytes += bytes;
        if (zero) CudaDriverApi.cuMemsetD8_v2(p, 0, (nuint)bytes).ThrowOnError();
        return p;
    }

    private void Upload(GgufFile gguf)
    {
        var tensors = gguf.TensorsByName;
        // Admission (CLAUDE.md rule 2): resident weights must fit free VRAM; refuse rather than spill into shared memory.
        long need = tensors.Values.Where(d => d.Name.StartsWith("blk.") || d.Name is OlalaTensors.TokenEmbd or OlalaTensors.Output)
                                  .Sum(d => TensorBytes(d));
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total).ThrowOnError();
        long reserve = 512L << 20;   // scratch, state, driver
        if (need + reserve > (long)free)
            throw new InvalidOperationException($"CUDA olala needs about {(need + reserve) / (1 << 20)} MiB of VRAM for the resident weights and scratch but only {(long)free / (1 << 20)} MiB of {(long)total / (1 << 20)} MiB are free. "
                                                + "The weights are never widened or spilled to host memory; free GPU memory or use the CPU backend.");

        Mat Mat_(string name)
        {
            var d = tensors[name];
            var kind = KindOf(name, d.QuantizationType);
            int inDim = (int)d.Shape[0], outDim = d.Shape.Rank > 1 ? (int)d.Shape[1] : 1;
            if (!CudaOlalaKernels.SupportsInDim(kind, inDim)) throw new NotSupportedException($"CUDA olala: tensor {name} has row length {inDim}, not a multiple of the {kind} block size.");
            if (Dequantize.RowByteSize(inDim, d.QuantizationType) != CudaOlalaKernels.RowBytes(kind, inDim)) throw new InvalidDataException($"row size mismatch for {name}");
            return new Mat(UploadBytes(gguf.TensorDataPointer(d), TensorBytes(d)), kind, inDim, outDim);
        }
        nint F32Dev(string name)
        {
            var d = tensors[name];
            if (d.QuantizationType != QuantizationType.F32)
                throw new InvalidDataException($"olala tensor {name} is {d.QuantizationType}; the contract keeps it F32 (biases, norms, MIMO projections, router, scalars).");
            return UploadBytes(gguf.TensorDataPointer(d), TensorBytes(d));
        }
        float Scalar(string name)
        {
            var d = tensors[name];
            if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"olala tensor {name} is {d.QuantizationType}; expected F32.");
            return *(float*)gguf.TensorDataPointer(d);
        }

        var emb = tensors[OlalaTensors.TokenEmbd];
        _embedKind = KindOf(OlalaTensors.TokenEmbd, emb.QuantizationType);
        if (_embedKind == OlalaWeightKind.Q6_K) throw new NotSupportedException("CUDA olala: a Q6_K token embedding is not supported.");
        if (!CudaOlalaKernels.SupportsInDim(_embedKind, _hidden)) throw new NotSupportedException("CUDA olala: embedding row length is not block aligned.");
        _embed = UploadBytes(gguf.TensorDataPointer(emb), TensorBytes(emb));
        _output = tensors.ContainsKey(OlalaTensors.Output) ? Mat_(OlalaTensors.Output) : new Mat(_embed, _embedKind, _hidden, _vocab);

        var ssmLayout = CudaOlalaStateLayout.Ssm(_o);
        var attnLayout = CudaOlalaStateLayout.Attn(_o, Config);
        for (int il = 0; il < _blocks.Length; il++)
        {
            string B(string s) => OlalaTensors.Block(il, s);
            var upD = tensors[B("ffn_up_exps.weight")]; var downD = tensors[B("ffn_down_exps.weight")];
            var upKind = KindOf(B("ffn_up_exps.weight"), upD.QuantizationType); var downKind = KindOf(B("ffn_down_exps.weight"), downD.QuantizationType);
            int lat = _o.MoeLatentSize, ff = _o.ExpertFfLength;
            if (!CudaOlalaKernels.SupportsInDim(upKind, lat) || !CudaOlalaKernels.SupportsInDim(downKind, ff))
                throw new NotSupportedException($"CUDA olala: expert row lengths {lat} / {ff} are not aligned to the {upKind} / {downKind} block size.");
            var routerD = tensors[B("ffn_gate_inp.weight")];
            if (routerD.QuantizationType != QuantizationType.F32)
                throw new NotSupportedException($"CUDA olala reads the router as F32 (the published GGUFs keep it F32); {B("ffn_gate_inp.weight")} is {routerD.QuantizationType}.");
            var moe = new MoeW
            {
                Router = UploadBytes(gguf.TensorDataPointer(routerD), TensorBytes(routerD)),
                ExpertBias = F32Dev(B("exp_probs_b")),
                LatentDown = Mat_(B("ffn_latent_down.weight")), LatentUp = Mat_(B("ffn_latent_up.weight")),
                SharedUp = Mat_(B("ffn_up_shexp.weight")), SharedDown = Mat_(B("ffn_down_shexp.weight")),
                UpExps = UploadBytes(gguf.TensorDataPointer(upD), TensorBytes(upD)), UpKind = upKind,
                DownExps = UploadBytes(gguf.TensorDataPointer(downD), TensorBytes(downD)), DownKind = downKind,
                UpExpertBytes = CudaOlalaKernels.RowBytes(upKind, lat) * ff,
                DownExpertBytes = CudaOlalaKernels.RowBytes(downKind, ff) * lat,
            };
            if (TensorBytes(upD) != moe.UpExpertBytes * _o.ExpertCount || TensorBytes(downD) != moe.DownExpertBytes * _o.ExpertCount)
                throw new InvalidDataException($"expert bank size mismatch in block {il}");
            var blk = new Blk
            {
                Index = il, IsSsm = _o.IsSsmLayer(il), Moe = moe,
                GeoMixScale = Scalar(B("geo_mixer_scale")), GeoMixBias = Scalar(B("geo_mixer_bias")),
                GeoMlpScale = Scalar(B("geo_mlp_scale")), GeoMlpBias = Scalar(B("geo_mlp_bias")),
                MixerOut = Mat_(B("attn_output.weight")),
            };
            if (blk.IsSsm)
            {
                blk.SsmIn = Mat_(B("ssm_in.weight")); blk.SsmInDyn = Mat_(B("ssm_in_dyn.weight"));
                blk.BNorm = F32Dev(B("ssm_b_norm.weight")); blk.CNorm = F32Dev(B("ssm_c_norm.weight"));
                blk.BBias = F32Dev(B("ssm_b_bias")); blk.CBias = F32Dev(B("ssm_c_bias"));
                blk.DtBias = F32Dev(B("ssm_dt_bias")); blk.D = F32Dev(B("ssm_d"));
                blk.MimoX = F32Dev(B("ssm_mimo_x")); blk.MimoZ = F32Dev(B("ssm_mimo_z")); blk.MimoO = F32Dev(B("ssm_mimo_o"));
                blk.Ssm = Alloc(ssmLayout.SsmBytes, true, true); blk.Angle = Alloc(ssmLayout.AngleBytes, true, true);
                blk.TrapK = Alloc(ssmLayout.TrapKBytes, true, true); blk.TrapV = Alloc(ssmLayout.TrapVBytes, true, true);
            }
            else
            {
                blk.Q = Mat_(B("attn_q.weight")); blk.WaK = Mat_(B("attn_wa_k.weight")); blk.WaV = Mat_(B("attn_wa_v.weight"));
                blk.WbK = Mat_(B("attn_wb_k.weight")); blk.WbV = Mat_(B("attn_wb_v.weight"));
                blk.ShiftK = Mat_(B("attn_shift_k.weight")); blk.ShiftV = Mat_(B("attn_shift_v.weight"));
                blk.Lambda = Mat_(B("attn_lambda.weight")); blk.Gate = Mat_(B("attn_gate.weight"));
                blk.QNorm = F32Dev(B("attn_q_norm.weight")); blk.KNorm = F32Dev(B("attn_k_norm.weight")); blk.Scaler = F32Dev(B("attn_softmax_scaler"));
                blk.LastK = Alloc(attnLayout.LastKBytes, true, true); blk.LastV = Alloc(attnLayout.LastVBytes, true, true);
            }
            _blocks[il] = blk;
        }
        _stream.Synchronize();
    }

    private static long TensorBytes(GgufTensorDescriptor d)
    {
        long rows = 1;
        for (int i = 1; i < d.Shape.Rank; i++) rows *= d.Shape[i];
        return Dequantize.RowByteSize(d.Shape[0], d.QuantizationType) * rows;
    }

    // ───────────────────────────── scratch / state ─────────────────────────────

    private nint Scratch(string key, long bytes)
    {
        bytes = Math.Max(bytes, 16);
        if (_scratch.TryGetValue(key, out var old))
        {
            if (old.Bytes >= bytes) return old.Ptr;
            _scratch.Remove(key);
            CudaDriverApi.cuMemFree_v2(old.Ptr).ThrowOnError();
        }
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        _scratch[key] = (ptr, bytes);
        return ptr;
    }

    private void EnsureHistory(Blk b, long tokens)
    {
        if (tokens <= b.Capacity) return;
        long cap = Math.Min(Math.Max(tokens, Math.Max(1024, b.Capacity * 2)), Config.MaxSequenceLength);
        cap = Math.Max(cap, tokens);
        long kvW = (long)_o.NumNoiseHeads * Config.HeadDim, bytes = cap * kvW * _actBytes;
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
        long extra = 2 * bytes - 2 * b.Capacity * kvW * _actBytes;
        if (extra + (b.Capacity > 0 ? 2 * b.Capacity * kvW * _actBytes : 0) > (long)free)
            throw new InvalidOperationException($"CUDA olala: the K/V history for {tokens} tokens does not fit in free VRAM ({(long)free / (1 << 20)} MiB free); K/V is never spilled to host memory.");
        nint nk = Alloc(bytes, false), nv = Alloc(bytes, false);
        if (b.Capacity > 0)
        {
            long old = b.Capacity * kvW * _actBytes;
            CudaDriverApi.cuMemcpyDtoDAsync_v2(nk, b.KCache, (nuint)old, _stream.Handle).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(nv, b.VCache, (nuint)old, _stream.Handle).ThrowOnError();
            _stream.Synchronize();
            _allocs.Remove(b.KCache); _allocs.Remove(b.VCache);
            CudaDriverApi.cuMemFree_v2(b.KCache).ThrowOnError(); CudaDriverApi.cuMemFree_v2(b.VCache).ThrowOnError();
            _historyBytes -= 2 * old;
        }
        b.KCache = nk; b.VCache = nv; _historyBytes += 2 * bytes;
        b.Capacity = cap;
    }

    /// <inheritdoc/>
    public void ResetSequenceState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _context.MakeCurrent(); _stream.Synchronize();
        var ssm = CudaOlalaStateLayout.Ssm(_o); var attn = CudaOlalaStateLayout.Attn(_o, Config);
        foreach (var b in _blocks)
        {
            if (b.IsSsm)
            {
                Zero(b.Ssm, ssm.SsmBytes); Zero(b.Angle, ssm.AngleBytes); Zero(b.TrapK, ssm.TrapKBytes); Zero(b.TrapV, ssm.TrapVBytes);
            }
            else { Zero(b.LastK, attn.LastKBytes); Zero(b.LastV, attn.LastVBytes); }
        }
        _length = 0;
    }

    private static void Zero(nint p, long bytes) => CudaDriverApi.cuMemsetD8_v2(p, 0, (nuint)Math.Max(bytes, 16)).ThrowOnError();

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, lastTokenLogitsOnly);

    private ITensor ForwardCore(ReadOnlySpan<int> ids, ReadOnlySpan<int> positions, bool lastTokenLogitsOnly)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int T = ids.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (positions[0] == 0 && _length != 0) ResetSequenceState();
        for (int i = 0; i < T; i++)
        {
            if ((uint)ids[i] >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ID {ids[i]} at position {i} is out of range [0, {_vocab}).");
            if (positions[i] != _length + i)
                throw new ArgumentException($"olala keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {_length + i}.");
        }
        if ((long)_length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        _context.MakeCurrent();

        int rowsOut = lastTokenLogitsOnly ? 1 : T;
        var result = UnmanagedTensor.Allocate(new TensorShape(rowsOut, _vocab), DType.Float32, -1);
        try
        {
            int chunk = Math.Max(1, ChunkTokens);
            for (int start = 0; start < T; start += chunk)
            {
                int c = Math.Min(chunk, T - start);
                bool last = start + c == T;
                int rows = lastTokenLogitsOnly ? (last ? 1 : 0) : c;
                int outRow = lastTokenLogitsOnly ? 0 : start;
                ForwardChunk(ids.Slice(start, c), _length, rows, result, outRow);
                _length += c;
            }
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private void ForwardChunk(ReadOnlySpan<int> ids, int posBase, int logitRows, UnmanagedTensor result, int resultRow)
    {
        int c = ids.Length, H = _hidden;
        nint s = _stream.Handle;
        long act = _actBytes;
        nint idDev = Scratch("ids", c * 4L);
        fixed (int* p = ids) CudaDriverApi.cuMemcpyHtoD_v2(idDev, (nint)p, (nuint)(c * 4)).ThrowOnError();
        nint res = Scratch("res", c * H * act), mix = Scratch("mix", c * H * act);

        _k.Embed(_embedKind, _embed, H, idDev, res, c, s);
        if (posBase == 0) Trace?.Invoke("embed", res, c, H);
        foreach (var b in _blocks)
        {
            if (b.IsSsm) SsmLayer(b, res, mix, c);
            else AttentionLayer(b, res, mix, c, posBase);
            if (posBase == 0) Trace?.Invoke($"blk.{b.Index}.mixer_out", mix, c, H);
            _k.Geodesic(res, mix, H, b.GeoMixScale, b.GeoMixBias, b.Index, c, s);
            if (posBase == 0) Trace?.Invoke($"blk.{b.Index}.post_mixer", res, c, H);
            MoeLayer(b.Moe, res, mix, c);
            if (posBase == 0) Trace?.Invoke($"blk.{b.Index}.mlp_out", mix, c, H);
            _k.Geodesic(res, mix, H, b.GeoMlpScale, b.GeoMlpBias, b.Index, c, s);
            if (posBase == 0) Trace?.Invoke($"blk.{b.Index}.l_out", res, c, H);
        }
        if (logitRows > 0)
        {
            nint logits = Scratch("logits", (long)logitRows * _vocab * 4);
            nint headIn = res + (nint)((long)(c - logitRows) * H * act);
            _k.Gemv(_output.Kind, _output.Ptr, H, _vocab, headIn, logits, logitRows, outF32: true, epi: 0, aux: 0, s);
            _stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer + (nint)((long)resultRow * _vocab * 4), logits, (nuint)((long)logitRows * _vocab * 4)).ThrowOnError();
        }
        else _stream.Synchronize();
    }

    private void SsmLayer(Blk b, nint x, nint mixOut, int c)
    {
        int H = _hidden, Hs = _o.NumSsmHeads, P = _o.MambaHeadDim, N = _o.StateSize, R = _o.MimoRank, A = _o.NumRopeAngles;
        int ph = 2 * P + 3, dynW = 2 * R * N + A;
        long act = _actBytes; nint s = _stream.Handle;
        nint zxdt = Scratch("zxdt", c * (long)Hs * ph * act), dyn = Scratch("dyn", c * (long)dynW * act), y = Scratch("ssm_y", c * (long)Hs * P * act);
        _k.Gemv(b.SsmIn.Kind, b.SsmIn.Ptr, b.SsmIn.In, b.SsmIn.Out, x, zxdt, c, false, 0, 0, s);
        _k.Gemv(b.SsmInDyn.Kind, b.SsmInDyn.Ptr, b.SsmInDyn.In, b.SsmInDyn.Out, x, dyn, c, false, 0, 0, s);
        _k.Mamba(zxdt, dyn, c, b.Ssm, b.Angle, b.TrapK, b.TrapV, b.BNorm, b.CNorm, b.BBias, b.CBias, b.DtBias, b.D, b.MimoX, b.MimoZ, b.MimoO,
                 y, Hs, P, N, R, A, _eps, s);
        if (_length == 0) Trace?.Invoke($"blk.{b.Index}.ssm_y", y, c, Hs * P);
        _k.Gemv(b.MixerOut.Kind, b.MixerOut.Ptr, b.MixerOut.In, b.MixerOut.Out, y, mixOut, c, false, 0, 0, s);
    }

    private void AttentionLayer(Blk b, nint x, nint mixOut, int c, int posBase)
    {
        int hd = Config.HeadDim, nk = _o.NumNoiseHeads, nq = _o.NumSignalHeads + nk, rank = _o.TpaRank, group = nq / nk;
        int qW = nq * hd, gateW = _o.NumSignalHeads * hd;
        long act = _actBytes; nint s = _stream.Handle;
        EnsureHistory(b, (long)posBase + c);
        nint q = Scratch("q", c * (long)qW * act);
        nint ak = Scratch("ak", c * (long)nk * rank * act), av = Scratch("av", c * (long)nk * rank * act);
        nint bk = Scratch("bk", c * (long)rank * hd * act), bv = Scratch("bv", c * (long)rank * hd * act);
        nint sk = Scratch("sk", c * (long)nk * act), sv = Scratch("sv", c * (long)nk * act), lam = Scratch("lam", c * (long)nk * act);
        nint gate = Scratch("gate", c * (long)gateW * act), attn = Scratch("attn", c * (long)qW * act), yv = Scratch("attn_y", c * (long)gateW * act);
        void G(Mat m, nint dst) => _k.Gemv(m.Kind, m.Ptr, m.In, m.Out, x, dst, c, false, 0, 0, s);
        G(b.Q, q); G(b.WaK, ak); G(b.WaV, av); G(b.WbK, bk); G(b.WbV, bv); G(b.ShiftK, sk); G(b.ShiftV, sv); G(b.Lambda, lam); G(b.Gate, gate);
        _k.TpaKv(ak, av, bk, bv, sk, sv, b.LastK, b.LastV, b.KCache, b.VCache, b.KNorm, posBase, c, nk, rank, hd, _eps, s);
        _k.TpaCarry(ak, av, bk, bv, b.LastK, b.LastV, c, nk, rank, hd, s);
        _k.Attention(q, b.KCache, b.VCache, attn, b.QNorm, b.Scaler, posBase, c, nq, nk, hd, _window, _softcap, _eps, s);
        _k.Combine(attn, lam, gate, yv, c, nk, group, hd, _o.GateBias, s);
        if (_length == 0) Trace?.Invoke($"blk.{b.Index}.attn_y", yv, c, gateW);
        _k.Gemv(b.MixerOut.Kind, b.MixerOut.Ptr, b.MixerOut.In, b.MixerOut.Out, yv, mixOut, c, false, 0, 0, s);
    }

    /// <summary>Latent MoE + shared expert: <c>y = LatentUp(sum_s w_s Down_e(relu(Up_e(LatentDown x))^2)) + SharedDown(relu(SharedUp x)^2)</c>.</summary>
    private void MoeLayer(MoeW m, nint x, nint y, int c)
    {
        int H = _hidden, E = _o.ExpertCount, K = _o.ExpertUsedCount, lat = _o.MoeLatentSize, ff = _o.ExpertFfLength, sff = m.SharedUp.Out;
        long act = _actBytes; nint s = _stream.Handle;
        nint idx = Scratch("route_idx", (long)c * K * 4), wts = Scratch("route_w", (long)c * K * 4);
        nint x0 = Scratch("x0", c * (long)lat * act), hid = Scratch("moe_hid", (long)c * K * ff * act), ylat = Scratch("ylat", c * (long)lat * act);
        nint routed = Scratch("routed", c * (long)H * act), sh = Scratch("shared_h", c * (long)sff * act);
        _k.Route(m.Router, m.ExpertBias, x, idx, wts, H, E, K, _o.ExpertWeightsScale, c, s);
        _k.Gemv(m.LatentDown.Kind, m.LatentDown.Ptr, m.LatentDown.In, m.LatentDown.Out, x, x0, c, false, 0, 0, s);
        _k.MoeUp(m.UpKind, m.UpExps, m.UpExpertBytes, lat, ff, idx, x0, hid, c, K, s);
        _k.MoeDown(m.DownKind, m.DownExps, m.DownExpertBytes, ff, lat, idx, wts, hid, ylat, c, K, s);
        _k.Gemv(m.LatentUp.Kind, m.LatentUp.Ptr, m.LatentUp.In, m.LatentUp.Out, ylat, routed, c, false, 0, 0, s);
        _k.Gemv(m.SharedUp.Kind, m.SharedUp.Ptr, m.SharedUp.In, m.SharedUp.Out, x, sh, c, false, 1, 0, s);
        _k.Gemv(m.SharedDown.Kind, m.SharedDown.Ptr, m.SharedDown.In, m.SharedDown.Out, sh, y, c, false, 2, routed, s);
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        _spillMonitor.Dispose();
        if (_disposed) return;
        _disposed = true;
        CudaDriverApi.cuCtxSetCurrent(_context.Handle);
        CudaDriverApi.cuStreamSynchronize(_stream.Handle);
        foreach (var b in _scratch.Values) CudaDriverApi.cuMemFree_v2(b.Ptr);
        foreach (nint p in _allocs) CudaDriverApi.cuMemFree_v2(p);
        _scratch.Clear(); _allocs.Clear();
        _k.Dispose(); _stream.Dispose(); _context.Dispose();
    }
}
