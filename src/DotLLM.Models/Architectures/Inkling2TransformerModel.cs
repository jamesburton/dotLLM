using System.Runtime.CompilerServices;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Models.Architectures.DSpark;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Load-time storage choices of <see cref="Inkling2TransformerModel"/> that the user rules on (issue #2026, epic #2025). User rulings 2026-11-26: the loop controller and the conv state default to
/// BF16 (converted once at load from the GGUF's F32); F32 stays as an A/B override only, and F16 conv state is ruled out.
/// </summary>
/// <param name="LoopController">
/// Storage of the loop-controller weights (<c>loop_msg_query</c>, <c>loop_*_ctrl_down/gain/forget</c>; F32 in the GGUF, ~113 MB). Default <see cref="QuantizationType.BF16"/> is a load-time copy (56.8 MB); F32 reads them in
/// place (A/B override); <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.F16"/> and <see cref="QuantizationType.BF16"/> make a smaller copy at load.
/// </param>
/// <param name="ConvState">Storage of the carried short-convolution inputs.</param>
public sealed record Inkling2Options(
    QuantizationType LoopController = QuantizationType.BF16,
    Inkling2ConvStateStorage ConvState = Inkling2ConvStateStorage.BF16)
{
    /// <summary>
    /// Options from <c>DOTLLM_INKLING2_LOOP_CONTROLLER</c> (<c>f32|q8_0|f16|bf16</c>) and <c>DOTLLM_INKLING2_CONV_STATE</c> (<c>f32|bf16</c>; F16 is ruled out);
    /// unset variables keep the defaults. Used by the measurement harness; a bad value throws rather than silently running the default.
    /// </summary>
    public static Inkling2Options FromEnvironment()
    {
        var o = new Inkling2Options();
        string? ctl = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_LOOP_CONTROLLER");
        if (!string.IsNullOrEmpty(ctl))
            o = o with
            {
                LoopController = ctl.ToLowerInvariant() switch
                {
                    "f32" => QuantizationType.F32,
                    "q8_0" or "q8" => QuantizationType.Q8_0,
                    "f16" => QuantizationType.F16,
                    "bf16" => QuantizationType.BF16,
                    _ => throw new ArgumentException($"DOTLLM_INKLING2_LOOP_CONTROLLER='{ctl}': expected f32, q8_0, f16 or bf16."),
                },
            };
        string? cs = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_CONV_STATE");
        if (!string.IsNullOrEmpty(cs))
            o = o with
            {
                ConvState = cs.ToLowerInvariant() switch
                {
                    "f32" => Inkling2ConvStateStorage.F32,
                    "bf16" => Inkling2ConvStateStorage.BF16,
                    _ => throw new ArgumentException($"DOTLLM_INKLING2_CONV_STATE='{cs}': expected f32 or bf16."),
                },
            };
        return o;
    }
}

/// <summary>Deliberately wrong implementations used to prove the oracle gate can fail (CLAUDE.md "Prove the Test Can Fail"). Test-only.</summary>
internal enum Inkling2Mutant
{
    None, TwoPasses, NoFinalSoftcap, NoSink, NoRelativeBias, NoLogScale, NoExpertScale, NoControl,
    /// <summary>DSpark taps keyed by cache slot instead of layer visit (#2039).</summary>
    TapBySlot,
    /// <summary>DSpark taps one visit late (the layer OUTPUT instead of its input).</summary>
    TapOffByOne,
    /// <summary>Visit 48 tapped before the output norm.</summary>
    TapPreNorm,
    /// <summary>A rejected speculative round leaves the recurrent state advanced (RestoreRecurrentStateToRow is a no-op).</summary>
    SkipRestore,
}

/// <summary>Receives <c>(name, data, rows, cols)</c> tensors from <see cref="Inkling2TransformerModel"/>'s trace hook (tests / bisection).</summary>
internal delegate void Inkling2TraceSink(string name, ReadOnlySpan<float> data, int rows, int cols);

/// <summary>
/// CPU reference forward for Thinking Machines Inkling 2 (<c>inkling2_loop</c>, Inkling-2-Mini; issue #2026). Authoritative semantics: llama.cpp
/// <c>src/models/inkling2.cpp</c> and <c>inkling2-loop.cpp</c> (+ <c>ggml-cpu/ops.cpp</c> for the ops); the layer-by-layer description with sources is
/// <c>docs/INKLING2.md</c>.
/// </summary>
/// <remarks>
/// <para>Correctness over speed. Weights are read in place from the mmap'd GGUF (Q8_0, NVFP4 with its per-expert scales, F32); activations are
/// transient F32 scratch (CPU kernels take F32 activations; nothing wider than the GGUF's own tensors is stored). Persistent state is narrow: the
/// KV history is F16 and the conv state follows <see cref="Inkling2Options.ConvState"/>.</para>
/// <para>The model owns its sequence state (KV + conv state per cache slot); positions must continue it (<c>positions[i] == state.Length + i</c>)
/// and a call at position 0 restarts it. An external <see cref="IKvCache"/> is ignored.</para>
/// </remarks>
public sealed unsafe partial class Inkling2TransformerModel : IModel
{
    private readonly record struct MatRef(nint Ptr, QuantizationType Qt, int In, int Out);

    private readonly record struct F32Ref(nint Ptr, int Count)
    {
        public ReadOnlySpan<float> Span => new((void*)Ptr, Count);
        public bool IsEmpty => Count == 0;
    }

    private sealed class LoopWeights
    {
        public required MatRef MsgQuery;                  // [4][H] as 4 rows
        public required MatRef[] Down, Gain, Forget;      // per controller Q, K, V, FFN
    }

    private sealed class Layer
    {
        public required bool Local;
        public required int Extent;                       // rows of the relative-bias table
        public required MatRef Wq, Wk, Wv, Wo, Gate, Rel, Sink;
        public MatRef KConvProj, VConvProj;               // global layers only
        public required F32Ref QClip, RelClip, RelBias, KConv, VConv, PostConv, FfnPostConv, FfnResQuery, FfnNorm, ExpBias;
        public F32Ref AttnNorm, QInpNorm, KInpNorm, VInpNorm, AttnResQuery;
        public required MatRef Router;
        public required nint GateUp, Down;
        public required QuantizationType GateUpQt, DownQt;
        public required long GateUpExpertBytes, DownExpertBytes;
        public F32Ref GateUpScale, DownScale;
        public LoopWeights? Loop;
    }

    private readonly GgufFile? _gguf;
    private readonly Layer[] _layers;
    private readonly nint _tokenEmbed;
    private readonly MatRef _output;
    private readonly F32Ref _tokEmbNorm, _outputNorm, _outputResQuery;
    private readonly Inkling2Config _c;
    private readonly Inkling2Options _options;
    private readonly int _hidden, _vocab, _heads, _kvHeads, _headDim, _nPhys, _kvW, _qW;
    private readonly float _eps;
    private readonly ComputeThreadPool? _pool;
    private readonly bool _ownsPool;
    private readonly List<nint> _owned = new();
    private readonly Stack<Inkling2Scratch> _scratchPool = new();
    private readonly Inkling2SequenceState _defaultState;
    private long _narrowedBytes, _f32ControllerBytes;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>The storage options this model was loaded with.</summary>
    public Inkling2Options Options => _options;

    /// <summary>Bytes of the loop-controller weights as stored (the F32 original is 113.7 MB on Inkling-2-Mini).</summary>
    public long LoopControllerBytes => _options.LoopController == QuantizationType.F32 ? _f32ControllerBytes : _narrowedBytes;

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _defaultState.Bytes;

    /// <summary>Test-only: run a deliberately wrong variant (see <see cref="Inkling2Mutant"/>).</summary>
    internal Inkling2Mutant Mutant { get; set; }

    /// <summary>Test/diagnostic hook: receives named intermediates (null = off, zero cost).</summary>
    internal Inkling2TraceSink? Trace { get; set; }

    private Inkling2TransformerModel(ModelConfig config, GgufFile? gguf, Layer[] layers, nint tokenEmbed, MatRef output,
                                     F32Ref tokEmbNorm, F32Ref outputNorm, F32Ref outputResQuery, Inkling2Options options,
                                     ComputeThreadPool? pool, bool ownsPool, List<nint> owned, long narrowedBytes, long f32ControllerBytes)
    {
        Config = config; _gguf = gguf; _layers = layers; _tokenEmbed = tokenEmbed; _output = output;
        _tokEmbNorm = tokEmbNorm; _outputNorm = outputNorm; _outputResQuery = outputResQuery;
        _options = options; _pool = pool; _ownsPool = ownsPool; _owned = owned;
        _narrowedBytes = narrowedBytes; _f32ControllerBytes = f32ControllerBytes;
        _c = config.Inkling2!;
        _hidden = config.HiddenSize; _vocab = config.VocabSize; _heads = config.NumAttentionHeads; _kvHeads = config.NumKvHeads;
        _headDim = config.HeadDim; _nPhys = config.NumLayers; _eps = config.NormEpsilon;
        _kvW = _kvHeads * _headDim; _qW = _heads * _headDim;
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>Allocates a fresh sequence state (native memory; dispose when the sequence ends).</summary>
    public Inkling2SequenceState CreateState() => new(Config, _options.ConvState);

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Loads an Inkling 2 model from an opened GGUF with the options of <see cref="Inkling2Options.FromEnvironment"/>. The file must outlive the model.</summary>
    public static Inkling2TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
        => LoadFromGguf(gguf, config, threading, Inkling2Options.FromEnvironment());

    /// <summary>Loads an Inkling 2 model from an opened GGUF. The file must outlive the model.</summary>
    public static Inkling2TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading, Inkling2Options options)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(options);
        if (config.Architecture != DotLLM.Core.Configuration.Architecture.Inkling2 || config.Inkling2 is not { } c)
            throw new ArgumentException("Inkling2TransformerModel requires an Architecture.Inkling2 configuration.", nameof(config));
        if (options.LoopController is not (QuantizationType.F32 or QuantizationType.Q8_0 or QuantizationType.F16 or QuantizationType.BF16))
            throw new ArgumentException($"Loop-controller storage {options.LoopController} is not supported (F32, Q8_0, F16, BF16).", nameof(options));

        var tensors = gguf.TensorsByName;
        var problems = Inkling2Tensors.FindProblems(tensors, config, out var consumedScales);
        if (problems.Count > 0)
            throw new InvalidDataException("inkling2 tensor table does not match the model contract: " + string.Join("; ", problems.Take(8))
                                           + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));
        // Only the per-expert scales of the two expert banks are applied; any other NVFP4 weight scale would be silently ignored, so refuse it.
        Nvfp4Support.RejectUnappliedScales(tensors, consumedScales, "The inkling2 loader");

        var owned = new List<nint>();
        long narrowed = 0, f32Controller = 0;

        MatRef Mat(string name)
        {
            var d = tensors[name];
            return new MatRef(gguf.TensorDataPointer(d), d.QuantizationType, d.Shape[0], d.Shape.Rank > 1 ? d.Shape[1] : 1);
        }
        F32Ref F32(string name)
        {
            var d = tensors[name];
            long count = 1; for (int i = 0; i < d.Shape.Rank; i++) count *= d.Shape[i];
            return new F32Ref(gguf.TensorDataPointer(d), checked((int)count));
        }
        // A loop-controller matrix: F32 in place, or a smaller copy (never wider than the source).
        MatRef Controller(string name)
        {
            var m = Mat(name);
            f32Controller += (long)m.In * m.Out * sizeof(float);
            if (options.LoopController == QuantizationType.F32) return m;
            long count = (long)m.In * m.Out;
            var src = new ReadOnlySpan<float>((void*)m.Ptr, checked((int)count));
            long bytes = Dequantize.RowByteSize(count, options.LoopController);
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            owned.Add(dst);
            var dstSpan = new Span<byte>((void*)dst, checked((int)bytes));
            if (options.LoopController == QuantizationType.BF16)
            {
                var u = MemoryMarshal.Cast<byte, ushort>(dstSpan);
                for (int i = 0; i < src.Length; i++) u[i] = Inkling2ConvState.ToBf16(src[i]);
            }
            else
            {
                if (count % 32 != 0 || m.In % 32 != 0) throw new InvalidDataException($"{name}: {options.LoopController} needs a row length that is a multiple of 32, got {m.In}.");
                Quantize.FromFloat32(src, count, options.LoopController, dstSpan);
            }
            narrowed += bytes;
            return new MatRef(dst, options.LoopController, m.In, m.Out);
        }

        var embDesc = tensors[Inkling2Tensors.TokenEmbd];
        nint embPtr = gguf.TensorDataPointer(embDesc);
        if (embDesc.QuantizationType != QuantizationType.Q8_0 && embDesc.QuantizationType != QuantizationType.F32 && embDesc.QuantizationType != QuantizationType.F16
            && embDesc.QuantizationType != QuantizationType.BF16)
            throw new NotSupportedException($"token_embd.weight is {embDesc.QuantizationType}; only Q8_0 / F32 / F16 / BF16 embeddings are implemented.");
        var output = Mat(Inkling2Tensors.Output);

        var layers = new Layer[config.NumLayers];
        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => Inkling2Tensors.Block(il, s);
            bool local = c.IsLocalLayer[il];
            var gu = tensors[B("ffn_gate_up_exps.weight")]; var dn = tensors[B("ffn_down_exps.weight")];
            var layer = new Layer
            {
                Local = local,
                Extent = local ? c.SlidingWindow : c.RelativeExtent,
                Wq = Mat(B("attn_q.weight")), Wk = Mat(B("attn_k.weight")), Wv = Mat(B("attn_v.weight")), Wo = Mat(B("attn_output.weight")),
                Gate = Mat(B("attn_gate.weight")), Rel = Mat(B("attn_rel.weight")), Sink = Mat(B("attn_sink_proj.weight")),
                QClip = F32(B("attn_q_clip.weight")), RelClip = F32(B("attn_rel_clip.weight")), RelBias = F32(B("attn_rel_bias.weight")),
                KConv = F32(B("attn_k_conv.weight")), VConv = F32(B("attn_v_conv.weight")), PostConv = F32(B("attn_post_conv.weight")),
                FfnPostConv = F32(B("ffn_post_conv.weight")), FfnResQuery = F32(B("ffn_res_query.weight")), FfnNorm = F32(B("ffn_norm.weight")),
                ExpBias = F32(B("exp_probs_b.bias")),
                Router = Mat(B("ffn_gate_inp.weight")),
                GateUp = gguf.TensorDataPointer(gu), GateUpQt = gu.QuantizationType,
                Down = gguf.TensorDataPointer(dn), DownQt = dn.QuantizationType,
                GateUpExpertBytes = Dequantize.RowByteSize(gu.Shape[0], gu.QuantizationType) * gu.Shape[1],
                DownExpertBytes = Dequantize.RowByteSize(dn.Shape[0], dn.QuantizationType) * dn.Shape[1],
            };
            if (tensors.ContainsKey(B("ffn_gate_up_exps.scale"))) layer.GateUpScale = F32(B("ffn_gate_up_exps.scale"));
            if (tensors.ContainsKey(B("ffn_down_exps.scale"))) layer.DownScale = F32(B("ffn_down_exps.scale"));
            if (local) layer.AttnNorm = F32(B("attn_norm.weight"));
            else
            {
                layer.QInpNorm = F32(B("attn_q_inp_norm.weight")); layer.KInpNorm = F32(B("attn_k_inp_norm.weight"));
                layer.VInpNorm = F32(B("attn_v_inp_norm.weight"));
                layer.KConvProj = Mat(B("attn_k_conv_proj.weight")); layer.VConvProj = Mat(B("attn_v_conv_proj.weight"));
            }
            if (il > 0) layer.AttnResQuery = F32(B("attn_res_query.weight"));
            if (c.LoopPasses > 1 && il >= c.LoopStart && il < c.LoopEnd)
            {
                var down = new MatRef[4]; var gain = new MatRef[4]; var forget = new MatRef[4];
                for (int k = 0; k < 4; k++)
                {
                    string kind = Inkling2Tensors.CtrlKinds[k];
                    down[k] = Controller(B($"loop_{kind}_ctrl_down.weight"));
                    gain[k] = Controller(B($"loop_{kind}_ctrl_gain.weight"));
                    forget[k] = Controller(B($"loop_{kind}_ctrl_forget.weight"));
                }
                layer.Loop = new LoopWeights { MsgQuery = Controller(B("loop_msg_query.weight")), Down = down, Gain = gain, Forget = forget };
            }
            layers[il] = layer;
        }

        var pool = CreatePool(threading);
        return new Inkling2TransformerModel(config, gguf, layers, embPtr, output,
            F32(Inkling2Tensors.TokenEmbdNorm), F32(Inkling2Tensors.OutputNorm), F32(Inkling2Tensors.OutputResQuery), options,
            pool, ownsPool: pool is not null, owned, narrowed, f32Controller);
    }

    private static ComputeThreadPool? CreatePool(ThreadingConfig threading)
    {
        if (!threading.IsParallel) return null;
        int n = threading.EffectiveThreadCount;
        if (threading.EnableNumaPinning || threading.EnablePCorePinning)
        {
            var topology = NumaTopology.Detect();
            if (threading.EnablePCorePinning && topology.IsHybrid) n = Math.Min(n, topology.PerformanceCoreIds.Count);
            return new ComputeThreadPool(n, topology, threading);
        }
        return new ComputeThreadPool(n, topology: null, threading);
    }

    // ───────────────────────────── GEMM ─────────────────────────────

    private void GemmSpan(MatRef w, ReadOnlySpan<float> x, Span<float> y, int n)
    {
        if (x.Length < (long)n * w.In) throw new ArgumentException("input too small.", nameof(x));
        if (y.Length < (long)n * w.Out) throw new ArgumentException("output too small.", nameof(y));
        fixed (float* xp = x)
        fixed (float* yp = y)
            Gemm(w.Ptr, w.Qt, xp, yp, w.Out, w.In, n, _pool);
    }

    /// <summary><c>c[t, i] = dot(W[i, :], b[t, :])</c>; every weight type is read in place (no expansion): quantized and BF16 weights through their GEMMs.</summary>
    internal static void GemmPublic(nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool) => Gemm(weights, qt, b, c, m, k, n, pool);

    private static void Gemm(nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.Q8_0: MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.NVFP4: MatMul.GemmNvfp4((byte*)weights, b, c, m, k, n, pool); return;
            case QuantizationType.F32: MatMul.GemmF32((float*)weights, b, c, m, k, n, pool); return;
            case QuantizationType.F16: MatMul.GemmF16(weights, b, c, m, k, n, pool); return;
            default: MatMul.GemmDequantRows((byte*)weights, qt, b, c, m, k, n, pool); return;
        }
    }

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly);

    /// <summary>Forward on an explicit sequence state (tests, multi-sequence callers).</summary>
    public ITensor ForwardWithState(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, Inkling2SequenceState state, bool lastTokenLogitsOnly = false)
        => ForwardCore(tokenIds, positions, deviceId, state, lastTokenLogitsOnly);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, Inkling2SequenceState state, bool lastTokenLogitsOnly,
                                IInkling2Tap? tap = null, bool recordRows = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (ReferenceEquals(state, _defaultState) && positions[0] == 0 && state.Length != 0) state.Reset();
        for (int i = 0; i < T; i++)
            if (positions[i] != state.Length + i)
                throw new ArgumentException($"inkling2 keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {state.Length + i}.");
        if (state.Length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");

        Inkling2Scratch sc = RentScratch();
        int posStart = state.Length;
        try
        {
            state.RecordRows = recordRows;
            if (recordRows) _recordedPos0 = posStart;
            return ForwardBody(tokenIds, deviceId, state, lastTokenLogitsOnly, sc, tap);
        }
        finally { state.RecordRows = false; ReturnScratch(sc); }
    }

    private Inkling2Scratch RentScratch()
    {
        Inkling2Scratch? sc = null;
        lock (_scratchPool) { if (_scratchPool.Count > 0) sc = _scratchPool.Pop(); }
        sc ??= new Inkling2Scratch();
        sc.Begin();
        return sc;
    }

    private void ReturnScratch(Inkling2Scratch sc)
    {
        sc.End();
        lock (_scratchPool) _scratchPool.Push(sc);
    }

    private ITensor ForwardBody(ReadOnlySpan<int> tokenIds, int deviceId, Inkling2SequenceState st, bool lastOnly, Inkling2Scratch sc, IInkling2Tap? tap)
    {
        int visit = 0;
        int T = tokenIds.Length, H = _hidden;
        int pos0 = st.Length;
        _pool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

        // Token embeddings: Q8_0 rows dequantised one at a time into the activation buffer, then the gain-ed RMS norm (llama.cpp: tok_norm).
        Span<float> cur = sc.Call.Alloc<float>(T * H);
        var embType = _gguf!.TensorsByName[Inkling2Tensors.TokenEmbd].QuantizationType;
        long rowBytes = Dequantize.RowByteSize(H, embType);
        for (int t = 0; t < T; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {_vocab}).");
            var row = cur.Slice(t * H, H);
            Dequantize.ToFloat32(_tokenEmbed + (nint)(id * rowBytes), H, embType, row);
            Inkling2Ops.RmsNorm(row, _tokEmbNorm.Span, _eps, row);
        }
        Trace?.Invoke("embed", cur, T, H);

        var blocks = sc.Blocks;   // RMS-normalised (gamma-free) block inputs, immutable once pushed (llama.cpp mix_stack cache)
        int start = _c.LoopStart, end = _c.LoopEnd;
        int coreStart = _c.LoopPasses > 1 ? start : _nPhys;

        for (int il = 0; il < coreStart; il++) { tap?.Capture(Mutant == Inkling2Mutant.TapOffByOne ? visit + 1 : visit, cur, T); visit++; cur = ForwardLayer(il, il, cur, sc, default, null, T, pos0, st); }

        if (_c.LoopPasses > 1)
        {
            Span<float> prelude = sc.Call.Alloc<float>(T * H);   // every pass restarts from the prelude output and the prelude blocks
            cur.CopyTo(prelude);
            cur = prelude;
            int preludeBlocks = blocks.Count;
            int core = end - start;
            Span<float> bankOut = default;
            for (int pass = 0; pass < (Mutant == Inkling2Mutant.TwoPasses ? 2 : _c.LoopPasses); pass++)
            {
                if (pass > 0)
                {
                    // The bank of the previous pass: its blocks after the first core block, and its output (llama.cpp inkling2-loop.cpp).
                    var bank = sc.Bank;
                    bank.Clear();
                    for (int i = preludeBlocks + 1; i < blocks.Count; i++) bank.Add(blocks[i]);
                    blocks.RemoveRange(preludeBlocks, blocks.Count - preludeBlocks);
                    bankOut = sc.Call.Alloc<float>(T * H);
                    cur.CopyTo(bankOut);
                    cur = prelude;
                }
                for (int ip = start; ip < end; ip++)
                {
                    int slot = pass == 0 ? ip : _nPhys + (pass - 1) * core + (ip - start);
                    Span<float> messages = default;
                    LoopWeights? loop = null;
                    if (pass > 0)
                    {
                        loop = _layers[ip].Loop!;
                        sc.Msg.Reset();
                        if (Mutant != Inkling2Mutant.NoControl)
                        {
                            // loop_msg_query is stored narrow (BF16 by default); the four query rows are widened into native scratch per visit.
                            var q = sc.Msg.Alloc<float>(loop.MsgQuery.In * loop.MsgQuery.Out);
                            Dequantize.ToFloat32(loop.MsgQuery.Ptr, q.Length, loop.MsgQuery.Qt, q);
                            messages = DepthMix(sc, sc.Bank, bankOut, q, 4, T, sc.Msg);
                        }
                    }
                    tap?.Capture(Mutant == Inkling2Mutant.TapBySlot ? slot : visit, cur, T); visit++;
                    cur = ForwardLayer(slot, ip, cur, sc, messages, loop, T, pos0, st);
                    Trace?.Invoke($"pass{pass}.l{ip}", cur, T, H);
                }
            }
            for (int il = end; il < _nPhys; il++) { tap?.Capture(visit++, cur, T); cur = ForwardLayer(il, il, cur, sc, default, null, T, pos0, st); }
        }

        // Output: depth mix with output_res_query, gain-ed RMS norm, LM head (unpadded rows only), final logit softcap.
        sc.Msg.Reset();
        Span<float> mixed = DepthMix(sc, blocks, cur, _outputResQuery.Span, 1, T, sc.Msg);
        for (int t = 0; t < T; t++)
            Inkling2Ops.RmsNorm(mixed.Slice(t * H, H), _outputNorm.Span, _eps, mixed.Slice(t * H, H));
        Trace?.Invoke("result_norm", mixed, T, H);
        tap?.Capture(visit++, Mutant == Inkling2Mutant.TapPreNorm ? cur : mixed, T);   // visit 48: the final normed output of every row

        int rows = lastOnly ? 1 : T;
        int firstRow = T - rows;
        var result = UnmanagedTensor.Allocate(new TensorShape(rows, _vocab), DType.Float32, deviceId);
        var logits = new Span<float>((void*)result.DataPointer, rows * _vocab);
        fixed (float* xp = &mixed[firstRow * H])
        fixed (float* yp = logits)
            Gemm(_output.Ptr, _output.Qt, xp, yp, _vocab, H, rows, _pool);
        float cap = _c.FinalLogitSoftcap;
        if (cap != 0f && Mutant != Inkling2Mutant.NoFinalSoftcap)
            for (int i = 0; i < logits.Length; i++) logits[i] = MathF.Tanh(logits[i] * (1f / cap)) * cap;
        st.Length += T;
        return result;
    }

    /// <summary>
    /// Depth mixing for T tokens (see <see cref="Inkling2Ops.DepthMixToken"/>): <paramref name="blocks"/> are already RMS-normalised, the
    /// current state is normalised here. Returns <c>[T][nq][H]</c> allocated from <paramref name="dst"/>.
    /// </summary>
    private Span<float> DepthMix(Inkling2Scratch sc, List<nint> blocks, ReadOnlySpan<float> cur, ReadOnlySpan<float> queries, int nq, int T, Inkling2Arena dst)
    {
        int H = _hidden, count = blocks.Count + 1;
        var output = dst.Alloc<float>(T * nq * H);
        var states = sc.Layer.Alloc<float>(count * H);
        for (int t = 0; t < T; t++)
        {
            for (int b = 0; b < blocks.Count; b++) new ReadOnlySpan<float>((void*)blocks[b], T * H).Slice(t * H, H).CopyTo(states.Slice(b * H, H));
            Inkling2Ops.RmsNorm(cur.Slice(t * H, H), default, Inkling2Ops.MixEps, states.Slice(blocks.Count * H, H));
            Inkling2Ops.DepthMixToken(states, count, queries, nq, H, output.Slice(t * nq * H, nq * H));
        }
        return output;
    }

    private Span<float> NormedCopy(ReadOnlySpan<float> x, int T, Inkling2Arena dst)
    {
        int H = _hidden;
        var r = dst.Alloc<float>(T * H);
        for (int t = 0; t < T; t++) Inkling2Ops.RmsNorm(x.Slice(t * H, H), default, Inkling2Ops.MixEps, r.Slice(t * H, H));
        return r;
    }

    /// <summary>
    /// The loop-controller combine for T tokens (llama.cpp <c>graph::control</c>): the rank-r gate is a projection of
    /// <c>[rms(cur); rms(message)]</c>; <c>out = (cur - tanh(forget) * cur) + tanh(gain) * message</c>. <paramref name="msgs"/> is
    /// <c>[T][4][H]</c> and <paramref name="ctrl"/> selects the controller (Q, K, V, FFN).
    /// </summary>
    private Span<float> Control(Inkling2Scratch sc, ReadOnlySpan<float> cur, ReadOnlySpan<float> msgs, LoopWeights loop, int ctrl, int T)
    {
        int H = _hidden, rank = _c.LoopGateRank;
        var a = sc.Layer;
        var gateIn = a.Alloc<float>(T * 2 * H);
        var msg = a.Alloc<float>(T * H);
        for (int t = 0; t < T; t++)
        {
            msgs.Slice((t * 4 + ctrl) * H, H).CopyTo(msg.Slice(t * H, H));
            Inkling2Ops.RmsNorm(cur.Slice(t * H, H), default, Inkling2Ops.MixEps, gateIn.Slice(t * 2 * H, H));
            Inkling2Ops.RmsNorm(msg.Slice(t * H, H), default, Inkling2Ops.MixEps, gateIn.Slice(t * 2 * H + H, H));
        }
        var gate = a.Alloc<float>(T * rank);
        GemmSpan(loop.Down[ctrl], gateIn, gate, T);
        var gainPre = a.Alloc<float>(T * H);
        var forgetPre = a.Alloc<float>(T * H);
        GemmSpan(loop.Gain[ctrl], gate, gainPre, T);
        GemmSpan(loop.Forget[ctrl], gate, forgetPre, T);
        var o = a.Alloc<float>(T * H);
        for (int t = 0; t < T; t++)
            Inkling2Ops.ControlCombine(cur.Slice(t * H, H), msg.Slice(t * H, H), gainPre.Slice(t * H, H), forgetPre.Slice(t * H, H), o.Slice(t * H, H));
        return o;
    }

    private Span<float> RmsNormRows(Inkling2Scratch sc, ReadOnlySpan<float> x, F32Ref gamma, int T)
    {
        int H = _hidden;
        var r = sc.Layer.Alloc<float>(T * H);
        for (int t = 0; t < T; t++) Inkling2Ops.RmsNorm(x.Slice(t * H, H), gamma.Span, _eps, r.Slice(t * H, H));
        return r;
    }

    /// <summary>One transformer layer visit (llama.cpp <c>graph::build_layer</c>). Returns the layer output (valid until two visits later); appends to the scratch block list when this layer starts a new block.</summary>
    private Span<float> ForwardLayer(int slot, int ip, ReadOnlySpan<float> cur, Inkling2Scratch sc, ReadOnlySpan<float> messages, LoopWeights? loop, int T, int pos0, Inkling2SequenceState st)
    {
        var lw = _layers[ip];
        int H = _hidden, hd = _headDim, nh = _heads, nkv = _kvHeads, kvW = _kvW, qW = _qW;
        var ss = st.Slots[slot];
        ss.EnsureCapacity(pos0 + T);
        var blocks = sc.Blocks;
        var A = sc.Layer;
        A.Reset();
        var outArena = sc.Out[sc.Visit++ & 1];
        outArena.Reset();
        ReadOnlySpan<float> inpSA = cur;

        // Attention inputs: layer 0 reads the embedding; later layers mix the earlier blocks with cur using learned queries
        // (one for local layers, three — Q, K, V — for global ones).
        ReadOnlySpan<float> in0 = cur, in1 = cur, in2 = cur;
        if (ip > 0)
        {
            int nq = lw.Local ? 1 : 3;
            var mixed = DepthMix(sc, blocks, cur, lw.AttnResQuery.Span, nq, T, A);
            in0 = PickQuery(A, mixed, lw.Local ? 0 : 0, nq, T);
            in1 = PickQuery(A, mixed, lw.Local ? 0 : 1, nq, T);
            in2 = PickQuery(A, mixed, lw.Local ? 0 : 2, nq, T);
        }
        if (!messages.IsEmpty)
        {
            in0 = Control(sc, in0, messages, loop!, 0, T);
            in1 = Control(sc, in1, messages, loop!, 1, T);
            in2 = Control(sc, in2, messages, loop!, 2, T);
        }

        if (lw.Local && !messages.IsEmpty)
        {
            in0 = RmsNormRows(sc, in0, lw.AttnNorm, T);
            in1 = RmsNormRows(sc, in1, lw.AttnNorm, T);
            in2 = RmsNormRows(sc, in2, lw.AttnNorm, T);
        }
        else if (lw.Local)
        {
            in0 = in1 = in2 = RmsNormRows(sc, in0, lw.AttnNorm, T);
        }
        else
        {
            in0 = RmsNormRows(sc, in0, lw.QInpNorm, T);
            in1 = RmsNormRows(sc, in1, lw.KInpNorm, T);
            in2 = RmsNormRows(sc, in2, lw.VInpNorm, T);
        }

        // ── self-attention ──
        var q4 = A.Alloc<float>(T * qW);
        var kRaw = A.Alloc<float>(T * kvW);
        var vRaw = A.Alloc<float>(T * kvW);
        int relLen = _c.RelativeLength;
        var rel = A.Alloc<float>(T * relLen * nh);
        var sink = A.Alloc<float>(T * nh);
        var gate = A.Alloc<float>(T * qW);
        GemmSpan(lw.Wq, in0, q4, T);
        GemmSpan(lw.Wk, in1, kRaw, T);
        GemmSpan(lw.Wv, in2, vRaw, T);
        GemmSpan(lw.Rel, in0, rel, T);
        GemmSpan(lw.Sink, in0, sink, T);
        GemmSpan(lw.Gate, in0, gate, T);
        if (Mutant == Inkling2Mutant.NoSink) sink.Clear();

        Span<float> kDyn = default, vDyn = default;
        if (!lw.Local)
        {
            kDyn = A.Alloc<float>(T * _c.ConvKernel * nkv);
            vDyn = A.Alloc<float>(T * _c.ConvKernel * nkv);
            GemmSpan(lw.KConvProj, in1, kDyn, T);
            GemmSpan(lw.VConvProj, in2, vDyn, T);
        }
        var convScratch = A.Alloc<float>(Math.Max(ss.KConv.ScratchFloats(T), ss.AttnPostConv.ScratchFloats(T)));
        ss.KConv.Run(T, kRaw, lw.KConv.Span, kDyn, nkv, kRaw, convScratch);
        ss.VConv.Run(T, vRaw, lw.VConv.Span, vDyn, nkv, vRaw, convScratch);

        var qClip = lw.QClip.Span; var relClip = lw.RelClip.Span;
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < nh; h++)
            {
                Inkling2Ops.ClipL2(q4.Slice(t * qW + h * hd, hd), qClip.Slice(h * hd, hd), _c.QkClip);
                Inkling2Ops.ClipL2(rel.Slice((t * nh + h) * relLen, relLen), relClip.Slice(h * relLen, relLen), _c.RelativeClip);
            }
            for (int g = 0; g < nkv; g++)
            {
                Inkling2Ops.ClipL2(kRaw.Slice(t * kvW + g * hd, hd), default, _c.QkClip);
                Inkling2Ops.RmsNorm(vRaw.Slice(t * kvW + g * hd, hd), default, _eps, vRaw.Slice(t * kvW + g * hd, hd));
            }
            if (Mutant == Inkling2Mutant.NoRelativeBias) rel.Slice(t * nh * relLen, nh * relLen).Clear();
            if (!lw.Local && Mutant != Inkling2Mutant.NoLogScale)
            {
                float tau = Inkling2Ops.LogScale(pos0 + t, _c.LogScaleFloor, _c.LogScaleAlpha);
                TensorPrimitives.Multiply(q4.Slice(t * qW, qW), tau, q4.Slice(t * qW, qW));
                TensorPrimitives.Multiply(rel.Slice(t * nh * relLen, nh * relLen), tau, rel.Slice(t * nh * relLen, nh * relLen));
            }
            ss.Store(pos0 + t, kRaw.Slice(t * kvW, kvW), vRaw.Slice(t * kvW, kvW));
        }
        Trace?.Invoke($"slot{slot}.Q", q4, T, qW);
        Trace?.Invoke($"slot{slot}.K", kRaw, T, kvW);
        Trace?.Invoke($"slot{slot}.V", vRaw, T, kvW);

        var attn = A.Alloc<float>(T * qW);
        Attend(lw, ss, A, q4, rel, sink, attn, T, pos0);
        for (int i = 0; i < attn.Length; i++) attn[i] *= Inkling2Ops.Sigmoid(gate[i]);

        var attnOut = A.Alloc<float>(T * H);
        GemmSpan(lw.Wo, attn, attnOut, T);
        ss.AttnPostConv.Run(T, attnOut, lw.PostConv.Span, default, 0, attnOut, convScratch);
        Trace?.Invoke($"slot{slot}.attn_out", attnOut, T, H);

        // The first layer of each block starts a new residual stream (its input joins the mixed blocks); the others add the residual.
        Span<float> ffnInp;
        if (ip % _c.AttnResBlockSize == 0)
        {
            var nb = NormedCopy(inpSA, T, sc.Call);
            blocks.Add((nint)Unsafe.AsPointer(ref MemoryMarshal.GetReference(nb)));
            ffnInp = attnOut;
        }
        else
        {
            ffnInp = A.Alloc<float>(T * H);
            TensorPrimitives.Add(inpSA, attnOut, ffnInp);
        }

        // ── MoE ──
        var m = DepthMix(sc, blocks, ffnInp, lw.FfnResQuery.Span, 1, T, A);
        if (!messages.IsEmpty) m = Control(sc, m, messages, loop!, 3, T);
        m = RmsNormRows(sc, m, lw.FfnNorm, T);
        var moe = A.Alloc<float>(T * H);
        ForwardMoe(lw, A, m, moe, T);
        ss.FfnPostConv.Run(T, moe, lw.FfnPostConv.Span, default, 0, moe, convScratch);
        Trace?.Invoke($"slot{slot}.ffn_out", moe, T, H);

        var outp = outArena.Alloc<float>(T * H);
        TensorPrimitives.Add(ffnInp, moe, outp);
        return outp;
    }

    /// <summary>The <paramref name="q"/>-th query result of every token of a <c>[T][nq][H]</c> mix, as a contiguous <c>[T][H]</c> copy.</summary>
    private Span<float> PickQuery(Inkling2Arena A, ReadOnlySpan<float> mixed, int q, int nq, int T)
    {
        int H = _hidden;
        var x = A.Alloc<float>(T * H);
        for (int t = 0; t < T; t++) mixed.Slice((t * nq + q) * H, H).CopyTo(x.Slice(t * H, H));
        return x;
    }

    /// <summary>
    /// Attention of T tokens over the slot's history: <c>softmax over keys and one zero sink of (q.k/sqrt(d) + bias[d].r - sink_logit)</c>, i.e.
    /// <c>exp(l_k) / (1 + sum exp(l_k))</c> with <c>l_k = q.k * scale + bias(distance) - t</c>. Local layers see keys at distance &lt; window; global
    /// layers see every earlier key, with zero bias beyond the relative extent.
    /// </summary>
    private void Attend(Layer lw, Inkling2SlotState ss, Inkling2Arena A, Span<float> q, ReadOnlySpan<float> rel, ReadOnlySpan<float> sink, Span<float> outp, int T, int pos0)
    {
        int hd = _headDim, nh = _heads, nkv = _kvHeads, qW = _qW, group = nh / nkv, relLen = _c.RelativeLength;
        float scale = 1f / MathF.Sqrt(hd);
        var relBias = lw.RelBias.Span;
        int maxKeys = lw.Local ? Math.Min(_c.SlidingWindow, pos0 + T) : pos0 + T;
        Span<float> logits = A.Alloc<float>(nh * maxKeys);
        Span<float> kRow = A.Alloc<float>(hd), vRow = A.Alloc<float>(hd);
        for (int t = 0; t < T; t++)
        {
            int p = pos0 + t;
            int lo = lw.Local ? Math.Max(0, p - _c.SlidingWindow + 1) : 0;
            int n = p - lo + 1;
            for (int g = 0; g < nkv; g++)
            {
                for (int j = 0; j < n; j++)
                {
                    ss.LoadK(lo + j, g, hd, kRow);
                    int d = p - (lo + j);
                    for (int s = 0; s < group; s++)
                    {
                        int h = g * group + s;
                        float l = TensorPrimitives.Dot(q.Slice(t * qW + h * hd, hd), kRow) * scale
                                  + Inkling2Ops.RelativeBias(relBias, lw.Extent, relLen, rel.Slice((t * nh + h) * relLen, relLen), d)
                                  - sink[t * nh + h];
                        logits[h * maxKeys + j] = l;
                    }
                }
            }
            for (int h = 0; h < nh; h++)
            {
                var l = logits.Slice(h * maxKeys, n);
                float max = MathF.Max(TensorPrimitives.Max(l), 0f);          // the implicit zero sink joins the max and the sum
                double sum = MathF.Exp(-max);
                for (int j = 0; j < n; j++) { l[j] = MathF.Exp(l[j] - max); sum += l[j]; }
                float inv = (float)(1.0 / sum);
                for (int j = 0; j < n; j++) l[j] *= inv;
            }
            for (int g = 0; g < nkv; g++)
            {
                for (int s = 0; s < group; s++) outp.Slice(t * qW + (g * group + s) * hd, hd).Clear();
                for (int j = 0; j < n; j++)
                {
                    ss.LoadV(lo + j, g, hd, vRow);
                    for (int s = 0; s < group; s++)
                    {
                        int h = g * group + s;
                        var o = outp.Slice(t * qW + h * hd, hd);
                        TensorPrimitives.MultiplyAdd(vRow, logits[h * maxKeys + j], o, o);
                    }
                }
            }
        }
    }

    // ───────────────────────────── MoE ─────────────────────────────

    /// <summary>
    /// SITU MoE (llama.cpp <c>build_moe_ffn</c> with <c>LLM_FFN_SITU</c>, sigmoid gating, selection bias, normalised weights): router logits in F32,
    /// <c>p = sigmoid(logits)</c>, top-k of <c>p + exp_probs_b</c>, the selected UNBIASED <c>p</c> normalised to sum 1 (+1e-20) and multiplied by
    /// <c>expert_weights_scale</c>. Each expert is a merged gate_up bank (gate rows first) then SITU then <c>down</c>; NVFP4 banks carry a per-expert
    /// <c>.scale</c> that multiplies the matmul output (llama.cpp <c>build_lora_mm_id</c>).
    /// </summary>
    private void ForwardMoe(Layer lw, Inkling2Arena A, ReadOnlySpan<float> x, Span<float> y, int T)
    {
        int H = _hidden, E = _c.ExpertCount, K = _c.ExpertUsedCount, ff = _c.ExpertFfLength;
        var logits = A.Alloc<float>(T * E);
        var selExpert = A.Alloc<int>(T * K);
        var selWeight = A.Alloc<float>(T * K);
        GemmSpan(lw.Router, x.Slice(0, T * H), logits, T);
        Span<float> scored = stackalloc float[E];
        Span<float> sig = stackalloc float[E];
        var bias = lw.ExpBias.Span;
        for (int t = 0; t < T; t++)
        {
            for (int e = 0; e < E; e++)
            {
                sig[e] = Inkling2Ops.Sigmoid(logits[t * E + e]);
                scored[e] = sig[e] + bias[e];
            }
            SelectTopK(scored, K, selExpert.Slice(t * K, K));
            float sum = 0f;
            for (int i = 0; i < K; i++) sum += sig[selExpert[t * K + i]];
            sum += 1e-20f;
            for (int i = 0; i < K; i++)
                selWeight[t * K + i] = sig[selExpert[t * K + i]] / sum * _c.ExpertWeightsScale;
        }
        y.Slice(0, T * H).Clear();
        ExecuteExperts(lw, A, x, y, selExpert, selWeight, T, K, E, H, ff);
    }

    private void ExecuteExperts(Layer lw, Inkling2Arena A, ReadOnlySpan<float> x, Span<float> y, ReadOnlySpan<int> selExpert, ReadOnlySpan<float> selWeight, int T, int K, int E, int H, int ff)
    {
        // Bucket (token, slot) assignments by expert so each expert runs one batched GEMM over its tokens.
        var count = A.Alloc<int>(E + 1); var cursor = A.Alloc<int>(E + 1);
        var tokenOf = A.Alloc<int>(T * K); var weightOf = A.Alloc<float>(T * K);
        count.Clear();
        for (int i = 0; i < T * K; i++) count[selExpert[i] + 1]++;
        for (int e = 0; e < E; e++) count[e + 1] += count[e];
        count.CopyTo(cursor);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < K; i++)
            {
                int e = selExpert[t * K + i], at = cursor[e]++;
                tokenOf[at] = t; weightOf[at] = selWeight[t * K + i];
            }
        var guScale = Mutant == Inkling2Mutant.NoExpertScale ? default : lw.GateUpScale;
        var dnScale = Mutant == Inkling2Mutant.NoExpertScale ? default : lw.DownScale;
        float beta = _c.SituBeta, lb = _c.SituLinearBeta;
        int maxN = 0;
        for (int e = 0; e < E; e++) maxN = Math.Max(maxN, count[e + 1] - count[e]);
        var xin = A.Alloc<float>(maxN * H);
        var gu = A.Alloc<float>(maxN * 2 * ff);
        var act = A.Alloc<float>(maxN * ff);
        var yo = A.Alloc<float>(maxN * H);
        for (int e = 0; e < E; e++)
        {
            int n = count[e + 1] - count[e], start = count[e];
            if (n == 0) continue;
            for (int i = 0; i < n; i++) x.Slice(tokenOf[start + i] * H, H).CopyTo(xin.Slice(i * H, H));
            fixed (float* xp = xin) fixed (float* gp = gu) fixed (float* ap = act) fixed (float* yp = yo)
            {
                Gemm(lw.GateUp + (nint)(e * lw.GateUpExpertBytes), lw.GateUpQt, xp, gp, 2 * ff, H, n, _pool);
                if (!guScale.IsEmpty) TensorPrimitives.Multiply(gu.Slice(0, n * 2 * ff), guScale.Span[e], gu.Slice(0, n * 2 * ff));
                for (int i = 0; i < n; i++)
                {
                    var row = gu.Slice(i * 2 * ff, 2 * ff);
                    var a = act.Slice(i * ff, ff);
                    for (int j = 0; j < ff; j++) a[j] = Inkling2Ops.Situ(row[j], row[ff + j], beta, lb);
                }
                Gemm(lw.Down + (nint)(e * lw.DownExpertBytes), lw.DownQt, ap, yp, H, ff, n, _pool);
                if (!dnScale.IsEmpty) TensorPrimitives.Multiply(yo.Slice(0, n * H), dnScale.Span[e], yo.Slice(0, n * H));
            }
            for (int i = 0; i < n; i++)
            {
                float w = weightOf[start + i];
                var dst = y.Slice(tokenOf[start + i] * H, H);
                TensorPrimitives.MultiplyAdd(yo.Slice(i * H, H), w, dst, dst);
            }
        }
    }

    /// <summary>Indices of the <paramref name="k"/> largest values, descending; ties resolve to the lower index.</summary>
    internal static void SelectTopK(ReadOnlySpan<float> v, int k, Span<int> idx)
    {
        Span<float> best = stackalloc float[k];
        int filled = 0;
        for (int e = 0; e < v.Length; e++)
        {
            float s = v[e];
            if (filled == k && s <= best[k - 1]) continue;
            int pos = filled < k ? filled++ : k - 1;
            while (pos > 0 && best[pos - 1] < s) { best[pos] = best[pos - 1]; idx[pos] = idx[pos - 1]; pos--; }
            best[pos] = s; idx[pos] = e;
        }
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        _dspark?.Dispose();
        DotLLM.Cpu.Kernels.CpuKernelScratch.Trim(null);
        if (_ownsPool) _pool?.Dispose();
        _defaultState.Dispose();
        lock (_scratchPool) { foreach (var sc in _scratchPool) sc.Dispose(); _scratchPool.Clear(); }
        foreach (var p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        GC.SuppressFinalize(this);
    }
}
