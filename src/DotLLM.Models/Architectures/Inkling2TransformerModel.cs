using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Load-time storage choices of <see cref="Inkling2TransformerModel"/> that the user rules on (issue #2026, epic #2025). The defaults are the
/// GGUF's own storage so a plain load reproduces the file; the narrower variants exist to be measured (KL / greedy identity) first.
/// </summary>
/// <param name="LoopController">
/// Storage of the loop-controller weights (<c>loop_msg_query</c>, <c>loop_*_ctrl_down/gain/forget</c>; F32 in the GGUF, ~113 MB). F32 reads them in
/// place; <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.F16"/> and <see cref="QuantizationType.BF16"/> make a smaller copy at load.
/// </param>
/// <param name="ConvState">Storage of the carried short-convolution inputs.</param>
public sealed record Inkling2Options(
    QuantizationType LoopController = QuantizationType.F32,
    Inkling2ConvStateStorage ConvState = Inkling2ConvStateStorage.F32)
{
    /// <summary>
    /// Options from <c>DOTLLM_INKLING2_LOOP_CONTROLLER</c> (<c>f32|q8_0|f16|bf16</c>) and <c>DOTLLM_INKLING2_CONV_STATE</c> (<c>f32|f16|bf16</c>);
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
                    "f16" => Inkling2ConvStateStorage.F16,
                    "bf16" => Inkling2ConvStateStorage.BF16,
                    _ => throw new ArgumentException($"DOTLLM_INKLING2_CONV_STATE='{cs}': expected f32, f16 or bf16."),
                },
            };
        return o;
    }
}

/// <summary>Deliberately wrong implementations used to prove the oracle gate can fail (CLAUDE.md "Prove the Test Can Fail"). Test-only.</summary>
internal enum Inkling2Mutant
{
    None, TwoPasses, NoFinalSoftcap, NoSink, NoRelativeBias, NoLogScale, NoExpertScale, NoControl,
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
public sealed unsafe class Inkling2TransformerModel : IModel
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

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, Inkling2SequenceState state, bool lastTokenLogitsOnly)
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
        return ForwardBody(tokenIds, deviceId, state, lastTokenLogitsOnly);
    }

    private ITensor ForwardBody(ReadOnlySpan<int> tokenIds, int deviceId, Inkling2SequenceState st, bool lastOnly)
    {
        int T = tokenIds.Length, H = _hidden;
        int pos0 = st.Length;
        _pool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

        // Token embeddings: Q8_0 rows dequantised one at a time into the activation buffer, then the gain-ed RMS norm (llama.cpp: tok_norm).
        float[] cur = new float[T * H];
        var embType = _gguf!.TensorsByName[Inkling2Tensors.TokenEmbd].QuantizationType;
        long rowBytes = Dequantize.RowByteSize(H, embType);
        for (int t = 0; t < T; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {_vocab}).");
            var row = cur.AsSpan(t * H, H);
            Dequantize.ToFloat32(_tokenEmbed + (nint)(id * rowBytes), H, embType, row);
            Inkling2Ops.RmsNorm(row, _tokEmbNorm.Span, _eps, row);
        }
        Trace?.Invoke("embed", cur, T, H);

        var blocks = new List<float[]>();   // RMS-normalised (gamma-free) block inputs, immutable once pushed (llama.cpp mix_stack cache)
        int start = _c.LoopStart, end = _c.LoopEnd;
        int coreStart = _c.LoopPasses > 1 ? start : _nPhys;

        for (int il = 0; il < coreStart; il++) cur = ForwardLayer(il, il, cur, blocks, null, null, T, pos0, st);

        if (_c.LoopPasses > 1)
        {
            float[] prelude = cur;
            var preludeBlocks = new List<float[]>(blocks);
            int core = end - start;
            for (int pass = 0; pass < (Mutant == Inkling2Mutant.TwoPasses ? 2 : _c.LoopPasses); pass++)
            {
                List<float[]>? bank = null; float[]? bankOut = null;
                if (pass > 0)
                {
                    // The bank of the previous pass: its blocks after the first core block, and its output (llama.cpp inkling2-loop.cpp).
                    bank = blocks.GetRange(preludeBlocks.Count + 1, blocks.Count - preludeBlocks.Count - 1);
                    bankOut = cur;
                    cur = prelude;
                    blocks = new List<float[]>(preludeBlocks);
                }
                for (int ip = start; ip < end; ip++)
                {
                    int slot = pass == 0 ? ip : _nPhys + (pass - 1) * core + (ip - start);
                    float[]? messages = null;
                    LoopWeights? loop = null;
                    if (pass > 0)
                    {
                        loop = _layers[ip].Loop!;
                        if (Mutant != Inkling2Mutant.NoControl) messages = DepthMix(bank!, bankOut!, MatRows(loop.MsgQuery), 4, T);
                    }
                    cur = ForwardLayer(slot, ip, cur, blocks, messages, loop, T, pos0, st);
                    Trace?.Invoke($"pass{pass}.l{ip}", cur, T, H);
                }
            }
            for (int il = end; il < _nPhys; il++) cur = ForwardLayer(il, il, cur, blocks, null, null, T, pos0, st);
        }

        // Output: depth mix with output_res_query, gain-ed RMS norm, LM head (unpadded rows only), final logit softcap.
        float[] mixed = DepthMix(blocks, cur, _outputResQuery.Span.ToArray(), 1, T);
        for (int t = 0; t < T; t++)
            Inkling2Ops.RmsNorm(mixed.AsSpan(t * H, H), _outputNorm.Span, _eps, mixed.AsSpan(t * H, H));
        Trace?.Invoke("result_norm", mixed, T, H);

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

    /// <summary>The rows of a (possibly narrowed) matrix as F32, for the small tensors that are used as dot-product queries.</summary>
    private float[] MatRows(MatRef m)
    {
        long count = (long)m.In * m.Out;
        var r = new float[count];
        Dequantize.ToFloat32(m.Ptr, count, m.Qt, r);
        return r;
    }

    /// <summary>
    /// Depth mixing for T tokens (see <see cref="Inkling2Ops.DepthMixToken"/>): <paramref name="blocks"/> are already RMS-normalised, the
    /// current state is normalised here. Returns <c>[T][nq][H]</c>.
    /// </summary>
    private float[] DepthMix(List<float[]> blocks, float[] cur, float[] queries, int nq, int T)
    {
        int H = _hidden, count = blocks.Count + 1;
        var output = new float[T * nq * H];
        float[] states = ArrayPool<float>.Shared.Rent(count * H);
        try
        {
            for (int t = 0; t < T; t++)
            {
                for (int b = 0; b < blocks.Count; b++) blocks[b].AsSpan(t * H, H).CopyTo(states.AsSpan(b * H, H));
                Inkling2Ops.RmsNorm(cur.AsSpan(t * H, H), default, Inkling2Ops.MixEps, states.AsSpan(blocks.Count * H, H));
                Inkling2Ops.DepthMixToken(states, count, queries, nq, H, output.AsSpan(t * nq * H, nq * H));
            }
        }
        finally { ArrayPool<float>.Shared.Return(states); }
        return output;
    }

    private float[] NormedCopy(float[] x, int T)
    {
        int H = _hidden;
        var r = new float[T * H];
        for (int t = 0; t < T; t++) Inkling2Ops.RmsNorm(x.AsSpan(t * H, H), default, Inkling2Ops.MixEps, r.AsSpan(t * H, H));
        return r;
    }

    /// <summary>
    /// The loop-controller combine for T tokens (llama.cpp <c>graph::control</c>): the rank-r gate is a projection of
    /// <c>[rms(cur); rms(message)]</c>; <c>out = (cur - tanh(forget) * cur) + tanh(gain) * message</c>. <paramref name="msgs"/> is
    /// <c>[T][4][H]</c> and <paramref name="ctrl"/> selects the controller (Q, K, V, FFN).
    /// </summary>
    private float[] Control(float[] cur, float[] msgs, LoopWeights loop, int ctrl, int T)
    {
        int H = _hidden, rank = _c.LoopGateRank;
        var gateIn = new float[T * 2 * H];
        var msg = new float[T * H];
        for (int t = 0; t < T; t++)
        {
            msgs.AsSpan((t * 4 + ctrl) * H, H).CopyTo(msg.AsSpan(t * H, H));
            Inkling2Ops.RmsNorm(cur.AsSpan(t * H, H), default, Inkling2Ops.MixEps, gateIn.AsSpan(t * 2 * H, H));
            Inkling2Ops.RmsNorm(msg.AsSpan(t * H, H), default, Inkling2Ops.MixEps, gateIn.AsSpan(t * 2 * H + H, H));
        }
        var gate = new float[T * rank];
        GemmSpan(loop.Down[ctrl], gateIn, gate, T);
        var gainPre = new float[T * H];
        var forgetPre = new float[T * H];
        GemmSpan(loop.Gain[ctrl], gate, gainPre, T);
        GemmSpan(loop.Forget[ctrl], gate, forgetPre, T);
        var o = new float[T * H];
        for (int t = 0; t < T; t++)
            Inkling2Ops.ControlCombine(cur.AsSpan(t * H, H), msg.AsSpan(t * H, H), gainPre.AsSpan(t * H, H), forgetPre.AsSpan(t * H, H), o.AsSpan(t * H, H));
        return o;
    }

    private float[] RmsNormRows(float[] x, F32Ref gamma, int T)
    {
        int H = _hidden;
        var r = new float[T * H];
        for (int t = 0; t < T; t++) Inkling2Ops.RmsNorm(x.AsSpan(t * H, H), gamma.Span, _eps, r.AsSpan(t * H, H));
        return r;
    }

    /// <summary>One transformer layer visit (llama.cpp <c>graph::build_layer</c>). Returns the layer output; appends to <paramref name="blocks"/> when this layer starts a new block.</summary>
    private float[] ForwardLayer(int slot, int ip, float[] cur, List<float[]> blocks, float[]? messages, LoopWeights? loop, int T, int pos0, Inkling2SequenceState st)
    {
        var lw = _layers[ip];
        int H = _hidden, hd = _headDim, nh = _heads, nkv = _kvHeads, kvW = _kvW, qW = _qW;
        float[] inpSA = cur;
        var ss = st.Slots[slot];
        ss.EnsureCapacity(pos0 + T);

        // Attention inputs: layer 0 reads the embedding; later layers mix the earlier blocks with cur using learned queries
        // (one for local layers, three — Q, K, V — for global ones).
        float[][] inputs = [cur, cur, cur];
        if (ip > 0)
        {
            int nq = lw.Local ? 1 : 3;
            float[] mixed = DepthMix(blocks, cur, lw.AttnResQuery.Span.ToArray(), nq, T);
            for (int j = 0; j < 3; j++)
            {
                int q = lw.Local ? 0 : j;
                var x = new float[T * H];
                for (int t = 0; t < T; t++) mixed.AsSpan((t * nq + q) * H, H).CopyTo(x.AsSpan(t * H, H));
                inputs[j] = x;
            }
        }
        if (messages is not null)
            for (int j = 0; j < 3; j++) inputs[j] = Control(inputs[j], messages, loop!, j, T);

        if (lw.Local && messages is not null)
            for (int j = 0; j < 3; j++) inputs[j] = RmsNormRows(inputs[j], lw.AttnNorm, T);
        else if (lw.Local)
        {
            var n = RmsNormRows(inputs[0], lw.AttnNorm, T);
            inputs[0] = inputs[1] = inputs[2] = n;
        }
        else
        {
            inputs[0] = RmsNormRows(inputs[0], lw.QInpNorm, T);
            inputs[1] = RmsNormRows(inputs[1], lw.KInpNorm, T);
            inputs[2] = RmsNormRows(inputs[2], lw.VInpNorm, T);
        }

        // ── self-attention ──
        var q4 = new float[T * qW];
        var kRaw = new float[T * kvW];
        var vRaw = new float[T * kvW];
        int relLen = _c.RelativeLength;
        var rel = new float[T * relLen * nh];
        var sink = new float[T * nh];
        var gate = new float[T * qW];
        GemmSpan(lw.Wq, inputs[0], q4, T);
        GemmSpan(lw.Wk, inputs[1], kRaw, T);
        GemmSpan(lw.Wv, inputs[2], vRaw, T);
        GemmSpan(lw.Rel, inputs[0], rel, T);
        GemmSpan(lw.Sink, inputs[0], sink, T);
        GemmSpan(lw.Gate, inputs[0], gate, T);
        if (Mutant == Inkling2Mutant.NoSink) Array.Clear(sink);

        float[] kDyn = [], vDyn = [];
        if (!lw.Local)
        {
            kDyn = new float[T * _c.ConvKernel * nkv];
            vDyn = new float[T * _c.ConvKernel * nkv];
            GemmSpan(lw.KConvProj, inputs[1], kDyn, T);
            GemmSpan(lw.VConvProj, inputs[2], vDyn, T);
        }
        ss.KConv.Run(T, kRaw, lw.KConv.Span, kDyn, nkv, kRaw);
        ss.VConv.Run(T, vRaw, lw.VConv.Span, vDyn, nkv, vRaw);

        var qClip = lw.QClip.Span; var relClip = lw.RelClip.Span;
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < nh; h++)
            {
                Inkling2Ops.ClipL2(q4.AsSpan(t * qW + h * hd, hd), qClip.Slice(h * hd, hd), _c.QkClip);
                Inkling2Ops.ClipL2(rel.AsSpan((t * nh + h) * relLen, relLen), relClip.Slice(h * relLen, relLen), _c.RelativeClip);
            }
            for (int g = 0; g < nkv; g++)
            {
                Inkling2Ops.ClipL2(kRaw.AsSpan(t * kvW + g * hd, hd), default, _c.QkClip);
                Inkling2Ops.RmsNorm(vRaw.AsSpan(t * kvW + g * hd, hd), default, _eps, vRaw.AsSpan(t * kvW + g * hd, hd));
            }
            if (Mutant == Inkling2Mutant.NoRelativeBias) rel.AsSpan(t * nh * relLen, nh * relLen).Clear();
            if (!lw.Local && Mutant != Inkling2Mutant.NoLogScale)
            {
                float tau = Inkling2Ops.LogScale(pos0 + t, _c.LogScaleFloor, _c.LogScaleAlpha);
                TensorPrimitives.Multiply(q4.AsSpan(t * qW, qW), tau, q4.AsSpan(t * qW, qW));
                TensorPrimitives.Multiply(rel.AsSpan(t * nh * relLen, nh * relLen), tau, rel.AsSpan(t * nh * relLen, nh * relLen));
            }
            ss.Store(pos0 + t, kRaw.AsSpan(t * kvW, kvW), vRaw.AsSpan(t * kvW, kvW));
        }
        Trace?.Invoke($"slot{slot}.Q", q4, T, qW);
        Trace?.Invoke($"slot{slot}.K", kRaw, T, kvW);
        Trace?.Invoke($"slot{slot}.V", vRaw, T, kvW);

        var attn = new float[T * qW];
        Attend(lw, ss, q4, rel, sink, attn, T, pos0);
        for (int i = 0; i < attn.Length; i++) attn[i] *= Inkling2Ops.Sigmoid(gate[i]);

        var attnOut = new float[T * H];
        GemmSpan(lw.Wo, attn, attnOut, T);
        ss.AttnPostConv.Run(T, attnOut, lw.PostConv.Span, default, 0, attnOut);
        Trace?.Invoke($"slot{slot}.attn_out", attnOut, T, H);

        // The first layer of each block starts a new residual stream (its input joins the mixed blocks); the others add the residual.
        float[] ffnInp;
        if (ip % _c.AttnResBlockSize == 0)
        {
            blocks.Add(NormedCopy(inpSA, T));
            ffnInp = attnOut;
        }
        else
        {
            ffnInp = new float[T * H];
            TensorPrimitives.Add(inpSA, attnOut, ffnInp);
        }

        // ── MoE ──
        float[] m = DepthMix(blocks, ffnInp, lw.FfnResQuery.Span.ToArray(), 1, T);
        if (messages is not null) m = Control(m, messages, loop!, 3, T);
        m = RmsNormRows(m, lw.FfnNorm, T);
        var moe = new float[T * H];
        ForwardMoe(lw, m, moe, T);
        ss.FfnPostConv.Run(T, moe, lw.FfnPostConv.Span, default, 0, moe);
        Trace?.Invoke($"slot{slot}.ffn_out", moe, T, H);

        var outp = new float[T * H];
        TensorPrimitives.Add(ffnInp, moe, outp);
        return outp;
    }

    /// <summary>
    /// Attention of T tokens over the slot's history: <c>softmax over keys and one zero sink of (q.k/sqrt(d) + bias[d].r - sink_logit)</c>, i.e.
    /// <c>exp(l_k) / (1 + sum exp(l_k))</c> with <c>l_k = q.k * scale + bias(distance) - t</c>. Local layers see keys at distance &lt; window; global
    /// layers see every earlier key, with zero bias beyond the relative extent.
    /// </summary>
    private void Attend(Layer lw, Inkling2SlotState ss, float[] q, float[] rel, float[] sink, float[] outp, int T, int pos0)
    {
        int hd = _headDim, nh = _heads, nkv = _kvHeads, qW = _qW, group = nh / nkv, relLen = _c.RelativeLength;
        float scale = 1f / MathF.Sqrt(hd);
        var relBias = lw.RelBias.Span;
        int maxKeys = lw.Local ? Math.Min(_c.SlidingWindow, pos0 + T) : pos0 + T;
        float[] logits = ArrayPool<float>.Shared.Rent(nh * maxKeys);
        float[] kRow = new float[hd], vRow = new float[hd];
        try
        {
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
                            float l = TensorPrimitives.Dot(q.AsSpan(t * qW + h * hd, hd), kRow) * scale
                                      + Inkling2Ops.RelativeBias(relBias, lw.Extent, relLen, rel.AsSpan((t * nh + h) * relLen, relLen), d)
                                      - sink[t * nh + h];
                            logits[h * maxKeys + j] = l;
                        }
                    }
                }
                for (int h = 0; h < nh; h++)
                {
                    var l = logits.AsSpan(h * maxKeys, n);
                    float max = MathF.Max(TensorPrimitives.Max(l), 0f);          // the implicit zero sink joins the max and the sum
                    double sum = MathF.Exp(-max);
                    for (int j = 0; j < n; j++) { l[j] = MathF.Exp(l[j] - max); sum += l[j]; }
                    float inv = (float)(1.0 / sum);
                    for (int j = 0; j < n; j++) l[j] *= inv;
                }
                for (int g = 0; g < nkv; g++)
                {
                    for (int s = 0; s < group; s++) outp.AsSpan(t * qW + (g * group + s) * hd, hd).Clear();
                    for (int j = 0; j < n; j++)
                    {
                        ss.LoadV(lo + j, g, hd, vRow);
                        for (int s = 0; s < group; s++)
                        {
                            int h = g * group + s;
                            var o = outp.AsSpan(t * qW + h * hd, hd);
                            TensorPrimitives.MultiplyAdd(vRow, logits[h * maxKeys + j], o, o);
                        }
                    }
                }
            }
        }
        finally { ArrayPool<float>.Shared.Return(logits); }
    }

    // ───────────────────────────── MoE ─────────────────────────────

    /// <summary>
    /// SITU MoE (llama.cpp <c>build_moe_ffn</c> with <c>LLM_FFN_SITU</c>, sigmoid gating, selection bias, normalised weights): router logits in F32,
    /// <c>p = sigmoid(logits)</c>, top-k of <c>p + exp_probs_b</c>, the selected UNBIASED <c>p</c> normalised to sum 1 (+1e-20) and multiplied by
    /// <c>expert_weights_scale</c>. Each expert is a merged gate_up bank (gate rows first) then SITU then <c>down</c>; NVFP4 banks carry a per-expert
    /// <c>.scale</c> that multiplies the matmul output (llama.cpp <c>build_lora_mm_id</c>).
    /// </summary>
    private void ForwardMoe(Layer lw, float[] x, float[] y, int T)
    {
        int H = _hidden, E = _c.ExpertCount, K = _c.ExpertUsedCount, ff = _c.ExpertFfLength;
        float[] logits = ArrayPool<float>.Shared.Rent(T * E);
        int[] selExpert = ArrayPool<int>.Shared.Rent(T * K);
        float[] selWeight = ArrayPool<float>.Shared.Rent(T * K);
        try
        {
            GemmSpan(lw.Router, x.AsSpan(0, T * H), logits, T);
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
                SelectTopK(scored, K, selExpert.AsSpan(t * K, K));
                float sum = 0f;
                for (int i = 0; i < K; i++) sum += sig[selExpert[t * K + i]];
                sum += 1e-20f;
                for (int i = 0; i < K; i++)
                    selWeight[t * K + i] = sig[selExpert[t * K + i]] / sum * _c.ExpertWeightsScale;
            }
            Array.Clear(y, 0, T * H);
            ExecuteExperts(lw, x, y, selExpert, selWeight, T, K, E, H, ff);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(logits); ArrayPool<int>.Shared.Return(selExpert); ArrayPool<float>.Shared.Return(selWeight);
        }
    }

    private void ExecuteExperts(Layer lw, float[] x, float[] y, int[] selExpert, float[] selWeight, int T, int K, int E, int H, int ff)
    {
        // Bucket (token, slot) assignments by expert so each expert runs one batched GEMM over its tokens.
        int[] count = ArrayPool<int>.Shared.Rent(E + 1), cursor = ArrayPool<int>.Shared.Rent(E + 1);
        int[] tokenOf = ArrayPool<int>.Shared.Rent(T * K); float[] weightOf = ArrayPool<float>.Shared.Rent(T * K);
        try
        {
            Array.Clear(count, 0, E + 1);
            for (int i = 0; i < T * K; i++) count[selExpert[i] + 1]++;
            for (int e = 0; e < E; e++) count[e + 1] += count[e];
            Array.Copy(count, cursor, E + 1);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < K; i++)
                {
                    int e = selExpert[t * K + i], at = cursor[e]++;
                    tokenOf[at] = t; weightOf[at] = selWeight[t * K + i];
                }
            var guScale = Mutant == Inkling2Mutant.NoExpertScale ? default : lw.GateUpScale;
            var dnScale = Mutant == Inkling2Mutant.NoExpertScale ? default : lw.DownScale;
            float beta = _c.SituBeta, lb = _c.SituLinearBeta;
            for (int e = 0; e < E; e++)
            {
                int n = count[e + 1] - count[e], start = count[e];
                if (n == 0) continue;
                float[] xin = ArrayPool<float>.Shared.Rent(n * H);
                float[] gu = ArrayPool<float>.Shared.Rent(n * 2 * ff);
                float[] act = ArrayPool<float>.Shared.Rent(n * ff);
                float[] yo = ArrayPool<float>.Shared.Rent(n * H);
                try
                {
                    for (int i = 0; i < n; i++) x.AsSpan(tokenOf[start + i] * H, H).CopyTo(xin.AsSpan(i * H, H));
                    fixed (float* xp = xin) fixed (float* gp = gu) fixed (float* ap = act) fixed (float* yp = yo)
                    {
                        Gemm(lw.GateUp + (nint)(e * lw.GateUpExpertBytes), lw.GateUpQt, xp, gp, 2 * ff, H, n, _pool);
                        if (!guScale.IsEmpty) TensorPrimitives.Multiply(gu.AsSpan(0, n * 2 * ff), guScale.Span[e], gu.AsSpan(0, n * 2 * ff));
                        for (int i = 0; i < n; i++)
                        {
                            var row = gu.AsSpan(i * 2 * ff, 2 * ff);
                            var a = act.AsSpan(i * ff, ff);
                            for (int j = 0; j < ff; j++) a[j] = Inkling2Ops.Situ(row[j], row[ff + j], beta, lb);
                        }
                        Gemm(lw.Down + (nint)(e * lw.DownExpertBytes), lw.DownQt, ap, yp, H, ff, n, _pool);
                        if (!dnScale.IsEmpty) TensorPrimitives.Multiply(yo.AsSpan(0, n * H), dnScale.Span[e], yo.AsSpan(0, n * H));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        float w = weightOf[start + i];
                        var dst = y.AsSpan(tokenOf[start + i] * H, H);
                        TensorPrimitives.MultiplyAdd(yo.AsSpan(i * H, H), w, dst, dst);
                    }
                }
                finally { ArrayPool<float>.Shared.Return(xin); ArrayPool<float>.Shared.Return(gu); ArrayPool<float>.Shared.Return(act); ArrayPool<float>.Shared.Return(yo); }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(count); ArrayPool<int>.Shared.Return(cursor);
            ArrayPool<int>.Shared.Return(tokenOf); ArrayPool<float>.Shared.Return(weightOf);
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
        DotLLM.Cpu.Kernels.CpuKernelScratch.Trim(null);
        if (_ownsPool) _pool?.Dispose();
        _defaultState.Dispose();
        foreach (var p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        GC.SuppressFinalize(this);
    }
}
