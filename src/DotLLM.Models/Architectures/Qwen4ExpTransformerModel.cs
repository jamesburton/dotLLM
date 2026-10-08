using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>Receives <c>(name, data, rows, cols)</c> tensors from <see cref="Qwen4ExpTransformerModel"/>'s trace hook.</summary>
internal delegate void Qwen4ExpTraceSink(string name, ReadOnlySpan<float> data, int rows, int cols);

/// <summary>
/// CPU reference ("oracle") forward for Qwen4-Exp / Qwen3.8-Flash-Next (<c>qwen4exp</c>): 48 blocks of
/// <c>(GDN | QSA) + 512-expert softmax MoE</c> on a 4-stream gated residual, an n-gram hash embedding injected before layer 1,
/// and a head mixer in place of the final norm. Correctness over speed: it is the numerical reference the Vulkan/CUDA
/// ports are validated against (HF <c>qwen4_exp</c> → this → GPU).
/// </summary>
/// <remarks>
/// <para>Per block (<see cref="Qwen4ExpGatedResidual"/> documents the maths): optional PLE add, GR read → token mixer →
/// GR write, GR read → MoE → GR write. The token mixer is Gated-DeltaNet (the Qwen3.5 GDN with a <b>sigmoid</b> output gate,
/// the one numerical difference llama.cpp's <c>qwen4exp</c> graph documents) or QSA attention (<see cref="Qwen4ExpQsaLayer"/>).
/// MoE: softmax over all experts → top-k → renormalise (no bias), plus one shared expert gated by <c>sigmoid(w·x)</c>.</para>
/// <para>Positions must continue the sequence state (<c>positions[i] == state.Length + i</c>); the model-owned default state
/// restarts when a call begins at position 0. The MTP block, vision tower and image-token PLE stand-in are not implemented.</para>
/// </remarks>
public sealed unsafe class Qwen4ExpTransformerModel : IModel
{
    private readonly record struct MatRef(nint Ptr, QuantizationType Qt, int In, int Out);

    private sealed class GrWeights
    {
        public required float[] Norm;
        public required MatRef Down, Up;
        public MatRef? Inject;
    }

    private sealed class Block
    {
        public required GrWeights AttnGr, FfnGr;
        public GdnTokenMixingWeights? Gdn;
        public int GdnOrdinal = -1;
        public Qwen4ExpQsaLayer? Qsa;
        public int QsaOrdinal = -1;
        public Qwen4ExpPleBranch? Ple;
        public int PleOrdinal = -1;
        public required MoeLayerWeights Moe;
    }

    private readonly GgufFile? _gguf;
    private readonly Block[] _blocks;
    private readonly GrWeights _head;
    private readonly nint _tokenEmbed;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly MatRef _output;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly Qwen4ExpConfig _q4;
    private readonly int _hidden, _hc, _lowRank, _numGdn, _numQsa;
    private readonly float _eps;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsPool;
    private readonly List<nint> _owned;
    private readonly Qwen4ExpSequenceState _defaultState;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _defaultState.Bytes;

    /// <summary>Test/diagnostic hook: receives <c>(name, data, rows, cols)</c> after each block and for the head (null = off, zero cost).</summary>
    internal Qwen4ExpTraceSink? Trace { get; set; }

    private Qwen4ExpTransformerModel(ModelConfig config, GgufFile? gguf, Block[] blocks, GrWeights head, nint tokenEmbed,
                                     QuantizationType tokenEmbedQt, MatRef output, ComputeThreadPool? pool, bool ownsPool,
                                     List<nint> owned, Qwen4ExpLoraContext lora)
    {
        _lora = lora;
        Config = config; _gguf = gguf; _blocks = blocks; _head = head; _tokenEmbed = tokenEmbed; _tokenEmbedQt = tokenEmbedQt;
        _output = output; _threadPool = pool; _ownsPool = ownsPool; _owned = owned;
        _gdn = config.GdnConfig!.Value;
        _q4 = config.Qwen4Exp!;
        _hidden = config.HiddenSize;
        _hc = _q4.HyperConnectionCount;
        _lowRank = _q4.HyperConnectionLowRank;
        _eps = config.NormEpsilon;
        _numGdn = blocks.Count(b => b.Gdn is not null);
        _numQsa = blocks.Count(b => b.Qsa is not null);
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>Allocates a fresh sequence state (native memory; dispose it when the sequence ends).</summary>
    public Qwen4ExpSequenceState CreateState()
    {
        var ple = new Qwen4ExpPleState?[_blocks.Length];
        var qsa = new Qwen4ExpQsaState?[_blocks.Length];
        for (int i = 0; i < _blocks.Length; i++)
        {
            ple[i] = _blocks[i].Ple?.CreateState();
            qsa[i] = _blocks[i].Qsa?.CreateState();
        }
        return new Qwen4ExpSequenceState(new GdnStateCache(_gdn, _numGdn), ple, qsa);
    }

    /// <summary>Toggles the QSA <see cref="Qwen4ExpQsaLayer.ForceDense"/> diagnostic on every QSA layer.</summary>
    /// <param name="value">True to bypass the indexer and attend densely.</param>
    public void SetForceDenseAttention(bool value)
    {
        foreach (var b in _blocks) if (b.Qsa is not null) b.Qsa.ForceDense = value;
    }

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Loads a Qwen4Exp model from an opened (possibly multi-shard) GGUF. The file must outlive the model.</summary>
    public static Qwen4ExpTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (config.Architecture != DotLLM.Core.Configuration.Architecture.Qwen4Exp || config.Qwen4Exp is not { } q4
            || config.GdnConfig is null || config.Moe is null || config.HybridLayout is null)
            throw new ArgumentException("Qwen4ExpTransformerModel requires a fully populated Architecture.Qwen4Exp configuration.", nameof(config));
        var tensors = gguf.TensorsByName;
        var problems = Qwen4ExpTensors.FindProblems(tensors, config, includeTrunk: true, includeMtp: config.NextnPredictLayers > 0);
        if (problems.Count > 0)
            throw new InvalidDataException("qwen4exp tensor table does not match the model contract: " +
                                           string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));
        var owned = new List<nint>();
        try
        {
            return Build(config, q4, gguf, tensors, threading, owned);
        }
        catch
        {
            foreach (nint p in owned) NativeMemory.AlignedFree((void*)p);
            throw;
        }
    }

    private static Qwen4ExpTransformerModel Build(ModelConfig config, Qwen4ExpConfig q4, GgufFile gguf,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ThreadingConfig threading, List<nint> owned)
    {
        int hidden = config.HiddenSize, hc = q4.HyperConnectionCount, hcDim = hc * hidden, lr = q4.HyperConnectionLowRank;
        var layout = config.HybridLayout!;
        var gdnCfg = config.GdnConfig!.Value;
        var pool = CreatePool(threading);
        var lora = new Qwen4ExpLoraContext(pool);

        MatRef Mat(string name)
        {
            var d = tensors[name];
            return new MatRef(gguf.TensorDataPointer(d), d.QuantizationType, d.Shape[0], d.Shape.Rank > 1 ? d.Shape[1] : 1);
        }
        float[] F32(string name, long count)
        {
            var d = tensors[name];
            var r = new float[count];
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, r);
            return r;
        }
        GrWeights Gr(string prefix, string norm, string down, string up, string? inject) => new()
        {
            Norm = F32(prefix + norm, hcDim),
            Down = Mat(prefix + down),
            Up = Mat(prefix + up),
            Inject = inject is null ? null : Mat(prefix + inject),
        };

        var embDesc = tensors[Qwen4ExpTensors.TokenEmbd];
        nint embPtr = gguf.TensorDataPointer(embDesc);
        MatRef output = tensors.ContainsKey(Qwen4ExpTensors.Output)
            ? Mat(Qwen4ExpTensors.Output)
            : new MatRef(embPtr, embDesc.QuantizationType, embDesc.Shape[0], embDesc.Shape[1]);
        var head = Gr("", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", null);

        // Rope tables for the QSA layers (attention and indexer share the rotary dim and theta).
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        float theta = config.RoPEConfig?.Theta ?? 10000.0f;
        int maxPos = config.MaxSequenceLength;
        float[] ropeCos = new float[(long)maxPos * (ropeDim / 2)], ropeSin = new float[ropeCos.Length];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, ropeCos, ropeSin);

        var blocks = new Block[config.NumLayers];
        int gdnOrd = 0, qsaOrd = 0, pleOrd = 0;
        for (int il = 0; il < config.NumLayers; il++)
        {
            string b = $"blk.{il}.";
            var block = new Block
            {
                AttnGr = Gr(b, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight"),
                FfnGr = Gr(b, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"),
                Moe = LoadMoe(il, gguf, tensors, config, owned),
            };

            if (layout.LayerKind[il] == HybridLayerKind.GatedDeltaNet)
            {
                block.Gdn = LoadGdn(b, gguf, tensors, gdnCfg);
                block.GdnOrdinal = gdnOrd++;
            }
            else
            {
                int nKv = layout.HeadCountKv[il];
                block.Qsa = new Qwen4ExpQsaLayer(hidden, config.NumAttentionHeads, nKv, config.HeadDim, ropeDim,
                    q4.IndexerHeadCount, q4.IndexerKeyLength, q4.IndexerBlockSize, q4.IndexerTopK, config.NormEpsilon,
                    F32(b + "attn_q_norm.weight", config.HeadDim), F32(b + "attn_k_norm.weight", config.HeadDim),
                    F32(b + "indexer.q_norm.weight", q4.IndexerKeyLength), F32(b + "indexer.k_norm.weight", q4.IndexerKeyLength),
                    ropeCos, ropeSin,
                    Projection(Mat(b + "attn_q.weight"), pool, lora, il, "q_proj"), Projection(Mat(b + "attn_k.weight"), pool, lora, il, "k_proj"),
                    Projection(Mat(b + "attn_v.weight"), pool, lora, il, "v_proj"), Projection(Mat(b + "attn_output.weight"), pool, lora, il, "o_proj"),
                    Projection(Mat(b + "indexer.q_proj.weight"), pool), Projection(Mat(b + "indexer.k_proj.weight"), pool));
                block.QsaOrdinal = qsaOrd++;
            }

            if (q4.Ple is { } ple && ple.Layers.Contains(il))
            {
                // module index = position in the (ascending) layer list: selects this module's own hash constants (HF ple_layer_index)
                int module = ple.Layers.ToList().IndexOf(il);
                var tdesc = tensors[Qwen4ExpTensors.PerLayerTokenEmbd];
                int rowDim = ple.RowDim;
                var convRaw = F32(b + "ple_conv1d.weight", (long)ple.ConvKernel * hcDim);   // [C][K] (K fastest)
                block.Ple = new Qwen4ExpPleBranch(
                    gguf.TensorDataPointer(tdesc), tdesc.QuantizationType, tdesc.Shape[1], rowDim,
                    ple.NgramSize, ple.HeadsPerNgram, ple.EosTokenId, ple.ConvKernel,
                    ple.MultipliersOf(module).Select(v => unchecked((long)v)).ToArray(),
                    ple.HeadOffsetsOf(module).Select(v => checked((long)v)).ToArray(),
                    ple.HeadVocabSizesOf(module).Select(v => checked((long)v)).ToArray(),
                    hc, hidden, config.NormEpsilon,
                    F32(b + "ple_norm_key.weight", hcDim), F32(b + "ple_norm_query.weight", hcDim), F32(b + "ple_norm_conv.weight", hcDim),
                    Qwen4ExpPle.TransposeConvWeight(convRaw, hcDim, ple.ConvKernel),
                    Projection(Mat(b + "ple_key.weight"), pool), Projection(Mat(b + "ple_value.weight"), pool));
                block.PleOrdinal = pleOrd++;
            }
            blocks[il] = block;
        }

        return new Qwen4ExpTransformerModel(config, gguf, blocks, head, embPtr, embDesc.QuantizationType, output,
                                            pool, ownsPool: pool is not null, owned, lora);
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

    private static GdnTokenMixingWeights LoadGdn(string b, GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, GatedDeltaNetConfig g)
    {
        int convDim = (2 * g.NKHead + g.NVHead) * g.DState;
        float[] F(string n, int count)
        {
            var d = t[n]; var r = new float[count];
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, r);
            return r;
        }
        var qkv = t[b + "attn_qkv.weight"]; var gate = t[b + "attn_gate.weight"]; var alpha = t[b + "ssm_alpha.weight"];
        var beta = t[b + "ssm_beta.weight"]; var outp = t[b + "ssm_out.weight"];
        return new GdnTokenMixingWeights
        {
            QkvWeight = gguf.TensorDataPointer(qkv), QkvQuantType = qkv.QuantizationType, QkvInputDim = qkv.Shape[0], QkvOutputDim = qkv.Shape[1],
            GateWeight = gguf.TensorDataPointer(gate), GateQuantType = gate.QuantizationType, GateInputDim = gate.Shape[0], GateOutputDim = gate.Shape[1],
            A = F(b + "ssm_a", g.NVHead),
            AlphaWeight = gguf.TensorDataPointer(alpha), AlphaQuantType = alpha.QuantizationType, AlphaInputDim = alpha.Shape[0], AlphaOutputDim = alpha.Shape[1],
            BetaWeight = gguf.TensorDataPointer(beta), BetaQuantType = beta.QuantizationType, BetaInputDim = beta.Shape[0], BetaOutputDim = beta.Shape[1],
            Conv1dWeight = F(b + "ssm_conv1d.weight", g.DConv * convDim),
            Conv1dBias = new float[convDim],
            DtBias = F(b + "ssm_dt.bias", g.NVHead),
            SsmNormWeight = F(b + "ssm_norm.weight", g.DState),
            OutWeight = gguf.TensorDataPointer(outp), OutQuantType = outp.QuantizationType, OutInputDim = outp.Shape[0], OutOutputDim = outp.Shape[1],
        };
    }

    private static MoeLayerWeights LoadMoe(int il, GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t,
                                           ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe!;
        string p = $"blk.{il}.";
        int hidden = config.HiddenSize, ne = moe.NumExperts, inter = moe.MoeIntermediateSize;
        int shared = moe.SharedExpertIntermediateSize ?? inter;

        float[] ToF32(string name, long count)
        {
            var d = t[name]; var r = new float[count];
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, r);
            return r;
        }
        nint SharedF32(string name, long count)
        {
            var d = t[name];
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            owned.Add(dst);
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, new Span<float>((void*)dst, (int)count));
            return dst;
        }

        var gateD = t[p + "ffn_gate_exps.weight"]; var upD = t[p + "ffn_up_exps.weight"]; var downD = t[p + "ffn_down_exps.weight"];
        var sg = t[p + "ffn_gate_shexp.weight"]; var su = t[p + "ffn_up_shexp.weight"]; var sd = t[p + "ffn_down_shexp.weight"];
        var layer = new MoeLayerWeights(
            gate: ToF32(p + "ffn_gate_inp.weight", (long)ne * hidden),
            w1: new nint[ne], w2: new nint[ne], w3: new nint[ne],
            numExperts: ne, numExpertsPerTok: moe.NumExpertsPerTok, hiddenSize: hidden, intermediateSize: inter,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: [SharedF32(p + "ffn_gate_shexp.weight", (long)shared * hidden)],
            sharedUpProj: [SharedF32(p + "ffn_up_shexp.weight", (long)shared * hidden)],
            sharedDownProj: [SharedF32(p + "ffn_down_shexp.weight", (long)hidden * shared)],
            sharedIntermediateSize: shared,
            sharedExpertGate: ToF32(p + "ffn_gate_inp_shexp.weight", hidden),
            gateExpsRaw: gguf.TensorDataPointer(gateD), gateExpsRawQt: gateD.QuantizationType, gateExpsMDim: inter, gateExpsKDim: hidden,
            upExpsRaw: gguf.TensorDataPointer(upD), upExpsRawQt: upD.QuantizationType, upExpsMDim: inter, upExpsKDim: hidden,
            downExpsRaw: gguf.TensorDataPointer(downD), downExpsRawQt: downD.QuantizationType, downExpsMDim: hidden, downExpsKDim: inter,
            sharedGateRaw: [gguf.TensorDataPointer(sg)], sharedGateRawQt: sg.QuantizationType,
            sharedUpRaw: [gguf.TensorDataPointer(su)], sharedUpRawQt: su.QuantizationType,
            sharedDownRaw: [gguf.TensorDataPointer(sd)], sharedDownRawQt: sd.QuantizationType);
        return layer;
    }

    // ───────────────────────────── projections ─────────────────────────────

    private static Qwen4ExpProjection Projection(MatRef w, ComputeThreadPool? pool, Qwen4ExpLoraContext? lora = null, int layer = -1, string? loraName = null)
    {
        if (lora is null || loraName is null)
            return (input, output, tokens) => GemmSpan(w, input, output, tokens, pool);
        // LoRA site (#845): the base GEMM, then y += scale * (x B) A when the current forward carries an adapter that targets this site.
        return (input, output, tokens) =>
        {
            GemmSpan(w, input, output, tokens, pool);
            lora.Apply(layer, loraName, input, output, tokens, w.In, w.Out);
        };
    }

    private static void GemmSpan(MatRef w, ReadOnlySpan<float> x, Span<float> y, int n, ComputeThreadPool? pool)
    {
        if (x.Length < (long)n * w.In) throw new ArgumentException("input too small.", nameof(x));
        if (y.Length < (long)n * w.Out) throw new ArgumentException("output too small.", nameof(y));
        fixed (float* xp = x)
        fixed (float* yp = y)
            Gemm(w.Ptr, w.Qt, xp, yp, w.Out, w.In, n, pool);
    }

    private static void Gemm(nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.Q8_0: MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q5_0: MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q2_K: MatMul.GemmQ2_K((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q3_K: MatMul.GemmQ3_K((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q4_K: MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q5_K: MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q6_K: MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.IQ4_XS: MatMul.GemmIQ4_XS((byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.IQ2_XXS:
            case QuantizationType.IQ2_XS:
            case QuantizationType.IQ2_S:
            case QuantizationType.IQ3_XXS:
            case QuantizationType.IQ3_S:
            case QuantizationType.IQ1_S:
                MatMul.GemmIQCodebook(qt, (byte*)weights, b, c, m, k, n, pool, null); return;
            case QuantizationType.F32: MatMul.GemmF32((float*)weights, b, c, m, k, n, pool); return;
            case QuantizationType.F16: MatMul.GemmF16(weights, b, c, m, k, n, pool); return;
            case QuantizationType.Q4_0:
            case QuantizationType.Q4_1:
            case QuantizationType.Q5_1:
            case QuantizationType.IQ4_NL:
                MatMul.GemmLegacyQuantOrDequant((byte*)weights, qt, b, c, m, k, n, pool, null); return;
            default:
                MatMul.GemmDequantRows((byte*)weights, qt, b, c, m, k, n, pool: null); return;
        }
    }

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, null, lastTokenLogitsOnly: false, snapRows: 0);

    /// <inheritdoc/>
    /// <remarks>Runs on the model-owned default state; a non-null <paramref name="kvCache"/> carries the QSA K/V rows.</remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, snapRows: 0);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly, snapRows: 0);

    /// <inheritdoc/>
    /// <remarks>
    /// LoRA (#845): the adapter's delta <c>y += (alpha / rank) (x B) A</c> is added after the base GEMM of every adapted projection - QSA
    /// q/k/v/o, Gated-DeltaNet in_proj_qkv/z/a/b + out_proj, routed-expert gate/up/down - and never merged into the weights, so adapters switch
    /// per call. An adapter targeting anything else, a name on the wrong layer kind or a mis-shaped factor is rejected before any compute
    /// (<see cref="NotSupportedException"/> / <see cref="ArgumentException"/>), see <c>Qwen4ExpLoraContext.Validate</c>. Model-owned state.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, snapRows: 0, adapter: adapter);

    /// <summary>The caller-owned-state forward with a LoRA adapter (see the remarks of the adapter overload above).</summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           Qwen4ExpSequenceState state, IKvCache? kvCache, bool lastTokenLogitsOnly, ILoraAdapter? adapter)
        => ForwardCore(tokenIds, positions, deviceId, state, kvCache, lastTokenLogitsOnly, snapRows: 0, adapter: adapter);

    private ILoraAdapter? _validatedAdapter;
    private readonly Qwen4ExpLoraContext _lora;

    private void ValidateAdapter(ILoraAdapter adapter)
    {
        if (ReferenceEquals(adapter, _validatedAdapter)) return;
        Qwen4ExpLoraContext.Validate(adapter, Config);
        _validatedAdapter = adapter;
    }

    /// <summary>
    /// Forward over a caller-owned sequence state with its QSA K/V rows in <paramref name="kvCache"/> (the engine path). Positions
    /// must continue <paramref name="state"/> (<c>positions[i] == state.Length + i</c>) and the cache must already hold at least
    /// <c>state.Length</c> rows (it may hold more after a speculative rollback: stale rows are overwritten).
    /// </summary>
    /// <param name="tokenIds">Input tokens.</param>
    /// <param name="positions">Absolute positions.</param>
    /// <param name="deviceId">Device id for the returned tensor.</param>
    /// <param name="state">Sequence state; advanced by <c>tokenIds.Length</c>.</param>
    /// <param name="kvCache">Engine KV cache carrying the QSA K/V rows (slot = QSA ordinal); null keeps them in <paramref name="state"/>.</param>
    /// <param name="lastTokenLogitsOnly">Return only the last row's logits (<c>[1, vocab]</c>); otherwise every row.</param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           Qwen4ExpSequenceState state, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, state, kvCache, lastTokenLogitsOnly, snapRows: 0);

    /// <summary>
    /// Forward over a caller-owned sequence state (chunked prefill / decode) with the QSA K/V rows kept in the state itself.
    /// Positions must continue <paramref name="state"/> (<c>positions[i] == state.Length + i</c>); the model-owned default state
    /// restarts when the call begins at position 0.
    /// </summary>
    /// <param name="tokenIds">Input tokens.</param>
    /// <param name="positions">Absolute positions.</param>
    /// <param name="deviceId">Device id for the returned tensor.</param>
    /// <param name="state">Sequence state; advanced by <c>tokenIds.Length</c>.</param>
    /// <param name="lastTokenLogitsOnly">Return only the last row's logits (<c>[1, vocab]</c>); otherwise every row.</param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           Qwen4ExpSequenceState state, bool lastTokenLogitsOnly = false)
        => ForwardCore(tokenIds, positions, deviceId, state, null, lastTokenLogitsOnly, snapRows: 0);

    /// <summary>
    /// Token id that marks a position whose input embedding the caller supplies instead of the token-embedding row (an image patch from a
    /// vision tower). The supplied rows go in <c>externalEmbeddings</c>, in position order, <c>hidden</c> floats each.
    /// </summary>
    /// <remarks>
    /// The PLE branch hashes TOKEN IDS, which an embedding row does not have: HF passes the original <c>input_ids</c> (the image
    /// placeholder, <c>image_token_id</c>) as <c>ple_input_ids</c>, and llama.cpp substitutes the GGUF <c>ple.image_token_id</c>
    /// stand-in (falling back to the EOS id for files written before the key existed). This model does the latter for sentinel positions,
    /// so a file whose <c>ple.image_token_id</c> equals the placeholder id reproduces HF exactly.
    /// </remarks>
    public const int ExternalEmbeddingToken = -1;

    /// <summary>
    /// Forward over a caller-owned sequence state where some positions carry externally supplied embeddings (image tokens): every
    /// <paramref name="tokenIds"/> entry equal to <see cref="ExternalEmbeddingToken"/> takes its input embedding from the next
    /// <c>hidden</c> floats of <paramref name="externalEmbeddings"/>, and is seen by the PLE hash as the image stand-in id.
    /// </summary>
    /// <param name="tokenIds">Token ids; <see cref="ExternalEmbeddingToken"/> marks an external-embedding position.</param>
    /// <param name="positions">Absolute positions (continuing <paramref name="state"/>).</param>
    /// <param name="deviceId">Device id for the returned tensor.</param>
    /// <param name="state">Sequence state; advanced by <c>tokenIds.Length</c>.</param>
    /// <param name="kvCache">Engine KV cache for the QSA rows, or null.</param>
    /// <param name="lastTokenLogitsOnly">Return only the last row's logits.</param>
    /// <param name="externalEmbeddings">One <c>hidden</c>-wide row per sentinel position, in order.</param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           Qwen4ExpSequenceState state, IKvCache? kvCache, bool lastTokenLogitsOnly, ReadOnlySpan<float> externalEmbeddings)
        => ForwardCore(tokenIds, positions, deviceId, state, kvCache, lastTokenLogitsOnly, snapRows: 0, externalEmbeddings);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                Qwen4ExpSequenceState state, IKvCache? kvCache, bool lastTokenLogitsOnly, int snapRows,
                                ReadOnlySpan<float> externalEmbeddings = default, ILoraAdapter? adapter = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (ReferenceEquals(state, _defaultState) && positions[0] == 0 && state.Length != 0) state.Reset();
        for (int i = 0; i < T; i++)
            if (positions[i] != state.Length + i)
                throw new ArgumentException($"qwen4exp keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {state.Length + i}.");
        if (state.Length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        if (kvCache is not null) ValidateKvCache(kvCache, state.Length);
        if (snapRows > 0 && !ReferenceEquals(state, _defaultState))
            throw new InvalidOperationException("Row snapshots are only recorded on the model-owned state.");
        int external = 0;
        for (int i = 0; i < T; i++) if (tokenIds[i] == ExternalEmbeddingToken) external++;
        if ((long)external * _hidden != externalEmbeddings.Length)
            throw new ArgumentException($"{external} external-embedding position(s) need {(long)external * _hidden} floats, got {externalEmbeddings.Length}.", nameof(externalEmbeddings));
        _snapValid = false;   // any forward moves the live state on: earlier row snapshots no longer describe it
        foreach (var p in state.Ple) p?.InvalidateRows();
        foreach (var q in state.Qsa) q?.Indexer.InvalidateRows();
        if (adapter is not null) ValidateAdapter(adapter);
        if (snapRows > 0) BeginRowRecording(state, snapRows, T);
        _lora.Adapter = adapter;   // consulted by the projections for the duration of this call only (not reentrant, like TransformerModel)
        try
        {
            return ForwardBody(tokenIds, deviceId, state, kvCache, lastTokenLogitsOnly, snapRows, externalEmbeddings);
        }
        finally
        {
            _lora.Adapter = null;
            if (snapRows > 0) EndRowRecording(state);
        }
    }

    /// <summary>Rejects an engine KV cache whose geometry cannot carry the QSA layers (clear message instead of a silent misread).</summary>
    private void ValidateKvCache(IKvCache kvCache, int stateLength)
    {
        if (kvCache.CurrentLength < stateLength)
            throw new InvalidOperationException(
                $"The KV cache holds {kvCache.CurrentLength} rows but the sequence state is at position {stateLength}: " +
                "they must advance together (Rollback the cache and restore the state to the same length).");
        if (kvCache is IPerLayerKvCache per)
        {
            if (per.LayerCount < _numQsa)
                throw new ArgumentException($"qwen4exp needs {_numQsa} KV layers (one per QSA layer); the cache has {per.LayerCount}.", nameof(kvCache));
            foreach (var b in _blocks)
                if (b.Qsa is { } qsa && per.KvStrideOf(b.QsaOrdinal) != qsa.KvStride)
                    throw new ArgumentException(
                        $"qwen4exp QSA layer {b.QsaOrdinal} needs KV stride {qsa.KvStride}; the cache slot has {per.KvStrideOf(b.QsaOrdinal)}.", nameof(kvCache));
        }
    }

    private ITensor ForwardBody(ReadOnlySpan<int> tokenIds, int deviceId, Qwen4ExpSequenceState state, IKvCache? kvCache,
                                bool lastTokenLogitsOnly, int snapRows, ReadOnlySpan<float> externalEmbeddings)
    {
        int T = tokenIds.Length;
        _threadPool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);
        int H = _hidden, S = _hc, row = S * H, vocab = Config.VocabSize;

        float[] emb = ArrayPool<float>.Shared.Rent(T * H);
        float[] res = ArrayPool<float>.Shared.Rent(T * row);
        float[] xn = ArrayPool<float>.Shared.Rent(T * row);
        float[] low = ArrayPool<float>.Shared.Rent(T * _lowRank);
        float[] mix = ArrayPool<float>.Shared.Rent(T * row);
        float[] h = ArrayPool<float>.Shared.Rent(T * H);
        float[] y = ArrayPool<float>.Shared.Rent(T * H);
        float[] gains = ArrayPool<float>.Shared.Rent(T * S);
        // The ids the PLE hash sees: image positions (no token id) take the GGUF stand-in, everything else is the token itself.
        int[]? pleIdBuf = null;
        try
        {
            ReadOnlySpan<int> pleIds = tokenIds;
            if (!externalEmbeddings.IsEmpty && Config.Qwen4Exp!.Ple is { } pleCfg)
            {
                pleIdBuf = ArrayPool<int>.Shared.Rent(T);
                int stand = pleCfg.ImageTokenId ?? pleCfg.EosTokenId;
                for (int t = 0; t < T; t++) pleIdBuf[t] = tokenIds[t] == ExternalEmbeddingToken ? stand : tokenIds[t];
                pleIds = pleIdBuf.AsSpan(0, T);
            }
            EmbedTokens(tokenIds, emb, externalEmbeddings);
            Qwen4ExpGatedResidual.Broadcast(emb, S, H, res, T);
            if (Trace is { } tr0) tr0("embed", emb.AsSpan(0, T * H), T, H);

            for (int il = 0; il < _blocks.Length; il++)
            {
                var blk = _blocks[il];
                if (blk.Ple is { } ple)
                    ple.Apply(pleIds, state.Ple[il]!, res.AsSpan(0, T * row));

                // ── token mixer ──
                GrRead(blk.AttnGr, res, xn, low, mix, h, gains, T, wantInject: true);
                if (blk.Gdn is { } g)
                    ForwardGdn(g, blk.GdnOrdinal, il, h, y, T, state.Gdn, snapRows);
                else
                    blk.Qsa!.Forward(h.AsSpan(0, T * H), T, state.Qsa[il]!, y.AsSpan(0, T * H), kvCache, blk.QsaOrdinal);
                Qwen4ExpGatedResidual.Write(res.AsSpan(0, T * row), y, gains, S, H, T);

                // ── MoE ──
                GrRead(blk.FfnGr, res, xn, low, mix, h, gains, T, wantInject: true);
                ForwardMoe(blk.Moe, il, h, y, T);
                Qwen4ExpGatedResidual.Write(res.AsSpan(0, T * row), y, gains, S, H, T);

                if (Trace is { } tr) tr($"blk.{il}.l_out", res.AsSpan(0, T * row), T, row);
                if (TensorDump.Enabled) DumpResidual(il, res, T, row);
            }

            // ── head mixer (replaces the final norm) + LM head ──
            GrRead(_head, res, xn, low, mix, h, gains, T, wantInject: false);
            if (Trace is { } trh) trh("hidden_final", h.AsSpan(0, T * H), T, H);

            int rows = lastTokenLogitsOnly ? 1 : T;
            int firstRow = T - rows;
            var result = UnmanagedTensor.Allocate(new TensorShape(rows, vocab), DType.Float32, deviceId);
            var logits = new Span<float>((void*)result.DataPointer, rows * vocab);
            GemmSpan(_output, h.AsSpan(firstRow * H, rows * H), logits, rows, _threadPool);
            state.Length += T;
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(emb); ArrayPool<float>.Shared.Return(res); ArrayPool<float>.Shared.Return(xn);
            ArrayPool<float>.Shared.Return(low); ArrayPool<float>.Shared.Return(mix); ArrayPool<float>.Shared.Return(h);
            ArrayPool<float>.Shared.Return(y); ArrayPool<float>.Shared.Return(gains);
            if (pleIdBuf is not null) ArrayPool<int>.Shared.Return(pleIdBuf);
        }
    }

    private static void DumpResidual(int il, float[] res, int T, int row)
    {
        fixed (float* p = res) TensorDump.Dump2D($"blk.{il}.l_out", p, T, row);
    }

    /// <summary>GR read: group-RMS → low-rank mix → block input <paramref name="h"/>; and the write gains when requested.</summary>
    private void GrRead(GrWeights gr, float[] res, float[] xn, float[] low, float[] mix, float[] h, float[] gains, int T, bool wantInject)
    {
        int H = _hidden, S = _hc, row = S * H;
        Qwen4ExpGatedResidual.GroupRmsNorm(res, gr.Norm, S, H, _eps, xn, T);
        GemmSpan(gr.Down, xn, low, T, _threadPool);
        Qwen4ExpGatedResidual.ActivateLowRank(low.AsSpan(0, T * _lowRank), S);
        GemmSpan(gr.Up, low, mix, T, _threadPool);
        Qwen4ExpGatedResidual.MixAndMean(mix.AsSpan(0, T * row), xn, S, H, h, T);
        if (wantInject && gr.Inject is { } inj)
        {
            GemmSpan(inj, xn, gains, T, _threadPool);
            Qwen4ExpGatedResidual.InjectGains(gains.AsSpan(0, T * S), S);
        }
    }

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float[] dest, ReadOnlySpan<float> externalEmbeddings = default)
    {
        int H = _hidden, nextExternal = 0;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if (id == ExternalEmbeddingToken)
            {
                externalEmbeddings.Slice(nextExternal++ * H, H).CopyTo(dest.AsSpan(t * H, H));
                continue;
            }
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {Config.VocabSize}).");
            long rowBytes = Dequantize.RowByteSize(H, _tokenEmbedQt);
            Dequantize.ToFloat32(_tokenEmbed + (nint)(id * rowBytes), H, _tokenEmbedQt, dest.AsSpan(t * H, H));
        }
    }

    /// <summary>
    /// Gated DeltaNet token mixer: the Qwen3.5 GDN with a SIGMOID output gate (<c>norm(core) * sigmoid(z)</c>) - llama.cpp
    /// <c>qwen4exp.cpp build_norm_gated</c> ("the one numerical difference from Qwen3.5's GDN") and HF
    /// <c>output_gate_type="sigmoid"</c>. The body is the shared <see cref="GdnTokenMixer"/>; this wrapper only rents the scratch.
    /// </summary>
    private void ForwardGdn(GdnTokenMixingWeights w, int ordinal, int absoluteLayer, float[] x, float[] y, int T, GdnStateCache cache, int snapRows)
    {
        int nV = _gdn.NVHead, nK = _gdn.NKHead, dS = _gdn.DState, dC = _gdn.DConv;
        int convDim = (2 * nK + nV) * dS, vDim = nV * dS, kDim = nK * dS;
        float[] qkv = ArrayPool<float>.Shared.Rent(T * convDim);
        float[] z = ArrayPool<float>.Shared.Rent(T * vDim);
        float[] alpha = ArrayPool<float>.Shared.Rent(T * nV);
        float[] beta = ArrayPool<float>.Shared.Rent(T * nV);
        float[] convIn = ArrayPool<float>.Shared.Rent((dC - 1 + T) * convDim);
        float[] q = ArrayPool<float>.Shared.Rent(T * kDim);
        float[] k = ArrayPool<float>.Shared.Rent(T * kDim);
        float[] v = ArrayPool<float>.Shared.Rent(T * vDim);
        float[] core = ArrayPool<float>.Shared.Rent(T * vDim);
        try
        {
            int rows = snapRows > 0 ? Math.Min(snapRows, T - 1) : 0;
            var rec = rows > 0
                ? new GdnRowRecording(rows, RowSnapshotConvLayer(ordinal), RowRecKLayer(ordinal), RowRecGLayer(ordinal), RowRecDLayer(ordinal))
                : default;
            _gdnLoraLayer = absoluteLayer;
            GdnTokenMixer.Forward(w, _gdn, absoluteLayer, ordinal, T, _hidden, _eps, GdnOutputGate.Sigmoid,
                                  x.AsSpan(0, T * _hidden), y.AsSpan(0, T * _hidden), _gdnGemm ??= GdnGemmAdapter,
                                  new GdnMixerScratch(qkv, z, alpha, beta, convIn, q, k, v, core), cache, rec);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(qkv); ArrayPool<float>.Shared.Return(z); ArrayPool<float>.Shared.Return(alpha);
            ArrayPool<float>.Shared.Return(beta); ArrayPool<float>.Shared.Return(convIn); ArrayPool<float>.Shared.Return(q);
            ArrayPool<float>.Shared.Return(k); ArrayPool<float>.Shared.Return(v); ArrayPool<float>.Shared.Return(core);
        }
    }

    private GdnGemm? _gdnGemm;

    private void GdnGemmAdapter(GdnProjection projection, nint weight, QuantizationType qt, ReadOnlySpan<float> input, Span<float> output, int outDim, int inDim, int seqLen)
    {
        if (input.Length < (long)seqLen * inDim) throw new ArgumentException("input too small.", nameof(input));
        if (output.Length < (long)seqLen * outDim) throw new ArgumentException("output too small.", nameof(output));
        fixed (float* xp = input)
        fixed (float* yp = output)
            Gemm(weight, qt, xp, yp, outDim, inDim, seqLen, _threadPool);
        if (_lora.Adapter is not null)
            _lora.Apply(_gdnLoraLayer, GdnLoraName(projection), input, output, seqLen, inDim, outDim);
    }

    private int _gdnLoraLayer;

    private static string GdnLoraName(GdnProjection p) => p switch
    {
        GdnProjection.Qkv => "in_proj_qkv", GdnProjection.Gate => "in_proj_z", GdnProjection.Alpha => "in_proj_a",
        GdnProjection.Beta => "in_proj_b", _ => "out_proj",
    };

    /// <summary>512-expert softmax MoE: router softmax over ALL experts → top-k → renormalise; routed experts + shared expert × sigmoid(gate).</summary>
    private void ForwardMoe(MoeLayerWeights moe, int layer, float[] x, float[] y, int T)
    {
        int ne = moe.NumExperts, k = moe.NumExpertsPerTok, H = _hidden, inter = moe.IntermediateSize, total = T * k;
        int[] assignExpert = ArrayPool<int>.Shared.Rent(total);
        float[] assignWeight = ArrayPool<float>.Shared.Rent(total);
        int[] cursors = ArrayPool<int>.Shared.Rent(ne + 1);
        int[] bucketTokens = ArrayPool<int>.Shared.Rent(total);
        int[] bucketSlots = ArrayPool<int>.Shared.Rent(total);
        int[] unique = ArrayPool<int>.Shared.Rent(Math.Min(ne, total) + 1);
        try
        {
            int uniqueCount = MoeSwiGluMlp.Route(
                x.AsSpan(0, T * H), moe.Gate, assignExpert, assignWeight, cursors, bucketTokens, bucketSlots, unique,
                ne, k, H, T, moe.NormTopKProb);

            long gateRow = Dequantize.RowByteSize((long)inter * H, moe.GateExpsRawQt);
            long upRow = Dequantize.RowByteSize((long)inter * H, moe.UpExpsRawQt);
            long downRow = Dequantize.RowByteSize((long)H * inter, moe.DownExpsRawQt);
            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                x.AsSpan(0, T * H),
                moe.GateExpsRaw, moe.GateExpsRawQt, gateRow, ReadOnlySpan<nint>.Empty,
                moe.UpExpsRaw, moe.UpExpsRawQt, upRow, ReadOnlySpan<nint>.Empty,
                moe.DownExpsRaw, moe.DownExpsRawQt, downRow, ReadOnlySpan<nint>.Empty,
                assignExpert, assignWeight, cursors, bucketTokens, bucketSlots, unique, uniqueCount,
                y.AsSpan(0, T * H),
                ne, k, H, inter, T,
                moe.SharedGateProj, moe.SharedUpProj, moe.SharedDownProj, moe.SharedIntermediateSize,
                moe.SharedExpertGate is null ? ReadOnlySpan<float>.Empty : moe.SharedExpertGate,
                loraAdapter: _lora.Adapter, loraLayer: layer, threadPool: _threadPool);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(assignExpert); ArrayPool<float>.Shared.Return(assignWeight);
            ArrayPool<int>.Shared.Return(cursors); ArrayPool<int>.Shared.Return(bucketTokens);
            ArrayPool<int>.Shared.Return(bucketSlots); ArrayPool<int>.Shared.Return(unique);
        }
    }

    // ───────────────────────────── engine contracts ─────────────────────────────

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => CreateState();

    /// <summary>
    /// Per-sequence loop over the forward: each request's <see cref="SequenceForwardRequest.GdnState"/> (a
    /// <see cref="Qwen4ExpSequenceState"/>) is threaded through, with the QSA K/V rows in the request's KV cache. Returns the LAST
    /// row's logits (<c>[1, vocab]</c>) per request — the scheduler samples row <c>Shape[0] - 1</c> only. GDN, PLE and the
    /// indexer are per-token recurrent and per-sequence, so fusing across sequences buys nothing on the CPU oracle; interleaved
    /// sequences therefore equal separate runs exactly.
    /// </summary>
    /// <exception cref="ArgumentException">A multi-sequence batch with a request lacking its own state (it would share the
    /// model-owned default state), or a state of another model type.</exception>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].GdnState is null && requests.Count >= 2)
                throw new ArgumentException(
                    $"Multi-seq ForwardBatch requires each SequenceForwardRequest to carry its own GdnState (request {i} has none): " +
                    "the model-owned default state would be shared across sequences. Allocate one with CreateSequenceState().",
                    nameof(requests));
            if (requests[i].GdnState is not null and not Qwen4ExpSequenceState)
                throw new ArgumentException(
                    $"Request {i} carries a {requests[i].GdnState!.GetType().Name}; qwen4exp needs a Qwen4ExpSequenceState.", nameof(requests));
        }

        var results = new List<ITensor>(requests.Count);
        try
        {
            foreach (var r in requests)
            {
                var st = (Qwen4ExpSequenceState?)r.GdnState ?? _defaultState;
                results.Add(ForwardCore(r.TokenIds.Span, r.Positions.Span, deviceId, st, r.KvCache, lastTokenLogitsOnly: true, snapRows: 0, adapter: r.Adapter));
            }
        }
        catch
        {
            foreach (var t in results) t.Dispose();
            throw;
        }
        return results;
    }

    // ── checkpoint / rollback of the model-owned state ──

    private Qwen4ExpSequenceState? _spareCheckpoint;

    /// <summary>A pooled checkpoint shell; disposing returns its buffers to the owning model.</summary>
    private sealed class StateCheckpoint(Qwen4ExpTransformerModel owner, Qwen4ExpSequenceState shell) : IDisposable
    {
        private Qwen4ExpSequenceState? _shell = shell;
        public Qwen4ExpSequenceState Shell => _shell ?? throw new ObjectDisposedException(nameof(StateCheckpoint));
        public Qwen4ExpTransformerModel Owner => owner;

        public void Dispose()
        {
            var s = Interlocked.Exchange(ref _shell, null);
            if (s is null) return;
            owner._defaultState.ReleaseSource(s);   // the live state may still lazily read its GDN from this shell
            if (Interlocked.CompareExchange(ref owner._spareCheckpoint, s, null) is not null) s.Dispose();
        }
    }

    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    /// <remarks>
    /// Logically a full, independent copy (GDN ~113 MiB at the released size, the PLE window + conv history, the QSA indexers'
    /// pooled keys + tails and, when the state keeps its K/V rows itself, those too), so it stays a valid prefix snapshot after
    /// the live state has moved to a different history (#840: physically incremental). The GDN buffers are exchanged with the
    /// pooled shell instead of copied and the live state defers its content from the checkpoint until the next forward, whose
    /// first step reads it (fused copy); pooled keys / own K/V rows are copied only where their content stamps differ from what
    /// the shell last synced. K/V rows in an engine KV cache are position-addressed: the caller rolls that cache back.
    /// </remarks>
    public object? CheckpointRecurrentState()
    {
        var shell = Interlocked.Exchange(ref _spareCheckpoint, null) ?? CreateState();
        _defaultState.CaptureInto(shell);
        return new StateCheckpoint(this, shell);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        if (checkpoint is null) return;
        if (checkpoint is not StateCheckpoint cp || !ReferenceEquals(cp.Owner, this))
            throw new ArgumentException(
                $"{GetType().Name}.RestoreRecurrentState expects a checkpoint from this model's CheckpointRecurrentState; got {checkpoint.GetType().Name}.",
                nameof(checkpoint));
        _defaultState.RestoreFrom(cp.Shell);
        _snapValid = false;
        foreach (var p in _defaultState.Ple) p?.InvalidateRows();
        foreach (var q in _defaultState.Qsa) q?.Indexer.InvalidateRows();
    }

    // ── per-row recurrent snapshots (speculative verify without replay) ──

    // Row snapshots (#842): NOT a full GDN state per row. The state before the chunk is kept once (_rowBase, obtained by exchanging
    // buffers with the live state: no copy), and per row and GDN layer only what rebuilds it by replay: the L2-normalised keys, the decays
    // and the scan's deltas (nV * (1 + 2 * dS) floats vs nV * dS^2 for a full state: ~64x less at the released size).
    private GdnStateCache? _rowBase;
    private readonly Qwen4ExpNativeBuffer _rowRecK = new(), _rowRecG = new(), _rowRecD = new(), _rowSnapConv = new();
    private int _snapStrideRows;      // rows per layer region in the scratch (this recording)
    private int _snapRowsRecorded;    // rows 0 .. _snapRowsRecorded-1 recorded (state after the last row is the live state)
    private int _snapBase;            // state.Length before the recorded chunk
    private bool _snapValid;

    /// <summary>The GDN part of <see cref="RecurrentRowSnapshotBytes"/>: the pre-chunk state, the per-row keys / decays / deltas and the conv windows.</summary>
    public long RecurrentRowSnapshotGdnBytes
        => (_rowBase?.AllocatedBytes ?? 0) + _rowRecK.Bytes + _rowRecG.Bytes + _rowRecD.Bytes + _rowSnapConv.Bytes;

    /// <summary>Bytes currently held by the per-row recurrent snapshot scratch (all components).</summary>
    public long RecurrentRowSnapshotBytes
    {
        get
        {
            long b = RecurrentRowSnapshotGdnBytes;
            foreach (var p in _defaultState.Ple) b += p is null ? 0 : p.Bytes - p.StateBytes;
            foreach (var q in _defaultState.Qsa) b += q?.Indexer.SnapshotScratchBytes ?? 0;
            return b;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Scratch is ONE pre-chunk GDN state (~113 MiB at the released size, kept for the model's lifetime, captured without a copy) plus
    /// about 1.8 MiB per row (keys, decays, deltas and the conv window of the 36 GDN layers), instead of ~113 MiB per row. A restore
    /// to row r replays r + 1 rank-1 updates from the pre-chunk state; the result is bit-identical to the state the scan itself leaves
    /// after that row (and to the full-state snapshots this replaced).
    /// </remarks>
    public bool SupportsRecurrentRowSnapshots => true;

    /// <inheritdoc/>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                                 IKvCache? kvCache, IMtpState? mtpState)
    {
        if (mtpState is not null)
            throw new NotSupportedException("qwen4exp has no MTP head on the CPU oracle yet (mtpState must be null).");
        int rows = Math.Max(tokenIds.Length - 1, 0);
        var logits = ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, snapRows: rows);
        _snapBase = positions[0];
        _snapRowsRecorded = rows;
        _snapValid = true;
        return logits;
    }

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if (!_snapValid || (uint)row > (uint)_snapRowsRecorded)
            throw new InvalidOperationException(
                $"No recurrent snapshot for row {row}: the last ForwardWithRecurrentSnapshots recorded {(_snapValid ? _snapRowsRecorded : 0)} " +
                "row(s), or a later forward invalidated them.");
        if (row == _snapRowsRecorded) return;   // the live state IS the state after the last row

        var st = _defaultState;
        for (int l = 0; l < _numGdn; l++)
        {
            st.Gdn.DiscardPending(l);   // fully overwritten below: no point materialising a lazy copy first
            GatedDeltaNetScan.Replay(st.Gdn.GetGdnStateForUpdate(l), _rowBase!.GetGdnState(l), RowRecKLayer(l), RowRecGLayer(l), RowRecDLayer(l),
                                     _gdn.NVHead, _gdn.NKHead, _gdn.DState, row + 1);
            RowSnapshotConv(l, row).CopyTo(st.Gdn.GetConvStateForUpdate(l));
        }
        foreach (var p in st.Ple) p?.RestoreRow(row);
        foreach (var q in st.Qsa) q?.Indexer.RestoreToRow(row);
        st.Length = _snapBase + row + 1;
        _snapValid = false;
    }

    private void BeginRowRecording(Qwen4ExpSequenceState state, int rows, int chunkLength)
    {
        int recorded = Math.Min(rows, chunkLength - 1);
        _snapStrideRows = Math.Max(recorded, 1);
        if (_numGdn > 0)
        {
            int kDim = _gdn.NKHead * _gdn.DState;
            _rowRecK.EnsureExact((long)_numGdn * _snapStrideRows * kDim);
            _rowRecG.EnsureExact((long)_numGdn * _snapStrideRows * _gdn.NVHead);
            _rowRecD.EnsureExact((long)_numGdn * _snapStrideRows * _gdn.NVHead * _gdn.DState);
            _rowSnapConv.EnsureExact((long)_numGdn * _snapStrideRows * state.Gdn.ConvStateElements);
            if (recorded > 0)
            {
                // Keep the pre-chunk state without copying it: the base takes the live buffers, the live state lazily reads the base in its
                // first scan step (fused). After the forward the base holds exactly the state before the chunk.
                _rowBase ??= new GdnStateCache(_gdn, _numGdn);
                state.Gdn.MaterializePending();
                _rowBase.SwapBuffersWith(state.Gdn);
                state.Gdn.DeferCopyFrom(_rowBase);
            }
        }
        foreach (var p in state.Ple) if (p is not null) p.RecordRowCount = recorded;
        foreach (var q in state.Qsa) if (q is not null) q.Indexer.RecordRowCount = recorded;
    }

    private void EndRowRecording(Qwen4ExpSequenceState state)
    {
        if (_rowBase is not null && state.Gdn.IsPendingOn(_rowBase)) state.Gdn.MaterializePending();   // the forward did not reach every layer
        foreach (var p in state.Ple) if (p is not null) p.RecordRowCount = 0;
        foreach (var q in state.Qsa) if (q is not null) q.Indexer.RecordRowCount = 0;
    }

    private Span<float> RowRecKLayer(int ordinal)
    {
        int e = _gdn.NKHead * _gdn.DState;
        return _rowRecK.Slice((long)ordinal * _snapStrideRows * e, _snapStrideRows * e);
    }

    private Span<float> RowRecGLayer(int ordinal)
        => _rowRecG.Slice((long)ordinal * _snapStrideRows * _gdn.NVHead, _snapStrideRows * _gdn.NVHead);

    private Span<float> RowRecDLayer(int ordinal)
    {
        int e = _gdn.NVHead * _gdn.DState;
        return _rowRecD.Slice((long)ordinal * _snapStrideRows * e, _snapStrideRows * e);
    }

    private Span<float> RowSnapshotConvLayer(int ordinal)
    {
        int e = _defaultState.Gdn.ConvStateElements;
        return _rowSnapConv.Slice((long)ordinal * _snapStrideRows * e, _snapStrideRows * e);
    }

    private Span<float> RowSnapshotConv(int ordinal, int row)
    {
        int e = _defaultState.Gdn.ConvStateElements;
        return _rowSnapConv.Slice(((long)ordinal * _snapStrideRows + row) * e, e);
    }

    // ── per-sequence prefix snapshots (scheduler recurrent prefix cache) ──

    /// <inheritdoc/>
    public bool SupportsSequencePrefixSnapshot => true;

    /// <summary>State clone + the QSA K/V rows of the prefix, independent of both inputs.</summary>
    private sealed class PrefixSnapshot : IDisposable
    {
        public required Qwen4ExpSequenceState State;
        public required Qwen4ExpNativeBuffer[] Keys, Values;
        public required int Length;

        public void Dispose()
        {
            State.Dispose();
            foreach (var b in Keys) b.Dispose();
            foreach (var b in Values) b.Dispose();
        }
    }

    /// <inheritdoc/>
    public IDisposable? SnapshotSequencePrefix(IKvCache kvCache, IRecurrentSequenceState? state, int prefixLen)
    {
        if (state is not Qwen4ExpSequenceState s) return null;
        ArgumentNullException.ThrowIfNull(kvCache);
        if (s.Length != prefixLen)
            throw new ArgumentException($"The state has consumed {s.Length} tokens, not the {prefixLen}-token prefix.", nameof(prefixLen));
        var keys = new Qwen4ExpNativeBuffer[_numQsa];
        var values = new Qwen4ExpNativeBuffer[_numQsa];
        var clone = CreateState();
        try
        {
            foreach (var b in _blocks)
            {
                if (b.Qsa is null) continue;
                int slot = b.QsaOrdinal, stride = b.Qsa.KvStride;
                keys[slot] = new Qwen4ExpNativeBuffer(); values[slot] = new Qwen4ExpNativeBuffer();
                keys[slot].EnsureExact(Math.Max(1L, (long)prefixLen * stride));
                values[slot].EnsureExact(Math.Max(1L, (long)prefixLen * stride));
                if (kvCache is IQuantizedKvCache qkv)
                {
                    // Quantised cache: the snapshot holds the DEQUANTISED rows (fp32), so restoring re-quantises them into the target.
                    Qwen4ExpKvRows.RequireSupported(qkv);
                    if (qkv.CurrentLength < prefixLen) throw new ArgumentException($"The KV cache holds {qkv.CurrentLength} rows; the prefix is {prefixLen}.", nameof(kvCache));
                    for (int p = 0; p < prefixLen; p++)
                        Qwen4ExpKvRows.ReadRow(qkv, slot, p, stride, keys[slot].Pointer + (long)p * stride, values[slot].Pointer + (long)p * stride);
                    continue;
                }
                var kr = kvCache.GetKeysRef(slot); var vr = kvCache.GetValuesRef(slot);
                if (kr.Dim0 < prefixLen) throw new ArgumentException($"The KV cache holds {kr.Dim0} rows; the prefix is {prefixLen}.", nameof(kvCache));
                new ReadOnlySpan<float>((void*)kr.DataPointer, prefixLen * stride).CopyTo(keys[slot].Slice(0, prefixLen * stride));
                new ReadOnlySpan<float>((void*)vr.DataPointer, prefixLen * stride).CopyTo(values[slot].Slice(0, prefixLen * stride));
            }
            clone.CopyFrom(s);
        }
        catch
        {
            clone.Dispose();
            foreach (var b in keys) b?.Dispose();
            foreach (var b in values) b?.Dispose();
            throw;
        }
        return new PrefixSnapshot { State = clone, Keys = keys, Values = values, Length = prefixLen };
    }

    /// <inheritdoc/>
    public void RestoreSequencePrefix(IDisposable snapshot, IKvCache kvCache, IRecurrentSequenceState? state)
    {
        if (snapshot is not PrefixSnapshot snap || state is not Qwen4ExpSequenceState s)
            throw new ArgumentException("Snapshot / state are not this model's types.");
        ArgumentNullException.ThrowIfNull(kvCache);
        if (kvCache.MaxLength < snap.Length) throw new ArgumentException("The KV cache is smaller than the snapshot prefix.", nameof(kvCache));
        if (kvCache.CurrentLength != 0) kvCache.Rollback(0);
        int[] pos = new int[snap.Length];
        for (int i = 0; i < pos.Length; i++) pos[i] = i;
        foreach (var b in _blocks)
        {
            if (b.Qsa is null) continue;
            int slot = b.QsaOrdinal, stride = b.Qsa.KvStride;
            // Sub-chunked for a windowed quantised cache (an update longer than its window would quantise unwritten ring slots).
            Qwen4ExpKvRows.Update(kvCache, snap.Keys[slot].Pointer, snap.Values[slot].Pointer, snap.Length, stride, pos, slot);
        }
        s.CopyFrom(snap.State);
    }

    // ── state accounting for memory planning ──

    /// <summary>
    /// Logical per-sequence state size after <paramref name="contextLength"/> tokens: the constant part (GDN ~113 MiB at the
    /// released size, PLE history, indexer tails) plus the context-proportional pooled keys and QSA K/V rows.
    /// </summary>
    /// <param name="contextLength">Tokens consumed by the sequence.</param>
    public Qwen4ExpStateBytes EstimateSequenceStateBytes(int contextLength) => Qwen4ExpStateBytes.Estimate(Config, contextLength);

    /// <summary>
    /// Like <see cref="EstimateSequenceStateBytes(int)"/> with the QSA K/V rows in a quantised engine KV cache (#841): <paramref name="keyDType"/> /
    /// <paramref name="valueDType"/> (Q8_0 or Q4_0) older than the fp32 <paramref name="windowSize"/> rows.
    /// </summary>
    public Qwen4ExpStateBytes EstimateSequenceStateBytes(int contextLength, KvCacheDType keyDType, KvCacheDType valueDType, int windowSize)
        => Qwen4ExpStateBytes.Estimate(Config, contextLength, keyDType, valueDType, windowSize);

    /// <summary>
    /// Size of a checkpoint taken at <paramref name="contextLength"/> tokens (what <see cref="CheckpointRecurrentState"/> copies,
    /// assuming the QSA K/V rows live in an engine KV cache): everything except <see cref="Qwen4ExpStateBytes.Kv"/>.
    /// </summary>
    /// <param name="contextLength">Tokens consumed.</param>
    public long CheckpointBytes(int contextLength) { var e = EstimateSequenceStateBytes(contextLength); return e.Total - e.Kv; }

    /// <summary>Resident bytes of the model-owned default state, by owner.</summary>
    public Qwen4ExpStateBytes DefaultStateBytes => _defaultState.ResidentBytes;

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsPool) _threadPool?.Dispose();
        _defaultState.Dispose();
        _spareCheckpoint?.Dispose();
        _rowBase?.Dispose(); _rowRecK.Dispose(); _rowRecG.Dispose(); _rowRecD.Dispose(); _rowSnapConv.Dispose();
        foreach (nint p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        GC.SuppressFinalize(this);
    }
}
