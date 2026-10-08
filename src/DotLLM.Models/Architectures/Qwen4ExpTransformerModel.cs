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
/// Per-sequence state of a <see cref="Qwen4ExpTransformerModel"/>: the Gated-DeltaNet recurrent state of every GDN layer, the
/// PLE hash window + dilated-conv history, and each QSA layer's K/V + pooled-indexer-key cache. Fully self-contained (the
/// model does not use the engine <see cref="IKvCache"/>): chunked prefill and token-by-token decode thread one instance.
/// </summary>
public sealed class Qwen4ExpSequenceState : IRecurrentSequenceState
{
    internal GdnStateCache Gdn { get; }
    internal Qwen4ExpPleState?[] Ple { get; }
    internal Qwen4ExpQsaState?[] Qsa { get; }

    /// <summary>Tokens consumed so far (the next position).</summary>
    public int Length { get; internal set; }

    internal Qwen4ExpSequenceState(GdnStateCache gdn, Qwen4ExpPleState?[] ple, Qwen4ExpQsaState?[] qsa)
    {
        Gdn = gdn; Ple = ple; Qsa = qsa;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        Gdn.Reset();
        foreach (var p in Ple) p?.Reset();
        foreach (var q in Qsa) q?.Reset();
        Length = 0;
    }

    /// <summary>Resident bytes of the QSA K/V + indexer caches and PLE state (the part that grows with context).</summary>
    public long Bytes => Gdn.AllocatedBytes + Ple.Sum(p => p?.Bytes ?? 0) + Qsa.Sum(q => q?.Bytes ?? 0);

    /// <inheritdoc/>
    public void Dispose() => Gdn.Dispose();
}

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
    private readonly int _hidden, _hc, _lowRank, _numGdn;
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
                                     List<nint> owned)
    {
        Config = config; _gguf = gguf; _blocks = blocks; _head = head; _tokenEmbed = tokenEmbed; _tokenEmbedQt = tokenEmbedQt;
        _output = output; _threadPool = pool; _ownsPool = ownsPool; _owned = owned;
        _gdn = config.GdnConfig!.Value;
        _q4 = config.Qwen4Exp!;
        _hidden = config.HiddenSize;
        _hc = _q4.HyperConnectionCount;
        _lowRank = _q4.HyperConnectionLowRank;
        _eps = config.NormEpsilon;
        _numGdn = blocks.Count(b => b.Gdn is not null);
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>Allocates a fresh sequence state.</summary>
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
        if (q4.Ple is { } ple0 && ple0.Layers.Count > 1)
            throw new NotSupportedException("qwen4exp files carry one set of PLE hash constants; several PLE layers cannot be represented.");

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
                    Projection(Mat(b + "attn_q.weight"), pool), Projection(Mat(b + "attn_k.weight"), pool),
                    Projection(Mat(b + "attn_v.weight"), pool), Projection(Mat(b + "attn_output.weight"), pool),
                    Projection(Mat(b + "indexer.q_proj.weight"), pool), Projection(Mat(b + "indexer.k_proj.weight"), pool));
                block.QsaOrdinal = qsaOrd++;
            }

            if (q4.Ple is { } ple && ple.Layers.Contains(il))
            {
                var tdesc = tensors[Qwen4ExpTensors.PerLayerTokenEmbd];
                int rowDim = ple.RowDim;
                var convRaw = F32(b + "ple_conv1d.weight", (long)ple.ConvKernel * hcDim);   // [C][K] (K fastest)
                block.Ple = new Qwen4ExpPleBranch(
                    gguf.TensorDataPointer(tdesc), tdesc.QuantizationType, tdesc.Shape[1], rowDim,
                    ple.NgramSize, ple.HeadsPerNgram, ple.EosTokenId, ple.ConvKernel,
                    ple.LayerMultipliers.Select(v => unchecked((long)v)).ToArray(),
                    ple.HeadOffsets.Select(v => checked((long)v)).ToArray(),
                    ple.HeadVocabSizes.Select(v => checked((long)v)).ToArray(),
                    hc, hidden, config.NormEpsilon,
                    F32(b + "ple_norm_key.weight", hcDim), F32(b + "ple_norm_query.weight", hcDim), F32(b + "ple_norm_conv.weight", hcDim),
                    Qwen4ExpPle.TransposeConvWeight(convRaw, hcDim, ple.ConvKernel),
                    Projection(Mat(b + "ple_key.weight"), pool), Projection(Mat(b + "ple_value.weight"), pool));
                block.PleOrdinal = pleOrd++;
            }
            blocks[il] = block;
        }

        return new Qwen4ExpTransformerModel(config, gguf, blocks, head, embPtr, embDesc.QuantizationType, output,
                                            pool, ownsPool: pool is not null, owned);
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

    private static Qwen4ExpProjection Projection(MatRef w, ComputeThreadPool? pool) =>
        (input, output, tokens) => GemmSpan(w, input, output, tokens, pool);

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
        => Forward(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><paramref name="kvCache"/> is non-null: the model keeps its own state (see remarks).</exception>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        RejectEngineKvCache(kvCache);
        return Forward(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly: false);
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><paramref name="kvCache"/> is non-null: the model keeps its own state (see remarks).</exception>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           bool lastTokenLogitsOnly)
    {
        RejectEngineKvCache(kvCache);
        return Forward(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly);
    }

    /// <summary>
    /// The oracle owns its sequence state (QSA K/V + indexer keys, GDN, PLE). An engine-owned <see cref="IKvCache"/> would silently
    /// not advance while that state does, and the default <c>ForwardBatch</c> loop would make concurrent sequences share the
    /// model-owned default state (the #261 failure class) — so both are refused until the engine integration of issue #817.
    /// </summary>
    private static void RejectEngineKvCache(IKvCache? kvCache)
    {
        if (kvCache is not null)
            throw new NotSupportedException(
                "Qwen4ExpTransformerModel keeps its own sequence state (QSA K/V, pooled indexer keys, GDN, PLE) and cannot run against an " +
                "engine KV cache yet; call Forward(tokens, positions, deviceId[, lastTokenLogitsOnly]) or the Qwen4ExpSequenceState overload. " +
                "Engine/scheduler integration is tracked in issue #817.");
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Always: batched/scheduler dispatch needs the engine state integration of issue #817.</exception>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
        => throw new NotSupportedException(
            "Qwen4ExpTransformerModel does not support ForwardBatch: requests carry an engine KV cache and the model-owned default state " +
            "would be shared by concurrent sequences. Engine/scheduler integration is tracked in issue #817.");

    /// <summary>
    /// Forward over a caller-owned sequence state (chunked prefill / decode). Positions must continue
    /// <paramref name="state"/> (<c>positions[i] == state.Length + i</c>); the model-owned default state restarts when the
    /// call begins at position 0.
    /// </summary>
    /// <param name="tokenIds">Input tokens.</param>
    /// <param name="positions">Absolute positions.</param>
    /// <param name="deviceId">Device id for the returned tensor.</param>
    /// <param name="state">Sequence state; advanced by <c>tokenIds.Length</c>.</param>
    /// <param name="lastTokenLogitsOnly">Return only the last row's logits (<c>[1, vocab]</c>); otherwise every row.</param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           Qwen4ExpSequenceState state, bool lastTokenLogitsOnly = false)
    {
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (ReferenceEquals(state, _defaultState) && positions[0] == 0 && state.Length != 0) state.Reset();
        for (int i = 0; i < T; i++)
            if (positions[i] != state.Length + i)
                throw new ArgumentException($"qwen4exp keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {state.Length + i}.");
        if (state.Length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");

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
        try
        {
            EmbedTokens(tokenIds, emb);
            Qwen4ExpGatedResidual.Broadcast(emb, S, H, res, T);
            if (Trace is { } tr0) tr0("embed", emb.AsSpan(0, T * H), T, H);

            for (int il = 0; il < _blocks.Length; il++)
            {
                var blk = _blocks[il];
                if (blk.Ple is { } ple)
                    ple.Apply(tokenIds, state.Ple[il]!, res.AsSpan(0, T * row));

                // ── token mixer ──
                GrRead(blk.AttnGr, res, xn, low, mix, h, gains, T, wantInject: true);
                if (blk.Gdn is { } g)
                    ForwardGdn(g, blk.GdnOrdinal, h, y, T, state.Gdn);
                else
                    blk.Qsa!.Forward(h.AsSpan(0, T * H), T, state.Qsa[il]!, y.AsSpan(0, T * H));
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

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float[] dest)
    {
        int H = _hidden;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {Config.VocabSize}).");
            long rowBytes = Dequantize.RowByteSize(H, _tokenEmbedQt);
            Dequantize.ToFloat32(_tokenEmbed + (nint)(id * rowBytes), H, _tokenEmbedQt, dest.AsSpan(t * H, H));
        }
    }

    /// <summary>
    /// Gated DeltaNet token mixer: the Qwen3.5 GDN with a SIGMOID output gate (<c>norm(core) * sigmoid(z)</c>) — llama.cpp
    /// <c>qwen4exp.cpp build_norm_gated</c> ("the one numerical difference from Qwen3.5's GDN") and HF
    /// <c>output_gate_type="sigmoid"</c>.
    /// </summary>
    private void ForwardGdn(GdnTokenMixingWeights w, int ordinal, float[] x, float[] y, int T, GdnStateCache cache)
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
            var xs = x.AsSpan(0, T * _hidden);
            fixed (float* xp = xs)
            {
                fixed (float* o = qkv) Gemm(w.QkvWeight, w.QkvQuantType, xp, o, w.QkvOutputDim, w.QkvInputDim, T, _threadPool);
                fixed (float* o = z) Gemm(w.GateWeight, w.GateQuantType, xp, o, w.GateOutputDim, w.GateInputDim, T, _threadPool);
                fixed (float* o = alpha) Gemm(w.AlphaWeight, w.AlphaQuantType, xp, o, w.AlphaOutputDim, w.AlphaInputDim, T, _threadPool);
                fixed (float* o = beta) Gemm(w.BetaWeight, w.BetaQuantType, xp, o, w.BetaOutputDim, w.BetaInputDim, T, _threadPool);
            }

            // g = exp(softplus(alpha + dt_bias) * A); beta = sigmoid(beta)
            for (int t = 0; t < T; t++)
                for (int vh = 0; vh < nV; vh++)
                {
                    float a = alpha[t * nV + vh] + w.DtBias[vh];
                    float sp = MathF.Log(1f + MathF.Exp(a));
                    alpha[t * nV + vh] = MathF.Exp(sp * w.A[vh]);
                }
            TensorPrimitives.Sigmoid(beta.AsSpan(0, T * nV), beta.AsSpan(0, T * nV));

            // causal conv over [conv_state | qkv] then SiLU
            var convState = cache.GetConvState(ordinal);
            convState.CopyTo(convIn.AsSpan(0, (dC - 1) * convDim));
            qkv.AsSpan(0, T * convDim).CopyTo(convIn.AsSpan((dC - 1) * convDim));
            Conv1dCausal.Execute(convIn.AsSpan(0, (dC - 1 + T) * convDim), w.Conv1dWeight, w.Conv1dBias,
                                 qkv.AsSpan(0, T * convDim), dC, convDim, T);
            SiLu.Execute(qkv.AsSpan(0, T * convDim), qkv.AsSpan(0, T * convDim));
            for (int r = 0; r < dC - 1; r++)
                convIn.AsSpan((T + r) * convDim, convDim).CopyTo(convState.Slice(r * convDim, convDim));

            for (int t = 0; t < T; t++)
            {
                qkv.AsSpan(t * convDim, kDim).CopyTo(q.AsSpan(t * kDim, kDim));
                qkv.AsSpan(t * convDim + kDim, kDim).CopyTo(k.AsSpan(t * kDim, kDim));
                qkv.AsSpan(t * convDim + 2 * kDim, vDim).CopyTo(v.AsSpan(t * vDim, vDim));
            }
            GatedDeltaNetScan.L2NormalizeHeads(q.AsSpan(0, T * kDim), dS);
            GatedDeltaNetScan.L2NormalizeHeads(k.AsSpan(0, T * kDim), dS);

            GatedDeltaNetScan.Execute(cache.GetGdnState(ordinal), q.AsSpan(0, T * kDim), k.AsSpan(0, T * kDim),
                                      v.AsSpan(0, T * vDim), alpha.AsSpan(0, T * nV), beta.AsSpan(0, T * nV),
                                      core.AsSpan(0, T * vDim), nV, nK, dS, T);

            // per-head RMSNorm(core, ssm_norm) * sigmoid(z)
            for (int t = 0; t < T; t++)
                for (int vh = 0; vh < nV; vh++)
                {
                    int off = t * vDim + vh * dS;
                    RmsNorm.Execute(core.AsSpan(off, dS), w.SsmNormWeight, _eps, core.AsSpan(off, dS));
                }
            Span<float> zs = z.AsSpan(0, T * vDim);
            TensorPrimitives.Sigmoid(zs, zs);
            TensorPrimitives.Multiply(core.AsSpan(0, T * vDim), zs, core.AsSpan(0, T * vDim));

            fixed (float* cp = core) fixed (float* yp = y)
                Gemm(w.OutWeight, w.OutQuantType, cp, yp, w.OutOutputDim, w.OutInputDim, T, _threadPool);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(qkv); ArrayPool<float>.Shared.Return(z); ArrayPool<float>.Shared.Return(alpha);
            ArrayPool<float>.Shared.Return(beta); ArrayPool<float>.Shared.Return(convIn); ArrayPool<float>.Shared.Return(q);
            ArrayPool<float>.Shared.Return(k); ArrayPool<float>.Shared.Return(v); ArrayPool<float>.Shared.Return(core);
        }
    }

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
                loraAdapter: null, loraLayer: layer, threadPool: _threadPool);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(assignExpert); ArrayPool<float>.Shared.Return(assignWeight);
            ArrayPool<int>.Shared.Return(cursors); ArrayPool<int>.Shared.Return(bucketTokens);
            ArrayPool<int>.Shared.Return(bucketSlots); ArrayPool<int>.Shared.Return(unique);
        }
    }

    // ───────────────────────────── lifetime ─────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsPool) _threadPool?.Dispose();
        _defaultState.Dispose();
        foreach (nint p in _owned) NativeMemory.AlignedFree((void*)p);
        _owned.Clear();
        GC.SuppressFinalize(this);
    }
}
