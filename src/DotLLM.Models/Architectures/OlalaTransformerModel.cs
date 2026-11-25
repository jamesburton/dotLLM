using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>Receives <c>(name, data, rows, cols)</c> tensors from <see cref="OlalaTransformerModel"/>'s trace hook (tests / bisection).</summary>
internal delegate void OlalaTraceSink(string name, ReadOnlySpan<float> data, int rows, int cols);

/// <summary>
/// CPU reference forward for OVHcloud Olala (<c>olala</c>, Olala-v1.0-mini-7A1B; issue #1970). 36 blocks of
/// <c>geodesic(mixer) → geodesic(latent MoE + shared expert)</c> with no pre-norms and no final norm; the mixer of block <c>i</c> is a
/// Mamba-3 MIMO layer (<c>M</c>, <see cref="OlalaOps.Mamba3MimoStep"/>) or a differential tensor-product-attention layer (<c>V</c>).
/// Authoritative semantics: HF <c>modeling_olala.py</c>; GGUF layout: the OVH llama.cpp fork. See <c>docs/OLALA.md</c>.
/// </summary>
/// <remarks>
/// <para>Correctness over speed. Weights are read in place from the mmap'd GGUF through the existing quantized GEMMs (BF16 / Q8_0 /
/// Q6_K / F32 — never expanded into a stored wider copy); activations are F32 (this model carries activation outliers up to ~1e14, so
/// F16 activation storage is not an option). The Mamba recurrent state is F32 by per-architecture policy
/// (<see cref="OlalaStatePolicy"/>, issue #1971); the attention K/V history is BF16 (HF also casts K and V to bf16 before attention).</para>
/// <para>The model owns its sequence state (SSM, angle, trapezoid and token-shift state, attention K/V); positions must continue it
/// (<c>positions[i] == state.Length + i</c>) and a call at position 0 restarts it. An external <see cref="IKvCache"/> is ignored.</para>
/// </remarks>
public sealed unsafe class OlalaTransformerModel : IModel
{
    private readonly record struct MatRef(nint Ptr, QuantizationType Qt, int In, int Out);

    private sealed class MoeWeights
    {
        public required float[] Router;            // [E][hidden] F32
        public required float[] ExpertBias;        // [E]
        public required MatRef LatentDown, LatentUp, SharedUp, SharedDown;
        public required nint UpExps, DownExps;
        public required QuantizationType UpQt, DownQt;
        public required long UpExpertBytes, DownExpertBytes;
    }

    private sealed class Block
    {
        public required int Index;
        public required bool IsSsm;
        public required float GeoMixScale, GeoMixBias, GeoMlpScale, GeoMlpBias;
        public required MatRef MixerOut;
        public required MoeWeights Moe;
        // M layer
        public MatRef SsmIn, SsmInDyn;
        public OlalaOps.Mamba3Params? Mamba;
        // V layer
        public MatRef Q, WaK, WaV, WbK, WbV, ShiftK, ShiftV, Lambda, Gate;
        public float[]? QNorm, KNorm, SoftmaxScaler;
        public int Ordinal;                        // index among the layers of its kind (state slot)
    }

    private readonly GgufFile? _gguf;
    private readonly Block[] _blocks;
    private readonly nint _tokenEmbed;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly MatRef _output;
    private readonly OlalaConfig _o;
    private readonly OlalaOps.Mamba3Geometry _geo;
    private readonly int _hidden, _vocab, _window;
    private readonly float _eps, _softcap;
    private readonly ComputeThreadPool? _pool;
    private readonly bool _ownsPool;
    private readonly OlalaSequenceState _defaultState;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _defaultState.Bytes;

    /// <summary>Test/diagnostic hook: receives named intermediates (null = off, zero cost).</summary>
    internal OlalaTraceSink? Trace { get; set; }

    private OlalaTransformerModel(ModelConfig config, GgufFile? gguf, Block[] blocks, nint tokenEmbed, QuantizationType tokenEmbedQt,
                                  MatRef output, ComputeThreadPool? pool, bool ownsPool)
    {
        Config = config; _gguf = gguf; _blocks = blocks; _tokenEmbed = tokenEmbed; _tokenEmbedQt = tokenEmbedQt;
        _output = output; _pool = pool; _ownsPool = ownsPool;
        _o = config.Olala!;
        _hidden = config.HiddenSize; _vocab = config.VocabSize;
        _window = config.SlidingWindowSize ?? int.MaxValue;
        _eps = config.NormEpsilon;
        _softcap = config.AttnLogitSoftcap ?? 0f;
        _geo = new OlalaOps.Mamba3Geometry(_o.NumSsmHeads, _o.MambaHeadDim, _o.StateSize, _o.MimoRank, _o.NumRopeAngles, _eps);
        _defaultState = CreateState();
    }

    // ───────────────────────────── state ─────────────────────────────

    /// <summary>Allocates a fresh sequence state (native memory; dispose when the sequence ends).</summary>
    public OlalaSequenceState CreateState() => new(_o, Config);

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Loads an Olala model from an opened GGUF (BF16, Q8_0-safe, K-quant or F32 tensors). The file must outlive the model.</summary>
    public static OlalaTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (config.Architecture != Architecture.Olala || config.Olala is not { } o)
            throw new ArgumentException("OlalaTransformerModel requires an Architecture.Olala configuration.", nameof(config));
        var tensors = gguf.TensorsByName;
        var problems = OlalaTensors.FindProblems(tensors, config);
        if (problems.Count > 0)
            throw new InvalidDataException("olala tensor table does not match the model contract: " + string.Join("; ", problems.Take(8))
                                           + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));

        MatRef Mat(string name)
        {
            var d = tensors[name];
            return new MatRef(gguf.TensorDataPointer(d), d.QuantizationType, d.Shape[0], d.Shape.Rank > 1 ? d.Shape[1] : 1);
        }
        float[] F32(string name)
        {
            var d = tensors[name];
            long count = 1; for (int i = 0; i < d.Shape.Rank; i++) count *= d.Shape[i];
            var r = new float[count];
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, r);
            return r;
        }
        float Scalar(string name) => F32(name)[0];

        var embDesc = tensors[OlalaTensors.TokenEmbd];
        nint embPtr = gguf.TensorDataPointer(embDesc);
        var output = tensors.ContainsKey(OlalaTensors.Output)
            ? Mat(OlalaTensors.Output)
            : new MatRef(embPtr, embDesc.QuantizationType, embDesc.Shape[0], embDesc.Shape[1]);

        var blocks = new Block[config.NumLayers];
        int ssmOrd = 0, attnOrd = 0;
        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => OlalaTensors.Block(il, s);
            var upD = tensors[B("ffn_up_exps.weight")]; var downD = tensors[B("ffn_down_exps.weight")];
            var moe = new MoeWeights
            {
                Router = F32(B("ffn_gate_inp.weight")),
                ExpertBias = F32(B("exp_probs_b")),
                LatentDown = Mat(B("ffn_latent_down.weight")),
                LatentUp = Mat(B("ffn_latent_up.weight")),
                SharedUp = Mat(B("ffn_up_shexp.weight")),
                SharedDown = Mat(B("ffn_down_shexp.weight")),
                UpExps = gguf.TensorDataPointer(upD), UpQt = upD.QuantizationType,
                DownExps = gguf.TensorDataPointer(downD), DownQt = downD.QuantizationType,
                UpExpertBytes = Dequantize.RowByteSize(upD.Shape[0], upD.QuantizationType) * upD.Shape[1],
                DownExpertBytes = Dequantize.RowByteSize(downD.Shape[0], downD.QuantizationType) * downD.Shape[1],
            };
            var blk = new Block
            {
                Index = il,
                IsSsm = o.IsSsmLayer(il),
                GeoMixScale = Scalar(B("geo_mixer_scale")), GeoMixBias = Scalar(B("geo_mixer_bias")),
                GeoMlpScale = Scalar(B("geo_mlp_scale")), GeoMlpBias = Scalar(B("geo_mlp_bias")),
                MixerOut = Mat(B("attn_output.weight")),
                Moe = moe,
            };
            if (blk.IsSsm)
            {
                blk.Ordinal = ssmOrd++;
                blk.SsmIn = Mat(B("ssm_in.weight"));
                blk.SsmInDyn = Mat(B("ssm_in_dyn.weight"));
                blk.Mamba = new OlalaOps.Mamba3Params(
                    BNorm: F32(B("ssm_b_norm.weight")), CNorm: F32(B("ssm_c_norm.weight")),
                    BBias: F32(B("ssm_b_bias")), CBias: F32(B("ssm_c_bias")),
                    DtBias: F32(B("ssm_dt_bias")), D: F32(B("ssm_d")),
                    MimoX: F32(B("ssm_mimo_x")), MimoZ: F32(B("ssm_mimo_z")), MimoO: F32(B("ssm_mimo_o")));
            }
            else
            {
                blk.Ordinal = attnOrd++;
                blk.Q = Mat(B("attn_q.weight")); blk.WaK = Mat(B("attn_wa_k.weight")); blk.WaV = Mat(B("attn_wa_v.weight"));
                blk.WbK = Mat(B("attn_wb_k.weight")); blk.WbV = Mat(B("attn_wb_v.weight"));
                blk.ShiftK = Mat(B("attn_shift_k.weight")); blk.ShiftV = Mat(B("attn_shift_v.weight"));
                blk.Lambda = Mat(B("attn_lambda.weight")); blk.Gate = Mat(B("attn_gate.weight"));
                blk.QNorm = F32(B("attn_q_norm.weight")); blk.KNorm = F32(B("attn_k_norm.weight"));
                blk.SoftmaxScaler = F32(B("attn_softmax_scaler"));
            }
            blocks[il] = blk;
        }

        var pool = CreatePool(threading);
        return new OlalaTransformerModel(config, gguf, blocks, embPtr, embDesc.QuantizationType, output, pool, ownsPool: pool is not null);
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

    /// <summary>
    /// <c>c[t, i] = dot(W[i, :], b[t, :])</c> for <c>m</c> rows of <c>k</c> weights. Quantized and BF16 weights are read in place: Q8_0 / K-quant
    /// kernels quantize the activations per block with a float or F16 scale (fine for the bounded inputs of the q8_0 tensors of safe2); BF16
    /// goes through a row-streaming dequantize-and-dot with F32 activations (the unbounded-input tensors of safe2 are exactly the BF16/Q6_K ones).
    /// </summary>
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
    public ITensor ForwardWithState(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, OlalaSequenceState state, bool lastTokenLogitsOnly = false)
        => ForwardCore(tokenIds, positions, deviceId, state, lastTokenLogitsOnly);

    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, OlalaSequenceState state, bool lastTokenLogitsOnly)
    {
        ArgumentNullException.ThrowIfNull(state);
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length) throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        if (ReferenceEquals(state, _defaultState) && positions[0] == 0 && state.Length != 0) state.Reset();
        for (int i = 0; i < T; i++)
            if (positions[i] != state.Length + i)
                throw new ArgumentException($"olala keeps its own sequence state: positions[{i}] = {positions[i]} but the state is at {state.Length + i}.");
        if (state.Length + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Sequence would exceed the context length {Config.MaxSequenceLength}.");
        return ForwardBody(tokenIds, deviceId, state, lastTokenLogitsOnly);
    }

    private ITensor ForwardBody(ReadOnlySpan<int> tokenIds, int deviceId, OlalaSequenceState st, bool lastOnly)
    {
        int T = tokenIds.Length, H = _hidden;
        _pool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);
        float[] res = ArrayPool<float>.Shared.Rent(T * H);
        float[] mix = ArrayPool<float>.Shared.Rent(T * H);
        try
        {
            // Embedding rows, L2-normalised (normalize_embeddings_ngpt): the residual then stays on the unit sphere.
            long rowBytes = Dequantize.RowByteSize(H, _tokenEmbedQt);
            for (int t = 0; t < T; t++)
            {
                int id = tokenIds[t];
                if ((uint)id >= (uint)_vocab)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {_vocab}).");
                var row = res.AsSpan(t * H, H);
                Dequantize.ToFloat32(_tokenEmbed + (nint)(id * rowBytes), H, _tokenEmbedQt, row);
                OlalaOps.L2Normalize(row);
            }
            Trace?.Invoke("embed", res.AsSpan(0, T * H), T, H);

            foreach (var blk in _blocks)
            {
                if (blk.IsSsm) ForwardSsm(blk, res, mix, T, st);
                else ForwardAttention(blk, res, mix, T, st);
                Trace?.Invoke($"blk.{blk.Index}.mixer_out", mix.AsSpan(0, T * H), T, H);
                for (int t = 0; t < T; t++)
                    OlalaOps.GeodesicUpdate(res.AsSpan(t * H, H), mix.AsSpan(t * H, H), res.AsSpan(t * H, H), blk.GeoMixScale, blk.GeoMixBias, blk.Index);
                Trace?.Invoke($"blk.{blk.Index}.post_mixer", res.AsSpan(0, T * H), T, H);

                ForwardMoe(blk.Moe, res, mix, T);
                Trace?.Invoke($"blk.{blk.Index}.mlp_out", mix.AsSpan(0, T * H), T, H);
                for (int t = 0; t < T; t++)
                    OlalaOps.GeodesicUpdate(res.AsSpan(t * H, H), mix.AsSpan(t * H, H), res.AsSpan(t * H, H), blk.GeoMlpScale, blk.GeoMlpBias, blk.Index);
                Trace?.Invoke($"blk.{blk.Index}.l_out", res.AsSpan(0, T * H), T, H);
                AssertUnitNorm(res, T, H, blk.Index);
            }

            int rows = lastOnly ? 1 : T;
            int firstRow = T - rows;
            var result = UnmanagedTensor.Allocate(new TensorShape(rows, _vocab), DType.Float32, deviceId);
            var logits = new Span<float>((void*)result.DataPointer, rows * _vocab);
            GemmSpan(_output, res.AsSpan(firstRow * H, rows * H), logits, rows);
            st.Length += T;
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(res);
            ArrayPool<float>.Shared.Return(mix);
        }
    }

    /// <summary>
    /// The embeddings are unit vectors and each geodesic update is a rotation (norm-preserving up to the 1e-6 clamp), so the residual
    /// stays at norm 1 after every block; a deviation means a geodesic / embedding bug without needing an oracle. Debug builds only.
    /// </summary>
    [Conditional("DEBUG")]
    private static void AssertUnitNorm(float[] res, int T, int H, int layer)
    {
        for (int t = 0; t < T; t++)
        {
            double ss = 0;
            for (int i = 0; i < H; i++) ss += (double)res[t * H + i] * res[t * H + i];
            Debug.Assert(Math.Abs(Math.Sqrt(ss) - 1.0) < 1e-3, $"olala residual norm {Math.Sqrt(ss)} after block {layer} (token {t}) is not 1");
        }
    }

    // ───────────────────────────── M layer ─────────────────────────────

    private void ForwardSsm(Block blk, float[] x, float[] mixOut, int T, OlalaSequenceState st)
    {
        int H = _hidden, inW = _geo.Heads * _geo.PerHead, dynW = _geo.DynWidth, yW = _geo.Heads * _geo.HeadDim;
        float[] zxdt = ArrayPool<float>.Shared.Rent(T * inW);
        float[] dyn = ArrayPool<float>.Shared.Rent(T * dynW);
        float[] y = ArrayPool<float>.Shared.Rent(T * yW);
        try
        {
            GemmSpan(blk.SsmIn, x.AsSpan(0, T * H), zxdt, T);
            GemmSpan(blk.SsmInDyn, x.AsSpan(0, T * H), dyn, T);
            var s = st.Ssm[blk.Ordinal];
            for (int t = 0; t < T; t++)
                OlalaOps.Mamba3MimoStep(_geo, blk.Mamba!, zxdt.AsSpan(t * inW, inW), dyn.AsSpan(t * dynW, dynW),
                                        s.Ssm.Span, s.Angle.Span, s.K.Span, s.V.Span, y.AsSpan(t * yW, yW), parallel: _pool is not null);
            Trace?.Invoke($"blk.{blk.Index}.ssm_y", y.AsSpan(0, T * yW), T, yW);
            GemmSpan(blk.MixerOut, y, mixOut, T);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(zxdt); ArrayPool<float>.Shared.Return(dyn); ArrayPool<float>.Shared.Return(y);
        }
    }

    // ───────────────────────────── V layer ─────────────────────────────

    /// <summary>
    /// Differential tensor-product attention (HF <c>OlalaDifferentialTensorProductAttentionV2</c> + the block gate): K and V are
    /// <c>A (kv heads x rank) @ B (rank x head_dim) / rank</c>, blended with the previous token's raw K/V by a learned sigmoid
    /// (token shift, zero at position 0), K is RMS-normed, Q is RMS-normed and scaled by <c>softmax_scaler * log(min(pos + 1, window))</c>;
    /// scores use scale <c>1/head_dim</c>, a tanh soft cap and a causal window; every group of four query heads is three signal heads
    /// and one noise head combined as <c>signal - sigmoid(lambda) * noise</c>, gated by <c>silu(gate_proj(x) + gate_bias)</c>, then
    /// <c>attn_output</c>. No RoPE.
    /// </summary>
    private void ForwardAttention(Block blk, float[] x, float[] mixOut, int T, OlalaSequenceState st)
    {
        int H = _hidden, hd = Config.HeadDim, nq = _o.NumSignalHeads + _o.NumNoiseHeads, nk = _o.NumNoiseHeads, rank = _o.TpaRank;
        int group = nq / nk, snr = group - 1, outHeads = _o.NumSignalHeads;
        int qW = nq * hd, kvW = nk * hd, gateW = outHeads * hd;
        var a = st.Attn[blk.Ordinal];
        a.EnsureCapacity(st.Length + T);

        float[] q = ArrayPool<float>.Shared.Rent(T * qW);
        float[] ak = ArrayPool<float>.Shared.Rent(T * nk * rank), av = ArrayPool<float>.Shared.Rent(T * nk * rank);
        float[] bk = ArrayPool<float>.Shared.Rent(T * rank * hd), bv = ArrayPool<float>.Shared.Rent(T * rank * hd);
        float[] sk = ArrayPool<float>.Shared.Rent(T * nk), sv = ArrayPool<float>.Shared.Rent(T * nk), lam = ArrayPool<float>.Shared.Rent(T * nk);
        float[] gate = ArrayPool<float>.Shared.Rent(T * gateW);
        float[] yv = ArrayPool<float>.Shared.Rent(T * gateW);
        try
        {
            var xs = x.AsSpan(0, T * H);
            GemmSpan(blk.Q, xs, q, T);
            GemmSpan(blk.WaK, xs, ak, T); GemmSpan(blk.WaV, xs, av, T);
            GemmSpan(blk.WbK, xs, bk, T); GemmSpan(blk.WbV, xs, bv, T);
            GemmSpan(blk.ShiftK, xs, sk, T); GemmSpan(blk.ShiftV, xs, sv, T);
            GemmSpan(blk.Lambda, xs, lam, T);
            GemmSpan(blk.Gate, xs, gate, T);

            float[] rawK = ArrayPool<float>.Shared.Rent(kvW), rawV = ArrayPool<float>.Shared.Rent(kvW);
            float[] prevK = ArrayPool<float>.Shared.Rent(kvW), prevV = ArrayPool<float>.Shared.Rent(kvW);
            float[] kk = ArrayPool<float>.Shared.Rent(kvW), vv = ArrayPool<float>.Shared.Rent(kvW);
            float[] scores = ArrayPool<float>.Shared.Rent(Math.Min(_window, st.Length + T) + 2);
            float[] attnOut = ArrayPool<float>.Shared.Rent(qW);
            try
            {
                a.LastK.Span.Slice(0, kvW).CopyTo(prevK); a.LastV.Span.Slice(0, kvW).CopyTo(prevV);
                float invRank = 1f / rank;
                for (int t = 0; t < T; t++)
                {
                    int pos = st.Length + t;
                    // TPA reconstruction of the raw (pre-shift) K and V of this token.
                    for (int h = 0; h < nk; h++)
                        for (int d = 0; d < hd; d++)
                        {
                            float k = 0f, v = 0f;
                            for (int r = 0; r < rank; r++)
                            {
                                k += ak[t * nk * rank + h * rank + r] * bk[t * rank * hd + r * hd + d];
                                v += av[t * nk * rank + h * rank + r] * bv[t * rank * hd + r * hd + d];
                            }
                            rawK[h * hd + d] = k * invRank; rawV[h * hd + d] = v * invRank;
                        }
                    // Token shift: alpha = sigmoid(proj), forced to 0 (no previous token) at position 0.
                    for (int h = 0; h < nk; h++)
                    {
                        float alK = pos == 0 ? 0f : OlalaOps.Sigmoid(sk[t * nk + h]);
                        float alV = pos == 0 ? 0f : OlalaOps.Sigmoid(sv[t * nk + h]);
                        for (int d = 0; d < hd; d++)
                        {
                            kk[h * hd + d] = alK * (pos == 0 ? 0f : prevK[h * hd + d]) + (1f - alK) * rawK[h * hd + d];
                            vv[h * hd + d] = alV * (pos == 0 ? 0f : prevV[h * hd + d]) + (1f - alV) * rawV[h * hd + d];
                        }
                    }
                    // The pre-shift K/V of this token are the next token's "previous".
                    rawK.AsSpan(0, kvW).CopyTo(prevK); rawV.AsSpan(0, kvW).CopyTo(prevV);

                    OlalaOps.RmsNormSlices(kk.AsSpan(0, kvW), blk.KNorm!, hd, _eps);
                    var qt = q.AsSpan(t * qW, qW);
                    OlalaOps.RmsNormSlices(qt, blk.QNorm!, hd, _eps);
                    float logPos = MathF.Log(MathF.Min(pos + 1f, _window));
                    for (int h = 0; h < nq; h++)
                        TensorPrimitives.Multiply(qt.Slice(h * hd, hd), blk.SoftmaxScaler![h] * logPos, qt.Slice(h * hd, hd));

                    a.Store(pos, kk, vv, kvW);

                    int lo = Math.Max(0, pos - _window);
                    int nKeys = pos - lo + 1;
                    float scale = 1f / hd;
                    for (int h = 0; h < nq; h++)
                    {
                        int kvh = h / group;
                        var qh = qt.Slice(h * hd, hd);
                        float max = float.NegativeInfinity;
                        for (int j = 0; j < nKeys; j++)
                        {
                            float s = a.DotK(lo + j, kvh, hd, kvW, qh) * scale;
                            if (_softcap > 0f) s = MathF.Tanh(s / _softcap) * _softcap;
                            scores[j] = s;
                            if (s > max) max = s;
                        }
                        double sum = 0;
                        for (int j = 0; j < nKeys; j++) { float e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                        float inv = (float)(1.0 / sum);
                        var o = attnOut.AsSpan(h * hd, hd);
                        o.Clear();
                        for (int j = 0; j < nKeys; j++) a.AddV(lo + j, kvh, hd, kvW, scores[j] * inv, o);
                    }

                    // Differential combine and gate: per kv group, heads [0, snr) are signal and the last one is noise.
                    for (int g = 0; g < nk; g++)
                    {
                        float lambda = OlalaOps.Sigmoid(lam[t * nk + g]);
                        for (int s = 0; s < snr; s++)
                            for (int d = 0; d < hd; d++)
                            {
                                int outIdx = (g * snr + s) * hd + d;
                                float combined = attnOut[(g * group + s) * hd + d] - lambda * attnOut[(g * group + snr) * hd + d];
                                yv[t * gateW + outIdx] = combined * OlalaOps.Silu(gate[t * gateW + outIdx] + _o.GateBias);
                            }
                    }
                }
                prevK.AsSpan(0, kvW).CopyTo(a.LastK.Span); prevV.AsSpan(0, kvW).CopyTo(a.LastV.Span);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rawK); ArrayPool<float>.Shared.Return(rawV); ArrayPool<float>.Shared.Return(prevK);
                ArrayPool<float>.Shared.Return(prevV); ArrayPool<float>.Shared.Return(kk); ArrayPool<float>.Shared.Return(vv);
                ArrayPool<float>.Shared.Return(scores); ArrayPool<float>.Shared.Return(attnOut);
            }
            Trace?.Invoke($"blk.{blk.Index}.attn_y", yv.AsSpan(0, T * gateW), T, gateW);
            GemmSpan(blk.MixerOut, yv, mixOut, T);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(q); ArrayPool<float>.Shared.Return(ak); ArrayPool<float>.Shared.Return(av);
            ArrayPool<float>.Shared.Return(bk); ArrayPool<float>.Shared.Return(bv); ArrayPool<float>.Shared.Return(sk);
            ArrayPool<float>.Shared.Return(sv); ArrayPool<float>.Shared.Return(lam); ArrayPool<float>.Shared.Return(gate);
            ArrayPool<float>.Shared.Return(yv);
        }
    }

    // ───────────────────────────── MoE ─────────────────────────────

    /// <summary>
    /// Latent MoE + shared expert (HF <c>OlalaMoE</c>): router <c>sigmoid(x W^T)</c> in F32, top-k of <c>score + expert_bias</c>, the selected
    /// UN-biased scores normalised to sum 1 and scaled by <c>expert_weights_scale</c>; the routed experts run on the latent
    /// <c>down(x)</c> (<c>relu(up e x)^2</c> then <c>down e</c>), the weighted sum goes back up to the hidden width; the shared expert is
    /// <c>fc2(relu(fc1 x)^2)</c> on <c>x</c>; the result is routed + shared.
    /// </summary>
    private void ForwardMoe(MoeWeights m, float[] x, float[] y, int T)
    {
        int H = _hidden, E = _o.ExpertCount, K = _o.ExpertUsedCount, lat = m.LatentDown.Out, ff = _o.ExpertFfLength;
        float[] logits = ArrayPool<float>.Shared.Rent(T * E);
        float[] x0 = ArrayPool<float>.Shared.Rent(T * lat);
        float[] routedLat = ArrayPool<float>.Shared.Rent(T * lat);
        float[] routed = ArrayPool<float>.Shared.Rent(T * H);
        int[] selExpert = ArrayPool<int>.Shared.Rent(T * K);
        float[] selWeight = ArrayPool<float>.Shared.Rent(T * K);
        try
        {
            // Router in F32 on the F32 residual.
            fixed (float* wr = m.Router)
            fixed (float* xp = x)
            fixed (float* lp = logits)
                MatMul.GemmF32(wr, xp, lp, E, H, T, _pool);
            Span<float> scored = stackalloc float[E];
            Span<float> sig = stackalloc float[E];
            for (int t = 0; t < T; t++)
            {
                for (int e = 0; e < E; e++)
                {
                    sig[e] = OlalaOps.Sigmoid(logits[t * E + e]);
                    scored[e] = sig[e] + m.ExpertBias[e];
                }
                SelectTopK(scored, K, selExpert.AsSpan(t * K, K));
                float sum = 0f;
                for (int i = 0; i < K; i++) sum += sig[selExpert[t * K + i]];
                for (int i = 0; i < K; i++)
                    selWeight[t * K + i] = sig[selExpert[t * K + i]] / (sum + 1e-20f) * _o.ExpertWeightsScale;
            }

            GemmSpan(m.LatentDown, x.AsSpan(0, T * H), x0, T);
            Array.Clear(routedLat, 0, T * lat);
            ExecuteExperts(m, x0, routedLat, selExpert, selWeight, T, K, E, lat, ff);
            GemmSpan(m.LatentUp, routedLat.AsSpan(0, T * lat), routed, T);

            // Shared expert on the residual itself.
            int sff = m.SharedUp.Out;
            float[] sh = ArrayPool<float>.Shared.Rent(T * sff);
            try
            {
                GemmSpan(m.SharedUp, x.AsSpan(0, T * H), sh, T);
                ReluSquared(sh.AsSpan(0, T * sff));
                GemmSpan(m.SharedDown, sh.AsSpan(0, T * sff), y, T);
            }
            finally { ArrayPool<float>.Shared.Return(sh); }
            TensorPrimitives.Add(y.AsSpan(0, T * H), routed.AsSpan(0, T * H), y.AsSpan(0, T * H));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(logits); ArrayPool<float>.Shared.Return(x0); ArrayPool<float>.Shared.Return(routedLat);
            ArrayPool<float>.Shared.Return(routed); ArrayPool<int>.Shared.Return(selExpert); ArrayPool<float>.Shared.Return(selWeight);
        }
    }

    private void ExecuteExperts(MoeWeights m, float[] x0, float[] outLat, int[] selExpert, float[] selWeight, int T, int K, int E, int lat, int ff)
    {
        // Bucket (token, slot) assignments by expert so each expert runs one batched GEMM over its tokens.
        int[] count = new int[E + 1];
        for (int i = 0; i < T * K; i++) count[selExpert[i] + 1]++;
        for (int e = 0; e < E; e++) count[e + 1] += count[e];
        int[] cursor = (int[])count.Clone();
        int[] tokenOf = new int[T * K]; float[] weightOf = new float[T * K];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < K; i++)
            {
                int e = selExpert[t * K + i], at = cursor[e]++;
                tokenOf[at] = t; weightOf[at] = selWeight[t * K + i];
            }
        for (int e = 0; e < E; e++)
        {
            int n = count[e + 1] - count[e], start = count[e];
            if (n == 0) continue;
            float[] xin = ArrayPool<float>.Shared.Rent(n * lat);
            float[] h = ArrayPool<float>.Shared.Rent(n * ff);
            float[] yo = ArrayPool<float>.Shared.Rent(n * lat);
            try
            {
                for (int i = 0; i < n; i++) x0.AsSpan(tokenOf[start + i] * lat, lat).CopyTo(xin.AsSpan(i * lat, lat));
                fixed (float* xp = xin) fixed (float* hp = h) fixed (float* yp = yo)
                {
                    Gemm(m.UpExps + (nint)(e * m.UpExpertBytes), m.UpQt, xp, hp, ff, lat, n, _pool);
                    ReluSquared(h.AsSpan(0, n * ff));
                    Gemm(m.DownExps + (nint)(e * m.DownExpertBytes), m.DownQt, hp, yp, lat, ff, n, _pool);
                }
                for (int i = 0; i < n; i++)
                {
                    float w = weightOf[start + i];
                    var dst = outLat.AsSpan(tokenOf[start + i] * lat, lat);
                    var src = yo.AsSpan(i * lat, lat);
                    for (int c = 0; c < lat; c++) dst[c] += w * src[c];
                }
            }
            finally { ArrayPool<float>.Shared.Return(xin); ArrayPool<float>.Shared.Return(h); ArrayPool<float>.Shared.Return(yo); }
        }
    }

    private static void ReluSquared(Span<float> v)
    {
        for (int i = 0; i < v.Length; i++) { float r = MathF.Max(v[i], 0f); v[i] = r * r; }
    }

    /// <summary>Indices of the <paramref name="k"/> largest values, descending; ties resolve to the lower index.</summary>
    private static void SelectTopK(ReadOnlySpan<float> v, int k, Span<int> idx)
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
        GC.SuppressFinalize(this);
    }
}
