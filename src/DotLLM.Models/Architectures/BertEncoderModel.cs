using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using NativeMemory = System.Runtime.InteropServices.NativeMemory;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// BERT-class bidirectional encoder (<c>bert</c>, <c>nomic-bert</c>) on the CPU backend (issue #739).
/// </summary>
/// <remarks>
/// <para>Graph follows llama.cpp <c>src/models/bert.cpp</c> step for step (that file is the oracle):</para>
/// <list type="number">
///   <item><description>Embedding: token row + <c>token_types</c> row 0 (type ids are hard-wired to zero)
///     + (<c>bert</c> only) <c>position_embd[pos]</c>, then LayerNorm with weight and bias.</description></item>
///   <item><description>Per layer (post-LN): Q/K/V projection (fused <c>attn_qkv</c> or separate, optional
///     biases), RoPE NeoX on Q/K for <c>nomic-bert</c>, bidirectional SDPA at <c>1/sqrt(head_dim)</c>,
///     output projection (+bias), residual add, <c>attn_output_norm</c>; FFN (<c>bert</c>: up+bias, GELU,
///     down+bias; <c>nomic-bert</c>: <c>down(silu(gate(x)) * up(x))</c>), residual add,
///     <c>layer_output_norm</c>.</description></item>
///   <item><description>No final norm: the last layer's output is the hidden state pooled by
///     <c>EmbeddingPooler</c> (<c>result_embd</c> in llama.cpp).</description></item>
/// </list>
/// <para>This is an <b>embedding-only</b> model: <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/>
/// throws, there is no LM head, no KV cache and no sampler.</para>
/// </remarks>
public sealed unsafe class BertEncoderModel : IModel, IEmbeddingModel
{
    private readonly GgufFile _gguf;
    private readonly Layer[] _layers;
    private readonly nint _tokenEmbd;
    private readonly QuantizationType _tokenEmbdQt;
    private readonly nint _typeEmbd;           // row 0 only; 0 when absent
    private readonly QuantizationType _typeEmbdQt;
    private readonly nint _posEmbd;            // bert only
    private readonly QuantizationType _posEmbdQt;
    private readonly int _posRows;
    private readonly float[] _embNormW;
    private readonly float[] _embNormB;
    private readonly float _eps;
    private readonly bool _nomic;
    private readonly ComputeThreadPool? _pool;
    private readonly float[]? _ropeCos;
    private readonly float[]? _ropeSin;
    private readonly int _ropeRows;
    private bool _disposed;

    /// <summary>Test seam: erf-exact GELU instead of llama.cpp's tanh approximation.</summary>
    internal static bool UseErfGelu { get; set; }

    /// <summary>Test seam: skip Q8 activation quantisation for quantized weights (dequantize-and-dot with F32 activations).</summary>
    internal static bool ForceF32Activations { get; set; }

    private sealed class Proj
    {
        public nint Weight;
        public QuantizationType Qt;
        public int M;   // output dim
        public int K;   // input dim
        public float[]? Bias;
    }

    private sealed class Layer
    {
        public Proj? Qkv, Q, K, V, O, Up, Gate, Down;
        public required float[] AttnNormW, AttnNormB, OutNormW, OutNormB;
    }

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => 0;

    /// <inheritdoc/>
    public PoolingType? DeclaredPoolingType => Config.PoolingType;

    private BertEncoderModel(
        GgufFile gguf, ModelConfig config, Layer[] layers,
        nint tokenEmbd, QuantizationType tokenEmbdQt,
        nint typeEmbd, QuantizationType typeEmbdQt,
        nint posEmbd, QuantizationType posEmbdQt, int posRows,
        float[] embNormW, float[] embNormB, ComputeThreadPool? pool,
        float[]? ropeCos, float[]? ropeSin, int ropeRows)
    {
        _gguf = gguf;
        Config = config;
        _layers = layers;
        _tokenEmbd = tokenEmbd; _tokenEmbdQt = tokenEmbdQt;
        _typeEmbd = typeEmbd; _typeEmbdQt = typeEmbdQt;
        _posEmbd = posEmbd; _posEmbdQt = posEmbdQt; _posRows = posRows;
        _embNormW = embNormW; _embNormB = embNormB;
        _eps = config.NormEpsilon;
        _nomic = config.Architecture == Architecture.NomicBert;
        _pool = pool;
        _ropeCos = ropeCos; _ropeSin = ropeSin; _ropeRows = ropeRows;
    }

    /// <summary>Loads a <c>bert</c> / <c>nomic-bert</c> GGUF onto the CPU backend.</summary>
    public static BertEncoderModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig? threading = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (config.Architecture is not (Architecture.Bert or Architecture.NomicBert))
            throw new ArgumentException($"BertEncoderModel requires Architecture.Bert/NomicBert, got {config.Architecture}.", nameof(config));

        var dataBase = new GgufDataBase(gguf);
        var t = gguf.TensorsByName;
        int h = config.HiddenSize;
        bool nomic = config.Architecture == Architecture.NomicBert;

        GgufTensorDescriptor Req(string name)
            => t.TryGetValue(name, out var d) ? d
               : throw new InvalidDataException($"BERT GGUF is missing required tensor '{name}'.");

        float[] Vec(string name, int n)
        {
            var d = Req(name);
            var r = new float[n];
            Dequantize.ToFloat32(dataBase.Of(d), n, d.QuantizationType, r);
            return r;
        }

        float[]? OptVec(string name, int n) => t.ContainsKey(name) ? Vec(name, n) : null;

        Proj? OptProj(string weight, string? bias)
        {
            if (!t.TryGetValue(weight, out var d)) return null;
            int k = d.Shape[0], m = d.Shape[1];
            return new Proj
            {
                Weight = dataBase.Of(d),
                Qt = d.QuantizationType,
                K = k,
                M = m,
                Bias = bias is not null ? OptVec(bias, m) : null,
            };
        }

        Proj ReqProj(string weight, string? bias)
            => OptProj(weight, bias) ?? throw new InvalidDataException($"BERT GGUF is missing required tensor '{weight}'.");

        var tokDesc = Req("token_embd.weight");
        if (tokDesc.Shape[0] != h)
            throw new InvalidDataException($"token_embd.weight row width {tokDesc.Shape[0]} != hidden size {h}.");

        nint typePtr = 0; var typeQt = QuantizationType.F32;
        if (t.TryGetValue("token_types.weight", out var typeDesc))
        {
            typePtr = dataBase.Of(typeDesc);
            typeQt = typeDesc.QuantizationType;
        }

        nint posPtr = 0; var posQt = QuantizationType.F32; int posRows = 0;
        if (!nomic)
        {
            var posDesc = Req("position_embd.weight");
            posPtr = dataBase.Of(posDesc);
            posQt = posDesc.QuantizationType;
            posRows = posDesc.Shape[1];
        }

        var layers = new Layer[config.NumLayers];
        for (int i = 0; i < layers.Length; i++)
        {
            string p = $"blk.{i}.";
            var l = new Layer
            {
                AttnNormW = Vec(p + "attn_output_norm.weight", h),
                AttnNormB = Vec(p + "attn_output_norm.bias", h),
                OutNormW = Vec(p + "layer_output_norm.weight", h),
                OutNormB = Vec(p + "layer_output_norm.bias", h),
                O = ReqProj(p + "attn_output.weight", p + "attn_output.bias"),
                Up = ReqProj(p + "ffn_up.weight", p + "ffn_up.bias"),
                Down = ReqProj(p + "ffn_down.weight", p + "ffn_down.bias"),
                Gate = OptProj(p + "ffn_gate.weight", null),
            };
            l.Qkv = OptProj(p + "attn_qkv.weight", p + "attn_qkv.bias");
            if (l.Qkv is null)
            {
                l.Q = ReqProj(p + "attn_q.weight", p + "attn_q.bias");
                l.K = ReqProj(p + "attn_k.weight", p + "attn_k.bias");
                l.V = ReqProj(p + "attn_v.weight", p + "attn_v.bias");
            }
            if (nomic && l.Gate is null)
                throw new InvalidDataException($"nomic-bert layer {i} is missing ffn_gate.weight (SwiGLU).");
            layers[i] = l;
        }

        float[]? cos = null, sin = null; int ropeRows = 0;
        if (nomic)
        {
            var rope = config.RoPEConfig ?? throw new InvalidDataException("nomic-bert config has no RoPE configuration.");
            ropeRows = config.MaxSequenceLength;
            int half = config.HeadDim / 2;
            cos = new float[ropeRows * half];
            sin = new float[ropeRows * half];
            RoPE.PrecomputeFrequencyTable(ropeRows, config.HeadDim, rope.Theta, cos, sin);
        }

        ComputeThreadPool? pool = null;
        if (threading is { IsParallel: true } th)
            pool = new ComputeThreadPool(th.EffectiveThreadCount, topology: null, th);

        return new BertEncoderModel(
            gguf, config, layers,
            dataBase.Of(tokDesc), tokDesc.QuantizationType,
            typePtr, typeQt, posPtr, posQt, posRows,
            Vec("token_embd_norm.weight", h), Vec("token_embd_norm.bias", h), pool,
            cos, sin, ropeRows);
    }

    // ───────────────────────────── IModel: embedding-only ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => throw NoLmHead();

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => throw NoLmHead();

    private NotSupportedException NoLmHead() => new(
        $"{Config.Architecture} is an embedding-only encoder: it has no LM head, KV cache or sampler. "
        + "Use POST /v1/embeddings or `dotllm embed`.");

    // ───────────────────────────── forward ─────────────────────────────

    /// <inheritdoc/>
    public ITensor ForwardHidden(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int n = tokenIds.Length;
        if (n == 0) throw new ArgumentException("Empty token sequence.", nameof(tokenIds));
        if (positions.Length != n) throw new ArgumentException("positions and tokenIds must have equal length.", nameof(positions));
        if (n > Config.MaxSequenceLength)
            throw new ArgumentException($"Sequence of {n} tokens exceeds the model's context length {Config.MaxSequenceLength}.", nameof(tokenIds));

        int h = Config.HiddenSize, ff = Config.IntermediateSize;
        int nh = Config.NumAttentionHeads, hd = Config.HeadDim;

        float* x = Alloc((long)n * h);
        float* qkv = Alloc((long)n * 3 * h);
        float* q = Alloc((long)n * h);
        float* k = Alloc((long)n * h);
        float* v = Alloc((long)n * h);
        float* attn = Alloc((long)n * h);
        float* tmp = Alloc((long)n * h);
        float* up = Alloc((long)n * ff);
        float* gate = _nomic ? Alloc((long)n * ff) : null;
        try
        {
            Embed(tokenIds, positions, x, h);

            var posSpan = positions;
            for (int li = 0; li < _layers.Length; li++)
            {
                var l = _layers[li];

                // ── attention ──
                if (l.Qkv is not null)
                {
                    Gemm(l.Qkv, x, qkv, n);
                    AddBias(l.Qkv.Bias, qkv, n, l.Qkv.M);
                    for (int t = 0; t < n; t++)
                    {
                        float* row = qkv + (long)t * 3 * h;
                        new ReadOnlySpan<float>(row, h).CopyTo(new Span<float>(q + (long)t * h, h));
                        new ReadOnlySpan<float>(row + h, h).CopyTo(new Span<float>(k + (long)t * h, h));
                        new ReadOnlySpan<float>(row + 2 * h, h).CopyTo(new Span<float>(v + (long)t * h, h));
                    }
                }
                else
                {
                    Gemm(l.Q!, x, q, n); AddBias(l.Q!.Bias, q, n, h);
                    Gemm(l.K!, x, k, n); AddBias(l.K!.Bias, k, n, h);
                    Gemm(l.V!, x, v, n); AddBias(l.V!.Bias, v, n, h);
                }

                if (_nomic)
                {
                    for (int t = 0; t < n; t++)
                        if ((uint)posSpan[t] >= (uint)_ropeRows)
                            throw new ArgumentOutOfRangeException(nameof(positions), $"Position {posSpan[t]} exceeds the RoPE table ({_ropeRows}).");
                    RoPE.Execute(new Span<float>(q, n * h), new Span<float>(k, n * h), posSpan,
                        nh, nh, hd, Config.RoPEConfig!.Value.DimensionCount, _ropeCos, _ropeSin, RoPEType.NeoX);
                }

                Attention.Execute(
                    new ReadOnlySpan<float>(q, n * h), new ReadOnlySpan<float>(k, n * h), new ReadOnlySpan<float>(v, n * h),
                    new Span<float>(attn, n * h),
                    n, n, nh, nh, hd, 0, 1.0f / MathF.Sqrt(hd), default, null, 0f,
                    AttentionMaskMode.Bidirectional, 0, default);

                Gemm(l.O!, attn, tmp, n);
                AddBias(l.O!.Bias, tmp, n, h);
                // residual + post-attention LayerNorm
                TensorPrimitives.Add(new ReadOnlySpan<float>(tmp, n * h), new ReadOnlySpan<float>(x, n * h), new Span<float>(tmp, n * h));
                LayerNorm(tmp, l.AttnNormW, l.AttnNormB, x, n, h);   // x := LN(attn_out + x_in)... see below

                // x now holds ffn_inp; keep a copy in tmp for the FFN residual.
                new ReadOnlySpan<float>(x, n * h).CopyTo(new Span<float>(tmp, n * h));

                // ── FFN ──
                if (_nomic)
                {
                    Gemm(l.Gate!, x, gate, n);
                    Gemm(l.Up!, x, up, n);
                    SiLu.Execute(new ReadOnlySpan<float>(gate, n * ff), new Span<float>(gate, n * ff));
                    TensorPrimitives.Multiply(new ReadOnlySpan<float>(gate, n * ff), new ReadOnlySpan<float>(up, n * ff), new Span<float>(up, n * ff));
                }
                else
                {
                    Gemm(l.Up!, x, up, n);
                    AddBias(l.Up!.Bias, up, n, ff);
                    Gelu(up, n * ff);
                }

                Gemm(l.Down!, up, attn, n);          // reuse attn as the FFN output buffer
                AddBias(l.Down!.Bias, attn, n, h);
                TensorPrimitives.Add(new ReadOnlySpan<float>(attn, n * h), new ReadOnlySpan<float>(tmp, n * h), new Span<float>(attn, n * h));
                LayerNorm(attn, l.OutNormW, l.OutNormB, x, n, h);
            }

            var result = UnmanagedTensor.Allocate(new TensorShape(n, h), DType.Float32, deviceId);
            new ReadOnlySpan<float>(x, n * h).CopyTo(new Span<float>((void*)result.DataPointer, n * h));
            return result;
        }
        finally
        {
            Free(x); Free(qkv); Free(q); Free(k); Free(v); Free(attn); Free(tmp); Free(up); Free(gate);
        }
    }

    private void Embed(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, float* x, int h)
    {
        var typeRow = h <= 4096 ? stackalloc float[h] : new float[h];
        if (_typeEmbd != 0)
            Dequantize.ToFloat32(_typeEmbd, h, _typeEmbdQt, typeRow);
        else
            typeRow.Clear();

        var posRow = h <= 4096 ? stackalloc float[h] : new float[h];

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at index {t} is out of range [0, {Config.VocabSize}).");

            var dst = new Span<float>(x + (long)t * h, h);
            long rowBytes = Dequantize.RowByteSize(h, _tokenEmbdQt);
            Dequantize.ToFloat32(_tokenEmbd + (nint)((long)id * rowBytes), h, _tokenEmbdQt, dst);
            TensorPrimitives.Add(dst, typeRow, dst);

            if (!_nomic)
            {
                int pos = positions[t];
                if ((uint)pos >= (uint)_posRows)
                    throw new ArgumentOutOfRangeException(nameof(positions), $"Position {pos} exceeds the position-embedding table ({_posRows} rows).");
                long pRowBytes = Dequantize.RowByteSize(h, _posEmbdQt);
                Dequantize.ToFloat32(_posEmbd + (nint)((long)pos * pRowBytes), h, _posEmbdQt, posRow);
                TensorPrimitives.Add(dst, posRow, dst);
            }
        }

        LayerNorm(x, _embNormW, _embNormB, x, tokenIds.Length, h);
    }

    // ───────────────────────────── ops ─────────────────────────────

    private static void AddBias(float[]? bias, float* data, int rows, int width)
    {
        if (bias is null) return;
        for (int t = 0; t < rows; t++)
        {
            var row = new Span<float>(data + (long)t * width, width);
            TensorPrimitives.Add(row, bias, row);
        }
    }

    /// <summary>
    /// <c>ggml_norm</c> + weight/bias: biased variance, <c>y = (x - mean) / sqrt(var + eps) * w + b</c>,
    /// sums accumulated in double. Safe when <paramref name="src"/> and <paramref name="dst"/> alias.
    /// </summary>
    private void LayerNorm(float* src, float[] w, float[] b, float* dst, int rows, int width)
    {
        for (int t = 0; t < rows; t++)
        {
            float* s = src + (long)t * width;
            float* d = dst + (long)t * width;
            double sum = 0;
            for (int i = 0; i < width; i++) sum += s[i];
            float mean = (float)(sum / width);
            double sq = 0;
            for (int i = 0; i < width; i++)
            {
                float c = s[i] - mean;
                sq += (double)c * c;
            }
            float scale = 1.0f / MathF.Sqrt((float)(sq / width) + _eps);
            for (int i = 0; i < width; i++)
                d[i] = (s[i] - mean) * scale * w[i] + b[i];
        }
    }

    private static void Gelu(float* data, int count)
    {
        if (UseErfGelu)
        {
            for (int i = 0; i < count; i++)
                data[i] = 0.5f * data[i] * (1.0f + (float)Erf(data[i] * 0.70710678118654752440));
            return;
        }

        // ggml_gelu: tanh approximation (GELU_COEF_A = 0.044715, SQRT_2_OVER_PI).
        for (int i = 0; i < count; i++)
        {
            float x = data[i];
            data[i] = 0.5f * x * (1.0f + MathF.Tanh(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));
        }
    }

    /// <summary>erf to ~1e-15 (W. J. Cody-style series / continued fraction split), used by the test seam only.</summary>
    private static double Erf(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 2.5)
        {
            // Maclaurin series: erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1)).
            double term = ax, sum = ax, x2 = ax * ax;
            for (int n = 1; n < 200; n++)
            {
                term *= -x2 / n;
                double add = term / (2 * n + 1);
                sum += add;
                if (Math.Abs(add) < 1e-17) break;
            }
            double r = 1.1283791670955126 * sum;
            return x < 0 ? -r : r;
        }
        // Continued fraction for erfc.
        double f = 0;
        for (int k = 60; k >= 1; k--) f = k / 2.0 / (ax + f);
        double erfc = Math.Exp(-ax * ax) / (1.7724538509055159 * (ax + f));
        return x < 0 ? erfc - 1.0 : 1.0 - erfc;
    }

    private void Gemm(Proj p, float* input, float* output, int n)
    {
        nint weights = p.Weight;
        var qt = p.Qt;
        int m = p.M, k = p.K;
        if (ForceF32Activations && qt is not (QuantizationType.F32 or QuantizationType.F16))
        {
            MatMul.GemmDequantRows((byte*)weights, qt, input, output, m, k, n, pool: _pool);
            return;
        }

        switch (qt)
        {
            case QuantizationType.F32: MatMul.GemmF32((float*)weights, input, output, m, k, n, _pool); return;
            case QuantizationType.F16: MatMul.GemmF16(weights, input, output, m, k, n, _pool); return;
            case QuantizationType.Q8_0: MatMul.GemmQ8_0((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q5_0: MatMul.GemmQ5_0((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q2_K: MatMul.GemmQ2_K((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q3_K: MatMul.GemmQ3_K((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q4_K: MatMul.GemmQ4_K((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q5_K: MatMul.GemmQ5_K((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q6_K: MatMul.GemmQ6_K((byte*)weights, input, output, m, k, n, _pool, null); return;
            case QuantizationType.Q4_0:
            case QuantizationType.Q4_1:
            case QuantizationType.Q5_1:
            case QuantizationType.IQ4_NL:
                MatMul.GemmLegacyQuantOrDequant((byte*)weights, qt, input, output, m, k, n, _pool, null); return;
            default:
                MatMul.GemmDequantRows((byte*)weights, qt, input, output, m, k, n, pool: _pool); return;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float* Alloc(long floats) => (float*)NativeMemory.AlignedAlloc((nuint)(floats * sizeof(float)), 64);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Free(float* p)
    {
        if (p != null) NativeMemory.AlignedFree(p);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _pool?.Dispose();
    }
}
