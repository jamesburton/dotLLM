using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Scalar / TensorPrimitives building blocks of the Inkling 2 CPU reference forward (issue #2026). Every function mirrors one llama.cpp
/// graph fragment of <c>src/models/inkling2.cpp</c> (see <c>docs/INKLING2.md</c>) and is deliberately free of model state so a unit test
/// can pin it against a hand-written definition.
/// </summary>
public static class Inkling2Ops
{
    /// <summary>Epsilon of the gamma-free RMS norms inside depth mixing and the loop controllers (llama.cpp hard-codes <c>1e-6f</c>).</summary>
    public const float MixEps = 1e-6f;

    /// <summary>Logistic function.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>
    /// <c>y = x / sqrt(mean(x^2) + eps) * gamma</c> (gamma optional). The sum of squares runs in double like ggml's <c>ggml_float</c>.
    /// <paramref name="y"/> may alias <paramref name="x"/>.
    /// </summary>
    /// <param name="x">Row.</param>
    /// <param name="gamma">Per-channel gain, or empty for none.</param>
    /// <param name="eps">Epsilon added to the mean square.</param>
    /// <param name="y">Destination, same length as <paramref name="x"/>.</param>
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, float eps, Span<float> y)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++) sum += (double)(x[i] * x[i]);
        float scale = 1f / MathF.Sqrt((float)(sum / x.Length) + eps);
        if (gamma.IsEmpty)
        {
            for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale;
        }
        else
        {
            for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale * gamma[i];
        }
    }

    /// <summary>
    /// The L2 clip of Q, K and the relative-position projection (llama.cpp <c>graph::clip</c>): multiply by the optional per-channel
    /// <paramref name="scale"/>, then scale the row down to an L2 norm of at most <paramref name="threshold"/>, i.e.
    /// <c>x * t / max(|x|, t)</c> (computed as <c>l2_norm(x, eps = t)</c> followed by <c>* t</c>).
    /// </summary>
    /// <param name="x">Row, modified in place.</param>
    /// <param name="scale">Per-channel scale (same length), or empty.</param>
    /// <param name="threshold">Maximum L2 norm.</param>
    public static void ClipL2(Span<float> x, ReadOnlySpan<float> scale, float threshold)
    {
        if (!scale.IsEmpty)
            for (int i = 0; i < x.Length; i++) x[i] *= scale[i];
        double sum = 0;
        for (int i = 0; i < x.Length; i++) sum += (double)(x[i] * x[i]);
        float s = 1f / MathF.Max(MathF.Sqrt((float)sum), threshold);
        for (int i = 0; i < x.Length; i++) x[i] = x[i] * s * threshold;
    }

    /// <summary>
    /// SITU, the MoE activation: <c>beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb)</c>; the up softcap is skipped when
    /// <paramref name="linearBeta"/> is not positive (llama.cpp <c>LLM_FFN_SITU</c> in <c>build_moe_ffn</c>).
    /// </summary>
    /// <param name="gate">Gate projection.</param>
    /// <param name="up">Up projection.</param>
    /// <param name="beta">Gate softcap.</param>
    /// <param name="linearBeta">Up softcap (0 = none).</param>
    /// <returns>The activated value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Situ(float gate, float up, float beta, float linearBeta)
    {
        float act = MathF.Tanh(gate * (1f / beta)) * beta;
        act *= Sigmoid(gate);
        if (linearBeta > 0f) up = MathF.Tanh(up * (1f / linearBeta)) * linearBeta;
        return act * up;
    }

    /// <summary>
    /// Depth ("attention-residual") mixing of one token (llama.cpp <c>graph::mix</c>). The states are the RMS-normalised earlier block
    /// outputs plus the RMS-normalised current state (all gamma-free, eps <see cref="MixEps"/>), <paramref name="states"/> holding
    /// <c>count</c> rows of <c>n</c>; per query, <c>w = softmax(states . q / sqrt(n))</c> and the result is <c>sum_i w_i * state_i</c>
    /// (the mix is over the NORMALISED states, so the output has no residual-scale of its own). The state order is irrelevant.
    /// </summary>
    /// <param name="states">Normalised state rows, <c>count * n</c>.</param>
    /// <param name="count">Number of states.</param>
    /// <param name="queries"><c>nq</c> query rows of <c>n</c>.</param>
    /// <param name="nq">Number of queries.</param>
    /// <param name="n">Row width.</param>
    /// <param name="output"><c>nq</c> rows of <c>n</c>.</param>
    public static void DepthMixToken(ReadOnlySpan<float> states, int count, ReadOnlySpan<float> queries, int nq, int n, Span<float> output)
    {
        Span<float> w = stackalloc float[count];
        float scale = 1f / MathF.Sqrt(n);
        for (int q = 0; q < nq; q++)
        {
            var qv = queries.Slice(q * n, n);
            float max = float.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                float s = TensorPrimitives.Dot(states.Slice(i * n, n), qv) * scale;
                w[i] = s;
                if (s > max) max = s;
            }
            float sum = 0f;
            for (int i = 0; i < count; i++) { w[i] = MathF.Exp(w[i] - max); sum += w[i]; }
            float inv = 1f / sum;
            var o = output.Slice(q * n, n);
            o.Clear();
            for (int i = 0; i < count; i++)
                TensorPrimitives.MultiplyAdd(states.Slice(i * n, n), w[i] * inv, o, o);
        }
    }

    /// <summary>
    /// Loop-controller combine (llama.cpp <c>graph::control</c>) after the rank-r gate has been computed:
    /// <c>out = (cur - tanh(forget_pre) * cur) + tanh(gain_pre) * message</c>.
    /// </summary>
    /// <param name="cur">Current input row.</param>
    /// <param name="message">Cross-pass message row.</param>
    /// <param name="gainPre">Gain projection before tanh.</param>
    /// <param name="forgetPre">Forget projection before tanh.</param>
    /// <param name="output">Destination (may alias <paramref name="cur"/>).</param>
    public static void ControlCombine(ReadOnlySpan<float> cur, ReadOnlySpan<float> message, ReadOnlySpan<float> gainPre, ReadOnlySpan<float> forgetPre, Span<float> output)
    {
        for (int i = 0; i < cur.Length; i++)
        {
            float c = cur[i];
            float forget = MathF.Tanh(forgetPre[i]);
            float gain = MathF.Tanh(gainPre[i]);
            output[i] = (c - forget * c) + gain * message[i];
        }
    }

    /// <summary>
    /// The attention-score bias of one (token, head): <c>bias[d] . r</c> for key distance <c>d &lt; extent</c>, zero beyond (the zero pad row of
    /// <c>llama_kv_cache::set_input_relative_pos</c> / <c>ggml_pad(attn_rel_bias)</c>).
    /// </summary>
    /// <param name="relBias">Table <c>[extent][relLen]</c>.</param>
    /// <param name="extent">Rows of the table.</param>
    /// <param name="relLen">Width of a row.</param>
    /// <param name="r">Relative-position vector of the head (<c>relLen</c>).</param>
    /// <param name="distance">Key distance <c>pos_query - pos_key</c> (&gt;= 0).</param>
    /// <returns>The additive score bias.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float RelativeBias(ReadOnlySpan<float> relBias, int extent, int relLen, ReadOnlySpan<float> r, int distance)
        => (uint)distance < (uint)extent ? TensorPrimitives.Dot(relBias.Slice(distance * relLen, relLen), r) : 0f;

    /// <summary>Log-scaling factor of the global layers: <c>1 + alpha * ln(max(1, (pos + 1) / floor))</c>.</summary>
    /// <param name="pos">Token position.</param>
    /// <param name="floor">Log-scale floor.</param>
    /// <param name="alpha">Log-scale alpha.</param>
    /// <returns>Multiplier of Q and the relative projection.</returns>
    public static float LogScale(int pos, int floor, float alpha)
        => alpha * MathF.Log(MathF.Max(1f, (pos + 1f) * (1f / floor))) + 1f;
}
