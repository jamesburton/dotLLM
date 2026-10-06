using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Independent, deliberately naive double-precision reference forward for <see cref="SyntheticGraniteGguf"/>
/// (llama.cpp <c>granite.cpp</c> / <c>granite-moe.cpp</c> graph semantics). Shares no code with the engine. Every
/// Granite scalar is an explicit switch so tests can ablate it and prove the fixture is sensitive to it.
/// </summary>
internal static class GraniteGgufReference
{
    internal sealed record Options
    {
        public bool EmbeddingScale { get; init; } = true;
        public bool AttentionScale { get; init; } = true;
        public bool ResidualScale { get; init; } = true;
        public bool LogitScale { get; init; } = true;
    }

    /// <summary>Last-token logits for <paramref name="tokens"/> (positions 0..n-1).</summary>
    internal static double[] LastTokenLogits(SyntheticGraniteGguf.Weights wts, int[] tokens, Options? opt = null)
    {
        opt ??= new Options();
        var c = wts.Config;
        int H = c.Hidden, hd = c.HeadDim, nh = c.Heads, nkv = c.KvHeads, T = tokens.Length;
        double attnScale = opt.AttentionScale && c.AttentionScale > 0 ? c.AttentionScale : 1.0 / Math.Sqrt(hd);
        double resScale = opt.ResidualScale && c.ResidualScale > 0 ? c.ResidualScale : 1.0;
        double embScale = opt.EmbeddingScale && c.EmbeddingScale > 0 ? c.EmbeddingScale : 1.0;
        double logitDiv = opt.LogitScale && c.LogitScale > 0 ? c.LogitScale : 1.0;

        var x = new double[T][];
        for (int t = 0; t < T; t++)
        {
            x[t] = new double[H];
            for (int i = 0; i < H; i++) x[t][i] = wts.TokenEmbd[tokens[t] * H + i] * embScale;
        }

        for (int l = 0; l < c.Layers; l++)
        {
            var L = wts.Layers[l];
            var q = new double[T][]; var k = new double[T][]; var v = new double[T][];
            for (int t = 0; t < T; t++)
            {
                var h = Rms(x[t], L.AttnNorm, c.NormEps);
                q[t] = MatVec(L.Q, h, nh * hd);
                k[t] = MatVec(L.K, h, nkv * hd);
                v[t] = MatVec(L.V, h, nkv * hd);
                for (int hh = 0; hh < nh; hh++) RopeNorm(q[t], hh * hd, hd, t, c.RopeBase);
                for (int hh = 0; hh < nkv; hh++) RopeNorm(k[t], hh * hd, hd, t, c.RopeBase);
            }
            for (int t = 0; t < T; t++)
            {
                var attn = new double[nh * hd];
                for (int hh = 0; hh < nh; hh++)
                {
                    int kvh = hh / (nh / nkv);
                    var sc = new double[t + 1];
                    for (int j = 0; j <= t; j++)
                    {
                        double s = 0;
                        for (int d = 0; d < hd; d++) s += q[t][hh * hd + d] * k[j][kvh * hd + d];
                        sc[j] = s * attnScale;
                    }
                    double mx = sc.Max(), sum = 0;
                    for (int i = 0; i < sc.Length; i++) { sc[i] = Math.Exp(sc[i] - mx); sum += sc[i]; }
                    for (int j = 0; j <= t; j++)
                        for (int d = 0; d < hd; d++)
                            attn[hh * hd + d] += sc[j] / sum * v[j][kvh * hd + d];
                }
                var o = MatVec(L.O, attn, H);
                for (int i = 0; i < H; i++) x[t][i] += resScale * o[i];
            }
            for (int t = 0; t < T; t++)
            {
                var h = Rms(x[t], L.FfnNorm, c.NormEps);
                double[] ffn = c.IsMoe ? Moe(L, c, h) : Dense(L.Gate!, L.Up!, L.Down!, h, c.FeedForward, H);
                for (int i = 0; i < H; i++) x[t][i] += resScale * ffn[i];
            }
        }

        var fin = Rms(x[T - 1], wts.OutputNorm, c.NormEps);
        var logits = new double[c.Vocab];
        for (int vtok = 0; vtok < c.Vocab; vtok++)
        {
            double s = 0;
            for (int i = 0; i < H; i++) s += wts.Output[vtok * H + i] * fin[i];
            logits[vtok] = s / logitDiv;
        }
        return logits;
    }

    private static double[] Dense(float[] gate, float[] up, float[] down, double[] h, int ff, int H)
    {
        var g = MatVec(gate, h, ff); var u = MatVec(up, h, ff);
        var act = new double[ff];
        for (int i = 0; i < ff; i++) act[i] = g[i] / (1.0 + Math.Exp(-g[i])) * u[i];
        return MatVec(down, act, H);
    }

    /// <summary>llama.cpp granite-moe: softmax over ALL experts, top-k, renormalise, SwiGLU experts, weighted sum.</summary>
    private static double[] Moe(SyntheticGraniteGguf.Layer L, SyntheticGraniteConfig c, double[] h)
    {
        int E = c.Experts, ff = c.FeedForward, H = c.Hidden;
        var logits = MatVec(L.Router!, h, E);
        double mx = logits.Max(), sum = 0;
        var p = new double[E];
        for (int e = 0; e < E; e++) { p[e] = Math.Exp(logits[e] - mx); sum += p[e]; }
        for (int e = 0; e < E; e++) p[e] /= sum;
        var order = Enumerable.Range(0, E).OrderByDescending(e => p[e]).Take(c.ExpertsUsed).ToArray();
        double norm = order.Sum(e => p[e]);
        var acc = new double[H];
        foreach (int e in order)
        {
            var gate = L.GateExps!.AsSpan(e * ff * H, ff * H).ToArray();
            var up = L.UpExps!.AsSpan(e * ff * H, ff * H).ToArray();
            var down = L.DownExps!.AsSpan(e * H * ff, H * ff).ToArray();
            var y = Dense(gate, up, down, h, ff, H);
            for (int i = 0; i < H; i++) acc[i] += p[e] / norm * y[i];
        }
        return acc;
    }

    private static double[] Rms(double[] x, float[] w, float eps)
    {
        double ss = 0;
        foreach (double d in x) ss += d * d;
        double inv = 1.0 / Math.Sqrt(ss / x.Length + eps);
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = x[i] * inv * w[i];
        return r;
    }

    private static double[] MatVec(float[] m, double[] x, int rows)
    {
        int cols = x.Length;
        var r = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double s = 0;
            for (int j = 0; j < cols; j++) s += m[(long)i * cols + j] * x[j];
            r[i] = s;
        }
        return r;
    }

    /// <summary>Adjacent-pair ("Norm") RoPE: the layout llama.cpp's converter produces for granite (it permutes Q/K).</summary>
    private static void RopeNorm(double[] v, int off, int hd, int pos, double theta)
    {
        for (int i = 0; i < hd / 2; i++)
        {
            double ang = pos * Math.Pow(theta, -2.0 * i / hd);
            double cs = Math.Cos(ang), sn = Math.Sin(ang);
            double a = v[off + 2 * i], b = v[off + 2 * i + 1];
            v[off + 2 * i] = a * cs - b * sn;
            v[off + 2 * i + 1] = a * sn + b * cs;
        }
    }
}
