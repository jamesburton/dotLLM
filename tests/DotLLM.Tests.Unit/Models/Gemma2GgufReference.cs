using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Independent, deliberately naive double-precision reference forward for the
/// <see cref="SyntheticGemma2Gguf"/> fixtures (llama.cpp <c>gemma</c> / <c>gemma2</c> graph
/// semantics). Shares NO code with the engine, so agreement is evidence, not tautology. Every
/// Gemma-2 feature is an explicit switch so tests can ablate it and prove the fixture is
/// sensitive to it (a control arm that cannot disagree proves nothing).
/// </summary>
internal static class Gemma2GgufReference
{
    /// <summary>Which features the reference applies. Defaults = the real model.</summary>
    internal sealed record Options
    {
        public bool AttnSoftcap { get; init; } = true;
        public bool FinalSoftcap { get; init; } = true;
        public bool SlidingWindow { get; init; } = true;
        public bool PostNorms { get; init; } = true;
        public bool QueryPreAttnScalarFromHidden { get; init; } = true;
        public bool EmbedScale { get; init; } = true;
        /// <summary>Gemma-3: per-head Q/K RMSNorm before RoPE.</summary>
        public bool QkNorm { get; init; } = true;
        /// <summary>Gemma-3: windowed layers use rope.freq_base_swa; false = every layer uses the global base.</summary>
        public bool DualRope { get; init; } = true;
        /// <summary>Gemma-3: linear RoPE scaling on the global layers only; false = unscaled.</summary>
        public bool LinearScale { get; init; } = true;
        /// <summary>Gemma-3: window pattern N (every Nth layer global); false = treat as Gemma-2's even/odd alternation.</summary>
        public bool Pattern6 { get; init; } = true;
    }

    /// <summary>Last-token logits for <paramref name="tokens"/> (positions 0..n-1).</summary>
    internal static double[] LastTokenLogits(SyntheticGemma2Gguf.Weights wts, int[] tokens, Options? opt = null)
    {
        opt ??= new Options();
        var c = wts.Config;
        bool g3 = c.Arch == "gemma3";
        bool g2 = c.Arch == "gemma2" || g3;           // four-norm layout
        bool caps = c.Arch == "gemma2";                // soft-caps are Gemma-2 only
        int H = c.Hidden, hd = c.HeadDim, nh = c.Heads, nkv = c.KvHeads, ff = c.FeedForward, T = tokens.Length;

        // llama.cpp: 27B (46 layers) scales by 1/sqrt(n_embd/n_head), everything else by 1/sqrt(head_dim).
        double qpas = g2 && opt.QueryPreAttnScalarFromHidden && c.Layers == (g3 ? 62 : 46) ? (double)(H / nh) : hd;
        double scale = 1.0 / Math.Sqrt(qpas);

        var x = new double[T][];
        for (int t = 0; t < T; t++)
        {
            x[t] = new double[H];
            for (int i = 0; i < H; i++)
                x[t][i] = wts.TokenEmbd[tokens[t] * H + i] * (opt.EmbedScale ? Math.Sqrt(H) : 1.0);
        }

        for (int l = 0; l < c.Layers; l++)
        {
            var L = wts.Layers[l];
            // ── attention ──
            var q = new double[T][]; var k = new double[T][]; var v = new double[T][];
            for (int t = 0; t < T; t++)
            {
                var h = Rms(x[t], L.AttnNorm, c.NormEps);
                q[t] = MatVec(L.Q, h, nh * hd);
                k[t] = MatVec(L.K, h, nkv * hd);
                v[t] = MatVec(L.V, h, nkv * hd);
                if (g3 && opt.QkNorm)
                {
                    for (int hh = 0; hh < nh; hh++) NormHead(q[t], hh * hd, hd, L.QNorm!, c.NormEps);
                    for (int hh = 0; hh < nkv; hh++) NormHead(k[t], hh * hd, hd, L.KNorm!, c.NormEps);
                }
                // Gemma-3: global layers (every Nth) use rope.freq_base with linear scale 1/factor; windowed layers
                // use rope.freq_base_swa at scale 1. Gemma 1/2 use theta 10000 everywhere.
                int pat = c.SlidingPattern > 0 ? c.SlidingPattern : 6;
                bool globalLayer = g3 && (opt.Pattern6 ? (l % pat) == pat - 1 : l % 2 != 0);
                double theta = !g3 ? 10000.0
                    : (globalLayer || !opt.DualRope) ? c.RopeBase : (c.RopeBaseSwa > 0 ? c.RopeBaseSwa : 10000.0);
                double lin = g3 && globalLayer && opt.LinearScale && c.RopeLinearFactor > 0 ? 1.0 / c.RopeLinearFactor : 1.0;
                for (int hh = 0; hh < nh; hh++) Rope(q[t], hh * hd, hd, t, theta, lin);
                for (int hh = 0; hh < nkv; hh++) Rope(k[t], hh * hd, hd, t, theta, lin);
            }
            // gemma2: even layers sliding (llama.cpp set_swa_pattern(2)); gemma3: il % N < N-1 sliding
            int npat = c.SlidingPattern > 0 ? c.SlidingPattern : 6;
            bool windowed = g3
                ? opt.SlidingWindow && c.SlidingWindow > 0 && (opt.Pattern6 ? (l % npat) < npat - 1 : l % 2 == 0)
                : g2 && opt.SlidingWindow && c.SlidingWindow > 0 && l % 2 == 0;
            for (int t = 0; t < T; t++)
            {
                var attn = new double[nh * hd];
                for (int hh = 0; hh < nh; hh++)
                {
                    int kvh = hh / (nh / nkv);
                    int lo = windowed ? Math.Max(0, t - c.SlidingWindow + 1) : 0;
                    var sc = new double[t - lo + 1];
                    for (int j = lo; j <= t; j++)
                    {
                        double s = 0;
                        for (int d = 0; d < hd; d++) s += q[t][hh * hd + d] * k[j][kvh * hd + d];
                        s *= scale;
                        if (caps && opt.AttnSoftcap && c.AttnSoftcap > 0) s = c.AttnSoftcap * Math.Tanh(s / c.AttnSoftcap);
                        sc[j - lo] = s;
                    }
                    double mx = sc.Max(), sum = 0;
                    for (int i = 0; i < sc.Length; i++) { sc[i] = Math.Exp(sc[i] - mx); sum += sc[i]; }
                    for (int j = lo; j <= t; j++)
                        for (int d = 0; d < hd; d++)
                            attn[hh * hd + d] += sc[j - lo] / sum * v[j][kvh * hd + d];
                }
                var o = MatVec(L.O, attn, H);
                if (g2 && opt.PostNorms) o = Rms(o, L.PostAttnNorm!, c.NormEps);
                for (int i = 0; i < H; i++) x[t][i] += o[i];
            }
            // ── FFN ──
            for (int t = 0; t < T; t++)
            {
                var h = Rms(x[t], L.FfnNorm, c.NormEps);
                var g = MatVec(L.Gate, h, ff); var u = MatVec(L.Up, h, ff);
                var act = new double[ff];
                for (int i = 0; i < ff; i++) act[i] = GeluTanh(g[i]) * u[i];
                var d = MatVec(L.Down, act, H);
                if (g2 && opt.PostNorms) d = Rms(d, L.PostFfnNorm!, c.NormEps);
                for (int i = 0; i < H; i++) x[t][i] += d[i];
            }
        }

        var fin = Rms(x[T - 1], wts.OutputNorm, c.NormEps);
        var logits = new double[c.Vocab];
        for (int vtok = 0; vtok < c.Vocab; vtok++)
        {
            double s = 0;
            for (int i = 0; i < H; i++) s += wts.TokenEmbd[vtok * H + i] * fin[i];
            if (caps && opt.FinalSoftcap && c.FinalSoftcap > 0) s = c.FinalSoftcap * Math.Tanh(s / c.FinalSoftcap);
            logits[vtok] = s;
        }
        return logits;
    }

    private static double[] Rms(double[] x, float[] w, float eps)
    {
        double ss = 0;
        foreach (double d in x) ss += d * d;
        double inv = 1.0 / Math.Sqrt(ss / x.Length + eps);
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = x[i] * inv * w[i];   // baked (1+w): plain gain
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

    /// <summary>Per-head RMSNorm (gain baked as 1+w), in place.</summary>
    private static void NormHead(double[] v, int off, int hd, float[] w, float eps)
    {
        double ss = 0;
        for (int i = 0; i < hd; i++) ss += v[off + i] * v[off + i];
        double inv = 1.0 / Math.Sqrt(ss / hd + eps);
        for (int i = 0; i < hd; i++) v[off + i] = v[off + i] * inv * w[i];
    }

    /// <summary>NeoX (rotate_half) RoPE over the full head, in place. angle = pos * linScale * theta^(-2i/hd).</summary>
    private static void Rope(double[] v, int off, int hd, int pos, double theta = 10000.0, double linScale = 1.0)
    {
        int half = hd / 2;
        for (int i = 0; i < half; i++)
        {
            double ang = pos * linScale * Math.Pow(theta, -2.0 * i / hd);
            double cs = Math.Cos(ang), sn = Math.Sin(ang);
            double a = v[off + i], b = v[off + i + half];
            v[off + i] = a * cs - b * sn;
            v[off + i + half] = b * cs + a * sn;
        }
    }

    private static double GeluTanh(double x) =>
        0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}
