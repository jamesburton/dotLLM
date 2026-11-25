using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Edge0;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// Independent float64 oracle of one Edge0 latent-MoE layer, written from the strata <c>latent_moe_reference</c> pseudo-code and
/// sharing no code with <see cref="Edge0LatentMoe"/>: it dequantizes every weight it touches to F32 (a test-only expansion) and
/// uses F32 activations, so a mutant in the production path cannot also be in the oracle.
/// </summary>
internal static unsafe class Edge0LatentMoeReference
{
    public static float[] Dequant(nint p, long count, QuantizationType qt)
    {
        var r = new float[count];
        Dequantize.ToFloat32(p, count, qt, r);
        return r;
    }

    /// <summary>Routed part only (no shared expert), one token. <paramref name="x"/> is <c>[hidden]</c>; <paramref name="labels"/> and
    /// <paramref name="colScale"/> (<c>s</c>, null = none) are the caller's ground truth, not read back from the pack reader.</summary>
    public static float[] Routed(Edge0Pack pack, int layer, float[] router, int topK, float[] x, int[] labels, float[]? colScale,
        out int[] ids, out double[] weights)
    {
        int H = pack.Hidden, L = pack.Latent, F = pack.ExpertFf, E = pack.Experts;
        // router: softmax over all experts, stable descending sort, top-k, renormalise
        var logit = new double[E];
        for (int e = 0; e < E; e++) { double s = 0; for (int i = 0; i < H; i++) s += (double)router[e * H + i] * x[i]; logit[e] = s; }
        double mx = logit.Max();
        var p = logit.Select(v => Math.Exp(v - mx)).ToArray();
        double sum = p.Sum();
        for (int e = 0; e < E; e++) p[e] /= sum;
        ids = Enumerable.Range(0, E).OrderByDescending(e => p[e]).ThenBy(e => e).Take(topK).ToArray();
        double ts = ids.Sum(e => p[e]);
        weights = ids.Select(e => p[e] / ts).ToArray();

        var xl = new Dictionary<int, double[]>();
        double[] Latent(int g)
        {
            if (xl.TryGetValue(g, out var cached)) return cached;
            var D = Dequant(pack.D(layer, g), (long)L * H, pack.LatentDenseType);
            var v = new double[L];
            for (int r = 0; r < L; r++)
            {
                double s = 0;
                for (int i = 0; i < H; i++) s += (double)D[r * H + i] * x[i];
                v[r] = colScale is null ? s : s * (1.0 / colScale[r]);   // the pack stores s; xl is multiplied by 1/s
            }
            return xl[g] = v;
        }

        var z = new Dictionary<int, double[]>();
        for (int j = 0; j < topK; j++)
        {
            int e = ids[j], g = labels[e];
            var xlg = Latent(g);
            var gate = Dequant(pack.GateBank(layer) + (nint)(e * pack.ExpertRecordBytes), (long)F * L, pack.GateUpType);
            var up = Dequant(pack.UpBank(layer) + (nint)(e * pack.ExpertRecordBytes), (long)F * L, pack.GateUpType);
            var down = Dequant(pack.DownBank(layer) + (nint)(e * pack.ExpertRecordBytes), (long)L * F, pack.DownType);
            var h = new double[F];
            for (int r = 0; r < F; r++)
            {
                double a = 0, b = 0;
                for (int i = 0; i < L; i++) { a += (double)gate[r * L + i] * xlg[i]; b += (double)up[r * L + i] * xlg[i]; }
                h[r] = a / (1 + Math.Exp(-a)) * b;
            }
            if (!z.TryGetValue(g, out var zg)) z[g] = zg = new double[L];
            for (int r = 0; r < L; r++)
            {
                double s = 0;
                for (int i = 0; i < F; i++) s += (double)down[r * F + i] * h[i];
                zg[r] += weights[j] * s;
            }
        }

        var outv = new double[H];
        foreach (var (g, zg) in z)
        {
            var U = Dequant(pack.U(layer, g), (long)H * L, pack.LatentDenseType);
            for (int r = 0; r < H; r++)
            {
                double s = 0;
                for (int i = 0; i < L; i++) s += (double)U[r * L + i] * zg[i];
                outv[r] += s;
            }
        }
        return outv.Select(v => (float)v).ToArray();
    }
}
