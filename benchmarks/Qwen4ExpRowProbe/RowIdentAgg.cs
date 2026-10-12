using System.Text;

/// <summary>Aggregates, per verify-row index, how often a candidate logit row reproduces the plain 1-row reference (#820 greedy identity).</summary>
internal sealed class RowIdentAgg(int rows)
{
    private sealed class Cell
    {
        public int N, Bit, Arg;
        public double SumMax;
        public double SumKl;   // KL(reference || candidate) over the full vocabulary
        public double MaxMax;
        public int TieFlips;   // rows where the candidate's argmax differs AND the reference gap between the two picks is < 1.0
    }

    // [arm][row]
    private readonly Cell[][] _c = [Enumerable.Range(0, rows).Select(_ => new Cell()).ToArray(), Enumerable.Range(0, rows).Select(_ => new Cell()).ToArray()];

    public void Add(int arm, int row, float[] reference, float[] candidate)
    {
        var c = _c[arm][row];
        c.N++;
        bool bit = true;
        double mx = 0;
        int ra = 0, ca = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (reference[i] != candidate[i]) bit = false;
            double d = Math.Abs((double)reference[i] - candidate[i]);
            if (d > mx) mx = d;
            if (reference[i] > reference[ra]) ra = i;
            if (candidate[i] > candidate[ca]) ca = i;
        }
        {
            double mr = reference[ra], mc = candidate[ca], zr = 0, zc = 0;
            for (int i = 0; i < reference.Length; i++) { zr += Math.Exp(reference[i] - mr); zc += Math.Exp(candidate[i] - mc); }
            double lr = Math.Log(zr) + mr, lc = Math.Log(zc) + mc, kl = 0;
            for (int i = 0; i < reference.Length; i++) kl += Math.Exp(reference[i] - lr) * ((reference[i] - lr) - (candidate[i] - lc));
            c.SumKl += kl;
        }
        if (bit) c.Bit++;
        if (ra == ca) c.Arg++;
        else c.TieFlips++;
        c.SumMax += mx;
        if (mx > c.MaxMax) c.MaxMax = mx;
    }

    public string Report()
    {
        var sb = new StringBuilder();
        string[] names = ["A: 1-row replay on the same state", "B: R-row verify forward on the same state"];
        for (int arm = 0; arm < 2; arm++)
        {
            sb.AppendLine(names[arm] + " vs plain 1-row decode reference");
            for (int j = 0; j < rows; j++)
            {
                var c = _c[arm][j];
                if (c.N == 0) continue;
                sb.AppendLine($"  row {j}: n={c.N} bit-identical {c.Bit}/{c.N} ({100.0 * c.Bit / c.N:F0}%), argmax equal {c.Arg}/{c.N} ({100.0 * c.Arg / c.N:F1}%), mean max|dlogit| {c.SumMax / c.N:F4}, worst {c.MaxMax:F3}, mean KL {c.SumKl / c.N:E2}");
            }
        }
        return sb.ToString();
    }
}
