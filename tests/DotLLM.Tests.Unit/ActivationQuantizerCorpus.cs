namespace DotLLM.Tests.Unit;

/// <summary>Shared exact-tie / near-tie corpus for the activation-quantizer rounding tests (#1340). Pure data, no device dependency.</summary>
internal static class ActivationQuantizerCorpus
{
    private const int Group = 32;

    /// <summary>Builds the corpus: [blocks * 32] floats.</summary>
    internal static float[] BuildCorpus(out string[] labels)
    {
        var data = new List<float>();
        var lab = new List<string>();
        void Block(string name, Func<int, float> f)
        {
            lab.Add(name);
            for (int i = 0; i < Group; i++) data.Add(f(i));
        }

        // Exact ties: d = 2^e exactly (amax = 127 * 2^e), so 1/d is exact and x * (1/d) is exactly k + 0.5.
        foreach (int e in new[] { -10, -6, -3, 0, 2, 5 })
        {
            float d = MathF.Pow(2f, e);
            for (int sign = 0; sign < 2; sign++)
            {
                float s = sign == 0 ? 1f : -1f;
                for (int k0 = 0; k0 < 126; k0 += 31)
                {
                    int kk = k0;
                    Block($"tie e={e} s={s} k0={kk}", i =>
                    {
                        if (i == 0) return s * 127f * d;                    // amax element, pins d = 2^e
                        int k = (kk + i - 1) % 126;                          // 0..125  -> (k + .5) in 0.5 .. 125.5
                        return s * (k + 0.5f) * d;
                    });
                }
            }
            // alternate signs inside a block
            Block($"tie mixed e={e}", i => i == 0 ? 127f * d : ((i & 1) == 0 ? 1f : -1f) * (i + 0.5f) * d);
        }

        // Degenerate blocks.
        Block("all zero", _ => 0f);
        Block("all -0.0", _ => -0.0f);
        Block("amax tiny normal", i => i == 0 ? 1e-30f : 0.4e-30f * (i % 3));
        Block("one element", i => i == 7 ? -3.25f : 0f);

        // Generic scales: values on and one ULP either side of every k + 0.5 boundary for an awkward d.
        var rng = new Random(1340);
        for (int b = 0; b < 6000; b++)
        {
            float amax = (float)(Math.Exp(rng.NextDouble() * 12 - 6));    // 0.0025 .. 400
            float d = amax / 127f;
            float id = 1f / d;
            var vals = new float[Group];
            vals[0] = amax * (rng.Next(2) == 0 ? 1f : -1f);
            for (int i = 1; i < Group; i++)
            {
                int k = rng.Next(0, 126);
                float boundary = (k + 0.5f) / id;
                int mode = rng.Next(4);
                float v = boundary;
                if (mode == 1) v = MathF.BitIncrement(boundary);
                else if (mode == 2) v = MathF.BitDecrement(boundary);
                else if (mode == 3) v = (float)(rng.NextDouble() * 2 - 1) * amax;
                if (MathF.Abs(v) > amax) v = amax;
                vals[i] = rng.Next(2) == 0 ? v : -v;
            }
            Block($"generic b={b}", i => vals[i]);
        }

        labels = lab.ToArray();
        return data.ToArray();
    }
}
