namespace DotLLM.Tests.Unit;

/// <summary>
/// Shared exact-tie / near-tie corpus for the block-quantizer rounding tests (#1340, #1499, #1500). Pure data, no device
/// dependency. <c>qmax</c> is the integer the block amax maps to: 127 for Q8_0 / Q8_1, 7 for Q4_0.
/// </summary>
internal static class ActivationQuantizerCorpus
{
    private const int Group = 32;

    /// <summary>Builds the Q8 (qmax = 127) corpus: [blocks * 32] floats.</summary>
    internal static float[] BuildCorpus(out string[] labels) => BuildCorpus(127, out labels);

    /// <summary>
    /// Builds the corpus for a quantizer whose scale is <c>d = amax / qmax</c>: exact ties (power-of-two <c>d</c>, so
    /// <c>x * (1/d) == k + 0.5</c> exactly), degenerate blocks, and 6000 generic-scale blocks with values on and one ULP
    /// either side of every <c>k + 0.5</c> boundary.
    /// </summary>
    internal static float[] BuildCorpus(int qmax, out string[] labels)
    {
        var data = new List<float>();
        var lab = new List<string>();
        void Block(string name, Func<int, float> f)
        {
            lab.Add(name);
            for (int i = 0; i < Group; i++) data.Add(f(i));
        }
        int kmax = qmax - 1;          // largest k with k + 0.5 < qmax

        // Exact ties: d = 2^e exactly (amax = qmax * 2^e), so 1/d is exact and x * (1/d) is exactly k + 0.5.
        foreach (int e in new[] { -10, -6, -3, 0, 2, 5 })
        {
            float d = MathF.Pow(2f, e);
            for (int sign = 0; sign < 2; sign++)
            {
                float s = sign == 0 ? 1f : -1f;
                for (int k0 = 0; k0 < kmax; k0 += Math.Max(1, kmax / 4))
                {
                    int kk = k0;
                    Block($"tie e={e} s={s} k0={kk}", i =>
                    {
                        if (i == 0) return s * qmax * d;                     // amax element, pins d = 2^e
                        int k = (kk + i - 1) % kmax;                         // 0..kmax-1 -> (k + .5) in 0.5 .. qmax-0.5
                        return s * (k + 0.5f) * d;
                    });
                }
            }
            // alternate signs inside a block
            Block($"tie mixed e={e}", i => i == 0 ? qmax * d : ((i & 1) == 0 ? 1f : -1f) * (((i - 1) % kmax) + 0.5f) * d);
        }

        // Degenerate blocks.
        Block("all zero", _ => 0f);
        Block("all -0.0", _ => -0.0f);
        Block("amax tiny normal", i => i == 0 ? 1e-30f : 0.4e-30f * (i % 3));
        Block("one element", i => i == 7 ? -3.25f : 0f);

        // Generic scales: values on and one ULP either side of every k + 0.5 boundary for an awkward d.
        var rng = new Random(1340 + qmax);
        for (int b = 0; b < 6000; b++)
        {
            float amax = (float)(Math.Exp(rng.NextDouble() * 12 - 6));    // 0.0025 .. 400
            float d = amax / qmax;
            float id = 1f / d;
            var vals = new float[Group];
            vals[0] = amax * (rng.Next(2) == 0 ? 1f : -1f);
            for (int i = 1; i < Group; i++)
            {
                int k = rng.Next(0, kmax);
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

    /// <summary>
    /// Half-representable corpus for the FP16-input CUDA quantizers: the exact-tie blocks survive the rounding to Half
    /// unchanged (<c>qmax * 2^e</c> and <c>(k + 0.5) * 2^e</c> need at most 8 significant bits), the generic blocks are
    /// re-placed on and around the boundaries of the Half-rounded amax so near-ties stay near ties, and the
    /// "amax tiny normal" block (below Half's range) is dropped. Returns the float values that the Half payload decodes to,
    /// so a float reference run on it is the oracle for the device.
    /// </summary>
    internal static float[] BuildHalfCorpus(int qmax, out string[] labels)
    {
        var data = new List<float>();
        var lab = new List<string>();
        void Block(string name, Func<int, float> f)
        {
            lab.Add(name);
            for (int i = 0; i < Group; i++) data.Add((float)(Half)f(i));
        }
        int kmax = qmax - 1;

        foreach (int e in new[] { -10, -6, -3, 0, 2, 5 })
        {
            float d = MathF.Pow(2f, e);
            for (int sign = 0; sign < 2; sign++)
            {
                float s = sign == 0 ? 1f : -1f;
                for (int k0 = 0; k0 < kmax; k0 += Math.Max(1, kmax / 4))
                {
                    int kk = k0;
                    Block($"tie e={e} s={s} k0={kk}", i =>
                    {
                        if (i == 0) return s * qmax * d;
                        int k = (kk + i - 1) % kmax;
                        return s * (k + 0.5f) * d;
                    });
                }
            }
            Block($"tie mixed e={e}", i => i == 0 ? qmax * d : ((i & 1) == 0 ? 1f : -1f) * (((i - 1) % kmax) + 0.5f) * d);
        }

        Block("all zero", _ => 0f);
        Block("all -0.0", _ => -0.0f);
        Block("one element", i => i == 7 ? -3.25f : 0f);

        // Generic: amax is Half-exact by construction; boundaries are placed in float, then rounded to Half - the rounding
        // moves each value to the nearest Half grid point, which for the smaller magnitudes lands exactly on or beside the
        // boundary that the float reference sees for the same Half payload.
        var rng = new Random(1499 + qmax);
        for (int b = 0; b < 6000; b++)
        {
            float amax = (float)(Half)(float)(Math.Exp(rng.NextDouble() * 10 - 5));   // 0.0067 .. 148, Half-exact
            float d = amax / qmax;
            float id = 1f / d;
            var vals = new float[Group];
            vals[0] = amax * (rng.Next(2) == 0 ? 1f : -1f);
            for (int i = 1; i < Group; i++)
            {
                int k = rng.Next(0, kmax);
                float boundary = (k + 0.5f) / id;
                int mode = rng.Next(4);
                float v = boundary;
                if (mode == 1) v = (float)Half.BitIncrement((Half)boundary);
                else if (mode == 2) v = (float)Half.BitDecrement((Half)boundary);
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
