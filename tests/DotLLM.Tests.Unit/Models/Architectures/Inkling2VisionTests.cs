using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Inkling-2-Mini vision tower + projector, CPU (issue #2035) against llama.cpp b12780 CPU: per-op checkpoint parity with <c>llama-mtmd-debug -p encode</c>
/// (every named checkpoint: first/last 3 values of each dimension + the tensor sum) and full-embedding parity on the real preprocessing path
/// (<c>mtmd_encode_chunk</c> output of the oracle program, F32). Every gate is shown to fail under a mutant (<see cref="Inkling2VisionMutant"/>).
/// </summary>
public sealed class Inkling2VisionTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VisionTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] EarlyStages =
        ["patch_embd", "posemb_interp", "embeddings", "pre_ln", "layer_inp_normed-0", "Qcur_pos-0", "Kcur_pos-0", "Vcur-0"];

    private static Inkling2VisionEncoder LoadEncoder()
    {
        string? mm = Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        return Inkling2VisionEncoder.Load(mm!, new ThreadingConfig(8));
    }

    /// <summary>
    /// Checkpoint gate. The first stages (before the Q8_0 activation noise of the matmuls accumulates) must agree to the printed precision (4 decimals);
    /// every later stage within 8% of its corner scale AND with a tensor sum within 2e-3 of N*rms (a wrong op moves the sum by O(1) of that scale).
    /// </summary>
    private static List<string> CheckpointViolations(Inkling2VisionEncoder enc, int n)
    {
        var oracle = Inkling2VisionFixtures.LoadTrace($"encode-rainbow{n}.json");
        Skip.If(oracle is null, "vision oracle (encode-rainbow*.json) not generated");
        var img = Inkling2VisionFixtures.Rainbow(n);
        var got = Inkling2VisionFixtures.Capture(enc, () => { using var e = enc.EncodeNormalized(img, n, n); });
        var v = new List<string>();
        string Map(string o) => o == "node_13" ? "posemb_interp" : o;
        foreach (var (name, cp) in oracle!)
        {
            string mine = Map(name);
            if (!got.TryGetValue(mine, out var data)) { v.Add($"{name}: missing"); continue; }
            long want = 1; foreach (int d in cp.Ne) want *= d;
            if (data.Length != want) { v.Add($"{name}: {data.Length} values, expected {want}"); continue; }
            var d2 = Inkling2VisionFixtures.Compare(cp, data);
            bool early = EarlyStages.Contains(mine);
            if (early ? d2.MaxCornerErr > 2e-4 : d2.MaxCornerRel > 0.08)
                v.Add($"{name}: corner err {d2.MaxCornerErr:E2} (rel {d2.MaxCornerRel:E2})");
            if (!double.IsNaN(d2.SumErr) && d2.SumErr / d2.SumScale > 2e-3)
                v.Add($"{name}: sum err {d2.SumErr:E2} vs scale {d2.SumScale:E2}");
        }
        return v;
    }

    [SkippableTheory]
    [InlineData(80)]    // 32 -> 4 position-embedding interpolation
    [InlineData(640)]   // 32 x 32 grid: the identity branch
    public void Checkpoints_MatchLlamaCppMtmdDebug(int n)
    {
        using var enc = LoadEncoder();
        var violations = CheckpointViolations(enc, n);
        Assert.True(violations.Count == 0, string.Join("; ", violations.Take(8)));
    }

    public static TheoryData<Inkling2VisionMutant> CheckpointMutants => new()
    {
        Inkling2VisionMutant.PosembAlignCorners, Inkling2VisionMutant.PosembTransposed, Inkling2VisionMutant.RopeAxesSwapped,
        Inkling2VisionMutant.RopeNeox, Inkling2VisionMutant.MergeOrder, Inkling2VisionMutant.MergeSkipped, Inkling2VisionMutant.NoLayerScale,
        Inkling2VisionMutant.GeluTanh, Inkling2VisionMutant.NoCls,
    };

    /// <summary>"Prove the test can fail": each deliberately wrong op must trip the checkpoint gate at n=80 (the interpolating case).</summary>
    [SkippableTheory]
    [MemberData(nameof(CheckpointMutants))]
    public void CheckpointGate_KillsMutant(Inkling2VisionMutant mutant)
    {
        using var enc = LoadEncoder();
        enc.Mutant = mutant;
        var violations = CheckpointViolations(enc, 80);
        _output.WriteLine($"{mutant}: {violations.Count} violation(s): {string.Join("; ", violations.Take(3))}");
        Assert.True(violations.Count > 0, $"mutant {mutant} survived the checkpoint gate");
    }

    // ───────────── full embeddings on the real preprocessing path ─────────────

    private sealed record EmbdOracle(DecodedImage Image, float[] Embd, int Tokens, int Dim);

    private static EmbdOracle? LoadEmbd(string name)
    {
        string dir = Path.Combine(Inkling2VisionFixtures.OracleDir, name);
        string ppm = Path.Combine(dir, "image.ppm"), embd = Path.Combine(dir, "embd.f32");
        if (!File.Exists(ppm) || !File.Exists(embd)) return null;
        var bytes = File.ReadAllBytes(embd);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        const int dim = 2304;
        return new EmbdOracle(ImageDecoder.Load(ppm), f, f.Length / dim, dim);
    }

    internal sealed record EmbdDiff(double RelRms, double MinCosine, double MaxAbs);

    internal static EmbdDiff CompareEmbeddings(ImageEmbeddings got, float[] want, int dim)
    {
        double num = 0, den = 0, maxAbs = 0, minCos = 1;
        var row = new float[dim];
        for (int r = 0; r < got.Tokens; r++)
        {
            got.ReadRow(r, row);
            double dot = 0, na = 0, nb = 0;
            for (int c = 0; c < dim; c++)
            {
                double a = row[c], b = want[r * dim + c];
                num += (a - b) * (a - b); den += b * b; maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
                dot += a * b; na += a * a; nb += b * b;
            }
            minCos = Math.Min(minCos, dot / Math.Sqrt(na * nb));
        }
        return new EmbdDiff(Math.Sqrt(num / den), minCos, maxAbs);
    }

    /// <summary>
    /// Gate: relative RMS error under 6% and every row's cosine above 0.99. The floor is NOT zero and is not a bug: across 23 layers of Q8_0
    /// activation quantisation a ~1e-4 relative input change (F16 rounding of the im2col pixels on vs off) already moves the embeddings by 2.1% relRMS
    /// (<see cref="Report_NoiseFloorAndMutantDistances"/>), the same as the gap to llama.cpp (1.8-3.3% over the three images). Sharper checks are the
    /// early checkpoints and the end-to-end KL; this gate catches gross errors (every mutant is above 18%).
    /// </summary>
    private static bool EmbeddingsPass(EmbdDiff d) => d.RelRms < 0.06 && d.MinCosine > 0.99;

    [SkippableTheory]
    [InlineData("rainbow190x110")]   // non-square, not a multiple of 40: padded to 200x120, 5x3 tokens
    [InlineData("rainbow80")]
    [InlineData("rainbow640")]
    public void Embeddings_MatchLlamaCppEncodeChunk(string name)
    {
        var o = LoadEmbd(name);
        Skip.If(o is null, "embedding oracle (vision/<name>/embd.f32) not generated");
        using var enc = LoadEncoder();
        Assert.Equal(o!.Tokens, enc.CountTokens(o.Image.Width, o.Image.Height));
        using var got = enc.Encode(o.Image);
        Assert.Equal(o.Tokens, got.Tokens);
        var d = CompareEmbeddings(got, o.Embd, o.Dim);
        _output.WriteLine($"{name}: relRMS {d.RelRms:E2}, min cosine {d.MinCosine:F5}, max abs {d.MaxAbs:E2}, {got.Bytes} F16 bytes");
        Assert.True(EmbeddingsPass(d), $"relRMS {d.RelRms:E2}, min cosine {d.MinCosine:F5}");
    }

    public enum PreprocMutant { Bgr, PadNormalizedZero }

    /// <summary>Preprocessing mutants on the non-square image: channel order, and the pad value (-1 raw is the reference's quirk).</summary>
    [SkippableTheory]
    [InlineData(PreprocMutant.Bgr)]
    [InlineData(PreprocMutant.PadNormalizedZero)]
    public void EmbeddingGate_KillsPreprocessingMutant(PreprocMutant m)
    {
        var o = LoadEmbd("rainbow190x110");
        Skip.If(o is null, "embedding oracle not generated");
        using var enc = LoadEncoder();
        ImageEmbeddings got;
        if (m == PreprocMutant.Bgr) { enc.SwapRgb = true; got = enc.Encode(o!.Image); }
        else
        {
            var (w, h, _, _) = Inkling2ImagePreprocessor.PlanSize(o!.Image.Width, o.Image.Height);
            var buf = new float[w * h * 3];
            Inkling2ImagePreprocessor.Normalize(o.Image, buf, padRaw: 127.5f);   // pad = 0 after normalisation, the plausible wrong alternative to -1 raw
            got = enc.EncodeNormalized(buf, w, h);
        }
        using (got)
        {
            var d = CompareEmbeddings(got, o.Embd, o.Dim);
            _output.WriteLine($"{m}: relRMS {d.RelRms:E2}, min cosine {d.MinCosine:F5}, max abs {d.MaxAbs:E2}");
            Assert.False(EmbeddingsPass(d), $"preprocessing mutant {m} survived (relRMS {d.RelRms:E2}, min cosine {d.MinCosine:F5})");
        }
    }

    /// <summary>
    /// Op-level gate that the noisy embedding comparison cannot be: the erf GELU against a double-precision erf over [-8, 8]. The tanh variant
    /// (a plausible wrong implementation: it moves the embeddings by 2.3% vs the 2.1% floor) differs by up to 4.7e-4, which this catches.
    /// </summary>
    [Fact]
    public void GeluErf_MatchesDoublePrecisionReference_AndTheTanhVariantIsDistinguishable()
    {
        static double ErfRef(double x)
        {
            double ax = Math.Abs(x);
            double r;
            if (ax < 2.5)
            {
                double term = ax, sum = ax, x2 = ax * ax;
                for (int n = 1; n < 200; n++) { term *= -x2 / n; double add = term / (2 * n + 1); sum += add; if (Math.Abs(add) < 1e-18) break; }
                r = 1.1283791670955126 * sum;
            }
            else
            {
                double f = 0;
                for (int k = 60; k >= 1; k--) f = k / 2.0 / (ax + f);
                r = 1.0 - Math.Exp(-ax * ax) / (1.7724538509055159 * (ax + f));
            }
            return x < 0 ? -r : r;
        }
        double maxErf = 0, maxTanh = 0;
        for (float x = -8f; x <= 8f; x += 0.01f)
        {
            double want = 0.5 * x * (1 + ErfRef(x * 0.7071067811865476));
            maxErf = Math.Max(maxErf, Math.Abs(Inkling2VisionEncoder.GeluErfScalar(x) - want));
            maxTanh = Math.Max(maxTanh, Math.Abs(Inkling2VisionEncoder.GeluTanhScalar(x) - want));
        }
        _output.WriteLine($"max |erf GELU - reference| {maxErf:E2}; max |tanh GELU - reference| {maxTanh:E2}");
        Assert.True(maxErf < 1e-6, $"erf GELU error {maxErf:E2}");
        Assert.True(maxTanh > 1e-4, "the tanh variant is not distinguishable by this check");
    }

    /// <summary>Not a gate: prints the unmutated noise floor and each mutant's distance (recorded in the PR).</summary>
    [SkippableFact]
    public void Report_NoiseFloorAndMutantDistances()
    {
        var o = LoadEmbd("rainbow190x110");
        Skip.If(o is null, "embedding oracle not generated");
        using var enc = LoadEncoder();
        foreach (var m in Enum.GetValues<Inkling2VisionMutant>())
        {
            enc.Mutant = m;
            using var got = enc.Encode(o!.Image);
            var d = CompareEmbeddings(got, o.Embd, o.Dim);
            _output.WriteLine($"{m,-20} relRMS {d.RelRms:E2}  min cosine {d.MinCosine:F5}  max abs {d.MaxAbs:E2}");
        }
        enc.Mutant = Inkling2VisionMutant.None;
        enc.SkipIm2colF16Rounding = true;
        using var exact = enc.Encode(o!.Image);
        var e = CompareEmbeddings(exact, o.Embd, o.Dim);
        _output.WriteLine($"no F16 im2col rounding relRMS {e.RelRms:E2}  min cosine {e.MinCosine:F5}");
        // Chaos probe: how far does a ~1e-4 relative input change (F16 im2col rounding on/off) move the embeddings, vs the oracle gap above?
        enc.SkipIm2colF16Rounding = false;
        using var baseline = enc.Encode(o.Image);
        var baseF = new float[exact.Tokens * exact.Dimension];
        var tmp = new float[exact.Dimension];
        for (int r = 0; r < baseline.Tokens; r++) { baseline.ReadRow(r, tmp); Array.Copy(tmp, 0, baseF, r * exact.Dimension, exact.Dimension); }
        var self = CompareEmbeddings(exact, baseF, o.Dim);
        _output.WriteLine($"self-distance (F16 im2col on vs off) relRMS {self.RelRms:E2}  min cosine {self.MinCosine:F5}");
    }
}
