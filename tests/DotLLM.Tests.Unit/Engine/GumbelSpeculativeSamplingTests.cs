using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Position-seeded Gumbel-max sampling for speculative verification (#1358). Proves (1) the noise primitive (known-answer Philox, 24-bit
/// uniform never produces an infinite variate, a 32-bit mutant does), (2) the draw is an exact sample from the prepared distribution
/// (chi-square), (3) MTP speculation with it emits the SAME tokens as plain decoding for the same seed (text identity, stronger than
/// distribution equality), (4) acceptance is not worse than the rejection-sampling method, (5) greedy is untouched.
/// </summary>
public sealed class GumbelSpeculativeSamplingTests
{
    private const int V = 16;
    private const int MaxSeqLen = 256;

    // Chi-square critical values use a fixed-seed design, so these tests are deterministic, not flaky.

    // ── noise primitive ──

    [Fact]
    public void Philox_KnownAnswer_ZeroCounterZeroKey()
        => Assert.Equal(0x6627e8d5u, GumbelNoise.Word(seed: 0, tokenIndex: 0, index: 0));

    [Fact]
    public void Uniform24_IsAlwaysInsideTheOpenUnitInterval_AndNoiseIsFinite()
    {
        foreach (uint w in new uint[] { 0, 1, 255, 256, 0x7FFFFFFF, 0xFFFFFF00, 0xFFFFFFFE, 0xFFFFFFFF })
        {
            float u = GumbelNoise.Uniform24(w);
            Assert.InRange(u, 1f / 16777216f, 1f - 1f / 16777216f);
            Assert.True(float.IsFinite(GumbelNoise.FromUniform(u)), $"word {w:X8} -> u={u}");
        }
        // the broken 32-bit conversion rounds the top of the range to exactly 1.0 -> +inf noise
        Assert.Equal(1f, GumbelNoise.Uniform32Unsafe(0xFFFFFFFF));
        Assert.True(float.IsPositiveInfinity(GumbelNoise.FromUniform(GumbelNoise.Uniform32Unsafe(0xFFFFFFFF))));
    }

    [Fact]
    public void Uniform32Mutant_IsCaught_ByTheForeignTokenRateAtRealVocabulary()
    {
        // Real vocabulary size; the target puts all its mass on 3 tokens and ~nothing on the other 151,933. A mutant draw lands on a "foreign"
        // token about V * 2^-25 = 4.5e-3 of the time (~11 in 2500 draws); the 24-bit form never does.
        const int Vocab = 151_936, Draws = 2500;
        var logits = new float[Vocab];
        int foreignOk = 0, foreignBad = 0;
        for (int n = 0; n < Draws; n++)
        {
            for (int arm = 0; arm < 2; arm++)
            {
                Array.Fill(logits, -40f);
                logits[0] = 0f; logits[1] = -0.5f; logits[2] = -1f;
                // pre-filter exactly like a top-k prepared row would: only entries that can matter are finite; foreign entries stay finite on purpose
                int tok = GumbelNoise.Sample(logits, seed: 77, tokenIndex: n, unsafeUniform32: arm == 1);
                if (tok > 2) { if (arm == 0) foreignOk++; else foreignBad++; }
            }
        }
        Assert.Equal(0, foreignOk);
        Assert.True(foreignBad >= 3, $"mutant produced only {foreignBad} foreign tokens in {Draws} draws");
    }

    // ── exactness of a single draw ──

    [Theory]
    [InlineData(1.0f, 0, 1.0f, 0f)]
    [InlineData(0.7f, 5, 1.0f, 0f)]
    [InlineData(1.3f, 0, 0.8f, 0f)]
    [InlineData(1.0f, 0, 1.0f, 0.1f)]
    public void Draw_MatchesThePreparedDistribution_ChiSquare(float temperature, int topK, float topP, float minP)
    {
        const int Trials = 60000;
        var options = new InferenceOptions { Temperature = temperature, TopK = topK, TopP = topP, MinP = minP, Seed = 5, GumbelSampling = true };
        var pipeline = new SamplerPipeline(options);
        Assert.True(pipeline.UsesGumbel);
        var raw = new float[V];
        for (int i = 0; i < V; i++) raw[i] = 2.5f * MathF.Sin(i * 1.7f + 0.3f);

        // expected = softmax of the prepared logits (what the categorical sampler would draw from)
        var prepared = (float[])raw.Clone();
        new SamplerPipeline(options).PrepareDistribution(prepared, new List<int>());
        double[] p = Softmax(prepared);

        var counts = new int[V];
        var history = new List<int>();
        for (int n = 0; n < Trials; n++)
        {
            while (history.Count < n) history.Add(0);          // generation index n
            var row = (float[])raw.Clone();
            counts[pipeline.Sample(row, history)]++;
        }
        AssertChiSquare(counts, p, Trials);
        for (int i = 0; i < V; i++) if (p[i] == 0) Assert.Equal(0, counts[i]);   // excluded tokens never win
    }

    // ── speculation == plain, same seed ──

    private static void SoftRow(int t, Span<float> row)
    {
        row.Fill(float.NegativeInfinity);
        row[(t + 1) % V] = MathF.Log(0.55f);
        row[(t + 2) % V] = MathF.Log(0.30f);
        row[(t + 3) % V] = MathF.Log(0.15f);
    }

    private static void WideRow(int t, Span<float> row)
    {
        for (int i = 0; i < V; i++) row[i] = 2f * MathF.Sin((t * 7 + i) * 1.3f);
    }

    private static List<int> RunPlain(SamplerPipeline pipeline, Action<int, Span<float>> rowFn, int start, int newTokens)
    {
        var ids = new List<int> { start };
        for (int i = 0; i < newTokens; i++)
        {
            var row = new float[V];
            rowFn(ids[^1], row);
            ids.Add(pipeline.Sample(row, ids));
        }
        return ids;
    }

    private static (List<int> Ids, int Rounds) RunSpec(SamplerPipeline pipeline, Action<int, Span<float>> rowFn, Func<int, int> draftFn,
        int start, int newTokens, int k)
    {
        using var model = new MtpSpeculativeDecoderTests.MockMtpModel(t => 0, draftFn, supportsMtp: true, targetRow: rowFn);
        var decoder = new MtpSpeculativeDecoder();
        var ids = new List<int> { start };
        using var kv = new SimpleKvCache(1, 1, 4, MaxSeqLen);
        using var mtp = new CpuMtpState(8, 1, 4, maxSteps: MaxSeqLen);
        int position = 0, rounds = 0;
        var outBuf = new int[k + 1];
        while (ids.Count - 1 < newTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, ids, null, position, V, k, outBuf);
            rounds++;
            for (int i = 0; i < r.AcceptedCount && ids.Count - 1 < newTokens; i++) ids.Add(outBuf[i]);
            position += r.AcceptedCount;
        }
        return (ids.Take(newTokens + 1).ToList(), rounds);
    }

    [Theory]
    [InlineData(1.0f, 0, 1.0f, 1.0f, 3)]
    [InlineData(0.8f, 6, 1.0f, 1.0f, 4)]
    [InlineData(1.0f, 0, 0.9f, 1.0f, 2)]
    [InlineData(1.0f, 8, 1.0f, 1.3f, 4)]     // repetition penalty reads the accepted history
    public void SpeculativeText_IsIdenticalToPlainText_ForTheSameSeed(float temperature, int topK, float topP, float repetitionPenalty, int k)
    {
        int rowsSpeculated = 0;
        for (int seed = 1; seed <= 60; seed++)
        {
            InferenceOptions Opts() => new()
            {
                Temperature = temperature, TopK = topK, TopP = topP, RepetitionPenalty = repetitionPenalty, Seed = seed, GumbelSampling = true,
            };
            var plain = RunPlain(new SamplerPipeline(Opts()), WideRow, start: seed % V, newTokens: 40);
            // draft = the token with the highest logit of the row (a realistic, frequently-right point-mass draft)
            Func<int, int> draft = t => { var r = new float[V]; WideRow(t, r); return Array.IndexOf(r, r.Max()); };
            var (spec, rounds) = RunSpec(new SamplerPipeline(Opts()), WideRow, draft, start: seed % V, newTokens: 40, k);
            Assert.Equal(plain, spec);
            rowsSpeculated += 40 - rounds;      // tokens saved by speculation
        }
        Assert.True(rowsSpeculated > 60, $"speculation saved only {rowsSpeculated} rounds: the draft path was not exercised");
    }

    [Fact]
    public void SpeculativeContinuations_FollowTheChainProbabilities_ChiSquare()
    {
        const int Trials = 30000, Len = 3;
        var counts = new int[27];
        int tokens = 0, rounds = 0;
        for (int trial = 0; trial < Trials; trial++)
        {
            var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 1.0f, Seed = 9000 + trial, GumbelSampling = true });
            var (ids, r) = RunSpec(pipeline, SoftRow, t => (t + 1) % V, start: 0, newTokens: Len, k: 3);
            rounds += r; tokens += Len;
            int a = (ids[1] - 0 + V) % V, b = (ids[2] - ids[1] + V) % V, c = (ids[3] - ids[2] + V) % V;
            counts[(a - 1) * 9 + (b - 1) * 3 + (c - 1)]++;
        }
        double[] step = [0.55, 0.30, 0.15];
        var p = new double[27];
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) p[a * 9 + b * 3 + c] = step[a] * step[b] * step[c];
        AssertChiSquare(counts, p, Trials);
        Assert.True(tokens / (double)rounds > 1.3, $"{tokens} tokens in {rounds} rounds");
    }

    [Fact]
    public void Acceptance_IsNotWorseThanRejectionSampling_ForAPointMassDraft()
    {
        // Both rules accept a point-mass draft d with probability p(d), so tokens per round must agree up to sampling noise.
        const int Trials = 6000, Len = 12;
        double Rate(bool gumbel)
        {
            int rounds = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 1.0f, Seed = 400 + trial, GumbelSampling = gumbel });
                rounds += RunSpec(pipeline, SoftRow, t => (t + 1) % V, start: 0, newTokens: Len, k: 3).Rounds;
            }
            return Trials * Len / (double)rounds;
        }
        double gumbel = Rate(true), rejection = Rate(false);
        Assert.True(gumbel >= rejection - 0.02, $"tokens/round gumbel={gumbel:F3} rejection={rejection:F3}");
    }

    [Fact]
    public void Greedy_IsUnchanged_EvenWithTheGumbelFlag()
    {
        var opts = new InferenceOptions { Temperature = 0f, GumbelSampling = true };
        var pipeline = new SamplerPipeline(opts);
        Assert.False(pipeline.UsesGumbel);
        var plain = RunPlain(new SamplerPipeline(new InferenceOptions { Temperature = 0f }), WideRow, 1, 30);
        Func<int, int> draft = t => { var r = new float[V]; WideRow(t, r); return Array.IndexOf(r, r.Max()); };
        Assert.Equal(plain, RunSpec(pipeline, WideRow, draft, 1, 30, 4).Ids);
        Assert.Equal(plain, RunPlain(pipeline, WideRow, 1, 30));
    }

    // ── helpers ──

    private static double[] Softmax(float[] logits)
    {
        double max = logits.Where(float.IsFinite).Max(), sum = 0;
        var p = new double[logits.Length];
        for (int i = 0; i < p.Length; i++) { p[i] = float.IsFinite(logits[i]) ? Math.Exp(logits[i] - max) : 0; sum += p[i]; }
        for (int i = 0; i < p.Length; i++) p[i] /= sum;
        return p;
    }

    private static void AssertChiSquare(int[] counts, double[] p, int trials)
    {
        double chi = 0; int cells = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            double e = p[i] * trials;
            if (e < 5) continue;
            chi += (counts[i] - e) * (counts[i] - e) / e; cells++;
        }
        int df = cells - 1;
        Assert.True(chi < CritValue(df), $"chi-square {chi:F1} over {cells} cells (df {df}) exceeds the 99.9% critical value {CritValue(df):F1}");
    }

    private static double CritValue(int df)
    {
        // Wilson-Hilferty approximation of the 99.9th percentile (z = 3.0902).
        double z = 3.0902, a = 2.0 / (9.0 * df);
        return df * Math.Pow(1 - a + z * Math.Sqrt(a), 3);
    }
}
