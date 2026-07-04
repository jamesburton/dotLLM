using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Unit coverage for <see cref="SelfCondSoftEmbed"/> — the shared (CPU-oracle + Vulkan-host)
/// DiffusionGemma self-conditioning soft-embed with top-K sparsification (issue #121):
/// top-K selection determinism (value desc, index asc), subset-renormalised softmax, the
/// mass-concentration case where dense and sparse agree to numerical noise, the spread case
/// where sparse deviates in a fully predictable way, and the dense-path routing guarantees
/// (K &lt;= 0 and K &gt;= vocab are byte-identical to each other).
/// </summary>
public sealed class SelfCondSoftEmbedTests
{
    // ── SelectTopK ──────────────────────────────────────────────────────────

    [Fact]
    public void SelectTopK_ReturnsKLargest_InAscendingIndexOrder()
    {
        float[] logits = [0.5f, 9f, -2f, 7f, 3f, 8f, 1f, -0.5f];
        Span<int> idx = stackalloc int[3];
        SelfCondSoftEmbed.SelectTopK(logits, 3, idx);
        // Top-3 values are 9 (idx 1), 8 (idx 5), 7 (idx 3) → ascending index order.
        Assert.Equal(new[] { 1, 3, 5 }, idx.ToArray());
    }

    [Fact]
    public void SelectTopK_BoundaryTies_PreferLowerIndex()
    {
        // Four entries tie at 5.0; K=3 must keep the LOWEST-index tied entries after the
        // strictly-larger 6.0. Expected: idx 0 (6.0), then the two lowest tied ids 1, 2.
        float[] logits = [6f, 5f, 5f, 5f, 5f, -1f];
        Span<int> idx = stackalloc int[3];
        SelfCondSoftEmbed.SelectTopK(logits, 3, idx);
        Assert.Equal(new[] { 0, 1, 2 }, idx.ToArray());
    }

    [Fact]
    public void SelectTopK_AllTied_KeepsFirstK()
    {
        float[] logits = new float[16]; // all zero — total tie
        Span<int> idx = stackalloc int[5];
        SelfCondSoftEmbed.SelectTopK(logits, 5, idx);
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, idx.ToArray());
    }

    [Fact]
    public void SelectTopK_KEqualsLength_ReturnsEveryIndex()
    {
        float[] logits = [3f, -1f, 2f, 0f];
        Span<int> idx = stackalloc int[4];
        SelfCondSoftEmbed.SelectTopK(logits, 4, idx);
        Assert.Equal(new[] { 0, 1, 2, 3 }, idx.ToArray());
    }

    // ── RenormSoftmax ───────────────────────────────────────────────────────

    [Fact]
    public void RenormSoftmax_MatchesManualComputation_AndSumsToOne()
    {
        float[] logits = [1f, 2f, 3f, 0f, -1f];
        int[] indices = [0, 1, 2]; // subset {1, 2, 3}
        Span<float> probs = stackalloc float[3];
        SelfCondSoftEmbed.RenormSoftmax(logits, indices, probs);

        // Manual: max = 3; e = [exp(-2), exp(-1), 1]; p = e / Σe.
        double e0 = Math.Exp(-2.0), e1 = Math.Exp(-1.0), e2 = 1.0;
        double sum = e0 + e1 + e2;
        Assert.Equal((float)(e0 / sum), probs[0], 6);
        Assert.Equal((float)(e1 / sum), probs[1], 6);
        Assert.Equal((float)(e2 / sum), probs[2], 6);
        Assert.Equal(1f, probs[0] + probs[1] + probs[2], 5);
    }

    // ── Compute: dense vs sparse ────────────────────────────────────────────

    private const int Vocab = 64;
    private const int Hidden = 8;

    /// <summary>Deterministic synthetic F32 embedding table [Vocab × Hidden].</summary>
    private static float[] MakeTable()
    {
        var table = new float[Vocab * Hidden];
        var rng = new Random(42);
        for (int i = 0; i < table.Length; i++)
            table[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return table;
    }

    private static unsafe float[] RunCompute(float[] table, float[] logits, int canvasLen, int topK)
    {
        var soft = new float[canvasLen * Hidden];
        fixed (float* tp = table)
        {
            SelfCondSoftEmbed.Compute(
                logits, canvasLen, Vocab, (nint)tp, QuantizationType.F32, Hidden, topK, soft);
        }
        return soft;
    }

    /// <summary>
    /// Mass-concentration case: one logit per column dwarfs the rest (+30 ⇒ residual mass
    /// &lt; exp(−30) ≈ 1e−13 per token). Dense and sparse (K = 4) soft-embeds must agree to
    /// float noise — the discriminating "sparse ≈ dense when the distribution is peaked" case.
    /// </summary>
    [Fact]
    public void Compute_MassConcentrated_SparseAgreesWithDense()
    {
        float[] table = MakeTable();
        const int canvasLen = 3;
        var logits = new float[canvasLen * Vocab];
        var rng = new Random(7);
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() - 0.5);
        int[] peaks = [5, 17, 40];
        for (int c = 0; c < canvasLen; c++) logits[c * Vocab + peaks[c]] += 30f;

        float[] dense = RunCompute(table, logits, canvasLen, topK: 0);
        float[] sparse = RunCompute(table, logits, canvasLen, topK: 4);

        for (int i = 0; i < dense.Length; i++)
            Assert.True(MathF.Abs(dense[i] - sparse[i]) <= 1e-5f,
                $"idx {i}: dense={dense[i]:F7} sparse={sparse[i]:F7}");
    }

    /// <summary>
    /// Spread case: ALL logits equal ⇒ dense soft-embed is the mean of all Vocab embedding
    /// rows, while sparse top-K (tie-break: lowest ids win) is the mean of the FIRST K rows —
    /// an exactly predictable deviation. Verifies both the deviation and its exact value.
    /// </summary>
    [Fact]
    public void Compute_SpreadDistribution_SparseDeviatesPredictably()
    {
        float[] table = MakeTable();
        const int k = 8;
        var logits = new float[Vocab]; // single column, all-zero logits

        float[] dense = RunCompute(table, logits, canvasLen: 1, topK: 0);
        float[] sparse = RunCompute(table, logits, canvasLen: 1, topK: k);

        // Expected sparse: mean of embedding rows 0..k-1 (uniform 1/k over the k lowest ids).
        var expected = new float[Hidden];
        for (int v = 0; v < k; v++)
            for (int h = 0; h < Hidden; h++)
                expected[h] += table[v * Hidden + h] / k;

        float maxDevFromExpected = 0f, maxDevFromDense = 0f;
        for (int h = 0; h < Hidden; h++)
        {
            maxDevFromExpected = MathF.Max(maxDevFromExpected, MathF.Abs(sparse[h] - expected[h]));
            maxDevFromDense = MathF.Max(maxDevFromDense, MathF.Abs(sparse[h] - dense[h]));
        }
        Assert.True(maxDevFromExpected <= 1e-5f,
            $"sparse must equal the mean of the first {k} rows (max dev {maxDevFromExpected:E3}).");
        // With random ±1 rows the first-8 mean and the all-64 mean differ materially —
        // the sparse path must NOT silently reproduce dense here.
        Assert.True(maxDevFromDense > 1e-3f,
            $"sparse must deviate from dense on a spread distribution (max dev {maxDevFromDense:E3}).");
    }

    /// <summary>K ≥ vocab and K ≤ 0 both route to the dense reference — bit-identical output.</summary>
    [Theory]
    [InlineData(Vocab)]
    [InlineData(Vocab + 100)]
    [InlineData(-1)]
    public void Compute_KOutsideSparseRange_ByteIdenticalToDense(int k)
    {
        float[] table = MakeTable();
        const int canvasLen = 2;
        var logits = new float[canvasLen * Vocab];
        var rng = new Random(1234);
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

        float[] dense = RunCompute(table, logits, canvasLen, topK: 0);
        float[] routed = RunCompute(table, logits, canvasLen, topK: k);

        for (int i = 0; i < dense.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(dense[i]), BitConverter.SingleToInt32Bits(routed[i]));
    }

    // ── ResolveTopK precedence ──────────────────────────────────────────────

    [Fact]
    public void ResolveTopK_EnvWins_ThenConfig_ThenDefault()
    {
        var cfg = new DiffusionConfig { MaskTokenId = 4, SelfCondTopK = 128 };

        // Parseable env value takes precedence (including <= 0 ⇒ dense).
        Assert.Equal(64, SelfCondSoftEmbed.ResolveTopK("64", cfg));
        Assert.Equal(0, SelfCondSoftEmbed.ResolveTopK("0", cfg));
        Assert.Equal(-3, SelfCondSoftEmbed.ResolveTopK("-3", cfg));

        // Unset / blank / unparseable env falls back to the config.
        Assert.Equal(128, SelfCondSoftEmbed.ResolveTopK(null, cfg));
        Assert.Equal(128, SelfCondSoftEmbed.ResolveTopK("", cfg));
        Assert.Equal(128, SelfCondSoftEmbed.ResolveTopK("  ", cfg));
        Assert.Equal(128, SelfCondSoftEmbed.ResolveTopK("not-a-number", cfg));

        // No config at all ⇒ documented default (256).
        Assert.Equal(SelfCondSoftEmbed.DefaultTopK, SelfCondSoftEmbed.ResolveTopK(null, null));
        Assert.Equal(256, SelfCondSoftEmbed.DefaultTopK);
    }

    /// <summary>The record default must match the helper's documented default.</summary>
    [Fact]
    public void DiffusionConfig_SelfCondTopK_DefaultsTo256()
    {
        var cfg = new DiffusionConfig { MaskTokenId = 4 };
        Assert.Equal(256, cfg.SelfCondTopK);
    }
}
