using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Unit tests for <see cref="EmbeddingPooler"/> (issue #451).
/// </summary>
/// <remarks>
/// The hidden states here are chosen so that the three pooling modes produce <b>different</b>
/// answers and no two modes coincide — a fixture where rows are equal, or where seqLen is 1,
/// cannot tell <c>last</c> from <c>cls</c> from <c>mean</c> and would pass for any of the three
/// implementations (the degenerate-shape trap called out in the project guide).
/// </remarks>
public sealed class EmbeddingPoolerTests
{
    // [seqLen=3, hiddenSize=2], row-major. No two rows equal; the mean equals no row.
    private static readonly float[] Hidden = [1f, 2f, 10f, -4f, 100f, 8f];

    [Fact]
    public void Last_selects_the_final_row()
    {
        var dest = new float[2];
        EmbeddingPooler.Pool(Hidden, seqLen: 3, hiddenSize: 2, PoolingType.Last, dest);
        Assert.Equal([100f, 8f], dest);
    }

    [Fact]
    public void Cls_selects_the_first_row()
    {
        var dest = new float[2];
        EmbeddingPooler.Pool(Hidden, seqLen: 3, hiddenSize: 2, PoolingType.Cls, dest);
        Assert.Equal([1f, 2f], dest);
    }

    [Fact]
    public void Mean_averages_every_row_with_uniform_weights()
    {
        var dest = new float[2];
        EmbeddingPooler.Pool(Hidden, seqLen: 3, hiddenSize: 2, PoolingType.Mean, dest);
        Assert.Equal(111f / 3f, dest[0], 6);
        Assert.Equal(6f / 3f, dest[1], 6);
    }

    [Fact]
    public void Mean_uses_only_the_first_seqLen_rows()
    {
        // The buffer holds 3 rows; pooling 2 must ignore the third.
        var dest = new float[2];
        EmbeddingPooler.Pool(Hidden, seqLen: 2, hiddenSize: 2, PoolingType.Mean, dest);
        Assert.Equal(11f / 2f, dest[0], 6);
        Assert.Equal(-2f / 2f, dest[1], 6);
    }

    [Fact]
    public void Last_honours_seqLen_rather_than_the_buffer_length()
    {
        var dest = new float[2];
        EmbeddingPooler.Pool(Hidden, seqLen: 2, hiddenSize: 2, PoolingType.Last, dest);
        Assert.Equal([10f, -4f], dest);
    }

    [Fact]
    public void Mean_is_numerically_stable_over_a_long_sequence()
    {
        // 4096 rows of a large constant plus a tiny alternating perturbation: a naive float32
        // accumulator loses the perturbation entirely.
        const int seqLen = 4096;
        var hidden = new float[seqLen];
        for (int i = 0; i < seqLen; i++)
            hidden[i] = 1e7f + (i % 2 == 0 ? 1f : -1f);

        var dest = new float[1];
        EmbeddingPooler.Pool(hidden, seqLen, hiddenSize: 1, PoolingType.Mean, dest);
        Assert.Equal(1e7f, dest[0], 0);
    }

    [Fact]
    public void L2Normalize_produces_a_unit_vector_preserving_direction()
    {
        var v = new float[] { 3f, 4f };
        EmbeddingPooler.L2Normalize(v);
        Assert.Equal(0.6f, v[0], 6);
        Assert.Equal(0.8f, v[1], 6);
        Assert.Equal(1.0, Math.Sqrt((double)v[0] * v[0] + (double)v[1] * v[1]), 6);
    }

    [Fact]
    public void L2Normalize_maps_a_zero_vector_to_zero_not_NaN()
    {
        // llama.cpp: norm = sum > 0.0 ? 1.0/sum : 0.0f — no NaNs.
        var v = new float[] { 0f, 0f, 0f };
        EmbeddingPooler.L2Normalize(v);
        Assert.All(v, x => Assert.Equal(0f, x));
    }

    [Theory]
    [InlineData(PoolingType.None)]
    [InlineData(PoolingType.Rank)]
    public void Unsupported_pooling_types_throw(PoolingType pooling)
    {
        var dest = new float[2];
        Assert.Throws<NotSupportedException>(() =>
            EmbeddingPooler.Pool(Hidden, seqLen: 3, hiddenSize: 2, pooling, dest));
    }

    [Fact]
    public void Pool_rejects_a_hidden_buffer_that_is_too_small()
    {
        var dest = new float[2];
        Assert.Throws<ArgumentException>(() =>
            EmbeddingPooler.Pool(Hidden, seqLen: 4, hiddenSize: 2, PoolingType.Last, dest));
    }

    [Fact]
    public void Pool_rejects_a_destination_that_is_too_small()
    {
        var dest = new float[1];
        Assert.Throws<ArgumentException>(() =>
            EmbeddingPooler.Pool(Hidden, seqLen: 3, hiddenSize: 2, PoolingType.Last, dest));
    }

    [Fact]
    public void Pool_rejects_a_zero_length_sequence()
    {
        var dest = new float[2];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EmbeddingPooler.Pool(Hidden, seqLen: 0, hiddenSize: 2, PoolingType.Last, dest));
    }

    [Fact]
    public void Resolve_prefers_the_explicit_request()
        => Assert.Equal(PoolingType.Cls, EmbeddingPooler.Resolve(PoolingType.Cls, PoolingType.Mean));

    [Fact]
    public void Resolve_falls_back_to_the_checkpoint_declaration()
        => Assert.Equal(PoolingType.Mean, EmbeddingPooler.Resolve(null, PoolingType.Mean));

    [Fact]
    public void Resolve_falls_back_to_last_when_the_checkpoint_declares_nothing()
        => Assert.Equal(PoolingType.Last, EmbeddingPooler.Resolve(null, null));

    [Fact]
    public void Resolve_honours_a_declared_None_rather_than_silently_rewriting_it()
    {
        // The caller must be told the checkpoint asks for something unrepresentable, not handed
        // a plausible-looking vector computed with a different rule.
        Assert.Equal(PoolingType.None, EmbeddingPooler.Resolve(null, PoolingType.None));
    }
}
