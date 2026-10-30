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

/// <summary>EOS handling for last-token pooling on decoder embedders (#1398).</summary>
public sealed class EmbeddingPoolerAppendEosTests
{
    private const int Eos = 151643;

    [Fact]
    public void Last_pooling_decoder_appends_eos_and_does_not_mutate_input()
    {
        int[] tokens = [10, 20, 30];
        var r = EmbeddingPooler.AppendEos(tokens, PoolingType.Last, modelAppendsEos: true, Eos, addEosOverride: null);
        Assert.Equal([10, 20, 30, Eos], r);
        Assert.Equal([10, 20, 30], tokens);
    }

    [Theory]
    [InlineData(PoolingType.Mean)]
    [InlineData(PoolingType.Cls)]
    public void Non_last_pooling_leaves_tokens_alone(PoolingType p)
    {
        int[] tokens = [10, 20];
        Assert.Same(tokens, EmbeddingPooler.AppendEos(tokens, p, true, Eos, null));
    }

    [Fact]
    public void Encoder_with_last_pooling_does_not_append()
    {
        int[] tokens = [10, 20];
        Assert.Same(tokens, EmbeddingPooler.AppendEos(tokens, PoolingType.Last, modelAppendsEos: false, Eos, null));
    }

    [Fact]
    public void Existing_trailing_eos_is_not_doubled()
    {
        int[] tokens = [10, Eos];
        Assert.Same(tokens, EmbeddingPooler.AppendEos(tokens, PoolingType.Last, true, Eos, null));
    }

    [Fact]
    public void Override_forces_and_suppresses()
    {
        int[] tokens = [10, 20];
        Assert.Equal([10, 20, Eos], EmbeddingPooler.AppendEos(tokens, PoolingType.Mean, false, Eos, addEosOverride: true));
        Assert.Same(tokens, EmbeddingPooler.AppendEos(tokens, PoolingType.Last, true, Eos, addEosOverride: false));
    }

    [Fact]
    public void Negative_eos_id_disables_appending()
    {
        int[] tokens = [10];
        Assert.Same(tokens, EmbeddingPooler.AppendEos(tokens, PoolingType.Last, true, -1, null));
    }

    [Fact]
    public void Last_row_differs_with_and_without_eos_row()
    {
        // Discriminating: the pooled vector is the EOS row, not the final text row.
        float[] hidden = [1f, 2f, 10f, -4f, 100f, 8f];
        var withEos = new float[2]; var without = new float[2];
        EmbeddingPooler.Pool(hidden, 3, 2, PoolingType.Last, withEos);
        EmbeddingPooler.Pool(hidden, 2, 2, PoolingType.Last, without);
        Assert.NotEqual(without, withEos);
    }
}
