using System.Text.Json;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Unit tests for <see cref="EmbeddingInputParser"/> — the OpenAI <c>input</c> union of
/// <c>POST /v1/embeddings</c> (issue #451).
/// </summary>
public sealed class EmbeddingInputParserTests
{
    private const int VocabSize = 100;

    /// <summary>Deterministic stand-in tokenizer: one token per character, id = length-dependent.</summary>
    private static int[] Encode(string text) => text.Select(c => (int)c % VocabSize).ToArray();

    private static EmbeddingInputParser.Result Parse(string json)
        => EmbeddingInputParser.Parse(JsonDocument.Parse(json).RootElement, Encode, VocabSize);

    [Fact]
    public void A_string_is_one_sequence()
    {
        var r = Parse("\"ab\"");
        Assert.True(r.Ok);
        Assert.Single(r.Sequences!);
        Assert.Equal(Encode("ab"), r.Sequences![0]);
    }

    [Fact]
    public void An_array_of_strings_is_many_sequences()
    {
        var r = Parse("[\"ab\",\"cde\"]");
        Assert.True(r.Ok);
        Assert.Equal(2, r.Sequences!.Count);
        Assert.Equal(2, r.Sequences[0].Length);
        Assert.Equal(3, r.Sequences[1].Length);
    }

    /// <summary>
    /// The disambiguation that matters: a flat number array is ONE pre-tokenised sequence, not
    /// three single-token sequences. Getting this backwards would return the wrong number of
    /// embeddings while still looking well-formed.
    /// </summary>
    [Fact]
    public void A_flat_number_array_is_one_pretokenised_sequence()
    {
        var r = Parse("[1,2,3]");
        Assert.True(r.Ok);
        Assert.Single(r.Sequences!);
        Assert.Equal([1, 2, 3], r.Sequences![0]);
    }

    [Fact]
    public void A_nested_number_array_is_many_pretokenised_sequences()
    {
        var r = Parse("[[1,2],[3,4,5]]");
        Assert.True(r.Ok);
        Assert.Equal(2, r.Sequences!.Count);
        Assert.Equal([1, 2], r.Sequences[0]);
        Assert.Equal([3, 4, 5], r.Sequences[1]);
    }

    [Theory]
    [InlineData("\"\"", "empty string")]
    [InlineData("[]", "empty array")]
    [InlineData("[[]]", "empty token array")]
    [InlineData("[\"a\",\"\"]", "empty string in array")]
    [InlineData("null", "null")]
    [InlineData("42", "bare number")]
    [InlineData("{}", "object")]
    [InlineData("[\"a\",1]", "mixed string/number")]
    [InlineData("[1,\"a\"]", "mixed number/string")]
    [InlineData("[[1],\"a\"]", "mixed array/string")]
    [InlineData("[1.5,2]", "non-integer token id")]
    [InlineData("[-1,2]", "negative token id")]
    [InlineData("[1,100]", "token id == vocabSize")]
    [InlineData("[[1,200]]", "nested out-of-vocabulary token id")]
    [InlineData("[true]", "boolean element")]
    public void Malformed_input_is_rejected_with_a_message(string json, string why)
    {
        var r = Parse(json);
        Assert.False(r.Ok, $"expected rejection for {why}: {json}");
        Assert.False(string.IsNullOrWhiteSpace(r.Error));
        Assert.Null(r.Sequences);
    }

    [Fact]
    public void The_highest_valid_token_id_is_accepted()
    {
        var r = Parse("[99]");
        Assert.True(r.Ok);
        Assert.Equal([99], r.Sequences![0]);
    }

    [Fact]
    public void Undefined_input_is_rejected()
    {
        var r = EmbeddingInputParser.Parse(default, Encode, VocabSize);
        Assert.False(r.Ok);
    }

    /// <summary>
    /// Each sequence keeps its own identity and order — a parser that flattened or reordered
    /// would break the OpenAI contract that <c>data[i]</c> corresponds to <c>input[i]</c>.
    /// </summary>
    [Fact]
    public void Sequence_order_is_preserved()
    {
        var r = Parse("[[5],[6,7],[8,9,10]]");
        Assert.True(r.Ok);
        Assert.Equal([5], r.Sequences![0]);
        Assert.Equal([6, 7], r.Sequences[1]);
        Assert.Equal([8, 9, 10], r.Sequences[2]);
    }
}
