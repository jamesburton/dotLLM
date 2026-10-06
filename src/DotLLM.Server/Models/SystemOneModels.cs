using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Jev-compatible <c>POST /v1/systemone</c> request (issue #708): a state plus typed questions. Same wire shape as TypeSafe's endpoint,
/// OpenJev servers, and the Microsoft Agent Framework <c>Microsoft.Agents.AI.TypeSafe</c> provider.
/// </summary>
public sealed record SystemOneRequest
{
    /// <summary>Model alias. Informational: the loaded model answers regardless (Jev clients send <c>jev-latest</c>).</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>The facts to decide about: a JSON string or any JSON value (embedded verbatim).</summary>
    [JsonPropertyName("state")]
    public JsonElement State { get; init; }

    /// <summary>Question id -> question. Evaluated in order.</summary>
    [JsonPropertyName("questions")]
    public Dictionary<string, SystemOneQuestion>? Questions { get; init; }
}

/// <summary>One typed question: <c>noul</c> (boolean probability), <c>choice</c> (named options) or <c>score</c> (ordered levels).</summary>
public sealed record SystemOneQuestion
{
    /// <summary><c>noul</c>, <c>choice</c> or <c>score</c>.</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    /// <summary>What to decide.</summary>
    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    /// <summary>noul: optional <c>{"true","false"}</c> descriptions; choice: <c>{name: description}</c>; score: an array of level descriptions.</summary>
    [JsonPropertyName("criteria")]
    public JsonElement Criteria { get; init; }
}

/// <summary>Answer to one question; only the fields of its <see cref="Type"/> are populated.</summary>
public sealed record SystemOneAnswer
{
    /// <summary><c>noul</c>, <c>choice</c> or <c>score</c>.</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    /// <summary>noul: probability the statement is true.</summary>
    [JsonPropertyName("noul")]
    public double? Noul { get; init; }

    /// <summary>choice: the most probable option's name.</summary>
    [JsonPropertyName("choice")]
    public string? Choice { get; init; }

    /// <summary>choice: option name -> probability; score: level index ("0".."n-1") -> probability.</summary>
    [JsonPropertyName("probabilities")]
    public Dictionary<string, double>? Probabilities { get; init; }

    /// <summary>choice / score: <c>1 - H(p) / ln K</c>.</summary>
    [JsonPropertyName("confidence")]
    public double? Confidence { get; init; }

    /// <summary>score: probability-weighted level index in <c>[0, n-1]</c>.</summary>
    [JsonPropertyName("score")]
    public double? Score { get; init; }

    /// <summary>score: level index -> description.</summary>
    [JsonPropertyName("legend")]
    public Dictionary<string, string>? Legend { get; init; }
}

/// <summary>Token accounting for a <c>/v1/systemone</c> call. There is no decode loop, so <see cref="OutputTokens"/> is 0.</summary>
public sealed record SystemOneUsage
{
    /// <summary>Prompt tokens across all questions.</summary>
    [JsonPropertyName("input_tokens")]
    public long InputTokens { get; init; }

    /// <summary>Always 0: answers are read from logits, nothing is generated.</summary>
    [JsonPropertyName("output_tokens")]
    public long OutputTokens { get; init; }
}

/// <summary>Jev-compatible <c>/v1/systemone</c> response.</summary>
public sealed record SystemOneResponse
{
    /// <summary>The model that answered.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>Question id -> answer.</summary>
    [JsonPropertyName("answers")]
    public Dictionary<string, SystemOneAnswer>? Answers { get; init; }

    /// <summary>Token usage.</summary>
    [JsonPropertyName("usage")]
    public SystemOneUsage? Usage { get; init; }
}
