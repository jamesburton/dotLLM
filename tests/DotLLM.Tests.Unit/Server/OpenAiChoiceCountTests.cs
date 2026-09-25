using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #460. <c>n</c> was accepted, bound, and never read — a search across <c>src/</c> for any use of
/// the property returned nothing — so <c>n: 3</c> silently produced one choice with no error and
/// no indication the request had not been honoured. Same class of defect as #456
/// (<c>tool_choice</c> parsed and discarded).
/// </summary>
/// <remarks>
/// <para>
/// It also carried the #462 source-generation trap: as an initializer <c>= 1</c> on a record with
/// a <c>required</c> member, <c>N</c> deserialized as <b>0</b>, not 1. Invisible only because
/// nothing read it — the first person to implement <c>n</c> on the old shape would have written a
/// zero-choice loop. <c>N</c> is now <c>int?</c> with the default resolved in code.
/// </para>
/// <para>
/// <b>What these tests do not cover.</b> That a served request actually returns <c>n</c> choices
/// needs a loaded model, so it belongs to the live conformance matrix, not here. These pin the
/// three pieces that can be got wrong silently: the parsed default, the range gate, and the
/// per-choice seed — plus the usage arithmetic, which a client reconciles spend against.
/// </para>
/// </remarks>
public sealed class OpenAiChoiceCountTests
{
    private static ChatCompletionRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    /// <summary>
    /// Criterion 3 of the issue: state what <c>N</c> actually deserializes to when <c>n</c> is
    /// omitted. It is <see langword="null"/>, and the effective count resolves to 1 in code —
    /// never via an initializer, which source-generation would drop.
    /// </summary>
    [Fact]
    public void NOmitted_DeserializesAsNull_AndResolvesToOneChoice()
    {
        var request = Parse("""{"messages":[{"role":"user","content":"hi"}]}""");

        Assert.Null(request.N);
        Assert.Equal(1, request.ChoiceCount);
    }

    [Fact]
    public void NSupplied_IsCarriedThrough()
    {
        var request = Parse("""{"messages":[{"role":"user","content":"hi"}],"n":3}""");

        Assert.Equal(3, request.N);
        Assert.Equal(3, request.ChoiceCount);
    }

    /// <summary>
    /// A value this server cannot honour is refused rather than quietly reinterpreted — which is
    /// the whole defect. Against the pre-fix validator every one of these was accepted.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(RequestValidator.MaxChoices + 1)]
    [InlineData(1000)]
    public void OutOfRangeN_IsRejected(int n)
    {
        var request = Parse($$"""{"messages":[{"role":"user","content":"hi"}],"n":{{n}}}""");

        string? error = RequestValidator.ValidateChatRequest(request);

        Assert.NotNull(error);
        Assert.Contains("n", error, System.StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(RequestValidator.MaxChoices)]
    public void InRangeN_IsAccepted(int n)
    {
        var request = Parse($$"""{"messages":[{"role":"user","content":"hi"}],"n":{{n}}}""");

        Assert.Null(RequestValidator.ValidateChatRequest(request));
    }

    // ───────────────────────── per-choice sampling ─────────────────────────

    /// <summary>
    /// A seeded request must not return <c>n</c> identical completions. Choice 0 keeps the
    /// caller's seed exactly, so single-choice determinism — which callers do depend on — is
    /// untouched, and later choices offset it.
    /// </summary>
    [Fact]
    public void SeededRequest_GivesEachChoiceADistinctSeed_ButChoiceZeroKeepsTheCallersSeed()
    {
        var options = new InferenceOptions { Seed = 42 };

        Assert.Equal(42, ChatCompletionEndpoint.SeedForChoice(options, 0).Seed);

        var seen = new System.Collections.Generic.HashSet<int>();
        for (int i = 0; i < RequestValidator.MaxChoices; i++)
        {
            int? seed = ChatCompletionEndpoint.SeedForChoice(options, i).Seed;
            Assert.NotNull(seed);
            Assert.True(seen.Add(seed!.Value), $"choice {i} reused seed {seed}");
        }
    }

    /// <summary>
    /// An unseeded request already varies per choice — the sampler pipeline builds a fresh
    /// <c>Random</c> when <c>Seed</c> is null — so nothing is invented for it. Inventing a seed
    /// here would make an explicitly non-deterministic request deterministic.
    /// </summary>
    [Fact]
    public void UnseededRequest_StaysUnseeded()
    {
        var options = new InferenceOptions { Seed = null };

        for (int i = 0; i < 4; i++)
            Assert.Null(ChatCompletionEndpoint.SeedForChoice(options, i).Seed);
    }

    // ───────────────────────── usage arithmetic ─────────────────────────

    /// <summary>
    /// OpenAI counts the prompt <b>once</b> however many choices were produced, and sums
    /// completion tokens across them. Getting this wrong overstates a caller's spend by a factor
    /// of <c>n</c> on the prompt, which for a long prompt and a short answer is most of the bill.
    /// </summary>
    [Fact]
    public void MultiChoiceUsage_CountsThePromptOnceAndSumsCompletions()
    {
        InferenceResponse[] results =
        [
            Response(promptTokens: 100, generated: 7),
            Response(promptTokens: 100, generated: 11),
            Response(promptTokens: 100, generated: 3),
        ];

        var usage = ChatCompletionEndpoint.BuildMultiChoiceUsage(results);

        Assert.Equal(100, usage.PromptTokens);
        Assert.Equal(21, usage.CompletionTokens);
        Assert.Equal(121, usage.TotalTokens);
    }

    [Fact]
    public void SingleChoiceUsage_IsUnchanged()
    {
        var usage = ChatCompletionEndpoint.BuildMultiChoiceUsage([Response(promptTokens: 9, generated: 4)]);

        Assert.Equal(9, usage.PromptTokens);
        Assert.Equal(4, usage.CompletionTokens);
        Assert.Equal(13, usage.TotalTokens);
    }

    private static InferenceResponse Response(int promptTokens, int generated) => new()
    {
        GeneratedTokenIds = new int[generated],
        Text = new string('x', generated),
        FinishReason = FinishReason.Stop,
        PromptTokenCount = promptTokens,
        GeneratedTokenCount = generated,
    };
}
