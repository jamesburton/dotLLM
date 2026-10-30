using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1292: request fields map to the engine's thinking budget, the server default applies when a request is silent, and
/// max_tokens is raised to budget + answer reserve so an answer after a spent budget is not truncated.
/// </summary>
public sealed class ThinkingBudgetMappingTests
{
    private static readonly ReasoningPlan Thinking = new(ReasoningFormat.Auto, promptOpened: true);

    [Fact]
    public void OpenAiRequest_ReadsThinkingBudgetTokens_AndItsAlias_AndTheMessage()
    {
        var a = JsonSerializer.Deserialize("""{"messages":[],"thinking_budget_tokens":64,"reasoning_budget_message":"Time's up."}""",
            ServerJsonContext.Default.ChatCompletionRequest)!;
        var b = JsonSerializer.Deserialize("""{"messages":[],"reasoning_budget_tokens":32}""", ServerJsonContext.Default.ChatCompletionRequest)!;
        var none = JsonSerializer.Deserialize("""{"messages":[]}""", ServerJsonContext.Default.ChatCompletionRequest)!;

        Assert.Equal(64, a.ThinkingBudgetTokens);
        Assert.Equal("Time's up.", a.ReasoningBudgetMessage);
        Assert.Equal(32, b.ReasoningBudgetTokens);
        Assert.Null(none.ThinkingBudgetTokens);
        Assert.Null(none.ReasoningBudgetTokens);
    }

    [Fact]
    public void AnthropicRequest_EnforcesBudgetTokens_OnlyWhenThinkingIsEnabled()
    {
        var on = JsonSerializer.Deserialize("""{"model":"m","max_tokens":10,"messages":[],"thinking":{"type":"enabled","budget_tokens":2048}}""",
            ServerJsonContext.Default.AnthropicMessagesRequest)!;
        var off = JsonSerializer.Deserialize("""{"model":"m","max_tokens":10,"messages":[],"thinking":{"type":"disabled","budget_tokens":2048}}""",
            ServerJsonContext.Default.AnthropicMessagesRequest)!;

        Assert.Equal(2048, on.ThinkingBudgetTokens);
        Assert.Null(off.ThinkingBudgetTokens);
    }

    [Fact]
    public void Plan_RequestWins_ServerDefaultFillsIn_NegativeIsUnlimited_AndNoSplitMeansNoBudget()
    {
        Assert.Equal(100, Thinking.ThinkingBudget(100, serverDefault: 5));
        Assert.Equal(5, Thinking.ThinkingBudget(null, serverDefault: 5));
        Assert.Null(Thinking.ThinkingBudget(-1, serverDefault: 5));
        Assert.Null(Thinking.ThinkingBudget(null, serverDefault: -1));
        Assert.Null(ReasoningPlan.Disabled.ThinkingBudget(100, serverDefault: -1));
    }

    [Fact]
    public void Plan_InstallsTheBudgetWithTheModelsTags()
    {
        var options = Thinking.WithThinkingBudget(new InferenceOptions(), 64, "Done.");

        Assert.Equal(new ThinkingBudget(64, "<think>", "</think>", StartsInside: true, "Done."), options.ThinkingBudget);
        Assert.Null(Thinking.WithThinkingBudget(new InferenceOptions(), null, "Done.").ThinkingBudget);
    }

    [Theory]
    [InlineData(512, 1000, 8192, 9192)]   // raised: budget + reserve
    [InlineData(20000, 1000, 8192, 20000)] // already larger: kept
    [InlineData(512, 1000, 0, 512)]       // reserve 0 = llama.cpp: budget counts inside max_tokens
    [InlineData(512, 0, 8192, 512)]       // budget 0 closes at once: nothing to reserve for
    public void AnswerReserve_RaisesMaxTokens(int maxTokens, int budget, int reserve, int expected)
        => Assert.Equal(expected, ReasoningSupport.WithAnswerReserve(maxTokens, budget, reserve));

    [Fact]
    public void AnswerReserve_NoBudget_LeavesMaxTokens()
        => Assert.Equal(512, ReasoningSupport.WithAnswerReserve(512, null, 8192));
}
