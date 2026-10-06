using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #450: the remaining OpenAI Chat Completions request-surface gaps —
/// <c>stream_options.include_usage</c>, <c>parallel_tool_calls</c>, and the accept-and-ignore
/// fields (<c>user</c>, <c>store</c>, <c>service_tier</c>, <c>reasoning_effort</c>) that a strict
/// DTO must not 400 on.
/// </summary>
public sealed class OpenAiRequestSurfaceTests
{
    private static ChatCompletionRequest ParseChat(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    private static CompletionRequest ParseCompletion(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.CompletionRequest)!;

    // ───────────────────────── stream_options ─────────────────────────

    [Fact]
    public void ChatRequest_ParsesStreamOptionsIncludeUsage()
    {
        var request = ParseChat("""
            {"messages":[],"stream":true,"stream_options":{"include_usage":true}}
            """);

        Assert.True(request.StreamOptions?.IncludeUsage);
        Assert.True(request.WantsUsageChunk);
    }

    [Fact]
    public void ChatRequest_StreamOptionsAbsent_MeansNoUsageChunk()
    {
        Assert.False(ParseChat("""{"messages":[],"stream":true}""").WantsUsageChunk);
        Assert.False(ParseChat("""
            {"messages":[],"stream":true,"stream_options":{"include_usage":false}}
            """).WantsUsageChunk);
    }

    [Fact]
    public void CompletionRequest_ParsesStreamOptionsIncludeUsage()
    {
        Assert.True(ParseCompletion("""
            {"prompt":"hi","stream":true,"stream_options":{"include_usage":true}}
            """).WantsUsageChunk);
    }

    /// <summary>
    /// OpenAI's final usage chunk carries <c>usage</c> and an EMPTY <c>choices</c> array. SDKs
    /// key on exactly that to tell it apart from a content chunk, so the shape is load-bearing.
    /// </summary>
    [Fact]
    public void UsageChunk_HasEmptyChoicesAndUsage()
    {
        var chunk = ChatCompletionEndpoint.BuildUsageChunk("req-1", "model-x", promptTokens: 7, completionTokens: 3);

        string json = JsonSerializer.Serialize(chunk, ServerJsonContext.Default.ChatCompletionChunk);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.Equal("chat.completion.chunk", root.GetProperty("object").GetString());
        Assert.Equal(JsonValueKind.Array, root.GetProperty("choices").ValueKind);
        Assert.Equal(0, root.GetProperty("choices").GetArrayLength());

        var usage = root.GetProperty("usage");
        Assert.Equal(7, usage.GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(3, usage.GetProperty("completion_tokens").GetInt32());
        Assert.Equal(10, usage.GetProperty("total_tokens").GetInt32());
    }

    // ───────────────────────── parallel_tool_calls ─────────────────────────

    [Fact]
    public void ChatRequest_ParsesParallelToolCalls()
    {
        Assert.False(ParseChat("""{"messages":[],"parallel_tool_calls":false}""").ParallelToolCalls);
        // Absent means the OpenAI default: parallel calls allowed.
        Assert.Null(ParseChat("""{"messages":[]}""").ParallelToolCalls);
    }

    /// <summary>
    /// <c>parallel_tool_calls: false</c> means the assistant emits at most one tool call. The
    /// model is not constrained during decode, so the server enforces it on the way out.
    /// </summary>
    [Fact]
    public void ApplyParallelToolCalls_KeepsOnlyTheFirst_WhenDisabled()
    {
        var calls = new[] { Call("a"), Call("b"), Call("c") };

        var limited = ChatCompletionEndpoint.ApplyParallelToolCalls(calls, parallelToolCalls: false);

        Assert.NotNull(limited);
        Assert.Single(limited!);
        Assert.Equal("a", limited[0].FunctionName);
    }

    [Fact]
    public void ApplyParallelToolCalls_PassesThrough_WhenEnabledOrUnset()
    {
        var calls = new[] { Call("a"), Call("b") };

        Assert.Equal(2, ChatCompletionEndpoint.ApplyParallelToolCalls(calls, parallelToolCalls: true)!.Length);
        Assert.Equal(2, ChatCompletionEndpoint.ApplyParallelToolCalls(calls, parallelToolCalls: null)!.Length);
    }

    [Fact]
    public void ApplyParallelToolCalls_TolerantOfNullAndEmpty()
    {
        Assert.Null(ChatCompletionEndpoint.ApplyParallelToolCalls(null, parallelToolCalls: false));
        Assert.Empty(ChatCompletionEndpoint.ApplyParallelToolCalls([], parallelToolCalls: false)!);
    }

    private static DotLLM.Tokenizers.ToolCall Call(string name) => new(name, name, "{}");

    // ───────────────────────── accept-and-ignore ─────────────────────────

    /// <summary>
    /// A client that always sends <c>user</c>/<c>store</c>/<c>service_tier</c>/
    /// <c>reasoning_effort</c>/<c>metadata</c> must not get a 400. These are now declared on the
    /// DTO so the intent is explicit and documented rather than resting on
    /// <c>JsonUnmappedMemberHandling</c> staying at its default.
    /// </summary>
    [Fact]
    public void ChatRequest_AcceptsAndIgnoresVendorFields()
    {
        var request = ParseChat("""
            {
              "messages": [{"role":"user","content":"hi"}],
              "user": "user-123",
              "store": true,
              "service_tier": "auto",
              "reasoning_effort": "high",
              "metadata": {"trace":"abc"}
            }
            """);

        Assert.Equal("user-123", request.User);
        Assert.True(request.Store);
        Assert.Equal("auto", request.ServiceTier);
        Assert.Equal("high", request.ReasoningEffort);
        Assert.NotNull(request.Metadata);
    }

    [Fact]
    public void CompletionRequest_AcceptsAndIgnoresVendorFields()
    {
        var request = ParseCompletion("""
            {"prompt":"hi","user":"u","store":false,"service_tier":"default","reasoning_effort":"low"}
            """);

        Assert.Equal("u", request.User);
        Assert.False(request.Store);
    }

    /// <summary>
    /// The genuinely-unknown case still has to be tolerated too — an SDK version newer than this
    /// server will send fields nobody here has heard of.
    /// </summary>
    [Fact]
    public void ChatRequest_ToleratesEntirelyUnknownFields()
    {
        var request = ParseChat("""
            {"messages":[{"role":"user","content":"hi"}],"some_field_from_2027":{"nested":[1,2]}}
            """);

        Assert.Single(request.Messages);
    }
}
