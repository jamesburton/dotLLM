using System.Text;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Endpoint tests for <c>POST /v1/messages/count_tokens</c> (#449). The handler is driven
/// directly with a <see cref="DefaultHttpContext"/> and a <see cref="ServerState"/> whose
/// tokenizer and chat template are fakes, so no model is loaded: counting needs neither
/// weights nor a forward pass, which is precisely the property under test.
/// </summary>
public sealed class AnthropicCountTokensTests
{
    // --- fakes ---------------------------------------------------------------

    /// <summary>Counts whitespace-separated words, so a count is predictable from the prompt.</summary>
    private sealed class WordTokenizer : ITokenizer
    {
        public int Calls;

        public int VocabSize => 32;
        public int BosTokenId => 1;
        public int EosTokenId => 2;

        public int[] Encode(string text)
        {
            Calls++;
            int n = Split(text).Length;
            var ids = new int[n];
            for (int i = 0; i < n; i++) ids[i] = 3 + (i % 8);
            return ids;
        }

        public string Decode(ReadOnlySpan<int> tokenIds) => new('x', tokenIds.Length);
        public string DecodeToken(int tokenId) => "x";
        public int CountTokens(string text) => Encode(text).Length;

        private static string[] Split(string text) =>
            text.Split([' ', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>Renders <c>role: content</c> lines — enough to see what reached the prompt.</summary>
    private sealed class EchoChatTemplate : IChatTemplate
    {
        public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
        {
            var sb = new StringBuilder();
            foreach (var m in messages)
                sb.Append(m.Role).Append(": ").Append(m.Content).Append('\n');
            if (options.Tools is { Length: > 0 })
                foreach (var t in options.Tools)
                    sb.Append("tooldef: ").Append(t.Name).Append('\n');
            if (options.AddGenerationPrompt)
                sb.Append("assistant:\n");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Satisfies <c>ServerState.Model is not null</c> (the residency "already active" check).
    /// Every compute member throws: counting tokens must never reach the model.
    /// </summary>
    private sealed class NonComputingModel : IModel
    {
        public ModelConfig Config { get; } = new()
        {
            Architecture = Architecture.Llama,
            VocabSize = 32,
            HiddenSize = 8,
            IntermediateSize = 16,
            NumLayers = 1,
            NumAttentionHeads = 2,
            NumKvHeads = 1,
            HeadDim = 4,
            MaxSequenceLength = 512,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) =>
            throw new InvalidOperationException("count_tokens must not run a forward pass");

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) =>
            throw new InvalidOperationException("count_tokens must not run a forward pass");

        public void Dispose() { }
    }

    // --- harness -------------------------------------------------------------

    private sealed record Harness(ServerState State, WordTokenizer Tokenizer, EchoChatTemplate Template);

    private static Harness NewState(bool ready = true)
    {
        var tokenizer = new WordTokenizer();
        var template = new EchoChatTemplate();
        var model = ready ? new NonComputingModel() : null;
        var state = new ServerState
        {
            // "none" is the sentinel EnsureActiveAsync treats as "nothing to activate", so the
            // not-ready harness fails activation instead of hitting the model resolver.
            Options = ready
                ? new ServerOptions { Model = "test-model", ModelId = "test-model" }
                : new ServerOptions { Model = "none", ModelId = "none" },
            Model = model,
            Config = model?.Config,
            Tokenizer = tokenizer,
            ChatTemplate = template,
            IsReady = ready,
        };
        return new Harness(state, tokenizer, template);
    }

    private static async Task<(int Status, JsonDocument Body)> PostAsync(
        Harness h, string json, params (string Name, string Value)[] headers)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "POST";
        ctx.Request.Path = "/v1/messages/count_tokens";
        foreach (var (name, value) in headers)
            ctx.Request.Headers.Append(name, value);

        var body = new MemoryStream();
        ctx.Response.Body = body;

        var request = JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;
        await MessagesEndpoint.HandleCountTokensAsync(request, h.State, ctx);

        body.Position = 0;
        return (ctx.Response.StatusCode, JsonDocument.Parse(body));
    }

    private static string Prompt(Harness h, params ChatMessage[] messages) =>
        h.Template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

    // --- tests ---------------------------------------------------------------

    [Fact]
    public async Task CountTokens_ReturnsTokenCountOfTheTemplatedPrompt()
    {
        var h = NewState();
        var (status, body) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"one two three four"}]}
        """);

        Assert.Equal(200, status);
        int expected = h.Tokenizer.CountTokens(
            Prompt(h, new ChatMessage { Role = "user", Content = "one two three four" }));
        // "user:" + 4 words + "assistant:" == 6 under WordTokenizer; the literal guards against
        // the expectation and the implementation drifting together through a shared helper.
        Assert.Equal(6, expected);
        Assert.Equal(expected, body.RootElement.GetProperty("input_tokens").GetInt32());
    }

    [Fact]
    public async Task CountTokens_CountsTheSystemPromptAndPriorTurns()
    {
        // The discriminating case: a body whose count differs only because system/history
        // reached the prompt. A handler that counted just the last user message would pass
        // the previous test and fail this one.
        var h = NewState();
        var (_, withoutSystem) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"one two three four"}]}
        """);
        var (_, withSystem) = await PostAsync(h, """
        {"model":"test-model",
         "system":[{"type":"text","text":"be terse and kind"},{"type":"text","text":"answer briefly"}],
         "messages":[{"role":"user","content":"hello there"},
                     {"role":"assistant","content":"hi"},
                     {"role":"user","content":"one two three four"}]}
        """);

        int a = withoutSystem.RootElement.GetProperty("input_tokens").GetInt32();
        int b = withSystem.RootElement.GetProperty("input_tokens").GetInt32();

        // system: 4 + 2 words + "system:" ... computed exactly rather than "greater than".
        int expected = h.Tokenizer.CountTokens(Prompt(h,
            new ChatMessage { Role = "system", Content = "be terse and kind\nanswer briefly" },
            new ChatMessage { Role = "user", Content = "hello there" },
            new ChatMessage { Role = "assistant", Content = "hi" },
            new ChatMessage { Role = "user", Content = "one two three four" }));
        Assert.Equal(expected, b);
        Assert.True(b > a, $"system + history must raise the count ({b} vs {a})");
    }

    [Fact]
    public async Task CountTokens_CountsToolDefinitions()
    {
        var h = NewState();
        var (_, withoutTools) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"weather?"}]}
        """);
        var (status, withTools) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"weather?"}],
         "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}
        """);

        Assert.Equal(200, status);
        Assert.True(
            withTools.RootElement.GetProperty("input_tokens").GetInt32() >
            withoutTools.RootElement.GetProperty("input_tokens").GetInt32(),
            "tool definitions are part of the prompt and must be counted");
    }

    [Fact]
    public async Task CountTokens_MatchesWhatTheGeneratingRouteWouldBill()
    {
        // The invariant the SDK's users rely on:
        //   count_tokens(body).input_tokens == messages.create(body).usage.input_tokens
        // /v1/messages takes its input_tokens from ValidatePromptLength over the same prompt.
        var h = NewState();
        const string json = """
        {"model":"test-model","max_tokens":16,
         "system":"be terse",
         "messages":[{"role":"user","content":"count these words please"}]}
        """;
        var (_, body) = await PostAsync(h, json);

        var request = JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;
        string prompt = h.Template.Apply(
            AnthropicConverter.ToMessages(request),
            new ChatTemplateOptions { AddGenerationPrompt = true });
        Assert.Null(RequestValidator.ValidatePromptLength(
            prompt, h.Tokenizer, h.State.Config!.MaxSequenceLength, 16, out _, out int billed));

        Assert.Equal(billed, body.RootElement.GetProperty("input_tokens").GetInt32());
    }

    [Fact]
    public async Task CountTokens_WithoutMaxTokens_Succeeds()
    {
        // The count_tokens body has no max_tokens at all (MessageCountTokensParams in the
        // official SDK), so the /v1/messages "max_tokens: field required" rule must not apply.
        var h = NewState();
        var (status, body) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"hi there"}]}
        """);

        Assert.Equal(200, status);
        Assert.True(body.RootElement.GetProperty("input_tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task CountTokens_InvalidBody_IsRejectedWithTheAnthropicErrorEnvelope()
    {
        var h = NewState();
        var (status, body) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"system","content":"nope"}]}
        """);

        Assert.Equal(400, status);
        Assert.Equal("error", body.RootElement.GetProperty("type").GetString());
        Assert.Equal("invalid_request_error", body.RootElement.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task CountTokens_UnknownAnthropicVersion_IsRejectedBeforeTokenizing()
    {
        var h = NewState();
        var (status, body) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """, ("anthropic-version", "2099-01-01"));

        Assert.Equal(400, status);
        Assert.Equal("invalid_request_error", body.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("anthropic-version", body.RootElement.GetProperty("error").GetProperty("message").GetString()!);
        // Rejected before any work was done on the request body.
        Assert.Equal(0, h.Tokenizer.Calls);
    }

    [Fact]
    public async Task CountTokens_SupportedVersionAndUnknownBeta_Succeeds()
    {
        var h = NewState();
        var (status, _) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """, ("anthropic-version", "2023-06-01"), ("anthropic-beta", "made-up-beta-2030-01-01"));

        Assert.Equal(200, status);
    }

    [Fact]
    public async Task CountTokens_ApiKeyHeaderOnly_IsAccepted()
    {
        // The official SDK authenticates with x-api-key and sends no Authorization header.
        // dotLLM has no auth layer, so this must simply work — the test exists so that adding
        // one later cannot silently lock the Anthropic SDK out of this surface.
        var h = NewState();
        var (status, _) = await PostAsync(h, """
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """, ("anthropic-version", "2023-06-01"), ("x-api-key", "sk-ant-whatever"));

        Assert.Equal(200, status);
    }

    [Fact]
    public async Task CountTokens_NoModelLoaded_FailsWithTheAnthropicErrorEnvelope()
    {
        var h = NewState(ready: false);
        var (status, body) = await PostAsync(h, """
        {"messages":[{"role":"user","content":"hi"}]}
        """);

        // Whatever the status, the body must be the Anthropic envelope: an SDK client parses
        // {"type":"error","error":{...}} and throws on the OpenAI {"error":"..."} shape.
        Assert.True(status is 400 or 503, $"status={status}");
        Assert.Equal("error", body.RootElement.GetProperty("type").GetString());
        Assert.False(string.IsNullOrEmpty(
            body.RootElement.GetProperty("error").GetProperty("message").GetString()));
        Assert.Equal(0, h.Tokenizer.Calls);
    }
}
