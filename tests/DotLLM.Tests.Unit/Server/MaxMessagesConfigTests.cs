using System.Text;
using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #835: the request message-count cap is configurable (flag + <c>DOTLLM_MAX_MESSAGES</c>), defaults to 8192,
/// 0 = unlimited, and both chat endpoints honour it. Tests use a non-default limit (5) so a hard-coded
/// 1024 or 8192 anywhere fails them.
/// </summary>
[Collection("MaxMessagesEnv")] // ServerOptions.Parse reads the process environment
public sealed class MaxMessagesConfigTests
{
    private const int Limit = 5;

    private static ServerOptions Opts(int? max) => new() { Model = "m.gguf", MaxMessages = max };

    private static ChatCompletionRequest OpenAi(int n)
    {
        var msgs = new ChatMessageDto[n];
        for (int i = 0; i < n; i++) msgs[i] = new ChatMessageDto { Role = "user", Content = "hi" };
        return new ChatCompletionRequest { Messages = msgs };
    }

    private static AnthropicMessagesRequest Anthropic(int n)
    {
        var msgs = new StringBuilder();
        for (int i = 0; i < n; i++)
        {
            if (i > 0) msgs.Append(',');
            msgs.Append("{\"role\":\"user\",\"content\":\"hi\"}");
        }
        return JsonSerializer.Deserialize(
            "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[" + msgs + "]}",
            ServerJsonContext.Default.AnthropicMessagesRequest)!;
    }

    private static T WithEnv<T>(string? value, Func<T> body)
    {
        string? old = Environment.GetEnvironmentVariable(ServerOptions.MaxMessagesEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(ServerOptions.MaxMessagesEnvVar, value);
            return body();
        }
        finally { Environment.SetEnvironmentVariable(ServerOptions.MaxMessagesEnvVar, old); }
    }

    // ── option resolution ──

    [Fact]
    public void Default_Is8192()
    {
        Assert.Null(Opts(null).MaxMessages);
        Assert.Equal(8192, Opts(null).EffectiveMaxMessages);
        Assert.Equal(8192, ServerOptions.DefaultMaxMessages);
    }

    [Fact]
    public void Option_Overrides() => Assert.Equal(Limit, Opts(Limit).EffectiveMaxMessages);

    [Fact]
    public void Zero_IsUnlimited() => Assert.Equal(0, Opts(0).EffectiveMaxMessages);

    [Theory]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    public void Negative_Cli_Rejected(int v) =>
        Assert.Throws<ArgumentException>(() => ServerOptions.ResolveMaxMessages(v, null));

    [Theory]
    [InlineData("-3")]
    [InlineData("abc")]
    [InlineData("1.5")]
    public void Bad_Env_Rejected(string v) =>
        Assert.Throws<ArgumentException>(() => ServerOptions.ResolveMaxMessages(null, v));

    [Fact]
    public void Resolve_NeitherSet_IsNull() => Assert.Null(ServerOptions.ResolveMaxMessages(null, null));

    [Theory]
    [InlineData("")]
    [InlineData("  ")]
    public void Resolve_BlankEnv_IsNull(string v) => Assert.Null(ServerOptions.ResolveMaxMessages(null, v));

    [Fact]
    public void Resolve_EnvOnly() => Assert.Equal(7, ServerOptions.ResolveMaxMessages(null, " 7 "));

    [Fact]
    public void Resolve_CliWinsOverEnv()
    {
        Assert.Equal(3, ServerOptions.ResolveMaxMessages(3, "99"));
        Assert.Equal(0, ServerOptions.ResolveMaxMessages(0, "99"));
        // A bad env value is irrelevant once the CLI value decides.
        Assert.Equal(3, ServerOptions.ResolveMaxMessages(3, "junk"));
    }

    [Fact]
    public void Parse_Flag_Sets()
    {
        var o = WithEnv(null, () => ServerOptions.Parse(["--model", "m.gguf", "--max-messages", "42"]));
        Assert.Equal(42, o.MaxMessages);
    }

    [Fact]
    public void Parse_NoFlagNoEnv_Default()
    {
        var o = WithEnv(null, () => ServerOptions.Parse(["--model", "m.gguf"]));
        Assert.Equal(8192, o.EffectiveMaxMessages);
    }

    [Fact]
    public void Parse_EnvUsed_WhenNoFlag()
    {
        var o = WithEnv("11", () => ServerOptions.Parse(["--model", "m.gguf"]));
        Assert.Equal(11, o.MaxMessages);
    }

    [Fact]
    public void Parse_FlagWinsOverEnv()
    {
        var o = WithEnv("11", () => ServerOptions.Parse(["--model", "m.gguf", "--max-messages", "0"]));
        Assert.Equal(0, o.MaxMessages);
    }

    [Fact]
    public void Parse_NegativeFlag_Throws() =>
        Assert.Throws<ArgumentException>(() =>
            WithEnv(null, () => ServerOptions.Parse(["--model", "m.gguf", "--max-messages", "-1"])));

    // ── OpenAI endpoint validator ──

    [Fact]
    public void OpenAi_AtLimit_Passes() =>
        Assert.Null(RequestValidator.ValidateChatRequest(OpenAi(Limit), Opts(Limit).EffectiveMaxMessages));

    [Fact]
    public void OpenAi_OverLimit_Fails_NamingLimitAndFix()
    {
        var error = RequestValidator.ValidateChatRequest(OpenAi(Limit + 1), Opts(Limit).EffectiveMaxMessages);
        Assert.NotNull(error);
        Assert.Contains($"exceeds maximum of {Limit} messages", error, StringComparison.Ordinal);
        Assert.Contains($"got {Limit + 1}", error, StringComparison.Ordinal);
        Assert.Contains("--max-messages", error, StringComparison.Ordinal);
        Assert.Contains("DOTLLM_MAX_MESSAGES", error, StringComparison.Ordinal);
    }

    [Fact]
    public void OpenAi_DefaultCap_Allows1025_RejectsAbove8192()
    {
        int cap = Opts(null).EffectiveMaxMessages;
        Assert.Null(RequestValidator.ValidateChatRequest(OpenAi(1025), cap));
        Assert.Null(RequestValidator.ValidateChatRequest(OpenAi(8192), cap));
        Assert.NotNull(RequestValidator.ValidateChatRequest(OpenAi(8193), cap));
    }

    [Fact]
    public void OpenAi_Unlimited_AllowsLarge() =>
        Assert.Null(RequestValidator.ValidateChatRequest(OpenAi(20000), Opts(0).EffectiveMaxMessages));

    // ── Anthropic endpoint validator ──

    [Fact]
    public void Anthropic_AtLimit_Passes() =>
        Assert.Null(MessagesEndpoint.ValidateRequest(Anthropic(Limit), maxMessages: Opts(Limit).EffectiveMaxMessages));

    [Fact]
    public void Anthropic_OverLimit_Fails_NamingLimitAndFix()
    {
        var error = MessagesEndpoint.ValidateRequest(Anthropic(Limit + 1), maxMessages: Opts(Limit).EffectiveMaxMessages);
        Assert.NotNull(error);
        Assert.StartsWith("messages: array exceeds", error, StringComparison.Ordinal);
        Assert.Contains($"maximum of {Limit} messages", error, StringComparison.Ordinal);
        Assert.Contains("--max-messages", error, StringComparison.Ordinal);
        Assert.Contains("DOTLLM_MAX_MESSAGES", error, StringComparison.Ordinal);
    }

    [Fact]
    public void Anthropic_CountTokensRoute_HonoursLimit()
    {
        int cap = Opts(Limit).EffectiveMaxMessages;
        Assert.Null(MessagesEndpoint.ValidateRequest(Anthropic(Limit), requireMaxTokens: false, maxMessages: cap));
        Assert.NotNull(MessagesEndpoint.ValidateRequest(Anthropic(Limit + 1), requireMaxTokens: false, maxMessages: cap));
    }

    [Fact]
    public void Anthropic_DefaultAndUnlimited()
    {
        Assert.Null(MessagesEndpoint.ValidateRequest(Anthropic(1025)));
        Assert.NotNull(MessagesEndpoint.ValidateRequest(Anthropic(8193)));
        Assert.Null(MessagesEndpoint.ValidateRequest(Anthropic(9000), maxMessages: 0));
    }
}

[CollectionDefinition("MaxMessagesEnv", DisableParallelization = true)]
public sealed class MaxMessagesEnvCollection;
