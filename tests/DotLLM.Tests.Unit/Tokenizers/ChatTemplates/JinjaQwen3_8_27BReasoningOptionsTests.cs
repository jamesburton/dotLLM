using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// #767: the reasoning switches reach the template through <see cref="ChatTemplateOptions"/> (not a raw
/// context dict), and the output is byte-identical to the reference Python Jinja2 goldens in
/// <c>qwen3.8-27b-render-cases.json</c> (the same fixture <see cref="JinjaQwen3_8_27BReferenceRenderTests"/> uses).
/// A raw-dict test cannot see a plumbing bug such as a null-valued <c>enable_thinking</c> key, which is
/// <i>defined</i> to the template and silently turns thinking off.
/// </summary>
public class JinjaQwen3_8_27BReasoningOptionsTests
{
    private static string Fixture(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    private static readonly JinjaChatTemplate Template =
        new(Fixture("qwen3.8-27b-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");

    /// <summary>Golden cases expressible without tool definitions.</summary>
    public static IEnumerable<object[]> CaseNames()
    {
        using var doc = JsonDocument.Parse(Fixture("qwen3.8-27b-render-cases.json"));
        foreach (var c in doc.RootElement.EnumerateArray())
        {
            var ctx = c.GetProperty("context");
            if (ctx.TryGetProperty("tools", out _)) continue;
            yield return [c.GetProperty("name").GetString()!];
        }
    }

    private static JsonElement Case(JsonDocument doc, string name) =>
        doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);

    private static ChatMessage[] Messages(JsonElement ctx) =>
        ctx.GetProperty("messages").EnumerateArray().Select(m => new ChatMessage
        {
            Role = m.GetProperty("role").GetString()!,
            Content = m.GetProperty("content").GetString()!,
            ReasoningContent = m.TryGetProperty("reasoning_content", out var r) ? r.GetString() : null,
        }).ToArray();

    private static ChatTemplateOptions ExplicitOptions(JsonElement ctx) => new()
    {
        AddGenerationPrompt = ctx.GetProperty("add_generation_prompt").GetBoolean(),
        EnableThinking = ctx.TryGetProperty("enable_thinking", out var e) ? e.GetBoolean() : null,
        ReasoningEffort = ctx.TryGetProperty("reasoning_effort", out var r) ? r.GetString() : null,
        PreserveThinking = ctx.TryGetProperty("preserve_thinking", out var p) ? p.GetBoolean() : null,
    };

    // The same switches, delivered as chat_template_kwargs instead of the typed members.
    private static ChatTemplateOptions KwargsOptions(JsonElement ctx)
    {
        var kwargs = new Dictionary<string, JsonElement>();
        foreach (var key in new[] { "enable_thinking", "reasoning_effort", "preserve_thinking" })
            if (ctx.TryGetProperty(key, out var v)) kwargs[key] = v.Clone();
        return new ChatTemplateOptions
        {
            AddGenerationPrompt = ctx.GetProperty("add_generation_prompt").GetBoolean(),
            TemplateKwargs = kwargs,
        };
    }

    private static void AssertMatches(JsonElement c, ChatTemplateOptions options, JsonElement ctx)
    {
        var messages = Messages(ctx);
        if (c.TryGetProperty("expectedError", out var err))
        {
            var ex = Assert.ThrowsAny<Exception>(() => Template.Apply(messages, options));
            Assert.Contains(err.GetString()!, ex.Message, StringComparison.Ordinal);
        }
        else
        {
            Assert.Equal(c.GetProperty("expected").GetString(), Template.Apply(messages, options));
        }
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void TypedOptions_MatchReferenceJinja2(string caseName)
    {
        using var doc = JsonDocument.Parse(Fixture("qwen3.8-27b-render-cases.json"));
        var c = Case(doc, caseName);
        var ctx = c.GetProperty("context");
        AssertMatches(c, ExplicitOptions(ctx), ctx);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void ChatTemplateKwargs_MatchReferenceJinja2(string caseName)
    {
        using var doc = JsonDocument.Parse(Fixture("qwen3.8-27b-render-cases.json"));
        var c = Case(doc, caseName);
        var ctx = c.GetProperty("context");
        AssertMatches(c, KwargsOptions(ctx), ctx);
    }

    [Fact]
    public void EnableThinkingFalse_ClosesTheThinkBlock_DefaultOpensIt()
    {
        ChatMessage[] msgs = [new() { Role = "user", Content = "hi" }];
        Assert.EndsWith("<|im_start|>assistant\n<think>\n",
            Template.Apply(msgs, new ChatTemplateOptions()), StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n",
            Template.Apply(msgs, new ChatTemplateOptions { EnableThinking = false }), StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n",
            Template.Apply(msgs, new ChatTemplateOptions { EnableThinking = true }), StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitMember_WinsOverSameNamedKwarg()
    {
        ChatMessage[] msgs = [new() { Role = "user", Content = "hi" }];
        using var doc = JsonDocument.Parse("""{"enable_thinking": true}""");
        var kwargs = new Dictionary<string, JsonElement> { ["enable_thinking"] = doc.RootElement.GetProperty("enable_thinking").Clone() };
        string off = Template.Apply(msgs, new ChatTemplateOptions { EnableThinking = false, TemplateKwargs = kwargs });
        Assert.EndsWith("</think>\n\n", off, StringComparison.Ordinal);
    }

    [Fact]
    public void Kwargs_CannotOverrideStructuralVariables()
    {
        ChatMessage[] msgs = [new() { Role = "user", Content = "real question" }];
        using var doc = JsonDocument.Parse("""{"messages": [], "add_generation_prompt": false, "bos_token": "X"}""");
        var kwargs = doc.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone());
        string prompt = Template.Apply(msgs, new ChatTemplateOptions { TemplateKwargs = kwargs });
        Assert.Contains("real question", prompt, StringComparison.Ordinal);
        Assert.EndsWith("<think>\n", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void UnsetOptions_LeaveVariablesUndefined_NotNull()
    {
        // enable_thinking is undefined => thinking ON. A null-valued key would be "defined" and not `true`.
        ChatMessage[] msgs = [new() { Role = "user", Content = "q" }];
        string prompt = Template.Apply(msgs, new ChatTemplateOptions { EnableThinking = null, ReasoningEffort = null });
        Assert.EndsWith("<think>\n", prompt, StringComparison.Ordinal);
        Assert.Contains("Reasoning effort is set to xhigh", prompt, StringComparison.Ordinal);
    }
}
