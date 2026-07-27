using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public sealed class QwenToolFormatVerificationTests
{
    private static string FixtureDir()
    {
        // Walk up from the test bin dir to the repo root, then into scripts/lora/fixtures.
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "dotLLM.slnx")))
            dir = Path.GetDirectoryName(dir);
        Assert.NotNull(dir);
        return Path.Combine(dir!, "scripts", "lora", "fixtures");
    }

    private sealed record Inputs(JsonElement messages, JsonElement tools,
        string assistant_tool_call, string bos_token, string eos_token);

    [Fact]
    public void DotLlm_Renders_Qwen3_Tool_Template_Identically_To_Reference()
    {
        var fix = FixtureDir();
        var template = File.ReadAllText(Path.Combine(fix, "qwen3_chat_template.jinja"));
        var expected = File.ReadAllText(Path.Combine(fix, "qwen3_tooluse_reference.txt"));
        var inp = JsonSerializer.Deserialize<Inputs>(
            File.ReadAllText(Path.Combine(fix, "format_inputs.json")))!;

        using var messagesEnumerator = inp.messages.EnumerateArray();
        var messages = messagesEnumerator
            .Select(m => new ChatMessage {
                Role = m.GetProperty("role").GetString()!,
                Content = m.GetProperty("content").GetString()! })
            .ToList();
        using var toolsEnumerator = inp.tools.EnumerateArray();
        var tools = toolsEnumerator
            .Select(t => new ToolDefinition(
                t.GetProperty("name").GetString()!,
                t.GetProperty("description").GetString()!,
                t.GetProperty("parameters").GetString()!))
            .ToArray();

        var tmpl = new JinjaChatTemplate(template, inp.bos_token, inp.eos_token);
        var actual = tmpl.Apply(messages, new ChatTemplateOptions {
            AddGenerationPrompt = true, Tools = tools });

        // Normalize trailing whitespace/newlines only — content must match.
        static string N(string s) => s.Replace("\r\n", "\n").TrimEnd();
        Assert.Equal(N(expected), N(actual));
    }

    [Fact]
    public void Hermes_Parser_RoundTrips_The_Assistant_Tool_Call()
    {
        var fix = FixtureDir();
        var inp = JsonSerializer.Deserialize<Inputs>(
            File.ReadAllText(Path.Combine(fix, "format_inputs.json")))!;

        var calls = new HermesToolCallParser().TryParse(inp.assistant_tool_call);

        Assert.NotNull(calls);
        Assert.Single(calls!);
        Assert.Equal("get_weather", calls![0].FunctionName);
        using var args = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("Paris", args.RootElement.GetProperty("city").GetString());
    }
}
