using System.Text.Json;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public sealed class BitNetToolFormatVerificationTests
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

    private static string TemplateDir()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "dotLLM.slnx")))
            dir = Path.GetDirectoryName(dir);
        Assert.NotNull(dir);
        return Path.Combine(dir!, "scripts", "lora", "templates");
    }

    private sealed record Inputs(JsonElement messages, JsonElement tools,
        string assistant_tool_call, string bos_token, string eos_token);

    [Fact]
    public void DotLlm_Renders_BitNet_Tool_Template_Identically_To_Reference()
    {
        var fix = FixtureDir();
        var expected = File.ReadAllText(Path.Combine(fix, "bitnet_tooluse_reference.txt"));
        var inp = JsonSerializer.Deserialize<Inputs>(
            File.ReadAllText(Path.Combine(fix, "bitnet_format_inputs.json")))!;

        var messages = inp.messages.EnumerateArray()
            .Select(m => new ChatMessage {
                Role = m.GetProperty("role").GetString()!,
                Content = m.GetProperty("content").GetString()! })
            .ToList();
        var tools = inp.tools.EnumerateArray()
            .Select(t => new ToolDefinition(
                t.GetProperty("name").GetString()!,
                t.GetProperty("description").GetString()!,
                t.GetProperty("parameters").GetString()!))
            .ToArray();

        // Use the embedded const (approach A) — same source as bitnet_tooluse.jinja.
        var tmpl = new JinjaChatTemplate(
            GgufChatTemplateFactory.BitNetToolAwareTemplateText,
            inp.bos_token,
            inp.eos_token);
        var actual = tmpl.Apply(messages, new ChatTemplateOptions {
            AddGenerationPrompt = true, Tools = tools });

        // Normalize trailing whitespace/newlines only — content must match.
        static string N(string s) => s.Replace("\r\n", "\n").TrimEnd();
        Assert.Equal(N(expected), N(actual));
    }

    [Fact]
    public void Embedded_Const_Matches_Jinja_File_On_Disk()
    {
        // Guard that the embedded C# const stays byte-identical to the .jinja file.
        var templatePath = Path.Combine(TemplateDir(), "bitnet_tooluse.jinja");
        var fileTemplate = File.ReadAllText(templatePath);

        static string N(string s) => s.Replace("\r\n", "\n").TrimEnd();

        // Both should render the same output for a fixed input (indirection through render,
        // not raw text comparison, since the file has actual newlines and the const uses \n).
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant." },
            new() { Role = "user", Content = "What is the weather in Tokyo?" },
        };
        var tools = new[] { new ToolDefinition(
            "get_weather",
            "Get the current weather for a city.",
            "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},\"required\":[\"city\"]}") };
        const string bos = "<|begin_of_text|>";
        const string eos = "<|eot_id|>";

        var fromFile = new JinjaChatTemplate(fileTemplate, bos, eos)
            .Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });
        var fromConst = new JinjaChatTemplate(
            GgufChatTemplateFactory.BitNetToolAwareTemplateText, bos, eos)
            .Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });

        Assert.Equal(N(fromFile), N(fromConst));
    }

    [Fact]
    public void Hermes_Parser_RoundTrips_The_BitNet_Assistant_Tool_Call()
    {
        var fix = FixtureDir();
        var inp = JsonSerializer.Deserialize<Inputs>(
            File.ReadAllText(Path.Combine(fix, "bitnet_format_inputs.json")))!;

        var calls = new HermesToolCallParser().TryParse(inp.assistant_tool_call);

        Assert.NotNull(calls);
        Assert.Single(calls!);
        Assert.Equal("get_weather", calls![0].FunctionName);
        using var args = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("Tokyo", args.RootElement.GetProperty("city").GetString());
    }

    [Fact]
    public void ToolCallParserFactory_Returns_Hermes_For_BitNet()
    {
        // Confirm the parser factory routes BitNet to HermesToolCallParser.
        var parser = ToolCallParserFactory.Create(DotLLM.Core.Configuration.Architecture.BitNet);
        Assert.IsType<HermesToolCallParser>(parser);
    }
}
