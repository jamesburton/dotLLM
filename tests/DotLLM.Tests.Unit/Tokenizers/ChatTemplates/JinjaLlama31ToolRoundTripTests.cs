using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// #771 (Llama-3.1-8B tool round trip). The harness saw the model "ignore the tool result"; the diagnosis was
/// whether our template rendering differs from the authority. The golden tail below is byte-identical to what BOTH
/// reference Jinja2 3.1.6 and llama.cpp's <c>/apply-template</c> (b9672, minja) produce for the same messages.
/// The tool message is a JSON STRING, and the template does
/// <c>{% if message.content is mapping or message.content is iterable %}{{ message.content | tojson }}</c>:
/// in Jinja2 a string IS iterable, so the result is rendered as a quoted, escaped JSON string. dotLLM used to
/// answer "not iterable" for strings and rendered it raw, AND kept the template source's trailing newline
/// (jinja2 keep_trailing_newline=False drops it), so every Llama-3.x prompt ended in "\n\n\n". With both fixed the
/// 8B Q4_K_M round trip answers "...17 degrees Celsius and there is light rain" (it used to ignore the result).
/// </summary>
public class JinjaLlama31ToolRoundTripTests
{
    private static string Template(string name) => File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    private static readonly ToolDefinition[] Tools =
    [
        new("get_weather", "Get the current weather for a city.",
            """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""),
    ];

    [Fact]
    public void ToolRoundTrip_TailMatchesReferenceJinja2AndLlamaCpp()
    {
        var template = new JinjaChatTemplate(Template("llama-3.1-chat-template.jinja"), "<|begin_of_text|>", "<|eot_id|>");
        var messages = new[]
        {
            new ChatMessage { Role = "user", Content = "What is the weather in Paris right now? Use the tool." },
            new ChatMessage
            {
                Role = "assistant", Content = null!,
                ToolCalls = [new ToolCall("call_0", "get_weather", """{"city":"Paris"}""")],
            },
            new ChatMessage { Role = "tool", ToolCallId = "call_0", Content = """{"temp_c": 17, "conditions": "light rain"}""" },
        };

        string prompt = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = Tools });

        const string expectedTail =
            "What is the weather in Paris right now? Use the tool.<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n" +
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eot_id|>" +
            "<|start_header_id|>ipython<|end_header_id|>\n\n" +
            "\"{\\\"temp_c\\\": 17, \\\"conditions\\\": \\\"light rain\\\"}\"<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
        Assert.EndsWith(expectedTail, prompt, StringComparison.Ordinal);
    }

    /// <summary>
    /// Jinja2's keep_trailing_newline=False: llama.cpp and HF drop ONE trailing newline of the template source.
    /// Llama-3.x's GGUF template ends "{%- endif %}\n"; keeping it appended a third "\n" after the generation prompt.
    /// </summary>
    [Theory]
    [InlineData("a{{ 'b' }}\n", "ab")]
    [InlineData("a{{ 'b' }}\r\n", "ab")]
    [InlineData("a{{ 'b' }}\n\n", "ab\n")]   // only ONE is dropped
    [InlineData("a{{ 'b' }}", "ab")]
    public void OneTrailingNewlineOfTheTemplateSource_IsDropped(string source, string expected)
        => Assert.Equal(expected, new JinjaChatTemplate(source, "", "")
            .Apply([new ChatMessage { Role = "user", Content = "x" }], new ChatTemplateOptions()));

    [Fact]
    public void Llama31_GenerationPrompt_EndsWithExactlyTwoNewlines()
    {
        var template = new JinjaChatTemplate(Template("llama-3.1-chat-template.jinja"), "<|begin_of_text|>", "<|eot_id|>");
        string prompt = template.Apply([new ChatMessage { Role = "user", Content = "hi" }],
            new ChatTemplateOptions { AddGenerationPrompt = true });
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", prompt, StringComparison.Ordinal);
        Assert.False(prompt.EndsWith("\n\n\n", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("{{ 'abc' is iterable }}", "True")]
    [InlineData("{{ [1] is iterable }}", "True")]
    [InlineData("{{ 5 is iterable }}", "False")]
    [InlineData("{{ 'abc' is mapping }}", "False")]
    public void IterableTest_FollowsJinja2(string source, string expected)
    {
        var template = new JinjaChatTemplate(source, "", "");
        Assert.Equal(expected, template.Apply([new ChatMessage { Role = "user", Content = "x" }], new ChatTemplateOptions()));
    }
}
