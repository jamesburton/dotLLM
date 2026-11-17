using System.Globalization;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Hf;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

public sealed class HfChatTemplateFactoryTests : IDisposable
{
    private const string Template = "{{ bos_token }}{% for message in messages %}{{ message.role + ':' + message.content + eos_token }}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";
    private const string ToolTemplate = "{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}<tool_call>" + Template;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"dotllm-hf-template-{Guid.NewGuid():N}");
    private static readonly ChatMessage[] Messages =
    [
        new() { Role = "system", Content = "A" },
        new() { Role = "user", Content = "B" },
        new() { Role = "assistant", Content = "C" },
    ];
    private static readonly ToolDefinition[] Tools =
    [
        new("weather", "City weather", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""),
    ];

    [Fact]
    public void StringTemplate_UsesHfTokenizerIdsAndExactTurnFormatting()
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(Template);
        var template = HfChatTemplateFactory.TryCreate(_directory, tokenizer)!;
        string prompt = template.Apply(Messages, new ChatTemplateOptions { AddGenerationPrompt = true });
        const string expected = "<s>system:A</s>user:B</s>assistant:C</s>assistant:";
        Assert.Equal(expected, prompt);
        Assert.Equal(300, tokenizer.BosTokenId);
        Assert.Equal(301, tokenizer.EosTokenId);
        int[] expectedIds = [300, .. AsciiIds("system:A"), 301, .. AsciiIds("user:B"), 301,
            .. AsciiIds("assistant:C"), 301, .. AsciiIds("assistant:")];
        Assert.Equal(expectedIds, tokenizer.Encode(prompt));
        Assert.Equal("<s>system:A</s>user:B</s>assistant:C</s>",
            template.Apply(Messages, new ChatTemplateOptions { AddGenerationPrompt = false }));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NamedTemplates_SelectToolUseAndWireDefinitionsAndParser(bool list)
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(list
            ? new[] { new { name = "default", template = "DEFAULT:" + Template }, new { name = "tool_use", template = ToolTemplate } }
            : (object)new Dictionary<string, string>(StringComparer.Ordinal) { ["default"] = "DEFAULT:" + Template, ["tool_use"] = ToolTemplate });
        var plain = HfChatTemplateFactory.TryCreate(_directory, tokenizer)!;
        var selected = HfChatTemplateFactory.TryCreate(_directory, tokenizer, useTools: true)!;
        Assert.StartsWith("DEFAULT:<s>", plain.Apply(Messages, new ChatTemplateOptions()), StringComparison.Ordinal);
        Assert.False(plain.SupportsTools);
        Assert.True(selected.SupportsTools);
        string prompt = selected.Apply(Messages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = Tools });
        Assert.StartsWith("<tools>", prompt, StringComparison.Ordinal);
        int end = prompt.IndexOf("</tools>", StringComparison.Ordinal);
        using var declarations = JsonDocument.Parse(prompt[7..end]);
        var function = declarations.RootElement[0].GetProperty("function");
        Assert.Equal("weather", function.GetProperty("name").GetString());
        Assert.Equal("City weather", function.GetProperty("description").GetString());
        Assert.Equal("string", function.GetProperty("parameters").GetProperty("properties").GetProperty("city").GetProperty("type").GetString());
        var parser = ToolCallParserFactory.Create(Architecture.Llama, selected.TemplateSource);
        Assert.IsType<XmlToolCallParser>(parser);
        var call = Assert.Single(parser.TryParse("""<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>""", Tools)!);
        Assert.Equal("weather", call.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", call.Arguments);
    }

    [Fact]
    public void NamedCollectionWithoutToolUse_UsesDefaultForTools()
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(new Dictionary<string, string>(StringComparer.Ordinal) { ["default"] = ToolTemplate });
        var template = HfChatTemplateFactory.TryCreate(_directory, tokenizer, useTools: true)!;
        Assert.Equal(ToolTemplate, template.TemplateSource);
        Assert.True(template.SupportsTools);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StandaloneTemplates_OverrideEmbeddedEntriesWhenPresent(bool hasConfig)
    {
        var tokenizer = WriteTokenizer();
        if (hasConfig)
            WriteConfig(new[] { new { name = "default", template = "OLD" }, new { name = "tool_use", template = "OLD TOOL" } });
        File.WriteAllText(Path.Combine(_directory, "chat_template.jinja"), Template);
        string additional = Path.Combine(_directory, "additional_chat_templates");
        Directory.CreateDirectory(additional);
        File.WriteAllText(Path.Combine(additional, "tool_use.jinja"), ToolTemplate);
        Assert.Equal(Template, HfChatTemplateFactory.TryCreate(_directory, tokenizer)!.TemplateSource);
        Assert.Equal(ToolTemplate, HfChatTemplateFactory.TryCreate(_directory, tokenizer, useTools: true)!.TemplateSource);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("{}")]
    [InlineData("{\"chat_template\":null}")]
    [InlineData("{\"chat_template\":\"\"}")]
    public void MissingTemplate_ReturnsNullForExistingCallerFallbacks(string? config)
    {
        var tokenizer = WriteTokenizer();
        if (config is not null) File.WriteAllText(Path.Combine(_directory, "tokenizer_config.json"), config);
        Assert.Null(HfChatTemplateFactory.TryCreate(_directory, tokenizer));
        Assert.Equal("User: B\nAssistant: ", GgufChatTemplateFactory.CreatePlainFallback(tokenizer).Apply(
            [new() { Role = "user", Content = "B" }], new ChatTemplateOptions { AddGenerationPrompt = true }));
    }

    [Theory]
    [InlineData("{")]
    [InlineData("[]")]
    [InlineData("{\"chat_template\":7}")]
    [InlineData("{\"chat_template\":true}")]
    [InlineData("{\"chat_template\":[]}")]
    [InlineData("{\"chat_template\":{}}")]
    [InlineData("{\"chat_template\":{\"custom\":\"hello\"}}")]
    [InlineData("{\"chat_template\":{\"default\":7}}")]
    [InlineData("{\"chat_template\":[{\"name\":\"default\"}]}")]
    [InlineData("{\"chat_template\":[{\"name\":\"default\",\"template\":\"\"}]}")]
    [InlineData("{\"chat_template\":[{\"name\":\"default\",\"template\":\"a\"},{\"name\":\"default\",\"template\":\"b\"}]}")]
    public void InvalidDeclaration_FailsClearlyWithoutFallback(string config)
    {
        var tokenizer = WriteTokenizer();
        File.WriteAllText(Path.Combine(_directory, "tokenizer_config.json"), config);
        var error = Assert.Throws<InvalidDataException>(() => HfChatTemplateFactory.TryCreate(_directory, tokenizer));
        Assert.Contains("tokenizer_config.json", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("{% if messages %}")]
    [InlineData("{% unsupported %}")]
    public void InvalidJinja_PropagatesTheSameErrorAsGguf(string source)
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(source);
        Assert.Throws<JinjaException>(() => HfChatTemplateFactory.TryCreate(_directory, tokenizer));
    }

    [Theory]
    [InlineData("tools are mentioned in prose", false)]
    [InlineData("{# tools #}{{ 'tools' }}", false)]
    [InlineData("{{ tools | tojson }}", true)]
    [InlineData("{% if xml_tools %}{{ xml_tools | tojson }}{% endif %}", true)]
    [InlineData("{% if python_tools %}{{ python_tools | tojson }}{% endif %}", false)]
    public void ToolPrompting_RequiresAWiredVariable(string source, bool expected)
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(source);
        var template = HfChatTemplateFactory.TryCreate(_directory, tokenizer)!;
        Assert.Equal(expected, template.SupportsTools);
        if (expected)
            Assert.Contains("weather", template.Apply(Messages, new ChatTemplateOptions { Tools = Tools }), StringComparison.Ordinal);
    }

    [Fact]
    public void UserControlTokens_RemainGuarded()
    {
        var tokenizer = WriteTokenizer();
        WriteConfig(Template);
        var template = HfChatTemplateFactory.TryCreate(_directory, tokenizer)!;
        string prompt = template.Apply([new() { Role = "user", Content = "<s>" }], new ChatTemplateOptions());
        Assert.Equal(1, tokenizer.Encode(prompt).Count(id => id == tokenizer.BosTokenId));
    }

    [Fact]
    public void TokenizerWithoutBosOrEos_UsesEmptyStrings()
    {
        Directory.CreateDirectory(_directory);
        WriteConfig("{{ bos_token }}B{{ eos_token }}");
        var tokenizer = HfBpeTokenizerFactory.Create("""{"model":{"type":"BPE","vocab":{"B":0},"merges":[]},"pre_tokenizer":{"type":"ByteLevel"}}""");
        Assert.Equal(-1, tokenizer.BosTokenId);
        Assert.Equal("B", HfChatTemplateFactory.TryCreate(_directory, tokenizer)!.Apply([], new ChatTemplateOptions()));
    }

    private ITokenizer WriteTokenizer()
    {
        Directory.CreateDirectory(_directory);
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < 256; i++) vocab.Add($"<0x{i.ToString("X2", CultureInfo.InvariantCulture)}>", i);
        vocab.Add("<s>", 300);
        vocab.Add("</s>", 301);
        string json = JsonSerializer.Serialize(new
        {
            model = new { type = "BPE", byte_fallback = true, vocab, merges = Array.Empty<string>() },
            pre_tokenizer = new { type = "Metaspace", replacement = "▁", prepend_scheme = "never" },
            added_tokens = new[] { new { id = 300, content = "<s>", special = true }, new { id = 301, content = "</s>", special = true } },
        });
        File.WriteAllText(Path.Combine(_directory, "tokenizer.json"), json);
        return HfBpeTokenizerFactory.TryLoadFromDirectory(_directory)!;
    }

    private void WriteConfig(object template)
        => File.WriteAllText(Path.Combine(_directory, "tokenizer_config.json"), JsonSerializer.Serialize(new { chat_template = template }));

    private static IEnumerable<int> AsciiIds(string text) => text.Select(c => (int)c);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
