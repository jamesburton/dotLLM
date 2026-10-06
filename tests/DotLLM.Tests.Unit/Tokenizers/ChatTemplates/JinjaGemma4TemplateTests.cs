using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Issue #770: the real Gemma-4 (E4B GGUF) chat template uses an inline-if WITHOUT else
/// (<c>{{- ',' if not loop.last -}}</c>), which used to fail to parse and made <c>dotllm serve</c>
/// fall back to a plain transcript (garbage chat). Golden string is from reference Jinja2 3.1.6.
/// </summary>
public class JinjaGemma4TemplateTests
{
    private static string LoadFixture() => File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "gemma-4-e4b-chat-template.jinja"));

    [Fact]
    public void FullTemplate_Parses_And_Renders_SystemUserChat_LikeReferenceJinja2()
    {
        var template = new JinjaChatTemplate(LoadFixture(), bosToken: "<bos>", eosToken: "<eos>");
        var messages = new[]
        {
            new ChatMessage { Role = "system", Content = "Be brief." },
            new ChatMessage { Role = "user", Content = "What is 2+2?" },
        };
        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });
        Assert.Equal("<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n", result);
    }
}
