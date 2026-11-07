using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers.ChatTemplates;

/// <summary>
/// #1667 (TwIL-8B S0): the official <c>TwIL-8B-Q4_K_M-balanced.gguf</c> chat template (ChatML-style, opens <c>&lt;think&gt;</c>)
/// rendered by the Jinja subset, against goldens produced by python <c>jinja2</c> with HF's environment
/// (<c>trim_blocks=True, lstrip_blocks=True</c>) on the same template text. Header/vocabulary only: no weights are loaded.
/// The model is under the webAI Non-Commercial licence, so the GGUF is never committed; the tests skip when it is absent.
/// </summary>
public sealed class Twil8bChatTemplateTests
{
    private static (GgufFile File, ITokenizer Tok, IChatTemplate Template)? Open(out string skip)
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_TWIL8B_GGUF", "webAI-Official", "TwIL-8B", "TwIL-8B-Q4_K_M-balanced.gguf");
        skip = loc.SkipMessage("TwIL-8B Q4_K_M balanced");
        if (!loc.Found) return null;
        var gguf = GgufFile.Open(loc.Path!);
        var tok = GgufTokenizerFactory.Load(gguf.Metadata);
        var tpl = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok);
        Assert.NotNull(tpl);
        return (gguf, tok, tpl!);
    }

    private static string Render(IChatTemplate t, ChatTemplateOptions o, params (string Role, string Content)[] msgs)
        => t.Apply(msgs.Select(m => new ChatMessage { Role = m.Role, Content = m.Content }).ToList(), o);

    [SkippableFact]
    public void Template_MatchesJinja2Goldens_AndOpensThink()
    {
        var o = Open(out string skip);
        Skip.If(o is null, skip);
        using var gguf = o!.Value.File;
        var t = o.Value.Template;

        // Goldens: python jinja2 (trim_blocks, lstrip_blocks) over the GGUF template text.
        Assert.Equal("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
            Render(t, new ChatTemplateOptions(), ("user", "What is 2+2?")));
        Assert.Equal("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>",
            Render(t, new ChatTemplateOptions { EnableThinking = false }, ("system", "Be brief."), ("user", "Hi")));
        Assert.Equal("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n",
            Render(t, new ChatTemplateOptions(), ("user", "Hi"), ("assistant", "<think>\nplan\n</think>\nHello!"), ("user", "Bye")));
        Assert.Equal("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi\n\n{reasoning effort: low}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            Render(t, new ChatTemplateOptions { ReasoningEffort = "low" }, ("user", "Hi")));

        // The generation prompt ends inside an open think block, so reasoning is split from content.
        string prompt = Render(t, new ChatTemplateOptions(), ("user", "Hi"));
        Assert.True(ReasoningFormats.PromptOpensThinking(prompt));
        Assert.False(ReasoningFormats.PromptOpensThinking(
            Render(t, new ChatTemplateOptions { EnableThinking = false }, ("user", "Hi"))));
    }

    [SkippableFact]
    public void ImEnd_IsTheDeclaredEos_AndASingleToken_AndNoBosIsAdded()
    {
        var o = Open(out string skip);
        Skip.If(o is null, skip);
        using var gguf = o!.Value.File;
        var tok = o.Value.Tok;

        Assert.Equal(100257, tok.EosTokenId);
        Assert.Equal("<|im_end|>", tok.DecodeToken(100257));
        Assert.Contains(100257, EndOfGenerationTokens.Resolve(tok));
        Assert.Equal(new[] { 100257 }, tok.Encode("<|im_end|>"));
        Assert.Equal(new[] { 100256 }, tok.Encode("<|im_start|>"));
        Assert.Equal(new[] { 100274 }, tok.Encode("<think>"));
        // llama.cpp evaluated the same file without BOS (no add_bos_token key); a leading BOS here would shift every parity number.
        Assert.NotEqual(100283, tok.Encode("hello")[0]);
    }
}
