using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Covers <see cref="GgufChatTemplateFactory.TryCreate(GgufMetadata, ITokenizer, Architecture)"/>
/// with a genuinely-unsupported declared <c>tokenizer.chat_template</c> — the exact call site
/// <see cref="DotLLM.Server.ServerStartup.LoadModel"/> wraps in a try/catch (#273) so a model
/// whose GGUF ships a template <c>JinjaParser</c> can't handle fails loudly instead of taking
/// down the whole server process. This test proves the precondition the fix depends on: the
/// factory does NOT swallow the parse failure itself, so <c>JinjaException</c> really does
/// propagate out to the caller for ServerStartup to catch.
/// </summary>
public class GgufChatTemplateFactoryTests
{
    /// <summary>Minimal <see cref="ITokenizer"/> stub — only BOS/EOS decoding is exercised here.</summary>
    private sealed class StubTokenizer : ITokenizer
    {
        public int[] Encode(string text) => [];
        public string Decode(ReadOnlySpan<int> tokenIds) => "";
        public string DecodeToken(int tokenId) => tokenId == BosTokenId ? "<s>" : "</s>";
        public int VocabSize => 32000;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int CountTokens(string text) => 0;
    }

    private static GgufMetadata BuildMetadata(string chatTemplate)
    {
        var data = new GgufTestData(version: 3);
        data.AddString("tokenizer.chat_template", chatTemplate);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var raw = GgufReader.ReadMetadata(reader, header);
        return new GgufMetadata(raw);
    }

    [Fact]
    public void TryCreate_ValidTemplate_Succeeds()
    {
        var metadata = BuildMetadata("{% for m in messages %}{{ m.content }}{% endfor %}");
        var result = GgufChatTemplateFactory.TryCreate(metadata, new StubTokenizer(), Architecture.Llama);
        Assert.NotNull(result);
    }

    [Fact]
    public void TryCreate_MacroTemplate_NowSucceeds()
    {
        // Pre-#273 fix, this threw JinjaException("Unexpected statement keyword: Macro") — the
        // exact crash the issue reports for real Qwen3-family GGUFs. With macro support added,
        // this is no longer an "unsupported" template at all.
        var metadata = BuildMetadata(
            "{% macro greet(name) %}Hi {{ name }}{% endmacro %}{{ greet('World') }}");
        var result = GgufChatTemplateFactory.TryCreate(metadata, new StubTokenizer(), Architecture.Llama);
        Assert.NotNull(result);
    }

    [Fact]
    public void TryCreate_TemplateWithGenuinelyUnsupportedStatement_ThrowsJinjaException()
    {
        // "block" is not a keyword JinjaLexer/JinjaParser recognizes at all (unlike macro, which
        // is now supported) — this is the class of input ServerStartup.LoadModel's catch(JinjaException)
        // must still guard against for whatever the next unsupported real-world template throws.
        var metadata = BuildMetadata("{% block content %}hi{% endblock %}");

        var ex = Assert.Throws<JinjaException>(
            () => GgufChatTemplateFactory.TryCreate(metadata, new StubTokenizer(), Architecture.Llama));
        // "block" isn't a recognized keyword token at all, so the lexer emits it as a plain
        // Identifier and the parser's default statement-keyword branch rejects it.
        Assert.Contains("Unexpected statement keyword", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TryCreate_MalformedTemplate_UnclosedTag_ThrowsJinjaException()
    {
        var metadata = BuildMetadata("{% if x %}unterminated");

        Assert.Throws<JinjaException>(
            () => GgufChatTemplateFactory.TryCreate(metadata, new StubTokenizer(), Architecture.Llama));
    }
}
