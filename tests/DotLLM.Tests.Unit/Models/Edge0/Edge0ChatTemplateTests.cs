using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// G5 (#1975): the tokenizer and chat template embedded in Edge0's <c>model-skeleton.gguf</c> (the Qwen template of the base model with the
/// <c>reasoning_effort</c> / <c>enable_thinking</c> switches) parse in dotLLM's Jinja engine, render what the template text says, and the
/// tokenizer maps the control tokens to single ids. Expected strings are written from the template TEXT, not from the renderer's output.
/// </summary>
public sealed class Edge0ChatTemplateTests(ITestOutputHelper output)
{
    private const string Xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    private const string Low = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf");

    private static string Render(DotLLM.Tokenizers.ChatTemplates.JinjaChatTemplate t, ChatTemplateOptions o)
        => t.Apply([new ChatMessage { Role = "user", Content = "Hi" }], o);

    [SkippableFact]
    public void Template_ParsesAndRendersEffortAndThinkingSwitches_AndControlTokensAreSingleIds()
    {
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k/model-skeleton.gguf (6 GB)"));
        using var gguf = GgufFile.Open(fx.Path!);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var t = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok, DotLLM.Core.Configuration.Architecture.Qwen4Exp);
        Assert.NotNull(t);   // a missing/unparseable template would fall back to the plain transcript (no <|im_start|>) in the CLI

        const string user = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        // thinking on, effort default (xhigh): instructions in a system turn, the generation prompt opens <think>
        Assert.Equal($"<|im_start|>system\n{Xhigh}<|im_end|>\n{user}<think>\n", Render(t, new ChatTemplateOptions()));
        Assert.Equal($"<|im_start|>system\n{Low}<|im_end|>\n{user}<think>\n", Render(t, new ChatTemplateOptions { ReasoningEffort = "low" }));
        // medium carries no instruction text
        Assert.Equal($"{user}<think>\n", Render(t, new ChatTemplateOptions { ReasoningEffort = "medium" }));
        // thinking off (the template's own "none"): an empty think block is pre-filled
        Assert.Equal($"{user}<think>\n\n</think>\n\n", Render(t, new ChatTemplateOptions { EnableThinking = false }));
        // "none" is not an effort level of THIS template: it must be refused loudly, not rendered as something else
        Assert.ThrowsAny<Exception>(() => Render(t, new ChatTemplateOptions { ReasoningEffort = "none" }));

        // tokenizer: control tokens are single ids; generation EOS (<|im_end|>) is not the PLE EOS (<|endoftext|> 248044)
        var ids = tok.Encode("<|im_start|>user\nHi<|im_end|>\n");
        output.WriteLine("ids: " + string.Join(",", ids));
        Assert.Equal(248045, ids[0]);   // <|im_start|>
        Assert.Contains(248046, ids);   // <|im_end|>
        Assert.Equal(248046, tok.EosTokenId);
        Assert.Equal(248044, gguf.Metadata.GetInt32OrDefaultCompat("tokenizer.ggml.bos_token_id"));
        Assert.False(gguf.Metadata.GetBoolOrDefault("tokenizer.ggml.add_bos_token", true));
    }
}

internal static class MetadataCompat
{
    /// <summary>The GGUF stores the id as UInt32; read it either way.</summary>
    public static int GetInt32OrDefaultCompat(this GgufMetadata m, string key)
        => m.TryGetValue(key, out var v) ? Convert.ToInt32(v.Value, System.Globalization.CultureInfo.InvariantCulture) : -1;
}
