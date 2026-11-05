using System.Text.Json;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// #1604 against the REAL glm4moe vocabulary (REAP-82B GGUF header only, no weights): the end-of-turn ids the server stops on,
/// the embedded chat template, and that the golden prompts tokenise to exactly the ids the Hugging Face tokenizer produces (the
/// template's own <c>[gMASK]&lt;sop&gt;</c> is the only BOS; the engine must not add another). Skips when the file is absent.
/// </summary>
public class Glm45AirRealTokenizerTests
{
    private static GgufFile? Open()
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_GLM45_AIR_REAP_GGUF", "bartowski", "cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF", "cerebras_GLM-4.5-Air-REAP-82B-A12B-IQ4_XS.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("GLM-4.5-Air-REAP-82B IQ4_XS"));
        return GgufFile.Open(loc.Path!);
    }

    [SkippableFact]
    public void StopsOnEndoftext_User_AndObservation()
    {
        using var gguf = Open()!;
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        int[] ids = EndOfGenerationTokens.Resolve(tokenizer);

        Assert.Equal("<|endoftext|>", tokenizer.DecodeToken(tokenizer.EosTokenId));
        foreach ((int id, string text) in new[] { (151329, "<|endoftext|>"), (151336, "<|user|>"), (151338, "<|observation|>") })
        {
            Assert.Contains(id, ids);
            Assert.Equal(text, tokenizer.DecodeToken(id));
        }
    }

    [SkippableFact]
    public void EmbeddedChatTemplate_IsTheOneWeRenderGoldensFor()
    {
        using var gguf = Open()!;
        string template = gguf.Metadata.GetString("tokenizer.chat_template");
        Assert.Equal(ReferenceRenderCases.Fixture("glm-4.5-air-gguf-embedded-chat-template.jinja"), template);
    }

    [SkippableFact]
    public void GoldenPrompts_TokeniseToTheHuggingFaceIds()
    {
        using var gguf = Open()!;
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);

        using var cases = JsonDocument.Parse(ReferenceRenderCases.Fixture("glm-4.5-air-render-cases.json"));
        using var hf = JsonDocument.Parse(ReferenceRenderCases.Fixture("glm-4.5-air-token-ids.json"));
        int checkedCount = 0;
        foreach (var h in hf.RootElement.EnumerateArray())
        {
            string name = h.GetProperty("name").GetString()!;
            string prompt = cases.RootElement.EnumerateArray().Single(c => c.GetProperty("name").GetString() == name).GetProperty("expected").GetString()!;
            int[] want = h.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.True(want.SequenceEqual(tokenizer.Encode(prompt)), $"token ids differ from HF for '{name}'");
            checkedCount++;
        }
        Assert.True(checkedCount >= 7);
    }
}
