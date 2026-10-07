using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// #776 against the REAL vocabularies: the unit tests use a fake tokenizer, which cannot show that
/// <c>Encode("&lt;eos&gt;")</c> on Gemma-4's real tokenizer returns exactly one id. If it did not,
/// <see cref="EndOfGenerationTokens.Resolve"/> would silently add nothing and the <c>&lt;eos&gt;&lt;eos&gt;...</c> leak
/// would remain. No weights are loaded: only the GGUF header/vocabulary is read (CPU, no GPU lock).
/// </summary>
public class EndOfGenerationRealTokenizerTests
{
    [SkippableFact]
    public void Gemma4_E4B_StopsOnEos_Id1_InAdditionToDeclaredTurnEnd()
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_GEMMA4_E4B_GGUF", "unsloth", "gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("gemma-4-E4B-it Q4_K_M"));

        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);

        int[] ids = EndOfGenerationTokens.Resolve(tokenizer);

        Assert.Equal(106, tokenizer.EosTokenId);                         // declared EOS is <turn|>
        Assert.Equal("<turn|>", tokenizer.DecodeToken(tokenizer.EosTokenId));
        Assert.Equal(tokenizer.EosTokenId, ids[0]);
        Assert.Contains(1, ids);                                         // <eos>: the token that ended every tool-call turn
        Assert.Equal("<eos>", tokenizer.DecodeToken(1));
    }

    [SkippableFact]
    public void Glm47Flash_StopsOnUserEot_AndObservationEom()
    {
        // #797: eos is <|endoftext|>, but the model ends a turn with <|user|> (eot) and a tool call with
        // <|observation|> (eom); both are declared in the GGUF and were ignored, so generation ran on.
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_GLM47_FLASH_GGUF", "unsloth", "GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("GLM-4.7-Flash Q4_K_M"));

        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);

        int[] ids = EndOfGenerationTokens.Resolve(tokenizer);

        Assert.Equal("<|endoftext|>", tokenizer.DecodeToken(tokenizer.EosTokenId));
        Assert.Contains(154827, ids);
        Assert.Equal("<|user|>", tokenizer.DecodeToken(154827));
        Assert.Contains(154829, ids);
        Assert.Equal("<|observation|>", tokenizer.DecodeToken(154829));
    }

    [SkippableFact]
    public void Llama32_1B_StopsOnEotAndEom()
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("Llama-3.2-1B-Instruct Q8_0"));

        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);

        int[] ids = EndOfGenerationTokens.Resolve(tokenizer);

        Assert.Equal(128009, ids[0]);     // <|eot_id|>
        Assert.Contains(128008, ids);     // <|eom_id|>
    }
}
