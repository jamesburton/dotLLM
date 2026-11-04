using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using DotLLM.Tests.Integration.Fixtures;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1591: on Vulkan a later request adopts the previous request's device KV cache, cuts it back to the longest common
/// prompt prefix and prefills only the suffix. At temperature 0 the same kernels run on the same rows, so the continuation
/// must equal a cold prefill of the same prompt.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanPrefixReuseTests
{
    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private static string Chat(string q) =>
        "<|im_start|>user\n" + string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 40)) + q
        + "<|im_end|>\n<|im_start|>assistant\n";

    [SkippableFact]
    public void SecondRequest_ReusesCommonPrefix_AndMatchesColdPrefill()
    {
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var _ = model as IDisposable;
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16 };

        var cold = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), mtpEnabled: false);
        using var prefix = new PrefixCache(2);
        var warm = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), prefix, mtpEnabled: false,
            prefixCacheHeadroomTokens: 1024);

        warm.Generate(Chat("Question 1: say OK."), opts);
        long live = device.LiveAllocationCount;
        foreach (string q in new[] { "Question 2: say OK.", "Question 33: name a colour.", "Question 2: say OK." })
        {
            var expected = cold.Generate(Chat(q), opts);
            var actual = warm.Generate(Chat(q), opts);
            int common = tokenizer.Encode(Chat(q)).AsSpan().CommonPrefixLength(tokenizer.Encode(Chat("Question 1: say OK.")));
            Assert.True(actual.Timings.CachedTokenCount > 0, "expected a prefix hit");
            Assert.True(actual.Timings.CachedTokenCount <= common, "cached more rows than the prompts share");
            Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);
        }
        Assert.Equal(live, device.LiveAllocationCount);          // the one retained cache is reused, nothing leaks
    }
}
