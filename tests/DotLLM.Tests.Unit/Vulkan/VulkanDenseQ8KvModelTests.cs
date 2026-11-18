using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1884 end to end on Qwen2.5-0.5B (dense, hd64, GQA 7): q8_0 or q4_0 K and V run on an in-place block <see cref="VulkanKvCache"/>, with
/// no F32 dequant scratch.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>The prefill dispatches the Q8_0 GQA flash twin (counter), and the cache is the in-place one.</item>
/// <item>Greedy tokens match the CPU <c>QuantizedKvCache</c> (the same KV rule), and for Q8_0 the old <see cref="VulkanQuantizedKvCache"/>
/// path too (same Q8_0 rule, same values; only the attention kernel differs).</item>
/// <item>A second turn reuses the first turn's prefix from the in-place cache, grows it, and matches a cold in-place run.</item>
/// </list>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanDenseQ8KvModelTests
{
    private readonly ITestOutputHelper _out;
    public VulkanDenseQ8KvModelTests(ITestOutputHelper output) => _out = output;

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private static string Turn1() => "<|im_start|>user\n" + string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 30))
        + "Summarise that in one sentence.<|im_end|>\n<|im_start|>assistant\n";

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void BlockKv_RunsInPlace_MatchesTheDequantScratchPath_AndReusesItsPrefix(KvCacheDType dtype)
    {
        var storage = dtype == KvCacheDType.Q4_0 ? VulkanKvStorage.Q4_0 : VulkanKvStorage.Q8_0;
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var _ = model as IDisposable;
        var vk = Assert.IsType<VulkanTransformerModel>(model);
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16, Logprobs = true };

        // Reference: the dequant-scratch cache (before the switch, so the factory is unaffected).
        var scratch = new TextGenerator(model, tokenizer, (_, size) => vk.CreateQuantizedKvCache(spvDir, size, dtype, dtype), mtpEnabled: false);
        var expected = scratch.Generate(Turn1(), opts);

        // The CPU oracle: QuantizedKvCache uses exactly the Q8_0 / Q4_0 KV rule the in-place cache copies (#1884; CUDA uses it too).
        InferenceResponse cpu;
        using (var cpuModel = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config, DotLLM.Core.Configuration.ThreadingConfig.Auto))
        {
            var cpuGen = new TextGenerator(cpuModel, tokenizer,
                (c, size) => new DotLLM.Engine.KvCache.QuantizedKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(c), size, dtype, dtype, 0), mtpEnabled: false);
            cpu = cpuGen.Generate(Turn1(), opts);
        }
        _out.WriteLine($"cpu {dtype}: lp0 {cpu.Logprobs![0].Logprob:R} text: {cpu.Text}");

        Assert.True(vk.TryUseInPlaceQuantizedKv(dtype, dtype, spvDir));
        Assert.False(vk.TryUseInPlaceQuantizedKv(KvCacheDType.Q8_0, KvCacheDType.Q4_0, spvDir));   // mixed: not yet in place
        using (var probe = kvFactory(64))
            Assert.Equal(storage, Assert.IsType<VulkanKvCache>(probe).Storage);

        var inPlace = new TextGenerator(model, tokenizer, (_, size) => kvFactory(size), mtpEnabled: false);
        long gqa0 = vk.PrefillGqaFlashCount;
        var actual = inPlace.Generate(Turn1(), opts);
        long gqaRuns = vk.PrefillGqaFlashCount - gqa0;
        double d = Math.Abs(expected.Logprobs![0].Logprob - actual.Logprobs![0].Logprob);
        _out.WriteLine($"in-place {dtype}: GQA flash prefills {gqaRuns}; first-token logprob scratch {expected.Logprobs[0].Logprob:R} in-place {actual.Logprobs[0].Logprob:R} (|d| {d:E2})");
        _out.WriteLine($"text scratch: {expected.Text}");
        _out.WriteLine($"text inplace: {actual.Text}");
        Assert.True(gqaRuns > 0, $"the prefill did not run the {dtype} GQA flash twin");
        Assert.Equal(expected.GeneratedTokenIds[0], actual.GeneratedTokenIds[0]);
        if (dtype == KvCacheDType.Q8_0)
        {
            Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);   // same Q8_0 rule, same values
            Assert.True(d < 0.05, $"first-token logprob moved {d}");
        }
        // The CPU oracle (QuantizedKvCache, the same KV rule): greedy tokens equal. On Q4_0 the first-token logprob is NOT a tight gauge: a 0.5B
        // model with 4-bit KV amplifies f16-vs-f32 scale and coopmat rounding differences (measured: CPU -1.478, old Vulkan path -0.781, in place
        // -0.373, all three with the same text), so the token sequence is the gate.
        Assert.Equal(cpu.GeneratedTokenIds, actual.GeneratedTokenIds);
        // Run to run bit-identical: the Q4_0 write once took the transfer barrier, so attention raced the quantiser and five runs gave five
        // first-token logprobs (-0.43, -1.57, -0.93, -0.82, -0.58). A missing write -> read barrier fails this with high probability.
        var again = inPlace.Generate(Turn1(), opts);
        for (int i = 0; i < actual.Logprobs!.Length; i++)
            Assert.Equal(actual.Logprobs[i].Logprob, again.Logprobs![i].Logprob);

        // Prefix reuse on the in-place cache: turn 2 adopts turn 1's cache, grows it, and matches a cold in-place run.
        using var prefix = new PrefixCache(2, PrefixCachePolicy.Agent);
        var warm = new TextGenerator(model, tokenizer, (_, size) => kvFactory(size), prefix, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
        var first = warm.Generate(Turn1(), opts);
        string t2 = Turn1() + first.Text + "<|im_end|>\n<|im_start|>user\n" + string.Concat(Enumerable.Repeat("Then name one colour. ", 20))
            + "<|im_end|>\n<|im_start|>assistant\n";
        long g0 = VulkanKvCache.GrowthsForTest;
        var second = warm.Generate(t2, opts);
        _out.WriteLine($"turn 2: prompt {second.PromptTokenCount}, cached {second.Timings.CachedTokenCount}, growths {VulkanKvCache.GrowthsForTest - g0}");
        var cold = inPlace.Generate(t2, opts);
        _out.WriteLine($"turn 2 warm: {second.Text} | lp0 {second.Logprobs![0].Logprob:R}");
        _out.WriteLine($"turn 2 cold: {cold.Text} | lp0 {cold.Logprobs![0].Logprob:R}");
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_1884_F16_CONTROL") == "1")
        {
            vk.KvStorageOverride = VulkanKvStorage.F16;
            using var p16 = new PrefixCache(2, PrefixCachePolicy.Agent);
            var w16 = new TextGenerator(model, tokenizer, (_, size) => kvFactory(size), p16, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
            w16.Generate(Turn1(), opts);
            var s16 = w16.Generate(t2, opts);
            var c16 = new TextGenerator(model, tokenizer, (_, size) => kvFactory(size), mtpEnabled: false).Generate(t2, opts);
            _out.WriteLine($"F16 control warm: {s16.Text} | lp0 {s16.Logprobs![0].Logprob:R}");
            _out.WriteLine($"F16 control cold: {c16.Text} | lp0 {c16.Logprobs![0].Logprob:R}");
            vk.KvStorageOverride = storage;
        }
        Assert.True(second.Timings.CachedTokenCount > 0, "turn 2 missed");
        Assert.True(VulkanKvCache.GrowthsForTest - g0 >= 1, $"turn 2 should have grown the {dtype} cache");

        // The discriminating check: a warm run that never grew (headroom sized for turn 2) runs the same kernels on the same rows, so
        // it must be bit-identical to the grown one. Warm vs cold is not: the suffix runs other attention kernels than one long prefill,
        // and on Q8_0 that moves a near-tie (measured: cold stops after the first word, warm continues; the F16 control's texts agree).
        using var roomyCache = new PrefixCache(2, PrefixCachePolicy.Agent);
        var roomy = new TextGenerator(model, tokenizer, (_, size) => kvFactory(size), roomyCache, mtpEnabled: false, prefixCacheHeadroomTokens: 4096);
        roomy.Generate(Turn1(), opts);
        long g1 = VulkanKvCache.GrowthsForTest;
        var ungrown = roomy.Generate(t2, opts);
        Assert.Equal(g1, VulkanKvCache.GrowthsForTest);
        Assert.Equal(second.Timings.CachedTokenCount, ungrown.Timings.CachedTokenCount);
        Assert.Equal(ungrown.GeneratedTokenIds, second.GeneratedTokenIds);
        for (int i = 0; i < ungrown.Logprobs!.Length; i++)
            Assert.Equal(ungrown.Logprobs[i].Logprob, second.Logprobs![i].Logprob);
        Assert.Equal(cold.GeneratedTokenIds[0], second.GeneratedTokenIds[0]);
        double dCold = Math.Abs(cold.Logprobs![0].Logprob - second.Logprobs![0].Logprob);
        Assert.True(dCold < (dtype == KvCacheDType.Q4_0 ? 0.3 : 0.1), $"turn 2 first-token logprob moved {dCold} vs cold");
    }
}
