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
/// Issue #1781: <see cref="VulkanKvCache.TryGrowTo"/> lets a retained prefix-cache entry serve a next turn that outgrew it (an agent's
/// large tool result) instead of re-prefilling the whole conversation. The growth keeps the committed rows byte for byte in the cache's
/// own format, is admitted against free device memory, and a refusal leaves the cache as it was.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCacheGrowthTests
{
    private static readonly int[] Strides = [8, 16];   // per-layer widths differ, so a wrong per-layer row size shows

    private static float Val(int layer, int row, int ch, bool value)
        => (layer * 1000 + row * 10 + ch) * 0.0173f * (value ? -1f : 1f) + 0.00031f;

    private static (float[] K, float[] V) Rows(int layer, int start, int count)
    {
        int s = Strides[layer];
        var k = new float[count * s];
        var v = new float[count * s];
        for (int r = 0; r < count; r++)
            for (int c = 0; c < s; c++)
            {
                k[r * s + c] = Val(layer, start + r, c, false);
                v[r * s + c] = Val(layer, start + r, c, true);
            }
        return (k, v);
    }

    private static void Fill(VulkanKvCache cache, int start, int count)
    {
        for (int l = 0; l < Strides.Length; l++)
        {
            var (k, v) = Rows(l, start, count);
            cache.WriteRows(l, start, count, k, v);
        }
    }

    private static void AssertRows(VulkanKvCache cache, int start, int count, VulkanKvStorage storage)
    {
        for (int l = 0; l < Strides.Length; l++)
        {
            int s = Strides[l];
            var (ek, ev) = Rows(l, start, count);
            var rk = new float[count * s];
            var rv = new float[count * s];
            cache.ReadRows(l, start, count, rk, rv);
            for (int i = 0; i < rk.Length; i++)
            {
                float k = storage == VulkanKvStorage.F16 ? (float)(Half)ek[i] : ek[i];
                float v = storage == VulkanKvStorage.F16 ? (float)(Half)ev[i] : ev[i];
                Assert.True(k == rk[i], $"K layer {l} element {i}: expected {k}, got {rk[i]}");
                Assert.True(v == rv[i], $"V layer {l} element {i}: expected {v}, got {rv[i]}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void TryGrowTo_KeepsTheCommittedRows_AndTheNewRowsAreWritable(VulkanKvStorage storage)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, Strides, 24, storage);
        Fill(cache, 0, 24);
        cache.Rollback(19);                                      // as the engine does: cut to the match, then grow
        long allocations = dev.LiveAllocationCount;

        Assert.True(cache.TryGrowTo(70));

        Assert.Equal(70, cache.MaxLength);
        Assert.Equal(19, cache.CurrentLength);
        Assert.Equal(allocations, dev.LiveAllocationCount);       // the old buffers were freed
        Assert.Equal(2L * 70 * (8 + 16) * storage.ElementBytes(), cache.AllocatedBytes);
        AssertRows(cache, 0, 19, storage);
        Fill(cache, 19, 51);                                      // past the old capacity, up to the new one
        AssertRows(cache, 0, 70, storage);
    }

    [SkippableFact]
    public void TryGrowTo_RefusedByAdmission_LeavesTheCacheAsItWas()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, Strides, 24, VulkanKvStorage.F32);
        Fill(cache, 0, 24);
        long asked = -1;
        VulkanKvCache.GrowthAdmissionOverride = (_, bytes) => { asked = bytes; return false; };
        try
        {
            Assert.False(cache.TryGrowTo(70));
        }
        finally
        {
            VulkanKvCache.GrowthAdmissionOverride = null;
        }

        // Peak extra bytes: the growth of every layer plus the largest old layer's K and V in transit.
        Assert.Equal(2L * (70 - 24) * (8 + 16) * 4 + 2L * 24 * 16 * 4, asked);
        Assert.Equal(24, cache.MaxLength);
        AssertRows(cache, 0, 24, VulkanKvStorage.F32);
        Assert.True(cache.TryGrowTo(24));                         // no growth needed: always fine
    }

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private static string Turn(string history) => history + "<|im_start|>assistant\n";

    /// <summary>
    /// End to end on the real model: the second turn appends a tool result far longer than the headroom the retained cache was sized
    /// with. The cache grows, every row of the first turn is reused and the greedy continuation equals a cold prefill. With the growth
    /// refused, the request falls back to a full prefill (cached 0) and is still correct.
    /// </summary>
    [SkippableFact]
    public void NextTurnLongerThanTheHeadroom_GrowsTheCache_AndMatchesColdPrefill()
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
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 12 };
        var cold = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), mtpEnabled: false);

        string turn1 = "<|im_start|>user\n" + string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 20))
            + "Fetch the page.<|im_end|>\n";
        string toolResult = "<|im_start|>user\n<tool_response>\n"
            + string.Concat(Enumerable.Range(0, 60).Select(i => $"Row {i}: the value of item {i * 7 % 13} is {i * 31 % 97}. "))
            + "\n</tool_response><|im_end|>\n";

        foreach (bool refuse in new[] { false, true })
        {
            using var prefix = new PrefixCache(2, PrefixCachePolicy.Agent);
            var warm = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), prefix, mtpEnabled: false,
                prefixCacheHeadroomTokens: 16);
            var first = warm.Generate(Turn(turn1), opts);
            string turn2 = Turn(turn1 + "<|im_start|>assistant\n" + first.Text + "<|im_end|>\n" + toolResult);
            int firstRows = tokenizer.Encode(Turn(turn1)).Length + first.GeneratedTokenIds.Length - 1;
            int added = tokenizer.Encode(turn2).Length - firstRows;
            Assert.True(added > 16 + opts.MaxTokens, $"the tool result must outgrow the headroom (added {added})");

            long growths = VulkanKvCache.GrowthsForTest;
            if (refuse) VulkanKvCache.GrowthAdmissionOverride = (_, _) => false;
            try
            {
                var actual = warm.Generate(turn2, opts);
                var expected = cold.Generate(turn2, opts);
                Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);
                if (refuse)
                {
                    Assert.Equal(0, actual.Timings.CachedTokenCount);
                    Assert.Equal(growths, VulkanKvCache.GrowthsForTest);
                }
                else
                {
                    // The reply's round trip through the template may re-tokenize its last tokens, and a reply that stopped on its
                    // end-of-turn token forwarded that token too: allow a short tail either way.
                    Assert.InRange(actual.Timings.CachedTokenCount, firstRows - 8, firstRows + 2);
                    Assert.Equal(growths + 1, VulkanKvCache.GrowthsForTest);
                }
            }
            finally
            {
                VulkanKvCache.GrowthAdmissionOverride = null;
            }
        }
    }
}
