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

    /// <summary>
    /// Issue #1843 on Vulkan, as on CUDA: with a second, unrelated entry retained, a hit whose growth is refused evicts that entry
    /// and retries the growth, keeping the hit, instead of re-prefilling the conversation. Admission is refused until the
    /// neighbour is gone, as when it holds the memory the growth needs.
    /// </summary>
    [SkippableFact]
    public void RefusedGrowth_WithANeighbourRetained_EvictsIt_AndKeepsTheHit()
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
        string unrelated = "Harbour log, evening watch. " + string.Concat(Enumerable.Repeat("Seven sailors sorted silver spoons. ", 15))
            + "Count the sailors.<|im_end|>\n";
        string toolResult = "<|im_start|>user\n<tool_response>\n"
            + string.Concat(Enumerable.Range(0, 60).Select(i => $"Row {i}: the value of item {i * 7 % 13} is {i * 31 % 97}. "))
            + "\n</tool_response><|im_end|>\n";

        using var prefix = new PrefixCache(2, PrefixCachePolicy.Lru);
        var warm = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), prefix, mtpEnabled: false,
            prefixCacheHeadroomTokens: 16);
        var first = warm.Generate(Turn(turn1), opts);
        warm.Generate(Turn(unrelated), opts);
        Assert.Equal(2, prefix.EntryCount);
        string turn2 = Turn(turn1 + "<|im_start|>assistant\n" + first.Text + "<|im_end|>\n" + toolResult);
        int firstRows = tokenizer.Encode(Turn(turn1)).Length + first.GeneratedTokenIds.Length - 1;

        long growths = VulkanKvCache.GrowthsForTest, evictions = prefix.AdmissionEvictions;
        VulkanKvCache.GrowthAdmissionOverride = (_, _) => prefix.AdmissionEvictions > evictions;
        InferenceResponse actual;
        try { actual = warm.Generate(turn2, opts); }
        finally { VulkanKvCache.GrowthAdmissionOverride = null; }

        Assert.Equal(evictions + 1, prefix.AdmissionEvictions);
        Assert.Equal(1, prefix.EntryCount);
        Assert.Equal(growths + 1, VulkanKvCache.GrowthsForTest);
        Assert.InRange(actual.Timings.CachedTokenCount, firstRows - 8, firstRows + 2);
        Assert.Equal(cold.Generate(turn2, opts).GeneratedTokenIds, actual.GeneratedTokenIds);
    }

    /// <summary>
    /// Issue #1843, Vulkan side of CUDA's quantized growth (#1594): a retained <see cref="VulkanQuantizedKvCache"/> grows for a next
    /// turn longer than its headroom, keeps every first-turn row, and the continuation equals a cold quantized prefill and a warm run
    /// that never grew (same blocks, same kernels: bit-identical tokens).
    /// </summary>
    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q4_0)]
    public void Quantized_NextTurnLongerThanTheHeadroom_GrowsTheCache_AndMatchesColdAndUngrown(KvCacheDType kType, KvCacheDType vType)
    {
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var __ = model as IDisposable;
        var vk = Assert.IsType<VulkanTransformerModel>(model);
        Func<DotLLM.Core.Models.ModelConfig, int, DotLLM.Core.Attention.IKvCache> quantized =
            (_, size) => vk.CreateQuantizedKvCache(spvDir, size, kType, vType);
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 12 };
        var cold = new TextGenerator(model, tokenizer, quantized, mtpEnabled: false);

        string turn1 = "<|im_start|>user\n" + string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 20))
            + "Fetch the page.<|im_end|>\n";
        string toolResult = "<|im_start|>user\n<tool_response>\n"
            + string.Concat(Enumerable.Range(0, 60).Select(i => $"Row {i}: the value of item {i * 7 % 13} is {i * 31 % 97}. "))
            + "\n</tool_response><|im_end|>\n";

        InferenceResponse Warm(int headroom, out long grew, out int firstRows)
        {
            using var prefix = new PrefixCache(2, PrefixCachePolicy.Agent);
            var warm = new TextGenerator(model, tokenizer, quantized, prefix, mtpEnabled: false, prefixCacheHeadroomTokens: headroom);
            var first = warm.Generate(Turn(turn1), opts);
            firstRows = tokenizer.Encode(Turn(turn1)).Length + first.GeneratedTokenIds.Length - 1;
            string turn2 = Turn(turn1 + "<|im_start|>assistant\n" + first.Text + "<|im_end|>\n" + toolResult);
            long g0 = VulkanKvCache.GrowthsForTest;
            var r = warm.Generate(turn2, opts);
            grew = VulkanKvCache.GrowthsForTest - g0;
            return r;
        }

        var grown = Warm(16, out long growths, out int rows);
        var ungrown = Warm(4096, out long noGrowths, out _);
        string t2 = Turn(turn1 + "<|im_start|>assistant\n" + cold.Generate(Turn(turn1), opts).Text + "<|im_end|>\n" + toolResult);
        var expected = cold.Generate(t2, opts);

        Assert.Equal(1, growths);
        Assert.Equal(0, noGrowths);
        Assert.InRange(grown.Timings.CachedTokenCount, rows - 8, rows + 2);
        Assert.Equal(ungrown.Timings.CachedTokenCount, grown.Timings.CachedTokenCount);
        Assert.Equal(ungrown.GeneratedTokenIds, grown.GeneratedTokenIds);
        Assert.Equal(expected.GeneratedTokenIds, grown.GeneratedTokenIds);
    }

    /// <summary>
    /// Issue #1843: <see cref="VulkanQuantizedKvCache.TrySeedPrefixFrom"/> copies the scale and quant blocks as stored, so a suffix
    /// over a seeded cache gives bit-identical logits to the same suffix over the cut-back source.
    /// </summary>
    [SkippableFact]
    public void Quantized_Seed_GivesTheCutBackLogits()
    {
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var __ = model as IDisposable;
        var vk = Assert.IsType<VulkanTransformerModel>(model);

        int[] tokens = tokenizer.Encode("<|im_start|>user\n" + string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 12))
            + "Fetch the page.<|im_end|>\n<|im_start|>assistant\n");
        int n = tokens.Length, cut = n / 2 + 3;
        int[] positions = Enumerable.Range(0, n).ToArray();
        unsafe float[] Suffix(DotLLM.Core.Attention.IKvCache kv)
        {
            using var t = model.Forward(tokens.AsSpan(cut), positions.AsSpan(cut), -1, kv);
            int vocab = config.VocabSize, rowsOut = t.Shape[0];
            var row = new float[vocab];
            new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rowsOut - 1) * vocab, vocab).CopyTo(row);
            return row;
        }

        using var src = vk.CreateQuantizedKvCache(spvDir, n, KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        model.Forward(tokens, positions, -1, src).Dispose();
        using var seeded = vk.CreateQuantizedKvCache(spvDir, n, KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Assert.True(seeded.TrySeedPrefixFrom(src, cut));
        Assert.Equal(cut, seeded.CurrentLength);
        float[] fromSeed = Suffix(seeded);

        src.Rollback(cut);
        float[] reference = Suffix(src);
        Assert.Equal(reference, fromSeed);

        using var otherType = vk.CreateQuantizedKvCache(spvDir, n, KvCacheDType.Q4_0, KvCacheDType.Q4_0);
        Assert.False(otherType.TrySeedPrefixFrom(src, cut));   // another block format: refused, unchanged
        Assert.Equal(0, otherType.CurrentLength);
    }
}
