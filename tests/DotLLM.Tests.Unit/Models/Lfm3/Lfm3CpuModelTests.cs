using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm3;

/// <summary>
/// #1559: LFM3 (<c>lfm3</c>) config extraction and CPU forward, against the real LFM3-1.7B Q8_0 GGUF (skipped when absent;
/// set <c>DOTLLM_LFM3_GGUF</c> or <c>hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf</c>). The reference token ids come from
/// llama.cpp (<c>dde879bc</c>, <c>-dev none</c>, greedy, repeat penalty off).
/// </summary>
public sealed unsafe class Lfm3CpuModelTests
{
    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_LFM3_GGUF", "LiquidAI", "LFM3-1.7B-GGUF", "LFM3-1.7B-Q8_0.gguf").Path;

    [SkippableFact]
    public void RealHeader_ExtractsTriHybridLayout()
    {
        string? path = FindGguf();
        Skip.If(path is null, "LFM3 GGUF not found: set DOTLLM_LFM3_GGUF or fetch LiquidAI/LFM3-1.7B-GGUF.");
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Lfm3, cfg.Architecture);
        Assert.Equal(24, cfg.NumLayers);
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.Equal(16, cfg.NumAttentionHeads);
        Assert.Equal(4, cfg.NumKvHeads);
        Assert.Equal(128, cfg.HeadDim);
        Assert.Equal(7680, cfg.IntermediateSize);
        Assert.Equal(7, cfg.ShortConvCacheLength);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.Equal(10000f, cfg.RoPEConfig.Value.Theta);            // sliding layers
        Assert.Equal(1_000_000f, cfg.GlobalRoPEConfig!.Value.Theta); // global layers
        Assert.Null(cfg.Moe);

        // conv at 0,1,3,5,...; sliding attention at 4,8,12,16,20; global attention at 2,6,10,14,18,22.
        for (int i = 0; i < 24; i++)
        {
            bool attn = i is 2 or 4 or 6 or 8 or 10 or 12 or 14 or 16 or 18 or 20 or 22;
            bool sliding = i is 4 or 8 or 12 or 16 or 20;
            Assert.Equal(attn ? HybridLayerKind.Attention : HybridLayerKind.ShortConv, cfg.HybridLayout!.LayerKind[i]);
            Assert.Equal(sliding ? 256 : (int?)null, cfg.PerLayerSlidingWindow![i]);
            Assert.Equal(7680, cfg.HybridLayout.FeedForwardLength[i]);
        }
    }

    [SkippableFact]
    public void Greedy_MatchesLlamaCppTokenIds()
    {
        string? path = FindGguf();
        Skip.If(path is null, "LFM3 GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Lfm3TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(4));

        // "The capital of France is" -> llama.cpp greedy: " Paris.\n\nThe capital of Japan is Tokyo.\n\nThe capital of Australia is Canberra"
        int[] prompt = [597, 5205, 302, 3980, 355];
        int[] expected = [4741, 63908, 597, 5205, 302, 3888, 355, 18943, 63908, 597, 5205, 302, 5881, 355, 2076, 59405];

        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 64);
        using var state = model.CreateSequenceState()!;
        var positions = Enumerable.Range(0, prompt.Length).ToArray();
        int next = ArgMax(model.Forward(prompt, positions, -1, kv, (ISsmState)state), config.VocabSize, rowsToSkip: prompt.Length - 1);
        var got = new List<int> { next };
        for (int i = 1; i < expected.Length; i++)
        {
            next = ArgMax(model.Forward([next], [prompt.Length + i - 1], -1, kv, (ISsmState)state), config.VocabSize, rowsToSkip: 0);
            got.Add(next);
        }
        Assert.Equal(expected, got);
    }

    /// <summary>
    /// Chunked prefill + single-token decode through the KV cache and conv state must reproduce the cacheless full-sequence rows. 300
    /// tokens cross the 256-token sliding window, so a window applied in only one path, a conv state that drops a row, or a global-layer
    /// cache that loses a position all show up as a large logit difference.
    /// </summary>
    [SkippableFact]
    public void IncrementalPath_ReproducesFullSequenceLogits_AcrossTheSlidingWindow()
    {
        string? path = FindGguf();
        Skip.If(path is null, "LFM3 GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Lfm3TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(4));

        const int total = 300;
        var tokens = new int[total];
        uint s = 12345;
        for (int i = 0; i < total; i++) { s = s * 1664525u + 1013904223u; tokens[i] = 1000 + (int)((s >> 8) % 100000); }
        var positions = Enumerable.Range(0, total).ToArray();

        int V = config.VocabSize;
        float[][] full = new float[total][];
        model.ResetSequenceState();
        using (var all = model.Forward(tokens, positions, -1))
            for (int i = 0; i < total; i++)
                full[i] = new ReadOnlySpan<float>((float*)all.DataPointer + (long)i * V, V).ToArray();

        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 512);
        using var state = model.CreateSequenceState()!;
        int[] chunkEnds = [100, 200, 280];
        int start = 0;
        foreach (int end in chunkEnds)
        {
            using var rows = model.Forward(tokens.AsSpan(start, end - start), positions.AsSpan(start, end - start), -1, kv, (ISsmState)state);
            // The last row of the chunk, and the first (which depends on the conv state + cache left by the previous chunk).
            AssertClose(full[end - 1], new ReadOnlySpan<float>((float*)rows.DataPointer + (long)(end - start - 1) * V, V), $"chunk end {end}");
            AssertClose(full[start], new ReadOnlySpan<float>((float*)rows.DataPointer, V), $"chunk start {start}");
            start = end;
        }
        for (int i = 280; i < total; i++)
        {
            using var row = model.Forward(tokens.AsSpan(i, 1), positions.AsSpan(i, 1), -1, kv, (ISsmState)state);
            AssertClose(full[i], new ReadOnlySpan<float>((float*)row.DataPointer, V), $"decode {i}");
        }
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        float maxAbs = 0, maxDiff = 0;
        int ea = 0, aa = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(expected[i]));
            maxDiff = Math.Max(maxDiff, Math.Abs(expected[i] - actual[i]));
            if (expected[i] > expected[ea]) ea = i;
            if (actual[i] > actual[aa]) aa = i;
        }
        Assert.True(maxDiff <= 2e-3f * maxAbs, $"{what}: max |dlogit| {maxDiff} vs scale {maxAbs}");
        // A real argmax flip is only tolerated when the two candidates are within the numeric noise.
        if (ea != aa) Assert.True(Math.Abs(expected[ea] - expected[aa]) <= 4 * maxDiff + 1e-6f, $"{what}: argmax {ea} vs {aa}");
    }

    private static int ArgMax(ITensor t, int vocab, int rowsToSkip)
    {
        using (t)
        {
            var span = new ReadOnlySpan<float>((float*)t.DataPointer + (long)rowsToSkip * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++) if (span[i] > span[best]) best = i;
            return best;
        }
    }
}
