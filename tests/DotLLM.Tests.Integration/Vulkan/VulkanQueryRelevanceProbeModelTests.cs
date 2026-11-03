using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1531 step 4 on the real model (Qwen3-4B-Instruct-2507 Q4_K_M): the device query-relevance probe attached to a cached forward.
/// Checks what a kernel-level reference cannot: the probe reads the post-RoPE Q the model actually used, leaves the forward untouched,
/// normalises over every visible key, and points at the context tokens an induction-style query repeats.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanQueryRelevanceProbeModelTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQueryRelevanceProbeModelTests(ITestOutputHelper output) => _out = output;

    private static unsafe float[] Copy(ITensor t)
    {
        var r = new float[t.ElementCount];
        new ReadOnlySpan<float>((void*)t.DataPointer, r.Length).CopyTo(r);
        return r;
    }

    [SkippableFact]
    public void Qwen3_4B_Probe_IsInertNormalisedAndFindsTheRepeatedPhrase()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fx.Path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, VulkanModelLoader.ResolveSpvDir());
        string spv = VulkanModelLoader.ResolveSpvDir();

        // Context: 72 filler tokens with a distinctive 6-token phrase at 40..45; query repeats the first four phrase tokens, then continues.
        const int L = 72;
        int[] ctx = Enumerable.Range(0, L).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
        int[] phrase = { 21003, 17771, 23459, 19211, 25003, 16661 };
        for (int i = 0; i < phrase.Length; i++) ctx[40 + i] = phrase[i];
        int[] query = { 9001, 21003, 17771, 23459 };   // "<filler> A B C": the induction target is the token after C, i.e. context 43
        int[] ctxPos = Enumerable.Range(0, L).ToArray();
        int[] qPos = Enumerable.Range(L, query.Length).ToArray();

        float[] Run(VulkanQueryRelevanceProbe? probe, out float[] logits, int[]? q = null)
        {
            q ??= query;
            using var cache = (VulkanKvCache)vk.CreateKvCache(160);
            using (vk.Forward(ctx, ctxPos, 0, cache)) { }
            vk.AttachQueryRelevanceProbe(probe);
            try
            {
                using ITensor t = vk.Forward(q, qPos, 0, cache);
                logits = Copy(t);
            }
            finally { vk.AttachQueryRelevanceProbe(null); }
            Assert.Equal(L + q.Length, cache.CurrentLength);
            return probe?.ReadScores() ?? [];
        }

        Run(null, out var plain);

        using var probeA = vk.CreateQueryRelevanceProbe(spv);
        probeA.Begin(L, query.Length, cfg.NumAttentionHeads);
        float[] scores = Run(probeA, out var withProbe);
        Assert.Equal(plain, withProbe);   // inert: the forward is bit-identical with the probe attached

        // Normalisation: scoring ALL visible keys (context + the query rows' own) gives total mass 1 per (layer, row), and the context part
        // is exactly what the context-only probe reports.
        using var probeB = vk.CreateQueryRelevanceProbe(spv);
        probeB.Begin(L + query.Length, query.Length, cfg.NumAttentionHeads);
        float[] all = Run(probeB, out _);
        double total = all.Sum(x => (double)x);
        Assert.InRange(total, 0.999, 1.001);
        for (int j = 0; j < L; j++) Assert.Equal(all[j], scores[j]);

        var top = Enumerable.Range(0, L).OrderByDescending(j => scores[j]).Take(8).ToArray();
        // The first token is an attention sink (most of the mass, whatever the query); judge relevance on the rest of the context.
        double Share(float[] sc) => Enumerable.Range(40, phrase.Length).Sum(j => (double)sc[j]) / sc.Skip(1).Sum(x => (double)x);
        double shareRepeat = Share(scores);

        // Control arm: a query of the same length that does NOT repeat the phrase must give the phrase no special weight.
        int[] filler = { 9001, 9002, 9003, 9004 };
        using var probeC = vk.CreateQueryRelevanceProbe(spv);
        probeC.Begin(L, filler.Length, cfg.NumAttentionHeads);
        double shareControl = Share(Run(probeC, out _, filler));
        _out.WriteLine($"[1531 step4] top-8 context positions: {string.Join(", ", top.Select(j => $"{j}:{scores[j]:E2}"))}");
        _out.WriteLine($"[1531 step4] phrase window [40,46) share of non-sink context mass: repeat query {shareRepeat:F3}, filler control {shareControl:F3}, uniform {phrase.Length / (double)(L - 1):F3}; total mass over all visible keys {total:F4}");
        Assert.True(shareRepeat > 1.5 * shareControl && shareRepeat > 2 * phrase.Length / (double)(L - 1),
            $"the repeated phrase should draw clearly more attention than under a control query: {shareRepeat:F3} vs {shareControl:F3}");
    }
}
