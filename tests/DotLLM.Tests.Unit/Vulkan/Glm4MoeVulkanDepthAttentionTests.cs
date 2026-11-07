using System.Diagnostics;
using System.Reflection;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1612: the GQA-grouped split-KV decode pass on the real glm4moe file (REAP-82B IQ4_XS, 96 / 8 heads = two 6-head sub-groups
/// per KV head) against the per-query-head pass, from the SAME cache state at several depths. Opt-in and slow (~44 GiB load):
/// <c>DOTLLM_GLM4MOE_DEPTH_AB=1</c> plus the file (<c>DOTLLM_GLM4MOE_GGUF</c> or the fixture resolver). Needs the Strix:GPU and
/// Strix:MEMORY locks.
/// </summary>
/// <remarks>
/// <para>Per depth (default 512, 4096, 8192; <c>DOTLLM_GLM4MOE_DEPTHS</c>): the first decode step's logits under both passes (KL, argmax),
/// 32 greedy tokens per arm (first divergence reported; exact identity over a long generation is not an invariant on a quantised
/// model, see #331), and the per-arm wall time of the 32 steps, arms interleaved A B B A in one process. The grouped arm must have
/// taken the grouped pass (<see cref="VulkanSplitKvAttentionKernel.GqaDispatches"/>).</para>
/// <para>Real wikitext ids come from the GGUF tokenizer over the LF corpus (<c>DOTLLM_WIKITEXT_LF</c>, default
/// <c>~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw</c>).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Glm4MoeVulkanDepthAttentionTests
{
    private const int Steps = 32;
    private const int PrefillChunk = 512;

    private readonly ITestOutputHelper _out;
    public Glm4MoeVulkanDepthAttentionTests(ITestOutputHelper output) => _out = output;

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    /// <summary>KL(softmax(p) || softmax(q)) in nats.</summary>
    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double mx = x.Max(), s = 0;
            foreach (float f in x) s += Math.Exp(f - mx);
            double ls = Math.Log(s);
            return x.Select(f => f - mx - ls).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < lp.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static float[] Step(VulkanTransformerModel m, VulkanKvCache cache, ReadOnlySpan<int> ids, int pos0)
    {
        int[] pos = Enumerable.Range(pos0, ids.Length).ToArray();
        using ITensor t = m.Forward(ids, pos, deviceId: -1, cache);
        return Copy(t);
    }

    private static string? ResolveModel()
    {
        string? p = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        if (!string.IsNullOrEmpty(p) && File.Exists(p)) return p;
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_GLM45_AIR_REAP_GGUF", "bartowski", "cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF", "cerebras_GLM-4.5-Air-REAP-82B-A12B-IQ4_XS.gguf");
        return loc.Found ? loc.Path : null;
    }

    [SkippableFact]
    public void GroupedSplitKv_MatchesPerHead_AtDepth()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_DEPTH_AB") == "1", "DOTLLM_GLM4MOE_DEPTH_AB=1 to enable (loads ~44 GiB).");
        string? path = ResolveModel();
        Skip.If(path is null, "GLM-4.5-Air REAP IQ4_XS not found (DOTLLM_GLM4MOE_GGUF)");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int[] depths = (Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_DEPTHS") ?? "512,4096,8192")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).OrderBy(d => d).ToArray();
        int maxDepth = depths[^1];

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        _out.WriteLine($"heads {cfg.NumAttentionHeads} / kv {cfg.NumKvHeads} / head_dim {cfg.HeadDim} / layers {cfg.NumLayers}");
        int group = cfg.NumAttentionHeads / cfg.NumKvHeads;
        Assert.True(group > VulkanSplitKvAttentionKernel.GqaMaxGroup, $"this test is about a sub-grouped shape; group is {group}");

        int[] ids = WikitextIds(gguf, maxDepth + 1);

        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir);
        _out.WriteLine("post-load: " + device.MemorySnapshot());

        var kernel = (VulkanSplitKvAttentionKernel?)typeof(VulkanTransformerModel)
            .GetField("_splitKvAttentionKvF16", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model);
        Assert.NotNull(kernel);
        Assert.True(kernel!.PreferGqa, "glm4moe's dense F16 host must opt into the grouped pass (#1612)");
        Assert.True(kernel.UsesGqa(cfg.NumAttentionHeads, cfg.NumKvHeads, cfg.HeadDim), "the grouped pass must serve the 96 / 8 / 128 shape");
        _out.WriteLine($"grouped v2: {kernel.UsesGqa2(cfg.NumAttentionHeads, cfg.NumKvHeads, cfg.HeadDim)}; sub-groups {VulkanSplitKvAttentionKernel.GqaSubGroups(cfg.NumAttentionHeads, cfg.NumKvHeads)}");

        using var cache = model.CreateKvCache(maxDepth + Steps + 16);
        Assert.Equal(VulkanKvStorage.F16, cache.Storage);

        int filled = 0;
        _out.WriteLine("depth\tarm\trun\tms/step\ttok/s\tfirstDivergence\tgqaDispatches");
        foreach (int depth in depths)
        {
            var swPre = Stopwatch.StartNew();
            while (filled < depth)
            {
                int n = Math.Min(PrefillChunk, depth - filled);
                Step(model, cache, ids.AsSpan(filled, n), filled);
                filled += n;
            }
            _out.WriteLine($"prefilled to {filled} in {swPre.Elapsed.TotalSeconds:F1} s");
            Assert.Equal(depth, cache.CurrentLength);

            // Arms interleaved A B B A; every run starts from the identical cache state (rollback to `depth` after each).
            float[]? firstPerHead = null, firstGrouped = null;
            int[]? seqPerHead = null, seqGrouped = null;
            double bestPerHead = double.MaxValue, bestGrouped = double.MaxValue;
            foreach (bool grouped in new[] { false, true, true, false })
            {
                kernel.PreferGqa = grouped;
                long before = kernel.GqaDispatches;
                var seq = new int[Steps];
                float[]? first = null;
                int tok = ids[depth];
                var sw = Stopwatch.StartNew();
                for (int s = 0; s < Steps; s++)
                {
                    float[] logits = Step(model, cache, [tok], depth + s);
                    first ??= logits;
                    tok = Argmax(logits);
                    seq[s] = tok;
                }
                double ms = sw.Elapsed.TotalMilliseconds / Steps;
                long dispatches = kernel.GqaDispatches - before;
                cache.Rollback(depth);
                Assert.Equal(depth, cache.CurrentLength);

                if (grouped)
                {
                    Assert.True(dispatches > 0, "the grouped arm did not take the grouped pass");
                    firstGrouped ??= first;
                    seqGrouped ??= seq;
                    bestGrouped = Math.Min(bestGrouped, ms);
                }
                else
                {
                    Assert.Equal(0, dispatches);
                    firstPerHead ??= first;
                    seqPerHead ??= seq;
                    bestPerHead = Math.Min(bestPerHead, ms);
                }
                int firstDiv = grouped ? FirstDivergence(seqPerHead, seq) : FirstDivergence(seqGrouped, seq);
                _out.WriteLine($"{depth}\t{(grouped ? "grouped" : "per-head")}\t{ms:F1}\t{1000 / ms:F2}\t{(firstDiv < 0 ? "none/na" : firstDiv.ToString())}\t{dispatches}");
            }

            double kl = Kl(firstPerHead!, firstGrouped!);
            int aP = Argmax(firstPerHead!), aG = Argmax(firstGrouped!);
            int div = FirstDivergence(seqPerHead!, seqGrouped!);
            _out.WriteLine($"depth {depth}: first-step KL(per-head||grouped) {kl:E3}, argmax {aP}/{aG}, greedy identity over {Steps}: {(div < 0 ? "identical" : $"first divergence at step {div}")}; " +
                           $"best ms/step per-head {bestPerHead:F1} vs grouped {bestGrouped:F1} ({bestPerHead / bestGrouped:F2}x)");
            // Reassociation-only difference between the passes: well under the file's own activation-precision floor (KL ~0.03 vs llama.cpp, #1600).
            Assert.True(kl < 1e-3, $"depth {depth}: grouped pass diverges from the per-head pass at the first step: KL {kl}");
            Assert.Equal(aP, aG);
        }

        // Per-head arms re-enabled the per-head pass last; leave the model on its default.
        kernel.PreferGqa = true;
    }

    private static int FirstDivergence(int[]? a, int[] b)
    {
        if (a is null) return -1;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (a[i] != b[i]) return i;
        return -1;
    }

    private int[] WikitextIds(GgufFile gguf, int count)
    {
        string corpus = Environment.GetEnvironmentVariable("DOTLLM_WIKITEXT_LF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        Skip.IfNot(File.Exists(corpus), $"wikitext LF corpus not found at {corpus} (DOTLLM_WIKITEXT_LF)");
        string text = File.ReadAllText(corpus);
        // ~4 chars per token; take enough text, then cut the ids.
        int chars = Math.Min(text.Length, count * 8 + 4096);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(text[..chars]);
        _out.WriteLine($"tokenised {chars} chars -> {ids.Length} ids (need {count})");
        Assert.True(ids.Length >= count, $"corpus too short: {ids.Length} ids for {count}");
        return ids[..count];
    }
}
