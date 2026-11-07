using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1562: Vulkan LFM3 against the CPU model on the real LFM3-1.7B Q8_0 GGUF. 300 tokens cross the 256-token sliding window, so a
/// window dropped by one of the Vulkan attention kernels, a wrong partial-rope layout on the global layers, a missing per-head gate
/// or a missing sandwich norm all show as large logit differences. Hold the Strix:GPU lock when running (<c>Category=GPU</c>).
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class Lfm3VulkanParityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    /// <summary>2560 wikitext-2 token ids (LFM3 tokenizer): natural activations, and long enough that the global layers' unrotated RoPE
    /// tail (frequencies 1e6^(-i/64) for pairs 32..63, ~1e-3 rad per position at i = 32) is observable.</summary>
    private static int[] NaturalTokens()
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Models", "Lfm3", "Fixtures", "lfm3-wikitext-tokens.txt"))
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_LFM3_GGUF", "LiquidAI", "LFM3-1.7B-GGUF", "LFM3-1.7B-Q8_0.gguf").Path;

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    private static float[] Row(ITensor t, int row)
    {
        int v = t.Shape[1];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * v, v).ToArray();
    }

    /// <summary>
    /// KL(cpu || vulkan) over the softmax of the two logit rows, plus an argmax check that tolerates only a flip between near-tied
    /// candidates. Random-token input gives broad distributions whose raw logits differ by several percent between the CPU's Q8-activation
    /// kernels and Vulkan's F32 activations, so a max-logit tolerance is the wrong instrument; a divergent window, rope, gate or norm
    /// moves the KL by orders of magnitude (measured with mutants, see the PR).
    /// </summary>
    private void AssertClose(float[] cpu, float[] vk, double maxKl, string what)
    {
        double mc = cpu.Max(), mv = vk.Max(), zc = 0, zv = 0;
        for (int i = 0; i < cpu.Length; i++) { zc += Math.Exp(cpu[i] - mc); zv += Math.Exp(vk[i] - mv); }
        double lzc = Math.Log(zc) + mc, lzv = Math.Log(zv) + mv, kl = 0;
        int ac = 0, av = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            double lpc = cpu[i] - lzc, lpv = vk[i] - lzv;
            kl += Math.Exp(lpc) * (lpc - lpv);
            if (cpu[i] > cpu[ac]) ac = i;
            if (vk[i] > vk[av]) av = i;
        }
        output.WriteLine($"{what}: KL {kl:E3}");
        Assert.True(kl <= maxKl, $"{what}: KL(cpu||vulkan) = {kl:E3} (> {maxKl:E1})");
        if (ac != av) Assert.True(cpu[ac] - cpu[av] <= 0.25f, $"{what}: argmax {ac} vs {av} with CPU gap {cpu[ac] - cpu[av]:F3}");
    }

    private static ITensor Step(IModel model, ReadOnlyMemory<int> tokens, ReadOnlyMemory<int> positions, IKvCache kv, ISsmState state)
        => model.ForwardBatch([new SequenceForwardRequest { TokenIds = tokens, Positions = positions, KvCache = kv, SsmState = state, LastTokenLogitsOnly = true }], -1)[0];

    [SkippableFact]
    public void Vulkan_MatchesCpu_FullSequenceAndIncrementalAcrossTheSlidingWindow()
    {
        string? path = FindGguf();
        Skip.If(path is null, "LFM3 GGUF not found: set DOTLLM_LFM3_GGUF or fetch LiquidAI/LFM3-1.7B-GGUF.");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cpu = Lfm3TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(4));
        using var device = VulkanDevice.Create();
        var (vkModel, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, SpvDir());
        using var vk = (IDisposable)vkModel;

        // Natural text, 2560 tokens: crosses the 256-token sliding window ~10x and gives the global layers 2000+ positions of relative offset.
        int[] tokens = NaturalTokens();
        int total = tokens.Length;
        var positions = Enumerable.Range(0, total).ToArray();

        int[] chunkEnds = [1000, 2000, 2500];
        const int decodeSteps = 24;
        int last = chunkEnds[^1] + decodeSteps;

        // CPU oracle: the same schedule through the CPU KV cache + conv state (pinned equal to the cacheless forward by Lfm3CpuModelTests).
        var cpuRows = new List<float[]>();
        using (var ckv = new SimpleKvCache(cpu.AttentionLayerCount, config.NumKvHeads, config.HeadDim, tokens.Length + 8))
        using (var cst = cpu.CreateSequenceState()!)
        {
            int cs = 0;
            foreach (int end in chunkEnds)
            {
                using var r = Step(cpu, tokens.AsMemory(cs, end - cs), positions.AsMemory(cs, end - cs), ckv, (ISsmState)cst);
                cpuRows.Add(Row(r, r.Shape[0] - 1));
                cs = end;
            }
            for (int i = chunkEnds[^1]; i < last; i++)
            {
                using var r = Step(cpu, tokens.AsMemory(i, 1), positions.AsMemory(i, 1), ckv, (ISsmState)cst);
                cpuRows.Add(Row(r, 0));
            }
        }

        // Vulkan: chunked prefill + single-token decode through the sparse KV cache and device conv state.
        long clipped0 = VulkanLfm3TransformerModel.SlidingClippedDecodes, flash0 = VulkanLfm3TransformerModel.SlidingFlashPrefills;
        vkModel.ResetSequenceState();
        using var kv = kvFactory(tokens.Length + 8);
        using var state = vkModel.CreateSequenceState()!;
        int start = 0, k = 0;
        foreach (int end in chunkEnds)
        {
            using var lastRow = Step(vkModel, tokens.AsMemory(start, end - start), positions.AsMemory(start, end - start), kv, (ISsmState)state);
            AssertClose(cpuRows[k++], Row(lastRow, lastRow.Shape[0] - 1), 0.005, $"prefill chunk ending {end}");
            start = end;
        }
        for (int i = chunkEnds[^1]; i < last; i++)
        {
            using var row = Step(vkModel, tokens.AsMemory(i, 1), positions.AsMemory(i, 1), kv, (ISsmState)state);
            AssertClose(cpuRows[k++], Row(row, row.Shape[0] - 1), 0.005, $"decode {i}");
        }
        // #1572 path proof: a parity pass is vacuous for the sliding-window fix unless the clipped decode range and the GQA flash prefill actually ran.
        if (!VulkanLfm3TransformerModel.SlidingOptDisabled)
        {
            Assert.True(VulkanLfm3TransformerModel.SlidingClippedDecodes > clipped0, "no sliding-layer decode used a clipped KV range");
            Assert.True(VulkanLfm3TransformerModel.SlidingFlashPrefills > flash0, "no sliding-layer prefill used the GQA flash kernel");
        }
    }

    /// <summary>
    /// #1649: a prefill cut into query-row slices (budget forced tiny so it happens at 2560 rows) must match the same prefill run as
    /// one forward, across the 256-token sliding window and the global layers. Also asserts slicing really happened, so a gate that
    /// silently stops slicing cannot pass this as "unchunked equals unchunked".
    /// </summary>
    [SkippableFact]
    public void RowSlicedPrefill_MatchesSingleForward()
    {
        string? path = FindGguf();
        Skip.If(path is null, "LFM3 GGUF not found: set DOTLLM_LFM3_GGUF or fetch LiquidAI/LFM3-1.7B-GGUF.");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (vkModel, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, SpvDir());
        using var vk = (IDisposable)vkModel;
        var lfm3 = (VulkanLfm3TransformerModel)vkModel;
        // 2557 rows: not divisible by the slice count, so the remainder-distribution (extra rows) is exercised.
        int[] tokens = NaturalTokens().Take(2557).ToArray();
        var positions = Enumerable.Range(0, tokens.Length).ToArray();

        float[] Run(VulkanPrefillSubmitBudget budget)
        {
            lfm3.SubmitBudgetForTest = budget;
            lfm3.ResetSequenceState();
            using var kv = kvFactory(tokens.Length + 8);
            using var state = lfm3.CreateSequenceState()!;
            using var r = Step(lfm3, tokens, positions, kv, (ISsmState)state);
            var row = Row(r, r.Shape[0] - 1);
            using var d = Step(lfm3, tokens.AsMemory(0, 1), positions.AsMemory(tokens.Length - 1, 1), kv, (ISsmState)state);
            return row;
        }

        int before = lfm3.RowSliceForwardsForTest;
        var whole = Run(new VulkanPrefillSubmitBudget(1000));
        Assert.Equal(before, lfm3.RowSliceForwardsForTest);
        var sliced = Run(new VulkanPrefillSubmitBudget(0.02));
        Assert.True(lfm3.RowSliceForwardsForTest - before >= 4, "budget did not force row slicing");
        AssertClose(whole, sliced, 1e-4, "row-sliced vs single-forward prefill");
        int aw = Array.IndexOf(whole, whole.Max()), asl = Array.IndexOf(sliced, sliced.Max());
        Assert.Equal(aw, asl);
    }
}
