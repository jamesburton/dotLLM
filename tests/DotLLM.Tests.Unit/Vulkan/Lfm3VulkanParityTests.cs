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
    private static readonly int[] NaturalTokens =
    [
        207, 550, 6102, 61512, 385, 550, 67274, 6102, 61512, 385, 355, 401, 3478, 2815, 1228, 8086, 309, 24900, 14498,
        1200, 904, 819, 267, 21021, 1762, 21, 40, 37892, 3170, 383, 278, 8086, 3587, 440, 8047, 296, 229, 585, 24, 1200,
        969, 452, 5614, 467, 267, 37892, 3170, 296, 278, 1431, 2864, 795, 4484, 467, 14538, 29574, 540, 1228, 643, 452,
        5312, 296, 229, 585, 25, 462, 278, 7565, 7419, 18931, 1200, 904, 819, 267, 21021, 3170, 296, 278, 8086, 3587, 31975,
        2409, 1015, 290, 296, 229, 585, 26, 1200, 529, 229, 585, 28, 61512, 385, 28078, 267, 3170, 405, 496, 32327, 496,
        296, 278, 13636, 496, 336, 51647, 951, 91, 15590, 496, 302, 278, 8086, 3587, 440, 7987, 53770, 3791, 449, 54618,
        16023, 17501, 4409, 32107, 309, 55589, 12675, 81, 1200, 904, 452, 6054, 296, 278, 229, 585, 29, 24900, 43420, 302,
        278, 13544, 42717, 2273, 1431, 32569, 34735, 1228, 643, 452, 5312, 462, 278, 3082, 399, 18931, 296, 48298, 309,
        278, 8597, 644, 61634, 38147, 296, 4595, 1200, 904, 452, 10378, 467, 2409, 336, 3389, 1255, 309, 54618, 16023,
        6122, 1084, 719, 1233, 1228, 1450, 1731, 895, 10823, 1228, 15649, 13186, 1228, 44863, 15328, 775, 1228, 39614,
        586, 17413, 309, 124708, 6828, 1200, 64019, 529, 229, 585, 30, 1228, 61512, 385, 54618, 16023, 1084, 719, 1233,
        296, 278, 1431, 43799, 1373, 4484, 467, 4409, 56756, 31043, 1200, 904, 8124, 383, 267, 229, 585, 30, 13636, 302,
        278, 8086, 3587, 1228, 50269, 1228, 5614, 467, 267, 3170, 296, 278, 229, 585, 31, 24900, 3392, 302, 1712, 310,
        6620, 411, 10378, 467, 7541, 381, 408, 425, 1019, 1200, 1712, 310, 6620, 411, 452, 5312, 462, 18001, 18931, 296,
        278, 4595, 49249, 302, 28696, 370, 35236, 309, 30187, 3844, 1200, 61512, 385, 54618, 296, 1046, 6922, 296, 229,
        585, 32, 1228, 5751
    ];

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
        => model.ForwardBatch([new SequenceForwardRequest { TokenIds = tokens, Positions = positions, KvCache = kv, SsmState = state }], -1)[0];

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

        // The first 300 token ids of the wikitext-2 test split under the LFM3 tokenizer (natural text: broad, realistic activations).
        int[] tokens = NaturalTokens;
        int total = tokens.Length;
        var positions = Enumerable.Range(0, total).ToArray();

        // CPU oracle rows (cacheless, all rows).
        cpu.ResetSequenceState();
        float[][] cpuRows = new float[total][];
        using (var all = cpu.Forward(tokens, positions, -1))
            for (int i = 0; i < total; i++) cpuRows[i] = Row(all, i);

        // Vulkan: chunked prefill + single-token decode through the sparse KV cache and device conv state.
        vkModel.ResetSequenceState();
        using var kv = kvFactory(512);
        using var state = vkModel.CreateSequenceState()!;
        int start = 0;
        foreach (int end in new[] { 100, 200, 280 })
        {
            using var last = Step(vkModel, tokens.AsMemory(start, end - start), positions.AsMemory(start, end - start), kv, (ISsmState)state);
            AssertClose(cpuRows[end - 1], Row(last, last.Shape[0] - 1), 0.02, $"prefill chunk ending {end}");
            start = end;
        }
        for (int i = 280; i < total; i++)
        {
            using var row = Step(vkModel, tokens.AsMemory(i, 1), positions.AsMemory(i, 1), kv, (ISsmState)state);
            AssertClose(cpuRows[i], Row(row, row.Shape[0] - 1), 0.02, $"decode {i}");
        }
    }
}
