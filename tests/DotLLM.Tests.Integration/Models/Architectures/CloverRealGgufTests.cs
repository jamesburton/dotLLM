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
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Shared fixture plumbing for the real Clover-35B-A3B-Preview GGUF (issue #1794). Set <c>DOTLLM_CLOVER_GGUF</c>, or keep the file in
/// the HF hub cache (<c>hf download Diffuspace/Clover-35B-A3B-Preview-GGUF Clover-35B-A3B-Preview-Q4_K_M.gguf</c>). Never downloads.
/// </summary>
internal static class CloverRealGguf
{
    /// <summary>"The capital of France is", tokenized by llama.cpp (<c>llama-tokenize --ids</c>, no BOS: add_bos_token = false).</summary>
    public static readonly int[] Prompt = [760, 6511, 314, 9338, 369];

    /// <summary>" Paris": llama.cpp's greedy next token for <see cref="Prompt"/> (master 32bdc3a + the #1794 oracle patch, <c>-dev none</c>).</summary>
    public const int Paris = 11751;

    /// <summary>"." : llama.cpp's greedy token after " Paris" (<c>The capital of France is Paris.</c> = [.., 11751, 13]).</summary>
    public const int Period = 13;

    public static string? Resolve() => TestFixtureResolver.ResolveFile(
        "DOTLLM_CLOVER_GGUF", "Diffuspace", "Clover-35B-A3B-Preview-GGUF",
        "Clover-35B-A3B-Preview-Q4_K_M.gguf", "Clover-35B-A3B-Preview-Q8_0.gguf", "Clover-35B-A3B-Preview-Q4_0.gguf").Path;

    public static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    public static (int Top, float Margin) Top1(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        float second = float.NegativeInfinity;
        for (int i = 0; i < row.Length; i++) if (i != best && row[i] > second) second = row[i];
        return (best, row[best] - second);
    }

    /// <summary>KL(p || q) of the softmaxes of two logit rows, in nats.</summary>
    public static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp), lzq = Math.Log(zq), kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - mp - lzp, lq = q[i] - mq - lzq;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }

    public static float[] CpuLastRow(GgufFile gguf, ModelConfig config) => CpuLastRow(gguf, config, Prompt);

    public static float[] CpuLastRow(GgufFile gguf, ModelConfig config, int[] tokens)
    {
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        int[] positions = Enumerable.Range(0, tokens.Length).ToArray();
        using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 64);
        using ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv);
        return LastRow(logits, config.VocabSize);
    }
}

/// <summary>
/// CPU causal-path gate on the real Clover GGUF (#1794): the next token after "The capital of France is" must be llama.cpp's " Paris".
/// The sensitivity arm reloads the file without the Clover sidecar, i.e. reads <c>ssm_a</c> (which this file stores as the raw
/// <c>A_log</c>) as qwen35moe's <c>-exp(A_log)</c>: that model emits garbage, so the gate provably sees the transform.
/// </summary>
public sealed class CloverRealGgufCpuTests
{
    private readonly ITestOutputHelper _output;

    public CloverRealGgufCpuTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Cpu_CausalNextToken_MatchesLlamaCpp_AndNeedsTheALogTransform()
    {
        string? path = CloverRealGguf.Resolve();
        Skip.If(path is null, "Clover GGUF not found: set DOTLLM_CLOVER_GGUF or hf-download Diffuspace/Clover-35B-A3B-Preview-GGUF.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        var clover = Assert.IsType<CloverConfig>(config.Clover);
        Assert.Equal(32, clover.BlockSize);
        Assert.Equal(151669, clover.MaskTokenId);
        Assert.True(clover.MoeBlockCoresetEnabled);

        var (top, margin) = CloverRealGguf.Top1(CloverRealGguf.CpuLastRow(gguf, config));
        _output.WriteLine($"[clover cpu] top1={top} margin={margin:F3}");
        Assert.Equal(CloverRealGguf.Paris, top);

        // Untransformed, the positive A_log entries make exp(g) > 1: the GDN state blows up to inf/NaN within a few tokens, and the
        // NaN router logits then trip MoeSwiGluMlp.Route (IndexOutOfRangeException, #1797). Either outcome is the divergence this
        // arm exists to prove; a finite " Paris" would mean the gate cannot see the transform.
        try
        {
            float[] wrong = CloverRealGguf.CpuLastRow(gguf, config with { Clover = null });
            var (wrongTop, wrongMargin) = CloverRealGguf.Top1(wrong);
            bool finite = wrong.All(float.IsFinite);
            _output.WriteLine($"[clover cpu, no A_log transform] finite={finite} top1={wrongTop} margin={wrongMargin:F3}");
            Assert.False(finite && wrongTop == CloverRealGguf.Paris, "the untransformed model still predicts \" Paris\"");
        }
        catch (IndexOutOfRangeException ex)
        {
            _output.WriteLine($"[clover cpu, no A_log transform] forward failed (non-finite routing): {ex.Message}");
        }
    }
}

/// <summary>
/// Vulkan causal-path gate on the real Clover GGUF (#1794), teacher-forced over a 66-token paragraph: Vulkan prefills the first 16 tokens
/// (the grouped MoE prefill path) and DECODES the other 50 one at a time (the fused single-token MoE layer and the Q4_K / Q6_K / Q8_0 down
/// MMVQ kernels, which perplexity never reaches), and every row is scored against the CPU host's one-pass logits.
/// </summary>
/// <remarks>
/// Single positions cannot carry this gate: Clover's causal distributions are high-entropy, so ANY two implementations differ at mean KL
/// ~0.05 and on ~11% of top-1s over wikitext-2 (llama.cpp CPU vs llama.cpp Vulkan: 0.052 / 90.0%; Qwen3.6-35B-A3B under the same pairs:
/// 0.007 / 96%). The bounds are aggregates set above that floor; the substitution arm (RoPE theta 1e7 -> 1e4, a finite, plausible wrong
/// config) shows they can fail. No non-finite arm on the GPU: NaN router logits would feed garbage expert ids to the device.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CloverRealGgufVulkanTests
{
    /// <summary>"The Roman Empire was the post-Republican state of ancient Rome. ..." tokenized by llama.cpp (no BOS).</summary>
    private static readonly int[] Paragraph =
    [
        760, 12386, 19825, 557, 279, 1675, 12, 66139, 1528, 314, 13437, 21047, 13, 1049, 369, 8524, 15513, 310, 2977, 279, 4031, 321,
        17435, 20632, 539, 279, 47091, 2614, 4756, 57177, 579, 23571, 314, 12924, 5728, 1172, 279, 43051, 344, 303, 220, 17, 22, 17501,
        13, 1049, 5064, 37167, 303, 4357, 11, 4634, 9871, 11, 321, 10550, 13229, 11, 321, 557, 20632, 539, 945, 692, 1052, 13,
    ];

    private const int PrefillRows = 16;
    private const double MaxMeanKl = 0.08;   // measured 0.029 (2026-11-14); corpus floor ~0.05; wrong-theta arm 0.166
    private const double MinTop1 = 0.75;

    private readonly ITestOutputHelper _output;

    public CloverRealGgufVulkanTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Vulkan_TeacherForcedPrefillAndDecode_TrackTheCpuHost()
    {
        string? path = CloverRealGguf.Resolve();
        Skip.If(path is null, "Clover GGUF not found: set DOTLLM_CLOVER_GGUF or hf-download Diffuspace/Clover-35B-A3B-Preview-GGUF.");
        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPV directory not found.");
        VulkanDevice device;
        try { device = VulkanDevice.Create(); }
        catch (Exception ex) { Skip.If(true, $"Vulkan unavailable: {ex.Message}"); return; }

        using (device)
        {
            using var gguf = GgufFile.Open(path!);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            float[][] cpu = CpuRows(gguf, config);

            var (kl, top1) = Score(cpu, VulkanRows(device, gguf, config, spvDir!));
            _output.WriteLine($"[clover vulkan] rows {PrefillRows - 1}..{Paragraph.Length - 2}: meanKL(cpu||vk)={kl:E3} top1={top1:F3}");
            Assert.True(kl < MaxMeanKl, $"mean KL(cpu || vulkan) = {kl:E3}");
            Assert.True(top1 >= MinTop1, $"top-1 agreement {top1:F3}");

            // Substitution arm: the same host with a plausible wrong RoPE base must fail at least one bound.
            var wrong = config with { RoPEConfig = config.RoPEConfig!.Value with { Theta = 10_000f } };
            var (wkl, wtop1) = Score(cpu, VulkanRows(device, gguf, wrong, spvDir!));
            _output.WriteLine($"[clover vulkan, rope theta 1e4] meanKL={wkl:E3} top1={wtop1:F3}");
            Assert.True(wkl >= MaxMeanKl || wtop1 < MinTop1, $"wrong-theta run passed the gate (KL {wkl:E3}, top1 {wtop1:F3}): the bounds cannot fail");
        }
    }

    private static unsafe float[][] CpuRows(GgufFile gguf, ModelConfig config)
    {
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 128);
        using ITensor logits = model.Forward(Paragraph, Enumerable.Range(0, Paragraph.Length).ToArray(), deviceId: -1, kv);
        Assert.Equal(Paragraph.Length, logits.Shape[0]);
        int v = config.VocabSize;
        var rows = new float[Paragraph.Length][];
        for (int r = 0; r < rows.Length; r++)
        {
            rows[r] = new float[v];
            new ReadOnlySpan<float>((float*)logits.DataPointer + (long)r * v, v).CopyTo(rows[r]);
        }
        return rows;
    }

    /// <summary>Row r predicts token r+1. Rows 0..14 come from the 16-row prefill (only its last row is returned); 15..64 from decode.</summary>
    private static float[]?[] VulkanRows(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
    {
        var rows = new float[]?[Paragraph.Length];
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        using var kv = model.CreateKvCache(128);
        using (ITensor pre = model.Forward(Paragraph.AsSpan(0, PrefillRows), Enumerable.Range(0, PrefillRows).ToArray(), deviceId: -1, kv))
            rows[PrefillRows - 1] = CloverRealGguf.LastRow(pre, config.VocabSize);
        for (int t = PrefillRows; t < Paragraph.Length - 1; t++)
        {
            using ITensor step = model.Forward(Paragraph.AsSpan(t, 1), [t], deviceId: -1, kv);
            rows[t] = CloverRealGguf.LastRow(step, config.VocabSize);
        }
        return rows;
    }

    private static (double MeanKl, double Top1) Score(float[][] cpu, float[]?[] vk)
    {
        double kl = 0;
        int agree = 0, n = 0;
        for (int r = 0; r < vk.Length; r++)
        {
            if (vk[r] is not { } row) continue;
            kl += CloverRealGguf.Kl(cpu[r], row);
            if (CloverRealGguf.Top1(cpu[r]).Top == CloverRealGguf.Top1(row).Top) agree++;
            n++;
        }
        return (kl / n, (double)agree / n);
    }

    private static string? ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null;
    }
}
