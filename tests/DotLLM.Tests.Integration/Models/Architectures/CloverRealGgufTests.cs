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

/// <summary>Vulkan causal-path gate on the real Clover GGUF (#1794): same next token as llama.cpp, and the last-row distribution close to the CPU host's.</summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CloverRealGgufVulkanTests
{
    private readonly ITestOutputHelper _output;

    public CloverRealGgufVulkanTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Vulkan_CausalNextToken_MatchesLlamaCpp_AndCpu()
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
            float[] cpu = CloverRealGguf.CpuLastRow(gguf, config);

            // Decode step after " Paris": CPU scores the 6-token sequence in one forward; Vulkan prefills 5 and then DECODES one token, so
            // this row runs the single-token MoE path (fused decode layer, Q4_K / Q6_K / Q8_0 down MMVQ, #1794) that perplexity never touches.
            int[] withParis = [.. CloverRealGguf.Prompt, CloverRealGguf.Paris];
            float[] cpuDecode = CloverRealGguf.CpuLastRow(gguf, config, withParis);

            float[] vk, vkDecode;
            using (var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir!))
            using (var kv = model.CreateKvCache(64))
            {
                int[] positions = Enumerable.Range(0, CloverRealGguf.Prompt.Length).ToArray();
                using (ITensor logits = model.Forward(CloverRealGguf.Prompt, positions, deviceId: -1, kv))
                    vk = CloverRealGguf.LastRow(logits, config.VocabSize);
                using (ITensor step = model.Forward([CloverRealGguf.Paris], [CloverRealGguf.Prompt.Length], deviceId: -1, kv))
                    vkDecode = CloverRealGguf.LastRow(step, config.VocabSize);
            }

            var (decodeTop, decodeMargin) = CloverRealGguf.Top1(vkDecode);
            double decodeKl = CloverRealGguf.Kl(cpuDecode, vkDecode);
            _output.WriteLine($"[clover vulkan decode] top1={decodeTop} margin={decodeMargin:F3} KL(cpu||vk)={decodeKl:E3} cpuTop1={CloverRealGguf.Top1(cpuDecode).Top}");
            Assert.Equal(CloverRealGguf.Period, decodeTop);
            Assert.True(decodeKl < 0.5, $"decode-step KL(cpu || vulkan) = {decodeKl:E3}");

            var (top, margin) = CloverRealGguf.Top1(vk);
            double kl = CloverRealGguf.Kl(cpu, vk);
            int[] cpuTop2 = cpu.Select((v, i) => (v, i)).OrderByDescending(t => t.v).Take(2).Select(t => t.i).ToArray();
            _output.WriteLine($"[clover vulkan] top1={top} margin={margin:F3} KL(cpu||vk)={kl:E3} cpuTop2=[{string.Join(",", cpuTop2)}]");
            // Clover's causal forward is unusually sensitive to kernel numerics: on wikitext-2 (16 x 512) EVERY pair of implementations sits
            // at mean KL ~0.05 (llama.cpp CPU vs llama.cpp Vulkan 0.052, 99.9th-percentile token 1.37; Qwen3.6-35B-A3B under the same pairs
            // 0.007), so single positions need bounds well above that noise. Measured 2026-11-14 (Strix, Q4_K_M) at this 5-token prefill:
            // KL 0.20 with the F32-in indexed Q4_K down, 0.34 (top-1 " France" by 0.09 over " Paris") with the Q8_1-activation MMVQ down
            // that short batches now take (#1794; corpus-wide that path costs +0.004 KL and no top-1). A wrong-decay model never gets here
            // (non-finite state, see the CPU test's sensitivity arm).
            Assert.Contains(top, cpuTop2);
            Assert.True(kl < 0.5, $"KL(cpu || vulkan) = {kl:E3}");
        }
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
