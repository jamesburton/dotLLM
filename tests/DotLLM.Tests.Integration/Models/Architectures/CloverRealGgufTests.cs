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

    public static float[] CpuLastRow(GgufFile gguf, ModelConfig config)
    {
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        int[] positions = Enumerable.Range(0, Prompt.Length).ToArray();
        using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 64);
        using ITensor logits = model.Forward(Prompt, positions, deviceId: -1, kv);
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

        var (wrongTop, wrongMargin) = CloverRealGguf.Top1(CloverRealGguf.CpuLastRow(gguf, config with { Clover = null }));
        _output.WriteLine($"[clover cpu, no A_log transform] top1={wrongTop} margin={wrongMargin:F3}");
        Assert.NotEqual(CloverRealGguf.Paris, wrongTop);
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

            float[] vk;
            using (var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir!))
            {
                int[] positions = Enumerable.Range(0, CloverRealGguf.Prompt.Length).ToArray();
                using ITensor logits = model.Forward(CloverRealGguf.Prompt, positions, deviceId: -1);
                vk = CloverRealGguf.LastRow(logits, config.VocabSize);
            }

            var (top, margin) = CloverRealGguf.Top1(vk);
            double kl = CloverRealGguf.Kl(cpu, vk);
            _output.WriteLine($"[clover vulkan] top1={top} margin={margin:F3} KL(cpu||vk)={kl:E3}");
            Assert.Equal(CloverRealGguf.Paris, top);
            // Measured 2026-11-14 (Strix, Q4_K_M): see the PR. The bound is two orders above it and far below a wrong-decay model.
            Assert.True(kl < 0.05, $"KL(cpu || vulkan) = {kl:E3}");
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
