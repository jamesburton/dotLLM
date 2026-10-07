using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #787 / #789: OLMoE-1B-7B Q4_K_M on Vulkan. A 512-token prefill used to end in VK_ERROR_DEVICE_LOST (the scalar indexed Q4_K expert matmul ran long enough
/// to trip the driver watchdog); the expert-grouped coopmat prefill must complete it AND agree with the CPU path. The scalar path is compared only at a context the
/// watchdog tolerates (the A/B must not itself kill the device). Gated on <c>DOTLLM_OLMOE_GGUF</c> / the HF-hub copy; take the GPU lock before running.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class OlmoeVulkanGroupedPrefillTests
{
    private readonly ITestOutputHelper _output;
    public OlmoeVulkanGroupedPrefillTests(ITestOutputHelper output) => _output = output;

    private static string? ModelPath()
    {
        string? p = Environment.GetEnvironmentVariable("DOTLLM_OLMOE_GGUF");
        if (!string.IsNullOrWhiteSpace(p) && File.Exists(p)) return p;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--bartowski--OLMoE-1B-7B-0924-Instruct-GGUF", "snapshots");
        if (!Directory.Exists(hub)) return null;
        return Directory.EnumerateFiles(hub, "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", SearchOption.AllDirectories).FirstOrDefault();
    }

    private const string Prose =
        "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the "
        + "banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked "
        + "provinces from Britain to Syria. Trade flourished along the Mediterranean, and aqueducts carried water to cities. ";

    private static int[] PromptOfLength(DotLLM.Tokenizers.ITokenizer tok, int n)
    {
        var ids = new List<int>();
        if (tok.BosTokenId >= 0) ids.Add(tok.BosTokenId);
        string text = Prose;
        while (ids.Count < n) { ids.AddRange(tok.Encode(text)); text += Prose; if (text.Length > 64 * Prose.Length) break; }
        return ids.Take(n).ToArray();
    }

    private static unsafe float[] LastRow(DotLLM.Core.Models.IModel m, int vocab, int[] ids, Func<int, DotLLM.Core.Attention.IKvCache> mk)
    {
        using var kv = mk(ids.Length + 2);
        using ITensor logits = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static double Rms(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s / a.Length);
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static unsafe (int[] Tokens, float[] Margins) GreedyWithMargins(VulkanTransformerModel vk, DotLLM.Core.Models.ModelConfig config, int[] promptIds)
    {
        const int steps = 16;
        int vocab = config.VocabSize;
        var toks = new List<int>(); var margins = new List<float>();
        using var kv = vk.CreateKvCache(promptIds.Length + steps + 2);
        for (int step = 0; step < steps; step++)
        {
            int[] ids = step == 0 ? promptIds : [toks[^1]];
            int[] pos = step == 0 ? Enumerable.Range(0, promptIds.Length).ToArray() : [promptIds.Length + step - 1];
            using ITensor logits = vk.Forward(ids, pos, -1, kv);
            var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;
            float second = float.NegativeInfinity;
            for (int i = 0; i < vocab; i++) if (i != best && row[i] > second) second = row[i];
            toks.Add(best); margins.Add(row[best] - second);
        }
        return (toks.ToArray(), margins.ToArray());
    }

    [SkippableFact]
    public void Vulkan_Olmoe_GroupedPrefill_Completes512_AndMatchesScalarAndCpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_OLMOE_GGUF to OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf.");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] shortIds = PromptOfLength(tokenizer, 320);   // scalar path survives this (ctx 384 worked before #789)
        int[] longIds = PromptOfLength(tokenizer, 520);    // the #787 repro size
        Assert.Equal(320, shortIds.Length);
        Assert.Equal(520, longIds.Length);

        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
        Skip.IfNot(vk.GroupedMoeKernelsBuilt, "device lacks wave64 coopmat (or DOTLLM_VK_MOE_GROUPED=0): grouped path not built.");

        // CPU reference (in-process, no cache: OLMoE is 7B total / 1B active).
        float[] cpuShort, cpuLong;
        using (var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config))
        {
            Func<int, DotLLM.Core.Attention.IKvCache> mkCpu = size => new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), size);
            cpuShort = LastRow(cpu, config.VocabSize, shortIds, mkCpu);
            cpuLong = LastRow(cpu, config.VocabSize, longIds, mkCpu);
        }

        // 320 tokens: grouped vs scalar, in-process A/B.
        vk.GroupedMoeEnabled = true;
        int before = vk.GroupedMoeDispatchCount;
        float[] gShort = LastRow(vk, config.VocabSize, shortIds, size => vk.CreateKvCache(size));
        int layers = vk.GroupedMoeDispatchCount - before;
        Assert.Equal(config.NumLayers, layers);   // every OLMoE layer is MoE and took the grouped path (Q4_K / Q6_K / Q5_K banks all have kernels)
        vk.GroupedMoeEnabled = false;
        before = vk.GroupedMoeDispatchCount;
        float[] sShort = LastRow(vk, config.VocabSize, shortIds, size => vk.CreateKvCache(size));
        Assert.Equal(before, vk.GroupedMoeDispatchCount);
        double rmsGs = Rms(gShort, sShort), rmsScale = Rms(sShort, new float[sShort.Length]);
        _output.WriteLine($"p320: rms(grouped-scalar)={rmsGs:F4} (logit rms {rmsScale:F3}); rms(grouped-cpu)={Rms(gShort, cpuShort):F4} rms(scalar-cpu)={Rms(sShort, cpuShort):F4}; "
            + $"argmax grouped={ArgMax(gShort)} scalar={ArgMax(sShort)} cpu={ArgMax(cpuShort)}");
        Assert.Equal(ArgMax(sShort), ArgMax(gShort));
        Assert.True(rmsGs < 0.05 * rmsScale + 0.05, $"grouped vs scalar logit rms {rmsGs}");

        // Greedy continuation from the 320-token prompt, grouped prefill vs scalar prefill (decode kernels identical). A divergence is accepted only at a
        // near-tie of the scalar path (top1-top2 margin < 0.5 logits), since F16 operands move the logits by ~1% of their rms.
        vk.GroupedMoeEnabled = true;
        var (gTok, _) = GreedyWithMargins(vk, config, shortIds);
        vk.GroupedMoeEnabled = false;
        var (sTok, sMargin) = GreedyWithMargins(vk, config, shortIds);
        int firstDiv = -1;
        for (int i = 0; i < gTok.Length; i++) if (gTok[i] != sTok[i]) { firstDiv = i; break; }
        _output.WriteLine($"greedy 16 tokens from p320: first divergence = {firstDiv}" + (firstDiv >= 0 ? $" (scalar margin {sMargin[firstDiv]:F3})" : " (identical)"));
        Assert.True(firstDiv < 0 || sMargin[firstDiv] < 0.5f, $"greedy diverged at step {firstDiv} with scalar margin {(firstDiv >= 0 ? sMargin[firstDiv] : 0f)}");

        // 520 tokens (the #787 repro): grouped only - the scalar path would lose the device here.
        vk.GroupedMoeEnabled = true;
        before = vk.GroupedMoeDispatchCount;
        var sw = Stopwatch.StartNew();
        float[] gLong = LastRow(vk, config.VocabSize, longIds, size => vk.CreateKvCache(size));
        double sec = sw.Elapsed.TotalSeconds;
        Assert.Equal(config.NumLayers, vk.GroupedMoeDispatchCount - before);
        _output.WriteLine($"p520 completed in {sec:F2}s ({520 / sec:F0} tok/s incl. first-call overhead); rms(grouped-cpu)={Rms(gLong, cpuLong):F4}; argmax vk={ArgMax(gLong)} cpu={ArgMax(cpuLong)}");
        Assert.Equal(ArgMax(cpuLong), ArgMax(gLong));
        Assert.True(Rms(gLong, cpuLong) < 0.1 * Rms(cpuLong, new float[cpuLong.Length]) + 0.1, "p520 logits diverge from CPU beyond quantised-GEMM noise");
    }
}
