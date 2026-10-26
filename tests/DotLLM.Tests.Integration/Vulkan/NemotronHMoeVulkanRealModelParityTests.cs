using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #402 acceptance on real bytes: NVIDIA Nemotron 3.5 Lightning 30B-A3B (ollama / ggml-org Q4_K_M, <c>nemotron_h_moe</c>) on Vulkan
/// against the CPU forward, which #926 pinned to llama.cpp (KL 4e-5 .. 0.011 on identical ids). Expert banks are Q5_0 / Q8_0 here,
/// so this also covers the Q5_0 -> Q5_1 repack and the indexed MMVQ kernels on real quantised weights. Gated on
/// <c>DOTLLM_NEMOTRON_MOE_GGUF</c> or <c>~/.dotllm/models/nvidia/nemotron-3.5-lightning/</c>; hold the GPU lock.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class NemotronHMoeVulkanRealModelParityTests(ITestOutputHelper output)
{
    private const string Prose =
        "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, " +
        "oil based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred " +
        "cities, and the price of a book fell dramatically. Scholars could now compare identical copies of a text, which changed " +
        "how errors were found and how knowledge accumulated. The next paragraph explains";

    [SkippableFact]
    public void Vulkan_MatchesCpu_OnRealWeights_PrefillAndDecode()
    {
        Skip.If(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN"), "1", StringComparison.Ordinal), "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        var fixture = KnownTestFixtures.NemotronLightningQ4KM;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.NemotronLightningDescription));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.EncodeRaw(Prose);
        Assert.InRange(ids.Length, 48, 200);
        int[] positions = Enumerable.Range(0, ids.Length).ToArray();
        int vocab = config.VocabSize;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        // CPU oracle rows (all positions).
        float[] cpu;
        using (var cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config))
            cpu = AllRows(cpuModel, ids, positions, -1);

        VulkanNemotronHTransformerModel.MoePathCounters.Reset();
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
        using var _ = (IDisposable)model;
        float[] vk = AllRows(model, ids, positions, 0);

        // The intended kernels ran (a silent fallback or an F32-widened bank would not count here).
        long mmvq = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.Mmvq);
        Assert.True(mmvq > 0, "indexed MMVQ expert kernels never ran");

        int n = ids.Length;
        var kl = new double[n];
        int agree = 0;
        for (int p = 0; p < n; p++)
        {
            kl[p] = Kl(cpu, p, vk, p, vocab);
            if (ArgMax(cpu, p, vocab) == ArgMax(vk, p, vocab)) agree++;
        }
        double meanKl = kl.Average();
        // Control: the same metric against the NEXT position must be far larger, or a high-agreement result proves nothing.
        double shifted = Enumerable.Range(0, n - 1).Average(p => Kl(cpu, p, vk, p + 1, vocab));
        output.WriteLine($"prefill {n} tokens: mean KL(cpu||vulkan)={meanKl:E3} max={kl.Max():E3} top1 {agree}/{n}; shifted-position control KL={shifted:F3}");
        Assert.True(meanKl < 0.02, $"mean KL {meanKl:E3} (llama.cpp-vs-CPU floor on this model is 4e-5..0.011)");
        Assert.True(agree >= n * 0.9, $"top-1 agreement {agree}/{n}");
        Assert.True(shifted > 20 * Math.Max(meanKl, 1e-4), $"control KL {shifted:F3} does not separate from {meanKl:E3}");

        // Decode: prefill the first part, then single-token steps through the engine KV cache must reproduce the prefill rows.
        int head = n - 8;
        model.ResetSequenceState();
        using IKvCache kv = kvFactory(n + 8);
        using (model.Forward(ids.AsSpan(0, head), positions.AsSpan(0, head), 0, kv)) { }
        int flips = 0;
        double decodeKl = 0;
        for (int p = head; p < n; p++)
        {
            using var row = model.Forward(ids.AsSpan(p, 1), positions.AsSpan(p, 1), 0, kv);
            float[] one = Copy(row, vocab);
            decodeKl += Kl(vk, p, one, 0, vocab);
            if (ArgMax(vk, p, vocab) != ArgMax(one, 0, vocab)) flips++;
        }
        decodeKl /= 8;
        output.WriteLine($"decode 8 steps vs Vulkan prefill rows: mean KL={decodeKl:E3} top-1 flips={flips}");
        Assert.True(decodeKl < 0.02, $"decode-vs-prefill mean KL {decodeKl:E3}");
        Assert.True(flips <= 1, $"{flips} top-1 flips between decode and prefill");
    }

    private static float[] AllRows(IModel m, int[] ids, int[] positions, int deviceId)
    {
        m.TrySetAllRowLogitsLimit(ids.Length + 1);
        using var logits = m.Forward(ids, positions, deviceId);
        long rows = logits.ElementCount / m.Config.VocabSize;
        Assert.Equal(ids.Length, rows);
        return Copy(logits, (int)logits.ElementCount);
    }

    private static unsafe float[] Copy(DotLLM.Core.Tensors.ITensor t, int count)
    {
        var a = new float[count];
        new ReadOnlySpan<float>((void*)t.DataPointer, count).CopyTo(a);
        return a;
    }

    private static int ArgMax(float[] rows, int row, int vocab)
    {
        int best = 0;
        int o = row * vocab;
        for (int i = 1; i < vocab; i++) if (rows[o + i] > rows[o + best]) best = i;
        return best;
    }

    /// <summary>KL(p || q) of softmax(rows a [ra]) against softmax(rows b [rb]), double precision.</summary>
    private static double Kl(float[] a, int ra, float[] b, int rb, int vocab)
    {
        double ma = double.NegativeInfinity, mb = double.NegativeInfinity;
        for (int i = 0; i < vocab; i++) { ma = Math.Max(ma, a[ra * vocab + i]); mb = Math.Max(mb, b[rb * vocab + i]); }
        double za = 0, zb = 0;
        for (int i = 0; i < vocab; i++) { za += Math.Exp(a[ra * vocab + i] - ma); zb += Math.Exp(b[rb * vocab + i] - mb); }
        double la = Math.Log(za), lb = Math.Log(zb), kl = 0;
        for (int i = 0; i < vocab; i++)
        {
            double lpa = a[ra * vocab + i] - ma - la;
            double lpb = b[rb * vocab + i] - mb - lb;
            kl += Math.Exp(lpa) * (lpa - lpb);
        }
        return kl;
    }
}
