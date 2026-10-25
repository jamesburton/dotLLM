using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #949: the DeepSeek/GLM shared-expert gate/up/down must stay packed (Q5_K / Q6_K on GLM-4.7-Flash Q4_K_M) on the
/// device instead of widening to F32. The SAME file is loaded twice, packed (default) and with the
/// <c>DOTLLM_VK_SHARED_EXPERT_PACKED=0</c> control (read at load time); the CPU model is the oracle. Counters prove the packed
/// layers exist only on the default model (a regression that silently widens them fails the count assertion), the weight
/// footprint must shrink by more than 1.2 GB, and KL vs CPU / greedy text must not get worse than the F32 control.
/// Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class MlaSharedExpertPackedRealModelTests
{
    private readonly ITestOutputHelper _output;
    public MlaSharedExpertPackedRealModelTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts =
    [
        "The three primary colors are",
        "The capital of France is Paris, and the capital of Germany is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +",
        "In 1969, the Apollo 11 mission landed the first humans on the Moon. The commander of the mission was",
        string.Concat(Enumerable.Repeat("The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. ", 4)) + "In summary, the Roman Empire was",
    ];

    private static string? Find(string envVar, string org, string repo, string file)
        => DotLLM.Tests.Integration.Fixtures.TestFixtureResolver.ResolveFile(envVar, org, repo, file).Path;

    [SkippableFact]
    public void SharedExperts_StayPacked_AndKlVsCpuIsNotWorseThanF32Control()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = Find("DOTLLM_GLM_GGUF", "unsloth", "GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf");
        Skip.If(path is null, "GLM-4.7-Flash Q4_K_M not found (set DOTLLM_GLM_GGUF).");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);
        int[][] ids = Prompts.Select(p => Encode(tokenizer, p, addBos)).ToArray();

        const int Steps = 16;
        var cpuLogits = new float[ids.Length][][];
        var cpuGreedy = new int[ids.Length][];
        using (var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config))
        {
            for (int p = 0; p < ids.Length; p++)
            {
                using (ITensor l = cpu.Forward(ids[p], Enumerable.Range(0, ids[p].Length).ToArray(), -1,
                    new SimpleKvCache(KvGeometry.FromConfig(config), ids[p].Length + 2)))
                    cpuLogits[p] = Rows(l, vocab);
                cpuGreedy[p] = Greedy(cpu, config, ids[p], Steps, size => new SimpleKvCache(KvGeometry.FromConfig(config), size)).Tokens;
            }
        }

        using var device = VulkanDevice.Create();

        VulkanTransformerModel control; Func<int, IKvCache> mkControl;
        string? prev = Environment.GetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED");
        Environment.SetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED", "0");
        try
        {
            var (m, mk) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
            control = (VulkanTransformerModel)m; mkControl = mk;
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED", prev); }
        using var controlDisp = control as IDisposable;

        var (pm, mkPacked) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
        using var packedDisp = pm as IDisposable;
        var packed = (VulkanTransformerModel)pm;

        int moeLayers = Enumerable.Range(0, config.NumLayers).Count(i => config.Moe!.IsMoeLayer(i));
        _output.WriteLine($"MoE layers {moeLayers}; packed shared layers: default {packed.SharedExpertPackedLayerCount}, control {control.SharedExpertPackedLayerCount}");
        Assert.Equal(moeLayers, packed.SharedExpertPackedLayerCount);       // fails if the packed path is skipped
        Assert.Equal(0, control.SharedExpertPackedLayerCount);              // the control is a real control
        double savedGb = (control.WeightBytes - packed.WeightBytes) / 1073741824.0;
        _output.WriteLine($"weight bytes: control {control.WeightBytes / 1073741824.0:F2} GiB, packed {packed.WeightBytes / 1073741824.0:F2} GiB, saved {savedGb:F2} GiB");
        Assert.True(savedGb > 1.2, $"expected > 1.2 GiB saved, got {savedGb:F2}");

        // Gate (#949): the MEAN KL is NOT a usable gate on this model. A few teacher-forced rows are ill-conditioned (massive-activation
        // tokens, CPU max-logit ~110 on every row) and flip on rounding-level differences in ANY arm: on 4544 rows (40 prompts, see
        // MlaSharedExpertOutlierStudyTests) control / packed / packed+legacy-GEMV have outlier (KL>1) rates 21 / 23 / 22 per 4544, medians
        // 0.00371 / 0.00369 / 0.00370, 99%-trimmed means 0.00679 / 0.00651 / 0.00657, and the three GPU arms agree with EACH OTHER on most
        // outlier rows while all disagree with the CPU oracle (flips go both ways). So the gate is: median and 99%-trimmed KL no worse than
        // the F32 control (+ small slack), outlier-row count no more than control + 2, top-1 agreement, and greedy identity below.
        foreach (var (name, chunk) in new[] { ("decode(1)", 1), ("prefill(all)", 0) })
        {
            var (klP, agP, n, rowsP) = Evaluate(packed, mkPacked, ids, cpuLogits, vocab, chunk);
            var (klC, agC, _, rowsC) = Evaluate(control, mkControl, ids, cpuLogits, vocab, chunk);
            double medP = Median(rowsP), medC = Median(rowsC), trP = Trimmed(rowsP), trC = Trimmed(rowsC);
            int outP = rowsP.Count(v => v > 1.0), outC = rowsC.Count(v => v > 1.0);
            _output.WriteLine($"{name,-13} rows={n,4} mean KL(cpu||vk) packed {klP:F5} control {klC:F5} | median {medP:F5} / {medC:F5} | trimmed99 {trP:F5} / {trC:F5} | outliers(KL>1) {outP} / {outC} | top-1 packed {agP}/{n} control {agC}/{n}");
            Assert.True(medP <= Math.Max(medC * 1.5, medC + 0.002), $"{name}: packed median KL {medP:F5} vs control {medC:F5}");
            Assert.True(trP <= Math.Max(trC * 1.5, trC + 0.003), $"{name}: packed 99%-trimmed KL {trP:F5} vs control {trC:F5}");
            Assert.True(outP <= outC + 2, $"{name}: packed outlier rows {outP} vs control {outC}");
            Assert.True(agP >= agC - Math.Max(2, n / 20), $"{name}: top-1 {agP} vs {agC} of {n}");
        }

        int identP = 0, identC = 0;
        for (int p = 0; p < ids.Length; p++)
        {
            var gP = Greedy(packed, config, ids[p], Steps, mkPacked);
            var gC = Greedy(control, config, ids[p], Steps, mkControl);
            int dP = FirstDiff(gP.Tokens, cpuGreedy[p]), dC = FirstDiff(gC.Tokens, cpuGreedy[p]);
            if (dP < 0) identP++;
            if (dC < 0) identC++;
            _output.WriteLine($"greedy p{p}: first divergence vs CPU packed {dP} control {dC}: '{tokenizer.Decode(gP.Tokens)}'");
            if (dP >= 0) Assert.True(gP.Margins[dP] < 1.0f, $"prompt {p}: packed diverged from CPU at step {dP} with a clear margin {gP.Margins[dP]:F3}");
        }
        _output.WriteLine($"greedy identical to CPU over {Steps} tokens: packed {identP}/{ids.Length}, control {identC}/{ids.Length}");
        Assert.True(identP >= identC - 1);
    }

    private (double Kl, int Agree, int Rows, double[] PerRow) Evaluate(VulkanTransformerModel vk, Func<int, IKvCache> mk, int[][] ids, float[][][] cpu, int vocab, int chunkSize)
    {
        double klSum = 0; int agree = 0, rows = 0; var per = new List<double>();
        for (int p = 0; p < ids.Length; p++)
        {
            int len = ids[p].Length;
            int chunk = chunkSize == 0 ? len : chunkSize;
            using var kv = mk(len + 2);
            for (int s = 0; s < len; s += chunk)
            {
                int n = Math.Min(chunk, len - s);
                using ITensor l = vk.Forward(ids[p].AsSpan(s, n), Enumerable.Range(s, n).ToArray(), -1, kv);
                float[] row = Rows(l, vocab)[^1];
                float[] want = cpu[p][s + n - 1];
                double kl1 = Kl(want, row);
                klSum += kl1; per.Add(kl1);
                if (ArgMax(row) == ArgMax(want)) agree++;
                rows++;
            }
        }
        return (klSum / rows, agree, rows, per.ToArray());
    }

    private static double Median(double[] v) { var s = v.OrderBy(x => x).ToArray(); return s[s.Length / 2]; }

    private static double Trimmed(double[] v) { var s = v.OrderBy(x => x).ToArray(); return s.Take(Math.Max(1, (int)(s.Length * 0.99))).Average(); }

    private static int[] Encode(DotLLM.Tokenizers.ITokenizer tok, string text, bool addBos)
    {
        int[] enc = tok.Encode(text);
        if (!addBos || tok.BosTokenId < 0) return enc;
        var ids = new int[enc.Length + 1];
        ids[0] = tok.BosTokenId;
        Array.Copy(enc, 0, ids, 1, enc.Length);
        return ids;
    }

    private static unsafe float[][] Rows(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var res = new float[rows][];
        for (int r = 0; r < rows; r++)
            res[r] = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)r * vocab, vocab).ToArray();
        return res;
    }

    private static (int[] Tokens, float[] Margins) Greedy(IModel m, ModelConfig config, int[] prompt, int steps, Func<int, IKvCache> mk)
    {
        int vocab = config.VocabSize;
        var toks = new List<int>(); var margins = new List<float>();
        using var kv = mk(prompt.Length + steps + 2);
        for (int s = 0; s < steps; s++)
        {
            int[] ids = s == 0 ? prompt : [toks[^1]];
            int[] pos = s == 0 ? Enumerable.Range(0, prompt.Length).ToArray() : [prompt.Length + s - 1];
            using ITensor l = m.Forward(ids, pos, -1, kv);
            float[] row = Rows(l, vocab)[^1];
            int best = ArgMax(row);
            float second = float.NegativeInfinity;
            for (int i = 0; i < row.Length; i++) if (i != best && row[i] > second) second = row[i];
            toks.Add(best); margins.Add(row[best] - second);
        }
        return (toks.ToArray(), margins.ToArray());
    }

    private static int FirstDiff(int[] a, int[] b)
    {
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (a[i] != b[i]) return i;
        return -1;
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lzp, lq = q[i] - lzq;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }
}
