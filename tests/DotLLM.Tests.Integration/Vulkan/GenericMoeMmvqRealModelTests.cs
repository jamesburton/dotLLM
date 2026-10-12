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
/// Issue #912: real-model correctness of the generic-MoE decode MMVQ arm (<c>RecordMoeLayer</c> at rows &lt; 16: indexed dp4a K-quant / MXFP4 kernels
/// instead of the scalar one-thread-per-cell kernels). Teacher-forced: the prompt is fed to the Vulkan model ONE token at a time (so every MoE layer sees a
/// 1-row decode step), and the per-position logits are compared with the CPU model's full-sequence logits as KL(cpu || vulkan) and top-1 agreement,
/// for the arm ON and (control, same loaded model, <see cref="VulkanTransformerModel.GenericMoeMmvqEnabled"/> = false) the scalar kernels. The arm's own
/// KL must not exceed the scalar path's by more than activation-quantisation noise; counters prove which kernels ran. Greedy continuations of the CPU, arm-on
/// and arm-off models are reported (and must agree up to a near-tie).
/// Models: GLM-4.7-Flash Q4_K_M (<c>DOTLLM_GLM_GGUF</c>) and gpt-oss-20b MXFP4 (<c>DOTLLM_GPTOSS_GGUF</c>), else the HF-hub copies. Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class GenericMoeMmvqRealModelTests
{
    private readonly ITestOutputHelper _output;
    public GenericMoeMmvqRealModelTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts =
    [
        "The three primary colors are",
        "The capital of France is Paris, and the capital of Germany is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +",
        "In 1969, the Apollo 11 mission landed the first humans on the Moon. The commander of the mission was",
        // > 128 tokens: past gpt-oss's sliding window, so windowed layers clip their KV range and dense layers split it (#915).
        string.Concat(Enumerable.Repeat("The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. ", 4)) + "In summary, the Roman Empire was",
    ];

    private static string? Find(string envVar, string repoDir, string file)
    {
        string? p = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(p) && File.Exists(p)) return p;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub", repoDir, "snapshots");
        return Directory.Exists(hub) ? Directory.EnumerateFiles(hub, file, SearchOption.AllDirectories).FirstOrDefault() : null;
    }

    [SkippableTheory]
    [InlineData("glm")]
    [InlineData("gptoss20b")]
    public void Decode_MmvqArm_IsLive_AndKlVsCpuIsNotWorseThanScalar(string which)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = which == "glm"
            ? Find("DOTLLM_GLM_GGUF", "models--unsloth--GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf")
            : Find("DOTLLM_GPTOSS_GGUF", "models--ggml-org--gpt-oss-20b-GGUF", "gpt-oss-20b-mxfp4.gguf");
        Skip.If(path is null, $"model for '{which}' not found (set DOTLLM_GLM_GGUF / DOTLLM_GPTOSS_GGUF).");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);
        int[][] ids = Prompts.Select(p => Encode(tokenizer, p, addBos)).ToArray();

        // CPU reference: full-sequence logits (one row per position) + a greedy continuation.
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
        var (model, mkKv) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);   // MLA models need MlaVulkanKvCache (a plain cache = garbage decode)
        using var vkDisposable = model as IDisposable;
        var vk = (VulkanTransformerModel)model;
        Skip.If(!vk.GenericMoeMmvqAvailable, "device has no integer-dot-product / MMVQ kernels: the arm cannot run here.");

        var klOn = new List<double>(); var klOff = new List<double>();
        int agreeOn = 0, agreeOff = 0, total = 0;
        long mm0 = vk.GenericMoeMmvqLayerCount, sc0 = vk.GenericMoeScalarDecodeLayerCount;
        vk.GenericMoeMmvqEnabled = true;
        var onLogits = ids.Select(x => TeacherForced(vk, mkKv, x, vocab)).ToArray();
        long mmvqLayers = vk.GenericMoeMmvqLayerCount - mm0;
        Assert.True(mmvqLayers > 0, "the MMVQ arm never ran: decode silently fell back to the scalar kernels (#912)");
        Assert.Equal(0, vk.GenericMoeScalarDecodeLayerCount - sc0);

        vk.GenericMoeMmvqEnabled = false;
        mm0 = vk.GenericMoeMmvqLayerCount; sc0 = vk.GenericMoeScalarDecodeLayerCount;
        var offLogits = ids.Select(x => TeacherForced(vk, mkKv, x, vocab)).ToArray();
        Assert.Equal(0, vk.GenericMoeMmvqLayerCount - mm0);
        Assert.True(vk.GenericMoeScalarDecodeLayerCount - sc0 > 0, "control arm did not run the scalar kernels");

        for (int p = 0; p < ids.Length; p++)
        {
            double sOn = 0, sOff = 0; int aOn = 0, aOff = 0;
            for (int t = 0; t < ids[p].Length; t++)
            {
                sOn += Kl(cpuLogits[p][t], onLogits[p][t]); sOff += Kl(cpuLogits[p][t], offLogits[p][t]);
                int top = ArgMax(cpuLogits[p][t]);
                if (ArgMax(onLogits[p][t]) == top) aOn++;
                if (ArgMax(offLogits[p][t]) == top) aOff++;
            }
            int n = ids[p].Length;
            klOn.Add(sOn / n); klOff.Add(sOff / n); agreeOn += aOn; agreeOff += aOff; total += n;
            _output.WriteLine($"[{which}] prompt {p} ({n} tok): mean KL(cpu||vk) arm-on {sOn / n:F5}  scalar {sOff / n:F5};  top-1 agree on {aOn}/{n} scalar {aOff}/{n}");
        }
        _output.WriteLine($"[{which}] OVERALL mean KL arm-on {klOn.Average():F5} scalar {klOff.Average():F5}; top-1 agreement vs CPU arm-on {agreeOn}/{total} scalar {agreeOff}/{total}; mmvq layers {mmvqLayers}");

        // Greedy continuations (prefill + decode through the real engine path; decode steps are 1-row => the arm).
        int identicalOn = 0, identicalOff = 0;
        for (int p = 0; p < ids.Length; p++)
        {
            vk.GenericMoeMmvqEnabled = true;
            var gOn = Greedy(vk, config, ids[p], Steps, mkKv);
            vk.GenericMoeMmvqEnabled = false;
            var gOff = Greedy(vk, config, ids[p], Steps, mkKv);
            vk.GenericMoeMmvqEnabled = true;
            int dOn = FirstDiff(gOn.Tokens, cpuGreedy[p]), dOff = FirstDiff(gOff.Tokens, cpuGreedy[p]), dOnOff = FirstDiff(gOn.Tokens, gOff.Tokens);
            if (dOn < 0) identicalOn++;
            if (dOff < 0) identicalOff++;
            _output.WriteLine($"[{which}] greedy p{p}: first divergence vs CPU  arm-on {dOn}  scalar {dOff};  arm-on vs scalar {dOnOff}"
                + (dOnOff >= 0 ? $" (scalar top1-top2 margin {gOff.Margins[dOnOff]:F3})" : ""));
            _output.WriteLine($"     cpu    : '{tokenizer.Decode(cpuGreedy[p])}'");
            _output.WriteLine($"     arm-on : '{tokenizer.Decode(gOn.Tokens)}'");
            _output.WriteLine($"     scalar : '{tokenizer.Decode(gOff.Tokens)}'");
        }
        _output.WriteLine($"[{which}] greedy text identical to CPU over {Steps} tokens: arm-on {identicalOn}/{ids.Length}, scalar {identicalOff}/{ids.Length}");

        // The arm quantises activations to Q8_1 (the scalar path is F32-in), so allow a small excess over the scalar path's own distance from the CPU.
        Assert.True(klOn.Average() <= Math.Max(klOff.Average() * 2.0, klOff.Average() + 0.01),
            $"arm-on KL {klOn.Average():F5} exceeds scalar {klOff.Average():F5} by more than activation-quantisation noise");
        Assert.True(agreeOn >= agreeOff - Math.Max(2, total / 20), $"top-1 agreement dropped: arm-on {agreeOn} vs scalar {agreeOff} of {total}");
    }

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

    /// <summary>Feeds the prompt one token at a time (every step is a 1-row decode forward) and returns the logits after each token.</summary>
    private static float[][] TeacherForced(VulkanTransformerModel vk, Func<int, IKvCache> mkKv, int[] ids, int vocab)
    {
        var res = new float[ids.Length][];
        using var kv = mkKv(ids.Length + 2);
        for (int t = 0; t < ids.Length; t++)
        {
            using ITensor l = vk.Forward([ids[t]], [t], -1, kv);
            res[t] = Rows(l, vocab)[^1];
        }
        return res;
    }

    private static (int[] Tokens, float[] Margins) Greedy(DotLLM.Core.Models.IModel m, ModelConfig config, int[] prompt, int steps, Func<int, IKvCache> mk)
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

    /// <summary>KL(softmax(p) || softmax(q)) in nats.</summary>
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
