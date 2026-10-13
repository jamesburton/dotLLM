using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #987: the Qwen3.6 hybrid shared-expert gate/up/down (Q8_0 in the real file) must stay packed Q8_0 on the device instead of
/// F16 copies. The SAME file is loaded twice (packed default / <c>DOTLLM_VK_MOE_SHARED_PACKED=0</c> control, read at load time);
/// the CPU model is the oracle. Counters prove the fast path exists only on the default model; KL is gated per the #949 pattern
/// (median, 99%-trimmed, outlier count, top-1), never mean KL. Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Qwen36SharedExpertPackedRealModelTests
{
    private readonly ITestOutputHelper _output;
    public Qwen36SharedExpertPackedRealModelTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts =
    [
        "The three primary colors are",
        "The capital of France is Paris, and the capital of Germany is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +",
        "In 1969, the Apollo 11 mission landed the first humans on the Moon. The commander of the mission was",
        string.Concat(Enumerable.Repeat("The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. ", 2)) + "In summary, the Roman Empire was",
    ];

    private const int Tail = 6;      // prefix lengths (len-Tail+1 .. len) per prompt are scored
    private const int Steps = 12;

    private static string? Find(string envVar, string repoDir, string file)
    {
        string? p = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(p) && File.Exists(p)) return p;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub", repoDir, "snapshots");
        return Directory.Exists(hub) ? Directory.EnumerateFiles(hub, file, SearchOption.AllDirectories).FirstOrDefault() : null;
    }

    [SkippableFact]
    public void SharedExperts_StayPackedQ8_AndKlVsCpuIsNotWorseThanF16Control()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = Find("DOTLLM_QWEN36_A3B_Q4_K_M_GGUF", "models--unsloth--Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");
        Skip.If(path is null, "Qwen3.6-35B-A3B UD-Q4_K_M not found (set DOTLLM_QWEN36_A3B_Q4_K_M_GGUF).");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);
        int[][] ids = Prompts.Select(p => Encode(tokenizer, p, addBos)).ToArray();

        // CPU oracle: uncached growing-prefix forwards (the CPU hybrid model has no incremental cache usable here).
        var cpuRows = new Dictionary<(int P, int L), float[]>();
        var cpuGreedy = new int[ids.Length][];
        var cpuWatch = System.Diagnostics.Stopwatch.StartNew();
        using (var cpu = ModelLoader.CreateCpuModelFromGguf(gguf, config))
        {
            for (int p = 0; p < ids.Length; p++)
            {
                for (int L = Math.Max(1, ids[p].Length - Tail + 1); L <= ids[p].Length; L++)
                    cpuRows[(p, L)] = LastRow(cpu, ids[p].AsSpan(0, L).ToArray(), vocab);
                var seq = new List<int>(ids[p]);
                for (int s = 0; s < Steps; s++) seq.Add(ArgMax(LastRow(cpu, seq.ToArray(), vocab)));
                cpuGreedy[p] = seq.Skip(ids[p].Length).ToArray();
            }
        }

        _output.WriteLine($"CPU oracle: {cpuWatch.Elapsed.TotalSeconds:F0} s");
        using var device = VulkanDevice.Create();

        string? prev = Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_SHARED_PACKED");
        VulkanQwen3MoeHybridTransformerModel control; Func<int, IKvCache> mkControl;
        Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_SHARED_PACKED", "0");
        try
        {
            var (m, mk) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
            control = (VulkanQwen3MoeHybridTransformerModel)m; mkControl = mk;
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_SHARED_PACKED", prev); }
        using var controlDisp = control as IDisposable;
        var (pm, mkPacked) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
        using var packedDisp = pm as IDisposable;
        var packed = (VulkanQwen3MoeHybridTransformerModel)pm;

        // Resident MoE bundles (and so the shared-expert upload) are built lazily on the first forward: warm both models up first.
        foreach (var (wm, wmk, env) in new[] { ((VulkanQwen3MoeHybridTransformerModel)packed, mkPacked, prev), (control, mkControl, "0") })
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_SHARED_PACKED", env);   // the flag is read when the layer bundle is built
            try
            {
                wm.ResetSequenceState();
                using var wkv = wmk(ids[0].Length + 2);
                using (wm.Forward(ids[0], Enumerable.Range(0, ids[0].Length).ToArray(), -1, wkv)) { }
                // every layer that a later prompt may touch is built by one pass over a long prompt; prompts 0..4 cover all 40 layers (all layers are MoE)
            }
            finally { Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_SHARED_PACKED", prev); }
        }
        _output.WriteLine($"packed shared layers: default {packed.SharedExpertPackedLayerCount}, control {control.SharedExpertPackedLayerCount}; " +
                          $"shared buffer bytes: control {control.SharedExpertBufferBytes / 1048576.0:F0} MiB, packed {packed.SharedExpertBufferBytes / 1048576.0:F0} MiB");
        Assert.True(packed.SharedExpertPackedLayerCount > 0, "packed path did not run");
        Assert.Equal(0, control.SharedExpertPackedLayerCount);
        Assert.True(packed.SharedExpertBufferBytes < control.SharedExpertBufferBytes);

        foreach (var (name, decodeLast) in new[] { ("prefill(all)", false), ("decode(1)", true) })
        {
            var (klP, agP, n, rowsP) = Evaluate(packed, mkPacked, ids, cpuRows, vocab, decodeLast);
            var (klC, agC, _, rowsC) = Evaluate(control, mkControl, ids, cpuRows, vocab, decodeLast);
            double medP = Median(rowsP), medC = Median(rowsC), trP = Trimmed(rowsP), trC = Trimmed(rowsC);
            int outP = rowsP.Count(v => v > 1.0), outC = rowsC.Count(v => v > 1.0);
            _output.WriteLine($"{name,-13} rows={n,3} mean KL packed {klP:F5} control {klC:F5} | median {medP:F5} / {medC:F5} | trimmed99 {trP:F5} / {trC:F5} | outliers(KL>1) {outP} / {outC} | top-1 packed {agP}/{n} control {agC}/{n}");
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

    private (double Kl, int Agree, int Rows, double[] PerRow) Evaluate(VulkanQwen3MoeHybridTransformerModel vk, Func<int, IKvCache> mk, int[][] ids,
        Dictionary<(int P, int L), float[]> cpu, int vocab, bool decodeLast)
    {
        double klSum = 0; int agree = 0, rows = 0; var per = new List<double>();
        for (int p = 0; p < ids.Length; p++)
        for (int L = Math.Max(1, ids[p].Length - Tail + 1); L <= ids[p].Length; L++)
        {
            vk.ResetSequenceState();
            using var kv = mk(L + 2);
            float[] row;
            if (decodeLast && L > 1)
            {
                using (vk.Forward(ids[p].AsSpan(0, L - 1), Enumerable.Range(0, L - 1).ToArray(), -1, kv)) { }
                using ITensor l = vk.Forward(ids[p].AsSpan(L - 1, 1), [L - 1], -1, kv);
                row = Rows(l, vocab)[^1];
            }
            else
            {
                using ITensor l = vk.Forward(ids[p].AsSpan(0, L), Enumerable.Range(0, L).ToArray(), -1, kv);
                row = Rows(l, vocab)[^1];
            }
            float[] want = cpu[(p, L)];
            double kl1 = Kl(want, row);
            klSum += kl1; per.Add(kl1);
            if (ArgMax(row) == ArgMax(want)) agree++;
            rows++;
        }
        return (klSum / rows, agree, rows, per.ToArray());
    }

    private static float[] LastRow(IModel m, int[] tokens, int vocab)
    {
        m.ResetSequenceState();
        using ITensor l = m.Forward(tokens, Enumerable.Range(0, tokens.Length).ToArray(), -1);
        return Rows(l, vocab)[^1];
    }

    private static (int[] Tokens, float[] Margins) Greedy(IModel m, ModelConfig config, int[] prompt, int steps, Func<int, IKvCache> mk)
    {
        int vocab = config.VocabSize;
        m.ResetSequenceState();
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
