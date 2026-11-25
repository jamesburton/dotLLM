using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0-145B on Vulkan against the CPU path (G3, #1981). Two opt-in stages that share a reference file, run in SEPARATE processes so the CPU and
/// the GPU never hold the 105 GB mapping's working set together and the CPU stage never overlaps a GPU run:
/// <list type="number">
/// <item><c>DOTLLM_EDGE0_VK_E2E=cpu</c> + <c>DOTLLM_EDGE0_REF</c>=path: runs the CPU model on the NLL passage and three short prompts, records per-row argmax,
/// the passage mean NLL, the last-row logits of each prompt and a 24-token greedy continuation.</item>
/// <item><c>DOTLLM_EDGE0_VK_E2E=vulkan</c> + the same path: runs the Vulkan model on the same inputs and compares: greedy text, NLL, KL(cpu || vulkan).</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
public sealed class VulkanEdge0EndToEndTests(ITestOutputHelper output)
{
    private const string Passage =
        "The mitochondrion is an organelle found in most eukaryotic cells. Its main job is to convert nutrients into adenosine triphosphate, " +
        "the molecule that cells use as a source of chemical energy. In software, a similar story is told about caches: a small, fast memory " +
        "sits in front of a large, slow one, and a good replacement policy keeps the hot data close. The city of Lyon lies at the confluence " +
        "of the Rhone and the Saone rivers in the east of France.";

    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "def reverse_string(s):\n    return",
        "Water boils at a temperature of",
    ];

    private const int GreedyTokens = 24;

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf",
        "native_experts.txt", "experts.bin", "ple-fp8.gguf");

    private sealed record Run(int[] Ids, int[] RowArgmax, double MeanNll, float[] LastLogits, int[] Greedy);

    private static unsafe int Argmax(ReadOnlySpan<float> row) { int b = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[b]) b = i; return b; }

    private static unsafe Run Execute(IModel model, ReadOnlySpan<int> ids, bool nll, Action<int>? allRows)
    {
        model.ResetSequenceState();
        allRows?.Invoke(ids.Length);
        var pos = new int[ids.Length];
        for (int i = 0; i < pos.Length; i++) pos[i] = i;
        using var logits = model.Forward(ids, pos, 0);
        int vocab = (int)logits.Shape[1];
        int rows = (int)logits.Shape[0];
        if (rows != ids.Length) throw new InvalidOperationException($"forward returned {rows} logit rows for {ids.Length} tokens (all-row logits unavailable)");
        var l = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab);
        var argmax = new int[rows];
        double total = 0;
        for (int t = 0; t < rows; t++)
        {
            var row = l.Slice(t * vocab, vocab);
            argmax[t] = Argmax(row);
            if (nll && t + 1 < rows)
            {
                float max = float.NegativeInfinity; foreach (float v in row) max = Math.Max(max, v);
                double sum = 0; foreach (float v in row) sum += Math.Exp(v - max);
                total -= row[ids[t + 1]] - max - Math.Log(sum);
            }
        }
        float[] last = l.Slice((rows - 1) * vocab, vocab).ToArray();
        var greedy = new int[GreedyTokens];
        int next = argmax[^1];
        for (int s = 0; s < GreedyTokens; s++)
        {
            greedy[s] = next;
            using var lg = model.Forward([next], [ids.Length + s], 0);
            next = Argmax(new ReadOnlySpan<float>((void*)lg.DataPointer, vocab));
        }
        return new Run(ids.ToArray(), argmax, nll ? total / (rows - 1) : double.NaN, last, greedy);
    }

    private static void Save(string path, List<Run> runs)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(runs.Count);
        foreach (var r in runs)
        {
            w.Write(r.Ids.Length); foreach (int i in r.Ids) w.Write(i);
            foreach (int a in r.RowArgmax) w.Write(a);
            w.Write(r.MeanNll);
            w.Write(r.LastLogits.Length); foreach (float f in r.LastLogits) w.Write(f);
            foreach (int g in r.Greedy) w.Write(g);
        }
    }

    private static List<Run> Load(string path)
    {
        using var rd = new BinaryReader(File.OpenRead(path));
        int n = rd.ReadInt32();
        var list = new List<Run>();
        for (int k = 0; k < n; k++)
        {
            int t = rd.ReadInt32(); var ids = new int[t]; for (int i = 0; i < t; i++) ids[i] = rd.ReadInt32();
            var am = new int[t]; for (int i = 0; i < t; i++) am[i] = rd.ReadInt32();
            double nll = rd.ReadDouble();
            int v = rd.ReadInt32(); var last = new float[v]; for (int i = 0; i < v; i++) last[i] = rd.ReadSingle();
            var g = new int[GreedyTokens]; for (int i = 0; i < g.Length; i++) g[i] = rd.ReadInt32();
            list.Add(new Run(ids, am, nll, last, g));
        }
        return list;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), sp = 0, sq = 0;
        foreach (float v in p) sp += Math.Exp(v - mp);
        foreach (float v in q) sq += Math.Exp(v - mq);
        double lsp = Math.Log(sp) + mp, lsq = Math.Log(sq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lsp, lq = q[i] - lsq;
            double pp = Math.Exp(lp);
            if (pp > 0) kl += pp * (lp - lq);
        }
        return kl;
    }

    private static int[][] Inputs(string skeleton)
    {
        using var g = GgufFile.Open(skeleton);
        var tok = GgufBpeTokenizerFactory.Load(g.Metadata);
        var list = new List<int[]> { tok.Encode(Passage).ToArray() };
        foreach (string p in Prompts) list.Add(tok.Encode(p).ToArray());
        return list.ToArray();
    }

    [SkippableFact]
    public void Stage_CpuReference()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "cpu", "DOTLLM_EDGE0_VK_E2E=cpu (CPU minutes; hold the cpu resource lock)");
        string refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_REF") ?? throw new InvalidOperationException("set DOTLLM_EDGE0_REF");
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k"));
        var inputs = Inputs(fx.Path!);
        var (model, gguf, _) = ModelLoader.LoadFromGguf(fx.Path!, DotLLM.Core.Configuration.ThreadingConfig.Auto);
        using (gguf) using (model)
        {
            var runs = new List<Run>();
            for (int i = 0; i < inputs.Length; i++)
            {
                runs.Add(Execute(model, inputs[i], nll: i == 0, null));
                output.WriteLine($"cpu input {i}: {inputs[i].Length} tokens, nll {runs[^1].MeanNll:F4}, greedy {string.Join(' ', runs[^1].Greedy.Take(8))}");
            }
            Save(refPath, runs);
        }
    }

    [SkippableFact]
    public void Stage_VulkanAgainstCpuReference()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "vulkan", "DOTLLM_EDGE0_VK_E2E=vulkan (loads ~52 GiB onto the GPU; hold the gpu lock)");
        string refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_REF") ?? throw new InvalidOperationException("set DOTLLM_EDGE0_REF");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k"));
        var cpu = Load(refPath);
        var inputs = Inputs(fx.Path!);
        using var gguf = GgufFile.Open(fx.Path!);
        using var device = VulkanDevice.Create();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var vk = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), spvDir);
        output.WriteLine($"vulkan load: {sw.Elapsed.TotalSeconds:F1} s; device live {device.LiveBytesTotal() / (double)(1L << 30):F2} GiB");

        for (int i = 0; i < inputs.Length; i++)
        {
            var r = Execute(vk, inputs[i], nll: i == 0, n => vk.TrySetAllRowLogitsLimit(n));
            var c = cpu[i];
            double kl = Kl(c.LastLogits, r.LastLogits);
            int rowAgree = 0; for (int t = 0; t < c.RowArgmax.Length; t++) if (c.RowArgmax[t] == r.RowArgmax[t]) rowAgree++;
            int greedyAgree = 0; for (int t = 0; t < GreedyTokens; t++) if (c.Greedy[t] == r.Greedy[t]) greedyAgree++;
            int firstDiv = Array.FindIndex(Enumerable.Range(0, GreedyTokens).ToArray(), t => c.Greedy[t] != r.Greedy[t]);
            output.WriteLine($"input {i}: {inputs[i].Length} tok; KL(cpu||vk) last row {kl:E3}; row argmax agree {rowAgree}/{c.RowArgmax.Length}; " +
                             $"greedy agree {greedyAgree}/{GreedyTokens} (first divergence {(firstDiv < 0 ? "none" : firstDiv.ToString())}); " +
                             (i == 0 ? $"NLL cpu {c.MeanNll:F4} vk {r.MeanNll:F4}" : ""));
            if (i == 0) Assert.True(Math.Abs(c.MeanNll - r.MeanNll) < 0.05, $"NLL cpu {c.MeanNll:F4} vs vulkan {r.MeanNll:F4}");
            Assert.True(kl < 0.05, $"input {i}: KL {kl:E3}");
            Assert.True(r.RowArgmax[^1] == c.RowArgmax[^1], $"input {i}: next-token argmax differs");
        }
    }
}
