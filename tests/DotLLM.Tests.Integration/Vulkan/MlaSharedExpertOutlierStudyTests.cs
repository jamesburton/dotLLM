using DotLLM.Core.Attention;
using System.Globalization;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #949 outlier study. Opt-in (<c>DOTLLM_GLM_OUTLIER_STUDY=1</c>): ~24+ natural-text prompts of varied length, teacher-forced
/// decode (1 row per forward) on three GPU arms against the CPU oracle: control (F32 shared experts), packed, and packed + legacy
/// Q5_K GEMV (a third, differently-rounded packed arm). Reports per-arm KL distribution, outlier rate (KL &gt; 1), top-1, max-logit
/// magnitudes and the pairwise KL between arms on the outlier rows. Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class MlaSharedExpertOutlierStudyTests
{
    private readonly ITestOutputHelper _o;
    public MlaSharedExpertOutlierStudyTests(ITestOutputHelper o) => _o = o;

    [SkippableFact]
    public void OutlierStudy()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GLM_OUTLIER_STUDY") == "1", "opt-in");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub", "models--unsloth--GLM-4.7-Flash-GGUF", "snapshots");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM_GGUF");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            path = Directory.Exists(hub) ? Directory.EnumerateFiles(hub, "GLM-4.7-Flash-Q4_K_M.gguf", SearchOption.AllDirectories).FirstOrDefault() : null;
        Skip.If(path is null, "GLM file not found");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);

        string docs = Environment.GetEnvironmentVariable("DOTLLM_STUDY_DOCS") ?? @"C:\Development\dotLLM\docs";
        int[] lens = [24, 48, 96, 160, 240];
        var prompts = new List<int[]>();
        int li = 0;
        foreach (string f in Directory.EnumerateFiles(docs, "*.md").OrderBy(x => x, StringComparer.Ordinal))
        {
            string text = File.ReadAllText(f);
            for (int w = 0; w < 2 && prompts.Count < 40; w++)
            {
                int start = Math.Min(text.Length / 2 * w + 200, Math.Max(0, text.Length - 3000));
                int[] enc = tok.Encode(text.Substring(start, Math.Min(3000, text.Length - start)));
                int len = lens[li++ % lens.Length];
                if (enc.Length < len) continue;
                bool bos = addBos && tok.BosTokenId >= 0;
                var ids = new int[len + (bos ? 1 : 0)];
                int o = 0;
                if (bos) ids[o++] = tok.BosTokenId;
                Array.Copy(enc, 0, ids, o, len);
                prompts.Add(ids);
            }
        }
        int totalRows = prompts.Sum(p => p.Length);
        _o.WriteLine($"prompts {prompts.Count}, teacher-forced decode rows {totalRows}");

        var cpuRows = new List<float[]>();
        using (var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config))
        {
            foreach (int[] ids in prompts)
            {
                using var kv = new SimpleKvCache(KvGeometry.FromConfig(config), ids.Length + 2);
                using ITensor l = cpu.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
                cpuRows.AddRange(Rows(l, vocab));
            }
        }
        _o.WriteLine("cpu done");

        using var device = VulkanDevice.Create();
        var arms = new (string Name, string? Packed, string? Legacy)[]
        {
            ("control", "0", null), ("packed", null, null), ("packed+legacyQ5K", null, "1"),
        };
        var armRows = new List<float[]>[arms.Length];
        for (int a = 0; a < arms.Length; a++)
        {
            string? pPrev = Environment.GetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED");
            string? lPrev = Environment.GetEnvironmentVariable("DOTLLM_VK_Q5_K_GEMV_LEGACY");
            Environment.SetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED", arms[a].Packed);
            Environment.SetEnvironmentVariable("DOTLLM_VK_Q5_K_GEMV_LEGACY", arms[a].Legacy);
            try
            {
                var (m, mk) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
                using var disp = m as IDisposable;
                var rows = new List<float[]>(totalRows);
                foreach (int[] ids in prompts)
                {
                    using var kv = mk(ids.Length + 2);
                    for (int s = 0; s < ids.Length; s++)
                    {
                        using ITensor l = m.Forward(ids.AsSpan(s, 1), [s], -1, kv);
                        rows.Add(Rows(l, vocab)[^1]);
                    }
                }
                armRows[a] = rows;
            }
            finally
            {
                Environment.SetEnvironmentVariable("DOTLLM_VK_SHARED_EXPERT_PACKED", pPrev);
                Environment.SetEnvironmentVariable("DOTLLM_VK_Q5_K_GEMV_LEGACY", lPrev);
            }
            _o.WriteLine($"arm {arms[a].Name} done");
        }

        int n = totalRows;
        var kl = new double[arms.Length][];
        for (int a = 0; a < arms.Length; a++)
        {
            kl[a] = new double[n];
            int agree = 0;
            for (int i = 0; i < n; i++)
            {
                kl[a][i] = Kl(cpuRows[i], armRows[a][i]);
                if (ArgMax(cpuRows[i]) == ArgMax(armRows[a][i])) agree++;
            }
            var s = kl[a].OrderBy(x => x).ToArray();
            int trim = (int)(n * 0.99);
            _o.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"STAT {arms[a].Name,-18} mean {kl[a].Average():F5} median {s[n / 2]:F5} trimmed99 {s.Take(trim).Average():F5} p95 {s[(int)(n * 0.95)]:F5} p99 {s[(int)(n * 0.99)]:F5} max {s[^1]:F3} outliers(KL>1) {s.Count(x => x > 1)}/{n} (KL>0.1: {s.Count(x => x > 0.1)}) top1 {agree}/{n}"));
        }

        var ml = cpuRows.Select(r => (double)r.Max()).OrderBy(x => x).ToArray();
        _o.WriteLine(string.Create(CultureInfo.InvariantCulture, $"CPU max-logit: median {ml[n / 2]:F1} p95 {ml[(int)(n * .95)]:F1} max {ml[^1]:F1}; rows with cpu max-logit > 60: {ml.Count(x => x > 60)}"));

        var idx = Enumerable.Range(0, n).Where(i => kl.Any(k => k[i] > 0.3)).OrderByDescending(i => kl.Max(k => k[i])).Take(30).ToList();
        _o.WriteLine($"rows with any arm KL>0.3: {Enumerable.Range(0, n).Count(i => kl.Any(k => k[i] > 0.3))}");
        int[] pStart = new int[prompts.Count]; int acc = 0;
        for (int p = 0; p < prompts.Count; p++) { pStart[p] = acc; acc += prompts[p].Length; }
        foreach (int i in idx)
        {
            int p = Array.FindLastIndex(pStart, st => st <= i);
            double Kp(int x, int y) => Kl(armRows[x][i], armRows[y][i]);
            _o.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"ROW prompt {p} pos {i - pStart[p]}/{prompts[p].Length} | KL cpu||: control {kl[0][i]:F3} packed {kl[1][i]:F3} packedLeg {kl[2][i]:F3} | pairwise ctrl||packed {Kp(0, 1):F3} ctrl||leg {Kp(0, 2):F3} packed||leg {Kp(1, 2):F3} | maxlogit cpu {cpuRows[i].Max():F1} ctrl {armRows[0][i].Max():F1} packed {armRows[1][i].Max():F1} leg {armRows[2][i].Max():F1} | top1 cpu {ArgMax(cpuRows[i])} ctrl {ArgMax(armRows[0][i])} packed {ArgMax(armRows[1][i])} leg {ArgMax(armRows[2][i])}"));
        }
        for (int x = 0; x < arms.Length; x++)
            for (int y = x + 1; y < arms.Length; y++)
            {
                var d = Enumerable.Range(0, n).Select(i => Kl(armRows[x][i], armRows[y][i])).OrderBy(v => v).ToArray();
                _o.WriteLine(string.Create(CultureInfo.InvariantCulture, $"PAIR {arms[x].Name}||{arms[y].Name} mean {d.Average():F5} median {d[n / 2]:F5} p99 {d[(int)(n * .99)]:F4} max {d[^1]:F3} >1: {d.Count(v => v > 1)}"));
            }
    }

    private static unsafe float[][] Rows(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var res = new float[rows][];
        for (int r = 0; r < rows; r++)
            res[r] = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)r * vocab, vocab).ToArray();
        return res;
    }

    private static int ArgMax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++) { double lp = p[i] - lzp, lq = q[i] - lzq; kl += Math.Exp(lp) * (lp - lq); }
        return kl;
    }
}
