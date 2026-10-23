using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in real-weights check of CPU-expert offload (#1187, Flint F7): the whole 49-layer Q4_1 Flint on a 12 GB GPU with
/// its routed experts on the CPU (read from the mmap). Greedy-decodes the five F1 validation chat prompts on CUDA, then
/// teacher-forces the same ids through the dotLLM CPU model and reports per-position KL(cpu || cuda) and top-1 agreement,
/// plus 512-token prefill and decode throughput, VRAM and host private bytes.
/// </summary>
/// <remarks>
/// Env: <c>DOTLLM_FLINT_GGUF</c> (required), <c>DOTLLM_FLINT_N_CPU_MOE</c> (<c>all</c> = <c>--cpu-moe</c>, a number =
/// <c>--n-cpu-moe</c>, <c>auto</c> (default) = the loader's fit), <c>DOTLLM_FLINT_GEN</c> (greedy tokens per prompt, default 32).
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaFlintCpuMoeRealProbe
{
    private static readonly (string Name, (string Role, string Content)[] Msgs)[] Prompts =
    [
        ("arith", [("user", "What is 17 * 23? Answer with just the number.")]),
        ("code", [("user", "Write a Python function that checks whether a string is a palindrome.")]),
        ("prose", [("system", "You are a concise assistant."), ("user", "Explain in two sentences why the sky is blue.")]),
        ("multi", [("user", "My name is Ada."), ("assistant", "Nice to meet you, Ada!"),
            ("user", "What is my name, and what is a good name for a pet cat?")]),
        ("cjk", [("user", "用一句话介绍一下你自己。")]),
    ];

    private readonly ITestOutputHelper _out;

    public CudaFlintCpuMoeRealProbe(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void FullModel_CpuExperts_VsCpuModel_AndThroughput()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_FLINT_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "set DOTLLM_FLINT_GGUF to the Flint GGUF");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        int gen = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_FLINT_GEN"), out int g) ? g : 32;
        string nCpu = Environment.GetEnvironmentVariable("DOTLLM_FLINT_N_CPU_MOE") ?? "auto";
        var options = new MoeOffloadOptions
        {
            CpuMoeLayers = nCpu == "all" ? MoeOffloadOptions.AllLayers : int.TryParse(nCpu, out int n) ? n : 0,
            ContextTokens = 4096,
            Log = s => _out.WriteLine(s),
        };

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var tmpl = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok, cfg.Architecture) ?? throw new InvalidOperationException("no chat template");
        int vocab = cfg.VocabSize;

        var seqs = new List<(string Name, int[] Ids, int PromptLen, List<float[]> CudaRows)>();
        CudaMemoryInfo.TryGet(0, out long free0, out long total);
        long priv0 = Process.GetCurrentProcess().PrivateMemorySize64;
        var sw = Stopwatch.StartNew();
        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0, moeOffload: options);
        double loadS = sw.Elapsed.TotalSeconds;
        long minFree = long.MaxValue;
        using (model)
        {
            var cuda = Assert.IsType<CudaTransformerModel>(model);
            CudaMemoryInfo.TryGet(0, out long freeLoaded, out _);
            long privLoaded = Process.GetCurrentProcess().PrivateMemorySize64;
            _out.WriteLine($"load {loadS:F1} s, CpuMoeLayers {cuda.CpuMoeLayers}; VRAM after load {(free0 - freeLoaded) / 1048576} MiB of {total / 1048576}; "
                + $"private bytes {priv0 / 1048576} -> {privLoaded / 1048576} MiB");

            var decodeMs = new List<double>();
            foreach (var (name, msgs) in Prompts)
            {
                string prompt = tmpl.Apply(msgs.Select(m => new ChatMessage { Role = m.Role, Content = m.Content }).ToList(),
                    new ChatTemplateOptions { AddGenerationPrompt = true });
                var ids = tok.EncodeRaw(prompt).ToList();
                int promptLen = ids.Count;
                var rows = new List<float[]>();
                using var kv = kvFactory(4096);
                using (ITensor p = cuda.Forward(ids.ToArray(), Enumerable.Range(0, promptLen).ToArray(), deviceId: -1, kv))
                    rows.Add(LastRow(p, vocab));
                for (int i = 0; i < gen; i++)
                {
                    int next = ArgMax(rows[^1]);
                    if (i == gen - 1) break;
                    ids.Add(next);
                    sw.Restart();
                    using ITensor d = cuda.Forward([next], [ids.Count - 1], deviceId: -1, kv);
                    rows.Add(LastRow(d, vocab));
                    decodeMs.Add(sw.Elapsed.TotalMilliseconds);
                }
                CudaMemoryInfo.TryGet(0, out long f, out _);
                minFree = Math.Min(minFree, f);
                seqs.Add((name, ids.ToArray(), promptLen, rows));
                _out.WriteLine($"{name}: prompt {promptLen}, CUDA greedy: {Ascii(tok.Decode(ids.Skip(promptLen).ToArray()))}");
            }

            // Throughput: one 512-token prefill of real text, then 64 greedy decode steps.
            string text = File.ReadAllText(Path.Combine(RepoRoot(), "docs", "ARCHITECTURE.md")).Replace("\r\n", "\n");
            int[] pre = tok.EncodeRaw(text).Take(512).ToArray();
            using (var kv = kvFactory(4096))
            {
                long cpuUs0 = cuda.CpuMoeMicroseconds;
                sw.Restart();
                float[] last;
                using (ITensor p = cuda.Forward(pre, Enumerable.Range(0, pre.Length).ToArray(), deviceId: -1, kv))
                    last = LastRow(p, vocab);
                double prefillS = sw.Elapsed.TotalSeconds;
                long cpuUsPre = cuda.CpuMoeMicroseconds - cpuUs0;
                int pos = pre.Length;
                cpuUs0 = cuda.CpuMoeMicroseconds;
                sw.Restart();
                for (int i = 0; i < 64; i++)
                {
                    using ITensor d = cuda.Forward([ArgMax(last)], [pos++], deviceId: -1, kv);
                    last = LastRow(d, vocab);
                }
                double decS = sw.Elapsed.TotalSeconds;
                CudaMemoryInfo.TryGet(0, out long f, out _);
                minFree = Math.Min(minFree, f);
                _out.WriteLine($"prefill {pre.Length} tok: {pre.Length / prefillS:F1} tok/s (CPU leg {cpuUsPre / 1000.0:F0} ms of {prefillS * 1000:F0} ms); "
                    + $"decode 64 @ {pre.Length}: {64 / decS:F2} tok/s ({decS * 1000 / 64:F1} ms/token, CPU leg {(cuda.CpuMoeMicroseconds - cpuUs0) / 64000.0:F1} ms/token)");
            }
            _out.WriteLine($"decode over prompts: {decodeMs.Skip(1).Average():F1} ms/token median {decodeMs.Order().ElementAt(decodeMs.Count / 2):F1}");
            _out.WriteLine($"peak VRAM used (device-wide) {(free0 - minFree) / 1048576} MiB; private bytes after runs {Process.GetCurrentProcess().PrivateMemorySize64 / 1048576} MiB "
                + $"(WDDM counts device allocations in private bytes 1:1, see CudaFlintRealWeightsProbe); peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576} MiB");
        }

        // CPU reference: teacher-force the CUDA greedy ids.
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, cfg, new ThreadingConfig(0));
        int agreeAll = 0, totalAll = 0;
        double klMax = 0, klSum = 0;
        foreach (var (name, ids, promptLen, rows) in seqs)
        {
            cpuModel.ResetSequenceState();
            cpuModel.TrySetAllRowLogitsLimit(ids.Length + 1);
            using ITensor lg = cpuModel.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), deviceId: -1);
            int agree = 0;
            double kmax = 0, ksum = 0;
            for (int r = 0; r < rows.Count; r++)
            {
                var cpuRow = new ReadOnlySpan<float>((float*)lg.DataPointer + (long)(promptLen - 1 + r) * vocab, vocab).ToArray();
                if (ArgMax(cpuRow) == ArgMax(rows[r])) agree++;
                double kl = Kl(cpuRow, rows[r]);
                kmax = Math.Max(kmax, kl);
                ksum += kl;
            }
            _out.WriteLine($"{name}: top-1 {agree}/{rows.Count}, KL(cpu||cuda) mean {ksum / rows.Count:E2} max {kmax:E2}");
            agreeAll += agree;
            totalAll += rows.Count;
            klMax = Math.Max(klMax, kmax);
            klSum += ksum;
        }
        _out.WriteLine($"ALL: top-1 {agreeAll}/{totalAll}, KL mean {klSum / totalAll:E2} max {klMax:E2}");
    }

    private static string RepoRoot()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d is not null && !File.Exists(Path.Combine(d.FullName, "CLAUDE.md"))) d = d.Parent;
        return d?.FullName ?? throw new DirectoryNotFoundException("repo root");
    }

    private static string Ascii(string s) => new(s.Take(120).Select(c => c is >= ' ' and <= '~' ? c : '.').ToArray());

    private static float[] LastRow(ITensor t, int vocab)
    {
        long rows = t.Shape.ElementCount / vocab;
        return new ReadOnlySpan<float>((float*)t.DataPointer + (rows - 1) * vocab, vocab).ToArray();
    }

    private static int ArgMax(float[] r)
    {
        int best = 0;
        for (int i = 1; i < r.Length; i++) if (r[i] > r[best]) best = i;
        return best;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        foreach (float v in p) zp += Math.Exp(v - mp);
        foreach (float v in q) zq += Math.Exp(v - mq);
        double kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - mp - Math.Log(zp), lq = q[i] - mq - Math.Log(zq);
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }
}
