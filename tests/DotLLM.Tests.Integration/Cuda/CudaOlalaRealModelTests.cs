using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Olala-v1.0-mini-7A1B q8_0-safe2 on CUDA (#1984) against the CPU forward on the prompts the CPU PR used (capital / code / chat; token ids come
/// from the HF oracle metadata <c>real-strict-gguf-norms-*</c>). The CPU logits of the SAME GGUF are dumped once on the dev box
/// (<see cref="Dump_CpuReference"/>, <c>DOTLLM_OLALA_DUMP_CPU_REF=1</c>) and shipped with the oracle dir, so the GPU box does not need a second
/// 7 GB CPU load. Skipped when the GGUF, the GPU or the reference files are absent: always read the <c>Skipped:</c> count.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CudaOlalaRealModelTests
{
    private static readonly string[] Prompts = ["capital", "code", "chat"];
    private readonly ITestOutputHelper _out;
    public CudaOlalaRealModelTests(ITestOutputHelper output) => _out = output;

    private static string RefPath(string prompt, string ext) => Path.Combine(OlalaOracleData.Dir()!, $"real-cpu-q8-{prompt}.{ext}");

    /// <summary>Writes the CPU q8_0-safe2 reference: prefill rows + teacher-forced decode rows (HF greedy ids) + the CPU's own free-running greedy ids.</summary>
    [SkippableFact]
    public void Dump_CpuReference()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_OLALA_DUMP_CPU_REF") != "1", "set DOTLLM_OLALA_DUMP_CPU_REF=1 to (re)generate the CPU reference");
        string? gguf = OlalaOracleData.Q8Safe2(); Skip.If(gguf is null, "q8_0-safe2 GGUF not found");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var cpu = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(file, cfg, ThreadingConfig.Auto));
        foreach (string prompt in Prompts)
        {
            var c = OlalaOracleData.Load($"real-strict-gguf-norms-{prompt}"); if (c is null) continue;
            cpu.ResetSequenceState();
            var rows = new List<float[]>();
            using (var full = cpu.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1)) rows.Add(OlalaOracleData.Rows(full, c.L, c.Vocab));
            for (int j = 1; j < c.N; j++)
            {
                using var s = cpu.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
                rows.Add(OlalaOracleData.Rows(s, 1, c.Vocab));
            }
            using (var fs = File.Create(RefPath(prompt, "logits.f32")))
                foreach (var r in rows) { var b = new byte[r.Length * 4]; Buffer.BlockCopy(r, 0, b, 0, b.Length); fs.Write(b); }
            cpu.ResetSequenceState();
            var gen = new List<int>();
            using (var pre = cpu.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, true)) gen.Add(OlalaOracleData.Argmax(OlalaOracleData.Rows(pre, 1, c.Vocab)));
            for (int j = 1; j < c.N; j++)
            {
                using var s = cpu.Forward([gen[^1]], [c.L + j - 1], -1, null, true);
                gen.Add(OlalaOracleData.Argmax(OlalaOracleData.Rows(s, 1, c.Vocab)));
            }
            File.WriteAllText(RefPath(prompt, "greedy.json"), JsonSerializer.Serialize(gen));
            _out.WriteLine($"{prompt}: wrote {c.L + c.N - 1} reference rows, greedy {string.Join(",", gen.Take(8))}...");
        }
    }

    private static (float[] Rows, int[] Greedy)? LoadRef(string prompt, OlalaOracleData.Case c)
    {
        if (OlalaOracleData.Dir() is null || !File.Exists(RefPath(prompt, "logits.f32")) || !File.Exists(RefPath(prompt, "greedy.json"))) return null;
        byte[] raw = File.ReadAllBytes(RefPath(prompt, "logits.f32"));
        var rows = new float[raw.Length / 4]; Buffer.BlockCopy(raw, 0, rows, 0, raw.Length);
        Assert.Equal((long)(c.L + c.N - 1) * c.Vocab, rows.Length);
        return (rows, JsonSerializer.Deserialize<int[]>(File.ReadAllText(RefPath(prompt, "greedy.json")))!);
    }

    [SkippableFact]
    public void Q8Safe2_Cuda_MatchesCpuReference_LogitsAndGreedy()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? gguf = OlalaOracleData.Q8Safe2(); Skip.If(gguf is null, "q8_0-safe2 GGUF not found");
        var cases = Prompts.Select(p => (p, c: OlalaOracleData.Load($"real-strict-gguf-norms-{p}"))).Where(x => x.c is not null).ToArray();
        Skip.If(cases.Length == 0 || cases.Any(x => LoadRef(x.p, x.c!) is null), "CPU reference files (real-cpu-q8-*) not generated: DOTLLM_OLALA_DUMP_CPU_REF=1");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var model = Assert.IsType<CudaOlalaTransformerModel>(CudaModelLoader.CreateFromGguf(file, cfg).Model);
        foreach (var (prompt, c0) in cases)
        {
            var c = c0!; var (refRows, refGreedy) = LoadRef(prompt, c)!.Value;
            int V = c.Vocab;
            // prefill rows
            model.ResetSequenceState();
            var klPre = new List<double>(); int top1Pre = 0;
            using (var full = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1))
            {
                float[] got = OlalaOracleData.Rows(full, c.L, V);
                for (int t = 0; t < c.L; t++)
                {
                    var r = refRows.AsSpan(t * V, V); var g = got.AsSpan(t * V, V);
                    foreach (float v in g) Assert.True(float.IsFinite(v), $"{prompt}: non-finite logit at row {t}");
                    klPre.Add(OlalaOracleData.Kl(r, g));
                    if (OlalaOracleData.Argmax(r) == OlalaOracleData.Argmax(g)) top1Pre++;
                }
            }
            // teacher-forced decode rows
            model.ResetSequenceState();
            using (model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, true)) { }
            var klDec = new List<double>(); int top1Dec = 0;
            for (int j = 1; j < c.N; j++)
            {
                using var s = model.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
                var g = OlalaOracleData.Rows(s, 1, V); var r = refRows.AsSpan((c.L + j - 1) * V, V);
                klDec.Add(OlalaOracleData.Kl(r, g));
                if (OlalaOracleData.Argmax(r) == OlalaOracleData.Argmax(g)) top1Dec++;
            }
            // free-running greedy
            model.ResetSequenceState();
            var gen = new List<int>();
            using (var pre = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, true)) gen.Add(OlalaOracleData.Argmax(OlalaOracleData.Rows(pre, 1, V)));
            for (int j = 1; j < c.N; j++)
            {
                using var s = model.Forward([gen[^1]], [c.L + j - 1], -1, null, true);
                gen.Add(OlalaOracleData.Argmax(OlalaOracleData.Rows(s, 1, V)));
            }
            int firstDiff = Enumerable.Range(0, c.N).FirstOrDefault(i => gen[i] != refGreedy[i], -1);
            klPre.Sort(); klDec.Sort();
            double Med(List<double> l) => l[l.Count / 2];
            _out.WriteLine($"{prompt}: prefill {c.L} rows: median KL {Med(klPre):E2}, worst {klPre[^1]:E2}, top-1 {top1Pre}/{c.L}");
            _out.WriteLine($"{prompt}: decode {c.N - 1} steps teacher-forced: median KL {Med(klDec):E2}, worst {klDec[^1]:E2}, top-1 {top1Dec}/{c.N - 1}");
            _out.WriteLine($"{prompt}: greedy {c.N} tokens vs CPU: {(firstDiff < 0 ? "identical" : $"first divergence at token {firstDiff}")}");
            // Gates, set from the measured RTX 3060 run (2026-11-25): CUDA keeps BF16 activations, the CPU F32, so agreement is bounded by BF16 noise.
            // Measured: capital median KL pre 5.9e-4 / dec 1.5e-3, top-1 5/5 + 22/23, greedy diverges at token 4; code 4.8e-4 / 2.8e-5, 13/13 + 23/23, identical;
            // chat (300 tokens, numerically chaotic even for HF) median KL pre 2.2e-2 / dec 5.6e-4, top-1 262/300 + 23/23, identical.
            bool longPrompt = c.L > 100;
            Assert.True(Med(klDec) < 5e-3, $"{prompt}: decode median KL {Med(klDec):E2}");
            Assert.True(top1Dec >= c.N - 2, $"{prompt}: decode top-1 {top1Dec}/{c.N - 1}");
            if (longPrompt)
            {
                Assert.True(Med(klPre) < 5e-2, $"{prompt}: prefill median KL {Med(klPre):E2}");
                Assert.True(top1Pre >= c.L * 0.8, $"{prompt}: prefill top-1 {top1Pre}/{c.L}");
            }
            else
            {
                Assert.True(Med(klPre) < 3e-3, $"{prompt}: prefill median KL {Med(klPre):E2}");
                Assert.True(top1Pre >= c.L - 1, $"{prompt}: prefill top-1 {top1Pre}/{c.L}");
            }
            Assert.True(firstDiff < 0 || firstDiff >= 4, $"{prompt}: greedy diverges from the CPU at token {firstDiff}");
        }
    }

    /// <summary>Opt-in throughput report (<c>DOTLLM_OLALA_BENCH=1</c>): prefill and decode tok/s of the chat prompt, several repetitions. Ranges, not points.</summary>
    [SkippableFact]
    public void Q8Safe2_Cuda_ThroughputReport()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_OLALA_BENCH") != "1", "set DOTLLM_OLALA_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? gguf = OlalaOracleData.Q8Safe2(); Skip.If(gguf is null, "q8_0-safe2 GGUF not found");
        var c = OlalaOracleData.Load("real-strict-gguf-norms-chat"); Skip.If(c is null, "chat oracle case missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var model = Assert.IsType<CudaOlalaTransformerModel>(CudaModelLoader.CreateFromGguf(file, cfg).Model);
        int[] pos = Enumerable.Range(0, c!.L).ToArray();
        var pp = new List<double>(); var tg = new List<double>();
        for (int rep = 0; rep < 6; rep++)
        {
            model.ResetSequenceState();
            var sw = Stopwatch.StartNew();
            using (model.Forward(c.Ids, pos, -1, null, true)) { }
            sw.Stop();
            double ppTps = c.L / sw.Elapsed.TotalSeconds;
            sw.Restart();
            int steps = 48, cur = c.GreedyIds[0];
            for (int j = 0; j < steps; j++)
            {
                using var s = model.Forward([cur], [c.L + j], -1, null, true);
                cur = OlalaOracleData.Argmax(OlalaOracleData.Rows(s, 1, c.Vocab));
            }
            sw.Stop();
            double tgTps = steps / sw.Elapsed.TotalSeconds;
            if (rep > 0) { pp.Add(ppTps); tg.Add(tgTps); }   // rep 0 is the warm-up (JIT of the PTX, allocations)
            _out.WriteLine($"rep {rep}: prefill {c.L} tokens {ppTps:F0} tok/s, decode {steps} steps {tgTps:F1} tok/s");
        }
        _out.WriteLine($"RANGE prefill {pp.Min():F0}-{pp.Max():F0} tok/s, decode {tg.Min():F1}-{tg.Max():F1} tok/s (5 reps after warm-up, includes one logits D2H per step)");
    }
}
