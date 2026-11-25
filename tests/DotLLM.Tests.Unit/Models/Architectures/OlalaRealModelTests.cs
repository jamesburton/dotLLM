using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Core.Configuration;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Olala-v1.0-mini-7A1B real checkpoints (#1970) against the HF float32 oracle (<c>olala_oracle.py real</c>). Skipped unless the GGUFs
/// (HF hub cache <c>OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF</c>, or <c>DOTLLM_OLALA_BF16_GGUF</c> / <c>DOTLLM_OLALA_Q8_GGUF</c>) and the oracle files exist.
/// Each test loads a 7-14 GB model; they are heavy and run on the CPU.
/// </summary>
public sealed class OlalaRealModelTests
{
    private readonly ITestOutputHelper _output;
    public OlalaRealModelTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts = ["capital", "code", "chat"];

    private static string? FindGguf(string envVar, string file)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--OVHaiLLM--Olala-v1.0-mini-7A1B-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps))
        {
            string p = Path.Combine(d, file);
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private static string? Bf16() => FindGguf("DOTLLM_OLALA_BF16_GGUF", "Olala-v1.0-mini-7A1B-bf16.gguf");
    private static string? Q8() => FindGguf("DOTLLM_OLALA_Q8_GGUF", "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf");

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    private static ThreadingConfig Threads() => ThreadingConfig.Auto;

    [SkippableTheory]
    [InlineData("strict-gguf-norms", 5e-3, 2.0)]   // HF F32 (query F32) with the GGUF's own norm gains: engine differences only
    [InlineData("hf", 5e-2, 3.0)]   // informational: median gate relaxed below                  // HF as shipped (bf16 query/attention output, exact 1+w): adds the HF artifact and the converter's bf16 rounding of 1+w
    public void Bf16Gguf_MatchesHfOracle(string variant, double maxKl, double maxLogitAbs)
    {
        string? gguf = Bf16();
        Skip.If(gguf is null, "bf16 GGUF not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var model = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(file, cfg, Threads()));
        RunCases(model, variant, maxKl, maxLogitAbs, requireGreedyIdentity: variant != "hf");
    }

    private void RunCases(OlalaTransformerModel model, string variant, double maxKl, double maxLogitAbs, bool requireGreedyIdentity)
    {
        bool any = false;
        foreach (string prompt in Prompts)
        {
            var c = OlalaOracleTests.LoadCase($"real-{variant}-{prompt}");
            if (c is null) continue;
            any = true;
            model.ResetSequenceState();
            using var full = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1);
            float[] got = Rows(full, c.L, c.Vocab);
            double worstKl = 0, worstAbs = 0; int top1 = 0;
            var kls = new List<double>();
            for (int t = 0; t < c.L; t++)
            {
                var r = c.Full.AsSpan(t * c.Vocab, c.Vocab); var g = got.AsSpan(t * c.Vocab, c.Vocab);
                double rowKl = OlalaOracleTests.Kl(r, g); kls.Add(rowKl);
                worstKl = Math.Max(worstKl, rowKl);
                for (int i = 0; i < c.Vocab; i++) worstAbs = Math.Max(worstAbs, Math.Abs(r[i] - g[i]));
                if (OlalaOracleTests.Argmax(r) == OlalaOracleTests.Argmax(g)) top1++;
            }
            _output.WriteLine($"{variant}/{prompt}: prefill {c.L} rows: worst KL {worstKl:E2}, worst |dlogit| {worstAbs:E2}, top-1 {top1}/{c.L}");

            // teacher-forced decode over the HF greedy tokens, and the free-running greedy continuation
            model.ResetSequenceState();
            using var pre = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, lastTokenLogitsOnly: true);
            var generated = new List<int>();
            double decKl = 0; int decTop1 = 0;
            float[] row = Rows(pre, 1, c.Vocab);
            for (int j = 0; j < c.N; j++)
            {
                var r = c.Greedy.AsSpan(j * c.Vocab, c.Vocab);
                double stepKl = OlalaOracleTests.Kl(r, row), stepAbs = 0;
                decKl = Math.Max(decKl, stepKl); kls.Add(stepKl);
                for (int i = 0; i < c.Vocab; i++) stepAbs = Math.Max(stepAbs, Math.Abs(r[i] - row[i]));
                worstAbs = Math.Max(worstAbs, stepAbs);
                _output.WriteLine($"  {variant}/{prompt} step {j}: KL {stepKl:E2} max|dlogit| {stepAbs:E2}");
                if (OlalaOracleTests.Argmax(r) == OlalaOracleTests.Argmax(row)) decTop1++;
                if (j + 1 < c.N)
                {
                    using var step = model.Forward([c.GreedyIds[j]], [c.L + j], -1, null, lastTokenLogitsOnly: true);
                    row = Rows(step, 1, c.Vocab);
                }
            }
            _output.WriteLine($"{variant}/{prompt}: decode {c.N} steps (teacher-forced): worst KL {decKl:E2}, top-1 {decTop1}/{c.N}");
            // Discrete events (an MoE top-6 flip between two near-tied experts under float noise) make single rows jump while the bulk stays at
            // float level, so the gate is: a small median, few rows above 1e-4, and a bounded worst row.
            kls.Sort();
            double median = kls[kls.Count / 2]; int above = kls.Count(k => k > 1e-4);
            _output.WriteLine($"{variant}/{prompt}: median KL {median:E2}, rows above 1e-4: {above}/{kls.Count}, worst {Math.Max(worstKl, decKl):E2}");
            // The 300-token chat prompt is numerically chaotic even for HF itself (its own one-token-step path differs from its chunked
            // forward by up to ~3 logits on this prompt: activation outliers of 1e9-1e14 amplify float reassociation), so it gets a
            // noise-floor gate (median and top-1 only) instead of the float-level gate of the short prompts.
            bool longPrompt = c.L > 100;
            if (!longPrompt)
            {
                Assert.True(median < (variant == "hf" ? 1e-3 : 1e-5), $"{variant}/{prompt}: median KL {median:E2}");
                Assert.True(variant == "hf" || above <= Math.Max(1, kls.Count / 20), $"{variant}/{prompt}: {above}/{kls.Count} rows above 1e-4");
                Assert.True(worstKl < maxKl && decKl < maxKl, $"{variant}/{prompt}: KL prefill {worstKl:E2} decode {decKl:E2} > {maxKl:E1}");
                Assert.True(worstAbs < maxLogitAbs, $"{variant}/{prompt}: max |dlogit| {worstAbs:E2} > {maxLogitAbs}");
                Assert.True(top1 >= c.L - 1 && decTop1 >= c.N - 1, $"{variant}/{prompt}: top-1 prefill {top1}/{c.L} decode {decTop1}/{c.N}");
            }
            else if (variant != "hf")   // 'hf' on the long prompt is informational: HF's own bf16 casts move it by KL ~1 on single rows
            {
                Assert.True(median < 5e-4, $"{variant}/{prompt}: median KL {median:E2}");
                Assert.True(top1 >= c.L * 0.97 && decTop1 >= c.N - 1, $"{variant}/{prompt}: top-1 prefill {top1}/{c.L} decode {decTop1}/{c.N}");
            }

            // free-running greedy from the prefill
            model.ResetSequenceState();
            using var pre2 = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, lastTokenLogitsOnly: true);
            generated.Add(OlalaOracleTests.Argmax(Rows(pre2, 1, c.Vocab)));
            for (int j = 1; j < c.N; j++)
            {
                using var step = model.Forward([generated[^1]], [c.L + j - 1], -1, null, lastTokenLogitsOnly: true);
                generated.Add(OlalaOracleTests.Argmax(Rows(step, 1, c.Vocab)));
            }
            int firstDiff = Enumerable.Range(0, c.N).FirstOrDefault(i => generated[i] != c.GreedyIds[i], -1);
            _output.WriteLine($"{variant}/{prompt}: greedy {(firstDiff < 0 ? "identical" : $"first divergence at token {firstDiff}")} over {c.N} tokens");
            if (requireGreedyIdentity) Assert.Equal(c.GreedyIds, generated);
        }
        Skip.If(!any, $"no real oracle files for variant {variant} (olala_oracle.py real --variant ...)");
    }

    [SkippableFact]
    public void Q8Safe2_AgainstBf16_StaysWithinPublishedDivergence()
    {
        string? bf = Bf16(), q8 = Q8();
        Skip.If(bf is null || q8 is null, "bf16 / q8_0-safe2 GGUFs not in the HF cache");
        // Teacher-forced on the HF greedy tokens of each oracle prompt: KL(bf16 || q8_0-safe2) per position and top-1 agreement.
        var cases = Prompts.Select(p => OlalaOracleTests.LoadCase($"real-strict-gguf-norms-{p}")).Where(c => c is not null).Select(c => c!).ToArray();
        Skip.If(cases.Length == 0, "no real oracle files");
        var refRows = new List<float[]>();
        using (var f = GgufFile.Open(bf!))
        using (var m = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(f, GgufModelConfigExtractor.Extract(f.Metadata), Threads())))
            foreach (var c in cases) refRows.Add(Positions(m, c));
        double klSum = 0; int n = 0, agree = 0; bool nan = false;   // gated on the short prompts; the 300-token chat prompt is reported only
        using (var f = GgufFile.Open(q8!))
        using (var m = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(f, GgufModelConfigExtractor.Extract(f.Metadata), Threads())))
            for (int ci = 0; ci < cases.Length; ci++)
            {
                var c = cases[ci]; float[] q = Positions(m, c), r = refRows[ci];
                int rows = q.Length / c.Vocab;
                double caseKl = 0; int caseAgree = 0;
                for (int t = 0; t < rows; t++)
                {
                    var qr = q.AsSpan(t * c.Vocab, c.Vocab);
                    foreach (float v in qr) if (!float.IsFinite(v)) nan = true;
                    double kl = OlalaOracleTests.Kl(r.AsSpan(t * c.Vocab, c.Vocab), qr); caseKl += kl;
                    bool same = OlalaOracleTests.Argmax(r.AsSpan(t * c.Vocab, c.Vocab)) == OlalaOracleTests.Argmax(qr);
                    if (same) caseAgree++;
                    if (c.L <= 100) { klSum += kl; n++; if (same) agree++; }
                }
                _output.WriteLine($"  q8_0-safe2 vs bf16 / {c.Name}: {rows} positions, mean KL {caseKl / rows:E3}, same top-1 {caseAgree}/{rows}");
            }
        _output.WriteLine($"q8_0-safe2 vs bf16 (dotLLM CPU, {n} positions): mean KL {klSum / n:E3}, same top-1 {agree}/{n} = {100.0 * agree / n:F1}%  (OVH: KLD 0.0040, 96.9% over 16 wikitext chunks)");
        Assert.False(nan, "q8_0-safe2 produced non-finite logits (activation overflow?)");
        Assert.True(klSum / n < 0.03, $"mean KL {klSum / n:E3}");
        Assert.True(agree >= 0.90 * n, $"top-1 agreement {agree}/{n}");
    }

    private static float[] Positions(OlalaTransformerModel m, OlalaOracleTests.Case c)
    {
        // prompt rows then the HF greedy continuation, one forward each (prefill rows + teacher-forced decode rows)
        m.ResetSequenceState();
        var all = new List<float>();
        using (var full = m.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1)) all.AddRange(Rows(full, c.L, c.Vocab));
        for (int j = 0; j < c.N - 1; j++)
        {
            using var s = m.Forward([c.GreedyIds[j]], [c.L + j], -1, null, lastTokenLogitsOnly: true);
            all.AddRange(Rows(s, 1, c.Vocab));
        }
        return all.ToArray();
    }
}
