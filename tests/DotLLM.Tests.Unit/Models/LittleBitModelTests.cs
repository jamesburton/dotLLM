using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Issue #864 step 2: the Qwen3-0.6B / 0.55 bpw LittleBit checkpoint run end to end on the CPU path as a dense Qwen3 with
/// factorized projections, against a PyTorch (bf16, trainer forward) reference generated offline by
/// <c>tools/littlebit-qat/gen_dotllm_reference.py</c> (fixture: 3 prompts x 40 greedy tokens + top-8 logits per position).
/// Checkpoint: <c>DOTLLM_LITTLEBIT_CKPT</c> or <c>~/.dotllm/test-cache/littlebit-qwen3-0.6b-055</c>; skips cleanly when absent.
/// Serialised: the dense-control test toggles a process-wide environment variable.
/// </summary>
[Collection("SequentialFileIO")]
public sealed class LittleBitModelTests(ITestOutputHelper output)
{
    private const string ControlEnv = "DOTLLM_LITTLEBIT_DENSE_CONTROL";

    private static string? CheckpointDir()
    {
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_LITTLEBIT_CKPT");
        if (string.IsNullOrEmpty(dir))
            dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "littlebit-qwen3-0.6b-055");
        return File.Exists(Path.Combine(dir, "model.safetensors")) && File.Exists(Path.Combine(dir, "config.json")) ? dir : null;
    }

    private static JsonElement Reference()
        => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Cpu", "Kernels", "LittleBitData",
            "littlebit_generation_reference.json"))).RootElement;

    private sealed record Metrics(int Positions, int Top1Agree, double MaxLogitDiff, int Top1Agree32, double MaxLogitDiff32,
        int FreeRunMatched, int FreeRunMatched32, int FreeRunTotal, float[][] LastLogits);

    private static unsafe float[] Rows(IModel m, int[] ids, int fromRow, int count)
    {
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        using ITensor t = m.Forward(ids, pos, -1);
        int vocab = t.Shape[1];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)fromRow * vocab, count * vocab).ToArray();
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static double MaxTopDiff(ReadOnlySpan<float> lg, JsonElement step)
    {
        var ids = step.GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var vals = step.GetProperty("top_logits").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        double d = 0;
        for (int i = 0; i < ids.Length; i++) d = Math.Max(d, Math.Abs(lg[ids[i]] - vals[i]));
        return d;
    }

    private Metrics Evaluate(string dir)
    {
        var (model, src, _) = ModelLoader.LoadFromSafetensors(dir, new ThreadingConfig(0));
        using (src as IDisposable)
        using (model)
        {
            model.TrySetAllRowLogitsLimit(1024);
            int pos = 0, agree = 0, agree32 = 0, matched = 0, matched32 = 0, total = 0;
            double maxDiff = 0, maxDiff32 = 0;
            var last = new List<float[]>();
            foreach (var c in Reference().GetProperty("cases").EnumerateArray())
            {
                int[] prompt = c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                int[] gen = c.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                // Teacher-forced on torch's own greedy path: row (prompt.Length-1+s) predicts gen[s].
                int[] full = prompt.Concat(gen.Take(gen.Length - 1)).ToArray();
                float[] flat = Rows(model, full, prompt.Length - 1, gen.Length);
                int vocab = flat.Length / gen.Length;
                int s2 = 0;
                var steps32 = c.GetProperty("steps_fp32").EnumerateArray().ToArray();
                foreach (var step in c.GetProperty("steps").EnumerateArray())
                {
                    var lg = new ReadOnlySpan<float>(flat, s2 * vocab, vocab);
                    pos++;
                    int am = Argmax(lg);
                    if (am == step.GetProperty("argmax").GetInt32()) agree++;
                    if (am == steps32[s2].GetProperty("argmax").GetInt32()) agree32++;
                    maxDiff = Math.Max(maxDiff, MaxTopDiff(lg, step));
                    maxDiff32 = Math.Max(maxDiff32, MaxTopDiff(lg, steps32[s2]));
                    s2++;
                }
                last.Add(flat[((gen.Length - 1) * vocab)..]);

                // Free-running greedy (no cache: recompute the whole sequence) vs torch's greedy continuation.
                var seq = new List<int>(prompt);
                int match = 0;
                for (int s = 0; s < gen.Length; s++)
                {
                    float[] lg = Rows(model, seq.ToArray(), seq.Count - 1, 1);
                    int next = Argmax(lg);
                    if (next == gen[s]) match++;
                    seq.Add(next);
                }
                int[] gen32 = c.GetProperty("generated_ids_fp32").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                int match32 = seq.Skip(prompt.Length).Zip(gen32).Count(z => z.First == z.Second);
                matched += match; matched32 += match32; total += gen.Length;
                output.WriteLine($"  prompt '{c.GetProperty("prompt").GetString()!.Replace("\n", "\\n")}': free-run greedy matches torch-bf16 on " +
                                 $"{match}/{gen.Length} tokens (identical={seq.Skip(prompt.Length).SequenceEqual(gen)}), torch-fp32 on {match32}/{gen.Length} (identical={seq.Skip(prompt.Length).SequenceEqual(gen32)})");
            }
            return new Metrics(pos, agree, maxDiff, agree32, maxDiff32, matched, matched32, total, last.ToArray());
        }
    }

    private void Report(string label, Metrics m)
        => output.WriteLine($"{label}: vs torch-fp32 top-1 {m.Top1Agree32}/{m.Positions}, max|logit diff| (torch top-8) {m.MaxLogitDiff32:F4}, free-run {m.FreeRunMatched32}/{m.FreeRunTotal}; " +
                            $"vs torch-bf16 top-1 {m.Top1Agree}/{m.Positions}, max|logit diff| {m.MaxLogitDiff:F3}, free-run {m.FreeRunMatched}/{m.FreeRunTotal}");

    [Fact]
    public void Factorized_TeacherForcedTop1_AndGreedyAgreeWithTorch()
    {
        string? dir = CheckpointDir();
        if (dir is null) { output.WriteLine("SKIP: LittleBit checkpoint absent (set DOTLLM_LITTLEBIT_CKPT)."); return; }
        var m = Evaluate(dir);
        Report("factorized", m);
        // Like-for-like oracle is torch in fp32 (the bf16 torch forward flips near-tied argmaxes against ANY fp32 engine:
        // torch-bf16 itself only agrees with torch-fp32 on 115/120 of these positions).
        Assert.True(m.Top1Agree32 >= m.Positions * 0.99, $"top-1 agreement with torch-fp32 {m.Top1Agree32}/{m.Positions}");
        Assert.True(m.Top1Agree >= m.Positions * 0.94, $"top-1 agreement with torch-bf16 {m.Top1Agree}/{m.Positions}");
    }

    [Fact]
    public void F32DecodedControl_MatchesFactorizedKernel()
    {
        string? dir = CheckpointDir();
        if (dir is null) { output.WriteLine("SKIP: LittleBit checkpoint absent (set DOTLLM_LITTLEBIT_CKPT)."); return; }
        var fact = Evaluate(dir);
        Metrics dense;
        Environment.SetEnvironmentVariable(ControlEnv, "1");
        try { dense = Evaluate(dir); }
        finally { Environment.SetEnvironmentVariable(ControlEnv, null); }
        double worst = 0;
        for (int i = 0; i < fact.LastLogits.Length; i++)
            for (int v = 0; v < fact.LastLogits[i].Length; v++)
                worst = Math.Max(worst, Math.Abs(fact.LastLogits[i][v] - dense.LastLogits[i][v]));
        Report("F32-decoded dense control", dense);
        output.WriteLine($"factorized kernel vs F32-decoded dense, last-position logits, max abs diff = {worst:E2}");
        Assert.True(worst < 5e-2, $"factorized vs F32-decoded dense logits differ by {worst}");
    }

    [Fact]
    public void FactorizedCheckpoint_IsRejectedByNonCpuLoaders()
    {
        string? dir = CheckpointDir();
        if (dir is null) { output.WriteLine("SKIP: LittleBit checkpoint absent."); return; }
        var (src, cfg) = ModelLoader.OpenSafetensorsAndConfig(dir);
        using (src as IDisposable)
        {
            // The shared weight loader (used by Vulkan/CUDA/HIP) must refuse unless the CPU path opts in.
            var ex = Assert.Throws<NotSupportedException>(() => TransformerWeightsSafetensorsLoader.Load(src, cfg));
            Assert.Contains("CPU backend only", ex.Message);
            // ...and the CPU opt-in refuses other architectures.
            var other = cfg with { Architecture = Architecture.Llama };
            var ex2 = Assert.Throws<NotSupportedException>(() => TransformerWeightsSafetensorsLoader.Load(src, other, null, allowFactorized: true));
            Assert.Contains("dense Qwen3", ex2.Message);
        }
    }
}
