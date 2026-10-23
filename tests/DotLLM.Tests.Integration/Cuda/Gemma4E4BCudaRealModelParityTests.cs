using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #734 acceptance on CUDA: real gemma-4-E4B-it (Q4_K_M) vs the CPU reference produced by the Vulkan twin test
/// (<c>DOTLLM_GEMMA4E4B_REF</c> JSON: prompt ids, CPU greedy tokens, CPU prefill logits). Free-running greedy identity,
/// cached-decode teacher-forced top-1 agreement, prefill logit parity and throughput. Needs a 12 GB-class GPU.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma4E4BCudaRealModelParityTests
{
    private sealed record Ref(int[] PromptIds, int[] Generated, float[] PrefillLogits);

    private readonly ITestOutputHelper _output;
    public Gemma4E4BCudaRealModelParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void Cuda_E4B_MatchesCpuReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA device.");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4E4B_GGUF");
        string? refFile = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4E4B_REF");
        Skip.If(string.IsNullOrWhiteSpace(path) || !File.Exists(path), "Set DOTLLM_GEMMA4E4B_GGUF.");
        Skip.If(string.IsNullOrWhiteSpace(refFile) || !File.Exists(refFile), "Set DOTLLM_GEMMA4E4B_REF to the CPU reference JSON.");
        var refs = JsonSerializer.Deserialize<Ref[]>(File.ReadAllText(refFile!))!;

        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = CudaModelLoader.LoadFromGguf(path!, deviceId: 0);
        using var _g = gguf;
        using var _m = model;
        _output.WriteLine($"CUDA load: {loadSw.Elapsed.TotalSeconds:F1} s");
        int vocab = config.VocabSize;

        int tfSteps = 0, tfAgree = 0, greedyMatched = 0, greedyTotal = 0;
        double worstPrefill = 0;
        foreach (var r in refs)
        {
            // Free-running greedy through the cache.
            using var kv = model.CreateKvCache(r.PromptIds.Length + r.Generated.Length + 2);
            var gen = new List<int>();
            float[] prefill = Row(model.Forward(r.PromptIds, Enumerable.Range(0, r.PromptIds.Length).ToArray(), -1, kv), vocab);
            for (int i = 0; i < prefill.Length; i++) worstPrefill = Math.Max(worstPrefill, Math.Abs(prefill[i] - r.PrefillLogits[i]));
            Assert.Equal(ArgMax(r.PrefillLogits), ArgMax(prefill));
            gen.Add(ArgMax(prefill));
            for (int k = 1; k < r.Generated.Length; k++)
                gen.Add(ArgMax(Row(model.Forward([gen[^1]], [r.PromptIds.Length + k - 1], -1, kv), vocab)));
            int firstDiv = -1;
            for (int i = 0; i < gen.Count; i++) if (gen[i] != r.Generated[i]) { firstDiv = i; break; }
            greedyTotal += gen.Count;
            greedyMatched += firstDiv < 0 ? gen.Count : firstDiv;
            _output.WriteLine($"prompt: firstDivergence={firstDiv}");

            // Teacher-forced cached decode.
            using var kv2 = model.CreateKvCache(r.PromptIds.Length + r.Generated.Length + 2);
            float[] row = Row(model.Forward(r.PromptIds, Enumerable.Range(0, r.PromptIds.Length).ToArray(), -1, kv2), vocab);
            for (int k = 0; k < r.Generated.Length; k++)
            {
                tfSteps++;
                if (ArgMax(row) == r.Generated[k]) tfAgree++;
                else _output.WriteLine($"  teacher-forced disagreement at step {k}: cpu token {r.Generated[k]}, cuda top1 {ArgMax(row)}, "
                    + $"cuda logit gap {row[ArgMax(row)] - row[r.Generated[k]]:F3}");
                if (k + 1 < r.Generated.Length)
                    row = Row(model.Forward([r.Generated[k]], [r.PromptIds.Length + k], -1, kv2), vocab);
            }
        }
        _output.WriteLine($"greedy matched {greedyMatched}/{greedyTotal}; teacher-forced top-1 {tfAgree}/{tfSteps}; worst prefill |dLogit|={worstPrefill:F4}");

        int[] bench = Enumerable.Range(0, 128).Select(i => 1000 + (i * 37) % 5000).ToArray();
        using (var kvb = model.CreateKvCache(256))
        {
            using (model.Forward(bench.AsSpan(0, 8), Enumerable.Range(0, 8).ToArray(), -1, kvb)) { }
        }
        using (var kvb = model.CreateKvCache(256))
        {
            var sw = Stopwatch.StartNew();
            using (model.Forward(bench, Enumerable.Range(0, 128).ToArray(), -1, kvb)) { }
            double pre = sw.Elapsed.TotalSeconds;
            sw.Restart();
            for (int i = 0; i < 16; i++)
                using (model.Forward([1234 + i], [128 + i], -1, kvb)) { }
            double dec = sw.Elapsed.TotalSeconds;
            _output.WriteLine($"CUDA throughput: prefill {128 / pre:F0} tok/s ({pre * 1000:F0} ms/128 tok), decode {16 / dec:F2} tok/s ({dec / 16 * 1000:F0} ms/tok)");
        }

        Assert.True(tfAgree >= tfSteps * 0.95, $"teacher-forced top-1 agreement {tfAgree}/{tfSteps} < 95%");
    }

    private static unsafe float[] Row(ITensor logits, int vocab)
    {
        using (logits)
            return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
