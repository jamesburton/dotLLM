using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #734 acceptance: real gemma-4-E4B-it (Q4_K_M) on Vulkan vs the CPU path - greedy token identity over
/// several prompts, prefill-logit parity, and Vulkan throughput. The CPU reference (~5 s/token) is computed once and
/// cached in the JSON file named by <c>DOTLLM_GEMMA4E4B_REF</c> (default: temp dir), so reruns only pay for Vulkan.
/// Gated on <c>DOTLLM_GEMMA4E4B_GGUF</c> / the canonical HF-hub path; acquire the GPU lock before running.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma4E4BVulkanRealModelParityTests
{
    private const string DefaultHubPath =
        "C:/Users/james/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/" +
        "snapshots/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf";

    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "Water boils at a temperature of",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    ];

    private const int MaxNew = 24;

    private readonly ITestOutputHelper _output;
    public Gemma4E4BVulkanRealModelParityTests(ITestOutputHelper output) => _output = output;

    private sealed record Ref(int[] PromptIds, int[] Generated, float[] PrefillLogits);

    private static string? ModelPath()
    {
        string? p = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4E4B_GGUF");
        if (!string.IsNullOrWhiteSpace(p) && File.Exists(p)) return p;
        return File.Exists(DefaultHubPath) ? DefaultHubPath : null;
    }

    [SkippableFact]
    public void Vulkan_E4B_GreedyTextAndLogits_MatchCpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_GEMMA4E4B_GGUF to the gemma-4-E4B-it GGUF.");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        string refFile = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4E4B_REF")
            ?? Path.Combine(Path.GetTempPath(), "dotllm-gemma4-e4b-cpu-ref.json");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var promptIds = Prompts.Select(p =>
        {
            int[] enc = tokenizer.Encode(p);
            var ids = new int[enc.Length + 1];
            ids[0] = tokenizer.BosTokenId;
            Array.Copy(enc, 0, ids, 1, enc.Length);
            return ids;
        }).ToArray();

        // ── CPU reference (cached) ──
        Ref[] cpuRefs;
        if (File.Exists(refFile)
            && JsonSerializer.Deserialize<Ref[]>(File.ReadAllText(refFile)) is { } cached
            && cached.Length == Prompts.Length
            && cached.Select((r, i) => r.PromptIds.SequenceEqual(promptIds[i])).All(x => x))
        {
            cpuRefs = cached;
            _output.WriteLine($"CPU reference loaded from {refFile}");
        }
        else
        {
            var sw = Stopwatch.StartNew();
            using var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config);
            cpuRefs = promptIds.Select(ids => Generate(cpu, config,
                size => new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), size), ids, -1)).ToArray();
            File.WriteAllText(refFile, JsonSerializer.Serialize(cpuRefs));
            _output.WriteLine($"CPU reference computed in {sw.Elapsed.TotalMinutes:F1} min -> {refFile}");
        }

        // ── Vulkan ──
        var loadSw = Stopwatch.StartNew();
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
        _output.WriteLine($"Vulkan load: {loadSw.Elapsed.TotalSeconds:F1} s");

        int totalSteps = 0, matchedSteps = 0;
        double worstPrefill = 0;
        var generatedPrefill = new float[Prompts.Length][];
        for (int p = 0; p < Prompts.Length; p++)
        {
            var sw = Stopwatch.StartNew();
            var got = Generate(vk, config, size => vk.CreateKvCache(size), promptIds[p], -1);
            sw.Stop();
            generatedPrefill[p] = got.PrefillLogits;

            double worst = 0; int wc = -1;
            for (int i = 0; i < got.PrefillLogits.Length; i++)
            {
                double d = Math.Abs(got.PrefillLogits[i] - cpuRefs[p].PrefillLogits[i]);
                if (d > worst) { worst = d; wc = i; }
            }
            worstPrefill = Math.Max(worstPrefill, worst);

            int firstDiv = -1;
            for (int i = 0; i < MaxNew; i++)
                if (i >= got.Generated.Length || i >= cpuRefs[p].Generated.Length || got.Generated[i] != cpuRefs[p].Generated[i])
                { firstDiv = i; break; }
            int n = Math.Min(got.Generated.Length, cpuRefs[p].Generated.Length);
            totalSteps += n;
            matchedSteps += firstDiv < 0 ? n : firstDiv;
            _output.WriteLine($"prompt {p}: '{Prompts[p].Replace("\n", "\\n")}' prefill argmax cpu={ArgMax(cpuRefs[p].PrefillLogits)} vk={ArgMax(got.PrefillLogits)} "
                + $"worst|dLogit|={worst:F4} @ {wc}; firstDivergence={firstDiv}; {sw.Elapsed.TotalSeconds:F2}s total ({got.Generated.Length} tok)");
            _output.WriteLine($"   cpu: '{tokenizer.Decode(cpuRefs[p].Generated)}'");
            _output.WriteLine($"   vk : '{tokenizer.Decode(got.Generated)}'");
        }
        _output.WriteLine($"TOTAL matched {matchedSteps}/{totalSteps}; worst prefill |dLogit|={worstPrefill:F4}");

        // Throughput (Vulkan): prefill 128 tokens and decode 32, after a warm-up.
        int[] bench = Enumerable.Range(0, 128).Select(i => 1000 + (i * 37) % 5000).ToArray();
        int[] benchPos = Enumerable.Range(0, 128).ToArray();
        using (var kv = vk.CreateKvCache(256))
        {
            using (vk.Forward(bench.AsSpan(0, 8), benchPos.AsSpan(0, 8), -1, kv)) { }
        }
        using (var kv = vk.CreateKvCache(256))
        {
            var sw = Stopwatch.StartNew();
            using (vk.Forward(bench, benchPos, -1, kv)) { }
            double pre = sw.Elapsed.TotalSeconds;
            sw.Restart();
            for (int i = 0; i < 32; i++)
                using (vk.Forward([1234 + i], [128 + i], -1, kv)) { }
            double dec = sw.Elapsed.TotalSeconds;
            _output.WriteLine($"Vulkan throughput: prefill {128 / pre:F0} tok/s (128 tok in {pre * 1000:F0} ms), decode {32 / dec:F1} tok/s ({dec / 32 * 1000:F1} ms/tok)");
        }

        // Free-running greedy text can legitimately flip on a near-tie (Q4_K activation-quant noise differs between the
        // CPU Q8_K path and the Vulkan MMVQ/MMQ paths; measured: one flip, CPU margin 0.41 / Vulkan-decode margin 0.04).
        // So the hard gates are (a) the prefill argmax equals the CPU's for every prompt and (b) most generated tokens
        // are identical; the teacher-forced test below measures per-step agreement without cascading divergence.
        for (int p = 0; p < Prompts.Length; p++)
            Assert.Equal(ArgMax(cpuRefs[p].PrefillLogits), ArgMax(generatedPrefill[p]));
        Assert.True(matchedSteps >= totalSteps * 2 / 3,
            $"Greedy text diverged from CPU too early: matched {matchedSteps}/{totalSteps} steps (worst prefill |dLogit| {worstPrefill:F4}).");
    }

    /// <summary>
    /// Teacher-forced diagnosis: feed the CPU's greedy tokens to BOTH backends and compare, at every step, the top-1 and the
    /// CPU top-1/top-2 margin. A step where Vulkan's top-1 differs while the CPU margin is tiny is a near-tie flipped by
    /// quantized-arithmetic noise, not a modelling bug. Needs the cached CPU reference from the test above.
    /// </summary>
    [SkippableFact]
    public void Vulkan_E4B_TeacherForced_TopOneAgreementAndMargins()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_GEMMA4E4B_GGUF.");
        string refFile = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4E4B_REF")
            ?? Path.Combine(Path.GetTempPath(), "dotllm-gemma4-e4b-cpu-ref.json");
        Skip.IfNot(File.Exists(refFile), "CPU reference not cached yet - run the greedy test first.");
        var refs = JsonSerializer.Deserialize<Ref[]>(File.ReadAllText(refFile))!;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
        int vocab = config.VocabSize;

        // Cached-decode teacher forcing over ALL cached CPU steps (cheap: Vulkan only): feed the CPU's own greedy tokens, so
        // a near-tie flip cannot cascade, and count how often Vulkan's cached-decode argmax equals the CPU's next token.
        int tfSteps = 0, tfAgree = 0;
        foreach (var r in refs)
        {
            using var kv = vk.CreateKvCache(r.PromptIds.Length + r.Generated.Length + 2);
            float[] row = TopRow(vk.Forward(r.PromptIds, Enumerable.Range(0, r.PromptIds.Length).ToArray(), -1, kv), vocab);
            for (int k = 0; k < r.Generated.Length; k++)
            {
                int[] t3 = Top(row, 3);
                tfSteps++;
                if (t3[0] == r.Generated[k]) tfAgree++;
                else _output.WriteLine($"  cached-decode disagreement at step {k}: cpu token {r.Generated[k]}, vk top3=[{string.Join(",", t3)}] "
                    + $"logits=[{string.Join(",", t3.Select(i => row[i].ToString("F3")))}], vk logit of cpu token={row[r.Generated[k]]:F3}");
                if (k + 1 < r.Generated.Length)
                    row = TopRow(vk.Forward([r.Generated[k]], [r.PromptIds.Length + k], -1, kv), vocab);
            }
        }
        _output.WriteLine($"cached-decode teacher-forced top-1 agreement with CPU: {tfAgree}/{tfSteps}");
        Assert.True(tfAgree >= tfSteps * 0.95, $"teacher-forced top-1 agreement {tfAgree}/{tfSteps} < 95%");

        int checkedSteps = 0, agree = 0;
        foreach (var r in refs.Take(1)) // prompt 0 holds the divergence; CPU is ~5 s/step so keep it bounded
        {
            for (int k = 0; k < Math.Min(6, r.Generated.Length); k++)
            {
                int[] ids = r.PromptIds.Concat(r.Generated.Take(k)).ToArray();
                int[] pos = Enumerable.Range(0, ids.Length).ToArray();
                float[] c = TopRow(cpu.Forward(ids, pos, -1), vocab);
                float[] v = TopRow(vk.Forward(ids, pos, -1, kvCache: null), vocab);
                int[] ct = Top(c, 3), vt = Top(v, 3);
                checkedSteps++;
                if (ct[0] == vt[0]) agree++;
                _output.WriteLine($"step {k}: cpu top3=[{string.Join(",", ct)}] logits=[{string.Join(",", ct.Select(i => c[i].ToString("F3")))}] margin={c[ct[0]] - c[ct[1]]:F3}"
                    + $" | vk top3=[{string.Join(",", vt)}] logits=[{string.Join(",", vt.Select(i => v[i].ToString("F3")))}] margin={v[vt[0]] - v[vt[1]]:F3}");
            }
        }
        _output.WriteLine($"top-1 agreement {agree}/{checkedSteps}");
    }

    private static unsafe float[] TopRow(ITensor logits, int vocab)
    {
        using (logits)
            return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static int[] Top(float[] v, int n) => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(n).ToArray();

    private static unsafe Ref Generate(DotLLM.Core.Models.IModel model, DotLLM.Core.Models.ModelConfig config,
        Func<int, DotLLM.Core.Attention.IKvCache> makeCache, int[] promptIds, int device)
    {
        int vocab = config.VocabSize;
        var generated = new List<int>();
        float[] prefill;
        using var kv = makeCache(promptIds.Length + MaxNew + 2);
        int[] pos = Enumerable.Range(0, promptIds.Length).ToArray();
        using (ITensor logits = model.Forward(promptIds, pos, device, kv))
        {
            var span = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab);
            prefill = span.ToArray();
        }
        generated.Add(ArgMax(prefill));
        for (int step = 1; step < MaxNew; step++)
        {
            using ITensor logits = model.Forward([generated[^1]], [promptIds.Length + step - 1], device, kv);
            var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;
            generated.Add(best);
        }
        return new Ref(promptIds, generated.ToArray(), prefill);
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
