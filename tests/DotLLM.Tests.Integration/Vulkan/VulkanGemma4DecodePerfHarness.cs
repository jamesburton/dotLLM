using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Non-asserting throughput harness for the autoregressive gemma4 Vulkan forward.
/// Quantifies the per-layer-strided KV-cache win by timing token-by-token decode
/// TWO ways on the same warmed model:
///   (a) CACHED — prefill once, then decode one token at a time reading the cache;
///   (b) CACHELESS RECOMPUTE — re-run the whole growing prompt every step (the
///       pre-cache behaviour, O(seq) per token).
/// Gated by <c>DOTLLM_VULKAN_PERF=1</c> so it never adds to the default sweep.
/// Uses the portable synthetic gemma4 <c>Bench</c> fixture — no checkpoint needed.
/// </summary>
[Trait("Category", "GPU")]
public sealed class VulkanGemma4DecodePerfHarness
{
    private readonly ITestOutputHelper _output;

    public VulkanGemma4DecodePerfHarness(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void MeasureGemma4DecodeLatency()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF") == "1",
            "DOTLLM_VULKAN_PERF=1 not set.");
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        int decodeSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_DECODE_STEPS", 24);
        int warmupSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_WARMUP", 4);
        int promptLen = ParseEnvInt("DOTLLM_VULKAN_PERF_PROMPT", 8);

        // Bench preset: 8 layers, hidden 1024, 16 experts top-4, dual head dim
        // (sliding 64 / global 128) — compute-bound enough to be representative.
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_perf_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, SyntheticGemma4Gguf.Bench);

        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            int vocab = cfg.VocabSize;

            var loadSw = Stopwatch.StartNew();
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
            loadSw.Stop();
            _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

            // Deterministic prompt within the sliding window so the cached and
            // cacheless paths see the same attention window each step.
            var prompt = new int[promptLen];
            prompt[0] = 2; // BOS
            for (int i = 1; i < promptLen; i++) prompt[i] = 3 + (i % (vocab - 8));
            var promptPos = new int[promptLen];
            for (int i = 0; i < promptLen; i++) promptPos[i] = i;

            // ── (a) CACHED decode ───────────────────────────────────────
            using var cache = model.CreateKvCache(maxSeqLen: promptLen + warmupSteps + decodeSteps + 8);
            int next;
            var prefillSw = Stopwatch.StartNew();
            using (var logits = model.Forward(prompt, promptPos, -1, cache))
            { prefillSw.Stop(); next = Argmax(logits, vocab); }
            _output.WriteLine($"cached_prefill_len={promptLen} cached_prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F2}");

            int pos = promptLen;
            for (int i = 0; i < warmupSteps; i++)
            {
                using var logits = model.Forward(new[] { next }, new[] { pos }, -1, cache);
                next = Argmax(logits, vocab); pos++;
            }
            double cachedTotal = 0, cachedMin = double.PositiveInfinity;
            for (int i = 0; i < decodeSteps; i++)
            {
                var sw = Stopwatch.StartNew();
                using (var logits = model.Forward(new[] { next }, new[] { pos }, -1, cache))
                { sw.Stop(); next = Argmax(logits, vocab); }
                pos++;
                double ms = sw.Elapsed.TotalMilliseconds;
                cachedTotal += ms; if (ms < cachedMin) cachedMin = ms;
            }
            double cachedAvg = cachedTotal / decodeSteps;

            // ── (b) CACHELESS RECOMPUTE decode (pre-cache behaviour) ─────
            var growing = new List<int>(prompt);
            var growingPos = new List<int>(promptPos);
            // advance to the same starting length the cached loop measured from
            for (int i = 0; i < warmupSteps; i++) { growing.Add(3); growingPos.Add(growing.Count - 1); }
            double reTotal = 0, reMin = double.PositiveInfinity;
            for (int i = 0; i < decodeSteps; i++)
            {
                growing.Add(3); growingPos.Add(growing.Count - 1);
                int[] ids = growing.ToArray(); int[] ps = growingPos.ToArray();
                var sw = Stopwatch.StartNew();
                using (var _ = model.Forward(ids, ps, -1, kvCache: null)) { sw.Stop(); }
                double ms = sw.Elapsed.TotalMilliseconds;
                reTotal += ms; if (ms < reMin) reMin = ms;
            }
            double reAvg = reTotal / decodeSteps;

            _output.WriteLine("=== gemma4 decode (Bench: 8L, hidden 1024, 16e/top4) ===");
            _output.WriteLine($"cached_decode_avg_ms={cachedAvg:F2} min={cachedMin:F2} tok_per_sec={1000.0 / cachedAvg:F1}");
            _output.WriteLine($"cacheless_recompute_avg_ms={reAvg:F2} min={reMin:F2} tok_per_sec={1000.0 / reAvg:F1} (final seq {growing.Count})");
            _output.WriteLine($"cache_speedup={reAvg / cachedAvg:F2}x over {decodeSteps} steps");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private static unsafe int Argmax(ITensor logits, int vocab)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
        int baseIdx = total - vocab, idx = 0; float best = span[baseIdx];
        for (int i = 1; i < vocab; i++) if (span[baseIdx + i] > best) { best = span[baseIdx + i]; idx = i; }
        return idx;
    }

    private static int ParseEnvInt(string key, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(key), out int n) && n > 0 ? n : fallback;

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
