using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using System.Diagnostics;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-weight KV-cached autoregressive DECODE of the Gemma-4-26B-A4B GGUF on the
/// Vulkan backend with routed experts kept QUANTIZED on device (fused gate_up Q4_K +
/// down Q5_1/Q8_0 indexed-MoE shaders — issue #137). This is the throughput-facing
/// complement to <c>Gemma4GgufForwardTests.Gemma4_26B_Vulkan_QuantizedExperts_LoadsAndForwards</c>
/// (which validates a single cacheless forward): prefill a short factual prompt, then
/// greedy-decode token-by-token through the per-layer-strided <see cref="VulkanKvCache"/>,
/// asserting a coherent completion and reporting per-step latency (min/mean ms → tok/s).
/// </summary>
/// <remarks>
/// <para>Gated on <c>DOTLLM_GEMMA4_GGUF</c> (path to the ~15.7 GB
/// <c>gemma-4-26B-A4B-it-UD-Q4_K_M.gguf</c>) + a Vulkan device; self-skips otherwise.
/// Decode step count via <c>DOTLLM_GEMMA4_DECODE_STEPS</c> (default 32).</para>
/// <para><b>UMA measurement rule (Strix Halo):</b> absolute decode tok/s swings with
/// host memory-bandwidth contention; only compare numbers from back-to-back runs in the
/// same session and prefer <c>decode_min_ms</c> as the contention-free indicator.</para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class Gemma4VulkanRealDecodeTests
{
    /// <summary>
    /// Gemma-4-26B fixture, resolved via <see cref="KnownTestFixtures.Gemma4_26B_A4B_Q4KM"/>:
    /// <c>$DOTLLM_GEMMA4_GGUF</c>, then the dotLLM test cache, then the HF hub cache (#308).
    /// </summary>
    private static FixtureLocation Gemma4Fixture => KnownTestFixtures.Gemma4_26B_A4B_Q4KM;

    private readonly ITestOutputHelper _output;

    public Gemma4VulkanRealDecodeTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void Gemma4_26B_Vulkan_KvCachedDecode_GeneratesCoherentText()
    {
        FixtureLocation fixture = Gemma4Fixture;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.Gemma4_26BDescription));
        string path = fixture.Path!;
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        int decodeSteps = int.TryParse(
            Environment.GetEnvironmentVariable("DOTLLM_GEMMA4_DECODE_STEPS"), out int n) && n > 0 ? n : 32;
        string spvDir = ResolveSpvDir();

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Gemma4, config.Architecture);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        _output.WriteLine($"[gemma4 26B Vulkan decode] {path}");
        _output.WriteLine($"  load_s={loadSw.Elapsed.TotalSeconds:F2} (experts quantized-resident on device)");

        // Completion-shaped factual prompt (instruct model, see Gemma4GgufForwardTests).
        const string prompt = "The Eiffel Tower is located in";
        int[] encoded = tokenizer.Encode(prompt);
        int[] promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);
        int promptLen = promptIds.Length;
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) positions[i] = i;

        int vocab = config.VocabSize;
        using var kv = model.CreateKvCache(maxSeqLen: promptLen + decodeSteps + 4);

        // ── Prefill ────────────────────────────────────────────────────────
        var prefillSw = Stopwatch.StartNew();
        int next;
        using (ITensor logits = model.Forward(promptIds, positions, deviceId: -1, kvCache: kv))
        {
            prefillSw.Stop();
            next = ArgMaxLastRow(logits, vocab);
        }
        Assert.Equal(promptLen, kv.CurrentLength);
        _output.WriteLine($"  prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F1} ({promptLen} tokens)");

        // ── Greedy decode loop ─────────────────────────────────────────────
        var generated = new List<int> { next };
        var stepMs = new double[decodeSteps];
        Span<int> oneId = stackalloc int[1];
        Span<int> onePos = stackalloc int[1];
        var stepSw = new Stopwatch();
        for (int s = 0; s < decodeSteps; s++)
        {
            oneId[0] = next;
            onePos[0] = promptLen + s;
            stepSw.Restart();
            using ITensor logits = model.Forward(oneId, onePos, deviceId: -1, kvCache: kv);
            next = ArgMaxLastRow(logits, vocab);
            stepSw.Stop();
            stepMs[s] = stepSw.Elapsed.TotalMilliseconds;
            if (next == tokenizer.EosTokenId) break;
            generated.Add(next);
        }

        string text = tokenizer.Decode(generated.ToArray());
        int measured = generated.Count - 1 >= 1 ? Math.Min(decodeSteps, generated.Count) : 1;
        double minMs = double.MaxValue, sumMs = 0;
        int counted = 0;
        // Skip step 0 (first decode carries one-time descriptor/pipeline warm-up).
        for (int s = 1; s < measured; s++)
        {
            if (stepMs[s] <= 0) continue;
            minMs = Math.Min(minMs, stepMs[s]);
            sumMs += stepMs[s];
            counted++;
        }
        double meanMs = counted > 0 ? sumMs / counted : double.NaN;
        _output.WriteLine($"  decode_steps={generated.Count}  decode_min_ms={minMs:F2}  decode_mean_ms={meanMs:F2}");
        _output.WriteLine($"  tok_s_from_min={1000.0 / minMs:F2}  tok_s_from_mean={1000.0 / meanMs:F2}");
        _output.WriteLine($"  text: '{prompt}{text}'");

        // ── Assertions ─────────────────────────────────────────────────────
        // Coherence gate: the factual completion must name Paris (greedy, wide margin
        // — validated on CPU and on the cacheless Vulkan forward for this checkpoint).
        Assert.Contains("Paris", text, StringComparison.OrdinalIgnoreCase);
        // Non-degenerate: the loop must not emit one token forever.
        Assert.True(generated.Distinct().Count() >= 3,
            $"Degenerate decode — {generated.Count} tokens but only {generated.Distinct().Count()} distinct: '{text}'");
    }

    private static unsafe int ArgMaxLastRow(ITensor logits, int vocab)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        float* p = (float*)logits.DataPointer + (total - vocab);
        int best = 0;
        float bestV = p[0];
        for (int v = 1; v < vocab; v++) if (p[v] > bestV) { bestV = p[v]; best = v; }
        return best;
    }

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
