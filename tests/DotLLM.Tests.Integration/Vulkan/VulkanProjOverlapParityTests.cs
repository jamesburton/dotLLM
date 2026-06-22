using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Op-swap bit-exact parity for the projection-overlap decode PoC (issue #336,
/// <c>VULKAN-THROUGHPUT-PLAN.md</c> §7). The SAME loaded model is decoded with the overlap
/// ON vs OFF (toggled via <see cref="VulkanTransformerModel.EnableProjOverlapEnvVar"/> between
/// forwards, which the forward re-reads each layer) and must produce <b>bit-exact</b> logits
/// and an identical argmax at every step.
/// </summary>
/// <remarks>
/// The overlap path quantizes the shared activation (NormOutput) ONCE and issues the K and V
/// MMVQ GEMVs back-to-back with no barrier between them — both read the immutable shared Q8_1
/// scratch (RAR) and write disjoint outputs, so each output row is computed by the same kernel
/// inner loop over the same bytes as the serial per-op path. The numerics must be unchanged
/// (L∞ == 0). This discriminates a wrong shared-scratch wiring or a missed hazard, which a
/// loose-tolerance CPU-vs-Vulkan test would not catch. Per <c>debug-gpu-divergence-op-swap</c>:
/// swap exactly one thing (the overlap) against a known-good baseline.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanProjOverlapParityTests
{
    private const int DecodeSteps = 8;

    private readonly ITestOutputHelper _output;
    public VulkanProjOverlapParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void SmolLM135M_Q8_0_ProjOverlap_MatchesSerial_BitExact()
    {
        string? path = ResolveGgufPath("DOTLLM_SMOLLM_Q8_GGUF",
            "C:/Users/james/.dotllm/models/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf");
        Skip.If(path is null, "SmolLM-135M Q8_0 GGUF not found (set DOTLLM_SMOLLM_Q8_GGUF).");
        RunParity(path!, Architecture.Llama, "SmolLM-135M-Q8_0", "The capital of France is");
    }

    [SkippableFact]
    public void Llama32_3B_Q8_0_ProjOverlap_MatchesSerial_BitExact()
    {
        string? path = ResolveGgufPath("DOTLLM_LLAMA32_3B_Q8_0_GGUF",
            "C:/Users/james/.dotllm/test-cache/bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf");
        Skip.If(path is null, "Llama-3.2-3B Q8_0 GGUF not found (set DOTLLM_LLAMA32_3B_Q8_0_GGUF).");
        RunParity(path!, Architecture.Llama, "Llama-3.2-3B-Q8_0", "The capital of France is");
    }

    [SkippableFact]
    public void Llama31_8B_Q4_K_M_ProjOverlap_MatchesSerial_BitExact()
    {
        string? path = ResolveGgufPath("DOTLLM_LLAMA31_8B_Q4_K_M_GGUF",
            "C:/Development/gguf-cache/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
        Skip.If(path is null, "Llama-3.1-8B Q4_K_M GGUF not found (set DOTLLM_LLAMA31_8B_Q4_K_M_GGUF).");
        RunParity(path!, Architecture.Llama, "Llama-3.1-8B-Q4_K_M", "The capital of France is");
    }

    private void RunParity(string path, Architecture expectedArch, string label, string prompt)
    {
        SkipIfVulkanUnavailable(out string spvDir);
        _output.WriteLine($"[{label}] gguf: {path}");

        string? prev = Environment.GetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar);
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(expectedArch, config.Architecture);
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

            int vocab = config.VocabSize;
            int[] promptIds = tokenizer.Encode(prompt).ToArray();
            Assert.NotEmpty(promptIds);
            _output.WriteLine($"[{label}] prompt -> {promptIds.Length} tokens");

            // Two lockstep KV caches so each decode step (seqLen == 1, where overlap fires)
            // runs from identical cache state.
            using var cacheSerial = model.CreateKvCache(maxSeqLen: promptIds.Length + DecodeSteps + 2);
            using var cacheOverlap = model.CreateKvCache(maxSeqLen: promptIds.Length + DecodeSteps + 2);

            int[] prefillPos = new int[promptIds.Length];
            for (int i = 0; i < prefillPos.Length; i++) prefillPos[i] = i;

            Environment.SetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar, null); // serial
            int nextToken = ArgmaxForward(model, promptIds, prefillPos, vocab, cacheSerial);
            Environment.SetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar, "1"); // overlap
            int nextTokenOv = ArgmaxForward(model, promptIds, prefillPos, vocab, cacheOverlap);
            Assert.Equal(nextToken, nextTokenOv); // prefill (seqLen>1) is identical — overlap is decode-only

            int nextPos = promptIds.Length;
            for (int step = 0; step < DecodeSteps; step++)
            {
                int[] single = { nextToken };
                int[] pos = { nextPos };

                Environment.SetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar, null);
                float[] serial = ForwardLastRow(model, single, pos, vocab, cacheSerial);

                Environment.SetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar, "1");
                float[] overlap = ForwardLastRow(model, single, pos, vocab, cacheOverlap);

                float maxAbs = 0f;
                int worst = 0;
                for (int i = 0; i < vocab; i++)
                {
                    float d = Math.Abs(serial[i] - overlap[i]);
                    if (d > maxAbs) { maxAbs = d; worst = i; }
                }
                int serialArg = Argmax(serial), overlapArg = Argmax(overlap);
                _output.WriteLine($"[{label}] decode {step}: L∞={maxAbs:G6} (idx {worst}) serialArg={serialArg} overlapArg={overlapArg}");

                Assert.Equal(serialArg, overlapArg);
                Assert.True(maxAbs == 0f,
                    $"[{label}] decode {step}: overlap not bit-exact vs serial (L∞={maxAbs:G6} at vocab idx {worst}).");

                nextToken = overlapArg;
                nextPos++;
            }
            _output.WriteLine($"[{label}] all {DecodeSteps} decode steps bit-exact.");
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.EnableProjOverlapEnvVar, prev);
        }
    }

    private static unsafe float[] ForwardLastRow(
        VulkanTransformerModel model, int[] tokenIds, int[] positions, int vocab, IKvCache cache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, cache);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static int ArgmaxForward(
        VulkanTransformerModel model, int[] tokenIds, int[] positions, int vocab, IKvCache cache)
        => Argmax(ForwardLastRow(model, tokenIds, positions, vocab, cache));

    private static int Argmax(float[] xs)
    {
        int best = 0; float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }

    private static string? ResolveGgufPath(string envVar, string conventional)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
        if (File.Exists(conventional)) return conventional;
        return null;
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");
    }

    private static bool IsVulkanRuntimeAvailable()
    {
        try { using var d = VulkanDevice.Create(); return true; }
        catch { return false; }
    }

    private static string ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null!;
    }
}
