using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end parity tests for the Vulkan forward pass against the CPU
/// reference on real-weight GGUF checkpoints. Sister class to
/// <see cref="RealHfSafetensorsEndToEndVulkanTests"/>: the SafeTensors path
/// upcasts source weights to F32 at upload (memory-doubling), while the
/// GGUF path keeps the source quant format on device thanks to the K-quant
/// kernels added in Phase 1 (Q4_K_M, Q5_K_M, Q6_K_M) plus the existing
/// Q8_0 path. This makes large quantised models — most importantly
/// DeepSeek-V2-Lite — runnable on memory-constrained Vulkan hosts where
/// the SafeTensors BF16→F32 expansion would OOM.
/// </summary>
/// <remarks>
/// <para>
/// Each test self-skips when its GGUF is not present at the conventional
/// path (or env-var override). Conventional paths follow the same
/// <c>~/.dotllm/test-cache/&lt;org&gt;/&lt;repo&gt;/&lt;file&gt;</c> layout used by the
/// existing CPU GGUF tests. The Q4_K_M / Q5_K_M / Q6_K_M / Q8_0 kernels are
/// production-shippable per the Phase 1 commits (afb2272, 15099b9,
/// 29a1459 and their wiring siblings).
/// </para>
/// <para>
/// Tolerances mirror the SmolLM-135M Q8_0 parity test
/// (<see cref="VulkanTransformerModelTests"/>): L∞ ≤ 3.0 absolute,
/// top-K=10 jaccard ≥ 0.5, ≥ 5/9 strict argmax matches across prefill +
/// 8 decode steps. The CPU path computes Q-format × Q8_1-quantised
/// activations on the way through projections; the Vulkan path keeps the
/// weights in source quant on device but runs F32 activations — so the
/// two paths agree on the model but differ slightly on per-projection
/// arithmetic. The argmax floor is the load-bearing assertion.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufVulkanParityTests
{
    private const float LogitsAbsTol = 3.0f;
    private const int TopKForJaccard = 10;
    private const float TopKJaccardFloor = 0.5f;
    private const int DecodeStepsToCheck = 8;
    private const int StrictArgmaxFloor = 5; // out of prefill + 8 decode = 9 steps

    private readonly ITestOutputHelper _output;

    public RealGgufVulkanParityTests(ITestOutputHelper output) => _output = output;

    // ────────────────────────────────────────────────────────────────────
    // Llama-3.2-1B Q8_0 (dense Llama, exercises Q8_0 path on a real model
    // larger than SmolLM-135M)
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Llama32_1B_Q8_0_VulkanForward_MatchesCpuReference()
    {
        string? path = ResolveGgufPath(
            envVar: "DOTLLM_LLAMA32_1B_Q8_0_GGUF",
            conventional: "C:/Users/james/.dotllm/test-cache/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        if (path is null)
        {
            _output.WriteLine("[SKIP] Llama-3.2-1B Q8_0 GGUF not found.");
            return;
        }
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Llama-3.2-1B-Q8_0",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bielik-1.5B Q4_K_M (dense Llama, exercises Q4_K_M path)
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Bielik15B_Q4_K_M_VulkanForward_MatchesCpuReference()
    {
        string? path = ResolveGgufPath(
            envVar: "DOTLLM_BIELIK_15B_Q4_K_M_GGUF",
            conventional: "C:/Users/james/.dotllm/test-cache/second-state/Bielik-1.5B-v3.0-Instruct-GGUF/Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf");
        if (path is null)
        {
            _output.WriteLine("[SKIP] Bielik-1.5B Q4_K_M GGUF not found.");
            return;
        }
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Bielik-1.5B-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // DeepSeek-V2-Lite Q4_K_M (MLA + MoE, ~10.4 GB GGUF)
    //
    // The SafeTensors variant cannot run on memory-constrained hosts because
    // the BF16→F32 expansion balloons to 58.5 GB (RealHfSafetensorsEndToEndVulkanTests
    // self-skips it). The Q4_K_M GGUF stays in source quant on device thanks
    // to Phase 1 K-quant Vulkan kernels — total VRAM footprint ≈ 10.4 GB.
    // Production-relevance: most DeepSeek-V2 deployments ship K-quants.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void DeepSeekV2Lite_Q4_K_M_VulkanForward_MatchesCpuReference()
    {
        string? path = ResolveGgufPath(
            envVar: "DOTLLM_DEEPSEEK_V2_LITE_Q4_K_M_GGUF",
            conventional: "C:/Users/james/.dotllm/test-cache/mradermacher/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite.Q4_K_M.gguf");
        if (path is null)
        {
            _output.WriteLine(
                "[SKIP] DeepSeek-V2-Lite Q4_K_M GGUF not found. Set "
                + "DOTLLM_DEEPSEEK_V2_LITE_Q4_K_M_GGUF or download to "
                + "~/.dotllm/test-cache/mradermacher/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite.Q4_K_M.gguf "
                + "(~10.4 GB) from huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF.");
            return;
        }
        RunGgufParityTest(path, expectedArch: Architecture.DeepSeekV2, label: "DeepSeek-V2-Lite-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ════════════════════════════════════════════════════════════════════
    // Driver
    // ════════════════════════════════════════════════════════════════════

    private void RunGgufParityTest(string path, Architecture expectedArch, string label, string prompt)
    {
        SkipIfVulkanUnavailable(out string spvDir);

        _output.WriteLine($"[{label}] gguf: {path}");

        // Both backends mmap the same GGUF file; weights are then either
        // dequantised (CPU Q8_0 path runs Q-format arithmetic; CPU K-quant
        // path dequantises to F32) or uploaded raw to device (Vulkan keeps
        // K-quant / Q8_0 in source bytes thanks to Phase 1).
        using var cpuGguf = GgufFile.Open(path);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        Assert.Equal(expectedArch, cpuConfig.Architecture);

        // The Vulkan backend rejects MlaConfig.UseLatentCache /
        // UseHybridMlaCache (CPU-only Phase B / Phase C). HF's DeepSeek
        // extractor defaults UseHybridMlaCache=true; strip those flags for
        // the Vulkan-side config.
        ModelConfig vkConfig = NormalizeForVulkan(cpuConfig);

        var cpuLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        using var cpuModel = TransformerModel.LoadFromGguf(cpuGguf, cpuConfig);
        cpuLoadWatch.Stop();
        _output.WriteLine(
            $"[{label}] CPU load ({cpuLoadWatch.Elapsed.TotalSeconds:F1} s): "
            + $"vocab={cpuConfig.VocabSize} hidden={cpuConfig.HiddenSize} layers={cpuConfig.NumLayers}");

        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var vkGguf = GgufFile.Open(path);
        var vkLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        VulkanTransformerModel? vkModel = null;
        try
        {
            vkModel = VulkanTransformerModel.LoadFromGguf(vkGguf, vkConfig, spvDir);
        }
        catch (DotLLM.Vulkan.Interop.VulkanException ex) when (
            ex.ErrorCode == -1 || ex.ErrorCode == -2 || ex.ErrorCode == -5)
        {
            Skip.If(true,
                $"[{label}] Vulkan load failed with {ex.Message}. Source-quant "
                + "weights still exceeded available device-local memory on this "
                + "host. Re-run on a host with more VRAM.");
        }
        vkLoadWatch.Stop();
        _output.WriteLine($"[{label}] Vulkan load ({vkLoadWatch.Elapsed.TotalSeconds:F1} s)");

        try
        {
            int[] promptIds = tokenizer.Encode(prompt).ToArray();
            Assert.NotEmpty(promptIds);
            _output.WriteLine($"[{label}] prompt: '{prompt}' -> {promptIds.Length} tokens [{string.Join(',', promptIds)}]");

            int strictArgmaxMatches = 0;
            int stepsChecked = 0;
            int vocab = cpuConfig.VocabSize;

            // Growing-context reprefill loop (no KV cache) — same pattern as
            // the SafeTensors sibling tests. Sidesteps any cache-mode
            // divergence between CPU (Phase C latent default for DeepSeek-V2)
            // and Vulkan (Phase A expanded only).
            var tokens = new List<int>(promptIds.Length + DecodeStepsToCheck);
            tokens.AddRange(promptIds);

            for (int step = 0; step <= DecodeStepsToCheck; step++)
            {
                int[] tokenIds = tokens.ToArray();
                int[] positions = new int[tokenIds.Length];
                for (int i = 0; i < positions.Length; i++) positions[i] = i;

                float[] cpuLogits = RunForwardCpuLastRow(cpuModel, tokenIds, positions, vocab);
                float[] vkLogits = RunForwardVulkanLastRow(vkModel!, tokenIds, positions, vocab);

                AssertLogitsMatch(cpuLogits, vkLogits, step, label);
                int cpuArgmax = Argmax(cpuLogits);
                int vkArgmax = Argmax(vkLogits);
                bool argmaxMatch = cpuArgmax == vkArgmax;
                if (argmaxMatch) strictArgmaxMatches++;
                stepsChecked++;

                _output.WriteLine(
                    $"[{label}] step {step}: cpu_argmax={cpuArgmax} vk_argmax={vkArgmax}{(argmaxMatch ? " [match]" : " [diff]")}");

                tokens.Add(cpuArgmax);
            }

            Assert.True(strictArgmaxMatches >= StrictArgmaxFloor,
                $"[{label}] strict argmax match floor {StrictArgmaxFloor}/{stepsChecked} not met: "
                + $"got {strictArgmaxMatches}/{stepsChecked}.");
            _output.WriteLine($"[{label}] strict argmax matches: {strictArgmaxMatches}/{stepsChecked}");
        }
        finally
        {
            vkModel?.Dispose();
            vkGguf.Dispose();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════

    private static ModelConfig NormalizeForVulkan(ModelConfig cpuConfig)
    {
        if (cpuConfig.MlaConfig is null) return cpuConfig;
        if (!cpuConfig.MlaConfig.UseLatentCache && !cpuConfig.MlaConfig.UseHybridMlaCache)
            return cpuConfig;
        var mla = cpuConfig.MlaConfig with
        {
            UseLatentCache = false,
            UseHybridMlaCache = false,
        };
        return cpuConfig with { MlaConfig = mla };
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
        // Matches the resolution logic in VulkanTransformerModelTests / the
        // SafeTensors sibling. SPV blobs ship next to the runtime DLL.
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");
    }

    private static bool IsVulkanRuntimeAvailable()
    {
        try
        {
            using var d = VulkanDevice.Create();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string ResolveSpvDir()
    {
        // The repo ships SPV blobs at native/vulkan/spv/ relative to the
        // repo root. Tests run from bin/Debug/net10.0/, so walk up to the
        // repo root.
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null!;
    }

    private static unsafe float[] RunForwardCpuLastRow(IModel model, int[] tokenIds, int[] positions, int vocab)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static unsafe float[] RunForwardVulkanLastRow(VulkanTransformerModel model, int[] tokenIds, int[] positions, int vocab)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private void AssertLogitsMatch(float[] cpuLogits, float[] vkLogits, int step, string label)
    {
        Assert.Equal(cpuLogits.Length, vkLogits.Length);
        float maxAbs = 0f;
        int worstIdx = 0;
        for (int i = 0; i < cpuLogits.Length; i++)
        {
            float diff = Math.Abs(cpuLogits[i] - vkLogits[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }
        float jaccard = TopKJaccard(cpuLogits, vkLogits, TopKForJaccard);
        _output.WriteLine($"[{label}] step {step}: L∞={maxAbs:F4} (idx {worstIdx}); top-{TopKForJaccard} jaccard={jaccard:F2}");
        Assert.True(maxAbs <= LogitsAbsTol,
            $"[{label}] step {step}: L∞ {maxAbs:F4} exceeds {LogitsAbsTol:F2}.");
        Assert.True(jaccard >= TopKJaccardFloor,
            $"[{label}] step {step}: top-{TopKForJaccard} jaccard {jaccard:F2} below floor {TopKJaccardFloor:F2}.");
    }

    private static float TopKJaccard(float[] a, float[] b, int k)
    {
        var aTop = TopKIndices(a, k);
        var bTop = TopKIndices(b, k);
        int inter = 0;
        foreach (int i in aTop) if (bTop.Contains(i)) inter++;
        return inter / (float)k;
    }

    private static int[] TopKIndices(float[] xs, int k)
    {
        var idx = new int[xs.Length];
        for (int i = 0; i < xs.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => xs[b].CompareTo(xs[a]));
        return idx.Take(k).ToArray();
    }

    private static int Argmax(float[] xs)
    {
        int best = 0; float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }
}
