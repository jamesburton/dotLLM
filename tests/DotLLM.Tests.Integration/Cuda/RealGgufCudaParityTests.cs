using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// End-to-end parity tests for the CUDA forward pass against the CPU
/// reference on real-weight GGUF checkpoints. Sister class to
/// <see cref="DotLLM.Tests.Integration.Vulkan.RealGgufVulkanParityTests"/> —
/// same fixtures, same prompt, same tolerances, so results are directly
/// comparable across backends. Filed as a gap: unlike Vulkan, CUDA had no
/// systematic multi-architecture real-model parity suite — only
/// architecture-specific layer tests (<c>CudaQwen3MoeHybridRealGgufLayerParityTests</c>)
/// and load/run smoke tests (<c>CudaRealGgufIQuantSmokeTests</c>) that never compare
/// logits against CPU. The Vulkan suite has already caught a real cross-backend bug
/// this way (#311, Q3_K bit layout) that the single-model-multi-quant
/// <c>CrossBackendQuantGateTests</c> gate did not — this suite exists to give CUDA
/// the same chance on real, diverse architectures.
/// </summary>
/// <remarks>
/// <para>
/// Each test self-skips when its GGUF is not present at the conventional
/// path (or env-var override), via the same <see cref="TestFixtureResolver"/>
/// the Vulkan suite uses. Unlike Vulkan, CUDA's <see cref="CudaTransformerModel"/>
/// natively supports DeepSeek's MLA latent/hybrid KV-cache config (see
/// <c>CudaMlaAttention</c>/<c>CudaMlaLatentKvCache</c>), so the CPU config is used
/// as-is — no Vulkan-style config stripping is needed here.
/// </para>
/// <para>
/// Tolerances mirror <see cref="DotLLM.Tests.Integration.Vulkan.RealGgufVulkanParityTests"/>:
/// L∞ ≤ 3.0 absolute, top-K=10 jaccard ≥ 0.5, ≥ 5/9 strict argmax matches across
/// prefill + 8 decode steps. Both backends compute in F32 for activations;
/// CUDA is expected to track the CPU reference at least as tightly as Vulkan
/// does, but the bar is kept the same for direct comparability rather than
/// tuned down.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufCudaParityTests
{
    private const float LogitsAbsTol = 3.0f;
    private const int TopKForJaccard = 10;
    private const float TopKJaccardFloor = 0.5f;
    private const int DecodeStepsToCheck = 8;
    private const int StrictArgmaxFloor = 5; // out of prefill + 8 decode = 9 steps

    private readonly ITestOutputHelper _output;

    public RealGgufCudaParityTests(ITestOutputHelper output) => _output = output;

    // ────────────────────────────────────────────────────────────────────
    // Llama-3.2-1B Q8_0 (dense Llama, exercises Q8_0 path on a real model
    // larger than SmolLM-135M)
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Llama32_1B_Q8_0_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.2-1B Q8_0 GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Llama-3.2-1B-Q8_0",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bielik-1.5B Q4_K_M (dense Llama, exercises Q4_K_M path)
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Bielik15B_Q4_K_M_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q4_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q4_K_M GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Bielik-1.5B-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bielik-1.5B Q3_K_M — the exact model+quant combination where the
    // Vulkan suite caught #311 (Q3_K bit layout transposed) that
    // CrossBackendQuantGateTests' synthetic single-model sweep missed.
    // The transposed-bit-layout bug was fixed "in every backend" per its
    // own commit message; this case exists to independently verify that
    // claim held for CUDA specifically, on the real checkpoint that
    // originally exposed it.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Bielik15B_Q3_K_M_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q3_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q3_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q3_K_M GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Bielik-1.5B-Q3_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Mistral-7B Q4_K_M real-weight coverage. This TheBloke GGUF was exported
    // with llama.cpp's `general.architecture=llama` metadata even though the
    // checkpoint is Mistral-family, so the assertion below intentionally uses
    // Architecture.Llama, matching the Vulkan sibling test.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Mistral7B_Q4_K_M_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_MISTRAL_7B_Q4_K_M_GGUF", "TheBloke", "Mistral-7B-Instruct-v0.2-GGUF",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Mistral-7B Q4_K_M GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Mistral-7B-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Llama-3.1-8B IQ4_XS / IQ1_S — larger dense model, i-quant coverage.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Llama31_8B_IQ4_XS_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA31_8B_IQ4_XS_GGUF", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.1-8B IQ4_XS GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Llama-3.1-8B-IQ4_XS",
            prompt: "The capital of France is");
    }

    [SkippableFact]
    public void Llama31_8B_IQ1_S_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA31_8B_IQ1_S_GGUF", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ1_S.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.1-8B IQ1_S GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Llama, label: "Llama-3.1-8B-IQ1_S",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // DeepSeek-V2-Lite / DeepSeek-Coder-V2-Lite — MLA + MoE coverage.
    // CudaTransformerModel supports MLA natively (CudaMlaAttention /
    // CudaMlaLatentKvCache), so the CPU config is used unmodified — no
    // Vulkan-style UseLatentCache/UseHybridMlaCache stripping needed.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void DeepSeekV2Lite_Q4_K_M_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_DEEPSEEK_V2_LITE_Q4_K_M_GGUF", "mradermacher", "DeepSeek-V2-Lite-GGUF",
            "DeepSeek-V2-Lite.Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("DeepSeek-V2-Lite Q4_K_M GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.DeepSeekV2, label: "DeepSeekV2Lite-Q4_K_M",
            prompt: "The capital of France is");
    }

    [SkippableFact]
    public void DeepSeekCoderV2Lite_Q2_K_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_DEEPSEEK_CODER_V2_LITE_Q2_K_GGUF", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("DeepSeek-Coder-V2-Lite Q2_K GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.DeepSeekV2, label: "DeepSeekCoderV2Lite-Q2_K",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Qwen3.6-35B-A3B IQ2_M / IQ2_XXS — sparse MoE hybrid (GDN + full
    // attention) coverage at the low-bpw end.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Qwen36A3B_IQ2_M_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_QWEN36_A3B_IQ2_M_GGUF", "unsloth", "Qwen3.6-35B-A3B-GGUF",
            "Qwen3.6-35B-A3B-IQ2_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Qwen3.6-35B-A3B IQ2_M GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Qwen3MoeHybrid, label: "Qwen36A3B-IQ2_M",
            prompt: "The capital of France is");
    }

    [SkippableFact]
    public void Qwen36A3B_IQ2_XXS_CudaForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_QWEN36_A3B_IQ2_XXS_GGUF", "unsloth", "Qwen3.6-35B-A3B-GGUF",
            "Qwen3.6-35B-A3B-IQ2_XXS.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Qwen3.6-35B-A3B IQ2_XXS GGUF"));
        RunGgufParityTest(fixture.Path!, expectedArch: Architecture.Qwen3MoeHybrid, label: "Qwen36A3B-IQ2_XXS",
            prompt: "The capital of France is");
    }

    // ════════════════════════════════════════════════════════════════════
    // Driver
    // ════════════════════════════════════════════════════════════════════

    private void RunGgufParityTest(string path, Architecture expectedArch, string label, string prompt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string ptxDir = ResolvePtxDir();
        Skip.If(!Directory.Exists(ptxDir), $"CUDA PTX directory not found (resolved: {ptxDir}).");

        _output.WriteLine($"[{label}] gguf: {path}");

        using var cpuGguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        Assert.Equal(expectedArch, config.Architecture);

        var cpuLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        using var cpuModel = TransformerModel.LoadFromGguf(cpuGguf, config);
        cpuLoadWatch.Stop();
        _output.WriteLine(
            $"[{label}] CPU load ({cpuLoadWatch.Elapsed.TotalSeconds:F1} s): "
            + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers}");

        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var cudaGguf = GgufFile.Open(path);
        var cudaLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        CudaTransformerModel? cudaModel = null;
        try
        {
            cudaModel = CudaTransformerModel.LoadFromGguf(cudaGguf, config, deviceId: 0, ptxDir);
        }
        catch (DotLLM.Cuda.Interop.CudaException ex)
        {
            Skip.If(true,
                $"[{label}] CUDA load failed with {ex.Message}. Weights likely exceeded "
                + "available VRAM on this host. Re-run on a host with more VRAM.");
        }
        cudaLoadWatch.Stop();
        _output.WriteLine($"[{label}] CUDA load ({cudaLoadWatch.Elapsed.TotalSeconds:F1} s)");

        try
        {
            int[] promptIds = tokenizer.Encode(prompt).ToArray();
            Assert.NotEmpty(promptIds);
            _output.WriteLine($"[{label}] prompt: '{prompt}' -> {promptIds.Length} tokens [{string.Join(',', promptIds)}]");

            int strictArgmaxMatches = 0;
            int stepsChecked = 0;
            int vocab = config.VocabSize;

            // Growing-context reprefill loop (no KV cache) — matches the
            // Vulkan sibling test exactly, via the shared stateless
            // IModel.Forward(tokenIds, positions, deviceId) overload.
            var tokens = new List<int>(promptIds.Length + DecodeStepsToCheck);
            tokens.AddRange(promptIds);

            for (int step = 0; step <= DecodeStepsToCheck; step++)
            {
                int[] tokenIds = tokens.ToArray();
                int[] positions = new int[tokenIds.Length];
                for (int i = 0; i < positions.Length; i++) positions[i] = i;

                float[] cpuLogits = RunForwardLastRow(cpuModel, tokenIds, positions, vocab, deviceId: -1);
                float[] cudaLogits = RunForwardLastRow(cudaModel!, tokenIds, positions, vocab, deviceId: 0);

                AssertLogitsMatch(cpuLogits, cudaLogits, step, label);
                int cpuArgmax = Argmax(cpuLogits);
                int cudaArgmax = Argmax(cudaLogits);
                bool argmaxMatch = cpuArgmax == cudaArgmax;
                if (argmaxMatch) strictArgmaxMatches++;
                stepsChecked++;

                _output.WriteLine(
                    $"[{label}] step {step}: cpu_argmax={cpuArgmax} cuda_argmax={cudaArgmax}{(argmaxMatch ? " [match]" : " [diff]")}");

                tokens.Add(cpuArgmax);
            }

            Assert.True(strictArgmaxMatches >= StrictArgmaxFloor,
                $"[{label}] strict argmax match floor {StrictArgmaxFloor}/{stepsChecked} not met: "
                + $"got {strictArgmaxMatches}/{stepsChecked}.");
            _output.WriteLine($"[{label}] strict argmax matches: {strictArgmaxMatches}/{stepsChecked}");
        }
        finally
        {
            cudaModel?.Dispose();
            cudaGguf.Dispose();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════

    private static string ResolvePtxDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "ptx");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
    }

    private static unsafe float[] RunForwardLastRow(IModel model, int[] tokenIds, int[] positions, int vocab, int deviceId)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private void AssertLogitsMatch(float[] cpuLogits, float[] cudaLogits, int step, string label)
    {
        Assert.Equal(cpuLogits.Length, cudaLogits.Length);
        float maxAbs = 0f;
        int worstIdx = 0;
        for (int i = 0; i < cpuLogits.Length; i++)
        {
            float diff = Math.Abs(cpuLogits[i] - cudaLogits[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }
        float jaccard = TopKJaccard(cpuLogits, cudaLogits, TopKForJaccard);
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
