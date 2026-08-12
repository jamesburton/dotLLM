using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit.Abstractions;
using Xunit;

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
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.2-1B Q8_0 GGUF"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Llama-3.2-1B-Q8_0",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bielik-1.5B Q4_K_M (dense Llama, exercises Q4_K_M path)
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Bielik15B_Q4_K_M_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q4_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q4_K_M GGUF"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Bielik-1.5B-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // Mistral-7B Q4_K_M real-weight coverage. This TheBloke GGUF was exported
    // with llama.cpp's `general.architecture=llama` metadata even though the
    // checkpoint is Mistral-family, so the assertion below intentionally uses
    // Architecture.Llama. The separate SafeTensors gate covers native
    // Architecture.Mistral config extraction when a HF checkpoint is present.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Mistral7B_Q4_K_M_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_MISTRAL_7B_Q4_K_M_GGUF", "TheBloke", "Mistral-7B-Instruct-v0.2-GGUF",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Mistral-7B Q4_K_M GGUF"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Mistral-7B-Q4_K_M",
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

    // ────────────────────────────────────────────────────────────────────
    // Llama-3.1-8B IQ4_XS (dense Llama, exercises IQ4_XS path — most-used
    // IQ-family quant in production deployments).
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Llama31_8B_IQ4_XS_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA31_8B_IQ4_XS_GGUF", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.1-8B IQ4_XS GGUF (~4.5 GB)"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Llama-3.1-8B-IQ4_XS",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // IQ1_S real-GGUF smoke. The smallest GGUF quant — production-cached
    // models are rare (extreme accuracy loss); this gate exists so any
    // available IQ1_S GGUF on disk (e.g. quant-experimentation cache)
    // exercises the IQ1_S projection path end-to-end vs the CPU oracle.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Llama31_8B_IQ1_S_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA31_8B_IQ1_S_GGUF", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ1_S.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.1-8B IQ1_S GGUF (~2.0 GB; any IQ1_S GGUF may be substituted via the env var)"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Llama-3.1-8B-IQ1_S",
            prompt: "The capital of France is");
    }

    [SkippableFact]
    public void DeepSeekV2Lite_Q4_K_M_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_DEEPSEEK_V2_LITE_Q4_K_M_GGUF", "mradermacher", "DeepSeek-V2-Lite-GGUF",
            "DeepSeek-V2-Lite.Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("DeepSeek-V2-Lite Q4_K_M GGUF (~10.4 GB)"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.DeepSeekV2, label: "DeepSeek-V2-Lite-Q4_K_M",
            prompt: "The capital of France is");
    }

    // ────────────────────────────────────────────────────────────────────
    // DeepSeek-Coder-V2-Lite Q2_K (MLA + MoE, ~5 GB GGUF) — exercises Q2_K
    // Vulkan kernels (densest K-quant, 84-byte super-block, ~2.6 bpw). Real-
    // weight smoke for the Q2_K path; gated on the conventional cache path.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void DeepSeekCoderV2Lite_Q2_K_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_DEEPSEEK_CODER_V2_LITE_Q2_K_GGUF", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("DeepSeek-Coder-V2-Lite Q2_K GGUF"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.DeepSeekV2, label: "DeepSeek-Coder-V2-Lite-Q2_K",
            prompt: "def fibonacci(n):");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bielik-1.5B Q3_K_M (~660 MB GGUF) — exercises Q3_K Vulkan kernels.
    //
    // This test is what surfaced #311: Q3_K's bit layout was transposed in
    // BOTH the scale high-bits and the element ordering, in every backend and
    // in the kernel-test fixture, so the decoded weights were noise (corr
    // 0.006 vs the Q8_0 build of the same model). Cross-backend parity still
    // "held" at the kernel level because every side shared the bug; only a
    // real 32-layer GGUF forward amplified the resulting garbage far enough
    // (L∞ 5.09, top-10 Jaccard 0.40) to break the end-to-end bound.
    //
    // Canonical-semantics coverage now lives in
    // DequantizeKQuantTests.Q3_K_DenseRandomBlocks_MatchLlamaCppReference;
    // this test remains the cross-backend check. See docs/QUANTIZATION.md.
    // ────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Bielik15B_Q3_K_M_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q3_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q3_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q3_K_M GGUF"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Llama, label: "Bielik-1.5B-Q3_K_M",
            prompt: "The capital of France is");
    }

    // ════════════════════════════════════════════════════════════════════
    // Qwen3.6-A3B-IQ2_M / IQ2_XXS — IQ2 family on Vulkan (~11 GB GGUFs).
    // Gated; download-on-demand from huggingface.co/unsloth or similar
    // mradermacher GGUF mirrors. The MOSTLY_IQ2_M file-type lands on the
    // IQ2_S block layout in ggml; both kernels share the same dispatch.
    // ════════════════════════════════════════════════════════════════════

    [SkippableFact]
    public void Qwen36A3B_IQ2_M_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_QWEN36_A3B_IQ2_M_GGUF", "unsloth", "Qwen3.6-35B-A3B-GGUF",
            "Qwen3.6-35B-A3B-IQ2_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Qwen3.6-A3B IQ2_M GGUF (~11.5 GB)"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Qwen3MoeHybrid, label: "Qwen3.6-A3B-IQ2_M",
            prompt: "The capital of France is");
    }

    [SkippableFact]
    public void Qwen36A3B_IQ2_XXS_VulkanForward_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_QWEN36_A3B_IQ2_XXS_GGUF", "unsloth", "Qwen3.6-35B-A3B-GGUF",
            "Qwen3.6-35B-A3B-IQ2_XXS.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Qwen3.6-A3B IQ2_XXS GGUF (~10.8 GB)"));
        string path = fixture.Path!;
        RunGgufParityTest(path, expectedArch: Architecture.Qwen3MoeHybrid, label: "Qwen3.6-A3B-IQ2_XXS",
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

        // #328: go through the per-architecture DISPATCHERS, not the dense loaders.
        // `TransformerModel.LoadFromGguf` / `VulkanTransformerModel.LoadFromGguf` are the
        // dense-attention implementations; calling them directly meant this suite never
        // exercised the dispatch point #259 added, and a hybrid checkpoint (Qwen3.6-A3B)
        // died inside the dense weight loader with `blk.0.attn_output.weight not present`
        // instead of either running or reporting a real reason.
        var cpuLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        using IModel cpuModel = ModelLoader.CreateCpuModelFromGguf(cpuGguf, cpuConfig);
        cpuLoadWatch.Stop();
        _output.WriteLine(
            $"[{label}] CPU load ({cpuLoadWatch.Elapsed.TotalSeconds:F1} s): "
            + $"vocab={cpuConfig.VocabSize} hidden={cpuConfig.HiddenSize} layers={cpuConfig.NumLayers}");

        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var vkGguf = GgufFile.Open(path);
        var vkLoadWatch = System.Diagnostics.Stopwatch.StartNew();
        // The dispatcher takes an existing device and does NOT take ownership of it, so the
        // test owns it and must outlive the model (disposed in the outer finally).
        using var vkDevice = VulkanDevice.Create();
        IModel? vkModel = null;
        try
        {
            (vkModel, _) = VulkanModelLoader.CreateFromGguf(vkDevice, vkGguf, vkConfig, spvDir);
        }
        catch (DotLLM.Vulkan.Interop.VulkanException ex) when (
            ex.ErrorCode == -1 || ex.ErrorCode == -2 || ex.ErrorCode == -5)
        {
            Skip.If(true,
                $"[{label}] Vulkan load failed with {ex.Message}. Source-quant "
                + "weights still exceeded available device-local memory on this "
                + "host. Re-run on a host with more VRAM.");
        }
        catch (InsufficientMemoryException ex)
        {
            // #326. The routed MoE expert banks would be host-dequantised to F32 because at
            // least one uses a quantization Vulkan cannot keep device-resident (#327). That is
            // a host-capacity limit, not a CPU↔Vulkan divergence — reporting it as a parity
            // FAIL misrepresents the sweep. The preflight message itemises the offending banks.
            Skip.If(true, $"[{label}] {ex.Message}");
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
            //
            // #328 — RECURRENT STATE. Every iteration re-forwards the WHOLE prefix, so each
            // iteration is an independent sequence. The uncached `Forward(tokens, positions,
            // deviceId)` overload carries no state container and falls back to the model-owned
            // recurrent state (`_gdnCache` / `_ssmCache`), which persists across calls — correct
            // for prefill-then-decode of one sequence, silently wrong here. On a recurrent
            // architecture (Qwen3MoeHybrid GDN, Nemotron-H SSM) step n would otherwise inherit
            // steps 0..n-1's recurrence *on top of* re-reading the same tokens, and the two
            // backends could then drift for reasons unrelated to kernel parity. This is the same
            // hazard perplexity scoring hit in #261; the fix is the same hook.
            //
            // Called unconditionally: it is a documented no-op on stateless (dense) models, and
            // for a recurrent model that forgot to implement it the IModel default THROWS — so a
            // future architecture cannot inherit a silent no-op here.
            var tokens = new List<int>(promptIds.Length + DecodeStepsToCheck);
            tokens.AddRange(promptIds);

            float[]? firstStepCpuLogits = null;
            float[]? firstStepVkLogits = null;
            int[]? firstStepTokenIds = null;
            int[]? firstStepPositions = null;

            for (int step = 0; step <= DecodeStepsToCheck; step++)
            {
                int[] tokenIds = tokens.ToArray();
                int[] positions = new int[tokenIds.Length];
                for (int i = 0; i < positions.Length; i++) positions[i] = i;

                cpuModel.ResetSequenceState();
                vkModel!.ResetSequenceState();

                float[] cpuLogits = RunForwardCpuLastRow(cpuModel, tokenIds, positions, vocab);
                float[] vkLogits = RunForwardVulkanLastRow(vkModel!, tokenIds, positions, vocab);

                if (step == 0)
                {
                    firstStepCpuLogits = (float[])cpuLogits.Clone();
                    firstStepVkLogits = (float[])vkLogits.Clone();
                    firstStepTokenIds = tokenIds;
                    firstStepPositions = positions;
                }

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

            // #328 — the state-independence check that makes the reset above load-bearing.
            //
            // Re-running step 0's exact inputs after the whole loop must reproduce step 0's
            // exact outputs, BIT-FOR-BIT, on both backends: nothing about the inputs changed,
            // and a correctly reset model is a pure function of (weights, tokens, positions).
            // Without the reset a recurrent model returns different logits here, because nine
            // forwards' worth of GDN/SSM recurrence has accumulated. This is what turns "we
            // called ResetSequenceState" into "the reset actually restores the initial state on
            // this architecture, on this backend" — and it discriminates: it is false before the
            // fix and true after, on any recurrent model, without needing a known-good reference.
            // On a dense model it is trivially true and costs one extra prefill per backend.
            cpuModel.ResetSequenceState();
            vkModel!.ResetSequenceState();
            float[] cpuReplay = RunForwardCpuLastRow(cpuModel, firstStepTokenIds!, firstStepPositions!, vocab);
            float[] vkReplay = RunForwardVulkanLastRow(vkModel!, firstStepTokenIds!, firstStepPositions!, vocab);
            AssertBitIdentical(firstStepCpuLogits!, cpuReplay, label, "CPU");
            AssertBitIdentical(firstStepVkLogits!, vkReplay, label, "Vulkan");
            _output.WriteLine(
                $"[{label}] sequence-state reset verified on both backends "
                + $"(recurrent={cpuModel.RequiresPerSequenceState})");

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

    /// <summary>
    /// Asserts two logit vectors are bit-identical. Deliberately exact (#328): this compares a
    /// forward against a REPLAY of itself on the same backend with the same inputs, so any
    /// difference at all means the model was not in the same state — there is no legitimate
    /// numerical slack to allow for, and a tolerance here would hide exactly what it checks.
    /// </summary>
    private static void AssertBitIdentical(float[] first, float[] replay, string label, string backend)
    {
        Assert.Equal(first.Length, replay.Length);
        for (int i = 0; i < first.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(replay[i]))
                continue;
            Assert.Fail(
                $"[{label}] {backend}: re-running step 0's inputs after the decode loop did not "
                + $"reproduce step 0's logits (index {i}: {first[i]:R} -> {replay[i]:R}). The model "
                + "did not return to its initial state, so every parity number above was measured "
                + "against accumulated recurrent state rather than the sequence under test. See "
                + "IModel.ResetSequenceState and issues #328 / #261.");
        }
    }

    private static unsafe float[] RunForwardVulkanLastRow(IModel model, int[] tokenIds, int[] positions, int vocab)
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
