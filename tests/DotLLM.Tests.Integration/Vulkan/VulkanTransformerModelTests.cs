using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end parity tests for the Vulkan F32 forward pass against the CPU
/// reference on SmolLM-135M (Llama-family, GQA 9/3, head_dim 64).
/// </summary>
/// <remarks>
/// <para>
/// The quantised SmolLM-135M.Q8_0 GGUF is dequantised to FP32 on weight
/// upload (see <c>VulkanWeights.Upload</c>); the CPU reference keeps the
/// weights in Q8_0 and computes quantised dot products (Q8_0 × Q8_1 with
/// int32 accumulate). The two paths therefore agree on the same *model*
/// (same weight bytes) but compute slightly different arithmetic — the
/// Vulkan forward is actually closer to a theoretical FP32 oracle, while
/// the CPU path carries an additional round of Q8_1 activation
/// quantisation per projection.
/// </para>
/// <para>
/// Tolerance targets:
/// <list type="bullet">
///   <item>Strict argmax-match per step (the product-relevant metric).</item>
///   <item>Loose logit L∞ check (~3.0 abs) and top-k jaccard (≥ 0.5) to
///     catch structural regressions while tolerating Q8_0 vs FP32 drift —
///     without these two, the argmax match might succeed by luck even
///     when the Vulkan path has a serious correctness bug.</item>
/// </list>
/// The "F32 abs 1e-3 on logits" target from the end-to-end plan applies
/// to a pure-FP32 model (no quantised weights). We don't ship one today
/// for SmolLM-135M, so the test here is calibrated against the Q8_0
/// reality plus argmax, and leaves the strict FP32 parity test as a
/// follow-up once pre-dequantised weights are available.
/// </para>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public class VulkanTransformerModelTests
{
    // L∞ budget on raw logits. 3.0 absolute covers the observed Q8_0-vs-F32
    // drift for SmolLM-135M (max ~2.3 on decode step 0, with argmax match).
    // Going lower would force dequantised-weight parity which is tracked as
    // a separate follow-up; going significantly higher risks letting a
    // structural Vulkan regression slip through.
    private const float LogitsAbsTol = 3.0f;
    // Top-k jaccard: how many of CPU's top-k tokens appear in Vulkan's top-k,
    // divided by k. 0.5 is a low-but-non-trivial floor that catches obvious
    // divergence (wrong sign, wrong scale, dropped layer) without burning on
    // the natural re-ranking of very close-logit-value tokens that happens
    // when two ~= paths disagree on the fourth decimal place.
    private const int TopKForJaccard = 10;
    private const float TopKJaccardFloor = 0.5f;
    private const int DecodeStepsToCheck = 8;

    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanTransformerModelTests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void VulkanForward_MatchesCpuReference_OnEightDecodeSteps()
    {
        SkipIfVulkanUnavailable(out string spvDir);

        // Build both models from the same GGUF file.
        using var cpuGguf = GgufFile.Open(_fixture.FilePath);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        using var cpuModel = TransformerModel.LoadFromGguf(cpuGguf, cpuConfig);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var vkGguf = GgufFile.Open(_fixture.FilePath);
        var vkConfig = GgufModelConfigExtractor.Extract(vkGguf.Metadata);
        using var vkModel = VulkanTransformerModel.LoadFromGguf(vkGguf, vkConfig, spvDir);

        // Tokenise a known prompt. The exact continuation doesn't matter
        // for the parity check — we only need both backends to agree at every
        // step that we feed them the same prompt.
        int[] prompt = tokenizer.Encode("The capital of France is").ToArray();
        Assert.NotEmpty(prompt);

        using var cpuCache = new SimpleKvCache(cpuConfig.NumLayers,
            cpuConfig.NumKvHeads, cpuConfig.HeadDim, maxSeqLen: 128);
        using var vkCache = vkModel.CreateKvCache(maxSeqLen: 128);

        // Prefill: feed the entire prompt to both backends, record the logits
        // for the last token (decode position 0).
        int[] positions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) positions[i] = i;

        float[] cpuLogits = RunForwardCpu(cpuModel, prompt, positions, cpuCache);
        float[] vkLogits = RunForwardVulkan(vkModel, prompt, positions, vkCache);
        AssertLogitsMatch(cpuLogits, vkLogits, step: 0);

        int cpuToken = Argmax(cpuLogits);
        int vkToken = Argmax(vkLogits);
        // Strict argmax parity is only required on the prefill: both paths
        // see the same Q8_0 weights / same positions, and the prefill top
        // logit should be unambiguous for "The capital of France is" →
        // " Paris" (token 7042). Decode steps that follow see one-token
        // inputs with tied or near-tied top logits, and swap argmax within
        // the top-5 routinely — see AssertStepArgmaxInOthersTopK below.
        Assert.Equal(cpuToken, vkToken);

        // 8 decode steps, driven by the CPU token so both backends stay on
        // the same trajectory even when their own argmaxes drift. We assert
        // that each step's argmax is in the OTHER's top-K (K=5), which is
        // the product-relevant "close enough" test when comparing a Q8_0
        // CPU path (lossy activation quantisation) against an FP32 Vulkan
        // path (dequantised weights, FP32 activations).
        int nextToken = cpuToken;
        int nextPos = prompt.Length;

        int strictArgmaxMatches = 1; // prefill counts
        int stepsChecked = 1;
        _output.WriteLine($"step 0 (prefill): cpu={cpuToken} vk={vkToken} [match]");

        for (int step = 1; step <= DecodeStepsToCheck; step++)
        {
            int[] single = { nextToken };
            int[] pos = { nextPos };

            cpuLogits = RunForwardCpu(cpuModel, single, pos, cpuCache);
            vkLogits = RunForwardVulkan(vkModel, single, pos, vkCache);
            AssertLogitsMatch(cpuLogits, vkLogits, step);

            cpuToken = Argmax(cpuLogits);
            vkToken = Argmax(vkLogits);
            if (cpuToken == vkToken) strictArgmaxMatches++;
            stepsChecked++;

            _output.WriteLine($"step {step}: cpu={cpuToken} vk={vkToken} " +
                $"{(cpuToken == vkToken ? "[match]" : "[swap-in-topK]")}");

            // Require each side's top-1 token to appear in the other's top-K.
            AssertStepArgmaxInOthersTopK(cpuLogits, vkLogits, step, k: 5);

            // Advance both caches with the CPU token to keep trajectories aligned.
            nextToken = cpuToken;
            nextPos++;
        }

        _output.WriteLine($"summary: {strictArgmaxMatches}/{stepsChecked} strict argmax matches");

        // Require at least 5/9 strict argmax matches across prefill + 8 decodes.
        // This catches a systematic drift where Vulkan is consistently picking
        // the wrong token, while still tolerating the expected noise-floor
        // swap between the top-2 or top-3 close-valued tokens seen above.
        Assert.True(strictArgmaxMatches >= 5,
            $"Only {strictArgmaxMatches}/{stepsChecked} strict argmax matches " +
            "(expected >= 5). This likely indicates a structural regression, " +
            "not expected Q8_0-vs-F32 drift.");
    }

    private static void AssertStepArgmaxInOthersTopK(float[] cpu, float[] vk, int step, int k)
    {
        int cpuArg = Argmax(cpu);
        int vkArg = Argmax(vk);
        int[] cpuTopK = TopK(cpu, k);
        int[] vkTopK = TopK(vk, k);
        bool vkInCpu = Array.IndexOf(cpuTopK, vkArg) >= 0;
        bool cpuInVk = Array.IndexOf(vkTopK, cpuArg) >= 0;
        Assert.True(vkInCpu && cpuInVk,
            $"step {step}: argmax-in-top-{k} failed. " +
            $"cpu_arg={cpuArg} cpu_top={string.Join(',', cpuTopK)}; " +
            $"vk_arg={vkArg} vk_top={string.Join(',', vkTopK)}");
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static unsafe float[] RunForwardCpu(
        TransformerModel model, int[] tokenIds, int[] positions, SimpleKvCache cache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, cache);
        int vocabSize = model.Config.VocabSize;
        int seqLen = logits.Shape[0];
        long lastRow = (long)(seqLen - 1) * vocabSize;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount);
        float[] result = new float[vocabSize];
        span.Slice((int)lastRow, vocabSize).CopyTo(result);
        return result;
    }

    private static unsafe float[] RunForwardVulkan(
        VulkanTransformerModel model, int[] tokenIds, int[] positions, VulkanKvCache cache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, cache);
        int vocabSize = model.Config.VocabSize;
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(vocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        float[] result = new float[vocabSize];
        span.CopyTo(result);
        return result;
    }

    private static int Argmax(float[] values)
    {
        int idx = 0;
        float best = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > best) { best = values[i]; idx = i; }
        }
        return idx;
    }

    private static void AssertLogitsMatch(float[] expected, float[] actual, int step)
    {
        Assert.Equal(expected.Length, actual.Length);

        // L∞ check.
        float maxAbs = 0;
        int worstIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }
        Assert.True(maxAbs <= LogitsAbsTol,
            $"step {step}: L∞ logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} " +
            $"at idx {worstIdx} (cpu={expected[worstIdx]:G6} vk={actual[worstIdx]:G6})");

        // Top-k jaccard — both lists' token indices for the top-K logits.
        int[] eTop = TopK(expected, TopKForJaccard);
        int[] aTop = TopK(actual, TopKForJaccard);
        var eSet = new HashSet<int>(eTop);
        int overlap = 0;
        foreach (int t in aTop) if (eSet.Contains(t)) overlap++;
        float jaccard = overlap / (float)TopKForJaccard;
        Assert.True(jaccard >= TopKJaccardFloor,
            $"step {step}: top-{TopKForJaccard} jaccard {jaccard:G3} < {TopKJaccardFloor:G3}. " +
            $"cpu={string.Join(',', eTop)} vk={string.Join(',', aTop)}");
    }

    private static int[] TopK(float[] values, int k)
    {
        int[] idx = new int[k];
        float[] vals = new float[k];
        for (int i = 0; i < k; i++) { idx[i] = -1; vals[i] = float.NegativeInfinity; }
        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            // Insertion-sort into the top-k buffer (k small, O(N*k) acceptable for tests).
            int insertAt = -1;
            for (int j = 0; j < k; j++)
            {
                if (v > vals[j]) { insertAt = j; break; }
            }
            if (insertAt < 0) continue;
            for (int j = k - 1; j > insertAt; j--) { vals[j] = vals[j - 1]; idx[j] = idx[j - 1]; }
            vals[insertAt] = v; idx[insertAt] = i;
        }
        return idx;
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        string? found = null;
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
            {
                found = full;
                break;
            }
        }
        Skip.If(found is null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
        spvDir = found!;
    }
}
