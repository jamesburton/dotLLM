using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end parity for the MMVQ shared-activation-quant optimisation on the
/// real SmolLM-135M.Q8_0 GGUF (dense GQA Llama-family, head_dim 64). Runs the
/// full Vulkan decode forward through the restructured Q/K/V and gate/up call
/// sites (<c>RecordSharedInputMmvqGroup</c>) TWICE — once with sharing on
/// (default) and once with <c>DOTLLM_VULKAN_MMVQ_NO_SHARE=1</c> — and asserts
/// the logits are BIT-IDENTICAL across prefill plus several decode steps.
/// </summary>
/// <remarks>
/// <para>
/// This is the discriminating wiring test for the optimisation: sharing
/// quantizes the activation once per same-input group instead of per
/// projection, but the quantize is deterministic and every GEMV is identical,
/// so the two paths must agree bit-for-bit. A wiring bug — wrong barrier order,
/// a clobbered shared scratch, the fused-vs-share branch inverted, or a
/// mismatched input dim — corrupts some projection's GEMV and breaks the
/// equality. (The kernel-level invariant is covered by
/// <c>VulkanMatMulQ8_0MmvqKernelTests.Mmvq_SharedQuant_BitIdenticalToPerProjection</c>;
/// this test additionally exercises the production model wiring.)
/// </para>
/// <para>
/// Q8_0 weights stay on device in their source byte layout
/// (<c>VulkanWeights</c> default <c>dequantToFp32=false</c>), so the decode
/// projections genuinely dispatch through the MMVQ path on integer-dot-capable
/// devices. Skipped where MMVQ is unavailable (the shared and per-projection
/// paths then collapse to the same F32-in GEMV and the test is vacuous).
/// </para>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public sealed class VulkanMmvqSharedQuantParityTests
{
    private const int DecodeStepsToCheck = 6;

    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanMmvqSharedQuantParityTests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void SharedQuant_BitIdenticalToPerProjection_OnSmolLmDecode()
    {
        SkipVulkanOrMmvqUnavailable(out string spvDir);

        int[] prompt;
        using (var gguf = GgufFile.Open(_fixture.FilePath))
        {
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            prompt = tokenizer.Encode("The capital of France is").ToArray();
        }
        Assert.NotEmpty(prompt);

        // SHARE = default (env unset). NO_SHARE = env set. The flag is read at
        // model construction, so build each model under the desired setting.
        float[][] shared = RunDecode(spvDir, prompt, noShare: false);
        float[][] perProj = RunDecode(spvDir, prompt, noShare: true);

        Assert.Equal(shared.Length, perProj.Length);
        for (int step = 0; step < shared.Length; step++)
        {
            float[] a = shared[step];
            float[] b = perProj[step];
            Assert.Equal(a.Length, b.Length);
            int mismatch = -1;
            for (int i = 0; i < a.Length; i++)
            {
                if (!a[i].Equals(b[i])) { mismatch = i; break; }
            }
            Assert.True(mismatch < 0,
                $"step {step}: shared vs per-projection logits differ at index {mismatch} " +
                $"(shared={(mismatch >= 0 ? a[mismatch] : 0):G9}, " +
                $"perProj={(mismatch >= 0 ? b[mismatch] : 0):G9}). " +
                "The restructured MMVQ group wiring is not bit-identical to per-projection quant.");
            _output.WriteLine($"step {step}: bit-identical ({a.Length} logits)");
        }
    }

    /// <summary>
    /// Companion to <see cref="SharedQuant_BitIdenticalToPerProjection_OnSmolLmDecode"/>
    /// for the fused dual-output dispatch (issue #71): the FFN gate_proj/up_proj
    /// pair collapses into one <c>MatMulQ8_0MmvqDualKernel</c> dispatch by default;
    /// <c>DOTLLM_VULKAN_MMVQ_NO_DUAL=1</c> forces the pre-#71 two-dispatch form
    /// (sharing still on). Both must produce bit-identical logits — the dual
    /// kernel changes dispatch shape only, not arithmetic.
    /// </summary>
    [SkippableFact]
    public void Dual_BitIdenticalToSeparateDispatches_OnSmolLmDecode()
    {
        SkipVulkanOrMmvqUnavailable(out string spvDir);

        int[] prompt;
        using (var gguf = GgufFile.Open(_fixture.FilePath))
        {
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            prompt = tokenizer.Encode("The capital of France is").ToArray();
        }
        Assert.NotEmpty(prompt);

        // Dual is opt-in (off by default — see VulkanTransformerModel.IsMmvqDualEnabled,
        // a measured non-win on this hardware). Force it on for one run to verify the
        // fused dispatch is still bit-identical to the default two-dispatch form.
        float[][] dual = RunDecode(spvDir, prompt, noShare: false, dualOn: true);
        float[][] separate = RunDecode(spvDir, prompt, noShare: false, dualOn: false);

        Assert.Equal(dual.Length, separate.Length);
        for (int step = 0; step < dual.Length; step++)
        {
            float[] a = dual[step];
            float[] b = separate[step];
            Assert.Equal(a.Length, b.Length);
            int mismatch = -1;
            for (int i = 0; i < a.Length; i++)
            {
                if (!a[i].Equals(b[i])) { mismatch = i; break; }
            }
            Assert.True(mismatch < 0,
                $"step {step}: dual vs separate-dispatch logits differ at index {mismatch} " +
                $"(dual={(mismatch >= 0 ? a[mismatch] : 0):G9}, " +
                $"separate={(mismatch >= 0 ? b[mismatch] : 0):G9}). " +
                "The fused dual MMVQ dispatch is not bit-identical to the two-dispatch form.");
            _output.WriteLine($"step {step}: bit-identical ({a.Length} logits)");
        }
    }

    // Builds a fresh Vulkan model under the requested NO_SHARE/DUAL setting and
    // runs prefill + DecodeStepsToCheck single-token decodes, returning the
    // last-row logits per step. The decode trajectory is driven by EACH run's OWN
    // argmax — which is fine because (a) both runs are deterministic and (b) if
    // they ever diverge, the bit-identical assert fires at that step before the
    // trajectories can drift apart.
    private float[][] RunDecode(string spvDir, int[] prompt, bool noShare, bool dualOn = false)
    {
        string? originalShare = Environment.GetEnvironmentVariable(
            VulkanTransformerModel.MmvqNoShareEnvVar);
        string? originalDual = Environment.GetEnvironmentVariable(
            VulkanTransformerModel.MmvqDualEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(
                VulkanTransformerModel.MmvqNoShareEnvVar, noShare ? "1" : null);
            Environment.SetEnvironmentVariable(
                VulkanTransformerModel.MmvqDualEnvVar, dualOn ? "1" : null);

            using var gguf = GgufFile.Open(_fixture.FilePath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
            using var cache = model.CreateKvCache(maxSeqLen: 128);

            var steps = new float[DecodeStepsToCheck + 1][];

            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            steps[0] = LastRowLogits(model, prompt, positions, cache);

            int nextToken = Argmax(steps[0]);
            int nextPos = prompt.Length;
            for (int step = 1; step <= DecodeStepsToCheck; step++)
            {
                steps[step] = LastRowLogits(model, new[] { nextToken }, new[] { nextPos }, cache);
                nextToken = Argmax(steps[step]);
                nextPos++;
            }

            return steps;
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.MmvqNoShareEnvVar, originalShare);
            Environment.SetEnvironmentVariable(VulkanTransformerModel.MmvqDualEnvVar, originalDual);
        }
    }

    private static unsafe float[] LastRowLogits(
        VulkanTransformerModel model, int[] tokenIds, int[] positions, VulkanKvCache cache)
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

    private static int Argmax(float[] v)
    {
        int arg = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[arg]) arg = i;
        return arg;
    }

    private static void SkipVulkanOrMmvqUnavailable(out string spvDir)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        using (var device = VulkanDevice.Create())
            Skip.IfNot(device.HasIntegerDotProduct,
                "Device lacks integer-dot-product — MMVQ path unavailable; share/no-share collapse to the same GEMV.");

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
        Skip.If(found is null, "Compiled SPV shaders not found.");
        spvDir = found!;
    }
}
