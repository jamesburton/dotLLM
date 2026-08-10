using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end parity for the fused SwiGLU + Q8_1-quantize down_proj decode
/// path (issue #71) on the real SmolLM-135M.Q8_0 GGUF. Runs the full Vulkan
/// decode forward TWICE — once with the fusion on (default) and once with
/// <c>DOTLLM_VULKAN_DISABLE_FUSED_SWIGLU_QUANT=1</c> (the pre-#71 standalone
/// SwiGLU -> quantize -> MMVQ+residual chain) — and asserts the logits are
/// BIT-IDENTICAL across prefill plus several decode steps.
/// </summary>
/// <remarks>
/// This is the discriminating wiring test for the fusion: the kernel-level
/// invariant is covered by
/// <c>VulkanSwiGluQuantizeQ8_1FusedKernelTests.Launch_BitIdenticalToStandalonePair</c>;
/// this test additionally exercises the production model wiring (the
/// <c>TryRecordFusedSwiGluQuantizeDownResidual</c> gate + the residual-fused
/// MMVQ store), which a kernel-only test cannot catch (e.g. a wrong buffer
/// passed at the call site, or the qualification guard admitting an
/// unsupported layer shape).
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public sealed class VulkanSwiGluQuantizeFusedParityTests
{
    private const int DecodeStepsToCheck = 6;
    private const string DisableSwiGluQuantEnvVar = "DOTLLM_VULKAN_DISABLE_FUSED_SWIGLU_QUANT";

    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanSwiGluQuantizeFusedParityTests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void Fused_BitIdenticalToStandaloneChain_OnSmolLmDecode()
    {
        SkipVulkanOrMmvqUnavailable(out string spvDir);

        int[] prompt;
        using (var gguf = GgufFile.Open(_fixture.FilePath))
        {
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            prompt = tokenizer.Encode("The capital of France is").ToArray();
        }
        Assert.NotEmpty(prompt);

        float[][] fused = RunDecode(spvDir, prompt, disableFused: false);
        float[][] standalone = RunDecode(spvDir, prompt, disableFused: true);

        Assert.Equal(fused.Length, standalone.Length);
        for (int step = 0; step < fused.Length; step++)
        {
            float[] a = fused[step];
            float[] b = standalone[step];
            Assert.Equal(a.Length, b.Length);
            int mismatch = -1;
            for (int i = 0; i < a.Length; i++)
            {
                if (!a[i].Equals(b[i])) { mismatch = i; break; }
            }
            Assert.True(mismatch < 0,
                $"step {step}: fused vs standalone-chain logits differ at index {mismatch} " +
                $"(fused={(mismatch >= 0 ? a[mismatch] : 0):G9}, " +
                $"standalone={(mismatch >= 0 ? b[mismatch] : 0):G9}). " +
                "The fused SwiGLU+quantize+down_proj chain is not bit-identical to the unfused form.");
            _output.WriteLine($"step {step}: bit-identical ({a.Length} logits)");
        }
    }

    private float[][] RunDecode(string spvDir, int[] prompt, bool disableFused)
    {
        string? original = Environment.GetEnvironmentVariable(DisableSwiGluQuantEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(DisableSwiGluQuantEnvVar, disableFused ? "1" : null);

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
            Environment.SetEnvironmentVariable(DisableSwiGluQuantEnvVar, original);
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
                "Device lacks integer-dot-product — MMVQ path unavailable; fused/standalone collapse to the same F32-in GEMV.");

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
