using System.Text;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #532, model level — the step the issue asks for once the kernel
/// level comes back clean. Runs the exact shape that flips the CPU backend in #525:
/// a 5-token prefill split <c>[3,2]</c> versus single-pass <c>[5]</c>, on a QUANTIZED
/// model, comparing the final LOGITS ROW rather than the sampled token (argmax
/// insensitivity is what hid the CPU defect for two sessions).
/// </summary>
/// <remarks>
/// On CPU the same shape moves the top-1 logprob by 0.043 at Q8_0. A Vulkan zero here
/// is the model-level counterpart of the clean kernel measurements; any non-zero that
/// is NOT attention (the kernels measured bitwise-invariant at this shape) would point
/// at a batch-size-dependent GEMM dispatch, the Vulkan analog of #525 section 6.
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection("VulkanKernels")]
public sealed class Probe532VulkanModelChunkedPrefillTests
{
    // "The capital of France is" tokenized for Llama-3.2 (BOS + 5).
    private static readonly int[] Prompt = [128000, 791, 6864, 315, 9822, 374];

    private readonly ITestOutputHelper _out;
    public Probe532VulkanModelChunkedPrefillTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void ChunkedPrefill_FinalLogitsRow_MatchesSinglePass()
    {
        string? path = FindLlama32_1B_Q8_0();
        Skip.If(path is null, "Llama-3.2-1B-Instruct Q8_0 GGUF not found (set DOTLLM_LLAMA32_1B_Q8_0_GGUF).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

        float[] baseline = RunChunked(model, config.VocabSize, [Prompt.Length]);

        var sb = new StringBuilder();
        sb.AppendLine("=== Vulkan model-level chunked prefill, final logits row (bitwise) ===");
        sb.AppendLine($"model={Path.GetFileName(path)} promptLen={Prompt.Length} vocab={config.VocabSize}");

        long worst = 0;
        foreach (int[] split in new[]
                 {
                     new[] { 6 },            // identity control: must be exactly 0
                     new[] { 3, 3 },
                     new[] { 2, 2, 2 },
                     new[] { 1, 5 },
                     new[] { 3, 2, 1 },
                     new[] { 1, 1, 1, 1, 1, 1 },
                     new[] { 2, 4 },
                     new[] { 4, 2 },
                     new[] { 5, 1 },
                     new[] { 1, 1, 4 },   // odd EARLY chunks, even final
                     new[] { 3, 1, 2 },   // odd early chunks, even final
                     new[] { 2, 1, 3 },
                 })
        {
            float[] got = RunChunked(model, config.VocabSize, split);
            long d = 0; float maxAbs = 0;
            for (int i = 0; i < baseline.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(baseline[i]) != BitConverter.SingleToInt32Bits(got[i]))
                { d++; maxAbs = MathF.Max(maxAbs, MathF.Abs(baseline[i] - got[i])); }
            }
            worst = Math.Max(worst, d);
            sb.AppendLine($"  chunks=[{string.Join(",", split)}]: differing={d,6}/{baseline.Length,-7} " +
                          $"maxAbs={maxAbs:E3}  argmax={ArgMax(baseline)}/{ArgMax(got)}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: Vulkan chunked prefill is BIT-EXACT against single-pass at the shape that flips CPU."
            : $"VERDICT: Vulkan chunked prefill DIVERGES from single-pass (worst {worst} logits).");
        _out.WriteLine(sb.ToString());
    }

    private static unsafe float[] RunChunked(VulkanTransformerModel model, int vocabSize, int[] chunkSizes)
    {
        using var kv = model.CreateKvCache(Prompt.Length + 8);
        float[] last = new float[vocabSize];
        int pos = 0;
        foreach (int n in chunkSizes)
        {
            int[] toks = Prompt[pos..(pos + n)];
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = pos + i;
            using ITensor logits = model.Forward(toks, positions, deviceId: -1, kv);
            float* row = (float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocabSize;
            new Span<float>(row, vocabSize).CopyTo(last);
            pos += n;
        }
        return last;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private static string? FindLlama32_1B_Q8_0()
    {
        string? probeOverride = Environment.GetEnvironmentVariable("PROBE532_MODEL_GGUF");
        if (!string.IsNullOrEmpty(probeOverride))
            return File.Exists(probeOverride) ? probeOverride : null;

        string? overridePath = Environment.GetEnvironmentVariable("DOTLLM_LLAMA32_1B_Q8_0_GGUF");
        if (!string.IsNullOrEmpty(overridePath))
            return File.Exists(overridePath) ? overridePath : null;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        return candidates.FirstOrDefault(File.Exists);
    }
}
