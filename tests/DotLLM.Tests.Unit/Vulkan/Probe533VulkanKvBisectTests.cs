using System.Text;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #533, step 2 — per-layer KV-cache bisect.
/// </summary>
/// <remarks>
/// The kernel-level row-count invariance sweep
/// (<see cref="Probe533VulkanRowCountInvarianceTests"/>) came back clean for every
/// matmul, rmsnorm, rope and attention kernel. So the divergence is introduced by
/// the model's own orchestration, not by an isolated kernel. This probe runs a
/// first chunk of <c>c</c> tokens against a single-pass 6-token prefill and
/// compares the KV-cache rows <c>0..c-1</c> of EVERY layer bitwise. The first
/// layer that differs, and which rows differ, localizes the op.
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection("VulkanKernels")]
public sealed class Probe533VulkanKvBisectTests
{
    private static readonly int[] Prompt = [128000, 791, 6864, 315, 9822, 374];

    private readonly ITestOutputHelper _out;
    public Probe533VulkanKvBisectTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void KvRows_FirstChunk_MatchSinglePass()
    {
        string? path = FindModel();
        Skip.If(path is null, "model GGUF not found (set PROBE533_MODEL_GGUF).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        VulkanDevice device = model.Device;

        var sb = new StringBuilder();
        sb.AppendLine("=== #533 per-layer KV bisect ===");
        sb.AppendLine($"model={Path.GetFileName(path)} layers={config.NumLayers}");

        // Baseline: single-pass 6-token prefill.
        (float[][] baseK, float[][] baseV, float[] baseLogits) = RunAndCapture(model, device, config, Prompt.Length);

        int stride0 = KvStride(model, config, 0);
        foreach (int c in new[] { 1, 2, 3, 4, 5, 6 })
        {
            (float[][] k, float[][] v, _) = RunAndCapture(model, device, config, c);
            int firstBadLayer = -1;
            for (int layer = 0; layer < config.NumLayers && firstBadLayer < 0; layer++)
            {
                (long dk, _, _) = CompareRows(baseK[layer], k[layer], c, stride0);
                (long dv, _, _) = CompareRows(baseV[layer], v[layer], c, stride0);
                if (dk + dv > 0) firstBadLayer = layer;
            }
            sb.AppendLine($"  chunk={c} ({(c % 2 == 0 ? "even" : "odd ")}): " +
                          (firstBadLayer < 0 ? "KV IDENTICAL across all 16 layers" : $"first bad layer={firstBadLayer}"));
            // Per-row profile for the first three layers.
            for (int layer = 0; layer < 3; layer++)
            {
                var parts = new List<string>();
                for (int r = 0; r < c; r++)
                {
                    (long dk, float mk, _) = CompareRows(baseK[layer], k[layer], 1, stride0, rowOffset: r);
                    (long dv, float mv, _) = CompareRows(baseV[layer], v[layer], 1, stride0, rowOffset: r);
                    parts.Add($"r{r}:K{dk}/{stride0}({mk:E1}) V{dv}({mv:E1})");
                }
                sb.AppendLine($"          L{layer}: {string.Join("  ", parts)}");
            }
        }
        _ = baseLogits;

        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Token-flip question. A 0.2 logit shift only changes emitted text if it
    /// crosses a top-2 margin, and argmax held at 12366 for the single #532
    /// prompt. This arm greedily decodes 64 tokens after an even-chunked and an
    /// odd-chunked prefill of the same prompt and reports the first position
    /// where the two token sequences differ.
    /// </summary>
    [SkippableFact]
    public void GreedyContinuation_EvenVsOddChunkedPrefill()
    {
        string? path = FindModel();
        Skip.If(path is null, "model GGUF not found (set PROBE533_MODEL_GGUF).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

        const int gen = 64;
        var sb = new StringBuilder();
        sb.AppendLine("=== #533 greedy continuation, even vs odd chunked prefill ===");
        sb.AppendLine($"model={Path.GetFileName(path)} prompt={Prompt.Length} tokens, generate={gen}");

        int[] baseline = Generate(model, config, [Prompt.Length], gen);
        foreach (int[] split in new[] { new[] { 2, 2, 2 }, new[] { 3, 3 }, new[] { 1, 5 }, new[] { 5, 1 }, new[] { 2, 1, 3 } })
        {
            int[] got = Generate(model, config, split, gen);
            int first = -1;
            for (int i = 0; i < gen; i++) if (baseline[i] != got[i]) { first = i; break; }
            sb.AppendLine($"  chunks=[{string.Join(",", split)}] ({(Array.TrueForAll(split, s => s % 2 == 0) ? "all even" : "has odd")}): " +
                          (first < 0 ? "IDENTICAL 64 tokens" : $"FIRST DIFFERENT TOKEN at step {first}: {baseline[first]} -> {got[first]}"));
        }
        _out.WriteLine(sb.ToString());
    }

    private static unsafe int[] Generate(
        VulkanTransformerModel model, Core.Models.ModelConfig config, int[] chunkSizes, int gen)
    {
        using var kv = model.CreateKvCache(Prompt.Length + gen + 8);
        int pos = 0;
        float[] last = new float[config.VocabSize];
        foreach (int n in chunkSizes)
        {
            int[] toks = Prompt[pos..(pos + n)];
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = pos + i;
            using ITensor logits = model.Forward(toks, positions, deviceId: -1, kv);
            float* row = (float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * config.VocabSize;
            new Span<float>(row, config.VocabSize).CopyTo(last);
            pos += n;
        }

        var outTokens = new int[gen];
        for (int t = 0; t < gen; t++)
        {
            int next = ArgMax(last);
            outTokens[t] = next;
            using ITensor logits = model.Forward([next], [pos], deviceId: -1, kv);
            new Span<float>((float*)logits.DataPointer, config.VocabSize).CopyTo(last);
            pos++;
        }
        return outTokens;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>
    /// Runs a fresh prefill of the first <paramref name="n"/> prompt tokens and
    /// snapshots every layer's K and V cache buffers plus the last logits row.
    /// </summary>
    private static unsafe (float[][] K, float[][] V, float[] Logits) RunAndCapture(
        VulkanTransformerModel model, VulkanDevice device, Core.Models.ModelConfig config, int n)
    {
        using var kv = model.CreateKvCache(Prompt.Length + 8);
        var vkCache = (VulkanKvCache)kv;
        int[] toks = Prompt[..n];
        int[] positions = new int[n];
        for (int i = 0; i < n; i++) positions[i] = i;

        float[] last = new float[config.VocabSize];
        using (ITensor logits = model.Forward(toks, positions, deviceId: -1, kv))
        {
            float* row = (float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * config.VocabSize;
            new Span<float>(row, config.VocabSize).CopyTo(last);
        }

        var ks = new float[config.NumLayers][];
        var vs = new float[config.NumLayers][];
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            var kb = vkCache.GetKeysBuffer(layer);
            var vb = vkCache.GetValuesBuffer(layer);
            ks[layer] = new float[kb.Size / sizeof(float)];
            vs[layer] = new float[vb.Size / sizeof(float)];
            device.Download(kb, ks[layer]);
            device.Download(vb, vs[layer]);
        }
        return (ks, vs, last);
    }

    private static int KvStride(VulkanTransformerModel model, Core.Models.ModelConfig config, int layer)
    {
        using var kv = model.CreateKvCache(1);
        return ((VulkanKvCache)kv).KvStrideOf(layer);
    }

    private static (long diff, float maxAbs, int firstRow) CompareRows(
        float[] a, float[] b, int rows, int stride, int rowOffset = 0)
    {
        long diff = 0; float maxAbs = 0; int firstRow = -1;
        for (int r = rowOffset; r < rowOffset + rows; r++)
            for (int i = 0; i < stride; i++)
            {
                int idx = r * stride + i;
                if (idx >= a.Length || idx >= b.Length) break;
                if (BitConverter.SingleToInt32Bits(a[idx]) != BitConverter.SingleToInt32Bits(b[idx]))
                {
                    diff++;
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(a[idx] - b[idx]));
                    if (firstRow < 0) firstRow = r;
                }
            }
        return (diff, maxAbs, firstRow);
    }

    private static string? FindModel()
    {
        string? o = Environment.GetEnvironmentVariable("PROBE533_MODEL_GGUF");
        if (!string.IsNullOrEmpty(o)) return File.Exists(o) ? o : null;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        return candidates.FirstOrDefault(File.Exists);
    }
}
