using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.CrossBackend;

/// <summary>
/// Issue #278: the cached <c>seqLen == 1</c> (GEMV) decode leg for IQ2_XS on
/// Vulkan, exercised with a REAL <see cref="IKvCache"/> on both backends —
/// the one scenario <see cref="CrossBackendQuantGateTests.Backend_AgreesWithCpu"/>
/// does NOT cover. That gate's own decode leg is explicitly cache-less (a
/// fresh growing-prefix forward every step, per its class remarks) precisely
/// because every backend implements the 3-arg <c>IModel.Forward</c>
/// uniformly; it never dispatches the fused RoPE+KV-cache-write path or reads
/// back through <see cref="VulkanKvCache"/> / <see cref="SimpleKvCache"/>.
/// This test fills that gap: prefill a real, long-ish prompt through each
/// backend's own real KV cache, take one more cached single-token step, and
/// compare logits cosine similarity — the literal "cached seqLen==1 (GEMV)"
/// scenario the issue names, at real production shapes (Llama-3.2-1B,
/// hidden_size 2048, vocab 128256) on the real <c>--pure</c> IQ2_XS fixture.
/// </summary>
/// <remarks>
/// Root-cause status (2026-08-07 investigation): could NOT reproduce a
/// negative-cosine (or any anomalous) divergence on the current `dev` tip.
/// The IQ2_XS Vulkan GEMV/MMVQ shaders were verified element-for-element
/// against llama.cpp's <c>dequantize_row_iq2_xs</c> (bit-index layout, sign
/// table, sub-scale nibble split all match); production-scale kernel-level
/// parity (<c>VulkanMatMulIq2XsGemvF32KernelTests</c> /
/// <c>VulkanMatMulIq2XsMmvqKernelTests</c>, both extended by this same
/// change to reach real k=2048/m=128256 shapes) passes cleanly; and the
/// existing cross-backend gate's own cache-less decode leg passes at
/// mean(1-cos)=2.1e-4. This test is the closest literal reproduction of the
/// issue's named scenario and also passes. It is kept as permanent coverage
/// for the gap the issue correctly identified (no real-KV-cache leg existed
/// for any Vulkan IQ2 type before this change) rather than as a red test,
/// since the described defect does not currently reproduce.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class Iq2XsVulkanCachedDecodeGateTests
{
    private const string Corpus =
        "The quick brown fox jumps over the lazy dog near the old stone bridge. "
        + "Scientists have long studied how small variations in early conditions can lead to "
        + "very different outcomes over time, a phenomenon popularly known as the butterfly "
        + "effect. In computing, this sensitivity shows up whenever a tiny rounding error "
        + "compounds across many sequential steps of a calculation.";

    private const int DecodeSteps = 8;

    /// <summary>
    /// Bound on mean <c>(1 - cosine_similarity)</c> across <see cref="DecodeSteps"/>
    /// real cached-decode steps. Matches
    /// <see cref="CrossBackendQuantGateTests.OneMinusCosineTolerance"/>'s rationale:
    /// well above ordinary FP-reduction-order noise (~1e-3 territory per #276),
    /// far below what a structurally wrong kernel produces (near-orthogonal or
    /// anti-correlated, 1-cos approaching or exceeding 1.0).
    /// </summary>
    private const double OneMinusCosineTolerance = 0.05;

    private readonly ITestOutputHelper _output;

    public Iq2XsVulkanCachedDecodeGateTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void CachedSeqLen1Decode_Iq2Xs_Vulkan_AgreesWithCpu()
    {
        string? path = CrossBackendQuantGateTests.ResolveFixturePath(QuantizationType.IQ2_XS);
        Skip.If(path is null, $"IQ2_XS: fixture not found (see docs/QUANT_FIXTURES.md).");

        bool vulkanOk;
        try { using var probe = VulkanDevice.Create(); vulkanOk = true; }
        catch { vulkanOk = false; }
        Skip.IfNot(vulkanOk, "Vulkan runtime not available on this host.");

        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPIR-V directory not found.");

        using var cpuGguf = GgufFile.Open(path!);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(cpuGguf, cpuConfig, ThreadingConfig.Auto);

        using var vkGguf = GgufFile.Open(path!);
        var vkConfig = GgufModelConfigExtractor.Extract(vkGguf.Metadata);
        using var device = VulkanDevice.Create();
        var (vkModel, vkKvCacheFactory) = VulkanModelLoader.CreateFromGguf(device, vkGguf, vkConfig, spvDir!);
        try
        {
            int[] promptIds = tokenizer.Encode(Corpus);
            Assert.True(promptIds.Length > 8, "corpus too short to score.");
            int vocab = cpuConfig.VocabSize;

            // Real KV caches, real prefill, on both backends independently.
            using var cpuCache = new SimpleKvCache(KvGeometry.FromConfig(cpuConfig), maxSeqLen: promptIds.Length + DecodeSteps + 4);
            using var vkCache = vkKvCacheFactory(promptIds.Length + DecodeSteps + 4);

            int[] positions = BuildPositions(promptIds.Length);
            using (cpuModel.Forward(promptIds, positions, deviceId: -1, cpuCache)) { }
            using (vkModel.Forward(promptIds, positions, deviceId: 0, vkCache)) { }
            Assert.Equal(promptIds.Length, cpuCache.CurrentLength);
            Assert.Equal(promptIds.Length, vkCache.CurrentLength);

            double sumOneMinusCos = 0;
            int nextToken = promptIds[^1];

            for (int step = 0; step < DecodeSteps; step++)
            {
                int pos = cpuCache.CurrentLength; // == vkCache.CurrentLength, asserted below.
                Assert.Equal(pos, vkCache.CurrentLength);

                int[] oneTok = { nextToken };
                int[] onePos = { pos };

                float[] cpuLogits;
                using (ITensor l = cpuModel.Forward(oneTok, onePos, deviceId: -1, cpuCache))
                    cpuLogits = LastRow(l, vocab);

                float[] vkLogits;
                using (ITensor l = vkModel.Forward(oneTok, onePos, deviceId: 0, vkCache))
                    vkLogits = LastRow(l, vocab);

                double cos = CosineSimilarity(cpuLogits, vkLogits);
                sumOneMinusCos += 1.0 - cos;
                _output.WriteLine($"[IQ2_XS/Vulkan cached decode] step {step}: pos={pos} cos={cos:F6}");

                nextToken = Argmax(cpuLogits);
            }

            double meanOneMinusCos = sumOneMinusCos / DecodeSteps;
            _output.WriteLine(
                $"[IQ2_XS/Vulkan cached decode] mean(1-cos) over {DecodeSteps} real cached steps: "
                + $"{meanOneMinusCos:E4} (bound {OneMinusCosineTolerance})");

            Assert.True(meanOneMinusCos <= OneMinusCosineTolerance,
                $"[IQ2_XS/Vulkan] cached-KV decode-step logits diverged: mean(1-cos)={meanOneMinusCos:E4} "
                + $"exceeds bound {OneMinusCosineTolerance}. Real KV cache on both sides, seqLen==1 GEMV path.");
        }
        finally
        {
            vkModel.Dispose();
            device.Dispose();
        }
    }

    private static int[] BuildPositions(int len)
    {
        var p = new int[len];
        for (int i = 0; i < len; i++) p[i] = i;
        return p;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static int Argmax(float[] xs)
    {
        int best = 0; float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int seqLen = logits.Shape.Rank == 2 ? logits.Shape[0] : 1;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static string? ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }
}
