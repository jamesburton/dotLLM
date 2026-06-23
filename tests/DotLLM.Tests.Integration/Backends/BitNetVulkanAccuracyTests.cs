using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using System.Numerics.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// End-to-end (model-level) accuracy gate for the BitNet b1.58 (I2_S ternary) Vulkan forward
/// against the CPU golden reference. Complements the kernel-level Vulkan I2_S GEMV/GEMM parity
/// tests by exercising the full integrated path: I2_S weight loading (raw on-device + tail scale),
/// the I2_S matmul dispatch (GEMV decode + GEMM prefill), the BitNet Sub-LN RMSNorms (attn/ffn),
/// and the gated squared-ReLU activation — all wired into <see cref="VulkanTransformerModel"/>.
/// </summary>
/// <remarks>
/// <see cref="SkippableFact"/> — skips when <c>DOTLLM_BITNET_GGUF</c> is unset/missing or no Vulkan
/// device is present. On the display-driving Arc iGPU set <c>DOTLLM_VULKAN_DEVICE_VENDOR=0x8086</c>;
/// these prompts are short so they complete well inside the 2s TDR window. The CPU (FP32) and Vulkan
/// (FP32-compute) I2_S forwards are independent implementations, so the gate is argmax match +
/// cosine &gt; 0.999 — the same bar as the CPU↔CUDA gate in
/// <c>BitNetAccuracyTests.CpuVsCuda_LastTokenLogits_Match</c>.
/// </remarks>
public sealed class BitNetVulkanAccuracyTests
{
    private readonly ITestOutputHelper _output;

    public BitNetVulkanAccuracyTests(ITestOutputHelper output) => _output = output;

    private static string? ModelPath =>
        Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF");

    private const string ParityPrompt = "The capital of France is";

    [SkippableFact]
    public unsafe void CpuVsVulkan_Prefill_LastTokenLogits_Match()
    {
        Skip.If(ModelPath is null || !File.Exists(ModelPath), "BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;

        int[] tokenIds = tokenizer.Encode(ParityPrompt);
        int[] positions = Positions(tokenIds.Length);
        Assert.True(tokenIds.Length > 1, "Prefill parity needs a multi-token prompt (exercises the I2_S GEMM path).");

        // ── CPU reference (golden) — full [seqLen, vocab]; take the last row. ──
        float[] cpuVec = new float[vocab];
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, config))
        using (ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1))
        {
            long lastRow = (long)(tokenIds.Length - 1) * vocab;
            new ReadOnlySpan<float>((float*)cpuLogits.DataPointer + lastRow, vocab).CopyTo(cpuVec);
        }

        // ── Vulkan under test — the multi-token prefill exercises the I2_S GEMM path;
        //    the causal forward returns the last token's logits only ([1, vocab]). ──
        string spvDir = ResolveSpvDir();
        float[] vkVec = new float[vocab];
        using (var vkModel = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir))
        using (ITensor vkLogits = vkModel.Forward(tokenIds, positions, -1))
        {
            Assert.Equal(vocab, vkLogits.Shape[1]);
            new ReadOnlySpan<float>((float*)vkLogits.DataPointer, vocab).CopyTo(vkVec);
        }

        AssertLogitParity(cpuVec, vkVec, tokenizer, "prefill/GEMM");
    }

    [SkippableFact]
    public unsafe void CpuVsVulkan_SingleToken_Logits_Match()
    {
        Skip.If(ModelPath is null || !File.Exists(ModelPath), "BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;

        // A single-token forward drives RecordMatmul's seqLen==1 branch end-to-end:
        // the I2_S GEMV decode path through every projection of the full model.
        int[] tokenIds = [tokenizer.Encode(ParityPrompt)[0]];
        int[] positions = [0];

        float[] cpuVec = new float[vocab];
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, config))
        using (ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1))
        {
            new ReadOnlySpan<float>((float*)cpuLogits.DataPointer, vocab).CopyTo(cpuVec);
        }

        string spvDir = ResolveSpvDir();
        float[] vkVec = new float[vocab];
        using (var vkModel = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir))
        using (ITensor vkLogits = vkModel.Forward(tokenIds, positions, -1))
        {
            new ReadOnlySpan<float>((float*)vkLogits.DataPointer, vocab).CopyTo(vkVec);
        }

        AssertLogitParity(cpuVec, vkVec, tokenizer, "decode/GEMV");
    }

    [SkippableFact]
    public unsafe void CpuVsVulkan_BatchedPrefill_LastTokenLogits_Match()
    {
        Skip.If(ModelPath is null || !File.Exists(ModelPath), "BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;

        // Two distinct prompts ⇒ ≥2 "simple" requests, which routes ForwardBatch through
        // ForwardBatchSimple (the fused multi-sequence batched path) rather than per-seq
        // Forward — the only path that exercises that path's I2_S matmuls + Sub-LN + ReLU².
        string[] prompts = ["The capital of France is", "The sun rises in the"];
        int[][] tokens = [tokenizer.Encode(prompts[0]), tokenizer.Encode(prompts[1])];
        foreach (var t in tokens) Assert.True(t.Length > 1, "each batched prompt must be multi-token.");

        // ── CPU reference (golden), per sequence. ──
        float[][] cpuVecs = new float[prompts.Length][];
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, config))
        {
            for (int s = 0; s < prompts.Length; s++)
            {
                cpuVecs[s] = new float[vocab];
                using ITensor cpuLogits = cpuModel.Forward(tokens[s], Positions(tokens[s].Length), deviceId: -1);
                long lastRow = (long)(tokens[s].Length - 1) * vocab;
                new ReadOnlySpan<float>((float*)cpuLogits.DataPointer + lastRow, vocab).CopyTo(cpuVecs[s]);
            }
        }

        // ── Vulkan ForwardBatch (fused batched path under test). ──
        string spvDir = ResolveSpvDir();
        float[][] vkVecs = new float[prompts.Length][];
        using (var vkModel = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir))
        {
            var caches = new IKvCache[prompts.Length];
            try
            {
                var requests = new SequenceForwardRequest[prompts.Length];
                for (int s = 0; s < prompts.Length; s++)
                {
                    caches[s] = vkModel.CreateKvCache(maxSeqLen: tokens[s].Length + 1);
                    requests[s] = new SequenceForwardRequest
                    {
                        TokenIds = tokens[s],
                        Positions = Positions(tokens[s].Length),
                        KvCache = caches[s],
                    };
                }

                IReadOnlyList<ITensor> results = vkModel.ForwardBatch(requests, deviceId: -1);
                Assert.Equal(prompts.Length, results.Count);
                for (int s = 0; s < prompts.Length; s++)
                {
                    vkVecs[s] = new float[vocab];
                    new ReadOnlySpan<float>((float*)results[s].DataPointer, vocab).CopyTo(vkVecs[s]);
                    results[s].Dispose();
                }
            }
            finally
            {
                foreach (var c in caches) c?.Dispose();
            }
        }

        for (int s = 0; s < prompts.Length; s++)
            AssertLogitParity(cpuVecs[s], vkVecs[s], tokenizer, $"batched seq {s} ('{prompts[s]}')");
    }

    // ── Helpers ──

    private void AssertLogitParity(float[] cpuVec, float[] vkVec, BpeTokenizer tokenizer, string label)
    {
        int cpuArgmax = ArgMax(cpuVec);
        int vkArgmax = ArgMax(vkVec);
        double cosine = CosineSimilarity(cpuVec, vkVec);
        (float maxAbs, float meanAbs) = AbsDiff(cpuVec, vkVec);

        _output.WriteLine($"[{label}] CPU argmax={cpuArgmax} ('{tokenizer.DecodeToken(cpuArgmax).Trim()}')  "
            + $"Vulkan argmax={vkArgmax} ('{tokenizer.DecodeToken(vkArgmax).Trim()}')");
        _output.WriteLine($"[{label}] cosine(cpu, vulkan)={cosine:F6}  max|Δ|={maxAbs:F4}  mean|Δ|={meanAbs:F4}");

        Assert.True(cpuArgmax == vkArgmax,
            $"BitNet CPU/Vulkan argmax mismatch ({label}): CPU={cpuArgmax} Vulkan={vkArgmax} (cosine={cosine:F6}). "
            + "Indicates an I2_S forward divergence between the CPU and Vulkan paths.");
        Assert.True(cosine > 0.999,
            $"BitNet CPU/Vulkan cosine {cosine:F6} below 0.999 ({label}) — significant end-to-end divergence.");
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var cand in candidates)
        {
            string full = Path.GetFullPath(cand);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }

    private static int[] Positions(int count)
    {
        int[] p = new int[count];
        for (int i = 0; i < count; i++) p[i] = i;
        return p;
    }

    private static int ArgMax(float[] vec)
        => TensorPrimitives.IndexOfMax(new ReadOnlySpan<float>(vec));

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    private static (float maxAbs, float meanAbs) AbsDiff(float[] a, float[] b)
    {
        float maxAbs = 0;
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxAbs) maxAbs = d;
            sum += d;
        }
        return (maxAbs, (float)(sum / a.Length));
    }
}
