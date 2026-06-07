using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the Qwen3MoeHybrid <c>ForwardBatch</c> override after the
/// per-sequence GDN-state plumbing landed. Qwen3MoeHybrid is a hybrid Gated
/// DeltaNet + sparse MoE architecture; the GDN recurrent state was previously
/// single-instance on the model and forced a <see cref="NotSupportedException"/>
/// on multi-seq dispatch. With <see cref="VulkanGdnStateCache"/> now per-seq
/// (passed via <see cref="SequenceForwardRequest.GdnState"/>), multi-seq dispatch
/// is correctness-safe — the override loops per-seq Forwards but each one threads
/// its own state through the GDN scan, so sequences do not corrupt each other.
/// </summary>
/// <remarks>
/// <para>
/// Four tests cover the supported axes:
/// </para>
/// <list type="number">
///   <item>Empty request list — returns empty.</item>
///   <item>Single sequence — must equal per-seq Forward.</item>
///   <item>Multi-seq without per-seq GdnState — throws ArgumentException to
///         surface the misuse (model-owned GDN cache would leak state).</item>
///   <item>Multi-seq with per-seq GdnState — logits match the reference produced
///         by two independent per-seq Forwards on fresh model instances, within
///         the dense-host tolerance envelope (abs 5e-3 / rel 1e-3).</item>
/// </list>
/// <para>
/// All tests self-skip when no Qwen3MoeHybrid GGUF is cached. Synthetic fixtures
/// for the 40+-tensor-per-layer Qwen3MoeHybrid hybrid layout are deferred (same
/// rationale as the IQ3 forward tests).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridTransformerModelForwardBatchTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void VulkanQwen3MoeHybridForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveGgufPath();
        Skip.If(path is null,
            "Qwen3.6-A3B GGUF not cached. Set DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF or stage the file " +
            "at the conventional cache path.");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
            device, gguf, config, spvDir);

        var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1);
        Assert.NotNull(results);
        Assert.Empty(results);
    }

    [SkippableFact]
    public void VulkanQwen3MoeHybridForwardBatch_SingleSeq_EqualsForward()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveGgufPath();
        Skip.If(path is null,
            "Qwen3.6-A3B GGUF not cached. Set DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF or stage the file " +
            "at the conventional cache path.");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tokenIds = [1, 100, 200, 300];
        int[] positions = [0, 1, 2, 3];

        // Per-seq Forward reference (fresh model — recurrent state begins empty).
        int referenceArgmax;
        float referenceMax;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            (referenceArgmax, referenceMax) = ArgmaxOf(logits);
        }

        // ForwardBatch with count==1 and GdnState=null delegates to the fallback
        // Forward overload using the model-owned _gdnCache — same path as Forward.
        int batchedArgmax;
        float batchedMax;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokenIds.AsMemory(),
                    Positions = positions.AsMemory(),
                    KvCache = null!,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Single(results);
                (batchedArgmax, batchedMax) = ArgmaxOf(results[0]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        // Single-seq batched delegates to Forward; argmax + max-logit must match exactly.
        Assert.Equal(referenceArgmax, batchedArgmax);
        Assert.Equal(referenceMax, batchedMax);
    }

    [SkippableFact]
    public void VulkanQwen3MoeHybridForwardBatch_MultiSeq_NullGdnState_ThrowsArgument()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveGgufPath();
        Skip.If(path is null,
            "Qwen3.6-A3B GGUF not cached. Set DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF or stage the file " +
            "at the conventional cache path.");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
            device, gguf, config, spvDir);

        int[] tokensA = [1, 100, 200];
        int[] positionsA = [0, 1, 2];
        int[] tokensB = [5, 50, 500];
        int[] positionsB = [0, 1, 2];

        var requests = new[]
        {
            new SequenceForwardRequest
            {
                TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(), KvCache = null!,
            },
            new SequenceForwardRequest
            {
                TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = null!,
            },
        };

        // Without per-seq GdnState, the model-owned cache would leak between sequences.
        // The override rejects this with an ArgumentException naming the missing slot.
        var ex = Assert.Throws<ArgumentException>(() =>
            model.ForwardBatch(requests, deviceId: -1));
        Assert.Contains("GdnState", ex.Message);
    }

    [SkippableFact]
    public void VulkanQwen3MoeHybridForwardBatch_MultiSeq_PerSeqGdnState_MatchesReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveGgufPath();
        Skip.If(path is null,
            "Qwen3.6-A3B GGUF not cached. Set DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF or stage the file " +
            "at the conventional cache path.");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tokensA = [1, 100, 200];
        int[] positionsA = [0, 1, 2];
        int[] tokensB = [5, 50, 500];
        int[] positionsB = [0, 1, 2];

        // Reference: two fresh-model Forwards (each with its own implicit per-model
        // GDN cache). Captures the per-seq logits produced when no cross-sequence
        // state corruption can occur.
        float[] referenceA, referenceB;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            using ITensor logitsA = model.Forward(tokensA, positionsA, deviceId: -1);
            referenceA = CopyLogits(logitsA);
        }
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            using ITensor logitsB = model.Forward(tokensB, positionsB, deviceId: -1);
            referenceB = CopyLogits(logitsB);
        }

        // Under test: a single model running ForwardBatch with two requests, each
        // carrying its own VulkanGdnStateCache. The per-seq state isolation means
        // sequence A's GDN scan cannot contaminate sequence B's logits.
        float[] batchedA, batchedB;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            using var gdnA = model.CreateGdnStateCache();
            using var gdnB = model.CreateGdnStateCache();
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokensA.AsMemory(),
                    Positions = positionsA.AsMemory(),
                    KvCache = null!,
                    GdnState = gdnA,
                },
                new SequenceForwardRequest
                {
                    TokenIds = tokensB.AsMemory(),
                    Positions = positionsB.AsMemory(),
                    KvCache = null!,
                    GdnState = gdnB,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Equal(2, results.Count);
                batchedA = CopyLogits(results[0]);
                batchedB = CopyLogits(results[1]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        AssertLogitsClose(referenceA, batchedA, "seqA");
        AssertLogitsClose(referenceB, batchedB, "seqB");
    }

    private static void AssertLogitsClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        for (int c = 0; c < reference.Length; c++)
        {
            float r = reference[c];
            float a = actual[c];
            float diff = MathF.Abs(r - a);
            float bar = AbsTol + RelTol * MathF.Abs(r);
            Assert.True(diff <= bar,
                $"{label} col={c}: reference={r:F6} vs batched={a:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static string? ResolveGgufPath()
    {
        // Conventional cache path (Strix Halo dev box) — the Qwen3.6-A3B GGUF is too
        // large (~28 GB) to keep in the repo or CI. The cache convention matches the
        // integration smoke test (Qwen3MoeHybridQ6KResidentSmokeTests).
        string? env = Environment.GetEnvironmentVariable("DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF");
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

        string conventional = "C:/Users/james/.dotllm/test-cache/unsloth/Qwen3.6-A3B-GGUF/Qwen3.6-A3B-UD-Q6_K_XL.gguf";
        if (File.Exists(conventional)) return conventional;
        return null;
    }

    private static unsafe (int idx, float val) ArgmaxOf(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
        int bestIdx = 0;
        float bestVal = span[0];
        for (int i = 1; i < total; i++)
        {
            if (span[i] > bestVal) { bestVal = span[i]; bestIdx = i; }
        }
        return (bestIdx, bestVal);
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
