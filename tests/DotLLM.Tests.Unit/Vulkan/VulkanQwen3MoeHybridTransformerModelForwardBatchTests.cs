using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the Phase 5f-mirror Qwen3MoeHybrid <c>ForwardBatch</c> override.
/// Qwen3MoeHybrid is a hybrid Gated DeltaNet + sparse MoE architecture; the GDN
/// recurrent state lives on the model as a single <see cref="VulkanGdnStateCache"/>
/// instance (not per-sequence) AND every layer routes through MoE FFN (per the
/// task partitioning, MoE-active layers force the per-seq complex path). The
/// override therefore THROWS <see cref="NotSupportedException"/> for
/// <c>requests.Count &gt;= 2</c> and only handles the trivial empty / single-seq
/// cases — same shape as the Mamba-3 host.
/// </summary>
/// <remarks>
/// <para>
/// Three tests cover the supported / unsupported axes. The single-seq parity
/// test self-skips when no Qwen3MoeHybrid GGUF is cached (set
/// <c>DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF</c> or stage the file at the conventional
/// cache path). Empty + multi-seq tests also rely on the same cached GGUF for
/// model construction.
/// </para>
/// <para>
/// The lm_head fan-out scratch is NOT shipped for this host (unlike the
/// NemotronH mirror) because the GDN single-state container would corrupt
/// per-seq Forwards in the same way as Mamba-3 — the scaffolding lands when
/// per-seq GDN state isolation is wired up.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridTransformerModelForwardBatchTests
{
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
        int[] referenceArgmaxes = new int[1];
        float referenceMax;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            (referenceArgmaxes[0], referenceMax) = ArgmaxOf(logits);
        }

        // ForwardBatch with count==1 — must delegate to Forward and produce the same
        // argmax on a freshly-built model (with freshly-empty recurrent state).
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
                    KvCache = null!, // Qwen3MoeHybrid's per-seq Forward accepts null; the GDN path doesn't use it
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

        // Single-seq batched path is a trivial delegation; argmax must match exactly.
        Assert.Equal(referenceArgmaxes[0], batchedArgmax);
        // And the max-logit value should be bit-equal (no reduction-order drift
        // because the same code path runs).
        Assert.Equal(referenceMax, batchedMax);
    }

    [SkippableFact]
    public void VulkanQwen3MoeHybridForwardBatch_MultiSeq_ThrowsNotSupported()
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

        // GDN single-state container + MoE-on-every-layer means no fusion target is
        // safe. Override throws.
        var ex = Assert.Throws<NotSupportedException>(() =>
            model.ForwardBatch(requests, deviceId: -1));
        Assert.Contains("VulkanGdnStateCache", ex.Message);
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
}
