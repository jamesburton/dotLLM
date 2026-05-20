using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the Mamba-3 <c>ForwardBatch</c> override. Mamba-3 is a pure SSM stack —
/// every layer threads a per-token recurrent state — so multi-sequence dispatch is
/// only safe when each request carries its own <see cref="VulkanMamba3State"/> via
/// <see cref="SequenceForwardRequest.MambaState"/>. The override threads each
/// request's MambaState through the SSD scan so multi-seq dispatch keeps recurrent
/// state isolated. Mirrors the <see cref="IGdnState"/> pattern from Qwen3MoeHybrid.
/// </summary>
/// <remarks>
/// Four tests cover the supported / unsupported axes:
/// <list type="number">
///   <item>Empty request list — returns empty.</item>
///   <item>Single sequence — must equal per-seq Forward exactly.</item>
///   <item>Multi-seq with NULL MambaState on any request — throws ArgumentException.</item>
///   <item>Multi-seq with per-seq MambaState — logits match running each seq through
///     Forward on a fresh model (state-isolated parity).</item>
/// </list>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelForwardBatchTests : IDisposable
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanMamba3TransformerModelForwardBatchTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-vk-fwdbatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void VulkanMamba3ForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, "m3-fwdbatch-empty.safetensors");
        VulkanMamba3TransformerModelForwardTests.WriteFixture(path, numLayers: 1, seed: 11);
        ModelConfig config = VulkanMamba3TransformerModelForwardTests.BuildConfig(numLayers: 1);

        using var sf = SafetensorsFile.Open(path);
        using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);

        var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1);
        Assert.NotNull(results);
        Assert.Empty(results);
    }

    [SkippableFact]
    public void VulkanMamba3ForwardBatch_SingleSeq_EqualsForward()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, "m3-fwdbatch-single.safetensors");
        VulkanMamba3TransformerModelForwardTests.WriteFixture(path, numLayers: 1, seed: 23);
        ModelConfig config = VulkanMamba3TransformerModelForwardTests.BuildConfig(numLayers: 1);

        int seqLen = 4;
        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VulkanMamba3TransformerModelForwardTests.VocabSize;
            positions[i] = i;
        }

        // Reference: per-seq Forward on a fresh model.
        float[] reference;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            reference = CopyLogits(logits);
        }

        // Under test: ForwardBatch with one request on a fresh model. Single-seq
        // delegates directly to Forward — MambaState slot may be null and the model
        // falls back to its model-owned default container (equivalent to the
        // reference path).
        float[] batched;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
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
                Assert.Equal(1, results[0].Shape[0]);
                Assert.Equal(VulkanMamba3TransformerModelForwardTests.VocabSize, results[0].Shape[1]);
                batched = CopyLogits(results[0]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        Assert.Equal(reference.Length, batched.Length);
        for (int c = 0; c < reference.Length; c++)
        {
            float r = reference[c];
            float a = batched[c];
            float diff = MathF.Abs(r - a);
            float bar = AbsTol + RelTol * MathF.Abs(r);
            Assert.True(diff <= bar,
                $"single-seq batch col={c}: reference={r:F6} vs batched={a:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void VulkanMamba3ForwardBatch_MultiSeq_NullMambaState_ThrowsArgument()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, "m3-fwdbatch-multi-null.safetensors");
        VulkanMamba3TransformerModelForwardTests.WriteFixture(path, numLayers: 1, seed: 37);
        ModelConfig config = VulkanMamba3TransformerModelForwardTests.BuildConfig(numLayers: 1);

        int[] tokensA = [1, 2, 3];
        int[] positionsA = [0, 1, 2];
        int[] tokensB = [5, 7, 11];
        int[] positionsB = [0, 1, 2];

        using var sf = SafetensorsFile.Open(path);
        using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
        var requests = new[]
        {
            new SequenceForwardRequest
            {
                TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(), KvCache = null!,
                // MambaState deliberately null — should trigger the guard.
            },
            new SequenceForwardRequest
            {
                TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = null!,
            },
        };

        // Multi-seq dispatch without per-seq MambaState would silently corrupt
        // recurrent state across sequences — override surfaces it loudly.
        var ex = Assert.Throws<ArgumentException>(() =>
            model.ForwardBatch(requests, deviceId: -1));
        Assert.Contains("MambaState", ex.Message);
    }

    [SkippableFact]
    public void VulkanMamba3ForwardBatch_MultiSeq_PerSeqMambaState_MatchesReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, "m3-fwdbatch-multi-perseq.safetensors");
        VulkanMamba3TransformerModelForwardTests.WriteFixture(path, numLayers: 1, seed: 41);
        ModelConfig config = VulkanMamba3TransformerModelForwardTests.BuildConfig(numLayers: 1);

        int[] tokensA = [0, 1, 2, 3];
        int[] positionsA = [0, 1, 2, 3];
        int[] tokensB = [2, 0, 1, 3];
        int[] positionsB = [0, 1, 2, 3];

        // Reference: run each sequence through a SEPARATE model instance via Forward.
        // The model-owned _recurrent gets primed by the first sequence on each
        // instance; using fresh model instances mirrors the "isolated per-seq state"
        // semantics that ForwardBatch + per-seq MambaState is meant to produce.
        float[] refA, refB;
        {
            using var sf = SafetensorsFile.Open(path);
            using var modelA = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logitsA = modelA.Forward(tokensA, positionsA, deviceId: -1);
            refA = CopyLogits(logitsA);
        }
        {
            using var sf = SafetensorsFile.Open(path);
            using var modelB = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logitsB = modelB.Forward(tokensB, positionsB, deviceId: -1);
            refB = CopyLogits(logitsB);
        }

        // Under test: single model, multi-seq ForwardBatch with per-seq MambaState
        // containers. Logits must match the reference — i.e. per-seq state is truly
        // isolated and the recurrent scan does not leak across sequences.
        float[] batA, batB;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using var stateA = model.CreateMambaState();
            using var stateB = model.CreateMambaState();
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(),
                    KvCache = null!, MambaState = stateA,
                },
                new SequenceForwardRequest
                {
                    TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(),
                    KvCache = null!, MambaState = stateB,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Equal(2, results.Count);
                batA = CopyLogits(results[0]);
                batB = CopyLogits(results[1]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        AssertLogitsClose(refA, batA, "seqA");
        AssertLogitsClose(refB, batB, "seqB");
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

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
