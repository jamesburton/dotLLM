using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the Phase 5f-mirror Mamba-3 <c>ForwardBatch</c> override. Mamba-3 is a pure
/// SSM stack — every layer threads a per-token recurrent state — and the model's
/// <c>VulkanMamba3State</c> is single-instance (not per-sequence). Multi-sequence
/// dispatch is therefore not safe today (per-seq Forwards would corrupt the recurrent
/// state across sequences), and the override throws <see cref="NotSupportedException"/>
/// for <c>requests.Count &gt;= 2</c>. The override DOES handle empty / single-seq
/// requests so the public API shape matches the dense / NemotronH hosts and so the
/// scheduler can call it uniformly across architectures.
/// </summary>
/// <remarks>
/// <para>
/// Three tests cover the supported / unsupported axes:
/// <list type="number">
///   <item>Empty request list — returns empty.</item>
///   <item>Single sequence — must equal per-seq Forward exactly.</item>
///   <item>Multi-seq request — throws NotSupportedException with the documented message.</item>
/// </list>
/// </para>
/// <para>
/// The lm_head fan-out scratch infrastructure (the <c>captureLastNormedRowTo</c> hook on
/// the model's internal <c>RunFinalNormAndLmHead</c>) is ready for the follow-up that
/// introduces per-sequence recurrent-state isolation; this test class will gain
/// multi-seq parity tests at that point.
/// </para>
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

        // Under test: ForwardBatch with one request on a fresh model. Must equal Forward
        // bit-exactly — the override delegates count==1 directly to Forward without
        // touching the lm_head fan-out path.
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
                    KvCache = null!, // Mamba-3 ignores kvCache (uses recurrent state); required by record contract though
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
    public void VulkanMamba3ForwardBatch_MultiSeq_ThrowsNotSupported()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, "m3-fwdbatch-multi.safetensors");
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
            },
            new SequenceForwardRequest
            {
                TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = null!,
            },
        };

        // Pure SSM stack with a single shared recurrent state means multi-seq batched
        // dispatch would corrupt state across sequences. Override is documented to throw.
        var ex = Assert.Throws<NotSupportedException>(() =>
            model.ForwardBatch(requests, deviceId: -1));
        Assert.Contains("VulkanMamba3State", ex.Message);
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
