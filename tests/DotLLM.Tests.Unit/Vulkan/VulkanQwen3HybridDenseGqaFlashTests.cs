using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #441: the Vulkan dense hybrid (Bonsai 2) routes head_dim-256 prefill attention to the GQA-grouped tiled kernel, and the
/// whole model's logits track the pre-#441 attention path within the backend's tolerance, for both KV formats.
/// </summary>
/// <remarks>
/// The fixture is Bonsai 2's head width with PQ2_0 projections and a group of 6 query heads per KV head over 2 KV heads (12 / 2),
/// so a wrong head mapping changes the logits. The schedule prefills at offset 0, then at offsets that are not tile multiples, with
/// lengths that are not tile multiples either. Tolerance is the one the Vulkan hybrid-dense forward is held to against the CPU
/// reference (abs 5e-3 + rel 1e-3).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseGqaFlashTests : IDisposable
{
    private const int Window = 4096;
    private const int Vocab = SyntheticQwen35HybridDenseMtpGguf.VocabSize;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-gqa-flash-{Guid.NewGuid():N}");

    /// <summary>Creates the scratch directory for the fixture.</summary>
    public VulkanQwen3HybridDenseGqaFlashTests() => Directory.CreateDirectory(_scratch);

    /// <inheritdoc/>
    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>The GQA kernel engages on prefill and its logits track the pre-#441 path.</summary>
    /// <param name="storage">KV storage.</param>
    [SkippableTheory]
    [InlineData(VulkanKvStorage.F16)]
    [InlineData(VulkanKvStorage.F32)]
    public void Prefill_GqaFlash_TracksPrevious(VulkanKvStorage storage)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "m.gguf"), withMtp: false,
            contextLength: Window, blockCount: 4, pq2_0Projections: true, attentionHeads: 12, kvHeads: 2);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(256, config.HeadDim);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        model.ConfigureKvStorage(storage);

        model.GqaFlashDisabledForTest = true;
        var want = RunSchedule(model);
        Assert.Equal(0, model.GqaFlashDispatchesForTest);
        model.GqaFlashDisabledForTest = false;
        var got = RunSchedule(model);
        Skip.If(model.GqaFlashDispatchesForTest == 0, $"the GQA flash kernel is unavailable on {device.DeviceName}");

        for (int s = 0; s < want.Count; s++)
        {
            for (int i = 0; i < Vocab; i++)
            {
                float diff = MathF.Abs(want[s][i] - got[s][i]);
                Assert.True(diff <= AbsTol + RelTol * MathF.Abs(want[s][i]),
                    $"{storage} step {s} logits[{i}]: previous path {want[s][i]:R} vs GQA flash {got[s][i]:R}");
            }
        }
    }

    /// <summary>Prefills of 70, 37 and 133 rows (positions 0, 70, 107) and a decode step; returns each forward's last-row logits.</summary>
    private static List<float[]> RunSchedule(VulkanQwen3HybridDenseTransformerModel model)
    {
        model.ResetSequenceState();
        using var gdn = model.CreateGdnStateCache();
        using var kv = model.CreateKvCache(8);
        var rows = new List<float[]>();
        int pos = 0;
        foreach (int n in new[] { 70, 37, 133, 1 })
        {
            rows.Add(Forward(model, Tokens(n, salt: pos + 3), pos, kv, gdn));
            pos += n;
        }

        return rows;
    }

    private static unsafe float[] Forward(
        VulkanQwen3HybridDenseTransformerModel model, int[] tokens, int start, VulkanNemotronHKvCache kv, VulkanGdnStateCache gdn)
    {
        Assert.True(kv.TryGrowTo(start + tokens.Length));
        int[] positions = Enumerable.Range(start, tokens.Length).ToArray();
        using ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv, gdn, mtpState: null);
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * Vocab, Vocab).ToArray();
    }

    private static int[] Tokens(int n, int salt)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 1 + ((i * 7) + (salt * 5) + (i / 3)) % (Vocab - 2);
        return t;
    }
}
