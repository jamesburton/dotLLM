using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan.Tiering;

/// <summary>
/// Expert tiering end to end on a synthetic qwen4exp checkpoint (#939): the routed experts live in the file, a small slot pool per layer caches them, and the
/// logits must be BIT-IDENTICAL to the fully resident model run through the same single-row path (same kernels, same bytes, only slot numbers differ).
/// Counters prove the tiered path ran: a model that ignored the tier would read empty slot banks and produce different logits, and a stuck cache would show no misses.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpExpertTierTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpExpertTierTests(ITestOutputHelper output) => _out = output;

    private static float[] StepLogits(VulkanQwen4ExpTransformerModel m, VulkanQwen4ExpSequenceState st, int token, int pos)
        => Q4eRig.Row(m.Forward([token], [pos], -1, st), 0);

    [SkippableTheory]
    [InlineData(11, 24)]   // 512 experts, Q4_K gate/up + Q5_1 down: the fused single-token chain
    [InlineData(11, 12)]   // tighter pool: more evictions
    public void TieredDecode_IsBitIdenticalToResident_AndReallyMissedAndEvicted(int variant, int slots)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-tier-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "q4e.gguf");
            File.WriteAllBytes(path, VulkanQwen4ExpParityTests.Build(variant));
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var resident = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, null, null, expertSlots: 0);
            using var tiered = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, null, null, expertSlots: slots);
            Assert.False(resident.ExpertsTiered);
            Assert.True(tiered.ExpertsTiered);
            Assert.False(tiered.CanAttachMtpHead);
            var tier = tiered.ExpertTier!;
            Assert.Equal(slots, tier.Slots);

            const int T = 40;
            var ids = VulkanQwen4ExpParityTests.Ids(T, cfg.VocabSize, seed: 91);
            using var rs = resident.CreateState();
            using var ts = tiered.CreateState();
            for (int i = 0; i < T; i++)
            {
                var a = StepLogits(resident, rs, ids[i], i);
                var b = StepLogits(tiered, ts, ids[i], i);
                int differing = 0;
                for (int k = 0; k < a.Length; k++) if (BitConverter.SingleToInt32Bits(a[k]) != BitConverter.SingleToInt32Bits(b[k])) differing++;
                Assert.True(differing == 0, $"step {i}: {differing} of {a.Length} logits differ between the resident and the tiered model (must be bit-identical)");
            }
            _out.WriteLine(tiered.ExpertTierSummary);
            int layers = cfg.NumLayers;
            Assert.Equal((long)T * layers, tier.LayerVisits);                      // one host step per MoE layer per token
            Assert.Equal(tier.LayerVisits * cfg.Moe!.NumExpertsPerTok, tier.Hits + tier.Misses);
            Assert.True(tier.Misses > 0, "the tiered model never loaded an expert from the file");
            Assert.True(tier.Hits > 0, "the slot cache never served a lookup (cache not retaining anything)");
            Assert.True(tier.Evictions > 0, "no eviction happened: the pool was never under pressure, so the LRU path is untested");
            Assert.True(tier.BytesRead > 0);
            // Each miss loads three pieces; with the sector over-read the file bytes read are at least the expert bytes.
            long expertBytes = tier.Layers.Sum(l => l.PieceBytes.Sum());
            Assert.True(tier.BytesRead >= tier.Misses * (expertBytes / layers), "fewer bytes were read than the missed experts occupy");
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }

    [SkippableFact]
    public void ResidentModel_RecordsNoTierWork()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        Assert.False(rig.Vk.ExpertsTiered);
        Assert.Null(rig.Vk.ExpertTier);
        Assert.True(rig.Vk.CanAttachMtpHead);
        Assert.Null(rig.Vk.ExpertTierSummary);
    }

    [SkippableFact]
    public void MoreSlotsThanExperts_IsClamped_AndFewerThanTopKIsRefused()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-tier-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "q4e.gguf");
            File.WriteAllBytes(path, VulkanQwen4ExpParityTests.Build(11));
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, null, null, expertSlots: cfg.Moe!.NumExpertsPerTok - 1));
            using var all = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, null, null, expertSlots: 100000);
            Assert.Equal(cfg.Moe!.NumExperts, all.ExpertTier!.Slots);
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }
}
