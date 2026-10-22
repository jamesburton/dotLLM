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
    public unsafe void SlotsHoldExactlyTheFileBytesOfTheirExpert_AfterDecodeSteps()
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
            using var tiered = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, null, null, expertSlots: 16);
            var tier = tiered.ExpertTier!;
            using var st = tiered.CreateState();
            var ids = VulkanQwen4ExpParityTests.Ids(6, cfg.VocabSize, seed: 5);
            for (int i = 0; i < ids.Length; i++) StepLogits(tiered, st, ids[i], i);

            int checkedSlots = 0;
            for (int il = 0; il < tier.Layers.Count; il++)
            {
                var L = tier.Layers[il];
                string[] names = [$"blk.{il}.ffn_gate_exps.weight", $"blk.{il}.ffn_up_exps.weight", $"blk.{il}.ffn_down_exps.weight"];
                for (int p = 0; p < 3; p++)
                {
                    long bank = L.SlotBanks[p].Size;
                    var host = new byte[bank];
                    var floats = new float[(bank + 3) / 4];
                    device.Download(L.SlotBanks[p], floats.AsSpan());
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(floats.AsSpan()).Slice(0, (int)bank).CopyTo(host);
                    nint src = gguf.TensorDataPointer(names[p]);
                    for (int slot = 0; slot < L.Cache.Slots; slot++)
                    {
                        int e = L.Cache.ExpertOf(slot);
                        if (e < 0) continue;
                        var want = new ReadOnlySpan<byte>((void*)(src + (nint)(e * L.PieceBytes[p])), (int)L.PieceBytes[p]);
                        var got = host.AsSpan((int)(slot * L.PieceBytes[p]), (int)L.PieceBytes[p]);
                        if (!got.SequenceEqual(want))
                        {
                            int firstDiff = 0; while (firstDiff < got.Length && got[firstDiff] == want[firstDiff]) firstDiff++;
                            bool zeros = true; foreach (byte bb in got) if (bb != 0) { zeros = false; break; }
                            int match = -1;
                            for (int e2 = 0; e2 < cfg.Moe!.NumExperts && match < 0; e2++)
                                if (got.SequenceEqual(new ReadOnlySpan<byte>((void*)(src + (nint)(e2 * L.PieceBytes[p])), (int)L.PieceBytes[p]))) match = e2;
                            // shifted match: does the slot hold the wanted bytes displaced by some delta?
                            int shift = int.MinValue;
                            var wantArr = want.ToArray();
                            for (int sh = -8192; sh <= 8192 && shift == int.MinValue; sh++)
                            {
                                int a0 = Math.Max(0, sh), n = Math.Min(got.Length, got.Length + sh) - a0 - 8;
                                if (n > 64 && got.Slice(a0, 64).SequenceEqual(wantArr.AsSpan(a0 - sh, 64))) shift = sh;
                            }
                            Assert.Fail($"layer {il} piece {p}: slot {slot} does not hold expert {e}'s bytes (len {got.Length}, first diff at {firstDiff}, all zeros: {zeros}, equals expert {match}, shift {(shift == int.MinValue ? "none" : shift.ToString())}, got[0..8]={BitConverter.ToString(got.Slice(0, 8).ToArray())} want[0..8]={BitConverter.ToString(wantArr, 0, 8)})");
                        }
                        checkedSlots++;
                    }
                }
            }
            _out.WriteLine($"{checkedSlots} (layer, piece, slot) regions verified byte-for-byte against the file");
            Assert.True(checkedSlots > 0);
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
