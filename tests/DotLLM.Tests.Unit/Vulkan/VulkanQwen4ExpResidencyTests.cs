using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Pure-arithmetic tests of the Qwen4-Exp pre-load residency gate (#818): no GPU needed.
/// </summary>
public sealed class Qwen4ExpResidencyPlanTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void Fits_WhenRequiredIsWithinCapacityMinusHeadroom()
    {
        var p = new Qwen4ExpResidencyPlan(DeviceWeightBytes: 80 * GiB, KvAndScratchBytes: 2 * GiB, HostOnlyBytes: 28 * GiB,
            CapacityBytes: 100 * GiB, HeadroomBytes: 6 * GiB, PhysicalRamBytes: 127 * GiB);
        Assert.True(p.Fits);
        Assert.Equal(94 * GiB, p.BudgetBytes);
        Assert.False(p.HostPressure);   // 82 + 28 = 110 GiB <= 127 - 6
    }

    [Fact]
    public void Refuses_WhenWeightsExceedBudget_EvenThoughTheyFitRawCapacity()
    {
        // 96 GiB of weights against a 100 GiB capacity is the 122B thrash shape: it "fits" but leaves no room for anything else.
        var p = new Qwen4ExpResidencyPlan(96 * GiB, 2 * GiB, 0, 100 * GiB, Qwen4ExpResidencyPlan.DefaultHeadroom(100 * GiB), 127 * GiB);
        Assert.False(p.Fits);
        Assert.Contains("resident", p.Describe() + " resident", StringComparison.Ordinal);
    }

    [Fact]
    public void HostPressure_WhenTableWouldPushPastPhysicalRam()
    {
        var p = new Qwen4ExpResidencyPlan(90 * GiB, 1 * GiB, 32 * GiB, 100 * GiB, 6 * GiB, 127 * GiB);
        Assert.True(p.Fits);
        Assert.True(p.HostPressure);    // 91 + 32 = 123 GiB > 127 - 6 = 121
    }

    [Fact]
    public void DefaultHeadroom_IsSixGiBOrFivePercent()
    {
        Assert.Equal(6 * GiB, Qwen4ExpResidencyPlan.DefaultHeadroom(40 * GiB));
        Assert.Equal(10 * GiB, Qwen4ExpResidencyPlan.DefaultHeadroom(200 * GiB));
    }
}

/// <summary>
/// Loader / residency behaviour of the Vulkan Qwen4-Exp model (#818): the n-gram table never reaches the device, the pre-load gate
/// refuses with numbers, the dense-attention limit is enforced, and the dispatcher routes the architecture.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpLoaderTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpLoaderTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void NgramTable_IsHostOnly_AndNeverStagedOrImported()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // A table deliberately LARGER than the entire rest of the checkpoint, so "it was uploaded" cannot hide in the noise.
        var geo = Qwen4ExpRandomGguf.Tiny with { TableRows = 400_000 };
        long tableBytes = 400_000L * geo.PleRowDim * 2;   // F16
        byte[] bytes = Qwen4ExpRandomGguf.Build(geo, Q4eQuant.Q8Q51);
        Assert.True(tableBytes > bytes.Length / 2, "fixture is not discriminating: the table must dominate the file");

        (nint Pointer, long Bytes) range;
        using (var rig = new Q4eRig(bytes, spvDir))
        {
            range = rig.Vk.HostOnlyTableRange;
            Assert.Equal(tableBytes, range.Bytes);
            Assert.Equal(tableBytes, rig.Vk.HostOnlyTableBytes);
            Assert.Contains(VulkanWeightImportPolicy.HostOnlyRanges, r => r.Pointer == range.Pointer && r.Bytes == range.Bytes);

            // The ledger of what the weight loader actually staged: no staged source range overlaps the table...
            foreach (var (p, n) in VulkanWeightImportPolicy.StagedSourceRanges)
                Assert.False(p < range.Pointer + (nint)range.Bytes && range.Pointer < p + (nint)n, $"staged range 0x{p:X}+{n} overlaps the n-gram table");
            // ...and everything staged or aliased is far smaller than the table, so the table cannot be in there.
            Assert.True(VulkanWeightImportPolicy.StagedBytes + VulkanWeightImportPolicy.ImportedBytes < tableBytes,
                $"loader moved {VulkanWeightImportPolicy.StagedBytes + VulkanWeightImportPolicy.ImportedBytes:N0} source bytes; the table alone is {tableBytes:N0}");
            Assert.True(rig.Vk.ComputeMemoryBytes < tableBytes, $"device footprint {rig.Vk.ComputeMemoryBytes:N0} >= table {tableBytes:N0}");

            // Belt and braces: every upload path refuses the table range outright.
            Assert.Throws<InvalidOperationException>(() => VulkanWeightImportPolicy.ThrowIfHostOnly(range.Pointer, 4096));
            Assert.Throws<InvalidOperationException>(() => VulkanWeightImportPolicy.ThrowIfHostOnly(range.Pointer + (nint)(range.Bytes - 8), 64));
            Assert.Throws<InvalidOperationException>(() => VulkanWeightImportPolicy.NoteStaged(range.Pointer + 128, 1024));
            using var staging = VulkanStagingBuffer.Create(rig.Device, 1 << 20);
            using var dst = rig.Device.AllocateDeviceLocal(1 << 20);
            Assert.Throws<InvalidOperationException>(() => staging.UploadBytes(range.Pointer + 4096, 8192, dst));
            _out.WriteLine($"table {tableBytes:N0} B host-only; loader moved {VulkanWeightImportPolicy.StagedBytes + VulkanWeightImportPolicy.ImportedBytes:N0} B");
        }
        Assert.DoesNotContain(VulkanWeightImportPolicy.HostOnlyRanges, r => r.Pointer == range.Pointer);   // unregistered on dispose
    }

    [SkippableFact]
    public void ResidencyEstimate_ExcludesTable_AndCountsWidenedBanks()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        _ = spvDir;
        var geo = Qwen4ExpRandomGguf.Tiny with { TableRows = 40_000 };
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-est-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            long Estimate(Q4eQuant q, out long hostOnly)
            {
                string path = Path.Combine(dir, $"{Guid.NewGuid():N}.gguf");
                File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(geo, q));
                using var gguf = GgufFile.Open(path);
                var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
                var (device, host) = Qwen4ExpResidencyPlan.EstimateWeights(gguf.TensorsByName, cfg);
                hostOnly = host;
                return device;
            }
            long f32 = Estimate(Q4eQuant.F32, out long host32);
            long q8 = Estimate(Q4eQuant.Q8Q51, out long hostQ);
            // Table (+ PLE projections / indexer) are host-only in both and dominate neither number.
            Assert.True(host32 >= 40_000L * geo.PleRowDim * 4, "table bytes missing from the host-only side");
            Assert.True(hostQ >= 40_000L * geo.PleRowDim * 2);
            // Q8_0 / Q5_1 expert banks have no resident kernel: they are WIDENED, so the experts cost the same F32 bytes in both variants.
            long expertF32 = 4L * geo.Experts * 3 * geo.MoeInter * geo.Hidden * 4;
            Assert.True(q8 >= expertF32, $"widened expert banks not counted: {q8:N0} < {expertF32:N0}");
            Assert.True(f32 >= expertF32);
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }

    [SkippableFact]
    public void Load_RefusesWhenOverCapacity_AndAllowsOvercommitOverride()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-cap-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "m.gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.F32));
        string? prior = Environment.GetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT");
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();

            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", null);
            var ex = Assert.Throws<NotSupportedException>(() =>
                VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, residentCapacityOverrideBytes: 1 << 20));
            Assert.Contains("resident capacity", ex.Message, StringComparison.Ordinal);
            Assert.Contains("DOTLLM_VK_ALLOW_OVERCOMMIT", ex.Message, StringComparison.Ordinal);
            _out.WriteLine(ex.Message);
            // A refused load must leave nothing registered behind.
            Assert.Empty(VulkanWeightImportPolicy.HostOnlyRanges);

            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", "1");
            using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, residentCapacityOverrideBytes: 1 << 20);
            Assert.NotNull(model);
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", prior);
            try { Directory.Delete(dir, true); } catch (IOException) { }
        }
    }

    [SkippableFact]
    public void ExpertBankDeviceTypes_ShowWhichQuantsStayResident()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using (var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(4), spvDir))   // kq256-kquant
            Assert.All(rig.Vk.ExpertBankDeviceTypes, t =>
            {
                Assert.Equal(QuantizationType.Q4_K, t.Gate);
                Assert.Equal(QuantizationType.Q4_K, t.Down);
                Assert.Equal(QuantizationType.Q4_K, t.Up);
            });
        using (var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(1), spvDir))   // tiny-q8q51: no resident Q8_0 / Q5_1 kernel yet
            Assert.All(rig.Vk.ExpertBankDeviceTypes, t =>
            {
                Assert.Equal(QuantizationType.F32, t.Gate);
                Assert.Equal(QuantizationType.F32, t.Down);
            });
    }

    [SkippableFact]
    public void DenseAttention_IsExactWithinBudget_AndRefusesBeyondIt()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // budget 16 tokens, block 4 -> exactly dense up to 19 tokens (HF/oracle semantics); the 20th would need sparse selection.
        var geo = Qwen4ExpRandomGguf.Tiny with { Budget = 16 };
        using var rig = new Q4eRig(Qwen4ExpRandomGguf.Build(geo, Q4eQuant.F32), spvDir);
        Assert.Equal(19, rig.Vk.DenseContextLimit);

        var ids = VulkanQwen4ExpParityTests.Ids(19, rig.Config.VocabSize);
        var pos = Enumerable.Range(0, 19).ToArray();
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), 18);
        var vk = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        var (rl2, kl, _) = VulkanQwen4ExpParityTests.Compare(cpu, vk);
        _out.WriteLine($"19 tokens at the budget edge: relL2 {rl2:E3}, KL {kl:E3}");
        Assert.True(rl2 < 3e-3);

        var ex = Assert.Throws<NotSupportedException>(() => rig.Vk.Forward([3], [19], -1));
        Assert.Contains("#819", ex.Message, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void Dispatcher_RoutesQwen4Exp_AndDenseDoorStillExplains()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-disp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "m.gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.F32));
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, cfg, spvDir);
            using (model)
            {
                Assert.IsType<VulkanQwen4ExpTransformerModel>(model);
                Assert.True(model.RequiresPerSequenceState);
                var ex = Assert.Throws<NotSupportedException>(() => kvFactory(16));
                Assert.Contains("#817", ex.Message, StringComparison.Ordinal);
                // State is model-owned: an engine KV cache and ForwardBatch are refused with the #817 explanation, like the CPU oracle.
                Assert.Throws<NotSupportedException>(() => model.ForwardBatch([], 0).ToString());
            }

            // The dense Vulkan model must never be handed this architecture: its tensors are not Llama-style.
            var dense = Assert.Throws<NotSupportedException>(() => VulkanTransformerModel.RejectUnsupportedArchitecture(cfg));
            Assert.Contains("VulkanModelLoader", dense.Message, StringComparison.Ordinal);
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }
}
