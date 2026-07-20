using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.Scheduler;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Validates the cross-device KV handoff (<see cref="StagedKvHandoffTransfer"/>, #360) across TWO physical
/// Vulkan devices: a prefill cache on device 0, a decode cache on device 1, contents staged
/// device→host→device over the <see cref="IHostStagedKvCache"/> seam. This is the dual-GPU proof for the
/// Vulkan backend — the sibling of <c>CudaCrossDeviceKvTransferTests</c> (proven on Kaggle T4 ×2, #361).
/// It runs on the Framework box (GPU0 = NVIDIA RTX 3060 discrete, GPU1 = Intel Arc integrated), exercising
/// a genuine discrete-VRAM → host → integrated-UMA transfer across vendors, and skips everywhere with &lt; 2
/// Vulkan devices (Strix Halo = single iGPU; T5500 = single RTX 3060).
/// </summary>
/// <remarks>
/// Vulkan KV storage is FP32 device-local (no FP16 round-trip, unlike the CUDA cache), so the post-transfer
/// compare is exact for any value; the fill still uses small distinct integers so a mismatch localizes to a
/// specific (layer, position, channel). Device 0 and 1 are bound independently via
/// <see cref="VulkanDevice.Create(int)"/> — the per-replica placement #360 added so one process can host
/// prefill and decode on different GPUs behind the <see cref="DisaggregatedScheduler"/> handoff seam.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCrossDeviceKvTransferTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int Stride = NumKvHeads * HeadDim;
    private const int MaxSeqLen = 64;
    private const int Length = 7;

    private readonly ITestOutputHelper _out;

    public VulkanCrossDeviceKvTransferTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void StagedTransfer_AcrossTwoVulkanDevices_PreservesContents()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2,
            "Cross-device transfer needs >= 2 Vulkan physical devices (e.g. Framework iGPU + RTX 3060)");

        // Prefill on device 0, decode on device 1 — explicit per-replica placement (#360).
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"prefill device 0: {dev0.DeviceName} (type={dev0.DeviceType}, vendor=0x{dev0.VendorId:X4})");
        _out.WriteLine($"decode  device 1: {dev1.DeviceName} (type={dev1.DeviceType}, vendor=0x{dev1.VendorId:X4})");

        var geom = KvGeometry.Uniform(NumLayers, NumKvHeads, HeadDim);

        // Expected per-layer K/V — small distinct integers so a mismatch points at (layer, position, channel).
        var expectedK = new float[NumLayers][];
        var expectedV = new float[NumLayers][];
        for (int l = 0; l < NumLayers; l++)
        {
            expectedK[l] = new float[Length * Stride];
            expectedV[l] = new float[Length * Stride];
            for (int p = 0; p < Length; p++)
                for (int d = 0; d < Stride; d++)
                {
                    float val = l * 1000 + p * 10 + d;
                    expectedK[l][p * Stride + d] = val;
                    expectedV[l][p * Stride + d] = -val;
                }
        }

        // Build + fill the prefill cache on device 0. Ownership transfers to
        // StagedKvHandoffTransfer.Transfer below (it disposes source); the `using`
        // here is a belt-and-suspenders idempotent second Dispose on scope exit.
        using var source = new VulkanKvCache(dev0, geom, MaxSeqLen);
        for (int l = 0; l < NumLayers; l++)
            source.UploadLayer(l, Length, expectedK[l], expectedV[l]);
        Assert.Equal(Length, source.CurrentLength);

        var config = new ModelConfig
        {
            VocabSize = 32,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = Architecture.Llama,
        };

        // Hand the sequence off to device 1 via the staged transfer (device0 → host → device1). Disposes source.
        using IKvCache dest = StagedKvHandoffTransfer.Instance.Transfer(
            source, config, (_, maxSeq) => new VulkanKvCache(dev1, geom, maxSeq));

        Assert.Equal(Length, dest.CurrentLength);
        var staged = Assert.IsAssignableFrom<IHostStagedKvCache>(dest);

        var gotK = new float[Length * Stride];
        var gotV = new float[Length * Stride];
        for (int l = 0; l < NumLayers; l++)
        {
            staged.DownloadLayer(l, gotK, gotV);
            Assert.Equal(expectedK[l], gotK);
            Assert.Equal(expectedV[l], gotV);
        }
    }
}
