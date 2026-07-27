using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Engine.Scheduler;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the cross-device KV handoff (<see cref="StagedKvHandoffTransfer"/>, #360) across TWO physical
/// CUDA devices: a prefill cache on GPU 0, a decode cache on GPU 1, contents staged device→host→device via
/// <see cref="IHostStagedKvCache"/>. This is the dual-GPU proof the local boxes cannot give (Strix Halo =
/// single iGPU; T5500 = single RTX 3060); it runs on a Kaggle "GPU T4 ×2" session (#361) and skips
/// elsewhere. The fill uses small integers that are exactly representable in FP16, so the post-transfer
/// compare is byte-exact despite the FP16 device storage.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaCrossDeviceKvTransferTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int Stride = NumKvHeads * HeadDim;
    private const int MaxSeqLen = 64;
    private const int Length = 7;

    [SkippableFact]
    public void StagedTransfer_AcrossTwoCudaDevices_PreservesContents()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(CudaDevice.GetDeviceCount() < 2, "Cross-device transfer needs >= 2 CUDA GPUs (e.g. Kaggle T4 x2)");

        using var ctx0 = CudaContext.Create(0);
        using var ctx1 = CudaContext.Create(1);
        var geom = KvGeometry.Uniform(NumLayers, NumKvHeads, HeadDim);

        // Expected per-layer K/V — small integers (exactly representable in FP16, so the round-trip is exact).
        var expectedK = new float[NumLayers][];
        var expectedV = new float[NumLayers][];
        for (int l = 0; l < NumLayers; l++)
        {
            expectedK[l] = new float[Length * Stride];
            expectedV[l] = new float[Length * Stride];
            for (int p = 0; p < Length; p++)
                for (int d = 0; d < Stride; d++)
                {
                    float val = l * 1000 + p * 10 + d; // <= 1078 < 2048 → exact in FP16
                    expectedK[l][p * Stride + d] = val;
                    expectedV[l][p * Stride + d] = -val;
                }
        }

        // Build + fill the prefill cache on GPU 0.
        ctx0.MakeCurrent();
        // Not wrapped in `using`: ownership transfers to StagedKvHandoffTransfer.Transfer below,
        // which disposes `source` as part of the device0->host->device1 handoff (see comment there).
        var source = new CudaKvCache(geom, MaxSeqLen, ctx0);
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

        // Hand the sequence off to GPU 1 via the staged transfer (device0 → host → device1). Disposes source.
        using IKvCache dest = StagedKvHandoffTransfer.Instance.Transfer(
            source, config, (_, maxSeq) => new CudaKvCache(geom, maxSeq, ctx1));

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
