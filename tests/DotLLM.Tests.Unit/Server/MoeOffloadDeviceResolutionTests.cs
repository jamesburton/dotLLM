using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1070: an explicit CUDA request for a <c>nemotron_h_moe</c> model whose weights exceed free VRAM reaches the loader
/// (which offloads the experts) instead of being refused by the weights-vs-free preflight, unless <c>--moe-offload off</c>.
/// No hardware: the device inventory and the loader are injected.
/// </summary>
public sealed class MoeOffloadDeviceResolutionTests
{
    private const long GiB = 1L << 30;

    private static DeviceListResponse SmallCuda() => new()
    {
        Backends =
        [
            new BackendInfoDto
            {
                Name = "cuda", Available = true, DeviceCount = 1, Servable = true,
                Devices = [new DeviceInfoDto { Index = 0, Name = "gpu0", DeviceString = "gpu:0", TotalMemoryBytes = 12 * GiB, FreeMemoryBytes = 10 * GiB }],
            },
            new BackendInfoDto
            {
                Name = "cpu", Available = true, DeviceCount = 1, Servable = true,
                Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }],
            },
        ],
    };

    private static ModelConfig Config(Architecture arch) => new()
    {
        Architecture = arch,
        VocabSize = 16,
        HiddenSize = 64,
        IntermediateSize = 64,
        NumLayers = 1,
        NumAttentionHeads = 1,
        NumKvHeads = 1,
        HeadDim = 64,
        MaxSequenceLength = 64,
        AttentionType = AttentionType.GQA,
        PositionEncodingType = PositionEncodingType.None,
        ActivationFunction = ActivationFunction.ReluSquared,
        NormType = NormType.RMSNorm,
        NormEpsilon = 1e-5f,
        TiedEmbeddings = false,
        ChatTemplate = null,
    };

    [Fact]
    public void ExpertOffloadPossible_OnlyForNemotronHMoe_AndNotWhenOff()
    {
        var auto = new MoeOffloadOptions();
        Assert.True(DeviceModelLoader.ExpertOffloadPossible(Config(Architecture.NemotronHMoe), auto));
        Assert.True(DeviceModelLoader.ExpertOffloadPossible(Config(Architecture.NemotronHMoe), auto with { Mode = MoeOffloadMode.On }));
        Assert.False(DeviceModelLoader.ExpertOffloadPossible(Config(Architecture.NemotronHMoe), auto with { Mode = MoeOffloadMode.Off }));
        Assert.False(DeviceModelLoader.ExpertOffloadPossible(Config(Architecture.NemotronH), auto));
        Assert.False(DeviceModelLoader.ExpertOffloadPossible(null, auto));
    }

    [Fact]
    public void ExplicitCuda_ModelLargerThanFreeVram_ReachesTheLoaderWhenExpertsCanBeOffloaded()
    {
        var tried = new List<string>();
        var result = DeviceModelLoader.LoadWith(null!, Config(Architecture.NemotronHMoe), "cuda", null, default, 20 * GiB, "bolt.gguf",
            _ => { }, _ => { }, SmallCuda(), (d, _) => { tried.Add(d); return (null!, (Func<int, IKvCache>?)null); },
            new MoeOffloadOptions());
        Assert.Equal("gpu:0", result.ResolvedDevice);
        Assert.Equal(["gpu:0"], tried);
    }

    [Theory]
    [InlineData(Architecture.NemotronHMoe, MoeOffloadMode.Off)]
    [InlineData(Architecture.NemotronH, MoeOffloadMode.Auto)]
    public void ExplicitCuda_ModelLargerThanFreeVram_IsStillRefusedWithoutOffload(Architecture arch, MoeOffloadMode mode)
    {
        var tried = new List<string>();
        Assert.Throws<DeviceUnavailableException>(() => DeviceModelLoader.LoadWith(null!, Config(arch), "cuda", null, default, 20 * GiB,
            "m.gguf", _ => { }, _ => { }, SmallCuda(), (d, _) => { tried.Add(d); return (null!, (Func<int, IKvCache>?)null); },
            new MoeOffloadOptions { Mode = mode }));
        Assert.Empty(tried);
    }
}
