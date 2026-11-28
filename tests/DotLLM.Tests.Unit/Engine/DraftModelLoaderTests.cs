using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #2064: <c>run</c> / <c>chat</c> / <c>serve</c> loaded every draft on the CPU, whatever the target ran on. These pin where
/// <see cref="DraftModelLoader"/> puts it, with the per-device loaders injected so no GPU is needed.
/// </summary>
public sealed class DraftModelLoaderTests
{
    private static readonly ModelConfig Dense = new()
    {
        Architecture = Architecture.Llama, VocabSize = 32, HiddenSize = 8, IntermediateSize = 16, NumLayers = 1,
        NumAttentionHeads = 2, NumKvHeads = 2, HeadDim = 4, MaxSequenceLength = 64,
    };

    private sealed class StubModel(int rowLimit, bool widens) : IModel
    {
        private int _limit = rowLimit;
        public int RequestedRows { get; private set; }
        public ModelConfig Config => Dense;
        public long ComputeMemoryBytes => 0;
        public int MaxAllRowLogitsLength => _limit;

        public bool TrySetAllRowLogitsLimit(int maxSeqLen)
        {
            RequestedRows = maxSeqLen;
            if (!widens) return false;
            _limit = Math.Max(_limit, maxSeqLen);
            return true;
        }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
            => throw new NotSupportedException();

        public void Dispose() { }
    }

    private static (DraftLoadResult? Result, List<string> Loads, List<string> Warnings) Load(
        string targetDevice, ModelConfig draftConfig, IModel? target = null, string? failOn = null)
    {
        var loads = new List<string>();
        var warnings = new List<string>();
        var result = DraftModelLoader.LoadWith(
            target ?? new StubModel(1, widens: true), targetDevice, draftConfig, speculativeCandidates: 3, warnings.Add,
            device =>
            {
                loads.Add(device);
                if (device == failOn) throw new InvalidOperationException("out of device memory");
                return (new StubModel(1, widens: false), device == "cpu" ? null : _ => null!);
            });
        return (result, loads, warnings);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    [InlineData("vulkan:1")]
    [InlineData("gpu:0")]
    public void Draft_LoadsOnTheTargetsDevice(string targetDevice)
    {
        var (result, loads, warnings) = Load(targetDevice, Dense);

        Assert.Equal([targetDevice], loads);
        Assert.Equal(targetDevice, result!.Device);
        // A GPU draft needs the cache its host builds: without the factory TextGenerator hands it a host SimpleKvCache.
        Assert.Equal(targetDevice != "cpu", result.KvCacheFactory is not null);
        Assert.Empty(warnings);
    }

    [Fact]
    public void OneRowTarget_LoadsNoDraftAndSaysWhy()
    {
        var target = new StubModel(1, widens: false);
        var (result, loads, warnings) = Load("gpu:0", Dense, target);

        Assert.Null(result);
        Assert.Empty(loads);
        Assert.Equal(4, target.RequestedRows);
        Assert.Contains("draft model not loaded", Assert.Single(warnings));
    }

    [Theory]
    [InlineData(Architecture.Qwen3MoeHybrid)]
    [InlineData(Architecture.Qwen3HybridDense)]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.Lfm2Moe)]
    [InlineData(Architecture.Lfm3)]
    public void NonDenseDraft_StaysOnTheCpu(Architecture architecture)
    {
        var (result, loads, warnings) = Load("vulkan", Dense with { Architecture = architecture });

        Assert.Equal(["cpu"], loads);
        Assert.Equal("cpu", result!.Device);
        Assert.Contains("CPU", Assert.Single(warnings));
    }

    [Fact]
    public void GpuLoadFailure_FallsBackToTheCpuLoudly()
    {
        var (result, loads, warnings) = Load("vulkan", Dense, failOn: "vulkan");

        Assert.Equal(["vulkan", "cpu"], loads);
        Assert.Equal("cpu", result!.Device);
        string warning = Assert.Single(warnings);
        Assert.Contains("out of device memory", warning);
        Assert.Contains("CPU", warning);
    }

    [Fact]
    public void CpuLoadFailure_Propagates()
        => Assert.Throws<InvalidOperationException>(() => Load("cpu", Dense, failOn: "cpu"));
}
