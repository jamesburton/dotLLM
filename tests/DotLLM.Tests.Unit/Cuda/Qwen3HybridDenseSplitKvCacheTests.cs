using DotLLM.Core.Attention;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// PR #1026 review (SF3): the partial-offload split KV holds its GPU half in the CUDA model's single-slot storage, so it must
/// report non-exclusive storage like its GPU handle. Otherwise the engine keeps a recurrent prefix snapshot across another
/// request, and a later hit attends over that request's rows (--gpu-layers below all).
/// </summary>
public sealed class Qwen3HybridDenseSplitKvCacheTests
{
    [Fact]
    public void SplitKv_ReportsTheGpuHandlesNonExclusiveStorage()
    {
        using IKvCache split = new Qwen3HybridDenseSplitKvCache(new CudaHybridKvCacheHandle(32), new SimpleKvCache(1, 1, 4, 32));

        Assert.False(split.HasExclusiveStorage);
    }

    [Fact]
    public void SplitKv_DoesNotGrow()
    {
        using IKvCache split = new Qwen3HybridDenseSplitKvCache(new CudaHybridKvCacheHandle(32), new SimpleKvCache(1, 1, 4, 32));

        Assert.True(split.TryGrowTo(32));
        Assert.False(split.TryGrowTo(64));
    }
}
