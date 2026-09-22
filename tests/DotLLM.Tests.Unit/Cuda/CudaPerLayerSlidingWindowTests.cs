using DotLLM.Core.Models;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Verifies CudaTransformerModel's per-layer sliding-window resolution matches the CPU
/// reference semantics (TransformerModel.GetLayerSlidingWindow, TransformerModel.cs:706-715):
/// PerLayerSlidingWindow list wins; else pattern N>0 windows layers where layer%N &lt; N-1
/// (gpt-oss N=2: even windowed, odd dense); else uniform. Kernel convention: 0 = dense.
/// </summary>
public class CudaPerLayerSlidingWindowTests
{
    [Theory]
    // gpt-oss shape: window=128, pattern=2 → layer0 windowed, layer1 dense, layer2 windowed...
    [InlineData(128, 2, 0, 128)]
    [InlineData(128, 2, 1, 0)]
    [InlineData(128, 2, 2, 128)]
    [InlineData(128, 2, 3, 0)]
    // pattern<=0 → uniform
    [InlineData(4096, 0, 0, 4096)]
    [InlineData(4096, 0, 5, 4096)]
    // no window at all
    [InlineData(0, 2, 0, 0)]
    public void ResolveLayerWindow_PatternSemanticsMatchCpu(
        int windowSize, int pattern, int layer, int expected)
    {
        int actual = CudaSlidingWindowResolver.Resolve(
            windowSize == 0 ? null : windowSize, pattern, perLayer: null, layer);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ResolveLayerWindow_PerLayerListWins()
    {
        var perLayer = new int?[] { null, 256, null, 64 };
        Assert.Equal(0,   CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 0));
        Assert.Equal(256, CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 1));
        Assert.Equal(0,   CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 2));
        Assert.Equal(64,  CudaSlidingWindowResolver.Resolve(128, 2, perLayer, 3));
    }
}
