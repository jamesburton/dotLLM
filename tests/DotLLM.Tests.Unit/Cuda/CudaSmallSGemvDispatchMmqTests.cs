using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// The #490 dispatch policy — which token counts go to the packed PQ2_0 prefill GEMM and on which
/// tile. Pure policy, so it runs without a GPU.
/// </summary>
public sealed class CudaSmallSGemvDispatchMmqTests : IDisposable
{
    public void Dispose()
    {
        CudaSmallSGemvDispatch.MmqOverride = null;
        CudaSmallSGemvDispatch.MmqTileOverride = null;
        CudaSmallSGemvDispatch.MaxColumnsOverride = null;
    }

    [Fact]
    public void Mmq_IsOptIn()
    {
        CudaSmallSGemvDispatch.MmqOverride = null;
        Assert.False(CudaSmallSGemvDispatch.CoversMmq(64),
            "the MMQ path must stay off unless DOTLLM_CUDA_PQ2_0_MMQ=1 (or the test override) asks for it");
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(8, false)]    // the GEMV's range
    [InlineData(9, true)]
    [InlineData(512, true)]
    public void CoversMmq_StartsWhereTheGemvStops(int seqLen, bool expected)
    {
        CudaSmallSGemvDispatch.MmqOverride = true;
        CudaSmallSGemvDispatch.MaxColumnsOverride = null;
        Assert.Equal(expected, CudaSmallSGemvDispatch.CoversMmq(seqLen));
    }

    /// <summary>
    /// <c>DOTLLM_CUDA_SMALL_S_MAX=0</c> turns the GEMV range off. Single-token decode must NOT then
    /// fall into a GEMM tile that idles 31 of its 32 columns — the floor is <c>max(1, MaxColumns)</c>.
    /// </summary>
    [Fact]
    public void CoversMmq_NeverClaimsSingleTokenDecode_EvenWithTheGemvRangeDisabled()
    {
        CudaSmallSGemvDispatch.MmqOverride = true;
        CudaSmallSGemvDispatch.MaxColumnsOverride = 0;
        Assert.False(CudaSmallSGemvDispatch.CoversMmq(1));
        Assert.True(CudaSmallSGemvDispatch.CoversMmq(2));
    }

    [Theory]
    [InlineData(9, CudaKernels.Pq2_0MmqTileColumnsNarrow)]
    [InlineData(16, CudaKernels.Pq2_0MmqTileColumnsNarrow)]
    [InlineData(17, CudaKernels.Pq2_0MmqTileColumnsWide)]
    [InlineData(1024, CudaKernels.Pq2_0MmqTileColumnsWide)]
    public void MmqTileColumns_PicksTheNarrowTileOnlyUpTo16(int seqLen, int expected)
    {
        CudaSmallSGemvDispatch.MmqTileOverride = null;
        Assert.Equal(expected, CudaSmallSGemvDispatch.MmqTileColumns(seqLen));
    }

    [Fact]
    public void MmqTileColumns_OverridePinsTheTile()
    {
        CudaSmallSGemvDispatch.MmqTileOverride = CudaKernels.Pq2_0MmqTileColumnsWide;
        Assert.Equal(CudaKernels.Pq2_0MmqTileColumnsWide, CudaSmallSGemvDispatch.MmqTileColumns(9));
        CudaSmallSGemvDispatch.MmqTileOverride = CudaKernels.Pq2_0MmqTileColumnsNarrow;
        Assert.Equal(CudaKernels.Pq2_0MmqTileColumnsNarrow, CudaSmallSGemvDispatch.MmqTileColumns(512));
    }
}
