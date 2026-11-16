using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1848: every runtime-sized CUDA device allocation is admitted against free VRAM through <see cref="CudaAdmission"/> and
/// refused with the standard <see cref="DeviceMemoryBudget.DescribeRefusal"/> message instead of letting WDDM spill it into shared
/// system memory. These run without a GPU: a fake free-memory source (<see cref="CudaAdmission.FreeBytesOverride"/>, scoped to the
/// test's async flow) refuses before any driver call, so removing a check makes the constructor reach <c>cuMemAlloc</c> (no driver
/// here: <see cref="DllNotFoundException"/>; on a CUDA box: an allocation that succeeds), and the test fails either way.
/// </summary>
public sealed class CudaAdmissionTests
{
    private const long MiB = 1L << 20;

    // TwIL-8B / Granite-8B shape (#1841): 40 layers, 8 KV heads x 128 = 1,024-wide K and V rows.
    private static readonly KvGeometry Twil = KvGeometry.Uniform(40, 8, 128);

    private static T WithFree<T>(long? freeBytes, Func<T> body)
    {
        var before = CudaAdmission.FreeBytesOverride;
        CudaAdmission.FreeBytesOverride = () => freeBytes;
        try { return body(); }
        finally { CudaAdmission.FreeBytesOverride = before; }
    }

    private static DeviceMemoryAdmissionException Refused(long? freeBytes, Action allocate)
        => WithFree(freeBytes, () => Assert.Throws<DeviceMemoryAdmissionException>(allocate));

    private static void AssertStandardRefusal(string message, string what)
    {
        Assert.StartsWith($"Not enough free GPU memory for the {what}", message);
        Assert.Contains("Refusing rather than letting the driver spill into shared system memory", message);
        Assert.Contains(DeviceMemoryBudget.ReserveEnvVar, message);
    }

    [Fact]
    public void TryFit_CountsTheReserve_AndMemoryTheCallerReleasesFirst()
    {
        long reserve = CudaAdmission.ReserveBytes;
        WithFree(reserve + 100 * MiB, () =>
        {
            Assert.True(CudaAdmission.TryFit(100 * MiB, out long free));
            Assert.Equal(reserve + 100 * MiB, free);
            Assert.False(CudaAdmission.TryFit(100 * MiB + 1, out _));
            Assert.True(CudaAdmission.TryFit(150 * MiB, out _, releasedBytes: 50 * MiB));   // the buffer it replaces counts as free
            Assert.False(CudaAdmission.TryFit(150 * MiB, out _, releasedBytes: 49 * MiB));
            return 0;
        });
    }

    [Fact]
    public void NoReading_Refuses()
        => WithFree(null, () =>
        {
            Assert.False(CudaAdmission.TryFit(1, out _));
            Assert.True(CudaAdmission.TryFit(0, out _));   // nothing to allocate is never refused
            return 0;
        });

    [Fact]
    public void DenseF16KvCache_IsRefusedBeforeAllocating()
    {
        // 40 layers x 2 (K, V) x 1,024 x 2 B = 160 KiB per position (#1841): 8,192 positions are 1,280 MiB.
        Assert.Equal(160 * 1024, CudaKvCache.DeviceBytes(Twil, 1));
        Assert.Equal(1280 * MiB, CudaKvCache.DeviceBytes(Twil, 8192));

        var ex = Refused(1024 * MiB, () => new CudaKvCache(Twil, 8192));
        AssertStandardRefusal(ex.Message, "KV cache (f16) for 8192 tokens");
        Assert.Contains("it needs 1280 MiB, 1024 MiB is free", ex.Message);
        Assert.Contains("--ctx-size", ex.Message);
    }

    [Fact]
    public void QuantizedKvCache_CountsTheFullHistoryDequantScratch_AndIsRefused()
    {
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        // q8_0: 1,024 / 32 blocks x 34 B = 1,088 B per K or V row -> 40 x 2,176 = 85 KiB, plus the FP16 K+V dequant scratch over
        // the whole history at the widest stride: 2 x 1,024 x 2 B = 4 KiB (#1841's 85 + 4 KiB per token).
        Assert.Equal(40L * 2 * 1088 + 2 * 1024 * 2, CudaQuantizedKvCache.BytesPerPosition(Twil, q8));
        var windowed = q8 with { MixedPrecisionWindowSize = 64 };
        Assert.Equal(1000 * CudaQuantizedKvCache.BytesPerPosition(Twil, q8) + 40L * 2 * 64 * 1024 * 2,
            CudaQuantizedKvCache.DeviceBytes(Twil, 1000, windowed));

        var ex = Refused(64 * MiB, () => new CudaQuantizedKvCache(Twil, 8192, q8));
        AssertStandardRefusal(ex.Message, "KV cache (q8_0/q8_0) for 8192 tokens");
    }

    [Fact]
    public void TurboQuantKvCache_IsRefusedBeforeAllocating()
    {
        float[] centroids = new float[16];
        float[] signs = Enumerable.Repeat(1f, 128).ToArray();
        var ex = Refused(1 * MiB, () => new CudaTurboQuantKvCache(40, 8, 128, 8192, 4, centroids, signs, signs, 1f / MathF.Sqrt(128)));
        AssertStandardRefusal(ex.Message, "KV cache (tq4) for 8192 tokens");
    }

    [Fact]
    public void MlaKvCache_IsRefusedBeforeAllocating()
    {
        var ex = Refused(1 * MiB, () => new CudaMlaKvCache(27, 8192, 16, 128, 128, 64, MlaPrecision.F16));
        AssertStandardRefusal(ex.Message, "MLA KV cache (f16) for 8192 tokens");
    }

    [Fact]
    public void ForwardScratch_TakesThePowerOfTwo_ThenTheExactSize_ThenRefuses()
    {
        static long Bytes(int capacity) => capacity * 1000L;

        // Plenty of room: the power of two, capped at the served context and never below the request.
        Assert.Equal(16384, CudaForwardState.ChooseCapacity(10_000, int.MaxValue, Bytes, _ => true));
        Assert.Equal(12_288, CudaForwardState.ChooseCapacity(10_000, 12_288, Bytes, _ => true));
        Assert.Equal(10_000, CudaForwardState.ChooseCapacity(10_000, 4096, Bytes, _ => true));
        // The power of two does not fit but the exact size does (10K rounded to 16K costs ~0.9 GB on an 8B model).
        Assert.Equal(10_000, CudaForwardState.ChooseCapacity(10_000, int.MaxValue, Bytes, b => b <= Bytes(12_000)));
        // Nothing fits.
        Assert.Equal(-1, CudaForwardState.ChooseCapacity(10_000, int.MaxValue, Bytes, b => b < Bytes(10_000)));
    }

    [Fact]
    public void Weights_ThatDoNotFit_AreRefusedWithTheOffloadRemedies()
    {
        long reserve = 256 * MiB;
        Assert.Null(CudaTransformerModel.DescribeWeightsRefusal(4900 * MiB, 4900 * MiB + reserve, reserve));
        Assert.Null(CudaTransformerModel.DescribeWeightsRefusal(4900 * MiB, null, reserve));   // no reading: the upload reports it

        string? message = CudaTransformerModel.DescribeWeightsRefusal(4900 * MiB, 4900 * MiB + reserve - 1, reserve);
        Assert.NotNull(message);
        AssertStandardRefusal(message!, "model weights (~4900 MiB)");
        Assert.Contains("--gpu-layers", message);
        Assert.DoesNotContain("--ctx-size", message);   // context is not the fix for weights
    }

    [Fact]
    public void PagedRequest_PastTheFreeBlocks_IsRefusedAtAdmission()
    {
        long perToken = CudaKvCache.DeviceBytes(Twil, 1);
        Assert.Null(CudaTransformerModel.DescribePoolRefusal(17_408, 16, 1088, 1088, perToken));
        Assert.Null(CudaTransformerModel.DescribePoolRefusal(17_393, 16, 1088, 1088, perToken));   // 1,088 blocks exactly

        string? message = CudaTransformerModel.DescribePoolRefusal(17_409, 16, 1088, 1088, perToken);
        Assert.NotNull(message);
        Assert.StartsWith("Not enough free GPU memory for the paged KV cache for 17409 tokens (1089 blocks; 1088 of the pool's 1088 are free)", message);
        Assert.NotNull(CudaTransformerModel.DescribePoolRefusal(160, 16, 9, 1088, perToken));   // other sequences hold the rest
    }

    [Fact]
    public void DenseContextBytesPerToken_IsKvForTheType_PlusUnchunkedPrefillScratch()
    {
        const long act = 150_000;
        var f16 = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Assert.Equal(160 * 1024 + act, CudaTransformerModel.DenseContextBytesPerToken(Twil, f16, act, chunkedPrefill: false));
        Assert.Equal(160 * 1024, CudaTransformerModel.DenseContextBytesPerToken(Twil, f16, act, chunkedPrefill: true));
        Assert.Equal(CudaQuantizedKvCache.BytesPerPosition(Twil, q8) + act,
            CudaTransformerModel.DenseContextBytesPerToken(Twil, q8, act, chunkedPrefill: false));
    }

    [Fact]
    public void G3ScratchElements_SquareAndOffset()
    {
        Assert.Equal(32L * 4096 * 4096, CudaG3Attention.SquareScratchElems(4096, 32));
        // Offset path: chunked by query rows under OffsetMaxScoreElems (64M elements).
        long perRow = 32L * (4113 + 647);
        int chunk = (int)Math.Min(CudaG3Attention.DefaultOffsetMaxScoreElems / perRow, 647);
        Assert.Equal(32L * chunk * (4113 + 647), CudaG3Attention.OffsetScratchElems(647, 4113, 32));
        Assert.True(CudaG3Attention.OffsetScratchElems(647, 4113, 32) <= CudaG3Attention.DefaultOffsetMaxScoreElems);
    }
}
