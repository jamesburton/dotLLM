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
[Collection(CudaCollection.Name)]
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
        Assert.Equal(160 * 1024, CudaKvCache.BytesFor(Twil, 1));
        Assert.Equal(1280 * MiB, CudaKvCache.BytesFor(Twil, 8192));

        var ex = Refused(1024 * MiB, () => new CudaKvCache(Twil, 8192));
        AssertStandardRefusal(ex.Message, "KV cache (f16) for 8192 tokens");
        Assert.Contains("it needs 1280 MiB, 1024 MiB is free", ex.Message);
        Assert.Contains("--ctx-size", ex.Message);
    }

    [Fact]
    public void QuantizedKvCache_CountsOnlyItsBlockBytes_AndIsRefused()
    {
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        // q8_0: 1,024 / 32 blocks x 34 B = 1,088 B per K or V row -> 40 x 2,176 = 85 KiB per token. Since #1850 attention reads the
        // rows in place, so the 4 KiB/token FP16 K+V dequant scratch #1841 counted is gone.
        Assert.Equal(40L * 2 * 1088, CudaQuantizedKvCache.BytesPerPosition(Twil, q8));
        Assert.Equal(85 * 1024, CudaQuantizedKvCache.BytesPerPosition(Twil, q8));
        var q4 = new KvCacheConfig(KvCacheDType.Q4_0, KvCacheDType.Q4_0);
        Assert.Equal(40L * 2 * (1024 / 32 * 18), CudaQuantizedKvCache.BytesPerPosition(Twil, q4));
        var windowed = q8 with { MixedPrecisionWindowSize = 64 };
        Assert.Equal(1000 * CudaQuantizedKvCache.BytesPerPosition(Twil, q8) + 40L * 2 * 64 * 1024 * 2,
            CudaQuantizedKvCache.BytesFor(Twil, 1000, windowed));

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
        long perToken = CudaKvCache.BytesFor(Twil, 1);
        Assert.Null(CudaTransformerModel.DescribePoolRefusal(17_408, 16, 1088, 1088, perToken));
        Assert.Null(CudaTransformerModel.DescribePoolRefusal(17_393, 16, 1088, 1088, perToken));   // 1,088 blocks exactly

        string? message = CudaTransformerModel.DescribePoolRefusal(17_409, 16, 1088, 1088, perToken);
        Assert.NotNull(message);
        Assert.StartsWith("Not enough free GPU memory for the paged KV cache for 17409 tokens (1089 blocks; 1088 of the pool's 1088 are free)", message);
        Assert.Contains("it needs 2723 MiB and the pool has 2720 MiB free", message);
        Assert.DoesNotContain("kept in reserve", message);   // the pool is not admitted against the VRAM reserve
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

    // FitDenseContext with 1 KiB per token: every MiB left over is exactly 1,024 tokens (one granule).
    private const long PerToken = 1024;
    private const long G3Plane = CudaG3Attention.RetainedMaxElems * CudaG3Attention.ScratchBytesPerElement;   // 384 MiB

    [Fact]
    public void DenseFit_Unchunked_CountsHeldScratchAsFree_AndChargesTheRetainedG3Plane()
    {
        Assert.Equal(384 * MiB, G3Plane);
        // 3,000 free + 200 held forward scratch (replaced by the growth) - 256 reserve - 100 lazy - 384 G3 plane = 2,460 MiB.
        var fit = CudaTransformerModel.FitDenseContext(3000 * MiB, 256 * MiB, 100 * MiB, 200 * MiB, chunkScratchBytes: null,
            g3PlaneHeldBytes: 0, g3PlaneMaxBytes: G3Plane, PerToken, native: int.MaxValue);
        Assert.Equal(2460 * 1024, fit.MaxContext);
        Assert.Equal(3200 * MiB, fit.FreeBytes);
        Assert.Equal((100 + 384) * MiB, fit.FixedBytes);
        Assert.Null(fit.SizingError);
    }

    [Fact]
    public void DenseFit_Chunked_ChargesTheChunkScratchGrowth_NotTheHeldScratchAsFree()
    {
        // 3,000 - 256 - 100 - 384 - (500 chunk scratch - 200 held) = 1,960 MiB.
        var fit = CudaTransformerModel.FitDenseContext(3000 * MiB, 256 * MiB, 100 * MiB, 200 * MiB, chunkScratchBytes: 500 * MiB,
            g3PlaneHeldBytes: 0, g3PlaneMaxBytes: G3Plane, PerToken, native: int.MaxValue);
        Assert.Equal(1960 * 1024, fit.MaxContext);
    }

    [Fact]
    public void DenseFit_ChargesOnlyTheG3PlaneNotYetHeld_AndNoneWhenG3CannotRun()
    {
        // A 100 MiB plane is already out of the free figure: only the other 284 MiB is charged.
        var held = CudaTransformerModel.FitDenseContext(3000 * MiB, 256 * MiB, 0, 0, null, 100 * MiB, G3Plane, PerToken, int.MaxValue);
        Assert.Equal((3000 - 256 - 284) * 1024, held.MaxContext);
        var off = CudaTransformerModel.FitDenseContext(3000 * MiB, 256 * MiB, 0, 0, null, 0, 0, PerToken, int.MaxValue);
        Assert.Equal((3000 - 256) * 1024, off.MaxContext);
        Assert.Equal(131_072, CudaTransformerModel.FitDenseContext(3000 * MiB, 0, 0, 0, null, 0, 0, PerToken, 131_072).MaxContext);
    }

    [Theory]
    // #1862: G3 off charges nothing, whatever the chunk.
    [InlineData(false, 1024, 32, 0L)]
    [InlineData(false, 0, 32, 0L)]
    // Unchunked: the retained plane only (the square plane grows with the prompt and is admitted per prefill).
    [InlineData(true, 0, 32, 384L)]
    // The default 1,024 chunk: 32 x 1,024^2 = 32M elements (192 MiB) sits inside the retained 64M-element plane.
    [InlineData(true, 1024, 32, 384L)]
    // Past ~1,448 tokens (32 heads) the first chunk's square plane is larger than the retained one, and is charged instead.
    [InlineData(true, 2048, 32, 768L)]
    [InlineData(true, 4096, 32, 3072L)]
    [InlineData(true, 2048, 8, 384L)]
    public void G3PlaneCharge_IsTheRetainedPlane_OrTheFirstChunksSquarePlaneWhenLarger(bool mayRun, int chunk, int heads, long expectedMiB)
        => Assert.Equal(expectedMiB * MiB, CudaTransformerModel.G3PlaneChargeBytes(mayRun, chunk, heads));

    [Fact]
    public void DenseFit_TheChunkedFixedCost_RoughlyDoublesTheF16Window()
    {
        // #1862, TwIL-8B-shaped: f16 KV is 160 KiB a token and an unchunked prefill adds ~146 KiB a token of forward scratch. 6,000 MiB
        // free after warm-up, 100 MiB of forward scratch held, 400 MiB of it at the 1,024 chunk, G3 able to run, 32 query heads.
        const long act = 150_000;
        var f16 = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);
        var unchunked = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, chunkScratchBytes: null,
            g3PlaneHeldBytes: 0, CudaTransformerModel.G3PlaneChargeBytes(true, 0, 32),
            CudaTransformerModel.DenseContextBytesPerToken(Twil, f16, act, chunkedPrefill: false), native: 131_072);
        var chunked = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, chunkScratchBytes: 400 * MiB,
            g3PlaneHeldBytes: 0, CudaTransformerModel.G3PlaneChargeBytes(true, 1024, 32),
            CudaTransformerModel.DenseContextBytesPerToken(Twil, f16, act, chunkedPrefill: true), native: 131_072);
        // Unchunked: (6,100 - 256 - 384) MiB / 313,840 B = 18,242 -> 17,408. Chunked: (6,000 - 256 - 384 - 300) MiB / 163,840 B = 32,384 -> 31,744.
        Assert.Equal(17_408, unchunked.MaxContext);
        Assert.Equal(31_744, chunked.MaxContext);
        Assert.Equal((384 + 300) * MiB, chunked.FixedBytes);
    }

    [Fact]
    public void G3ServesOnlyTheF16Cache_SoAQuantizedFitChargesNoPlane()
    {
        // #1850: q8_0 / q4_0 read in place, so G3's plane is never used over them. TurboQuant is never served by G3 (#1888): its KV is
        // codes read in place, not the f16 cache.
        Assert.True(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32)));
        Assert.False(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0)));
        Assert.False(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.Q4_0, KvCacheDType.Q4_0)));
        Assert.False(CudaTransformerModel.G3ServesKvCache(new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant)));

        // TwIL-8B-shaped q8_0 at the 1,024 chunk, as above: the 384 MiB plane is ~4,700 tokens of q8_0 window.
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        long perToken = CudaTransformerModel.DenseContextBytesPerToken(Twil, q8, 150_000, chunkedPrefill: true);
        var charged = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, 400 * MiB, 0,
            CudaTransformerModel.G3PlaneChargeBytes(true, 1024, 32), perToken, native: 131_072);
        var notCharged = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, 400 * MiB, 0,
            CudaTransformerModel.G3PlaneChargeBytes(CudaTransformerModel.G3ServesKvCache(q8), 1024, 32), perToken, native: 131_072);
        Assert.True(notCharged.MaxContext - charged.MaxContext >= 4096,
            $"q8_0 window {notCharged.MaxContext} without the plane vs {charged.MaxContext} with it");
    }

    /// <summary>
    /// #1923: when the F16 flash prefill takes every prefill of 16+ rows (hd64 / hd128, GQA group up to 8, uniform geometry,
    /// enabled, kernel loaded), G3 only ever sees shorter prefills, so the f16 context fit must not charge its 384 MiB plane:
    /// that is ~2,500 tokens of f16 window on Qwen3-4B. Any shape the flash entry does not cover keeps the charge.
    /// </summary>
    [Fact]
    public void F16FlashPrefillCoveringTheModel_RemovesTheG3PlaneCharge()
    {
        var f16 = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);
        // The claim rests on G3's offset path never taking a prefill the flash entry would (it needs >= 32 rows, flash >= 16). With
        // a lower offset minimum an uncharged offset plane of up to 384 MiB could be asked for (#1923 review M3).
        Assert.True(CudaG3Attention.OffsetMinQueryRows >= CudaTransformerModel.KvqFlashPrefillMinRows,
            $"G3 offset minimum {CudaG3Attention.OffsetMinQueryRows} < flash minimum {CudaTransformerModel.KvqFlashPrefillMinRows}");
        bool Covers(bool enabled = true, bool uniform = true, bool loaded = true, int heads = 32, int kvHeads = 8, int headDim = 128)
            => CudaTransformerModel.F16FlashPrefillCovers(enabled, uniform, loaded, heads, kvHeads, headDim);

        Assert.True(Covers());                                  // Qwen3-4B
        Assert.True(Covers(heads: 14, kvHeads: 2, headDim: 64)); // Qwen2.5-0.5B, group 7
        Assert.False(Covers(enabled: false));                   // DOTLLM_CUDA_F16_FLASH_PREFILL=0
        Assert.False(Covers(uniform: false));                   // Gemma-4 style per-layer geometry
        Assert.False(Covers(loaded: false));                    // pre-Ampere / missing PTX
        Assert.False(Covers(headDim: 256));
        Assert.False(Covers(heads: 36, kvHeads: 4));            // group 9

        Assert.False(CudaTransformerModel.G3PlaneCharged(f16, f16FlashCoversModel: true));
        Assert.True(CudaTransformerModel.G3PlaneCharged(f16, f16FlashCoversModel: false));
        Assert.False(CudaTransformerModel.G3PlaneCharged(new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0), false));

        // What MeasureServingContextFit passes: G3PlaneChargeBytes(G3PlaneCharged(kv, covers) && g3MayRun, chunk, heads).
        long Charge(bool covers) => CudaTransformerModel.G3PlaneChargeBytes(CudaTransformerModel.G3PlaneCharged(f16, covers), 1024, 32);
        Assert.Equal(0, Charge(covers: true));
        Assert.Equal(384 * MiB, Charge(covers: false));

        var geom = KvGeometry.Uniform(36, 8, 128);   // Qwen3-4B
        long perToken = CudaTransformerModel.DenseContextBytesPerToken(geom, f16, 150_000, chunkedPrefill: true);
        var withPlane = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, 400 * MiB, 0, Charge(false), perToken,
            native: 262_144);
        var withoutPlane = CudaTransformerModel.FitDenseContext(6000 * MiB, 256 * MiB, 0, 100 * MiB, 400 * MiB, 0, Charge(true), perToken,
            native: 262_144);
        Assert.True(withoutPlane.MaxContext - withPlane.MaxContext >= 2048,
            $"f16 window {withoutPlane.MaxContext} without the plane vs {withPlane.MaxContext} with it");
    }

    [Fact]
    public void DenseFit_WithoutAReading_FitsNothing_AndSaysWhy()
    {
        var fit = CudaTransformerModel.FitDenseContext(null, 256 * MiB, 0, 0, null, 0, G3Plane, PerToken, int.MaxValue);
        Assert.Equal(0, fit.MaxContext);
        Assert.Contains("cuMemGetInfo", fit.SizingError);
    }

    [Fact]
    public void RefusalFromEnsureFits_CarriesTheNumbers()
    {
        var ex = Refused(100 * MiB, () => CudaAdmission.EnsureFits("test buffer", 200 * MiB));
        Assert.Equal(200 * MiB, ex.RequiredBytes);
        Assert.Equal(100 * MiB, ex.FreeBytes);
        Assert.Equal(CudaAdmission.ReserveBytes, ex.ReserveBytes);
        Assert.False(ex.CouldFitAfterReleasing(50 * MiB));
        Assert.True(ex.CouldFitAfterReleasing(100 * MiB + CudaAdmission.ReserveBytes));
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
