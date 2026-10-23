using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1099: <see cref="CudaWideProjection.RunTiled"/> — the bounded LM-head fallback the MoE hybrid and
/// Nemotron-H hosts use for quant types without a packed multi-row kernel — must equal the whole-matrix
/// dequant + cuBLAS it replaces. Every output element is the same k-length dot product of the same F16
/// operands, so the bar is F16 output rounding (one ulp, 2^-10 relative: cuBLAS may pick a different reduction
/// order for a narrower n), not a model tolerance. FP16 accumulation (<c>CudaGemm.Use16FCompute</c>) is pinned off.
/// </summary>
/// <remarks>
/// The shape is chosen to discriminate: <c>m</c> is not a multiple of the tile, so the last tile is
/// partial; there are several tiles; and <c>seqLen &gt; 1</c>, so a wrong output stride (<c>ldc</c>) or
/// row offset lands values in the wrong token row instead of being invisible.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaWideProjectionTests
{
    private readonly ITestOutputHelper _out;

    public CudaWideProjectionTests(ITestOutputHelper output) => _out = output;

    /// <summary>Tiled vs whole-matrix, F16 and F32 outputs, on random Q8_0 weights.</summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void RunTiled_MatchesWholeMatrixDequant_Q8_0(bool f32Out)
    {
        string ptxDir = CudaPQ2_0GemvMultiTests.SkipUnlessMultiKernel();
        const int m = 1000, k = 256, seqLen = 5, tileRows = 300;   // 4 tiles, the last one 100 rows
        var rng = new Random(1099 + (f32Out ? 1 : 0));
        byte[] w = RandomQ8_0(rng, m, k);
        var x = new Half[seqLen * k];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)(rng.NextSingle() * 2 - 1);

        bool use16F = CudaGemm.Use16FCompute;
        CudaGemm.Use16FCompute = false;
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var cublas = CudaCublasHandle.Create();
        using var kernels = new CudaKernels(ptxDir);
        nint s = stream.Handle;
        int outBytes = f32Out ? sizeof(float) : sizeof(ushort);
        nint dW = Alloc(w.Length), dX = Alloc((long)x.Length * 2);
        nint dWhole = Alloc((long)m * k * 2), dTile = Alloc((long)tileRows * k * 2);
        nint dYWhole = Alloc((long)seqLen * m * outBytes), dYTiled = Alloc((long)seqLen * m * outBytes);
        try
        {
            fixed (byte* p = w) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)p, (nuint)w.Length).ThrowOnError();
            fixed (Half* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)(x.Length * 2)).ThrowOnError();

            kernels.LaunchDequantToF16(dW, QuantizationType.Q8_0, dWhole, m * k, s);
            if (f32Out) CudaGemm.LinearF16InF32Out(cublas.Handle, dX, dWhole, dYWhole, seqLen, k, m, s);
            else CudaGemm.LinearF16(cublas.Handle, dX, dWhole, dYWhole, seqLen, k, m, s);

            int tiles = CudaWideProjection.RunTiled(kernels, cublas.Handle, dW, QuantizationType.Q8_0,
                dX, dYTiled, f32Out, m, k, seqLen, dTile, tileRows, s);
            stream.Synchronize();
            Assert.Equal(4, tiles);

            float[] whole = Download(dYWhole, seqLen * m, f32Out), tiled = Download(dYTiled, seqLen * m, f32Out);
            float maxDiff = 0;
            for (int i = 0; i < whole.Length; i++)
            {
                float d = MathF.Abs(whole[i] - tiled[i]);
                maxDiff = MathF.Max(maxDiff, d);
                Assert.True(d <= MathF.Abs(whole[i]) * (1f / 1024f) + 1e-6f,
                    $"token {i / m} row {i % m}: whole {whole[i]:R}, tiled {tiled[i]:R}");
            }
            _out.WriteLine($"f32Out={f32Out}: {tiles} tiles, max|tiled-whole| {maxDiff:E3}");
        }
        finally
        {
            foreach (nint p in new[] { dW, dX, dWhole, dTile, dYWhole, dYTiled })
                CudaDriverApi.cuMemFree_v2(p);
            CudaGemm.Use16FCompute = use16F;
        }
    }

    /// <summary>Tile sizing: whole rows within the bound, at least one, at most <c>m</c>.</summary>
    [Fact]
    public void TileRows_FitsBound()
    {
        Assert.Equal(6553, CudaWideProjection.TileRows(248320, 5120, CudaWideProjection.TileFloorElems));
        Assert.Equal(1, CudaWideProjection.TileRows(10, 5120, 100));
        Assert.Equal(10, CudaWideProjection.TileRows(10, 64, 1L << 30));
        Assert.False(CudaWideProjection.SupportsRowTiles(QuantizationType.PQ2_0));
        Assert.False(CudaWideProjection.SupportsRowTiles(QuantizationType.I2_S));
        Assert.True(CudaWideProjection.SupportsRowTiles(QuantizationType.Q6_K));
    }

    private static byte[] RandomQ8_0(Random rng, int m, int k)
    {
        int blocks = m * (k / 32);
        byte[] w = new byte[blocks * 34];
        for (int b = 0; b < blocks; b++)
        {
            BitConverter.TryWriteBytes(w.AsSpan(b * 34, 2), (Half)(0.002f + rng.NextSingle() * 0.01f));
            for (int i = 0; i < 32; i++) w[b * 34 + 2 + i] = (byte)(sbyte)rng.Next(-127, 128);
        }
        return w;
    }

    private static unsafe float[] Download(nint dev, int count, bool f32)
    {
        var result = new float[count];
        if (f32)
        {
            fixed (float* p = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(count * 4)).ThrowOnError();
            return result;
        }
        var h = new Half[count];
        fixed (Half* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(count * 2)).ThrowOnError();
        for (int i = 0; i < count; i++) result[i] = (float)h[i];
        return result;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        return p;
    }
}
