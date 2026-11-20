using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Tests.Unit.Engine.KvCache;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1892 Task 5: the CUDA in-place TurboQuant cache stores the shared layout (codes padded to <c>CodeUintsPerVec</c> uints,
/// norms in the decided format), byte-identical to the CPU codec. Runs on the framework RTX 3060 (no CUDA on the Strix box).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantInPlaceKvCacheTests
{
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantInPlaceKvCacheTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    /// <summary>Bits 2/3/4/8 (and 1) over head dims 64/128/256, plus hd16 where a vector's code bytes are NOT a whole number of words
    /// (bits 3: 6 bytes in 2 uints), the only shape that tells the padded per-vector stride from a tighter one.</summary>
    public static IEnumerable<object[]> Configs()
    {
        foreach (int b in new[] { 1, 2, 3, 4, 8 })
            foreach (int d in new[] { 16, 64, 128, 256 })
                yield return new object[] { b, d };
    }

    private sealed class Fixture : IDisposable
    {
        public readonly CudaContext Ctx;
        public readonly CudaStream Stream;
        public readonly CudaKernels Kernels;
        public readonly TurboQuantByteExactCorpus.Corpus Corpus;
        public readonly CudaTurboQuantInPlaceKvCache Cache;
        public readonly int Seq, Heads, Bits, HeadDim;
        private readonly nint _dK, _dV;

        public Fixture(int bits, int headDim, TqNormFormat? norm = null, int layers = 1)
        {
            Ctx = CudaContext.Create(0);
            Stream = CudaStream.Create();
            Kernels = new CudaKernels(FindPtxDir()!);
            Skip.IfNot(Kernels.TurboQuantF16Available, "turboquant f16 kernels not loaded");
            Bits = bits; HeadDim = headDim;
            Corpus = TurboQuantByteExactCorpus.Build(bits, headDim, halfInputs: true);
            Seq = TurboQuantByteExactCorpus.SeqLen; Heads = TurboQuantByteExactCorpus.NumKvHeads;
            var ck = Corpus.K.Codec; var cv = Corpus.V.Codec;
            Cache = new CudaTurboQuantInPlaceKvCache(layers, Heads, headDim, Seq, ck.MseBits, ck.Centroids, ck.RotationSigns,
                cv.RotationSigns, ck.InvSqrtD, norm, context: Ctx);
            _dK = UploadHalf(Corpus.K.Data); _dV = UploadHalf(Corpus.V.Data);
        }

        public void EncodeLayer(int layer)
        {
            var positions = new int[Seq];
            for (int i = 0; i < Seq; i++) positions[i] = i;
            Cache.UpdateDevice(_dK, _dV, positions, Seq, layer, Stream.Handle, Kernels);
            Stream.Synchronize();
        }

        public void Dispose()
        {
            Free(_dK); Free(_dV);
            Cache?.Dispose(); Kernels?.Dispose(); Stream?.Dispose(); Ctx?.Dispose();
        }
    }

    private static Fixture Open(int bits, int headDim, TqNormFormat? norm = null, int layers = 1)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.If(FindPtxDir() == null, "PTX files not found");
        return new Fixture(bits, headDim, norm, layers);
    }

    [SkippableTheory]
    [MemberData(nameof(Configs))]
    public void Codes_MatchCpuEncode_ByteForByte(int bits, int headDim)
    {
        using var f = Open(bits, headDim);
        f.EncodeLayer(0);
        var layout = f.Cache.Layout;
        int n = f.Seq * f.Heads, cu = layout.CodeUintsPerVec;
        Assert.Equal((f.Corpus.K.Codec.CodeBytesPerVector + 3) / 4, cu);
        string failures = "";
        foreach (bool values in new[] { false, true })
        {
            var side = values ? f.Corpus.V : f.Corpus.K;
            var gotCodes = new uint[n * cu]; var gotNorms = new byte[n * layout.NormBytes];
            f.Cache.DownloadCodeUints(0, values, gotCodes);
            f.Cache.DownloadNormBytes(0, values, gotNorms);
            var (cpuCodes, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, layout.Norm);
            var (cb, nbad, total) = TurboQuantByteExactCorpus.Compare(cpuCodes, cpuNorms, MemoryMarshal.AsBytes(gotCodes.AsSpan()), gotNorms, cu * 4, layout.NormBytes);
            _out.WriteLine($"in-place bits={bits} hd={headDim} {(values ? "V" : "K")}: code mismatches {cb}/{total}, norm mismatches {nbad}/{total} (tie coords {side.TieCoords})");
            if (cb != 0 || nbad != 0) failures += $"{(values ? "V" : "K")}: {cb} code / {nbad} norm of {total}; ";
        }
        Assert.True(failures.Length == 0, $"in-place CUDA encode differs from the CPU codec at bits={bits} hd={headDim}: {failures}");
    }

    [SkippableTheory]
    [InlineData(2, 128)]
    [InlineData(3, 128)]
    [InlineData(4, 64)]
    [InlineData(8, 128)]
    public void Norms_InDecidedFormat(int bits, int headDim)
    {
        using var f = Open(bits, headDim);
        Assert.Equal(TurboQuantLayout.DefaultNorm, f.Cache.Layout.Norm);
        Assert.Equal(TqNormFormat.F16, f.Cache.NormFormat);
        Assert.Equal(2, f.Cache.Layout.NormBytes);
        f.EncodeLayer(0);
        foreach (bool values in new[] { false, true })
        {
            var side = values ? f.Corpus.V : f.Corpus.K;
            var (_, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, TqNormFormat.F16);
            var got = new byte[cpuNorms.Length];
            f.Cache.DownloadNormBytes(0, values, got);
            Assert.Equal(cpuNorms, got);   // byte-equal: an F32-sized store would put zero high halves at every odd 16-bit slot
        }
    }

    [SkippableFact]
    public void Surface_ExposesDistinctConstantsAndStablePerLayerBuffers()
    {
        using var f = Open(bits: 3, headDim: 128, layers: 2);
        var c = f.Cache;
        Assert.NotEqual(0, c.CentroidsPtr);
        Assert.NotEqual(c.SignsKPtr, c.SignsVPtr);
        var centroids = new float[8]; var sk = new float[128]; var sv = new float[128];
        c.DownloadFloats(c.CentroidsPtr, centroids);
        c.DownloadFloats(c.SignsKPtr, sk);
        c.DownloadFloats(c.SignsVPtr, sv);
        Assert.Equal(f.Corpus.K.Codec.Centroids.ToArray(), centroids);
        Assert.Equal(f.Corpus.K.Codec.RotationSigns.ToArray(), sk);
        Assert.Equal(f.Corpus.V.Codec.RotationSigns.ToArray(), sv);
        Assert.NotEqual(sk, sv);                       // independent K / V seeds
        var seen = new HashSet<nint>();
        for (int l = 0; l < 2; l++)
            foreach (nint p in new[] { c.GetKeyCodesPtr(l), c.GetValueCodesPtr(l), c.GetKeyNormsPtr(l), c.GetValueNormsPtr(l) })
                Assert.True(p != 0 && seen.Add(p), "every per-layer buffer is non-null and distinct");

        // Pointers stay valid across updates (no growth), and a second layer's encode leaves the first untouched.
        nint k0 = c.GetKeyCodesPtr(0);
        f.EncodeLayer(1);
        Assert.Equal(k0, c.GetKeyCodesPtr(0));
        var zero = new uint[f.Seq * f.Heads * c.CodeUintsPerVec];
        var got = new uint[zero.Length];
        c.DownloadCodeUints(0, false, got);
        Assert.Equal(zero, got);
        Assert.Equal(f.Seq, c.CurrentLength);
        Assert.Equal(TurboQuantLayout.DefaultNorm, c.Layout.Norm);
        Assert.Equal(CudaTurboQuantInPlaceKvCache.AdmissionBytes(c.Layout, 2, f.Seq), c.AllocatedBytes);
    }

    [SkippableFact]
    public void HostPaths_RefuseLoudly()
    {
        using var f = Open(bits: 4, headDim: 64);
        Assert.Throws<NotSupportedException>(() => f.Cache.GetKeys(0));
        Assert.Throws<NotSupportedException>(() => f.Cache.GetValuesRef(0));
        Assert.Throws<NotSupportedException>(() => f.Cache.Update(default(DotLLM.Core.Tensors.TensorRef), default(DotLLM.Core.Tensors.TensorRef), new int[] { 0 }, 0));
    }

    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError(); return p; }
    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
    private static nint UploadHalf(float[] src)
    {
        var h = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) h[i] = (Half)src[i];
        nint d = Alloc(h.Length * 2L);
        fixed (Half* p = h) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)(h.Length * 2)).ThrowOnError();
        return d;
    }
}

/// <summary>#1892 Task 5: construction rules, byte accounting and routing that need no GPU.</summary>
public sealed class CudaTurboQuantInPlaceKvCacheContractTests
{
    private static (float[] centroids, float[] signsK, float[] signsV, float invSqrtD) Constants(int bits, int hd)
    {
        var ck = new TurboQuantCodec(hd, bits, 0x5DEECE66DUL);
        var cv = new TurboQuantCodec(hd, bits, 0x5DEECE66DUL ^ TurboQuantCodec.VSeedMask);
        return (ck.Centroids.ToArray(), ck.RotationSigns.ToArray(), cv.RotationSigns.ToArray(), ck.InvSqrtD);
    }

    [Fact]
    public void Window_AboveZero_Throws()
    {
        var (c, sk, sv, isd) = Constants(4, 128);
        var ex = Assert.Throws<ArgumentException>(() => new CudaTurboQuantInPlaceKvCache(1, 1, 128, 16, 4, c, sk, sv, isd, window: 8));
        Assert.Contains("window", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EqualKvSigns_Throw()
    {
        var (c, sk, _, isd) = Constants(4, 128);
        Assert.Throws<ArgumentException>(() => new CudaTurboQuantInPlaceKvCache(1, 1, 128, 16, 4, c, sk, sk, isd));
    }

    [Theory]
    [InlineData(96)]
    [InlineData(512)]
    public void UnsupportedHeadDim_Throws(int hd)
    {
        var (c, sk, sv, isd) = Constants(4, 128);
        Assert.Throws<ArgumentException>(() => new CudaTurboQuantInPlaceKvCache(1, 1, hd, 16, 4, c, sk, sv, isd));
    }

    /// <summary>The admission asks for the exact byte count; a refusal names it, and nothing is allocated (no GPU is touched).</summary>
    [Fact]
    public void Admission_UsesTheExactByteCount_AndRefusesBeforeAllocating()
    {
        var (c, sk, sv, isd) = Constants(3, 128);
        var layout = new TurboQuantLayout(128, 4, 3, TurboQuantLayout.DefaultNorm);
        const int layers = 5, rows = 1000;
        long expected = layers * 2L * (layout.CodeBytesPerLayer(rows) + layout.NormBytesPerLayer(rows))
                        + 4L * (8 + 2 * 128);
        Assert.Equal(expected, CudaTurboQuantInPlaceKvCache.AdmissionBytes(layout, layers, rows));

        var before = CudaAdmission.FreeBytesOverride;
        long refusals = CudaAdmission.RefusalCount;
        CudaAdmission.FreeBytesOverride = () => expected - 1 + CudaAdmission.ReserveBytes;   // exactly one byte short
        try
        {
            var ex = Assert.Throws<DeviceMemoryAdmissionException>(
                () => new CudaTurboQuantInPlaceKvCache(layers, 4, 128, rows, 3, c, sk, sv, isd));
            Assert.Equal(expected, ex.RequiredBytes);
            Assert.Equal(refusals + 1, CudaAdmission.RefusalCount);
            CudaAdmission.FreeBytesOverride = () => expected + CudaAdmission.ReserveBytes;    // exactly enough
            Assert.True(CudaAdmission.TryFit(expected, out _));
        }
        finally { CudaAdmission.FreeBytesOverride = before; }
    }

    /// <summary>Steady-state cost per cached token (all layers, K and V) at hd128, 1 layer x 1 kv head.</summary>
    [Theory]
    [InlineData(2, 68)]    // 2 * (32 code bytes + 2 norm bytes)
    [InlineData(3, 100)]   // 2 * (48 + 2)
    [InlineData(4, 132)]   // 2 * (64 + 2)
    [InlineData(8, 260)]   // 2 * (128 + 2)
    public void BytesPerToken_Hd128(int bits, long expected)
    {
        var layout = new TurboQuantLayout(128, 1, bits, TurboQuantLayout.DefaultNorm);
        Assert.Equal(expected, CudaTurboQuantInPlaceKvCache.BytesPerToken(layout, 1));
        Assert.Equal(expected * 32 * 8, CudaTurboQuantInPlaceKvCache.BytesPerToken(layout with { NumKvHeads = 8 }, 32));
    }

    private static KvCacheConfig Cfg(KvCacheDType k, KvCacheDType v, bool qjl = false, int bits = 4)
        => new(k, v, 0, bits, 0x5DEECE66DUL, qjl);

    [Theory]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, false, 128, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, false, 64, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, false, 256, true)]
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, true, 128, false)]    // tq*q
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, false, 96, false)]    // not a power of two
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, false, 512, false)]   // above 256
    [InlineData(KvCacheDType.TurboQuant, KvCacheDType.Q8_0, false, 128, false)]         // mixed
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.TurboQuant, false, 128, false)]         // V-only
    [InlineData(KvCacheDType.F32, KvCacheDType.F32, false, 128, false)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0, false, 128, false)]
    public void Routing_OnlyBothSideMseTurboQuantGoesInPlace(KvCacheDType k, KvCacheDType v, bool qjl, int hd, bool expected)
    {
        var geom = KvGeometry.Uniform(4, 4, hd);
        Assert.Equal(expected, CudaTransformerModel.UsesInPlaceTurboQuant(Cfg(k, v, qjl), geom, hd));
    }

    [Fact]
    public void Routing_NonUniformGeometry_IsNotInPlace()
    {
        var geom = KvGeometry.PerLayer(new[] { 256, 512 });
        Assert.False(CudaTransformerModel.UsesInPlaceTurboQuant(Cfg(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant), geom, 128));
    }
}
