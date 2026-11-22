using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Tests.Unit.Engine.KvCache;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1892: CUDA TurboQuant encode is byte-identical to the CPU codec (codes and norm bytes), and the cache stores
/// norms in the decided format. Corpus has exact F32 distance ties injected after normalise+rotate. Runs on the
/// framework RTX 3060 (no CUDA on the Strix box).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantByteExactTests
{
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantByteExactTests(ITestOutputHelper output) => _out = output;

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

    public static System.Collections.Generic.IEnumerable<object[]> Configs() => TurboQuantByteExactCorpus.BitsAndHeadDims();

    /// <summary>F32 kernel, un-rounded corpus: codes + F16 norm bytes equal the CPU codec's.</summary>
    [SkippableTheory]
    [MemberData(nameof(Configs))]
    public void Encode_IsByteIdenticalToCpu_CudaF32Kernel(int bits, int headDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TurboQuantAvailable, "turboquant.ptx not present/loaded");

        var corpus = TurboQuantByteExactCorpus.Build(bits, headDim, halfInputs: false);
        var format = TurboQuantLayout.DefaultNorm;
        int seq = TurboQuantByteExactCorpus.SeqLen, heads = TurboQuantByteExactCorpus.NumKvHeads;
        int n = seq * heads;
        string failures = "";

        foreach (bool values in new[] { false, true })
        {
            var side = values ? corpus.V : corpus.K;
            var codec = side.Codec;
            int cu = (codec.CodeBytesPerVector + 3) / 4;
            int nb = TurboQuantNormCodec.BytesPerNorm(format);
            nint dSrc = 0, dCent = 0, dSign = 0, dCodes = 0, dNorms = 0;
            try
            {
                dSrc = Alloc(side.Data.Length * sizeof(float));
                dCent = Alloc(codec.Centroids.Length * sizeof(float));
                dSign = Alloc(headDim * sizeof(float));
                dCodes = Alloc((n * cu + 1) * sizeof(uint));
                dNorms = Alloc(n * nb);
                Upload(dSrc, side.Data);
                Upload(dCent, codec.Centroids.ToArray());
                Upload(dSign, codec.RotationSigns.ToArray());
                kernels.LaunchTurboQuantEncodeF32(dSrc, dCent, dSign, dCodes, dNorms, seq, headDim, heads, codec.MseBits, cu,
                    codec.LevelCount, 0, codec.InvSqrtD, stream.Handle, format);
                stream.Synchronize();
                var gotCodes = new uint[n * cu]; var gotNorms = new byte[n * nb];
                fixed (uint* p = gotCodes) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dCodes, (nuint)(gotCodes.Length * sizeof(uint))).ThrowOnError();
                fixed (byte* p = gotNorms) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dNorms, (nuint)gotNorms.Length).ThrowOnError();
                var (cpuCodes, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, format);
                var (cb, nbad, total) = TurboQuantByteExactCorpus.Compare(cpuCodes, cpuNorms, MemoryMarshal.AsBytes(gotCodes.AsSpan()), gotNorms, cu * 4, nb);
                _out.WriteLine($"CUDA-F32 bits={bits} hd={headDim} {(values ? "V" : "K")}: code mismatches {cb}/{total}, norm mismatches {nbad}/{total} (tie coords {side.TieCoords})");
                if (cb != 0 || nbad != 0) failures += $"{(values ? "V" : "K")}: {cb} code / {nbad} norm of {total}; ";
            }
            finally { Free(dSrc); Free(dCent); Free(dSign); Free(dCodes); Free(dNorms); }
        }
        Assert.True(failures.Length == 0, $"CUDA F32 encode differs from the CPU codec at bits={bits} hd={headDim}: {failures}");
    }

    /// <summary>The production path: CudaTurboQuantKvCache.UpdateDevice (half K/V in, default F16 norms), on a half-rounded corpus.</summary>
    [SkippableTheory]
    [MemberData(nameof(Configs))]
    public void Encode_IsByteIdenticalToCpu_CudaCacheF16(int bits, int headDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TurboQuantF16Available, "turboquant f16 kernels not loaded");

        var corpus = TurboQuantByteExactCorpus.Build(bits, headDim, halfInputs: true);
        var format = TurboQuantLayout.DefaultNorm;
        int seq = TurboQuantByteExactCorpus.SeqLen, heads = TurboQuantByteExactCorpus.NumKvHeads;
        int n = seq * heads;
        var ck = corpus.K.Codec; var cv = corpus.V.Codec;
        using var cache = new CudaTurboQuantKvCache(1, heads, headDim, seq, ck.MseBits, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD);
        Assert.Equal(TqNormFormat.F16, cache.NormFormat);

        nint dK = UploadHalf(corpus.K.Data), dV = UploadHalf(corpus.V.Data);
        try
        {
            var positions = new int[seq];
            for (int i = 0; i < seq; i++) positions[i] = i;
            cache.UpdateDevice(dK, dV, positions, seq, 0, stream.Handle, kernels);
            stream.Synchronize();

            int cu = cache.CodeUintsPerVec;
            int nb = TurboQuantNormCodec.BytesPerNorm(format);
            string failures = "";
            foreach (bool values in new[] { false, true })
            {
                var side = values ? corpus.V : corpus.K;
                var gotCodes = new uint[n * cu]; var gotNorms = new byte[n * nb];
                cache.DownloadCodeUints(0, values, gotCodes);
                cache.DownloadNormBytes(0, values, gotNorms);
                var (cpuCodes, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, format);
                var (cb, nbad, total) = TurboQuantByteExactCorpus.Compare(cpuCodes, cpuNorms, MemoryMarshal.AsBytes(gotCodes.AsSpan()), gotNorms, cu * 4, nb);
                _out.WriteLine($"CUDA-F16cache bits={bits} hd={headDim} {(values ? "V" : "K")}: code mismatches {cb}/{total}, norm mismatches {nbad}/{total} (tie coords {side.TieCoords})");
                if (cb != 0 || nbad != 0) failures += $"{(values ? "V" : "K")}: {cb} code / {nbad} norm of {total}; ";
            }
            Assert.True(failures.Length == 0, $"CUDA cache encode differs from the CPU codec at bits={bits} hd={headDim}: {failures}");
        }
        finally { Free(dK); Free(dV); }
    }

    /// <summary>Stored norm bytes equal the shared codec's encoding of the encode-time norm, for each GPU-supported format.</summary>
    [SkippableTheory]
    [InlineData(TqNormFormat.F32Reference)]
    [InlineData(TqNormFormat.F16)]
    [InlineData(TqNormFormat.Bf16)]
    public void Oracle_NormsRoundTripThroughTheSharedCodec(TqNormFormat format)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TurboQuantF16Available, "turboquant f16 kernels not loaded");

        const int hd = 128, bits = 4;
        var corpus = TurboQuantByteExactCorpus.Build(bits, hd, halfInputs: true);
        int seq = TurboQuantByteExactCorpus.SeqLen, heads = TurboQuantByteExactCorpus.NumKvHeads;
        var ck = corpus.K.Codec; var cv = corpus.V.Codec;
        using var cache = new CudaTurboQuantKvCache(1, heads, hd, seq, ck.MseBits, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD, format);
        nint dK = UploadHalf(corpus.K.Data), dV = UploadHalf(corpus.V.Data);
        try
        {
            var positions = new int[seq];
            for (int i = 0; i < seq; i++) positions[i] = i;
            cache.UpdateDevice(dK, dV, positions, seq, 0, stream.Handle, kernels);
            stream.Synchronize();
            var (_, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(corpus.K, format);
            var got = new byte[cpuNorms.Length];
            cache.DownloadNormBytes(0, values: false, got);
            Assert.Equal(cpuNorms, got);
        }
        finally { Free(dK); Free(dV); }
    }

    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError(); return p; }
    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
    private static void Upload(nint dst, float[] src) { fixed (float* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(float))).ThrowOnError(); }
    private static nint UploadHalf(float[] src)
    {
        var h = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) h[i] = (Half)src[i];
        nint d = Alloc(h.Length * 2L);
        fixed (Half* p = h) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)(h.Length * 2)).ThrowOnError();
        return d;
    }
}
