using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Runtime parity for the CUDA TurboQuant (MSE-stage) dequant + encode kernels against the CPU
/// <see cref="TurboQuantCodec"/> — the CUDA analogue of the Vulkan kernel tests. Dequant is checked
/// bit-exact (codes built by the CPU codec, F32 in/out, no-FMA PTX), and encode is checked by the
/// GPU encode→dequant round-trip (relMse within the codec bound) plus high code agreement with the
/// CPU codec. Gated on a CUDA GPU + turboquant.ptx (runs on T5500, not the Strix Halo box).
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaTurboQuantKernelTests
{
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantKernelTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
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

    private static double ExpectedEps(int bits) => bits switch
    {
        2 => 0.1175, 3 => 0.03454, 4 => 0.009497, _ => throw new ArgumentOutOfRangeException(nameof(bits)),
    };

    [SkippableTheory]
    [InlineData(64, 4, 3, 5)]    // headDim, bits, numKvHeads, positions
    [InlineData(128, 4, 8, 6)]   // Llama-3.1-8B geometry
    [InlineData(128, 2, 2, 4)]
    [InlineData(256, 3, 2, 3)]
    public void Dequant_MatchesCpuCodec_BitExact(int headDim, int bits, int numKvHeads, int positions)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TurboQuantAvailable, "turboquant.ptx not present/loaded");

        var codec = new TurboQuantCodec(headDim, bits, seed: 0x7A2B_C0DEUL, useQjl: false);
        int codeBytes = codec.CodeBytesPerVector;
        int codeUintsPerVec = (codeBytes + 3) / 4;
        int numVectors = positions * numKvHeads;
        int stride = numKvHeads * headDim;
        var rng = new Random(0xB16_5EED ^ (headDim * 131 + bits * 17 + numKvHeads));

        var codeBytesBuf = new byte[(numVectors * codeUintsPerVec + 1) * sizeof(uint)];
        var norms = new float[numVectors];
        var expected = new float[positions * stride];
        var tmp = new byte[codeBytes];
        var dec = new float[headDim];
        for (int pos = 0; pos < positions; pos++)
            for (int h = 0; h < numKvHeads; h++)
            {
                int hv = pos * numKvHeads + h;
                float[] x = ScaledGaussian(rng, headDim, (pos + 1) * 10f + h * 3f + 1f);
                norms[hv] = codec.Encode(x, tmp);
                tmp.CopyTo(codeBytesBuf.AsSpan(hv * codeUintsPerVec * sizeof(uint), codeBytes));
                codec.Decode(tmp, norms[hv], dec);
                Array.Copy(dec, 0, expected, pos * stride + h * headDim, headDim);
            }
        float[] centroids = codec.Centroids.ToArray();
        float[] signs = codec.RotationSigns.ToArray();

        var actual = new float[expected.Length];
        nint dCodes = 0, dNorms = 0, dCent = 0, dSign = 0, dDst = 0;
        try
        {
            dCodes = Alloc(codeBytesBuf.Length);
            dNorms = Alloc(norms.Length * sizeof(float));
            dCent = Alloc(centroids.Length * sizeof(float));
            dSign = Alloc(signs.Length * sizeof(float));
            dDst = Alloc(expected.Length * sizeof(float));
            UploadBytes(dCodes, codeBytesBuf);
            UploadFloats(dNorms, norms);
            UploadFloats(dCent, centroids);
            UploadFloats(dSign, signs);

            kernels.LaunchTurboQuantDequantF32(dCodes, dNorms, dCent, dSign, dDst,
                numVectors, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, codec.InvSqrtD, stream.Handle);
            stream.Synchronize();
            DownloadFloats(actual, dDst);
        }
        finally { Free(dCodes); Free(dNorms); Free(dCent); Free(dSign); Free(dDst); }

        double maxAbs = 0;
        for (int i = 0; i < expected.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs((double)expected[i] - actual[i]));
        _out.WriteLine($"d={headDim} b={bits} heads={numKvHeads} pos={positions}: maxAbsDiff={maxAbs:E3}");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs((double)expected[i] - actual[i]) <= 2e-3 + 1e-4 * Math.Abs(expected[i]),
                $"mismatch at {i}: cpu={expected[i]} gpu={actual[i]}");
    }

    [SkippableTheory]
    [InlineData(64, 4, 3, 5)]
    [InlineData(128, 4, 8, 6)]
    [InlineData(128, 2, 2, 4)]
    [InlineData(256, 3, 2, 3)]
    public void EncodeThenDequant_RoundTrips_AndAgreesWithCpuCodes(int headDim, int bits, int numKvHeads, int positions)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TurboQuantAvailable, "turboquant.ptx not present/loaded");

        var codec = new TurboQuantCodec(headDim, bits, seed: 0x51A7_BEEFUL, useQjl: false);
        int codeBytes = codec.CodeBytesPerVector;
        int codeUintsPerVec = (codeBytes + 3) / 4;
        int numVectors = positions * numKvHeads;
        int stride = numKvHeads * headDim;
        var rng = new Random(0x0DE_C0DE ^ (headDim * 91 + bits * 13 + numKvHeads));

        var srcFlat = new float[positions * stride];
        var cpuCodes = new byte[numVectors * codeBytes];
        var tmp = new byte[codeBytes];
        for (int pos = 0; pos < positions; pos++)
            for (int h = 0; h < numKvHeads; h++)
            {
                int hv = pos * numKvHeads + h;
                float[] x = ScaledGaussian(rng, headDim, (pos + 1) * 8f + h * 2.5f + 1f);
                Array.Copy(x, 0, srcFlat, pos * stride + h * headDim, headDim);
                codec.Encode(x, tmp);
                Array.Copy(tmp, 0, cpuCodes, hv * codeBytes, codeBytes);
            }
        float[] centroids = codec.Centroids.ToArray();
        float[] signs = codec.RotationSigns.ToArray();

        var gpuRecon = new float[srcFlat.Length];
        var gpuCodeUints = new uint[numVectors * codeUintsPerVec + 1];
        var gpuNorms = new float[numVectors];
        nint dSrc = 0, dCent = 0, dSign = 0, dCodes = 0, dNorms = 0, dDst = 0;
        try
        {
            dSrc = Alloc(srcFlat.Length * sizeof(float));
            dCent = Alloc(centroids.Length * sizeof(float));
            dSign = Alloc(signs.Length * sizeof(float));
            dCodes = Alloc(gpuCodeUints.Length * sizeof(uint));
            dNorms = Alloc(numVectors * sizeof(float));
            dDst = Alloc(srcFlat.Length * sizeof(float));
            UploadFloats(dSrc, srcFlat);
            UploadFloats(dCent, centroids);
            UploadFloats(dSign, signs);
            // zero codes (incl. dequant guard uint)
            UploadBytes(dCodes, new byte[gpuCodeUints.Length * sizeof(uint)]);

            kernels.LaunchTurboQuantEncodeF32(dSrc, dCent, dSign, dCodes, dNorms,
                positions, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, codec.LevelCount, 0, codec.InvSqrtD, stream.Handle);
            stream.Synchronize();
            kernels.LaunchTurboQuantDequantF32(dCodes, dNorms, dCent, dSign, dDst,
                numVectors, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, codec.InvSqrtD, stream.Handle);
            stream.Synchronize();
            DownloadFloats(gpuRecon, dDst);
            DownloadUints(gpuCodeUints, dCodes);
        }
        finally { Free(dSrc); Free(dCent); Free(dSign); Free(dCodes); Free(dNorms); Free(dDst); }

        double sumRel = 0;
        for (int pos = 0; pos < positions; pos++)
            for (int h = 0; h < numKvHeads; h++)
            {
                int off = pos * stride + h * headDim;
                double err = 0, sq = 0;
                for (int i = 0; i < headDim; i++)
                {
                    double d = (double)srcFlat[off + i] - gpuRecon[off + i];
                    err += d * d; sq += (double)srcFlat[off + i] * srcFlat[off + i];
                }
                sumRel += err / sq;
            }
        double relMse = sumRel / numVectors;
        double eps = ExpectedEps(bits);

        int agree = 0, totalCoords = numVectors * headDim;
        for (int hv = 0; hv < numVectors; hv++)
            for (int c = 0; c < headDim; c++)
                if (ReadCodeUint(gpuCodeUints, hv, codeUintsPerVec, c, codec.MseBits)
                    == ReadCodeByte(cpuCodes, hv, codeBytes, c, codec.MseBits)) agree++;
        double agreeFrac = (double)agree / totalCoords;
        _out.WriteLine($"d={headDim} b={bits} heads={numKvHeads} pos={positions}: relMse={relMse:F4} (eps={eps:F4}), code-agree={agreeFrac:P2}");

        Assert.InRange(relMse, 0.4 * eps, 1.7 * eps);
        Assert.True(agreeFrac > 0.97, $"GPU vs CPU code agreement too low: {agreeFrac:P2}");
    }

    // ── CUDA device-memory helpers ──
    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError(); return p; }
    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
    private static void UploadBytes(nint dst, byte[] src) { fixed (byte* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)src.Length).ThrowOnError(); }
    private static void UploadFloats(nint dst, float[] src) { fixed (float* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(float))).ThrowOnError(); }
    private static void DownloadFloats(float[] dst, nint src) { fixed (float* p = dst) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(dst.Length * sizeof(float))).ThrowOnError(); }
    private static void DownloadUints(uint[] dst, nint src) { fixed (uint* p = dst) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(dst.Length * sizeof(uint))).ThrowOnError(); }

    private static int ReadCodeUint(uint[] codes, int hv, int codeUintsPerVec, int coord, int mseBits)
    {
        long baseBit = (long)hv * codeUintsPerVec * 32 + (long)coord * mseBits;
        int widx = (int)(baseBit >> 5), boff = (int)(baseBit & 31);
        uint val = codes[widx] >> boff;
        if (boff + mseBits > 32) val |= codes[widx + 1] << (32 - boff);
        return (int)(val & ((1u << mseBits) - 1u));
    }

    private static int ReadCodeByte(byte[] codes, int hv, int codeBytes, int coord, int mseBits)
    {
        int bitPos = hv * codeBytes * 8 + coord * mseBits, idx = 0;
        for (int b = 0; b < mseBits; b++) { int p = bitPos + b; if ((codes[p >> 3] & (1 << (p & 7))) != 0) idx |= 1 << b; }
        return idx;
    }

    private static float[] ScaledGaussian(Random rng, int n, float targetNorm)
    {
        var v = new float[n]; double sq = 0;
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            v[i] = g; sq += (double)g * g;
        }
        float scale = targetNorm / (float)Math.Sqrt(sq);
        for (int i = 0; i < n; i++) v[i] *= scale;
        return v;
    }
}
