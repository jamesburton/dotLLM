using System;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Validates the Vulkan TurboQuant (MSE-stage) <b>encode</b> shader. Encode + the slice-4a dequant
/// shader form the full GPU round-trip; we check two things against the CPU codec
/// (<see cref="TurboQuantCodec"/>):
/// (1) GPU encode→decode reconstructs the input within the codec's MSE bound (the codec is correct
///     end-to-end on GPU), and
/// (2) GPU codes agree with CPU codes on the vast majority of coordinates — they differ only by
///     occasional one-level flips at quantization-cell boundaries (GPU fp32 norm vs CPU fp64), which
///     a gross encode bug (wrong rotation/centroid/packing) could not produce.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTurboQuantEncodeF32KernelTests
{
    private readonly ITestOutputHelper _output;
    public VulkanTurboQuantEncodeF32KernelTests(ITestOutputHelper output) => _output = output;

    // Standard-normal Lloyd–Max normalized MSE per bit-width (matches TurboQuantCodecTests).
    private static double ExpectedEps(int bits) => bits switch
    {
        2 => 0.1175, 3 => 0.03454, 4 => 0.009497, _ => throw new ArgumentOutOfRangeException(nameof(bits)),
    };

    [SkippableTheory]
    [InlineData(64, 4, 3, 5)]    // headDim, bits, numKvHeads, positions
    [InlineData(128, 4, 8, 6)]   // Llama-3.1-8B geometry
    [InlineData(128, 2, 2, 4)]
    [InlineData(256, 3, 2, 3)]
    public void EncodeThenDequant_RoundTrips_AndAgreesWithCpuCodes(int headDim, int bits, int numKvHeads, int positions)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var codec = new TurboQuantCodec(headDim, bits, seed: 0x51A7_BEEFUL, useQjl: false);
        int codeBytes = codec.CodeBytesPerVector;
        int codeUintsPerVec = (codeBytes + 3) / 4;
        int numVectors = positions * numKvHeads;
        int stride = numKvHeads * headDim;
        var rng = new Random(0x0DE_C0DE ^ (headDim * 91 + bits * 13 + numKvHeads));

        // Source K/V activations [positions, stride] and the CPU reference (codes + reconstruction).
        var srcFlat = new float[positions * stride];
        var cpuCodes = new byte[numVectors * codeBytes];
        var cpuRecon = new float[positions * stride];
        var tmp = new byte[codeBytes];
        var dec = new float[headDim];
        for (int pos = 0; pos < positions; pos++)
        {
            for (int h = 0; h < numKvHeads; h++)
            {
                int hv = pos * numKvHeads + h;
                float[] x = ScaledGaussian(rng, headDim, (pos + 1) * 8f + h * 2.5f + 1f);
                Array.Copy(x, 0, srcFlat, pos * stride + h * headDim, headDim);
                float n = codec.Encode(x, tmp);
                Array.Copy(tmp, 0, cpuCodes, hv * codeBytes, codeBytes);
                codec.Decode(tmp, n, dec);
                Array.Copy(dec, 0, cpuRecon, pos * stride + h * headDim, headDim);
            }
        }

        float[] centroids = codec.Centroids.ToArray();
        float[] signs = codec.RotationSigns.ToArray();

        using var device = VulkanDevice.Create();
        using var enc = TurboQuantEncodeF32Kernel.Create(device, spvDir);
        using var deq = TurboQuantDequantF32Kernel.Create(device, spvDir);

        using var bufSrc = device.Allocate((long)srcFlat.Length * sizeof(float));
        using var bufCent = device.Allocate((long)centroids.Length * sizeof(float));
        using var bufSign = device.Allocate((long)signs.Length * sizeof(float));
        using var bufCodes = device.Allocate(((long)numVectors * codeUintsPerVec + 1) * sizeof(uint)); // +1 guard for dequant
        using var bufNorms = device.Allocate((long)numVectors * sizeof(float));
        using var bufDst = device.Allocate((long)srcFlat.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(new byte[(int)bufCodes.Size]), bufCodes); // zero (incl. dequant guard uint)
        device.Upload(new ReadOnlySpan<float>(srcFlat), bufSrc);
        device.Upload(new ReadOnlySpan<float>(centroids), bufCent);
        device.Upload(new ReadOnlySpan<float>(signs), bufSign);

        enc.Launch(bufSrc, bufCent, bufSign, bufCodes, bufNorms,
                   positions, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, startPos: 0, codec.InvSqrtD);
        deq.Launch(bufCodes, bufNorms, bufCent, bufSign, bufDst,
                   numVectors, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, codec.InvSqrtD);

        var gpuRecon = new float[srcFlat.Length];
        device.Download(bufDst, gpuRecon);

        // (1) GPU encode→decode reconstructs the input within the codec's MSE bound.
        double sumRel = 0;
        for (int pos = 0; pos < positions; pos++)
        {
            for (int h = 0; h < numKvHeads; h++)
            {
                int off = pos * stride + h * headDim;
                double err = 0, sq = 0;
                for (int i = 0; i < headDim; i++)
                {
                    double d = (double)srcFlat[off + i] - gpuRecon[off + i];
                    err += d * d;
                    sq += (double)srcFlat[off + i] * srcFlat[off + i];
                }
                sumRel += err / sq;
            }
        }
        double relMse = sumRel / numVectors;
        double eps = ExpectedEps(bits);

        // (2) GPU codes agree with CPU codes on the vast majority of coordinates.
        var gpuNorms = new float[numVectors];
        device.Download(bufNorms, gpuNorms);
        var gpuCodeUints = new uint[numVectors * codeUintsPerVec + 1];
        device.Download(bufCodes, AsFloatView(gpuCodeUints)); // download raw bytes via float view
        int agree = 0, totalCoords = numVectors * headDim;
        for (int hv = 0; hv < numVectors; hv++)
            for (int c = 0; c < headDim; c++)
                if (ReadCodeUintLayout(gpuCodeUints, hv, codeUintsPerVec, c, codec.MseBits)
                    == ReadCodeByteLayout(cpuCodes, hv, codeBytes, c, codec.MseBits))
                    agree++;
        double agreeFrac = (double)agree / totalCoords;

        _output.WriteLine($"d={headDim} b={bits} heads={numKvHeads} pos={positions}: " +
                          $"relMse={relMse:F4} (eps_b={eps:F4}), code-agree={agreeFrac:P2}");

        Assert.InRange(relMse, 0.4 * eps, 1.7 * eps);              // round-trip quality tracks the bound
        Assert.True(agreeFrac > 0.97, $"GPU vs CPU code agreement too low: {agreeFrac:P2}");
    }

    // The codes buffer is downloaded through the float-typed Download API; reinterpret as uints.
    private static Span<float> AsFloatView(uint[] uints) =>
        System.Runtime.InteropServices.MemoryMarshal.Cast<uint, float>(uints.AsSpan());

    private static int ReadCodeUintLayout(uint[] codes, int hv, int codeUintsPerVec, int coord, int mseBits)
    {
        long baseBit = (long)hv * codeUintsPerVec * 32 + (long)coord * mseBits;
        int widx = (int)(baseBit >> 5), boff = (int)(baseBit & 31);
        uint val = codes[widx] >> boff;
        if (boff + mseBits > 32) val |= codes[widx + 1] << (32 - boff);
        return (int)(val & ((1u << mseBits) - 1u));
    }

    private static int ReadCodeByteLayout(byte[] codes, int hv, int codeBytes, int coord, int mseBits)
    {
        int bitPos = hv * codeBytes * 8 + coord * mseBits;
        int idx = 0;
        for (int b = 0; b < mseBits; b++)
        {
            int p = bitPos + b;
            if ((codes[p >> 3] & (1 << (p & 7))) != 0) idx |= 1 << b;
        }
        return idx;
    }

    private static float[] ScaledGaussian(Random rng, int n, float targetNorm)
    {
        var v = new float[n];
        double sq = 0;
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            v[i] = g;
            sq += (double)g * g;
        }
        float scale = targetNorm / (float)Math.Sqrt(sq);
        for (int i = 0; i < n; i++) v[i] *= scale;
        return v;
    }
}
