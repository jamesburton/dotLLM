using System;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity test for the Vulkan TurboQuant (MSE-stage) dequant kernel against the CPU oracle
/// <see cref="TurboQuantCodec"/>.<c>Decode</c> (useQjl = false). The shader and the codec read the
/// same codes/norm and reproduce the same pipeline (centroid lookup → unnormalized Walsh–Hadamard →
/// ×invSqrtD×sign×norm), so outputs match to within tight float tolerance. A wrong rotation /
/// centroid / sign in the shader would diverge by O(norm) per element and fail.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTurboQuantDequantF32KernelTests
{
    private readonly ITestOutputHelper _output;
    public VulkanTurboQuantDequantF32KernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(64, 4, 3, 5)]    // headDim, bits, numKvHeads, positions
    [InlineData(128, 4, 8, 6)]   // Llama-3.1-8B geometry (headDim 128, 8 KV heads)
    [InlineData(128, 2, 2, 4)]
    [InlineData(256, 3, 2, 3)]
    public void Dequant_MatchesCpuCodec(int headDim, int bits, int numKvHeads, int positions)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var codec = new TurboQuantCodec(headDim, bits, seed: 0x7A2B_C0DEUL, useQjl: false);
        int codeBytes = codec.CodeBytesPerVector;
        int codeUintsPerVec = (codeBytes + 3) / 4;
        int numVectors = positions * numKvHeads;
        int stride = numKvHeads * headDim;
        var rng = new Random(0xB16_5EED ^ (headDim * 131 + bits * 17 + numKvHeads));

        // Build the compressed store in the shader's uint-aligned-per-vector layout, plus the CPU
        // reference reconstruction.
        var codeBytesBuf = new byte[(numVectors * codeUintsPerVec + 1) * sizeof(uint)]; // +1 guard uint
        var norms = new float[numVectors];
        var expected = new float[positions * stride];
        var tmp = new byte[codeBytes];
        var dec = new float[headDim];

        for (int pos = 0; pos < positions; pos++)
        {
            for (int h = 0; h < numKvHeads; h++)
            {
                int hv = pos * numKvHeads + h;
                float target = (pos + 1) * 10f + h * 3f + 1f; // distinct per (pos,head)
                float[] x = ScaledGaussian(rng, headDim, target);

                norms[hv] = codec.Encode(x, tmp);
                tmp.CopyTo(codeBytesBuf.AsSpan(hv * codeUintsPerVec * sizeof(uint), codeBytes));

                codec.Decode(tmp, norms[hv], dec);
                Array.Copy(dec, 0, expected, pos * stride + h * headDim, headDim);
            }
        }

        float[] centroids = codec.Centroids.ToArray();
        float[] signs = codec.RotationSigns.ToArray();

        using var device = VulkanDevice.Create();
        using var kernel = TurboQuantDequantF32Kernel.Create(device, spvDir);

        using var bufCodes = device.Allocate(codeBytesBuf.Length);
        using var bufNorms = device.Allocate((long)numVectors * sizeof(float));
        using var bufCent = device.Allocate((long)centroids.Length * sizeof(float));
        using var bufSign = device.Allocate((long)signs.Length * sizeof(float));
        using var bufDst = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(codeBytesBuf), bufCodes);
        device.Upload(new ReadOnlySpan<float>(norms), bufNorms);
        device.Upload(new ReadOnlySpan<float>(centroids), bufCent);
        device.Upload(new ReadOnlySpan<float>(signs), bufSign);

        kernel.Launch(bufCodes, bufNorms, bufCent, bufSign, bufDst,
                      numVectors, headDim, numKvHeads, codec.MseBits, codeUintsPerVec, codec.InvSqrtD);

        var actual = new float[expected.Length];
        device.Download(bufDst, actual);

        double maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs((double)expected[i] - actual[i]));
        _output.WriteLine($"d={headDim} b={bits} heads={numKvHeads} pos={positions}: maxAbsDiff={maxAbs:E3}");

        // Same arithmetic in the same order on both paths — expect tight agreement. Bound well below
        // the per-element magnitude (norms ~ tens) so a transposed/dropped rotation cannot slip by.
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs((double)expected[i] - actual[i]) <= 2e-3 + 1e-4 * Math.Abs(expected[i]),
                $"mismatch at {i}: cpu={expected[i]} gpu={actual[i]}");
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
