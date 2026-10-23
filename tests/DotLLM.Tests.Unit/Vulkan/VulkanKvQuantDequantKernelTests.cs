using System;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Round-trip parity tests for the Vulkan KV-cache quantize → dequant shader pipeline.
/// Uploads FP32 data to GPU, runs the quantize kernel, then the dequant kernel, downloads
/// the result, and compares against a scalar CPU reference that implements the identical
/// split-layout algorithm (float scale, not half). Both paths do the same arithmetic, so
/// results should agree to within tight float tolerance — a broken shader would diverge by
/// O(max_abs) per element.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvQuantDequantKernelTests
{
    private const int BlockSize = 32;

    private readonly ITestOutputHelper _output;
    public VulkanKvQuantDequantKernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, 32)]
    [InlineData(KvCacheDType.Q8_0, 128)]
    [InlineData(KvCacheDType.Q8_0, 1024)]
    [InlineData(KvCacheDType.Q4_0, 32)]
    [InlineData(KvCacheDType.Q4_0, 128)]
    [InlineData(KvCacheDType.Q4_0, 1024)]
    public void RoundTrip_MatchesCpuReference(KvCacheDType dtype, int elementCount)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xCA5E ^ ((int)dtype * 37 + elementCount));
        float[] src = RandomFloats(rng, elementCount, magnitude: 5.0f);

        float[] cpuExpected = CpuRoundTrip(src, dtype);

        int totalBlocks = elementCount / BlockSize;
        int quantUints = dtype == KvCacheDType.Q8_0 ? 8 : 4;

        using var device = VulkanDevice.Create();
        using var quantizeKernel = KvQuantizeKernel.Create(device, spvDir, dtype);
        using var dequantKernel = KvDequantKernel.Create(device, spvDir, dtype);

        using var srcBuf = device.Allocate((long)elementCount * sizeof(float));
        using var scalesBuf = device.Allocate((long)totalBlocks * sizeof(float));
        using var quantsBuf = device.Allocate((long)totalBlocks * quantUints * sizeof(uint));
        using var dstBuf = device.Allocate((long)elementCount * sizeof(float));

        device.Upload(new ReadOnlySpan<float>(src), srcBuf);

        quantizeKernel.Launch(srcBuf, scalesBuf, quantsBuf, totalBlocks, srcOffset: 0, dstOffset: 0);
        dequantKernel.Launch(scalesBuf, quantsBuf, dstBuf, totalBlocks, srcOffset: 0, dstOffset: 0);

        var actual = new float[elementCount];
        device.Download(dstBuf, actual);

        double maxAbs = 0, sumAbs = 0;
        long mismatches = 0;
        for (int i = 0; i < elementCount; i++)
        {
            double d = Math.Abs((double)cpuExpected[i] - actual[i]);
            sumAbs += d;
            if (d > maxAbs) maxAbs = d;
            if (d > 1e-4 + 1e-5 * Math.Abs(cpuExpected[i])) mismatches++;
        }

        double meanAbs = sumAbs / elementCount;
        _output.WriteLine($"{dtype} n={elementCount}: maxAbs={maxAbs:E3} meanAbs={meanAbs:E3} mismatches={mismatches}");

        Assert.True(mismatches == 0, $"GPU/CPU mismatch at {mismatches}/{elementCount} elements (maxAbs={maxAbs:E3})");
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, 256)]
    [InlineData(KvCacheDType.Q4_0, 256)]
    public void RoundTrip_WithOffset(KvCacheDType dtype, int elementCount)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEEF ^ (int)dtype);
        int totalBlocks = elementCount / BlockSize;
        int quantUints = dtype == KvCacheDType.Q8_0 ? 8 : 4;
        int offsetBlocks = 3;
        int srcOffsetElements = offsetBlocks * BlockSize;

        float[] src = RandomFloats(rng, elementCount + srcOffsetElements, magnitude: 3.0f);
        float[] cpuExpected = CpuRoundTrip(src.AsSpan(srcOffsetElements, elementCount).ToArray(), dtype);

        using var device = VulkanDevice.Create();
        using var quantizeKernel = KvQuantizeKernel.Create(device, spvDir, dtype);
        using var dequantKernel = KvDequantKernel.Create(device, spvDir, dtype);

        int allocBlocks = totalBlocks + offsetBlocks;
        using var srcBuf = device.Allocate((long)(elementCount + srcOffsetElements) * sizeof(float));
        using var scalesBuf = device.Allocate((long)allocBlocks * sizeof(float));
        using var quantsBuf = device.Allocate((long)allocBlocks * quantUints * sizeof(uint));
        using var dstBuf = device.Allocate((long)elementCount * sizeof(float));

        device.Upload(new ReadOnlySpan<float>(src), srcBuf);

        quantizeKernel.Launch(srcBuf, scalesBuf, quantsBuf, totalBlocks,
            srcOffset: srcOffsetElements, dstOffset: offsetBlocks);
        dequantKernel.Launch(scalesBuf, quantsBuf, dstBuf, totalBlocks,
            srcOffset: offsetBlocks, dstOffset: 0);

        var actual = new float[elementCount];
        device.Download(dstBuf, actual);

        double maxAbs = 0;
        long mismatches = 0;
        for (int i = 0; i < elementCount; i++)
        {
            double d = Math.Abs((double)cpuExpected[i] - actual[i]);
            if (d > maxAbs) maxAbs = d;
            if (d > 1e-4 + 1e-5 * Math.Abs(cpuExpected[i])) mismatches++;
        }

        _output.WriteLine($"{dtype} n={elementCount} offset={offsetBlocks}blks: maxAbs={maxAbs:E3} mismatches={mismatches}");
        Assert.True(mismatches == 0, $"GPU/CPU mismatch with offset: {mismatches}/{elementCount}");
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void AllZeros_RoundTripsToZero(KvCacheDType dtype)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elementCount = 64;
        int totalBlocks = elementCount / BlockSize;
        int quantUints = dtype == KvCacheDType.Q8_0 ? 8 : 4;

        using var device = VulkanDevice.Create();
        using var quantizeKernel = KvQuantizeKernel.Create(device, spvDir, dtype);
        using var dequantKernel = KvDequantKernel.Create(device, spvDir, dtype);

        using var srcBuf = device.Allocate((long)elementCount * sizeof(float));
        using var scalesBuf = device.Allocate((long)totalBlocks * sizeof(float));
        using var quantsBuf = device.Allocate((long)totalBlocks * quantUints * sizeof(uint));
        using var dstBuf = device.Allocate((long)elementCount * sizeof(float));

        device.Upload(new ReadOnlySpan<float>(new float[elementCount]), srcBuf);

        quantizeKernel.Launch(srcBuf, scalesBuf, quantsBuf, totalBlocks, 0, 0);
        dequantKernel.Launch(scalesBuf, quantsBuf, dstBuf, totalBlocks, 0, 0);

        var actual = new float[elementCount];
        device.Download(dstBuf, actual);

        for (int i = 0; i < elementCount; i++)
            Assert.Equal(0.0f, actual[i]);
    }

    // ── CPU scalar reference (split-layout, float scale — matches shader exactly) ──

    private static float[] CpuRoundTrip(float[] src, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => CpuRoundTripQ8_0(src),
        KvCacheDType.Q4_0 => CpuRoundTripQ4_0(src),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };

    private static float[] CpuRoundTripQ8_0(float[] src)
    {
        int blockCount = src.Length / BlockSize;
        var result = new float[src.Length];

        for (int b = 0; b < blockCount; b++)
        {
            int off = b * BlockSize;
            float amax = 0;
            for (int i = 0; i < BlockSize; i++)
                amax = Math.Max(amax, Math.Abs(src[off + i]));

            float scale = amax / 127.0f;
            float invScale = scale != 0 ? 1.0f / scale : 0;

            for (int i = 0; i < BlockSize; i++)
            {
                int q = Math.Clamp((int)MathF.Round(src[off + i] * invScale), -128, 127);
                result[off + i] = scale * q;
            }
        }
        return result;
    }

    private static float[] CpuRoundTripQ4_0(float[] src)
    {
        int blockCount = src.Length / BlockSize;
        var result = new float[src.Length];

        for (int b = 0; b < blockCount; b++)
        {
            int off = b * BlockSize;
            float amax = 0;
            for (int i = 0; i < BlockSize; i++)
                amax = Math.Max(amax, Math.Abs(src[off + i]));

            float scale = amax / 7.0f;
            float invScale = scale != 0 ? 1.0f / scale : 0;

            for (int j = 0; j < 16; j++)
            {
                int lo = Math.Clamp((int)MathF.Round(src[off + 2 * j] * invScale) + 8, 0, 15);
                int hi = Math.Clamp((int)MathF.Round(src[off + 2 * j + 1] * invScale) + 8, 0, 15);
                result[off + 2 * j] = scale * (lo - 8);
                result[off + 2 * j + 1] = scale * (hi - 8);
            }
        }
        return result;
    }

    private static float[] RandomFloats(Random rng, int count, float magnitude)
    {
        var buf = new float[count];
        for (int i = 0; i < count; i++)
            buf[i] = ((float)rng.NextDouble() * 2 - 1) * magnitude;
        return buf;
    }
}
