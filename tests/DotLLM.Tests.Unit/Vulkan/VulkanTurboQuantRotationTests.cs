using System;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Tests.Unit.Engine.KvCache;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1892 Task 3: the Vulkan Q/O rotation shaders (<c>tq_rotate_q</c>, <c>tq_rotate_o</c>) against
/// <see cref="TurboQuantCodec"/>'s <c>Rotate</c> / <c>InverseRotate</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTurboQuantRotationTests
{
    private readonly ITestOutputHelper _output;
    public VulkanTurboQuantRotationTests(ITestOutputHelper output) => _output = output;

    public static TheoryData<int, int, int> Geometries => new()
    {
        { 64, 3, 2 }, { 128, 8, 4 }, { 256, 2, 3 },
    };

    [SkippableTheory]
    [MemberData(nameof(Geometries))]
    public void RotateQuery_MatchesCodecRotate(int headDim, int numHeads, int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var codecK = new TurboQuantCodec(headDim, 4, TurboQuantRotationTests.Seed);
        float[] signs = codecK.RotationSigns.ToArray();
        int n = rows * numHeads * headDim;
        float[] q = TurboQuantRotationTests.Gaussian(new Random(headDim * 31 + numHeads), n, 3f);
        float[] expected = (float[])q.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecK.Rotate(expected.AsSpan(v * headDim, headDim));

        using var device = VulkanDevice.Create();
        using var kernel = TqRotateKernel.Create(device, spvDir);
        using var bufQ = device.Allocate((long)n * sizeof(float));
        using var bufSign = device.Allocate((long)signs.Length * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(signs, bufSign);
        kernel.LaunchQuery(bufQ, rows, numHeads, headDim, bufSign);
        var got = new float[n];
        device.Download(bufQ, got);
        AssertClose(expected, got);
    }

    [SkippableTheory]
    [MemberData(nameof(Geometries))]
    public void RotateOutput_InvertsCodec(int headDim, int numHeads, int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var codecV = new TurboQuantCodec(headDim, 4, TurboQuantRotationTests.Seed ^ TurboQuantRotationTests.VSeedXor);
        float[] signs = codecV.RotationSigns.ToArray();
        int n = rows * numHeads * headDim;
        float[] a = TurboQuantRotationTests.Gaussian(new Random(headDim * 17 + rows), n, 2f);
        float[] expected = (float[])a.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecV.InverseRotate(expected.AsSpan(v * headDim, headDim));

        using var device = VulkanDevice.Create();
        using var kernel = TqRotateKernel.Create(device, spvDir);
        using var bufO = device.Allocate((long)n * sizeof(float));
        using var bufSign = device.Allocate((long)signs.Length * sizeof(float));
        device.Upload(a, bufO);
        device.Upload(signs, bufSign);
        kernel.LaunchOutput(bufO, rows, numHeads, headDim, bufSign);
        var got = new float[n];
        device.Download(bufO, got);
        AssertClose(expected, got);
    }

    private void AssertClose(float[] expected, float[] got)
    {
        double max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            max = Math.Max(max, Math.Abs(expected[i] - got[i]));
            Assert.True(Math.Abs(expected[i] - got[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(expected[i])),
                $"i={i}: codec={expected[i]} gpu={got[i]}");
        }
        _output.WriteLine($"maxAbsDiff={max:E3}");
    }

    // Record: a power-of-two head dim up to TqRotateKernel.MaxHeadDim (256). 96 and 512 must throw before anything is recorded.
    [SkippableTheory]
    [InlineData(96)]
    [InlineData(512)]
    public void UnsupportedHeadDim_IsRejected(int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = TqRotateKernel.Create(device, spvDir);
        using var data = device.Allocate((long)2 * 3 * headDim * sizeof(float));
        using var signs = device.Allocate((long)headDim * sizeof(float));
        Assert.Throws<ArgumentException>(() => kernel.LaunchQuery(data, 2, 3, headDim, signs));
        Assert.Throws<ArgumentException>(() => kernel.LaunchOutput(data, 2, 3, headDim, signs));
    }
}
