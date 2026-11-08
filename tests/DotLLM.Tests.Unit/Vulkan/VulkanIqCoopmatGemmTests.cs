using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1677: the native dense IQ1_S / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S coopmat GEMMs against the CPU dequantizer's double-precision
/// product, and against the F16 scratch path (<see cref="IqF16PrefillMatmul"/>: dequantise to F16, F16 coopmat GEMM), which stages
/// bit-identical F16 operands and so must agree to accumulation order. The second comparison is the discriminating one: any deviation
/// in the staged expression (sign, sub-scale nibble, grid half, delta) shows up as a 1e-2..1e-1 relative error against it, far above
/// the 1e-4 bound (mutants in the PR body).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanIqCoopmatGemmTests
{
    public static IEnumerable<object[]> Cases()
    {
        (int m, int k, int n)[] shapes =
        {
            (70, 256, 32),     // partial M and N tiles, one super-block
            (130, 512, 129),   // M and N each cross a 128 tile by one
            (64, 3072, 40),    // Llama-3.2-3B hidden as K
            (33, 8192, 64),    // Llama-3.2-3B intermediate as K (32 super-blocks)
            (256, 1024, 256),  // whole tiles only (the direct-store path)
            (70, 256, 2),      // the smallest prefill batch
        };
        foreach (var qt in MatMulIqGemmCoopmatKernel.Formats)
            foreach (var (m, k, n) in shapes)
                yield return new object[] { qt, m, k, n };
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void NativeGemm_MatchesCpuReference_AndF16Path(QuantizationType qt, int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulIqGemmCoopmatKernel.IsSupportedOn(device, spvDir, qt), "no wave64 cooperative-matrix device");

        var rng = new Random((int)qt * 977 + m * 131 + k * 7 + n);
        byte[] w = RandomIqRows(rng, qt, m, k);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] wf = new float[m * k];
        unsafe
        {
            fixed (byte* pw = w)
                Dequantize.ToFloat32((nint)pw, wf.Length, qt, wf);
        }
        float[] expected = new float[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[t * k + c];
                expected[t * m + r] = (float)acc;
            }

        using var native = MatMulIqGemmCoopmatKernel.Create(device, spvDir, qt);
        using var f16Gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        using var f16Path = IqF16PrefillMatmul.TryCreate(device, spvDir, f16Gemm, (long)m * k)
            ?? throw new InvalidOperationException("F16 prefill helper unavailable");
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        using var bufY16 = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        long before = Interlocked.Read(ref MatMulIqGemmCoopmatKernel.Dispatches);
        native.Launch(bufW, bufX, bufY, m, k, n);
        Assert.Equal(before + 1, Interlocked.Read(ref MatMulIqGemmCoopmatKernel.Dispatches));
        bool compareF16 = n >= f16Path.MinTokens;   // below its threshold the F16 path declines by design
        if (compareF16)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            Assert.True(f16Path.TryRecord(ctx.CommandBuffer, qt, bufW, bufX, bufY16, m, k, n), "the F16 path declined");
            ctx.SubmitAndWait();
        }
        float[] actual = new float[n * m], viaF16 = new float[n * m];
        device.Download(bufY, actual);
        device.Download(bufY16, viaF16);

        float maxAbs = expected.Max(MathF.Abs);
        Assert.True(maxAbs > 0, "degenerate fixture");
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 4e-3f * maxAbs,
                $"{qt} m={m} k={k} n={n} [{i}]: expected {expected[i]}, native {actual[i]} (max |y| {maxAbs})");
            if (compareF16)
                Assert.True(MathF.Abs(actual[i] - viaF16[i]) <= 1e-4f * maxAbs,
                    $"{qt} m={m} k={k} n={n} [{i}]: F16 path {viaF16[i]}, native {actual[i]} (same F16 operands)");
        }
    }

    /// <summary>Every byte pattern is a valid block for these formats (every grid / sign index is in range), so random bytes with a sane fp16 d per super-block are a complete fixture.</summary>
    [Theory]
    [InlineData(QuantizationType.IQ1_S, 50)]
    [InlineData(QuantizationType.IQ2_XS, 74)]
    [InlineData(QuantizationType.IQ2_S, 82)]
    [InlineData(QuantizationType.IQ3_XXS, 98)]
    [InlineData(QuantizationType.IQ3_S, 110)]
    public void BlockBytes_MatchTheCpuFormatTable(QuantizationType qt, int expectedBytes)
    {
        Assert.Equal(expectedBytes, MatMulIqGemmCoopmatKernel.BlockBytesOf(qt));
        Assert.Equal(expectedBytes, RandomIqRows(new Random(1), qt, 1, 256).Length);
        Assert.True(MatMulIqGemmCoopmatKernel.Handles(qt));
        Assert.False(MatMulIqGemmCoopmatKernel.Handles(QuantizationType.IQ2_XXS));   // has its own kernel (#1069)
        Assert.False(MatMulIqGemmCoopmatKernel.Handles(QuantizationType.IQ4_XS));    // #601
    }

    /// <summary>
    /// Random but valid rows of <paramref name="qt"/>: fp16 d in [2.5e-4, 7.5e-4) per super-block (weights of the same order as the other IQ
    /// fixtures), every other byte random (all grid / sign / scale indices of these formats are in range for any bit pattern).
    /// </summary>
    internal static byte[] RandomIqRows(Random rng, QuantizationType qt, int rows, int cols)
    {
        int blockBytes = MatMulIqGemmCoopmatKernel.BlockBytesOf(qt);
        int blocks = rows * (cols / 256);
        byte[] bytes = new byte[blocks * blockBytes];
        rng.NextBytes(bytes);
        for (int b = 0; b < blocks; b++)
            BitConverter.TryWriteBytes(bytes.AsSpan(b * blockBytes, 2), (Half)(2.5e-4 + rng.NextDouble() * 5e-4));
        return bytes;
    }
}
