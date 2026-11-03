using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The NVFP4 routed-bank indexed matmul (#1579): reads the packed bank in place and multiplies expert <c>e</c>'s result by
/// <c>scale[e]</c>. Oracle = the CPU <see cref="Dequantize"/> of the same bytes, dotted in double, scale applied after the dot.
/// Per-expert scales are distinct (0.3..6) so "scale ignored", "expert 0's scale for all" and "scale indexed by row not expert" are all
/// far from correct (asserted via substitution on the oracle, so the fixture provably discriminates).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNvfp4MoeIndexedTests
{
    [SkippableTheory]
    [InlineData(5, 6, 8, 64)]
    [InlineData(40, 7, 33, 192)]
    [InlineData(9, 16, 64, 2048)]   // Flint hidden
    public void IndexedMatmul_AppliesPerExpertScale(int n, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x1579 + n * 31 + numExperts * 17 + m * 11 + k);
        int blocksPerRow = k / 64;
        long rowBytes = (long)blocksPerRow * 36;
        byte[] bank = new byte[(long)numExperts * m * rowBytes];
        rng.NextBytes(bank);
        for (long off = 0; off < bank.Length; off += 36)
            for (int s = 0; s < 4; s++) bank[off + s] = (byte)rng.Next(0x18, 0x30);   // valid UE4M3, moderate magnitudes

        float[] deq = new float[(long)numExperts * m * k];
        unsafe
        {
            fixed (byte* p = bank) Dequantize.ToFloat32((nint)p, deq.Length, QuantizationType.NVFP4, deq);
        }

        float[] scale = new float[numExperts];
        for (int e = 0; e < numExperts; e++) scale[e] = 0.3f + 5.7f * e / Math.Max(1, numExperts - 1);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = rng.Next(numExperts);

        float[] Oracle(Func<int, int, float> scaleOf)
        {
            var r = new float[n * m];
            for (int row = 0; row < n; row++)
                for (int o = 0; o < m; o++)
                {
                    double acc = 0;
                    long wb = ((long)indices[row] * m + o) * k;
                    for (int j = 0; j < k; j++) acc += deq[wb + j] * x[row * k + j];
                    r[row * m + o] = (float)(acc * scaleOf(row, indices[row]));
                }
            return r;
        }

        float[] expected = Oracle((_, e) => scale[e]);
        float[] noScale = Oracle((_, _) => 1f);
        float[] expert0 = Oracle((_, _) => scale[0]);
        float[] byRow = Oracle((row, _) => scale[row % numExperts]);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulNvfp4F32Kernel.Create(device, spvDir);
        using var bankBuf = device.Allocate(bank.Length);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));
        using var sBuf = device.Allocate((long)scale.Length * sizeof(float));
        device.Upload(bank, bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        device.Upload(scale, sBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, sBuf, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);
        for (int i = 0; i < expected.Length; i++)
        {
            float bar = 1e-3f + 2e-4f * MathF.Abs(expected[i]) + 1e-5f * k * scale[indices[i / m]];
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= bar,
                $"row={i / m}, col={i % m}: ref={expected[i]:F6} vs vulkan={actual[i]:F6}");
        }

        // Substitution proof: each wrong implementation differs from the correct result by much more than the kernel's error.
        foreach (var (name, wrong) in new[] { ("no scale", noScale), ("expert 0 for all", expert0), ("by row", byRow) })
        {
            double num = 0, den = 0;
            for (int i = 0; i < expected.Length; i++) { double d = wrong[i] - expected[i]; num += d * d; den += (double)expected[i] * expected[i]; }
            Assert.True(Math.Sqrt(num / den) > 0.2, $"fixture cannot tell '{name}' from correct");
        }
    }

    [SkippableTheory]
    [InlineData(7, 100, 64)]
    [InlineData(12, 200, 256)]
    [InlineData(16, 70, 2048)]   // Flint hidden
    public void GroupedCoopmat_AppliesPerExpertScale(int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulLegacyQuantCoopmatKernel.IsSupportedOn(device, spvDir, MoeGroupedLegacyQuant.Nvfp4), "needs wave64 coopmat");
        var rng = new Random(0x1580 + numExperts * 7 + m + k);
        long rowBytes = (long)(k / 64) * 36;
        byte[] bank = new byte[(long)numExperts * m * rowBytes];
        rng.NextBytes(bank);
        for (long off = 0; off < bank.Length; off += 36)
            for (int sb = 0; sb < 4; sb++) bank[off + sb] = (byte)rng.Next(0x18, 0x30);
        float[] deq = new float[(long)numExperts * m * k];
        unsafe { fixed (byte* p = bank) Dequantize.ToFloat32((nint)p, deq.Length, QuantizationType.NVFP4, deq); }

        int[] counts = new int[numExperts];
        for (int e = 0; e < numExperts; e++) counts[e] = e % 5 == 3 ? 0 : rng.Next(1, 40);
        uint[] offsets = new uint[numExperts + 1];
        for (int e = 0; e < numExperts; e++) offsets[e + 1] = offsets[e] + (uint)counts[e];
        int rows = (int)offsets[numExperts];
        float[] scale = new float[numExperts];
        for (int e = 0; e < numExperts; e++) scale[e] = 0.3f + 5.7f * e / Math.Max(1, numExperts - 1);
        float[] x = new float[rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[rows * m];
        var noScale = new float[rows * m];
        var expert0 = new float[rows * m];
        for (int e = 0; e < numExperts; e++)
            for (uint r = offsets[e]; r < offsets[e + 1]; r++)
                for (int o = 0; o < m; o++)
                {
                    double acc = 0; long wb = ((long)e * m + o) * k;
                    for (int j = 0; j < k; j++) acc += deq[wb + j] * x[r * k + j];
                    expected[r * m + o] = (float)(acc * scale[e]);
                    noScale[r * m + o] = (float)acc;
                    expert0[r * m + o] = (float)(acc * scale[0]);
                }

        using var kernel = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, MoeGroupedLegacyQuant.Nvfp4);
        using var bankBuf = device.Allocate(bank.Length);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var offBuf = device.Allocate((long)offsets.Length * sizeof(uint));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));
        using var sBuf = device.Allocate((long)scale.Length * sizeof(float));
        device.Upload(bank, bankBuf); device.Upload(x, xBuf); device.Upload(MemoryMarshal.AsBytes<uint>(offsets), offBuf); device.Upload(scale, sBuf);
        kernel.Launch(bankBuf, xBuf, offBuf, yBuf, sBuf, applyScale: true, m, k, rows, numExperts, maxRowsPerExpert: counts.Max());
        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        double num = 0, den = 0, numNo = 0, numE0 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i]; num += d * d; den += (double)expected[i] * expected[i];
            numNo += (noScale[i] - expected[i]) * (double)(noScale[i] - expected[i]);
            numE0 += (expert0[i] - expected[i]) * (double)(expert0[i] - expected[i]);
        }
        Assert.True(Math.Sqrt(num / den) < 0.01, $"grouped kernel rel err {Math.Sqrt(num / den)} (F16 operands: ~1e-3 expected)");
        Assert.True(Math.Sqrt(numNo / den) > 0.2 && Math.Sqrt(numE0 / den) > 0.2, "fixture cannot tell a missing / expert-0 scale from correct");
    }
}
