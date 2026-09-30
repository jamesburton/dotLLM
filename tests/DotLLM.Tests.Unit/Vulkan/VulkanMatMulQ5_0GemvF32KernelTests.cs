using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #574: the Q5_0 decode GEMV (coalesced default and block-per-lane legacy) against a
/// double-precision reference built from the CPU dequantiser over the exact bytes the GPU sees.
/// Weights are RANDOM Q5_0 blocks (random scale, random qh, random nibbles), so every qh bit and both
/// nibble halves are exercised. Both variants are F32-in, so the bar is float rounding, not the F16
/// operand floor of the coopmat GEMM.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ5_0GemvF32KernelTests
{
    private const int BlockBytes = 22;
    private const int Group = 32;
    private const double AbsTol = 2e-3;
    private const double RelTol = 1e-3;

    [SkippableTheory]
    [InlineData(1, 32, false)]
    [InlineData(7, 64, false)]
    [InlineData(33, 96, false)]         // odd block count: a partial window of blocks
    [InlineData(300, 4480, false)]      // Nemotron-H hidden width
    [InlineData(64, 15680, false)]      // ffn_down contraction width
    [InlineData(1, 32, true)]
    [InlineData(300, 4480, true)]
    [InlineData(64, 15680, true)]
    public void Launch_MatchesCpuReference(int m, int k, bool legacy)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        string? saved = Environment.GetEnvironmentVariable(MatMulQ5_0GemvF32Kernel.LegacyEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(MatMulQ5_0GemvF32Kernel.LegacyEnvVar, legacy ? "1" : null);
            RunAndAssert(device, spvDir, m, k);
        }
        finally
        {
            Environment.SetEnvironmentVariable(MatMulQ5_0GemvF32Kernel.LegacyEnvVar, saved);
        }
    }

    private static void RunAndAssert(VulkanDevice device, string spvDir, int m, int k)
    {
        var rng = new Random(0x5A11 + m * 17 + k);
        int blocksPerRow = k / Group;
        byte[] w = new byte[(long)m * blocksPerRow * BlockBytes];
        for (long b = 0; b < (long)m * blocksPerRow; b++)
        {
            int o = (int)(b * BlockBytes);
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)(0.001f + rng.NextSingle() * 0.01f));
            rng.NextBytes(w.AsSpan(o + 2, BlockBytes - 2));
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        float[] wf = new float[(long)m * k];
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * blocksPerRow * BlockBytes, k,
                    QuantizationType.Q5_0, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }

        double[] expected = new double[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[c];
            expected[r] = acc;
        }

        using var kernel = MatMulQ5_0GemvF32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k);
        float[] actual = new float[m];
        device.Download(bufY, actual);

        for (int r = 0; r < m; r++)
            Assert.True(Math.Abs(actual[r] - expected[r]) <= AbsTol + RelTol * Math.Abs(expected[r]),
                $"row={r}: gpu={actual[r]:F6} vs ref={expected[r]:F6}");
    }
}
