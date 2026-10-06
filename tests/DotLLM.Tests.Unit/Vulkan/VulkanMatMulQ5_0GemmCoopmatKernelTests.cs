using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #568: the Q5_0 blocked 128x128 coopmat GEMM against a double-precision reference built from the
/// CPU dequantiser over the exact bytes the GPU sees. Weights are RANDOM Q5_0 blocks (random scale,
/// random qh, random nibbles), not quantised floats, so every qh bit and both nibble halves are
/// exercised: a swapped nibble half or a shifted qh bit produces O(1) errors, far above the F16
/// operand tolerance.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ5_0GemmCoopmatKernelTests
{
    private const int BlockBytes = 22;
    private const int Group = 32;

    // Same F16-operand precision floor as the Q8_0 coopmat kernel (see its tests for the derivation).
    private const double AbsTol = 5e-3;
    private const double RelTol = 5e-3;

    [SkippableTheory]
    [InlineData(1, 1, 32)]          // single-cell output
    [InlineData(2, 4, 32)]          // one block per row
    [InlineData(17, 33, 64)]        // ragged in both dims
    [InlineData(3, 15, 128)]        // below a single fragment
    [InlineData(129, 129, 256)]     // one past a full blocked tile in both dims
    [InlineData(128, 128, 256)]     // exactly one blocked tile
    [InlineData(64, 1000, 4480)]    // Nemotron-H hidden width, ragged M
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ5_0GemmCoopmatKernel.IsSupportedOn(device, spvDir),
            "Q5_0 coopmat kernel unsupported here (needs cooperative matrix, wave64, compiled SPIR-V).");

        var rng = new Random(0x5EED + n * 31 + m * 17 + k);
        int blocksPerRow = k / Group;
        byte[] w = new byte[(long)m * blocksPerRow * BlockBytes];
        for (long b = 0; b < (long)m * blocksPerRow; b++)
        {
            int o = (int)(b * BlockBytes);
            // Realistic scale (real Q5_0 weights are ~0.01-0.15) keeps the F16-operand error at K=4480 inside
            // the tolerance; the bytes after it are fully random.
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)(0.001f + rng.NextSingle() * 0.01f));
            rng.NextBytes(w.AsSpan(o + 2, BlockBytes - 2));
        }

        float[] b32 = new float[n * k];
        for (int i = 0; i < b32.Length; i++) b32[i] = rng.NextSingle() * 2f - 1f;

        // Reference: dequantise the exact bytes on the CPU, multiply in double.
        float[] wf = new float[(long)m * k];
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * blocksPerRow * BlockBytes, k,
                    QuantizationType.Q5_0, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }

        double[] expected = new double[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * b32[t * k + c];
                expected[t * m + r] = acc;
            }

        using var kernel = MatMulQ5_0GemmCoopmatKernel.Create(device, spvDir);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b32, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= AbsTol + RelTol * Math.Abs(expected[i]),
                $"t={i / m} row={i % m}: gpu={actual[i]:F6} vs ref={expected[i]:F6}");
    }

    [SkippableFact]
    public void LegacyEnvVar_DisablesTheKernel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ5_0GemmCoopmatKernel.IsSupportedOn(device, spvDir), "kernel unsupported here");

        string? saved = Environment.GetEnvironmentVariable(MatMulQ5_0GemmCoopmatKernel.LegacyEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(MatMulQ5_0GemmCoopmatKernel.LegacyEnvVar, "1");
            Assert.False(MatMulQ5_0GemmCoopmatKernel.IsSupportedOn(device, spvDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(MatMulQ5_0GemmCoopmatKernel.LegacyEnvVar, saved);
        }
    }
}
