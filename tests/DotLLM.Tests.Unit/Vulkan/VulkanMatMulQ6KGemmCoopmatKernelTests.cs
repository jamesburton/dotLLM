using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #578: the Q6_K blocked 128x128 coopmat GEMM against a double-precision reference built from the
/// CPU dequantiser over the exact bytes the GPU sees. Weights are RANDOM Q6_K blocks (random scale,
/// random qh, random nibbles), not quantised floats, so every sub-block scale/min path (j<4 and j>=4) and both
/// nibble halves are exercised: a swapped half or a mis-unpacked scale produces O(1) errors, far above the F16
/// operand tolerance.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ6KGemmCoopmatKernelTests
{
    private const int BlockBytes = 210;
    private const int Group = 256;

    // Same F16-operand precision floor as the Q8_0 coopmat kernel (see its tests for the derivation).
    private const double AbsTol = 5e-3;
    private const double RelTol = 5e-3;

    [SkippableTheory]
    [InlineData(1, 1, 256)]          // single-cell output
    [InlineData(2, 4, 256)]          // one block per row
    [InlineData(17, 33, 512)]        // ragged in both dims
    [InlineData(3, 15, 256)]        // below a single fragment
    [InlineData(129, 129, 512)]     // one past a full blocked tile in both dims
    [InlineData(128, 128, 512)]     // exactly one blocked tile
    [InlineData(64, 300, 10240)]    // Nemotron-H ssm_out contraction width
    [InlineData(128, 2560, 4608)]   // 20 tiles -> split-K x4 (Tev1 ffn-down shape class)
    [InlineData(128, 4096, 2560)]   // 32 tiles -> split-K x2
    [InlineData(100, 1000, 2560)]   // 8 tiles, ragged n and m, k=80 chunks -> split-K x4
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ6KGemmCoopmatKernel.IsSupportedOn(device, spvDir),
            "Q6_K coopmat kernel unsupported here (needs cooperative matrix, wave64, compiled SPIR-V).");

        var rng = new Random(0x5EED + n * 31 + m * 17 + k);
        int blocksPerRow = k / Group;
        byte[] w = new byte[(long)m * blocksPerRow * BlockBytes];
        for (long b = 0; b < (long)m * blocksPerRow; b++)
        {
            int o = (int)(b * BlockBytes);
            // Realistic scale (real Q6_K weights are ~0.01-0.15) keeps the F16-operand error at K=4480 inside
            // the tolerance; the bytes after it are fully random.
            rng.NextBytes(w.AsSpan(o, BlockBytes - 2));   // ql[128] + qh[64] + scales[16] (int8), fully random
            BitConverter.TryWriteBytes(w.AsSpan(o + BlockBytes - 2, 2), (Half)(0.00003f + rng.NextSingle() * 0.00007f));   // d (fp16, last 2 bytes)
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
                    QuantizationType.Q6_K, wf.AsSpan(r * k, k));
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

        using var kernel = MatMulQ6KGemmCoopmatKernel.Create(device, spvDir);
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
        Skip.IfNot(MatMulQ6KGemmCoopmatKernel.IsSupportedOn(device, spvDir), "kernel unsupported here");

        string? saved = Environment.GetEnvironmentVariable(MatMulQ6KGemmCoopmatKernel.LegacyEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(MatMulQ6KGemmCoopmatKernel.LegacyEnvVar, "1");
            Assert.False(MatMulQ6KGemmCoopmatKernel.IsSupportedOn(device, spvDir));
        }
        finally
        {
            Environment.SetEnvironmentVariable(MatMulQ6KGemmCoopmatKernel.LegacyEnvVar, saved);
        }
    }
}
