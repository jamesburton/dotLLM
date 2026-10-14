using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #576: the Q6_K decode GEMV (coalesced default and block-per-lane legacy) against a
/// double-precision reference built from the CPU dequantiser over the exact bytes the GPU sees.
/// Weights are RANDOM Q6_K blocks (random ql/qh and random signed int8 scales), so both
/// nibble halves and qh fields are exercised. Both variants are F32-in, so the bar is float rounding, not the F16
/// operand floor of the coopmat GEMM.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ6KGemvF32KernelTests
{
    private const int BlockBytes = 210;
    private const int Group = 256;
    private const double AbsTol = 2e-3;
    private const double RelTol = 1e-3;

    [SkippableTheory]
    [InlineData(1, 256, false)]
    [InlineData(7, 512, false)]
    [InlineData(33, 768, false)]        // odd super-block count: a partial window
    [InlineData(300, 10240, false)]     // Nemotron-H ssm_out contraction width
    [InlineData(64, 5120, false)]       // attn_output contraction width
    [InlineData(1, 256, true)]
    [InlineData(300, 10240, true)]
    [InlineData(64, 5120, true)]
    public void Launch_MatchesCpuReference(int m, int k, bool legacy)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        string? saved = Environment.GetEnvironmentVariable(MatMulQ6KGemvF32Kernel.LegacyEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(MatMulQ6KGemvF32Kernel.LegacyEnvVar, legacy ? "1" : null);
            RunAndAssert(device, spvDir, m, k);
        }
        finally
        {
            Environment.SetEnvironmentVariable(MatMulQ6KGemvF32Kernel.LegacyEnvVar, saved);
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
            rng.NextBytes(w.AsSpan(o, BlockBytes - 2));   // ql[128] + qh[64] + scales[16] (int8), fully random
            BitConverter.TryWriteBytes(w.AsSpan(o + BlockBytes - 2, 2), (Half)(0.0002f + rng.NextSingle() * 0.0006f));   // d (fp16, last 2 bytes)
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        float[] wf = new float[(long)m * k];
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * blocksPerRow * BlockBytes, k,
                    QuantizationType.Q6_K, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }

        double[] expected = new double[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[c];
            expected[r] = acc;
        }

        using var kernel = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
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
