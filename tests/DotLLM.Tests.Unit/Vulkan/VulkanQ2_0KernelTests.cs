using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Upstream ggml Q2_0 on Vulkan (#2074): F32-in GEMV, tiled GEMM, 128x128x4 coopmat GEMM and the embedding gather, each against a double
/// reference built from the CPU dequantiser over the exact bytes the GPU sees. Blocks are RANDOM (random scale, all four codes): the real
/// Bonsai 2.5 tensors never carry code 3 (+2), so only random bytes can tell the ggml map (-1,0,+1,+2) from a ternary one, and a swapped
/// 2-bit position or a half-block mix-up is an O(1) error.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ2_0KernelTests
{
    private const int BlockBytes = 18;
    private const int Group = 64;

    private static byte[] RandomBlocks(int rows, int k, int seed)
    {
        var rng = new Random(seed);
        int bpr = k / Group;
        byte[] w = new byte[(long)rows * bpr * BlockBytes];
        for (long b = 0; b < (long)rows * bpr; b++)
        {
            int o = (int)(b * BlockBytes);
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)(0.001f + rng.NextSingle() * 0.01f));
            rng.NextBytes(w.AsSpan(o + 2, BlockBytes - 2));
        }
        return w;
    }

    private static float[] Dequant(byte[] w, int rows, int k)
    {
        float[] wf = new float[(long)rows * k];
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < rows; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * (k / Group) * BlockBytes, k, QuantizationType.Q2_0, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }
        return wf;
    }

    private static double[] Reference(float[] wf, float[] b, int n, int m, int k)
    {
        var e = new double[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * b[t * k + c];
                e[t * m + r] = acc;
            }
        return e;
    }

    private static float[] RandomActs(int n, int k, int seed)
    {
        var rng = new Random(seed);
        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
        return b;
    }

    private static void Check(float[] actual, double[] expected, double absTol, double relTol, int m)
    {
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= absTol + relTol * Math.Abs(expected[i]),
                $"t={i / m} row={i % m}: gpu={actual[i]:F6} vs ref={expected[i]:F6}");
    }

    [SkippableTheory]
    [InlineData(1, 64)]
    [InlineData(5, 192)]
    [InlineData(37, 640)]
    [InlineData(70, 5120)]
    public void Gemv_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var w = RandomBlocks(m, k, 7 + m + k);
        var x = RandomActs(1, k, 9);
        var expected = Reference(Dequant(w, m, k), x, 1, m, k);
        using var kernel = MatMulQ4_0GemvF32Kernel.Create(device, spvDir, QuantizationType.Q2_0);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k);
        float[] actual = new float[m];
        device.Download(bufY, actual);
        Check(actual, expected, 1e-3, 1e-4, m);   // F32 arithmetic: only summation order differs
    }

    [SkippableTheory]
    [InlineData(1, 1, 64)]
    [InlineData(3, 17, 128)]
    [InlineData(20, 33, 640)]
    public void TiledGemm_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var w = RandomBlocks(m, k, 11 + m + k);
        var b = RandomActs(n, k, 12);
        var expected = Reference(Dequant(w, m, k), b, n, m, k);
        using var kernel = MatMulQ4_0GemmF32Kernel.Create(device, spvDir, QuantizationType.Q2_0);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        float[] actual = new float[n * m];
        device.Download(bufC, actual);
        Check(actual, expected, 1e-3, 1e-4, m);
    }

    [SkippableTheory]
    [InlineData(2, 4, 64)]
    [InlineData(17, 33, 192)]
    [InlineData(129, 129, 384)]
    [InlineData(128, 128, 640)]
    [InlineData(64, 1000, 5120)]
    public void CoopmatGemm_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir, QuantizationType.Q2_0), "Q2_0 coopmat kernel unsupported here.");
        var w = RandomBlocks(m, k, 13 + m + k);
        var b = RandomActs(n, k, 14);
        var expected = Reference(Dequant(w, m, k), b, n, m, k);
        using var kernel = MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir, QuantizationType.Q2_0);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        float[] actual = new float[n * m];
        device.Download(bufC, actual);
        Check(actual, expected, 5e-3, 5e-3, m);   // F16 A operand
    }

    [SkippableFact]
    public void EmbedGather_IsBitIdenticalToTheCpuDequantiser()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        const int vocab = 23, hidden = 192;
        Assert.True(EmbedGatherF32Kernel.IsSupported(QuantizationType.Q2_0, hidden, spvDir));
        Assert.False(EmbedGatherF32Kernel.IsSupported(QuantizationType.Q2_0, 96, spvDir));   // not whole 64-element blocks
        var w = RandomBlocks(vocab, hidden, 21);
        var wf = Dequant(w, vocab, hidden);
        int[] ids = [0, 22, 7, 7, 13];
        using var kernel = EmbedGatherF32Kernel.Create(device, spvDir, QuantizationType.Q2_0);
        using var bufT = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufI = device.Allocate(ids.Length * sizeof(int));
        using var bufD = device.Allocate((long)ids.Length * hidden * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufT);
        device.Upload(MemoryMarshal.AsBytes<int>(ids), bufI);
        kernel.Launch(bufT, bufI, bufD, ids.Length, hidden, vocab);
        float[] got = new float[ids.Length * hidden];
        device.Download(bufD, got);
        for (int t = 0; t < ids.Length; t++)
            for (int c = 0; c < hidden; c++)
                Assert.Equal(BitConverter.SingleToInt32Bits(wf[ids[t] * hidden + c]), BitConverter.SingleToInt32Bits(got[t * hidden + c]));
    }
}
