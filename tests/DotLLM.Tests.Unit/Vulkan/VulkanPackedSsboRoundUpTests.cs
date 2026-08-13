using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #361 — packed-weight SSBO allocations must be rounded up to a 4-byte
/// multiple. The packed-matmul shaders read their weight buffer through a
/// <c>uint</c>-addressed funnel; ten formats have a block size ≡ 2 (mod 4)
/// (Q8_0 34 B, Q5_0 22 B, IQ4_NL 18 B, Q3_K 110 B, Q6_K 210 B, …), so an odd
/// total block count leaves the buffer end 2 bytes inside the funnel's final
/// <c>uint</c>. Production upload used to allocate exact-sized; with
/// <c>robustBufferAccess</c> off and <c>VK_WHOLE_SIZE</c> bindings that final
/// read was out-of-bounds by specification (benign on this driver only by
/// allocation-granularity luck). The fix is allocation-side —
/// <c>VulkanWeights.AllocateAndUploadPacked</c> — because the 2 trailing bytes
/// are real weight data the shader must keep reading; clamping the read would
/// corrupt the last row instead.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanPackedSsboRoundUpTests
{
    /// <summary>
    /// The issue's synthetic probe sizes (66 / 330 / 99330 bytes — all ≡ 2 mod 4)
    /// plus already-aligned controls. The helper must (a) allocate a 4-byte-multiple
    /// buffer, (b) upload the packed bytes verbatim, (c) zero-fill the pad so the
    /// buffer has no uninitialized tail.
    /// </summary>
    [SkippableTheory]
    [InlineData(66)]     // odd Q8_0/IQ4_NL-family size, pad 2
    [InlineData(330)]    // 3 × 110-byte Q3_K blocks, pad 2
    [InlineData(99330)]  // 903 × 110-byte Q3_K blocks, pad 2
    [InlineData(21)]     // pad 3 (not reachable via 2-mod-4 blocks; exercises the general path)
    [InlineData(64)]     // already aligned — must NOT grow
    public unsafe void AllocateAndUploadPacked_RoundsUpUploadsAndZeroPads(int bytes)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        using var staging = VulkanStagingBuffer.Create(device, neededBytes: 1 << 16);

        var rng = new Random(0x361 + bytes);
        byte[] src = new byte[bytes];
        rng.NextBytes(src);
        // Make sure the last byte is nonzero so "verbatim upload" is discriminating
        // right at the boundary the round-up exists for.
        src[^1] = 0xA5;

        long expectedSize = (bytes + 3) & ~3L;

        fixed (byte* sp = src)
        {
            using var buf = VulkanWeights.AllocateAndUploadPacked(
                device, staging, (nint)sp, bytes);

            Assert.Equal(expectedSize, buf.Size);
            Assert.Equal(0L, buf.Size % 4);

            // Download exposes a float-span overload only; the padded size is a
            // 4-byte multiple by construction, so reinterpret.
            float[] readBackF = new float[expectedSize / 4];
            device.Download(buf, readBackF.AsSpan());
            ReadOnlySpan<byte> readBack = MemoryMarshal.AsBytes<float>(readBackF);

            Assert.True(readBack.Slice(0, bytes).SequenceEqual(src),
                $"uploaded bytes differ from source (size {bytes})");
            for (int i = bytes; i < (int)expectedSize; i++)
                Assert.True(readBack[i] == 0, $"pad byte {i} not zeroed: 0x{readBack[i]:X2}");
        }
    }

    /// <summary>
    /// Q5_0 GEMV at an odd total block count (the shape class no unit test covered —
    /// Q5_0 blocks are 22 bytes, so 7 rows × 3 blocks = 462 bytes ≡ 2 mod 4), run
    /// against a buffer allocated exactly the way production now allocates it, and
    /// checked element-wise against the CPU decode of the same bytes. Discriminates
    /// against a wrong fix that clamps or truncates the funnel's final read: the
    /// last row's tail nibbles live in those 2 trailing bytes.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 96)]   // 21 blocks, 462 bytes ≡ 2 (mod 4)
    [InlineData(3, 32)]   // 3 blocks, 66 bytes ≡ 2 (mod 4) — the issue's smallest probe
    public unsafe void Q5_0Gemv_OddBlockCount_ProductionAllocationMatchesCpuOracle(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int blockBytes = MatMulQ5_0GemvF32Kernel.Q5_0BlockBytes;
        const int groupSize = MatMulQ5_0GemvF32Kernel.Q5_0GroupSize;
        int blocks = m * (k / groupSize);
        int totalBytes = blocks * blockBytes;
        Assert.Equal(2, totalBytes % 4); // the test exists for exactly this class

        // Random-but-valid Q5_0 blocks: random payload, controlled fp16 d per block
        // (same recipe as VulkanWeightsQuantRepackTests).
        var rng = new Random(0x361_50 + m * 7 + k);
        byte[] src = new byte[totalBytes];
        rng.NextBytes(src);
        for (int b = 0; b < blocks; b++)
            MemoryMarshal.Write(src.AsSpan(b * blockBytes), (Half)((rng.NextSingle() - 0.5f) * 0.25f));

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // CPU oracle: decode the same bytes with the llama.cpp-anchored CPU dequant,
        // then a plain F32 dot per row.
        float[] wDequant = new float[m * k];
        float[] expected = new float[m];
        fixed (byte* sp = src)
            Dequantize.ToFloat32((nint)sp, (long)m * k, QuantizationType.Q5_0, wDequant);
        for (int row = 0; row < m; row++)
        {
            double acc = 0;
            for (int i = 0; i < k; i++) acc += (double)wDequant[row * k + i] * x[i];
            expected[row] = (float)acc;
        }

        using var device = VulkanDevice.Create();
        using var staging = VulkanStagingBuffer.Create(device, neededBytes: 1 << 16);
        using var kernel = MatMulQ5_0GemvF32Kernel.Create(device, spvDir);

        fixed (byte* sp = src)
        {
            // The production allocation policy under test — NOT a test-side round-up.
            using var bufW = VulkanWeights.AllocateAndUploadPacked(device, staging, (nint)sp, totalBytes);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufY = device.Allocate((long)m * sizeof(float));
            device.Upload(x, bufX);

            kernel.Launch(bufW, bufX, bufY, m, k);

            float[] actual = new float[m];
            device.Download(bufY, actual);

            for (int row = 0; row < m; row++)
            {
                float diff = Math.Abs(expected[row] - actual[row]);
                float tol = 1e-4f + 1e-3f * Math.Abs(expected[row]);
                Assert.True(diff <= tol,
                    $"row {row}: expected {expected[row]:G9}, got {actual[row]:G9} (diff {diff:G3}) m={m} k={k}");
            }
        }
    }
}
