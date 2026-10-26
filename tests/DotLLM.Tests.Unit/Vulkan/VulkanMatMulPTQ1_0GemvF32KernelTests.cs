using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan PTQ1_0 GEMV kernel (issue #436).
///
/// <para>The packing helper here is deliberately <b>not</b> a second transcription of the format.
/// It packs a ternary matrix and then asserts, via <see cref="Dequantize.DequantizePTQ1_0"/>,
/// that the bytes decode back to exactly what went in — and that CPU dequant is the routine gated
/// bit-exact against the fork's own <c>dequantize_row_ptq1_0</c> in
/// <c>PTQ1_0ForkOracleTests</c>. So the chain is: fork reference → CPU dequant → these packed
/// bytes → GPU kernel. A shader that decoded some self-consistent but wrong layout would have to
/// coincide with the fork's to pass.</para>
/// </summary>
/// <remarks>
/// Tolerance — PTQ1_0 values are exact ternary, so divergence from the sequential CPU reference
/// is fp16 group-scale rounding plus reduction order (GPU 128-thread tree reduce vs scalar sum).
/// Same 5e-3 / 1e-3 bounds as the PQ2_0 and I2_S GEMV parity tests.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulPTQ1_0GemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;
    private const int GroupSize = QuantFormat.TernaryGroupSize;   // 128
    private const int GroupBytes = QuantFormat.PTQ1_0BlockBytes;  // 28
    private const int QsBytes = 24;
    private const int QhBytes = 2;

    [SkippableTheory]
    [InlineData(1, 128)]      // one group -> 28 bytes packed
    [InlineData(8, 128)]
    [InlineData(4, 256)]      // two groups, distinct per-group scales
    [InlineData(16, 768)]
    [InlineData(2048, 256)]
    [InlineData(576, 1024)]
    [InlineData(2560, 2560)]
    public unsafe void Launch_MatchesScalarReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_51 ^ (m * 7 + k * 11));

        int groups = k / GroupSize;

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weights = PackPTQ1_0(ternary, scales, m, k);

        // The packing is validated against the bit-exact-gated CPU dequant before it is trusted
        // as the input to a GPU parity claim. Only the first row is checked — the packer is
        // row-uniform, and decoding all of a 2560x2560 case would dominate the test's runtime.
        var decoded = new float[k];
        fixed (byte* wp = weights)
            Dequantize.ToFloat32((nint)wp, k, QuantizationType.PTQ1_0, decoded);
        for (int c = 0; c < k; c++)
            Assert.Equal(ternary[c] * (float)scales[c / GroupSize], decoded[c], 1e-6f);

        // Ground-truth reference: y[r] = Σ_g scale(r,g) · Σ_{e∈g} ternary[r,e] · x[e].
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            int rowBase = r * k;
            for (int g = 0; g < groups; g++)
            {
                double groupScale = (float)scales[r * groups + g];
                double groupAcc = 0;
                int groupBase = g * GroupSize;
                for (int c = 0; c < GroupSize; c++)
                    groupAcc += ternary[rowBase + groupBase + c] * (double)x[groupBase + c];
                acc += groupScale * groupAcc;
            }
            expected[r] = (float)acc;
        }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulPTQ1_0GemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weights.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Assert.Equal(m, actual.Length);
        for (int r = 0; r < m; r++)
        {
            float diff = MathF.Abs(expected[r] - actual[r]);
            float tol = AbsTol + RelTol * MathF.Abs(expected[r]);
            Assert.True(diff <= tol,
                $"row {r} (m={m}, k={k}): expected {expected[r]:G9}, got {actual[r]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
        }
    }

    /// <summary>
    /// The prefill GEMM against the same reference, at N &gt; 1. Shares the decode with the GEMV,
    /// so what this actually discriminates is the (row, token) indexing and the C layout
    /// (<c>c[t·M + m]</c>, token-major) — a transposed output would still produce plausible
    /// magnitudes.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 128, 3)]
    [InlineData(16, 512, 7)]
    [InlineData(64, 256, 2)]
    public unsafe void GemmLaunch_MatchesScalarReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_52 ^ (m * 7 + k * 11 + n * 13));
        int groups = k / GroupSize;

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        float[] b = new float[(long)n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        byte[] weights = PackPTQ1_0(ternary, scales, m, k);

        float[] expected = new float[(long)n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int g = 0; g < groups; g++)
                {
                    double groupScale = (float)scales[r * groups + g];
                    double groupAcc = 0;
                    for (int c = 0; c < GroupSize; c++)
                        groupAcc += ternary[r * k + g * GroupSize + c] * (double)b[(long)t * k + g * GroupSize + c];
                    acc += groupScale * groupAcc;
                }
                expected[(long)t * m + r] = (float)acc;
            }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulPTQ1_0GemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weights.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(b, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[(long)n * m];
        device.Download(bufC, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"index {i} (m={m}, k={k}, n={n}): expected {expected[i]:G9}, got {actual[i]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
        }
    }

    /// <summary>
    /// The embedding gather must be <b>bit-identical</b> to a host dequant of the same rows — not
    /// merely close. It replaces a <c>vkCmdCopyBuffer</c> of rows that were dequantized on the
    /// host, so anything less than bit-identical changes the forward pass's input depending on
    /// which path the table happened to take.
    /// </summary>
    [SkippableFact]
    public unsafe void EmbedGather_IsBitIdenticalToHostDequant()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int Vocab = 97;
        const int Hidden = 512;
        int groups = Hidden / GroupSize;

        var rng = new Random(0x2A_53);
        sbyte[] ternary = new sbyte[Vocab * Hidden];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[Vocab * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        byte[] table = PackPTQ1_0(ternary, scales, Vocab, Hidden);

        int[] ids = [0, 5, 96, 42, 42, 1];
        int rowBytes = groups * GroupBytes;

        float[] expected = new float[ids.Length * Hidden];
        for (int t = 0; t < ids.Length; t++)
            fixed (byte* tp = &table[ids[t] * rowBytes])
                Dequantize.ToFloat32((nint)tp, Hidden, QuantizationType.PTQ1_0,
                    expected.AsSpan(t * Hidden, Hidden));

        using var device = VulkanDevice.Create();
        using var kernel = EmbedGatherF32Kernel.Create(device, spvDir, QuantizationType.PTQ1_0);

        using var bufTable = device.Allocate(((long)table.Length + 3) & ~3L);
        using var bufIds = device.Allocate((long)ids.Length * sizeof(int));
        using var bufDst = device.Allocate((long)ids.Length * Hidden * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(table), bufTable);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<int>(ids), bufIds);

        kernel.Launch(bufTable, bufIds, bufDst, ids.Length, Hidden, Vocab);

        float[] actual = new float[ids.Length * Hidden];
        device.Download(bufDst, actual);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.SingleToUInt32Bits(expected[i]), BitConverter.SingleToUInt32Bits(actual[i]));
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix plus per-(row, group) fp16 scales into the
    /// PTQ1_0 byte layout, following the fork's <c>quantize_row_ptq1_0_ref</c>: base-3 with the
    /// first value in the most significant trit, ceiling-encoded as <c>(q·256 + 242)/243</c>, and
    /// the <c>qh</c> tail shifted up one trit because it holds only four. The scale goes at
    /// offset 26 — the END of the block.
    /// </summary>
    private static byte[] PackPTQ1_0(sbyte[] ternary, Half[] scales, int m, int k)
    {
        int groups = k / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * rowBytes];
        int[] stages = [32, 16, 8];

        for (int r = 0; r < m; r++)
        {
            int rowBase = r * k;
            int rowByteBase = r * rowBytes;
            for (int g = 0; g < groups; g++)
            {
                int blockBase = rowByteBase + g * GroupBytes;
                int elemBase = rowBase + g * GroupSize;

                int src = 0;
                int j = 0;
                foreach (int c in stages)
                {
                    while (j + c <= QsBytes)
                    {
                        for (int mm = 0; mm < c; mm++)
                        {
                            int q = 0;
                            for (int nn = 0; nn < 5; nn++)
                                q = q * 3 + (ternary[elemBase + src + mm + nn * c] + 1);

                            buf[blockBase + j + mm] = (byte)((q * 256 + 242) / 243);
                        }

                        src += 5 * c;
                        j += c;
                    }
                }

                for (int h = 0; h < QhBytes; h++)
                {
                    int q = 0;
                    for (int mm = 0; mm < 4; mm++)
                        q = q * 3 + (ternary[elemBase + src + h + mm * QhBytes] + 1);

                    q *= 3;
                    buf[blockBase + QsBytes + h] = (byte)((q * 256 + 242) / 243);
                }

                BitConverter.TryWriteBytes(buf.AsSpan(blockBase + QsBytes + QhBytes, 2), scales[r * groups + g]);
            }
        }

        return buf;
    }
}
