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
