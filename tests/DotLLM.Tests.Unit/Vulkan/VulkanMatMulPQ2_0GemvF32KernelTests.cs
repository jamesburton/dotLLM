using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan PQ2_0 (PrismML Bonsai ternary) GEMV kernel against a
/// scalar ground-truth reference. The reference is computed directly from the unpacked ternary
/// values with each 128-element group scaled by its own fp16 scale
/// (<c>y[r] = Σ_g scale(r,g) · Σ_{c∈g} ternary[r,c] · x[c]</c>), so a passing test proves the GPU
/// kernel decodes the PQ2_0 bit-layout (byte <c>gp</c> packs group-relative positions
/// {gp,gp+32,gp+64,gp+96} at bits {6,4,2,0}, value = code−1), reads each group's leading fp16
/// scale from its 34-byte group header, applies it per-group (not once at the end, unlike I2_S),
/// and reduces correctly. Mirrors <c>VulkanMatMulI2SGemvF32KernelTests</c>.
/// </summary>
/// <remarks>
/// Tolerance — PQ2_0 codes are exact ternary (no per-element quant error beyond the fp16 group
/// scale's own rounding), so divergence from the sequential CPU reference is fp16-scale rounding
/// plus reduction order (GPU 128-thread tree reduce vs scalar sum). The same 5e-3 / 1e-3
/// tolerances as the I2_S / K-quant GEMV parity tests cover it.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulPQ2_0GemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    [SkippableTheory]
    [InlineData(1, 128)]      // one group -> 34 bytes packed
    [InlineData(8, 128)]
    [InlineData(4, 256)]      // two groups, distinct per-group scales
    [InlineData(16, 768)]
    [InlineData(2048, 256)]
    [InlineData(2560, 2560)]
    [InlineData(576, 1024)]
    public void Launch_MatchesScalarReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_50 ^ (m * 7 + k * 11));

        int groups = k / GroupSize;

        // Random ternary weights {-1,0,+1}, a random fp16 scale per (row, group), and a
        // random activation vector.
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsPQ2_0 = PackPQ2_0(ternary, scales, m, k);

        // Ground-truth reference: y[r] = Σ_g scale(r,g) · Σ_{c∈g} ternary[r,c] · x[c].
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
        using var kernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsPQ2_0.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsPQ2_0), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertClose(expected, actual, m, k);
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix + per-(row,group) fp16 scales into the
    /// PQ2_0 byte layout the kernel decodes: each 128-element group is 34 bytes — a leading
    /// little-endian fp16 scale, then 32 bytes where byte <c>gp</c> holds group-relative
    /// positions {gp, gp+32, gp+64, gp+96} at bit offsets {6, 4, 2, 0} (stored code = value + 1
    /// ∈ {0,1,2}). Row stride = (k/128)·34 bytes.
    /// </summary>
    private static byte[] PackPQ2_0(sbyte[] ternary, Half[] scales, int m, int k)
    {
        int groups = k / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * rowBytes];
        for (int r = 0; r < m; r++)
        {
            int rowBase = r * k;
            int rowByteBase = r * rowBytes;
            for (int g = 0; g < groups; g++)
            {
                int groupByteBase = rowByteBase + g * GroupBytes;
                BitConverter.GetBytes(scales[r * groups + g]).CopyTo(buf, groupByteBase);
                int codeBase = groupByteBase + 2;
                int groupElemBase = g * GroupSize;
                for (int p = 0; p < GroupSize; p++)
                {
                    int code = ternary[rowBase + groupElemBase + p] + 1;   // {-1,0,1} -> {0,1,2}
                    int byteInGroup = p % 32;
                    int shift = 6 - 2 * (p / 32);                          // sub-group 0->6,1->4,2->2,3->0
                    buf[codeBase + byteInGroup] |= (byte)(code << shift);
                }
            }
        }
        return buf;
    }

    private static void AssertClose(float[] expected, float[] actual, int m, int k)
    {
        Assert.Equal(m, actual.Length);
        for (int r = 0; r < m; r++)
        {
            float diff = MathF.Abs(expected[r] - actual[r]);
            float tol = AbsTol + RelTol * MathF.Abs(expected[r]);
            Assert.True(diff <= tol,
                $"row {r} (m={m}, k={k}): expected {expected[r]:G9}, got {actual[r]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
        }
    }
}
