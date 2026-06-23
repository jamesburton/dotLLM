using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan I2_S (BitNet b1.58 ternary) prefill GEMM kernel
/// against a scalar ground-truth reference (<c>C[t,m] = scale · Σ_c ternary[m,c] · B[t,c]</c>).
/// Same per-element decode as the GEMV kernel; the difference is the 16x16 output-cell tile
/// + N-batched B input and the <c>c[t·M + m]</c> (token-major) output layout. A passing test
/// proves the GEMM decodes the I2_S bit-layout, reads the per-tensor scale from the buffer tail,
/// and tiles correctly across partial tiles.
/// </summary>
/// <remarks>
/// Tolerance — I2_S codes are exact ternary (no per-element quant error), so divergence from the
/// sequential CPU reference is only reduction order. Same 5e-3 / 1e-3 tolerances as the GEMV test.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulI2SGemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(16, 128, 4)]      // tile-aligned, 1 block per row, batch=4
    [InlineData(32, 256, 8)]      // 2 blocks per row, batch=8
    [InlineData(64, 128, 16)]     // 1 block, batch=16 (tile boundary)
    [InlineData(48, 768, 12)]     // 6 blocks, partial-tile m=48 (3 tiles)
    [InlineData(15, 128, 3)]      // m < TILE_M, n < TILE_N — partial-tile fallthrough
    [InlineData(2560, 2560, 5)]   // BitNet hidden × hidden, small prefill batch
    public void Launch_MatchesScalarReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x12_5C_6E ^ (m * 7 + k * 11 + n * 13));

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = rng.NextSingle() * 0.05f + 0.01f;
        float[] inputB = new float[(long)n * k];
        for (int i = 0; i < inputB.Length; i++) inputB[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsI2S = PackI2S(ternary, m, k, scale);

        // Ground-truth reference: C[t·M + m] = scale · Σ_c ternary[m,c] · B[t,c].
        float[] expected = new float[(long)n * m];
        for (int t = 0; t < n; t++)
        {
            int bRowBase = t * k;
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                int wRowBase = r * k;
                for (int c = 0; c < k; c++) acc += ternary[wRowBase + c] * (double)inputB[bRowBase + c];
                expected[t * m + r] = (float)(acc * scale);
            }
        }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulI2SGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsI2S.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsI2S), bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[(long)n * m];
        device.Download(bufC, actual);

        for (int t = 0; t < n; t++)
        {
            for (int r = 0; r < m; r++)
            {
                int idx = t * m + r;
                float diff = MathF.Abs(expected[idx] - actual[idx]);
                float tol = AbsTol + RelTol * MathF.Abs(expected[idx]);
                Assert.True(diff <= tol,
                    $"cell t={t} m={r} (m={m}, k={k}, n={n}): expected {expected[idx]:G9}, got {actual[idx]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
            }
        }
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix into the I2_S byte layout the kernel decodes:
    /// per 128-element block (32 bytes), byte <c>gp</c> holds block-relative positions
    /// {gp, gp+32, gp+64, gp+96} at bit offsets {6, 4, 2, 0}; stored code = value + 1 ∈ {0,1,2}.
    /// Row stride = k/4 bytes; a single float32 scale is appended at offset m·(k/4).
    /// Mirrors <see cref="VulkanMatMulI2SGemvF32KernelTests"/>'s packer.
    /// </summary>
    private static byte[] PackI2S(sbyte[] ternary, int m, int k, float scale)
    {
        int rowBytes = k / 4;
        byte[] buf = new byte[(long)m * rowBytes + sizeof(float)];
        int blocks = k / 128;
        for (int r = 0; r < m; r++)
        {
            int rowBase = r * k;
            int rowByteBase = r * rowBytes;
            for (int b = 0; b < blocks; b++)
            {
                int blockElemBase = b * 128;
                int blockByteBase = rowByteBase + b * 32;
                for (int p = 0; p < 128; p++)
                {
                    int code = ternary[rowBase + blockElemBase + p] + 1;   // {-1,0,1} → {0,1,2}
                    int byteInBlock = p % 32;
                    int shift = 6 - 2 * (p / 32);                          // group 0→6, 1→4, 2→2, 3→0
                    buf[blockByteBase + byteInBlock] |= (byte)(code << shift);
                }
            }
        }
        BitConverter.GetBytes(scale).CopyTo(buf, m * rowBytes);   // per-tensor scale at the tail
        return buf;
    }
}
