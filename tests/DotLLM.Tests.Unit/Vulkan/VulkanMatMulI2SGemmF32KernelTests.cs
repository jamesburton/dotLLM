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
        => RunParity(I2SGemmVariant.Scalar, m, k, n);

    /// <summary>
    /// Same parity contract for the register-blocked variant (32x32 tile, 2x2 micro-tile per
    /// thread). The extra rows stress the 32-tile edges specifically — the baseline's 16x16
    /// rows would leave a 32-wide tile's partial-tile guards untested, and it is exactly the
    /// four separate bounds checks on the micro-tile corners that a mapping bug would break.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]      // m and n both below TILE — every micro-tile corner guarded
    [InlineData(32, 256, 8)]      // exactly one tile, 2 blocks per row
    [InlineData(64, 128, 16)]     // 2 tiles in m, partial in n
    [InlineData(48, 768, 12)]     // partial tile in both dims, 6 blocks
    [InlineData(33, 128, 33)]     // one element past a full tile in both dims
    [InlineData(17, 256, 47)]     // ragged in both dims, straddling the SUB=16 stride
    [InlineData(15, 128, 3)]      // below the micro-tile stride entirely
    [InlineData(2560, 2560, 5)]   // BitNet hidden × hidden, small prefill batch
    public void RegisterBlocked_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.RegisterBlocked, m, k, n);

    /// <summary>
    /// Same parity contract for the cooperative-matrix variant, at a tolerance widened for its
    /// F16 operands. Skipped on devices without <c>VK_KHR_cooperative_matrix</c>.
    /// </summary>
    /// <remarks>
    /// Tolerance rationale: the ternary A operand {-1, 0, +1} is exact in F16 and the per-tensor
    /// scale is applied to the F32 accumulator, so the ONLY F16 error is the activation cast —
    /// ~2^-11 relative per element, accumulating as a sqrt(K) random walk. At K=2560 that is
    /// ~50 x 2^-11 ~ 2.4e-2 relative on a unit-scale dot product before the (small) scale, hence
    /// 3e-2 absolute. Still tight enough to catch tile-layout, staging and store-path bugs, which
    /// are the failure modes this kernel actually risks. The F32 scalar variant remains the
    /// reference on every shape.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]      // exactly one coopmat tile
    [InlineData(32, 256, 8)]      // 2 tiles in m, 2 blocks per row
    [InlineData(64, 128, 16)]     // 4 tiles in m, one full tile in n
    [InlineData(48, 768, 12)]     // partial tile in n -> exercises the staged scatter store
    [InlineData(17, 256, 33)]     // ragged in both dims -> bounds-guarded store path
    [InlineData(15, 128, 3)]      // smaller than one tile in both dims
    [InlineData(2560, 2560, 5)]   // BitNet hidden × hidden, small prefill batch
    public void Coopmat_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.Coopmat, m, k, n, absTol: 3e-2f, relTol: 5e-3f);

    /// <summary>
    /// Parity for the 32-thread coopmat probe. Same tolerance rationale as
    /// <see cref="Coopmat_MatchesScalarReference"/>; this exists so the probe's
    /// workgroup-size-agnostic strided staging is proven correct before its timings are trusted.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 33)]     // ragged both dims -> bounds-guarded store path
    [InlineData(2560, 2560, 5)]   // BitNet hidden × hidden
    public void Coopmat32_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.Coopmat32, m, k, n, absTol: 3e-2f, relTol: 5e-3f);

    /// <summary>
    /// Parity for the coopmat warptile variant (2x2 grid of 16x16 fragments, 32x32 output tile).
    /// Same tolerance rationale as <see cref="Coopmat_MatchesScalarReference"/>.
    /// </summary>
    /// <remarks>
    /// The rows deliberately stress the 32-wide tile edges. With four fragments there are four
    /// distinct store offsets, and a mis-set fragment offset would corrupt exactly one quadrant of
    /// the tile — which only a shape that does NOT align to 32 in both dimensions can expose.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 128, 32)]     // exactly one 32x32 tile — all four fragments fully in bounds
    [InlineData(64, 256, 64)]     // 2x2 tiles, 2 blocks per row
    [InlineData(48, 768, 40)]     // partial tile both dims -> staged scatter, 6 blocks
    [InlineData(33, 128, 33)]     // one past a full tile -> only the (0,0) fragment fully valid
    [InlineData(17, 256, 47)]     // ragged both dims, straddles the fragment boundary at 16
    [InlineData(15, 128, 3)]      // smaller than a single fragment
    [InlineData(2560, 2560, 5)]   // BitNet hidden × hidden, n far below the tile
    public void CoopmatWarptile_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.CoopmatWarptile, m, k, n, absTol: 3e-2f, relTol: 5e-3f);

    private static void RunParity(
        I2SGemmVariant variant, int m, int k, int n, float absTol = AbsTol, float relTol = RelTol)
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
        Skip.If(variant.RequiresCooperativeMatrix && !device.HasCooperativeMatrix,
            "Device does not advertise VK_KHR_cooperative_matrix.");
        using var kernel = MatMulI2SGemmF32Kernel.Create(device, spvDir, variant);

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
                float tol = absTol + relTol * MathF.Abs(expected[idx]);
                Assert.True(diff <= tol,
                    $"{variant.SpvFileName} cell t={t} m={r} (m={m}, k={k}, n={n}): expected {expected[idx]:G9}, got {actual[idx]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
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
