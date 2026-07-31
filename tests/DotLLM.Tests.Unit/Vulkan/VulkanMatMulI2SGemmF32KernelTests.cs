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
    /// Same parity contract for the wide-load unpack variant, at the F32 tolerance — the unpack
    /// change is bit-exact, so this must match as tightly as the baseline, not merely closely.
    /// </summary>
    /// <remarks>
    /// These rows are chosen to attack the wide-load's specific risk: it assumes the four bytes a
    /// thread decodes form ONE aligned 32-bit word. That holds only because <c>rowBytes = K/4</c>
    /// is a multiple of 32. The varying K values (128 / 256 / 768 / 2560, i.e. rowBytes
    /// 32 / 64 / 192 / 640) exercise that alignment argument at several row strides, and the
    /// partial-tile rows confirm the hoisted bounds test still zero-fills correctly.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]      // rowBytes=32, minimum stride
    [InlineData(32, 256, 8)]      // rowBytes=64
    [InlineData(64, 128, 16)]
    [InlineData(48, 768, 12)]     // rowBytes=192, partial tile -> hoisted bounds test
    [InlineData(33, 128, 33)]     // one past a full tile in both dims
    [InlineData(17, 256, 47)]     // ragged both dims
    [InlineData(15, 128, 3)]      // below the micro-tile stride
    [InlineData(2560, 2560, 5)]   // rowBytes=640, BitNet hidden × hidden
    public void RegisterBlockedWide_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.RegisterBlockedWide, m, k, n);

    /// <summary>
    /// The wide-load unpack must be <b>bit-identical</b> to <see cref="I2SGemmVariant.RegisterBlocked"/>,
    /// not merely within tolerance.
    /// </summary>
    /// <remarks>
    /// It writes the same values into the same shared slots and accumulates in the same order, so
    /// every output bit must match exactly. Asserting exact equality is a far sharper discriminator
    /// than the tolerance check: a byte-extraction or endianness slip that permuted codes within a
    /// word would shift results by a small amount that a 5e-3 tolerance could absorb on random
    /// ternary data, but cannot survive bitwise comparison.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 47)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlockedWide_IsBitIdenticalToRegisterBlocked(int m, int k, int n)
        => AssertBitIdentical(I2SGemmVariant.RegisterBlockedWide, m, k, n);

    /// <summary>
    /// The bank-padded variant must also be bit-identical to the production kernel — it changes only
    /// the shared-memory row stride, never a value or an accumulation order.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 47)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlockedPadded_IsBitIdenticalToRegisterBlocked(int m, int k, int n)
        => AssertBitIdentical(I2SGemmVariant.RegisterBlockedPadded, m, k, n);

    /// <summary>
    /// The 4x4 ILP variant must be bit-identical to the production kernel: it changes the
    /// thread-to-output mapping and the number of threads, but not the per-cell accumulation order.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 47)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlocked4x4_IsBitIdenticalToRegisterBlocked(int m, int k, int n)
        => AssertBitIdentical(I2SGemmVariant.RegisterBlocked4x4, m, k, n);

    /// <summary>
    /// Parity for the 4x4 ILP variant against the scalar reference, stressing the SUB=8 micro-tile
    /// stride. With 4x4 cells per thread there are 16 separate bounds checks per thread, and a
    /// stride error shows up only on shapes not aligned to 32 in both dimensions.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 128, 32)]     // exactly one tile
    [InlineData(64, 256, 64)]     // 2x2 tiles
    [InlineData(48, 768, 12)]     // partial in both dims
    [InlineData(33, 128, 33)]     // one past a full tile
    [InlineData(17, 256, 47)]     // ragged, straddles the SUB=8 stride
    [InlineData(15, 128, 3)]      // below the micro-tile stride
    [InlineData(2560, 2560, 5)]   // BitNet hidden x hidden
    public void RegisterBlocked4x4_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.RegisterBlocked4x4, m, k, n);

    /// <summary>
    /// Parity for the F16-shared variant. NOT bit-exact, so this runs at a widened tolerance —
    /// but it is the only thing standing between "1.3x faster" and "1.3x faster and wrong".
    /// </summary>
    /// <remarks>
    /// Only the activations round: sharedW holds raw ternary {-1, 0, +1}, all exact in F16, so the
    /// weight side is lossless. The activation cast costs ~2^-11 relative per element, accumulating
    /// as a sqrt(K) random walk — at K=2560 that is ~50 x 2^-11 ~ 2.4e-2 relative on a unit-scale
    /// dot product, hence the 3e-2 absolute allowance, matching the coopmat variants which round
    /// activations the same way.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(33, 128, 33)]
    [InlineData(17, 256, 47)]
    [InlineData(15, 128, 3)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlockedF16Shared_MatchesScalarReference(int m, int k, int n)
        => RunParity(I2SGemmVariant.RegisterBlockedF16Shared, m, k, n, absTol: 3e-2f, relTol: 5e-3f);

    /// <summary>
    /// The F16 weight tile must be bit-identical to production: ternary is exactly representable in
    /// F16, so staging it there changes no value and no accumulation order. Exact equality is the
    /// right gate — it proves the F16 round-trip is lossless rather than merely close.
    /// </summary>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 47)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlockedWeightF16_IsBitIdenticalToRegisterBlocked(int m, int k, int n)
        => AssertBitIdentical(I2SGemmVariant.RegisterBlockedWeightF16, m, k, n);

    /// <summary>
    /// The int8 weight tile must be bit-identical to production: ternary is exactly representable in
    /// int8, so staging it there changes no value and no accumulation order.
    /// </summary>
    /// <remarks>
    /// Skipped when the device lacks 8-bit storage — pipeline creation throws in that case, which is
    /// the same condition the production fallback chain handles.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(32, 256, 8)]
    [InlineData(48, 768, 12)]
    [InlineData(17, 256, 47)]
    [InlineData(2560, 2560, 5)]
    public void RegisterBlockedWeightInt8_IsBitIdenticalToRegisterBlocked(int m, int k, int n)
        => AssertBitIdentical(I2SGemmVariant.RegisterBlockedWeightInt8, m, k, n);

    private static void AssertBitIdentical(I2SGemmVariant variant, int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x5E_ED ^ (m * 7 + k * 11 + n * 13));
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = rng.NextSingle() * 0.05f + 0.01f;
        float[] inputB = new float[(long)n * k];
        for (int i = 0; i < inputB.Length; i++) inputB[i] = rng.NextSingle() * 2f - 1f;
        byte[] weightsI2S = PackI2S(ternary, m, k, scale);

        using var device = VulkanDevice.Create();
        float[] baseline = RunVariant(device, spvDir, I2SGemmVariant.RegisterBlocked, weightsI2S, inputB, m, k, n);
        float[] candidate;
        try
        {
            candidate = RunVariant(device, spvDir, variant, weightsI2S, inputB, m, k, n);
        }
        catch (Exception ex) when (ex is InvalidOperationException or DotLLM.Vulkan.Interop.VulkanException)
        {
            // Device lacks the small-type storage feature this variant needs (e.g. 8-bit storage).
            throw new SkipException($"{variant.SpvFileName} not creatable on this device: {ex.Message}");
        }

        for (int i = 0; i < baseline.Length; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(candidate[i]),
                $"{variant.SpvFileName} cell {i} (m={m}, k={k}, n={n}) differs: register-blocked {baseline[i]:G9} vs variant {candidate[i]:G9}");
        }
    }

    private static float[] RunVariant(
        VulkanDevice device, string spvDir, I2SGemmVariant variant,
        byte[] weightsI2S, float[] inputB, int m, int k, int n)
    {
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
        return actual;
    }

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
