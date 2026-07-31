using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan I2_S (BitNet b1.58 ternary) GEMV kernel against a
/// scalar ground-truth reference. The reference is computed directly from the unpacked ternary
/// values (<c>y[r] = scale · Σ_c ternary[r,c] · x[c]</c>), so a passing test proves the GPU kernel
/// decodes the I2_S bit-layout (byte <c>gp</c> packs positions {gp,gp+32,gp+64,gp+96} at bits
/// {6,4,2,0}, value = code−1), reads the per-tensor scale from the buffer tail (offset m·K/4),
/// and reduces correctly. Mirrors the Q2_K/Q4_K GEMV parity tests.
/// </summary>
/// <remarks>
/// Tolerance — I2_S codes are exact ternary (no per-element quant error), so the only divergence
/// from the sequential CPU reference is reduction order (GPU 128-thread tree reduce vs scalar sum).
/// The same 5e-3 / 1e-3 tolerances as the other K-quant GEMV parity tests cover it.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulI2SGemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 128)]      // one block → 32 bytes packed
    [InlineData(8, 128)]
    [InlineData(4, 256)]      // two blocks
    [InlineData(16, 768)]
    [InlineData(2048, 256)]   // BitNet-class output dim
    [InlineData(2560, 2560)]  // BitNet hidden × hidden (o_proj-ish)
    [InlineData(576, 1024)]
    public void Launch_MatchesScalarReference(int m, int k)
        => RunParity("matmul_i2_s_f32_gemv.spv", m, k);

    /// <summary>
    /// Same parity contract for the subgroup-reduction variant, which replaces the 128-entry
    /// shared-memory tree reduce with <c>subgroupAdd</c> plus a single cross-subgroup step.
    /// Only the reduction differs, so what these rows actually guard is that the cross-subgroup
    /// step sums every subgroup's partial exactly once — a bug there shows up as a value scaled
    /// by a whole fraction, which the k-varying rows catch since they change the lane workload.
    /// </summary>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    [SkippableTheory]
    [InlineData(1, 128)]      // single block — most lanes contribute zero to the reduction
    [InlineData(8, 128)]
    [InlineData(4, 256)]      // two blocks
    [InlineData(16, 768)]
    [InlineData(2048, 256)]   // BitNet-class output dim
    [InlineData(2560, 2560)]  // BitNet hidden × hidden (o_proj-ish)
    [InlineData(576, 1024)]
    public void Subgroup_MatchesScalarReference(int m, int k)
        => RunParity("matmul_i2_s_f32_gemv_sg.spv", m, k);

    /// <summary>
    /// Parity for the multi-row decode GEMV (4 output rows per workgroup).
    /// </summary>
    /// <remarks>
    /// The rows deliberately include m values that are NOT multiples of 4 (1, 15, 2049, 577), because
    /// the variant's tail guard (<c>if (m &gt;= pc.M) break;</c> plus the guarded store) is the one
    /// thing a multi-row mapping can get wrong: a workgroup covering the final partial group must
    /// write only the rows that exist. m=1 is the extreme case — a whole workgroup for a single row.
    /// </remarks>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    [SkippableTheory]
    [InlineData(1, 128)]      // single row — 3 of 4 lanes must be suppressed
    [InlineData(4, 128)]      // exactly one full group
    [InlineData(8, 128)]
    [InlineData(15, 256)]     // ragged tail (15 = 3*4 + 3)
    [InlineData(16, 768)]
    [InlineData(577, 1024)]   // ragged at scale
    [InlineData(2049, 256)]   // ragged, large m
    [InlineData(2560, 2560)]  // BitNet hidden × hidden
    public void MultiRow_MatchesScalarReference(int m, int k)
        => RunParity("matmul_i2_s_f32_gemv_mr4.spv", m, k);

    /// <summary>Parity for the 8-row decode GEMV. Same tail-guard risk as the 4-row variant.</summary>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    [SkippableTheory]
    [InlineData(1, 128)]      // 7 of 8 lanes suppressed
    [InlineData(8, 128)]      // exactly one group
    [InlineData(15, 256)]     // ragged tail
    [InlineData(577, 1024)]
    [InlineData(2049, 256)]
    [InlineData(2560, 2560)]
    public void MultiRow8_MatchesScalarReference(int m, int k)
        => RunParity("matmul_i2_s_f32_gemv_mr8.spv", m, k);

    /// <summary>
    /// The multi-row GEMV variants must be <b>bit-identical</b> to the production kernel, not merely
    /// within tolerance: each output element accumulates over the same k in the same per-thread
    /// stride order and through the same tree reduce, so only the row-to-workgroup mapping changes.
    /// </summary>
    /// <remarks>
    /// This is the sharp gate for a multi-row mapping. A row-indexing slip would put a correct-looking
    /// value in the wrong output row, which a tolerance check against a scalar reference can mask when
    /// neighbouring rows have similar magnitudes, but bitwise comparison against the production kernel
    /// cannot.
    /// </remarks>
    /// <param name="spv">Variant SPIR-V to compare.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    [SkippableTheory]
    [InlineData("matmul_i2_s_f32_gemv_mr4.spv", 2560, 2560)]
    [InlineData("matmul_i2_s_f32_gemv_mr4.spv", 577, 1024)]
    [InlineData("matmul_i2_s_f32_gemv_mr4.spv", 2049, 256)]
    [InlineData("matmul_i2_s_f32_gemv_mr8.spv", 2560, 2560)]
    [InlineData("matmul_i2_s_f32_gemv_mr8.spv", 577, 1024)]
    [InlineData("matmul_i2_s_f32_gemv_mr8.spv", 2049, 256)]
    public void MultiRow_IsBitIdenticalToProduction(string spv, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x5E_ED ^ (m * 7 + k * 11));
        sbyte[] ternary = new sbyte[(long)m * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = rng.NextSingle() * 0.05f + 0.01f;
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
        byte[] weightsI2S = PackI2S(ternary, m, k, scale);

        using var device = VulkanDevice.Create();
        float[] baseline = RunVariantRaw(device, spvDir, "matmul_i2_s_f32_gemv.spv", weightsI2S, x, m, k);
        float[] candidate = RunVariantRaw(device, spvDir, spv, weightsI2S, x, m, k);

        for (int i = 0; i < baseline.Length; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(candidate[i]),
                $"{spv} row {i} (m={m}, k={k}) differs: production {baseline[i]:G9} vs variant {candidate[i]:G9}");
        }
    }

    private static float[] RunVariantRaw(
        VulkanDevice device, string spvDir, string spvFileName, byte[] weightsI2S, float[] x, int m, int k)
    {
        using var kernel = MatMulI2SGemvF32Kernel.Create(device, spvDir, spvFileName);
        long weightsBufBytes = ((long)weightsI2S.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weightsI2S), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k);
        float[] y = new float[m];
        device.Download(bufY, y);
        return y;
    }

    private static void RunParity(string spvFileName, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x12_5C ^ (m * 7 + k * 11));

        // Random ternary weights {-1,0,+1} and a random per-tensor scale + activation.
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = rng.NextSingle() * 0.05f + 0.01f;
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsI2S = PackI2S(ternary, m, k, scale);

        // Ground-truth reference: y[r] = scale · Σ_c ternary[r,c] · x[c].
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            int rowBase = r * k;
            for (int c = 0; c < k; c++) acc += ternary[rowBase + c] * (double)x[c];
            expected[r] = (float)(acc * scale);
        }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulI2SGemvF32Kernel.Create(device, spvDir, spvFileName);

        long weightsBufBytes = ((long)weightsI2S.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsI2S), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertClose(expected, actual, m, k);
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix into the I2_S byte layout the kernel decodes:
    /// per 128-element block (32 bytes), byte <c>gp</c> holds block-relative positions
    /// {gp, gp+32, gp+64, gp+96} at bit offsets {6, 4, 2, 0}; stored code = value + 1 ∈ {0,1,2}.
    /// Row stride = k/4 bytes; a single float32 scale is appended at offset m·(k/4).
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
