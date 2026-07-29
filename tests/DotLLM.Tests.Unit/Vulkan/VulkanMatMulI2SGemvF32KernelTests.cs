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
