using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #443: parity for <see cref="MoeGroupedCoopmatVariant.Blocked128x128x4"/>, the grouped-MoE
/// instantiation of the shared 128x128 blocked coopmat template.
/// </summary>
/// <remarks>
/// <para>
/// The reference is a CPU GEMM over the exact F16 bytes the GPU sees, so this is a real oracle
/// rather than a self-comparison.
/// </para>
/// <para>
/// <b>Ragged expert counts are the whole point.</b> The expert indirection resolves to a
/// workgroup-uniform base offset, but the thing that can still go wrong is the row bound: with
/// <c>BN = 128</c> a workgroup covers 128 packed rows while an expert may own three, so every
/// shape here mixes experts that are larger than a tile, smaller than a tile, and <b>empty</b>.
/// An empty expert is the case that most easily writes into the next expert's rows, and the
/// legacy 16x16 kernel's smaller tile hid how far such a write would reach.
/// </para>
/// </remarks>
public sealed class VulkanMoeGroupedMatmulBlockedTileTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    /// <summary>
    /// Parity against a CPU reference for a mix of expert row counts, including empty experts
    /// and counts above and below the 128-row tile.
    /// </summary>
    /// <param name="m">Output dim per expert.</param>
    /// <param name="k">Contraction dim; must be a multiple of 32.</param>
    /// <param name="counts">Rows owned by each expert, comma separated.</param>
    [SkippableTheory]
    [InlineData(128, 128, "128")]                  // exactly one blocked tile, one expert
    [InlineData(64, 64, "3,0,17,40")]              // ragged, an EMPTY expert, all under one tile
    [InlineData(160, 96, "200,1,0,63")]            // an expert past a full tile, next to an empty one
    [InlineData(33, 32, "5,7")]                    // M below one coopmat fragment
    [InlineData(256, 128, "129,0,2")]              // one row past a full blocked tile
    public void Blocked128x128x4_MatchesCpuReference(int m, int k, string counts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "Device does not advertise VK_KHR_cooperative_matrix.");

        var variant = MoeGroupedCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(variant.IsSupportedOn(device, spvDir),
            $"{variant.SpvFileName} not compiled (native/vulkan/build.ps1/build.sh).");

        int[] expertRows = Array.ConvertAll(counts.Split(','), int.Parse);
        int numExperts = expertRows.Length;
        int rows = 0;
        uint[] offsets = new uint[numExperts + 1];
        for (int e = 0; e < numExperts; e++)
        {
            offsets[e] = (uint)rows;
            rows += expertRows[e];
        }
        offsets[numExperts] = (uint)rows;

        var rng = new Random(0x443_30E + m * 31 + k * 7 + rows);
        int rowUints = k / 2;

        // Weight bank [E, M, K] F16, two elements per uint.
        uint[] bank = new uint[(long)numExperts * m * rowUints];
        float[] bankF32 = new float[(long)numExperts * m * k];
        for (int i = 0; i < bank.Length; i++)
        {
            float lo = rng.NextSingle() * 0.2f - 0.1f;
            float hi = rng.NextSingle() * 0.2f - 0.1f;
            var loH = (Half)lo;
            var hiH = (Half)hi;
            bank[i] = BitConverter.HalfToUInt16Bits(loH) | ((uint)BitConverter.HalfToUInt16Bits(hiH) << 16);
            bankF32[2L * i] = (float)loH;
            bankF32[2L * i + 1] = (float)hiH;
        }

        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        // CPU reference over the exact bytes the GPU sees.
        float[] expected = new float[(long)rows * m];
        for (int e = 0; e < numExperts; e++)
        {
            for (uint r = offsets[e]; r < offsets[e + 1]; r++)
            {
                for (int col = 0; col < m; col++)
                {
                    double acc = 0;
                    long wBase = ((long)e * m + col) * k;
                    for (int t = 0; t < k; t++) acc += (double)bankF32[wBase + t] * x[(long)r * k + t];
                    expected[(long)r * m + col] = (float)acc;
                }
            }
        }

        using var kernel = MoeGroupedMatmulF16CoopmatKernel.Create(device, spvDir, variant);
        using var bufW = device.Allocate((long)bank.Length * sizeof(uint));
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate((long)offsets.Length * sizeof(uint));
        using var bufY = device.Allocate((long)rows * m * sizeof(float));

        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(bank), bufW);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(offsets), bufOff);

        kernel.Launch(bufW, bufX, bufOff, bufY, m, k, rows, numExperts);

        float[] actual = new float[(long)rows * m];
        device.Download(bufY, actual);

        for (int r = 0; r < rows; r++)
        {
            for (int col = 0; col < m; col++)
            {
                long idx = (long)r * m + col;
                float diff = MathF.Abs(expected[idx] - actual[idx]);
                float tol = AbsTol + RelTol * MathF.Abs(expected[idx]);
                Assert.True(diff <= tol,
                    $"row {r} col {col} (m={m}, k={k}, counts={counts}): expected {expected[idx]:G9}, "
                    + $"got {actual[idx]:G9}, |delta|={diff:G9} > tol {tol:G9}");
            }
        }
    }
}
