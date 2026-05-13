using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for <see cref="CudaKernels.LaunchQuantizedGemv"/>
/// across every quantization type the CUDA backend ships kernels for: Q8_0, Q2_K, Q4_K,
/// Q5_K, Q6_K, IQ4_NL, IQ4_XS. The kernel takes FP16 input and produces FP16 output;
/// the CPU reference dequantises the weight row to FP32 and runs an FP32 dot product
/// with the FP32-promoted input, then rounds back to FP16 for comparison.
/// </summary>
/// <remarks>
/// <para>
/// This is the heart of the decode path. Without per-quant-type parity coverage, a bad
/// PTX rebuild or a missed scale-decode change can silently corrupt every projection
/// for one quant family while leaving the others fine — which is exactly the failure
/// mode the IQ4_XS bug hunt surfaced.
/// </para>
/// <para>
/// Tolerance bands are sized empirically: K=256/512 random Gaussian inputs accumulate
/// ~sqrt(K) FP16-rounding error in the FP32 reduction, plus FP16 output rounding.
/// Q6_K / Q8_0 / IQ4 are tighter (5-bit+ precision); Q4_K / Q5_K need a touch more
/// headroom. Mixed absolute+relative form via
/// <see cref="CudaKernelTestHarness.AssertClose"/> handles both small-output rows and
/// rows that accumulated large dot products.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class QuantizedGemvParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CudaKernelTestHarness _harness = new();

    public QuantizedGemvParityTests(ITestOutputHelper output) => _out = output;

    public void Dispose() => _harness.Dispose();

    /// <summary>
    /// One row per supported quant type: (type, M, K, absTol, relTol). M and K are
    /// chosen to satisfy the kernel's alignment requirement
    /// (<see cref="CudaKernels.MinKAlignmentFor"/> — 32 for block-32, 256 for K-quants).
    /// Small shapes (M ≤ 64, K ≤ 512) keep each test well under 200 ms.
    /// </summary>
    /// <remarks>
    /// Tolerance band notes (observed maxAbs / refMax on this hardware, M=64 K=256):
    /// <list type="bullet">
    ///   <item>Q8_0:    maxAbs ~3.3e-3, ref|max| ~10  → 5e-3 relative is sufficient.</item>
    ///   <item>Q2_K:    maxAbs ~4.4e-4, ref|max| ~2   → 5e-3 absolute is sufficient.</item>
    ///   <item>Q4_K:    maxAbs ~1.5e-2, ref|max| ~50  → needs 1e-2 relative tolerance.</item>
    ///   <item>Q5_K:    maxAbs ~6.2e-2, ref|max| ~204 → needs 1e-2 relative tolerance.</item>
    ///   <item>Q6_K:    maxAbs ~1.4e-2, ref|max| ~48  → needs 5e-3 relative tolerance.</item>
    ///   <item>IQ4_NL:  maxAbs ~3.9e-3, ref|max| ~12  → 5e-3 relative is sufficient.</item>
    ///   <item>IQ4_XS:  maxAbs comparable, ref|max| comparable → 5e-3 relative is sufficient.</item>
    /// </list>
    /// The absolute tolerance handles near-zero outputs; the relative tolerance handles
    /// large outputs where accumulated FP16-rounding error scales with the row sum.
    /// </remarks>
    public static IEnumerable<object[]> QuantTypes() =>
    [
        // Block-32 quants: K must be a multiple of 32. Q8_0 is the highest-precision
        // weight format — tightest tolerance.
        [QuantizationType.Q8_0, 64, 256, 2e-3f, 5e-3f],

        // K-quants (super-block = 256). All require K % 256 == 0.
        [QuantizationType.Q2_K, 64, 256, 5e-3f, 1e-2f],
        [QuantizationType.Q4_K, 64, 256, 5e-3f, 1e-2f],
        [QuantizationType.Q5_K, 64, 256, 5e-3f, 1e-2f],
        [QuantizationType.Q6_K, 64, 256, 2e-3f, 5e-3f],

        // I-quants: IQ4_NL is block-32; IQ4_XS is super-block-256.
        [QuantizationType.IQ4_NL, 64, 256, 2e-3f, 5e-3f],
        [QuantizationType.IQ4_XS, 64, 256, 2e-3f, 5e-3f],
    ];

    [SkippableTheory]
    [MemberData(nameof(QuantTypes))]
    public unsafe void QuantizedGemv_PerType_MatchesCpuReference(
        QuantizationType qt, int m, int k, float absTol, float relTol)
    {
        _harness.SkipIfUnavailable();
        Skip.IfNot(_harness.Kernels.HasQuantizedGemvKernel(qt),
            $"No GEMV kernel loaded for {qt} (stale PTX or missing symbol)");

        // Kernel alignment contract.
        Assert.Equal(0, k % CudaKernels.MinKAlignmentFor(qt));

        // ─── Build synthetic quantised weight matrix ───
        // Use random bytes for the qs/qh payload, but overwrite the per-block scale Halves
        // with small reasonable values so the resulting FP32 row entries stay in a sane
        // range (~[-1, +1]) — random qs interpreted with a random scale half can blow
        // up to Inf/NaN.
        long rowBytes = Dequantize.RowByteSize(k, qt);
        long weightBytes = (long)m * rowBytes;
        var rng = new Random(0xC0FFEE ^ (int)qt ^ m ^ k);
        byte[] hostW = new byte[weightBytes];
        rng.NextBytes(hostW);
        WriteReasonableScales(hostW, qt, m, k, rng);

        // ─── Build random FP16 input vector ───
        Half[] hostX = CudaKernelTestHarness.RandomF16(rng, k, scale: 0.4f);
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++) xF32[i] = (float)hostX[i];

        // ─── CPU reference: dequant row → F32 dot ───
        // Use Dequantize.ToFloat32 as the authoritative scalar oracle. Both Q-row dequant
        // and an FP32 reduction match what the GPU's quantized GEMV is meant to compute,
        // with the GPU additionally rounding the final accumulator to FP16.
        float[] yRefF32 = new float[m];
        float[] rowDequant = new float[k];
        fixed (byte* pW = hostW)
        {
            for (int row = 0; row < m; row++)
            {
                Dequantize.ToFloat32((nint)(pW + row * rowBytes), k, qt, rowDequant);
                float acc = 0;
                for (int i = 0; i < k; i++) acc += rowDequant[i] * xF32[i];
                yRefF32[row] = acc;
            }
        }

        // ─── GPU GEMV ───
        nint devW = _harness.Upload(hostW);
        nint devX = _harness.Upload(hostX);
        nint devY = _harness.Allocate((long)m * sizeof(ushort));

        _harness.Kernels.LaunchQuantizedGemv(devW, qt, devX, devY, m, k, _harness.StreamHandle);
        _harness.Synchronize();

        Half[] yGpuF16 = _harness.DownloadHalves(devY, m);
        float[] yGpuF32 = new float[m];
        for (int i = 0; i < m; i++) yGpuF32[i] = (float)yGpuF16[i];

        // Diagnostic line for easy tolerance tuning if a kernel changes.
        float maxAbs = 0, refMax = 0;
        for (int i = 0; i < m; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(yRefF32[i] - yGpuF32[i]));
            refMax = MathF.Max(refMax, MathF.Abs(yRefF32[i]));
        }
        _out.WriteLine($"{qt} M={m} K={k}: ref|max|={refMax:F3}  max-abs-diff={maxAbs:E4}  " +
                       $"absTol={absTol:E2}  relTol={relTol:E2}");

        CudaKernelTestHarness.AssertClose($"QuantizedGemv-{qt}",
            yRefF32, yGpuF32, absTol, relTol);
    }

    /// <summary>
    /// Overwrite the per-block scale half (and Q4_K/Q5_K's dmin half) with a small
    /// random magnitude so the dequantised values stay in a range where the GEMV
    /// output fits comfortably inside FP16 representable values. Without this step the
    /// scales — interpreted from random bytes — span the full FP16 range and the
    /// downstream FP32 dot product overflows on inversion.
    /// </summary>
    private static unsafe void WriteReasonableScales(byte[] hostW, QuantizationType qt,
                                                      int m, int k, Random rng)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);

        fixed (byte* pW = hostW)
        {
            switch (qt)
            {
                case QuantizationType.Q8_0:
                    {
                        // Block-32: each block is 2-byte half scale + 32 int8 qs.
                        const int blockSize = 32;
                        const int blockBytes = 34;
                        int blocksPerRow = k / blockSize;
                        for (int row = 0; row < m; row++)
                            for (int b = 0; b < blocksPerRow; b++)
                            {
                                byte* blk = pW + row * rowBytes + b * blockBytes;
                                *(Half*)blk = (Half)((rng.NextDouble() - 0.5) * 0.04);
                            }
                        break;
                    }

                case QuantizationType.IQ4_NL:
                    {
                        // Block-32: 2-byte half d + 16-byte qs.
                        const int blockSize = 32;
                        const int blockBytes = 18;
                        int blocksPerRow = k / blockSize;
                        for (int row = 0; row < m; row++)
                            for (int b = 0; b < blocksPerRow; b++)
                            {
                                byte* blk = pW + row * rowBytes + b * blockBytes;
                                *(Half*)blk = (Half)((rng.NextDouble() - 0.5) * 0.04);
                            }
                        break;
                    }

                case QuantizationType.Q2_K:
                    {
                        // Super-block-256, 84 bytes/block. d half @ +80, dmin half @ +82.
                        const int superBlockBytes = 84;
                        int sbPerRow = k / 256;
                        for (int row = 0; row < m; row++)
                            for (int sb = 0; sb < sbPerRow; sb++)
                            {
                                byte* blk = pW + row * rowBytes + sb * superBlockBytes;
                                *(Half*)(blk + 80) = (Half)((rng.NextDouble() - 0.5) * 0.04);
                                *(Half*)(blk + 82) = (Half)((rng.NextDouble() - 0.5) * 0.02);
                            }
                        break;
                    }

                case QuantizationType.Q4_K:
                    {
                        // Super-block-256, 144 bytes/block. d half @ +0, dmin half @ +2.
                        const int superBlockBytes = 144;
                        int sbPerRow = k / 256;
                        for (int row = 0; row < m; row++)
                            for (int sb = 0; sb < sbPerRow; sb++)
                            {
                                byte* blk = pW + row * rowBytes + sb * superBlockBytes;
                                *(Half*)(blk + 0) = (Half)((rng.NextDouble() - 0.5) * 0.04);
                                *(Half*)(blk + 2) = (Half)((rng.NextDouble() - 0.5) * 0.02);
                            }
                        break;
                    }

                case QuantizationType.Q5_K:
                    {
                        // Super-block-256, 176 bytes/block. d half @ +0, dmin half @ +2.
                        const int superBlockBytes = 176;
                        int sbPerRow = k / 256;
                        for (int row = 0; row < m; row++)
                            for (int sb = 0; sb < sbPerRow; sb++)
                            {
                                byte* blk = pW + row * rowBytes + sb * superBlockBytes;
                                *(Half*)(blk + 0) = (Half)((rng.NextDouble() - 0.5) * 0.04);
                                *(Half*)(blk + 2) = (Half)((rng.NextDouble() - 0.5) * 0.02);
                            }
                        break;
                    }

                case QuantizationType.Q6_K:
                    {
                        // Super-block-256, 210 bytes/block. d half @ +208 (last 2 bytes).
                        const int superBlockBytes = 210;
                        int sbPerRow = k / 256;
                        for (int row = 0; row < m; row++)
                            for (int sb = 0; sb < sbPerRow; sb++)
                            {
                                byte* blk = pW + row * rowBytes + sb * superBlockBytes;
                                *(Half*)(blk + 208) = (Half)((rng.NextDouble() - 0.5) * 0.005);
                            }
                        break;
                    }

                case QuantizationType.IQ4_XS:
                    {
                        // Super-block-256, 136 bytes/block. d half @ +0.
                        const int superBlockBytes = 136;
                        int sbPerRow = k / 256;
                        for (int row = 0; row < m; row++)
                            for (int sb = 0; sb < sbPerRow; sb++)
                            {
                                byte* blk = pW + row * rowBytes + sb * superBlockBytes;
                                *(Half*)blk = (Half)((rng.NextDouble() - 0.5) * 0.002);
                            }
                        break;
                    }

                default:
                    throw new ArgumentOutOfRangeException(nameof(qt), qt, "Unsupported quant type for parity test.");
            }
        }
    }
}
