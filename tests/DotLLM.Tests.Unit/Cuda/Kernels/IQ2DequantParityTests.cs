using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Parity coverage for the IQ2 family (IQ2_XXS / IQ2_XS / IQ2_S — IQ2_S also
/// stores the 2-bit tensors in <c>MOSTLY_IQ2_M</c> file-type GGUFs).
/// </summary>
/// <remarks>
/// <para>
/// Two independent checks per type:
/// </para>
/// <list type="number">
///   <item>
///     <description>
///       <b>Self-consistent scalar reference</b> — a manually-coded scalar
///       dequantiser inlined into the test (mirroring <c>ggml-quants.c</c>'s
///       <c>dequantize_row_iq2_*</c>) is asserted bit-for-bit equal to the
///       production <see cref="Dequantize.ToFloat32"/> implementation. This is
///       the "0-ULP round-trip vs gguf reference" gate from the task spec: if
///       the production kernel diverges from the C reference formula by a single
///       bit, this test fails.
///     </description>
///   </item>
///   <item>
///     <description>
///       <b>CUDA dequant matches CPU</b> — the on-GPU IQ2 dequant produces the
///       same float values as the CPU oracle, within FP16 round-trip tolerance.
///     </description>
///   </item>
/// </list>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed unsafe class IQ2DequantParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CudaKernelTestHarness _harness = new();

    public IQ2DequantParityTests(ITestOutputHelper output) => _out = output;

    public void Dispose() => _harness.Dispose();

    // ──────────────────── CPU bit-perfect oracle parity ────────────────────

    /// <summary>
    /// Hand-coded IQ2_XXS dequant from the C reference, asserted bit-for-bit
    /// equal to the production <see cref="Dequantize.DequantizeIQ2_XXS"/>.
    /// </summary>
    [Fact]
    public void IQ2_XXS_CpuMatchesScalarReference()
    {
        var rng = new Random(0xC0FFEE);
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * 256;
        byte[] packed = new byte[SuperBlocks * Dequantize.IQ2_XXS_BlockBytes];
        rng.NextBytes(packed);
        // Pin block scale to a known small magnitude so dequantised values stay
        // in a sane range — the rest of the block payload is random.
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * Dequantize.IQ2_XXS_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ2_XXS, production);

        float[] reference = ReferenceDequantIQ2_XXS(packed, SuperBlocks);

        AssertExactlyEqual("IQ2_XXS", reference, production);
    }

    /// <summary>
    /// Hand-coded IQ2_XS dequant from the C reference, asserted bit-for-bit
    /// equal to the production <see cref="Dequantize.DequantizeIQ2_XS"/>.
    /// </summary>
    [Fact]
    public void IQ2_XS_CpuMatchesScalarReference()
    {
        var rng = new Random(0xBEEF);
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * 256;
        byte[] packed = new byte[SuperBlocks * Dequantize.IQ2_XS_BlockBytes];
        rng.NextBytes(packed);
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * Dequantize.IQ2_XS_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ2_XS, production);

        float[] reference = ReferenceDequantIQ2_XS(packed, SuperBlocks);

        AssertExactlyEqual("IQ2_XS", reference, production);
    }

    /// <summary>
    /// Hand-coded IQ2_S dequant from the C reference, asserted bit-for-bit
    /// equal to the production <see cref="Dequantize.DequantizeIQ2_S"/>. IQ2_S
    /// is also the on-disk type for MOSTLY_IQ2_M file-type recipes.
    /// </summary>
    [Fact]
    public void IQ2_S_CpuMatchesScalarReference()
    {
        var rng = new Random(0xCAFE);
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * 256;
        byte[] packed = new byte[SuperBlocks * Dequantize.IQ2_S_BlockBytes];
        rng.NextBytes(packed);
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * Dequantize.IQ2_S_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ2_S, production);

        float[] reference = ReferenceDequantIQ2_S(packed, SuperBlocks);

        AssertExactlyEqual("IQ2_S", reference, production);
    }

    /// <summary>
    /// Spot-check IQ2_S dequant against pre-computed values for a single
    /// hand-constructed block. Every bit in the block layout is known, so a
    /// single mismatched output element is a layout-decoding bug.
    /// </summary>
    [Fact]
    public void IQ2_S_HandConstructedBlock_Element0Matches()
    {
        // Construct a single IQ2_S super-block with known bytes:
        //   d = 0.5 (Half = 0x3800)
        //   qsLow[0] = 0  -> grid index lo bits = 0
        //   qsSigns[0] = 0xFF -> all 8 elements flipped -> negative
        //   qh[0] = 0     -> high bits 0 -> grid index = 0
        //   scales[0] = 0x88  -> low nibble = 8 (db0 = 0.5*(0.5+8)*0.25 = 1.0625),
        //                       high nibble = 8 (db1 = 1.0625)
        // For l=0, j=0..7: grid index 0 in iq2s_grid = (0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08).
        // Expected y[0..7] = db0 * 0x08 * (-1) = 1.0625 * 8 * -1 = -8.5
        byte[] packed = new byte[Dequantize.IQ2_S_BlockBytes];
        fixed (byte* p = packed)
        {
            *(Half*)(p + 0) = (Half)0.5f;
            // qs[0..31] (qs_low) all zero, qs[32..63] (signs):
            p[2 + 32] = 0xFF;
            // qh all zero, scales[0] = 0x88
            p[2 + 64 + 8 + 0] = 0x88;
        }
        float[] dequant = new float[256];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, 256, QuantizationType.IQ2_S, dequant);

        // db0 = 0.5 * (0.5 + 8) * 0.25 = 1.0625
        // grid[0][j] = 0x08 = 8 for all 8 elements
        // sign[j] = -1 (qs_signs[0] = 0xFF flips all bits)
        // expected = -1.0625 * 8 = -8.5
        float expected = -8.5f;
        for (int j = 0; j < 8; j++)
        {
            Assert.Equal(expected, dequant[j]);
        }
    }

    // ──────────────────── CUDA dequant parity ────────────────────

    public static IEnumerable<object[]> IQ2Types() =>
    [
        [QuantizationType.IQ2_XXS],
        [QuantizationType.IQ2_XS],
        [QuantizationType.IQ2_S],
    ];

    [SkippableTheory]
    [MemberData(nameof(IQ2Types))]
    public void IQ2_CudaDequantMatchesCpu(QuantizationType qt)
    {
        _harness.SkipIfUnavailable();

        const int SuperBlocks = 8;
        const int Elem = SuperBlocks * 256;
        int blockBytes = qt switch
        {
            QuantizationType.IQ2_XXS => Dequantize.IQ2_XXS_BlockBytes,
            QuantizationType.IQ2_XS => Dequantize.IQ2_XS_BlockBytes,
            QuantizationType.IQ2_S => Dequantize.IQ2_S_BlockBytes,
            _ => throw new ArgumentOutOfRangeException(nameof(qt)),
        };

        var rng = new Random(0xDEAD ^ (int)qt);
        byte[] packed = new byte[SuperBlocks * blockBytes];
        rng.NextBytes(packed);
        // Stamp small block scales so dequantised values stay in FP16 range.
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * blockBytes) = (Half)((rng.NextDouble() - 0.5) * 0.002);
        }

        // CPU oracle (FP32).
        float[] cpuF32 = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, qt, cpuF32);

        // GPU F32 dequant.
        nint devSrc = _harness.Upload(packed);
        nint devDstF32 = _harness.Allocate((long)Elem * sizeof(float));
        _harness.Kernels.LaunchDequantToF32(devSrc, qt, devDstF32, Elem, _harness.StreamHandle);
        _harness.Synchronize();
        float[] gpuF32 = _harness.DownloadFloats(devDstF32, Elem);

        // GPU F32 path is bit-exact vs CPU (same float arithmetic order). Allow
        // a tiny tolerance for FMA fusion the PTX may apply.
        AssertCloseF32($"{qt}-F32", cpuF32, gpuF32, absTol: 1e-6f, relTol: 1e-6f);

        // GPU F16 dequant + convert back to F32 — within FP16 rounding (~5e-4 rel).
        nint devDstF16 = _harness.Allocate((long)Elem * sizeof(ushort));
        _harness.Kernels.LaunchDequantToF16(devSrc, qt, devDstF16, Elem, _harness.StreamHandle);
        _harness.Synchronize();
        Half[] gpuF16 = _harness.DownloadHalves(devDstF16, Elem);
        float[] gpuF16AsF32 = new float[Elem];
        for (int i = 0; i < Elem; i++) gpuF16AsF32[i] = (float)gpuF16[i];

        AssertCloseF32($"{qt}-F16", cpuF32, gpuF16AsF32, absTol: 1e-2f, relTol: 5e-3f);
    }

    // ──────────────────── Helpers ────────────────────

    /// <summary>
    /// Asserts every float in <paramref name="actual"/> equals <paramref name="expected"/>
    /// exactly (0 ULP). Used to gate the CPU production dequant against a hand-coded
    /// scalar implementation of the ggml-quants.c reference formula.
    /// </summary>
    private void AssertExactlyEqual(string name, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        int firstMismatch = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
            {
                if (firstMismatch < 0) firstMismatch = i;
            }
        }
        if (firstMismatch >= 0)
        {
            _out.WriteLine($"{name}: first mismatch at index {firstMismatch}: " +
                           $"expected={expected[firstMismatch]:G9}, actual={actual[firstMismatch]:G9}");
            Assert.Equal(expected[firstMismatch], actual[firstMismatch]);
        }
    }

    private void AssertCloseF32(string name, float[] expected, float[] actual,
                                 float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float rel = diff / MathF.Max(MathF.Abs(expected[i]), 1e-30f);
            maxAbs = MathF.Max(maxAbs, diff);
            maxRel = MathF.Max(maxRel, rel);
        }
        _out.WriteLine($"{name}: max-abs-diff={maxAbs:E4}  max-rel-diff={maxRel:E4}  " +
                       $"absTol={absTol:E2}  relTol={relTol:E2}");
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff <= absTol) continue;
            float rel = diff / MathF.Max(MathF.Abs(expected[i]), 1e-30f);
            Assert.True(rel <= relTol,
                $"{name}[{i}]: expected {expected[i]:G6}, got {actual[i]:G6} (diff={diff:E3})");
        }
    }

    // ──────────────────── Scalar reference dequantisers ────────────────────
    // Independent re-implementation of dequantize_row_iq2_{xxs,xs,s} from
    // ggml-quants.c. If these and Dequantize.DequantizeIQ2_* disagree on a
    // single float bit, one of them is wrong.

    private static float[] ReferenceDequantIQ2_XXS(byte[] packed, int superBlocks)
    {
        float[] y = new float[superBlocks * 256];
        ReadOnlySpan<byte> grid = Dequantize.Iq2XxsGrid;
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;
        int o = 0;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            int blockOffset = sb * Dequantize.IQ2_XXS_BlockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf(
                (ushort)(packed[blockOffset] | (packed[blockOffset + 1] << 8)));
            int qsOff = blockOffset + 2;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                uint a0 = ReadU32(packed, qsOff + 8 * ib32);
                uint a1 = ReadU32(packed, qsOff + 8 * ib32 + 4);
                float db = d * (0.5f + (a1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int gridIdx = (int)((a0 >> (8 * l)) & 0xff);
                    int signsIdx = (int)((a1 >> (7 * l)) & 0x7f);
                    int signs = ksigns[signsIdx];
                    for (int j = 0; j < 8; j++)
                    {
                        float sign = (signs & (1 << j)) != 0 ? -1f : 1f;
                        y[o++] = db * grid[gridIdx * 8 + j] * sign;
                    }
                }
            }
        }
        return y;
    }

    private static float[] ReferenceDequantIQ2_XS(byte[] packed, int superBlocks)
    {
        float[] y = new float[superBlocks * 256];
        ReadOnlySpan<byte> grid = Dequantize.Iq2XsGrid;
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;
        int o = 0;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            int blockOffset = sb * Dequantize.IQ2_XS_BlockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf(
                (ushort)(packed[blockOffset] | (packed[blockOffset + 1] << 8)));
            int qsOff = blockOffset + 2;
            int scalesOff = blockOffset + 2 + 64;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (packed[scalesOff + ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (packed[scalesOff + ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int q = packed[qsOff + (ib32 * 4 + l) * 2]
                          | (packed[qsOff + (ib32 * 4 + l) * 2 + 1] << 8);
                    int gridIdx = q & 0x1FF;
                    int signsIdx = q >> 9;
                    int signs = ksigns[signsIdx];
                    float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        float sign = (signs & (1 << j)) != 0 ? -1f : 1f;
                        y[o++] = dl * grid[gridIdx * 8 + j] * sign;
                    }
                }
            }
        }
        return y;
    }

    private static float[] ReferenceDequantIQ2_S(byte[] packed, int superBlocks)
    {
        float[] y = new float[superBlocks * 256];
        ReadOnlySpan<byte> grid = Dequantize.Iq2SGrid;
        int o = 0;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            int blockOffset = sb * Dequantize.IQ2_S_BlockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf(
                (ushort)(packed[blockOffset] | (packed[blockOffset + 1] << 8)));
            int qsLowOff = blockOffset + 2;
            int qsSignsOff = blockOffset + 2 + 32;
            int qhOff = blockOffset + 2 + 64;
            int scalesOff = blockOffset + 2 + 64 + 8;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (packed[scalesOff + ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (packed[scalesOff + ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int lo = packed[qsLowOff + ib32 * 4 + l];
                    int hi = (packed[qhOff + ib32] >> (2 * l)) & 0x3;
                    int gridIdx = lo | (hi << 8);
                    int signs = packed[qsSignsOff + ib32 * 4 + l];
                    float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        float sign = (signs & (1 << j)) != 0 ? -1f : 1f;
                        y[o++] = dl * grid[gridIdx * 8 + j] * sign;
                    }
                }
            }
        }
        return y;
    }

    private static uint ReadU32(byte[] data, int offset) =>
        (uint)data[offset]
        | ((uint)data[offset + 1] << 8)
        | ((uint)data[offset + 2] << 16)
        | ((uint)data[offset + 3] << 24);
}
