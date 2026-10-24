using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Element-layout tests for the CUDA dequant kernels added by issue #258:
/// BF16, MXFP4, IQ3_XXS, IQ3_S and IQ1_S.
///
/// Every fixture is built so that the plausible *wrong* layouts produce a
/// different element vector from the correct one — the failure mode that shipped
/// silently in the Q4_0/Q4_1 kernels (issue #254, commit 5d724b8d), where an
/// interleaved nibble unpack ran to completion and emitted tokens while
/// permuting every weight in the block.
///
/// Expectations are computed inline from the formulas in llama.cpp
/// <c>ggml/src/ggml-quants.c</c> (<c>dequantize_row_{iq3_xxs,iq3_s,iq1_s,mxfp4}</c>),
/// not by calling the CUDA kernel's C# sibling. The shared codebook tables are
/// taken from <see cref="Dequantize"/> because those are <c>ggml-common.h</c>
/// verbatim — the tables are inputs to the test, the index/scale/sign arithmetic
/// is what is under test.
///
/// Each test also cross-checks the CUDA output against the CPU scalar
/// dequantizer on the same bytes, which catches any disagreement the
/// hand-written expectation might share with the kernel.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaDequantIssue258LayoutTests
{
    private const int SuperBlock = 256;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the test class with the xUnit output helper.</summary>
    /// <param name="output">Sink for diagnostic output.</param>
    public CudaDequantIssue258LayoutTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// BF16 widening is <c>f32_bits = bf16_bits &lt;&lt; 16</c>. The fixture uses
    /// values whose two bytes differ, so a byte-swapped read (or a read that
    /// treated the payload as FP16) lands on a completely different float.
    /// </summary>
    [SkippableFact]
    public void Bf16_Dequant_ShiftsIntoTheHighHalfOfTheF32()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        // Chosen so every bf16 pattern has hi != lo byte and every value is exactly
        // representable in FP16 (the GPU dequant target) — 1.0, -2.5, 0.5, ...
        float[] values = [1.0f, -2.5f, 0.5f, -0.125f, 96.0f, -3.75f, 0.375f, -1.5f];
        const int Count = 64;

        byte[] packed = new byte[Count * 2];
        float[] expected = new float[Count];
        for (int i = 0; i < Count; i++)
        {
            float v = values[i % values.Length];
            // Truncating F32 -> BF16 (drop the low 16 bits). All the values above
            // have zero low mantissa bits, so the truncation is lossless and the
            // expectation is exact.
            uint bits = BitConverter.SingleToUInt32Bits(v);
            ushort bf = (ushort)(bits >> 16);
            Assert.NotEqual(bf & 0xFF, bf >> 8); // Fixture must be byte-order discriminating.
            packed[i * 2] = (byte)(bf & 0xFF);
            packed[i * 2 + 1] = (byte)(bf >> 8);
            expected[i] = BitConverter.UInt32BitsToSingle((uint)bf << 16);
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.BF16, Count);
        AssertMatches(expected, actual, "BF16", 1e-3f);
        AssertMatchesCpu(packed, QuantizationType.BF16, Count, actual, "BF16");
    }

    /// <summary>
    /// MXFP4 packs 32 elements into 16 bytes as two halves: element <c>j</c> is the
    /// low nibble of <c>qs[j]</c> and element <c>j + 16</c> is the high nibble —
    /// the exact convention Q4_0 got wrong in #254. Each byte here gets a low
    /// nibble that differs from its high nibble, so an interleaved
    /// (<c>out[2j]</c>/<c>out[2j+1]</c>) unpack cannot coincidentally agree.
    /// </summary>
    [SkippableFact]
    public void Mxfp4_Dequant_UsesTwoHalvesNibbleOrderAndE8m0HalfScale()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 17;
        const int BlockCount = 5;
        const int Count = 32 * BlockCount;

        ReadOnlySpan<sbyte> kvalues = Dequantize.Mxfp4Values;
        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[Count];

        for (int b = 0; b < BlockCount; b++)
        {
            int at = b * BlockBytes;
            // E8M0 exponent: 127 -> 0.5, 128 -> 1.0, ... Distinct per block so a
            // kernel that hoisted one block's scale across the tile would fail.
            byte e = (byte)(125 + b);
            packed[at] = e;
            float d = BitConverter.UInt32BitsToSingle((uint)(e - 1) << 23);

            for (int j = 0; j < 16; j++)
            {
                int lo = (j + b) & 0x0F;
                int hi = (j + b + 7) & 0x0F; // Never equal to lo.
                Assert.NotEqual(lo, hi);
                packed[at + 1 + j] = (byte)((hi << 4) | lo);

                expected[b * 32 + j] = kvalues[lo] * d;
                expected[b * 32 + j + 16] = kvalues[hi] * d;
            }
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.MXFP4, Count);
        AssertMatches(expected, actual, "MXFP4", 1e-3f);
        AssertMatchesCpu(packed, QuantizationType.MXFP4, Count, actual, "MXFP4");
    }

    /// <summary>
    /// IQ3_XXS: 8 elements per grid pair, laid out as <c>grid1[0..3]</c> then
    /// <c>grid2[0..3]</c>, with sign bits 0..3 applying to the first four and 4..7
    /// to the second four. The fixture gives every sub-block a different 4-bit
    /// scale, every pair a different 7-bit sign code, and every codeword a
    /// different grid row, so swapping the two grid rows, interleaving the eight
    /// outputs, or reusing one sign nibble all produce a different vector.
    /// </summary>
    [SkippableFact]
    public void Iq3Xxs_Dequant_MatchesGgmlGridSignAndScaleLayout()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 98;
        const int BlockCount = 3;
        const int Count = SuperBlock * BlockCount;

        ReadOnlySpan<byte> grid = Dequantize.Iq3XxsGrid;
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;

        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[Count];
        var rng = new Random(258);

        for (int b = 0; b < BlockCount; b++)
        {
            int at = b * BlockBytes;
            float d = 0.25f * (b + 1); // Exactly representable in FP16.
            BitConverter.GetBytes((Half)d).CopyTo(packed, at);

            // qs[64]: 8 sub-blocks x 8 codewords, all distinct grid rows.
            for (int i = 0; i < 64; i++)
                packed[at + 2 + i] = (byte)((i * 7 + b * 13 + 1) & 0xFF);

            // scales_and_signs[32]: per sub-block uint32 = 4x7-bit sign code + 4-bit scale.
            uint[] aux = new uint[8];
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                uint a = 0;
                for (int l = 0; l < 4; l++)
                {
                    // Sign codes with differing low/high nibbles of the resulting
                    // mask, so a kernel that applied one nibble to both halves fails.
                    uint code = (uint)((ib32 * 17 + l * 29 + b * 5 + 1) & 0x7F);
                    a |= code << (7 * l);
                }
                a |= (uint)(ib32 % 8) << 28; // Distinct 4-bit sub-scale per sub-block.
                aux[ib32] = a;
                BitConverter.GetBytes(a).CopyTo(packed, at + 2 + 64 + 4 * ib32);
            }

            // Expectation transcribed from dequantize_row_iq3_xxs in ggml-quants.c.
            int o = b * SuperBlock;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                uint aux32 = aux[ib32];
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++)
                {
                    byte signs = ksigns[(int)((aux32 >> (7 * l)) & 127)];
                    int g1 = packed[at + 2 + ib32 * 8 + 2 * l + 0] * 4;
                    int g2 = packed[at + 2 + ib32 * 8 + 2 * l + 1] * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        expected[o + j + 0] = db * grid[g1 + j] * ((signs & (1 << j)) != 0 ? -1f : 1f);
                        expected[o + j + 4] = db * grid[g2 + j] * ((signs & (1 << (j + 4))) != 0 ? -1f : 1f);
                    }
                    o += 8;
                }
            }

            _ = rng; // Deterministic fixture; rng retained for future variation.
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.IQ3_XXS, Count);
        AssertMatches(expected, actual, "IQ3_XXS", 5e-2f);
        AssertMatchesCpu(packed, QuantizationType.IQ3_XXS, Count, actual, "IQ3_XXS");
    }

    /// <summary>
    /// IQ3_S adds a 9th grid-index bit from <c>qh</c> and a paired scale byte
    /// (low nibble = even sub-block, high nibble = odd). The fixture sets every
    /// <c>qh</c> byte to a pattern with distinct bits so that dropping the high
    /// bit, using the wrong shift, or swapping the scale nibbles all change the
    /// output — and the codewords are chosen so grid indices land both below and
    /// above 256.
    /// </summary>
    [SkippableFact]
    public void Iq3S_Dequant_MatchesGgmlHighBitAndPairedScaleLayout()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 110;
        const int BlockCount = 3;
        const int Count = SuperBlock * BlockCount;

        ReadOnlySpan<byte> grid = Dequantize.Iq3SGrid;

        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[Count];

        for (int b = 0; b < BlockCount; b++)
        {
            int at = b * BlockBytes;
            float d = 0.125f * (b + 1);
            BitConverter.GetBytes((Half)d).CopyTo(packed, at);

            for (int i = 0; i < 64; i++)
                packed[at + 2 + i] = (byte)((i * 11 + b * 23 + 3) & 0xFF);   // qs
            for (int i = 0; i < 8; i++)
                packed[at + 66 + i] = (byte)(0x5A ^ (i * 0x33) ^ (b * 0x0F)); // qh: mixed bits
            for (int i = 0; i < 32; i++)
                packed[at + 74 + i] = (byte)((i * 19 + b * 7 + 1) & 0xFF);    // signs
            for (int i = 0; i < 4; i++)
                packed[at + 106 + i] = (byte)(((i * 3 + 1) & 0xF) | (((i * 5 + 2) & 0xF) << 4));

            // Scale nibbles must differ within a byte or the pairing is untestable.
            for (int i = 0; i < 4; i++)
                Assert.NotEqual(packed[at + 106 + i] & 0xF, packed[at + 106 + i] >> 4);

            // Expectation transcribed from dequantize_row_iq3_s in ggml-quants.c.
            int o = b * SuperBlock;
            int qsOff = 0, signsOff = 0, qhOff = 0;
            for (int ib32 = 0; ib32 < 8; ib32 += 2)
            {
                byte sc = packed[at + 106 + ib32 / 2];
                float db1 = d * (1 + 2 * (sc & 0xF));
                float db2 = d * (1 + 2 * (sc >> 4));

                for (int half = 0; half < 2; half++)
                {
                    float db = half == 0 ? db1 : db2;
                    int qhByte = packed[at + 66 + qhOff + half];
                    for (int l = 0; l < 4; l++)
                    {
                        int g1 = (packed[at + 2 + qsOff + 2 * l + 0] | ((qhByte << (8 - 2 * l)) & 256)) * 4;
                        int g2 = (packed[at + 2 + qsOff + 2 * l + 1] | ((qhByte << (7 - 2 * l)) & 256)) * 4;
                        byte signs = packed[at + 74 + signsOff + l];
                        for (int j = 0; j < 4; j++)
                        {
                            expected[o + j + 0] = db * grid[g1 + j] * ((signs & (1 << j)) != 0 ? -1f : 1f);
                            expected[o + j + 4] = db * grid[g2 + j] * ((signs & (1 << (j + 4))) != 0 ? -1f : 1f);
                        }
                        o += 8;
                    }
                    qsOff += 8;
                    signsOff += 4;
                }
                qhOff += 2;
            }
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.IQ3_S, Count);
        AssertMatches(expected, actual, "IQ3_S", 5e-2f);
        AssertMatchesCpu(packed, QuantizationType.IQ3_S, Count, actual, "IQ3_S");
    }

    /// <summary>
    /// IQ1_S unpacks 8 signed grid points from a uint64 codebook entry in element
    /// order (byte <c>j</c> of the entry is element <c>j</c> — no half-splitting),
    /// scaled by a per-sub-block <c>dl</c> and offset by a per-sub-block
    /// <c>±IQ1S_DELTA</c>. The fixture varies the 3-bit sub-scale, both delta
    /// signs, and all four 3-bit high-index fields, so a kernel that dropped the
    /// high bits, reversed the byte order within an entry, or hoisted one delta
    /// across the super-block produces a different vector.
    /// </summary>
    [SkippableFact]
    public void Iq1S_Dequant_MatchesGgmlCodebookDeltaAndSubScaleLayout()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 50;
        const int BlockCount = 3;
        const int Count = SuperBlock * BlockCount;
        const float Delta = 0.125f;

        ReadOnlySpan<ulong> grid = Dequantize.Iq1SGrid;

        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[Count];

        for (int b = 0; b < BlockCount; b++)
        {
            int at = b * BlockBytes;
            float d = 0.5f * (b + 1);
            BitConverter.GetBytes((Half)d).CopyTo(packed, at);

            for (int i = 0; i < 32; i++)
                packed[at + 2 + i] = (byte)((i * 13 + b * 29 + 5) & 0xFF); // qs

            ushort[] qh = new ushort[8];
            for (int ib = 0; ib < 8; ib++)
            {
                // Four distinct 3-bit high-index fields (bits 0..11), a distinct
                // 3-bit sub-scale (bits 12..14), and alternating delta sign (bit 15).
                ushort v = 0;
                for (int l = 0; l < 4; l++)
                    v |= (ushort)((uint)((ib + l + b) & 7) << (3 * l));
                v |= (ushort)((uint)((ib + b) & 7) << 12);
                if (((ib + b) & 1) != 0) v |= 0x8000;
                qh[ib] = v;
                BitConverter.GetBytes(v).CopyTo(packed, at + 34 + 2 * ib);
            }

            // Expectation transcribed from dequantize_row_iq1_s in ggml-quants.c.
            int o = b * SuperBlock;
            for (int ib = 0; ib < 8; ib++)
            {
                float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
                float delta = (qh[ib] & 0x8000) != 0 ? -Delta : Delta;
                for (int l = 0; l < 4; l++)
                {
                    int idx = packed[at + 2 + ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                    ulong entry = grid[idx];
                    for (int j = 0; j < 8; j++)
                    {
                        sbyte g = (sbyte)((entry >> (8 * j)) & 0xFF);
                        expected[o + j] = dl * (g + delta);
                    }
                    o += 8;
                }
            }
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.IQ1_S, Count);
        AssertMatches(expected, actual, "IQ1_S", 5e-3f);
        AssertMatchesCpu(packed, QuantizationType.IQ1_S, Count, actual, "IQ1_S");
    }

    private static float[] DequantOnGpu(byte[] packed, QuantizationType quantType, int elementCount)
    {
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dSrc = 0;
        nint dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)packed.Length).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();

            fixed (byte* p = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)packed.Length).ThrowOnError();

            kernels.LaunchDequantToF16(dSrc, quantType, dDst, elementCount, stream.Handle);
            stream.Synchronize();

            ushort[] f16 = new ushort[elementCount];
            fixed (ushort* p = f16)
            {
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(elementCount * sizeof(ushort)))
                    .ThrowOnError();
            }

            float[] result = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
                result[i] = (float)BitConverter.UInt16BitsToHalf(f16[i]);

            return result;
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }
    }

    /// <summary>
    /// Cross-checks the CUDA output against the CPU scalar dequantizer on the same
    /// bytes. The CPU path is FP32 throughout, so the tolerance absorbs the GPU's
    /// FP16 output rounding (relative, with an absolute floor for near-zero values).
    /// </summary>
    private void AssertMatchesCpu(byte[] packed, QuantizationType quantType, int elementCount,
                                   float[] gpu, string label)
    {
        float[] cpu = new float[elementCount];
        nint hostPtr = System.Runtime.InteropServices.Marshal.AllocHGlobal(packed.Length);
        try
        {
            System.Runtime.InteropServices.Marshal.Copy(packed, 0, hostPtr, packed.Length);
            Dequantize.ToFloat32(hostPtr, elementCount, quantType, cpu);
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(hostPtr);
        }

        int mismatches = 0;
        int first = -1;
        for (int i = 0; i < elementCount; i++)
        {
            float tol = MathF.Max(1e-4f, MathF.Abs(cpu[i]) * 1e-2f);
            if (MathF.Abs(cpu[i] - gpu[i]) > tol)
            {
                mismatches++;
                if (first < 0) first = i;
            }
        }

        _output.WriteLine($"{label} vs CPU reference: mismatches={mismatches}/{elementCount}");
        Assert.True(mismatches == 0,
            $"{label} CUDA dequant disagrees with the CPU scalar reference on {mismatches} of "
            + $"{elementCount} elements"
            + (first >= 0 ? $" (first at {first}: cpu={cpu[first]}, gpu={gpu[first]})" : string.Empty));
    }

    private void AssertMatches(float[] expected, float[] actual, string label, float tolerance)
    {
        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = MathF.Max(tolerance, MathF.Abs(expected[i]) * 1e-2f);
            if (MathF.Abs(expected[i] - actual[i]) > tol)
            {
                mismatches++;
                if (firstMismatch < 0) firstMismatch = i;
            }
        }

        _output.WriteLine($"{label} vs ggml spec: mismatches={mismatches}/{expected.Length}");
        if (firstMismatch >= 0)
        {
            _output.WriteLine(
                $"  first mismatch at {firstMismatch}: expected={expected[firstMismatch]:F6}, "
                + $"actual={actual[firstMismatch]:F6}");
        }

        Assert.True(
            mismatches == 0,
            $"{label} dequant layout wrong: {mismatches} of {expected.Length} elements differ "
            + (firstMismatch >= 0
                ? $"(first at index {firstMismatch}: expected {expected[firstMismatch]}, got {actual[firstMismatch]})"
                : string.Empty));
    }

    private static string ResolvePtxDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("DOTLLM_PTX_DIR");
        if (envDir is not null && Directory.Exists(envDir))
            return envDir;

        string repoRoot = Path.GetDirectoryName(typeof(CudaDequantIssue258LayoutTests).Assembly.Location)!;
        while (repoRoot.Length > 3 && !File.Exists(Path.Combine(repoRoot, "dotLLM.slnx")))
            repoRoot = Path.GetDirectoryName(repoRoot)!;

        return Path.Combine(repoRoot, "native", "ptx");
    }
}
