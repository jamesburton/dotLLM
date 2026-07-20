using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for the PrismML Bonsai ternary quantization format (GGUF type 42, "PQ2_0").
/// Layout: 128-element groups, each 34 bytes = scale(Half, 2 bytes) + codes[32](uint8,
/// 4 codes/byte, 2 bits each). Codes map 0→-1, 1→0, 2→+1. Unlike I2_S (a single per-tensor
/// scale trailing the whole packed payload), PQ2_0's scale is per-GROUP and comes BEFORE the
/// group's codes — both facts verified empirically against real
/// <c>Ternary-Bonsai-27B-Q2_0.gguf</c> tensor bytes (2026-07-20): decoding with
/// scale-then-codes gives 0% invalid ternary codes and a tight, plausible per-group scale
/// distribution across 200 sampled groups of a real attention-output weight matrix; every
/// other byte-ordering hypothesis (codes-then-scale, or separate contiguous code/scale
/// regions, mirroring I2_S's shape) produces invalid codes and/or wildly inconsistent scales.
/// </summary>
public sealed unsafe class PQ2_0Tests
{
    // ──────────────────── Byte-size accounting ────────────────────

    [Theory]
    [InlineData(128, 34)]       // one group -> 34 bytes (2 scale + 32 packed)
    [InlineData(2560, 680)]     // 20 groups
    [InlineData(6912, 1836)]    // 54 groups
    public void RowByteSize_IncludesPerGroupScale(long elementCount, long expected)
    {
        // Unlike I2_S, the scale is interleaved per-group, so row stride includes it.
        Assert.Equal(expected, Dequantize.RowByteSize(elementCount, QuantizationType.PQ2_0));
    }

    [Theory]
    [InlineData(128, 34)]
    [InlineData(245760, 65280)]       // real blk.0.ssm_alpha.weight element count
    [InlineData(31457280, 8355840)]   // real blk.N.attn_output.weight element count
    public void ComputeByteCount_MatchesRowByteSize_NoSeparateTensorTail(long elementCount, long expected)
    {
        // PQ2_0 has no separate per-tensor tail scale (contrast I2_S's +4) -- every group is
        // self-contained, so whole-tensor size is exactly groupCount * groupBytes.
        Assert.Equal(expected, QuantizationType.PQ2_0.ComputeByteCount(elementCount));
        Assert.Equal(expected, Dequantize.RowByteSize(elementCount, QuantizationType.PQ2_0));
    }

    // ──────────────────── Bit layout (hand-packed) ────────────────────

    /// <summary>
    /// Pins the exact byte layout independent of any packing helper: scale(Half) at offset 0,
    /// codes[32] starting at offset 2. Within the codes, byte at group_pos holds elements
    /// {gp, gp+32, gp+64, gp+96} at bit offsets {6,4,2,0} -- same bit convention as I2_S.
    /// Byte 0x92 = 0b10_01_00_10 -> codes {2,1,0,2} -> ternary {+1,0,-1,+1}.
    /// </summary>
    [Fact]
    public void DequantizePQ2_0_HandPackedGroup_DecodesTernaryTimesGroupScale()
    {
        const int n = 128;
        const float scale = 0.5f;
        byte* buf = (byte*)NativeMemory.AllocZeroed(34);
        try
        {
            *(Half*)buf = (Half)scale;  // scale FIRST
            buf[2] = 0x92;               // then codes; elements 0,32,64,96

            float[] dest = new float[n];
            Dequantize.ToFloat32((nint)buf, n, QuantizationType.PQ2_0, dest);

            Assert.Equal(+scale, dest[0], 1e-3f);  // code 2 -> +1
            Assert.Equal(0f, dest[32], 1e-3f);      // code 1 ->  0
            Assert.Equal(-scale, dest[64], 1e-3f); // code 0 -> -1
            Assert.Equal(+scale, dest[96], 1e-3f); // code 2 -> +1
            Assert.Equal(-scale, dest[1], 1e-3f);  // unset byte -> code 0 -> -1
        }
        finally { NativeMemory.Free(buf); }
    }

    [Fact]
    public void DequantizePQ2_0_TwoGroups_EachUsesItsOwnScale()
    {
        const int n = 256; // two groups
        byte* buf = (byte*)NativeMemory.AllocZeroed(68);
        try
        {
            *(Half*)buf = (Half)0.1f;
            buf[2] = 0xFF; // all codes = 3 (invalid ternary bit pattern, but exercises decode math: (3-1)=2)
            *(Half*)(buf + 34) = (Half)0.2f;
            buf[36] = 0x00; // all codes = 0 -> -1

            float[] dest = new float[n];
            Dequantize.ToFloat32((nint)buf, n, QuantizationType.PQ2_0, dest);

            Assert.Equal(2 * 0.1f, dest[0], 1e-3f);   // group 0's scale applies to group 0
            Assert.Equal(-1 * 0.2f, dest[128], 1e-3f); // group 1's scale applies to group 1, not group 0's
        }
        finally { NativeMemory.Free(buf); }
    }

    [Fact]
    public void DequantizePQ2_0_RoundTripsTestPacker()
    {
        var rng = new Random(11);
        const int n = 256; // two groups
        sbyte[] ternary = new sbyte[n];
        for (int i = 0; i < n; i++) ternary[i] = (sbyte)(rng.Next(3) - 1); // {-1,0,1}
        float[] scales = [0.0123f, 0.0456f];

        byte* buf = PackPQ2_0(ternary, scales);
        try
        {
            float[] dest = new float[n];
            Dequantize.ToFloat32((nint)buf, n, QuantizationType.PQ2_0, dest);
            for (int i = 0; i < n; i++)
            {
                float scale = scales[i / 128];
                Assert.Equal(ternary[i] * scale, dest[i], 1e-3f);
            }
        }
        finally { NativeMemory.Free(buf); }
    }

    [Fact]
    public void DequantizePQ2_0_RejectsNonGroupMultiple()
    {
        byte* buf = (byte*)NativeMemory.AllocZeroed(34);
        try
        {
            Assert.Throws<ArgumentException>(() =>
                Dequantize.ToFloat32((nint)buf, 100, QuantizationType.PQ2_0, new float[100]));
        }
        finally { NativeMemory.Free(buf); }
    }

    // ──────────────────── Ternary GEMV/GEMM ────────────────────

    [Fact]
    public void GemvPQ2_0_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(99);
        const int m = 5;    // output rows
        const int k = 256;  // input dim (2 groups)
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[m * (k / 128)];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.1f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackPQ2_0Rows(ternary, groupScales, m, k);
        try
        {
            float[] y = new float[m];
            fixed (float* xp = x)
            fixed (float* yp = y)
                MatMul.GemvPQ2_0(w, xp, yp, m, k, null);

            int groupsPerRow = k / 128;
            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++)
                {
                    float scale = groupScales[r * groupsPerRow + c / 128];
                    acc += ternary[r * k + c] * scale * x[c];
                }
                Assert.Equal(acc, y[r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    [Fact]
    public void GemmPQ2_0_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(4321);
        const int m = 5;     // output rows (weight rows)
        const int k = 256;   // input dim
        const int n = 3;     // tokens
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[m * (k / 128)];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.1f;

        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackPQ2_0Rows(ternary, groupScales, m, k);
        try
        {
            float[] c = new float[n * m];
            fixed (float* bp = b)
            fixed (float* cp = c)
                MatMul.GemmPQ2_0(w, bp, cp, m, k, n, null);

            int groupsPerRow = k / 128;
            for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int col = 0; col < k; col++)
                {
                    float scale = groupScales[r * groupsPerRow + col / 128];
                    acc += ternary[r * k + col] * scale * b[t * k + col];
                }
                Assert.Equal(acc, c[t * m + r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    // ──────────────────── Test helpers ────────────────────

    /// <summary>Packs an [m,k] ternary weight matrix + one scale per 128-element group per row
    /// into PQ2_0's row-major scale-then-codes layout (row stride <c>(k/128)*34</c> bytes).
    /// Test-only helper for GEMV/GEMM tests, mirroring I2STests' PackI2S row layout.</summary>
    private static byte* PackPQ2_0Rows(sbyte[] ternary, float[] groupScales, int m, int k)
    {
        int groupsPerRow = k / 128;
        Assert.Equal(m * groupsPerRow, groupScales.Length);
        int rowBytes = groupsPerRow * 34;
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(m * rowBytes));

        for (int r = 0; r < m; r++)
        {
            byte* rowBase = buf + r * rowBytes;
            for (int g = 0; g < groupsPerRow; g++)
            {
                byte* groupBase = rowBase + g * 34;
                *(Half*)groupBase = (Half)groupScales[r * groupsPerRow + g];
                byte* codes = groupBase + 2;
                int baseIdx = r * k + g * 128;
                for (int gp = 0; gp < 32; gp++)
                {
                    byte c0 = (byte)(ternary[baseIdx + gp] + 1);
                    byte c1 = (byte)(ternary[baseIdx + gp + 32] + 1);
                    byte c2 = (byte)(ternary[baseIdx + gp + 64] + 1);
                    byte c3 = (byte)(ternary[baseIdx + gp + 96] + 1);
                    codes[gp] = (byte)((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
                }
            }
        }
        return buf;
    }

    /// <summary>Packs ternary {-1,0,1} values + one scale per 128-element group into PQ2_0's
    /// scale-then-codes group layout. Test-only helper (mirrors I2STests' PackI2S).</summary>
    private static byte* PackPQ2_0(sbyte[] ternary, float[] groupScales)
    {
        int n = ternary.Length;
        int groupCount = n / 128;
        Assert.Equal(groupCount, groupScales.Length);
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(groupCount * 34));

        for (int g = 0; g < groupCount; g++)
        {
            byte* groupBase = buf + g * 34;
            *(Half*)groupBase = (Half)groupScales[g];
            byte* codes = groupBase + 2;
            int baseIdx = g * 128;
            for (int gp = 0; gp < 32; gp++)
            {
                byte c0 = (byte)(ternary[baseIdx + gp] + 1);
                byte c1 = (byte)(ternary[baseIdx + gp + 32] + 1);
                byte c2 = (byte)(ternary[baseIdx + gp + 64] + 1);
                byte c3 = (byte)(ternary[baseIdx + gp + 96] + 1);
                codes[gp] = (byte)((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
            }
        }
        return buf;
    }
}
