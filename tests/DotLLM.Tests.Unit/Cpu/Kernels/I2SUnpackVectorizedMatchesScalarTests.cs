using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Discriminating regression test for issue #128 (AVX2-vectorized <c>UnpackRowI8</c>). Compares
/// the production unpack (<see cref="MatMul.UnpackRowI8Public"/>, which runs the AVX2 fast path on
/// this box) against an independent, hand-rolled scalar reference of the documented bit layout,
/// across many random seeds and row sizes — including a single-block edge case and multi-block
/// rows that don't share block boundaries with AVX2's 32-byte processing granularity in any special
/// way (128-element I2_S blocks are always exactly 32 packed bytes = one Vector256&lt;byte&gt;, so
/// there is no partial-vector remainder to exercise, but this still catches a wrong shuffle/lane
/// order or misplaced shift constant, which a hand-picked easy case would not).
/// </summary>
public sealed unsafe class I2SUnpackVectorizedMatchesScalarTests
{
    /// <summary>
    /// Correctness coverage for <see cref="MatMul.BenchStreamingReadOnly"/> (the issue #196
    /// decode-bandwidth-profiling read-only streaming probe): its XOR checksum over every packed
    /// byte must match an independent scalar XOR reduction, across both the AVX2 vectorized
    /// accumulation path and the scalar tail.
    /// </summary>
    [Fact]
    public void BenchStreamingReadOnly_MatchesScalarChecksum()
    {
        var rng = new Random(7);
        const int m = 64, k = 512;
        int rowBytes = k / 4;
        byte[] buf = new byte[m * rowBytes];
        rng.NextBytes(buf);

        byte expected = 0;
        foreach (byte b in buf) expected ^= b;

        fixed (byte* p = buf)
        {
            byte actual = MatMul.BenchStreamingReadOnly(p, m, k);
            Assert.Equal(expected, actual);
        }
    }

    [Theory]
    [InlineData(128, 1)]     // single block
    [InlineData(128, 7)]
    [InlineData(256, 3)]     // two blocks
    [InlineData(384, 13)]    // three blocks
    [InlineData(2560, 11)]   // BitNet attn row (hidden dim)
    [InlineData(2560, 17)]
    [InlineData(6912, 23)]
    public void VectorizedUnpack_MatchesScalarReference_AcrossManySeeds(int k, int seedBase)
    {
        for (int trial = 0; trial < 25; trial++)
        {
            int seed = seedBase * 1000 + trial;
            var rng = new Random(seed);

            int rowBytes = k / 4;
            byte[] packed = new byte[rowBytes];
            rng.NextBytes(packed); // includes code value 3 (unused/invalid) — must still match scalar exactly

            byte* rowPtr = (byte*)NativeMemory.Alloc((nuint)rowBytes);
            sbyte* vecOut = (sbyte*)NativeMemory.Alloc((nuint)k);
            try
            {
                Marshal.Copy(packed, 0, (nint)rowPtr, rowBytes);

                MatMul.UnpackRowI8Public(rowPtr, vecOut, k);

                sbyte[] expected = ScalarReferenceUnpack(packed, k);

                for (int i = 0; i < k; i++)
                {
                    Assert.True(expected[i] == vecOut[i],
                        $"seed={seed} k={k} mismatch at index {i}: expected {expected[i]}, got {vecOut[i]}");
                }
            }
            finally
            {
                NativeMemory.Free(rowPtr);
                NativeMemory.Free(vecOut);
            }
        }
    }

    /// <summary>
    /// Independent scalar re-implementation of the documented I2_S bit layout (not calling into
    /// <c>MatMul</c> at all): within a 128-element block, byte at <c>gp</c> holds elements
    /// {gp, gp+32, gp+64, gp+96} at bit offsets {6,4,2,0}; ternary value = <c>code - 1</c>.
    /// </summary>
    private static sbyte[] ScalarReferenceUnpack(byte[] packed, int k)
    {
        const int blockSize = 128;
        sbyte[] dest = new sbyte[k];
        int blocks = k / blockSize;
        for (int blk = 0; blk < blocks; blk++)
        {
            int bpBase = blk * 32;
            int outBase = blk * blockSize;
            for (int gp = 0; gp < 32; gp++)
            {
                byte b = packed[bpBase + gp];
                dest[outBase + gp] = (sbyte)(((b >> 6) & 0x3) - 1);
                dest[outBase + gp + 32] = (sbyte)(((b >> 4) & 0x3) - 1);
                dest[outBase + gp + 64] = (sbyte)(((b >> 2) & 0x3) - 1);
                dest[outBase + gp + 96] = (sbyte)((b & 0x3) - 1);
            }
        }
        return dest;
    }
}
