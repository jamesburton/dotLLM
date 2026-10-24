using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only check of the integer identity the dp4a PQ2_0 GEMV (issue #485,
/// <c>native/kernels/pq2_0_gemv_dp4a.cu</c>) is built on, emulated instruction for instruction:
/// code-plane words <c>(codeWord &gt;&gt; 2i) &amp; 0x03030303</c> dotted (signed dp4a) against the
/// quantizer's chunk-permuted int8 words, the chain seeded with <c>-Σq</c>, must equal
/// <c>Σ (code − 1)·q</c> over a 32-element block. Runs without a GPU, so a layout mistake in either
/// the permutation or the plane extraction is caught before the PTX exists.
/// </summary>
public sealed class PQ2_0Dp4aLayoutEmulationTests
{
    [Fact]
    public void CodePlanesDotPermutedActivations_EqualsSignedTernaryDot()
    {
        var rng = new Random(485);
        for (int trial = 0; trial < 2000; trial++)
        {
            // One 128-element group's 32 code bytes (all four codes, including 3 -> +2) and the
            // int8 activations of its four 32-element blocks (full range, including -127/127).
            var codeBytes = new byte[32];
            rng.NextBytes(codeBytes);
            var q = new sbyte[128];
            for (int i = 0; i < 128; i++) q[i] = (sbyte)rng.Next(-127, 128);
            if (trial == 0) { Array.Fill(codeBytes, (byte)0xFF); Array.Fill(q, (sbyte)127); }   // max |isum|
            if (trial == 1) { Array.Fill(codeBytes, (byte)0x00); Array.Fill(q, (sbyte)-127); }

            for (int qb = 0; qb < 4; qb++)
            {
                // Direct: element e of the group has code at byte e/4, bits 2*(e%4); value = code - 1.
                int expected = 0, sum = 0;
                for (int i = 0; i < 32; i++)
                {
                    int e = qb * 32 + i;
                    int code = (codeBytes[e / 4] >> (2 * (e % 4))) & 3;
                    expected += (code - 1) * q[e];
                    sum += q[e];
                }

                // Kernel: one 8-byte load of code bytes 8qb..8qb+7 (little-endian uint2).
                uint cwx = BitConverter.ToUInt32(codeBytes, 8 * qb);
                uint cwy = BitConverter.ToUInt32(codeBytes, 8 * qb + 4);
                // Quantizer output for this block: 8 permuted int32 words.
                uint[] xw = PermutedWords(q, qb * 32);

                int isum = -sum;
                for (int i = 0; i < 4; i++)
                {
                    isum = Dp4a((cwx >> (2 * i)) & 0x03030303u, xw[i], isum);
                    isum = Dp4a((cwy >> (2 * i)) & 0x03030303u, xw[4 + i], isum);
                }
                Assert.Equal(expected, isum);
            }
        }
    }

    [Fact]
    public void PermutedToElement_IsTheInverseOfTheQuantizerPermutation()
    {
        var q = new sbyte[64];
        for (int i = 0; i < q.Length; i++) q[i] = (sbyte)i;
        var bytes = new byte[64];
        for (int blk = 0; blk < 2; blk++)
        {
            uint[] w = PermutedWords(q, blk * 32);
            for (int j = 0; j < 8; j++)
                BitConverter.GetBytes(w[j]).CopyTo(bytes, blk * 32 + 4 * j);
        }
        for (int p = 0; p < bytes.Length; p++)
            Assert.Equal(CudaPQ2_0GemvDp4aTests.Dp4aPermutedToElement(p), bytes[p]);
    }

    /// <summary>Mirror of the quantizer's store: chunk c, word i = bytes { q[16c+i], q[16c+4+i], q[16c+8+i], q[16c+12+i] }.</summary>
    private static uint[] PermutedWords(sbyte[] q, int blockStart)
    {
        var w = new uint[8];
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < 4; i++)
            {
                int b = blockStart + 16 * c;
                w[4 * c + i] = (byte)q[b + i]
                             | ((uint)(byte)q[b + 4 + i] << 8)
                             | ((uint)(byte)q[b + 8 + i] << 16)
                             | ((uint)(byte)q[b + 12 + i] << 24);
            }
        return w;
    }

    /// <summary>CUDA <c>__dp4a(int, int, int)</c>: signed byte-wise dot plus accumulator.</summary>
    private static int Dp4a(uint a, uint b, int c)
    {
        for (int i = 0; i < 4; i++)
            c += (sbyte)(a >> (8 * i)) * (sbyte)(b >> (8 * i));
        return c;
    }
}
