using System.Runtime.CompilerServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Anchors <see cref="Dequantize.DequantizeQ5_0Scalar"/> to a literal transcription of
/// llama.cpp's <c>dequantize_row_q5_0</c> over dense random blocks.
/// <para>
/// Structurally unlike the production indexing on purpose: the reference below keeps
/// llama.cpp's own control flow and its <c>xh_0 / xh_1</c> shift trick rather than
/// re-deriving bit positions. Q5_0 hides the 5th bit of each weight in a separate
/// <c>qh</c> word, which is the same class of indexing that Q3_K got transposed.
/// </para>
/// </summary>
public unsafe class DequantizeQ5_0AnchorTests
{
    private const int Qk = 32;
    private const int BlockBytes = 22;

    [Fact]
    public void Q5_0_DenseRandomBlocks_MatchLlamaCppReference()
    {
        const int blocks = 64;
        var rng = new Random(0x5C0DE);
        var raw = new byte[blocks * BlockBytes];
        rng.NextBytes(raw);
        // Keep d in a sane finite range: overwrite each block's fp16 delta.
        for (int b = 0; b < blocks; b++)
        {
            Half d = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            Unsafe.WriteUnaligned(ref raw[b * BlockBytes], d);
        }

        var actual = new float[blocks * Qk];
        var expected = new float[blocks * Qk];
        fixed (byte* p = raw)
        {
            Dequantize.DequantizeQ5_0Scalar((nint)p, blocks * Qk, actual);
            LlamaCppDequantizeRowQ5_0(p, blocks, expected);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i],
                $"element {i}: llama.cpp={expected[i]:R} dotLLM={actual[i]:R}");
    }

    /// <summary>
    /// Literal transcription of llama.cpp <c>ggml-quants.c dequantize_row_q5_0</c>,
    /// kept in its original control-flow shape on purpose.
    /// </summary>
    private static void LlamaCppDequantizeRowQ5_0(byte* x, int nb, Span<float> y)
    {
        for (int i = 0; i < nb; i++)
        {
            byte* blk = x + i * BlockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(blk);
            uint qh = Unsafe.ReadUnaligned<uint>(blk + 2);
            byte* qs = blk + 6;

            for (int j = 0; j < Qk / 2; j++)
            {
                byte xh_0 = (byte)(((qh >> (j + 0)) << 4) & 0x10);
                byte xh_1 = (byte)((qh >> (j + 12)) & 0x10);

                int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
                int x1 = ((qs[j] >> 4) | xh_1) - 16;

                y[i * Qk + j + 0] = x0 * d;
                y[i * Qk + j + Qk / 2] = x1 * d;
            }
        }
    }
}
