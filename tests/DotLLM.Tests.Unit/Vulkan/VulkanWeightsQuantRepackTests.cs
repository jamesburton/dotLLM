using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// CPU-only exactness tests for the Gemma-4 down-expert bank repacks introduced for the
/// real 26B GGUF (whose <c>ffn_down_exps</c> are Q5_0/Q8_0, not the synthetic fixture's
/// Q5_1): <see cref="VulkanWeights.ConvertQ5_0BlocksToQ5_1"/> must be bit-exact
/// (<c>d·(q−16) = d·q + m</c> with <c>m = −16·d</c> exactly representable in fp16), and
/// <see cref="VulkanWeights.CopyQ8_0BlocksScaled"/> must fold the per-expert scale into
/// each block's fp16 <c>d</c> within one fp16 rounding while leaving the int8 payload
/// untouched. No GPU required.
/// </summary>
public sealed class VulkanWeightsQuantRepackTests
{
    private const int GroupSize = 32;
    private const int Q5_0BlockBytes = 22;
    private const int Q5_1BlockBytes = 24;
    private const int Q8_0BlockBytes = 34;

    [Fact]
    public unsafe void ConvertQ5_0BlocksToQ5_1_DequantisesBitExact()
    {
        const int blocks = 257; // odd count, > several rows
        const int elems = blocks * GroupSize;
        var rng = new Random(40); // fixed seed — deterministic test

        byte[] src = new byte[blocks * Q5_0BlockBytes];
        rng.NextBytes(src);
        for (int b = 0; b < blocks; b++)
        {
            // Realistic per-block scale: weights' d is small; include negatives, zero
            // and a subnormal-half to cover the exponent-shift edge cases.
            float d = b switch
            {
                0 => 0f,
                1 => -0f,
                2 => 5.96e-8f, // subnormal half
                3 => -5.96e-8f,
                _ => (rng.NextSingle() - 0.5f) * 0.25f,
            };
            MemoryMarshal.Write(src.AsSpan(b * Q5_0BlockBytes), (Half)d);
        }

        byte[] dst = new byte[blocks * Q5_1BlockBytes];
        float[] fromQ5_0 = new float[elems];
        float[] fromQ5_1 = new float[elems];
        fixed (byte* sp = src, dp = dst)
        {
            VulkanWeights.ConvertQ5_0BlocksToQ5_1((nint)sp, (nint)dp, blocks);
            Dequantize.ToFloat32((nint)sp, elems, QuantizationType.Q5_0, fromQ5_0);
            Dequantize.ToFloat32((nint)dp, elems, QuantizationType.Q5_1, fromQ5_1);
        }

        // Bit-exact: d·q and 16·d are both exact in F32 (11-bit × 5-bit significands),
        // so d·q − 16d and d·(q−16) round identically.
        for (int i = 0; i < elems; i++)
            Assert.True(fromQ5_0[i].Equals(fromQ5_1[i]),
                $"elem {i}: Q5_0={fromQ5_0[i]:G9} != converted Q5_1={fromQ5_1[i]:G9}");
    }

    [Fact]
    public unsafe void CopyQ8_0BlocksScaled_FoldsScaleWithinOneHalfRounding()
    {
        const int blocks = 129;
        const int elems = blocks * GroupSize;
        const float scale = 0.03127f; // deliberately not a power of two
        var rng = new Random(41);

        byte[] src = new byte[blocks * Q8_0BlockBytes];
        rng.NextBytes(src);
        for (int b = 0; b < blocks; b++)
            MemoryMarshal.Write(src.AsSpan(b * Q8_0BlockBytes), (Half)((rng.NextSingle() - 0.5f) * 0.25f));

        byte[] dst = new byte[blocks * Q8_0BlockBytes];
        float[] original = new float[elems];
        float[] scaled = new float[elems];
        fixed (byte* sp = src, dp = dst)
        {
            VulkanWeights.CopyQ8_0BlocksScaled((nint)sp, (nint)dp, blocks, scale);
            Dequantize.ToFloat32((nint)sp, elems, QuantizationType.Q8_0, original);
            Dequantize.ToFloat32((nint)dp, elems, QuantizationType.Q8_0, scaled);
        }

        for (int b = 0; b < blocks; b++)
        {
            // Payload untouched.
            Assert.True(src.AsSpan(b * Q8_0BlockBytes + 2, 32).SequenceEqual(
                            dst.AsSpan(b * Q8_0BlockBytes + 2, 32)),
                $"block {b}: int8 payload changed");
            // d' == fp16(d · scale) exactly — the only rounding is the fold itself.
            Half d = MemoryMarshal.Read<Half>(src.AsSpan(b * Q8_0BlockBytes));
            Half expected = (Half)((float)d * scale);
            Half actual = MemoryMarshal.Read<Half>(dst.AsSpan(b * Q8_0BlockBytes));
            Assert.Equal(expected, actual);
        }

        // And the dequantised values are exactly fp16(d·scale) · q — an fp16 (11-bit)
        // by int8 (8-bit) product is exact in F32, so equality is bitwise.
        for (int i = 0; i < elems; i++)
        {
            int b = i / GroupSize;
            Half foldedD = MemoryMarshal.Read<Half>(dst.AsSpan(b * Q8_0BlockBytes));
            int q = (sbyte)src[b * Q8_0BlockBytes + 2 + (i % GroupSize)];
            Assert.True(((float)foldedD * q).Equals(scaled[i]),
                $"elem {i}: expected {(float)foldedD * q:G9}, got {scaled[i]:G9}");
        }
    }
}
