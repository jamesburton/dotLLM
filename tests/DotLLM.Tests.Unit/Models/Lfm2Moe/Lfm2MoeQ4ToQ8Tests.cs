using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

/// <summary>
/// #1015: the Vulkan LFM2-MoE loader serves Q4_0 tensors as Q8_0. The transcode must reproduce the dequantised F32 weights EXACTLY
/// (not approximately): a Q4_0 value is <c>(nibble - 8) * d</c>, an int8 times the same f16 scale.
/// </summary>
public sealed unsafe class Lfm2MoeQ4ToQ8Tests
{
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(20000)]   // spans several parallel chunks
    public void Transcode_DequantisesBitIdenticallyToQ4_0(int blocks)
    {
        var rng = new Random(1015 + blocks);
        byte* q4 = (byte*)NativeMemory.Alloc((nuint)(blocks * 18));
        byte* q8 = (byte*)NativeMemory.Alloc((nuint)(blocks * 34));
        try
        {
            for (int b = 0; b < blocks; b++)
            {
                Half d = (Half)(float)((rng.NextDouble() * 2 - 1) * 0.1);
                *(Half*)(q4 + b * 18) = d;
                for (int j = 0; j < 16; j++) q4[b * 18 + 2 + j] = (byte)rng.Next(256);
            }
            Q4_0ToQ8_0.Transcode((nint)q4, (nint)q8, blocks);

            var a = new float[blocks * 32];
            var c = new float[blocks * 32];
            Dequantize.ToFloat32((nint)q4, blocks * 32, QuantizationType.Q4_0, a);
            Dequantize.ToFloat32((nint)q8, blocks * 32, QuantizationType.Q8_0, c);
            for (int i = 0; i < a.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(c[i]), $"element {i}: Q4_0 {a[i]:G9} != transcoded Q8_0 {c[i]:G9}");
            Assert.Equal(blocks * 34L, Q4_0ToQ8_0.Q8Bytes(blocks * 18L));
        }
        finally
        {
            NativeMemory.Free(q4);
            NativeMemory.Free(q8);
        }
    }
}
