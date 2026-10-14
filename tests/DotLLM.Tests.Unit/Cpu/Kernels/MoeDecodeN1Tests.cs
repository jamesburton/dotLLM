using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// The single-token routed-expert decode path (two pool dispatches per layer over the flattened expert x row space) must be
/// bit-identical to the per-expert grouped path it replaces: it feeds the same <c>ComputeRows</c> kernels the same Q8 activations.
/// The pool-less call is the independent baseline (serial <c>ProcessRoutedExpert</c> loop, GEMM n=1).
/// </summary>
public sealed unsafe class MoeDecodeN1Tests
{
    private const int NumExperts = 12;
    private const int TopK = 5;
    private const int Hidden = 512;
    private const int Inter = 256;

    public static TheoryData<QuantizationType, QuantizationType, QuantizationType> Cases() => new()
    {
        { QuantizationType.IQ1_M, QuantizationType.IQ1_M, QuantizationType.IQ2_XXS },
        { QuantizationType.IQ2_XXS, QuantizationType.IQ2_XXS, QuantizationType.IQ2_XS },
        { QuantizationType.IQ2_XS, QuantizationType.IQ2_XS, QuantizationType.IQ3_S },
        { QuantizationType.Q4_K, QuantizationType.Q4_K, QuantizationType.Q6_K },
        { QuantizationType.Q5_K, QuantizationType.Q5_K, QuantizationType.Q4_K },
        { QuantizationType.Q2_K, QuantizationType.Q2_K, QuantizationType.Q3_K },
        { QuantizationType.Q8_0, QuantizationType.Q8_0, QuantizationType.Q8_0 },
        { QuantizationType.Q4_K, QuantizationType.Q4_K, QuantizationType.Q8_0 },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void DecodeFastPath_IsBitIdenticalToGroupedPath(QuantizationType gateQt, QuantizationType upQt, QuantizationType downQt)
    {
        var rng = new Random(9031 + (int)gateQt * 7 + (int)downQt);
        long gateExp = Dequantize.RowByteSize((long)Inter * Hidden, gateQt);
        long upExp = Dequantize.RowByteSize((long)Inter * Hidden, upQt);
        long downExp = Dequantize.RowByteSize((long)Hidden * Inter, downQt);
        byte* gate = Bank(rng, gateQt, gateExp * NumExperts);
        byte* up = Bank(rng, upQt, upExp * NumExperts);
        byte* down = Bank(rng, downQt, downExp * NumExperts);
        float[] router = new float[NumExperts * Hidden];
        for (int i = 0; i < router.Length; i++) router[i] = rng.NextSingle() * 2f - 1f;
        using var pool = new ComputeThreadPool(5);
        pool.SetDispatchMode(DispatchMode.SpinWait);
        long callsBefore = MoeSwiGluMlp.DecodeN1Calls;
        try
        {
            for (int trial = 0; trial < 6; trial++)
            {
                float[] x = new float[Hidden];
                for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
                float[] expected = Run(x, router, gate, gateQt, gateExp, up, upQt, upExp, down, downQt, downExp, null);
                float[] actual = Run(x, router, gate, gateQt, gateExp, up, upQt, upExp, down, downQt, downExp, pool);
                for (int i = 0; i < Hidden; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"{gateQt}/{upQt}/{downQt} trial {trial} elem {i}: grouped {expected[i]} vs decode {actual[i]}");
            }
        }
        finally
        {
            // Discriminates "bit-identical because it IS the grouped path" from "bit-identical because the fast path ran".
            Assert.True(MoeSwiGluMlp.DecodeN1Calls - callsBefore >= 6, "decode fast path did not run");
            NativeMemory.AlignedFree(gate); NativeMemory.AlignedFree(up); NativeMemory.AlignedFree(down);
        }
    }

    private static float[] Run(float[] x, float[] router,
        byte* gate, QuantizationType gateQt, long gateExp, byte* up, QuantizationType upQt, long upExp,
        byte* down, QuantizationType downQt, long downExp, ComputeThreadPool? pool)
    {
        int[] assignExpert = new int[TopK];
        float[] assignWeight = new float[TopK];
        int[] cursors = new int[NumExperts + 1];
        int[] tokens = new int[TopK];
        int[] slots = new int[TopK];
        int[] unique = new int[NumExperts];
        int uniqueCount = MoeSwiGluMlp.Route(x, router, assignExpert, assignWeight, cursors, tokens, slots, unique,
            NumExperts, TopK, Hidden, 1, normTopKProb: true);
        float[] output = new float[Hidden];
        MoeSwiGluMlp.ExecuteRoutedFromAssignments(
            x,
            (nint)gate, gateQt, gateExp, ReadOnlySpan<nint>.Empty,
            (nint)up, upQt, upExp, ReadOnlySpan<nint>.Empty,
            (nint)down, downQt, downExp, ReadOnlySpan<nint>.Empty,
            assignExpert, assignWeight, cursors, tokens, slots, unique, uniqueCount,
            output,
            NumExperts, TopK, Hidden, Inter, 1,
            ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, 0, ReadOnlySpan<float>.Empty,
            loraAdapter: null, loraLayer: 0, threadPool: pool);
        return output;
    }

    /// <summary>Random quantised bytes with every fp16 super-scale overwritten by a small finite value.</summary>
    private static byte* Bank(Random rng, QuantizationType qt, long bytes)
    {
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        for (long i = 0; i < bytes; i++) p[i] = (byte)rng.Next(256);
        var info = QuantFormat.TryGetInfo(qt)!.Value;
        int bb = info.BlockBytes;
        for (long off = 0; off + bb <= bytes; off += bb)
        {
            Half d = (Half)((rng.NextSingle() * 2f - 1f) * 0.01f);
            Half d2 = (Half)(rng.NextSingle() * 0.01f);
            byte* b = p + off;
            switch (qt)
            {
                case QuantizationType.Q8_0: *(Half*)b = d; break;
                case QuantizationType.Q4_K:
                case QuantizationType.Q5_K: *(Half*)b = d; *(Half*)(b + 2) = d2; break;
                case QuantizationType.Q2_K: *(Half*)(b + 80) = d; *(Half*)(b + 82) = d2; break;
                case QuantizationType.Q3_K: *(Half*)(b + 108) = d; break;
                case QuantizationType.Q6_K: *(Half*)(b + 208) = d; break;
                case QuantizationType.IQ1_M:
                    {
                        ushort bits = BitConverter.HalfToUInt16Bits(d);
                        ushort* sc = (ushort*)(b + 48);
                        for (int i = 0; i < 4; i++)
                            sc[i] = (ushort)((sc[i] & 0x0FFF) | (((bits >> (4 * i)) & 0xF) << 12));
                        break;
                    }
                default: *(Half*)b = d; break; // IQ2_XXS / IQ2_XS / IQ3_S: leading fp16 d
            }
        }
        return p;
    }
}
