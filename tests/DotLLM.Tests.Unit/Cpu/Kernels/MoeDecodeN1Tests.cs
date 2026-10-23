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
        => AssertDecodeParity(gateQt, upQt, downQt, packedLegacy: false, expectFastPath: true);

    [Theory]
    [InlineData(QuantizationType.Q4_0, QuantizationType.Q4_0, QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_1, QuantizationType.Q4_1, QuantizationType.Q5_1)]
    [InlineData(QuantizationType.Q5_1, QuantizationType.Q5_1, QuantizationType.IQ4_NL)]
    [InlineData(QuantizationType.IQ4_NL, QuantizationType.IQ4_NL, QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_0, QuantizationType.Q5_1, QuantizationType.Q6_K)]
    [InlineData(QuantizationType.Q5_0, QuantizationType.Q4_0, QuantizationType.Q8_0)]
    public void PackedLegacyDecode_IsBitIdenticalAndUsesFastPath(QuantizationType gateQt, QuantizationType upQt, QuantizationType downQt)
        => AssertDecodeParity(gateQt, upQt, downQt, packedLegacy: true, expectFastPath: true);

    [Theory]
    [InlineData(QuantizationType.Q4_0, QuantizationType.Q4_0, QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q5_0, QuantizationType.Q4_0, QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_K, QuantizationType.Q4_K, QuantizationType.Q4_0)]
    public void LegacyOptOut_PreservesGroupedNumerics(QuantizationType gateQt, QuantizationType upQt, QuantizationType downQt)
        => AssertDecodeParity(gateQt, upQt, downQt, packedLegacy: false, expectFastPath: false);

    [Fact]
    public void PackedLegacyDecode_UnevenWorkerPartitionsPreserveRowKernel()
        => AssertDecodeParity(QuantizationType.Q4_0, QuantizationType.Q4_0, QuantizationType.Q4_0,
            packedLegacy: true, expectFastPath: true, workers: 6);

    [Theory]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_1)]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void PackedLegacyPrefill_MatchesIndependentTokenRows(QuantizationType qt)
    {
        var rng = new Random(1029 + (int)qt);
        long expertBytes = Dequantize.RowByteSize((long)Inter * Hidden, qt);
        byte* gate = Bank(rng, qt, expertBytes * NumExperts);
        byte* up = Bank(rng, qt, expertBytes * NumExperts);
        byte* down = Bank(rng, qt, expertBytes * NumExperts);
        using var pool = new ComputeThreadPool(5);
        pool.SetDispatchMode(DispatchMode.SpinWait);
        try
        {
            float[] x = Enumerable.Range(0, Hidden * 3).Select(_ => rng.NextSingle() * 2f - 1f).ToArray();
            float[] router = Enumerable.Range(0, NumExperts * Hidden).Select(_ => rng.NextSingle() * 2f - 1f).ToArray();
            var actual = Run(x, router, gate, qt, expertBytes, up, qt, expertBytes, down, qt, expertBytes, pool, true);
            for (int t = 0; t < 3; t++)
            {
                var expected = Run(x.AsSpan(t * Hidden, Hidden).ToArray(), router,
                    gate, qt, expertBytes, up, qt, expertBytes, down, qt, expertBytes, null, true);
                // Multi-token GEMM and single-token GEMV use different accumulation orders.
                float scale = MathF.Max(1, expected.Max(MathF.Abs));
                for (int i = 0; i < Hidden; i++)
                    Assert.True(float.IsFinite(actual[t * Hidden + i]) && MathF.Abs(expected[i] - actual[t * Hidden + i]) <= scale * 1e-5f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(gate); NativeMemory.AlignedFree(up); NativeMemory.AlignedFree(down);
        }
    }

    private static void AssertDecodeParity(QuantizationType gateQt, QuantizationType upQt, QuantizationType downQt,
        bool packedLegacy, bool expectFastPath, int workers = 5)
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
        using var pool = new ComputeThreadPool(workers);
        pool.SetDispatchMode(DispatchMode.SpinWait);
        try
        {
            for (int trial = 0; trial < 6; trial++)
            {
                float[] x = new float[Hidden];
                for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
                float[] expected = Run(x, router, gate, gateQt, gateExp, up, upQt, upExp, down, downQt, downExp, null, packedLegacy);
                long callsBefore = MoeSwiGluMlp.DecodeN1Calls;
                float[] actual = Run(x, router, gate, gateQt, gateExp, up, upQt, upExp, down, downQt, downExp, pool, packedLegacy);
                if (expectFastPath)
                    Assert.True(MoeSwiGluMlp.DecodeN1Calls > callsBefore, "decode fast path did not run");
                else
                    Assert.Equal(callsBefore, MoeSwiGluMlp.DecodeN1Calls);
                for (int i = 0; i < Hidden; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"{gateQt}/{upQt}/{downQt} trial {trial} elem {i}: grouped {expected[i]} vs decode {actual[i]}");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(gate); NativeMemory.AlignedFree(up); NativeMemory.AlignedFree(down);
        }
    }

    private static float[] Run(float[] x, float[] router,
        byte* gate, QuantizationType gateQt, long gateExp, byte* up, QuantizationType upQt, long upExp,
        byte* down, QuantizationType downQt, long downExp, ComputeThreadPool? pool, bool packedLegacy)
    {
        int seqLen = x.Length / Hidden;
        int[] assignExpert = new int[TopK * seqLen];
        float[] assignWeight = new float[TopK * seqLen];
        int[] cursors = new int[NumExperts + 1];
        int[] tokens = new int[TopK * seqLen];
        int[] slots = new int[TopK * seqLen];
        int[] unique = new int[NumExperts];
        int uniqueCount = MoeSwiGluMlp.Route(x, router, assignExpert, assignWeight, cursors, tokens, slots, unique,
            NumExperts, TopK, Hidden, seqLen, normTopKProb: true);
        float[] output = new float[Hidden * seqLen];
        MoeSwiGluMlp.ExecuteRoutedFromAssignments(
            x,
            (nint)gate, gateQt, gateExp, ReadOnlySpan<nint>.Empty,
            (nint)up, upQt, upExp, ReadOnlySpan<nint>.Empty,
            (nint)down, downQt, downExp, ReadOnlySpan<nint>.Empty,
            assignExpert, assignWeight, cursors, tokens, slots, unique, uniqueCount,
            output,
            NumExperts, TopK, Hidden, Inter, seqLen,
            ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, 0, ReadOnlySpan<float>.Empty,
            loraAdapter: null, loraLayer: 0, threadPool: pool, packedLegacyExperts: packedLegacy);
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
                case QuantizationType.Q4_1:
                case QuantizationType.Q5_1: *(Half*)b = d; *(Half*)(b + 2) = d2; break;
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
