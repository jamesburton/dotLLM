using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1071: one routed <c>nemotron_h_moe</c> expert on the CPU (<see cref="NemotronHMoeExpertCompute"/>), the API the CPU
/// model's per-expert paths and the CUDA offload's CPU cold leg share, and the cold leg's expert-parallel workers and gather
/// index (pure CPU code of the CUDA host; nothing here touches a GPU).
/// <list type="bullet">
///   <item>The API is bit-identical to the type-specific CPU kernels it replaces in the CPU model (up, relu², down), for every
///   bank type Bolt-class files use, at n = 1 and n &gt; 1.</item>
///   <item>A null pool (the cold leg: one expert per worker) and a compute pool give bit-identical results.</item>
///   <item>Expert e is read from <c>bank + e · perExpertBytes</c>.</item>
///   <item>The worker pool computes a schedule's CPU groups into the rows the gather index points at, bit-identically for any
///   thread count.</item>
/// </list>
/// </summary>
public sealed unsafe class NemotronHMoeExpertComputeTests : IDisposable
{
    private const int ExpertDim = 256, Inter = 512, Experts = 4;
    private readonly List<nint> _allocs = [];

    public void Dispose()
    {
        foreach (nint p in _allocs) NativeMemory.Free((void*)p);
    }

    public static TheoryData<QuantizationType, QuantizationType> BankTypes => new()
    {
        { QuantizationType.Q4_K, QuantizationType.Q5_K },
        { QuantizationType.Q6_K, QuantizationType.Q8_0 },
        { QuantizationType.IQ2_XXS, QuantizationType.IQ2_XXS },
        { QuantizationType.IQ4_NL, QuantizationType.IQ4_XS },
        { QuantizationType.NVFP4, QuantizationType.NVFP4 },
        { QuantizationType.F32, QuantizationType.Q5_K },
    };

    [Theory]
    [MemberData(nameof(BankTypes))]
    public void Compute_IsBitIdenticalToTheKernelsTheCpuModelRanBefore(QuantizationType upQt, QuantizationType downQt)
    {
        var moe = Layer(upQt, downQt, seed: 11);
        foreach (int n in new[] { 1, 3, 8 })
        {
            float* x = Floats(n * ExpertDim, seed: n);
            for (int e = 0; e < Experts; e++)
            {
                float* got = Floats(n * ExpertDim);
                NemotronHMoeExpertCompute.Compute(moe, e, ExpertDim, x, n, Floats(n * Inter), got);

                float* mid = Floats(n * Inter), want = Floats(n * ExpertDim);
                ReferenceGemm(moe.UpExpsWeight + (nint)(e * moe.UpPerExpertBytes), upQt, x, mid, Inter, ExpertDim, n);
                ReluSquared.Execute(new Span<float>(mid, n * Inter), new Span<float>(mid, n * Inter));
                ReferenceGemm(moe.DownExpsWeight + (nint)(e * moe.DownPerExpertBytes), downQt, mid, want, ExpertDim, Inter, n);

                AssertBits(want, got, n * ExpertDim, $"{upQt}/{downQt} n={n} expert {e}");
                Assert.True(MaxAbs(got, n * ExpertDim) > 0, "the expert output is all zero: the fixture proves nothing");
            }
        }
    }

    [Theory]
    [MemberData(nameof(BankTypes))]
    public void Compute_NullPoolAndComputePool_AreBitIdentical_AndExpertsAreTheirOwnBankSlices(
        QuantizationType upQt, QuantizationType downQt)
    {
        var moe = Layer(upQt, downQt, seed: 12);
        const int N = 5;
        float* x = Floats(N * ExpertDim, seed: 5);
        using var pool = new ComputeThreadPool(4);
        for (int e = 0; e < Experts; e++)
        {
            float* a = Floats(N * ExpertDim), b = Floats(N * ExpertDim), c = Floats(N * ExpertDim);
            NemotronHMoeExpertCompute.Compute(moe, e, ExpertDim, x, N, Floats(N * Inter), a);
            NemotronHMoeExpertCompute.Compute(moe, e, ExpertDim, x, N, Floats(N * Inter), b, pool);
            AssertBits(a, b, N * ExpertDim, $"{upQt}/{downQt} expert {e}: pool");

            // The same expert copied out of its bank into a one-expert layer.
            var one = Copy((byte*)NemotronHMoeExpertCompute.UpPointer(moe, e), moe.UpPerExpertBytes);
            var oneDown = Copy((byte*)NemotronHMoeExpertCompute.DownPointer(moe, e), moe.DownPerExpertBytes);
            var single = Layer(upQt, downQt, seed: 12, upBank: one, downBank: oneDown, experts: 1);
            NemotronHMoeExpertCompute.Compute(single, 0, ExpertDim, x, N, Floats(N * Inter), c);
            AssertBits(a, c, N * ExpertDim, $"{upQt}/{downQt} expert {e}: bank slice");
        }
        bool threw = false;
        try { NemotronHMoeExpertCompute.Compute(moe, Experts, ExpertDim, x, 1, null, null); }
        catch (ArgumentOutOfRangeException) { threw = true; }
        Assert.True(threw, "an out-of-range expert must be rejected");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public void CpuColdLeg_ComputesTheScheduledGroupsIntoTheirRows_ForAnyThreadCount(int threads)
    {
        var moe = Layer(QuantizationType.Q4_K, QuantizationType.Q6_K, seed: 13, experts: 6);
        const int N = 7, TopK = 3, Slots = N * TopK;
        float* x = Floats(N * ExpertDim, seed: 7);

        // Schedule: 2 device groups then 3 CPU groups (experts 5, 1, 4), slot ids ascending within each group.
        int[] groupExperts = [0, 2, 5, 1, 4];
        int[][] groupSlots = [[0, 4, 9], [1, 5, 10, 13, 16, 20], [2, 3, 6, 11], [7, 8, 12, 14, 17], [15, 18, 19]];
        int[] groupStart = new int[groupExperts.Length + 1];
        int[] slotIds = new int[Slots];
        for (int g = 0, pos = 0; g < groupExperts.Length; g++)
        {
            groupStart[g] = pos;
            foreach (int s in groupSlots[g]) slotIds[pos++] = s;
            groupStart[g + 1] = pos;
        }
        Assert.Equal(Slots, groupStart[^1]);
        int firstCpu = 2, cpuStart = groupStart[firstCpu], cpuRows = Slots - cpuStart;

        float* rows = Floats(cpuRows * ExpertDim);
        using (var leg = new CudaNemotronHTransformerModel.CpuColdLeg(threads))
        {
            fixed (int* gs = groupStart, ids = slotIds)
                leg.Run(moe, ExpertDim, TopK, groupExperts, firstCpu, groupExperts.Length, gs, ids, x, rows);
        }

        // Each CPU slot's row is its expert applied to its token's input, as one sequential call per slot computes it.
        int[] idx = new int[Slots];
        fixed (int* ids = slotIds, ix = idx)
            CudaNemotronHTransformerModel.BuildColdGatherIndex(ids, cpuStart, Slots, ix);
        for (int g = firstCpu; g < groupExperts.Length; g++)
        {
            int cnt = groupSlots[g].Length;
            float* gIn = Floats(cnt * ExpertDim), want = Floats(cnt * ExpertDim);
            for (int j = 0; j < cnt; j++)
                new ReadOnlySpan<float>(x + (long)(groupSlots[g][j] / TopK) * ExpertDim, ExpertDim)
                    .CopyTo(new Span<float>(gIn + (long)j * ExpertDim, ExpertDim));
            NemotronHMoeExpertCompute.Compute(moe, groupExperts[g], ExpertDim, gIn, cnt, Floats(cnt * Inter), want);
            for (int j = 0; j < cnt; j++)
            {
                int slot = groupSlots[g][j];
                int row = idx[slot] - Slots;   // the gather reads the CPU rows after the device's
                Assert.InRange(row, 0, cpuRows - 1);
                AssertBits(want + (long)j * ExpertDim, rows + (long)row * ExpertDim, ExpertDim, $"slot {slot} (expert {groupExperts[g]})");
            }
        }

        // Device slots read their own rows.
        for (int g = 0; g < firstCpu; g++)
            foreach (int s in groupSlots[g]) Assert.Equal(s, idx[s]);
        Assert.Equal(Slots, idx.Distinct().Count());
    }

    [Fact]
    public void CpuColdLeg_SurfacesAWorkerFailure()
    {
        var moe = Layer(QuantizationType.Q8_0, QuantizationType.Q8_0, seed: 14, experts: 2);
        int[] groupExperts = [0, 7];   // expert 7 does not exist
        int[] groupStart = [0, 1, 2];
        int[] slotIds = [0, 1];
        float* x = Floats(ExpertDim), rows = Floats(2 * ExpertDim);
        using var leg = new CudaNemotronHTransformerModel.CpuColdLeg(2);
        InvalidOperationException? ex = null;
        try
        {
            fixed (int* gs = groupStart, ids = slotIds)
                leg.Run(moe, ExpertDim, 2, groupExperts, 0, 2, gs, ids, x, rows);
        }
        catch (InvalidOperationException e)
        {
            ex = e;
        }
        Assert.NotNull(ex);
        Assert.IsType<ArgumentOutOfRangeException>(ex!.InnerException);
    }

    // ───────────────────────── fixture ─────────────────────────

    private NemotronHMoeWeights Layer(QuantizationType upQt, QuantizationType downQt, int seed, nint upBank = 0, nint downBank = 0,
        int experts = Experts)
    {
        var rng = new Random(seed);
        long upPer = Dequantize.RowByteSize(ExpertDim, upQt) * Inter;
        long downPer = Dequantize.RowByteSize(Inter, downQt) * ExpertDim;
        if (upBank == 0) upBank = Bank(rng, upQt, experts * Inter, ExpertDim, upPer * experts);
        if (downBank == 0) downBank = Bank(rng, downQt, experts * ExpertDim, Inter, downPer * experts);
        return new NemotronHMoeWeights
        {
            GateInpWeight = 0, GateInpQuantType = QuantizationType.F32,
            SelectionBias = new float[experts],
            UpExpsWeight = upBank, UpExpsQuantType = upQt, UpPerExpertBytes = upPer,
            DownExpsWeight = downBank, DownExpsQuantType = downQt, DownPerExpertBytes = downPer,
            UpShexpWeight = 0, UpShexpQuantType = QuantizationType.F32,
            DownShexpWeight = 0, DownShexpQuantType = QuantizationType.F32,
            NumExperts = experts, NumExpertsPerTok = 2, MoeIntermediateSize = Inter, SharedIntermediateSize = 1,
            NormalizeWeights = true, WeightsScale = 1f, LatentSize = ExpertDim,
        };
    }

    /// <summary>A bank of <paramref name="rows"/> rows of length <paramref name="k"/> in <paramref name="qt"/>.</summary>
    private nint Bank(Random rng, QuantizationType qt, int rows, int k, long bytes)
    {
        byte[] b;
        switch (qt)
        {
            case QuantizationType.IQ2_XXS:
                b = new byte[bytes];
                rng.NextBytes(b);
                for (long blk = 0; blk < bytes; blk += 66) SetHalf(b, blk, 0.001f + (float)rng.NextDouble() * 0.001f);
                break;
            case QuantizationType.NVFP4:
                b = new byte[bytes];
                rng.NextBytes(b);
                for (long blk = 0; blk < bytes; blk += 36)
                    for (int s = 0; s < 4; s++) b[blk + s] = (byte)rng.Next(0x28, 0x38);
                break;
            default:
                float[] w = new float[(long)rows * k];
                for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
                b = qt switch
                {
                    QuantizationType.F32 => MemoryMarshal.AsBytes(w.AsSpan()).ToArray(),
                    QuantizationType.Q4_K => Q4KFixture.QuantizeRows(w, rows, k),
                    QuantizationType.Q5_K => Q5KFixture.QuantizeRows(w, rows, k),
                    QuantizationType.Q6_K => Q6KFixture.QuantizeRows(w, rows, k),
                    QuantizationType.Q8_0 => Quantize.FromFloat32(w, w.Length, QuantizationType.Q8_0),
                    QuantizationType.IQ4_NL => Iq4Fixture.QuantizeRowsIq4Nl(w, rows, k),
                    QuantizationType.IQ4_XS => Iq4Fixture.QuantizeRowsIq4Xs(w, rows, k),
                    _ => throw new NotSupportedException(qt.ToString()),
                };
                break;
        }
        Assert.Equal(bytes, b.LongLength);
        nint p = Copy(b);
        return p;
    }

    private static void SetHalf(byte[] b, long at, float v)
    {
        ushort h = BitConverter.HalfToUInt16Bits((Half)v);
        b[at] = (byte)(h & 0xFF);
        b[at + 1] = (byte)(h >> 8);
    }

    /// <summary>The type-specific CPU kernels the model's <c>Gemm</c> dispatched to before #1071 (pool = null).</summary>
    private static void ReferenceGemm(nint w, QuantizationType qt, float* b, float* c, int m, int k, int n)
    {
        byte* p = (byte*)w;
        switch (qt)
        {
            case QuantizationType.Q4_K: MatMul.GemmQ4_K(p, b, c, m, k, n, null, null); break;
            case QuantizationType.Q5_K: MatMul.GemmQ5_K(p, b, c, m, k, n, null, null); break;
            case QuantizationType.Q6_K: MatMul.GemmQ6_K(p, b, c, m, k, n, null, null); break;
            case QuantizationType.Q8_0: MatMul.GemmQ8_0(p, b, c, m, k, n, null, null); break;
            case QuantizationType.IQ2_XXS: MatMul.GemmIQCodebook(qt, p, b, c, m, k, n, null, null); break;
            case QuantizationType.IQ4_XS: MatMul.GemmIQ4_XS(p, b, c, m, k, n, null, null); break;
            case QuantizationType.IQ4_NL: MatMul.GemmLegacyQuantOrDequant(p, qt, b, c, m, k, n, null, null); break;
            case QuantizationType.NVFP4: MatMul.GemmNvfp4(p, b, c, m, k, n, null); break;
            case QuantizationType.F32: MatMul.GemmF32((float*)p, b, c, m, k, n, null); break;
            default: throw new NotSupportedException(qt.ToString());
        }
    }

    private float* Floats(int count, int seed = -1)
    {
        float* p = (float*)NativeMemory.AllocZeroed((nuint)Math.Max(count, 1), sizeof(float));
        _allocs.Add((nint)p);
        if (seed >= 0)
        {
            var rng = new Random(seed);
            for (int i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        return p;
    }

    private nint Copy(byte[] src)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)src.Length);
        _allocs.Add((nint)p);
        src.CopyTo(new Span<byte>(p, src.Length));
        return (nint)p;
    }

    private nint Copy(byte* src, long bytes)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)bytes);
        _allocs.Add((nint)p);
        Buffer.MemoryCopy(src, p, bytes, bytes);
        return (nint)p;
    }

    private static void AssertBits(float* want, float* got, int count, string what)
    {
        for (int i = 0; i < count; i++)
            Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                $"{what}: element {i}: want {want[i]:R}, got {got[i]:R}");
    }

    private static float MaxAbs(float* p, int count)
    {
        float m = 0;
        for (int i = 0; i < count; i++) m = MathF.Max(m, MathF.Abs(p[i]));
        return m;
    }
}
