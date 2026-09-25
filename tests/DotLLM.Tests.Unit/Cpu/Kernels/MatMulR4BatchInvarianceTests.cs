using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #530. <c>GemmInterleaved</c> used to dispatch on batch size: <c>n == 1</c> ran the
/// repacked (R4-interleaved) <c>ComputeRows*Interleaved</c> kernels, <c>n &gt; 1</c> fell through
/// to the original row-major weights for every non-Q8_0 quantization. Two implementations of the
/// same mathematics selected by row count, so a token's logits depended on how many tokens shared
/// its forward pass.
/// </summary>
/// <remarks>
/// These assert on raw kernel outputs, bit-exactly. Argmax is insensitive to a 1e-6 drift, which
/// is why the defect hid behind greedy-token comparisons for so long — so nothing here is allowed
/// a tolerance.
/// </remarks>
public sealed unsafe class MatMulR4BatchInvarianceTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_1BlockBytes = 36;
    private const int Q5_0BlockBytes = 22;
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int Q8_K_BlockBytes = 292;

    /// <summary>
    /// m = 38 exercises 9 full R4 groups plus a 2-row row-major tail, and is above
    /// <c>ParallelMinRows</c> (32) so the pooled paths are reachable too.
    /// </summary>
    private const int M = 38;

    /// <summary>k = 512: two K-quant super-blocks, 16 Q8_0/Q5_0 blocks.</summary>
    private const int K = 512;

    /// <summary>
    /// Quants whose R4 and row-major kernels are required to agree bit for bit. Q8_0 is absent on
    /// purpose — see <see cref="Q8_0_RowMajorVsRepacked_StaysWithinBound"/>.
    /// </summary>
    public static TheoryData<QuantizationType> LayoutParityQuants() => new()
    {
        QuantizationType.Q4_K,
        QuantizationType.Q5_K,
        QuantizationType.Q6_K,
        QuantizationType.Q5_0,
    };

    public static TheoryData<QuantizationType, int> QuantsByBatch()
    {
        var data = new TheoryData<QuantizationType, int>();
        foreach (var qt in new[]
                 {
                     QuantizationType.Q4_K, QuantizationType.Q5_K, QuantizationType.Q6_K,
                     QuantizationType.Q5_0, QuantizationType.Q8_0,
                 })
        {
            for (int n = 1; n <= 5; n++)
                data.Add(qt, n);
        }

        return data;
    }

    /// <summary>
    /// The batch-size sweep the issue asks for: a row's output must not depend on its batch-mates.
    /// Reference is the single-token arm (repacked <c>ComputeRows*Interleaved</c>); the arm under
    /// test is the multi-token GEMM, run for n = 1..5 over the same rows.
    /// </summary>
    [Theory]
    [MemberData(nameof(QuantsByBatch))]
    public void MultiTokenGemm_MatchesSingleToken_BitExact(QuantizationType qt, int n)
    {
        var f = new Fixture(qt, n, seed: 530 + n);
        try
        {
            f.RunMultiTokenGemm(f.Actual, pool: null);
            f.RunSingleTokenPerRow(f.Expected, pool: null);
            f.AssertBitExact($"{qt} n={n}: multi-token GEMM diverges from the single-token kernel");

            // Same property with the thread pool engaged on both arms: M = 38 is above
            // ParallelMinRows, so production reaches the pooled workers, and group boundaries
            // must not move with the thread count.
            using var pool = new ComputeThreadPool(4);
            f.RunMultiTokenGemm(f.Actual, pool);
            f.AssertBitExact($"{qt} n={n}: pooled multi-token GEMM diverges from the single-token kernel");
            f.RunSingleTokenPerRow(f.Actual, pool);
            f.AssertBitExact($"{qt} n={n}: pooled single-token kernel diverges from the serial one");
        }
        finally
        {
            f.Dispose();
        }
    }

    /// <summary>
    /// The defect itself, at kernel level: the row-major GEMM the old <c>n &gt; 1</c> arm fell
    /// through to must agree, bit for bit, with the repacked kernel the <c>n == 1</c> arm runs.
    /// Both consume the identical pre-quantized activations, so any difference is pure
    /// accumulation-order divergence between two implementations of one operation.
    /// </summary>
    /// <remarks>
    /// RED before the fix for Q4_K / Q5_K / Q6_K: the R4 kernel used to horizontally reduce and
    /// accumulate once per super-block, while the row-major kernel accumulated the whole K in
    /// vector accumulators. Q5_0 passes either way (its R4 kernel already accumulated across the
    /// whole K); Q8_0 is absent because whether it agrees is tier-dependent — see
    /// <see cref="Q8_0_RowMajorVsRepacked_StaysWithinBound"/> and #535.
    /// </remarks>
    [Theory]
    [MemberData(nameof(LayoutParityQuants))]
    public void RowMajorGemm_MatchesRepackedKernel_BitExact(QuantizationType qt)
    {
        const int n = 3;
        var f = new Fixture(qt, n, seed: 4530);
        try
        {
            f.RunRowMajorGemm(f.Actual);
            f.RunSingleTokenPerRow(f.Expected, pool: null);
            f.AssertBitExact($"{qt}: row-major GEMM diverges from the repacked kernel");
        }
        finally
        {
            f.Dispose();
        }
    }

    /// <summary>
    /// Bounds a gap #530 does NOT close (#535): Q8_0's R4 kernel
    /// (<c>VecDotQ8_0Avx2_4RowsR4</c>) and its row-major kernel can disagree. Asserts only the
    /// upper bound, because how much they disagree — including whether they disagree at all — is
    /// hardware-tier-dependent.
    /// </summary>
    /// <remarks>
    /// Measured, and a datum for #535: 4.304E-006 relative on a Zen 5 box (row-major on the VNNI
    /// tier, <c>ComputeRowsVnni</c>), 2.564E-006 on the same box with <c>DOTNET_EnableAVXVNNI=0</c>,
    /// and bit-exact agreement on the GitHub runner. So the two layouts DO agree on at least one
    /// shipping tier and disagree on others.
    ///
    /// An earlier version asserted <c>AnyBitDifference()</c> unconditionally and encoded a property
    /// of one machine: green locally, red in CI. Gating that assertion on the VNNI tier would have
    /// been a second guess of the same kind — disabling VNNI here does not make the two agree, so
    /// VNNI is not the predicate, and nothing available on this box identifies the one that is.
    /// The bound is what holds everywhere, so the bound is what this asserts; if the tiers ever
    /// converge, this test simply keeps passing and #535 closes on its own evidence.
    ///
    /// Nothing in the transformer's O / FFN / lm_head path depends on the two agreeing: both
    /// batch-size arms run the R4 kernel. The fused decode QKV path (<c>FusedDecodeGemv3</c>)
    /// does read the ORIGINAL row-major weights while prefill QKV runs R4, so Q8_0 QKV can depend
    /// on batch size wherever both layouts are in play — on SmolLM-135M it measures 0.0
    /// end-to-end (see ChunkedPrefillLogitsQ8_0Tests), so this stays a kernel-level finding.
    /// </remarks>
    [Fact]
    public void Q8_0_RowMajorVsRepacked_StaysWithinBound()
    {
        const int n = 3;
        var f = new Fixture(QuantizationType.Q8_0, n, seed: 4530);
        try
        {
            f.RunRowMajorGemm(f.Actual);
            f.RunSingleTokenPerRow(f.Expected, pool: null);

            Assert.True(f.MaxRelativeDifference() < 1e-5f,
                $"Q8_0 R4-vs-row-major divergence grew beyond the recorded bound: {f.WorstPair()}");
        }
        finally
        {
            f.Dispose();
        }
    }

    /// <summary>Runs one (quant, n) case of the sweep; internal so isolation repros can reuse it.</summary>
    internal static void RunOneCase(QuantizationType qt, int n, ComputeThreadPool? pool)
    {
        var f = new Fixture(qt, n, seed: 530 + n);
        try
        {
            f.RunMultiTokenGemm(f.Actual, pool);
            f.RunSingleTokenPerRow(f.Expected, pool);
            f.AssertBitExact($"{qt} n={n} pool={(pool is null ? "null" : "4")}");
        }
        finally
        {
            f.Dispose();
        }
    }

    // ──────────────────── Fixture ────────────────────

    private sealed class Fixture : IDisposable
    {
        private readonly QuantizationType _qt;
        private readonly int _n;
        private readonly int _blockCount;
        private readonly int _inputRowBytes;
        private readonly nint _rowMajor;
        private readonly float* _b;
        private readonly byte* _inputQ;
        private WeightRepacking.RepackedWeight _rw;

        public readonly float* Actual;
        public readonly float* Expected;

        public Fixture(QuantizationType qt, int n, int seed)
        {
            _qt = qt;
            _n = n;
            var rng = new Random(seed);

            int blockBytes = BlockBytes(qt);
            int groupSize = qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K
                ? 256
                : 32;
            _blockCount = K / groupSize;
            _inputRowBytes = _blockCount * InputBlockBytes(qt);

            _rowMajor = AllocRandomWeights(blockBytes, _blockCount, M, rng);
            _rw = WeightRepacking.RepackR4(_rowMajor, qt, M, K);
            Assert.NotEqual(0, _rw.Ptr);

            _b = (float*)NativeMemory.AlignedAlloc((nuint)(n * K * sizeof(float)), 64);
            for (int i = 0; i < n * K; i++)
                _b[i] = rng.NextSingle() * 2f - 1f;

            _inputQ = (byte*)NativeMemory.AlignedAlloc((nuint)(n * _inputRowBytes), 64);
            for (int t = 0; t < n; t++)
                QuantizeRow(qt, _b + t * K, _inputQ + t * _inputRowBytes);

            Actual = (float*)NativeMemory.AlignedAlloc((nuint)(n * M * sizeof(float)), 64);
            Expected = (float*)NativeMemory.AlignedAlloc((nuint)(n * M * sizeof(float)), 64);
            for (int i = 0; i < n * M; i++) { Actual[i] = float.NaN; Expected[i] = float.NaN; }
        }

        /// <summary>The post-fix multi-token arm: tiled GEMM over the repacked weights.</summary>
        public void RunMultiTokenGemm(float* c, ComputeThreadPool? pool)
        {
            byte* w = (byte*)_rw.Ptr;
            switch (_qt)
            {
                case QuantizationType.Q4_K:
                    MatMul.GemmR4TiledQ4_K(w, _b, _inputQ, c, _rw.FullGroupCount, _rw.TailRows,
                        _blockCount, M, K, _n, pool);
                    break;
                case QuantizationType.Q5_K:
                    MatMul.GemmR4TiledQ5_K(w, _b, _inputQ, c, _rw.FullGroupCount, _rw.TailRows,
                        _blockCount, M, K, _n, pool);
                    break;
                case QuantizationType.Q6_K:
                    MatMul.GemmR4TiledQ6_K(w, _b, _inputQ, c, _rw.FullGroupCount, _rw.TailRows,
                        _blockCount, M, K, _n, pool);
                    break;
                case QuantizationType.Q5_0:
                    MatMul.GemmR4TiledQ5_0(w, _b, _inputQ, c, _rw.FullGroupCount, _rw.TailRows,
                        _blockCount, M, K, _n, pool);
                    break;
                case QuantizationType.Q8_0:
                    MatMul.GemmR4TiledQ8_0(w, _b, _inputQ, c, _rw.FullGroupCount, _rw.TailRows,
                        _blockCount, M, K, _n, pool);
                    break;
                default:
                    throw new NotSupportedException(_qt.ToString());
            }
        }

        /// <summary>The pre-fix multi-token arm: row-major GEMM over the original weights.</summary>
        public void RunRowMajorGemm(float* c)
        {
            byte* w = (byte*)_rowMajor;
            switch (_qt)
            {
                case QuantizationType.Q4_K:
                    MatMul.GemmQ4_K(w, _b, c, M, K, _n, _inputQ);
                    break;
                case QuantizationType.Q5_K:
                    MatMul.GemmQ5_K(w, _b, c, M, K, _n, _inputQ);
                    break;
                case QuantizationType.Q6_K:
                    MatMul.GemmQ6_K(w, _b, c, M, K, _n, _inputQ);
                    break;
                case QuantizationType.Q5_0:
                    MatMul.GemmQ5_0(w, _b, c, M, K, _n, _inputQ);
                    break;
                case QuantizationType.Q8_0:
                    MatMul.GemmQ8_0(w, _b, c, M, K, _n, _inputQ);
                    break;
                default:
                    throw new NotSupportedException(_qt.ToString());
            }
        }

        /// <summary>The single-token arm: repacked ComputeRows, once per token.</summary>
        public void RunSingleTokenPerRow(float* c, ComputeThreadPool? pool)
        {
            byte* w = (byte*)_rw.Ptr;
            for (int t = 0; t < _n; t++)
            {
                byte* x = _inputQ + (long)t * _inputRowBytes;
                float* y = c + (long)t * M;
                switch (_qt)
                {
                    case QuantizationType.Q4_K:
                        MatMul.ComputeRowsQ4_KInterleaved(w, x, y, _rw.FullGroupCount, _rw.TailRows, _blockCount, pool);
                        break;
                    case QuantizationType.Q5_K:
                        MatMul.ComputeRowsQ5_KInterleaved(w, x, y, _rw.FullGroupCount, _rw.TailRows, _blockCount, pool);
                        break;
                    case QuantizationType.Q6_K:
                        MatMul.ComputeRowsQ6_KInterleaved(w, x, y, _rw.FullGroupCount, _rw.TailRows, _blockCount, pool);
                        break;
                    case QuantizationType.Q5_0:
                        MatMul.ComputeRowsQ5_0Interleaved(w, x, y, _rw.FullGroupCount, _rw.TailRows, _blockCount, pool);
                        break;
                    case QuantizationType.Q8_0:
                        MatMul.ComputeRowsQ8_0Interleaved(w, x, y, _rw.FullGroupCount, _rw.TailRows, _blockCount, pool);
                        break;
                    default:
                        throw new NotSupportedException(_qt.ToString());
                }
            }
        }

        public void AssertBitExact(string what)
        {
            for (int t = 0; t < _n; t++)
            {
                for (int r = 0; r < M; r++)
                {
                    float e = Expected[t * M + r];
                    float a = Actual[t * M + r];
                    Assert.False(float.IsNaN(e) || float.IsNaN(a), $"{what}: token {t} row {r} not written");
                    if (BitConverter.SingleToInt32Bits(e) != BitConverter.SingleToInt32Bits(a))
                    {
                        Assert.Fail($"{what}: token {t} row {r}: expected {e:R}, actual {a:R} " +
                                    $"(delta {MathF.Abs(e - a):E3})");
                    }
                }
            }
        }

        /// <summary>Worst offending element, so a failure shows noise vs functional break.</summary>
        public string WorstPair()
        {
            int worstIdx = 0;
            float worst = -1;
            float worstAbs = 0;
            for (int i = 0; i < _n * M; i++)
            {
                float denom = MathF.Max(MathF.Abs(Expected[i]), 1e-6f);
                float rel = MathF.Abs(Expected[i] - Actual[i]) / denom;
                worstAbs = MathF.Max(worstAbs, MathF.Abs(Expected[i] - Actual[i]));
                if (rel > worst) { worst = rel; worstIdx = i; }
            }

            return $"worst rel {worst:E3} at [{worstIdx}] expected={Expected[worstIdx]:R} " +
                   $"actual={Actual[worstIdx]:R}; max abs {worstAbs:E3}";
        }

        public float MaxRelativeDifference()
        {
            float worst = 0;
            for (int i = 0; i < _n * M; i++)
            {
                float denom = MathF.Max(MathF.Abs(Expected[i]), 1e-6f);
                worst = MathF.Max(worst, MathF.Abs(Expected[i] - Actual[i]) / denom);
            }

            return worst;
        }

        public void Dispose()
        {
            _rw.Dispose();
            NativeMemory.AlignedFree((void*)_rowMajor);
            NativeMemory.AlignedFree(_b);
            NativeMemory.AlignedFree(_inputQ);
            NativeMemory.AlignedFree(Actual);
            NativeMemory.AlignedFree(Expected);
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static int BlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_K => Q4_K_BlockBytes,
        QuantizationType.Q5_K => Q5_K_BlockBytes,
        QuantizationType.Q6_K => Q6_K_BlockBytes,
        QuantizationType.Q5_0 => Q5_0BlockBytes,
        QuantizationType.Q8_0 => Q8_0BlockBytes,
        _ => throw new NotSupportedException(qt.ToString()),
    };

    private static int InputBlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K => Q8_K_BlockBytes,
        QuantizationType.Q5_0 => Q8_1BlockBytes,
        QuantizationType.Q8_0 => Q8_0BlockBytes,
        _ => throw new NotSupportedException(qt.ToString()),
    };

    private static void QuantizeRow(QuantizationType qt, float* src, byte* dest)
    {
        switch (qt)
        {
            case QuantizationType.Q4_K:
            case QuantizationType.Q5_K:
            case QuantizationType.Q6_K:
                MatMul.QuantizeF32ToQ8_K(src, dest, K);
                break;
            case QuantizationType.Q5_0:
                MatMul.QuantizeF32ToQ8_1(src, dest, K);
                break;
            case QuantizationType.Q8_0:
                MatMul.QuantizeF32ToQ8_0(src, dest, K);
                break;
            default:
                throw new NotSupportedException(qt.ToString());
        }
    }

    /// <summary>
    /// Random quantized payload with plausible per-block scales — the exact bit pattern does not
    /// matter, only that both arms read the same bytes and that the scales are finite.
    /// </summary>
    private static nint AllocRandomWeights(int blockBytes, int blockCount, int rows, Random rng)
    {
        nuint totalBytes = (nuint)((long)rows * blockCount * blockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        byte[] buf = new byte[(int)totalBytes];
        rng.NextBytes(buf);
        fixed (byte* src = buf)
            NativeMemory.Copy(src, (void*)ptr, totalBytes);

        for (long b = 0; b < (long)rows * blockCount; b++)
        {
            byte* block = (byte*)ptr + b * blockBytes;
            switch (blockBytes)
            {
                case Q6_K_BlockBytes:
                    Unsafe.WriteUnaligned(block + 208, (Half)(rng.NextSingle() * 0.02f));
                    break;
                case Q4_K_BlockBytes:
                case Q5_K_BlockBytes:
                    Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                    Unsafe.WriteUnaligned(block + 2, (Half)(rng.NextSingle() * 0.1f));
                    break;
                case Q8_0BlockBytes:
                    Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                    // Q8_0 payload is sbyte. Real quantizers emit [-127, 127]; -128 (0x80) is
                    // out of domain and the VNNI tier's abs/sign emulation does not handle it
                    // (|-128| is still -128 in int8), so random bytes would compare two kernels
                    // on input neither is specified for.
                    for (int i = 2; i < Q8_0BlockBytes; i++)
                        if (block[i] == 0x80) block[i] = 0x81;
                    break;
                default: // Q5_0: Half scale at offset 0, then unsigned nibbles
                    Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                    break;
            }
        }

        return ptr;
    }
}
