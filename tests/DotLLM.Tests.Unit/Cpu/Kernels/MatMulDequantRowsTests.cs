using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Equivalence tests for issue #263: <see cref="MatMul.GemmDequantRows"/> replaced four copies of a
/// serial, dequantize-per-token fallback with one row-parallel kernel that decodes each weight row
/// once and reuses it across every input column.
/// </summary>
/// <remarks>
/// <para>
/// The rewrite is a scheduling change, not a numerical one: every output element is still
/// <c>TensorPrimitives.Dot(dequantize(row i), b + t*k)</c> over exactly the same inputs. The tests
/// therefore assert <b>bit-exact</b> equality against a literal transcription of the old serial
/// loop, not a tolerance — a tolerance would hide precisely the reordering mistakes that matter
/// (row/column transposition, a partition that skips or double-writes rows, a row buffer reused
/// across threads).
/// </para>
/// <para>
/// The reference deliberately shares <see cref="Dequantize.ToFloat32"/> with the kernel. Weight
/// decoding is not what changed and is covered by the per-format dequantize tests; re-deriving it
/// here would only test the transcription. What is under test is the loop structure around it.
/// </para>
/// </remarks>
public sealed unsafe class MatMulDequantRowsTests : IDisposable
{
    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    /// <summary>
    /// Every quantization format that reaches the dequantize-and-dot fallback, i.e. everything
    /// <see cref="MatMul.SupportsFusedDecode"/> rejects and for which no dedicated GEMM exists.
    /// </summary>
    public static TheoryData<QuantizationType> FallbackFormats()
    {
        var data = new TheoryData<QuantizationType>();
        data.Add(QuantizationType.BF16);
        data.Add(QuantizationType.Q4_0);
        data.Add(QuantizationType.Q4_1);
        data.Add(QuantizationType.Q5_1);
        data.Add(QuantizationType.IQ4_NL);
        data.Add(QuantizationType.Q2_K);
        data.Add(QuantizationType.Q3_K);
        data.Add(QuantizationType.IQ4_XS);
        data.Add(QuantizationType.IQ2_XXS);
        data.Add(QuantizationType.IQ2_XS);
        data.Add(QuantizationType.IQ2_S);
        data.Add(QuantizationType.IQ1_S);
        data.Add(QuantizationType.IQ3_XXS);
        data.Add(QuantizationType.IQ3_S);
        return data;
    }

    // ──────────────────── Bit-exact equivalence ────────────────────

    /// <summary>
    /// Single-column (decode) case: the row-parallel kernel must reproduce the serial fallback
    /// bit-for-bit for every affected format.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemvDequantRows_Parallel_IsBitExactWithSerialReference(QuantizationType qt)
        => AssertBitExact(qt, m: 96, k: 512, n: 1, seed: 11);

    /// <summary>
    /// Multi-column (prefill) case. This is where the loop was inverted — the old code walked
    /// tokens outermost and re-decoded the whole matrix per token, the new code walks rows
    /// outermost. A transposed write into <c>c</c> would survive the <c>n == 1</c> test above and
    /// only surface here.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemmDequantRows_MultiColumn_IsBitExactWithSerialReference(QuantizationType qt)
        => AssertBitExact(qt, m: 96, k: 512, n: 5, seed: 22);

    /// <summary>
    /// A row count that is neither a multiple of the partition quantum nor of the worker count,
    /// so the last worker gets a short range and some workers get none. Catches off-by-one
    /// partitioning that a tidy 96/4 split would not.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemmDequantRows_RaggedRowCount_IsBitExactWithSerialReference(QuantizationType qt)
        => AssertBitExact(qt, m: 37, k: 512, n: 3, seed: 33);

    /// <summary>
    /// Below <c>DequantRowsParallelMinRows</c> the kernel runs inline instead of dispatching.
    /// Both branches must produce the same answer.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemmDequantRows_BelowParallelThreshold_IsBitExactWithSerialReference(QuantizationType qt)
        => AssertBitExact(qt, m: 4, k: 512, n: 2, seed: 44);

    /// <summary>
    /// The serial (<c>pool: null</c>) entry point — used by the architectures that have no pool in
    /// scope — must agree with the pooled one exactly.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemmDequantRows_SerialAndPooled_Agree(QuantizationType qt)
    {
        const int m = 96, k = 512, n = 3;
        var fx = new Fixture(qt, m, k, n, seed: 55);
        try
        {
            MatMul.GemmDequantRows(fx.Weights, qt, fx.B, fx.C, m, k, n, _pool);
            MatMul.GemmDequantRows(fx.Weights, qt, fx.B, fx.Expected, m, k, n, pool: null);

            for (int i = 0; i < m * n; i++)
                Assert.Equal(fx.Expected[i], fx.C[i]);
        }
        finally
        {
            fx.Dispose();
        }
    }

    // ──────────────────── Liveness ────────────────────

    /// <summary>
    /// Guards the failure mode a pure equality assertion cannot see: if both the kernel and the
    /// reference wrote nothing, or wrote zeros everywhere, every comparison above would still
    /// pass. Result buffers are poisoned with NaN, so an unwritten element is caught by the
    /// equality checks; this test additionally requires the output to be non-degenerate.
    /// </summary>
    [Theory]
    [MemberData(nameof(FallbackFormats))]
    public void GemmDequantRows_ProducesNonDegenerateOutput(QuantizationType qt)
    {
        const int m = 96, k = 512, n = 3;
        var fx = new Fixture(qt, m, k, n, seed: 66);
        try
        {
            MatMul.GemmDequantRows(fx.Weights, qt, fx.B, fx.C, m, k, n, _pool);

            int nonZero = 0;
            for (int i = 0; i < m * n; i++)
            {
                Assert.False(float.IsNaN(fx.C[i]), $"{qt}: output[{i}] was never written");
                if (fx.C[i] != 0f) nonZero++;
            }

            Assert.True(nonZero > m * n / 2,
                $"{qt}: only {nonZero}/{m * n} outputs are non-zero — the kernel is not doing real work");
        }
        finally
        {
            fx.Dispose();
        }
    }

    // ──────────────────── Helpers ────────────────────

    private void AssertBitExact(QuantizationType qt, int m, int k, int n, int seed)
    {
        var fx = new Fixture(qt, m, k, n, seed);
        try
        {
            SerialReference(fx.Weights, qt, fx.B, fx.Expected, m, k, n);
            MatMul.GemmDequantRows(fx.Weights, qt, fx.B, fx.C, m, k, n, _pool);

            for (int t = 0; t < n; t++)
            {
                for (int i = 0; i < m; i++)
                {
                    int idx = t * m + i;

                    // float.NaN.Equals(float.NaN) is true, so a NaN-for-NaN match would satisfy
                    // the equality below without proving anything. Reject NaN outright.
                    Assert.False(float.IsNaN(fx.Expected[idx]),
                        $"{qt}: reference produced NaN at t={t} row={i} — fixture is degenerate");
                    Assert.True(fx.Expected[idx].Equals(fx.C[idx]),
                        $"{qt} m={m} k={k} n={n}: c[t={t}][row={i}] expected {fx.Expected[idx]}, got {fx.C[idx]}");
                }
            }
        }
        finally
        {
            fx.Dispose();
        }
    }

    /// <summary>
    /// Literal transcription of the pre-#263 fallback: tokens outermost, one weight row decoded
    /// per output element, one <c>TensorPrimitives.Dot</c> per element.
    /// </summary>
    private static void SerialReference(byte* weights, QuantizationType qt, float* b, float* c,
                                        int m, int k, int n)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int t = 0; t < n; t++)
            {
                var xSpan = new ReadOnlySpan<float>(b + t * k, k);
                for (int i = 0; i < m; i++)
                {
                    Dequantize.ToFloat32((nint)weights + i * (nint)rowBytes, k, qt, rowSpan);
                    c[t * m + i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Native buffers for one case. Weight bytes are pseudo-random: the reference and the kernel
    /// share the same decoder, so what the bit patterns decode to is irrelevant to the comparison,
    /// and random bytes exercise every branch of the block decoders rather than a tidy subset.
    /// </summary>
    private sealed class Fixture : IDisposable
    {
        public readonly byte* Weights;
        public readonly float* B;
        public readonly float* C;
        public readonly float* Expected;

        public Fixture(QuantizationType qt, int m, int k, int n, int seed)
        {
            var rng = new Random(seed);
            long rowBytes = Dequantize.RowByteSize(k, qt);

            Weights = (byte*)NativeMemory.AlignedAlloc((nuint)(rowBytes * m), 64);
            for (long i = 0; i < rowBytes * m; i++)
                Weights[i] = FiniteByte(rng);

            B = (float*)NativeMemory.AlignedAlloc((nuint)(n * k * sizeof(float)), 64);
            for (int i = 0; i < n * k; i++)
                B[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            C = AllocPoisoned(m * n);
            Expected = AllocPoisoned(m * n);
        }

        /// <summary>
        /// Draws a random byte that can never act as the high byte of a non-finite
        /// <see cref="Half"/> or <c>bfloat16</c>.
        /// </summary>
        /// <remarks>
        /// Every one of these formats stores at least one 16-bit float scale, at a block offset
        /// that varies per format. Unconstrained random bytes therefore produce NaN/Inf scales,
        /// and NaN outputs would make the bit-exactness assertions vacuous —
        /// <c>float.NaN.Equals(float.NaN)</c> is <see langword="true"/>, so a kernel that wrote
        /// nothing but NaN would compare equal to a reference that did the same. Excluding the 8
        /// byte values whose bits 2..6 are all set (<c>0x7C..0x7F</c>, <c>0xFC..0xFF</c>) makes
        /// every 16-bit float readable anywhere in the buffer finite, at the cost of 3% of the
        /// byte space.
        /// </remarks>
        private static byte FiniteByte(Random rng)
        {
            byte b = (byte)rng.Next(256);
            return (b & 0x7C) == 0x7C ? (byte)(b ^ 0x40) : b;
        }

        private static float* AllocPoisoned(int count)
        {
            float* p = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            for (int i = 0; i < count; i++) p[i] = float.NaN;
            return p;
        }

        public void Dispose()
        {
            NativeMemory.AlignedFree(Weights);
            NativeMemory.AlignedFree(B);
            NativeMemory.AlignedFree(C);
            NativeMemory.AlignedFree(Expected);
        }
    }
}
