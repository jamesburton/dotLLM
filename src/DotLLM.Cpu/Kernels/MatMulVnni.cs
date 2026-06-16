using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// AVX-512 / AVX VNNI (<c>vpdpbusd</c>) fused integer quantized matmul paths.
///
/// <para>
/// <b>Why VNNI.</b> The baseline Q8_0×Q8_0 dot (see <see cref="MatMul.VecDotQ8_0Avx2"/>)
/// emulates a signed×signed int8 dot with the abs/sign trick plus two
/// <c>vpmaddubsw</c>/<c>vpmaddwd</c> instructions per 32-lane block. VNNI replaces that with
/// a single <c>vpdpbusd</c> (unsigned×signed int8 → int32 accumulate, 32 MACs/instruction),
/// fusing the dequant-scale apply into the same loop and removing the int16 intermediate.
/// </para>
///
/// <para>
/// <b>Sign handling.</b> <c>vpdpbusd</c> needs an <em>unsigned</em> left operand and a
/// <em>signed</em> right operand, but both Q8_0 weights and the Q8_0/Q8_1 activation are
/// signed. We bias the activation to unsigned with the standard +128 trick:
/// <c>xu_i = qx_i + 128 ∈ [1,255]</c> (computed as <c>vx XOR 0x80</c>), so
/// <c>qx_i = xu_i − 128</c> and
/// <c>Σ qw_i·qx_i = Σ qw_i·xu_i − 128·Σ qw_i</c>.
/// The first term is <c>vpdpbusd(xu, qw)</c>; the correction <c>128·Σ qw_i</c> is a second
/// <c>vpdpbusd(const 128, qw)</c>. Both accumulate int32; the per-block fp scale
/// <c>d_w·d_x</c> is applied once per block in fp32. Validated bit-exact vs the scalar
/// integer reference.
/// </para>
///
/// <para>
/// <b>Hardware note (Zen 5 / Strix Halo).</b> On this box .NET 10 exposes 512-bit VNNI only
/// through <c>Avx10v1.V512</c>, which reports <c>IsSupported=false</c> here; the available
/// path is 256-bit <see cref="AvxVnni"/> (the CPU runs AVX-512 at 256-bit datapath width
/// anyway). One <c>vpdpbusd</c> on a YMM register consumes exactly one 32-byte Q8_0 block,
/// which maps cleanly onto the block layout with no tail-splitting inside a block.
/// </para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>XOR mask that converts a signed int8 to unsigned via +128 (wraparound add).</summary>
    private static readonly Vector256<byte> BiasXor80 = Vector256.Create((byte)0x80);

    /// <summary>Constant 128 used to recover the <c>−128·Σ qw</c> correction term.</summary>
    private static readonly Vector256<byte> Const128 = Vector256.Create((byte)128);

    /// <summary>
    /// Whether the VNNI fused integer Q8_0 path is usable at runtime.
    /// Requires 256-bit <c>vpdpbusd</c> (<see cref="AvxVnni"/>).
    /// </summary>
    internal static bool IsQ8_0VnniSupported
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => AvxVnni.IsSupported;
    }

    /// <summary>
    /// VNNI Q8_0×Q8_0 dot product for a single weight row.
    /// One <c>vpdpbusd</c> pair per 32-element block; fp scale applied per block.
    /// </summary>
    /// <param name="a">Q8_0 weight row: blockCount blocks of 34 bytes.</param>
    /// <param name="b">Q8_0 activation: blockCount blocks of 34 bytes.</param>
    /// <param name="blockCount">Number of 32-element blocks.</param>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Vnni(byte* a, byte* b, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> c128 = Const128;
        Vector256<byte> bias = BiasXor80;

        for (int block = 0; block < blockCount; block++)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float dw = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float dx = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock + 2);
            Vector256<byte> vx = Unsafe.ReadUnaligned<Vector256<byte>>(bBlock + 2);

            // Bias activation to unsigned: xu = qx + 128.
            Vector256<byte> xu = vx ^ bias;

            // sum(xu*qw) − 128*sum(qw) = sum(qx*qw).
            Vector256<int> pos = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, xu, vw);
            Vector256<int> neg = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, c128, vw);
            Vector256<int> isum = pos - neg;

            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            Vector256<float> scale = Vector256.Create(dw * dx);
            acc = Fma.MultiplyAdd(scale, fsum, acc);
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// VNNI Q8_0×Q8_0 dot product computing 4 weight rows against one shared activation.
    /// The biased activation and its <c>vpdpbusd</c> are reused; only the weight load and the
    /// two dot products differ per row.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Vnni_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* x, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;
        Vector256<byte> c128 = Const128;
        Vector256<byte> bias = BiasXor80;

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
            Vector256<byte> vx = Unsafe.ReadUnaligned<Vector256<byte>>(xBlock + 2);
            Vector256<byte> xu = vx ^ bias;

            ProcessVnniBlock(w0, block, xu, c128, dx, ref acc0);
            ProcessVnniBlock(w1, block, xu, c128, dx, ref acc1);
            ProcessVnniBlock(w2, block, xu, c128, dx, ref acc2);
            ProcessVnniBlock(w3, block, xu, c128, dx, ref acc3);
        }

        results[0] = HorizontalSumAvx2Float(acc0);
        results[1] = HorizontalSumAvx2Float(acc1);
        results[2] = HorizontalSumAvx2Float(acc2);
        results[3] = HorizontalSumAvx2Float(acc3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessVnniBlock(
        byte* w, int block, Vector256<byte> xu, Vector256<byte> c128, float dx,
        ref Vector256<float> acc)
    {
        byte* wBlock = w + block * Q8_0BlockBytes;
        float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
        Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);

        Vector256<int> pos = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, xu, vw);
        Vector256<int> neg = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, c128, vw);
        Vector256<int> isum = pos - neg;

        Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
        Vector256<float> scale = Vector256.Create(dw * dx);
        acc = Fma.MultiplyAdd(scale, fsum, acc);
    }

    /// <summary>
    /// VNNI variant of <see cref="ComputeRows(byte*, byte*, float*, int, int)"/>.
    /// Processes rows in groups of 4, reusing the biased activation across the group.
    /// Caller must have verified <see cref="IsQ8_0VnniSupported"/>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsVnni(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount)
    {
        int rowBytes = blockCount * Q8_0BlockBytes;
        int row = 0;
        for (; row + 3 < m; row += 4)
        {
            VecDotQ8_0Vnni_4Rows(
                weightsQ8 + row * rowBytes,
                weightsQ8 + (row + 1) * rowBytes,
                weightsQ8 + (row + 2) * rowBytes,
                weightsQ8 + (row + 3) * rowBytes,
                xQ8, blockCount, result + row);
        }
        for (; row < m; row++)
            result[row] = VecDotQ8_0Vnni(weightsQ8 + row * rowBytes, xQ8, blockCount);
    }
}
