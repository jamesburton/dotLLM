using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// 128-bit (SSE2/SSE3) F16 x F32 tier (issue #477 — Westmere / pre-AVX2 hardware, no F16C).
///
/// <para>Without F16C, <c>TensorPrimitives.ConvertToSingle(Half)</c> is already vectorized with
/// integer ops, so the existing convert-into-scratch + <c>TensorPrimitives.Dot</c> GEMV is ~3.3x
/// the scalar <c>(float)Half</c> loop. What it still pays is the scratch round-trip: the fused
/// <see cref="DotF16Sse"/> converts in registers and is a further ~1.7x (Strix Halo Zen 5 under
/// <c>DOTNET_EnableAVX=0</c>, single-threaded, ~10 GB/s of F16 weights).</para>
///
/// <para>For GEMM (n &gt; 1) the tier instead converts each weight row <em>once</em> and dots it
/// against all n activations (<see cref="GemmF16RowsSse"/>), rather than re-converting the row
/// per token.</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Fused F16 x F32 dot: converts Halves eight at a time with the exact integer
    /// <see cref="HalfToSingleSse2"/> and multiplies-adds against <paramref name="x"/>, so the
    /// weight row never round-trips through an F32 scratch buffer. Tail elements (<c>k % 8</c>)
    /// use the software conversion.
    /// </summary>
    /// <remarks>
    /// Measured single-row rather than four rows per lane on purpose: a 4-row variant sharing the
    /// activation loads was equal in cache and ~0.5x on a 57 MB lm_head (4 concurrent row streams),
    /// while the conversion — not the activation load — is the bottleneck either way.
    /// </remarks>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float DotF16Sse(ushort* w, float* x, int k)
    {
        Vector128<float> a = Vector128<float>.Zero, b = Vector128<float>.Zero;
        Vector128<float> c = Vector128<float>.Zero, d = Vector128<float>.Zero;
        Vector128<ushort> zero = Vector128<ushort>.Zero;
        int i = 0;
        for (; i + 15 < k; i += 16)
        {
            Vector128<ushort> h0 = Unsafe.ReadUnaligned<Vector128<ushort>>(w + i);
            Vector128<ushort> h1 = Unsafe.ReadUnaligned<Vector128<ushort>>(w + i + 8);
            a = Sse.Add(a, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackLow(h0, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i)));
            b = Sse.Add(b, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackHigh(h0, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i + 4)));
            c = Sse.Add(c, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackLow(h1, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i + 8)));
            d = Sse.Add(d, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackHigh(h1, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i + 12)));
        }
        a = Sse.Add(a, c);
        b = Sse.Add(b, d);
        for (; i + 7 < k; i += 8)
        {
            Vector128<ushort> h = Unsafe.ReadUnaligned<Vector128<ushort>>(w + i);
            a = Sse.Add(a, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackLow(h, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i)));
            b = Sse.Add(b, Sse.Multiply(HalfToSingleSse2(Sse2.UnpackHigh(h, zero).AsInt32()),
                Unsafe.ReadUnaligned<Vector128<float>>(x + i + 4)));
        }
        float sum = HorizontalSumSse(Sse.Add(a, b));
        for (; i < k; i++)
            sum += (float)BitConverter.UInt16BitsToHalf(w[i]) * x[i];
        return sum;
    }

    /// <summary>F16 GEMV body over rows <c>[0, m)</c> for the SSE tier: y[r] = Σ w[r, i]·x[i].</summary>
    [SkipLocalsInit]
    internal static void GemvF16Sse(ushort* w, float* x, float* y, int m, int k)
    {
        for (int row = 0; row < m; row++)
            y[row] = DotF16Sse(w + (long)row * k, x, k);
    }

    /// <summary>
    /// F16 GEMM over a tile of <paramref name="tileRows"/> weight rows for the SSE tier: each row is
    /// converted once into <paramref name="rowBuf"/> (k floats) and dotted with all
    /// <paramref name="n"/> activation rows. Writes <c>c[t·cStride + row]</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemmF16RowsSse(Half* tileW, int tileRows, float* b, float* c, int cStride,
        int k, int n, float* rowBuf)
    {
        var dest = new Span<float>(rowBuf, k);
        for (int row = 0; row < tileRows; row++)
        {
            System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                new ReadOnlySpan<Half>(tileW + (long)row * k, k), dest);
            for (int t = 0; t < n; t++)
                c[(long)t * cStride + row] = System.Numerics.Tensors.TensorPrimitives.Dot(
                    (ReadOnlySpan<float>)dest, new ReadOnlySpan<float>(b + (long)t * k, k));
        }
    }

    /// <summary>True when the F16 kernels should take the 128-bit tier (no AVX2; SSE3 present).</summary>
    private static bool F16SseTier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => !Avx2.IsSupported && Sse3.IsSupported;
    }
}
