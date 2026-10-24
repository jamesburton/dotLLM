using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Profiling-only passthroughs used by two investigations: issue #128 (vectorize I2_S
/// UnpackRowI8) exposes the otherwise-private scalar/AVX2 unpack loop so a benchmark harness can
/// measure its standalone cost vs the full GEMV call, and so tests can force-compare the scalar
/// and vectorized code paths directly regardless of which one
/// <see cref="MatMul.GemvI2_S(byte*, float*, float*, int, int, DotLLM.Cpu.Threading.ComputeThreadPool?)"/>
/// would pick at runtime; issue #196 (decode bandwidth profiling) adds
/// <see cref="BenchStreamingReadOnly"/>, an achievable-bandwidth probe used to determine whether
/// I2_S decode is memory-bandwidth-bound or compute-bound.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Issue #232 — runs the I2_S W2A8 <b>GEMM</b> (prefill) path with the 4x4 register-blocked
    /// inner kernel explicitly on or off, so a harness can A/B both forms <b>within a single run</b>
    /// (baselines drift enough between runs that cross-run ratios are unsound) and can assert the
    /// two are <b>bit-exact</b>. Bypasses the <c>DOTLLM_I2S_TILE</c> gate and the n==1 GEMV
    /// short-circuit; <paramref name="k"/> must be a multiple of 128.
    /// </summary>
    /// <param name="weights">Packed I2_S payload (<c>m·k/4</c> bytes) followed by the per-tensor f32 scale.</param>
    /// <param name="b">F32 activations [n × k], row-major.</param>
    /// <param name="c">F32 output [n × m], row-major.</param>
    /// <param name="m">Weight rows (output features).</param>
    /// <param name="k">Input dimension (multiple of 128).</param>
    /// <param name="n">Token count.</param>
    /// <param name="tiled"><c>true</c> for the register-blocked tile, <c>false</c> for the per-cell baseline.</param>
    /// <param name="threadPool">Optional worker pool; <c>null</c> runs single-threaded.</param>
    public static void BenchGemmI2_SW2A8(byte* weights, float* b, float* c, int m, int k, int n,
                                         bool tiled, DotLLM.Cpu.Threading.ComputeThreadPool? threadPool)
    {
        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);
        GemmI2_SW2A8(weights, b, c, m, k, n, scale, threadPool, tiled);
    }

    /// <summary>True when the 4x4 register-blocked I2_S GEMM tile (issue #232) is the default path on this box.</summary>
    public static bool I2SGemmTileEnabled => I2SGemmTile;

    /// <summary>Times just the UnpackRowI8 loop (AVX2 fast path, or scalar on non-AVX2 hardware) over <paramref name="m"/> rows (no dot).</summary>
    public static void BenchUnpackRowI8Only(byte* weights, int m, int k)
    {
        int rowBytes = k / 4;
        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            {
                for (int r = 0; r < m; r++)
                    UnpackRowI8(weights + (long)r * rowBytes, wI8, k);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Test-only forwarder to the private <see cref="UnpackRowI8"/> (which internally dispatches
    /// AVX2 vs scalar based on <c>Avx2.IsSupported</c> — on this box that's always the AVX2
    /// path). Used by <c>I2SUnpackVectorizedMatchesScalarTests</c> to compare against a hand-rolled
    /// scalar reference of the same documented bit layout, so a subtly-wrong SIMD shuffle/lane
    /// order is caught even though the scalar branch inside <see cref="UnpackRowI8"/> itself is
    /// unreachable on AVX2 hardware.
    /// </summary>
    public static void UnpackRowI8Public(byte* rowPtr, sbyte* dest, int k) => UnpackRowI8(rowPtr, dest, k);

    /// <summary>
    /// Reads every packed byte a full <see cref="GemvI2_S(byte*, float*, float*, int, int, DotLLM.Cpu.Threading.ComputeThreadPool?)"/>
    /// call would touch (<c>m·k/4</c> bytes), doing no unpack/decode — an empirical
    /// achievable-bandwidth probe for this exact access pattern. Returns an XOR checksum so the
    /// JIT cannot eliminate the reads as dead code.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static byte BenchStreamingReadOnly(byte* weights, int m, int k)
    {
        long totalBytes = (long)m * (k / 4);
        Vector256<byte> acc = Vector256<byte>.Zero;
        long i = 0;

        if (Avx2.IsSupported)
        {
            long vecEnd = totalBytes - (totalBytes % 32);
            for (; i < vecEnd; i += 32)
                acc ^= Unsafe.ReadUnaligned<Vector256<byte>>(weights + i);
        }

        Span<byte> lanes = stackalloc byte[32];
        acc.CopyTo(lanes);
        byte result = 0;
        foreach (byte lane in lanes) result ^= lane;

        for (; i < totalBytes; i++)
            result ^= weights[i];

        return result;
    }
}
