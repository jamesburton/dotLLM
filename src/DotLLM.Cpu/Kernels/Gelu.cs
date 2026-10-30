using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Tanh-approximate GELU (<c>ggml_gelu</c>: <c>0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))</c>) for BERT-class
/// encoders (#1403): vectorised through <see cref="TensorPrimitives"/> (shipping .NET 10 intrinsics only) in L1-sized
/// tiles and split across the compute pool.
/// </summary>
public static class Gelu
{
    private const float SqrtTwoOverPi = 0.79788456080286535588f;
    private const float Coeff = 0.044715f;
    private const int Tile = 512;           // 3 x 2 KiB scratch stays in L1
    private const int MinParallel = 1 << 14; // below this the dispatch costs more than it saves

    /// <summary>In-place tanh-approximate GELU over <paramref name="count"/> floats, optionally parallel on <paramref name="pool"/>.</summary>
    [SkipLocalsInit]
    public static unsafe void ExecuteTanh(float* data, int count, ComputeThreadPool? pool)
    {
        if (pool is null || pool.ThreadCount < 2 || count < MinParallel)
        {
            Range(data, 0, count);
            return;
        }
        var ctx = new Ctx { Data = data, Count = count };
        pool.Dispatch((nint)(&ctx), &Worker);
    }

    private unsafe struct Ctx { public float* Data; public int Count; }

    [SkipLocalsInit]
    private static unsafe void Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<Ctx>((void*)ctxPtr);
        int tiles = (ctx.Count + Tile - 1) / Tile;
        ComputeThreadPool.PartitionRange(tiles, threadIdx, threadCount, out int t0, out int t1);
        if (t0 >= t1) return;
        Range(ctx.Data, t0 * Tile, Math.Min(ctx.Count, t1 * Tile));
    }

    [SkipLocalsInit]
    private static unsafe void Range(float* data, int start, int end)
    {
        Span<float> a = stackalloc float[Tile];
        Span<float> b = stackalloc float[Tile];
        for (int i = start; i < end; i += Tile)
        {
            int n = Math.Min(Tile, end - i);
            var x = new Span<float>(data + i, n);
            var inner = a.Slice(0, n);
            var t = b.Slice(0, n);
            // inner = sqrt(2/pi) * x * (1 + 0.044715 x^2)
            TensorPrimitives.Multiply(x, x, inner);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)inner, Coeff, inner);
            TensorPrimitives.Add((ReadOnlySpan<float>)inner, 1.0f, inner);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)inner, (ReadOnlySpan<float>)x, inner);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)inner, SqrtTwoOverPi, inner);
            TensorPrimitives.Tanh((ReadOnlySpan<float>)inner, t);
            // x = x * (0.5 * tanh + 0.5)
            TensorPrimitives.Multiply((ReadOnlySpan<float>)t, 0.5f, t);
            TensorPrimitives.Add((ReadOnlySpan<float>)t, 0.5f, t);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)x, (ReadOnlySpan<float>)t, x);
        }
    }

    /// <summary>Scalar reference: the loop <c>BertEncoderModel</c> used before #1403 (<c>MathF.Tanh</c>).</summary>
    internal static void ExecuteTanhScalar(ReadOnlySpan<float> input, Span<float> result)
    {
        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            result[i] = 0.5f * x * (1.0f + MathF.Tanh(SqrtTwoOverPi * x * (1.0f + Coeff * x * x)));
        }
    }
}
