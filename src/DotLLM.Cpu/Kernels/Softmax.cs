using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Numerically stable softmax kernel: <c>softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))</c>.
/// Used in attention score normalization and final logit-to-probability conversion.
/// </summary>
public static class Softmax
{
    /// <summary>
    /// Computes softmax over the input span. Output sums to 1.0.
    /// Uses max-subtraction for numerical stability before exponentiation.
    /// </summary>
    /// <param name="input">Input span (logits).</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="input"/>.Length.
    /// May alias <paramref name="input"/> for in-place operation.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, Span<float> result)
    {
        // Subtract max for numerical stability, then exp.
        float max = TensorPrimitives.Max(input);
        TensorPrimitives.Add(input, -max, result);
        TensorPrimitives.Exp(result, result);

        // Normalize by sum.
        float sum = TensorPrimitives.Sum(result);
        TensorPrimitives.Multiply(result, 1.0f / sum, result);
    }

    /// <summary>
    /// Single-pass softmax for attention scores, built on <see cref="FastMath.ExpSumAndStore"/>
    /// (fused shift + exp + store + sum) rather than separate Add/Exp/Sum passes.
    /// <para>
    /// The name is historical: this path used an IEEE-754 bit-manipulation approximation of
    /// <c>exp</c> on the premise that "exp errors get normalized away when dividing by the sum".
    /// #501 measured that premise false (the error is relative and per-element, so it survives
    /// normalization) and made the exp precise by default; the approximation now only returns
    /// under <c>DOTLLM_FAST_EXP=1</c>. "Fast" here means fewer passes, not lower precision.
    /// Standard <see cref="Execute"/> should still be used for sampling.
    /// </para>
    /// </summary>
    /// <param name="input">Input span (logits).</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="input"/>.Length.
    /// May alias <paramref name="input"/> for in-place operation.</param>
    [SkipLocalsInit]
    public static void ExecuteFast(ReadOnlySpan<float> input, Span<float> result)
    {
        float max = TensorPrimitives.Max(input);
        float sum = FastMath.ExpSumAndStore(input, result, -max);
        TensorPrimitives.Multiply(result, 1.0f / sum, result);
    }

    /// <summary>
    /// Scalar reference implementation for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> input, Span<float> result)
    {
        // Find max for numerical stability.
        float max = float.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] > max) max = input[i];
        }

        // exp(x - max) and accumulate sum.
        float sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = MathF.Exp(input[i] - max);
            sum += result[i];
        }

        // Normalize.
        float invSum = 1.0f / sum;
        for (int i = 0; i < input.Length; i++)
            result[i] *= invSum;
    }
}
