using System.Numerics.Tensors;
using System.Runtime.Intrinsics;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Platform-independent <c>float</c> transcendentals for building lookup tables (#1309). <see cref="MathF"/> forwards to the C
/// runtime (UCRT on Windows, glibc on Linux, ...), and those differ by up to 1 ulp for some inputs. In a RoPE table that ulp is
/// harmless in F32, but quantized models quantize their activations (Q8_0 / Q8_1), which digitizes the drift into a different
/// int8 or a different MoE expert: <c>FlintSyntheticGoldenTests.Quantized_Forward_Matches_LlamaCpp_Golden</c> was 8e-6 from
/// llama.cpp on Windows and 0.0199 on Linux CI from the RoPE tables alone. Evaluating in <see cref="double"/> and rounding once
/// to float is correctly rounded in all but vanishingly rare cases, and is the same answer on every OS. Table-build only: do not
/// use on a per-token hot path.
/// </summary>
internal static class PortableMath
{
    /// <summary><c>e^x</c> evaluated in double, rounded once to float.</summary>
    internal static float Exp(float x) => (float)Math.Exp(x);

    /// <summary><c>x^y</c> evaluated in double, rounded once to float.</summary>
    internal static float Pow(float x, float y) => (float)Math.Pow(x, y);

    /// <summary><c>cos(x)</c> evaluated in double, rounded once to float.</summary>
    internal static float Cos(float x) => (float)Math.Cos(x);

    /// <summary><c>sin(x)</c> evaluated in double, rounded once to float.</summary>
    internal static float Sin(float x) => (float)Math.Sin(x);

    /// <summary>
    /// Element-wise <c>exp</c> that is the same on every OS for every span length (#1309). <see cref="TensorPrimitives.Exp(ReadOnlySpan{float}, Span{float})"/>
    /// is vectorized only for spans of at least one 128-bit vector (and only when SIMD is hardware accelerated); anything
    /// shorter falls back to <c>MathF.Exp</c>, i.e. the C runtime's <c>expf</c>, which differs between UCRT and glibc by an
    /// ulp. An attention row of 1..3 keys (the first tokens of a prompt) hit exactly that fallback, and the ulp was digitized
    /// into a different int8 by the next quantized GEMM: <c>FlintSyntheticGoldenTests.Quantized_Forward_Matches_LlamaCpp_Golden</c>
    /// failed at 0.0199 on Linux CI and passed on Windows. Short spans take the double-evaluated <see cref="Exp(float)"/>.
    /// </summary>
    internal static void Exp(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<float>.Count)
        {
            TensorPrimitives.Exp(x, destination);
            return;
        }
        if (destination.Length < x.Length) throw new ArgumentException("Destination is too short.", nameof(destination));
        for (int i = 0; i < x.Length; i++)
            destination[i] = Exp(x[i]);
    }
}
