using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Perf comparison for issue #204's PQ2_0 W2A8 (AVX2/AVX-VNNI) SIMD tier vs the pre-existing
/// scalar (unpack-to-float + <c>TensorPrimitives.Dot</c>) tier. Follows
/// <see cref="I2SDecodeBandwidthProfileBench"/>'s pattern: <c>Category=Benchmark</c> (excluded
/// from CI per <c>.github/workflows/ci.yml</c>, added in #196), xUnit <c>[Theory]</c>,
/// <see cref="Stopwatch"/>, median-of-N trials to smooth this box's known run-to-run noise (see
/// #202's closing comment — same box, same lesson).
///
/// <para>Shapes are real PrismML Bonsai-27B GEMV/GEMM decode dims (hidden=5120,
/// ffn=17408 — <c>qwen35.embedding_length</c>/<c>qwen35.feed_forward_length</c>, same source as
/// <c>CudaPQ2_0GemvTest</c>'s doc comment), mirroring the attention/ffn_gate/ffn_down shape
/// triple <see cref="I2SDecodeBandwidthProfileBench"/> uses for I2_S.</para>
///
/// <para>The scalar tier is measured by directly building a packed buffer and running it through
/// the pre-#204 code path is no longer reachable via the public <see cref="MatMul.GemvPQ2_0"/>
/// entry point on AVX2 hardware (it now always dispatches to W2A8) — so the scalar baseline here
/// calls <see cref="MatMul.GemvPQ2_0Scalar"/>, an internal test-only forwarding shim added
/// alongside this benchmark that always takes the float-unpack + <c>TensorPrimitives.Dot</c>
/// path regardless of ISA, letting the two tiers be compared head-to-head on the same box.</para>
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class PQ2_0DecodeBenchmark
{
    private readonly ITestOutputHelper _output;

    public PQ2_0DecodeBenchmark(ITestOutputHelper output) => _output = output;

    private const int Trials = 5;
    private const int IterationsPerTrial = 20;

    [Theory]
    [InlineData(5120, 5120, "attn_qproj (hidden×hidden)")]
    [InlineData(17408, 5120, "ffn_gate (ffn×hidden)")]
    [InlineData(5120, 17408, "ffn_down (hidden×ffn)")]
    public void GemvPQ2_0_W2A8VsScalar_MedianOf5(int m, int k, string label)
    {
        var rng = new Random(204);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long weightBytes = (long)m * rowBytes;

        sbyte[] ternary = new sbyte[(long)m * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[m * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;

        byte* weights = Pack(ternary, groupScales, m, k);
        float* x = (float*)NativeMemory.AllocZeroed((nuint)(k * sizeof(float)));
        float* resultSimd = (float*)NativeMemory.AllocZeroed((nuint)(m * sizeof(float)));
        float* resultScalar = (float*)NativeMemory.AllocZeroed((nuint)(m * sizeof(float)));

        try
        {
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            // Warm up (JIT) both tiers.
            MatMul.GemvPQ2_0(weights, x, resultSimd, m, k, null);
            MatMul.GemvPQ2_0Scalar(weights, x, resultScalar, m, k);

            // Correctness sanity within this benchmark: both tiers must broadly agree (mean
            // relative error, aggregated over all m rows) before we trust the timing numbers.
            // A per-element bound (as PQ2_0Tests.cs's small-shape correctness tests use) is too
            // strict at these large real-decode row counts: with m in the thousands, some rows
            // land near a true-sum cancellation (scalar value near zero), where Q8_0's per-block
            // quant noise dominates and blows up the *relative* error for that one row without
            // indicating an actual bug — PQ2_0Tests.cs already carries the precise per-shape
            // correctness bar; this check just guards against a gross mismatch here.
            AssertMeanRelativeErrorWithinBound(resultSimd, resultScalar, m, label);

            double simdMedianMs = MedianTrialMs(() => MatMul.GemvPQ2_0(weights, x, resultSimd, m, k, null));
            double scalarMedianMs = MedianTrialMs(() => MatMul.GemvPQ2_0Scalar(weights, x, resultScalar, m, k));

            double speedup = scalarMedianMs / simdMedianMs;

            _output.WriteLine($"[{label}] m={m} k={k} weightBytes={weightBytes} " +
                               $"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                               $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");
            _output.WriteLine($"  scalar (median of {Trials}): {scalarMedianMs:F4} ms/call");
            _output.WriteLine($"  W2A8 SIMD (median of {Trials}): {simdMedianMs:F4} ms/call");
            _output.WriteLine($"  speedup: {speedup:F2}x");
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(x);
            NativeMemory.Free(resultSimd);
            NativeMemory.Free(resultScalar);
        }
    }

    [Theory]
    [InlineData(5120, 5120, 8, "attn_qproj (n=8 tokens)")]
    [InlineData(17408, 5120, 8, "ffn_gate (n=8 tokens)")]
    public void GemmPQ2_0_W2A8VsScalar_MedianOf5(int m, int k, int n, string label)
    {
        var rng = new Random(204 + n);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;

        sbyte[] ternary = new sbyte[(long)m * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[m * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;

        byte* weights = Pack(ternary, groupScales, m, k);
        float* b = (float*)NativeMemory.AllocZeroed((nuint)((long)n * k * sizeof(float)));
        float* cSimd = (float*)NativeMemory.AllocZeroed((nuint)((long)n * m * sizeof(float)));
        float* cScalar = (float*)NativeMemory.AllocZeroed((nuint)((long)n * m * sizeof(float)));

        try
        {
            for (long i = 0; i < (long)n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            MatMul.GemmPQ2_0(weights, b, cSimd, m, k, n, null);
            MatMul.GemmPQ2_0Scalar(weights, b, cScalar, m, k, n);

            // See GemvPQ2_0_W2A8VsScalar_MedianOf5's identical comment: aggregate mean-relative
            // check, not a strict per-element bound (near-zero-cancellation rows at these row
            // counts make a per-element relative bound unreliable; PQ2_0Tests.cs owns the precise
            // per-shape correctness bar).
            AssertMeanRelativeErrorWithinBound(cSimd, cScalar, (long)n * m, label);

            double simdMedianMs = MedianTrialMs(() => MatMul.GemmPQ2_0(weights, b, cSimd, m, k, n, null));
            double scalarMedianMs = MedianTrialMs(() => MatMul.GemmPQ2_0Scalar(weights, b, cScalar, m, k, n));

            double speedup = scalarMedianMs / simdMedianMs;

            _output.WriteLine($"[{label}] m={m} k={k} n={n}");
            _output.WriteLine($"  scalar (median of {Trials}): {scalarMedianMs:F4} ms/call");
            _output.WriteLine($"  W2A8 SIMD (median of {Trials}): {simdMedianMs:F4} ms/call");
            _output.WriteLine($"  speedup: {speedup:F2}x");
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(b);
            NativeMemory.Free(cSimd);
            NativeMemory.Free(cScalar);
        }
    }

    /// <summary>
    /// Aggregate sanity check: mean(|diff|) / mean(|scalar|) across all elements must be small.
    /// Bounds overall Q8_0-quant-induced drift between the two tiers without being thrown off by
    /// individual near-zero-cancellation elements the way a per-element relative bound would be.
    /// </summary>
    private static void AssertMeanRelativeErrorWithinBound(float* simd, float* scalar, long count, string label)
    {
        double sumAbsDiff = 0, sumAbsScalar = 0;
        for (long i = 0; i < count; i++)
        {
            sumAbsDiff += MathF.Abs(simd[i] - scalar[i]);
            sumAbsScalar += MathF.Abs(scalar[i]);
        }
        double meanAbsDiff = sumAbsDiff / count;
        double meanAbsScalar = sumAbsScalar / count;
        double relError = meanAbsDiff / meanAbsScalar;

        // Q8_0 is ~1/127 relative quant step per activation element; a few-percent aggregate
        // relative error comfortably separates "expected quant noise" from an actual bug.
        Assert.True(relError <= 0.05,
            $"[{label}] mean relative error {relError:P2} exceeds 5% (meanAbsDiff={meanAbsDiff:E4}, meanAbsScalar={meanAbsScalar:E4})");
    }

    private static double MedianTrialMs(Action body)
    {
        double[] trialMs = new double[Trials];
        for (int t = 0; t < Trials; t++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < IterationsPerTrial; i++) body();
            sw.Stop();
            trialMs[t] = sw.Elapsed.TotalMilliseconds / IterationsPerTrial;
        }
        Array.Sort(trialMs);
        return trialMs[Trials / 2];
    }

    /// <summary>Packs ternary {-1,0,+1} into dotLLM PQ2_0 layout: per-128-group Half scale + codes.</summary>
    private static byte* Pack(sbyte[] ternary, float[] groupScales, int m, int k)
    {
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)((long)m * rowBytes));

        for (int r = 0; r < m; r++)
        {
            byte* rowBase = buf + (long)r * rowBytes;
            for (int g = 0; g < groupsPerRow; g++)
            {
                byte* groupBase = rowBase + g * 34;
                *(Half*)groupBase = (Half)groupScales[r * groupsPerRow + g];
                byte* codes = groupBase + 2;
                int baseIdx = r * k + g * 128;
                for (int gp = 0; gp < 32; gp++)
                {
                    byte c0 = (byte)(ternary[baseIdx + gp] + 1);
                    byte c1 = (byte)(ternary[baseIdx + gp + 32] + 1);
                    byte c2 = (byte)(ternary[baseIdx + gp + 64] + 1);
                    byte c3 = (byte)(ternary[baseIdx + gp + 96] + 1);
                    codes[gp] = (byte)((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
                }
            }
        }
        return buf;
    }
}
