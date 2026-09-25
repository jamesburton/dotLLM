using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #232 — opt-in prefill-scale A/B for the 4x4 register-blocked I2_S W2A8 GEMM tile.
/// Enable with <c>DOTLLM_I2S_TILE_BENCH=1</c>.
///
/// <para><b>Methodology (dictated by upstream kkokosa/dotLLM#416).</b></para>
/// <list type="bullet">
/// <item><b>Never L1-resident.</b> #416 rates L1-resident microbenchmarks invalid for this question
/// — the same 2x4 tile measured 0.97 (no effect) L1-resident and 0.86 at prefill scale. Every shape
/// below is a real BitNet-2B-4T projection at N=256 tokens, so the Q8_0 activation panel alone is
/// 2.2-6.0 MB (past this box's 48 KB L1D and 1 MB L2) and the weight panel 3.4-4.4 MB.</item>
/// <item><b>Within-run comparison only.</b> #416 records the same baseline at 380.6us and 425.5us
/// across runs. Both variants are therefore measured inside one process, interleaved
/// (A,B,B,A per round) so any monotonic clock ramp or thermal drift is shared, and the reported
/// figure is the per-variant <i>minimum</i> over all rounds.</item>
/// </list>
/// </summary>
public sealed unsafe class I2SRegisterBlockedGemmBench
{
    private readonly ITestOutputHelper _output;

    public I2SRegisterBlockedGemmBench(ITestOutputHelper output) => _output = output;

    // Real BitNet-2B-4T projection shapes (hidden=2560, ffn=6912) at prefill N=256.
    private static readonly (string Tag, int M, int K, int N)[] Shapes =
    [
        ("attn qkv/o  (M=2560 K=2560 N=256)", 2560, 2560, 256),
        ("ffn up/gate (M=6912 K=2560 N=256)", 6912, 2560, 256),
        ("ffn down    (M=2560 K=6912 N=256)", 2560, 6912, 256),
    ];

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Bench_I2SGemmRegisterBlockTile(bool threaded)
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_I2S_TILE_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_I2S_TILE_BENCH=1 to enable this benchmark.");

        using ComputeThreadPool? pool = threaded ? new ComputeThreadPool(Environment.ProcessorCount / 2) : null;

        _output.WriteLine($"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                          $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported} " +
                          $"AVX512F={System.Runtime.Intrinsics.X86.Avx512F.IsSupported} " +
                          $"tile-default={MatMul.I2SGemmTileEnabled} " +
                          $"threads={(pool is null ? 1 : Environment.ProcessorCount / 2)}");

        foreach (var (tag, m, k, n) in Shapes)
            RunShape(tag, m, k, n, pool);
    }

    /// <summary>
    /// Mechanism decomposition. Two effects can produce a win here and the issue explicitly warns
    /// they must not be assumed to co-hold:
    /// <list type="number">
    /// <item><b>Scale/ALU amortization</b> — the Q8_0 activation scale <c>d_b</c> is row-invariant
    /// for ternary weights, so its load + <c>Half</c>→float convert + broadcast is paid once per
    /// (block, token) instead of once per cell. This effect is present even when everything is
    /// L1-resident, and it does not grow with N.</item>
    /// <item><b>Activation-panel traffic</b> — the untiled loop re-streams the whole Q8_0 activation
    /// panel once per weight row; the 4-row tile divides that by 4. This effect is zero at small N
    /// (panel stays in L1) and grows as the panel outgrows the caches.</item>
    /// </list>
    /// Sweeping N at fixed M,K separates them: an L1-resident small-N point isolates (1), and the
    /// growth from there to N=256 is (2).
    /// </summary>
    [SkippableFact]
    public void Bench_I2SGemmTile_MechanismDecomposition()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_I2S_TILE_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_I2S_TILE_BENCH=1 to enable this benchmark.");

        _output.WriteLine("N-sweep at M=2560 K=2560, single-threaded. Q8_0 activation panel = N*2720 B.");
        foreach (int n in new[] { 4, 8, 16, 32, 64, 128, 256 })
            RunShape($"N={n,-4} panel={n * 2720 / 1024.0,7:F1} KiB", 2560, 2560, n, null);
    }

    private void RunShape(string tag, int m, int k, int n, ComputeThreadPool? pool)
    {
        const int rounds = 7;
        var rng = new Random(232);
        int rowBytes = k / 4;
        long weightBytes = (long)m * rowBytes + 4;

        byte* weights = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);
        float* b = (float*)NativeMemory.AllocZeroed((nuint)((long)n * k * sizeof(float)));
        float* cBase = (float*)NativeMemory.AllocZeroed((nuint)((long)m * n * sizeof(float)));
        float* cTile = (float*)NativeMemory.AllocZeroed((nuint)((long)m * n * sizeof(float)));
        try
        {
            byte[] rand = new byte[weightBytes];
            rng.NextBytes(rand);
            Marshal.Copy(rand, 0, (nint)weights, (int)weightBytes);
            *(float*)(weights + (long)m * rowBytes) = 0.0217f;
            for (long i = 0; i < (long)n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Warm-up / JIT both variants before any timing.
            MatMul.BenchGemmI2_SW2A8(weights, b, cBase, m, k, n, tiled: false, pool);
            MatMul.BenchGemmI2_SW2A8(weights, b, cTile, m, k, n, tiled: true, pool);

            double baseMin = double.MaxValue, tileMin = double.MaxValue;
            double baseSum = 0, tileSum = 0;

            for (int round = 0; round < rounds; round++)
            {
                // Interleaved A,B,B,A: each variant gets one early and one late slot per round,
                // so a monotonic clock ramp cannot favour either.
                double a1 = Time(() => MatMul.BenchGemmI2_SW2A8(weights, b, cBase, m, k, n, false, pool));
                double b1 = Time(() => MatMul.BenchGemmI2_SW2A8(weights, b, cTile, m, k, n, true, pool));
                double b2 = Time(() => MatMul.BenchGemmI2_SW2A8(weights, b, cTile, m, k, n, true, pool));
                double a2 = Time(() => MatMul.BenchGemmI2_SW2A8(weights, b, cBase, m, k, n, false, pool));

                baseMin = Math.Min(baseMin, Math.Min(a1, a2));
                tileMin = Math.Min(tileMin, Math.Min(b1, b2));
                baseSum += a1 + a2;
                tileSum += b1 + b2;
            }

            // Exactness is re-asserted here so a perf number can never be reported for a kernel
            // that silently diverged under these (larger) shapes.
            long mismatches = 0;
            for (long i = 0; i < (long)m * n; i++)
                if (BitConverter.SingleToInt32Bits(cBase[i]) != BitConverter.SingleToInt32Bits(cTile[i]))
                    mismatches++;

            double actKiB = (long)n * (k / 32) * 34 / 1024.0;
            double wKiB = weightBytes / 1024.0;
            double baseAvg = baseSum / (2 * rounds), tileAvg = tileSum / (2 * rounds);

            _output.WriteLine($"[{tag}] weights={wKiB / 1024:F1} MB  q8-activations={actKiB / 1024:F1} MB  bit-exact={mismatches == 0}");
            _output.WriteLine($"    baseline  min={baseMin:F2} ms  avg={baseAvg:F2} ms");
            _output.WriteLine($"    4x4 tile  min={tileMin:F2} ms  avg={tileAvg:F2} ms");
            _output.WriteLine($"    speedup (min/min)={baseMin / tileMin:F3}x   (avg/avg)={baseAvg / tileAvg:F3}x");
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(b);
            NativeMemory.Free(cBase);
            NativeMemory.Free(cTile);
        }
    }

    private static double Time(Action action)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }
}
