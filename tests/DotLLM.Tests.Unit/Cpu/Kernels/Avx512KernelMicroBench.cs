using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Directional microbenchmark: times each new AVX-512 quant/dequant path against its AVX2 sibling on the
/// same buffer, to confirm the AVX-512 variants are at least not slower (the gaps are bandwidth-light slivers,
/// so a small win or rough parity is the expected — and acceptable — outcome). Stopwatch-based (min of several
/// rounds), not BenchmarkDotNet — adequate for direction, not for precise ratios. Opt-in via
/// <c>DOTLLM_RUN_KERNEL_BENCH</c>; runs only on AVX-512 hardware.
/// </summary>
public sealed unsafe class Avx512KernelMicroBench
{
    private const int Elements = 8192;       // multiple of 256 (Q8_K) and 32 (others); ~one large row
    private const int Iters = 4000;          // calls per timed round
    private const int Rounds = 6;            // take the min round (least noise, fully tiered)

    [SkippableFact]
    public void Avx512_VsAvx2_Direction()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_KERNEL_BENCH")),
            "Kernel microbench is opt-in — set DOTLLM_RUN_KERNEL_BENCH=1 to run.");
        Skip.IfNot(Avx512F.IsSupported, "AVX-512F not available.");

        var rng = new Random(42);
        float[] f = new float[Elements];
        float[] w = new float[Elements];
        for (int i = 0; i < Elements; i++) { f[i] = (float)(rng.NextDouble() * 2 - 1) * 5f; w[i] = (float)(rng.NextDouble() * 2 - 1); }

        byte[] q8 = new byte[Elements / 32 * 34];   // Q8_0 (KV) blocks
        byte[] q4 = new byte[Elements / 32 * 18];   // Q4_0 blocks
        float[] outF = new float[Elements];
        byte[] outQ4 = new byte[Elements / 32 * 18];
        byte[] outFusedQ8_0 = new byte[Elements / 32 * 34];
        byte[] outFusedQ8_1 = new byte[Elements / 32 * 36];
        byte[] outFusedQ8_K = new byte[Elements / 256 * 292];

        fixed (float* fp = f)
        fixed (byte* q8p = q8)
        fixed (byte* q4p = q4)
        {
            KvQuantize.F32ToQ8_0(fp, q8p, Elements);
            KvQuantize.F32ToQ4_0Scalar(fp, q4p, Elements);
        }

        Console.WriteLine($"[Avx512KernelMicroBench] elements={Elements} iters={Iters} rounds={Rounds} (min ns/call)");
        Console.WriteLine($"  {"kernel",-26} {"AVX2",10} {"AVX-512",10}  speedup");

        fixed (float* fp = f)
        fixed (byte* q8p = q8)
        fixed (byte* q4p = q4)
        fixed (float* ofp = outF)
        fixed (byte* oq4 = outQ4)
        fixed (byte* of80 = outFusedQ8_0)
        fixed (byte* of81 = outFusedQ8_1)
        fixed (byte* of8k = outFusedQ8_K)
        {
            // Lambdas can't capture pointer-typed locals — capture pinned addresses as nint and re-cast
            // inside the lambda body (legal in this unsafe context; pins are held for the whole fixed block).
            nint q8a = (nint)q8p, q4a = (nint)q4p, fa = (nint)fp, ofa = (nint)ofp;
            nint oq4a = (nint)oq4, of80a = (nint)of80, of81a = (nint)of81, of8ka = (nint)of8k;

            Report("KvQuantize.Q8_0ToF32",
                () => KvQuantize.Q8_0ToF32Avx2((byte*)q8a, (float*)ofa, Elements),
                () => KvQuantize.Q8_0ToF32Avx512((byte*)q8a, (float*)ofa, Elements));

            Report("KvQuantize.Q4_0ToF32",
                () => KvQuantize.Q4_0ToF32Avx2((byte*)q4a, (float*)ofa, Elements),
                () => KvQuantize.Q4_0ToF32Avx512((byte*)q4a, (float*)ofa, Elements));

            Report("KvQuantize.F32ToQ4_0",
                () => KvQuantize.F32ToQ4_0Avx2((float*)fa, (byte*)oq4a, Elements),
                () => KvQuantize.F32ToQ4_0Avx512((float*)fa, (byte*)oq4a, Elements));

            Report("FusedOps.RmsNormQ8_0",
                () => FusedOps.RmsNormQuantizeQ8_0Avx2((float*)fa, w, 0.1f, (byte*)of80a, Elements),
                () => FusedOps.RmsNormQuantizeQ8_0Avx512((float*)fa, w, 0.1f, (byte*)of80a, Elements));

            Report("FusedOps.RmsNormQ8_1",
                () => FusedOps.RmsNormQuantizeQ8_1Avx2((float*)fa, w, 0.1f, (byte*)of81a, Elements),
                () => FusedOps.RmsNormQuantizeQ8_1Avx512((float*)fa, w, 0.1f, (byte*)of81a, Elements));

            Report("FusedOps.RmsNormQ8_K",
                () => FusedOps.RmsNormQuantizeQ8_KAvx2((float*)fa, w, 0.1f, (byte*)of8ka, Elements),
                () => FusedOps.RmsNormQuantizeQ8_KAvx512((float*)fa, w, 0.1f, (byte*)of8ka, Elements));
        }
    }

    private static void Report(string name, Action avx2, Action avx512)
    {
        double a2 = MinNsPerCall(avx2);
        double a5 = MinNsPerCall(avx512);
        double speedup = a5 > 0 ? a2 / a5 : 0;
        Console.WriteLine($"  {name,-26} {a2,10:F1} {a5,10:F1}  {speedup,5:F2}x");
    }

    private static double MinNsPerCall(Action call)
    {
        // Warm up (tier up).
        for (int i = 0; i < 500; i++) call();

        double best = double.MaxValue;
        var sw = new Stopwatch();
        for (int r = 0; r < Rounds; r++)
        {
            sw.Restart();
            for (int i = 0; i < Iters; i++) call();
            sw.Stop();
            double nsPerCall = sw.Elapsed.TotalMilliseconds * 1_000_000.0 / Iters;
            if (nsPerCall < best) best = nsPerCall;
        }
        return best;
    }
}
