using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Retained profiling harness — measures whether I2_S GEMV decode (post-#128 unpack-SIMD fix) is
/// memory-bandwidth-bound or still compute-bound on this box, gating the AVX-512 activation-LUT
/// dot kernel proposed upstream in issue #334. Not a correctness test; prints wall-clock numbers
/// via test output. The recorded baseline verdict lives in
/// <c>docs/superpowers/specs/2026-07-27-bitnet-decode-bandwidth-RESULTS.md</c>; this class is kept
/// (rather than deleted) as the before/after comparison harness for the AVX-512 activation-LUT
/// kernel that investigation recommends as the follow-up.
/// </summary>
[Trait("Category", "Benchmark")]
public sealed unsafe class I2SDecodeBandwidthProfileBench
{
    private readonly ITestOutputHelper _output;

    public I2SDecodeBandwidthProfileBench(ITestOutputHelper output) => _output = output;

    /// <summary>Number of distinct weight-buffer copies for the "cold" (>L3) measurement.</summary>
    private const int ColdBufferCount = 32;

    [Theory]
    [InlineData(2560, 2560, "attn_qproj")]
    [InlineData(6912, 2560, "ffn_gate")]
    [InlineData(2560, 6912, "ffn_down")]
    public void ProfileGemvI2SDecode_BandwidthVsCompute(int m, int k, string label)
    {
        var rng = new Random(42);
        int rowBytes = k / 4;
        long weightBytes = (long)m * rowBytes;
        const float scale = 0.02f;

        // Hot buffer: single allocation, reused every iteration (matches #131's methodology and a
        // resident weight tensor reused across consecutive decode steps).
        byte* hotWeights = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);

        // Cold buffers: enough distinct copies to exceed typical L3, round-robined across
        // iterations so each touch is a fresh cache line from DRAM.
        byte*[] coldWeights = new byte*[ColdBufferCount];

        float* x = (float*)NativeMemory.AllocZeroed((nuint)(k * sizeof(float)));
        float* result = (float*)NativeMemory.AllocZeroed((nuint)(m * sizeof(float)));

        try
        {
            byte[] randRow = new byte[weightBytes];
            rng.NextBytes(randRow);
            Marshal.Copy(randRow, 0, (nint)hotWeights, (int)weightBytes);

            for (int c = 0; c < ColdBufferCount; c++)
            {
                coldWeights[c] = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);
                rng.NextBytes(randRow);
                Marshal.Copy(randRow, 0, (nint)coldWeights[c], (int)weightBytes);
            }

            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            const int iters = 20;

            // Warm up (JIT).
            MatMul.GemvI2_S(hotWeights, x, result, m, k, scale, null);
            MatMul.BenchUnpackRowI8Only(hotWeights, m, k);
            MatMul.BenchStreamingReadOnly(hotWeights, m, k);

            double hotStreamMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchStreamingReadOnly(hotWeights, m, k);
            }, iters);

            double coldStreamMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchStreamingReadOnly(coldWeights[i % ColdBufferCount], m, k);
            }, iters);

            double unpackMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.BenchUnpackRowI8Only(hotWeights, m, k);
            }, iters);

            double decodeMs = Time(() =>
            {
                for (int i = 0; i < iters; i++)
                    MatMul.GemvI2_S(hotWeights, x, result, m, k, scale, null);
            }, iters);

            double hotStreamGBs = GBs(weightBytes, hotStreamMs);
            double coldStreamGBs = GBs(weightBytes, coldStreamMs);
            double decodeGBs = GBs(weightBytes, decodeMs);

            _output.WriteLine($"[{label}] m={m} k={k} weightBytes={weightBytes} " +
                               $"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                               $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");
            _output.WriteLine($"  hot streaming-only:  {hotStreamMs:F4} ms/call   {hotStreamGBs:F2} GB/s   (cache-resident ceiling)");
            _output.WriteLine($"  cold streaming-only: {coldStreamMs:F4} ms/call   {coldStreamGBs:F2} GB/s   (DRAM-forced ceiling)");
            _output.WriteLine($"  unpack-only:         {unpackMs:F4} ms/call");
            _output.WriteLine($"  full GemvI2_S decode:{decodeMs:F4} ms/call   {decodeGBs:F2} GB/s");
            _output.WriteLine($"  decode / hot-ceiling ratio:  {decodeGBs / hotStreamGBs:P1}");
            _output.WriteLine($"  decode / cold-ceiling ratio: {decodeGBs / coldStreamGBs:P1}");
        }
        finally
        {
            NativeMemory.Free(hotWeights);
            foreach (byte* p in coldWeights)
                if (p is not null) NativeMemory.Free(p);
            NativeMemory.Free(x);
            NativeMemory.Free(result);
        }
    }

    private static double Time(Action body, int iters)
    {
        var sw = Stopwatch.StartNew();
        body();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static double GBs(long bytes, double ms) => bytes / (ms / 1000.0) / 1e9;
}
