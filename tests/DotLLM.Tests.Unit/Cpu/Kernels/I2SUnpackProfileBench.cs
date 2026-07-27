using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// TEMPORARY profiling harness for issue #128 — measures what fraction of I2_S W2A8 GEMV time is
/// spent in the scalar UnpackRowI8 loop vs the full unpack+dot call. Not a correctness test; prints
/// wall-clock numbers via test output. Delete before finalizing if profiling shows unpack isn't the
/// bottleneck.
/// </summary>
public sealed unsafe class I2SUnpackProfileBench
{
    private readonly ITestOutputHelper _output;

    public I2SUnpackProfileBench(ITestOutputHelper output) => _output = output;

    [Fact]
    public void BenchStreamingReadOnly_MatchesScalarChecksum()
    {
        var rng = new Random(7);
        const int m = 64, k = 512;
        int rowBytes = k / 4;
        byte[] buf = new byte[m * rowBytes];
        rng.NextBytes(buf);

        byte expected = 0;
        foreach (byte b in buf) expected ^= b;

        fixed (byte* p = buf)
        {
            byte actual = MatMul.BenchStreamingReadOnly(p, m, k);
            Assert.Equal(expected, actual);
        }
    }

    [Theory]
    [InlineData(6912, 2560, "ffn_down-like (BitNet 3B: m=ffn, k=hidden)")]
    [InlineData(2560, 6912, "ffn_up-like (BitNet 3B: m=hidden, k=ffn)")]
    public void ProfileGemvI2SW2A8_UnpackFractionOfTotal(int m, int k, string label)
    {
        var rng = new Random(42);
        int rowBytes = k / 4;
        long weightBytes = (long)m * rowBytes + 4;
        byte* weights = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);
        float* x = (float*)NativeMemory.AllocZeroed((nuint)(k * sizeof(float)));
        float* result = (float*)NativeMemory.AllocZeroed((nuint)(m * sizeof(float)));
        try
        {
            byte[] randRow = new byte[weightBytes];
            rng.NextBytes(randRow);
            Marshal.Copy(randRow, 0, (nint)weights, (int)weightBytes);
            *(float*)(weights + (long)m * rowBytes) = 0.02f; // per-tensor scale

            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            const int iters = 20;

            // Warm up (JIT).
            MatMul.GemvI2_S(weights, x, result, m, k, null);
            MatMul.BenchUnpackRowI8Only(weights, m, k);

            var swTotal = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                MatMul.GemvI2_S(weights, x, result, m, k, null);
            swTotal.Stop();

            var swUnpack = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                MatMul.BenchUnpackRowI8Only(weights, m, k);
            swUnpack.Stop();

            double totalMs = swTotal.Elapsed.TotalMilliseconds / iters;
            double unpackMs = swUnpack.Elapsed.TotalMilliseconds / iters;
            double fraction = unpackMs / totalMs;

            _output.WriteLine($"[{label}] m={m} k={k} AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                               $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");
            _output.WriteLine($"  total GEMV/call: {totalMs:F4} ms   unpack-only/call: {unpackMs:F4} ms   unpack fraction: {fraction:P1}");
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(x);
            NativeMemory.Free(result);
        }
    }
}
