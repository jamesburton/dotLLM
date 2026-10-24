using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #1042: the threaded selective scan partitions heads across a pool. Each head runs the
/// serial loop body unchanged, so the threaded result must be BIT-identical to the serial one
/// (the CUDA scan is a bit-order-faithful port of the serial kernel; a tolerance here would hide
/// a reordering). Head counts are chosen to split unevenly across the pool, and B/C groups &gt; 1
/// so a head mapped to the wrong group would show.
/// </summary>
public sealed class Mamba2SelectiveScanThreadingTests
{
    /// <param name="nHead">Head count; 10 over 4 threads and 7 over 3 are uneven splits.</param>
    /// <param name="nGroup">B/C groups (must divide nHead).</param>
    /// <param name="seqLen">Tokens; 1 exercises the decode-shaped dispatch.</param>
    /// <param name="threads">Pool size.</param>
    [Theory]
    [InlineData(10, 2, 7, 4)]
    [InlineData(7, 7, 5, 3)]
    [InlineData(64, 8, 1, 6)]
    [InlineData(128, 8, 3, 14)]
    public void Threaded_IsBitIdenticalToSerial(int nHead, int nGroup, int seqLen, int threads)
    {
        const int headDim = 3, dState = 5;
        var rng = new Random(1042 + nHead * 31 + seqLen);
        int dInner = nHead * headDim;

        float[] state0 = Rand(rng, nHead * headDim * dState, 1f);
        float[] x = Rand(rng, seqLen * dInner, 1f);
        float[] dt = Rand(rng, seqLen * nHead, 2f);
        float[] a = Rand(rng, nHead, 1f).Select(v => -MathF.Abs(v) - 0.1f).ToArray();
        float[] b = Rand(rng, seqLen * nGroup * dState, 1f);
        float[] c = Rand(rng, seqLen * nGroup * dState, 1f);

        float[] stateSerial = (float[])state0.Clone();
        float[] ySerial = new float[seqLen * dInner];
        Mamba2SelectiveScan.Execute(stateSerial, x, dt, a, b, c, ySerial, nHead, headDim, dState, nGroup, seqLen);

        float[] stateThreaded = (float[])state0.Clone();
        float[] yThreaded = new float[seqLen * dInner];
        using (var pool = new ComputeThreadPool(threads))
            Mamba2SelectiveScan.Execute(stateThreaded, x, dt, a, b, c, yThreaded, nHead, headDim, dState, nGroup, seqLen, pool);

        AssertBitIdentical(ySerial, yThreaded, "y");
        AssertBitIdentical(stateSerial, stateThreaded, "state");
    }

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: serial {expected[i]:G9} vs threaded {actual[i]:G9}");
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return r;
    }
}
