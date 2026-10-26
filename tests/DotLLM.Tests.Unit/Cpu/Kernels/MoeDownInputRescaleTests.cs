using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Tests.Unit.Models.Flint;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1158: the routed-expert down projection rescales its input row by a power of two when <c>max|row| &gt; 1024</c>
/// (llama.cpp <c>build_amax_src1_f16</c>). Real Flint reaches ~13000 there; the Q8_1 block sum <c>s = d * Σq</c> is stored as F16,
/// overflowed to inf and turned the layer into NaN (an out-of-range expert index in the next layer's routing).
/// </summary>
public sealed unsafe class MoeDownInputRescaleTests
{
    [Fact]
    public void Small_Rows_Are_Untouched()
    {
        var row = new float[64];
        for (int i = 0; i < row.Length; i++) row[i] = (i - 32) * 32f;   // max |x| = 1024: the boundary stays untouched
        var copy = (float[])row.Clone();
        fixed (float* p = row)
            Assert.Equal(1f, MoeSwiGluMlp.RescaleDownInputForF16Sums(p, row.Length));
        Assert.Equal(copy, row);
    }

    [Fact]
    public void Large_Rows_Are_Divided_Exactly_By_A_Power_Of_Two()
    {
        var row = new float[64];
        var rng = new Random(1158);
        for (int i = 0; i < row.Length; i++) row[i] = (float)(rng.NextDouble() * 2 - 1) * 1000f;
        row[5] = 13036.78f;   // the real-model maximum: ceil(log2(13036.78 / 1024)) = 4
        var copy = (float[])row.Clone();
        float e;
        fixed (float* p = row)
            e = MoeSwiGluMlp.RescaleDownInputForF16Sums(p, row.Length);
        Assert.Equal(16f, e);
        for (int i = 0; i < row.Length; i++)
            Assert.Equal(copy[i], row[i] * e);   // exact: power-of-two scaling round-trips bit for bit
    }

    /// <summary>
    /// One Q4_1 expert whose SwiGLU output is ~2e5 per element: without the rescale every Q8_1 block sum of the down input
    /// overflows F16. Both routed paths (grouped GEMM and the n=1 decode fast path) must stay finite and match an F32 reference.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Q4_1_Expert_With_Huge_SwiGLU_Output_Stays_Finite(bool decodeFastPath)
    {
        const int hidden = 64, inter = 64;
        var rng = new Random(7);
        float[] gate = Fill(inter * hidden, () => 1.5f + 0.05f * (float)rng.NextDouble());
        float[] up = Fill(inter * hidden, () => 1.5f + 0.05f * (float)rng.NextDouble());
        float[] down = Fill(hidden * inter, () => 0.01f * (float)(rng.NextDouble() * 2 - 1));
        float[] x = Fill(hidden, () => 5f + (float)rng.NextDouble());
        byte[] gq = SyntheticFlintGguf.EncodeQ4_1(gate), uq = SyntheticFlintGguf.EncodeQ4_1(up), dq = SyntheticFlintGguf.EncodeQ4_1(down);

        var (expected, scale) = Reference(gq, uq, dq, x, hidden, inter);
        var output = new float[hidden];
        using var pool = decodeFastPath ? new ComputeThreadPool(4) : null;
        fixed (byte* g = gq) fixed (byte* u = uq) fixed (byte* d = dq)
        {
            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                hidden: x,
                gateExpsRawBase: (nint)g, gateExpsQt: QuantizationType.Q4_1, gateExpsRowBytes: gq.Length, gateExpsF32Ptrs: default,
                upExpsRawBase: (nint)u, upExpsQt: QuantizationType.Q4_1, upExpsRowBytes: uq.Length, upExpsF32Ptrs: default,
                downExpsRawBase: (nint)d, downExpsQt: QuantizationType.Q4_1, downExpsRowBytes: dq.Length, downExpsF32Ptrs: default,
                assignExpert: [0], assignWeight: [1f],
                bucketCursors: [0, 1], bucketTokens: [0], bucketSlots: [0],
                uniqueExperts: [0], uniqueExpertCount: 1,
                output: output,
                numExperts: 1, numExpertsPerTok: 1, hiddenSize: hidden, intermediateSize: inter, seqLen: 1,
                sharedGateProj: default, sharedUpProj: default, sharedDownProj: default,
                sharedIntermediateSize: 0, sharedExpertGate: default,
                loraAdapter: null, loraLayer: -1,
                threadPool: pool,
                packedLegacyExperts: true);
        }

        for (int i = 0; i < hidden; i++)
        {
            Assert.True(float.IsFinite(output[i]), $"output[{i}] = {output[i]}");
            // Each output is a cancelling sum of ~3000-sized terms, so the Q8_1 activation error is judged against sum |term|.
            Assert.True(MathF.Abs(output[i] - expected[i]) <= 0.01f * scale[i],
                $"output[{i}] = {output[i]}, expected {expected[i]} (sum |term| {scale[i]})");
        }
    }

    private static float[] Fill(int n, Func<float> f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = f();
        return a;
    }

    /// <summary>Dequantised weights, F32 activations, double accumulation; also returns sum |term| per output row.</summary>
    private static (float[] Y, float[] AbsSum) Reference(byte[] gq, byte[] uq, byte[] dq, float[] x, int hidden, int inter)
    {
        float[] G = Dequant(gq, inter * hidden), U = Dequant(uq, inter * hidden), D = Dequant(dq, hidden * inter);
        var h = new double[inter];
        for (int r = 0; r < inter; r++)
        {
            double gs = 0, us = 0;
            for (int c = 0; c < hidden; c++) { gs += G[r * hidden + c] * (double)x[c]; us += U[r * hidden + c] * (double)x[c]; }
            h[r] = gs / (1 + Math.Exp(-gs)) * us;
        }
        var y = new float[hidden];
        var abs = new float[hidden];
        for (int r = 0; r < hidden; r++)
        {
            double s = 0, a = 0;
            for (int c = 0; c < inter; c++) { s += D[r * inter + c] * h[c]; a += Math.Abs(D[r * inter + c] * h[c]); }
            y[r] = (float)s;
            abs[r] = (float)a;
        }
        return (y, abs);
    }

    private static float[] Dequant(byte[] q, int n)
    {
        var f = new float[n];
        fixed (byte* p = q)
            Dequantize.ToFloat32((nint)p, n, QuantizationType.Q4_1, f);
        return f;
    }
}
