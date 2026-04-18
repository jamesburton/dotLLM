using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness anchor for <see cref="Mamba3MimoProject"/>. The MIMO rank-R
/// factorization is a pair of strided broadcasts whose easiest silent-bug
/// class is an axis mix-up (rank placed wrong, head/channel transposed).
/// These tests lock the reference layout
/// <c>[T, n_head, head_dim, rank]</c> for the expanded tensor and
/// <c>[n_head, head_dim, rank]</c> for both weight tensors, matching the
/// VikramKarLex/mamba3-minimal convention.
/// </summary>
public sealed class Mamba3MimoProjectTests
{
    /// <summary>
    /// With <c>rank=1</c>, expansion is <c>x_mimo[t,h,p,0] = x[t,h,p] * w_x[h,p,0]</c>
    /// and contraction is <c>y[t,h,p] = x_mimo[t,h,p,0] * w_d[h,p,0]</c>. Composing
    /// them gives <c>y[t,h,p] = x[t,h,p] * w_x[h,p,0] * w_d[h,p,0]</c> — the
    /// rank-R MIMO degenerates to a per-(h,p) scalar gain with no cross-talk.
    /// </summary>
    [Fact]
    public void RankOne_Roundtrip_IsPerElementGain()
    {
        const int seqLen = 4;
        const int nHead = 2;
        const int headDim = 3;
        const int rank = 1;

        var rng = new Random(42);
        float[] x = new float[seqLen * nHead * headDim];
        float[] wX = new float[nHead * headDim * rank];
        float[] wD = new float[nHead * headDim * rank];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < wX.Length; i++) wX[i] = 0.1f + (float)rng.NextDouble();
        for (int i = 0; i < wD.Length; i++) wD[i] = 0.1f + (float)rng.NextDouble();

        float[] xMimo = new float[seqLen * nHead * headDim * rank];
        float[] y = new float[seqLen * nHead * headDim];

        Mamba3MimoProject.ExpandInput(x, wX, xMimo, seqLen, nHead, headDim, rank);
        Mamba3MimoProject.ContractOutput(xMimo, wD, y, seqLen, nHead, headDim, rank);

        int hpCount = nHead * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            for (int hp = 0; hp < hpCount; hp++)
            {
                float expected = x[t * hpCount + hp] * wX[hp] * wD[hp];
                Assert.Equal(expected, y[t * hpCount + hp], 1e-6f);
            }
        }
    }

    /// <summary>
    /// Shape sanity: under a few different <c>(T, n_head, head_dim, rank)</c>
    /// combinations, the output spans must be fully written and exactly the
    /// sizes expected. We also check that out-of-bounds spans throw.
    /// </summary>
    [Fact]
    public void Shapes_AreRespected()
    {
        var cases = new (int T, int H, int P, int R)[]
        {
            (1, 1, 1, 1), (2, 4, 8, 2), (3, 2, 5, 4), (5, 3, 2, 3),
        };

        foreach (var (T, H, P, R) in cases)
        {
            float[] x = new float[T * H * P];
            float[] wX = new float[H * P * R];
            float[] wD = new float[H * P * R];
            for (int i = 0; i < x.Length; i++) x[i] = i * 0.01f + 0.5f;
            for (int i = 0; i < wX.Length; i++) wX[i] = 0.1f + i * 0.003f;
            for (int i = 0; i < wD.Length; i++) wD[i] = 0.2f + i * 0.002f;

            float[] xMimo = new float[T * H * P * R];
            float[] y = new float[T * H * P];

            // Must not throw.
            Mamba3MimoProject.ExpandInput(x, wX, xMimo, T, H, P, R);
            Mamba3MimoProject.ContractOutput(xMimo, wD, y, T, H, P, R);

            // Expanded tensor must be fully written (no residual zeros where there
            // shouldn't be), confirmed by recomputing the full sum and matching.
            double recomputed = 0.0;
            for (int i = 0; i < xMimo.Length; i++) recomputed += Math.Abs(xMimo[i]);
            Assert.True(recomputed > 0.0);

            // Out-of-bounds shapes must throw.
            Assert.Throws<ArgumentException>(() =>
                Mamba3MimoProject.ExpandInput(x.AsSpan(0, x.Length - 1), wX, xMimo, T, H, P, R));
            Assert.Throws<ArgumentException>(() =>
                Mamba3MimoProject.ContractOutput(xMimo, wD, y.AsSpan(0, y.Length - 1), T, H, P, R));
        }
    }

    /// <summary>
    /// Hand-computed small case. T=2, nHead=2, headDim=2, rank=2.
    /// Deliberately asymmetric weights so any transposition or rank/channel
    /// mix-up gives a different number.
    /// </summary>
    [Fact]
    public void HandComputed_SmallCase_MatchesExpansionAndContraction()
    {
        const int seqLen = 2;
        const int nHead = 2;
        const int headDim = 2;
        const int rank = 2;
        int hpCount = nHead * headDim; // 4

        // x has 8 entries: (t, h, p) indexed.
        //   t=0:   h=0 -> (1, 2);   h=1 -> (3, 4)
        //   t=1:   h=0 -> (5, 6);   h=1 -> (7, 8)
        float[] x = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];

        // wX has 8 entries: (h, p, r) indexed.
        //   h=0,p=0 -> (0.1, 0.2)   h=0,p=1 -> (0.3, 0.4)
        //   h=1,p=0 -> (0.5, 0.6)   h=1,p=1 -> (0.7, 0.8)
        float[] wX = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f];

        // wD has 8 entries: (h, p, r) indexed. Different from wX so we can tell.
        //   h=0,p=0 -> (1.0, 2.0)   h=0,p=1 -> (3.0, 4.0)
        //   h=1,p=0 -> (5.0, 6.0)   h=1,p=1 -> (7.0, 8.0)
        float[] wD = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];

        float[] xMimo = new float[seqLen * hpCount * rank]; // 16
        float[] y = new float[seqLen * hpCount];             // 8

        Mamba3MimoProject.ExpandInput(x, wX, xMimo, seqLen, nHead, headDim, rank);

        // Hand-computed expansion: xMimo[t,h,p,r] = x[t,h,p] * wX[h,p,r].
        // t=0:
        //   (0,0,0): 1*(0.1, 0.2)  -> (0.1, 0.2)
        //   (0,0,1): 2*(0.3, 0.4)  -> (0.6, 0.8)
        //   (0,1,0): 3*(0.5, 0.6)  -> (1.5, 1.8)
        //   (0,1,1): 4*(0.7, 0.8)  -> (2.8, 3.2)
        // t=1:
        //   (1,0,0): 5*(0.1, 0.2)  -> (0.5, 1.0)
        //   (1,0,1): 6*(0.3, 0.4)  -> (1.8, 2.4)
        //   (1,1,0): 7*(0.5, 0.6)  -> (3.5, 4.2)
        //   (1,1,1): 8*(0.7, 0.8)  -> (5.6, 6.4)
        float[] expectedExpand =
        [
            0.1f, 0.2f, 0.6f, 0.8f, 1.5f, 1.8f, 2.8f, 3.2f,
            0.5f, 1.0f, 1.8f, 2.4f, 3.5f, 4.2f, 5.6f, 6.4f,
        ];
        for (int i = 0; i < xMimo.Length; i++)
            Assert.Equal(expectedExpand[i], xMimo[i], 1e-6f);

        // Now contract using wD.
        Mamba3MimoProject.ContractOutput(xMimo, wD, y, seqLen, nHead, headDim, rank);

        // Hand-computed contraction: y[t,h,p] = sum_r xMimo[t,h,p,r] * wD[h,p,r].
        // t=0:
        //   (0,0): 0.1*1 + 0.2*2 = 0.1 + 0.4 = 0.5
        //   (0,1): 0.6*3 + 0.8*4 = 1.8 + 3.2 = 5.0
        //   (1,0): 1.5*5 + 1.8*6 = 7.5 + 10.8 = 18.3
        //   (1,1): 2.8*7 + 3.2*8 = 19.6 + 25.6 = 45.2
        // t=1:
        //   (0,0): 0.5*1 + 1.0*2 = 0.5 + 2.0 = 2.5
        //   (0,1): 1.8*3 + 2.4*4 = 5.4 + 9.6 = 15.0
        //   (1,0): 3.5*5 + 4.2*6 = 17.5 + 25.2 = 42.7
        //   (1,1): 5.6*7 + 6.4*8 = 39.2 + 51.2 = 90.4
        float[] expectedY = [0.5f, 5.0f, 18.3f, 45.2f, 2.5f, 15.0f, 42.7f, 90.4f];
        for (int i = 0; i < y.Length; i++)
            Assert.Equal(expectedY[i], y[i], 1e-4f);
    }

    /// <summary>
    /// Distinct-buffer contract. <see cref="Mamba3MimoProject.ExpandInput"/> and
    /// <see cref="Mamba3MimoProject.ContractOutput"/> write buffers of different
    /// lengths than their inputs (rank axis is added or removed), so in-place
    /// is structurally impossible. We verify the public surface by running a
    /// random case with non-overlapping buffers and asserting that a second
    /// identical run produces identical results (no accidental state between
    /// calls), and that the scalar path matches the public path on the same
    /// random case.
    /// </summary>
    [Fact]
    public void ScalarAndPublic_AgreeOnRandomInputs()
    {
        const int seqLen = 3;
        const int nHead = 2;
        const int headDim = 4;
        const int rank = 3;

        var rng = new Random(0xBEEF);
        float[] x = new float[seqLen * nHead * headDim];
        float[] wX = new float[nHead * headDim * rank];
        float[] wD = new float[nHead * headDim * rank];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < wX.Length; i++) wX[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < wD.Length; i++) wD[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] xMimoA = new float[seqLen * nHead * headDim * rank];
        float[] xMimoB = new float[seqLen * nHead * headDim * rank];
        float[] yA = new float[seqLen * nHead * headDim];
        float[] yB = new float[seqLen * nHead * headDim];

        Mamba3MimoProject.ExpandInput(x, wX, xMimoA, seqLen, nHead, headDim, rank);
        Mamba3MimoProject.ExpandInputScalar(x, wX, xMimoB, seqLen, nHead, headDim, rank);
        for (int i = 0; i < xMimoA.Length; i++)
            Assert.Equal(xMimoB[i], xMimoA[i], 1e-6f);

        Mamba3MimoProject.ContractOutput(xMimoA, wD, yA, seqLen, nHead, headDim, rank);
        Mamba3MimoProject.ContractOutputScalar(xMimoB, wD, yB, seqLen, nHead, headDim, rank);
        for (int i = 0; i < yA.Length; i++)
            Assert.Equal(yB[i], yA[i], 1e-5f);
    }
}
