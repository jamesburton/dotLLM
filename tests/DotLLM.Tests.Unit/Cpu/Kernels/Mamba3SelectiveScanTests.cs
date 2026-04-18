using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public class Mamba3SelectiveScanTests
{
    /// <summary>
    /// Minimal T=2, n_head=1, head_dim=1, d_state=1, n_group=1 case with fully hand-computed
    /// expected values — the simplest possible configuration that exercises both γ-term and
    /// β-term contributions.
    /// </summary>
    [Fact]
    public void HandComputed_TwoTokens_OneHeadOneStateOneChannel()
    {
        // Setup: single head, single channel, single state width.
        const int nHead = 1, headDim = 1, dState = 1, nGroup = 1, seqLen = 2;
        int dInner = nHead * headDim;

        // Fresh scan state.
        float[] state = new float[nHead * headDim * dState];   // = {0}
        float[] prevBx = new float[nHead * headDim * dState];  // = {0}

        float[] x = [1.0f, 2.0f];
        float[] alpha = [0.5f, 0.5f];
        float[] beta = [0.0f, 0.3f];
        float[] gamma = [1.0f, 0.7f];
        float[] b = [0.5f, 0.8f];   // [T=2, nGroup=1, dState=1]
        float[] c = [1.0f, 1.0f];   // [T=2, nGroup=1, dState=1]

        float[] y = new float[seqLen * dInner];

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        // Hand-trace:
        // t=0: curBx = B[0]*x[0] = 0.5*1.0 = 0.5
        //      state = α·0 + β·0 + γ·curBx = 0 + 0 + 1.0*0.5 = 0.5
        //      y[0] = C[0]*state = 1.0*0.5 = 0.5
        //      prev_Bx ← 0.5
        Assert.Equal(0.5f, y[0], 6);

        // t=1: curBx = B[1]*x[1] = 0.8*2.0 = 1.6
        //      state = α·0.5 + β·0.5 + γ·1.6 = 0.5*0.5 + 0.3*0.5 + 0.7*1.6 = 0.25 + 0.15 + 1.12 = 1.52
        //      y[1] = C[1]*state = 1.0*1.52 = 1.52
        //      prev_Bx ← 1.6
        Assert.Equal(1.52f, y[1], 5);
        Assert.Equal(1.52f, state[0], 5);
        Assert.Equal(1.6f, prevBx[0], 6);
    }

    /// <summary>
    /// When β = 0 everywhere, the β-term vanishes and the scan reduces to
    /// <c>state = α·state + γ·B·x</c>. prev_Bx should still be updated for
    /// downstream correctness but contributes nothing to y.
    /// </summary>
    [Fact]
    public void Beta_AllZero_ReducesToGammaOnlyScan()
    {
        const int nHead = 2, headDim = 2, dState = 2, nGroup = 1, seqLen = 3;
        int dInner = nHead * headDim;

        float[] state = new float[nHead * headDim * dState];
        float[] prevBx = new float[nHead * headDim * dState];
        float[] stateRef = new float[nHead * headDim * dState];
        float[] prevBxRef = new float[nHead * headDim * dState];

        // Pseudo-random-ish inputs.
        float[] x = [0.1f, -0.2f, 0.3f, 0.4f,
                     -0.5f, 0.6f, 0.7f, -0.8f,
                     0.9f, -1.0f, 1.1f, 1.2f];
        float[] alpha = [0.9f, 0.85f, 0.9f, 0.85f, 0.9f, 0.85f];   // [T=3, nHead=2]
        float[] beta = new float[6];  // all zero
        float[] gamma = [0.1f, 0.15f, 0.2f, 0.1f, 0.15f, 0.2f];
        float[] b = [0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f];        // [T=3, nGroup=1, dState=2]
        float[] c = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f];

        float[] y = new float[seqLen * dInner];
        float[] yRef = new float[seqLen * dInner];

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        // Reference: plain γ-scan inlined.
        int headsPerGroup = nHead / nGroup;
        int stateStrideHead = headDim * dState;
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float alpha_th = alpha[t * nHead + h];
                float gamma_th = gamma[t * nHead + h];
                int g = h / headsPerGroup;
                for (int p = 0; p < headDim; p++)
                {
                    int xIdx = t * dInner + h * headDim + p;
                    int stateRowOffset = h * stateStrideHead + p * dState;
                    float sumf = 0f;
                    for (int k = 0; k < dState; k++)
                    {
                        int bIdx = t * nGroup * dState + g * dState + k;
                        float curBx = b[bIdx] * x[xIdx];
                        float s = stateRef[stateRowOffset + k] * alpha_th + curBx * gamma_th;
                        stateRef[stateRowOffset + k] = s;
                        prevBxRef[stateRowOffset + k] = curBx;
                        sumf += s * c[bIdx];
                    }
                    yRef[t * dInner + h * headDim + p] = sumf;
                }
            }
        }

        for (int i = 0; i < y.Length; i++)
            Assert.Equal(yRef[i], y[i], 6);
        for (int i = 0; i < state.Length; i++)
            Assert.Equal(stateRef[i], state[i], 6);
        for (int i = 0; i < prevBx.Length; i++)
            Assert.Equal(prevBxRef[i], prevBx[i], 6);
    }

    /// <summary>
    /// After the scan completes, <c>prevBx</c> must hold the current-step B·x values for
    /// every (h, p, k), so the next decode call can use them as its β-term input.
    /// </summary>
    [Fact]
    public void PrevBx_EqualsLastStepBTimesX()
    {
        const int nHead = 2, headDim = 3, dState = 2, nGroup = 2, seqLen = 4;
        int dInner = nHead * headDim;

        float[] state = new float[nHead * headDim * dState];
        float[] prevBx = new float[nHead * headDim * dState];

        var rng = new Random(42);
        float[] x = RandomArray(rng, seqLen * dInner);
        float[] alpha = RandomArray(rng, seqLen * nHead, min: 0.1f, max: 0.99f);
        float[] beta = RandomArray(rng, seqLen * nHead, min: -0.1f, max: 0.1f);
        float[] gamma = RandomArray(rng, seqLen * nHead, min: 0.01f, max: 0.3f);
        float[] b = RandomArray(rng, seqLen * nGroup * dState);
        float[] c = RandomArray(rng, seqLen * nGroup * dState);
        float[] y = new float[seqLen * dInner];

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        // After the scan, prevBx[h, p, k] must equal B_{T-1}[g, k] * x_{T-1}[h, p]
        // where g = h / (nHead/nGroup).
        int headsPerGroup = nHead / nGroup;
        int stateStrideHead = headDim * dState;
        int lastT = seqLen - 1;

        for (int h = 0; h < nHead; h++)
        {
            int g = h / headsPerGroup;
            for (int p = 0; p < headDim; p++)
            {
                float x_last = x[lastT * dInner + h * headDim + p];
                int stateRowOffset = h * stateStrideHead + p * dState;
                for (int k = 0; k < dState; k++)
                {
                    float b_last = b[lastT * nGroup * dState + g * dState + k];
                    float expected = b_last * x_last;
                    Assert.Equal(expected, prevBx[stateRowOffset + k], 6);
                }
            }
        }
    }

    /// <summary>
    /// Single-token scan from a fresh state must produce y = γ·B·x·C (β contributes 0
    /// because prev_Bx starts at zero), confirming the t=0 edge case doesn't accidentally
    /// read stale prev_Bx memory.
    /// </summary>
    [Fact]
    public void FreshStart_SingleToken_BetaTermIsZero()
    {
        const int nHead = 1, headDim = 2, dState = 2, nGroup = 1, seqLen = 1;
        int dInner = nHead * headDim;

        float[] state = new float[nHead * headDim * dState];   // zeros
        float[] prevBx = new float[nHead * headDim * dState];  // zeros

        float[] x = [0.3f, -0.4f];
        float[] alpha = [0.7f];
        float[] beta = [0.5f];     // non-zero, but contributes 0 because prev_Bx = 0
        float[] gamma = [0.2f];
        float[] b = [0.5f, 0.6f];
        float[] c = [1.0f, 1.0f];

        float[] y = new float[seqLen * dInner];

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        // For each channel p, y = γ · x[p] · Σ_k B[k] * C[k] (since state_prev = prev_Bx = 0)
        //   p=0: state[0,0,k] = γ·B[k]·x[0] = 0.2·B[k]·0.3 = {0.03, 0.036}
        //        y[0] = 1.0·0.03 + 1.0·0.036 = 0.066
        //   p=1: state[0,1,k] = 0.2·B[k]·(-0.4) = {-0.04, -0.048}
        //        y[1] = 1.0·(-0.04) + 1.0·(-0.048) = -0.088
        Assert.Equal(0.066f, y[0], 5);
        Assert.Equal(-0.088f, y[1], 5);
    }

    /// <summary>
    /// Degenerate scan: all α = 1, β = γ = 0 means state never updates and y is always zero
    /// (since fresh state is zero). prev_Bx should still accumulate.
    /// </summary>
    [Fact]
    public void AllZeroCoefficients_StateStaysZero_PrevBxStillAccumulates()
    {
        const int nHead = 1, headDim = 1, dState = 2, nGroup = 1, seqLen = 3;
        int dInner = nHead * headDim;

        float[] state = new float[2];
        float[] prevBx = new float[2];
        float[] x = [1.0f, 2.0f, 3.0f];
        float[] alpha = [1.0f, 1.0f, 1.0f];   // state[t] = state[t-1] + β·prev_Bx + γ·curBx
        float[] beta = new float[3];           // 0
        float[] gamma = new float[3];          // 0
        float[] b = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f];
        float[] c = [1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f];
        float[] y = new float[3];

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        // With α=1, β=γ=0: state never changes from 0, so y = 0.
        foreach (var v in y) Assert.Equal(0f, v, 6);
        foreach (var v in state) Assert.Equal(0f, v, 6);

        // prev_Bx still updates to the last token's B·x.
        Assert.Equal(0.5f * 3.0f, prevBx[0], 6);
        Assert.Equal(0.6f * 3.0f, prevBx[1], 6);
    }

    private static float[] RandomArray(Random rng, int length, float min = -1f, float max = 1f)
    {
        float[] result = new float[length];
        float scale = max - min;
        for (int i = 0; i < length; i++) result[i] = min + (float)rng.NextDouble() * scale;
        return result;
    }
}
