using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public class LoraDeltaParityTests
{
    [Fact]
    public void Apply_MatchesScalarReference()
    {
        const int seq = 2, dIn = 4, dOut = 3, r = 2;
        float scale = 0.5f;
        float[] x = [ 1,2,3,4,  -1,0,1,2 ];          // [seq, dIn]
        float[] b = [ 0.1f,0.2f,0.3f,0.4f,  0.5f,-0.1f,0.0f,0.2f ]; // [r, dIn]
        float[] a = [ 1,0,  0,1,  1,1 ];             // [dOut, r]
        var y = new float[seq * dOut];

        // Reference: tmp[t,k]=sum_i x[t,i]*b[k,i]; y[t,o]=scale*sum_k a[o,k]*tmp[t,k]
        var expected = new float[seq * dOut];
        for (int t = 0; t < seq; t++)
        {
            var tmp = new float[r];
            for (int k = 0; k < r; k++)
                for (int i = 0; i < dIn; i++) tmp[k] += x[t*dIn+i] * b[k*dIn+i];
            for (int o = 0; o < dOut; o++)
            {
                float acc = 0; for (int k = 0; k < r; k++) acc += a[o*r+k] * tmp[k];
                expected[t*dOut+o] = scale * acc;
            }
        }

        // LoraDelta.Apply(x, b, a, y, seqLen, inputDim, outputDim, rank, scale) — verified signature.
        LoraDelta.Apply(x, b, a, y, seqLen: seq, inputDim: dIn, outputDim: dOut, rank: r, scale: scale);

        for (int n = 0; n < y.Length; n++)
            Assert.Equal(expected[n], y[n], precision: 4);
    }
}
