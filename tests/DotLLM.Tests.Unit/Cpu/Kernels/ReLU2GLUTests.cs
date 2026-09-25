using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for the BitNet gated FFN activation: <c>result[i] = relu(gate[i])² * up[i]</c>,
/// where <c>relu(x) = max(0, x)</c>. This is the squared-ReLU ("relu2") gating used by
/// BitNet b1.58 in place of SwiGLU.
/// </summary>
public sealed class ReLU2GLUTests
{
    [Fact]
    public void PositiveGate_SquaresGateTimesUp()
    {
        // relu(2)² * 3 = 4 * 3 = 12
        float[] gate = [2.0f];
        float[] up = [3.0f];
        float[] result = new float[1];

        FusedOps.ReLU2GLU(gate, up, result);

        Assert.Equal(12.0f, result[0], 1e-5f);
    }

    [Fact]
    public void NegativeGate_ProducesZero()
    {
        // relu(-5)² * 7 = 0 * 7 = 0
        float[] gate = [-5.0f];
        float[] up = [7.0f];
        float[] result = new float[1];

        FusedOps.ReLU2GLU(gate, up, result);

        Assert.Equal(0f, result[0], 1e-6f);
    }

    [Fact]
    public void ZeroGate_ProducesZero()
    {
        float[] gate = [0f];
        float[] up = [9.0f];
        float[] result = new float[1];

        FusedOps.ReLU2GLU(gate, up, result);

        Assert.Equal(0f, result[0], 1e-6f);
    }

    [Fact]
    public void MultipleValues_MatchScalarReference()
    {
        float[] gate = [-3f, -0.5f, 0f, 0.5f, 3f];
        float[] up = [1f, 2f, 3f, 4f, 5f];
        float[] result = new float[5];
        float[] expected = new float[5];

        FusedOps.ReLU2GLUScalar(gate, up, expected);
        FusedOps.ReLU2GLU(gate, up, result);

        for (int i = 0; i < gate.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }

    [Fact]
    public void ScalarMatchesFused_OverRandomVector()
    {
        // Spans both the tiled body and the tail (length not a multiple of the tile).
        var rng = new Random(1234);
        const int n = 6912 + 37; // BitNet intermediate size + tail
        float[] gate = new float[n];
        float[] up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = rng.NextSingle() * 8f - 4f; // [-4, 4]
            up[i] = rng.NextSingle() * 8f - 4f;
        }

        float[] scalarResult = new float[n];
        float[] fusedResult = new float[n];

        FusedOps.ReLU2GLUScalar(gate, up, scalarResult);
        FusedOps.ReLU2GLU(gate, up, fusedResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(scalarResult[i], fusedResult[i], 1e-4f);
    }
}
