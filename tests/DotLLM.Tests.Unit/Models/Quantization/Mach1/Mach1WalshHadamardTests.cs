using DotLLM.Models.Quantization.Mach1;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Quantization.Mach1;

/// <summary>
/// Verifies <see cref="Mach1WalshHadamard.TransformRowsInPlace"/> against an
/// independently-constructed orthonormal Sylvester Hadamard matrix
/// (recursive definition, naive matrix-vector multiply) — a genuinely
/// different code path from the production butterfly network, so a real
/// indexing/normalization bug would show up as more than float noise.
/// </summary>
public sealed class Mach1WalshHadamardTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(64)]
    public void TransformRowsInPlace_MatchesNaiveSylvesterMatrixReference(int dim)
    {
        var rng = new Random(1234 + dim);
        const int rows = 3; // arbitrary, non-power-of-two row count on purpose
        var data = new float[rows * dim];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 4 - 2);

        float[] expected = new float[data.Length];
        double[,] hadamard = BuildOrthonormalSylvesterMatrix(dim);
        for (int r = 0; r < rows; r++)
        {
            for (int outIdx = 0; outIdx < dim; outIdx++)
            {
                double acc = 0;
                for (int inIdx = 0; inIdx < dim; inIdx++)
                    acc += hadamard[outIdx, inIdx] * data[r * dim + inIdx];
                expected[r * dim + outIdx] = (float)acc;
            }
        }

        var actual = (float[])data.Clone();
        Mach1WalshHadamard.TransformRowsInPlace(actual, rows, dim);

        for (int i = 0; i < actual.Length; i++)
        {
            float tol = Math.Max(1e-3f, Math.Abs(expected[i]) * 1e-4f);
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
        }
    }

    [Fact]
    public void TransformRowsInPlace_NonPowerOfTwoDim_Throws()
    {
        var data = new float[24];
        Assert.Throws<ArgumentException>(() => Mach1WalshHadamard.TransformRowsInPlace(data, 1, 24));
    }

    /// <summary>
    /// Builds the orthonormal (1/sqrt(dim)-scaled) Sylvester Hadamard matrix
    /// via its textbook recursive definition: H_1 = [1]; H_2n =
    /// [[H_n, H_n], [H_n, -H_n]].
    /// </summary>
    private static double[,] BuildOrthonormalSylvesterMatrix(int dim)
    {
        double[,] h = { { 1.0 } };
        while (h.GetLength(0) < dim)
        {
            int n = h.GetLength(0);
            var next = new double[n * 2, n * 2];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    next[i, j] = h[i, j];
                    next[i, j + n] = h[i, j];
                    next[i + n, j] = h[i, j];
                    next[i + n, j + n] = -h[i, j];
                }
            }
            h = next;
        }

        double scale = 1.0 / Math.Sqrt(dim);
        var result = new double[dim, dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                result[i, j] = h[i, j] * scale;
        return result;
    }
}
