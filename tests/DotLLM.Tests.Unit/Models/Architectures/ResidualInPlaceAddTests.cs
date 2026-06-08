using System.Buffers;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Verifies the bit-exactness property that the residual-in-place-add optimization
/// in <c>TransformerModel</c> relies on: the kernel <c>Add.Execute(a, b, result)</c>
/// produces an identical result whether <c>result</c> aliases <c>a</c> (the new
/// in-place path) or is a distinct buffer (the previous "saved residual copy" path).
/// <para>
/// If this invariant ever breaks (e.g. TensorPrimitives.Add changes semantics around
/// aliased outputs), the residual optimization in TransformerModel.Forward would
/// silently produce different numerics — this test guards that contract.
/// </para>
/// </summary>
public class ResidualInPlaceAddTests
{
    [Theory]
    [InlineData(4096, 1)]   // single-token decode at typical hidden size
    [InlineData(4096, 32)]  // small prefill batch
    [InlineData(4096, 512)] // long prefill — the case the issue cites (512 MB savings)
    [InlineData(2048, 128)] // smaller model size
    public void Add_InPlace_BitExactWithOutOfPlace(int hiddenSize, int seqLen)
    {
        var rng = new Random(hiddenSize ^ seqLen);
        int n = hiddenSize * seqLen;

        var hiddenIn = new float[n];
        var normOut = new float[n];
        for (int i = 0; i < n; i++)
        {
            hiddenIn[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            normOut[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Baseline (previous code path): save residual = hidden, then hidden = residual + normOut.
        var residual = ArrayPool<float>.Shared.Rent(n);
        var baselineHidden = new float[n];
        try
        {
            hiddenIn.AsSpan().CopyTo(residual.AsSpan(0, n));
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual, t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut, t * hiddenSize, hiddenSize),
                    new Span<float>(baselineHidden, t * hiddenSize, hiddenSize));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(residual);
        }

        // Optimization (new code path): hidden += normOut, in place, no saved residual.
        var optHidden = new float[n];
        hiddenIn.AsSpan().CopyTo(optHidden);
        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(optHidden, t * hiddenSize, hiddenSize);
            Add.Execute(row, new ReadOnlySpan<float>(normOut, t * hiddenSize, hiddenSize), row);
        }

        for (int i = 0; i < n; i++)
            Assert.Equal(baselineHidden[i], optHidden[i]);
    }
}
