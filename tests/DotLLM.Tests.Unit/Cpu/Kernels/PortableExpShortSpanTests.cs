using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1309: <c>TensorPrimitives.Exp</c> falls back to the C runtime's <c>expf</c> for spans shorter than one 128-bit vector, and
/// UCRT and glibc differ by an ulp for some inputs. Attention rows of 1..3 keys (the first tokens of every prompt) hit that
/// fallback, and quantized GEMMs digitize the ulp: the Flint quantized golden was 8e-6 from llama.cpp on Windows and 0.0199 on
/// Linux CI. These pin the short-span softmax exp to the double-evaluated, once-rounded value, so the result is the same on
/// every OS. (On Windows the old code happens to agree with it for most inputs; the test discriminates on glibc.)
/// </summary>
public sealed class PortableExpShortSpanTests
{
    private static float Reference(float x) => (float)Math.Exp(Math.Max(x, -87.3f));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ExpSumAndStore_ShortSpan_IsBitIdenticalToDoubleReference(int length)
    {
        if (FastMath.UseFastExp) return;   // the opt-in Schraudolph bit trick is intentionally approximate
        var rng = new Random(1309 + length);
        float[] input = new float[length], output = new float[length];
        for (int iter = 0; iter < 20_000; iter++)
        {
            for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * -12.0);
            float offset = (float)(rng.NextDouble() * 0.5);
            float sum = FastMath.ExpSumAndStore(input, output, offset);
            float expectedSum = 0;
            for (int i = 0; i < length; i++)
            {
                float expected = Reference(input[i] + offset);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(output[i]));
                expectedSum += expected;
            }
            Assert.Equal(expectedSum, sum, 1e-6f);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void SoftmaxRowWithSink_ShortRow_UsesPortableExp(int length)
    {
        var rng = new Random(77 + length);
        for (int iter = 0; iter < 20_000; iter++)
        {
            float[] row = new float[length];
            for (int i = 0; i < length; i++) row[i] = (float)(rng.NextDouble() * 6 - 3);
            float sink = (float)(rng.NextDouble() * 6 - 3);
            float[] orig = (float[])row.Clone();

            Attention.SoftmaxRowWithSink(row, sink);

            float m = MathF.Max(orig.Max(), sink);
            float[] e = orig.Select(v => Reference(v - m)).ToArray();
            float sum = e.Sum() + (float)Math.Exp(sink - m);
            for (int i = 0; i < length; i++)
                Assert.Equal(e[i] * (1f / sum), row[i], 1e-6f);
        }
    }
}
