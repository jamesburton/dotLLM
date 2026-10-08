using System.Text.Json;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #832 (exploratory spike): LittleBit factorized-linear reference + AVX2/VNNI kernels.
/// Fixtures were generated offline by <c>LittleBitData/gen_fixture.py</c> from the paper's equations (float64 numpy).
/// </summary>
public sealed unsafe class LittleBitTests(ITestOutputHelper output)
{
    private static LittleBitLayer BuildLayer(JsonElement c)
    {
        int dOut = c.GetProperty("d_out").GetInt32(), dIn = c.GetProperty("d_in").GetInt32(), r = c.GetProperty("r").GetInt32();
        var paths = new List<LittleBitPath>();
        foreach (var p in c.GetProperty("paths").EnumerateArray())
        {
            sbyte[] us = p.GetProperty("us").EnumerateArray().Select(e => (sbyte)e.GetInt32()).ToArray();
            sbyte[] vs = p.GetProperty("vs").EnumerateArray().Select(e => (sbyte)e.GetInt32()).ToArray();
            Half[] H(string n) => p.GetProperty(n).EnumerateArray().Select(e => (Half)e.GetSingle()).ToArray();
            paths.Add(LittleBitPath.FromSigns(dOut, dIn, r, us, vs, H("h"), H("g"), H("l")));
        }
        return new LittleBitLayer(paths.ToArray());
    }

    private static float[] Run(LittleBitLayer l, float[] x, ComputeThreadPool? pool, LittleBitKernel k)
    {
        var (dIn, r) = l.ScratchDims;
        using var sc = new LittleBitScratch(dIn, r);
        var y = new float[l.DOut];
        fixed (float* xp = x, yp = y) l.Gemv(xp, yp, sc, pool, k);
        return y;
    }

    private static double MaxRel(double[] expected, float[] got)
    {
        double scale = expected.Max(Math.Abs), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - got[i]) / scale);
        return worst;
    }

    [Fact]
    public void PackedBits_AreLsbFirst_OneIsMinusOne_VsIsTransposed()
    {
        // Us 2x3: row0 = [+,-,+], row1 = [-,-,+]; Vs 4x3 ([d_in x r]); check the documented bit layout.
        sbyte[] us = [1, -1, 1, -1, -1, 1];
        sbyte[] vs = [1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1];
        Half[] one2 = [(Half)1, (Half)1], one4 = [(Half)1, (Half)1, (Half)1, (Half)1], one3 = [(Half)1, (Half)1, (Half)1];
        using var p = LittleBitPath.FromSigns(2, 4, 3, us, vs, one2, one4, one3);
        Assert.Equal(0b010, p.UBits[0]);            // row 0: bit1 set
        Assert.Equal(0b011, p.UBits[p.RPad / 8]);   // row 1: bits 0,1 set
        // V latent row j=0 holds Vs[0..4, 0] = [+,-,+,-] -> bits 1,3
        Assert.Equal(0b1010, p.VBits[0]);
        Assert.Equal(0b1100, p.VBits[p.DInPad / 8]);   // j=1: Vs[:,1] = [+,+,-,-] -> bits 2,3
        for (int o = 0; o < 2; o++) for (int j = 0; j < 3; j++) Assert.Equal(us[o * 3 + j] > 0 ? 1 : -1, p.USign(o, j));
        for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) Assert.Equal(vs[i * 3 + j] > 0 ? 1 : -1, p.VSign(i, j));
        Assert.Equal(2L * 3 * (2 + 4 + 1) + 16L * (2 + 4 + 3), p.PaperBits);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void Fixture_ReferenceAndKernels_MatchNumpyFloat64(int caseIdx)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Cpu", "Kernels", "LittleBitData", "littlebit_fixture.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var c = doc.RootElement.GetProperty("cases")[caseIdx];
        using var layer = BuildLayer(c);
        using var pool = new ComputeThreadPool(3);
        var xs = c.GetProperty("x").EnumerateArray().ToArray();
        var ys = c.GetProperty("y").EnumerateArray().ToArray();
        for (int n = 0; n < xs.Length; n++)
        {
            float[] x = xs[n].EnumerateArray().Select(e => e.GetSingle()).ToArray();
            double[] y = ys[n].EnumerateArray().Select(e => e.GetDouble()).ToArray();
            var scalar = LittleBitReference.Gemv(layer, x);
            for (int i = 0; i < y.Length; i++) Assert.Equal(y[i], scalar[i], 1e-9 * (1 + Math.Abs(y[i])));
            Assert.True(MaxRel(y, Run(layer, x, null, LittleBitKernel.Avx2Float)) < 1e-5);
            Assert.True(MaxRel(y, Run(layer, x, pool, LittleBitKernel.Avx2Float)) < 1e-5);
            if (LittleBitLayer.VnniSupported)
                Assert.True(MaxRel(y, Run(layer, x, pool, LittleBitKernel.VnniInt8)) < 0.05);
        }
    }

    // d_out != d_in, r not a multiple of 8/32 (43, 270, 500 are the benchmark ranks), several activation vectors.
    [Theory]
    [InlineData(37, 200, 43, 2)]
    [InlineData(130, 77, 270, 2)]
    [InlineData(65, 1000, 500, 1)]
    [InlineData(8, 31, 9, 2)]
    [InlineData(1, 5, 1, 2)]
    public void RandomShapes_Kernels_MatchScalarAndDenseF32Control(int dOut, int dIn, int r, int npaths)
    {
        var rng = new Random(dOut * 31 + dIn);
        using var layer = new LittleBitLayer(Enumerable.Range(0, npaths).Select(_ => LittleBitPath.Random(dOut, dIn, r, rng)).ToArray());
        var dense = LittleBitReference.DecodeDense(layer);
        using var pool = new ComputeThreadPool(4);
        double worstInt8 = 0;
        for (int n = 0; n < 4; n++)   // NK != 1: several distinct activation vectors through the same layer
        {
            float[] x = Enumerable.Range(0, dIn).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            double[] refY = LittleBitReference.Gemv(layer, x);
            // F32-decoded control agrees with the scalar reference (validates the decode and the layout).
            Assert.True(MaxRel(refY, LittleBitReference.DenseGemv(dense, dOut, dIn, x)) < 1e-4);
            Assert.True(MaxRel(refY, Run(layer, x, null, LittleBitKernel.Avx2Float)) < 1e-4);
            Assert.True(MaxRel(refY, Run(layer, x, pool, LittleBitKernel.Avx2Float)) < 1e-4);
            if (LittleBitLayer.VnniSupported)
            {
                var yi = Run(layer, x, pool, LittleBitKernel.VnniInt8);
                // not bit-comparable: two int8 activation quantizations (x' and t). Bound the extra error.
                double err = MaxRel(refY, yi);
                worstInt8 = Math.Max(worstInt8, err);
                Assert.True(err < 0.06, $"int8 err {err}");
                // The integer kernel must be deterministic across threading.
                Assert.Equal(yi, Run(layer, x, null, LittleBitKernel.VnniInt8));
            }
        }
        output.WriteLine($"{dOut}x{dIn} r={r} paths={npaths}: worst int8-activation max-rel error {worstInt8:E2}");
    }
}
