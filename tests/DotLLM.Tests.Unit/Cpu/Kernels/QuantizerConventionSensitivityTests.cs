using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1499 / #1500: proves, without a GPU, that the tie corpora can tell the rounding contract apart from the conventions
/// the GPU quantizers used before the fix. Each pre-fix convention is modelled in IEEE float arithmetic (which is what
/// the C# JIT and <c>__fdiv_rn</c>/<c>__frcp_rn</c>/<c>__float2int_rn</c> compute) and compared lane-for-lane and
/// scale-bit-for-bit against the CPU reference quantizers. If a mutant produced zero differences on the corpus, the
/// device tests built on the same corpus could not fail either - this is the sensitivity control for them.
/// <para>
/// Contract (CPU == Vulkan == CUDA): <c>d = amax / qmax</c> IEEE, <c>id = 1 / d</c> IEEE, <c>q = roundHalfToEven(x * id)</c>.
/// Mutants: (M1) CUDA pre-#1499: multiply by <c>127 / amax</c> (one rounding) while storing <c>amax / 127</c>;
/// (M2) GLSL <c>round()</c> half-away-from-zero ties; (M3) RDNA's <c>amax * rcp(qmax)</c> lowering of the division.
/// </para>
/// </summary>
public sealed unsafe class QuantizerConventionSensitivityTests
{
    private const int Group = 32;
    private readonly ITestOutputHelper _out;
    public QuantizerConventionSensitivityTests(ITestOutputHelper output) => _out = output;

    public enum Convention { Contract, M1_MultiplyByQmaxOverAmax, M2_RoundHalfAway, M3_RcpLoweredScale }

    /// <summary>Models one block quantizer convention in float; returns int lanes and the float scale per block.</summary>
    private static (int[] Q, float[] D) Model(float[] x, int qmax, Convention c)
    {
        int blocks = x.Length / Group;
        var q = new int[x.Length];
        var d = new float[blocks];
        float rcpQmax = 1f / qmax;
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < Group; i++) amax = MathF.Max(amax, MathF.Abs(x[b * Group + i]));
            float scale = c == Convention.M3_RcpLoweredScale ? amax * rcpQmax : amax / qmax;
            float id = c switch
            {
                Convention.M1_MultiplyByQmaxOverAmax => amax > 0 ? qmax / amax : 0f,
                _ => scale != 0 ? 1f / scale : 0f,
            };
            d[b] = scale;
            for (int i = 0; i < Group; i++)
            {
                float v = x[b * Group + i] * id;
                int r = c == Convention.M2_RoundHalfAway
                    ? (int)MathF.Round(v, MidpointRounding.AwayFromZero)
                    : (int)MathF.Round(v, MidpointRounding.ToEven);
                q[b * Group + i] = Math.Clamp(r, -qmax, qmax);
            }
        }
        return (q, d);
    }

    private static (int[] Q, float[] D) CpuQ8_0(float[] x)
    {
        int blocks = x.Length / Group;
        var dest = new byte[blocks * 34];
        fixed (float* src = x) fixed (byte* dst = dest) MatMul.QuantizeF32ToQ8_0Scalar(src, dst, x.Length);
        var q = new int[x.Length];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = (float)BitConverter.ToHalf(dest, b * 34);
            for (int i = 0; i < Group; i++) q[b * Group + i] = (sbyte)dest[b * 34 + 2 + i];
        }
        return (q, d);
    }

    private static (int[] Q, float[] D) CpuQ4_0(float[] x)
    {
        int blocks = x.Length / Group;
        var dest = new byte[blocks * 18];
        fixed (float* src = x) fixed (byte* dst = dest) KvQuantize.F32ToQ4_0Scalar(src, dst, x.Length);
        var q = new int[x.Length];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = (float)BitConverter.ToHalf(dest, b * 18);
            for (int j = 0; j < 16; j++)
            {
                byte p = dest[b * 18 + 2 + j];
                q[b * Group + 2 * j] = (p & 0xF) - 8;
                q[b * Group + 2 * j + 1] = (p >> 4) - 8;
            }
        }
        return (q, d);
    }

    private (int Lanes, int Scales) Diff(string label, (int[] Q, float[] D) a, (int[] Q, float[] D) b, bool compareScaleAsHalf)
    {
        int lanes = 0, scales = 0;
        for (int i = 0; i < a.Q.Length; i++) if (a.Q[i] != b.Q[i]) lanes++;
        for (int i = 0; i < a.D.Length; i++)
        {
            bool same = compareScaleAsHalf
                ? BitConverter.HalfToInt16Bits((Half)a.D[i]) == BitConverter.HalfToInt16Bits((Half)b.D[i])
                : BitConverter.SingleToInt32Bits(a.D[i]) == BitConverter.SingleToInt32Bits(b.D[i]);
            if (!same) scales++;
        }
        _out.WriteLine($"{label}: lanes differ={lanes} scales differ={scales} (blocks={a.D.Length})");
        return (lanes, scales);
    }

    [Theory]
    [InlineData(127)]
    [InlineData(7)]
    public void Contract_IsBitIdenticalToCpuReference(int qmax)
    {
        float[] x = ActivationQuantizerCorpus.BuildCorpus(qmax, out _);
        var cpu = qmax == 127 ? CpuQ8_0(x) : CpuQ4_0(x);
        var model = Model(x, qmax, Convention.Contract);
        var (lanes, scales) = Diff($"contract vs CPU qmax={qmax}", model, cpu, compareScaleAsHalf: true);
        Assert.Equal((0, 0), (lanes, scales));
    }

    /// <summary>
    /// Each pre-fix convention must be visible on the corpus (float payload, as the Vulkan shaders see it). Scales are
    /// compared as float because the Vulkan split KV layout stores the float scale.
    /// </summary>
    [Theory]
    [InlineData(127, Convention.M1_MultiplyByQmaxOverAmax, true, false)]
    [InlineData(127, Convention.M2_RoundHalfAway, true, false)]
    [InlineData(127, Convention.M3_RcpLoweredScale, false, true)]
    [InlineData(7, Convention.M1_MultiplyByQmaxOverAmax, true, false)]
    [InlineData(7, Convention.M2_RoundHalfAway, true, false)]
    [InlineData(7, Convention.M3_RcpLoweredScale, false, true)]
    public void Mutant_DiffersFromContract_OnFloatCorpus(int qmax, Convention mutant, bool expectLaneDiffs, bool expectScaleDiffs)
    {
        float[] x = ActivationQuantizerCorpus.BuildCorpus(qmax, out _);
        var contract = Model(x, qmax, Convention.Contract);
        var m = Model(x, qmax, mutant);
        var (lanes, scales) = Diff($"{mutant} qmax={qmax}", m, contract, compareScaleAsHalf: false);
        if (expectLaneDiffs) Assert.True(lanes > 0, $"{mutant} is indistinguishable from the contract on the corpus lanes");
        if (expectScaleDiffs) Assert.True(scales > 0, $"{mutant} is indistinguishable from the contract on the corpus scales");
    }

    /// <summary>
    /// The CUDA quantizers take FP16 input and store an FP16 scale; the pre-fix M1 convention must still be visible on
    /// the Half corpus (the one the CUDA device test uses) - otherwise that device test could not fail.
    /// </summary>
    [Theory]
    [InlineData(127)]
    [InlineData(7)]
    public void Mutant_M1_DiffersFromContract_OnHalfCorpus(int qmax)
    {
        float[] x = ActivationQuantizerCorpus.BuildHalfCorpus(qmax, out _);
        var contract = Model(x, qmax, Convention.Contract);
        var cpu = qmax == 127 ? CpuQ8_0(x) : CpuQ4_0(x);
        Assert.Equal((0, 0), Diff($"half corpus contract vs CPU qmax={qmax}", contract, cpu, compareScaleAsHalf: true));
        var m1 = Model(x, qmax, Convention.M1_MultiplyByQmaxOverAmax);
        var (lanes, _) = Diff($"half corpus M1 qmax={qmax}", m1, contract, compareScaleAsHalf: true);
        Assert.True(lanes > 0, "M1 (127/amax multiplier) is indistinguishable from the contract on the Half corpus");
        var m2 = Model(x, qmax, Convention.M2_RoundHalfAway);
        Assert.True(Diff($"half corpus M2 qmax={qmax}", m2, contract, compareScaleAsHalf: true).Lanes > 0);
    }
}
